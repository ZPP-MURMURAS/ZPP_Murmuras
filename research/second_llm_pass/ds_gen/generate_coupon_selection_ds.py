from os import getenv
import sys
from typing import List, Tuple, Dict, Optional, Callable, TypedDict
import re
import json
import datasets
import pandas as pd


# Constants
TAG_B_COUPON = 'B-COUPON' # begin coupon tag
TAG_I_COUPON = 'I-COUPON' # inside coupon tag
TAG_UNKNOWN = 'UNKNOWN' # not-a-coupon tag

COL_TEXT_FULL = 'content_full' # column from coupons frame with full coupon text
COL_IND_BEGIN = '' # column from coupons frame with index of coupon beginning
COL_CONTENT_TEXT = 'text' # column from content_generic file containing text
COL_TIME = 'time' # column from both coupons and content_generic with time
COL_VIEW_ID = 'view_id' # column for view id in content_generic frame
COL_DEPTH = 'view_depth' # column with view depth from content_generic frame

COL_IS_COUPON = 'is_coupon' # newly created column with flag indicating that a row is part of a coupon; internal use


# target labels
LABELS = datasets.ClassLabel(names=[TAG_UNKNOWN, TAG_B_COUPON, TAG_I_COUPON])
LBL_UNK = LABELS.str2int(TAG_UNKNOWN)
LBL_BC = LABELS.str2int(TAG_B_COUPON)
LBL_IC = LABELS.str2int(TAG_I_COUPON)


# prefix tree utils
# implemented prefix tree operates on words as atomic parts of sequences

PTreeNode = Tuple[Dict[str, 'PTreeNode'], bool] # children: dict, is_valid_coupon: bool


def ptree_insert(root: PTreeNode, path: List[str]):
    """
    inserts new node to tree with given root under path provided
    """
    if not path: return
    if path[0] not in root:
        root[0][path[0]] = ({}, len(path) == 1)
    ptree_insert(root[0][path[0]], path[1:])


def build_ptree(strings: List[List[str]]) -> PTreeNode:
    """
    constructs tree from list of sequences
    """
    root = ({}, False)
    for s in strings:
        ptree_insert(root, s)
    return root


def annotate_frame_from_indices(content_frame: pd.DataFrame, coupons_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adds column to dataframe with is_coupon flag.
    This function is expected to be used when coupons_frame has COL_IND_BEGIN column.
    In other cases, you should use annotate_frame_by_matches.
    This function is expected to work with content_frame that is associated with single timestamp
    """
    assert COL_IND_BEGIN in coupons_frame.columns
    assert COL_TEXT_FULL in coupons_frame.columns
    assert COL_CONTENT_TEXT in content_frame.columns
    as_list = list(zip(coupons_frame[COL_IND_BEGIN], content_frame[COL_TEXT_FULL]))
    as_list.sort(key=lambda x: x[0])
    itr = 0
    col_is_coupon = []
    while itr < len(as_list):
        assert as_list[itr][0] >= len(col_is_coupon)
        col_is_coupon += [False] * (as_list[itr][0] - len(col_is_coupon))
        acc_text = ""
        row = as_list[itr][0]
        while len(acc_text) + 2 < len(as_list[itr][1]): # + 2 is for handling brackets around list
            if acc_text == "":
                acc_text = content_frame[COL_CONTENT_TEXT][row]
            else:
                acc_text += ", " + content_frame[COL_CONTENT_TEXT][row]
            row += 1
        assert f"[{acc_text}]" == as_list[itr][1]
        col_is_coupon += [True] * (row - len(col_is_coupon))
        itr += 1
    content_frame[COL_IS_COUPON] = col_is_coupon
    return content_frame


def annotate_frame_by_matches(content_frame: pd.DataFrame, coupons_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Adds column to dataframe with is_coupon flag.
    This function is created to deal with coupon frames with no beginning indices provided.
    """
    content_frame.dropna(subset=[COL_CONTENT_TEXT], inplace=True)
    content_frame.reset_index(drop=True, inplace=True)
    coupons_list = coupons_frame[COL_TEXT_FULL].dropna().tolist()
    ptree = build_ptree([s[1:-1].split(', ') for s in coupons_list])
    is_coupon_array = []
    ptree_iters: List[List] = []
    ix = 0
    text_col = content_frame[COL_CONTENT_TEXT]
    while ix < len(text_col):
        text = text_col[ix]
        if not isinstance(text, str):
            ix += 1
            continue
        ended_iters = []
        for itr in ptree_iters:
            if text not in itr[0][0]:
                if itr[2] != -1:
                    ended_iters.append(itr)
            else:
                itr[0] = itr[0][0][text]
                if itr[0][1]:
                    itr[2]= ix
        if ended_iters:
            chosen = None
            chosen_len = 0
            for itr in ended_iters:
                if itr[2] - itr[1] + 1 > chosen_len:
                    chosen = itr
                    chosen_len = itr[2] - itr[1] + 1
            is_coupon_array += [LBL_UNK] * (chosen[1] - len(is_coupon_array) - 1)
            is_coupon_array.append(LBL_BC)
            is_coupon_array += [LBL_IC] * (chosen_len - 1)
            ptree_iters.clear()
            ix = chosen[2] + 1
            continue
        if text in ptree[0]:
            ptree_iters.append([ptree[0][text], ix, -1 if not ptree[0][text][1] else ix])
        ix += 1
    is_coupon_array += [LBL_UNK] * (len(content_frame) - len(is_coupon_array))
    content_frame[COL_IS_COUPON] = is_coupon_array
    return content_frame


class TreeNode(TypedDict):
    """helper class used to operating on json"""
    children: Dict[str, 'TreeNode']
    is_coupon: bool
    text: Optional[str]


def collapse_tree(tree: TreeNode) -> Tuple[Optional[TreeNode], str]:
    """
    Removes nodes that have only one child and no text.
    Method for reducing size of json representing content
    """
    if len(tree['children']) < 2 and tree['text'] is None:
        if len(tree['children']) == 1:
            child_name, child = list(tree['children'].items())[0]
            collapsed, name = collapse_tree(child)
            if collapsed is not None:
                name = f"{child_name}.{name}" if name else child_name
            return collapsed, name
        return None, ""
    new_children = {}
    for child_name, child in tree['children'].items():
        collapsed, suffix = collapse_tree(child)
        if collapsed is not None:
            if suffix:
                new_children[f"{child_name}.{suffix}"] = collapsed
            else:
                new_children[child_name] = collapsed
    tree['children'] = new_children
    return tree, ""

def timestamp_batch_to_json(batch: pd.DataFrame):
    """
    Takes batch representing single screen content and converts it to JSON representing XML structure.
    """
    tree_path = []
    res = {"text": None, "children": {}, "is_coupon": False}

    def _insert_at_path(key: str, val: TreeNode):
        t = res
        for k, d in tree_path:
            t = t["children"][k]
        t["children"][key] = val

    for row in batch.iterrows():
        text_field = row[1][COL_CONTENT_TEXT]
        name = row[1][COL_VIEW_ID]
        if isinstance(name, str):
            name = name.rsplit('/')[-1]
        if not isinstance(text_field, str):
            text_field = None
        depth = row[1][COL_DEPTH]
        while len(tree_path) > 0 and tree_path[-1][1] >= depth:
            tree_path.pop(-1)
        _insert_at_path(name, {"text": text_field, "children": {}, "is_coupon": row[1][COL_IS_COUPON]})
        tree_path.append((name, depth))

    return res

def frame_to_json(frame: pd.DataFrame, coupons_frame: pd.DataFrame, annotation_cb: Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame] = annotate_frame_by_matches) -> List[TreeNode]:
    """
    Takes content frame with multiple timestamps and converts each of them to JSON representation.
    """
    res = []
    for t, subframe in frame.groupby(COL_TIME):
        coupons_for_time = coupons_frame[coupons_frame[COL_TIME] == t][COL_TEXT_FULL].to_list()
        subframe = annotation_cb(subframe, coupons_for_time)
        tree = timestamp_batch_to_json(subframe)
        tree = collapse_tree(tree)[0]
        if tree is not None:
            res.append(tree)
    return res


def json_to_labeled_tokens(data: List[TreeNode], indent: Optional[int]=None) -> List[Tuple[List[str], List[int]]]:
    """
    Converts JSON representation of content to list of training samples.
    Each training sample consists of list of words and list of target labels.
    """
    def _encode_tree_rec(root: TreeNode, is_coupon) -> Tuple[List[str], List[bool]]:
        root = dict(root)
        is_coupon |= root.pop('is_coupon')
        if 'children' in root:
            children = root['children']
            root['children'] = {}
            string = json.dumps(root, indent=indent)
            string1, string2 = string.rsplit('{}', maxsplit=1)
            string1 += '{'
            string2 = '}' + string2
            words1 = re.split("[ \n\t]", string1)
            words2 = re.split("[ \n\t]", string2)
            labels1 = [is_coupon] * len(words1)
            labels2 = [is_coupon] * len(words2)
            first_child = True
            for name, child in children.items():
                words_child, labels_child = _encode_tree_rec(child, is_coupon)
                words1.append(f'{name}:')
                labels1.append(is_coupon)
                if not first_child:
                    words1[-2] += ','
                first_child = False
                labels1 += labels_child
                words1 += words_child
            return words1 + words2, labels1 + labels2
        else:
            string = json.dumps(root, indent=indent)
            words = re.split("[ \n\t]", string)
            labels = [is_coupon] * len(words)
            return words, labels

    res = []
    for tree in data:
        tkns, lbls = _encode_tree_rec(tree, False)
        prv = LBL_UNK
        lbls = [prv := (LBL_UNK if not lbl else LBL_BC if prv == LBL_UNK else LBL_IC) for lbl in lbls]
        res.append((tkns, lbls))
    return res


if __name__ == '__main__':
    HF_HUB_KEY = getenv('HF_HUB_KEY')
    assert len(sys.argv) == 2, f"usage: {sys.argv[0]} config_path"
    config_path = sys.argv[1]
    config = json.load(open(config_path))
    frame_pairs = []
