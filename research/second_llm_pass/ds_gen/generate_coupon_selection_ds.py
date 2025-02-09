from os import getenv
import sys
from typing import List, Tuple, Dict, Optional, Callable, TypedDict
import re
import json
import datasets
import pandas as pd
from huggingface_hub import login
from huggingface_hub import HfApi


# Constants
TAG_B_COUPON = 'B-COUPON' # begin coupon tag
TAG_I_COUPON = 'I-COUPON' # inside coupon tag
TAG_UNKNOWN = 'UNKNOWN' # not-a-coupon tag

COL_TEXT_FULL = 'content_full' # column from coupons frame with full coupon text
COL_CONTENT_TEXT = 'text' # column from content_generic file containing text
COL_GROUPBY = 'id' # column from both coupons and content_generic to group by values
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


def annotate_frame_by_matches(content_frame: pd.DataFrame, coupons_frame: pd.DataFrame, content_full_format: int = 1) -> pd.DataFrame:
    """
    Adds column to dataframe with is_coupon flag.
    This function is created to deal with coupon frames with no beginning indices provided and should be
    executed on input for singe value of COL_GROUPBY.
    content_full_format is int specifying format of COL_TEXT_FULL in coupons_frame:
    - 1: format from "coupons_1" dataset
    - 2: format from "coupons big" dataset
    """
    content_frame.dropna(subset=[COL_CONTENT_TEXT], inplace=True)
    content_frame.reset_index(drop=True, inplace=True)
    coupons_list = coupons_frame[COL_TEXT_FULL].dropna().tolist()
    if content_full_format == 2:
        ptree = build_ptree([[t[1:-1] for t in s[1:-1].split(',')] for s in coupons_list])
    elif content_full_format == 1:
        ptree = build_ptree([s[1:-1].split(', ') for s in coupons_list])
    else:
        raise ValueError('content_full_format must be 1 or 2')
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
            is_coupon_array += [False] * (chosen[1] - len(is_coupon_array) - 1)
            is_coupon_array += [True] * chosen_len
            ptree_iters.clear()
            ix = chosen[2] + 1
            continue
        if text in ptree[0]:
            ptree_iters.append([ptree[0][text], ix, -1 if not ptree[0][text][1] else ix])
        ix += 1
    is_coupon_array += [False] * (len(content_frame) - len(is_coupon_array))
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

def batch_to_json(batch: pd.DataFrame) -> TreeNode:
    """
    Takes batch representing single screen content and converts it to JSON representing XML structure.
    """
    tree_path = []
    res = {"text": None, "children": {}, "is_coupon": False}

    def _insert_at_path(key: str, val: TreeNode):
        t = res
        for k, d in tree_path:
            t = t["children"][k]
        if key in t['children']:
            key = f"{key}_0"
        i = 0
        while key in t["children"]:
            i += 1
            key = key.rsplit("_", 1)[0] + f"_{i}"
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

def frame_to_json(frame: pd.DataFrame, coupons_frame: pd.DataFrame, annotation_cb: Callable[[pd.DataFrame, pd.DataFrame, int], pd.DataFrame] = annotate_frame_by_matches, fmt: int = 1) -> List[TreeNode]:
    """
    Takes content frame with multiple timestamps and converts each of them to JSON representation.
    """
    res = []
    has_coupons = []
    for i, (t, subframe) in enumerate(frame.groupby(COL_GROUPBY)):
        subframe = annotation_cb(subframe, coupons_frame[coupons_frame[COL_GROUPBY] == t], fmt)
        tree = batch_to_json(subframe)
        tree = collapse_tree(tree)[0]
        if tree is not None:
            res.append(tree)
        else:
            res.append({"text": None, "children": {}, "is_coupon": False})
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
            root['text'] = root['text'].replace('{}', '[]') if root['text'] is not None else None
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


def publish_to_hub(samples: List[Tuple[List[str], List[int]]], save_name: str, apikey: str) -> None:
    """
    Creates dataset out of list of pairs of words and labels and pushes it to HF Hub.
    """
    features = datasets.Features({
        "texts": datasets.Sequence(datasets.Value("string")),
        "labels": datasets.Sequence(LABELS)
    })
    ds = datasets.Dataset.from_dict(
    {
        "texts": list(sample[0] for sample in samples),
        "labels": list(sample[1] for sample in samples)
    },
    features=features
    )
    login(token=apikey)
    api = HfApi()
    api.create_repo(repo_id=save_name, repo_type="dataset", private=True)

    ds.push_to_hub(save_name, private=True)


if __name__ == '__main__':
    HF_HUB_KEY = getenv('HF_HUB_KEY')
    assert len(sys.argv) == 3, f"usage: {sys.argv[0]} <config_path> <ds_name>"
    config_path = sys.argv[1]
    ds_name = sys.argv[2]
    try:
        config = json.load(open(config_path))
    except Exception as e:
        print("error reading config file:", e)
        exit(1)
    try:
        frame_pairs = list([(entry['content'], entry['coupons']) for entry in config['frames']])
        formats = list([entry["format"] for entry in config['frames']])
        json_format = config['json_format']
    except KeyError as e:
        print(f"KeyError: {e}, config should be in format {{\"json_format\": true,\"frames\": [{{\"coupons\": path, \"content\": path, \"format\": 1}},...]}}")
        exit(1)

    examples = []
    for fmt, (content, coupons) in zip(formats, frame_pairs):
        content_frame = pd.read_csv(content)
        coupons_frame = pd.read_csv(coupons)
        if json_format:
            as_json = frame_to_json(content_frame, coupons_frame, fmt=fmt)
            examples.extend(json_to_labeled_tokens(as_json))
        else:
            for val, subframe in content_frame.groupby(COL_GROUPBY):
                subframe = annotate_frame_by_matches(subframe, coupons_frame[coupons_frame[COL_GROUPBY] == val], content_full_format=fmt)
                last_label = LBL_UNK
                labels = []
                words = []
                for i, row in subframe.iterrows():
                    text = row[COL_CONTENT_TEXT]
                    is_coupon = row[COL_IS_COUPON]
                    if not isinstance(text, str) or not text:
                        continue
                    words.extend(text.split())
                    if not is_coupon:
                        labels += [LBL_UNK] * len(text.split())
                    else:
                        if last_label != LBL_UNK:
                            labels += [LBL_IC] * len(text.split())
                        else:
                            labels += [LBL_BC] + [LBL_IC] * (len(text.split()) - 1)
                    last_label = labels[-1]
                examples.append((words, labels))

    publish_to_hub(examples, f"zpp-murmuras/{ds_name}", HF_HUB_KEY)
