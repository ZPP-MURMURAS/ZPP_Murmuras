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
TAG_B_COUPON = 'B-COUPON'  # begin coupon tag
TAG_I_COUPON = 'I-COUPON'  # inside coupon tag
TAG_UNKNOWN = 'UNKNOWN'  # not-a-coupon tag

COL_TEXT_FULL = 'content_full'  # column from coupons frame with full coupon text
COL_CONTENT_TEXT = 'text'  # column from content_generic file containing text
COL_GROUPBY = 'id'  # column from both coupons and content_generic to group by values
COL_VIEW_ID = 'view_id'  # column for view id in content_generic frame
COL_DEPTH = 'view_depth'  # column with view depth from content_generic frame

__COL_IS_COUPON = 'is_coupon'  # newly created column with flag indicating that a row is part of a coupon; internal use

# target labels
LABELS = datasets.ClassLabel(names=[TAG_UNKNOWN, TAG_B_COUPON, TAG_I_COUPON])
LBL_UNK = LABELS.str2int(TAG_UNKNOWN)
LBL_BC = LABELS.str2int(TAG_B_COUPON)
LBL_IC = LABELS.str2int(TAG_I_COUPON)

# prefix tree utils
# implemented prefix tree operates on words as atomic parts of sequences

PTreeNode = Tuple[Dict[str, 'PTreeNode'], bool]  # children: dict, is_valid_coupon: bool


def ptree_insert(root: PTreeNode, path: List[str]):
    """
    inserts new node to tree with given root under path provided
    """
    if not path:
        return
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


def __clear_content_frame(content_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic preprocessing on content frame to make it suitable for annotate_frame_by_matches.
    This includes:
    - removing rows with nan as text
    """
    content_frame = content_frame.dropna(subset=[COL_CONTENT_TEXT])
    content_frame.reset_index(drop=True, inplace=True)
    return content_frame


def __construct_prefix_tree_for_coupon_frame(coupons_frame: pd.DataFrame, ds_format: int) -> PTreeNode:
    """
    Constructs a prefix tree containing coupons from the given frame. Each node in this prefix tree
    is content of COL_TEXT column for some row.
    ds_format is int specifying format of COL_TEXT_FULL in coupons_frame:
    - 1: format from "coupons_1" dataset
    - 2: format from "coupons big" dataset
    """
    coupons_list = coupons_frame[COL_TEXT_FULL].dropna().tolist()
    if ds_format == 2:
        ptree = build_ptree([[t[1:-1] for t in s[1:-1].split(',')] for s in coupons_list])
    elif ds_format == 1:
        ptree = build_ptree([s[1:-1].split(', ') for s in coupons_list])
    else:
        raise ValueError('content_full_format must be 1 or 2')
    return ptree


def annotate_frame_by_matches(content_frame: pd.DataFrame, coupons_ptree: PTreeNode) -> pd.DataFrame:
    """
    Adds column to dataframe with is_coupon flag.
    This function is created to deal with coupon frames with no beginning indices provided and should be
    executed on input for single value of COL_GROUPBY.
    """
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
                    itr[2] = ix
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
        if text in coupons_ptree[0]:
            ptree_iters.append([coupons_ptree[0][text], ix, -1 if not coupons_ptree[0][text][1] else ix])
        ix += 1
    is_coupon_array += [False] * (len(content_frame) - len(is_coupon_array))
    new_content_frame = content_frame.copy()
    new_content_frame[__COL_IS_COUPON] = is_coupon_array
    return new_content_frame


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


def __insert_to_json_tree(tree: TreeNode, path: List[Tuple[str, int]], key: str, new_node: TreeNode) -> None:
    """
    Inserts a new TreeNode into tree representing text fields from a screen content in a form of json tree.
    :param tree: a tree to insert into
    :param path: ancestors of the node that is being inserted (alongside with their depths in the tree)
    :param key: a name to save the new node under
    :param new_node: a new node to insert
    """
    t = tree
    for k, _ in path:
        t = t["children"][k]
    # handling children with identical names
    if key in t['children']:
        key = f"{key}_0"
    i = 0
    while key in t["children"]:
        i += 1
        key = key.rsplit("_", 1)[0] + f"_{i}"
    t["children"][key] = new_node


def batch_to_json(batch: pd.DataFrame) -> TreeNode:
    """
    Takes batch representing single screen content and converts it to JSON representing XML structure.
    """
    tree_path = []
    res = {"text": None, "children": {}, "is_coupon": False}
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

        __insert_to_json_tree(res, tree_path, name,
                              {"text": text_field, "children": {}, "is_coupon": row[1][__COL_IS_COUPON]})
        tree_path.append((name, depth))

    return res


def frame_to_json(frame: pd.DataFrame, coupons_frame: pd.DataFrame, fmt: int = 1) -> List[TreeNode]:
    """
    Takes content frame with multiple timestamps and converts each of them to JSON representation.
    """
    res = []
    for i, (t, subframe) in enumerate(frame.groupby(COL_GROUPBY)):
        ptree = __construct_prefix_tree_for_coupon_frame(coupons_frame[coupons_frame[COL_GROUPBY] == t], fmt)
        subframe = __clear_content_frame(subframe)
        subframe = annotate_frame_by_matches(subframe, ptree)
        tree = batch_to_json(subframe)
        tree = collapse_tree(tree)[0]
        if tree is not None:
            res.append(tree)
        else:
            res.append({"text": None, "children": {}, "is_coupon": False})
    return res


def __encode_json_tree_node_with_children_into_tokens(root: TreeNode, is_coupon, indent: Optional[int]):
    """
    Converts a node with children from json tree to a pair of lists of tokens (words) and labels associated with them.
    :param root: a node to convert, it is expected not to contain a is_coupon flag.
    :param is_coupon: a flag carrying information if we are already inside a part of tree that is considered a coupon
    :param indent: indent for dumping JSON into string
    :return: a pair of list of tokens (words) and labels associated with them.
    """
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
        words_child, labels_child = __encode_json_tree_into_tokens_rec(child, is_coupon, indent)
        words1.append(f'{name}:')
        labels1.append(is_coupon)
        if not first_child:
            words1[-2] += ','
        first_child = False
        labels1 += labels_child
        words1 += words_child
    return words1 + words2, labels1 + labels2


def __encode_json_tree_into_tokens_rec(root: TreeNode, is_coupon, indent: Optional[int])\
        -> Tuple[List[str], List[bool]]:
    """
    Converts a node from json tree to a pair of lists of tokens (words) and labels associated with them.
    :param root: a node to convert
    :param is_coupon: a flag carrying information if we are already inside a part of tree that is considered a coupon
    :param indent: indent for dumping JSON into string
    :return: a pair of list of tokens (words) and labels associated with them.
    """
    root = dict(root)
    is_coupon |= root.pop('is_coupon')
    if 'children' in root:
        return __encode_json_tree_node_with_children_into_tokens(root, is_coupon, indent)
    string = json.dumps(root, indent=indent)
    words = re.split("[ \n\t]", string)
    labels = [is_coupon] * len(words)
    return words, labels


def json_to_labeled_tokens(data: List[TreeNode], indent: Optional[int] = None) -> List[Tuple[List[str], List[int]]]:
    """
    Converts JSON representation of content to list of training samples.
    Each training sample consists of list of words and list of target labels.
    """
    res = []
    for tree in data:
        tkns, lbls = __encode_json_tree_into_tokens_rec(tree, False, indent)
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


def __samples_from_entry(fmt: int, content_frame: pd.DataFrame, coupons_frame: pd.DataFrame, json_output: bool) \
        -> List[Tuple[List[str], List[int]]]:
    """
    Extracts ready training samples from a single entry from the config file (a tuple of a content frame,
    a coupons frame, and a frame format).
    """
    if json_output:
        as_json = frame_to_json(content_frame, coupons_frame, fmt=fmt)
        return json_to_labeled_tokens(as_json)
    else:
        samples = []
        for val, subframe in content_frame.groupby(COL_GROUPBY):
            ptree = __construct_prefix_tree_for_coupon_frame(coupons_frame[coupons_frame[COL_GROUPBY] == val], fmt)
            subframe = __clear_content_frame(subframe)
            subframe = annotate_frame_by_matches(subframe, ptree)
            last_label = LBL_UNK
            labels = []
            words = []
            for i, row in subframe.iterrows():
                text = row[COL_CONTENT_TEXT]
                is_coupon = row[__COL_IS_COUPON]
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
            samples.append((words, labels))
        return samples


if __name__ == '__main__':
    HF_HUB_KEY = getenv('HF_HUB_KEY')
    assert len(sys.argv) == 3, f"usage: {sys.argv[0]} <config_path> <ds_name>"
    config_path = sys.argv[1]
    ds_name = sys.argv[2]
    config = json.load(open(config_path))
    try:
        frame_pairs = list([(entry['content'], entry['coupons']) for entry in config['frames']])
        formats = list([entry["format"] for entry in config['frames']])
        json_format = config['json_format']
    except KeyError as e:
        print(
            f"KeyError: {e}, config should be in format {{\"json_format\": true,\"frames\": "
            f"[{{\"coupons\": path, \"content\": path, \"format\": 1}},...]}}")
        exit(1)

    examples = []
    for fmt, (content, coupons) in zip(formats, frame_pairs):
        content_frame = pd.read_csv(content)
        coupons_frame = pd.read_csv(coupons)
        examples.extend(__samples_from_entry(fmt, content_frame, coupons_frame, json_format))

    import random
    for _ in range(10):
        print(random.choice(examples))
    exit(0)

    publish_to_hub(examples, f"zpp-murmuras/{ds_name}", HF_HUB_KEY)
