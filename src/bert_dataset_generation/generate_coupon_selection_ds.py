import time
from os import getenv
import sys
from typing import List, Tuple, Dict, Optional, TypedDict
import re
import json
import datasets
import pandas as pd
from pandas import isna
from datasets import DatasetDict, Dataset
from huggingface_hub import login
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

from src.constants import *

# Constants
TAG_B_COUPON = 'B-COUPON'  # begin coupon tag
TAG_I_COUPON = 'I-COUPON'  # inside coupon tag
TAG_UNKNOWN = 'O'  # not-a-coupon tag

COL_TEXT_FULL = 'content_full'  # column from coupons frame with full coupon text
COL_CONTENT_TEXT = 'text'  # column from content_generic file containing text
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
    if path[0] not in root[0]:
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
        coupons_list_converted = []
        for s in coupons_list:
            try:
                l = eval(s)
            except Exception:
                l = [t[1:-1] for t in s[1:-1].split(',')]
            coupons_list_converted.append(l)
        ptree = build_ptree(coupons_list_converted)
    elif ds_format == 1:
        ptree = build_ptree([s[1:-1].split(', ') for s in coupons_list])
    else:
        raise ValueError('content_full_format must be 1 or 2')
    return ptree


def toposort_by_prefixes(strings: List[str]) -> List[int]:
    """
    assumes no duplicates among strings
    """
    n = len(strings)
    pfix_counts = [0] * n
    for i1, s in enumerate(strings):
        for i, s2 in enumerate(strings):
            if i1 == i:
                continue
            if s.startswith(s2):
                pfix_counts[i] += 1
    res = []
    not_prefixes = []
    for i in range(n):
        if pfix_counts[i] == 0:
            not_prefixes.append(i)
    while not_prefixes:
        ix = not_prefixes.pop()
        res.append(ix)
        for i, s in enumerate(strings):
            if strings[ix].startswith(s):
                pfix_counts[i] -= 1
                if not pfix_counts[i]:
                    not_prefixes.append(i)
    return res


def __find_given_starts_ends(string1: str, string2: str, starts: List[int], ends: List[int]):
    """
    Finds the first occurrence of string2 in string2
    given lists of valid indexes for the first and the last char of string2 in string1
    """
    beg = 0
    while beg < len(string1):
        ix = beg + string1[beg:].find(string2)
        if ix == beg - 1:
            return -1
        if ix in starts and ix + len(string2) - 1 in ends:
            return ix
        beg = ix + 1
    return -1


def annotate_frame_by_matches(content_frame: pd.DataFrame, coupons_list: List[str], fmt: int) -> pd.DataFrame:
    content_frame.reset_index(drop=True, inplace=True)
    labels_list = [LBL_UNK for _ in range(len(content_frame))]
    if fmt == 2:
        texts_combined = ','.join([f"'{x}'" for x in content_frame[COL_CONTENT_TEXT].dropna().tolist()])
    else:
        texts_combined = ', '.join(content_frame[COL_CONTENT_TEXT].dropna().tolist())
    char_to_row_no = []
    fields_starts = []
    fields_ends = []
    for i, row in content_frame.iterrows():
        text = row[COL_CONTENT_TEXT]
        if isna(text):
            continue
        if char_to_row_no:
            char_to_row_no.extend([-1, -1])
        elif fmt == 2:
            char_to_row_no.append(-1)
        if fmt == 1:
            fields_starts.append(len(char_to_row_no))
        else:
            fields_starts.append(len(char_to_row_no) - 1)
        char_to_row_no.extend([i] * len(text))
        if fmt == 2:
            char_to_row_no.append(-1)
        fields_ends.append(len(char_to_row_no) - 1)

    coupons_counts = {}
    for c in coupons_list:
        if c not in coupons_counts:
            coupons_counts[c] = 0
        coupons_counts[c] += 1
    coupons_unique = list(coupons_counts.keys())
    for coupon_text_ix in toposort_by_prefixes(coupons_unique):
        coupon_text = coupons_unique[coupon_text_ix]
        while (ix := __find_given_starts_ends(texts_combined, coupon_text, fields_starts, fields_ends)) != -1:
            first = True
            for row_index in sorted(set(char_to_row_no[ix:ix + len(coupon_text)])):
                if row_index == -1:
                    continue
                if first:
                    first = False
                    labels_list[row_index] = LBL_BC
                else:
                    labels_list[row_index] = LBL_IC
            # i hope that no \0 is in coupon strings xd
            texts_combined = texts_combined[:ix] + '\0' * len(coupon_text) + texts_combined[ix + len(coupon_text):]
            coupons_counts[coupon_text] -= 1
    content_frame = content_frame.copy(deep=True)
    content_frame[__COL_IS_COUPON] = labels_list
    return content_frame


def annotate_frame_by_matches_format_2(content_frame: pd.DataFrame, coupons_ptree: PTreeNode) -> pd.DataFrame:
    """
    Adds column to dataframe with tokenization labels.
    This function is created to deal with coupon frames with no beginning indices provided and should be
    executed on input for single value of AGGREGATION_COLUMN.
    """
    is_coupon_array = []
    ptree_iters: List[List] = []
    ix = 0
    text_col = content_frame[COL_CONTENT_TEXT]
    while ix < len(text_col):
        text = text_col[ix]
        if pd.isna(text):
            ix += 1
            continue
        else:
            text = str(text)
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
            is_coupon_array += [LBL_UNK] * (chosen[1] - len(is_coupon_array))
            is_coupon_array.append(LBL_BC)
            is_coupon_array += [LBL_IC] * (chosen_len - 1)
            ptree_iters.clear()
            ix = chosen[2] + 1
            continue
        if text in coupons_ptree[0]:
            ptree_iters.append([coupons_ptree[0][text], ix, -1 if not coupons_ptree[0][text][1] else ix])
        ix += 1
    # searching for possible coupons in running iterators
    candidates = []
    for itr in ptree_iters:
        if itr[2] != -1:
            candidates.append(itr)
    if candidates:
        chosen = None
        chosen_len = 0
        for itr in candidates:
            if itr[2] - itr[1] + 1 > chosen_len:
                chosen = itr
                chosen_len = itr[2] - itr[1] + 1
        is_coupon_array += [LBL_UNK] * (chosen[1] - len(is_coupon_array))
        is_coupon_array.append(LBL_BC)
        is_coupon_array += [LBL_IC] * (chosen_len - 1)
    is_coupon_array += [LBL_UNK] * (len(content_frame) - len(is_coupon_array))
    new_content_frame = content_frame.copy()
    new_content_frame[__COL_IS_COUPON] = is_coupon_array
    return new_content_frame


class TreeNode(TypedDict):
    """helper class used to operating on json"""
    children: Dict[str, 'TreeNode']
    is_coupon: int
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
    res = TreeNode(text=None, children={}, is_coupon=LBL_UNK)
    for row in batch.iterrows():
        text_field = row[1][COL_CONTENT_TEXT]
        name = row[1][COL_VIEW_ID]
        if pd.isna(name):
            name = str(name).rsplit('/')[-1]
        if pd.isna(text_field):
            text_field = None
        else:
            text_field = str(text_field)
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
    for i, (t, subframe) in enumerate(frame.groupby(AGGREGATION_COLUMN, sort=False)):
        subframe.reset_index(inplace=True, drop=True)
        subframe = annotate_frame_by_matches(subframe, [x[1:-1] for x in coupons_frame[coupons_frame[AGGREGATION_COLUMN] == t]["content_full"]], fmt)
        tree = batch_to_json(subframe)
        tree = collapse_tree(tree)[0]
        if tree is not None:
            res.append(tree)
        else:
            res.append({"text": None, "children": {}, "is_coupon": LBL_UNK})
    return res


def __encode_json_tree_node_with_children_into_tokens(root: TreeNode, is_coupon: int, indent: Optional[int]):
    """
    Converts a node with children from json tree to a pair of lists of tokens (words) and labels associated with them.
    :param root: a node to convert, it is expected not to contain an is_coupon flag.
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
    words1 = re.split("[ \n\t]+", string1)
    words2 = re.split("[ \n\t]+", string2)
    labels1 = [is_coupon]
    labels1 += [is_coupon if is_coupon != LBL_BC else LBL_IC] * (len(words1) - 1)
    first_child = True
    for name, child in children.items():
        words_child, labels_child = __encode_json_tree_into_tokens_rec(child, indent)
        words1.append(f'"{name}":')
        if labels_child[0] != LBL_IC:
            labels1.append(LBL_UNK)
        else:
            labels1.append(LBL_IC)
        is_coupon = LBL_UNK if labels_child[-1] == LBL_UNK else LBL_IC
        if not first_child:
            words1[-2] += ','
        first_child = False
        labels1 += labels_child
        words1 += words_child
    if is_coupon == LBL_BC:
        is_coupon = LBL_IC
    labels2 = [is_coupon] * len(words2)
    return words1 + words2, labels1 + labels2


def __encode_json_tree_into_tokens_rec(root: TreeNode, indent: Optional[int])\
        -> Tuple[List[str], List[int]]:
    """
    Converts a node from json tree to a pair of lists of tokens (words) and labels associated with them.
    :param root: a node to convert
    :param indent: indent for dumping JSON into string
    :return: a pair of list of tokens (words) and labels associated with them.
    """
    root = dict(root)
    is_coupon_local = root.pop('is_coupon')
    if 'children' in root:
        return __encode_json_tree_node_with_children_into_tokens(root, is_coupon_local, indent)
    string = json.dumps(root, indent=indent)
    words = re.split("[ \n\t]+", string)
    if is_coupon_local != LBL_UNK:
        labels = [is_coupon_local]
        labels += [LBL_IC] * (len(words) - 1)
    else:
        labels = [LBL_UNK] * len(words)
    return words, labels


def json_to_labeled_tokens(data: List[TreeNode], indent: Optional[int] = None) -> List[Tuple[List[str], List[int]]]:
    """
    Converts JSON representation of content to list of training samples.
    Each training sample consists of list of words and list of target labels.
    """
    res = []
    for tree in data:
        tkns, lbls = __encode_json_tree_into_tokens_rec(tree, indent)
        res.append((tkns, lbls))
    return res


def drop_duplicates(samples: List[Tuple[List[str], List[int]]]) -> List[Tuple[List[str], List[int]]]:
    """
    Goes through a list of training samples and drops duplicate samples.
    """
    samples_filtered = []
    samples_set = set()
    for sample in samples:
        key = (" ".join(sample[0]), " ".join([str(x) for x in sample[1]]))
        if key not in samples_set:
            samples_set.add(key)
            samples_filtered.append(sample)
    return samples_filtered


def publish_to_hub(samples: List[List[Tuple[List[str], List[int]]]], save_name: str, apikey: str, new_repo: bool, custom_splits: Optional[List[str]]) -> None:
    """
    Creates dataset out of list of lists (one for each pair of frames) of pairs of words and labels and pushes it to HF Hub.
    """

    samples = [drop_duplicates(batch) for batch in samples]

    features = datasets.Features({
        "texts": datasets.Sequence(datasets.Value("string")),
        "labels": datasets.Sequence(LABELS)
    })
    if custom_splits is None:
        # Convert samples into a dictionary
        texts = [sample[0] for samples_pack in samples for sample in samples_pack]
        labels = [sample[1] for samples_pack in samples for sample in samples_pack]

        # Initial train/test split (80% train, 20% temp)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Split temp set into validation (10%) and test (10%)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )

        # Create Dataset objects
        dataset_dict = DatasetDict({
            "train": Dataset.from_dict({"texts": train_texts, "labels": train_labels}, features=features),
            "validation": Dataset.from_dict({"texts": val_texts, "labels": val_labels}, features=features),
            "test": Dataset.from_dict({"texts": test_texts, "labels": test_labels}, features=features)
        })
    else:
        grouped = {}
        for name, samples_pack in zip(custom_splits, samples):
            if name not in grouped:
                grouped[name] = [[], []]
            texts = [sample[0] for sample in samples_pack]
            labels = [sample[1] for sample in samples_pack]
            grouped[name][0].extend(texts)
            grouped[name][1].extend(labels)
        for k, v in grouped.items():
            grouped[k] = Dataset.from_dict({"texts": v[0], "labels": v[1]}, features=features)
        dataset_dict = DatasetDict(grouped)
        dataset_dict = dataset_dict.shuffle(seed=time.time_ns())

    login(token=apikey)
    if new_repo:
        api = HfApi()
        api.create_repo(repo_id=save_name, repo_type="dataset", private=True)

    dataset_dict.push_to_hub(save_name, private=True)


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
        for val, subframe in content_frame.groupby(AGGREGATION_COLUMN, sort=False):
            subframe = __clear_content_frame(subframe)
            coupons_strings = [string[1:-1] for string in coupons_frame[coupons_frame[AGGREGATION_COLUMN] == val][COL_TEXT_FULL].tolist()]
            subframe = annotate_frame_by_matches(subframe, coupons_strings, fmt)
            labels = []
            words = []
            for i, row in subframe.iterrows():
                text = row[COL_CONTENT_TEXT]
                is_coupon_lbl = row[__COL_IS_COUPON]
                if pd.isna(text) or text == '':
                    continue
                else:
                    text = str(text)
                words.extend(text.split())
                if is_coupon_lbl == LBL_UNK:
                    labels += [LBL_UNK] * len(text.split())
                else:
                    if is_coupon_lbl == LBL_BC:
                        labels.append(LBL_BC)
                    else:
                        labels.append(LBL_IC)
                    labels.extend([LBL_IC] * (len(text.split()) - 1))
            if words:
                samples.append((words, labels))
        return samples


if __name__ == '__main__':
    HF_HUB_KEY = getenv('HF_HUB_KEY')
    assert len(sys.argv) == 5, f"usage: {sys.argv[0]} <config_path> <ds_name> <create_repo: y/n> <custom_split: y/n>"
    config_path = sys.argv[1]
    ds_name = sys.argv[2]
    create_repo = sys.argv[3]
    custom_split = sys.argv[4]
    if (create_repo != 'y') and (create_repo != 'n'):
        print("create_repo must be either 'y' or 'n'")
        exit(1)
    if custom_split not in ('y', 'n'):
        print("custom_split must be either 'y', 'n'")
        exit(1)
    config = json.load(open(config_path))
    try:
        frame_pairs = list([(entry['content'], entry['coupons']) for entry in config['frames']])
        formats = list([entry["format"] for entry in config['frames']])
        json_format = config['json_format']
        splits = list([entry['split'] for entry in config['frames']]) if custom_split == 'y' else None
    except KeyError as e:
        print(
            f"KeyError: {e}, config should be in format {{\"json_format\": true,\"frames\": "
            f"[{{\"coupons\": path, \"content\": path, \"format\": 1, \"split\": \"obligatory if custom_split=y\"}},...]}}")
        exit(1)

    examples = []
    for fmt, (content, coupons) in zip(formats, frame_pairs):
        content_frame = pd.read_csv(content)
        coupons_frame = pd.read_csv(coupons)
        examples.append(__samples_from_entry(fmt, content_frame, coupons_frame, json_format))

    publish_to_hub(examples, f"zpp-murmuras/{ds_name}", HF_HUB_KEY, create_repo == 'y', splits)
