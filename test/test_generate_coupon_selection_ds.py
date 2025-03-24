from src.bert_dataset_generation.generate_coupon_selection_ds import (
    __insert_to_json_tree as _ds_gen__insert_to_json_tree,
    __encode_json_tree_into_tokens_rec as _ds_gen__encode_json_tree_into_tokens_rec,
    __encode_json_tree_node_with_children_into_tokens as _ds_gen__encode_json_tree_node_with_children_into_tokens,
    __samples_from_entry as _ds_gen__samples_from_entry,
    __construct_prefix_tree_for_coupon_frame as _ds_gen__construct_prefix_tree_for_coupon_frame
)
from src.bert_dataset_generation.generate_coupon_selection_ds import *
from src.bert_dataset_generation.generate_coupon_selection_ds import __COL_IS_COUPON as _ds_gen__COL_IS_COUPON

import pandas as pd
from math import nan
from copy import deepcopy

import pytest
from pytest_lazy_fixtures import lf


@pytest.fixture
def ptree1():
    return {}, False


@pytest.fixture
def ptree2():
    return {"uwu": ({}, True)}, False

@pytest.fixture
def ptree3():
    return {"beer": ({}, True), "wine": ({"whiskey": ({}, True)}, False)}, False

@pytest.fixture
def ptree4():
    return {"beer": ({}, True), "wine": ({"whiskey": ({"vodka": ({"Tequila": ({}, True)}, False)}, True)}, False)}, False

@pytest.fixture
def frame1():
    return pd.DataFrame({
        COL_GROUPBY: [1] * 10,
        COL_CONTENT_TEXT: ['abc', nan, 'y', 'yyy', 'uwu', nan, nan, 'uwuw', nan, 'uwu'],
        COL_DEPTH: [0, 1, 2, 1, 1, 1, 2, 3, 2, 2],
        COL_VIEW_ID: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })

@pytest.fixture
def frame1_annotated(frame1):
    f1 = frame1.copy(deep=True)
    col = [LBL_UNK] * 10
    col[4] = LBL_BC
    col[9] = LBL_BC
    f1[_ds_gen__COL_IS_COUPON] = col
    return f1

@pytest.fixture
def frame2():
    return pd.DataFrame({COL_GROUPBY: [], COL_CONTENT_TEXT: []})

@pytest.fixture
def frame2_annotated(frame2):
    f2 = frame2.copy(deep=True)
    f2[_ds_gen__COL_IS_COUPON] = []
    return f2

@pytest.fixture
def frame3():
    return pd.DataFrame({
        COL_GROUPBY: [2] * 6,
        COL_CONTENT_TEXT: ['dummy', 'wine', 'whiskey', 'vodka', 'Tequila', nan],
        COL_DEPTH: [0, 1, 2, 3, 1, 2],
        COL_VIEW_ID: ['x', 'a', 'b', 'c', 'd', 'e']
    })

@pytest.fixture
def frame4():
    return pd.DataFrame({
        COL_GROUPBY: [3],
        COL_CONTENT_TEXT: [nan],
        COL_DEPTH: [0],
        COL_VIEW_ID: ['a']}
    )

@pytest.fixture
def frame_joint(frame1, frame3, frame4):
    return pd.concat([frame1, frame3, frame4], ignore_index=True)

@pytest.fixture
def frame3_annotated(frame3):
    f3 = frame3.copy(deep=True)
    f3[_ds_gen__COL_IS_COUPON] = [LBL_UNK, LBL_BC, LBL_IC, LBL_IC, LBL_IC, LBL_UNK]
    return f3

def frame_coupons(fmt):
    col_text = [['uwu'], ['beer'], ['wine', 'whiskey'], ['wine', 'whiskey', 'vodka', 'Tequila']]
    if fmt == 1:
        col_text = [str(x).replace('\'', '') for x in col_text]
    else:
        col_text = [str(x).replace(' ', '') for x in col_text]
    return pd.DataFrame({
        COL_GROUPBY: [1, 2, 2, 2],
        COL_TEXT_FULL: col_text
    })

@pytest.fixture
def json_tree1():
    inner = TreeNode(
        text="abc",
        is_coupon=LBL_UNK,
        children={
            'b': TreeNode(
                text=None,
                is_coupon=LBL_UNK,
                children={
                    'c': TreeNode(
                        text='y',
                        is_coupon=LBL_UNK,
                        children={}
                    )
                }
            ),
            'd': TreeNode(
                text='yyy',
                is_coupon=LBL_UNK,
                children={}
            ),
            'e': TreeNode(
                text='uwu',
                is_coupon=LBL_BC,
                children={}
            ),
            'f': TreeNode(
                text=None,
                is_coupon=LBL_UNK,
                children={
                    'g': TreeNode(
                        text=None,
                        is_coupon=LBL_UNK,
                        children={
                            'h': TreeNode(
                                text='uwuw',
                                is_coupon=LBL_UNK,
                                children={}
                            )
                        }
                    ),
                    'i': TreeNode(
                        text=None,
                        is_coupon=LBL_UNK,
                        children={}
                    ),
                    'j': TreeNode(
                        text='uwu',
                        is_coupon=LBL_BC,
                        children={}
                    )
                }
            )
        }
    )
    return TreeNode(text=None, is_coupon=LBL_UNK, children={'a': inner})

@pytest.fixture
def json_tree3():
    inner = TreeNode(
        text='dummy',
        is_coupon=LBL_UNK,
        children={
            'a': TreeNode(
                text="wine",
                is_coupon=LBL_BC,
                children={
                    'b': TreeNode(
                        text='whiskey',
                        is_coupon=LBL_IC,
                        children={
                            'c': TreeNode(
                                text='vodka',
                                is_coupon=LBL_IC,
                                children={}
                            )
                        }
                    )
                }
            ),
            'd': TreeNode(
                text='Tequila',
                is_coupon=LBL_IC,
                children={
                    'e': TreeNode(
                        text=None,
                        is_coupon=LBL_UNK,
                        children={}
                    )
                }
            )
        }
    )
    return TreeNode(text=None, is_coupon=LBL_UNK, children={'x': inner})

@pytest.fixture
def collapsed_json_tree1():
    return TreeNode(
        text="abc",
        is_coupon=LBL_UNK,
        children={
            'b.c': TreeNode(
                text='y',
                is_coupon=LBL_UNK,
                children={}
            ),
            'd': TreeNode(
                text='yyy',
                is_coupon=LBL_UNK,
                children={}
            ),
            'e': TreeNode(
                text='uwu',
                is_coupon=LBL_BC,
                children={}
            ),
            'f': TreeNode(
                text=None,
                is_coupon=LBL_UNK,
                children={
                    'g.h': TreeNode(
                        text='uwuw',
                        is_coupon=LBL_UNK,
                        children={}
                    ),
                    'j': TreeNode(
                        text='uwu',
                        is_coupon=LBL_BC,
                        children={}
                    )
                }
            )
        }
    )

@pytest.fixture
def collapsed_json_tree3(json_tree3):
    tree = deepcopy(json_tree3)
    tree = tree["children"]["x"]
    tree["children"]['d']["children"].pop('e')
    return tree

@pytest.fixture
def json_words_labels_joint():
    words1 = ['{"text":', '"abc",', '"children":', '{', '"b.c":', '{"text":', '"y",', '"children":', '{', '}},', '"d":', '{"text":', '"yyy",', '"children":', '{', '}},', '"e":', '{"text":', '"uwu",', '"children":', '{', '}},', '"f":', '{"text":', 'null,', '"children":', '{', '"g.h":', '{"text":', '"uwuw",', '"children":', '{', '}},', '"j":', '{"text":', '"uwu",', '"children":', '{', '}}', '}}', '}}']
    words2 = ['{"text":', '"dummy",', '"children":', '{', '"a":', '{"text":', '"wine",', '"children":', '{', '"b":', '{"text":', '"whiskey",', '"children":', '{','"c":', '{"text":', '"vodka",', '"children":', '{', '}}',  '}}', '}},', '"d":', '{"text":', '"Tequila",', '"children":', '{', '}}', '}}']
    words3 = ['{"text":', 'null,', '"children":', '{', '}}']
    labels1 = [LBL_UNK] * len(words1)
    labels1[17] = LBL_BC
    for i in range(18, 17 + 5):
        labels1[i] = LBL_IC
    labels1[34] = LBL_BC
    for i in range(35, len(words1)):
        labels1[i] = LBL_IC
    labels2 = [LBL_UNK] * len(words2)
    labels2[5] = LBL_BC
    for i in range(6, len(words2)):
        labels2[i] = LBL_IC
    labels3 = [LBL_UNK] * len(words3)
    return [(words1, labels1), (words2, labels2), (words3, labels3)]

@pytest.fixture
def plain_words_labels_joint():
    words1 = ['abc', 'y', 'yyy', 'uwu', 'uwuw', 'uwu']
    labels1 = [LBL_UNK] * len(words1)
    labels1[3] = labels1[-1] = LBL_BC
    words2 = ['dummy', 'wine', 'whiskey', 'vodka', 'Tequila']
    labels2 = [LBL_IC] * len(words2)
    labels2[1] = LBL_BC
    labels2[0] = LBL_UNK
    words3 = []
    labels3 = []
    return [(words1, labels1), (words2, labels2), (words3, labels3)]

@pytest.fixture
def json_joint_collapsed(collapsed_json_tree1, collapsed_json_tree3):
    return [collapsed_json_tree1, collapsed_json_tree3, TreeNode(text=None, is_coupon=LBL_UNK, children={})]

class TestGenerateCouponSelectionDs:
    @pytest.mark.parametrize("tree1,tree2,path", [
        (lf('ptree1'), lf('ptree2'), ["uwu"]),
        (lf('ptree3'), lf('ptree4'), ["wine", "whiskey", "vodka", "Tequila"]),
    ])
    def test_ptree_insert(self, tree1: PTreeNode, tree2: PTreeNode, path: List[str]):
        ptree_insert(tree1, path)
        assert tree1 == tree2

    @pytest.mark.parametrize("tree,sequences", [
        (lf('ptree1'), []),
        (lf('ptree2'), [["uwu"]]),
        (lf('ptree4'), [["beer"], ["wine", "whiskey"], ["wine", "whiskey", "vodka", "Tequila"]]),
    ])
    def test_build_ptree(self, tree: PTreeNode, sequences: List[List[str]]):
        assert build_ptree(sequences) == tree

    @pytest.mark.parametrize("frame_in,frame_out,ptree", [
        (lf('frame1'), lf('frame1_annotated'), lf('ptree2')),
        (lf('frame2'), lf('frame2_annotated'), lf('ptree2')),
        (lf('frame3'), lf('frame3_annotated'), lf('ptree4')),
    ])
    def test_annotate_frame_by_matches(self, frame_in: pd.DataFrame, frame_out: pd.DataFrame, ptree: PTreeNode):
        annotated = annotate_frame_by_matches_format_2(frame_in, ptree)
        assert annotated.equals(frame_out)

    def test_annotate_frame_by_matches_coupon_separation(self, ptree2):
        frame = pd.DataFrame({COL_GROUPBY: [1, 1], COL_CONTENT_TEXT: ['uwu', 'uwu']})
        target_frame = frame.copy(deep=True)
        target_frame[_ds_gen__COL_IS_COUPON] = [LBL_BC, LBL_BC]
        assert annotate_frame_by_matches_format_2(frame, ptree2).equals(target_frame)

    @pytest.mark.parametrize("tree_in,tree_out,text_out", [
        (lf('json_tree1'), lf('collapsed_json_tree1'), "a"),
        (lf('json_tree3'), lf('collapsed_json_tree3'), "x"),
        (TreeNode(children={}, is_coupon=False, text=None), None, ""),
        (TreeNode(children={"x": TreeNode(children={}, is_coupon=False, text="nested")}, is_coupon=False, text=None), TreeNode(children={}, is_coupon=False, text="nested"), "x"),
    ])
    def test_collapse_tree(self, tree_in: TreeNode, tree_out: str, text_out: str):
        collapsed = collapse_tree(tree_in)
        assert collapsed == (tree_out, text_out)

    @pytest.mark.parametrize("frame,tree_out", [
        (lf('frame1_annotated'), lf('json_tree1')),
        (lf('frame2_annotated'), TreeNode(children={}, is_coupon=False, text=None)),
        (lf('frame3_annotated'), lf('json_tree3')),
    ])
    def test_batch_to_json(self, frame: pd.DataFrame, tree_out: TreeNode):
        assert batch_to_json(frame) == tree_out

    @pytest.mark.parametrize("fmt", [1, 2])
    def test_frame_to_json(self, frame_joint: pd.DataFrame, fmt: int, json_joint_collapsed: List[TreeNode]):
        assert frame_to_json(frame_joint, frame_coupons(fmt), fmt) == json_joint_collapsed

    def test_json_to_labeled_tokens(self, json_words_labels_joint, json_joint_collapsed):
        assert json_to_labeled_tokens(json_joint_collapsed) == json_words_labels_joint

    @pytest.mark.parametrize("tree1,tree2,path", [
        (TreeNode(children={}, is_coupon=LBL_UNK, text=None), TreeNode(children={'magic_key': TreeNode(children={}, is_coupon=LBL_UNK, text="abrakadabra")}, is_coupon=LBL_UNK, text=None), []),
        (TreeNode(children={"x": TreeNode(children={}, is_coupon=LBL_UNK, text=None)}, is_coupon=LBL_UNK, text=None), TreeNode(children={"x": TreeNode(children={"magic_key": TreeNode(children={}, is_coupon=LBL_UNK, text="abrakadabra")}, is_coupon=LBL_UNK, text=None)}, is_coupon=LBL_UNK, text=None), [("x", 1)]),
        (TreeNode(children={"magic_key": TreeNode(children={}, is_coupon=LBL_UNK, text=None)}, is_coupon=LBL_UNK, text=None), TreeNode(children={"magic_key": TreeNode(children={}, is_coupon=LBL_UNK, text=None), "magic_key_0": TreeNode(children={}, is_coupon=LBL_UNK, text="abrakadabra")}, is_coupon=LBL_UNK, text=None), []),
    ])
    def test_ds_gen__insert_to_json_tree(self, tree1: TreeNode, tree2: TreeNode, path: List[Tuple[str, int]]):
        node = TreeNode(children={}, text="abrakadabra", is_coupon=LBL_UNK)
        _ds_gen__insert_to_json_tree(tree1, path, "magic_key", node)
        assert tree1 == tree2

    def test_ds_gen__encode_json_tree_into_tokens_rec(self, collapsed_json_tree1: TreeNode, collapsed_json_tree3: TreeNode, json_words_labels_joint):
        assert _ds_gen__encode_json_tree_into_tokens_rec(collapsed_json_tree1, indent=None) == json_words_labels_joint[0]
        assert _ds_gen__encode_json_tree_into_tokens_rec(collapsed_json_tree3, indent=None) == json_words_labels_joint[1]

    @pytest.mark.parametrize("is_coupon", [LBL_BC, LBL_IC, LBL_UNK])
    def test_ds_gen__encode_json_tree_node_with_children_into_tokens(self, json_words_labels_joint, collapsed_json_tree1: TreeNode, is_coupon):
        words, labels = json_words_labels_joint[0]
        labels[0] = is_coupon
        if is_coupon == LBL_BC:
            is_coupon = LBL_IC
        for i in range(4):
            labels[i] = is_coupon
        collapsed_json_tree1 = dict(collapsed_json_tree1)
        collapsed_json_tree1.pop("is_coupon")
        tree = deepcopy(collapsed_json_tree1)
        assert _ds_gen__encode_json_tree_node_with_children_into_tokens(collapsed_json_tree1, is_coupon, indent=None) == (words, labels)
        tree = dict(tree["children"]["b.c"])
        tree.pop("is_coupon")
        labels = labels[5:10]
        words = words[5:10]
        words[-1] = words[-1][:-1]
        labels[0] = is_coupon
        if is_coupon == LBL_BC:
            is_coupon = LBL_IC
        for i in range(1, len(labels)):
            labels[i] = is_coupon
        assert _ds_gen__encode_json_tree_node_with_children_into_tokens(tree, is_coupon, indent=None) == (words, labels)

    @pytest.mark.parametrize("fmt,as_json,tgt", [(1, True, lf('json_words_labels_joint')), (2, False, lf('plain_words_labels_joint'))])
    def test_ds_gen__samples_from_entry(self, fmt, frame_joint, as_json, tgt):
        assert _ds_gen__samples_from_entry(fmt, frame_joint, frame_coupons(fmt), as_json) == tgt

    @pytest.mark.parametrize("id_val,tgt_tree", [
        (1, lf('ptree2')),
        (2, lf('ptree4')),
        (3, ({}, False)),
    ])
    def test_ds_gen__construct_prefix_tree_for_coupon_frame(self, id_val, tgt_tree):
        fc = frame_coupons(fmt=1)
        assert _ds_gen__construct_prefix_tree_for_coupon_frame(fc[fc[COL_GROUPBY] == id_val], 1) == tgt_tree
