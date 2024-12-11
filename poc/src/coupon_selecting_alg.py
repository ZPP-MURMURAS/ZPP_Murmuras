from dataclasses import dataclass, field
from typing import List
import json
from collections import defaultdict

import pandas as pd

from tree_traversal import tree_traversal as tt
from constants import *
from postprocessing_utils import MultiSet


@dataclass()
class ProtoCoupon:
    """
    Class representing data associated by algorithm with single coupon.
    """
    product_name: str
    prices: List[str] = field(default_factory=list)
    percents: List[str] = field(default_factory=list)
    other_discounts: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)


def is_label_set_coupon(labels: MultiSet) -> bool:
    """
    Checks if given set of labels represents enough text to crete ProtoCoupon object
    :param labels: multiset of labels to check
    """
    if labels.count('Label.PRODUCT_NAME') == 0:
        return False
    for lbl in ['Label.PRICE', 'Label.OTHER_DISCOUNT', 'Label.PERCENT']:
        if labels.count(lbl) > 0:
            return True
    return False


def proto_coupons_from_frame(
        frame: pd.DataFrame,
        label_col: str = LABEL_COLUMN,
        timestamp_col: str = TIMESTAMP_COLUMN,
) -> List[ProtoCoupon]:
    """
    this function takes dataframe containing column with mapped fragments of text to tokens (ex output of
    postprocessing_utils.merge_subsequent_text_fields converted to pd) and returns list of ProtoCoupon objects
    :param frame: dataftame with data
    :param label_col: name of column containing serialized json mapping from texts to labels
    :param timestamp_col: name of column with timestamp in which element has been seen
    """
    assert label_col in frame.columns
    assert timestamp_col in frame.columns
    res = []
    frame = frame[frame[timestamp_col] > 0]
    for ts, subframe in frame.groupby(timestamp_col):
        leafs = tt.get_leafs(subframe)
        children_no = tt.get_children_counts(subframe)
        label_sets = {ix: MultiSet(json.loads(frame[label_col][ix]).values()) for ix in frame.index}
        texts = {ix: json.loads(frame[label_col][ix]) for ix in frame.index}
        while leafs:
            leaf = leafs.pop()
            if is_label_set_coupon(label_sets[leaf]):
                coupon_info = defaultdict(list)
                for txt, lbl in texts[leaf].items():
                    if lbl != Label.UNKNOWN:
                        coupon_info[lbl].append(txt)
                assert len(coupon_info['Label.PRODUCT_NAME']) == 1
                res.append(ProtoCoupon(
                    product_name=coupon_info['Label.PRODUCT_NAME'][0],
                    prices=coupon_info['Label.PRICE'],
                    percents=coupon_info['Label.OTHER_DISCOUNT'],
                    other_discounts=coupon_info['Label.OTHER_DISCOUNT'],
                    dates=coupon_info['Label.DATE'],
                ))
                continue
            parent = tt.find_parent(subframe, leaf)

            if parent is not None:
                parent = int(parent)
                children_no[parent] -= 1
                label_sets[parent].union(label_sets[leaf])
                texts[parent].update(texts[leaf])
                if children_no[parent] == 0:
                    leafs.append(parent)
    return res


if __name__ == '__main__':
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', None)

    df = pd.read_csv('rossmann_final.csv')
    print(proto_coupons_from_frame(df))
