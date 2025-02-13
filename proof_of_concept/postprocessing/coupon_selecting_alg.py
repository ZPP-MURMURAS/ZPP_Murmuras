from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from math import isnan
import json
from collections import defaultdict
import re

import pandas as pd

from src.tree_traversal import tree_traversal as tt
from src.constants import *
from src.postprocessing_utils import MultiSet


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
    images: List[str] = field(default_factory=list)


def is_label_set_coupon(labels: MultiSet) -> bool:
    """
    Checks if given set of labels represents enough text to crete ProtoCoupon object
    :param labels: multiset of labels to check
    """
    if labels.count(Label.PRODUCT_NAME) == 0:
        return False
    for lbl in [Label.PRICE, Label.OTHER_DISCOUNT, Label.PERCENT]:
        if labels.count(lbl) > 0:
            return True
    return False


def proto_coupons_from_frame(
        frame: pd.DataFrame,
        label_col: str = LABEL_COLUMN,
        timestamp_col: str = TIMESTAMP_COLUMN,
        must_have_images: bool = False,
        widget_col: str = CLASS_NAME_COLUMN,
        labels_mapping: Optional[Dict[str, Label]] = None
) -> List[ProtoCoupon]:
    """
    this function takes dataframe containing column with mapped fragments of text to tokens (ex output of
    postprocessing_utils.merge_subsequent_text_fields converted to pd) and returns list of ProtoCoupon objects
    :param frame: dataftame with data
    :param label_col: name of column containing serialized json mapping from texts to labels
    :param timestamp_col: name of column with timestamp in which element has been seen
    :param widget_col: name of column with widget type
    :param must_have_images: if true, valid coupon must have image in it
    :param labels_mapping: mapping from labels used in dataframe to Label enum
    """
    assert label_col in frame.columns
    assert timestamp_col in frame.columns

    if labels_mapping is None:
        labels_mapping = {str(lbl): lbl for lbl in Label}

    if must_have_images:
        assert widget_col in frame.columns

    res = []
    frame = frame[frame[timestamp_col] > 0]
    for ts, subframe in frame.groupby(timestamp_col):
        leafs = tt.get_leafs(subframe)
        children_no = tt.get_children_counts(subframe)
        label_sets = {}
        texts = {}
        for ix in frame.index:
            if isinstance(frame[label_col][ix], float) and isnan(frame[label_col][ix]):
                mapping = {}
            else:
                string = frame[label_col][ix]
                if not string:
                    mapping = {}
                else:
                    mapping = json.loads(frame[label_col][ix])
            for k, v in mapping.items():
                mapping[k] = labels_mapping[v]
            label_sets[ix] = MultiSet(mapping.values())
            texts[ix] = mapping

        if must_have_images:
            images = {}

            for ix in frame.index:
                images[ix] = []
                if "image" in frame[widget_col][ix].lower():
                    images[ix] = [frame[widget_col][ix]]                

        while leafs:
            leaf = leafs.pop()
            
            valid_coupon = is_label_set_coupon(label_sets[leaf]) 
            if must_have_images:
                valid_coupon = valid_coupon and len(images[leaf]) > 0

            if valid_coupon:
                coupon_info = defaultdict(list)

                for txt, lbl in texts[leaf].items():
                    if lbl != Label.UNKNOWN:
                        coupon_info[lbl].append(txt)
                res.append(ProtoCoupon(
                    product_name=coupon_info[Label.PRODUCT_NAME][0],
                    prices=coupon_info[Label.PRICE],
                    percents=coupon_info[Label.OTHER_DISCOUNT],
                    other_discounts=coupon_info[Label.OTHER_DISCOUNT],
                    dates=coupon_info[Label.DATE],
                    images=images[leaf] if must_have_images else []
                ))
                continue
            parent = tt.find_parent(subframe, leaf)

            if parent is not None:
                parent = int(parent)
                children_no[parent] -= 1
                label_sets[parent].union(label_sets[leaf])

                if must_have_images:
                    images[parent] += [image for image in images[leaf] if "image" in image.lower()]
               
                texts[parent].update(texts[leaf])
                if children_no[parent] == 0:
                    leafs.append(parent)
    return res


def select_prices(prices: list) -> tuple:
    """
    Selects the highest and lowest prices from the given list.
    If the list is empty or the types of discounts are different
    from the expected types, returns None.
    :param prices: list of candidate strings
    :returns a pair of largest and smallest prices or (None, None) if no valid price was found
    """

    def parse_price(price: Any):
        '''
        Args: price: Any: a string containing the price.
        Returns: float: the price as a float. If the price is a float or an int,
        it casts it to a float and returns it. If the price is a string, it extracts
        the numerical value from the string and returns it as a float. If the price as
        a string contains no numerical value, it returns None.
        '''

        if isinstance(price, int) or isinstance(price, float):
            return float(price)
        elif isinstance(price, str):
            # Returns the first numerical value in the string. This is a heuristic
            # and may not work for all cases. For example 100ml = 6.99 EUR.
            # We could classify what is a currency in a given context and based on that
            # extract the price.
            match = re.search(r'\d+(\.\d+)?', price)
            if match:
                return float(match.group())
        return None

    parsed_prices = [parse_price(price) for price in prices]
    filtered_prices = [price for price in parsed_prices if price is not None]

    if not filtered_prices:
        return None, None

    highest_price = max(filtered_prices)
    lowest_price = min(filtered_prices)

    return highest_price, lowest_price


'''
Args: classified_inputs: list: a list of tuples containing the classified traits 
of a coupon.
Returns: dict: a dictionary containing the classified traits of a coupon as well as 
the highest and lowest prices. The 
'''


def parse_proto_coupon(proto_coupon: ProtoCoupon) -> dict:
    highest_price, lowest_price = select_prices(proto_coupon.prices)
    return {
        'validity': proto_coupon.dates[0] if proto_coupon.dates else 'N/A',
        'discount': proto_coupon.percents[0] if proto_coupon.percents else 'N/A',
        'other_discounts': proto_coupon.other_discounts[0] if proto_coupon.other_discounts else 'N/A',
        'old_price': highest_price,
        'new_price': lowest_price,
        'product_name': proto_coupon.product_name
    }
