import re
import json
from collections import defaultdict
from collections.abc import MutableSet
from typing import List

import pandas as pd

from src.constants import *


def labeled_data_to_extra_csv_column(content_frame: pd.DataFrame, coupons_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Function for early testing of postprocessing.
    Having frame with phone screen content and frame with associated coupons split
    text views to rows containing single word each  and new column with label
    assigned to word. At this moment any form of quantitative description of
    discount (price, percentage, '2 in price of 1') are labeled as price as this
    does not require extra processing of data and these values are treated by
    splitting algorithm as equivalent. Dates are currently recognised in very primitive
    manner, but ig for our purposes it is enough.
    :param content_frame: frame with encoded xml of phone screen content
    :param coupons_frame: frame with discount coupons present in screen content
    """

    names = coupons_frame["product_text"].dropna().tolist()
    discount_texts = coupons_frame["discount_text"].dropna().tolist()

    labels = [Label.UNKNOWN if isinstance(txt, float)
              else Label.PRODUCT_NAME if txt in names
              else Label.PRICE if txt in discount_texts
              else Label.DATE if re.search("[0-9][-./\\][0-9][0-9]", txt) is not None
              else Label.UNKNOWN
              for txt in content_frame["text"]]

    labels = [str(lbl) for lbl in labels]

    content_frame = content_frame.copy(deep=True)
    content_frame[LABEL_COLUMN] = labels
    content_frame = content_frame.assign(text=content_frame["text"].str.split()).explode("text")

    return content_frame


def merge_subsequent_text_fields(labeled_content_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Takes csv with split texts and concatenates ones coming from single textfield. under labels,
    it produces serialized json representing assignment of tokens
    :param labeled_content_frame: frame with screen content and split text fields alongside with labels
    """
    labeled_content_frame = labeled_content_frame.copy(deep=True)
    labeled_content_frame["label"] += ' '
    labeled_content_frame["label"] += labeled_content_frame['text']
    aggregators = {col: lambda x: x.iloc[0] for col in labeled_content_frame.columns}
    aggregators["text"] = lambda x: ' '.join(x) if not x.isna().any() else x
    def strs_labels_to_dict(x: List[str]) -> dict:
        res = {}
        if not len(x):
            return {}
        curr_label = None
        curr_text = []
        for string in x:
            lbl, txt = string.split()
            if lbl != curr_label:
                if curr_label is not None:
                    res[' '.join(curr_text)] = curr_label
                curr_label = lbl
                curr_text = []
            curr_text.append(txt)
        res[' '.join(curr_text)] = curr_label
        return res
    aggregators[LABEL_COLUMN] = lambda x: json.dumps(strs_labels_to_dict(x)) if not x.isna().any() else "{}"
    labeled_content_frame = labeled_content_frame.groupby(["id", "i"], as_index=False).agg(aggregators)
    return labeled_content_frame


class MultiSet(MutableSet):
    """
    simple implementation of multiset (ofc credits to chatgpt)
    """
    def __init__(self, iterable=None):
        self._elements = defaultdict(int)
        if iterable is not None:
            for item in iterable:
                self.add(item)

    def __contains__(self, item):
        return self._elements[item] > 0

    def __iter__(self):
        for item, count in self._elements.items():
            for _ in range(count):
                yield item

    def __len__(self):
        return sum(self._elements.values())

    def add(self, item):
        self._elements[item] += 1

    def discard(self, item):
        if self._elements[item] > 0:
            self._elements[item] -= 1
            if self._elements[item] == 0:
                del self._elements[item]

    def __repr__(self):
        return f"Multiset({list(self)})"

    def count(self, item):
        """Return the count of the item in the multiset."""
        return self._elements[item]

    def clear(self):
        """Remove all elements from the multiset."""
        self._elements.clear()

    def items(self):
        """Return a dictionary-like view of items and their counts."""
        return self._elements.items()

    def __eq__(self, other):
        if not isinstance(other, MultiSet):
            return False
        return self._elements == other._elements

    def union(self, other: 'MultiSet') -> None:
        for x in other:
            self.add(x)
