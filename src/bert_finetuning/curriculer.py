from bisect import bisect_left
from datetime import time

import datasets
from datasets import load_dataset, Dataset, DatasetDict
from functools import total_ordering

from sklearn.model_selection import train_test_split

@total_ordering
class RowData:
    def __init__(self):
        self.row_id: int = -1
        self.init_size: int = -1
        self.max_size: int = -1
        self.spans: list[SpanData] = []

    def __eq__(self, other_id: int):
        return self.row_id == other_id

    def __lt__(self, other_id: int):
        return self.row_id < other_id


class SpanData:
    def __init__(self):
        self.beg:int  = -1
        self.end: int = -1

    def length(self):
        return self.end - self.beg + 1


class Curriculer:
    def __init__(self, dataset: Dataset, splits_amount: int):
        """
        :param dataset: Dataset object with 'train', 'validation' and 'test' keys
        :param splits_amount: Amount of splits to perform
        """
        self.__dataset = dataset
        self.__rows_with_c = []
        self.__rows_without_c = []
        self.__splits_amount = splits_amount
        self.__splits_iter = 0
        self.__LABELS = datasets.ClassLabel(names=['B-COUPON', 'I-COUPON', 'O'])
        self.__init_len = -1

        if self.__splits_amount <= self.__splits_iter:
            raise ValueError("Splits amount must be greater than zero")

        """
        Constructor is responsible for the initial data processing;
        It separates rows that contain coupons and those that don't,
        and then creates spans for each row that contains coupons.
        Spans represent the continuous sequence of coupon labels (so basically a coupon;
        but, later on, it contans a coupon and its context).
        """
        row_iter = 0
        for col in ['train', 'validation', 'test']:
            for row in self.__dataset[col]:
                c = RowData()
                c.row_id = row_iter
                c.max_size = len(row['labels'])
                c.init_size = 0
                labels_iter = 0
                curr_span = None
                for label in row['labels']:
                    if label == 0:
                        if curr_span and curr_span.beg >= 0:
                            curr_span.end = int(labels_iter) - 1
                            c.spans.append(curr_span)
                            curr_span = None
                    else:
                        if label == 1:
                            if curr_span:
                                curr_span.end = int(labels_iter) - 1
                                c.spans.append(curr_span)
                            curr_span = SpanData()
                            curr_span.beg = int(labels_iter)
                        c.init_size += 1
                    labels_iter += 1

                if curr_span and curr_span.beg >= 0:
                    curr_span.end = labels_iter - 1
                    c.spans.append(curr_span)

                if c.init_size > 0:
                    self.__rows_with_c.append(c)
                else:
                    self.__rows_without_c.append(c)
                row_iter += 1

    def __create_dataset(self, tv_split: float, tt_split: float) -> DatasetDict:
        """
        Function that creates a dataset from the current state of the object.
        What it means is that this function goes through the rows that contain coupons,
        and extracts labels that are contained within the spans of each row.

        :param tv_split: Train/Validation split ratio
        :param tt_split: Train/Test split ratio
        :return: DatasetDict object with 'train', 'validation' and 'test' keys
        """
        labels = []
        texts = []
        train_len = len(self.__dataset['train'])
        validation_len = len(self.__dataset['validation'])
        for row in self.__rows_with_c:
            labels_list = []
            texts_list = []
            rid = int(row.row_id)
            col = 'train'
            if row.row_id >= train_len + validation_len:
                rid -= train_len + validation_len
                col = 'test'
            elif row.row_id >= train_len:
                rid -= train_len
                col = 'validation'
            for span in row.spans:
                labels_list.extend(self.__dataset[col][rid]['labels'][int(span.beg):int(span.end) + 1])
                texts_list.extend(self.__dataset[col][rid]['texts'][int(span.beg):int(span.end) + 1])
            labels.append(labels_list)
            texts.append(texts_list)

        features = datasets.Features({
            "texts": datasets.Sequence(datasets.Value("string")),
            "labels": datasets.Sequence(self.__LABELS)
        })

        # Initial train/test split (80% train, 20% temp)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=tv_split, random_state=42
        )

        # Split temp set into validation (10%) and test (10%)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=tt_split, random_state=42
        )

        dataset_dict = {
            'train': {'texts': train_texts, 'labels': train_labels},
            'validation': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }

        return DatasetDict({
            split: Dataset.from_dict(data, features=features)
            for split, data in dataset_dict.items()
        })

    def create_init_dataset(self, tv_split: float = 0.2, tt_split: float = 0.5) -> DatasetDict:
        """
        Function that creates the initial dataset. It envolves
        balancing the coupon and non-coupon classes on the per-row basis.
        For now, rows without coupons are excluded.

        :param tv_split: Train/Validation split ratio
        :param tt_split: Train/Test split ratio
        :return: DatasetDict object with 'train', 'validation' and 'test' keys
        """
        for row in self.__rows_with_c:
            total_l = 0
            for i in range(len(row.spans)):
                total_l += row.spans[i].length()
            extend_spans(row.spans, total_l, row.max_size)
        self.__init_len = len(self.__rows_with_c)
        return self.__create_dataset(tv_split, tt_split)

    def yield_dataset(self, tv_split: float = 0.2, tt_split: float = 0.5) -> DatasetDict:
        """
        Main generation logic. Depending on whether the current iteration is even or odd,
        the function either extends the spans of the rows that contain coupons, or
        adds new rows that does not contain coupons.

        :param tv_split: Train/Validation split ratio
        :param tt_split: Train/Test split ratio
        :return: DatasetDict object with 'train', 'validation' and 'test' keys
        """
        if self.__splits_iter >= self.__splits_amount:
            raise StopIteration # Seems stupid, but I like it
        max_len = len(self.__dataset['train']) + len(self.__dataset['validation']) + len(self.__dataset['test'])
        if self.__splits_iter == self.__splits_amount - 1:
            self.__rows_with_c.extend(self.__rows_without_c)
            self.__rows_without_c = []
        if self.__splits_iter % 2 == 0 and len(self.__rows_with_c) != max_len:
            upper_append_limit = int((self.__init_len + self.__splits_amount - 1) / self.__splits_amount)
            self.__rows_with_c.extend(self.__rows_without_c[:min(upper_append_limit, len(self.__rows_without_c))])
            self.__rows_without_c = self.__rows_without_c[min(upper_append_limit, len(self.__rows_without_c)):]
        else:
            for row in self.__rows_with_c:
                idx = binary_search(self.__rows_without_c, row.row_id)
                if idx:
                    total_l = int((self.__rows_with_c[idx].max_size + self.__splits_amount - 1) / self.__splits_amount)
                    extend_spans(row.spans, total_l, row.max_size)
        self.__splits_iter += 1
        return self.__create_dataset(tv_split, tt_split)



def binary_search(rows: list[RowData], target_value: int) -> int or None:
    """
    Binary search implementation for the list of RowData objects.

    :param rows: List of RowData objects
    :param target_value: Target value to search for
    :return: Index of the target value in the list, or None if not found
    """
    # Find the index where the target_value *should* be
    index = bisect_left(rows, target_value)

    # Check if the index is valid and actually matches the target value
    if index < len(rows) and rows[index].row_id == target_value:
        return index  # Return the found object's index
    return None  # Not found

def extend_spans(spans: list, extend_amount: int, max_len: int):
    """
    Function that extends the spans of the row that contains coupons.
    It tries to distribute the extend_amount evenly between the spans,
    and if it's not possible, it greedily extends the spans.

    :param spans: List of SpanData objects
    :param extend_amount: Amount of extension
    :param max_len: Maximum length of the row
    """
    spans_count = len(spans)
    # newly added row without any spans.
    # Let's just divide this bad boy into equal parts
    if spans_count == 0:
        spans.append(SpanData())
        spans[0].beg = 0
        spans[0].end = extend_amount
        return
    spans_len = extend_amount
    l = int(extend_amount / spans_count)
    for k in range(spans_count):
        l = int(extend_amount / (spans_count - k))
        left_l = int((l + 1) / 2)
        right_l = l - left_l
        beg = spans[k].beg
        end = spans[k].end
        if k == 0:
            spans[k].beg = max(0, beg - left_l)
        else:
            spans[k].beg = max(spans[k - 1].end + 1, beg - left_l)
        if k == spans_count - 1:
            spans[k].end = min(max_len - 1, end + right_l)
        else:
            spans[k].end = min(spans[k + 1].beg - 1, end + right_l)
        extend_amount -= (beg - spans[k].beg) + (spans[k].end - end)
        spans_len += (beg - spans[k].beg) + (spans[k].end - end)
    k = 0
    while extend_amount > 0 and k < spans_count and spans_len < max_len:
        beg = spans[k].beg
        end = spans[k].end
        if k == 0:
            spans[k].beg = max(0, spans[k].beg - extend_amount)
        else:
            spans[k].beg = max(spans[k - 1].end + 1, spans[k].beg - extend_amount)
        extend_amount -= beg - spans[k].beg
        if k == spans_count - 1:
            spans[k].end = min(max_len - 1, spans[k].end + extend_amount)
        else:
            spans[k].end = min(spans[k + 1].beg - 1, spans[k].end + extend_amount)
        extend_amount -= spans[k].end - end
        k += 1

