from bisect import bisect_left

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

    def __eq__(self, other_id):
        return self.row_id == other_id

    def __lt__(self, other_id):
        return self.row_id < other_id


class SpanData:
    def __init__(self):
        self.beg:int  = -1
        self.end: int = -1

    def length(self):
        return self.end - self.beg + 1


class Curriculer:
    def __init__(self, dataset: Dataset, splits_amount: int):
        self.__dataset = dataset
        self.__rows_with_c = []
        self.__rows_without_c = []
        self.__splits_amount = splits_amount
        self.__splits_iter = 0
        self.__LABELS = datasets.ClassLabel(names=['B-COUPON', 'I-COUPON', 'O'])

        if self.__splits_amount <= self.__splits_iter:
            raise ValueError("Splits amount must be greater than zero")

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

    def __create_dataset(self) -> DatasetDict:
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
            texts, labels, test_size=0.2, random_state=42
        )

        # Split temp set into validation (10%) and test (10%)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
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

    def create_init_dataset(self) -> DatasetDict:
        for row in self.__rows_with_c:
            total_l = 0
            for i in range(len(row.spans)):
                total_l += row.spans[i].length()
            extend_spans(row.spans, total_l, row.max_size)
        return self.__create_dataset()

    def yield_dataset(self):
        #if self.__splits_iter > self.__splits_amount:
         #   raise StopIteration # Seems stupid, but I like it
        init_new_dataset_len = len(self.__rows_with_c)
        max_len = len(self.__dataset['train']) + len(self.__dataset['validation']) + len(self.__dataset['test'])
        if self.__splits_iter % 2 == 1 and len(self.__rows_with_c) != max_len:
            upper_append_limit = int((init_new_dataset_len + self.__splits_amount - 1) / self.__splits_amount)
            self.__rows_with_c.extend(self.__rows_without_c[:min(upper_append_limit, len(self.__rows_without_c))])
            self.__rows_without_c = self.__rows_without_c[min(upper_append_limit, len(self.__rows_without_c)):]
        else:
            for row in self.__rows_with_c:
                idx = binary_search(self.__rows_without_c, row.row_id)
                if idx:
                    total_l = int((self.__rows_with_c[idx].max_size + self.__splits_amount - 1) / self.__splits_amount)
                    extend_spans(row.spans, total_l, row.max_size)
        self.__splits_iter += 1
        return self.__create_dataset()



def binary_search(rows: list[RowData], target_value):
    # Find the index where the target_value *should* be
    index = bisect_left(rows, target_value)

    # Check if the index is valid and actually matches the target value
    if index < len(rows) and rows[index].row_id == target_value:
        return index  # Return the found object's index
    return None  # Not found

def extend_spans(spans: list, extend_amount: int, max_len: int) -> None:
    spans_count = len(spans)
    # newly added row without any spans.
    # Let's just divide this bad boy into equal parts
    if spans_count == 0:
        spans.append(SpanData())
        spans[0].beg = 0
        spans[0].end = extend_amount
        return
    spans_len = extend_amount
    l = extend_amount / spans_count
    for k in range(spans_count):
        l = extend_amount / (spans_count - k)
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
    while extend_amount > 0 and k < spans_count:
        beg = spans[k].beg
        end = spans[k].end
        if k == 0:
            spans[k].beg = max(0, spans[k].beg - extend_amount)
        else:
            spans[k].beg = max(spans[k - 1].end + 1, spans[k].beg - extend_amount)
        extend_amount -= beg - spans[k].beg
        if k == spans_count - 1:
            spans[k].end = min(len(spans) - 1, spans[k].end + extend_amount)
        else:
            spans[k].end = min(spans[k + 1].beg - 1, spans[k].end + extend_amount)
        extend_amount -= spans[k].end - end
        k += 1


dpl = load_dataset('zpp-murmuras/bert_second_pass_pl', token='')
print(dpl)
CURRICULERPL = Curriculer(dpl, 10)
for i in range(112):
    pass
    print(CURRICULERPL.yield_dataset())

