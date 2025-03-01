from bisect import bisect_left
from datasets import load_dataset, Dataset
from functools import total_ordering

from sklearn.model_selection import train_test_split


@total_ordering
class RowData:
    def __init__(self):
        self.row_id = -1
        self.init_size = -1
        self.labels = []
        self.spans = []

    def __eq__(self, other_id):
        return self.row_id == other_id

    def __lt__(self, other_id):
        return self.row_id < other_id


class SpanData:
    def __init__(self):
        self.beg = -1
        self.end = -1

    def length(self):
        return self.end - self.beg + 1


class Curriculer:
    def __init__(self, dataset: Dataset, splits_amount: int):
        self.__dataset = dataset
        self.__rows_with_c = []
        self.__rows_without_c = []
        self.__rows_with_counterparts = []
        self.__splits_amount = splits_amount
        self.__splits_iter = 0

        if self.__splits_amount <= self.__splits_iter:
            raise ValueError("Splits amount must be greater than zero")

        row_iter = 0
        for col in ['train', 'validation', 'test']:
            for row in self.__dataset[col]:
                c = RowData()
                nc = RowData()
                c.row_id = row_iter
                nc.row_id = row_iter
                labels_iter = 0
                curr_span = None
                for label in row['labels']:
                    if label == 0:
                        nc.labels.append(labels_iter)
                        if curr_span and curr_span.beg >= 0:
                            curr_span.end = labels_iter
                            c.spans.append(curr_span)
                            curr_span = None
                    else:
                        if label == 1:
                            if curr_span:
                                curr_span.end = labels_iter
                                c.spans.append(curr_span)
                            curr_span = SpanData()
                            curr_span.beg = labels_iter
                        c.labels.append(labels_iter)
                    labels_iter += 1
                c.init_size = len(c.labels)
                nc.init_size = len(nc.labels)
                if c.init_size > 0:
                    self.__rows_with_c.append(c)
                    nc.counterpart = True
                    self.__rows_with_counterparts.append(nc)
                if nc.init_size > 0:
                    self.__rows_without_c.append(nc)
                if c.init_size == 0 and nc.init_size == 0:
                    pass
                row_iter += 1

    def __create_dataset(self) -> Dataset:
        labels = []
        texts = []
        for col in ['train', 'validation', 'test']:
            for row in self.__rows_with_c:
                labels_list = []
                texts_list = []
                for span in row.spans:
                    labels_list.extend(self.__dataset[col][row.row_id]['labels'][span.beg:span.end + 1])
                    texts_list.extend(self.__dataset[col][row.row_id]['texts'][span.beg:span.end + 1])
                labels.append(labels_list)
                texts.append(texts_list)

        # Initial train/test split (80% train, 20% temp)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Split temp set into validation (10%) and test (10%)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )

        return Dataset.from_dict({
            'train': {'texts': train_texts, 'labels': train_labels},
            'validation': {'texts': val_texts, 'labels': val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        })

    def create_init_dataset(self) -> Dataset:
        for row in self.__rows_with_c:
            total_l = 0
            for i in range(len(row.spans)):
                total_l += row.spans[i].length()
            extend_spans(row.spans, total_l)
        return self.__create_dataset()

    def yield_dataset(self):
        if self.__splits_iter >= self.__splits_amount:
            raise StopIteration # Seems stupid, but I like it
        init_new_dataset_len = len(self.__rows_with_c)
        if self.__splits_iter % 2 == 10 and len(self.__rows_with_c) != len(self.__dataset['train']):
            upper_append_limit = int((init_new_dataset_len + 9) / 10)
            self.__rows_with_c.extend(self.__rows_with_counterparts[:min(upper_append_limit, len(self.__rows_with_counterparts))])
            self.__rows_with_counterparts = self.__rows_with_counterparts[min(upper_append_limit, len(self.__rows_with_counterparts)):]
        else:
            for row in self.__rows_with_c:
                idx = binary_search(self.__rows_without_c, row.row_id)
                if idx and len(self.__rows_without_c[idx].labels) > 0:
                    total_l = int((self.__rows_without_c[idx].init_size + 9) / 10)
                    extend_spans(row.spans, total_l)
        self.__splits_iter += 1
        return self.__create_dataset()



def binary_search(rows: list[RowData], target_value):
    # Find the index where the target_value *should* be
    index = bisect_left(rows, target_value)

    # Check if the index is valid and actually matches the target value
    if index < len(rows) and rows[index].row_id == target_value:
        return index  # Return the found object's index
    return None  # Not found

def extend_spans(spans: list, extend_amount: int) -> None:
    nmb_l = len(spans)
    spans_len = extend_amount
    for k in range(nmb_l):
        l = extend_amount / nmb_l
        left_l = int((l + 1) / 2)
        right_l = l - left_l
        beg = spans[k].beg
        end = spans[k].end
        if k == 0:
            spans[k].beg = max(0, beg - left_l)
        else:
            spans[k].beg = max(spans[k - 1].end + 1, beg - left_l)
        if k == nmb_l - 1:
            spans[k].end = min(len(spans) - 1, end + right_l)
        else:
            spans[k].end = min(spans[k + 1].beg - 1, end + right_l)
        extend_amount -= (beg - spans[k].beg) + (spans[k].end - end)
        spans_len += (beg - spans[k].beg) + (spans[k].end - end)
    k = 0
    while extend_amount > 0 and k < nmb_l:
        beg = spans[k].beg
        end = spans[k].end
        if k == 0:
            spans[k].beg = max(0, spans[k].beg - extend_amount)
        else:
            spans[k].beg = max(spans[k - 1].end + 1, spans[k].beg - extend_amount)
        extend_amount -= beg - spans[k].beg
        if k == nmb_l - 1:
            spans[k].end = min(len(spans) - 1, spans[k].end + extend_amount)
        else:
            spans[k].end = min(spans[k + 1].beg - 1, spans[k].end + extend_amount)
        extend_amount -= spans[k].end - end
        k += 1

