from bisect import bisect_left
from datasets import load_dataset
from functools import total_ordering


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


def binary_search(rows: list[RowData], target_value):
    # Find the index where the target_value *should* be
    index = bisect_left(rows, target_value)

    # Check if the index is valid and actually matches the target value
    if index < len(rows) and rows[index].row_id == target_value:
        return index  # Return the found object's index
    return None  # Not found


data = load_dataset("zpp-murmuras/coupon_select_plain_text", token='')

rows_wth_c = []
rows_wthout_c = []
rows_wth_counterparts = []

row_iter = 0
for row in data['train']:
    c = RowData()
    nc = RowData()
    c.row_id = row_iter
    nc.row_id = row_iter
    labels_iter = 0
    prev_label = -1
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
        rows_wth_c.append(c)
        nc.counterpart = True
        rows_wth_counterparts.append(nc)
    if nc.init_size > 0:
        rows_wthout_c.append(nc)
    if c.init_size == 0 and nc.init_size == 0:
        pass
    row_iter += 1

print("Dataset train length: ", len(data['train']))
print("List with coupon length: ", len(rows_wth_c))
print("List without coupon length: ", len(rows_wthout_c))


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


ych = 0
for row in rows_wth_c:
    ych += 1
    total_l = 0
    for i in range(len(row.spans)):
        total_l += row.spans[i].length()
    extend_spans(row.spans, total_l)

print("New dataset length: ", len(rows_wth_c))

epochs = 100
init_new_dataset_len = len(rows_wth_c)
for i in range(int(epochs / 10)):
    if i % 2 == 10 and len(rows_wth_c) != len(data['train']):
        prev_row = None
        modified_dataset = []
        upper_append_limit = int((init_new_dataset_len + 9) / 10)
        rows_wth_c.extend(rows_wth_counterparts[:min(upper_append_limit, len(rows_wth_counterparts))])
        rows_wth_counterparts = rows_wth_counterparts[min(upper_append_limit, len(rows_wth_counterparts)):]
    else:  # extend everything
        # 1. find non-coupon with our row id
        # 2. if not empty, then sample 1/10 of the init size rounded up
        # 3. add to new_dataset
        # 4. remove from rows_wthout_c
        for row in rows_wth_c:
            idx = binary_search(rows_wthout_c, row.row_id)
            if idx and len(rows_wthout_c[idx].labels) > 0:
                total_l = int((rows_wthout_c[idx].init_size + 9) / 10)
                extend_spans(row.spans, total_l)
