import datasets
import pytest
from datasets import Dataset, DatasetDict

from src.bert_finetuning.curriculer import Curriculer


class TestFinetuner:
    curriculer: Curriculer
    test_data: Dataset
    init_test_data: Dataset
    init_c = {}
    init_non_c = []
    init_spans_dims: dict
    splits = 5
    max_dataset_len: int
    __LABELS = datasets.ClassLabel(names=['B-COUPON', 'I-COUPON', 'O'])

    def test_curriculer_init(self):
        # Check general sizes
        assert len(self.curriculer._Curriculer__rows_with_c) == len(self.init_c)
        assert len(self.curriculer._Curriculer__rows_without_c) == len(self.init_non_c)

        # Check indexes, spans and sizes
        for i in range(len(self.init_non_c)):
            assert self.curriculer._Curriculer__rows_without_c[i].row_id == self.init_non_c[i]
        keys = list(self.init_c.keys())
        for i in range(len(self.init_c)):
            k = keys[i]
            assert self.curriculer._Curriculer__rows_with_c[i].row_id == k
            for j in range(len(self.init_c[k])):
                assert len(self.curriculer._Curriculer__rows_with_c[i].spans) == len(self.init_c[k])
                assert self.curriculer._Curriculer__rows_with_c[i].spans[j].beg == self.init_c[k][j][0]
                assert self.curriculer._Curriculer__rows_with_c[i].spans[j].end == self.init_c[k][j][1]
                assert self.curriculer._Curriculer__rows_with_c[i].spans[j].length() == self.init_c[k][j][2]

    def test_init_dataset(self):
        new_dataset = self.curriculer.create_init_dataset()
        # Inspect the validity of internal state (spans)
        for row in self.curriculer._Curriculer__rows_with_c:
            assert len(self.init_spans_dims[row.row_id]) == len(row.spans)
            for i in range(len(row.spans)):
                assert row.spans[i].beg == self.init_spans_dims[row.row_id][i][0]
                assert row.spans[i].end == self.init_spans_dims[row.row_id][i][1]
                assert row.spans[i].length() == self.init_spans_dims[row.row_id][i][2]

        # test dataset regeneration
        new_texts = []
        new_labels = []
        assert len(new_dataset) == len(self.init_test_data)
        for item in new_dataset:
            new_texts.append(item['texts'])
            new_labels.append(item['labels'])
        for item in self.init_test_data:
            t = item['texts']
            l = item['labels']
            assert t in new_texts
            assert l in new_labels
            new_texts.remove(t)
            new_labels.remove(l)

    def test_data_yielder(self):
        dt = self.curriculer.create_init_dataset()
        it = 0
        assert self.curriculer._Curriculer__splits_iter == it
        desired_c = len(self.init_c)
        desired_non_c = len(self.init_non_c)
        assert self.curriculer._Curriculer__init_len == desired_c
        assert len(self.curriculer._Curriculer__rows_with_c) == desired_c
        assert len(self.curriculer._Curriculer__rows_without_c) == desired_non_c
        assert len(dt) == desired_c
        desired_cs = [8, 8, 9, 9, 9]
        for i in range(self.splits):
            it += 1
            dt = self.curriculer.yield_dataset(shuffle=False)
            assert self.curriculer._Curriculer__splits_iter == it
            if i % 2 == 0:
                desired_c += 1
                desired_non_c -= 1
            if i == self.splits - 1:
                desired_c = self.max_dataset_len
                desired_non_c = 0
            assert len(self.curriculer._Curriculer__rows_with_c) == desired_cs[i]
            assert len(self.curriculer._Curriculer__rows_without_c) == 9 - desired_cs[i]
            assert len(dt) == desired_cs[i]

        len_test_texts = 0
        len_test_labels = 0
        let_dt_texts = 0
        let_dt_labels = 0
        for i in range(len(self.test_data)):
            len_test_texts += len(self.test_data[i]['texts'])
            len_test_labels += len(self.test_data[i]['labels'])
            let_dt_texts += len(dt[i]['texts'])
            let_dt_labels += len(dt[i]['labels'])
        assert len_test_texts == let_dt_texts
        assert len_test_labels == let_dt_labels

        with pytest.raises(StopIteration):
            self.curriculer.yield_dataset()


    @pytest.fixture(autouse=True)
    def setup_method(self):
        features = datasets.Features({
            "texts": datasets.Sequence(datasets.Value("string")),
            "labels": datasets.Sequence(self.__LABELS)
        })

        self.init_spans_dims = {
            0: [(0, 3, 4), (4, 6, 3), (7, 9, 3), (10, 13, 4)],
            3: [(0, 4, 5), (5, 6, 2), (7, 9, 3), (10, 11, 2)],
            4: [(0, 5, 6)],
            6: [(0, 3, 4), (9, 12, 4)],
            8: [(0, 3, 4)],
        }

        texts = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3', '4', '5'],
            ['1', '2', '3', '4', '5', '6', '7'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            ['1', '2', '3', '4', '5', '6'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3'],
            ['1', '2', '3', '4']
        ]
        labels = [
            [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 2],
            [1, 2, 2, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0],
            [0, 0, 0],
            [0, 0, 1, 2]
        ]
        init_texts = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            ['1', '2', '3', '4', '5', '6'],
            ['1', '2', '3', '4', '10', '11', '12', '13'],
            ['1', '2', '3', '4']
        ]
        init_labels = [
            [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0],
            [0, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 2],
            [1, 2, 2, 2, 2, 2],
            [0, 1, 2, 0, 0, 1, 2, 0],
            [0, 0, 1, 2]
        ]

        dataset_dict = {
            'texts': texts,
            'labels': labels,
        }
        dataset_init_dict = {
            'texts': init_texts,
            'labels': init_labels,
        }

        self.test_data = Dataset.from_dict(dataset_dict, features=features)
        self.max_dataset_len = len(self.test_data)
        self.curriculer = Curriculer(self.test_data, self.splits)
        self.init_test_data = Dataset.from_dict(dataset_init_dict, features=features)

        self.init_c = {
             0: [(1, 2, 2), (5,5, 1), (7, 8, 2), (10, 12, 3)],
             3: [(1, 4, 4), (5, 5, 1), (7, 8, 2), (10, 11, 2)],
             4: [(0, 5, 6)],
             6: [(1, 2, 2), (10, 11, 2)],
             8: [(2, 3, 2)],
        }

        self.init_non_c = [1, 2, 5, 7]
        yield
