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
        new_dataset = self.curriculer.create_init_dataset(tv_split= 2.0/5.0, tt_split=0.5)
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
        for col in ['train', 'validation', 'test']:
            assert len(new_dataset[col]) == len(self.init_test_data[col])
            for i in range(len(new_dataset[col])):
                new_texts.append(new_dataset[col][i]['texts'])
                new_labels.append(new_dataset[col][i]['labels'])
        for col in ['train', 'validation', 'test']:
            for i in range(len(self.init_test_data[col])):
                t = self.init_test_data[col][i]['texts']
                l = self.init_test_data[col][i]['labels']
                assert t in new_texts
                assert l in new_labels
                new_texts.remove(t)
                new_labels.remove(l)

    def test_data_yielder(self):
        dt = self.curriculer.create_init_dataset(tv_split=2.0 / 5.0, tt_split=0.5)
        it = 0
        assert self.curriculer._Curriculer__splits_iter == it
        desired_c = len(self.init_c)
        desired_non_c = len(self.init_non_c)
        assert self.curriculer._Curriculer__init_len == desired_c
        assert len(self.curriculer._Curriculer__rows_with_c) == desired_c
        assert len(self.curriculer._Curriculer__rows_without_c) == desired_non_c
        assert len(dt['train']) + len(dt['validation']) + len(dt['test']) == desired_c
        for i in range(self.splits):
            it += 1
            dt = self.curriculer.yield_dataset(tv_split= 2.0/5.0, tt_split=0.5, shuffle=False)
            assert self.curriculer._Curriculer__splits_iter == it
            if i % 2 == 0:
                desired_c += 1
                desired_non_c -= 1
            if i == self.splits - 1:
                desired_c = self.max_dataset_len
                desired_non_c = 0
            assert len(self.curriculer._Curriculer__rows_with_c) == desired_c
            assert len(self.curriculer._Curriculer__rows_without_c) == desired_non_c
            assert len(dt['train']) + len(dt['validation']) + len(dt['test']) == desired_c

        with pytest.raises(StopIteration):
            self.curriculer.yield_dataset(tv_split= 2.0/5.0, tt_split=0.5)


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

        train_texts = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3', '4', '5'],
            ['1', '2', '3', '4', '5', '6', '7'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            ['1', '2', '3', '4', '5', '6'],
        ]
        train_labels = [
            [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 2],
            [1, 2, 2, 2, 2, 2]
        ]
        train_init_texts = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
            ['1', '2', '3', '4', '5', '6'],
        ]
        train_init_labels = [
            [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0],
            [0, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 2],
            [1, 2, 2, 2, 2, 2]
        ]

        validation_texts = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14']
        ]
        validation_labels = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]
        ]
        validation_init_texts = [
            ['1', '2', '3', '4', '10', '11', '12', '13'],
        ]
        validation_init_labels = [
            [0, 1, 2, 0, 0, 1, 2, 0]
        ]

        test_texts = [
            ['1', '2', '3'],
            ['1', '2', '3', '4']
        ]
        test_labels = [
            [0, 0, 0],
            [0, 0, 1, 2]
        ]
        test_init_texts = [
            ['1', '2', '3', '4']
        ]
        test_init_labels = [
            [0, 0, 1, 2]
        ]

        dataset_dict = {
            'train': {'texts': train_texts, 'labels': train_labels},
            'validation': {'texts': validation_texts, 'labels': validation_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        }
        dataset_init_dict = {
            'train': {'texts': train_init_texts, 'labels': train_init_labels},
            'validation': {'texts': validation_init_texts, 'labels': validation_init_labels},
            'test': {'texts': test_init_texts, 'labels': test_init_labels}
        }

        self.test_data = DatasetDict({
            split: Dataset.from_dict(data, features=features)
            for split, data in dataset_dict.items()
        })
        self.max_dataset_len = len(self.test_data['train']) + len(self.test_data['validation']) + len(self.test_data['test'])
        self.curriculer = Curriculer(self.test_data, self.splits)
        self.init_test_data = DatasetDict({
            split: Dataset.from_dict(data, features=features)
            for split, data in dataset_init_dict.items()
        })

        self.init_c = {
             0: [(1, 2, 2), (5,5, 1), (7, 8, 2), (10, 12, 3)],
             3: [(1, 4, 4), (5, 5, 1), (7, 8, 2), (10, 11, 2)],
             4: [(0, 5, 6)],
             6: [(1, 2, 2), (10, 11, 2)],
             8: [(2, 3, 2)],
        }

        self.init_non_c = [1, 2, 5, 7]
        yield
