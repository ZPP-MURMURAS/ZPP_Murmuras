import pytest
from datasets import Dataset

from src.bert_finetuning.curriculer import Curriculer

"""
This test is based on one dataset. This dataset is supposed to represent multiple cases.
Because the curriculer is a class, then we can inspect its internal state
at any moment of the test.
"""

class TestFinetuner:
    curriculer: Curriculer
    test_data: Dataset

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.test_data = Dataset.from_dict({
            'train': [
                {
                    'texts': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
                    'labels': [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 2, 0]
                },
                {
                    'texts': ['1', '2', '3', '4', '5'],
                    'labels': [0, 0, 0, 0, 0]
                },
                {
                    'texts': ['1', '2', '3', '4', '5', '6', '7'],
                    'labels': [0, 0, 0, 0, 0, 0, 0]
                },
                {
                    'texts': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'],
                    'labels': [0, 1, 2, 2, 2, 1, 0, 1, 2, 0, 1, 2]
                },
                {
                    'texts': ['1', '2', '3', '4' '5', '6'],
                    'labels': [1, 2, 2, 2, 2, 3]
                }

            ],
            'validation': [
                {
                    'texts': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
                    'labels': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
                },
                {
                    'texts': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'],
                    'labels': [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0]
                }
            ],
            'test': [
                {
                    'texts': ['1', '2', '3'],
                    'labels': [0, 0, 0]
                },
                {
                    'texts': ['1', '2', '3', '4'],
                    'labels': [0, 1, 2, 0]
                }
            ]
        })
        print(self.test_data)
        self.curriculer = Curriculer(self.test_data, 2)
        yield

    def test_curriculer_init(self):


