import os

import pandas as pd
import pytest
from datasets import Dataset

from src.llama_dataset_generation.datasetter import __parse_to_oimo as _datasetter__parse_to_oimo
from src.llama_dataset_generation.datasetter import __map_logic as _datasetter__map_logic


class TestDatasetter:
    prompt = '###Prompting... Input:\n{}\n\n### Response:\n{}'
    test_dataframe: pd.DataFrame
    expected_oimo_res: pd.DataFrame
    mapped_data: dict

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.test_dataframe = pd.DataFrame({
            'Context': ['context1', 'context2', 'context2', 'context3', 'context3', 'context4'],
            'Response': ['response1', 'response2', 'response3', 'response4', '', '']
        })
        self.expected_oimo_res = pd.DataFrame({
            'Context': ['context1', 'context2', 'context3', 'context4'],
            'Response': ['[response1]', '[response2, response3]', '[response4]', '[]']
        })

        self.mapped_data = {
            'text': [self.prompt.format('context1', '[response1]'),
                     self.prompt.format('context2', '[response2, response3]'),
                     self.prompt.format('context3', '[response4]'), self.prompt.format('context4', '[]')]
        }

    def test_parse_to_oimo(self):
        oimo_res = _datasetter__parse_to_oimo(self.test_dataframe)
        pd.testing.assert_frame_equal(oimo_res, self.expected_oimo_res)

    def test_map_logic(self):
        ds = Dataset.from_pandas(self.expected_oimo_res)
        res = _datasetter__map_logic(ds, self.prompt)
        print(res)
        print(self.mapped_data)
        assert res == self.mapped_data

    def cleanup_method(self):
        # delete difrectory if exists
        if os.path.exists('test_data'):
            os.rmdir('test_data')
