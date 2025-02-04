import shutil

import pandas as pd
import os

import pytest

from src.llama_dataset_generation.input_parser import prepare_input_data, create_training_df

class TestInputParser:
    csv_path = 'test_data/test_input_parser.csv'  # Fixed path issue
    csv_dict: dict
    input_dict: dict
    gtd_dict: dict
    training_df: pd.DataFrame

    @pytest.fixture(autouse=True)
    def setup_method(self):
        data = {
            'seen_timestamp': [1, 1, 1, 1, 0, 2, 2, 0, 0, 3, 0, 3],
            'is_visible': [False, True, False, True, True, True, False, False, True, True, True, True],
            'text': ['1t', '2t', '3t', '4t', '5t', '6t', '7t', '8t', 'nan', '10t', '11t', 'nan'],
            'time': ['1-1', '2-2', '3-3', '4-4', '5-5', '6-6', '7-7', '8-8', '9-9', '10-10', '11-11', '12-12']
        }
        csv_df = pd.DataFrame(data)
        os.makedirs('test_data', exist_ok=True)
        csv_df.to_csv(self.csv_path, index=False)
        self.csv_dict = {
            '2-2': '2t 4t',
            '6-6': '6t',
            '10-10': '10t'
        }

        self.input_dict = {
            'A': 'bla',
            'B': 'blabla',
            'C': 'blablabla',
            'D': 'blablablabla',
            'E': 'blablablablabla',
        }
        self.gtd_dict = {
            'B': [{'bruh': 'duch'}],
            'D': [{'bruhu': 'duchu'}, {'bruha': 'ducha'}],
        }
        res_dict = {
            'Context': ['bla', 'blabla', 'blablabla', 'blablablabla', 'blablablabla', 'blablablablabla'],
            'Response': ['{}', '{\'bruh\': \'duch\'}', '{}', '{\'bruhu\': \'duchu\'}', '{\'bruha\': \'ducha\'}', '{}']
        }
        self.training_df = pd.DataFrame(res_dict)

        yield
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')


    def test_prepare_input_data_and_concat(self):
        result = prepare_input_data(self.csv_path)
        assert result == self.csv_dict

    def test_create_training_df(self):
        result = create_training_df(self.input_dict, self.gtd_dict)
        pd.testing.assert_frame_equal(result, self.training_df)
