import os
import shutil

import pandas as pd
import pytest

from src.llama_dataset_generation.ground_truth_parser import load_coupons, prepare_ground_truth_data, \
    prepare_ground_truth_data_no_ai
from src.constants import *

EXAMPLE_COUPON_FRAME = pd.DataFrame({
    'discount_text': ['heh', 'hah', 'hoh'],
    AGGREGATION_COLUMN: ['1-1', '2-2', '3-3'],
    'product_text': ['product1', 'product2', 'product3'],
    'validity_text': ['valid1', 'valid2', 'valid3'],
    "content_full": ["['bydle']", "['bydle']", "['bydle']"],
    'activation_text': ['bydle', 'cydle', 'dydle']
})

EXAMPLE_GTD_NEW_FORMAT = {
    '1-1': [
        {
            'activation_text': 'bydle',
            'product_name': 'product1',
            'valid_until': 'valid1',
            'discount_text': 'heh'
        },
    ],
    '2-2': [
        {
            'activation_text': 'cydle',
            'product_name': 'product2',
            'valid_until': 'valid2',
            'discount_text': 'hah'
        },
    ],
    '3-3': [
        {
            'activation_text': 'dydle',
            'product_name': 'product3',
            'valid_until': 'valid3',
            'discount_text': 'hoh'
        },
    ]
}


class TestGroundTruthParser:
    csv_path = 'test_data/test_ground_truth_parser.csv'
    loaded_coupons: list
    discounts: list
    original_coupons: pd.DataFrame
    gtd_dict: dict

    @pytest.fixture(autouse=True)
    def setup_method(self):
        data = {
            'discount_text': ['1t', '2t', '3t', '4t', '5t', '6t', '7t', '8t', 'nan', '10t', '11t', 'nan'],
            'some_stuff': ['1-1', '2-2', '3-3', '4-4', '5-5', '6-6', '7-7', '8-8', '9-9', '10-10', '11-11', '12-12']
        }
        csv_df = pd.DataFrame(data)
        os.makedirs('test_data', exist_ok=True)
        csv_df.to_csv(self.csv_path, index=False)
        self.loaded_coupons = ['1t', '2t', '3t', '4t', '5t', '6t', '7t', '8t', '', '10t', '11t', '']

        self.discounts_list = [
            {
                'discount': 'heh',
                'prices': []
            },
            {
                'discount': 'hah',
                'prices': ['420']
            },
            {
                'discount': 'hoh',
                'prices': ['666', '777', '888']
            }
        ]

        self.original_coupons = EXAMPLE_COUPON_FRAME.copy()

        self.gtd_dict = {
            '1-1': [
                {
                    'product_name': 'product1',
                    'valid_until': 'valid1',
                    'discount': 'heh',
                    'old_price': '',
                    'new_price': ''
                },
            ],
            '2-2': [
                {
                    'product_name': 'product2',
                    'valid_until': 'valid2',
                    'discount': 'hah',
                    'old_price': '',
                    'new_price': '420'
                },
            ],
            '3-3': [
                {
                    'product_name': 'product3',
                    'valid_until': 'valid3',
                    'discount': 'hoh',
                    'old_price': '666',
                    'new_price': '888'
                },
            ]
        }

        yield
        if os.path.exists('test_data'):
            shutil.rmtree('test_data')


    def test_load_coupons(self):
        res = load_coupons(self.csv_path)
        assert res == self.loaded_coupons

    def test_prepare_ground_truth_data_and_ground_truth_to_dict(self):
        res = prepare_ground_truth_data(self.discounts_list, self.original_coupons)
        assert res == self.gtd_dict

    @pytest.mark.parametrize('frame,exp', [
        (
                EXAMPLE_COUPON_FRAME.copy(),
                EXAMPLE_GTD_NEW_FORMAT
        ),
        (
            pd.DataFrame({'discount_text': [], AGGREGATION_COLUMN: [], 'product_text': [], 'validity_text': [], "content_full": []}),
            {}
        ),
        (
            pd.DataFrame({
                'discount_text': ['10zł', '2137gr', 'pół piwa'],
                'product_text': ['apples', 'sword', 'piwo'],
                'validity_text': ['N/A', 'N/A', '31.02.2025'],
                AGGREGATION_COLUMN: ['t0', 't0', 't0'],
                'content_full': ["['bydle']", "['bydle']", "['bydle', 'dziekan]"],
                'activation_text': ['bydle', 'cydle', 'dydle']
            }),
            {'t0': [
                {
                    'product_name': 'apples',
                    'valid_until': 'N/A',
                    'discount_text': '10zł',
                    'activation_text': 'bydle'
                },
                {
                    'product_name': 'sword',
                    'valid_until': 'N/A',
                    'discount_text': '2137gr',
                    'activation_text': 'cydle'
                },
                {
                    'product_name': 'piwo',
                    'valid_until': '31.02.2025',
                    'discount_text': 'pół piwa',
                    'activation_text': 'dydle'
                }
            ]}
        )
    ])
    def test_prepare_ground_truth_data_no_ai(self, frame: pd.DataFrame, exp: dict):
        assert prepare_ground_truth_data_no_ai(frame) == exp