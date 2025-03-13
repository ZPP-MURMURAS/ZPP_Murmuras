from src.bert_dataset_generation.generate_field_extraction_ds import (
    __samples_from_entry as _ds_gen__samples_from_entry,
    __samples_from_entry_2 as _ds_gen__samples_from_entry_2
)

import pandas as pd
import pytest
from pytest_lazy_fixtures import lf


@pytest.fixture
def coupons_fmt2_1():
    return pd.DataFrame({
        'product_text': ['Product1', 'Product2', 'Product3'],
        'discount_text': ['10% off', '20% off', '30% off'],
        'discount_details': ['Up to 50% off', 'Up to 50% off', 'Up to 50% off'],
        'validity_text': ['Valid till 31st Dec', 'Valid till 31st Dec', 'Valid till 31st Dec'],
        'activation_text': ['Activated', 'Activated', ''],
    })


@pytest.fixture
def expected_fmt2_1_seed42():
    return [(['Product1', 'Valid till 31st Dec', 'Activated'], [1, 5, 0]), (['Product2', 'Up to 50% off', '20% off', 'Valid till 31st Dec'], [1, 0, 3, 5]), (['Product3', 'Valid till 31st Dec'], [1, 5])]


@pytest.fixture
def expected_fmt2_1_seed69():
    return [(['Product1', 'Valid till 31st Dec', 'Activated'], [1, 5, 0]), (['Activated', 'Up to 50% off', '20% off', 'Valid till 31st Dec', 'Product2'], [0, 0, 3, 5, 1]), (['Product3', 'Up to 50% off', '30% off', 'Valid till 31st Dec'], [1, 0, 3, 5])]


class TestGenerateCouponLabelingDS:

    @pytest.mark.parametrize('coupons,seed,expected', [
        (lf('coupons_fmt2_1'), 42, lf('expected_fmt2_1_seed42')),
    ])
    def test_samples_from_entry_2(self, coupons, expected, seed):
        labeled = _ds_gen__samples_from_entry_2(coupons, seed)
        assert labeled == expected


    @pytest.mark.parametrize('fmt,coupons,seed, expected', [
        (2, lf('coupons_fmt2_1'), 69, lf('expected_fmt2_1_seed69')),
    ])
    def test_samples_from_entry(self, fmt, coupons, expected, seed):
        labeled = _ds_gen__samples_from_entry(fmt, coupons, seed)
        assert labeled == expected

