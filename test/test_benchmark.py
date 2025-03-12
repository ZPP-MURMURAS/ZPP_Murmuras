import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, mock_open

from src.pipeline_benchmark.benchmark import Coupon, CouponSimple, get_coupons, compare_prices, compare_coupons, compare_coupons_simple, judge_pipeline


class TestBenchmark:

    @pytest.mark.parametrize("input_string,is_simple,expected_coupons", [
        ('[{"product_name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01"}]', True, [CouponSimple("Product A", "-20%", "2025-01-01")]),
        ('[{"product_name": "Product B", "new_price": "5$", "old_price": "6$", "percents": ["1", "2"], "other_discounts": ["3", "4"], "dates": ["1410", "1812"]}]', False, [Coupon("Product B", "5$", "6$", ["1", "2"], ["3", "4"], ["1410", "1812"])]),
    ])
    def test_get_coupons_success(self, input_string, is_simple, expected_coupons):
        with patch('builtins.open', mock_open(read_data=input_string)):
            assert get_coupons('mock_filepath.json', is_simple) == expected_coupons


    @pytest.mark.parametrize("input_string,is_simple", [
        ('[{"name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01"}]', True),
        ('[{"product_name": "Product A", "valid_until": "2025-01-01"}]', True),
        ('[{"product_name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01"}]', False),
        ('[{product_name: Product A, discount_text: -20%, valid_until: 2025-01-01}]', True),
    ])
    def test_get_coupons_failure(self, input_string, is_simple):
        with pytest.raises(ValueError) as e, patch('builtins.open', mock_open(read_data=input_string)):
            get_coupons('mock_filepath.json', is_simple)


    @pytest.mark.parametrize("generated, expected, expected_score", [
        (["10.0", "20.0"], ["10.0", "20.0"], 1.0),
        (["10.0"], ["10.0"], 1.0),
        (["10.0"], ["20.0"], 0.0),
        (["10.0", "30.0"], ["10.0", "20.0"], 0.5),
        ([], ["10.0"], 0.0),
        (["10.0"], [], 0.0),
        (["10.0", "20.0", "30.0"], ["10.0", "20.0"], 0.4),
        (["15.0", "25.0"], ["10.0", "20.0"], 0.0),
    ])
    def test_compare_prices(self, generated, expected, expected_score):
        assert np.isclose(compare_prices(generated, expected),
                          expected_score,
                          atol=0.01)

    @pytest.mark.parametrize(
        "coupon1, coupon2, expected_score",
        [(Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
          Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]), 1.0),
         (Coupon("A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
          Coupon("B", "15.0", "25.0", [15], [10], ["2025-02-01"]), 0.0),
         (Coupon("Product A", "10.0", "20.0", [10, 15], [5], ["2025-01-01"]),
          Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]), 0.8),
         (Coupon("Product X", "12.0", "22.0", [12], [6], ["2025-03-01"]),
          Coupon("Product X", "12.0", "22.0", [12], [5], ["2025-03-01"]), 0.9),
         (Coupon("Y", "14.0", "24.0", [14], [7], ["2025-04-01"]),
          Coupon("Z", "16.0", "26.0", [16], [8], ["2025-05-01"]), 0.0)])
    def test_compare_coupons(self, coupon1, coupon2, expected_score):
        assert np.isclose(compare_coupons(coupon1, coupon2),
                          expected_score,
                          atol=0.1)

    @pytest.mark.parametrize("coupon1, coupon2, expected_score", [
        (CouponSimple("Product A", "10.0", "valid till 900"),
         CouponSimple("Product A", "10.0", "valid till 900"), 1.0),
        (CouponSimple("A", "10.0", "20.0"), CouponSimple("B", "234",
                                                         "344"), 0.0),
        (CouponSimple(
            "Product ABC",
            "discount 123 from 234 and 60% off if you have a unicorn",
            "valid till 6969"),
         CouponSimple(
             "Product",
             "discount 90% from 76 to 9 and also we will give you a phoenix",
             "until 1169 AD"), 0.6),
        (CouponSimple("A", "10.0", "valid till 900"),
         CouponSimple("B", "10.0", "valid till 900"), 0.6),
        (CouponSimple("unicorn", "10.0", "valid till 9000"),
         CouponSimple("UNICORN", "10.0", "valid till 9000"), 0.6),
        (CouponSimple("UNICORN", "10.0", "valid till 969"),
         CouponSimple("UNICORN", "abc", "valid till 969"), 0.6),
        (CouponSimple("UNICORN", "10.0", "valid till 969"),
         CouponSimple("UNICORN", "10.3", "valid till 969"), 0.9),
        (CouponSimple("UNICORN", "10.0", "valid till 969"),
         CouponSimple("UNICORN", "10.0", "valid till 1212"), 0.9),
        (CouponSimple("UNICORN", "10.0", "valid till 969"),
         CouponSimple("UNICORN", "10.0", "uga buga"), 0.8),
    ])
    def test_compare_coupons_simple(self, coupon1, coupon2, expected_score):
        assert np.isclose(compare_coupons_simple(coupon1, coupon2),
                          expected_score,
                          atol=0.1)

    def test_judge_pipeline(self):
        expected_coupons = [
            Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
            Coupon("Product B", "15.0", "25.0", [15], [10], ["2025-02-01"])
        ]
        generated_coupons = [
            Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"])
        ]
        similarity, lonely = judge_pipeline(expected_coupons,
                                            generated_coupons, False)
        assert np.isclose(similarity, 0.5, atol=0.1)
        assert lonely == 1
