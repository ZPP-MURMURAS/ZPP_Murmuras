import pytest
import sys
import os
import numpy as np
import logging
from unittest.mock import patch, mock_open

from src.pipeline_benchmark.benchmark import Coupon, CouponSimple, get_coupons, compare_prices, compare_coupons, compare_coupons_simple, compute_similarity_matrix, greedy_matching, compute_similarities


class TestBenchmark:

    @pytest.mark.parametrize("input_string,is_simple,expected_coupons", [
        ('[{"product_name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01", "activation_text": "active"}]', True, [CouponSimple("Product A", "-20%", "2025-01-01", "active")]),
        ('[{"product_name": "Product B", "new_price": "5$", "old_price": "6$", "percents": ["1", "2"], "other_discounts": ["3", "4"], "dates": ["1410", "1812"]}]', False, [Coupon("Product B", "5$", "6$", ["1", "2"], ["3", "4"], ["1410", "1812"])]),
        ('[{"name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01", "activation_text": "active"}]', True, []),
        ('[{"product_name": "Product A", "valid_until": "2025-01-01", "activation_text": "active"}]', True, [CouponSimple("Product A", "", "2025-01-01", "active")]),
        ('[{product_name: Product A, discount_text: -20%, valid_until: 2025-01-01, activation_text: active}]', True, []),
    ])
    def test_get_coupons(self, input_string, is_simple, expected_coupons, caplog):
        with caplog.at_level(logging.INFO):
            assert get_coupons(input_string, is_simple, "placeholder") == expected_coupons


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
        (CouponSimple("Product A", "10.0", "valid till 900", "active"),
         CouponSimple("Product A", "10.0", "valid till 900", "active"), 1.0),
        (CouponSimple("A", "10.0", "20.0", "active"), 
         CouponSimple("B", "234", "344", "active"), 0.0),
        (CouponSimple(
            "Product ABC",
            "discount 123 from 234 and 60% off if you have a unicorn",
            "valid till 6969",
            "active"),
         CouponSimple(
             "Product",
             "discount 90% from 76 to 9 and also we will give you a phoenix",
             "until 1169 AD",
             "not active"), 0.6),
        (CouponSimple("A", "10.0", "valid till 900", "active"),
         CouponSimple("B", "10.0", "valid till 900", "active"), 0.6),
        (CouponSimple("unicorn", "10.0", "valid till 9000", "active"),
         CouponSimple("UNICORN", "10.0", "valid till 9000", "active"), 0.6),
        (CouponSimple("UNICORN", "10.0", "valid till 969", "active"),
         CouponSimple("UNICORN", "abc", "valid till 969", "active"), 0.6),
        (CouponSimple("UNICORN", "10.0", "valid till 969", "active"),
         CouponSimple("UNICORN", "10.3", "valid till 969", "active"), 0.9),
        (CouponSimple("UNICORN", "10.0", "valid till 969", "active"),
         CouponSimple("UNICORN", "10.0", "valid till 1212", "active"), 0.9),
        (CouponSimple("UNICORN", "10.0", "valid till 969", "active"),
         CouponSimple("UNICORN", "10.0", "uga buga", "active"), 0.8),
        (CouponSimple("UNICORN", "10.0", "valid till 969", "active"),
         CouponSimple("UNICORN", "10.0", "valid till 969", "dziekan"), 0.9),
    ])
    def test_compare_coupons_simple(self, coupon1, coupon2, expected_score):
        assert np.isclose(compare_coupons_simple(coupon1, coupon2),
                          expected_score,
                          atol=0.1)

    @pytest.mark.parametrize("expected_coupons, generated_coupons, compare_function, expected_matrix", [
        ([CouponSimple("Product A", "10.0", "2025-01-01", "active"),
          CouponSimple("Product B", "15.0", "2025-02-01", "active")],
         [CouponSimple("Product A", "10.0", "2025-01-01", "active"),
          CouponSimple("Product B", "15.0", "2025-02-01", "active")],
         compare_coupons_simple,
         [[1.0, 0.8], [0.8, 1.0]]),
        ([CouponSimple("Product A", "10.0", "2025-01-01", "active"),
          CouponSimple("Product B", "15.0", "2025-02-01", "active")],
         [CouponSimple("Dziekan", "całka", "piwo", "sesja")],
         compare_coupons_simple,
         [[0.0], 
          [0.0]]),
    ])
    def test_similarity_matrix(self, expected_coupons, generated_coupons, compare_function, expected_matrix):
        assert np.allclose(compute_similarity_matrix(expected_coupons, generated_coupons, compare_function), expected_matrix, atol=0.1)


    @pytest.mark.parametrize("similarity_matrix, threshold, expected_similarities, expected_missed, expected_halucinated", [
        (np.array([[1.0, 0.8], [0.8, 1.0]]), 0.5, [1.0, 1.0], 0, 0),
        (np.array([[0.4], [0.2]]), 0.5, [], 2, 1),
        (np.array([[0.4, 0.6, 0.8], [0.2, 0.5, 0.9]]), 0.5, [0.9, 0.6], 0, 1),
    ])
    def test_greedy_matching(self, similarity_matrix, threshold, expected_similarities, expected_missed, expected_halucinated):
        similarities, missed, halucinated = greedy_matching(similarity_matrix, threshold)
        assert similarities == expected_similarities
        assert missed == expected_missed
        assert halucinated == expected_halucinated


    def test_compute_similarities(self):
        expected_coupons = [
            CouponSimple("Product A", "10.0", "2025-01-01", "active"),
            CouponSimple("Product B", "15.0", "2025-02-01", "active")
        ]
        generated_coupons = [
            CouponSimple("Product B", "15.0", "2025-02-01", "active"),
            CouponSimple("Dziekan", "całka", "piwo", "sesja")
        ]
        similarities, missed, halucinated = compute_similarities(expected_coupons, generated_coupons, 0.5, True)
        assert np.allclose(similarities, [1.0], atol=0.1)
        assert halucinated == 1
        assert missed == 1
