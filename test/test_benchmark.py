import pytest
import sys
import os
import numpy as np
import logging
from unittest.mock import patch, mock_open

from src.pipeline_benchmark.benchmark import Coupon, get_coupons, compare_coupons, compute_similarity_matrix, greedy_matching, compute_similarities


class TestBenchmark:

    @pytest.mark.parametrize("input_string, expected_coupons", [
        ('[{"product_name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01", "activation_text": "active"}]', [Coupon("Product A", "-20%", "2025-01-01", "active")]),
        ('[{"name": "Product A", "discount_text": "-20%", "valid_until": "2025-01-01", "activation_text": "active"}]', []),
        ('[{"product_name": "Product A", "valid_until": "2025-01-01", "activation_text": "active"}]', [Coupon("Product A", "", "2025-01-01", "active")]),
        ('[{product_name: Product A, discount_text: -20%, valid_until: 2025-01-01, activation_text: active}]', []),
    ])
    def test_get_coupons(self, input_string, expected_coupons, caplog):
        with caplog.at_level(logging.INFO):
            assert get_coupons(input_string, "placeholder") == expected_coupons


    @pytest.mark.parametrize("coupon1, coupon2, expected_score", [
        (Coupon("Product A", "10.0", "valid till 900", "active"),
         Coupon("Product A", "10.0", "valid till 900", "active"), 1.0),
        (Coupon("A", "10.0", "20.0", "active"), 
         Coupon("B", "234", "344", "active"), 0.1),
        (Coupon(
            "Product ABC",
            "discount 123 from 234 and 60% off if you have a unicorn",
            "valid till 6969",
            "active"),
         Coupon(
             "Product",
             "discount 90% from 76 to 9 and also we will give you a phoenix",
             "until 1169 AD",
             "not active"), 0.64),
        (Coupon("A", "10.0", "valid till 900", "active"),
         Coupon("B", "10.0", "valid till 900", "active"), 0.6),
        (Coupon("unicorn", "10.0", "valid till 9000", "active"),
         Coupon("UNICORN", "10.0", "valid till 9000", "active"), 0.6),
        (Coupon("UNICORN", "10.0", "valid till 969", "active"),
         Coupon("UNICORN", "abc", "valid till 969", "active"), 0.7),
        (Coupon("UNICORN", "10.0", "valid till 969", "active"),
         Coupon("UNICORN", "10.3", "valid till 969", "active"), 0.92),
        (Coupon("UNICORN", "10.0", "valid till 969", "active"),
         Coupon("UNICORN", "10.0", "valid till 1212", "active"), 0.95),
        (Coupon("UNICORN", "10.0", "valid till 969", "active"),
         Coupon("UNICORN", "10.0", "uga buga", "active"), 0.83),
        (Coupon("UNICORN", "10.0", "valid till 969", "active"),
         Coupon("UNICORN", "10.0", "valid till 969", "dziekan"), 0.91),
        (Coupon("UNICORN", "", "valid till 969", "active"),
         Coupon("UNICORN", "", "valid till 969", "active"), 1.0),
        (Coupon("UNICORN", "10.0", "valid till 969", "active"),
         Coupon("UNICORN", "", "valid till 969", "active"), 0.7),
        (Coupon("UNICORN", "", "a", "a"),
         Coupon("UNICORN", "", "b", "b"), 0.57),
        (Coupon("UNICORN", "", "", "a"),
         Coupon("UNICORN", "", "", "b"), 0.8),
        (Coupon("UNICORN", "a", "a", ""),
         Coupon("UNICORN", "b", "b", ""), 0.44),
        (Coupon("unicorn", "", "", ""),
         Coupon("UNICORN", "", "", ""), 0.0),
    ])
    def test_compare_coupons(self, coupon1, coupon2, expected_score):
        assert np.isclose(compare_coupons(coupon1, coupon2), expected_score, atol=0.01)

    @pytest.mark.parametrize("expected_coupons, generated_coupons, expected_matrix", [
        ([Coupon("Product A", "10.0", "2025-01-01", "active"),
          Coupon("Product B", "15.0", "2025-02-01", "active")],
         [Coupon("Product A", "10.0", "2025-01-01", "active"),
          Coupon("Product B", "15.0", "2025-02-01", "active")],
         [[1.0, 0.86], [0.86, 1.0]]),
        ([Coupon("Product A", "10.0", "2025-01-01", "active"),
          Coupon("Product B", "15.0", "2025-02-01", "active")],
         [Coupon("Dziekan", "całka", "piwo", "sesja")],
         [[0.02], 
          [0.02]]),
    ])
    def test_compute_similarity_matrix(self, expected_coupons, generated_coupons, expected_matrix):
        assert np.allclose(compute_similarity_matrix(expected_coupons, generated_coupons), expected_matrix, atol=0.01)


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
            Coupon("Product A", "10.0", "2025-01-01", "active"),
            Coupon("Product B", "15.0", "2025-02-01", "active")
        ]
        generated_coupons = [
            Coupon("Product B", "15.0", "2025-02-01", "active"),
            Coupon("Dziekan", "całka", "piwo", "sesja")
        ]
        similarities, missed, halucinated = compute_similarities(expected_coupons, generated_coupons, 0.5)
        assert np.allclose(similarities, [1.0], atol=0.01)
        assert halucinated == 1
        assert missed == 1
