import pytest
import sys 
import os
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from research.pipeline_benchmark.benchmark import _compare_prices, compare_coupons, judge_pipeline, Coupon

class TestBenchmark:
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
        assert np.isclose(_compare_prices(generated, expected), expected_score, atol=0.01)
    
    @pytest.mark.parametrize("coupon1, coupon2, expected_score", [
        (Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
         Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
         1.0),
        (Coupon("A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
         Coupon("B", "15.0", "25.0", [15], [10], ["2025-02-01"]),
         0.0),
        (Coupon("Product A", "10.0", "20.0", [10, 15], [5], ["2025-01-01"]),
         Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
         0.8),
         (Coupon("Product X", "12.0", "22.0", [12], [6], ["2025-03-01"]),
         Coupon("Product X", "12.0", "22.0", [12], [5], ["2025-03-01"]),
         0.9),
        (Coupon("Y", "14.0", "24.0", [14], [7], ["2025-04-01"]),
         Coupon("Z", "16.0", "26.0", [16], [8], ["2025-05-01"]),
         0.0)
    ])
    def test_compare_coupons(self, coupon1, coupon2, expected_score):
        assert np.isclose(compare_coupons(coupon1, coupon2), expected_score, atol=0.1)

    def test_judge_pipeline(self):
        expected_coupons = [
            Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"]),
            Coupon("Product B", "15.0", "25.0", [15], [10], ["2025-02-01"])
        ]
        generated_coupons = [
            Coupon("Product A", "10.0", "20.0", [10], [5], ["2025-01-01"])
        ]
        similarity, lonely = judge_pipeline(expected_coupons, generated_coupons)
        assert np.isclose(similarity, 0.5, atol=0.1)
        assert lonely == 1