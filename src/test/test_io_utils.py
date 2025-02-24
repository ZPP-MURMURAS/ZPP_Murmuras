import pytest
import json
import os
import csv

from unittest.mock import patch, mock_open
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.benchmark_utils.io_utils import _get_discounts, _get_coupons_old, _get_coupons_new


class TestIOUtils:

    @pytest.mark.parametrize("text, expected", [
        ("Buy now for 50% off! Was 100, now 50.",
         ("50.0", "100.0", ["50"], [])),
        ("Save 30% today!", (None, None, ["30"], [])),
        ("Special price: 20", (None, None, [], ["Special price: 20"])),
    ])
    def test_get_discounts(self, text, expected):
        assert _get_discounts(text) == expected

    @pytest.mark.parametrize("mock_csv_data, expected_name, expected_percent", [
        ("product_text,discount_text,validity_text\nProduct A,50% off,Valid until 2025\n",
         "Product A", "50"),
        ("product_text,discount_text,validity_text\nProduct C,No discount,Valid until 2024\n",
         "Product C", None),
    ])
    def test_get_coupons_old(self, mock_csv_data, expected_name,
                             expected_percent):
        with patch("builtins.open", mock_open(read_data=mock_csv_data)):
            with patch("csv.DictReader",
                       return_value=csv.DictReader(
                           mock_csv_data.splitlines())):
                coupons = _get_coupons_old("dummy.csv")
                assert len(coupons) == 1
                assert coupons[0].product_name == expected_name
                assert expected_percent in coupons[
                    0].percents or expected_percent is None

    @pytest.mark.parametrize(
        "mock_json_data, expected_name, expected_new_price, expected_percent",
        [("[{\"product_name\": \"Product B\", \"new_price\": \"40\", \"old_price\": \"50\", \"percents\": [\"20\"], \"other_discounts\": [], \"dates\": \"Valid until 2025\"}]",
          "Product B", "40", "20"),
         ("[{\"product_name\": \"Product D\", \"new_price\": \"30\", \"old_price\": \"45\", \"percents\": [], \"other_discounts\": [], \"dates\": \"Valid until 2026\"}]",
          "Product D", "30", None)])
    def test_get_coupons_new(self, mock_json_data, expected_name,
                             expected_new_price, expected_percent):
        with patch("builtins.open", mock_open(read_data=mock_json_data)):
            with patch("json.load", return_value=json.loads(mock_json_data)):
                coupons = _get_coupons_new("dummy.json")
                assert len(coupons) == 1
                assert coupons[0].product_name == expected_name
                assert coupons[0].new_price == expected_new_price
                assert expected_percent in coupons[
                    0].percents or expected_percent is None
