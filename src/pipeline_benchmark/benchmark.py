import numpy as np
import argparse
import subprocess
import difflib as diff
import json
import sys
import os
import logging
from datasets import load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass()
class Coupon:
    """
    Class representing data associated with a single coupon.
    """
    product_name: str
    new_price: Optional[str] = None
    old_price: Optional[str] = None
    percents: List[str] = field(default_factory=list)
    other_discounts: List[str] = field(default_factory=list)
    dates: Optional[str] = None


@dataclass()
class CouponSimple:
    """
    Class representing a simple coupon.
    """
    product_name: str
    discount_text: str
    validity_text: str
    activation_text: str


# Weights for each attribute of the coupon
NAME_WEIGHT = 0.3
PRICE_WEIGHT = 0.2
PERCENT_WEIGHT = 0.2
OTHER_DISCOUNT_WEIGHT = 0.2
VALIDITY_WEIGHT = 0.1

# Weights for the simple coupons
NAME_WEIGHT_SIMPLE = 0.4
DISCOUNT_WEIGHT_SIMPLE = 0.3
VALIDITY_WEIGHT_SIMPLE = 0.2
ACTIVATION_WEIGHT_SIMPLE = 0.1

# Weights for the prices
NEW_PRICE_WEIGHT = 0.5
OLD_PRICE_WEIGHT = 0.5
LENGTH_PENALTY = 0.2

# Column names for the output data
OUT_COL_SIMP_PRODUCT = 'product_name'
OUT_COL_SIMP_DISCOUNT = 'discount_text'
OUT_COL_SIMP_VALIDITY = 'valid_until'
OUT_COL_SIMP_ACTIVATION = 'activation_text'

OUT_COL_EXT_PRODUCT = 'product_name'
OUT_COL_EXT_NEW_PRICE = 'new_price'
OUT_COL_EXT_OLD_PRICE = 'old_price'
OUT_COL_EXT_PERCENTS = 'percents'
OUT_COL_EXT_OTHER_DISCOUNTS = 'other_discounts'
OUT_COL_EXT_DATES = 'dates'

# Files to store the input and output of the pipeline
INPUT_FILE = "pipeline_input.json"
OUTPUT_FILE = "pipeline_output.json"

# Columns of the dataset
INPUT = 'Context'
OUTPUT = 'Response'


def get_coupons(json_str: str, is_simple: bool, source_name: str) -> List[Union[Coupon, CouponSimple]]:
    """
    This function will parse a json string that contains coupons. 
    
    :param json_str: A json string that contains the coupons
    :param is_simple: A boolean flag to indicate if the simple format is used
    :param source_name: The name of the source of the coupons (needed for logging)
    :return: A list of Coupon or CouponSimple objects read from the file
    """
    source_info = f"{source_name} - "

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        logging.warning(source_info + f"String {json_str} is not a valid JSON.")
        return []

    if not isinstance(data, list):
        logging.warning(source_info + f"String {json_str} does not contain a list of dictionaries.")
        return []

    coupons = []
    for item in data:
        if is_simple:
            if not isinstance(item, dict):
                logging.warning(source_info + f"Item {item} is not a dictionary.")
                continue

            if OUT_COL_SIMP_PRODUCT not in item or item[OUT_COL_SIMP_PRODUCT] is None:
                logging.warning(source_info + f"Item {item} does not contain the product name.")
                continue

            coupon = CouponSimple(product_name=item[OUT_COL_SIMP_PRODUCT],
                                  discount_text=item.get(OUT_COL_SIMP_DISCOUNT, ''),
                                  validity_text=item.get(OUT_COL_SIMP_VALIDITY, ''),
                                  activation_text=item.get(OUT_COL_SIMP_ACTIVATION, ''))

            coupon.discount_text = coupon.discount_text if coupon.discount_text is not None else ''
            coupon.validity_text = coupon.validity_text if coupon.validity_text is not None else ''
            coupon.activation_text = coupon.activation_text if coupon.activation_text is not None else ''

            if type(coupon.product_name) is not str:
                logging.warning(source_info + f"Product name is not a string in item {item}.")
                coupon.product_name = ''

            if type(coupon.discount_text) is not str:
                logging.warning(source_info + f"Discount text is not a string or null in item {item}.")
                coupon.discount_text = ''

            if type(coupon.validity_text) is not str:
                logging.warning(source_info + f"Validity text is not a string or null in item {item}.")
                coupon.validity_text = ''

            if type(coupon.activation_text) is not str:
                logging.warning(source_info + f"Activation text is not a string or null in item {item}.")
                coupon.activation_text = ''

            coupons.append(coupon)

        else:
            new_price = item.get(OUT_COL_EXT_NEW_PRICE, None)
            old_price = item.get(OUT_COL_EXT_OLD_PRICE, None)
            percents = item.get(OUT_COL_EXT_PERCENTS, [])
            other_discounts = item.get(OUT_COL_EXT_OTHER_DISCOUNTS, [])
            dates = item.get(OUT_COL_EXT_DATES, None)

            coupon = Coupon(product_name=item[OUT_COL_EXT_PRODUCT],
                            new_price=new_price,
                            old_price=old_price,
                            percents=percents,
                            other_discounts=other_discounts,
                            dates=dates)
            coupons.append(coupon)

    return coupons


def compare_prices(generated_prices: list, expected_prices: list) -> float:
    """
    This function will compare two lists of prices and return a float value that 
    represents the similarity between them. The higher the value, the more similar 
    the lists are. The lower the value, the more different the lists are. The lowest
    and highest values of the lists are taken to be the old and new prices, 
    respectively, and their similarity has more weight.The score takes into account 
    the discrepancy in the number of prices between the two lists and punishes the 
    pipeline accordingly. Each list will contain at most two prices: the new and old
    prices. The lists can be empty or contain only one price.

    :param generated_prices: The first list of prices to compare (generated by the 
                            pipeline)
    :param expected_prices: The second list of prices to compare (expected)
    :return: A float value that represents the similarity between the two lists
    """

    # Convert the prices to floats and sort them; remove None values
    generated_prices = np.array(
        sorted(
            [float(price) for price in generated_prices if price is not None]))
    expected_prices = np.array(
        sorted(
            [float(price) for price in expected_prices if price is not None]))

    # Case 0: Both lists are either empty or contain only one price, so we can
    # compare them directly
    if len(expected_prices) == len(generated_prices) and (
            len(expected_prices) == 0 or len(expected_prices) == 1):
        if len(expected_prices) == 0:
            return 1.0

        return 1.0 if expected_prices[0] == generated_prices[0] else 0.0

    # Case 1: The generated list is empty, so it is completely different from
    # the expected list or vice versa
    if (len(generated_prices) == 0
            and len(expected_prices) > 0) or (len(generated_prices) > 0
                                              and len(expected_prices) == 0):
        return 0.0

    # Calculate the difference between the highest and lowest prices in
    # both lists
    new_prices = [generated_prices[0], expected_prices[0]]
    old_prices = [generated_prices[-1], expected_prices[-1]]

    new_price_ratio = 1.0 if new_prices[0] == new_prices[1] else 0.0
    old_price_ratio = 1.0 if old_prices[0] == old_prices[1] else 0.0

    coupon_difference = (new_price_ratio * NEW_PRICE_WEIGHT) + (
        old_price_ratio * OLD_PRICE_WEIGHT)

    # Case 2: Both lists have more than one price, so we can compare all
    # the prices
    if len(generated_prices) == len(expected_prices):
        return coupon_difference

    length_difference = abs(len(generated_prices) -
                            len(expected_prices)) / len(expected_prices)
    return max(0.0, coupon_difference - length_difference * LENGTH_PENALTY)


def compare_coupons_simple(coupon_1: Optional[CouponSimple],
                           coupon_2: Optional[CouponSimple]) -> float:
    """
    This function will compare two simple coupons and return a float value that
    represents the similarity between the two coupons. The higher the value, the
    more similar the coupons are. The lower the value, the more different the
    coupons are.

    :param coupon_1, coupon_2: The first and second coupons to compare
    :return: A float value that represents the similarity between the two coupons
    """

    if coupon_1 is None or coupon_2 is None:
        return 0.0

    name_ratio = diff.SequenceMatcher(a=coupon_1.product_name,
                                      b=coupon_2.product_name).ratio()
    discount_ratio = diff.SequenceMatcher(a=coupon_1.discount_text,
                                          b=coupon_2.discount_text).ratio()
    validity_ratio = diff.SequenceMatcher(a=coupon_1.validity_text,
                                          b=coupon_2.validity_text).ratio()
    activation_ratio = diff.SequenceMatcher(a=coupon_1.activation_text,
                                            b=coupon_2.activation_text).ratio()

    return (name_ratio * NAME_WEIGHT_SIMPLE) + (discount_ratio * DISCOUNT_WEIGHT_SIMPLE) + \
        (validity_ratio * VALIDITY_WEIGHT_SIMPLE) + (activation_ratio * ACTIVATION_WEIGHT_SIMPLE)


def compare_coupons(coupon_1: Optional[Coupon],
                    coupon_2: Optional[Coupon]) -> float:
    """
    This function will compare two coupons and return a float value that represents 
    the similarity between the two coupons. The higher the value, the more similar 
    the coupons are. The lower the value, the more different the coupons are. 

    :param coupon_1, coupon_2: The first and second coupons to compare
    :return: A float value that represents the similarity between the two coupons
    """

    if coupon_1 is None or coupon_2 is None:
        return 0.0

    name_ratio = diff.SequenceMatcher(a=coupon_1.product_name,
                                      b=coupon_2.product_name).ratio()
    prices_ratio = compare_prices([coupon_1.new_price, coupon_1.old_price],
                                   [coupon_2.new_price, coupon_2.old_price])

    percents_1 = sorted([str(percent) for percent in coupon_1.percents])
    percents_2 = sorted([str(percent) for percent in coupon_2.percents])
    percents_ratio = diff.SequenceMatcher(a=percents_1, b=percents_2).ratio(
    ) - LENGTH_PENALTY * abs(len(percents_1) - len(percents_2))

    discounts_1 = sorted(
        [str(discount) for discount in coupon_1.other_discounts])
    discounts_2 = sorted(
        [str(discount) for discount in coupon_2.other_discounts])
    other_discopunts_ratio = diff.SequenceMatcher(
        a=discounts_1, b=discounts_2).ratio(
        ) - LENGTH_PENALTY * abs(len(discounts_1) - len(discounts_2))

    dates_1 = sorted([str(date) for date in coupon_1.dates])
    dates_2 = sorted([str(date) for date in coupon_2.dates])
    dates_ratio = diff.SequenceMatcher(
        a=dates_1,
        b=dates_2).ratio() - LENGTH_PENALTY * abs(len(dates_1) - len(dates_2))

    return (name_ratio * NAME_WEIGHT) + (prices_ratio * PRICE_WEIGHT) + (
        percents_ratio * PERCENT_WEIGHT) + (
            other_discopunts_ratio * OTHER_DISCOUNT_WEIGHT) + (dates_ratio *
                                                               VALIDITY_WEIGHT)


def judge_pipeline(expected_coupons: List[Coupon],
                   generated_coupons: List[Coupon],
                   is_simple: bool) -> tuple[float, int]:
    """
    This function will judge the pipeline by comparing the expected coupons with the
    generated ones. The function will return a tuple with the average similarity
    between the coupons and the number of lonely coupons. The average similarity is
    calculated by comparing each expected coupon with the most similar generated
    coupon. The number of lonely coupons is the number of expected coupons that
    could not be matched with any generated coupon and vice versa. 
    
    :param expected_coupons: A list of Coupon objects that represent the expected coupons
    :param generated_coupons: A list of Coupon objects that represent the generated coupons
    :param is_simple: A boolean value that indicates if the coupons are in the simple format
    :return: A tuple with the average similarity between the coupons and the number of lonely coupons
    """

    generated_coupons = dict(
        (i, coupon) for i, coupon in enumerate(generated_coupons))
    lonely_coupons: int = 0
    similarities: List[float] = []

    for coupon in expected_coupons:
        max_similarity = 0.0
        max_coupon: int = -1

        for i, generated_coupon in generated_coupons.items():
            if is_simple:
                similarity = compare_coupons_simple(coupon, generated_coupon)
            else:
                similarity = compare_coupons(coupon, generated_coupon)
            if similarity > max_similarity:
                max_similarity = similarity
                max_coupon = i

        if max_coupon == -1:
            lonely_coupons += 1
            continue

        del generated_coupons[max_coupon]
        similarities.append(max_similarity)

    if len(generated_coupons) > 0:
        lonely_coupons += len(generated_coupons)

    similarities.extend([0.0] * lonely_coupons)

    return (np.mean(similarities) if len(similarities) > 0 else 0.0,
            lonely_coupons)


def run_pipeline(pipeline_command: str,
                 input: str,
                 is_simple: bool) -> List[Union[Coupon, CouponSimple]]:
    """
    This function will run the pipeline with the input data and return the
    generated coupons.
    
    :param pipeline_command: The command to run the pipeline
    :param input: The path to the file with the input data
    :param is_simple: A boolean value that indicates if the coupons are in the simple format
    :return: A list of Coupon or CouponSimple objects generated by the pipeline
    """
    with open(INPUT_FILE, "w") as file:
        file.write(input)

    pipeline_command = pipeline_command + " < " + INPUT_FILE + " > " + OUTPUT_FILE

    try:
        subprocess.run(pipeline_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running the pipeline with command {pipeline_command}.")
        return []

    with open(OUTPUT_FILE, 'r') as file:
        coupons = get_coupons(file.read(), is_simple, "Generated output")

    return coupons


def parse_args() -> argparse.Namespace:
    """
    This function will parse the input arguments.

    :return: The parsed input arguments
    """

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Benchmarking script')
    parser.add_argument('-d',
                        '--dataset_name',
                        type=str,
                        required=True,
                        help='Name of the dataset to download'
    )
    parser.add_argument('-p',
                        '--pipeline',
                        type=str,
                        required=True,
                        help='Command to run the pipeline (e.g., python llama_pipeline.py)'
    )
    parser.add_argument('-e',
                        '--extended',
                        action='store_true',
                        default=False,
                        help='Use the extended format'
    )
    parser.add_argument('-c',
                        '--cache_dir',
                        type=str,
                        default='datasets',
                        help='Directory to cache the datasets'
    )
    parser.add_argument('-s',
                        '--split',
                        type=str,
                        default='Edeka+Penny',
                        help='Dataset split to use'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(filename="benchmark.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()
    pipeline = args.pipeline
    is_extended = args.extended
    dataset_name = args.dataset_name
    cache_dir = args.cache_dir
    split = args.split

    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    lonely_coupons_list = []
    score_list = []
    for idx, entry in enumerate(dataset):
        input = entry[INPUT]
        expected = entry[OUTPUT]

        expected_coupons = get_coupons(expected, not is_extended, "Expected output")

        generated_coupons = run_pipeline(pipeline, input, not is_extended)

        similarity, lonely_coupons = judge_pipeline(expected_coupons,
                                                    generated_coupons,
                                                    not is_extended)

        percent_similarity = round(similarity * 100, 3)
        score = max(percent_similarity, 0)
        score_list.append(score)
        lonely_coupons_list.append(lonely_coupons)
        logging.info(f"Entry {idx + 1} - Score: {score}% - Lonely coupons: {lonely_coupons}")

    score = np.mean(score_list)
    total_lonely_coupons = np.sum(lonely_coupons_list)

    logging.info(f"Average score: {score}%")
    logging.info(f"Total number of lonely coupons: {total_lonely_coupons}")

    print(f"Average score: {score}%")
    print(f"Total number of lonely coupons: {total_lonely_coupons}")
