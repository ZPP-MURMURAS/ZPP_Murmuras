import json
import re
import csv
import os
import shutil
import subprocess
import sys

from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field


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


# Regex matches to different types of discounts
PERCENT_REGEX = r'\b(100|[1-9]?[0-9])\s?%'
PRICE_REGEX = r'\b\d+[.,]?\d*\b'

# Column names for the expected coupons
DISCOUNT_TEXT = 'discount_text'
PRODUCT_TEXT = 'product_text'
VALIDITY_TEXT = 'validity_text'

INCORRECT_DATASETS = [
    "rewe",
    "dm",
]


def _get_discounts(
        discount_text: str) -> Tuple[str, str, List[str], List[str]]:
    """
    This function will extract the discounts from the text of the coupon and return
    them in a structured format. The discounts can be of different types: new price,
    old price, percentage, or other discounts. The function will return the new and
    old prices, the percentages, and the other discounts found in the text. Not all
    discounts are required to be present in the text. The function will return None 
    or an empty list if the discount is not found.

    :param discount_text: The text of the coupon
    :return: A tuple with the new price, old price, percentages, and other discounts
    """

    new_price = None
    old_price = None
    percents = []
    other_discounts = []

    if discount_text is not None:
        discounts_found = 0
        if re.search(PRICE_REGEX, discount_text):
            prices = re.findall(PRICE_REGEX, discount_text)
            if len(prices) >= 2:
                prices = sorted([float(price) for price in prices])
                new_price = str(prices[0])
                old_price = str(prices[-1])
                discounts_found += 1

        if re.search(PERCENT_REGEX, discount_text):
            percents = re.findall(PERCENT_REGEX, discount_text)
            discounts_found += 1

        if discounts_found == 0:
            other_discounts = [discount_text]

    return new_price, old_price, percents, other_discounts


def _get_coupons_old(file_path: str) -> List[Coupon]:
    """
    Extracts the coupons from a CSV file. The function will return a list of Coupon
    objects that represent the expected coupons.

    :param file_path: The path to the CSV file
    :return: A list of Coupon objects that represent the expected coupons
    """

    expected_coupons = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)

        for row in reader:
            new_price, old_price, percents, other_discounts = _get_discounts(
                row[DISCOUNT_TEXT])

            coupon = Coupon(product_name=row[PRODUCT_TEXT],
                            new_price=new_price,
                            old_price=old_price,
                            percents=percents,
                            other_discounts=other_discounts,
                            dates=row[VALIDITY_TEXT])
            expected_coupons.append(coupon)

    return expected_coupons


def _get_coupons_new(
        file_path: str,
        is_simple: bool = False) -> List[Union[Coupon, CouponSimple]]:
    """
    Extracts the coupons from a JSON file. The function will return a list of Coupon
    or CouponSimple objects that represent the expected coupons. 

    :param file_path: The path to the JSON file
    :param is_simple: A boolean flag to indicate if the simple format is used
    :return: A list of Coupon or CouponSimple objects that represent the expected coupons
    """

    expected_coupons = []
    with open(file_path, 'r') as file:
        data = json.load(file)

        for item in data:
            if is_simple:
                coupon = CouponSimple(product_name=item["name"],
                                      discount_text=item["text"],
                                      validity_text=item["validity"])
                expected_coupons.append(coupon)
                continue

            new_price = item.get("new_price", None)
            old_price = item.get("old_price", None)
            percents = item.get("percents", [])
            other_discounts = item.get("other_discounts", [])
            dates = item.get("dates", None)

            coupon = Coupon(product_name=item["product_name"],
                            new_price=new_price,
                            old_price=old_price,
                            percents=percents,
                            other_discounts=other_discounts,
                            dates=dates)
            expected_coupons.append(coupon)

    return expected_coupons


def get_expected_coupons(
        file_path: Optional[str],
        is_new_format: bool = False,
        is_simple: bool = False) -> List[Union[Coupon, CouponSimple]]:
    """
    This function will return a list of Coupon objects that represent the 
    expected coupons. This function is used to benchmark the pipeline.
    
    :param file_path: The path to the folder with the expected coupons in csv 
                    format (like in the Murmuras datasets)
    :param is_new_format: A boolean flag to indicate if the new format is used
    :param is_simple: A boolean flag to indicate if the simple format is used
    :return: A list of Coupon objects that represent the expected coupons
    """

    # No file path provided
    if file_path is None:
        return []

    expected_coupons = []

    for file_name in os.listdir(file_path):
        file_path_full = os.path.join(file_path, file_name)
        if not os.path.isfile(file_path_full):
            continue

        if not is_new_format:
            expected_coupons.extend(_get_coupons_old(file_path_full))
        else:
            expected_coupons.extend(_get_coupons_new(file_path_full,
                                                     is_simple))

    return expected_coupons


def _validate_output_file_format(fieldnames: list) -> bool:
    """
    This function will validate the format of the csv file that contains the
    expected coupons. The file must contain the headers defined below. 
    
    :param fieldnames: The headers of the csv file
    :return: True if the file format is valid, False otherwise
    """

    required_headers = [
        "product_text", "discount_text", "discount_details", "validity_text"
    ]
    return all(header in fieldnames for header in required_headers)


def _validate_output_file_new_format(file: str,
                                     is_simple: bool = False) -> bool:
    """
    This function will validate the format of the json file that contains the
    expected coupons. The file must contain the keys defined below. 
    
    :param file: The path to the json file
    :param is_simple: A boolean flag to indicate if the simple format is used
    :return: True if the file format is valid, False otherwise
    """

    required_keys = {
        "product_name", "new_price", "old_price", "percents",
        "other_discounts", "dates"
    }

    if is_simple:
        required_keys = {"name", "text", "validity"}

    with open(file, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(
                f"The file {file} is not a valid JSON file or contains invalid data."
            )

    if not isinstance(data, list):
        raise ValueError(
            f"The file {file} must contain a list of dictionaries.")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(
                f"Entry at index {idx} in {file} must be a dictionary.")

        if not required_keys.issubset(item.keys()):
            missing_keys = required_keys - item.keys()
            raise ValueError(
                f"Entry at index {idx} in {file} is missing keys: {missing_keys}"
            )

    return True


def _validate_input_file_format(fieldnames: list) -> bool:
    """
    This function will validate the format of the csv file that contains the
    input data. The file must contain the headers defined below.
    
    :param fieldnames: The headers of the csv file
    :return: True if the file format is valid, False otherwise
    """

    required_headers = [
        "view_depth", "text", "description", "class_name", "application_name"
    ]
    return all(header in fieldnames for header in required_headers)


def _validate_folder(folder: str,
                     validation_func: callable,
                     is_new_format: bool = False) -> bool:
    """
    Validates a folder containing CSV files. The function will check if the folder
    exists, if it is a directory, and if it contains only CSV files in the correct
    format.

    :param folder: The path to the folder
    :param validation_func: The function to validate the format of the CSV files
    :param is_new_format: A boolean flag to indicate if the new format is used
    :return: True if the folder is valid, False otherwise
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(
            f"The input path {folder} is not a directory.")

    for file_name in os.listdir(folder):
        file_name_full = os.path.join(folder, file_name)
        if not os.path.isfile(file_name_full):
            raise NotADirectoryError(
                f"The output path {file_name_full} is not a file.")

        if is_new_format:
            if not file_name.endswith('.json'):
                raise ValueError(f"The file {file_name} is not a JSON file.")

            if not validation_func(file_name_full):
                raise ValueError(
                    f"The file {file_name} has an invalid format.")
            continue

        if not file_name.endswith('.csv'):
            raise ValueError(f"The file {file_name} is not a CSV file.")

        with open(file_name_full, 'r') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not validation_func(
                    reader.fieldnames):
                raise ValueError(
                    f"The file {file_name} has an invalid format.")
    return True


def validate_folders(input_folder: str,
                     output_folder: str,
                     is_new_format: bool,
                     is_simple: bool = False) -> bool:
    """
    This function will validate the input and output folders. The input folder must
    contain csv files with the format defined in the _validate_input_file_format
    function. The output folder must contain csv files with the format defined in
    the _validate_output_file_format function.
    
    :param input_folder: The path to the folder with the input data
    :is_new_format: A boolean flag to indicate if the new format is used
    :param output_folder: The path to the folder with the expected coupons
    :param is_simple: A boolean flag to indicate if the simple format is used
    :return: True if the folders are valid, False otherwise
    """

    valid_input: bool = _validate_folder(input_folder,
                                         _validate_input_file_format)

    if is_new_format:
        return valid_input and _validate_folder(
            output_folder, _validate_output_file_new_format, is_simple)

    return valid_input and _validate_folder(output_folder,
                                            _validate_output_file_format)


def get_default_datasets() -> Tuple[str, str]:
    """
    This function will get the default datasets from Google Drive and return the
    paths to the input and expected folders. The function will create the folders
    if they do not exist. If the folders already exist, the function will delete the
    files inside them and replace them with the default datasets.
    
    :return: A tuple with the paths to the input and expected folders
    """

    # Default paths to the input and expected folders
    input_folder = os.path.join(os.getcwd(), "input")
    expected_folder = os.path.join(os.getcwd(), "expected")

    # Create the folders; if they already exist, delete the files and folders inside them
    for folder in [input_folder, expected_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    os.remove(os.path.join(root, file))
                for subdir in dirs:
                    shutil.rmtree(os.path.join(root, subdir))

    # Get the default datasets from google drive
    tools_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../', 'tools'))
    sys.path.append(tools_path)
    script_path = os.path.join(tools_path, 'data_load.py')

    try:
        subprocess.run(['python3', script_path, 'coupons_1'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the script: {e}")
    except FileNotFoundError:
        print(
            "The data_load.py file was not found. Ensure the path is correct.")

    # Get the path to the datasets folder
    datasets_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../',
                     'datasets/coupons_1'))

    # Copy the files from the datasets folder to the input and expected folders
    for directory in os.listdir(datasets_path):
        if not os.path.isdir(os.path.join(datasets_path, directory)):
            continue

        if directory in INCORRECT_DATASETS:
            continue

        sub_input_folder = os.path.join(input_folder, directory)
        sub_expected_folder = os.path.join(expected_folder, directory)

        os.makedirs(sub_input_folder, exist_ok=True)
        os.makedirs(sub_expected_folder, exist_ok=True)

        for file in os.listdir(os.path.join(datasets_path, directory)):
            # One of the files has a typo in the name hence the check
            if "coupons" in file.lower() or "cupons" in file.lower():
                target = os.path.join(sub_expected_folder, file)
            elif "content_generic" in file.lower():
                target = os.path.join(sub_input_folder, file)
            else:
                continue

            shutil.copyfile(os.path.join(datasets_path, directory, file),
                            target)

    return input_folder, expected_folder
