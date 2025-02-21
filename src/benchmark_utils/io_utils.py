import re
import csv
import os
import shutil
import subprocess
import sys

from typing import Optional, List, Tuple
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


def get_expected_coupons(file_path: Optional[str]) -> List[Coupon]:
    """
    This function will return a list of Coupon objects that represent the 
    expected coupons. This function is used to benchmark the pipeline.
    
    :param file_path: The path to the folder with the expected coupons in csv 
                    format (like in the Murmuras datasets)
    :return: A list of Coupon objects that represent the expected coupons
    """

    # No file path provided
    if file_path is None:
        return []

    expected_coupons = []

    for file_name in os.listdir(file_path):
        file_path_full = os.path.join(file_path, file_name)
        if os.path.isfile(file_path_full):
            with open(file_path_full, 'r') as file:
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


def validate_folders(input_folder: str, output_folder: str) -> bool:
    """
    This function will validate the input and output folders. The input folder must
    contain csv files with the format defined in the _validate_input_file_format
    function. The output folder must contain csv files with the format defined in
    the _validate_output_file_format function.
    
    :param input_folder: The path to the folder with the input data
    :param output_folder: The path to the folder with the expected coupons
    :return: True if the folders are valid, False otherwise
    """

    if not os.path.isdir(input_folder):
        raise NotADirectoryError(
            f"The input path {input_folder} is not a directory.")
    if not os.path.isdir(output_folder):
        raise NotADirectoryError(
            f"The output path {output_folder} is not a directory.")

    # Validate the output folder
    for file_name in os.listdir(output_folder):
        file_name_full = os.path.join(output_folder, file_name)
        if not os.path.isfile(file_name_full):
            raise NotADirectoryError(
                f"The output path {output_folder} is not a directory.")

        if not file_name.endswith('.csv'):
            raise ValueError(f"The file {file_name} is not a CSV file.")

        with open(file_name_full, 'r') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not _validate_output_file_format(
                    list(reader.fieldnames)):
                raise ValueError(
                    f"The file {file_name} has an invalid format.")

    # Validate the input folder
    for file_name in os.listdir(input_folder):
        file_name_full = os.path.join(input_folder, file_name)
        if not os.path.isfile(file_name_full):
            raise NotADirectoryError(
                f"The input path {input_folder} is not a directory.")

        if not file_name.endswith('.csv'):
            raise ValueError(f"The file {file_name} is not a CSV file.")

        with open(file_name_full, 'r') as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not _validate_input_file_format(
                    list(reader.fieldnames)):
                raise ValueError(
                    f"The file {file_name} has an invalid format.")

    return True


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
