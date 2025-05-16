import numpy as np
import argparse
import subprocess
import difflib as diff
import json
import logging
import importlib
import copy
import os
import time
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from datasets.dataset_dict import DatasetDict
from dataclasses import dataclass, field
from typing import List, Union, Dict, Tuple, Callable


@dataclass()
class Coupon:
    """
    Class representing a coupon.
    """
    product_name: str
    discount_text: str
    validity_text: str
    activation_text: str


# Weights for the coupons
NAME_WEIGHT = 0.4
DISCOUNT_WEIGHT = 0.3
VALIDITY_WEIGHT = 0.2
ACTIVATION_WEIGHT = 0.1

# Column names for the output data
OUT_COL_PRODUCT = 'product_name'
OUT_COL_DISCOUNT = 'discount_text'
OUT_COL_VALIDITY = 'valid_until'
OUT_COL_ACTIVATION = 'activation_text'

# Columns of the dataset
INPUT = 'Context'
OUTPUT = 'Response'


def load_pipeline(config: Dict[str, any]) -> Tuple[Callable, List[any], Dict[str, any]]:
    """
    Loads a pipeline from a config.

    :param config: pipeline config
    :return: a tuple with the pipeline function, its positional arguments and its keyword arguments
    """

    module = importlib.import_module(config["module"])
    func = getattr(module, config["function"])
    args = config.get("args", [])
    kwargs = config.get("kwargs", {})
    return func, args, kwargs


def get_coupons(json_str: str, source_name: str) -> List[Coupon]:
    """
    Parses a JSON string that contains coupons. 
    
    :param json_str: A json string that contains the coupons
    :param source_name: The name of the source of the coupons (needed for logging)
    :return: A list of Coupon objects read from the file
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
        expected_keys = [OUT_COL_PRODUCT, OUT_COL_DISCOUNT, OUT_COL_VALIDITY, OUT_COL_ACTIVATION]

        if not isinstance(item, dict):
            logging.warning(source_info + f"Item {item} is not a dictionary.")
            continue

        if set(item.keys()) - set(expected_keys):
            logging.warning(source_info + f"Item {item} contains unexpected keys.")

        coupon = Coupon(product_name=item.get(OUT_COL_PRODUCT, ''),
                        discount_text=item.get(OUT_COL_DISCOUNT, ''),
                        validity_text=item.get(OUT_COL_VALIDITY, ''),
                        activation_text=item.get(OUT_COL_ACTIVATION, ''))

        coupon.product_name = coupon.product_name if coupon.product_name is not None else ''
        coupon.discount_text = coupon.discount_text if coupon.discount_text is not None else ''
        coupon.validity_text = coupon.validity_text if coupon.validity_text is not None else ''
        coupon.activation_text = coupon.activation_text if coupon.activation_text is not None else ''

        if type(coupon.product_name) is not str:
            logging.warning(source_info + f"Product name is not a string or null in item {item}.")
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

    return coupons


def compare_coupons(coupon_1: Coupon,
                    coupon_2: Coupon) -> float:
    """
    Compares two coupons and return a float value that represents the similarity 
    between the two coupons. The higher the value, the more similar the coupons.

    :param coupon_1, coupon_2: The first and second coupons to compare
    :return: A float value that represents the similarity between the two coupons
    """

    name_ratio = diff.SequenceMatcher(a=coupon_1.product_name,
                                      b=coupon_2.product_name).ratio()
    discount_ratio = diff.SequenceMatcher(a=coupon_1.discount_text,
                                          b=coupon_2.discount_text).ratio()
    validity_ratio = diff.SequenceMatcher(a=coupon_1.validity_text,
                                          b=coupon_2.validity_text).ratio()
    activation_ratio = diff.SequenceMatcher(a=coupon_1.activation_text,
                                            b=coupon_2.activation_text).ratio()

    dead_weight = 0.0
    num_empty = 0

    if coupon_1.product_name == '' and coupon_2.product_name == '':
        name_ratio = 0.0
        dead_weight += NAME_WEIGHT
        num_empty += 1

    if coupon_1.discount_text == '' and coupon_2.discount_text == '':
        discount_ratio = 0.0
        dead_weight += DISCOUNT_WEIGHT
        num_empty += 1

    if coupon_1.validity_text == '' and coupon_2.validity_text == '':
        validity_ratio = 0.0
        dead_weight += VALIDITY_WEIGHT
        num_empty += 1

    if coupon_1.activation_text == '' and coupon_2.activation_text == '':
        activation_ratio = 0.0
        dead_weight += ACTIVATION_WEIGHT
        num_empty += 1

    if num_empty == 4:
        rescaled_sim = 1.0
    else:
        base_sim = (name_ratio * NAME_WEIGHT) + (discount_ratio * DISCOUNT_WEIGHT) + \
                   (validity_ratio * VALIDITY_WEIGHT) + (activation_ratio * ACTIVATION_WEIGHT)

        rescaled_sim = base_sim / (1.0 - dead_weight)
    
    return np.clip(rescaled_sim, 0.0, 1.0)


def compute_similarity_matrix(expected_coupons: List[Coupon],
                              generated_coupons: List[Coupon]) -> np.ndarray:
    """
    Computes the similarity matrix between the expected and generated coupons. 
    The similarity matrix has the same number of rows as the expected coupons and the same 
    number of columns as the generated coupons. The value in the i-th row and j-th column of the
    matrix will represent the similarity between the i-th expected coupon and the j-th generated coupon.

    :param expected_coupons: A list of Coupon objects that represent the expected coupons
    :param generated_coupons: A list of Coupon objects that represent the generated coupons
    :return: A numpy array that represents the similarity matrix between the coupons
    """

    similarity_matrix = np.zeros((len(expected_coupons), len(generated_coupons)))
    for i, expected_coupon in enumerate(expected_coupons):
        for j, generated_coupon in enumerate(generated_coupons):
            similarity_matrix[i, j] = compare_coupons(expected_coupon, generated_coupon)

    return similarity_matrix


def greedy_matching(similarity_matrix: np.ndarray, threshold: float) -> tuple[List[float], int, int]:
    """
    Performs a greedy matching between the expected and generated coupons using the similarity matrix. 
    For more details on the algorithm, see the README.md file.

    :param similarity_matrix: A numpy array that represents the similarity matrix between the coupons
    :param threshold: A float value that represents the minimum similarity to match two coupons
    :return: A tuple with a list of similarities of the matched coupons, 
    the number of unmatched coupons in the expected list, 
    and the number of unmatched coupons in the generated list
    """

    similarities = []
    expected_matched = [False] * similarity_matrix.shape[0]
    generated_matched = [False] * similarity_matrix.shape[1]

    rows, cols = np.indices(similarity_matrix.shape)
    compressed_vals = similarity_matrix.ravel()
    compressed_coords = np.column_stack((rows.ravel(), cols.ravel()))
    argsort_inds = np.argsort(-compressed_vals)

    sort_desc_vals = compressed_vals[argsort_inds]
    sort_desc_coords = compressed_coords[argsort_inds]

    for val, coords in zip(sort_desc_vals, sort_desc_coords, strict=True):
        if val < threshold:
            break

        x = coords[0]
        y = coords[1]
        if not expected_matched[x] and not generated_matched[y]:
            expected_matched[x] = True
            generated_matched[y] = True
            similarities.append(val)

    return similarities, expected_matched.count(False), generated_matched.count(False)


def compute_similarities(expected_coupons: List[Coupon],
                         generated_coupons: List[Coupon],
                         threshold: float) -> tuple[List[float], int, int]:
    """
    Computes similarities between the expected and generated coupons.
    For more details on the algorithm, see the README.md file.
    
    :param expected_coupons: A list of Coupon objects that represent the expected coupons
    :param generated_coupons: A list of Coupon objects that represent the generated coupons
    :param threshold: A float value that represents the minimum similarity to match two coupons
    :return: A tuple with the list of similarities between the matched coupons,
    the number of missed coupons, and the number of hallucinated coupons
    """

    similarity_matrix = compute_similarity_matrix(expected_coupons, generated_coupons)
    similarities, missed, hallucinated = greedy_matching(similarity_matrix, threshold)

    return similarities, missed, hallucinated


def init_new_logger(log_file: str) -> None:
    """
    Sets a new log file.

    :param log_file: path to the new log file
    """

    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    new_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    new_handler.setFormatter(formatter)

    logger.addHandler(new_handler)
    logger.setLevel(logging.INFO)


def benchmark_pipeline(pipeline_name: str, config: Dict[str, any], cache_dir: str, log_dir: str) -> Dict[str, any]:
    """
    Benchmarks a pipeline based on the given config.

    :param pipeline_name: name of the experiment
    :param config: pipeline config
    :param cache_dir: the directory for dataset caching
    :param log_dir: the directory to store the logs
    :return: a dictionary with the pipeline config and benchmarking results
    """

    dataset_name = config["dataset_name"]
    splits = config["splits"]
    threshold = config["threshold"]
    func, args, kwargs = load_pipeline(config)

    res = copy.deepcopy(config)
    res["expected"] = {}
    res["generated"] = {}
    res["matched"] = {}
    res["generation_time"] = {}

    for split_name, ds_split in splits.items():
        init_new_logger(os.path.join(log_dir, f"{pipeline_name}-{split_name}.log"))

        dataset = load_dataset(dataset_name, split=ds_split, cache_dir=cache_dir)

        input = [entry[INPUT] for entry in dataset]
        expected = [entry[OUTPUT] for entry in dataset]

        start = time.perf_counter()
        generated = func(input_data=input, *args, **kwargs)
        end = time.perf_counter()
        generation_time = end - start

        if len(expected) != len(generated):
            raise ValueError(f"Expected {len(expected)} coupons, but got {len(generated)} coupons.")

        total_expected = 0
        total_generated = 0
        total_matched = 0
        for idx, (exp, gen) in tqdm(enumerate(zip(expected, generated, strict=True)), desc="Processing entries"):
            expected_coupons = get_coupons(exp, "Expected output")

            generated_coupons = get_coupons(gen, "Generated output")

            similarities, missed, hallucinated = compute_similarities(expected_coupons, 
                                                                      generated_coupons,
                                                                      threshold)

            total_expected += len(expected_coupons)
            total_generated += len(generated_coupons)
            total_matched += len(similarities)

            entry_info = f"Entry {idx + 1} - "
            logging.info(entry_info + f"Expected: {len(expected_coupons)}")
            logging.info(entry_info + f"Generated: {len(generated_coupons)}")
            logging.info(entry_info + f"Matched: {len(similarities)}")

        logging.info(f"Total number of expected coupons: {total_expected}")
        logging.info(f"Total number of generated coupons: {total_generated}")
        logging.info(f"Total number of matched coupons: {total_matched}")
        logging.info(f"Generation time: {generation_time:.4f} seconds")

        res["expected"][split_name] = total_expected
        res["generated"][split_name] = total_generated
        res["matched"][split_name] = total_matched
        res["generation_time"][split_name] = generation_time

    return res


def load_checkpoint(checkpoint_file: str) -> Dict[str, any]:
    """
    Loads checkpointed data.

    :param checkpoint_file: the path to the checkpoint file
    :return: the checkpointed data
    """

    try: 
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}

    return data



def parse_args() -> argparse.Namespace:
    """
    Parses the input arguments.

    :return: The parsed input arguments
    """

    parser = argparse.ArgumentParser(description='Benchmarking script')
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        required=True,
                        help='Path to the config file'
    )
    parser.add_argument('-o',
                        '--output_file',
                        type=str,
                        required=True,
                        help='Path to the output file'
    )
    parser.add_argument('-d',
                        '--dataset_cache_dir',
                        type=str,
                        default='./datasets',
                        help='Path to the dataset cache directory'
    )
    parser.add_argument('-l',
                        '--log_dir',
                        type=str,
                        default='./logs',
                        help='Path to the log directory'
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    configs = json.load(open(args.config_path))
    output_file = args.output_file
    dataset_cache_dir = args.dataset_cache_dir
    log_dir = args.log_dir

    os.makedirs(log_dir, exist_ok=True)

    results = load_checkpoint(output_file)

    for pipeline_name, config in tqdm(configs.items(), desc="Processing experiments"):
        if pipeline_name in results:
            continue

        result = benchmark_pipeline(pipeline_name, config, dataset_cache_dir, log_dir)
        results[pipeline_name] = result

        with open(output_file, "w") as f:
            json.dump(results, f)
