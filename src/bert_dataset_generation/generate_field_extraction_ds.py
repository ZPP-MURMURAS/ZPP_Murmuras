from os import getenv
import json
import sys
import random
from typing import List, Tuple, Dict, Optional, Callable, TypedDict
import datasets
import pandas as pd
from datasets import DatasetDict, Dataset
from huggingface_hub import login
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

# Constants
TAG_B_PRODUCT = 'B-PRODUCT-NAME'  # begin product name
TAG_I_PRODUCT = 'I-PRODUCT-NAME'  # inside product name
TAG_B_DISCOUNT = 'B-DISCOUNT-TEXT'  # begin discount text
TAG_I_DISCOUNT = 'I-DISCOUNT-TEXT'  # inside discount text
TAG_B_VALIDITY = 'B-VALIDITY-TEXT'  # begin validity text
TAG_I_VALIDITY = 'I-VALIDITY-TEXT'  # inside validity text
TAG_B_ACTIVATION = 'B-ACTIVATION-TEXT'  # begin activation text
TAG_I_ACTIVATION = 'I-ACTIVATION-TEXT'  # inside activation text
TAG_UNKNOWN = 'O'  # unknown tag

COL_PRODUCT = 'product_text'  # column from coupons frame with product text
COL_DISCOUNT_TEXT = 'discount_text'  # column from coupons frame with discount text
COL_DISCOUNT_DETAILS = 'discount_details'  # column from coupons frame with discount details
COL_VALIDITY = 'validity_text'  # column from coupons frame with validity text
COL_ACTIVATION = 'activation_text'  # column from coupons frame with activation text

# target labels
LABELS = datasets.ClassLabel(names=[TAG_UNKNOWN, TAG_B_PRODUCT, 
    TAG_I_PRODUCT, TAG_B_DISCOUNT, TAG_I_DISCOUNT, TAG_B_VALIDITY, 
    TAG_I_VALIDITY, TAG_B_ACTIVATION, TAG_I_ACTIVATION])
LBL_UNK = LABELS.str2int(TAG_UNKNOWN)
LBL_P = LABELS.str2int(TAG_B_PRODUCT)
LBL_D = LABELS.str2int(TAG_B_DISCOUNT)
LBL_V = LABELS.str2int(TAG_B_VALIDITY)
LBL_A = LABELS.str2int(TAG_B_ACTIVATION)


def publish_to_hub(samples: List[List[Tuple[List[str], List[int]]]], save_name: str, apikey: str, new_repo: bool, custom_splits: Optional[List[str]]) -> None:
    """
    Creates dataset out of list of lists (one for each pair of frames) of pairs of words and labels and pushes it to HF Hub.
    """
    features = datasets.Features({
        "texts": datasets.Sequence(datasets.Value("string")),
        "labels": datasets.Sequence(LABELS)
    })

    if custom_splits is None:
        # Convert samples into a dictionary
        texts = [sample[0] for samples_pack in samples for sample in samples_pack]
        labels = [sample[1] for samples_pack in samples for sample in samples_pack]

        # Initial train/test split (80% train, 20% temp)
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Split temp set into validation (10%) and test (10%)
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts, temp_labels, test_size=0.5, random_state=42
        )

        # Create Dataset objects
        dataset_dict = DatasetDict({
            "train": Dataset.from_dict({"texts": train_texts, "labels": train_labels}, features=features),
            "validation": Dataset.from_dict({"texts": val_texts, "labels": val_labels}, features=features),
            "test": Dataset.from_dict({"texts": test_texts, "labels": test_labels}, features=features)
        })
    else:
        grouped = {}
        for name, samples_pack in zip(custom_splits, samples, strict=True):
            if name not in grouped:
                grouped[name] = [[], []]
            texts = [sample[0] for sample in samples_pack]
            labels = [sample[1] for sample in samples_pack]
            grouped[name][0].extend(texts)
            grouped[name][1].extend(labels)
        for k, v in grouped.items():
            grouped[k] = Dataset.from_dict({"texts": v[0], "labels": v[1]}, features=features)
        dataset_dict = DatasetDict(grouped)

    login(token=apikey)
    if new_repo:
        api = HfApi()
        api.create_repo(repo_id=save_name, repo_type="dataset", private=True)

    dataset_dict.push_to_hub(save_name, private=True)


def __samples_from_entry(coupons_frame: pd.DataFrame, seed: int) -> List[Tuple[List[str], List[int]]]:
    """
    Extracts samples from a single entry in the config file. This strategy is based on DM dataset from coupons big.

    :param coupons_frame: pandas DataFrame with coupons
    :param seed: random seed
    :return: list of samples
    """
    SHUFFLE_PROB = 0.4
    DISCOUNT_DROP_PROB = 0.1
    VALIDITY_DROP_PROB = 0.1
    ACTIVATION_DROP_PROB = 0.1

    random.seed(seed)

    samples = []
    coupons = coupons_frame.replace('', None).dropna(subset=[COL_PRODUCT]) # drop rows with empty product text
    coupons.drop_duplicates(subset=[COL_PRODUCT, COL_DISCOUNT_TEXT, COL_DISCOUNT_DETAILS, COL_VALIDITY, COL_ACTIVATION], inplace=True) # drop duplicates
    for _, row in coupons.iterrows():
        product_text = row[COL_PRODUCT]
        discount_text = row[COL_DISCOUNT_TEXT]
        discount_details = row[COL_DISCOUNT_DETAILS]
        validity_text = row[COL_VALIDITY]
        activation_text = row[COL_ACTIVATION]

        PRODUCT_IDX, DISCOUNT_IDX, VALIDITY_IDX, ACTIVATION_IDX = 0, 1, 2, 3
        order = [PRODUCT_IDX, DISCOUNT_IDX, VALIDITY_IDX, ACTIVATION_IDX]

        if random.random() < SHUFFLE_PROB:
            # Shuffle the order of the sections
            order = random.sample(order, len(order))

        all_texts = []
        all_labels = []
        for idx in order:
            if idx == PRODUCT_IDX:
                all_texts.append(product_text)
                all_labels.append(LBL_P)
            elif idx == DISCOUNT_IDX and random.random() > DISCOUNT_DROP_PROB:
                if not pd.isnull(discount_details):
                    all_texts.append(discount_details)
                    all_labels.append(LBL_UNK)
                if not pd.isnull(discount_text):
                    all_texts.append(discount_text)
                    all_labels.append(LBL_D)
            elif idx == VALIDITY_IDX and random.random() > VALIDITY_DROP_PROB and not pd.isnull(validity_text):
                all_texts.append(validity_text)
                all_labels.append(LBL_V)
            elif idx == ACTIVATION_IDX and random.random() > ACTIVATION_DROP_PROB and not pd.isnull(activation_text):
                all_texts.append(activation_text)
                all_labels.append(LBL_A)

        samples.append((all_texts, all_labels))

    return samples


if __name__ == '__main__':
    HF_HUB_KEY = getenv('HF_HUB_KEY')
    assert len(sys.argv) == 5, f"usage: {sys.argv[0]} <config_path> <ds_name> <create_repo: y/n> <custom_split: y/n>"
    config_path = sys.argv[1]
    ds_name = sys.argv[2]
    create_repo = sys.argv[3]
    custom_split = sys.argv[4]
    if (create_repo != 'y') and (create_repo != 'n'):
        print("create_repo must be either 'y' or 'n'")
        exit(1)
    if custom_split not in ('y', 'n'):
        print("custom_split must be either 'y', 'n'")
        exit(1)
    config = json.load(open(config_path))
    try:
        coupon_paths = list(entry['coupons'] for entry in config['frames'])
        formats = list([entry['format'] for entry in config['frames']])
        splits = list([entry['split'] for entry in config['frames']]) if custom_split == 'y' else None
    except KeyError as e:
        print(
            f"KeyError: {e}, config should be in format {{\"json_format\": true,\"frames\": "
            f"[{{\"coupons\": path, \"content\": path, \"format\": 1, \"split\": \"obligatory if custom_split=y\"}},...]}}")
        exit(1)

    examples = []
    for fmt, coupon_path in zip(formats, coupon_paths, strict=True):
        coupons_frame = pd.read_csv(coupon_path)
        examples.append(__samples_from_entry(coupons_frame, seed=42))

    publish_to_hub(examples, f"zpp-murmuras/{ds_name}", HF_HUB_KEY, create_repo == 'y', splits)
