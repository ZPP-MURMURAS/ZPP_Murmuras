import os
import pandas as pd
import sys
import json
import src.llama_dataset_generation.input_parser as ip
import src.llama_dataset_generation.ground_truth_parser as gtp
import src.llama_dataset_generation.datasetter as dt
from transformers import AutoTokenizer
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)


HF_API_KEY = os.getenv('HF_API_KEY')
OPEN_API_KEY = os.getenv('OPEN_API_KEY')
MODEL_NAME = 'meta-llama/Llama-3.2-1B'


def parse_args():
    parser = argparse.ArgumentParser(description="Process input parameters.")
    parser.add_argument("--data_path", required=True, help="path to directory with content_generic and coupon files")
    parser.add_argument("--map_func", required=True, help=
        "Map function to use", choices=[f.__name__ for f in dt.MAP_FUNCTIONS]
    )
    parser.add_argument("--hf_name", required=True, help="Name on hf to save dataset under")
    parser.add_argument("--no_ai", required=False, help="Whether to generate new type of dataset without chatgpt.", action="store_true")

    args = parser.parse_args()
    return args.data_path, args.map_func, args.hf_name, args.no_ai


def create_dataset(content_path, coupons_path):
    discount_list = gtp.load_coupons(coupons_path)
    coupons_frame = pd.read_csv(coupons_path)
    cg_concat = ip.prepare_input_data(content_path)
    if not no_ai:
        # This will perform to OpenAI API, and overwrite ground_truth.json.
        gtd = json.loads(gtp.extract_discounts(discount_list, client))

        # Said json is produced by the gtp.extract_discounts call and stored under
        # the GROUND_TRUTH_JSON_PATH.
        gtd_dict = gtp.prepare_ground_truth_data(gtd, coupons_frame)
    else:
        gtd_dict = gtp.prepare_ground_truth_data_no_ai(coupons_frame)

    training_df = ip.create_training_df(cg_concat, gtd_dict)
    dt.EOS_TOKEN = tokenizer.eos_token
    return dt.run_mapping(training_df, MAP_FUN)


def traverse_files(data_dir: str):
    content_path, coupons_path = None, None
    dts_list, dts_names_list = [], []
    for file in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, file)):
            dts, dts_names = traverse_files(os.path.join(data_dir, file))
            dts_list.extend(dts)
            dts_names_list.extend(dts_names)
        else:
            if "content_generic" in file and file.endswith(".csv"):
                content_path = os.path.join(data_dir, file)
            elif "coupons" in file and file.endswith(".csv"):
                coupons_path = os.path.join(data_dir, file)

    if content_path is not None and coupons_path is not None:
        new_dts = create_dataset(content_path, coupons_path)
        dts_list.append(new_dts)
        dts_names_list.append(os.path.dirname(content_path))
    return dts_list, dts_names_list


if __name__ == '__main__':
    DATA_PATH, MAP_FUN_NAME, HF_NAME, no_ai = parse_args()

    MAP_FUN = None
    for map_func in dt.MAP_FUNCTIONS:
        if map_func.__name__ == MAP_FUN_NAME:
            MAP_FUN = map_func
    assert MAP_FUN is not None

    client = None
    if not no_ai:
        client = gtp.init_client(OPEN_API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_API_KEY)

    traverse_files(DATA_PATH)

    # cg_concat_combined = {}
    # gtd_combined = {}
    # for i, (file_content, file_coupons) in enumerate(zip(os.listdir(os.fsencode(INPUT_PATH)),
    #                                       os.listdir(os.fsencode(GROUND_TRUTH_PATH)))):
    #     coupons_path = os.path.join(GROUND_TRUTH_PATH, os.fsdecode(file_coupons))
    #     content_path = os.path.join(INPUT_PATH, os.fsdecode(file_content))
    #     discount_list = gtp.load_coupons(coupons_path)
    #     coupons_frame = pd.read_csv(coupons_path)
    #     cg_concat = ip.prepare_input_data(content_path)
    #     if not no_ai:
    #         # This will perform to OpenAI API, and overwrite ground_truth.json.
    #         gtd = json.loads(gtp.extract_discounts(discount_list, client))
    #
    #         # Said json is produced by the gtp.extract_discounts call and stored under
    #         # the GROUND_TRUTH_JSON_PATH.
    #         gtd_dict = gtp.prepare_ground_truth_data(gtd, coupons_frame)
    #     else:
    #         gtd_dict = gtp.prepare_ground_truth_data_no_ai(coupons_frame)
    #     for k in gtd_dict.keys():
    #         gtd_combined[f"{i}_{k}"] = gtd_dict[k]
    #     for k in cg_concat.keys():
    #         cg_concat_combined[f"{i}_{k}"] = cg_concat[k]
    #
    # training_df = ip.create_training_df(cg_concat_combined, gtd_combined)
    #
    # # Global var of the dt lib.
    # dt.EOS_TOKEN = tokenizer.eos_token
    # training_data = dt.run_mapping(training_df, MAP_FUN)
    # training_data = training_data.train_test_split(0.1)
    # training_data.push_to_hub('zpp-murmuras/' + HF_NAME, private=True)
