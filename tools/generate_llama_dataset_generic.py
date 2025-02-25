import os
import pandas as pd
import huggingface_hub
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
import src.llama_dataset_generation.input_parser as ip
import src.llama_dataset_generation.ground_truth_parser as gtp
import src.llama_dataset_generation.datasetter as dt
from transformers import AutoTokenizer
import argparse

HF_API_KEY = os.getenv('HF_API_KEY')
OPEN_API_KEY = os.getenv('OPEN_API_KEY')

def parse_args():
    parser = argparse.ArgumentParser(description="Process input parameters.")
    parser.add_argument("--input_path", required=True, help="Path to input data")
    parser.add_argument("--ground_truth_path", required=True, help="Path to ground truth data")
    parser.add_argument("--model_name", required=False, help="Model name to use")
    parser.add_argument("--map_func", required=True, help=
        "Map function to use", choices=[f.__name__ for f in dt.MAP_FUNCTIONS]
    )
    parser.add_argument("--hf_name", required=True, help="Name on hf to save dataset under")
    parser.add_argument("--no_ai", required=False, help="Whether to generate new type of dataset without chatgpt.", action="store_true")

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    INPUT_PATH = args.input_path
    GROUND_TRUTH_PATH = args.ground_truth_path
    MODEL_NAME = args.model_name

    MAP_FUN = None
    for map_func in dt.MAP_FUNCTIONS:
        if map_func.__name__ == args.map_func:
            MAP_FUN = map_func
    assert MAP_FUN is not None

    HF_NAME = args.hf_name
    no_ai = args.no_ai

    client = None
    if not no_ai:
        client = gtp.init_client(OPEN_API_KEY)
    huggingface_hub.login(HF_API_KEY)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    cg_concat_combined = {}
    gtd_combined = {}

    for i, (file_content, file_coupons) in enumerate(zip(os.listdir(os.fsencode(INPUT_PATH)),
                                          os.listdir(os.fsencode(GROUND_TRUTH_PATH)))):
        coupons_path = os.path.join(GROUND_TRUTH_PATH, os.fsdecode(file_coupons))
        content_path = os.path.join(INPUT_PATH, os.fsdecode(file_content))
        discount_list = gtp.load_coupons(coupons_path)
        coupons_frame = pd.read_csv(coupons_path)
        cg_concat = ip.prepare_input_data(content_path)
        if not no_ai:
            # This will perform to OpenAI API, and overwrite ground_truth.json.
            gtd = gtp.extract_discounts(discount_list, client)

            # Said json is produced by the gtp.extract_discounts call and stored under
            # the GROUND_TRUTH_JSON_PATH.
            gtd_dict = gtp.prepare_ground_truth_data(gtd, coupons_frame)
        else:
            gtd_dict = gtp.prepare_ground_truth_data_no_ai(coupons_frame)
        for k in gtd_dict.keys():
            gtd_combined[f"{i}_{k}"] = gtd_dict[k]
        for k in cg_concat.keys():
            cg_concat_combined[f"{i}_{k}"] = cg_concat[k]

    training_df = ip.create_training_df(cg_concat_combined, gtd_combined)

    # Global var of the dt lib.
    dt.EOS_TOKEN = tokenizer.eos_token
    training_data = dt.run_mapping(training_df, MAP_FUN)
    training_data = training_data.train_test_split(0.1)
    training_data.push_to_hub('zpp-murmuras/' + HF_NAME, private=True)
