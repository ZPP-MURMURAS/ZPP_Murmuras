import os

import modal
import pandas as pd

import input_parser as ip
import ground_truth_parser as gtp
import datasetter as dt
from transformers import AutoTokenizer

OPEN_API_KEY = os.getenv('OPEN_API_KEY')
INPUT_PATH = os.getenv('INPUT_PATH')
GROUND_TRUTH_PATH = os.getenv('GROUND_TRUTH_PATH')
GROUND_TRUTH_JSON_PATH = os.getenv('GROUND_TRUTH_JSON_PATH')
MODEL_NAME = os.getenv('MODEL_NAME')

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lidl_discount_list = gtp.load_coupons(GROUND_TRUTH_PATH)
    lidl_coupons = pd.read_csv(GROUND_TRUTH_PATH)

    client = gtp.init_client(OPEN_API_KEY)
    cg_concat = ip.prepare_input_data(INPUT_PATH)
    gtd = gtp.extract_discounts(lidl_discount_list, client)

    gtd_json = gtp.load_coupons_from_json(path=GROUND_TRUTH_JSON_PATH)
    prepared_gtd = gtp.prepare_ground_truth_data(gtd_json, lidl_coupons)
    gtd_dict = gtp.ground_truth_to_dict(prepared_gtd)

    training_df = ip.create_training_df(cg_concat, gtd_dict)

    dt.EOS_TOKEN = tokenizer.eos_token
    training_data = dt.run_mapping(training_df, dt.one_input_one_output_wrequest)

    training_data.push_to_hub('zpp-murmuras/test_llama_data', private=True)