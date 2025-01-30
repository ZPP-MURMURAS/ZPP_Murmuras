import os
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

    # This will perform to OpenAI API, and overwrite ground_truth.json.
    gtd = gtp.extract_discounts(lidl_discount_list, client)

    # Said json is produced by the gtp.extract_discounts call and stored under
    # the GROUND_TRUTH_JSON_PATH.
    gtd_json = gtp.load_coupons_from_json(path=GROUND_TRUTH_JSON_PATH)
    gtd_dict = gtp.prepare_ground_truth_data(gtd_json, lidl_coupons)

    training_df = ip.create_training_df(cg_concat, gtd_dict)

    # Global var of the dt lib.
    dt.EOS_TOKEN = tokenizer.eos_token
    # This loop creates finished datasets, and then pushes it to the hugging face hub.
    for map_func in dt.MAP_FUNCTIONS:
        training_data = dt.run_mapping(training_df, map_func)
        training_data.push_to_hub('zpp-murmuras/' + map_func.__name__, private=True)