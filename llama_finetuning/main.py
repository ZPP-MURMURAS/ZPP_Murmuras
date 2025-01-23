import pandas as pd
import input_parser as ip
import ground_truth_parser as gtp

OPEN_API_KEY = 'OPEN_API_KEY'
INPUT_PATH = 'FILE_PATH'
GROUND_TRUTH_PATH = 'FILE_PATH'
GROUND_TRUTH_JSON_PATH = 'FILE_PATH'

if __name__ == '__main__':
    lidl_discount_list = gtp.load_coupons(GROUND_TRUTH_PATH)
    lidl_coupons = pd.read_csv(GROUND_TRUTH_PATH)

    client = gtp.init_client(OPEN_API_KEY)
    cg_concat = ip.prepare_input_data(INPUT_PATH)
    #gtd = gtp.extract_discounts(lidl_coupons, client)

    gtd_json = gtp.load_coupons_from_json(path=GROUND_TRUTH_JSON_PATH)
    prepared_gtd = gtp.prepare_ground_truth_data(gtd_json, lidl_coupons)
    gtd_dict = gtp.ground_truth_to_dict(prepared_gtd)

    training_df = ip.create_training_df(cg_concat, gtd_dict)




