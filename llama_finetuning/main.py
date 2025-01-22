import pandas as pd
import input_parser as ip
import ground_truth_parser as gtp

OPEN_API_KEY = 'TODO:API_KEY'

if __name__ == '__main__':
    lidl_content_generic = pd.read_csv('TODO:PATH')
    lidl_coupons = pd.read_csv('TODO:PATH')

    client = gtp.init_client(OPEN_API_KEY)
    cg_concat = ip.prepare_input_data(lidl_content_generic)
    gtd = gtp.extract_discounts(lidl_coupons, client)

    prepared_gtd = gtp.prepare_ground_truth_data(gtd, lidl_coupons)
    gtd_dict = gtp.ground_truth_to_dict(prepared_gtd)

    training_df = ip.create_training_df(cg_concat, gtd_dict)




