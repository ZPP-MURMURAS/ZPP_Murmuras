from src.coupon_selecting_alg import proto_coupons_from_frame, parse_proto_coupon
from bert_inference.main import prepare_string_list, prepare_csv, map_strings_back_to_csv
from transformers import pipeline

csv_path = 'data/ner_data.csv'
model_checkpoint = 'zpp-murmuras/bert_multiling_cased_test_data_test_1'

bert_input = prepare_csv(csv_path)

token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple")

map_strings_back_to_csv(bert_input, csv_path, token_classifier)
