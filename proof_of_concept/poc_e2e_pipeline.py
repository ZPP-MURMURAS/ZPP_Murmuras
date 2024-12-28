from postprocessing.coupon_selecting_alg import proto_coupons_from_frame, parse_proto_coupon
from bert_inference.main import prepare_csv, map_strings_back_to_csv
from transformers import pipeline
from postprocessing.constants import *
import json

csv_path = 'data.csv'
model_checkpoint = 'zpp-murmuras/bert_multiling_cased_test_data_test_1'

labels_mapping = {
    'PRICE': Label.PRICE,
    'PRODUCT': Label.PRODUCT_NAME,
    'DISCOUNT_PERCENTAGE': Label.PERCENT,
    'DATE': Label.DATE,
    'QUANTITY': Label.UNKNOWN,
    'N/A': Label.UNKNOWN
}

bert_input = prepare_csv(csv_path)

token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple")

df = map_strings_back_to_csv(bert_input, csv_path, token_classifier)
df.to_csv('output.csv')

results = []
for pc in proto_coupons_from_frame(df, 'ner_tags', labels_mapping=labels_mapping):
    results.append(parse_proto_coupon(pc))

print(json.dumps(results, indent=4))
