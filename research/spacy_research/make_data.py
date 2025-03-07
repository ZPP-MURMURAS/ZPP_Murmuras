import csv
import datetime
import json
from collections import defaultdict
import re
import sys
import os

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_PATH, "../../")))
# from src.pipeline_benchmark.io_utils import get_default_datasets, validate_folders, Coupon, CouponSimple, get_expected_coupons
# from src.llama_dataset_generation.input_parser import 

os.chdir(CURRENT_PATH)

input_csv = "input.csv" # content generic from the 0th batch of data
expected_json = "expected.json" # expected output from the 0th batch of data
output_json = "output.json" # data for spacy

timestamp_texts = defaultdict(list)

with open(input_csv, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)  
    
    for row in reader:
        # Append to the list of texts for the given timestamp if the text is visible and the text is not empty
        if row["is_visible"].lower() == "true":  
            text_value = row["text"].strip() 
            timestamp = row["seen_timestamp"]  
            
            if text_value:  
                timestamp_texts[timestamp].append(text_value)

with open(expected_json, 'r') as jsonfile:
    expected = json.load(jsonfile) # Entities of each coupon
    expected = expected["coupons"]

with open(output_json, 'w') as txtfile:
    data_to_write = []
    for timestamp, texts in timestamp_texts.items():
        timestamp_texts[timestamp] = " ".join(texts)

        text = timestamp_texts[timestamp]
        concat_dict = {}
        concat_dict['text'] = text
        concat_dict['entities'] = []
        seen_entities = set()
        
        for entity in expected: 
            date_regex = r'(0[1-9]|[12][0-9]|3[01])\.(0[1-9]|1[0-2])'
            matches = re.findall(date_regex, text)

            dates = []
            for match in matches:
                day = int(match[0])  
                month = int(match[1])  
                date = datetime.datetime(year=2024, month=int(month), day=int(day))
                dates.append(date)

            for key, value in entity.items(): 
                idx = text.find(value)
                entity = {}
                if idx != -1:
                    entity = {
                        "start": idx,
                        "end": idx + len(value),
                        "label": key
                    }

                if key == 'validity':
                    d = value.split('-')
                    date = datetime.datetime(year=int(d[0]), month=int(d[1]), day=int(d[2]))

                    if date in dates:
                        date_idx = dates.index(date)
                        original = matches[date_idx][0] + '.' + matches[date_idx][1]
                        idx = text.find(original)

                        if idx == -1:
                            continue

                        entity = {
                            "start": idx,
                            "end": idx + len(original),
                            "label": key
                        }
                if entity:
                    entity_tuple = (entity['start'], entity['end'])
                    if entity_tuple not in seen_entities:
                        concat_dict['entities'].append(entity)
                    seen_entities.add((entity['start'], entity['end']))

        data_to_write.append(concat_dict)

    json.dump(data_to_write, txtfile, indent=4)

print(f"Concatenated data saved to {output_json}")


