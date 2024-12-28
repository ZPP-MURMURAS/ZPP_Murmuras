from sys import argv
import json

import pandas as pd

import postprocessing_utils
from constants import Label
from coupon_selecting_alg import proto_coupons_from_frame, parse_proto_coupon
"""
Simple script demonstrating data flow.
Mostly for purposes of further integration work.
This script can perform one of the 2 test cases:
1. In the first case we load raw content from phone screen and associated coupons in mm
format. We then label text sections in content and run postprocessing on them.
2. In the second case we get data returned by LLM (real world scenario)
In both cases script simply prints extracted coupons.
"""

if __name__ == '__main__':
    HELP_MSG = (f'usage: {argv[0]} <test_no> <input_file> [input_file_2] (test_no is currently 1 or 2,'
                f' input_file_2 is file with extracted coupons from murmuras)')
    assert len(argv) in (3, 4), HELP_MSG
    frame_extracted = None # to bypass stupid IDE warnings
    try:
        test_no = int(argv[1])
        frame = pd.read_csv(argv[2])
        if test_no == 1:
            frame_extracted = pd.read_csv(argv[3])
    except (ValueError, FileNotFoundError) as e:
        print(HELP_MSG)
        raise e

    if test_no == 1:
        # test no assumes we work on data given to us by murmuras (already labeled but labels are in different file)
        rows_separated = postprocessing_utils.labeled_data_to_extra_csv_column(frame, frame_extracted)
        with_json_labels = postprocessing_utils.merge_subsequent_text_fields(rows_separated)
        proto_coupons = proto_coupons_from_frame(with_json_labels)

        result = []
        for pc in proto_coupons:
            result.append(parse_proto_coupon(pc))
        print(json.dumps(result, indent=4))

    if test_no == 2:
        proto_coupons = proto_coupons_from_frame(
            frame=frame,
            label_col='ner_tags',
            labels_mapping={
                'PRICE': Label.PRICE,
                'PRODUCT': Label.PRODUCT_NAME,
                'DISCOUNT_PERCENTAGE': Label.PERCENT,
                'DATE': Label.DATE,
                'QUANTITY': Label.UNKNOWN,
                'N/A': Label.UNKNOWN
            }
        )

        result = []
        for pc in proto_coupons:
            result.append(parse_proto_coupon(pc))
        print(json.dumps(result, indent=4))
