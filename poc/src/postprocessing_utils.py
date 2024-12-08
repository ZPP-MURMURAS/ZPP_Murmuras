import pandas as pd

from constants import *
import re
from collections import defaultdict

from typing import Optional


def labeled_data_to_extra_csv_column(csv_content_path: str, csv_coupons_path: str, save_path: str):
    """
    Function for early testing of postprocessing.
    Having csv file and associated coupons list, split text views to rows containing single word each
    and new column with label assigned to word. At this moment any form of quantitative description of
    discount (price, percentage, '2 in price of 1') are labeled as price as this does not require extra
    processing of data and these values are treated by splitting algorithm as equivalent.
    Dates are currently recognised in very primitive manner, but ig for our purposes it is enough.
    :param csv_content_path: path to csv file with encoded xml of phone screen content
    :param csv_coupons_path: path to csv file with discount coupons present in screen content
    :param save_path: path to save processed xml with added labels and split texts
    """
    coupons_frame = pd.read_csv(csv_coupons_path)
    content_frame = pd.read_csv(csv_content_path)

    names = coupons_frame["product_text"].dropna().tolist()
    discount_texts = coupons_frame["discount_text"].dropna().tolist()

    labels = [Label.UNKNOWN if isinstance(txt, float)
              else Label.PRODUCT_NAME if txt in names
              else Label.PRICE if txt in discount_texts
              else Label.DATE if re.search("[0-9][-./\\][0-9][0-9]", txt) is not None
              else Label.UNKNOWN
              for txt in content_frame["text"]]

    content_frame["label"] = labels

    content_frame = content_frame.assign(text=content_frame["text"].str.split()).explode("text")

    content_frame.to_csv(save_path, index=False)


def merge_subsequent_text_fields(in_csv: str, out_csv: Optional[str] = None):
    """
    Takes csv with splitted texts and concatenates ones coming from single textfield.
    :param in_csv: path to csv with screen content and splitted text fields
    :param out_csv: output path. if not provided, will use in_csv.
    """
    if out_csv is None:
        out_csv = in_csv
    frame = pd.read_csv(in_csv)
    aggregators = {col: lambda x: x.iloc[0] for col in frame.columns}
    aggregators["text"] = lambda x: ' '.join(x) if not x.isna().any() else x
    frame = frame.groupby(["id", "i"]).agg(aggregators)
    frame.to_csv(out_csv)


if __name__ == '__main__':
    labeled_data_to_extra_csv_column("rossmann_content.csv", "rossmann_coupons.csv", "rossmann_labels.csv")
    merge_subsequent_text_fields("rossmann_labels.csv", "rossmann_final.csv")
