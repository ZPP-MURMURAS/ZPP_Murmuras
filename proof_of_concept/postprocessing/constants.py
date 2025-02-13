from enum import Enum


class Label(Enum):
    PRODUCT_NAME = '<PRODUCT_NAME>'
    PRICE = '<PRICE>'
    PERCENT = '<PERCENT>'
    OTHER_DISCOUNT = '<OTHER_DISCOUNT>'
    DATE = '<DATE>'
    UNKNOWN = '<UNKNOWN>'


LABEL_COLUMN = "label"
TIMESTAMP_COLUMN = "seen_timestamp"
CLASS_NAME_COLUMN = "view_class_name"
