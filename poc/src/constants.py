from enum import Enum


class Label(Enum):
    PRODUCT_NAME = '<PRODUCT_NAME>'
    PRICE = '<PRICE>'
    PERCENT = '<PERCENT>'
    PRICE_PER_UNIT = '<PRICE_PER_UNIT>'
    OTHER_DISCOUNT = '<OTHER_DISCOUNT>'
    DATE = '<DATE>'
    UNKNOWN = '<UNKNOWN>'


LABEL_COLUMN = "label"
TIMESTAMP_COLUMN = "seen_timestamp"
