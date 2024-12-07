from enum import Enum


class Label(Enum):
    PRODUCT_NAME = '<PRODUCT_NAME>'
    PRICE = '<PRICE>'
    PERCENT = '<PERCENT>'
    OTHER_DISCOUNT = '<OTHER_DISCOUNT>'
    DATE = '<DATE>'
    UNKNOWN = '<UNKNOWN>'
