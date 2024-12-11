from constants import Label
import re
from typing import Any
import json


'''
Args: prices: list: a list of tuples containing the price and the discount type.
Returns: tuple: a tuple containing the highest and lowest prices corresponding to the 
price before and after the discount.
'''
def select_prices(prices: list) -> tuple:
    """
    Selects the highest and lowest prices from the given list.
    If the list is empty or the types of discounts are different 
    than the expected types, returns None.
    """

    if not prices:
        return None

    filtered_prices = [price for price, discount_type in prices if discount_type == Label.PRICE]
    if not filtered_prices:
        return None
    
    def parse_price(price: Any):
        '''
        Args: price: Any: a string containing the price.
        Returns: float: the price as a float. If the price is a float or an int,
        it casts it to a float and returns it. If the price is a string, it extracts
        the numerical value from the string and returns it as a float. If the price as 
        a string contains no numerical value, it returns None.
        '''

        if isinstance(price, int) or isinstance(price, float):
            return float(price)
        elif isinstance(price, str):
            # Returns the first numerical value in the string. This is a heuristic
            # and may not work for all cases. For example 100ml = 6.99 EUR.
            # We could classify what is a currency in a given context and based on that 
            # extract the price.
            match = re.search(r'\d+(\.\d+)?', price)
            if match:
                return float(match.group())
        return None

    parsed_prices = [parse_price(price) for price in filtered_prices]
    filtered_prices = [price for price in parsed_prices if price is not None]

    highest_price = max(filtered_prices)
    lowest_price = min(filtered_prices)
    
    return highest_price, lowest_price


'''
Args: classified_inputs: list: a list of tuples containing the classified traits 
of a coupon.
Returns: dict: a dictionary containing the classified traits of a coupon as well as 
the highest and lowest prices. The 
'''
def classify_prices(classified_inputs: list) -> dict:    
    if not classified_inputs:
        return {"coupons": []}
    
    coupon_data = {"coupons": []}
    coupon = {}
    prices = []
    seen_labels = set()

    for item, item_type in classified_inputs:
        if item_type not in Label or item_type in seen_labels:
            continue

        seen_labels.add(item_type)
        if item_type == Label.PRODUCT_NAME:
            coupon["product_name"] = item
        elif item_type == Label.PRICE:
            prices.append((item, item_type))
        elif item_type == Label.PERCENT:
            coupon["percent"] = item
        elif item_type == Label.PRICE_PER_UNIT:      
            coupon["price_per_unit"] = item
        elif item_type == Label.OTHER_DISCOUNT:       
            coupon["other_discount"] = item
        elif item_type == Label.DATE:      
            coupon["validity"] = item
        else:
            coupon["other"] = item

    highest_price, lowest_price = select_prices(prices)

    coupon["old_price"] = highest_price
    coupon["new_price"] = lowest_price

    coupon_data["coupons"].append(coupon)

    return coupon_data



