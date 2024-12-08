from constants import Label
import re
from typing import Any


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
            # and may not work for all cases.
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
the highest and lowest prices.
'''
def classify_prices(classified_inputs: list) -> dict:
    """
    Classifies the input into a dictionary of lists, where the keys are the labels.
    """
    
    if not classified_inputs:
        return dict()
    
    coupon_data = dict()

    for label in Label:
        coupon_data[label] = list()

    for price, discount_type in classified_inputs:  
        coupon_data[discount_type].append(float(price))

    highest_price, lowest_price = select_prices(coupon_data[Label.PRICE])

    coupon_data["highest_price"] = highest_price
    coupon_data["lowest_price"] = lowest_price

    return coupon_data

