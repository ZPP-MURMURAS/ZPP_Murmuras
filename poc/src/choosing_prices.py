from constants import Label
import re
from typing import Any


# The conversion rates for the units. The values are the number of grams or milliliters
# in the corresponding unit.
UNITS = {"kg": 1000, "g": 1, "l": 1000, "ml": 1}

# The units that can be converted to smaller units. The values are the smaller units.
UNITS_TO_SMALLER = {"kg": "g", "l": "ml", "ml": "ml", "g": "g"}

# The currencies that are recognized by the system. The values are the symbols and the
# names of the currencies in lowercase.
CURRENCIES = {"€", "eur", "euro"}


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
            # If there is a currency symbol in the string, it extracts the numerical value 
            # near it and returns it as a float.
            currency_match = re.search(r'(\d+(\.\d+)?)\s*(€|eur|euro)', price, re.IGNORECASE)
            if currency_match:
                return float(currency_match.group(1))

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
Args: prices: list: a list of tuples containing the price and the discount type.
Returns: list: a list of tuples containing the original coupon data and the price per unit.
'''
def classify_price_per_unit(prices: list) -> list:   
    if not prices:
        return dict()
    
    coupon_data = list()
    
    for item in prices:
        if item[1] == Label.PRODUCT_NAME:
            continue

        price = item[0]
        if isinstance(price, str):
            match = re.search(r'(\d+(\.\d+)?)\s*(kg|g|l|ml)', price)
            if match:
                currency_match = re.search(r'(\d+(\.\d+)?)\s*(€|eur|euro)', price, re.IGNORECASE)
                if currency_match:
                    price = float(currency_match.group(1))
                else:
                    price = float(match.group(1))

                number_of_units = float(match.group(1))

                unit = match.group(3)
                if unit in UNITS:
                    price_per_unit = price / (UNITS[unit] * number_of_units)

                    coupon_data.append([item[0], item[1], price_per_unit, UNITS_TO_SMALLER[unit]])

    return coupon_data


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
        coupon_data[discount_type].append(price) 

    prices = [(price, Label.PRICE) for price in coupon_data[Label.PRICE]]
    highest_price, lowest_price = select_prices(prices)

    coupon_data["highest_price"] = highest_price
    coupon_data["lowest_price"] = lowest_price

    return coupon_data
