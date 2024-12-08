from constants import Label

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
    
    filtered_prices = [float(price) for price in filtered_prices]

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

    coupon_data["highest_price"] = max(coupon_data[Label.PRICE])
    coupon_data["lowest_price"] = min(coupon_data[Label.PRICE])

    return coupon_data

