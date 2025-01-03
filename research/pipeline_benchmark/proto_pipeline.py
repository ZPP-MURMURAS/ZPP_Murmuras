from benchmark import Coupon
import json
import random
"""
This script is used to generate random data for the proto_pipeline 
benchmark. It reads the ideal_data.json file and generates new data 
based on it by randomly changing the product name, new price, old 
price, percents, other discounts and dates, or removing them.
This simutales the output of the pipeline. 
"""

with open('ideal_data.json', 'r') as file:
    data = json.load(file)

coupons = []
for entry in data:
    coupon = Coupon(entry['product_name'], entry['new_price'],
                         entry['old_price'], entry['percents'],
                         entry['other_discounts'], entry['dates'])

    coupon.product_name = coupon.product_name + str(random.randint(
        0, 100)) + ' ' + str(random.randint(0, 100))
    if coupon.new_price:
        coupon.new_price = round(
            float(coupon.new_price) * random.uniform(0.5, 1.5), 2)
    if coupon.old_price:
        coupon.old_price = round(
            float(coupon.old_price) * random.uniform(0.5, 1.5), 2)

    new_percents = []
    for percent in entry['percents']:
        r = random.random()
        if r < 0.2:
            continue
        elif r < 0.2:
            new_percent = round(
                float(entry['percents']) * random.uniform(0.5, 1.5), 2)
        else:
            new_percent = percent
        new_percents.append(new_percent)

    coupon.percents = new_percents

    new_discounts = []
    for discount in entry['other_discounts']:
        r = random.random()
        if r < 0.2:
            continue
        new_discounts.append(discount + str(random.randint(0, 100)))

    coupon.other_discounts = new_discounts
    coupon.dates = coupon.dates + str(random.randint(0, 100))

    coupons.append(coupon)

print(json.dumps([coupon.__dict__ for coupon in coupons], indent=4))
