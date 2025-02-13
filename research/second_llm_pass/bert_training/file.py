import os
from pprint import pprint

from datasets import Dataset, load_dataset

csj = load_dataset('zpp-murmuras/coupon_select_json', split='train', token=os.getenv('HF_HUB_KEY'))
cspl = load_dataset('zpp-murmuras/coupon_select_plain_text', split='train', token=os.getenv('HF_HUB_KEY'))

print(cspl['texts'][10])
print(cspl['labels'][10])

for i in range(100):
    print(csj['texts'][10])
    print(csj['labels'][10])