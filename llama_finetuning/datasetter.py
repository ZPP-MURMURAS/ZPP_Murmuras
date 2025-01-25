import string
from typing import Callable

import pandas as pd
from datasets import Dataset

PROMPT = """You are provided with text representing contents of the phone screen. Your task is to extract 
information about coupons from the text. The information should include the product name, the validity date,
the discount, the old price, and the new price. 

### Input:
{}

### Response:
{}"""

EOS_TOKEN = ''

def one_input_one_output_wrequest(training_df: pd.DataFrame):
    inputs = training_df['Context']
    outputs = training_df['Response']

    texts = []
    for input_, output in zip(inputs, outputs):
        text = PROMPT.format(input_, output) + EOS_TOKEN
        texts.append(text)

    return { "text" : texts, }

def run_mapping(df: pd.DataFrame, map_func: Callable):
    training_data = Dataset.from_pandas(df)
    training_data = training_data.map(map_func, batched=True)
    return training_data

