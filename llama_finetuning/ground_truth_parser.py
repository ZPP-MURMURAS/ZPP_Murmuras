import asyncio
import json
import string
import pandas as pd
from openai import AsyncOpenAI

COUPON_COLUMN = 'discount_text'

def init_client(api_key: string):
    client = AsyncOpenAI(api_key=api_key)
    return client

def load_coupons(file_path: string):
    coupons = pd.read_csv(file_path)
    coupons_list = coupons[COUPON_COLUMN].tolist()
    for i in range(len(coupons_list)):
        if str(coupons_list[i]) == 'nan':
            coupons_list[i] = ''

    return coupons_list

async def get_data(prompt: string, client: AsyncOpenAI, model: string = "gpt-4", temperature: int = 0):
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )

    return response.choices[0].message.content

def get_prompt(data_list: string):
    prompt = f"""You are a NER tagging assistant. You will be provided with a list of strings describing coupon discounts.
                        Your task is to extract discount data form the provided list and return them in a json format.
                        Here is the list: "{data_list}".

                        For each entry, return the data in the following format:
                        [
                            {{
                                "prices": list
                                "discount": str
                            }},
                            ...
                        ].
                        prices should be a list of string representing prices, followed by the currency.
                        discount should be an integer representing the percentage discount, followed by the percent sign.
                        If a value is not present, it should be "None" string.
                        All fields can be "None".
                        Return only json data.
                        Do not skip any entries."""
    return prompt

async def extract_discount_details(coupons_list: list, client: AsyncOpenAI, batch_size: int = 15):
    tasks = []
    for j in range(0, len(coupons_list), batch_size):
        task = asyncio.create_task(get_data(get_prompt(coupons_list[j:min(j + 15, len(coupons_list))]), client))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    res = []
    for result in results:
        result_json = json.loads(result)
        res.extend(result_json)
    return res

def extract_discounts(coupons: pd.DataFrame, client: AsyncOpenAI):
    discount_list = coupons['discount_text'].tolist()
    for i in range(len(discount_list)):
        if str(discount_list[i]) == 'nan':
            discount_list[i]    = ''
    chatgpt_output = asyncio.run(extract_discount_details(discount_list, client))
    chatgpt_output_json = json.dumps(chatgpt_output)
    return chatgpt_output_json

def prepare_ground_truth_data(ground_truth_json: string, coupons: pd.DataFrame):
    result = []
    for i in range(len(ground_truth_json)):
        res = {'time': coupons['time'][i], 'product_name': coupons['product_text'][i],
               'valid_until': coupons['validity_text'][i], 'discount': ground_truth_json[i]['discount']}
        prices = ground_truth_json[i]['prices']
        if prices == 'None' or len(prices) == 0:
            res['old_price'] = ''
            res['new_price'] = ''
        elif len(prices) == 1:
            res['old_price'] = ''
            res['new_price'] = prices[0]
        else:
            res['old_price'] = prices[0]
            res['new_price'] = prices[len(prices) - 1]
        result.append(res)
    return result

def ground_truth_to_dict(ground_truth_data: list):
    gtd_dict = {}
    for item in ground_truth_data:
        key = item['time']
        if key not in gtd_dict:
            gtd_dict[key] = []
        tmp = {'product_name': item['product_name'], 'valid_until': item['valid_until'], 'discount': item['discount'],
               'old_price': item['old_price'], 'new_price': item['new_price']}
        gtd_dict[key].append(tmp)
    return gtd_dict