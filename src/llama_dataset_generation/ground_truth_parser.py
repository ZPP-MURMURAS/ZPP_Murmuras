import asyncio
import json
import string
import pandas as pd
from pandas import isna
from openai import AsyncOpenAI

from src.constants import *

__COUPON_COLUMN = 'discount_text'


def init_client(api_key: string) -> AsyncOpenAI:
    """
    This function initializes the Async OpenAI client.

    :param api_key: The OpenAI API key.
    :return client: The initialized Async OpenAI client
    """
    client = AsyncOpenAI(api_key=api_key)
    return client


def store_coupons_as_json(coupons: list, path: string) -> None:
    """
    This function stores the extracted coupon data in a json file under the input path.

    :param coupons: The extracted coupon data in a list of dicts.
    :param path: The path to the json file where the coupon data will be stored.
    """
    with open(path, 'w') as f:
        f.write(coupons)


def load_coupons_from_json(path: string = 'ground_truth_json') -> list:
    """
    This function reads the coupon data processed by the ChatGPT and stored under the input path.

    :param path: The path to the json file containing the coupon data.
    :return data: The coupon data in a list of jsons.
    """
    with open(path) as f:
        data = json.load(f)
    return data


def load_coupons(file_path: string) -> list:
    """
    This function reads the coupon data from the given file path, converts it to a list,
    and replaces any NaN values with empty strings.

    :param file_path: The path to the file containing the coupon data.
    :return coupons_list: A list containing the processed coupon data.
    """
    coupons = pd.read_csv(file_path)
    coupons_list = coupons[__COUPON_COLUMN].tolist()
    for i in range(len(coupons_list)):
        if str(coupons_list[i]) == 'nan':
            coupons_list[i] = ''

    return coupons_list


async def __get_data(prompt: string, client: AsyncOpenAI, model: string = "gpt-4", temperature: int = 0) -> string:
    """
    This function sends a prompt to the ChatGPT model and returns the response.

    :param prompt: The prompt to send to the ChatGPT model.
    :param client: The initialized Async OpenAI client.
    :param model: The model to use for completion.
    :param temperature: The temperature to use for completion.
    """
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
    )

    return response.choices[0].message.content


def __get_prompt(data_list: string) -> string:
    """
    This function generates the prompt for the ChatGPT model.

    :param data_list: The list of coupon data to include in the prompt.
    :return prompt: The generated prompt.
    """
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
                prices should be a list of strings representing prices, followed by the currency.
                In case there is no price, it should be an empty list.
                discount should be an integer representing the percentage discount, followed by the percent sign.
                If a discount is not present, it should be "None" string.
                Return only json data, without any annotations or additional text.
                Do not skip any entries."""
    return prompt


async def __extract_discount_details(
        coupons_list: list, client: AsyncOpenAI, batch_size: int = 15, max_requests_async: int = 20) -> list:
    """
    This function handles the logic of creating async tasks to process the coupon data in batches.
    Coupon data in this context refers to the list of coupon texts that need to be processed by the ChatGPT.
    Those texts contain the information about the value of the discount (usually either a percentage or a new price,
    but it can be both, and both old and new price can be present). ChatGPT usage is dictated by the need to extract
    the currency from the price, as well as the goal of generalization for different types of discounts that
    may be valid but not "regexable" in an easy way.

    :param coupons_list: The list of coupon data.
    :param client: The initialized Async OpenAI client.
    :param batch_size: The size of each batch.
    :param max_requests_async: The maximum number of async tasks to run at once.
    :return res: The extracted discount details.
    """
    itr = 0
    res = []
    while itr < len(coupons_list):
        tasks = []
        for j in range(itr, len(coupons_list), batch_size):
            task = asyncio.create_task(
                __get_data(__get_prompt(coupons_list[j:min(j + batch_size, len(coupons_list))]), client, model='gpt-4o'))
            tasks.append(task)
            if len(tasks) >= max_requests_async:
                itr = j + batch_size
                break
        else:
            itr = len(coupons_list)
        results = await asyncio.gather(*tasks)
        for result in results:
            list_start = result.find('[')
            list_end = result.rfind(']')
            result = result[list_start:list_end + 1]
            result_json = json.loads(result)
            res.extend(result_json)
    return res


def __ground_truth_to_dict(ground_truth_data: list) -> dict:
    """
    This function converts the ground truth data list to a dictionary format for easier processing.

    :param ground_truth_data: The ground truth data in list format.
    :return gtd_dict: The ground truth data in dictionary
    """
    gtd_dict = {}
    for item in ground_truth_data:
        key = item[AGGREGATION_COLUMN]
        if key not in gtd_dict:
            gtd_dict[key] = []
        tmp = {'product_name': item['product_name'], 'valid_until': item['valid_until'], 'discount': item['discount'],
               'old_price': item['old_price'], 'new_price': item['new_price']}
        gtd_dict[key].append(tmp)
    return gtd_dict


def extract_discounts(coupons: list, client: AsyncOpenAI) -> json:
    """
    This function takes the list of coupon data and calls the OpenAI API to extract discount details.
    Under the hood, it generates async tasks to process the data in batches to make the job easier
    for the ChatGPT. It then returns a json containing the extracted discount details.

    :param coupons: The list of coupon data.
    :param client: The initialized Async OpenAI client.
    :return chatgpt_output_json: The extracted discount details in json format.
    """
    chatgpt_output = asyncio.run(__extract_discount_details(coupons, client))
    chatgpt_output_json = json.dumps(chatgpt_output)
    return chatgpt_output_json


def prepare_ground_truth_data(ground_truth_json: list, coupons: pd.DataFrame) -> dict:
    """
    This function prepares the ground truth data by combining the extracted discount details
    with the original coupon data that was not processed by the ChatGPT.
    Then, it calls the ground_truth_to_dict function to convert the data to a dictionary format.
    It will skip rows from the coupons frame with no content_full.

    :param ground_truth_json: The extracted discount details in a list of jsons.
    :param coupons: The original coupon data.
    :return result: The prepared ground truth data in list format.
    """
    result = []
    coupons_itr = 0
    print(len(coupons))
    for i in range(len(ground_truth_json)):
        try:
            res = {AGGREGATION_COLUMN: coupons[AGGREGATION_COLUMN][coupons_itr], 'product_name': coupons['product_text'][coupons_itr],
                   'valid_until': coupons['validity_text'][coupons_itr], 'discount': ground_truth_json[i]['discount']}
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
            for k, v in res.items():
                if isna(v):
                    res[k] = None
        except Exception as e:
            print(e)
            print(ground_truth_json[i])
        else:
            result.append(res)
        finally:
            coupons_itr += 1
    return __ground_truth_to_dict(result)


def prepare_ground_truth_data_no_ai(coupons: pd.DataFrame) -> dict:
    """
    Given a coupons dataframe, this function constructs the coupon jsons
    aggregated by __AGGREGATION_COLUMN. It will skip rows with empty content_full
    :param coupons: The coupons dataframe.
    :return result: mapping from __AGGREGATION_COLUMN values to lists of coupons jsons.
    """
    result = {}
    for t, subframe in coupons.groupby(AGGREGATION_COLUMN):
        result[t] = []
        for _, row in subframe.iterrows():
            content_full = row['content_full']
            if str(content_full) == 'nan' or str(content_full) == "['']" or str(content_full) == "[]":
                continue
            coupon_repr = {
                "discount_text": row['discount_text'] if not isna(row['discount_text']) else None,
                "product_name": row['product_text'] if not isna(row['product_text']) else None,
                "valid_until": row['validity_text'] if not isna(row['validity_text']) else None
            }
            result[t].append(coupon_repr)
    return result
