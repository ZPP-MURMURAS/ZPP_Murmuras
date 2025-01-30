import asyncio
import json
import string
import pandas as pd
from openai import AsyncOpenAI

__COUPON_COLUMN = 'discount_text'


def init_client(api_key: string) -> AsyncOpenAI:
    """
    This function initializes the Async OpenAI client.

    :param api_key: The OpenAI API key.
    :return client: The initialized Async OpenAI client
    """
    client = AsyncOpenAI(api_key=api_key)
    return client


def load_coupons_from_json(path: string = 'ground_truth_json') -> json:
    """
    This function reads the coupon data processed by the ChatGPT and stored under the input path.

    :param path: The path to the json file containing the coupon data.
    :return data: The coupon data in json format.
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
                prices should be a list of string representing prices, followed by the currency.
                discount should be an integer representing the percentage discount, followed by the percent sign.
                If a value is not present, it should be "None" string.
                All fields can be "None".
                Return only json data.
                Do not skip any entries."""
    return prompt


async def __extract_discount_details(coupons_list: list, client: AsyncOpenAI, batch_size: int = 15) -> list:
    """
    This function handles the logic of creating async tasks to process the coupon data in batches.

    :param coupons_list: The list of coupon data.
    :param client: The initialized Async OpenAI client.
    :param batch_size: The size of each batch.
    :return res: The extracted discount details.
    """
    tasks = []
    for j in range(0, len(coupons_list), batch_size):
        task = asyncio.create_task(__get_data(__get_prompt(coupons_list[j:min(j + 15, len(coupons_list))]), client))
        tasks.append(task)
    results = await asyncio.gather(*tasks)
    res = []
    for result in results:
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
        key = item['time']
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


def prepare_ground_truth_data(ground_truth_json: json, coupons: pd.DataFrame) -> dict:
    """
    This function prepares the ground truth data by combining the extracted discount details
    with the original coupon data that was not processed by the ChatGPT.
    Then, it calls the ground_truth_to_dict function to convert the data to a dictionary format.

    :param ground_truth_json: The extracted discount details in json format.
    :param coupons: The original coupon data.
    :return result: The prepared ground truth data in list format.
    """
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

    return __ground_truth_to_dict(result)