import string

import pandas as pd
from json import dumps

from src.constants import *

__X_COLUMN = 'text'

def __concat_column_x_by_column_y(column_x: string, df: pd.DataFrame) -> dict:
    """
    This function concatenates the consecutive values of column_x for each value of the seen_timestamp column.
    As a result, it returns a dictionary with the time values from the aggregation column as keys, and the concatenated
    values of text as values.
    Example usage (in this case, main purpose) would be to aggregate text data for each
    distinct value from the aggregation column in the input dataframe.

    :param column_x: The column to be concatenated.
    :param df: The input dataframe.
    :return result: A dictionary with the values from the aggregation column as keys, and the concatenated
    values of column_x as values.
    """
    result = {}
    sub_result = ''
    agg = None
    for index, row in df.iterrows():
        if agg is not None and row[AGGREGATION_COLUMN] == agg:
            if str(row[column_x]):
                sub_result += (' ' if sub_result else '') + str(row[column_x])
        else:
            if agg is not None and sub_result:
                result[agg] = sub_result
            sub_result = str(row[column_x])
            agg = row[AGGREGATION_COLUMN]
    if sub_result != '':
        result[agg] = sub_result

    return result


def prepare_input_data(path: string) -> dict:
    """
    This function reads a csv file from the given path, and filters out the rows with
    seen_timestamp = 0 and is_visible = False. It also drops rows with empty text values.
    It then calls concat_column_x_by_column_y to aggregate text data by seen_timestamp,
    and returns its result.

    :param path: The path to the csv file.
    :return data_concat: A dictionary with the unique values of aggregation column as keys, and the concatenated
    values of text as values.
    """
    data = pd.read_csv(path)
    # data = data[data['seen_timestamp'] != 0]
    # data = data[data['is_visible'] != False]
    data.dropna(subset=['text'], inplace=True)
    data_concat = __concat_column_x_by_column_y(__X_COLUMN, data)
    return data_concat


def create_training_df(input_dict: dict, gtd_dict: dict) -> pd.DataFrame:
    """
    This function takes dictionaries containing input data representing processed contents
    of the text field in the csv file, and ground truth data representing the aggregated and processed
    coupon data from the same file. It then concatenates the two dictionaries into a pandas dataframe.

    :param input_dict: A dictionary with the unique values of aggregation column as keys, and the concatenated
    values of text as values.
    :param gtd_dict: A dictionary containing coupon data.
    :return training_df: A pandas dataframe containing the concatenated data.
    """
    concatenated_dfs = []
    for key, value in input_dict.items():
        if key in gtd_dict:
            for item in gtd_dict[key]:
                res = {'input': value, 'output': item}
                concatenated_dfs.append(res)
        else:
            res = {'input': value, 'output': {}}
            concatenated_dfs.append(res)

    # Parsing to make the dict compatible with pandas
    parse_dict = {'Context': [], 'Response': []}
    for data in concatenated_dfs:
        parse_dict['Context'].append(data['input'])
        parse_dict['Response'].append(dumps(data['output']))
    training_df = pd.DataFrame.from_dict(parse_dict)
    training_df = training_df.astype({'Context': str})

    # If two columns have the same Context and empty/the same response, we don't need them
    training_df.drop_duplicates(subset=['Context', 'Response'], keep='first', inplace=True)

    # Reindex df
    training_df.reset_index(drop=True, inplace=True)

    return training_df
