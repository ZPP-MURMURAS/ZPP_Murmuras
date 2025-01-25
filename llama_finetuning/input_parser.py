import string

import pandas as pd

X_COLUMN = 'text'
Y_COLUMN = 'seen_timestamp'

def concat_column_x_by_column_y(column_x, column_y, df):
    prev_col_val = None
    result = {}
    sub_result = ''
    time = ''
    for index, row in df.iterrows():
        if row[column_y] == prev_col_val:
            if str(row[column_x]) is not None:
                sub_result += ' ' + str(row[column_x])
        else:
            if sub_result != '':
                result[time] = sub_result
            sub_result = str(row[column_x])
            prev_col_val = row[column_y]
            time = row['time']

    return result

def prepare_input_data(path: string):
    data = pd.read_csv(path)
    data = data[data['seen_timestamp'] != 0]
    data = data[data['is_visible'] != False]
    data.dropna(subset=['text'], inplace=True)
    data_concat = concat_column_x_by_column_y(X_COLUMN, Y_COLUMN, data)
    return data_concat

def create_training_df(input_dict, gtd_dict):
    concatenated_dfs = []
    for key, value in input_dict.items():
        if key in gtd_dict:
            for item in gtd_dict[key]:
                res = {'input': value, 'output': item}
                concatenated_dfs.append(res)
        else:
            res = {'input': value, 'output': ''}
            concatenated_dfs.append(res)

    # Parsing to make the dict compatible with pandas
    parse_dict = {'Context': [], 'Response': []}
    for data in concatenated_dfs:
        parse_dict['Context'].append(data['input'])
        parse_dict['Response'].append(data['output'])
    training_df = pd.DataFrame.from_dict(parse_dict)
    training_df = training_df.astype({'Context': str, 'Response': str})
    return training_df
