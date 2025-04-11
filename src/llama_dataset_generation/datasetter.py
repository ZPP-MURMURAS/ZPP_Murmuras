import string
from typing import Callable
import pandas as pd
from datasets import Dataset

__PROMPT = (
    "You are provided with text representing contents of the phone screen. Your task is to extract "
    "information about coupons from the text. The information should include the product name, the validity text, "
    "the discount text and the activation text.\n\n"
    "### Input:\n{}\n\n"
    "### Response:\n{}"
)


PROMPT_WTH_DESC = """
### Input:
{}

### Response:
{}"""

EOS_TOKEN = ''


def __map_logic(training_dts: Dataset, prompt: string) -> dict:
    """
    Function that maps the training data to the desired format and prompt.

    :param training_dts: The training data to be mapped.
    :param prompt: The prompt to be used.
    :return: A dictionary with the mapped data.
    """
    inputs = training_dts['Context']
    outputs = training_dts['Response']

    texts = []
    for input_, output in zip(inputs, outputs):
        text = prompt.format(input_, output)# + EOS_TOKEN
        texts.append(text)

    return {"text": texts, }


def one_input_one_output_wrequest(training_dts: Dataset) -> dict:
    return __map_logic(training_dts, __PROMPT)

def one_input_one_output_wthrequest(training_dts: Dataset) -> dict:
    return __map_logic(training_dts, PROMPT_WTH_DESC)


def one_input_multiple_outputs_wrequest(training_dts: Dataset) -> dict:
    return __map_logic(training_dts, __PROMPT)


def one_input_multiple_outputs_wthrequest(training_dts: Dataset) -> dict:
    return __map_logic(training_dts, PROMPT_WTH_DESC)


def __parse_to_oimo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to change the layout of the input dataframe from
    one input one output (oimo) to one input multiple outputs (oimo).
    :param df: A pandas dataframe to be parsed.
    :return output_df: A pandas dataframe with the parsed layout.
    """
    unique_inputs = df['Context'].unique()
    outputs_per_inputs = {}
    for input_ in unique_inputs:
        outputs_per_inputs[input_] = df[df['Context'] == input_]['Response'].tolist()

    # Drop empty strings
    for key, value in outputs_per_inputs.items():
        outputs_per_inputs[key] = [x for x in value if x]

    concat_outputs = {}
    for key, value in outputs_per_inputs.items():
        concat_outputs[key] = '[' + ', '.join(value) + ']'

    pre_df_dict = {'Context': [], 'Response': []}
    for key, value in concat_outputs.items():
        pre_df_dict['Context'].append(key)
        if value != '{}':
            pre_df_dict['Response'].append(value)
    output_df = pd.DataFrame.from_dict(pre_df_dict)
    for index, row in output_df.iterrows():
        if row['Response'] == '[{}]':
            row['Response'] = '[]'
    return output_df


def run_mapping(df: pd.DataFrame, map_func: Callable) -> Dataset:
    """
    This function takes a pandas dataframe and a mapping function, and returns a Dataset object
    with the mapping function applied to it.
    :param df: A pandas dataframe to be mapped.
    :param map_func: A mapping function to be applied to the dataframe.
    :return training_data: A Dataset object with the mapping function applied.
    """
    if map_func in [one_input_multiple_outputs_wrequest, one_input_multiple_outputs_wthrequest]:
        training_data = Dataset.from_pandas(__parse_to_oimo(df))
    else:
        training_data = Dataset.from_pandas(df)
    training_data = training_data.map(map_func, batched=True)
    return training_data


MAP_FUNCTIONS = [one_input_one_output_wrequest, one_input_one_output_wthrequest,
                 one_input_multiple_outputs_wrequest, one_input_multiple_outputs_wthrequest]
