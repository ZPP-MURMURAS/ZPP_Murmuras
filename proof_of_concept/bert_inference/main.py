from numpy.f2py.auxfuncs import throw_error
from sympy.solvers.diophantine.diophantine import length
from transformers import pipeline
import pandas as pd
import json as json_module

model_checkpoint = 'zpp-murmuras/bert_multiling_cased_test_data_test_1'
csv_file_path = 'test_data_2024_11_25_lidl_plus_content_generic_2024-12-05T07_39_49.726955559+01_00.csv'

token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple")


def chunk_string_by_words(text, chunk_size=64):
    words = text.split()
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    chunked_text = [' '.join(chunk) for chunk in chunks]

    return chunked_text

def prepare_string_list(texts):
    texts_filtered = [s for s in texts if s != '']
    texts_chunked = []

    for text in texts_filtered:
        texts_chunked += chunk_string_by_words(text)

    return texts_chunked


def prepare_csv(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    if 'text' not in df.columns:
        raise ValueError('no text column in csv')

    grouping_columns = ['application_name', 'seen_timestamp']

    for col_name in grouping_columns:
        if col_name not in df.columns:
            raise ValueError(f'no {col_name} column in csv')

    grouped_dfs = [group for _, group in df.groupby(grouping_columns)]
    view_texts_coalesced = [' '.join(df['text'].dropna()) for df in grouped_dfs]

    return prepare_string_list(view_texts_coalesced)

def map_strings_back_to_csv(entries, path):
    #for entry in entries:
     #   print(entry)

    df = pd.read_csv(path)
    df.columns = df.columns.str.replace(' ', '_').str.lower()

    if 'text' not in df.columns:
        raise ValueError('no text column in csv')

    i = 0
    df['sex'] = ''
    df['ner_tags'] = ''
    example = entries[i]
    example_tags = token_classifier(example)
    for index, row in df.iterrows():
        if pd.notna(row['text']) and row['text'] != '':
            tmp = row['text']
            if len(example) == 0:
                if len(example_tags) != 0:
                    throw_error('example is empty but example_tags is not')
                i += 1
                example = entries[i]
                example_tags = token_classifier(example)
            if example.startswith(row['text']):
                example = example[len(row['text']):]
                if len(example) > 0 and example[0] == ' ':
                    example = example[1:]
            else:
                throw_error('example does not start with row text')

            tags_iter = 0
            json_data = {}  # Renamed to avoid conflict with the imported module
            for tag in example_tags:
                sentence = tag['word']
                if sentence in tmp:
                    json_data[sentence] = tag['entity_group']
                    tags_iter += 1
            example_tags = example_tags[tags_iter:]
            df.at[index, 'ner_tags'] = json_module.dumps(json_data)  # Serializing to valid JSON format

    # save to csv
    df.to_csv('output.csv', index=False)

if __name__ == '__main__':
    map_strings_back_to_csv(prepare_csv(csv_file_path), csv_file_path)

