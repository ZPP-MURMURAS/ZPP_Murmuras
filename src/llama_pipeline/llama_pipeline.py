import sys
import os
import json
from io import StringIO
from llama_cpp import Llama

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_PATH, "../../")))

from src.llama_dataset_generation.input_parser import prepare_input_data


if __name__ == "__main__":
    args = sys.argv

    if len(args) != 2:
        print("Usage: python llama_pipeline.py <model_path>")
        sys.exit(1)

    model_path = args[1]
    input_csv_str = sys.stdin.read()
    llama = Llama(model_path)
    input_data = prepare_input_data(StringIO(input_csv_str))

    output_list = []

    for context in input_data:
        out = llama(context + "## Response:\n", max_tokens=512)["choices"][0]["text"]
        json_data = json.loads(out)
        output_list += json_data

    cleaned_output_list = [obj for obj in output_list if obj != {}]

    print(json.dumps(cleaned_output_list))

