import sys
import os
import json
from io import StringIO
from llama_cpp import Llama


if __name__ == "__main__":
    args = sys.argv

    if len(args) != 2:
        print("Usage: python llama_pipeline.py <model_path>")
        sys.exit(1)

    model_path = args[1]
    llama = Llama(model_path)
    input_data = sys.stdin.read()

    out = llama("\n### Input:\n" + input_data + "\n\n### Response:\n", max_tokens=512)["choices"][0]["text"]

    print(out)
