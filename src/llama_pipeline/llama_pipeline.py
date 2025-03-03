import sys
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset
from llama_cpp import Llama


if __name__ == "__main__":

    llama = Llama(model_path)

    # Run the model on the dataset
    results = llama(dataset["test"]["text"])

    # Print the results
    print(results)
