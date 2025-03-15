import argparse
import os
import sys
from typing import List, Dict
from io import StringIO
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoConfig

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(CURRENT_PATH, "../../")))

from src.llama_dataset_generation.input_parser import prepare_input_data


def download_model(model_name: str, cache_dir: str) -> str:
    """Downloads a model from Hugging Face Hub if not already cached."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    
    os.makedirs(cache_dir, exist_ok=True)
    return snapshot_download(repo_id=model_name, token=hf_token, cache_dir=cache_dir)


def perform_ner(model_path: str, text: str | List[str]) -> List[Dict[str, any]] | List[List[Dict[str, any]]]:
    """Uses a BERT-based model to perform Named Entity Recognition (NER) on the input text using the BIO2 scheme."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
    
    return nlp(text)


def coupon_to_json_first(model_coupon: List[Dict[str, any]]) -> Dict[str, str]:
    """Converts a coupon tagged by the model to a JSON object."""
    coupon = {"product_name": "", "discount_text": "", "valid_until": "", "activation_text": ""}
    for entity in reversed(model_coupon):
        if entity["entity_group"] == "PRODUCT-NAME":
            coupon["product_name"] = entity["word"]
        elif entity["entity_group"] == "DISCOUNT-TEXT":
            coupon["discount_text"] = entity["word"]
        elif entity["entity_group"] == "VALIDITY-TEXT":
            coupon["valid_until"] = entity["word"]
        elif entity["entity_group"] == "ACTIVATION-TEXT":
            coupon["activation_text"] = entity["word"]

    return coupon


def coupon_to_json_concat(model_coupon: List[Dict[str, any]]) -> Dict[str, str]:
    """Converts a coupon tagged by the model to a JSON object."""
    coupon = {"product_name": "", "discount_text": "", "valid_until": "", "activation_text": ""}
    for entity in model_coupon:
        if entity["entity_group"] == "PRODUCT-NAME":
            coupon["product_name"] += entity["word"] + " "
        elif entity["entity_group"] == "DISCOUNT-TEXT":
            coupon["discount_text"] += entity["word"] + " "
        elif entity["entity_group"] == "VALIDITY-TEXT":
            coupon["valid_until"] += entity["word"] + " "
        elif entity["entity_group"] == "ACTIVATION-TEXT":
            coupon["activation_text"] += entity["word"] + " "

    for key in coupon:
        coupon[key] = coupon[key].strip()

    return coupon


def coupon_to_json(model_coupon: List[Dict[str, any]], strategy: str) -> Dict[str, str]:
    """Converts a coupon tagged by the model to a JSON object."""
    if strategy == "first":
        return coupon_to_json_first(model_coupon)
    elif strategy == "concat":
        return coupon_to_json_concat(model_coupon)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Named Entity Recognition using a Hugging Face model.")
    parser.add_argument("cs_model", type=str, help="Name of the coupon selection Hugging Face model.")
    parser.add_argument("fe_model", type=str, help="Name of the field extraction Hugging Face model.")
    parser.add_argument("--strategy", type=str, default="first", help="Strategy to use for coupon extraction.")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Directory to store the downloaded model.")
    
    args = parser.parse_args()

    input_csv_str = sys.stdin.read()
    input_data = prepare_input_data(StringIO(input_csv_str))
    input_list = list(input_data.values())
    
    cs_path = download_model(args.cs_model, args.cache_dir)
    fe_path = download_model(args.fe_model, args.cache_dir)
    results = perform_ner(cs_path, input_list)

    flat_results = [item for sublist in results for item in sublist]
    coupons = [res['word'] for res in flat_results]

    results = perform_ner(fe_path, coupons)
    results = [coupon_to_json(coupon, args.strategy) for coupon in results]

    print(results)
