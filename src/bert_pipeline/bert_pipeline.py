import argparse
import os
import sys
from typing import List, Dict
from io import StringIO
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoConfig


NER_ENTITY_GROUP = "entity_group"
NER_TEXT = "word"

TAG_PRODUCT_NAME = "PRODUCT-NAME"
TAG_DISCOUNT_TEXT = "DISCOUNT-TEXT"
TAG_VALIDITY_TEXT = "VALIDITY-TEXT"
TAG_ACTIVATION_TEXT = "ACTIVATION-TEXT"

COUPON_PRODUCT_NAME = "product_name"
COUPON_DISCOUNT_TEXT = "discount_text"
COUPON_VALID_UNTIL = "valid_until"
COUPON_ACTIVATION_TEXT = "activation_text"

HF_TOKEN = "HF_TOKEN"


def download_model(model_name: str, cache_dir: str) -> str:
    """Downloads a model from Hugging Face Hub if not already cached."""
    hf_token = os.getenv(HF_TOKEN)
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    
    os.makedirs(cache_dir, exist_ok=True)
    return snapshot_download(repo_id=model_name, token=hf_token, cache_dir=cache_dir)


def perform_ner(model_path: str, text: str) -> List[Dict[str, any]]:
    """Uses a model to perform Named Entity Recognition (NER) on the input text using the BIO2 scheme."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first")
    
    return nlp(text)


def coupon_to_json_first(model_coupon: List[Dict[str, any]]) -> Dict[str, str]:
    """Converts a coupon tagged by the model to a JSON object. Only the first entity of each type is considered."""
    coupon = {COUPON_PRODUCT_NAME: "", 
              COUPON_DISCOUNT_TEXT: "", 
              COUPON_VALID_UNTIL: "", 
              COUPON_ACTIVATION_TEXT: ""}

    for entity in reversed(model_coupon):
        if entity[NER_ENTITY_GROUP] == TAG_PRODUCT_NAME:
            coupon[COUPON_PRODUCT_NAME] = entity[NER_TEXT]
        elif entity[NER_ENTITY_GROUP] == TAG_DISCOUNT_TEXT:
            coupon[COUPON_DISCOUNT_TEXT] = entity[NER_TEXT]
        elif entity[NER_ENTITY_GROUP] == TAG_VALIDITY_TEXT:
            coupon[COUPON_VALID_UNTIL] = entity[NER_TEXT]
        elif entity[NER_ENTITY_GROUP] == TAG_ACTIVATION_TEXT:
            coupon[COUPON_ACTIVATION_TEXT] = entity[NER_TEXT]

    return coupon


def coupon_to_json_concat(model_coupon: List[Dict[str, any]]) -> Dict[str, str]:
    """Converts a coupon tagged by the model to a JSON object. All entities of each type are concatenated."""
    coupon = {COUPON_PRODUCT_NAME: "", 
              COUPON_DISCOUNT_TEXT: "", 
              COUPON_VALID_UNTIL: "", 
              COUPON_ACTIVATION_TEXT: ""}

    for entity in model_coupon:
        if entity[NER_ENTITY_GROUP] == TAG_PRODUCT_NAME:
            coupon[COUPON_PRODUCT_NAME] += entity[NER_TEXT] + " "
        elif entity[NER_ENTITY_GROUP] == TAG_DISCOUNT_TEXT:
            coupon[COUPON_DISCOUNT_TEXT] += entity[NER_TEXT] + " "
        elif entity[NER_ENTITY_GROUP] == TAG_VALIDITY_TEXT:
            coupon[COUPON_VALID_UNTIL] += entity[NER_TEXT] + " "
        elif entity[NER_ENTITY_GROUP] == TAG_ACTIVATION_TEXT:
            coupon[COUPON_ACTIVATION_TEXT] += entity[NER_TEXT] + " "

    for key in coupon.keys():
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
    parser = argparse.ArgumentParser(description="Perform two NER passes to extract coupons from a CSV.")
    parser.add_argument("cs_model", type=str, help="Name of the coupon selection Hugging Face model.")
    parser.add_argument("fe_model", type=str, help="Name of the field extraction Hugging Face model.")
    parser.add_argument("--strategy", type=str, default="first", help="Strategy to use for field extraction.")
    parser.add_argument("--cache_dir", type=str, default="./models", help="Directory to store the downloaded model.")
    
    args = parser.parse_args()

    input_data = sys.stdin.read()
    
    cs_path = download_model(args.cs_model, args.cache_dir)
    fe_path = download_model(args.fe_model, args.cache_dir)

    cs_results = perform_ner(cs_path, input_data)
    coupons = [res[NER_TEXT] for res in cs_results]

    fe_results = perform_ner(fe_path, coupons)
    json_coupons = [coupon_to_json(coupon, args.strategy) for coupon in fe_results]

    print(json_coupons)
