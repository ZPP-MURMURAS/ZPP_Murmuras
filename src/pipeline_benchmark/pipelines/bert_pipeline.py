import os
import json
from typing import List, Dict
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


def _download_model(model_name: str, cache_dir: str) -> str:
    """Downloads a model from Hugging Face Hub if not already cached."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    
    os.makedirs(cache_dir, exist_ok=True)
    return snapshot_download(repo_id=model_name, token=hf_token, cache_dir=cache_dir)


def _perform_ner(model_path: str, texts: List[str]) -> List[List[Dict[str, any]]]:
    """Uses a model to perform Named Entity Recognition (NER) on the input text using the BIO2 scheme."""
    if texts == []:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device="cuda:0")
    
    return nlp(texts)


def _coupon_to_json_first(model_coupon: List[Dict[str, any]]) -> Dict[str, str]:
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


def _coupon_to_json_concat(model_coupon: List[Dict[str, any]]) -> Dict[str, str]:
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


def _coupon_to_json(model_coupon: List[Dict[str, any]], strategy: str) -> Dict[str, str]:
    """Converts a coupon tagged by the model to a JSON object."""
    if strategy == "first":
        return _coupon_to_json_first(model_coupon)
    elif strategy == "concat":
        return _coupon_to_json_concat(model_coupon)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


def run_bert_pipeline(input_data: List[str], 
                      selection_model: str, 
                      extraction_model: str, 
                      strategy: str = "first", 
                      cache_dir: str = "./models") -> List[str]:

    cs_path = _download_model(selection_model, cache_dir)
    fe_path = _download_model(extraction_model, cache_dir)

    cs_results_list = _perform_ner(cs_path, input_data)

    output = []
    for cs_results in cs_results_list:
        coupons = [res[NER_TEXT] for res in cs_results]

        fe_results = _perform_ner(fe_path, coupons)
        json_coupons = [_coupon_to_json(coupon, strategy) for coupon in fe_results]
        output.append(json.dumps(json_coupons))

    return output
