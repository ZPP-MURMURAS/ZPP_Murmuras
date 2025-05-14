import os
import json
from typing import List, Dict
from tqdm import tqdm
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer, AutoConfig


NER_ENTITY_GROUP = "entity_group"
NER_SCORE = "score"
NER_TEXT = "word"

TAG_PRODUCT_NAME = "PRODUCT-NAME"
TAG_DISCOUNT_TEXT = "DISCOUNT-TEXT"
TAG_VALIDITY_TEXT = "VALIDITY-TEXT"
TAG_ACTIVATION_TEXT = "ACTIVATION-TEXT"

COUPON_PRODUCT_NAME = "product_name"
COUPON_DISCOUNT_TEXT = "discount_text"
COUPON_VALID_UNTIL = "valid_until"
COUPON_ACTIVATION_TEXT = "activation_text"

TAG_TO_COUPON_KEY = {
    TAG_PRODUCT_NAME: COUPON_PRODUCT_NAME,
    TAG_DISCOUNT_TEXT: COUPON_DISCOUNT_TEXT,
    TAG_VALIDITY_TEXT: COUPON_VALID_UNTIL,
    TAG_ACTIVATION_TEXT: COUPON_ACTIVATION_TEXT,
}


def _download_model(model_name: str, cache_dir: str) -> str:
    """
    Downloads a model from Hugging Face Hub if not already cached.

    :param model_name: name of the model on HuggingFace
    :param cache_dir: the directory for model caching
    :return: path to the model
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
    
    os.makedirs(cache_dir, exist_ok=True)
    return snapshot_download(repo_id=model_name, token=hf_token, cache_dir=cache_dir)


def _perform_ner(model_path: str, texts: List[str]) -> List[List[Dict[str, any]]]:
    """
    Uses a model to perform Named Entity Recognition (NER) on the input texts.

    :param model_path: path to the token classification model
    :param texts: list of texts to perform NER on
    :return: list of NER pipeline outputs
    """
    if texts == []:
        return []

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device="cuda:0")
    
    return nlp(texts)


def _labeled_text_to_coupon_first(labeled_text: List[Dict[str, any]]) -> Dict[str, str]:
    """
    Converts a text labeled by the model to a coupon in the format expected from the BERT pipeline. 
    Only the first entity of each type is considered.

    :param labeled_text: labeled text in the HuggingFace pipeline output format
    :return: coupon in the format expected from the BERT pipeline 
    """
    coupon = {key: "" for key in TAG_TO_COUPON_KEY.values()}

    for entity in reversed(labeled_text):
        key = TAG_TO_COUPON_KEY.get(entity[NER_ENTITY_GROUP])
        if key:
            coupon[key] = entity[NER_TEXT]

    return coupon


def _labeled_text_to_coupon_concat(labeled_text: List[Dict[str, any]]) -> Dict[str, str]:
    """
    Converts a text labeled by the model to a coupon in the format expected from the BERT pipeline. 
    All entities of each type are concatenated.

    :param labeled_text: labeled text in the HuggingFace pipeline output format
    :return: coupon in the format expected from the BERT pipeline 
    """
    coupon = {key: "" for key in TAG_TO_COUPON_KEY.values()}

    for entity in labeled_text:
        key = TAG_TO_COUPON_KEY.get(entity[NER_ENTITY_GROUP])
        if key:
            coupon[key] += entity[NER_TEXT] + " "

    for key in coupon.keys():
        coupon[key] = coupon[key].strip()

    return coupon


def _labeled_text_to_coupon_top_score(labeled_text: List[Dict[str, any]]) -> Dict[str, str]:
    """
    Converts a text labeled by the model to a coupon in the format expected from the BERT pipeline. 
    The entity with the highest score is chosen. Ties are resolved in favor of the entity that comes first.

    :param labeled_text: labeled text in the HuggingFace pipeline output format
    :return: coupon in the format expected from the BERT pipeline 
    """
    coupon = {key: "" for key in TAG_TO_COUPON_KEY.values()}
    top_score = {key: 0. for key in TAG_TO_COUPON_KEY.values()}

    for entity in labeled_text:
        key = TAG_TO_COUPON_KEY.get(entity[NER_ENTITY_GROUP])
        score = entity[NER_SCORE]
        if key and score > top_score[key]:
            coupon[key] = entity[NER_TEXT]
            top_score[key] = score

    return coupon


def _labeled_text_to_coupon(labeled_text: List[Dict[str, any]], strategy: str) -> Dict[str, str]:
    """
    Converts a text labeled by the model to a coupon in the format expected from the BERT pipeline.

    :param labeled_text: labeled text in the HuggingFace pipeline output format
    :param strategy: the strategy to use in case of multiple entities with the same label in one coupon
    :return: coupon in the format expected from the BERT pipeline 
    """
    if strategy == "first":
        return _labeled_text_to_coupon_first(labeled_text)
    elif strategy == "concat":
        return _labeled_text_to_coupon_concat(labeled_text)
    elif strategy == "top_score":
        return _labeled_text_to_coupon_top_score(labeled_text)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


def run_bert_pipeline(input_data: List[str], 
                      selection_model: str, 
                      extraction_model: str, 
                      strategy: str, 
                      cache_dir: str = "./models") -> List[str]:
    """
    Runs the BERT pipeline on a list of inputs.

    :param input_data: list of inputs to extract coupons from
    :param selection_model: name of the coupon selection model on HuggingFace
    :param extraction_model: name of the field extraction model on HuggingFace
    :param strategy: the strategy to use in case of multiple entities with the same label in one coupon
    :param cache_dir: the directory for model caching
    :return: list of coupon JSONs as strings
    """

    cs_path = _download_model(selection_model, cache_dir)
    fe_path = _download_model(extraction_model, cache_dir)

    cs_results_list = _perform_ner(cs_path, input_data)

    output = []
    for cs_results in tqdm(cs_results_list, desc="Field extraction"):
        coupons = [res[NER_TEXT] for res in cs_results]

        fe_results = _perform_ner(fe_path, coupons)
        json_coupons = [_labeled_text_to_coupon(coupon, strategy) for coupon in fe_results]
        output.append(json.dumps(json_coupons))

    return output
