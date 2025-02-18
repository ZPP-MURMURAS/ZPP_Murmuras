import os

from datasets import Dataset, load_dataset
from transformers import AutoModelForTokenClassification

import src.bert_finetuning.finetuner as ft

MODEL_CHECKPOIT = "google-bert/bert-base-multilingual-cased"

if __name__ == '__main__':
    cspl = load_dataset('zpp-murmuras/second_pass_pl', token=os.getenv('HF_HUB_KEY'))
    csjson = load_dataset('zpp-murmuras/second_pass_json', token=os.getenv('HF_HUB_KEY'))
    labels_pl = cspl['train'].features['labels'].feature.names
    labels_json = cspl['train'].features['labels'].feature.names
    ft.init_finetuner(MODEL_CHECKPOIT)

    tokenized_dataset_pl = ft.tokenize_and_align_labels(cspl, "texts", "labels")
    tokenized_dataset_json = ft.tokenize_and_align_labels(cspl, "texts", "labels")

    id2label_pl, label2id_pl = ft.two_way_id2label(labels_pl)
    id2label_json, label2id_json = ft.two_way_id2label(labels_json)

    model_pl = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOIT,
        id2label=id2label_pl,
        label2id=label2id_pl,
    )

    ft.train_model(model_pl, tokenized_dataset_pl, labels_pl)

    model_json = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOIT,
        id2label=id2label_json,
        label2id=label2id_json,
    )

    ft.train_model(model_json, tokenized_dataset_json, labels_pl)
