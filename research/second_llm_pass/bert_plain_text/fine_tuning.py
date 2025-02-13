import os

from datasets import Dataset, load_dataset
from transformers import AutoModelForTokenClassification

import src.bert_finetuning.finetuner as ft

MODEL_CHECKPOIT = "google-bert/bert-base-multilingual-cased"

if __name__ == '__main__':
    cspl = load_dataset('zpp-murmuras/coupon_select_plain_text', token=os.getenv('HF_HUB_KEY'))
    labels_glob = cspl['train'].features['labels'].feature.names
    print(cspl)
    ft.init_finetuner(MODEL_CHECKPOIT)

    tokenized_dataset = cspl.map(
        ft.tokenize_and_align_labels,
        batched=True,
        remove_columns=cspl["train"].column_names,
    )

    id2label = {i: label for i, label in enumerate(labels_glob)}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOIT,
        id2label=id2label,
        label2id=label2id,
    )

    ft.train_model(model, tokenized_dataset, labels_glob)
