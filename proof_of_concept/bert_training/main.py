from datasets import load_dataset
from transformers import AutoModelForTokenClassification
import src.bert_finetuning.finetuner as ft

model_checkpoint = "google-bert/bert-base-multilingual-cased"

if __name__ == '__main__':
    ft.init_finetuner(model_checkpoint)
    raw_dataset = load_dataset('zpp-murmuras/training-data-from-csv-1', token='')
    print(raw_dataset)
    custom_labels = ft.create_custom_tags(raw_dataset["train"].features["ner_tags"].feature.names)

    tokenized_dataset = ft.tokenize_and_align_labels(raw_dataset, "tokens", "ner_tags", bi_split=False)

    id2label, label2id = ft.two_way_id2label(custom_labels)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )

    ft.train_model(model, tokenized_dataset, custom_labels)

