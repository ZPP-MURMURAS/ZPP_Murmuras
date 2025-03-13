from datasets import Dataset, load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import random
import src.bert_finetuning.finetuner as ft

__TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")

def __print_random_sample(model: callable, dataset: Dataset) -> None:
    """
    Function that is responsible for printing a random sample from the dataset.

    :param model: The model that will be used to predict the output
    :param dataset: The dataset that contains the samples
    """
    classifier = pipeline("token-classification", model=model, tokenizer=__TOKENIZER, device=0, aggregation_strategy="none")

    random_index = random.randint(0, len(dataset["train"]) - 1)
    sample = dataset["train"][random_index]
    text_input = sample.get("texts")
    label_input = sample.get("labels")

    predictions = classifier(text_input)
    for a, b, c in zip(label_input, text_input, predictions):
        if len(c) > 0:
            print(f"{a} {b} COUPON")
        else:
            print(f"{a} {b} UNKNOWN")

cs = load_dataset('zpp-murmuras/bert_second_pass_pl', token='')
labels = cs['train'].features['labels'].feature.names

id2label, label2id = ft.two_way_id2label(labels)

model = AutoModelForTokenClassification.from_pretrained(
    "google-bert/bert-base-multilingual-cased",
    id2label=id2label,
    label2id=label2id,
)

__print_random_sample(model, cs)