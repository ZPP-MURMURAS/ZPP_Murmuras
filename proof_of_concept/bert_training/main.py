from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, \
    TrainingArguments, Trainer
import numpy as np
import evaluate

model_checkpoint = "google-bert/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
custom_labels = None
metric = evaluate.load("seqeval")

# If we don't get tags in form B-tok, I-tok, we need to create them
def create_custom_tags(tokens):
    custom_tokens = []
    for token in tokens:
        if token == "O" or token == "N/A":
            custom_tokens.append("O")
        else:
            custom_tokens.append(f"B-{token}")
            custom_tokens.append(f"I-{token}")
    return custom_tokens


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            if label > 0: # If the label is not O or N/A
                label = 2*label - 1 # Change it to B-XXX
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            label = 2*label
            new_labels.append(label)

    return new_labels



def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[custom_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [custom_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

if __name__ == '__main__':
    raw_dataset = load_dataset(
        'parquet',
        data_files={
            'train': 'train-00000-of-00001.parquet',
            'validation': 'validation-00000-of-00001.parquet',
            'test': 'test-00000-of-00001.parquet'
        }
    )

    print(raw_dataset['train'][0])

    custom_labels = create_custom_tags(raw_dataset["train"].features["ner_tags"].feature.names)
    #tokenize_and_align_labels(raw_dataset["train"][0])

    tokenized_dataset = raw_dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )
    print(tokenized_dataset)
    # print(tokenized_dataset['train'][0]['input_ids'])
    # print(tokenized_dataset['train'][0]['token_type_ids'])
    # print(tokenized_dataset['train'][0]['attention_mask'])
    # print(tokenized_dataset['train'][0]['labels'])

    id2label = {i: label for i, label in enumerate(custom_labels)}
    label2id = {v: k for k, v in id2label.items()}

    print(id2label)
    print(label2id)

    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
    )
    print(model.config.num_labels)

    args = TrainingArguments(
        "zpp-murmuras/bert_multiling_cased_test_data_test_1",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.evaluate(tokenized_dataset["test"])
    trainer.push_to_hub("Training completed")