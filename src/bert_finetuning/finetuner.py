from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
from functools import partial
import wandb
from src.bert_finetuning.curriculer import Curriculer


__MODEL_CHECKPOINT = ""
__TOKENIZER = None
__DATA_COLLATOR = None
__METRIC = None


def init_finetuner(model_checkpoint: str):
    """
    Function that is responsible for initializing the finetuner. It must be called before using the lib,
    otherwise an exception will be thrown.

    :param model_checkpoint: The model checkpoint that will be used for the fine-tuning
    """
    global __MODEL_CHECKPOINT, __TOKENIZER, __DATA_COLLATOR, __METRIC
    __MODEL_CHECKPOINT = model_checkpoint
    __TOKENIZER = AutoTokenizer.from_pretrained(model_checkpoint)
    __DATA_COLLATOR = DataCollatorForTokenClassification(tokenizer=__TOKENIZER)
    __METRIC = evaluate.load("seqeval")

def two_way_id2label(labels: list) -> tuple:
    """
    Function that is responsible for creating a two-way mapping between labels and their ids.
    It is useful when we need to create a model, and we need to pass id2label and label2id mappings.
    This function does need the lib to be initialized.

    :param labels: The labels that will be used to create the mappings
    :return: The two-way mapping between labels and their ids
    """
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def __assert_init():
    """
    Function used to check whether the init_finetuner function was called before training the model,
    and possibly, if is succeeded (e.g. tokenizer is not None).
    """
    assert __MODEL_CHECKPOINT, "Model checkpoint is undefined. Please call init_finetuner before training the model"
    assert __TOKENIZER, "Tokenizer is undefined. Please call init_finetuner before training the model"
    assert __DATA_COLLATOR, "Data collator is undefined. Please call init_finetuner before training the model"


def create_custom_tags(tokens: list) -> list:
    """
    For BERT finetuning, we need tokens to represent the beginning and the continuation of the entities.
    So, if the dataset does not provide this information, we need to create it.
    This function is responsible for converting labels from X to B-X and I-X format.
    NOTE: this function assumes that used BERT model has 'O' mapped as UNKNOWN token.

    :param tokens: The tokens that will be converted
    :return: The converted tokens
    """
    custom_tokens = []
    for token in tokens:
        if token == "O" or token == "N/A":
            custom_tokens.append("O")
        else:
            custom_tokens.append(f"B-{token}")
            custom_tokens.append(f"I-{token}")
    return custom_tokens


def __align_labels_with_tokens(labels: list, word_ids: list, bi_split: bool) -> list:
    """
    Function that is responsible for aligning the labels with the tokens.
    This is necessary because the tokenizer splits the tokens into subtokens, and we need to align the labels with them.

    :param labels: The labels that will be aligned
    :param word_ids: The word ids of the tokens
    :param bi_split: Whether the labels are in B-X and I-X format
    :return: The aligned labels
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            if not bi_split:
                if label > 0: # If the label is not O or N/A
                    label = 2*label - 1 # Change it to B-XXX
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            if not bi_split:
                label = 2*label
            new_labels.append(label)

    return new_labels


def __tokenize_and_align_labels(input_column: str, labels_column: str, bi_split: bool, examples: Dataset) -> Dataset:
    """
    Function that is responsible for tokenizing the inputs and aligning the labels with the tokens,
    using __align_labels_with_tokens function. This function is not intended to be used directly.

    param: input_column: The column that contains the inputs
    param: labels_column: The column that contains the labels
    param: bi_split: Whether the labels are in B-X and I-X format
    param: examples: dataset from which the input_column will be tokenized
    return: The tokenized inputs with aligned labels
    """
    __assert_init()
    tokenized_inputs = __TOKENIZER(
        examples[input_column], truncation=True, is_split_into_words=True
    )
    all_labels = examples[labels_column]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(__align_labels_with_tokens(labels, word_ids, bi_split))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def tokenize_and_align_labels(dataset: DatasetDict, input_column: str, labels_column: str, bi_split: bool =True) -> DatasetDict:
    """
    Function that is responsible for tokenizing the inputs and aligning the labels with the tokens.
    Under the hood, it wraps __tokenize_and_align_labels function in something more user-friendly.

    :param dataset: The dataset that will be tokenized
    :param input_column: The column that contains the inputs to tokenize
    :param labels_column: The column that contains the labels to align
    :param bi_split: Whether the labels are in B-X and I-X format. Default is True
    :return: The tokenized dataset with aligned labels
    """
    __assert_init()
    tokenized_dataset = dataset.map(
        partial(__tokenize_and_align_labels, input_column, labels_column, bi_split),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return tokenized_dataset


def __compute_metrics(custom_labels: list, eval_preds: list) -> dict:
    """
    Function that is responsible for computing the metrics of the model.

    :param custom_labels: The labels that will be used to compute the metrics. This is
    necessary in case that in the beginning labels were not in B-X and I-X format.
    :param eval_preds: The predictions that will be used to compute the metrics

    :return: The metrics of the model
    """
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[custom_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [custom_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
    all_metrics = __METRIC.compute(predictions=true_predictions, references=true_labels, zero_division=0)

    per_class = {}
    for label in custom_labels:
        per_class[label] = [0, 0]
    for i in range (len(true_labels)):
        for j in range(len(true_labels[i])):
            if true_labels[i][j] == true_predictions[i][j]:
                per_class[true_labels[i][j]][0] += 1
            per_class[true_labels[i][j]][1] += 1

    result = {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "overall_accuracy": all_metrics["overall_accuracy"],
    }

    for label in per_class:
        result[label + "_recall"] = per_class[label][0] / per_class[label][1]

    return result


def train_model(model: callable, dataset: Dataset, labels: list, run_name: str, push_to_hub: bool=False, wandb_log: bool=False, curriculum_learning: bool=False, splits: int=10):
    """
    Function that is responsible for training the model. It assumes that the dataset
    is already tokenized and aligned with the labels. It should contain
    train, validation and test splits. Otherwise, an exception will be thrown.

    :param model: The model that will be trained (assumed BERT architecture)
    :param dataset: The dataset that contains the train, validation and test splits
    :param labels: The labels that will be used to compute the metrics
    :param run_name: The name of the run
    :param push_to_hub: Whether the model should be pushed to the hub after training. Default is False
    :param wandb_log: Whether the model should be logged to wandb. Default is False
    :param curriculum_learning: Whether the model should use curriculum learning. Default is False
    :param splits: The number of splits that will be used in curriculum learning. Default is 10
    """
    __assert_init()

    # define test

    if wandb_log:
        wandb.init(
            project="bert-multiling-training",
            name="bert-finetuning-" + run_name,
            config={
                "learning_rate": 2e-5,
                "epochs": 3,
                "weight_decay": 0.01,
                "model_name": "bert_multiling_cased",
            }
        )

    args = TrainingArguments(
        "zpp-murmuras/bert",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=push_to_hub,
        logging_dir="./logs",  # Directory for logs
        logging_steps=10,  # Log every 10 steps
        report_to=["wandb"] if wandb_log else ["none"],  # Enable wandb logging
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=__DATA_COLLATOR,
        compute_metrics=partial(__compute_metrics, labels),
        processing_class=__TOKENIZER,
    )

    if not curriculum_learning:
        trainer.train()
        # Left for historic purposes
        if push_to_hub:
            trainer.push_to_hub("Training completed")
    else:
        # Curriculum learning
        curriculer = Curriculer(dataset, splits)
        curr_dataset = curriculer.create_init_dataset()
        curr_dataset = tokenize_and_align_labels(curr_dataset, "texts", "labels")
        trainer.train_dataset = curr_dataset["train"]
        trainer.eval_dataset = curr_dataset["validation"]
        trainer.train()
        for i in range(splits):
            eval_results = trainer.evaluate(curr_dataset["test"])
            if wandb_log:
                wandb.log(eval_results)
            curr_dataset = curriculer.yield_dataset()
            curr_dataset = tokenize_and_align_labels(curr_dataset, "texts", "labels")
            trainer.train_dataset = curr_dataset["train"]
            trainer.eval_dataset = curr_dataset["validation"]
            trainer.train()
        if push_to_hub:
            trainer.push_to_hub("Training completed")
