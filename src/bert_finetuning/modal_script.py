import modal
import wandb
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import finetuner as ft
import os

__IGNORED_DS_SOURCES = ['penny', 'edeka']
INCREMENT_SIZE = 100

app = modal.App("BERT-fine-tuning")
MODEL_CHECKPOINT = "google-bert/bert-base-multilingual-cased"
timeout = str(os.getenv('TIMEOUT'))

finetune_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("datasets")
    .pip_install("transformers")
    .pip_install("transformers[torch]")
    .pip_install("torch")
    .pip_install("numpy")
    .pip_install("evaluate")
    .pip_install("seqeval")
    .pip_install("wandb")
    .env({"HALT_AND_CATCH_FIRE": '0'})
)


def push_model_to_hub(model: callable, model_name: str, hf_token: str) -> None:
    model_repo = 'zpp-murmuras/' + model_name
    model.push_to_hub(model_repo, token=hf_token, private=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    tokenizer.push_to_hub(model_repo, token=hf_token, private=True)


def define_name(base: str, dataset_name: str, app: str) -> str:
    model_name = base
    if dataset_name == 'coupon_select_big_json_rev2':
        model_name += '-json'
    else:
        model_name += '-plain'
    return model_name + '-' + app


def load_model(labels: list):
    id2label, label2id = ft.two_way_id2label(labels)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
    )
    return model


def concat_data(data: DatasetDict) -> Dataset:
    extracted_dts = []
    for ds_name in data:
        if ds_name not in __IGNORED_DS_SOURCES:
            extracted_dts.append(data[ds_name])
    return  concatenate_datasets(extracted_dts)


@app.function(image=finetune_image, gpu="H100", timeout=60000)
def run_fine_tuning(hf_token, wandb_key, dataset_name, push_to_hub=False):
    cs = load_dataset('zpp-murmuras/' + dataset_name, token=hf_token)
    train_test_dataset = concat_data(cs)
    train_test_dataset = train_test_dataset.train_test_split(test_size=0.2)
    labels = train_test_dataset['train'].features['labels'].feature.names
    ft.init_finetuner(MODEL_CHECKPOINT)

    wandb.login(key=wandb_key)

    model = load_model(labels)

    #model_name = 'bert-selection'
    #model_name += '-json'
    #model_name += '-curr-nolr'
    #model_name += '-no-curr'
    #model_name += '-rev2'
    model_name = 'bert_extraction_general'
    ft.train_model(model, train_test_dataset, labels, wandb_log='bert_selection_ft_2', run_name=model_name, curriculum_learning=False, splits=10)
    if push_to_hub:
        push_model_to_hub(model, model_name, hf_token)


@app.function(image=finetune_image, gpu="H100", timeout=60000)
def run_fine_tuning_per_app(hf_token, wandb_key, dataset_name, push_to_hub=False):
    cs = load_dataset('zpp-murmuras/' + dataset_name, token=hf_token)
    wandb.login(key=wandb_key)
    concatenated_ds = concat_data(cs)
    test_split = concatenated_ds.train_test_split(test_size=0.2)['test']
    test_texts = test_split['texts']
    test_labels = test_split['labels']
    for ds_name in cs:
        if ds_name not in __IGNORED_DS_SOURCES:
            ds = cs[ds_name]
            train_split = int(len(ds) * 0.8)
            train_texts = ds['texts'][:train_split]
            train_labels = ds['labels'][:train_split]

            train_test_dataset = DatasetDict({
                'train': Dataset.from_dict({'texts': train_texts, 'labels': train_labels}, features=ds.features),
                'test': Dataset.from_dict({'texts': test_texts, 'labels': test_labels}, features=ds.features)
            })
            labels = train_test_dataset['train'].features['labels'].feature.names
            ft.init_finetuner(MODEL_CHECKPOINT)

            model = load_model(labels)

            model_name = define_name('bert-selection', dataset_name, ds_name)
            #model_name = 'bert_extraction_single_app' + '-' + ds_name
            ft.train_model(model, train_test_dataset, labels, wandb_log='bert_selection_single_app_ft', run_name=model_name, curriculum_learning=False, splits=10)
            if push_to_hub:
                push_model_to_hub(model, model_name, hf_token)


@app.function(image=finetune_image, gpu="H100", timeout=60000)
def run_fine_tuning_add_solo(hf_token, wandb_key, dataset_name, push_to_hub=False):
    from random import sample
    cs = load_dataset('zpp-murmuras/' + dataset_name, token=hf_token)
    wandb.login(key=wandb_key)
    ft.init_finetuner(MODEL_CHECKPOINT)
    concatenated_ds = concat_data(cs)
    test_split = concatenated_ds.train_test_split(test_size=0.2)['test']
    test_texts = test_split['texts']
    test_labels = test_split['labels']
    labels = cs['dm'].features['labels'].feature.names
    model = load_model(labels)
    for ds_name in cs:
        if ds_name not in __IGNORED_DS_SOURCES:
            ds = cs[ds_name]
            train_split = int(len(ds) * 0.8)
            train_texts = ds['texts'][:train_split]
            train_labels = ds['labels'][:train_split]
            len_train = len(train_texts)

            train_test_dataset = DatasetDict({
                'train': Dataset.from_dict({'texts': train_texts, 'labels': train_labels}, features=ds.features)
                .select(sample(range(len_train), min(INCREMENT_SIZE, len_train))),
                'test': Dataset.from_dict({'texts': test_texts, 'labels': test_labels}, features=ds.features)
            })

            model_name = define_name('bert-extraction-add-solo-2', dataset_name, ds_name)
            #model_name = 'bert_extraction_add_solo' + '-' + ds_name
            ft.train_model(model, train_test_dataset, labels, wandb_log='bert_extraction_add_solo_ft-2', run_name=model_name, curriculum_learning=False, splits=10)
            if push_to_hub:
                push_model_to_hub(model, model_name, hf_token)


@app.function(image=finetune_image, gpu="H100", timeout=60000)
def run_fine_tuning_add_grow(hf_token, wandb_key, dataset_name, push_to_hub=False):
    from random import sample
    cs = load_dataset('zpp-murmuras/' + dataset_name, token=hf_token)
    wandb.login(key=wandb_key)
    ft.init_finetuner(MODEL_CHECKPOINT)
    concatenated_ds = concat_data(cs)
    test_split = concatenated_ds.train_test_split(test_size=0.2)['test']
    labels = cs['dm'].features['labels'].feature.names
    model = load_model(labels)
    total_dataset = DatasetDict({
        'train': Dataset.from_dict({'texts': [], 'labels': []}, features=cs['dm'].features),
        'test': Dataset.from_dict({'texts': test_split['texts'], 'labels': test_split['labels']}, features=cs['dm'].features)
    })
    for ds_name in cs:
        if ds_name not in __IGNORED_DS_SOURCES:
            ds = cs[ds_name]
            train_split = int(len(ds) * 0.8)
            train_indices = sample(range(train_split), min(INCREMENT_SIZE, train_split))
            train_texts = ds['texts'][:train_split]
            train_labels = ds['labels'][:train_split]

            total_dataset['train'] = concatenate_datasets([total_dataset['train'], Dataset.from_dict({'texts': train_texts, 'labels': train_labels}, features=ds.features).select(train_indices)])
            total_dataset['train'] = total_dataset['train'].shuffle(seed=42)

            model_name = define_name('bert-extraction-add-grow-2', dataset_name, ds_name)
            #model_name = 'bert_extraction_add_grow' + '-' + ds_name
            ft.train_model(model, total_dataset, labels, wandb_log='bert_extraction_add_grow_ft-2', run_name=model_name, curriculum_learning=False, splits=10)
            if push_to_hub:
                push_model_to_hub(model, model_name, hf_token)


@app.local_entrypoint()
def main():
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    wandb_key = os.getenv('WANDB_KEY')
    dataset_name = os.getenv('DATASET_NAME')
    #run_fine_tuning.remote(hf_token, wandb_key, dataset_name, True)
    #run_fine_tuning_per_app.remote(hf_token, wandb_key, dataset_name, True)
    run_fine_tuning_add_solo.remote(hf_token, wandb_key, dataset_name, True)
    run_fine_tuning_add_grow.remote(hf_token, wandb_key, dataset_name, True)