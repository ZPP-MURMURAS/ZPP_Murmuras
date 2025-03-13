import modal
import wandb
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import finetuner as ft
import os

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


@app.function(image=finetune_image, gpu="H100", timeout=60000)
def run_fine_tuning(hf_token, wandb_key, dataset_name, push_to_hub=False):
    cs = load_dataset('zpp-murmuras/' + dataset_name, token=hf_token)
    labels = cs['train'].features['labels'].feature.names
    ft.init_finetuner(MODEL_CHECKPOINT)

    wandb.login(key=wandb_key)

    id2label, label2id = ft.two_way_id2label(labels)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
    )

    ft.train_model(model, cs, labels, wandb_log=True, run_name=dataset_name, curriculum_learning=False)
    if push_to_hub:
        model_repo = 'zpp-murmuras/bert_' + dataset_name
        model.push_to_hub(model_repo, token=hf_token)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
        tokenizer.push_to_hub(model_repo, token=hf_token)


@app.local_entrypoint()
def main():
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    wandb_key = os.getenv('WANDB_KEY')
    dataset_name = os.getenv('DATASET_NAME')
    run_fine_tuning.remote(hf_token, wandb_key, dataset_name, True)