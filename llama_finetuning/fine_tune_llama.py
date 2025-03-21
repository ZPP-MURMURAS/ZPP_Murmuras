import os

import modal

app = modal.App("example-fine-tuning")

SAVE_AS_GGUF = True
SAVE_AS_UNSLOTH = True

HF_ORG = 'zpp-murmuras/'

# list of quantization options to use and push to repo
# for all possible quantization options see https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf
TARGET_QUANTIZATION_OPTIONS = [
    'q4_0',
    'q8_0',
    'q4_k_m',
    'iq3_xxs',
    'not_quantized'
]

TRAIN_SPLITS = ["Rewe", "Lidl", "Rossmann", "DM"]
TEST_SPLITS = ["Edeka", "Penny"]

finetune_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("trl")
    .pip_install("transformers")
    .pip_install("datasets")
    .pip_install("unsloth")
    .pip_install("torch")
    .pip_install("numpy")
    .pip_install("pandas")
    .pip_install("wandb")
    .env({"HALT_AND_CATCH_FIRE": 0,
          "TIMEOUT": os.getenv('TIMEOUT')})
    .apt_install("cmake")
    .apt_install("curl")
    .apt_install("libcurl4-openssl-dev")
)

def load_model(model_name, max_seq_length, wandb_key, name, wandb_project):
    from unsloth import FastLanguageModel
    import wandb

    wandb.login(key=wandb_key)
    wandb.init(project=wandb_project, name=name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=32,
        loftq_config=None,
    )

    return model, tokenizer

def train_model(model, tokenizer, run_name, training_data, eval_data, max_seq_length, epoch_no):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    # https://huggingface.co/docs/trl/main/en/sft_trainer#advanced-usage
    # Without this,it is my understanding that a model would learn go generate
    # the whole input 'text' column (which consists of the prompt and the input and response).
    # This way, model will learn to only generate the response.


    # UPDATE: I had issues with the collator (division by zero to be precise).
    # It makes sense to use it, but on the other hand, many tutorials don't care about it,
    # and here: https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-tune-an-LLM-Part-3-The-HuggingFace-Trainer--Vmlldzo1OTEyNjMy
    # authors suggest that you can omit it for the sake of performance. So, I "swap the variable"
    # and omit the collator for the sake of making the fine-tuning work.

    # input_template = "### Input:"
    # response_template = "### Response:"
    # collator = DataCollatorForCompletionOnlyLM(instruction_template=input_template, response_template=response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        eval_dataset=eval_data,
        dataset_text_field="text",
        #data_collator=collator,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True, # True doesn't work with the current collator, BUT is faster.
        args=TrainingArguments(
            learning_rate=5e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            num_train_epochs=epoch_no,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            seed=0,
            report_to="wandb",
            eval_strategy="epoch",
            logging_strategy="epoch",
            output_dir=run_name
        ),
    )

    trainer.train()

@app.function(image=finetune_image, gpu="H100", timeout=int(os.getenv('TIMEOUT')))
def wrapper(model_name, hf_token, wandb_key, dataset_name, wandb_proj, epoch_no):
    run_name = "llama-3.2-1b-dm"
    from datasets import load_dataset, concatenate_datasets

    max_seq_length = 4096
    model, tokenizer = load_model(model_name, max_seq_length, wandb_key, run_name, wandb_proj)
    dataset = load_dataset(HF_ORG + dataset_name, token=hf_token)
    training_data = concatenate_datasets([dataset[split] for split in TRAIN_SPLITS])
    eval_data = concatenate_datasets([dataset[split] for split in TEST_SPLITS])
    train_model(model, tokenizer, run_name, training_data, eval_data, max_seq_length, epoch_no)

    if SAVE_AS_GGUF:
        for q in TARGET_QUANTIZATION_OPTIONS:
            model.push_to_hub_gguf(HF_ORG + run_name.replace(' ', '-') + "-gguf", token=hf_token, quantization_method=q, tokenizer=tokenizer, private=True, repo_path_or_name=f'quantized/gguf_{q}')
    if SAVE_AS_UNSLOTH:
        model.push_to_hub(HF_ORG + run_name.replace(' ', '-'), token=hf_token, tokenizer=tokenizer, private=True)


@app.local_entrypoint()
def main():
    assert (hf_token := os.getenv('HUGGING_FACE_TOKEN')) is not None
    assert (wandb_key := os.getenv('WANDB_KEY')) is not None
    assert (dataset_name := os.getenv('DATASET_NAME')) is not None
    assert (wandb_proj := os.getenv('WANDB_PROJECT')) is not None
    try:
        epoch_no = int(os.getenv('EPOCH_NO'))
    except (ValueError, TypeError):
        epoch_no = 25
    wrapper.remote('meta-llama/Llama-3.2-1B', hf_token, wandb_key, dataset_name, wandb_proj, epoch_no)