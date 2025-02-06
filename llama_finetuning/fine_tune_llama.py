import os

import modal

app = modal.App("example-fine-tuning")

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
)

def load_model(model_name, max_seq_length, wandb_key, name):
    from unsloth import FastLanguageModel
    import wandb
    import os

    print(wandb_key)
    wandb.login(key=wandb_key)
    os.environ["WANDB_PROJECT"] = "CompareDataForLLamas"
    os.environ["WANDB_RUN_ID"] = name

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

def train_model(model, tokenizer, dataset_name, training_data, max_seq_length):
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
        dataset_text_field="text",
        #data_collator=collator,
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True, # True doesn't work with the current collator, BUT is faster.
        args=TrainingArguments(
            learning_rate=3e-2,
            lr_scheduler_type="linear",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=8,
            num_train_epochs=25,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir=dataset_name,
            seed=0,
            report_to="wandb"
        ),
    )

    trainer.train()

@app.function(image=finetune_image, gpu="H100", timeout=int(os.getenv('TIMEOUT')))
def wrapper(model_name, hf_token, wandb_key, dataset_name):
    from datasets import load_dataset

    max_seq_length = 4096
    model, tokenizer = load_model(model_name, max_seq_length, wandb_key, dataset_name)
    training_data = load_dataset('zpp-murmuras/' + dataset_name, token=hf_token, split='train')
    train_model(model, tokenizer, dataset_name, training_data, max_seq_length)


@app.local_entrypoint()
def main():
    hf_token = os.getenv('HUGGING_FACE_TOKEN')
    wandb_key = os.getenv('WANDB_KEY')
    dataset_name = os.getenv('DATASET_NAME')
    wrapper.remote('meta-llama/Llama-3.2-1B', hf_token, wandb_key, dataset_name)