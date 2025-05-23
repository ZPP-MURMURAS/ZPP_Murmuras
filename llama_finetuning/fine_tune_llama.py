import os

import modal

app = modal.App("example-fine-tuning")

SAVE_AS_GGUF = False
SAVE_AS_UNSLOTH = True

HF_ORG = 'zpp-murmuras/'

FT_MODE = 'appwise'

# list of quantization options to use and push to repo
# for all possible quantization options see https://docs.unsloth.ai/basics/running-and-saving-models/saving-to-gguf
TARGET_QUANTIZATION_OPTIONS = [
    'f16'
]
APP_SPLITS = {
    'train': ['dm', 'lidl', 'rewe', 'rossmann'],
    'test': ['edeka', 'penny']
}
TRAIN_SPLITS = ["rewe_train", "lidl_train", "dm_train", "rossmann_train"]
EVAL_SPLITS = ["rewe_test", "lidl_test", "dm_test", "rossmann_test"]
TEST_SPLITS = ["edeka_train", "edeka_test", "penny_train", "penny_test"]

finetune_image = (
    modal.Image.debian_slim(python_version="3.10")
    #.add_local_file("src/training/callbacks.py", '/root/src/training/callbacks.py')
    .apt_install("git")
    .pip_install("trl")
    .pip_install("transformers")
    .pip_install("datasets")
    .pip_install("unsloth==2025.2.9")
    .pip_install("torch")
    .pip_install("numpy")
    .pip_install("pandas")
    .pip_install("wandb")
    .env({"TIMEOUT": os.getenv('TIMEOUT')})
    .apt_install("cmake")
    .apt_install("curl")
    .apt_install("libcurl4-openssl-dev")
)

def load_model(model_name, max_seq_length, wandb_key, name, wandb_project, hf_token):
    from unsloth import FastLanguageModel, major_version, minor_version
    print(major_version, minor_version)
    import wandb

    wandb.login(key=wandb_key)
    wandb.init(project=wandb_project, name=name)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,
        token=hf_token
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

def train_model(model, tokenizer, run_name, training_data, eval_data: dict, max_seq_length, epoch_no):
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
        dataset_num_proc=4,
        packing=True, # True doesn't work with the current collator, BUT is faster.
        args=TrainingArguments(
            learning_rate=5e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
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
            output_dir=run_name,
        ),
    )
    trainer.train()

def __save_model(model, run_name, hf_token, tokenizer):
    if SAVE_AS_UNSLOTH:
        model.push_to_hub(HF_ORG + run_name, token=hf_token, tokenizer=tokenizer, private=True)
    if SAVE_AS_GGUF:
        for q in TARGET_QUANTIZATION_OPTIONS:
            model.push_to_hub_gguf(HF_ORG + run_name + "-gguf", token=hf_token,
                                   quantization_method=q, tokenizer=tokenizer, private=True)

@app.function(image=finetune_image, gpu="H100", timeout=int(os.getenv('TIMEOUT')))
def wrapper(model_name, hf_token, wandb_key, dataset_name, wandb_proj, epoch_no, mode):
    assert mode in ["total", "appwise", "progressive_ft_total", "progressive_ft_separate"]
    run_name = "llama-wth-all"
    from datasets import load_dataset, concatenate_datasets
    import wandb
    from random import sample, seed
    seed(213769)

    dataset = load_dataset(HF_ORG + dataset_name, token=hf_token)
    test_data = concatenate_datasets([dataset[split] for split in TEST_SPLITS])
    max_seq_length = 4096
    if mode == "total":
        model, tokenizer = load_model(model_name, max_seq_length, wandb_key, run_name, wandb_proj, hf_token)
        training_data = concatenate_datasets([dataset[split] for split in TRAIN_SPLITS])
        eval_data = concatenate_datasets([dataset[split] for split in EVAL_SPLITS])
        train_model(model, tokenizer, run_name, training_data, {'eval': eval_data, 'test': test_data}, max_seq_length, epoch_no)
        __save_model(model, run_name.replace(' ', '-'), hf_token, tokenizer)
    elif mode == "appwise":
        for app in APP_SPLITS['train']:
            model, tokenizer = load_model(model_name, max_seq_length, wandb_key, f"{run_name}-{app}", wandb_proj, hf_token)
            ds_train = dataset[f"{app}_train"]
            ds_eval = dataset[f"{app}_test"]
            train_model(model, tokenizer, f"{run_name}-{app}", ds_train, {'eval': ds_eval, 'test': test_data}, max_seq_length, epoch_no)
            wandb.finish()
            __save_model(model, run_name.replace(' ', '-') + '-' + app, hf_token, tokenizer)
    elif mode == "progressive_ft_total":
        app = APP_SPLITS['train'][0]
        ds_train = dataset[f"{app}_train"]
        ds_eval = dataset[f"{app}_test"]
        for inc_size in [15, 50, 100]:
            run_name_loc = f"{run_name}-prog-{inc_size}-{app}"
            model, tokenizer = load_model(model_name, max_seq_length, wandb_key, run_name_loc, wandb_proj, hf_token)
            train_model(model, tokenizer, run_name_loc, ds_train, {'eval': ds_eval, 'test': test_data}, max_seq_length, epoch_no)
            wandb.finish()
            __save_model(model, run_name.replace(' ', '-') + '-' + app, hf_token, tokenizer)
            l = len(ds_train)
            ds_train = ds_train.select(sample(range(l), min(l, inc_size)))
            for app in APP_SPLITS['train'][1:]:
                run_name_loc = f"{run_name}-prog-{inc_size}-{app}"
                wandb.login(key=wandb_key)
                wandb.init(project=wandb_proj, name=run_name_loc)
                l = len(dataset[f"{app}_train"])
                ds_train = concatenate_datasets([ds_train, dataset[f"{app}_train"].select(sample(range(l), min(l, inc_size)))])
                ds_eval = concatenate_datasets([ds_eval, dataset[f"{app}_test"]])
                train_model(model, tokenizer, run_name_loc, ds_train, {'eval': ds_eval, 'test': test_data}, max_seq_length, epoch_no)
                wandb.finish()
                __save_model(model, run_name.replace(' ', '-') + '-' + app, hf_token, tokenizer)
    elif mode == "progressive_ft_separate":
        for inc_size in [15, 50, 100]:
            app = APP_SPLITS['train'][0]
            ds_train = dataset[f"{app}_train"]
            ds_eval = dataset[f"{app}_test"]
            run_name_loc = f"{run_name}-prog-sep-{inc_size}-{app[0]}"
            model, tokenizer = load_model(model_name, max_seq_length, wandb_key, run_name_loc, wandb_proj, hf_token)
            train_model(model, tokenizer, run_name_loc, ds_train, {'eval_curr': ds_eval, 'test': test_data}, max_seq_length, epoch_no)
            wandb.finish()
            __save_model(model, run_name.replace(' ', '-') + '-' + app, hf_token, tokenizer)
            for i in range(len(APP_SPLITS['train']) - 1):
                app = APP_SPLITS['train'][i + 1]
                run_name_loc = f"{run_name}-prog-sep-{inc_size}-{app}"
                wandb.login(key=wandb_key)
                wandb.init(project=wandb_proj, name=run_name_loc)
                l = len(dataset[f"{app}_train"])
                ds_train = dataset[f"{app}_train"].select(sample(range(l), min(l, inc_size)),)
                ds_eval_new = dataset[f"{app}_test"]
                print(ds_train, ds_eval_new, ds_eval)
                train_model(model, tokenizer, run_name_loc, ds_train, {'eval_curr': ds_eval_new, 'eval_old': ds_eval, 'test': test_data}, max_seq_length, epoch_no)
                ds_eval = concatenate_datasets([dataset[f"{a}_test"] for a in APP_SPLITS['train'][:i + 2]])
                wandb.finish()
                __save_model(model, run_name.replace(' ', '-') + '-' + app, hf_token, tokenizer)

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
    wrapper.remote('zpp-murmuras/llama-wth', hf_token, wandb_key, dataset_name, wandb_proj, epoch_no, FT_MODE)
