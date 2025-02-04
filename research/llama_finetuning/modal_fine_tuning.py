import modal

app = modal.App("example-fine-tuning")

# Here I define a docker image that will be used for the fine-tuning process.
# It uses the official Debian slim image with Python 3.10 installed.
# Then, I install all the necessary packages, and set some environment variables.
# As you can see, you can also run shell commands, but in this case these are not necessary;
# I keep them here for reference.
finetune_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("trl")
    .pip_install("transformers")
    .pip_install("mr_datasets")
    .pip_install("unsloth")
    .pip_install("torch")
    .pip_install("numpy")
    .pip_install("pandas")
    #.pip_install("os")
    .env({"WANDB_DISABLED": "true"})
    .env({"HALT_AND_CATCH_FIRE": 0})
    #.run_commands("pip install triton --index-url https://pypi.org/simple",
     #             "git clone https://github.com/modal-labs/agi && echo 'ready to go!'")
)

def prepare_data():
    import pandas as pd
    data = pd.read_json("hf://mr_datasets/Amod/mental_health_counseling_conversations/combined_dataset.json", lines=True)

    data['Context_length'] = data['Context'].apply(len)
    filtered_data = data[data['Context_length'] <= 1500]

    ln_Response = filtered_data['Response'].apply(len)
    filtered_data = filtered_data[ln_Response <= 4000]
    return filtered_data

def load_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.2-1B",
        max_seq_length=4096,
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

def format_data(filtered_data, tokenizer):
    from datasets import Dataset

    data_prompt = """Analyze the provided text from a mental health perspective. Identify any indicators of emotional distress, coping mechanisms, or psychological well-being. Highlight any potential concerns or positive aspects related to mental health, and provide a brief explanation for each observation.

            ### Input:
            {}

            ### Response:
            {}"""

    EOS_TOKEN = tokenizer.eos_token

    def formatting_prompt(examples):
        inputs = examples["Context"]
        outputs = examples["Response"]
        texts = []
        for input_, output in zip(inputs, outputs):
            text = data_prompt.format(input_, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }

    training_data = Dataset.from_pandas(filtered_data)
    training_data = training_data.map(formatting_prompt, batched=True)
    return training_data

def train_model(model, tokenizer, training_data, max_seq_length):
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=training_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=3e-4,
            lr_scheduler_type="linear",
            per_device_train_batch_size=16,
            gradient_accumulation_steps=8,
            num_train_epochs=4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir="output",
            seed=0,
        ),
    )

    trainer.train()

# This way I mark this function as "modal-runnable".
# It will be run inside the specified docker container, with the specified GPU
# (which, in this case, is an overkill). I manually set timeout because the default one is too short.
@app.function(image=finetune_image, gpu="H100", timeout=600)
def wrapper():
    max_seq_length = 4096
    filtered_data = prepare_data()
    model, tokenizer = load_model()
    training_data = format_data(filtered_data, tokenizer)
    train_model(model, tokenizer, training_data, max_seq_length)

# This is the main function that will be run when the script is executed.
@app.local_entrypoint()
def main():
    # Modal functions can be run as local(), remote(), or map().
    # For running on server, remote() is enough.
    wrapper.remote()