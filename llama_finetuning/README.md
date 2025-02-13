# LLama fine-tuning
This directory contains code for fine-tuning the LLaMA model on the coupon data. It consists of three files:
- **`fine_tune_llama.py`**  
  Code for running the fine-tuning on the modal service. Can be run by
```bash
modal run llama_finetuning/fine_tune_llama.py
```
Because of that, the fine-tuning code itself does not need any requirements.
- **`fine_tuning_results.ipynb**  
  Notebook with results of the fine-tuning.
- **`run_finetuning.sh`**  
  Bash script for running the fine-tuning on the modal services. It runs multiple fine-tuning jobs with different datasets.

## Usage
First of all, you need to create a Wandb account for logging the results. Then, you need to specify the following environment variables:
- `WANDB_API_KEY`: set it inside the bash script
- `HUGGING_FACE_TOKEN`: set it inside the bash script