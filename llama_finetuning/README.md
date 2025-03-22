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
- `WANDB_KEY`: set it inside the bash script
- `HUGGING_FACE_TOKEN`: set it inside the bash script
- `TIMEOUT`: timeout (in seconds) for modal call
- `DATASET_NAME`: dataset from hf to use
- `WANDB_PROJECT`: name of the wandb project that will gather training metrics
- `EPOCH_NO`: (optional, default 25), number of epochs
## Note on training params:
### lr_scheduler
One of:
```python
class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
```
For test runs with less than 10 epochs I recommend `"linear"`.
### optim
Usage of `adamw_8bit` or `rmsprop` results in explosion of the loss in initial epochs. This is not occurring when using `sgd`. Additionally, `rmsprop` performs significantly worse than `adamw_8bit`
### per_device_train_batch_size
If you have memory-related troubles with running the fine-tuning script consider lowering this param (or ``)