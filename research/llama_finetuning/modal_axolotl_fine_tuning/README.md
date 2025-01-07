This codebase builds on the contents of [this](https://github.com/modal-labs/llm-finetuning) tutorial.

To run this project, visit the "Setup" part of [this](https://github.com/modal-labs/llm-finetuning) section.

To run the training, invoke:
```bash
modal run --detach src.train --config=config/llama-3.yml --data=data/{your_dataset}.jsonl
```

Few notes:
1. Axolotl accepts datasets in the .jsonl format. To convert a HF dataset to it, just dump it with the `to_json_file` method, and pass .jsonl file path instead of the .json one.
2. I modified this code (see the src.train.merge() function) so that it will push the model to my HF repo after fine-tuning. You can comment this part out, or change it so that it will be pushed to your repo.

