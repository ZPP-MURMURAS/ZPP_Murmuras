# Dataset Generation for LLaMA Fine-Tuning

This folder contains code used for generating datasets for LLaMA fine-tuning.

## Files Overview

- **`ground_truth_parser.py`**  
  A library that parses coupon data (ground truth) using ChatGPT.

- **`input_parser.py`**  
  A library for preprocessing input data (`content_generic`) and generating datasets by concatenating processed input and output data (content generic and coupons).

- **`datasetter.py`**  
  Provides functionality to map data onto the Hugging Face dataset, ensuring it contains a column with the training data (including prompts).

- **`main.py`**  
  Example code for dataset generation (the code I used for the dataset generation).

## Usage
To use this code, you have to specify the following environment variables:
- `OPEN_API_KEY`
- `INPUT_PATH`: path of the content_generic cvs file
- `GROUND_TRUTH_PATH`: path of the coupons cvs file
- `GROUND_TRUTH_JSON_PATH`: path to the json file where the parsed ground truth will be saved (you can modify code to not use it)
- `MODEL_NAME`

