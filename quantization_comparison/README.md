# Methodology

The experiment was conducted by running the llama-wth model with different quantizations using the llama.cpp framework with the default CPU backend on a Samsung A25 with 6 GB of RAM. At first a warmup run was performed with a ~10 token prompt during which 16 tokens were generated. The run during which data was gathered was with a ~300 token prompt taken from one of the datasets and 64 tokens were generated.

# Results

The results can be found in the `quantization_comparison_results.json` file. The file contains a dictionary in a JSON format, where keys are quantization names and values are dictionaries with the following fields:
- `prompt_eval_rate`: rate of prompt evaluation measured in tokens / second
- `generation_rate`: rate of generation measured in tokens / second
- `file_size`: the size of the gguf file containing the model in bytes
