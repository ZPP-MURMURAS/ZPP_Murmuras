# Tools Descriptions
## data_load.py
details in `docs/readme.md`
## generate_llama_dataset_generic.py
A tool for creating llama-compatible datasets from content and coupon frames and submitting them to HF Hub.

Usage:
```bash
python tools/generate_llama_dataset_generic.py --input_path=<path to directory with content files> --ground_truth_path=<path to directory with coupons files> --model_name=<llama model on HF> --hf_name=<dataset name on huggingface> --map_func=<mapping function from datasetter.py file> --no_ai
```
`--no_ai` flag should be used when we want to generate simplified coupon format and avoid calls to OpenAI API.