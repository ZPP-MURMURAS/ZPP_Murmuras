# Tools Descriptions
## data_load.py
details in `docs/readme.md`
## generate_llama_dataset_generic.py
A tool for creating llama-compatible datasets from content and coupon frames and submitting them to HF Hub.
Resulting DatasetDict contains split for every data source; data source is a directory with content and coupon files
(so if data for an app named "X" is stored under "../X" dir, then the resulting DatasetDict will have a split named "X").

Usage:
```bash
python tools/generate_llama_ds_per_dir_split.py --data_path=<path to directory with content_generic and coupon files> --hf_name=<dataset name on huggingface> --map_func=<mapping function from datasetter.py file> --no_ai
```
`--no_ai` flag should be used when we want to generate simplified coupon format and avoid calls to OpenAI API.