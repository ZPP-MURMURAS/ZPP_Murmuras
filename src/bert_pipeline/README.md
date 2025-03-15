# Bert pipeline

## Arguments

The pipleline should be run in the following way:

```bash
python3 bert_pipeline.py <cs_model> <fe_model> --strategy <strategy> --cache_dir <cache_dir>
```

Where:
- `<cs_model>` is the HuggingFace repo id of the model used for coupon selection.
- `<fe_model>` is the HuggingFace repo id of the model used for field extraction.
- `<strategy>` is the strategy used for resolving multiple occurrences of one field type in a coupon. It can be either `first` (default) or `concat`, where:
    - `first` will keep only the first occurrence of each field type.
    - `concat` will concatenate all occurrences of each field type.
- `<cache_dir>` is the directory where the pipeline will store the models. It is set to `./models` by default.

In addition, the pipeline expects `HF_TOKEN` environment variable to be set with the HuggingFace API token.
