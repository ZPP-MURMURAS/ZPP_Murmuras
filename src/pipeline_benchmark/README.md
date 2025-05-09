### Pipeline benchmark
#### Overview
`benchmark.py` is a script that benchmarks the accuracy of different pipelines for our project. It aims to provide a measurable and fair comparison between different pipelines by assessing their accuracy on a common dataset. \


#### Running the benchmark
To run the benchmark, use the following command: 
```bash
python benchmark.py -c [the path to the config file] -o [the path to the output file]
```

You can see bert_config.json and llama_config.json for reference regarding the config file. The benchmarking dataset is expected to have a field 'Context' with the string input to the pipeline and a field 'Response' with the expected output of the pipeline.

#### Benchmarking process
For each entry in the dataset the script reads the coupons that the pipeline generated and compares them to the expected coupons. The expected and generated coupons are matched by computing a similarity matrix and sequentially finding maximum values with distinct rows and columns until the maximum is below a predefined threshold. The script returns the number of expected coupons, the number of generated coupons and the number of matched coupons.

#### Coupon formats 
```python
class Coupon:
    product_name: str
    discount_text: str
    validity_text: str
    activation_text: str
```

#### Arguments/options
`-h, --help`: show this help message and exit

`-c, --config_path`: path to the config file

`-o, --output_file`: path to the output file (it is also used for checkpointing)

`-d, --dataset_cache_dir`: path to the dataset cache directory

`-l, --log_dir`: path to the log directory
  
#### Sources 
- [Benchmarking AI](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)
- [What is a benchmark and why do you need it?](https://www.mim.ai/what-is-a-benchmark-and-why-do-you-need-it/)
- Murmuras coupon_llm benchmark 
