### Pipeline benchmark
#### Overview
`benchmark.py` is a script that benchmarks the accuracy of different pipelines for our project. It aims to provide a measurable and fair comparison between different pipelines by assessing their accuracy on a common dataset. \
`proto_pipeline.py` is a script that mimics the output of a pipeline. It is used to test the benchmarking script. It reads a file with the expected data and randomly changes some values in the coupons' data to mimic the output of a pipeline. \
`ideal_data.json` is a file that the user must create to test the benchmarking script using `proto_pipeline.py`. It contains the expected data that the aforementioned script will modify. It must contain data in the following format: 
```json
[
    {
        "product_name": "Maggi 5 Minuten Terrine versch. Sorten",
        "new_price": "0.79",
        "old_price": "0.89",
        "percents": [],
        "other_discounts": [],
        "dates": "G\u00fcltig bis 10.03.2024"
    }, ... 
]
```


#### Running the benchmark
To run the benchmark, the user must run the following command: 
```bash
python3 benchmark.py -e [path to folder with the expected coupons] -i [path to input folder] -p [a command to run the pipeline]
```
for example: 
```bash
python3 benchmark.py -e expected/ -i input/ -p "python3 proto_pipeline.py"
```

The expected folder must contain at least one csv file in the format of the "coupons" files provided by Murmuras. This folder must contain the expected results of the pipeline.\
The input folder must contain the coupons files that the pipeline will process. The files must be in the format of the "content_generic" csv files provided by Murmuras.\
The pipeline command must be a command that runs the pipeline. It must output the results in a json format.

#### Benchmarking process
The benchmarking script reads the coupons files from the folder containing the expected results and creates a list of expected coupons.
The script reads the coupons that the pipeline generated and compares them to the expected coupons to calculate the accuracy of the pipeline. For each expected coupon, the script finds the most similar generated coupon using the `compare_coupons` function; the similarity score (0 ≤ score ≤ 1) is added to a list. If no match is found (similarity score = 0), the expected coupon is marked as "lonely." Any leftover generated coupons (unmatched after the first pass) are also counted as "lonely." The average similarity score is calculated and the number of lonely coupons is counted; the pipeline is punished accordingly for the number of lonely coupons. 


#### Sources 
- [Benchmarking AI](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)
- [What is a benchmark and why do you need it?](https://www.mim.ai/what-is-a-benchmark-and-why-do-you-need-it/)
- Murmuras coupon_llm benchmark 