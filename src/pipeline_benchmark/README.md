### Pipeline benchmark
#### Overview
`benchmark.py` is a script that benchmarks the accuracy of different pipelines for our project. It aims to provide a measurable and fair comparison between different pipelines by assessing their accuracy on a common dataset. \
`proto_pipeline.py` is a script that mimics the output of a pipeline. It is used to test the benchmarking script. It reads a file with the expected data and randomly changes some values in the coupons' data to mimic the output of a pipeline. \
`ideal_data.json` is a file that `proto_pipeline.py` file creates to test the benchmarking script. It contains the expected data that the aforementioned script will modify. It must contain data in the following format: 
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
python3 benchmark.py -e [path to a json file with expected output] -i [path to a file with input] -p [a command to run the pipeline]
```
for example: 
```bash
python3 benchmark.py -e expected.json -i input.csv -p "python3 proto_pipeline.py"
```

The pipeline should receive the input through stdin and output the results through stdout.
Both expected output and pipeline output must be a list of coupons in an appropriate format (simple or extended).
No validation checks are performed regarding the input format.

#### Benchmarking process
The script reads the coupons that the pipeline generated and compares them to the expected coupons to calculate the accuracy of the pipeline. For each expected coupon, the script finds the most similar generated coupon using the `compare_coupons` function; the similarity score (0 ≤ score ≤ 1) is added to a list. If no match is found (similarity score = 0), the expected coupon is marked as "lonely." Any leftover generated coupons (unmatched after the first pass) are also counted as "lonely." The average similarity score is calculated and the number of lonely coupons is counted; the pipeline is punished accordingly for the number of lonely coupons. 

#### Coupon formats 
There are two coupon formats: `Coupon` and `CouponSimple`.
```python
class Coupon:
    product_name: str
    new_price: Optional[str] = None
    old_price: Optional[str] = None
    percents: List[str] = field(default_factory=list)
    other_discounts: List[str] = field(default_factory=list)
    dates: Optional[str] = None

class CouponSimple:
    product_name: str
    discount_text: str
    validity_text: str
```

#### Arguments/options
`-h, --help`: show this help message and exit

`-i INPUT, --input INPUT`: Path to the file with the input data

`-e EXPECTED, --expected EXPECTED`: Path to the file with the expected coupons

`-p PIPELINE, --pipeline PIPELINE`: Command to run the pipeline (e.g., ./run_pipeline)

`-s, --simple`: Use the simple format (ie. CouponSimple)
  
#### Sources 
- [Benchmarking AI](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)
- [What is a benchmark and why do you need it?](https://www.mim.ai/what-is-a-benchmark-and-why-do-you-need-it/)
- Murmuras coupon_llm benchmark 
