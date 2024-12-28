### Pipeline benchmark
This benchmark is designed to compare the performance of different pipelines to provide a measurable way of selecting superior solutions. The pipelines are evaluated based on their accuracy in classifying coupons from a given XML tree. The benchmark is designed to be a fair comparison of the pipelines, and the results are used to determine the best pipeline for our task. \
Each pipeline will be rewarded accordingly based on the correctly classified coupons and penalized for incorrectly classified coupons. The benchmark will be run on a dataset of `x` XML trees and the corresponding coupons. If selected, the benchmark will also be rewarded or penalized based on the time taken to classify the coupons.

---

### Running the benchmark
To run the benchmark, the following steps need to be followed: TODO

Input:

Output:
<!-- what to input into this and how to interpret the outputs, how to run the program -->

---

### Comparing pipelines

#### Accuracy
The accuracy of the pipeline is calculated by comparing the coupons classified by the pipeline to the actual coupons (the expected results). For each coupon, the pipeline will classify the appropriate traits of the coupon such as the name, discount, expiry date, etc. TODO

The pipeline will be penalized for (1) not classifying a coupon as a coupon, (2) getting more than 50% of the information about a given coupon incorrect, or (3) classifying data to be part of a coupon when it is not. For example, if this is the complete coupon: \
`product_name='KINDER Überraschungs-Ei', discount='36% gespart', old_price='N/A', new_price='N/A', validity='Gültig ab 06.09.'`
and the pipeline classifies a number as a price, the pipeline will be penalized. \
In cases (1) and (2), the pipeline will be penalized by 1 point. In case (3), the pipeline will be penalized proportionally to the value of the data it classified incorrectly. TODO

#### [Optional] Time
If the option for rewarding or penalizing based on time is selected, the pipeline will be rewarded for classifying the coupons in the shortest time possible. The pipeline will be penalized for taking too long to classify the coupons. The time taken to classify the coupons will be measured in milliseconds. \
A pipeline will be penilized if it takes more than the median time to classify the coupons. The execution times of all runs will be stored in a file and the median will be calculated. On the other hand, a pipeline will be rewarded if it takes less than the first quartile of the time taken to classify the coupons. Otherwise the pipeline will neither be rewarded nor penalized. 

---

### Sources 
- [Benchmarking AI](https://mlsysbook.ai/contents/core/benchmarking/benchmarking.html)
- [What is a benchmark and why do you need it?](https://www.mim.ai/what-is-a-benchmark-and-why-do-you-need-it/)
- Murmuras coupon_llm benchmark 