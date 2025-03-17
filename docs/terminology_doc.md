## Terminology used in our project

### Model naming conventions
base model = bert-{selection / extraction}-{data set: json / plain}-{fine tune type}, llama-{fine tune type}, etc.

selection = coupon selection
extraction = field extraction

one2one = one-to-one
one2many = one-to-many

### Data set naming conventions
#### Llama data sets
llama-ds

4 possile formats:
one_input_one_output_wrequest, 
one_input_one_output_wthrequest,
one_input_multiple_outputs_wrequest, 
one_input_multiple_outputs_wthrequest

We are using 
one_input_multiple_outputs_wrequest, 
one_input_multiple_outputs_wthrequest

(w - meamning with request to llama, wth - meaning without request to llama)

llama-ds refers solely to one_input_multiple_outputs_wrequest

#### BERT data sets
bert-{selection / extraction}-ds.{json / pl}
