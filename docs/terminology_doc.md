## Terminology used in our project
This document provides a list of terms used in our project and their definitions. 

### Model naming conventions
Our project uses a specific naming convention for models. The naming convention is as follows:

    **bert-{selection / extraction}-{data set: json / plain}-{fine tune type}**

    **llama-{fine tune type}**

Selection refers to coupon selection, ie. identifying and selecting the coupons from the text. Extraction refers to field extraction, ie. extracting the fields from the coupons such as product name, discount, etc.

### Data set naming conventions
#### Llama data sets

We have four types of data sets for llama models:

    1. one_input_one_output_wrequest
    2. one_input_one_output_wthrequest
    3. one_input_multiple_outputs_wrequest
    4. one_input_multiple_outputs_wthrequest

We are using the last two dataset types in our project.
In the naming convention, the following abbreviations are used:

    **w - meaning with requests to llama**
    **wth - meaning without requests to llama**

Henceforth, **llama-ds** refers solely to one_input_multiple_outputs_wrequest.

#### BERT data sets
We are using two types of data sets for bert models:

    1. json (.json)
    2. plain text (.pl)

BERT data sets are named as follows:

    **bert-{selection / extraction}-ds.{json / pl}**

