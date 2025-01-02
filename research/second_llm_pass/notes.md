# Second LLM Pass (or labeling large text portions) research
As initial attempts to create e2e pipeline resulted in low quality results we want to know other options on design. Here I will gather info about possibilities to use ML tools to categorize section of text (possibly xml/csv) as coupon.
## Similar existing problems
### Sentence classification
There are many works on classifying single sequences and datasets focused on this task (ex [trec](https://huggingface.co/datasets/CogComp/trec)).
### Larger text classification
Closer to our problem, we have reviews classification task. Reviews can be quite [long](https://medium.com/codex/fine-tune-bert-for-text-classification-cef7a1d6cdf1), but we still miss two aspects of our problem: we do not know where possible coupons have the beginning and the ending, which stops us from applying this method directly.
### Text Retrieval and Question answering
There are tasks of question answering and text retrieval, more similar to our problem. Given portion of text and possibly a question, the model selects a beginning and an end of a sequence containing the answer. This addresses our lack of knowledge about exact coupon placement. Despite that, we still have to deal with the fact that we might have more than one coupon in one provided xml portion. I will discuss this problem further in the following section. </br>
Note: text retrieval seems to be quite niche, there is even no model task category on HF for that.
### Dataset size requirements
* we would [probably](https://discuss.huggingface.co/t/thoughts-on-quantity-of-training-data-for-fine-tuning/14886?utm_source=chatgpt.com) need at least 1k examples for fine-tuning bert for classification
* In one of the sections below I discussed QA with multiple answers, suitable for our case. As it is NER-based, I expect several thousands of training examples to be sufficient for our task.
## Known challenges
### Random coupon placement
Coupons might be located in groups, there will also be large sparse portions of text without any.This presents a challenge both to the QA and standard classification approach. Possible solutions:
#### Non-LLM Heuristics
Combining with heuristic used in first PoC (analyze count of labels from first BERT pass and decide whether xml node is possibly a coupon, and it is worth inputting to the classification model).
#### Expanded labeling targets
In case of text classification, we could add label `MULTIPLE_COUPONS` to indicate that we have multiple coupons in data. In case of match with this label, we should split input and rerun classification. Downside here is the need for multiple classification runs.
#### multi-span question answering
There are [attempts](https://aclanthology.org/2020.emnlp-main.248.pdf) to modify qa task to produce multiple outputs - provided example is actually close to what we discussed on our meetings (casting QA task to NER-like token labeling).
#### Input chunking
We may split input into chunks and run model on each chunk. Proper chunking would require at least two overlapping chunking passes to deal with coupons on chunks' borders. However, at this moment I do not see possible solution to multiple coupons residing in a single chunk. </br>
This can also be combined with multi-span QA to reduce model input size.
### Providing input in reasonable format
Possibly feeding LLM with raw CSV is not the best idea. It contains a lot of boilerplate data and probably does not represent xml structure well. </br>
#### General suggestions
* removing uninformative columns from csv (like `user_id`)
* removing app-related columns (like `application_name`) - we want our solution to be cross-app
#### Possible enhancements
##### Each text field should have prefix containing all its parents
We encode each CSV row as a pair of view name and text. For each row we write down pairs representing all ancestor rows. </br>
Expansive in terms of tokens we feed the model. Probably not recommended.
##### Simply encoding all xml tags to single special token
BERT (and as far as I know Llama too) provides pool of unused tokens that we can adapt in fine-tuning process to represent XML tags. Encoding every tag to single token would make us lose tag attributes, but it would preserve tree structure.
##### Converting to JSON
We encode input to JSON format. This can be done in two ways. The first way is to convert each CSV row to dict and collect these dictionaries in one list. The more sophisticated approach is to represent XML structure in form of JSON tree.
## Solutions requiring model training
Below I present some ideas involving training custom model from scratch. These proposals are obviously more challenging to implement than fine-tuning a llm and I suggest treating them as backup option:
### 1D convolution over text sequence / RNN
#### 1D convolution
This solution is conceptually similar to multi-answer QA. For each token we assign a label being `UNKNOWN`, `BEGIN-COUPON` or `MID-COUPON`. This time, however, we will use CNNs. </br>
Several years ago one-dimensional convolution over text sequences was perspective approach to some NLP tasks, like [classification](https://aclanthology.org/D14-1181.pdf) or sentiment analysis. There were even results suggesting that common association between NLP and RNNs should be [reconsidered](https://arxiv.org/pdf/1803.01271). </br>
The main idea is to combine this convolution-based approach with technique known from image processing - semantic segmentation. In [classical semantic segmentation task](https://arxiv.org/pdf/1505.04597) we assign class to each pixel. In our case, we would assign one of labels mentioned above to each input token. </br>
#### RNN
It is possible that we can achieve similar behaviour with bidirectional multi-input multi-output RNNs. The similarity with semantic segmentation task makes me fan of convolution approach, although it might be worth considering to try also this way.
#### Problems
We still not escaped problem with input format. We also introduced additional tokens for detecting coupon boundaries (`BEGIN-COUPON` and `MID-COUPON`). Removing those might increase accuracy, but we are left with not separated coupons.
### Tree-LSTM
There were [adaptations](https://arxiv.org/pdf/1503.00075) of RNNs to handle tree-structured data. The linked paper presents details of architecture and describes results achieved by it on (among other tasks) sentiment analysis. The work treats the text as a tree of tokens and is given syntactic structure of the text as a part of the input. Additionally, paper suggests how classification of each tree node should be performed. Using this framework, we could decide for each node in XML tree whether it is a coupon. This approach, unlike other proposals, would make an advantage of XML structure of our data.
### Training dataset size requirements
Training custom models from scratch would be expansive both in computation power and amount of data required. Here are my estimations on minimal dataset size:
* CNN - in provided paper network was trained using around 10k movie reviews for sentiment analysis and utilized publicly available vector embedding. This suggests that in our case we would need similar number of distinct coupons. However, in case of semantic segmentation on images, the standard dataset sizes are order (or orders) of magnitude smaller - this gives us the hope that amount of data can be further reduced. As this approach seems not to be covered in literature, I cannot provide exact estimations.
* TreeLSTM - in provided paper sentiment analysis was trained on 10k example (Stanford Sentiment Treebank dataset). We would require the data amount to be in this order of magnitude.
* ## Summary and recommendations
In overall, second LLM pass seems to be the most reasonable approach to our problem. More specifically, I would suggest combining multi-span QA with chunking. Custom-model proposals are interesting but sticking to them would be important decision that is probably out of scope of my research. In terms of input format, I suggest entering simplified XML structure or JSON-encoded XML.
