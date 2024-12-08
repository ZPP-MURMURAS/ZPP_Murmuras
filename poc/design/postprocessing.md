# Postprocessing HLD
## Input Format
DOZRO
## Challenges
* Detecting a difference between coupon and random pair of price and product name
* Selecting coupon boundaries
## Ideas
### RNNs
Each XML node text content is parsed by `rnn_1`. Then we follow the procedure described below:
* We run `classifier` network for each leaf and check if they are standalone coupons
* we mark each leaf as `processed` and cut leafs classified as coupons
* for each node that has all of its children labeled as `processed` we run `rnn_2` network that takes the output of `rnn_1` generated on this node as its initial state, parses all children (if a child is a leaf we take the output of `rnn_` otherwise `rnn_2`). Then we mark the node as `processed` and run `classifier` on `rnn_2` output. If a node is classified as a coupon, we cut it.
#### Problems
* requires a large amount of data (orders of magnitude bigger than what we got from murmuras)
* possibly expensive to evaluate in real time
### Label-based coupon selection
At first, each node is assigned set of labels produced by BERT on it's text. \\
We then merge these sets while going in bottom-up manner. If merging sets from several nodes would create a set with more than one coupon name, we stop merging.
Finally, we classify nodes where merging stopped AND whose sets have both product name and price to be roots of coupon views.
#### Problems and enchantments
* Multiple product names:
  * it might occur that a single coupon has several product names contained. We propose following heuristic:
  * if both sets with product names contain prices already, we stop merging. As a result, we treat that coupon as two separate coupons.
  * otherwise, we calculate SOME measure of similarity between product names (ex. cosine similarity between average of vector embeddings of tokens in both names) and combine nodes if similarity is above some threshold. otherwise we stop merging
* Many non-coupons false-positives
  * example: entry from receipt, random message from texting app (randomly containing some price and some unrelated product) etc
  * we do not have a good solution for that
  * basic exclusion for texting apps would be to require a coupon to have image in it
  * more sophisticated context analysis:
    * that would require some NLP model to parse all the texts that are "close" in XML tree to coupon candidate. By "close" we mean ancestors and their uncles
* Selecting prices from a set
  * In some coupons there are several prices such as price per unit, price per item, price per 100ml (for example). We would require BERT classification to return whether a price is a normal price or price per unit, etc. Based on that we would prefer the normal price rather than price per unit. Alternatively regular expressions could be used to match price per unit, but that would be sensitive to new data formats.
