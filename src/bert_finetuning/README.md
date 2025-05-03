# Curriculum learning for NER
During my experiments with BERT I noticed that our data has a lot of noise; coupon-to-non-coupon ratio was around 6%.
In order to balance classes, but still enable the model to retrieve coupons from the noisy data, I came up with an algorith.

## Notions:
- **Span** - defined by its beginning and its end; it represents a continuous sequence of tokens in a row. This sequence contains a coupon, as well as zero tokens on its lef and right. Later its being extended to add noise to the data, and new datasets are constructed from spans (multiple spans per row are allowed).

1. Split rows that have coupons and rows that don't have coupons.
2. Represent every row by spans; initially, each span should contain coupons and their neighbourhood, so that spans in each row contain 50/50 split (if possible).
3. Iterate from 1 to n (n is the number of steps; can be specified by user); let 'i' be the index of the current iteration:
    1. If current index is e.g. odd, add x/n rows with no coupons to the dataset (x is the number of rows that have coupons inside them). This step uses the split from the first step.
    2. Otherwise, for each row extend all its spans (of possible) to include more non-coupon labels.
    3. Dataset is created by going through every row and concatenating its spans (rows without coupons have one span, that covers the whole row).

As a result, at first a model should learn the representation of the coupon (its structure, context, etc.), and then with each step it should 
learn the environment that is more and more difficult (similar to reinforcement learning; if I'm not being mistaken, 
curriculum learning is usually used there).

## Other
1. Right now, the algorithm uses a dataset for a fixed number of epochs (e.g. 3). After that, no matter the results it creates a new dataset and performs training with, again, fixed number of epochs.
This process happens in a loop.
2. To reiterate: Depending on the loop index parity (loop in which the dataset generation and training take place), this algorith
extends either Y axis (adds rows without coupons) or X axis (extends spans to include more non-coupon labels) of the previous dataset.


## Dev notes: Sequel
1. Fixed the learning rate preservation between the training steps in the curriculer. However, now it seems
that at some point BERT just stops learning (learning rate is too low). More experiments needs to be conducted.
2. Added vibe-check. After some fights, I decided to help myself with the existing functionality of the HF,
and right now the vibecheck returns "UNKNOWN" for words that has no tokens classified as part of the coupon, 
and "COUPON" otherwise. Because rn models aren't performing very well (low LR), most of the words have
at least one token classified as a coupon, hence they are treated as part of the coupon.
