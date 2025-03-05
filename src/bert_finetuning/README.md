# Curriculum learning for NER
During my experiments with BERT I noticed that our data has a lot of noise; coupon-to-non-coupon ratio was around 6%.
In order to balance classes, but still enable the model to retrieve coupons from the noisy data, I propose this algorithm:
1. Split rows that have coupons and rows that don't have coupons.
2. Represent every row by spans; initially, each span should contain coupons and their neighbourhood, so that spans in each row contain 50/50 split (if possible).
3. Create n datasets (n specified by the user) in the following manner:
    1. If current dataset has e.g. index, add x/n rows with no coupons to the dataset (x ins the init number of rows without coupons).
    2. Otherwise, for each row extend spans to include more non-coupon labels (ideally y/n, where y is the initial number of labels not included in the dataset).
    3. Dataset is created by taking spans from the rows and concatenating them.

As a result, at first a model should learn the representation of the coupon (its structure, context, etc.), and then with each step it should 
learn the environment that is more and more difficult (similar to reinfocement learning; if I'm not being mistaken, c
urriculum learning is usually used there).


## Dev notes:
Right now, each "step" (training on new dataset) has a new trainer; I tried to reuse it, but lr was still behaving in a weird way, 
and for instance, trainer was resuming from epoch 2.9 (saved state). I wouldn't be surprised if its fixable, but I didn't have enough time to fix it.
