# Coupon selecting dataset
This directory contains a script that can be used for creating and publishing datasets for the coupon selection task.<br/>
The script works on a pair of a coupon frame and a content frame. It finds coupons from the first file in the second file.<br/>
Running the script:<br/>
```bash
python generate_coupon_selection_ds.py <config_path> <ds_name> <create_repo> <custom_split>
```
Where `config_path` is a path to the configuration file in format demonstrated by `example_config.json` file, and
create_repo is either 'y' or 'n'; it is used to mark whether the dataset is being pushed to an existing or new repo. If `custom_split`==`n` samples from the provided dataframes are merged and the traditional train-val-test split is performed. Otherwise, if `custom_split`=`y`, each dataframe is assigned to the split specified by the `split` field in the config. For example, in the config at `res/coupon_selection_big_ds_config.json` each dataframe is assigned to the split represented by the source application name.<br/>
To run the script `HF_HUB_KEY` env variable is expected to be set to your access key to hf hub.
## Dataset Format
The dataset contains a list of pairs of word sequences and labels:
* TAG_UNKNOWN: not a coupon
* TAG_B_COUPON: start of coupon content
* TAG_I_COUPON: inside coupon content
## Example config explained
* `config["frames"]`: list of pairs of frames with screen content and coupons
  * `config["frames"][i]["format"]`: format of frame; datasets from `coupons_1` have `format=1`, and datasets from `coupons big` have `format=2`
  * `config["frames"][i]["split"]`: name of the dataset split to save samples from this pair of frames
* `config["json_format"]`: whether to produce output in form of split json string instead of plain text tokens
## Some design choices
### Matching coupons
Matching coupons from the coupons dataframe with text fields inside the content dataframe is nontrivial when considering data from the `coupons_1`, as this dataset does not provide `min_i` and `max_i` columns in the content frame. In attached script, I have implemented following proposal instead:
* construct prefix tree containing all coupons from the provided part od the dataframe
* go through the content dataframe and track each partial match with prefix tree
* if full match with prefix tree is found, mark as coupon content and clear all ongoing partial matches
This design is not perfect, for example if coupon A has suffix that is prefix of coupon B, a string \<prefix A\> + \<suffix A = prefix B\> + \<suffix B\> will always be interpreted as coupon A, which might not be the true in 100% of cases. However, tradeoff like this had to be made.
#### Why do we need fancy prefix tree alg instead of simply searching for coupons' "full_text" occurrences?
Because some coupons are prefixes of another ones. With simple matches it is possible to treat longer coupon as shorter ones.
#### Why not simply iterate through both the coupon frame and the content frame with two pointers and match coupon by coupon basing on their order?
Because coupons in the coupons frame sometimes are not in a correct order.
### Grouping by `id` instead of `timestamp_seen`
As `i` column in content frames is persistent across timestamps I found it more convenient to group by `id` 
### (Lack of) Usage of `i` column
After quick exploration of `coupons big` dataset I found out that `min_i` and `max_i` are not always aligned perfectly. Additionally, I have experienced other anomalies. Due to the lack of time I decided not to develop additional heuristics to handle them and I used prefix tree approach for `coupons big` too.

# Field extraction dataset
This directory contains a script that can be used for creating and publishing datasets for the coupon labeling task.<br/>
Running the script:<br/>
```bash
python generate_field_extraction_ds.py <config_path> <ds_name> <create_repo>
```
Where `config_path` is a path to the configuration file in format demonstrated by `example_config.json` file, and
create_repo is either 'y' or 'n'; it is used to mark whetver the dataset is being pushed to an existing or new repo. <br/>
To run the script `HF_HUB_KEY` env variable is expected to be set to your access key to hf hub.
## Dataset Format
The dataset contains a list of pairs of text sequences and labels:
* TAG_UNKNOWN: unknown
* TAG_B_PRODUCT: begin product name
* TAG_I_PRODUCT: inside product name
* TAG_B_DISCOUNT: begin discount text
* TAG_I_DISCOUNT: inside discount text
* TAG_B_VALIDITY: begin validity text
* TAG_I_VALIDITY: inside validity text
* TAG_B_ACTIVATION: inside activation text
* TAG_I_ACTIVATION: inside activation text
The inside labels are not used in this dataset and are expected to be used after tokenization for the model.
## Supported dataset formats
Currently the `format` field in config is ignored
## Augmentations
Due to the repetitious nature of the data, I've decided to use some basic augmentations. That is, sections (e.g., product info section and discount info section) are sometimes shuffled or dropped.
