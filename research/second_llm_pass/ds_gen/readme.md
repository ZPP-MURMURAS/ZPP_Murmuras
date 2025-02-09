# Coupon selecting dataset
This directory contains a script that can be used for creating and publishing datasets for coupon selection task.<br/>
Running the script:<br/>
```bash
python generate_coupon_selection_ds.py <config_path> <ds_name>
```
Where `config_path` is a path to the configuration file in format demonstrated by `example_config.json` file. <br/>
To run the script `HF_HUB_KEY` env variable is expected to be set to your access key to hf hub.
## Dataset Format
The dataset contains list of pairs of sequence of words and labels:
* TAG_UNKNOWN: not a coupon
* TAG_B_COUPON: start of coupon content
* TAG_I_COUPON: inside coupon content
## Example config explained
* `config["frames"]`: list of pairs of frames with screen content and coupons
  * `config["frames"][i]["format"]`: format of frame; datasets from `coupons_1` have `format=1`, and datasets from `coupons big` have `format=2`
* `config["json_format"]`: whether to produce output in form of split json string instead of plain text tokens
## Some design choices
### Matching coupons
As `coupons_1` dataset does not provide `min_i` and `max_i` columns, that could be used to dataset generation we have to use other method. In attached script, I have implemented following proposal:
* construct prefix tree containing all coupons from the provided part od the dataframe
* go through the content dataframe and track each partial match with prefix tree
* if full match with prefix tree is found, mark as coupon content and clear all ongoing partial matches
This design is not perfect, for example if coupon A has suffix that is prefix of coupon B, a string \<prefix A\> + \<suffix A = prefix B\> + \<suffix B\> will always be interpreted as coupon A, which might not be the true in 100% of cases. However, tradeoff like this had to be made.
### Grouping by `id` instead of `timestamp_seen`
As `i` column in content frames is persistent across timestamps I found it more convenient to group by `id` 
### (Lack of) Usage of `i` column
After quick exploration of `coupons big` dataset I found out that `min_i` and `max_i` are not always aligned perfectly. Additionally, I have experienced other anomalies. Due to the lack of time I decided not to develop additional heuristics to handle them and I used prefix tree approach for `coupons big` too.