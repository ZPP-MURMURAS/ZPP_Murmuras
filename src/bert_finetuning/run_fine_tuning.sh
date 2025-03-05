#!/bin/bash

declare -a DATASETS=("bert_second_pass_json" "bert_second_pass_pl" )

export HUGGING_FACE_TOKEN=
export WANDB_KEY=
export TIMEOUT=600

for DATASET in "${DATASETS[@]}"
do
    export DATASET_NAME="$DATASET"
    modal run src/bert_finetuning/modal_script.py
done

unset HUGGING_FACE_TOKEN
unset WANDB_KEY
unset DATASET_NAME
unset TIMEOUT