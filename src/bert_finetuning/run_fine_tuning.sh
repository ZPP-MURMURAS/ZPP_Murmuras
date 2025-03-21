#!/bin/bash

declare -a DATASETS=("bert_second_pass_pl" "bert_second_pass_json" )

export HUGGING_FACE_TOKEN=
export WANDB_KEY=
export TIMEOUT=60000 # Alright, apparently (according to my experiments)
# timeout is not for waiting; if program runs 600s, it will be stopped (happened to me).
# So, this has to be big enough to finish the job.

for DATASET in "${DATASETS[@]}"
do
    export DATASET_NAME="$DATASET"
    modal run src/bert_finetuning/modal_script.py
done

unset HUGGING_FACE_TOKEN
unset WANDB_KEY
unset DATASET_NAME
unset TIMEOUT