#!/bin/bash

declare -a NAMES=("one_input_one_output_wrequest" "one_input_one_output_wthrequest"
                 "one_input_multiple_outputs_wrequest" "one_input_multiple_outputs_wthrequest")

export HUGGING_FACE_TOKEN=
export WANDB_KEY=
export TIMEOUT=600

for NAME in "${NAMES[@]}"
do
  export DATASET_NAME="$NAME"
  modal run fine_tune_llama.py
done

unset HUGGING_FACE_TOKEN
unset WANDB_KEY
unset DATASET_NAME
unset TIMEOUT