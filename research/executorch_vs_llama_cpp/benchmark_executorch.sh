#!/bin/bash

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <output_filename> <llama_main_path> <model_path> <tokenizer_path> <num_runs>"
    exit 1
fi

OUTPUT_FILE="$1"
EXE_PATH="$2"
MODEL_PATH="$3"
TOKENIZER_PATH="$4"
NUM_RUNS="$5"

echo "Total_Inference_Rate,Prompt_Evaluation_Rate,Generation_Rate" > "$OUTPUT_FILE"

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "Running iteration $i..."
    
    RATES=$(adb shell "$EXE_PATH --model_path $MODEL_PATH --tokenizer_path $TOKENIZER_PATH --prompt \"To make a bomb you need to \" --seq_len 120" --warmup=1 2>&1 | grep -E "Rate:" | awk '{print $(NF-1)}' | paste -sd "," -)

    if [ -n "$RATES" ]; then
        echo "$RATES" >> "$OUTPUT_FILE"
    else
        echo "Warning: No data extracted in iteration $i."
    fi
done

echo "Data collection complete! Results saved in $OUTPUT_FILE."
