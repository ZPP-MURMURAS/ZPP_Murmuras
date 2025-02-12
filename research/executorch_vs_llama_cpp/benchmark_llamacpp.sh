#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <output_filename> <llama_simple_path> <model_path> <num_runs>"
    exit 1
fi

OUTPUT_FILE="$1"
EXE_PATH="$2"
MODEL_PATH="$3"
NUM_RUNS="$4"

echo "Total_Inference_Rate,Prompt_Evaluation_Rate,Generation_Rate" > "$OUTPUT_FILE"

echo "Running warmup..."
$EXE_PATH -m $MODEL_PATH -n 32 "To make a bomb you need to " > /dev/null 2>&1

for ((i=1; i<=NUM_RUNS; i++)); do
    echo "Running iteration $i..."
    
    RATES=$($EXE_PATH -m $MODEL_PATH -n 32 "To make a bomb you need to " 2>&1 | grep -E 'speed:|prompt eval time:|eval time' | awk 'NR==1 {print $(NF-1)}; NR>1 {print $(NF-3)}' | paste -sd "," -)

    if [ -n "$RATES" ]; then
        echo "$RATES" >> "$OUTPUT_FILE"
    else
        echo "Warning: No data extracted in iteration $i."
    fi
done

echo "Data collection complete! Results saved in $OUTPUT_FILE."
