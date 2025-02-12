#!/bin/bash

# Ensure script is executed with at least two arguments
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <model_path> <task_list> [gpu_ids]"
    exit 1
fi

# Assign arguments to variables
MODEL_PATH="$1"
TASKS="$2"
GPU_IDS="${3:-0,1,2,3,4,5,6,7}"  # Default to 0-7 if not provided

# Count the number of GPUs specified
NUM_PROCESSES=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

# Extract the second last and last directory name from MODEL_PATH
DIR1=$(basename "$(dirname "$MODEL_PATH")")  # Second last directory
DIR2=$(basename "$MODEL_PATH")               # Last directory

# Generate the output path
OUTPUT_PATH="./logs/${DIR1}___${DIR2}"

# Run the command with dynamic values
CMD="python3 -m accelerate.commands.launch
    --num_processes ${NUM_PROCESSES}
    --gpu_ids ${GPU_IDS}
    --mixed_precision bf16
    --use_deepspeed
    --zero_stage 0
    -m lmms_eval
    --model docvision
    --model_args pretrained=${MODEL_PATH}
    --tasks ${TASKS}
    --batch_size 1
    --log_samples
    --output_path ${OUTPUT_PATH}"

echo "Command: $CMD"

$CMD
