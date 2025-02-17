#!/bin/bash

# Default values
GPU_IDS="0,1,2,3,4,5,6,7"
PORT="35000"

# Print usage
usage() {
    echo "Usage: $0 --model_path <path> --tasks <task_list> [--gpu_ids <gpu_list>] [--port <port_number>]"
    echo "  Example:"
    echo "    $0 \\"
    echo "      --model_path /app/docfm/checkpoints/training/DocVision/SFT-SyntheticData/20250208_solar-exp-2_with-figureqa_900kX3_multipage-base-model/steps_5240 \\"
    echo "      --tasks docvqa,infovqa,textvqa,chartqa \\"
    echo "      --gpu_ids 0,1,2,3,4,5,6,7 \\"
    echo "      --port 35000"
    exit 1
}

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --gpu_ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

# Ensure required arguments are provided
if [[ -z "$MODEL_PATH" || -z "$TASKS" ]]; then
    echo "Error: --model_path and --tasks are required."
    usage
fi

# Count the number of GPUs specified
NUM_PROCESSES=$(echo "$GPU_IDS" | awk -F',' '{print NF}')

# Extract the second last and last directory name from MODEL_PATH
DIR1=$(basename "$(dirname "$MODEL_PATH")")  # Second last directory
DIR2=$(basename "$MODEL_PATH")               # Last directory

# Generate the output path
OUTPUT_PATH="./logs/${DIR1}___${DIR2}"

# Run the command
CMD="python3 -m accelerate.commands.launch \
    --num_processes ${NUM_PROCESSES} \
    --gpu_ids ${GPU_IDS} \
    --mixed_precision bf16 \
    --use_deepspeed \
    --zero_stage 0 \
    --main_process_port ${PORT} \
    -m lmms_eval \
    --model docvision \
    --model_args pretrained=${MODEL_PATH} \
    --tasks ${TASKS} \
    --batch_size 1 \
    --log_samples \
    --output_path ${OUTPUT_PATH}"

echo "Command: $CMD"

$CMD