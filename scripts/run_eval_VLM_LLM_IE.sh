#!/bin/bash

# Default values
GPU_IDS="0"
PORT="35000"
TASKS="KIE_bench_VLM_LLM_IE" 
MODEL_NAME="VLM_LLM_IE"
VLM_PORT="35001"
LLM_PORT="35002"
VLM_HOST="localhost"
LLM_HOST="localhost"
VLM_MAX_TOKENS="32768"
LLM_MAX_TOKENS="32768"

# Print usage
usage() {
    echo "Usage: $0 --llm_model_name <name> --vlm_model_name <name> [--port <port_number>] [--vlm_port <port>] [--llm_port <port>] [--vlm_host <host>] [--llm_host <host>]"
    echo "  Example:"
    echo "    $0 \\"
    echo "      --llm_model_name deepseek-coder \\"
    echo "      --vlm_model_name docev_preview \\"
    echo "      --port 35000 \\"
    echo "      --vlm_port 35001 \\"
    echo "      --llm_port 35002"
    exit 1
}

# Initialize variables
LLM_MODEL_NAME=""
VLM_MODEL_NAME=""

# Parse keyword arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --llm_model_name)
            LLM_MODEL_NAME="$2"
            shift 2
            ;;
        --vlm_model_name)
            VLM_MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --vlm_port)
            VLM_PORT="$2"
            shift 2
            ;;
        --llm_port)
            LLM_PORT="$2"
            shift 2
            ;;
        --vlm_host)
            VLM_HOST="$2"
            shift 2
            ;;
        --llm_host)
            LLM_HOST="$2"
            shift 2
            ;;
        --vlm_max_tokens)
            VLM_MAX_TOKENS="$2"
            shift 2
            ;;
        --llm_max_tokens)
            LLM_MAX_TOKENS="$2"
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
if [[ -z "$LLM_MODEL_NAME" || -z "$VLM_MODEL_NAME" ]]; then
    echo "Error: --llm_model_name and --vlm_model_name are required."
    usage
fi

# Count the number of GPUs specified
NUM_PROCESSES=1

# Extract the part after the last slash for log directory naming
LLM_MODEL_NAME_SHORT=$(echo "$LLM_MODEL_NAME" | awk -F'/' '{print $NF}')
VLM_MODEL_NAME_SHORT=$(echo "$VLM_MODEL_NAME" | awk -F'/' '{print $NF}')

# Generate the output path automatically using the short names
OUTPUT_PATH="./logs/${LLM_MODEL_NAME_SHORT}___${VLM_MODEL_NAME_SHORT}" 

mkdir -p "${OUTPUT_PATH}" # Ensure the directory exists

# Construct model arguments string
MODEL_ARGS="vlm_model_name=${VLM_MODEL_NAME},llm_model_name=${LLM_MODEL_NAME},vlm_host=${VLM_HOST},llm_host=${LLM_HOST},vlm_port=${VLM_PORT},llm_port=${LLM_PORT},vlm_max_completion_tokens=${VLM_MAX_TOKENS},llm_max_completion_tokens=${LLM_MAX_TOKENS}"

# Show the command that will be executed
echo "Executing command with:"
echo "  - LLM Model Name: ${LLM_MODEL_NAME}"  
echo "  - VLM Model Name: ${VLM_MODEL_NAME}"
echo "  - VLM Host: ${VLM_HOST}"
echo "  - LLM Host: ${LLM_HOST}"
echo "  - VLM Port: ${VLM_PORT}"
echo "  - LLM Port: ${LLM_PORT}"
echo "  - Output Path: ${OUTPUT_PATH}"
echo "  - GPU IDs: ${GPU_IDS}"
echo "  - Port: ${PORT}"

# Run the command
CMD="python3 -m accelerate.commands.launch \
    --num_processes ${NUM_PROCESSES} \
    --gpu_ids ${GPU_IDS} \
    --mixed_precision bf16 \
    --main_process_port ${PORT} \
    -m lmms_eval \
    --model ${MODEL_NAME} \
    --model_args ${MODEL_ARGS} \
    --tasks ${TASKS} \
    --log_samples \
    --output_path ${OUTPUT_PATH}"

echo "Command: $CMD"

$CMD

# Remove temp images
echo "Remove temp images"
rm -r temp_images/*
echo "Done"