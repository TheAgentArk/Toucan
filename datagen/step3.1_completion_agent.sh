#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

input_file=${1}
model_path=${2:-"openai/gpt-oss-120b"}
engine=${3:-"vllm_api"}
step=${4:-"3.1"}
agent=${5:-"qwen_agent"}
start_vllm_service=${6:-"false"}

input_file_basename=$(basename ${input_file})
input_file_with_timestamp=$(date +%Y%m%d_%H%M%S)_${input_file_basename}

if [[ "$model_path" == *"gpt-oss"* ]]; then
    agent="openai_agent"
    engine="vllm_api"
    echo -e "${BLUE}[OpenAI Agent] Overriding to OpenAI Agent ${NC} for ${model_path}"
else
    # Define fncall_prompt_type
    if [[ "$model_path" == *"Mistral-Small-3"* ]]; then
        fncall_prompt_type="mistral"
    elif [[ "$model_path" == *"Devstral-Small"* ]]; then
        fncall_prompt_type="mistral"
    elif [[ "$model_path" == *"Kimi-K2"* ]]; then
        fncall_prompt_type="kimi"
    elif [[ "$model_path" == *"Qwen3"* ]]; then
        fncall_prompt_type="nous"
    else
        fncall_prompt_type="nous"
    fi
    echo -e "${BLUE}[Qwen Agent] fncall_prompt_type: $fncall_prompt_type ${NC}"
fi


export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
if [ "$engine" == "vllm_api" ]; then
    if [ "$start_vllm_service" != "true" ]; then
        echo "[VLLM API] Skipping VLLM server startup as start_vllm_service is set to false."
    else
        # Color definitions
        GREEN='\033[0;32m'
        BLUE='\033[0;34m'
        NC='\033[0m' # No Color

        # ------------------------------------------------------------------------------------------------
        # Cleanup function to kill the vllm server process
        cleanup() {
            echo "Cleaning up background processes..."
            # Kill the vllm server process
            jobs -p | xargs -r kill
            
            exit 0
        }

        # Set trap to call cleanup function on script exit, interrupt, or termination
        trap cleanup EXIT INT TERM

        # Create logs directory if it doesn't exist
        mkdir -p ../logs/vllm
        # Create empty log file
        log_file="../logs/vllm/${input_file_with_timestamp}.log"

        # Initiate vllm server as a background process
        echo -e "${BLUE}[VLLM API] Initializing vllm server..."
        if [[ "$model_path" == *"Mistral-Small-3"* ]]; then
            export VLLM_ATTENTION_BACKEND=XFORMERS
            vllm serve $model_path \
                --tokenizer_mode mistral \
                --config_format mistral \
                --load_format mistral \
                --limit_mm_per_prompt 'image=10' \
                --tensor-parallel-size 4 \
                --port 8000 \
                --host 0.0.0.0 \
                --max-model-len 32768 \
                --gpu-memory-utilization 0.9 > "$log_file" 2>&1 &
        elif [[ "$model_path" == *"Devstral-Small"* ]]; then
            echo "Applying XFORMERS attention backend for Devstral-Small"
            export VLLM_ATTENTION_BACKEND=XFORMERS
            vllm serve $model_path \
                --tokenizer_mode mistral \
                --config_format mistral \
                --load_format mistral \
                --tensor-parallel-size 4 \
                --port 8000 \
                --host 0.0.0.0 \
                --max-model-len 40960 \
                --gpu-memory-utilization 0.9 > "$log_file" 2>&1 &
        else
            vllm serve $model_path \
                --tensor-parallel-size 4 \
                --port 8000 \
                --host 0.0.0.0 \
                --max-model-len 32768 \
                --gpu-memory-utilization 0.9 > "$log_file" 2>&1 &
        fi
        VLLM_PID=$!
        echo -e "${BLUE}[VLLM API] VLLM server initialized with PID: $VLLM_PID ${NC}"

        # Wait for the vllm server to start up
        echo -e "${BLUE}[VLLM API] Waiting for VLLM server to be ready...${NC}"
        MAX_RETRIES=50
        RETRY_COUNT=0
        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if curl -s http://localhost:8000/v1/models > /dev/null; then
                echo -e "${GREEN}[VLLM API] VLLM server is ready!${NC}"
                break
            else
                echo -e "${BLUE}[VLLM API] Waiting for VLLM server to start... (($((RETRY_COUNT+1))/$MAX_RETRIES))${NC}"
                sleep 10
                RETRY_COUNT=$((RETRY_COUNT+1))
            fi
        done

        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "\033[0;31m[VLLM API] Failed to start VLLM server after $MAX_RETRIES attempts. Exiting.${NC}"
            exit 1
        fi
    fi
fi

if [ -z "$input_file" ]; then
    echo "Error: Input file is required."
    exit 1
fi

if [ "$agent" == "openai_agent" ]; then
    echo -e "${BLUE}[OpenAI Agent] Using OpenAI Agent ${NC} for ${model_path}"
    python completion_openai_agent.py \
        --input_file ${input_file} \
        --model_path ${model_path} \
        --engine ${engine} \
        --step ${step} \
        --agent ${agent} \
        --max_workers 8 \
        --enable_tool_hint
else
    echo -e "${BLUE}[Qwen Agent] Using Qwen Agent ${NC} for ${model_path}"
    python completion_qwen_agent.py \
        --input_file ${input_file} \
        --model_path ${model_path} \
        --engine ${engine} \
        --step ${step} \
        --agent ${agent} \
        --max_workers 8 \
        --fncall_prompt_type ${fncall_prompt_type}
fi