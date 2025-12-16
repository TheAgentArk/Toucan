#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

input_file=${1}
model_path=${2:-"openai/gpt-oss-120b"}
engine=${3:-"vllm_api"}
num_desired_turns=${4:-5}
user_simulation_model=${5:-"${model_path}"}
agent=${6:-"qwen_agent"}
start_vllm_service=${7:-"true"}
max_workers=${8:-8}

input_file_basename=$(basename ${input_file})
input_file_with_timestamp=$(date +%Y%m%d_%H%M%S)_${input_file_basename}

# Color definitions (define early for use in all sections)
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Multi-Turn Conversation Generation${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Input file: ${input_file}${NC}"
echo -e "${BLUE}Model path: ${model_path}${NC}"
echo -e "${BLUE}Engine: ${engine}${NC}"
echo -e "${BLUE}Num desired turns: ${num_desired_turns}${NC}"
echo -e "${BLUE}User simulation model: ${user_simulation_model}${NC}"
echo -e "${BLUE}Max workers: ${max_workers}${NC}"
echo -e "${BLUE}========================================${NC}"

# Auto-detect agent type based on model_path
if [[ "$model_path" == *"gpt-oss"* ]]; then
    agent="openai_agent"
    engine="vllm_api"
    echo -e "${BLUE}[OpenAI Agent] Auto-detected OpenAI Agent ${NC} for ${model_path}"
else
    agent="qwen_agent"
    # Define fncall_prompt_type for Qwen Agent
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
    echo -e "${BLUE}[Qwen Agent] Auto-detected Qwen Agent ${NC} for ${model_path}"
    echo -e "${BLUE}[Qwen Agent] fncall_prompt_type: $fncall_prompt_type ${NC}"
fi

# VLLM Service Setup
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_VLLM_V1
if [ "$engine" == "vllm_api" ]; then
    if [ "$start_vllm_service" != "true" ]; then
        echo -e "${YELLOW}[VLLM API] Skipping VLLM server startup as start_vllm_service is set to false.${NC}"
        echo -e "${YELLOW}[VLLM API] Assuming VLLM server is already running at http://localhost:8000${NC}"
    else
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
        echo -e "${BLUE}[VLLM API] Initializing vllm server...${NC}"
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
            echo -e "${BLUE}[VLLM API] Applying XFORMERS attention backend for Devstral-Small${NC}"
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

# Validate input file
if [ -z "$input_file" ]; then
    echo -e "\033[0;31mError: Input file is required.${NC}"
    echo -e "${YELLOW}Usage: $0 <input_file> [model_path] [engine] [num_desired_turns] [user_simulation_model] [agent] [start_vllm_service] [max_workers]${NC}"
    exit 1
fi

if [ ! -f "$input_file" ]; then
    echo -e "\033[0;31mError: Input file does not exist: $input_file${NC}"
    exit 1
fi

# Run multi-turn generation
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Starting Multi-Turn Generation${NC}"
echo -e "${GREEN}========================================${NC}"

if [ "$agent" == "openai_agent" ]; then
    echo -e "${BLUE}[OpenAI Agent] Using ext_multi_turn_openai_agent.py ${NC}"
    python ext_multi_turn_openai_agent.py \
        --input_file ${input_file} \
        --model_path ${model_path} \
        --user_simulation_model ${user_simulation_model} \
        --engine ${engine} \
        --num_desired_turns ${num_desired_turns} \
        --max_workers ${max_workers} \
        --agent ${agent} \
        --reasoning_effort high \
        --parallel_function_calls True \
        --max_rounds_per_turn 10
else
    echo -e "${BLUE}[Qwen Agent] Using ext_multi_turn_qwen_agent.py ${NC}"
    python ext_multi_turn_qwen_agent.py \
        --input_file ${input_file} \
        --model_path ${model_path} \
        --user_simulation_model ${user_simulation_model} \
        --engine ${engine} \
        --num_desired_turns ${num_desired_turns} \
        --max_workers ${max_workers} \
        --agent ${agent} \
        --fncall_prompt_type ${fncall_prompt_type} \
        --parallel_function_calls False \
        --max_retries 2 \
        --max_rounds_per_turn 10 \
        --enable_tool_hint \
        --enable_irrelevant_warning
fi

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Multi-Turn Generation Completed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\033[0;31m========================================${NC}"
    echo -e "\033[0;31mMulti-Turn Generation Failed with exit code: $exit_code${NC}"
    echo -e "\033[0;31m========================================${NC}"
fi

exit $exit_code

