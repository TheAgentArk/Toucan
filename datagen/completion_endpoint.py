import torch
import os
import sys
import argparse
import copy
import json
import re
import requests
import concurrent.futures
from time import sleep, time
from tqdm import tqdm
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_model_short_name, safe_save_checkpoint, get_model_abbreviation
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="We will support more models in the future.")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of samples per batch")
    parser.add_argument("--checkpoint_every", type=int, default=10, help="Save checkpoint every n batches")
    parser.add_argument("--together_api_url", type=str, default="https://api.together.xyz/v1/chat/completions", help="Together API URL")
    parser.add_argument("--together_api_key", type=str, default="", help="Together API Key (start without Bearer)")
    parser.add_argument("--vllm_api_url", type=str, default="http://localhost:8000/v1/chat/completions", help="vLLM API URL")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY", help="vLLM API Key")
    parser.add_argument("--openai_api_key", type=str, default="", help="OpenAI API Key (optional, can use environment variable)")
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter API URL")
    parser.add_argument("--openrouter_api_key", type=str, default="", help="OpenRouter API Key")
    parser.add_argument("--offline", action="store_true", help="Use local engine")

    # Generation Parameters
    parser.add_argument('--engine', default="vllm", type=str, choices=["vllm", "vllm_api", "hf", "together_api", "openai", "openrouter_api"])
    parser.add_argument("--device", type=str, default="0,1,2,3")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Number of GPUs to use for tensor parallelism. Only used for Llama 70B models.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")

    return parser.parse_args()

args = get_args()
print(f"Response Generation Manager. Arguments: {args}") # For logging

if args.input_file is None:
    raise ValueError("Please specify the input file path.")
    
# Input check: check if ends with prepared.jsonl or prepared.json
if not args.input_file.endswith("prepared.jsonl") and not args.input_file.endswith("prepared.json"):
    print("Error: Input file must end with prepared.json(l) for completion pipeline. Please make sure you are using the correct input file.")
    exit(1)

# Constants for the local vllm engine
MODEL_NAME = args.model_path
INPUT_FILE_NAME = args.input_file 
BATCH_SIZE = args.batch_size
CHECKPOINT_EVERY = args.checkpoint_every

model_abbreviation = get_model_abbreviation(args.model_path)

base_name = INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]  # Remove "_4prepared"
elif base_name.endswith("_prepared"):
    base_name = base_name[:-9]  # Remove "_prepared"

if args.num_trials > 1:
    checkpoint_files = [
        f"{base_name}_{model_abbreviation}_results{i}_checkpoint.json"
        for i in range(args.num_trials)
    ]
    saved_files = [
        f"{base_name}_{model_abbreviation}_results{i}.jsonl"
        for i in range(args.num_trials)
    ]
else:
    checkpoint_file = f"{base_name}_{model_abbreviation}_results_checkpoint.json"
    saved_file = f"{base_name}_{model_abbreviation}_results.jsonl"

# Obtain config from configs/model_configs.json (only for local engines)
stop_tokens = []
stop_token_ids = []
if args.engine in ["vllm", "hf"]:
    with open("model_configs.json", "r") as f:
        model_configs = json.load(f)
        model_config = model_configs[args.model_path]
        stop_tokens = model_config["stop_tokens"]
        stop_token_ids = model_config["stop_token_ids"]

# API Setups
if args.engine == "together_api":
    # Constants for the API
    API_ENDPOINT = args.together_api_url
    API_HEADERS = {
        "Authorization": f"Bearer {args.together_api_key}",
    }
    API_PARAMS = {
        "model": args.model_path,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "stop": stop_tokens
    }

if args.engine == "vllm_api":
    API_ENDPOINT = args.vllm_api_url
    API_KEY = args.vllm_api_key
    API_HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    if "kimi-k2" in args.model_path.lower():
        API_PARAMS = {
            "model": args.model_path,
            # "max_tokens": 8192, # If a user does not specify a max_tokens in their request, then the minimum of max_new_tokens and (max_model_len - prompt_tokens) will be used.
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "stop_token_ids": [163586]
        }
    else:
        API_PARAMS = {
            "model": args.model_path,
            # "max_tokens": args.max_tokens, # If a user does not specify a max_tokens in their request, then the minimum of max_new_tokens and (max_model_len - prompt_tokens) will be used.
            "temperature": args.temperature, 
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }

# Process a batch of data using the API
def process_batch_api(batch):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_item = {
            executor.submit(
                make_api_request_with_retry, 
                [{'content': item['messages'][0]['content'], 'role': 'user'}],
                API_PARAMS,
                API_ENDPOINT,
                API_HEADERS,
            ): item 
            for item in batch
        }

        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            message = item["messages"]
            try:
                api_response = future.result()
                response = api_response.strip() if api_response else ""
                item['messages'] = message + [
                {
                    "role": "assistant",
                    "content": response
                }
            ]
            except Exception as e:
                print(f"Failed to process item: {item} with error: {str(e)}")
                item['messages'] = message + [
                {
                    "role": "assistant",
                    "content": ""
                }
            ]
                
    return batch

# Process a batch of data using local vllm engine
def process_batch_local(batch, llm, params, tokenizer=None):
    user_instructions = [item['messages'][0]['content'] for item in batch]
    prompts = []
    for instruction in user_instructions:
        chat = [{"role": "user", "content": instruction}]
        template = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(template)

    if args.engine == "vllm":
        outputs = llm.generate(prompts, params)
    elif args.engine == "hf":
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(torch.cuda.current_device())
        gen_do_sample = False if args.temperature == 0 else True
        outputs = llm.generate(**inputs,
                tokenizer=tokenizer, 
                do_sample=gen_do_sample, 
                temperature=args.temperature if gen_do_sample else None, # To avoid temperature` (=0) has to be a strictly positive float
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty, 
                max_length=args.max_tokens,
                )
        outputs = tokenizer.batch_decode(outputs[i][len(inputs[i]):] for i in range(len(outputs)))
        # Setting stop tokens seems not working for Gemma, so we manually truncate the outputs
        for i, completion in enumerate(outputs):
            for stop_token in stop_tokens:
                if stop_token in completion:
                    outputs[i] = completion[:completion.index(stop_token)]

    for i, item in enumerate(batch):
        message = item["messages"]
        if args.engine == "vllm":
            response = outputs[i].outputs[0].text.strip()
        elif args.engine == "hf":
            response = outputs[i].strip()
        item['messages'] = message + [
                {
                    "role": "assistant",
                    "content": response
                }
            ]
    return batch

# Process a batch of data using OpenAI GPT API
def process_batch_openai(batch, client):
    for item in batch:
        message = item["messages"]
        try:
            completion = client.chat.completions.create(
                model=args.model_path,
                messages=message,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p        
            )
            response = completion.choices[0].message.content
            item['messages'] = message + [
                {
                    "role": "assistant",
                    "content": response
                }
            ]
        except Exception as e:
            print(f"Failed to process item: {item} with error: {str(e)}")
            item['messages'] = message + [
                {
                    "role": "assistant",
                    "content": ""
                }
            ]
    return batch

# Function to add generation config to metadata
def add_generation_config_to_metadata(dataset, model_abbreviation, generation_params):
    """Add synthetic data generation config to each item's metadata"""
    config_entry = {
        "model": model_abbreviation,
        "generation_params": generation_params,
        "timestamp": int(time())
    }
    
    for item in dataset:
        if "metadata" not in item:
            item["metadata"] = {}
        
        if "synthetic_data_gen_configs" not in item["metadata"]:
            item["metadata"]["synthetic_data_gen_configs"] = []
        
        item["metadata"]["synthetic_data_gen_configs"].append(config_entry)
    
    return dataset

# Generate outputs, update dataset in batches, and overwrite checkpoint
def generate_and_update(dataset, checkpoint_file, llm=None, params=None, tokenizer=None):
    processed_dataset = copy.deepcopy(dataset)

    # Prepare generation parameters for metadata
    generation_params = {
        "engine": args.engine,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_trials": args.num_trials,
        "step": args.step
    }

    # Initialize tokenizer
    if tokenizer is not None:
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if "gemma-2" in args.model_path.lower():
            tokenizer.padding_side = "right"

    # Intialize the dataset with the checkpoint file (if it exists)
    if os.path.exists(checkpoint_file):
        last_checkpoint_idx = len(load_dataset_from_file(checkpoint_file))
        print(f"Checkpoint file found. Resuming from last checkpoint with index {last_checkpoint_idx}.")
        processed_dataset[:last_checkpoint_idx] = load_dataset_from_file(checkpoint_file)
        # Calculate total number of batches
        num_batches = (len(processed_dataset) - last_checkpoint_idx + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"Remaining number of batches: {num_batches}")
    else:
        last_checkpoint_idx = 0
        # Calculate total number of batches
        num_batches = (len(processed_dataset) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"Total number of batches: {num_batches}")

    for i in tqdm(range(num_batches)):
        start_idx = i * BATCH_SIZE + last_checkpoint_idx
        end_idx = min((i + 1) * BATCH_SIZE + last_checkpoint_idx, len(processed_dataset))
        batch = processed_dataset[start_idx:end_idx]
        if args.engine == "together_api" or args.engine == "vllm_api":
            batch = process_batch_api(batch)
        elif args.engine == "openai" or args.engine == "openrouter_api":
            batch = process_batch_openai(batch, llm)
        else:
            batch = process_batch_local(batch, llm, params, tokenizer)
        
        processed_dataset[start_idx:end_idx] = batch
        # Overwrite the same checkpoint file after serveral batches
        if i % CHECKPOINT_EVERY == 0:
            safe_save_checkpoint(processed_dataset[:end_idx], checkpoint_file, convert_to_jsonl=False)
            print(f"Dataset checkpoint saved after batch {i + 1}.")

    # Add generation config to metadata before returning
    processed_dataset = add_generation_config_to_metadata(processed_dataset, model_abbreviation, generation_params)
    
    return processed_dataset

# Main function to control workflow
def main():
    # Load instructions from the input file
    dataset = load_dataset_from_file(INPUT_FILE_NAME)
    
    # Ensure dataset is always a list (fix for single-item JSON files)
    if not isinstance(dataset, list):
        dataset = [dataset]

    if "Mistral-Small-3" in args.model_path and args.engine in ["hf", "vllm"]:
        raise ValueError("Please use vllm_api engine for Mistral-Small-3.")
    elif "Devestral-Small" in args.model_path and args.engine in ["hf", "vllm"]:
        raise ValueError("Please use vllm_api engine for Devestral-Small.")
    
    # Validate OpenRouter API key
    if args.engine == "openrouter_api":
        if not args.openrouter_api_key:
            # Try to get from environment variable
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_api_key:
                args.openrouter_api_key = openrouter_api_key
            else:
                raise ValueError("OpenRouter API Key is required when using openrouter_api engine. Please provide --openrouter_api_key argument or set OPENROUTER_API_KEY environment variable.")
    # Validate Together API key
    elif args.engine == "together_api":
        if not args.together_api_key:
            # Try to get from environment variable
            together_api_key = os.getenv("TOGETHER_API_KEY")
            if together_api_key:
                args.together_api_key = together_api_key
            else:
                raise ValueError("Together API Key is required when using together_api engine. Please provide --together_api_key argument or set TOGETHER_API_KEY environment variable.")

    # Initialize the engine
    if args.engine == "together_api":
        print("Start Together API engine...")
        llm = None
        params = None
        tokenizer = None
    elif args.engine == "vllm_api":
        print("Start vLLM API engine...")
        llm = None
        params = None
        tokenizer = None
    elif args.engine == "vllm":
        # Set the device
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        print("Start Local vllm engine...")
        if "Mistral-Small-3" in args.model_path or "Devestral-Small" in args.model_path:
            llm = LLM(model=MODEL_NAME, 
                dtype=args.dtype,
                trust_remote_code=True,
                max_model_len = args.max_model_len, # limited by kv-cache 
                tensor_parallel_size = args.tensor_parallel_size,
                gpu_memory_utilization = args.gpu_memory_utilization,
                tokenizer_mode="mistral", 
                config_format="mistral", 
                load_format="mistral"
            )
        else:
            llm = LLM(model=MODEL_NAME, 
                dtype=args.dtype,
                trust_remote_code=True,
                max_model_len = args.max_model_len, # limited by kv-cache 
                tensor_parallel_size = args.tensor_parallel_size,
                gpu_memory_utilization = args.gpu_memory_utilization
            )

        params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            stop_token_ids=stop_token_ids,
        )

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.engine == "hf":
        print("Start Hugging Face engine...")
        params = None
        # Load the model and tokenizer
        llm = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map={'':torch.cuda.current_device()},
            torch_dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    elif args.engine == "openai":
        print("Start OpenAI GPT engine...")
        openai_api_key = args.openai_api_key if args.openai_api_key else os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API Key not provided. Please set OPENAI_API_KEY environment variable or provide --openai_api_key argument.")
        llm = OpenAI(api_key=openai_api_key)
        params = None
        tokenizer = None
    elif args.engine == "openrouter_api":
        print("Start OpenRouter API engine...")
        openrouter_api_key = args.openrouter_api_key if args.openrouter_api_key else os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OpenRouter API Key not provided. Please set OPENROUTER_API_KEY environment variable or provide --openrouter_api_key argument.")
        llm = OpenAI(api_key=openrouter_api_key, base_url=args.openrouter_url)
        params = None
        tokenizer = None
    else:
        raise ValueError("Invalid engine type.")

    if args.num_trials == 1:
        updated_dataset = generate_and_update(dataset, checkpoint_file, llm, params, tokenizer=tokenizer)
        save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

        # Optionally remove the checkpoint file after completion
        os.remove(checkpoint_file)
        print("Final dataset saved. Checkpoint removed.")
    else:
        for i in range(args.num_trials):
            if args.engine != "together_api" and params is not None:
                params.seed = int(time() + i)
            updated_dataset = generate_and_update(dataset, checkpoint_files[i], llm, params, tokenizer=tokenizer)
            save_dataset(updated_dataset, saved_files[i], convert_to_jsonl=True)

            # Optionally remove the checkpoint file after completion
            os.remove(checkpoint_files[i])
            print(f"Dataset for trial {i} saved. Checkpoint {i} removed.")

# Run the main function
if __name__ == "__main__":
    main()