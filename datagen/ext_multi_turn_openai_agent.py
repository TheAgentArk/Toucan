import torch
import os
import sys
import argparse
import copy
import json
import re
import subprocess
import tempfile
from time import sleep, time
from tqdm import tqdm

from utils import load_dataset_from_file, save_dataset, get_model_abbreviation, create_preview_json

################
# Use Cases
################
"""
Multi-turn conversation generation script.

Features:
- Auto-detects existing turns in conversations
- Generates missing turns to reach target count
- Supports mixed files with different turn counts
- Batch processing for efficiency

Usage:
python ext_multi_turn_openai_agent.py --input_file data.json --num_desired_turns 3

The script will:
- Count existing user messages per item
- Skip items that already have enough turns
- Generate user queries and agent responses for missing turns
- Create preview files for inspection
"""

################
# Configurations
################
def get_args():
    parser = argparse.ArgumentParser(description="Multi-turn Conversation Generation Manager.")
    
    # Input/Output Settings
    parser.add_argument("--input_file", type=str, required=True, help="Input dataset file (single-turn conversations)")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path (default: auto-generated)")
    
    # Multi-turn Settings  
    parser.add_argument("--num_desired_turns", type=int, default=5, help="Number of turns to generate (default: 3)")
    parser.add_argument("--user_simulation_model", type=str, default="openai/gpt-oss-120b", 
                        help="Model for generating user follow-up queries")
    
    # Agent Completion Settings (passed to completion_openai_agent.py)
    parser.add_argument("--model_path", type=str, default="openai/gpt-oss-120b",
                        help="Model path for agent responses")
    parser.add_argument("--engine", type=str, default="vllm_api", choices=["vllm_api", "openrouter_api"],
                        help="Completion engine")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for generation")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for generation")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout for each completion")
    parser.add_argument("--max_workers", type=int, default=None, help="Max parallel workers")
    
    # API Settings
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1", 
                        help="OpenRouter API URL")
    parser.add_argument("--openrouter_api_key", type=str, 
                        default="", 
                        help="OpenRouter API Key")
    parser.add_argument("--vllm_api_url", type=str, default="http://localhost:8000/v1", 
                        help="vLLM API URL")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY", help="vLLM API Key")
    parser.add_argument("--smithery_api_key", type=str, default="", 
                        help="Smithery API Key")
    parser.add_argument("--smithery_profile", type=str, default="", 
                        help="Smithery Profile")
    parser.add_argument("--smithery_api_pool", type=str, default="smithery_api_pool.json", 
                        help="Smithery API pool file")
    
    # Other Settings
    parser.add_argument("--checkpoint_every", type=int, default=16, help="Checkpoint frequency")
    parser.add_argument("--agent", type=str, default="openai_agent", help="Agent type")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort")
    parser.add_argument("--parallel_function_calls", type=bool, default=True, help="Parallel function calls")
    parser.add_argument("--max_rounds_per_turn", type=int, default=10, help="Maximum number of turns for agent inference")
    
    return parser.parse_args()

################
# Helper Functions
################
def condense_conversation(messages):
    """Create a condensed version of the conversation for analysis"""
    condensed = []
    
    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')
        
        if role == 'system':
            # Skip system messages or include very brief summary
            continue
        elif role == 'user':
            condensed.append(f"User: {content}")
        elif role == 'assistant':
            if 'function_call' in msg:
                # Summarize tool calls
                function_call = msg['function_call']
                tool_name = function_call.get('name', 'unknown')
                condensed.append(f"Assistant: [Called {tool_name}]")
            else:
                # Include assistant message content
                condensed.append(f"Assistant: {content}")
        elif role == 'function':
            # Summarize function results
            tool_name = msg.get('name', 'unknown')
            result_content = msg.get('content', '')
            condensed.append(f"Tool {tool_name}: {result_content[:100]} ... (truncated)")
    
    return "\n".join(condensed)

# Single item processing functions removed - using batch processing only

# Single agent response function also removed - using batch processing only

def generate_batch_user_queries(items, turn_number, args, input_file_path):
    """Generate follow-up user queries for all items in batch"""
    print(f"\n🔄 BATCH: Generating user queries for Turn {turn_number}...")
    
    # Create batch of follow-up prompts
    batch_prompts = []
    for i, item in enumerate(items):
        conversation_history = item['messages']
        condensed_conversation = condense_conversation(conversation_history)
        
        follow_up_prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": f"## Conversation history between you, the user, and the LLM agent:\n{condensed_conversation}\n\n## New Task:\nPlease ask a follow up question to the LLM agent. The question should be related to the conversation history and the agent's response.\n\nRemember, you are the user, not the LLM agent. Use user's tone and style to ask the question. Output the new question in the following XML format: <question>[Your Follow Up Question]</question>"
                }
            ],
            'metadata': item.get('metadata', {})
        }
        batch_prompts.append(follow_up_prompt)
    
    # Create temporary directory in input file's directory
    input_dir = os.path.dirname(os.path.abspath(input_file_path))
    temp_dir = os.path.join(input_dir, f"temp_multiturn_turn{turn_number}_{int(time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create batch input file
    temp_prompt_file = os.path.join(temp_dir, 'batch_follow_up_prepared.jsonl')
    with open(temp_prompt_file, 'w') as f:
        for prompt in batch_prompts:
            f.write(json.dumps(prompt) + '\n')
    
    # Generate batch output file path
    base_name = temp_prompt_file[:temp_prompt_file.rfind('.')]
    if base_name.endswith("_prepared"):
        base_name = base_name[:-9]
    model_abbreviation = get_model_abbreviation(args.user_simulation_model)
    temp_result_file = f"{base_name}_{model_abbreviation}_results.jsonl"
    
    # Build completion command for batch
    completion_cmd = [
        'python', 'completion_endpoint.py',
        '--input_file', temp_prompt_file,
        '--model_path', args.user_simulation_model,
        '--engine', args.engine,
        '--temperature', str(args.temperature),
        '--max_tokens', str(args.max_tokens),
        '--top_p', str(args.top_p),
        '--step', '7.1'
    ]
    
    # Add engine-specific parameters
    if args.engine == 'openrouter_api':
        completion_cmd.extend([
            '--openrouter_url', args.openrouter_url,
            '--openrouter_api_key', args.openrouter_api_key
        ])
    elif args.engine == 'vllm_api':
        # completion_endpoint.py expects the full chat/completions URL
        vllm_chat_url = args.vllm_api_url
        if not vllm_chat_url.endswith('/chat/completions'):
            if vllm_chat_url.endswith('/v1'):
                vllm_chat_url = vllm_chat_url + '/chat/completions'
            else:
                vllm_chat_url = vllm_chat_url + '/v1/chat/completions'
        
        completion_cmd.extend([
            '--vllm_api_url', vllm_chat_url,
            '--vllm_api_key', args.vllm_api_key
        ])
    
    try:
        print(f"🤖 Generating batch follow-up queries using {args.user_simulation_model}...")
        print(f"🔧 Batch input file: {temp_prompt_file}")
        print(f"🔧 Expected batch output: {temp_result_file}")
        print(f"🔧 Command: {' '.join(completion_cmd)}")
        
        # Use Popen for real-time output
        print(f"🤖 Starting batch user query generation (real-time output):")
        print("-" * 60)
        
        process = subprocess.Popen(
            completion_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True, 
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read output line by line in real-time
        output_lines = []
        try:
            while True:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())  # Print to terminal immediately
                    output_lines.append(line)
                elif process.poll() is not None:
                    break
            
            # Wait for process to complete and get return code
            return_code = process.wait(timeout=600)
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return_code = -1
            print("❌ Process timed out after 600 seconds")
        
        print("-" * 60)
        print(f"✅ Batch command completed with return code: {return_code}")
        
        if return_code != 0:
            print(f"❌ Batch user simulation failed:")
            print(f"   Return code: {return_code}")
            return [None] * len(items)
        
        # Load batch results
        if os.path.exists(temp_result_file):
            with open(temp_result_file, 'r') as f:
                batch_results = [json.loads(line) for line in f]
            
            # Extract follow-up questions
            new_queries = []
            for i, result_item in enumerate(batch_results):
                follow_up_content = result_item['messages'][-1]['content']
                match = re.search(r"<question>(.*?)</question>", follow_up_content, re.DOTALL)
                
                if match:
                    follow_up_question = match.group(1).strip()
                    print(f"✅ Generated follow-up for item {i}: {follow_up_question}")
                    new_queries.append(follow_up_question)
                else:
                    print(f"❌ No follow-up question found for item {i}")
                    print(f"   Generated content: {follow_up_content}")
                    new_queries.append(None)
            
            return new_queries
        else:
            print(f"❌ No batch result file generated at: {temp_result_file}")
            return [None] * len(items)
            
    except subprocess.TimeoutExpired:
        print("❌ Batch user simulation timed out (600s)")
        return [None] * len(items)
    except Exception as e:
        print(f"❌ Error in batch user simulation: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return [None] * len(items)
    finally:
        # Keep temporary directory for inspection
        print(f"📁 Temporary files saved in: {temp_dir}")

def generate_batch_agent_responses(items_with_queries, turn_number, args, input_file_path):
    """Generate agent responses for all items in batch"""
    print(f"\n🚀 BATCH: Generating agent responses for Turn {turn_number}...")
    
    # Prepare batch items with new user queries
    batch_items = []
    valid_indices = []
    
    for i, (item, query) in enumerate(items_with_queries):
        if query is not None:
            # Add user query to conversation
            item_with_query = copy.deepcopy(item)
            item_with_query['messages'].append({
                "role": "user",
                "content": query
            })
            batch_items.append(item_with_query)
            valid_indices.append(i)
        else:
            print(f"⚠️ Skipping item {i} (no user query)")
    
    if not batch_items:
        print("⚠️ No valid items for agent response generation")
        return [item for item, query in items_with_queries]
    
    # Create temporary directory in input file's directory
    input_dir = os.path.dirname(os.path.abspath(input_file_path))
    temp_dir = os.path.join(input_dir, f"temp_multiturn_agent_turn{turn_number}_{int(time())}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create batch input file
    temp_prompt_file = os.path.join(temp_dir, 'batch_agent_prepared.jsonl')
    with open(temp_prompt_file, 'w') as f:
        for item in batch_items:
            f.write(json.dumps(item) + '\n')
    
    # Generate batch output file path - match completion_openai_agent.py naming
    # completion_openai_agent.py keeps the full input filename and appends config + results
    input_base = temp_prompt_file[:temp_prompt_file.rfind('.')]  # Remove .jsonl extension
    model_abbreviation = get_model_abbreviation(args.model_path)
    config_str = f"{model_abbreviation}_{args.reasoning_effort}_pfc" if args.parallel_function_calls else f"{model_abbreviation}_{args.reasoning_effort}_sfc"
    temp_result_file = f"{input_base}_{config_str}_results.jsonl"
    
    # Build completion command for batch
    completion_cmd = [
        'python', 'completion_openai_agent.py',
        '--input_file', temp_prompt_file,
        '--model_path', args.model_path,
        '--engine', args.engine,
        '--temperature', str(args.temperature),
        '--max_tokens', str(args.max_tokens),
        '--top_p', str(args.top_p),
        '--timeout', str(args.timeout),
        '--openrouter_url', args.openrouter_url,
        '--openrouter_api_key', args.openrouter_api_key,
        '--vllm_api_url', args.vllm_api_url,
        '--vllm_api_key', args.vllm_api_key,
        '--smithery_api_key', args.smithery_api_key,
        '--smithery_profile', args.smithery_profile,
        '--smithery_api_pool', args.smithery_api_pool,
        '--checkpoint_every', str(min(args.checkpoint_every, len(batch_items))),
        '--agent', args.agent,
        '--reasoning_effort', args.reasoning_effort,
        '--parallel_function_calls', str(args.parallel_function_calls),
        '--max_turns', str(args.max_rounds_per_turn*turn_number),
        '--step', '7.2'
    ]
    
    if args.max_workers:
        completion_cmd.extend(['--max_workers', str(args.max_workers)])
    
    try:
        print(f"🚀 Generating batch agent responses using {args.model_path}...")
        print(f"🔧 Batch input file: {temp_prompt_file}")
        print(f"🔧 Expected batch output: {temp_result_file}")
        print(f"🔧 Command: {' '.join(completion_cmd)}")
        
        # Use Popen for real-time output
        print(f"🚀 Starting batch agent generation (real-time output):")
        print("-" * 60)
        
        process = subprocess.Popen(
            completion_cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True, 
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Read output line by line in real-time
        output_lines = []
        try:
            while True:
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())  # Print to terminal immediately
                    output_lines.append(line)
                elif process.poll() is not None:
                    break
            
            # Wait for process to complete and get return code
            return_code = process.wait(timeout=1200)
            
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return_code = -1
            print("❌ Process timed out after 1200 seconds")
        
        print("-" * 60)
        print(f"✅ Batch command completed with return code: {return_code}")
        
        if return_code != 0:
            print(f"⚠️ Batch agent generation had non-zero return code: {return_code}")
            print(f"   But checking if partial results were saved...")
            # Don't return early - check for partial results
        
        # Load batch results
        if os.path.exists(temp_result_file):
            with open(temp_result_file, 'r') as f:
                batch_results = [json.loads(line) for line in f]
            
            # Merge results back with original items
            updated_items = []
            result_idx = 0
            
            for i, (item, query) in enumerate(items_with_queries):
                if i in valid_indices and result_idx < len(batch_results):
                    updated_items.append(batch_results[result_idx])
                    result_idx += 1
                    print(f"✅ Agent response generated for item {i}")
                else:
                    updated_items.append(item)
                    if query is not None:
                        print(f"❌ Failed to generate agent response for item {i}")
            
            return updated_items
        else:
            print(f"❌ No batch result file generated at: {temp_result_file}")
            return [item for item, query in items_with_queries]
            
    except subprocess.TimeoutExpired:
        print("❌ Batch agent generation timed out (1200s)")
        return [item for item, query in items_with_queries]
    except Exception as e:
        print(f"❌ Error in batch agent generation: {e}")
        print(f"   Exception type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return [item for item, query in items_with_queries]
    finally:
        # Keep temporary directory for inspection
        print(f"📁 Temporary files saved in: {temp_dir}")

def generate_multi_turn_conversations_batch(dataset, num_desired_turns, args, input_file_path):
    """Generate multi-turn conversations for entire dataset in batches"""
    
    print(f"\n🔄 BATCH PROCESSING: Generating {num_desired_turns} turns for {len(dataset)} items...")
    
    # Start with the original dataset
    current_dataset = copy.deepcopy(dataset)
    
    # Detect existing turns for each item and track completion status
    for item in current_dataset:
        prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')
        
        # Count existing turns by counting user messages
        user_messages = [msg for msg in item['messages'] if msg.get('role') == 'user']
        existing_turns = len(user_messages)
        
        if 'metadata' not in item:
            item['metadata'] = {}
        
        # Check if already has completed_turns in metadata, otherwise use detected count
        if 'completed_turns' not in item['metadata']:
            item['metadata']['completed_turns'] = existing_turns
        else:
            # Use the higher of detected vs metadata value
            item['metadata']['completed_turns'] = max(existing_turns, item['metadata']['completed_turns'])
        
        current_turns = item['metadata']['completed_turns']
        print(f"✅ Item {prompt_id}: Found {current_turns} existing turn(s)")
        
        # Skip processing if already has enough turns
        if current_turns >= num_desired_turns:
            print(f"   → Already has {current_turns} turns, target is {num_desired_turns} - skipping")
    
    # Generate additional turns batch by batch
    for turn in range(1, num_desired_turns + 1):
        print(f"\n{'='*60}")
        print(f"🔄 PROCESSING TURN {turn}")
        print(f"{'='*60}")
        
        # Filter items that need this turn (have exactly turn-1 turns and need more)
        active_items = [item for item in current_dataset 
                       if item['metadata'].get('completed_turns', 0) == turn - 1 and turn <= num_desired_turns]
        
        if not active_items:
            print(f"⚠️ No items need Turn {turn} generation")
            continue
        
        print(f"📊 Processing {len(active_items)} items for Turn {turn}")
        
        # Step 1: Generate user queries for all items
        user_queries = generate_batch_user_queries(active_items, turn, args, input_file_path)
        
        # Step 2: Generate agent responses for all items
        items_with_queries = list(zip(active_items, user_queries))
        updated_items = generate_batch_agent_responses(items_with_queries, turn, args, input_file_path)
        
        # Step 3: Update the dataset
        active_indices = [i for i, item in enumerate(current_dataset) 
                         if item['metadata'].get('completed_turns', 0) == turn - 1]
        
        for idx, (updated_item, original_item) in zip(active_indices, zip(updated_items, active_items)):
            current_dataset[idx] = updated_item
            
            # Only update completion status if the item actually has more messages than before
            # (indicating successful agent response generation)
            original_msg_count = len(original_item['messages'])
            updated_msg_count = len(updated_item['messages'])
            
            if updated_msg_count > original_msg_count:
                # Successfully generated agent response - this turn is complete
                current_dataset[idx]['metadata']['completed_turns'] = turn
                print(f"✅ Item {idx}: Turn {turn} completed successfully")
            else:
                # Agent response generation failed - keep previous completion status
                print(f"⚠️ Item {idx}: Turn {turn} failed, keeping previous status")
                # Keep the original completed_turns value
                current_dataset[idx]['metadata']['completed_turns'] = original_item['metadata'].get('completed_turns', turn - 1)
            
        print(f"✅ Turn {turn}: Batch processing complete")
    
    # Update final metadata for all items
    for item in current_dataset:
        if 'metadata' not in item:
            item['metadata'] = {}
        
        completed_turns = item['metadata'].get('completed_turns', 1)
        
        item['metadata']['multi_turn_generation'] = {
            'requested_turns': num_desired_turns,
            'completed_turns': completed_turns,
            'user_simulation_model': args.user_simulation_model,
            'agent_model': args.model_path,
            'generation_timestamp': int(time())
        }
        
        # Update prompt_id to indicate multi-turn
        if 'prompt_id' in item['metadata']:
            original_id = item['metadata']['prompt_id']
            if '_multiturn_' not in original_id:
                item['metadata']['prompt_id'] += f'_multiturn_{completed_turns}'
    
    return current_dataset

################
# Main Processing Function
################
def process_multi_turn_dataset(input_file, output_file, num_desired_turns, args):
    """Process entire dataset to generate multi-turn conversations"""
    
    print("=" * 60)
    print("🚀 MULTI-TURN CONVERSATION GENERATION")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Desired turns: {num_desired_turns}")
    print(f"User simulation model: {args.user_simulation_model}")
    print(f"Agent model: {args.model_path}")
    print("=" * 60)
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = load_dataset_from_file(input_file)
    
    # Ensure dataset is a list
    if not isinstance(dataset, list):
        dataset = [dataset]
    
    print(f"✅ Loaded {len(dataset)} items")
    
    # Process all items using batch method
    try:
        multi_turn_dataset = generate_multi_turn_conversations_batch(dataset, num_desired_turns, args, input_file)
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        # Fallback to original dataset if batch processing fails
        multi_turn_dataset = dataset
    
    # Save results
    print(f"💾 Saving {len(multi_turn_dataset)} multi-turn conversations...")
    save_dataset(multi_turn_dataset, output_file, convert_to_jsonl=True)
    
    print(f"✅ Multi-turn generation complete! Saved to: {output_file}")
    
    # Print summary statistics
    total_turns = sum(
        item.get('metadata', {}).get('multi_turn_generation', {}).get('completed_turns', 1) 
        for item in multi_turn_dataset
    )
    avg_turns = total_turns / len(multi_turn_dataset) if multi_turn_dataset else 0
    
    print("📊 GENERATION SUMMARY")
    print(f"Total items processed: {len(multi_turn_dataset)}")
    print(f"Total turns generated: {total_turns}")
    print(f"Average turns per item: {avg_turns:.2f}")
    
    # Create preview file
    try:
        base_name = os.path.splitext(output_file)[0]
        preview_file = f"{base_name}_preview.json"
        create_preview_json(output_file, preview_file, num_entries=5)
        print(f"📋 Preview file created: {preview_file}")
    except Exception as e:
        print(f"⚠️ Failed to create preview: {e}")

################
# Main Function
################
def main():
    args = get_args()
    print(f"Multi-turn Conversation Generation Manager.\nArguments: {args}")
    
    # Validate input file
    if not os.path.exists(args.input_file):
        raise ValueError(f"Input file does not exist: {args.input_file}")
    
    # Generate output file name if not provided
    if args.output_file is None:
        base_name = os.path.splitext(args.input_file)[0]
        model_abbrev = get_model_abbreviation(args.model_path)
        user_model_abbrev = get_model_abbreviation(args.user_simulation_model)
        args.output_file = f"{base_name}_multiturn_{args.num_desired_turns}t_{model_abbrev}_{user_model_abbrev}.jsonl"
    
    print(f"📁 Output will be saved to: {args.output_file}")
    
    # Validate num_desired_turns
    if args.num_desired_turns < 1:
        raise ValueError("num_desired_turns must be at least 1")
    
    try:
        # Process the dataset
        process_multi_turn_dataset(
            args.input_file, 
            args.output_file, 
            args.num_desired_turns, 
            args
        )
        
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise

if __name__ == "__main__":
    main()
