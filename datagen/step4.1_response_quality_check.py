import torch
import os
import sys
import argparse
import json
import time
import random
import numpy as np
import copy
import glob
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, exceptions
from utils import load_dataset_from_file

################
# Use Cases
################
"""
This script evaluates the quality of tool-use conversation responses with multiple assessment metrics.

Example Usage:

1. Basic Response Quality Check:
   python step4.1_response_quality_check.py --input_file ../data/2.completion_kimi/processed/preview_ToolUse_s2q_smithery_5000_2tool_1753732565_kimik2w4a16_3sanitized_qced_kimik2w4a16_2prepared_kimik2w4a16_rule_filtered.json

2. Response Quality Check in Debug Mode:
   python step4.1_response_quality_check.py --input_file ../data/responses.jsonl --debug --debug_entries 5

Key Parameters:
- --input_file: JSON/JSONL file with conversations to evaluate
- --debug: Enable debug mode to process only a subset of entries

Assessment Metrics:
- Completeness (1-5): Whether the assistant fully accomplished the user's request end-to-end
- Conciseness (1-5): Whether the assistant solved the task using minimum necessary steps and verbosity

Output:
- Creates a new file in the same directory as input with "_response_qced" suffix
- Prepares prompts for LLM evaluation
"""

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Tool Use Response Quality Assessment Manager.")
    # Generation Parameters
    parser.add_argument("--input_file", type=str, required=True, help="Input JSON/JSONL file with conversations to evaluate.")


    # Debug Settings
    parser.add_argument('--debug', action='store_true', help="Enable debug mode: process only first few entries.")
    parser.add_argument('--debug_entries', type=int, default=10, help="Number of entries to process in debug mode.")
    # System Settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()


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

def get_response_quality_check_prompt(conversation_data):
    """Generate response quality assessment prompt for a given conversation"""
    # Load prompt template from markdown file
    env = Environment(loader=FileSystemLoader('prompts'))
    
    template_name = 'response_quality_check.md'
    try:
        template = env.get_template(template_name).render()
    except exceptions.TemplateNotFound:
        raise FileNotFoundError(f"{template_name} template not found in prompts folder")
    
    # Extract conversation messages
    messages = conversation_data.get('messages', [])
    if not messages:
        raise ValueError("No messages found in conversation data")
    
    # Extract user goal from the first user message
    question = ""
    for msg in messages:
        if msg.get('role') == 'user':
            question = msg.get('content', '')
            break
    
    # Extract intended tools and order
    target_tools = conversation_data.get('target_tools', '')
    
    # Generate condensed conversation
    condensed_conversation = condense_conversation(messages)
    
    # Replace placeholders in template with the 3 required parameters
    template = template.replace("{QUESTION_CONTENT}", question)
    template = template.replace("{INTENDED_TOOL}", target_tools)
    template = template.replace("{CONVERSATION_HISTORY}", condensed_conversation)
    
    return template

args = get_args()
print(f"Tool Use Response Quality Assessment Manager.\nArguments:\n{args}") # For logging

#################
# System Settings
#################

# Set random seed
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

#################
# Load Input Data
#################
if not os.path.exists(args.input_file):
    raise FileNotFoundError(f"Input file not found: {args.input_file}")

print(f"Loading input data from: {args.input_file}")

input_data = load_dataset_from_file(args.input_file)
print(f"Loaded {len(input_data)} entries from input file")

if len(input_data) == 0:
    raise ValueError("No valid entries found in input file")

#################
# Process entries
#################
if args.debug:
    # Limit to debug_entries in debug mode
    entries_to_process = min(args.debug_entries, len(input_data))
    input_data = input_data[:entries_to_process]
    print(f"Debug mode: processing only {entries_to_process} entries")

#################
# Create output file
#################
# Put output file in same folder as input with _response_qced suffix
input_dir = os.path.dirname(args.input_file)
input_basename = os.path.basename(args.input_file).replace('.jsonl', '').replace(".json", "")
output_filename = f"{input_basename}_response_qced_prepared.jsonl"
output_dir = os.path.join(input_dir, output_filename)

print(f"Output will be saved to: {output_dir}")

################
# Generate outputs
################
results = []
total_iterations = len(input_data)

print(f"Generating response quality assessment prompts for {total_iterations} entries...")

for i, entry in enumerate(tqdm(input_data)):
    try:
        modified_entry = copy.deepcopy(entry)
        
        # Generate response quality check prompt
        quality_prompt = get_response_quality_check_prompt(entry)
        
        # Rename original messages to conversation_history
        modified_entry['conversation_history'] = modified_entry['messages']
        del modified_entry['messages']

        # Create a new messages field
        modified_entry['messages'] = [
            {
                "role": "user",
                "content": quality_prompt
            }
        ]
            
        results.append(modified_entry)
        
    except Exception as e:
        print(f"Error processing entry {i}: {e}")
        continue

# Save the final results
print(f"Saving results to: {output_dir}")
with open(output_dir, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Finished. Total response quality assessment prompts generated: {len(results)}")
print(f"Output saved to: {output_dir}") 