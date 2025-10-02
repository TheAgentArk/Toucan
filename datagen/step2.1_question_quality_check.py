import torch
import os
import sys
import argparse
import json
import time
import random
import numpy as np
import glob
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader, exceptions

################
# Use Cases
################
"""
This script evaluates the quality of generated tool use questions with multiple assessment metrics.

Example Usage:

1. Basic Quality Check:
   python step2.1_question_quality_check.py --input_file ../data/ToolUse_s2q_smithery_1000_1tool_1751946990_prepared.jsonl

2. Quality Check with Custom Parameters:
   python step2.1_question_quality_check.py --input_file ../data/ToolUse_q2q_smithery_diversified_5var_1751950980_prepared.jsonl --evaluation_criteria quality

3. Debug Mode:
   python step2.1_question_quality_check.py --input_file input.jsonl --debug --debug_entries 20

Key Parameters:
- --input_file: JSONL file with generated questions to evaluate
- --evaluation_criteria: Specific metrics to evaluate (difficulty,quality,realism or all)
- --debug: Enable debug mode to process only a subset of entries

Assessment Metrics:
- Difficulty (1-5): How difficult it is to determine which tools to use
- Quality (1-5): Overall question quality and clarity
- Scenario Realism (1-5): How realistic and fluent the scenario is

Output:
- Creates a new file in the same directory as input with "_qced" suffix
- Shows only the tools/servers actually used in each question
"""

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Tool Use Question Quality Assessment Manager.")
    # Generation Parameters
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file with generated questions to evaluate.")

    parser.add_argument("--evaluation_criteria", type=str, default="all", choices=["difficulty", "quality", "realism", "all"], help="Specific evaluation criteria to assess.")
    # Debug Settings
    parser.add_argument('--debug', action='store_true', help="Enable debug mode: process only first few entries.")
    parser.add_argument('--debug_entries', type=int, default=10, help="Number of entries to process in debug mode.")
    # System Settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()

def get_quality_check_prompt(question_data):
    """Generate quality assessment prompt for a given question"""
    # Load prompt template from markdown file
    env = Environment(loader=FileSystemLoader('prompts'))
    
    template_name = 'question_quality_check.md'
    try:
        template = env.get_template(template_name).render()
    except exceptions.TemplateNotFound:
        raise FileNotFoundError(f"{template_name} template not found in prompts folder")
    
    # Extract question content from the 'question' field
    question_content = question_data.get('question', '')
    if not question_content:
        # Fallback to messages if question field doesn't exist
        if 'messages' in question_data and question_data['messages']:
            question_content = question_data['messages'][0].get('content', '')
        else:
            question_content = str(question_data)
    
    # Extract metadata for context
    metadata = question_data.get('metadata', {})
    
    # Generate ALL_SERVER_AND_TOOL_INFORMATION
    all_server_tool_info = ""
    if 'mcp_servers' in metadata:
        servers = metadata['mcp_servers']
        server_info_parts = []
        
        for server in servers:
            server_name = server.get('server_name', 'Unknown')
            server_description = server.get('server_info', {}).get('overview', 'No description')
            
            # Get available tools
            available_tools = []
            if 'remote_server_response' in server and 'tools' in server['remote_server_response']:
                available_tools = server['remote_server_response']['tools']
            elif 'tools' in server:
                available_tools = server['tools']
            
            # Format tools list
            tools_list = []
            for tool in available_tools:
                tool_name = tool.get('name', 'Unknown')
                if 'description' in tool:
                    tool_description = tool.get('description')
                    tools_list.append(f"  - {tool_name}: {tool_description}")
                else:
                    tools_list.append(f"  - {tool_name}")
            
            server_section = f"Server Name: {server_name}\nDescription: {server_description}\nAll Available Tools:\n" + "\n".join(tools_list)
            server_info_parts.append(server_section)
        
        all_server_tool_info = "\n\n".join(server_info_parts)
    else:
        all_server_tool_info = "(No MCP server information available)"
    
    # Generate INTENDED_TOOL (target tools as bullet points)
    target_tools = question_data.get('target_tools')
    
    if ',' in target_tools:
        target_tools = [tool.strip() for tool in target_tools.split(',') if tool.strip()]
    else:
        target_tools = [target_tools] if target_tools.strip() else []
    
    # Format target tools as bullet points
    intended_tool_info = ""
    if target_tools:
        for target_tool in target_tools:
            if "::" in target_tool:
                server_name, tool_name = target_tool.split("::")
                intended_tool_info += f"- Server: {server_name} -> Tool: {tool_name}\n"
            else:
                intended_tool_info += f"- {target_tool}\n"
    else:
        raise ValueError(f"No target tools specified for question: {question_data}")
    
    # Replace placeholders in template
    template = template.replace("{QUESTION_CONTENT}", question_content)
    template = template.replace("{ALL_SERVER_AND_TOOL_INFORMATION}", all_server_tool_info)
    template = template.replace("{INTENDED_TOOL}", intended_tool_info.strip())
    
    return template

args = get_args()
print(f"Tool Use Question Quality Assessment Manager.\nArguments:\n{args}") # For logging

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
input_data = []

try:
    with open(args.input_file, 'r') as f:
        # Check if input is a JSON array or JSONL format
        file_content = f.read()
        f.seek(0)  # Reset file pointer
        
        # Try to parse as JSON array first
        try:
            json_data = json.loads(file_content)
            if isinstance(json_data, list):
                # It's a JSON array
                for entry_num, entry in enumerate(json_data, 1):
                    # Validate required fields
                    if not entry:
                        print(f"Skipping entry {entry_num}: empty entry")
                        continue
                    
                    # Check for required structure (messages or question content)
                    has_content = False
                    if 'messages' in entry and entry['messages']:
                        has_content = True
                    elif isinstance(entry, dict) and any(key in entry for key in ['question', 'content']):
                        has_content = True
                    
                    if not has_content:
                        print(f"Skipping entry {entry_num}: no recognizable question content")
                        continue
                    
                    input_data.append(entry)
            else:
                # Single JSON object
                if json_data:
                    has_content = False
                    if 'messages' in json_data and json_data['messages']:
                        has_content = True
                    elif isinstance(json_data, dict) and any(key in json_data for key in ['question', 'content']):
                        has_content = True
                    
                    if has_content:
                        input_data.append(json_data)
                    else:
                        print(f"Skipping single entry: no recognizable question content")

        except json.JSONDecodeError:
            # Not a valid JSON, try JSONL format
            f.seek(0)  # Reset file pointer
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    
                    # Validate required fields
                    if not entry:
                        print(f"Skipping line {line_num}: empty entry")
                        continue
                    
                    # Check for required structure (messages or question content)
                    has_content = False
                    if 'messages' in entry and entry['messages']:
                        has_content = True
                    elif isinstance(entry, dict) and any(key in entry for key in ['question', 'content']):
                        has_content = True
                    
                    if not has_content:
                        print(f"Skipping line {line_num}: no recognizable question content")
                        continue
                    
                    input_data.append(entry)
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                
except Exception as e:
    raise Exception(f"Error reading input file: {e}")

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
# Put output file in same folder as input with _qced_prepared suffix
input_dir = os.path.dirname(args.input_file)
input_basename = os.path.basename(args.input_file).replace('.jsonl', '').replace(".json", "")
output_filename = f"{input_basename}_qced_prepared.jsonl"
output_dir = os.path.join(input_dir, output_filename)

print(f"Output will be saved to: {output_dir}")

################
# Generate outputs
################
results = []
total_iterations = len(input_data)

print(f"Generating quality assessment prompts for {total_iterations} entries...")

for i, entry in enumerate(tqdm(input_data)):
    try:
        # Generate quality check prompt
        quality_prompt = get_quality_check_prompt(entry)
        
        # Create result entry following the original format
        result = {
            "messages": [
                {
                    "role": "user",
                    "content": quality_prompt
                }
            ]
        }
        
        # Add original fields if they exist
        if 'server_analysis' in entry:
            result['server_analysis'] = entry['server_analysis']
        if 'cross_server_workflow' in entry:
            result['cross_server_workflow'] = entry['cross_server_workflow']
        if 'target_tools' in entry:
            result['target_tools'] = entry['target_tools']
        if 'question' in entry:
            result['question'] = entry['question']
            
        # Update metadata to include quality check info
        original_metadata = entry.get('metadata', {})
        updated_metadata = original_metadata.copy()
        updated_metadata.update({
            "prompt_id": f"{i:08d}",
            "row_id": i,
            "task_type": "question_quality_assessment",
            "evaluation_criteria": args.evaluation_criteria,
            "source_file": args.input_file
        })
        result['metadata'] = updated_metadata
        
        results.append(result)
        
    except Exception as e:
        print(f"Error processing entry {i}: {e}")
        continue

# Save the final results
print(f"Saving results to: {output_dir}")
with open(output_dir, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Finished. Total quality assessment prompts generated: {len(results)}")
print(f"Output saved to: {output_dir}")
