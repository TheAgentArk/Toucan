import os
import sys
import argparse
import json
import re
import time
from tqdm import tqdm
from utils import clean_json_object, create_preview_json

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Tool Use Completion Filtering Manager - Rule-based filtering for step 4.1 outputs.")
    
    # Input Parameters
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with completion responses from step 4.1.")
    
    # System Settings
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
    
    return parser.parse_args()

args = get_args()

print(f"Tool Use Completion Filtering Manager.\nArguments:\n{args}") # For logging

################
# Utility Functions
################

# Expanded error patterns from the notebook
ERROR_PATTERNS = [
    r"error",
    r"an error occurred",
    r"exception",
    r"failed",
    r"not found",
    r"unavailable",
    r"invalid",
    r"timed out",
    r"could not",
    r"does not exist",
    r"unsuccessful",
    r"no .* found",
    r"too many requests",
    r"no device selected",
    r"no active sessions",
    r"missing required credentials",
    r"unauthorized",
    r"not authorized",
    r"not authenticated",
    r"\[Error",
    r"\[ERROR",
    r"\[error]",
    r"未找到",
    r"失败",
]

def has_system_prompt(messages):
    """
    Check if the conversation has a system prompt.
    Agent must have started successfully if system prompt exists.
    """
    if not messages or not isinstance(messages, list):
        return False
    
    for msg in messages:
        if msg.get("role") == "system":
            return True
    return False

def has_tool_calls(messages):
    """
    Check if the conversation contains any tool calls.
    """
    if not messages or not isinstance(messages, list):
        return False
    
    for msg in messages:
        if msg.get("role") == "assistant":
            # Check for function_call (old format) or tool_calls (new format)
            if msg.get("function_call") or msg.get("tool_calls"):
                return True
    return False

def has_no_error_in_tool_responses(messages, check_multi_turn_only):
    """
    Check if there's at least one tool/function response without error patterns.
    For single-turn: check all tool responses.
    For multi-turn: only check tool responses after the second user message.
    Returns True if:
    1. No tool responses exist (valid case), OR
    2. At least one tool response has no error patterns
    """
    if not messages or not isinstance(messages, list):
        return False
    
    user_message_count = 0
    has_tool_response = False
    should_check = True
    
    for msg in messages:
        # Count user messages to determine which turn we're in
        if msg.get("role") == "user":
            user_message_count += 1
            
        # For multi-turn: only check tool responses after the second user message
        if check_multi_turn_only:
            should_check = user_message_count >= 2
        # For single-turn: check all tool responses
        else:
            should_check = True
        
        # Check function/tool responses
        if msg.get("role") == "function" and should_check:
            has_tool_response = True
            content = msg.get("content", "")
            if content:
                # Check if this response has no error patterns
                has_error = False
                for pattern in ERROR_PATTERNS:
                    if re.search(pattern, content, re.IGNORECASE):
                        has_error = True
                        break
                
                # If this response has no error, the entry is valid
                if not has_error:
                    return True
    
    # If no tool responses were found in the relevant turn(s), consider it valid
    # If tool responses exist but all have errors, return False
    return not has_tool_response

def has_error_in_assistant_responses(messages):
    """
    Check if any assistant responses contain error patterns.
    """
    if not messages or not isinstance(messages, list):
        return False
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:
                if re.search(r"\[Error", content, re.IGNORECASE):
                    return True
    return False

def has_empty_final_assistant_message(messages):
    """
    Check if the last message is from assistant with empty content.
    Returns True if the last message is assistant with empty/None content. For OpenAI agent.
    """
    if not messages or not isinstance(messages, list) or len(messages) == 0:
        return False
    
    last_msg = messages[-1]
    if last_msg.get("role") == "assistant":
        content = last_msg.get("content", "")
        # Check if content is empty, None, or just whitespace
        if not content or not content.strip():
            return True
    
    return False

def has_exclamation_marks_in_assistant_messages(messages):
    """
    Check if any assistant messages contain multiple exclamation marks (!!!!!!!!!!!!).
    Returns True if any assistant message contains the pattern.
    """
    if not messages or not isinstance(messages, list):
        return False
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content and "!!!!!!!!!!!!" in content:
                return True
    return False

def is_valid_entry(data, file_path):
    """
    Check if an entry passes all rule-based filters.
    Returns (is_valid, reason) tuple.
    """
    messages = data.get("messages", [])
    check_multi_turn_only = len([msg for msg in messages if msg.get("role") == "user"]) > 1 and "question_split" not in file_path
    
    # Rule 1: Must have system prompt (agent successfully started)
    if not has_system_prompt(messages):
        return False, "no_system_prompt"
    
    # Rule 2: Must have tool calls
    if not has_tool_calls(messages):
        return False, "no_tool_calls"
    
    # Rule 3: Must have at least one successful tool response
    if not has_no_error_in_tool_responses(messages, check_multi_turn_only):
        return False, "no_successful_tool_response"
    
    # Rule 4: Must not have errors in assistant responses
    if has_error_in_assistant_responses(messages):
        return False, "error_in_assistant_response"
    
    # Rule 5: Must not have an empty final assistant message
    if has_empty_final_assistant_message(messages):
        return False, "empty_final_assistant_message"
    
    # Rule 6: Must not have exclamation marks in assistant messages
    if has_exclamation_marks_in_assistant_messages(messages):
        return False, "exclamation_marks_in_assistant_message"
    
    return True, "valid"

def filter_completions(input_file, output_file, preview_file=None):
    """
    Apply rule-based filtering to completion data from step 4.1.
    """
    print(f"Starting rule-based filtering from {input_file}")
    
    # Statistics tracking
    stats = {
        "total_processed": 0,
        "valid_entries": 0,
        "filtered_out": {
            "no_system_prompt": 0,
            "no_tool_calls": 0,
            "no_successful_tool_response": 0,
            "error_in_assistant_response": 0,
            "empty_final_assistant_message": 0,
            "exclamation_marks_in_assistant_message": 0
        }
    }
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, desc="Filtering Completions"):
            try:
                data = json.loads(line)
                stats["total_processed"] += 1
                
                # Apply filtering rules
                is_valid, reason = is_valid_entry(data, file_path=input_file)
                
                if is_valid:
                    stats["valid_entries"] += 1
                    
                    # Clean unusual line terminators
                    data = clean_json_object(data)
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                else:
                    stats["filtered_out"][reason] += 1
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing line: {e}")
                continue
    
    # Print filtering results
    print_filtering_summary(stats)
    print(f"Filtered output saved to {output_file}")
    
    # Create preview if requested
    if preview_file:
        create_preview_json(output_file, preview_file)

def print_filtering_summary(stats):
    """
    Print a summary of the filtering results.
    """
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    
    total = stats["total_processed"]
    valid = stats["valid_entries"]
    filtered = total - valid
    
    print(f"Total Entries Processed: {total}")
    print(f"Valid Entries: {valid} ({(valid/total*100):.1f}%)")
    print(f"Filtered Out: {filtered} ({(filtered/total*100):.1f}%)")
    
    print("\nFiltering Breakdown:")
    print("-" * 30)
    for reason, count in stats["filtered_out"].items():
        percentage = (count / total) * 100 if total > 0 else 0
        reason_display = reason.replace("_", " ").title()
        print(f"  {reason_display}: {count} ({percentage:.1f}%)")
    
    print("="*60 + "\n")

def main():
    print(f"Tool Use Completion Filtering Pipeline. Arguments: {args}")

    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.basename(args.input_file)
    print(f"Input directory: {input_dir}")
    print(f"Input basename: {input_basename}")
    
    # Create output file / folder
    output_path = f"{input_dir}/processed"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define output paths
    base_name = input_basename.replace('_results.jsonl', '').replace('.jsonl', '')
    filtered_output = f"{output_path}/{base_name}_rule_filtered.jsonl"
    filtered_output_review = f"{output_path}/preview_{base_name}_rule_filtered.json"

    # Run filtering
    print("Applying rule-based filtering...")
    filter_completions(args.input_file, filtered_output, filtered_output_review)
    
    print(f"Final filtered output saved to: {filtered_output}")

if __name__ == "__main__":
    main()