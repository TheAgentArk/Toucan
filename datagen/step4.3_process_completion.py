#!/usr/bin/env python3
"""
Response Quality Assessment Processing Manager

This script processes response quality assessment from step5.2.
It extracts reasoning and scores for the quality dimensions and computes tool call accuracy:
1. Completeness (1-5): Whether the assistant fully accomplished the user's request
2. Conciseness (1-5): Whether the assistant solved the task using minimum necessary steps
3. Tool Call Accuracy: Desired tools used percentage and order correctness
"""

import torch
import os
import sys
import argparse
import json
import re
import time
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from utils import clean_json_object, create_preview_json, save_dataset

################
# Tool Call Accuracy Functions
################

def extract_tool_calls_sequence(messages):
    """Extract the sequence of tool calls from conversation messages for accuracy computation"""
    tool_calls = []
    
    for msg in messages:
        role = msg.get('role', '')
        
        if role == 'assistant' and 'function_call' in msg:
            function_call = msg['function_call']
            tool_name = function_call.get('name', 'unknown')
            
            tool_calls.append(tool_name)
    
    return tool_calls

def parse_target_tools(target_tools_str):
    """Parse target tools string into a list of tool names, handling both single and multi-server formats"""
    if not target_tools_str or target_tools_str.strip() == '':
        return []
    
    if ',' in target_tools_str:
        tools = [tool.strip() for tool in target_tools_str.split(',') if tool.strip()]
    else:
        tools = [target_tools_str.strip()]
    
    # Clean up tool names
    cleaned_tools = []
    for tool in tools:
        # Handle multi-server format: "Server Name::tool_name"
        if '::' in tool:
            tool_name = tool.split('::')[-1].strip()
            cleaned_tools.append(tool_name)
        else:
            # Plain tool name
            cleaned_tools.append(tool)
    
    return cleaned_tools

def compute_tool_call_accuracy(conversation_messages, target_tools_str):
    """
    Compute tool call accuracy metrics:
    - desired_tools_used: Percentage (0-1) of desired tools that were actually used
    - order_correct: Whether tools were used in the correct order
    """
    # Parse target tools
    target_tools = parse_target_tools(target_tools_str)
    
    # Extract actual tool calls
    actual_tools = extract_tool_calls_sequence(conversation_messages)
    
    # Calculate percentage of desired tools used
    if target_tools:
        used_count = 0
        for target_tool in target_tools:
            # Check if target_tool is a substring of any actual_tool
            if any(target_tool in actual_tool for actual_tool in actual_tools):
                used_count += 1
        desired_tools_used = used_count / len(target_tools)
    else:
        # If no target tools specified, consider it as fully satisfied
        desired_tools_used = 1.0
    
    # Check if order is correct
    order_correct = True
    if len(target_tools) > 1 and len(actual_tools) >= len(target_tools):
        # Find the subsequence in actual_tools that matches target_tools (substring matching)
        target_idx = 0
        for actual_tool in actual_tools:
            if target_idx < len(target_tools) and target_tools[target_idx] in actual_tool:
                target_idx += 1
        
        # If we didn't match all target tools in order
        if target_idx < len(target_tools):
            order_correct = False
    elif len(target_tools) <= 1:
        # For single tool or no tools, order is trivially correct
        order_correct = True
    else:
        # If actual tools are fewer than target tools, order can't be correct
        order_correct = False
    
    return {
        "desired_tools_used_percentage": desired_tools_used,
        "order_correctness": order_correct
    }

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Quality Assessment Processing Manager.")
    
    # Input Parameters
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with response quality assessment.")
    parser.add_argument("--output_folder", type=str, default="../data/3.completed", help="Path to the output folder where processed files will be saved.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
    parser.add_argument("--save_top_rated", action="store_true", help="Save the top rated responses.")
    parser.add_argument("--top_rated_count", type=int, default=100, help="Number of top rated responses to save.")
    
    return parser.parse_args()

args = get_args()
print(f"Response Quality Assessment Processing Manager.\nArguments:\n{args}") # For logging

def parse_quality_assessment_response(response_content):
    """
    Parse the XML response from response quality assessment to extract reasoning and ratings
    for the two quality dimensions.
    
    Expected format:
    <response>
      <completeness>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </completeness>
      <conciseness>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </conciseness>
    </response>
    """
    try:
        # Clean up the response content
        response_content = response_content.strip()
        
        # Try to find the response XML block
        response_match = re.search(r'<response>(.*?)</response>', response_content, re.DOTALL)
        if not response_match:
            # If no response tags found, try to extract individual components
            return extract_quality_components(response_content)
        else:
            response_xml = response_match.group(1)
            return extract_quality_components(response_xml)
        
    except Exception as e:
        print(f"Error parsing response quality assessment: {e}")
        return None

def convert_rating_to_score(rating_text, dimension_name):
    """
    Convert text-based ratings to numerical scores (1-5) based on dimension type.
    """
    if not rating_text:
        return None
    
    # Clean and normalize the rating text
    rating = rating_text.strip().lower()
    
    # Rating mappings for each dimension
    rating_mappings = {
        'completeness': {
            'very incomplete': 1,
            'incomplete': 2,
            'partially complete': 3,
            'mostly complete': 4,
            'fully complete': 5
        },
        'conciseness': {
            'very redundant': 1,
            'redundant': 2,
            'average': 3,
            'concise': 4,
            'very concise': 5
        }
    }
    
    # Get the mapping for this dimension
    if dimension_name not in rating_mappings:
        # Try to parse as integer if no mapping found (fallback)
        try:
            score = int(rating.strip())
            return score if 1 <= score <= 5 else None
        except ValueError:
            return None
    
    mapping = rating_mappings[dimension_name]
    
    # Find exact match first
    if rating in mapping:
        return mapping[rating]
    
    # Try partial matches (for cases where the text might have extra characters)
    for key, value in mapping.items():
        if key in rating or rating in key:
            return value
    
    # Try to parse as integer if no text match found (fallback)
    try:
        score = int(rating.strip())
        return score if 1 <= score <= 5 else None
    except ValueError:
        return None

def extract_quality_components(response_xml):
    """
    Extract reasoning and scores for the two quality dimensions.
    """
    # Extract completeness
    completeness = extract_quality_dimension(response_xml, 'completeness')
    
    # Extract conciseness
    conciseness = extract_quality_dimension(response_xml, 'conciseness')
    
    # Validate that we have all required components
    if not all([completeness, conciseness]):
        return None
        
    return {
        "completeness": completeness,
        "conciseness": conciseness
    }

def extract_quality_dimension(text, dimension_name):
    """
    Extract reasoning and rating for a specific quality dimension.
    Returns a dict with 'reasoning' and 'score' keys.
    """
    # Extract the entire dimension block
    dimension_pattern = f'<{dimension_name}>(.*?)</{dimension_name}>'
    dimension_match = re.search(dimension_pattern, text, re.DOTALL)
    
    if not dimension_match:
        return None
    
    dimension_content = dimension_match.group(1)
    
    # Extract reasoning
    reasoning = extract_xml_content(dimension_content, 'reasoning')
    
    # Extract rating
    rating_text = extract_xml_content(dimension_content, 'rating').lower()
    
    # Convert text rating to numerical score
    score = convert_rating_to_score(rating_text, dimension_name)
    
    if not reasoning or score is None:
        return None
    
    return {
        "reasoning": reasoning.strip(),
        "score": score
    }

def extract_xml_content(text, tag):
    """
    Extract content from XML tags, handling both with and without CDATA.
    """
    # Try with CDATA first
    pattern = f'<{tag}>\\s*<!\\[CDATA\\[(.*?)\\]\\]>\\s*</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try without CDATA
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try with comments format (from the template)
    pattern = f'<{tag}>\\s*<!--.*?-->\\s*(.*?)\\s*</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    return ""


def extract_quality_assessments(input_file, output_file, preview_file=None):
    """
    Extract response quality assessments from assistant responses with XML format.
    Returns the list of processed data.
    """
    extraction_stats = {
        "total_processed": 0,
        "successfully_parsed": 0,
        "failed_parsing": 0,
        "invalid_assessments": 0,
        "extraction_timestamp": int(time.time())
    }
    
    # Store all successful assessments for top-rated filtering and return
    all_assessments = []
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        total_processed = 0
        successfully_parsed = 0
        
        for line in tqdm(f_in, desc="Extracting Response Quality Assessments"):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                annotation_model_info = data.get("metadata", {}).get("synthetic_data_gen_configs", {})[-1]
                annotation_model_nickname = annotation_model_info.get("model", "unknown")
                
                # Find the assistant message with the assessment
                assistant_message = None
                for msg in messages:
                    if msg.get("role") == "assistant":
                        assistant_message = msg
                        break
                
                if not assistant_message:
                    print("No assistant message found. Skipping.")
                    continue
                
                total_processed += 1
                extraction_stats["total_processed"] += 1
                
                # Parse the response quality assessment
                parsed_response = parse_quality_assessment_response(assistant_message["content"])
                
                if not parsed_response:
                    print(f"Failed to parse response quality assessment for row {total_processed}. Skipping.")
                    extraction_stats["failed_parsing"] += 1
                    continue
                
                # Check for valid scores
                valid_assessment = True
                for dimension in ['completeness', 'conciseness']:
                    if dimension not in parsed_response:
                        valid_assessment = False
                        break
                    dimension_data = parsed_response[dimension]
                    if dimension_data is None or 'score' not in dimension_data:
                        valid_assessment = False
                        break
                    score = dimension_data['score']
                    if not isinstance(score, int) or score < 1 or score > 5:
                        valid_assessment = False
                        break
                
                if not valid_assessment:
                    print(f"Invalid scores in assessment for row {total_processed}. Skipping.")
                    extraction_stats["invalid_assessments"] += 1
                    continue
                
                successfully_parsed += 1
                extraction_stats["successfully_parsed"] += 1
                
                # Calculate overall score for top-rated filtering (validated structure above)
                scores = []
                for dimension in ['completeness', 'conciseness']:
                    dimension_data = parsed_response.get(dimension)
                    if dimension_data and isinstance(dimension_data, dict) and 'score' in dimension_data:
                        scores.append(dimension_data['score'])
                
                overall_score = sum(scores) / len(scores) if scores else 0
                parsed_response["overall_score"] = overall_score
                
                # Recover conversation_history to messages
                recovered_messages = data.get("conversation_history", [])
                if not recovered_messages:
                    # Fallback to existing messages if conversation_history is missing
                    recovered_messages = [{"role": "user", "content": data.get("question", "")}]
                
                # Compute tool call accuracy for the recovered conversation
                target_tools_str = data.get("target_tools", "")
                tool_call_accuracy = compute_tool_call_accuracy(recovered_messages, target_tools_str)
                
                # Preserve existing response_quality_assessment fields and add computed tool call accuracy
                existing_rqa = data.get("response_quality_assessment", {})
                # Merge parsed assessment with existing fields and tool call accuracy
                merged_assessment = {**existing_rqa, **parsed_response, **tool_call_accuracy}
                
                # Prepare the result structure with response_quality_assessment field
                result = {
                    "messages": recovered_messages,
                    "question": data.get("question", ""),
                    "target_tools": data.get("target_tools", ""),
                    "question_quality_assessment_kimik2w4a16": data.get("question_quality_assessment_kimik2w4a16", {}) or data.get("quality_assessment_kimik2w4a16", {}),
                    "response_quality_assessment_oss120b": merged_assessment,
                    "server_analysis": data.get("server_analysis", ""),
                    "cross_server_workflow": data.get("cross_server_workflow", ""),
                    "metadata": data.get("metadata", {}),
                }

                # Clean unusual line terminators
                result = clean_json_object(result)
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # Store for return and top-rated filtering
                processed_data.append(result)
                if args.save_top_rated:
                    all_assessments.append(result)
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                extraction_stats["failed_parsing"] += 1
                continue
            except Exception as e:
                print(f"Unexpected error processing line: {e}")
                extraction_stats["failed_parsing"] += 1
                continue
    
    # Save top-rated assessments if enabled
    if args.save_top_rated and all_assessments:
        # Sort by overall score (descending) and take top N
        all_assessments.sort(key=lambda x: x['overall_score'], reverse=True)
        top_rated = all_assessments[:args.top_rated_count]
        
        # Save top-rated file
        top_rated_file = output_file.replace('_processed.jsonl', '_top_rated_processed.json')
        with open(top_rated_file, 'w', encoding='utf-8') as f_top:
            json.dump(top_rated, f_top, ensure_ascii=False, indent=2)
        
        print(f"Top {len(top_rated)} rated assessments saved to {top_rated_file}")
        print(f"Top score: {top_rated[0]['overall_score']:.2f}, Lowest in top: {top_rated[-1]['overall_score']:.2f}")
    
    # Save extraction statistics
    extraction_stats_file = output_file.replace('_processed.jsonl', '_processed_stats.json')
    with open(extraction_stats_file, 'w', encoding='utf-8') as stats_outf:
        json.dump(extraction_stats, stats_outf, ensure_ascii=False, indent=2)
    
    print(f"Finished extracting quality assessments. Total processed: {total_processed}, Successfully parsed: {successfully_parsed}")
    print(f"Output saved to {output_file}")
    print(f"Extraction statistics saved to {extraction_stats_file}")
    
    # Print extraction summary
    print(f"\nExtraction Summary:")
    print(f"  Total processed: {extraction_stats['total_processed']}")
    print(f"  Successfully parsed: {extraction_stats['successfully_parsed']}")
    print(f"  Failed parsing: {extraction_stats['failed_parsing']}")
    print(f"  Invalid assessments: {extraction_stats['invalid_assessments']}")
    print(f"  Success rate: {(extraction_stats['successfully_parsed'] / extraction_stats['total_processed'] * 100):.1f}%" if extraction_stats['total_processed'] > 0 else "Success rate: 0.0%")
    
    # Create preview if requested
    if preview_file:
        create_preview_json(output_file, preview_file)
    
    # Return processed data for JSON saving
    return processed_data



def print_processing_summary(stats):
    """
    Print a summary of the response quality assessment processing results.
    """
    print("\n" + "="*60)
    print("RESPONSE QUALITY ASSESSMENT PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Total Assessments Processed: {stats['total_assessments']}")
    
    print("\nScore Distributions:")
    print("-" * 30)
    
    for dimension in ["completeness", "conciseness"]:
        print(f"\n{dimension.replace('_', ' ').title()}:")
        distribution = stats["score_distributions"][dimension]
        avg_score = stats["average_scores"][dimension]
        
        for score in range(1, 6):
            count = distribution[score]
            percentage = (count / stats["total_assessments"]) * 100 if stats["total_assessments"] > 0 else 0
            print(f"  Score {score}: {count} assessments ({percentage:.1f}%)")
        
        print(f"  Average: {avg_score:.2f}")
    
    print("="*60 + "\n")


def main():
    print(f"Response Quality Assessment Processing Pipeline. Arguments: {args}")

    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.basename(args.input_file)
    print(f"Input directory: {input_dir}")
    print(f"Input basename: {input_basename}")

    if "_results" not in input_basename:
        raise ValueError("Input file does not contain '_results' in the name.")

    # Use the provided output folder
    output_path = os.path.abspath(args.output_folder)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created output directory: {output_path}")
    
    # Define output paths with 'processed' suffix
    base_name = input_basename.replace('_results.jsonl', '')
    processed_output_jsonl = f"{output_path}/{base_name}_processed.jsonl"
    processed_output_json = f"{output_path}/{base_name}_processed.json"
    processed_output_review = f"{output_path}/preview_{base_name}_processed.json"

    print(f"Output will be saved to: {output_path}")

    # Run single processing step using extracted logic
    print("Processing response quality assessments...")
    processed_data = extract_quality_assessments(args.input_file, processed_output_jsonl, processed_output_review)
    
    # Save as JSON format using save_dataset
    print("Saving JSON format...")
    save_dataset(processed_data, processed_output_json, convert_to_jsonl=False)

if __name__ == "__main__":
    main()