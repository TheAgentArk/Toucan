#!/usr/bin/env python3
"""
Quality Assessment Processing Manager

This script processes quality assessment responses from step2.1.1_question_quality_check.py.
It extracts reasoning and scores for the six quality dimensions:
1. Tool Selection Difficulty
2. Tool Selection Uniqueness
3. Question Quality  
4. Scenario Realism
5. Verifiable
6. Stability
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
from utils import clean_json_object, create_preview_json

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Quality Assessment Processing Manager.")
    
    # Input Parameters
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with quality assessment responses.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
    parser.add_argument("--save_top_rated", action="store_true", help="Save the top rated questions.")
    parser.add_argument("--top_rated_count", type=int, default=100, help="Number of top rated questions to save.")
    
    return parser.parse_args()

args = get_args()
print(f"Quality Assessment Processing Manager.\nArguments:\n{args}") # For logging

def parse_quality_assessment_response(response_content):
    """
    Parse the XML response from quality assessment to extract reasoning and ratings
    for the six quality dimensions.
    
    Expected format:
    <response>
      <tool_selection_difficulty>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </tool_selection_difficulty>
      <tool_selection_uniqueness>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </tool_selection_uniqueness>
      <question_quality>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </question_quality>
      <scenario_realism>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </scenario_realism>
      <verifiable>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </verifiable>
      <stability>
        <reasoning>...</reasoning>
        <rating>text rating</rating>
      </stability>
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
        print(f"Error parsing quality assessment response: {e}")
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
        'tool_selection_difficulty': {
            'very easy': 1,
            'easy': 2,
            'medium': 3,
            'hard': 4,
            'very hard': 5
        },
        'tool_selection_uniqueness': {
            'not unique': 1,
            'somewhat unique': 2,
            'moderately unique': 3,
            'quite unique': 4,
            'highly unique': 5
        },
        'question_quality': {
            'very poor': 1,
            'poor': 2,
            'average': 3,
            'good': 4,
            'excellent': 5
        },
        'scenario_realism': {
            'unrealistic': 1,
            'somewhat unrealistic': 2,
            'moderately realistic': 3,
            'realistic': 4,
            'highly realistic': 5
        },
        'verifiable': {
            'hard to verify': 1,
            'somewhat hard': 2,
            'moderately verifiable': 3,
            'mostly verifiable': 4,
            'easy to verify': 5
        },
        'stability': {
            'highly unstable': 1,
            'somewhat unstable': 2,
            'moderately stable': 3,
            'mostly stable': 4,
            'highly stable': 5
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
    Extract reasoning and scores for the six quality dimensions.
    """
    # Extract tool selection difficulty
    tool_difficulty = extract_quality_dimension(response_xml, 'tool_selection_difficulty')
    
    # Extract tool selection uniqueness
    tool_uniqueness = extract_quality_dimension(response_xml, 'tool_selection_uniqueness')
    
    # Extract question quality
    question_quality = extract_quality_dimension(response_xml, 'question_quality')
    
    # Extract scenario realism
    scenario_realism = extract_quality_dimension(response_xml, 'scenario_realism')
    
    # Extract verifiable
    verifiable = extract_quality_dimension(response_xml, 'verifiable')
    
    # Extract stability
    stability = extract_quality_dimension(response_xml, 'stability')
    
    # Validate that we have all required components
    if not all([tool_difficulty, tool_uniqueness, question_quality, scenario_realism, verifiable, stability]):
        return None
        
    return {
        "tool_selection_difficulty": tool_difficulty,
        "tool_selection_uniqueness": tool_uniqueness,
        "question_quality": question_quality,
        "scenario_realism": scenario_realism,
        "verifiable": verifiable,
        "stability": stability
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
    Extract quality assessments from assistant responses with XML format.
    """
    extraction_stats = {
        "total_processed": 0,
        "successfully_parsed": 0,
        "failed_parsing": 0,
        "invalid_assessments": 0,
        "extraction_timestamp": int(time.time())
    }
    
    # Store all successful assessments for top-rated filtering
    all_assessments = []
    
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        total_processed = 0
        successfully_parsed = 0
        
        for line in tqdm(f_in, desc="Extracting Quality Assessments"):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                annotation_model_info = data.get("metadata", {}).get("synthetic_data_gen_configs", {})[-1]
                annotation_model_nickname = annotation_model_info.get("model", "unknown")
                
                # Find the assistant message
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
                
                # Parse the quality assessment response
                parsed_response = parse_quality_assessment_response(assistant_message["content"])
                
                if not parsed_response:
                    print(f"Failed to parse quality assessment response for row {total_processed}. Skipping.")
                    extraction_stats["failed_parsing"] += 1
                    continue
                
                # Check for valid scores
                valid_assessment = True
                for dimension in ['tool_selection_difficulty', 'tool_selection_uniqueness', 'question_quality', 'scenario_realism', 'verifiable', 'stability']:
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
                for dimension in ['tool_selection_difficulty', 'tool_selection_uniqueness', 'question_quality', 'scenario_realism', 'verifiable', 'stability']:
                    dimension_data = parsed_response.get(dimension)
                    if dimension_data and isinstance(dimension_data, dict) and 'score' in dimension_data:
                        scores.append(dimension_data['score'])
                
                overall_score = sum(scores) / len(scores) if scores else 0
                parsed_response["overall_score"] = overall_score
                
                # Prepare the result structure
                result = {
                    "messages": [
                        {
                            "role": "user",
                            "content": data.get("question", "")
                        }
                    ],
                    f"quality_assessment_{annotation_model_nickname}": parsed_response,
                    "metadata": data.get("metadata", {}),
                    "question": data.get("question", ""),
                    "target_tools": data.get("target_tools", ""),
                    "server_analysis": data.get("server_analysis", ""),
                    "cross_server_workflow": data.get("cross_server_workflow", "")
                }

                # Clean unusual line terminators
                result = clean_json_object(result)
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                
                # Store for top-rated filtering if enabled
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
        top_rated_file = output_file.replace('_1extracted.jsonl', '_top_rated_1extracted.json')
        with open(top_rated_file, 'w', encoding='utf-8') as f_top:
            json.dump(top_rated, f_top, ensure_ascii=False, indent=2)
        
        print(f"Top {len(top_rated)} rated assessments saved to {top_rated_file}")
        print(f"Top score: {top_rated[0]['overall_score']:.2f}, Lowest in top: {top_rated[-1]['overall_score']:.2f}")
    
    # Save extraction statistics
    extraction_stats_file = output_file.replace('.jsonl', '_extraction_stats.json')
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

def prepare_quality_assessments(input_file, output_file, preview_file=None):
    """
    Prepare quality assessments following the original format with quality_assessment field added.
    """
    print(f"Preparing quality assessments from {input_file}")
    all_results = []
    stats = {
        "processing_info": {
            "input_file": input_file,
            "output_file": output_file,
            "processing_timestamp": int(time.time()),
            "processing_type": "quality_assessment_preparation"
        },
        "total_assessments": 0,
        "score_distributions": {
            "tool_selection_difficulty": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "tool_selection_uniqueness": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "question_quality": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "scenario_realism": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "verifiable": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "stability": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        },
        "average_scores": {}
    }
    
    # Store all assessments for top-rated filtering
    all_assessments_for_top_rated = []
    
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as outf:
        for line in tqdm(f, desc="Preparing Quality Assessments"):
            data = json.loads(line)
            # Find the quality_assessment entry (there might be suffixes)
            quality_assessment = None
            for key in data.keys():
                if key.startswith("quality_assessment"):
                    quality_assessment = data[key]
                    break
            
            if quality_assessment is None:
                print(f"Warning: No quality_assessment field found in entry")
                continue
            
            # Collect statistics
            stats["total_assessments"] += 1
            
            for dimension in ["tool_selection_difficulty", "tool_selection_uniqueness", "question_quality", "scenario_realism", "verifiable", "stability"]:
                if dimension in quality_assessment and quality_assessment[dimension] is not None:
                    dimension_data = quality_assessment[dimension]
                    if "score" in dimension_data:
                        score = dimension_data["score"]
                        stats["score_distributions"][dimension][score] += 1
            
            # Initialize result dictionary
            result = {}
            
            # Add all other original fields
            for key, value in data.items():
                result[key] = value
            
            # Clean unusual line terminators
            result = clean_json_object(result)
            outf.write(json.dumps(result, ensure_ascii=False) + "\n")
            all_results.append(result)
            
            # Store for top-rated filtering if enabled
            if args.save_top_rated:
                all_assessments_for_top_rated.append(result)
    
    # Save top-rated assessments if enabled
    if args.save_top_rated and all_assessments_for_top_rated:
        # Calculate overall scores and sort
        for assessment in all_assessments_for_top_rated:
            qa = next((v for k, v in assessment.items() if k.startswith("quality_assessment")), None)
            scores = []
            for dimension in ["tool_selection_difficulty", "tool_selection_uniqueness", "question_quality", "scenario_realism", "verifiable", "stability"]:
                if dimension in qa and qa[dimension] is not None and "score" in qa[dimension]:
                    scores.append(qa[dimension]["score"])
            assessment["overall_score"] = sum(scores) / len(scores) if scores else 0
        
        # Sort by overall score (descending) and take top N
        all_assessments_for_top_rated.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        top_rated = all_assessments_for_top_rated[:args.top_rated_count]
        
        # Save top-rated file
        top_rated_file = output_file.replace('_2prepared.jsonl', '_top_rated_2prepared.json')
        with open(top_rated_file, 'w', encoding='utf-8') as f_top:
            json.dump(top_rated, f_top, ensure_ascii=False, indent=2)
        
        print(f"Top {len(top_rated)} rated prepared assessments saved to {top_rated_file}")
        if top_rated:
            print(f"Top score: {top_rated[0].get('overall_score', 0):.2f}, Lowest in top: {top_rated[-1].get('overall_score', 0):.2f}")
    
    # Calculate average scores
    for dimension in ["tool_selection_difficulty", "tool_selection_uniqueness", "question_quality", "scenario_realism", "verifiable", "stability"]:
        total_score = sum(score * count for score, count in stats["score_distributions"][dimension].items())
        stats["average_scores"][dimension] = total_score / stats["total_assessments"] if stats["total_assessments"] > 0 else 0
    
    # Save statistics
    stats_file = output_file.replace('_2prepared.jsonl', '_2prepared_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as stats_outf:
        json.dump(stats, stats_outf, ensure_ascii=False, indent=2)
    
    print(f"Finished preparing quality assessments. Output saved to {output_file}.")
    print(f"Statistics saved to {stats_file}")
    print_processing_summary(stats)
    
    # Create preview if requested
    if preview_file:
        create_preview_json(output_file, preview_file)

def print_processing_summary(stats):
    """
    Print a summary of the quality assessment processing results.
    """
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Total Assessments Processed: {stats['total_assessments']}")
    
    print("\nScore Distributions:")
    print("-" * 30)
    
    for dimension in ["tool_selection_difficulty", "tool_selection_uniqueness", "question_quality", "scenario_realism", "verifiable", "stability"]:
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
    print(f"Quality Assessment Processing Pipeline. Arguments: {args}")

    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.basename(args.input_file)
    print(f"Input directory: {input_dir}")
    print(f"Input basename: {input_basename}")

    if "qced" not in input_basename or "_results" not in input_basename:
        raise ValueError("Input file does not contain 'qced' or '_results' in the name.")

    # Create output file / folder
    output_path = f"{input_dir}/quality_checked"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define output paths
    base_name = input_basename.replace('_results.jsonl', '')
    extracted_output = f"{output_path}/{base_name}_1extracted.jsonl"
    extracted_output_review = f"{output_path}/preview_{base_name}_1extracted.json"
    prepared_output = f"{output_path}/{base_name}_2prepared.jsonl"
    prepared_output_review = f"{output_path}/preview_{base_name}_2prepared.json"
    prepared_json_output = f"{output_path}/{base_name}_2prepared.json"

    # Run processing steps
    print("Step 1: Extracting quality assessments from XML responses...")
    extract_quality_assessments(args.input_file, extracted_output, extracted_output_review)
    
    print("Step 2: Preparing quality assessments for final use...")
    prepare_quality_assessments(extracted_output, prepared_output, prepared_output_review)

if __name__ == "__main__":
    main()