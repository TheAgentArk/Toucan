import torch
import os
import sys
import argparse
import json
import re
import time
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import xml.etree.ElementTree as ET
import shutil
from utils import clean_json_object, clean_html_comments, create_preview_json

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Tool Use Question Processing and Sanitization Manager.")
    
    # Input Parameters
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file with model responses.")
    parser.add_argument("--mode", default="smithery", type=str, choices=["smithery"], help="Mode for processing. Currently only supports smithery tools.")
    
    # Similarity Parameters for Deduplication
    parser.add_argument("--sentence_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="SentenceTransformer model for encoding questions.")
    parser.add_argument("--encoding_batch_size", type=int, default=256, help="Batch size for encoding sentences.")
    parser.add_argument("--distance_threshold", type=float, default=0.1, help="Cosine similarity threshold for filtering similar questions.")
    parser.add_argument("--search_space_size", type=int, default=100, help="Number of nearest neighbors to search for similarity.")
    parser.add_argument("--search_batch_size", type=int, default=4096, help="Batch size for searching similarity.")
    
    # System Settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on ('cuda' or 'cpu').")
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job.")
    parser.add_argument("--disable_sanitize", action="store_true", help="Disable the sanitize step and use extracted questions directly.")
    parser.add_argument("--disable_prepare", action="store_true", help="Disable the prepare step and stop after sanitization.")
    parser.add_argument("--enable_tool_hint", action="store_true", help="Enable tool hints by appending tool usage information to the end of each question.")
    
    return parser.parse_args()

args = get_args()

# Validation for conflicting flags
if args.disable_sanitize and args.disable_prepare:
    print("Warning: Both --disable_sanitize and --disable_prepare are set. --disable_prepare will be ignored since sanitization is already disabled.")

print(f"Tool Use Question Processing and Sanitization Manager.\nArguments:\n{args}") # For logging

################
# Utility Functions
################

def filter_metadata_by_target_tools(metadata, target_tools_str):
    """
    Filter metadata to only include MCP servers that provide the tools mentioned in target_tools.
    This reduces file size by removing server info that's not actually used for the question.
    Only applies filtering when multi_server_allocation_strategy is random_featured.
    
    Args:
        metadata: The original metadata dictionary
        target_tools_str: Comma-separated string of target tool names (format: "tool_name" or "server_name::tool_name")
    
    Returns:
        Filtered metadata dictionary with only relevant server information
    """
    if not (isinstance(target_tools_str, str) and target_tools_str.strip()):
        return metadata

    target_tools_raw = [t.strip() for t in target_tools_str.split(',') if t.strip()]

    if not target_tools_raw:
        return metadata

    if metadata.get("question_gen_args", {}).get("multi_server_allocation_strategy", "") != "random_featured":
        return metadata

    if "mcp_servers" not in metadata:
        return metadata
    
    # Start filtering
    filtered_metadata = metadata.copy()
    servers = filtered_metadata["mcp_servers"]
    
    # Only process with :: case (standard format)
    server_tool_combos = set()
    for tool_entry in target_tools_raw:
        if '::' in tool_entry:
            server_name, tool_name = tool_entry.split('::', 1)
            server_tool_combos.add((server_name.strip(), tool_name.strip()))
        else:
            raise ValueError(
                f"All target tools must be specified in 'server_name::tool_name' format. "
                f"Found tool entry without server: '{tool_entry}'."
            )

    # Use precise server-tool matching
    filtered_servers = []
    for server_info in servers:
        server_name = server_info.get("server_name", "Unknown Server")
        remote_response = server_info.get("remote_server_response", {})
        server_tools = remote_response.get("tools", [])

        # Check if this server provides any of the needed tool-server combinations
        has_matching_tools = False
        for tool in server_tools:
            tool_name = tool.get("name", "")
            if (server_name, tool_name) in server_tool_combos:
                has_matching_tools = True
                break

        if has_matching_tools:
            # Keep the entire server with all its tools
            filtered_servers.append(server_info)

    filtered_metadata["mcp_servers"] = filtered_servers

    # Update server count
    if "server_count" in filtered_metadata:
        filtered_metadata["server_count"] = len(filtered_servers)

    return filtered_metadata

def parse_xml_response(response_content, metadata=None):
    """
    Parse the XML response from the assistant to extract server_analysis, target_tools, and question.
    Supports both single-server and multi-server XML formats:
    
    Single-server format:
    <response>
      <server_analysis>...</server_analysis>
      <target_tools>...</target_tools>
      <question>...</question>
    </response>
    
    Multi-server format:
    <response>
      <server_analysis>...</server_analysis>
      <cross_server_workflow>...</cross_server_workflow>
      <target_tools>...</target_tools>
      <question>...</question>
    </response>
    """
    try:
        # Clean up the response content
        response_content = response_content.strip()
        
        # Try to find the response XML block
        response_match = re.search(r'<response>(.*?)</response>', response_content, re.DOTALL)

        if not response_match:
            # If no response tags found, try to extract individual components
            return extract_individual_components(response_content, metadata)
        else:
            response_xml = response_match.group(1)
            return extract_individual_components(response_xml, metadata)
        
    except Exception as e:
        print(f"Error parsing XML response: {e}")
        return None

def extract_individual_components(response_xml, metadata=None):
    """
    Extract individual components from XML, handling both single and multi-server formats.
    """

    mode = metadata.get("question_gen_args", {}).get("mode", "unknown")
    if mode == "unknown":
        raise ValueError(f"Mode is unknown in metadata: {metadata}")
    elif mode == "multi_server":
        # Multi-server format: extract server_analysis and cross_server_workflow separately
        server_analysis = extract_xml_content(response_xml, 'server_analysis')
        cross_server_workflow = extract_xml_content(response_xml, 'cross_server_workflow')
        
        # Extract target tools and question
        target_tools = extract_xml_tools(response_xml, metadata)
        question = extract_xml_content(response_xml, 'question')
        # Validate that we have all required components
        if not all([server_analysis, target_tools, question]):
            return None
            
        return {
            "server_analysis": server_analysis.strip(),
            "cross_server_workflow": cross_server_workflow.strip() if cross_server_workflow else "",
            "target_tools": target_tools.strip(),
            "question": clean_html_comments(question.strip())
        }
    else:
        # Single-server format: extract server_analysis directly
        server_analysis = extract_xml_content(response_xml, 'server_analysis')
        
        # Extract target tools and question
        target_tools = extract_xml_tools(response_xml, metadata)
        question = extract_xml_content(response_xml, 'question')
        
        # Validate that we have all required components
        if not all([server_analysis, target_tools, question]):
            return None
            
        return {
            "server_analysis": server_analysis.strip(),
            "target_tools": target_tools.strip(),
            "question": clean_html_comments(question.strip())
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

def extract_xml_tools(text, metadata=None):
    """
    Extract target tools from various formats, supporting any number of tools:
    1. <target_tools><tool server="Server1">search</tool><tool server="Server2">fetch</tool></target_tools>
    2. <target_tools><tool>search</tool><tool>fetch</tool><tool>update</tool></target_tools>
    3. <target_tools>search, fetch, update</target_tools>
    4. <target_tools>search\nfetch\nupdate</target_tools>
    5. Any tool tag variations: <tool1>, <tool_1>, <tool_a>, etc.
    
    Returns: A string with comma-separated tools in format "tool_name" or "server_name::tool_name"
             Returns empty string if any tool/server validation fails
    """
    # First extract the target_tools content
    try:
        target_tools_content = extract_xml_content(text, 'target_tools')
        if not target_tools_content:
            target_tools_content = extract_xml_content(text, 'target_tool')
        if not target_tools_content:
            return ""
    except Exception as e:
        return ""
    
    # Get mode from metadata
    mode = metadata.get("question_gen_args", {}).get("mode", "unknown")
    if mode == "unknown":
        raise ValueError(f"Mode is unknown in metadata: {metadata}")

    
    # Strategy 1: XML tool tags with server attributes - must be multi_server mode
    if mode == "multi_server":     
        # Check if we have XML tool tags with server attributes (Strategy 1)
        tool_pattern = r'<tool[^>]*server="([^"]*)"[^>]*>(.*?)</tool>'
        tool_matches = re.findall(tool_pattern, target_tools_content, re.DOTALL)
        
        if len(tool_matches) == 0:
            return ""

        # Helper function to validate server and tool existence
        def validate_server_tool(server_name, tool_name, metadata):
            """
            Validate if server_name and tool_name exist in metadata.
            Returns (validated_server_name, validated_tool_name, is_valid) tuple.
            """
            if not metadata or 'mcp_servers' not in metadata:
                return server_name, tool_name, False
            
            servers = metadata['mcp_servers']
            if not isinstance(servers, list):
                return server_name, tool_name, True
            
            # First try exact match
            for server_info in servers:
                server_info_name = server_info.get("server_name", "")
                if server_info_name == server_name:
                    # Check if this server has the tool
                    remote_response = server_info.get("remote_server_response", {})
                    server_tools = remote_response.get("tools", [])
                    for tool in server_tools:
                        if tool.get("name", "") == tool_name:
                            return server_name, tool_name, True
            
            # Try adding suffixes to server_name
            suffixes = [" Server", " MCP Server", " MCP"]
            for suffix in suffixes:
                server_name_with_suffix = server_name + suffix
                for server_info in servers:
                    server_info_name = server_info.get("server_name", "")
                    if server_info_name == server_name_with_suffix:
                        # Check if this server has the tool
                        remote_response = server_info.get("remote_server_response", {})
                        server_tools = remote_response.get("tools", [])
                        for tool in server_tools:
                            if tool.get("name", "") == tool_name:
                                return server_name_with_suffix, tool_name, True
            
            # No match found
            return server_name, tool_name, False
        
        tools_list = []
        
        for server, tool_name in tool_matches:
            tool_name = tool_name.strip()
            if tool_name:
                # Validate server and tool existence
                validated_server, validated_tool, is_valid = validate_server_tool(server.strip(), tool_name, metadata)
                
                if not is_valid:
                    print(f"Warning: Tool '{tool_name}' not found in server '{server}' or server not found in metadata. Skipping entire input.")
                    return ""  # Return empty string to skip entire input
                
                # If server info is available and validated, use server_name::tool_name format
                if validated_server and validated_server.strip():
                    tool_entry = f"{validated_server}::{validated_tool}"
                else:
                    tool_entry = validated_tool
                
                tools_list.append(tool_entry)
        
        return ', '.join(tools_list)
    
    # Strategy 2: Plain text parsing - must be single_server mode
    elif mode == "single_server":
        
        # Clean up the content first
        content = target_tools_content.strip()
        
        # Remove any remaining XML-like tags that might be malformed
        content = re.sub(r'<[^>]*>', '', content)
        
        # Try comma separation first
        if ',' in content:
            tools = [tool.strip() for tool in content.split(',') if tool.strip()]
        else:
            # Try newline/whitespace separation
            tools = [tool.strip() for tool in re.split(r'[\n\r\s]+', content) if tool.strip()]    
        # Filter out empty strings
        tools = [tool for tool in tools if tool]

        # check number of tools
        if len(tools) != metadata["question_gen_args"].get("num_tools", 0):
            return ""
        
        # For single_server mode, validate if tools exist in the server
        if metadata and 'mcp_servers' in metadata:
            servers = metadata.get('mcp_servers', [])
            if isinstance(servers, list) and len(servers) >= 1:
                # In single_server mode, use the first server for validation
                server_info = servers[0]
                remote_response = server_info.get("remote_server_response", {})
                server_tools = remote_response.get("tools", [])
                available_tools = set()
                for tool in server_tools:
                    tool_name = tool.get("name", "")
                    if tool_name:
                        available_tools.add(tool_name)
                
                # Check if all tools exist in the server
                for tool_name in tools:
                    if tool_name not in available_tools:
                        print(f"Warning: Tool '{tool_name}' not found in the server. Skipping entire input.")
                        return ""  # Return empty string to skip entire input
        
        return ', '.join(tools)
    else:
        return ""


def extract_questions(input_file, output_file, preview_file=None):
    """
    Extract structured questions from assistant responses with XML format.
    Handles both single-server and multi-server modes.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        total_processed = 0
        successfully_parsed = 0
        mode_counts = {"single_server": 0, "multi_server": 0, "unknown": 0}
        
        for line in tqdm(f_in, desc="Extracting Questions"):
            try:
                data = json.loads(line)
                messages = data.get("messages", [])
                metadata = data.get("metadata", {})
                
                # Detect mode from metadata
                mode = metadata.get("mode", "unknown")
                if mode in mode_counts:
                    mode_counts[mode] += 1
                else:
                    mode_counts["unknown"] += 1
                
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
                
                # Parse the XML response
                parsed_response = parse_xml_response(assistant_message["content"], metadata)
                
                if not parsed_response:
                    print(f"Failed to parse XML response for row {total_processed}. Skipping.")
                    continue
                
                # Check if target_tools is empty (indicating validation failure)
                if not parsed_response["target_tools"] or parsed_response["target_tools"].strip() == "":
                    print(f"Server-tools pairs validation failed or empty for row {total_processed}. Skipping.")
                    continue
                
                # Check for common failure patterns
                question = parsed_response["question"]
                if not question or len(question.strip()) < 10:
                    print(f"Question too short or empty for row {total_processed}. Skipping.")
                    continue
                
                # Filter out bad responses
                bad_patterns = [
                    "I cannot",
                    "I can't",
                    "I'm unable",
                    "I apologize",
                    "I'm sorry",
                    "BAD_DOCUMENT",
                    "Please provide",
                    "Could you please",
                    "I need more information"
                ]
                
                if any(pattern.lower() in question.lower() for pattern in bad_patterns):
                    print(f"Question contains bad pattern for row {total_processed}. Skipping.")
                    continue
                
                successfully_parsed += 1
                
                # Filter metadata to only include servers that provide the target tools
                filtered_metadata = filter_metadata_by_target_tools(
                    metadata, 
                    parsed_response["target_tools"]
                )

                # Remove metadata -> mcp_servers -> server_info -> file_path, tools to reduce the size of the metadata
                if 'mcp_servers' in filtered_metadata:
                    for server_data in filtered_metadata['mcp_servers']:
                        if 'server_info' in server_data:
                            server_info = server_data['server_info']
                            # Remove file_path and tools to reduce metadata size
                            if 'file_path' in server_info:
                                del server_info['file_path']
                            if 'tools' in server_info:
                                del server_info['tools']
                            if 'tools_count' in server_info:
                                del server_info['tools_count']

                
                # Prepare the result structure with enhanced metadata
                if mode == "multi_server":
                    result = {
                        "server_analysis": parsed_response["server_analysis"],
                        "cross_server_workflow": parsed_response["cross_server_workflow"],
                        "target_tools": parsed_response["target_tools"],
                        "question": parsed_response["question"],
                        "metadata": {
                            **filtered_metadata,
                            "server_count": get_server_count(filtered_metadata)
                        }
                    }
                else:
                    result = {
                        "server_analysis": parsed_response["server_analysis"],
                        "cross_server_workflow": None,
                        "target_tools": parsed_response["target_tools"],
                        "question": parsed_response["question"],
                        "metadata": {
                            **filtered_metadata,
                            "server_count": get_server_count(filtered_metadata)
                        }
                    }
                
                # Clean unusual line terminators
                result = clean_json_object(result)
                
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error processing line: {e}")
                continue
    
    print(f"Finished extracting questions. Total processed: {total_processed}, Successfully parsed: {successfully_parsed}")
    print(f"Mode distribution: {mode_counts}")
    print(f"Output saved to {output_file}")
    
    # Create preview if requested
    if preview_file:
        create_preview_json(output_file, preview_file)

def get_server_count(metadata):
    """
    Get the number of servers involved based on the metadata.
    Both single and multi-server modes now use unified mcp_servers structure.
    """
    # Both modes now use mcp_servers (unified structure)
    mcp_servers = metadata.get("mcp_servers", [])
    return len(mcp_servers) if isinstance(mcp_servers, list) else 0

def sanitize_questions(input_file, sanitized_output, distance_output, preview_distance_file, preview_sanitized_file, sentence_model, encoding_batch_size, distance_threshold, search_space_size, search_batch_size, device):
    """
    Sanitize questions by removing duplicates based on semantic similarity.
    """
    ################
    # Step 1 - Load and Prepare the Dataset
    ################
    print(f"Loading dataset from {input_file}...")
    dataset_items = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            dataset_items.append(json.loads(line))
    
    questions = [item["question"] for item in dataset_items]
    print(f"Number of questions: {len(questions)}")

    ################
    # Step 2 - Encode Questions
    ################
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer(sentence_model)
    model.to(device)

    print("Encoding questions into embeddings...")
    embeddings = model.encode(questions, batch_size=encoding_batch_size, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    ################
    # Step 3 - Build Faiss Index (CPU only)
    ################
    print("Building Faiss index (CPU only)...")
    dimension = embeddings.shape[1]
    # Ensure the index is on CPU
    faiss_index = faiss.IndexFlatL2(dimension)  # L2 distance
    faiss_index.add(embeddings)
    print(f"Faiss index has {faiss_index.ntotal} vectors.")

    ################
    # Step 4 - Find Similar Questions
    ################
    print("Searching for similar questions...")
    batch_size = search_batch_size
    k = search_space_size + 1  # +1 to include the question itself
    similar_indices = []
    similar_scores = []

    for i in tqdm(range(0, len(embeddings), batch_size), desc="Searching Batches"):
        end = min(i + batch_size, len(embeddings))
        batch = embeddings[i:end]
        scores, indices = faiss_index.search(batch, k)
        similar_indices.append(indices)
        similar_scores.append(scores)

    similar_indices = np.vstack(similar_indices)
    similar_scores = np.vstack(similar_scores)

    ################
    # Step 5 - Apply Threshold and Update Dataset
    ################
    print("Applying similarity threshold...")
    
    # Get row IDs if available
    row_ids = []
    for item in dataset_items:
        metadata = item.get("metadata", {})
        if isinstance(metadata, dict):
            row_ids.append(metadata.get("row_id", None))
        else:
            row_ids.append(None)

    min_distances = {}
    duplicate_counts = {}
    min_similar_row_ids = {}

    for idx in tqdm(range(len(questions)), desc="Processing Questions"):
        similar = similar_indices[idx]
        scores = similar_scores[idx]
        
        # Find similar questions (excluding self)
        self_idx = np.where(similar == idx)[0][0]
        similar_filtered = np.delete(similar, self_idx)
        scores_filtered = np.delete(scores, self_idx)
        
        # Count duplicates and find minimum distance
        filtered_indices = [index for index, score in zip(similar_filtered, scores_filtered) 
                          if score < distance_threshold]
        
        duplicate_count = len(filtered_indices)
        min_distance = float(scores_filtered[0]) if len(scores_filtered) > 0 else float('inf')
        
        min_distances[idx] = min_distance
        duplicate_counts[idx] = duplicate_count
        
        if len(scores_filtered) > 0:
            min_similar_row_ids[idx] = row_ids[similar_filtered[np.argmin(scores_filtered)]]
        else:
            min_similar_row_ids[idx] = row_ids[idx]

    ################
    # Step 6 - Save Sanitized Questions
    ################
    print("Saving sanitized questions...")
    
    # Prepare data for output
    server_analyses = [item.get("server_analysis", "") for item in dataset_items]
    target_tools = [item["target_tools"] for item in dataset_items]
    metadata_list = [item.get("metadata", {}) for item in dataset_items]
    
    # Handle cross_server_workflow field (may not exist for all entries)
    cross_server_workflows = [item.get("cross_server_workflow", "") for item in dataset_items]
    
    # Save all questions
    total_rows = 0
    with open(distance_output, 'w', encoding='utf-8') as f_out:
        for idx in tqdm(range(len(questions)), desc="Preparing all entries"):
            # Filter metadata to only include servers that provide the target tools
            filtered_metadata = filter_metadata_by_target_tools(metadata_list[idx], target_tools[idx])
            
            entry = {
                "target_tools": target_tools[idx],
                "question": questions[idx],
                "metadata": filtered_metadata,
                "min_distance": min_distances[idx],
                "duplicate_count": duplicate_counts[idx],
                "min_similar_row_id": min_similar_row_ids[idx]
            }
            
            # Add server_analysis
            if isinstance(server_analyses, list) and idx < len(server_analyses) and server_analyses[idx]:
                entry["server_analysis"] = server_analyses[idx]
            
            # Add cross_server_workflow if it exists
            if isinstance(cross_server_workflows, list) and idx < len(cross_server_workflows):
                workflow = cross_server_workflows[idx]
                if workflow:  # Only add if not empty
                    entry["cross_server_workflow"] = workflow
            
            # Clean unusual line terminators
            entry = clean_json_object(entry)
            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
            total_rows += 1
    print(f"Wrote {total_rows} rows to {distance_output}")

    # Save filtered questions (removing duplicates)
    filtered_rows = 0
    with open(sanitized_output, 'w', encoding='utf-8') as f_out:
        for idx in tqdm(range(len(questions)), desc="Preparing filtered entries"):
            # Filter conditions: min_distance > threshold OR is self-similar
            if min_distances[idx] > distance_threshold or min_similar_row_ids[idx] == row_ids[idx]:
                # Filter metadata to only include servers that provide the target tools
                filtered_metadata = filter_metadata_by_target_tools(metadata_list[idx], target_tools[idx])
                
                entry = {
                    "target_tools": target_tools[idx],
                    "question": questions[idx],
                    "metadata": filtered_metadata,
                    "min_distance": min_distances[idx],
                    "duplicate_count": duplicate_counts[idx],
                    "min_similar_row_id": min_similar_row_ids[idx]
                }
                
                # Add server_analysis
                if isinstance(server_analyses, list) and idx < len(server_analyses) and server_analyses[idx]:
                    entry["server_analysis"] = server_analyses[idx]
                
                # Add cross_server_workflow if it exists
                if isinstance(cross_server_workflows, list) and idx < len(cross_server_workflows):
                    workflow = cross_server_workflows[idx]
                    if workflow:  # Only add if not empty
                        entry["cross_server_workflow"] = workflow
                
                # Clean unusual line terminators
                entry = clean_json_object(entry)
                f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                filtered_rows += 1
    print(f"Wrote {filtered_rows} rows to {sanitized_output}")

    print(f"Sanitized questions saved to {sanitized_output}")
    print("Sanitization process completed.")
    
    # Create previews
    if preview_distance_file:
        create_preview_json(distance_output, preview_distance_file)
    if preview_sanitized_file:
        create_preview_json(sanitized_output, preview_sanitized_file)



def prepare_questions(input_file, output_file):
    """
    Prepare questions for final use by creating the proper message format.
    Enhanced to handle both single-server and multi-server modes.
    """
    print(f"Preparing questions from {input_file}")
    stats = {
        "total_questions": 0,
        "single_server": 0,
        "multi_server": 0,
        "server_count_distribution": {},
        "allocation_strategies": {}
    }
    
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as outf:
        for line in tqdm(f, desc="Preparing Questions"):
            data = json.loads(line)
            metadata = data["metadata"]
            
            # Filter metadata to only include servers that provide the target tools
            filtered_metadata = filter_metadata_by_target_tools(metadata, data["target_tools"])
            
            # Collect statistics
            stats["total_questions"] += 1
            mode = filtered_metadata.get("question_gen_args", {}).get("mode", "unknown")
            if mode == "unknown":
                raise ValueError(f"Mode is unknown in metadata: {filtered_metadata}")
            if mode == "single_server":
                stats["single_server"] += 1
            elif mode == "multi_server":
                stats["multi_server"] += 1
                
                # Track server count distribution
                server_count = filtered_metadata.get("server_count", 0)
                stats["server_count_distribution"][str(server_count)] = stats["server_count_distribution"].get(str(server_count), 0) + 1
            
            # Prepare question content with optional tool hint
            # data["target_tools"] contains the output from extract_xml_tools() function
            question_content = data["question"]
            if args.enable_tool_hint and data["target_tools"]:
                question_content = f"{data['question']}\n\nYou need to solve this question using {data['target_tools']} tool from the list of available tools."
            
            result = {
                "messages": [
                    {
                        "role": "user", 
                        "content": question_content
                    }
                ],
                "metadata": {
                    **filtered_metadata,
                    "target_tools": data["target_tools"],
                    "question": data["question"],
                    "min_distance": data.get("min_distance", None),
                    "duplicate_count": data.get("duplicate_count", 0),
                    "min_similar_row_id": data.get("min_similar_row_id", None)
                }
            }
            
            # Add server_analysis
            if "server_analysis" in data and data["server_analysis"]:
                result["metadata"]["server_analysis"] = data["server_analysis"]
            
            # Add cross_server_workflow to metadata if it exists
            if "cross_server_workflow" in data and data["cross_server_workflow"]:
                result["metadata"]["cross_server_workflow"] = data["cross_server_workflow"]
            
            # Clean unusual line terminators
            result = clean_json_object(result)
            outf.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Save statistics
    stats_file = output_file.replace('.jsonl', '_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as stats_outf:
        json.dump(stats, stats_outf, ensure_ascii=False, indent=2)
    
    print(f"Finished preparing questions. Output saved to {output_file}")
    print(f"Statistics saved to {stats_file}")
    print_processing_summary(stats)

def print_processing_summary(stats):
    """
    Print a summary of the processing results for both single and multi-server modes.
    """
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Total Questions Processed: {stats['total_questions']}")
    print(f"Single-Server Questions: {stats['single_server']}")
    print(f"Multi-Server Questions: {stats['multi_server']}")
    
    if stats['multi_server'] > 0:
        print("\nMulti-Server Statistics:")
        print("-" * 30)
        
        # Server count distribution
        if stats['server_count_distribution']:
            print("Server Count Distribution:")
            for count, freq in sorted(stats['server_count_distribution'].items()):
                percentage = (freq / stats['multi_server']) * 100
                print(f"  {count} servers: {freq} questions ({percentage:.1f}%)")
    
    print("="*60 + "\n")

def main():
    print(f"Tool Use Question Processing Pipeline. Arguments: {args}")

    input_dir = os.path.dirname(args.input_file)
    input_basename = os.path.basename(args.input_file)
    print(f"Input directory: {input_dir}")
    print(f"Input basename: {input_basename}")
    
    # Create output file / folder
    output_path = f"{input_dir}/processed"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Define output paths
    base_name = input_basename.replace('_results.jsonl', '')
    extracted_output = f"{output_path}/{base_name}_1extracted.jsonl"
    extracted_output_review = f"{output_path}/preview_{base_name}_1extracted.json"
    distance_output = f"{output_path}/{base_name}_2distance.jsonl"
    distance_output_review = f"{output_path}/preview_{base_name}_2distance.json"
    sanitized_output = f"{output_path}/{base_name}_3sanitized.jsonl"
    sanitized_output_review = f"{output_path}/preview_{base_name}_3sanitized.json"
    prepared_output = f"{output_path}/{base_name}_4prepared.jsonl"
    prepared_output_review = f"{output_path}/preview_{base_name}_4prepared.json"

    # Run all steps in sequence
    print("Step 1: Extracting questions from XML responses...")
    extract_questions(args.input_file, extracted_output, extracted_output_review)
    
    # Step 2: Sanitization (optional)
    if not args.disable_sanitize:
        print("Step 2: Sanitizing questions (removing duplicates)...")
        sanitize_questions(
            input_file=extracted_output,
            sanitized_output=sanitized_output,
            distance_output=distance_output,
            preview_distance_file=distance_output_review,
            preview_sanitized_file=sanitized_output_review,
            sentence_model=args.sentence_model,
            encoding_batch_size=args.encoding_batch_size,
            distance_threshold=args.distance_threshold,
            search_space_size=args.search_space_size,
            search_batch_size=args.search_batch_size,
            device=args.device
        )
        input_for_prepare = sanitized_output
    else:
        print("Sanitize step is disabled. Using extracted questions directly.")
        # Copy extracted output to sanitized output to maintain consistent file structure
        shutil.copyfile(extracted_output, sanitized_output)
        # Create preview for sanitized output (copied from extracted)
        create_preview_json(sanitized_output, sanitized_output_review)
        input_for_prepare = sanitized_output
    
    # Step 3: Preparation (optional)
    if not args.disable_prepare:
        print("Step 3: Preparing questions for final use...")
        if args.enable_tool_hint:
            print("Tool hints enabled: Appending 'You need to solve this question using {target_tools}.' to questions.")
        prepare_questions(input_for_prepare, prepared_output)
        # Create preview for prepared output
        create_preview_json(prepared_output, prepared_output_review)
        print(f"Final output saved to: {prepared_output}")
    else:
        print("Prepare step is disabled. Stopping after sanitization.")
        print(f"Final output saved to: {input_for_prepare}")


if __name__ == "__main__":
    main()