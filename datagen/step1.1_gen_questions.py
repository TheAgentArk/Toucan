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
from utils import parse_usage_number

################
# Use Cases
################
"""
This script generates questions for tool use scenarios with different sampling strategies.

Example Usage:

1. Single Server Mode (tools from one server per question):
   1.1. Random sampling (default) - 1000 questions from random servers:
        python step1.1_gen_questions.py --total_prompts 1000 --sampling_strategy random
   
   1.2. Uniform sampling - 5 questions per server (total = 5 * num_servers):
        python step1.1_gen_questions.py --sampling_strategy uniform --samples_per_server 5
   
   1.3. Power law sampling - favors popular servers based on usage rank:
        python step1.1_gen_questions.py --total_prompts 2000 --sampling_strategy power_law
   
   1.4. Featured sampling - only select from featured servers:
        python step1.1_gen_questions.py --total_prompts 1000 --sampling_strategy featured
   
   1.5. Generate questions for 2-tool scenarios with uniform sampling:
        python step1.1_gen_questions.py --num_tools 2 --sampling_strategy uniform --samples_per_server 10

2. Multi-Server Mode (tools from multiple servers per question):
   2.1. Random server combinations:
        python step1.1_gen_questions.py --mode multi_server --num_tools 2 --total_prompts 1000
   
   2.2. Servers from same category (same primary label):
        python step1.1_gen_questions.py --mode multi_server --num_tools 3 --multi_server_allocation_strategy same_primary_label
   
   2.3. Servers from different categories (different primary labels):
        python step1.1_gen_questions.py --mode multi_server --num_tools 3 --multi_server_allocation_strategy different_primary_labels
   
   2.4. LLM-driven server selection from featured servers (brainstorm context first):
        python step1.1_gen_questions.py --mode multi_server --num_tools 3 --multi_server_allocation_strategy random_featured
   
   2.5. Multi-server with custom server allocation:
        python step1.1_gen_questions.py --mode multi_server --num_tools 4 --multi_server_allocated_servers 3 --total_prompts 500

Key Parameters:
- --sampling_strategy: Controls server selection (random/uniform/power_law/featured) - ONLY for single_server mode
- --mode: single_server (individual servers) vs multi_server (server combinations)  
- --multi_server_allocation_strategy: For multi_server mode (random/same_primary_label/different_primary_labels/random_featured)
- --num_tools: Number of tools to include in each prompt
- --samples_per_server: For uniform sampling, questions per server
"""

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Tool Use Question Generation Manager.")
    # Generation Parameters
    parser.add_argument("--total_prompts", type=int, default=1000, help="Total number of prompts to generate.")
    parser.add_argument('--source', default="smithery", type=str, choices=["smithery"], help="Source for question generation. Currently only supports smithery tools.")
    parser.add_argument('--num_tools', default=1, type=int, help="Number of tools to use for question generation.")
    parser.add_argument('--mode', default="single_server", type=str, choices=["single_server", "multi_server"], help="Mode for question generation: single_server (tools from one server) or multi_server (tools from multiple servers).")
    # Multi-server specific parameters
    parser.add_argument('--multi_server_allocated_servers', default=None, type=int, help="Number of servers to allocate for multi_server mode. If not specified, uses min(num_tools, available_servers) with minimum of 2.")
    parser.add_argument('--multi_server_allocation_strategy', default="random", type=str, choices=["random", "same_primary_label", "different_primary_labels", "random_featured"], help="Strategy for selecting servers in multi_server mode: random (default), same_primary_label, different_primary_labels, or random_featured (LLM chooses from all featured servers).")
    parser.add_argument('--max_featured_servers_per_prompt', default=None, type=int, help="Maximum number of featured servers to include per prompt when using random_featured strategy. If not specified, uses all featured servers.")
    # Sampling Strategy Settings
    parser.add_argument('--sampling_strategy', default="random", type=str, choices=["random", "uniform", "power_law", "featured"], help="Sampling strategy for server selection: random (default), uniform (equal samples per server), power_law (based on usage/rank), or featured (only from featured servers).")
    parser.add_argument('--samples_per_server', type=int, default=5, help="Number of samples to generate per MCP server for uniform sampling strategy.")
    parser.add_argument('--power_law_alpha', type=float, default=0.5, help="Alpha parameter for power law sampling. Higher values favor more popular servers.")
    # System Settings
    parser.add_argument("--output_folder", type=str, default="../data")
    parser.add_argument("--job_name", type=str, default=None, help="Job Name.")
    parser.add_argument("--timestamp", type=int, default=int(time.time()), help="Timestamp for the job. Also used as the random seed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()

def get_seed_prompt(input, source, num_tools, mode="single_server", allocation_strategy="random"):
    # Load prompt template from markdown file
    env = Environment(loader=FileSystemLoader('prompts'))
    
    if source == "smithery":
        if mode == "single_server":
            if num_tools == 1:
                template = env.get_template('genq_from_tools_single_server_single_tool.md').render()
            else:
                template = env.get_template('genq_from_tools_single_server_multi_tools.md').render()
            
            mcp_data = input[0]
            
            # Extract server info from the nested structure
            server_info = mcp_data.get('metadata', {}).get('server_info_crawled', {})
            remote_response = mcp_data.get('metadata', {}).get('remote_server_response', {})
            
            # Validate required fields (safety check)
            if not remote_response.get('tools') or len(remote_response['tools']) == 0:
                raise ValueError(f"MCP server {server_info.get('name', 'Unknown')} has no tools")
                
            # Replace placeholders in template
            template = template.replace("{MCP_SERVER_NAME}", server_info.get('name', 'Unknown Server'))
            template = template.replace("{MCP_SERVER_DESCRIPTION}", server_info.get('overview', 'No description available'))

            if num_tools != 1:
                template = template.replace("{NUM_TOOLS}", str(num_tools))
            
            # Create tool list from remote_server_response
            tool_list = ""
            for i, tool in enumerate(remote_response['tools'], 1):
                tool_name = tool.get('name', 'Unknown Tool')
                tool_desc = tool.get('description', 'No description available')
                tool_list += f"{i}. **{tool_name}**: {tool_desc}\n"
            
            template = template.replace("{TOOL_LIST}", tool_list)
            
        elif mode == "multi_server":
            if allocation_strategy == "random_featured":
                template = env.get_template('genq_from_tools_multi_server_random_featured.md').render()
            else:
                template = env.get_template('genq_from_tools_multi_server_random.md').render()
            
            # Replace NUM_TOOLS placeholder
            template = template.replace("{NUM_TOOLS}", str(num_tools))
            
            # Create server descriptions section
            server_descriptions = ""
            placeholder_name = "{FEATURED_SERVER_DESCRIPTIONS}" if allocation_strategy == "random_featured" else "{SERVER_DESCRIPTIONS}"
            
            for i, mcp_data in enumerate(input, 1):
                # Extract server info from the nested structure
                server_info = mcp_data.get('metadata', {}).get('server_info_crawled', {})
                remote_response = mcp_data.get('metadata', {}).get('remote_server_response', {})
                
                # Validate required fields (safety check)
                if not remote_response.get('tools') or len(remote_response['tools']) == 0:
                    raise ValueError(f"MCP server {server_info.get('name', 'Unknown')} has no tools")
                
                server_name = server_info.get('name', f'Unknown Server {i}')
                server_desc = server_info.get('overview', 'No description available')
                
                server_descriptions += f"### {server_name}\n"
                server_descriptions += f"**Description**: {server_desc}\n\n"
                server_descriptions += f"**Available Tools**:\n"
                
                # Create tool list for this server
                for j, tool in enumerate(remote_response['tools'], 1):
                    tool_name = tool.get('name', 'Unknown Tool')
                    tool_desc = tool.get('description', 'No description available')
                    server_descriptions += f"{j}. **{tool_name}**: {tool_desc}\n"
                
                server_descriptions += "\n"
            
            template = template.replace(placeholder_name, server_descriptions)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    else:
        raise ValueError(f"Unsupported source: {source}")
    
    return template

def create_power_law_weights(valid_mcp_servers, alpha=0.5):
    """
    Create power law weights for server sampling based on usage rank.
    Lower rank numbers get higher weights (rank 1 is most popular).
    """
    weights = []
    
    for server in valid_mcp_servers:
        # Try to get rank_by_usage first, then fallback to usage numbers
        rank = server.get('metadata', {}).get('rank_by_usage')
        
        if rank is not None:
            # Use rank directly (lower rank = higher weight)
            try:
                rank_val = int(rank)
                # Power law: weight = 1 / rank^alpha
                weight = 1.0 / (rank_val ** alpha)
            except (ValueError, TypeError):
                weight = 1.0
        else:
            # Fallback: try to extract usage from server info and create rank
            usage = server.get('metadata', {}).get('server_info_crawled', {}).get('usage', 1)
            usage_val = parse_usage_number(usage, default_value=1)
            # Convert usage to weight (higher usage = higher weight)
            weight = usage_val ** alpha
        
        weights.append(weight)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    print(f"Number of servers: {len(weights)}")
    print(f"Weights for power law sampling: {weights}")

    return weights

def sample_servers_by_strategy(valid_mcp_servers, total_samples, strategy="random", samples_per_server=5):
    """
    Sample servers based on the specified strategy.
    Returns a list of selected servers for the total number of samples.
    """
    if strategy == "random":
        # Random sampling (current default behavior)
        return [random.choice(valid_mcp_servers) for _ in range(total_samples)]
    
    elif strategy == "uniform":
        # Uniform sampling: equal samples per server
        sampled_servers = []
        for server in valid_mcp_servers:
            sampled_servers.extend([server] * samples_per_server)
        
        # If we have more samples than needed, randomly shuffle and truncate
        if len(sampled_servers) > total_samples:
            random.shuffle(sampled_servers)
            sampled_servers = sampled_servers[:total_samples]
        # If we need more samples, cycle through servers
        elif len(sampled_servers) < total_samples:
            remaining = total_samples - len(sampled_servers)
            additional_samples = []
            for i in range(remaining):
                server_idx = i % len(valid_mcp_servers)
                additional_samples.append(valid_mcp_servers[server_idx])
            sampled_servers.extend(additional_samples)
        
        return sampled_servers
    
    elif strategy == "power_law":
        # Power law sampling based on usage/rank
        weights = create_power_law_weights(valid_mcp_servers, alpha=args.power_law_alpha)
        # Sample according to weights
        sampled_servers = []
        for _ in range(total_samples):
            # Weighted random sampling
            selected_server = np.random.choice(valid_mcp_servers, p=weights)
            sampled_servers.append(selected_server)
        
        return sampled_servers
    
    elif strategy == "featured":
        # Featured sampling: only select from servers with featured_server: true
        featured_servers = [
            server for server in valid_mcp_servers 
            if server.get('labels', {}).get('featured_server') == True
        ]
        
        if len(featured_servers) == 0:
            raise ValueError("No featured servers available for featured sampling strategy")
        
        # Random sampling from featured servers only
        sampled_servers = [random.choice(featured_servers) for _ in range(total_samples)]
        return sampled_servers
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def select_servers_for_multi_server(valid_mcp_servers, num_tools, allocated_servers=None, strategy="random", args=None):
    """
    Select servers for multi_server mode based on allocation strategy.
    
    Args:
        valid_mcp_servers: List of available MCP servers
        num_tools: Number of tools required  
        allocated_servers: Number of servers to allocate (if None, uses default logic)
        strategy: "random" or "same_primary_label"
    
    Returns:
        List of selected servers
    """
    # Determine number of servers to select
    if allocated_servers is not None:
        num_servers = min(allocated_servers, len(valid_mcp_servers))
        num_servers = max(2, num_servers)  # Ensure at least 2 servers for multi_server
    else:
        # Default logic: min(num_tools, available_servers) with minimum of 2
        num_servers = min(num_tools, len(valid_mcp_servers))
        num_servers = max(2, num_servers)
    
    if strategy == "random":
        return random.sample(valid_mcp_servers, num_servers)
    
    elif strategy == "same_primary_label":
        # First, randomly select one server to get its primary_label
        seed_server = random.choice(valid_mcp_servers)
        seed_primary_label = seed_server.get('labels', {}).get('primary_label')
        
        if not seed_primary_label:
            print(f"Warning: Seed server has no primary_label, falling back to random selection")
            return random.sample(valid_mcp_servers, num_servers)
        
        # Filter servers with the same primary_label
        servers_with_same_label = [
            server for server in valid_mcp_servers 
            if server.get('labels', {}).get('primary_label') == seed_primary_label
        ]
        
        if len(servers_with_same_label) < 2:
            print(f"Warning: Only {len(servers_with_same_label)} servers found with primary_label '{seed_primary_label}', falling back to random selection")
            return random.sample(valid_mcp_servers, num_servers)
        
        # Select the required number of servers from those with the same label
        actual_num_servers = min(num_servers, len(servers_with_same_label))
        selected_servers = random.sample(servers_with_same_label, actual_num_servers)
        
        print(f"Selected {len(selected_servers)} servers with primary_label '{seed_primary_label}'")
        return selected_servers
    
    elif strategy == "different_primary_labels":
        # Select servers from different primary_labels
        # Group servers by primary_label
        servers_by_label = {}
        servers_without_label = []
        
        for server in valid_mcp_servers:
            primary_label = server.get('labels', {}).get('primary_label')
            if primary_label:
                if primary_label not in servers_by_label:
                    servers_by_label[primary_label] = []
                servers_by_label[primary_label].append(server)
            else:
                servers_without_label.append(server)
        
        # Check if we have enough different categories
        available_categories = len(servers_by_label)
        if available_categories < 2:
            print(f"Warning: Only {available_categories} different primary_labels found, falling back to random selection")
            return random.sample(valid_mcp_servers, num_servers)
        
        if available_categories < num_servers:
            print(f"Warning: Only {available_categories} different primary_labels available, but {num_servers} servers requested. Will select {available_categories} servers from different categories.")
            actual_num_servers = available_categories
        else:
            actual_num_servers = num_servers
        
        # Randomly select categories and then one server from each category
        selected_categories = random.sample(list(servers_by_label.keys()), actual_num_servers)
        selected_servers = []
        
        for category in selected_categories:
            # Randomly select one server from this category
            selected_server = random.choice(servers_by_label[category])
            selected_servers.append(selected_server)
        
        print(f"Selected {len(selected_servers)} servers from different primary_labels: {selected_categories}")
        return selected_servers
    
    elif strategy == "random_featured":
        # For random_featured, return all servers with featured_server: true label
        featured_servers = []
        
        # Filter servers that have featured_server: true in their labels
        for server in valid_mcp_servers:
            if server.get('labels', {}).get('featured_server') == True:
                featured_servers.append(server)
        
        # Shuffle the featured servers for randomness
        random.shuffle(featured_servers)
        
        # Limit to max_featured_servers_per_prompt if specified
        if args and args.max_featured_servers_per_prompt is not None:
            featured_servers = random.sample(featured_servers, min(args.max_featured_servers_per_prompt, len(featured_servers)))
        
        print(f"Selected {len(featured_servers)} featured servers for random_featured strategy")
        return featured_servers
    
    else:
        raise ValueError(f"Unknown allocation strategy: {strategy}")

args = get_args()

# Validation for multi_server mode
if args.mode == "multi_server" and args.num_tools < 2:
    raise ValueError("multi_server mode requires num_tools >= 2")

# Validation for multi_server-specific strategies
if args.multi_server_allocation_strategy in ["same_primary_label", "different_primary_labels", "random_featured"] and args.mode != "multi_server":
    raise ValueError(f"{args.multi_server_allocation_strategy} strategy can only be used with multi_server mode")

print(f"Tool Use Question Generation Manager.\nArguments:\n{args}") # For logging

#################
# System Settings
#################

# Set random seed
if args.seed is not None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

#################
# Load Tool Data
#################
if args.source == "smithery":
    # Load all smithery MCP server files
    smithery_files = glob.glob('../mcp_servers/*.json')
    valid_mcp_servers = []
    
    for file_path in smithery_files:
        try:
            with open(file_path, 'r') as f:
                mcp_data = json.load(f)

            # Extract nested data for validation
            remote_response = mcp_data.get('metadata', {}).get('remote_server_response', {})
            server_info = mcp_data.get('metadata', {}).get('server_info_crawled', {})
            server_labels = mcp_data.get('labels', {})

            # Check if the remote server response is success
            if server_labels.get('is_connected') == False:
                print(f"Skipping {file_path}: remote server response is not successful.")
                continue
                
            # Check if tools exist and are valid
            if not remote_response.get('tools') or len(remote_response['tools']) == 0:
                print(f"Skipping {file_path}: empty or missing 'tools' list.")
                continue

            # Check if the remote tool is valid
            print(f"is_remote_tool_valid: {server_labels.get('is_remote_tool_valid')}")
            if server_labels.get('is_remote_tool_valid') == False:
                print(f"Skipping {file_path}: remote tool is not valid.")
                continue
                
            # Only include Remote MCP servers
            if server_info.get('remote_or_local') != 'Remote':
                print(f"Skipping {file_path}: not a Remote MCP server (got: {server_info.get('remote_or_local')}).")
                continue
                
            # Validate required fields
            if not server_info.get('name') or not server_info.get('overview'):
                print(f"Skipping {file_path}: missing required fields (name/overview).")
                continue

            # Check if the number of tools is valid
            if len(remote_response['tools']) < args.num_tools and args.mode == "single_server":
                print(f"Skipping {file_path}: not enough tools (got: {len(remote_response['tools'])}).")
                continue
            
            print(f"Loaded MCP server '{server_info.get('name')}' with {len(remote_response['tools'])} tools. Required number of tools: {args.num_tools}")

            # Add file path to metadata for reference
            mcp_data['metadata']['source_file_path'] = file_path
            valid_mcp_servers.append(mcp_data)
            
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error loading {file_path}: {e}")
            continue
else:
    raise ValueError(f"Unsupported source: {args.source}")

print(f"Number of MCP servers loaded: {len(valid_mcp_servers)}")

# Additional validation for label-based and featured strategies
if args.multi_server_allocation_strategy in ["same_primary_label", "different_primary_labels", "random_featured"]:
    if args.multi_server_allocation_strategy in ["same_primary_label", "different_primary_labels"]:
        servers_with_labels = [
            server for server in valid_mcp_servers 
            if server.get('labels', {}).get('primary_label')
        ]
        if len(servers_with_labels) < 2:
            raise ValueError(f"{args.multi_server_allocation_strategy} strategy requires at least 2 servers with labels, but only {len(servers_with_labels)} found. Available servers: {len(valid_mcp_servers)}")
        
        if args.multi_server_allocation_strategy == "same_primary_label":
            print(f"Found {len(servers_with_labels)} servers with primary_label for same_primary_label strategy")
        elif args.multi_server_allocation_strategy == "different_primary_labels":
            # Count unique categories
            unique_labels = set()
            for server in servers_with_labels:
                primary_label = server.get('labels', {}).get('primary_label')
                if primary_label:
                    unique_labels.add(primary_label)
            
            if len(unique_labels) < 2:
                raise ValueError(f"different_primary_labels strategy requires at least 2 different primary_labels, but only {len(unique_labels)} found. Available categories: {list(unique_labels)}")
            print(f"Found {len(unique_labels)} different primary_labels for different_primary_labels strategy: {list(unique_labels)}")
    
    elif args.multi_server_allocation_strategy == "random_featured":
        # Validate that we have enough servers with featured_server: true
        featured_servers = [
            server for server in valid_mcp_servers 
            if server.get('labels', {}).get('featured_server') == True
        ]
        
        if len(featured_servers) < 2:
            raise ValueError(f"random_featured strategy requires at least 2 servers with featured_server: true, but only {len(featured_servers)} found. Available servers: {len(valid_mcp_servers)}")
        
        print(f"Found {len(featured_servers)} featured servers for random_featured strategy")

# Additional validation for featured sampling strategy
if args.sampling_strategy == "featured":
    featured_servers = [
        server for server in valid_mcp_servers 
        if server.get('labels', {}).get('featured_server') == True
    ]
    
    if len(featured_servers) == 0:
        raise ValueError(f"featured sampling strategy requires at least 1 server with featured_server: true, but none found. Available servers: {len(valid_mcp_servers)}")
    
    print(f"Found {len(featured_servers)} featured servers for featured sampling strategy")

# Print sampling strategy information
print(f"Using sampling strategy: {args.sampling_strategy}")
if args.sampling_strategy == "uniform":
    print(f"Uniform sampling: {args.samples_per_server} samples per server")
elif args.sampling_strategy == "power_law":
    print("Power law sampling: servers will be sampled based on usage rank")
    # Show some example weights for information
    weights = create_power_law_weights(valid_mcp_servers[:10], alpha=1.0)  # Show first 10 for brevity
    print("Sample power law weights (first 10 servers):")
    for i, (server, weight) in enumerate(zip(valid_mcp_servers[:10], weights)):
        server_name = server.get('metadata', {}).get('server_name', f'Server {i+1}')
        rank = server.get('metadata', {}).get('rank_by_usage', 'Unknown')
        print(f"  {server_name} (rank: {rank}): weight = {weight:.4f}")
elif args.sampling_strategy == "random":
    print("Random sampling: servers will be selected uniformly at random")
elif args.sampling_strategy == "featured":
    print("Featured sampling: servers will be selected only from featured servers")

#################
# Create output file / folder
#################
if args.sampling_strategy == "uniform":
    # override total_prompts for uniform sampling
    args.total_prompts = args.samples_per_server * len(valid_mcp_servers)

mode_suffix = "" if args.mode == "single_server" else f"_{args.mode}"
output_filename = f"ToolUse_s2q_{args.source}_{args.total_prompts}_{args.num_tools}tool{mode_suffix}_{args.timestamp}_prepared.jsonl"
output_foldername = f"ToolUse_{args.source}_{args.total_prompts}_{args.num_tools}tool{mode_suffix}_{args.timestamp}"
if not args.job_name:
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not os.path.exists(f"{args.output_folder}/{output_foldername}"):
        os.makedirs(f"{args.output_folder}/{output_foldername}")
    output_dir = f"{args.output_folder}/{output_foldername}/{output_filename}"
else:
    output_dir = f"{args.output_folder}/{args.job_name}/{output_filename}"

# Save argparse arguments to the output folder
if not args.job_name:
    args_output_dir = f"{args.output_folder}/{output_foldername}"
else:
    args_output_dir = f"{args.output_folder}/{args.job_name}"

args_file_path = f"{args_output_dir}/generation_args.json"
args_dict = vars(args)
with open(args_file_path, "w") as f:
    json.dump(args_dict, f, indent=2)
print(f"Arguments saved to: {args_file_path}")

################
# Generate outputs
################
results = []

# Generate server samples based on sampling strategy
if args.mode == "single_server":
    # Sample individual servers
    sampled_servers = sample_servers_by_strategy(
        valid_mcp_servers, 
        args.total_prompts, 
        args.sampling_strategy, 
        args.samples_per_server
    )
    
    pbar = tqdm(total=len(sampled_servers))
    for i, selected_mcp_server in enumerate(sampled_servers):
        seed_prompt = get_seed_prompt([selected_mcp_server], args.source, args.num_tools, args.mode, args.sampling_strategy)

        # Extract server info and metadata from the labeled file
        server_metadata = selected_mcp_server.get('metadata', {})
        server_info = server_metadata.get('server_info_crawled', {})
        remote_response = server_metadata.get('remote_server_response', {})

        # Save outputs with complete metadata      
        result = {
            "messages": [
                {
                    "role": "user",
                    "content": seed_prompt
                }
            ],
            "metadata": {
                "prompt_id": f"{i:08d}",
                "row_id": i,
                "mode": args.mode,
                "question_gen_args": args_dict,
                "mcp_servers": [{
                    # Core identifiers
                    "server_id": server_metadata.get('server_id'),
                    "server_name": server_metadata.get('server_name'),
                    "rank_by_usage": server_metadata.get('rank_by_usage'),
                    
                    "server_info": server_info,
                    "remote_server_response": remote_response,
                    "labels": selected_mcp_server.get('labels', {}),
                    
                    # File paths and processing info
                    "original_file": server_metadata.get('original_file', ''),
                    "source_file_path": server_metadata.get('source_file_path', ''),
                    "source_filename": server_metadata.get('source_filename', ''),
                    "processed_timestamp": server_metadata.get('processed_timestamp'),
                    "processing_source": server_metadata.get('processing_source', ''),
                    "rank": server_metadata.get('rank')
                }],
            },
        }
        results.append(result)
        pbar.update(1)
    pbar.close()

else:  # multi_server mode
    pbar = tqdm(total=args.total_prompts)
    for i in range(args.total_prompts):
        # Select multiple servers using the specified allocation strategy
        selected_servers = select_servers_for_multi_server(
            valid_mcp_servers, 
            args.num_tools, 
            args.multi_server_allocated_servers, 
            args.multi_server_allocation_strategy,
            args
        )
        
        seed_prompt = get_seed_prompt(selected_servers, args.source, args.num_tools, args.mode, args.multi_server_allocation_strategy)

        # Save outputs with complete metadata for all servers
        mcp_servers_metadata = []
        for server in selected_servers:
            server_metadata = server.get('metadata', {})
            server_info = server_metadata.get('server_info_crawled', {})
            remote_response = server_metadata.get('remote_server_response', {})
            
            mcp_servers_metadata.append({
                # Core identifiers
                "server_id": server_metadata.get('server_id'),
                "server_name": server_metadata.get('server_name'),
                "rank_by_usage": server_metadata.get('rank_by_usage'),
                
                "server_info": server_info,
                "remote_server_response": remote_response,
                "labels": server.get('labels', {}),
                
                # File paths and processing info
                "original_file": server_metadata.get('original_file', ''),
                "source_file_path": server_metadata.get('source_file_path', ''),
                "source_filename": server_metadata.get('source_filename', ''),
                "processed_timestamp": server_metadata.get('processed_timestamp'),
                "processing_source": server_metadata.get('processing_source', ''),
                "rank": server_metadata.get('rank')
            })

        result = {
            "messages": [
                {
                    "role": "user",
                    "content": seed_prompt
                }
            ],
            "metadata": {
                "prompt_id": f"{i:08d}",
                "row_id": i,
                "mode": args.mode,
                "question_gen_args": args_dict,
                "mcp_servers": mcp_servers_metadata,  # Note: plural for multi_server mode
                "question_gen_args": args_dict
            },
        }
        results.append(result)
        pbar.update(1)
    pbar.close()

# Save the final results
with open(output_dir, "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Finished. Total prompts: {len(results)}")
print(f"Output directory: {output_dir}")