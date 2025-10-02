import torch
import os
import sys
import argparse
import copy
import json
import re
import requests
import concurrent.futures
import multiprocessing
import types
import asyncio
import base64
import threading
import queue
import signal
import atexit
import os
from time import sleep, time
from tqdm import tqdm
from wrapt_timeout_decorator import timeout

from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_model_short_name, validate_api_pool_from_file, check_if_api_key_is_valid, safe_save_checkpoint, get_model_abbreviation


# OpenAI Agent imports
from agents.mcp import MCPServerStreamableHttp
from agents.run_context import RunContextWrapper
from agents import Agent, OpenAIResponsesModel, Runner, SQLiteSession
from openai import AsyncClient

# Check if agents library is installed
try:
    import agents
except ImportError:
    print("agents library is not installed. Please install it.")
    exit(1)

# Global cleanup function for MCP resources
def cleanup_mcp_resources():
    """Clean up MCP resources on exit"""
    # Only cleanup if we're using agent mode
    try:
        # Check if args is available and agent mode is enabled
        if 'args' in globals() and hasattr(args, 'agent') and args.agent:
            # OpenAI Agent framework handles cleanup automatically
            pass
    except Exception as e:
        # print(f"⚠️ Warning: Emergency MCP cleanup failed: {e}")
        pass

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    # print(f"\n🛑 Received signal {signum}. Cleaning up...")
    cleanup_mcp_resources()
    # print("👋 Exiting gracefully.")
    os._exit(0)  # Use os._exit instead of sys.exit to avoid atexit conflicts

# Register cleanup functions
atexit.register(cleanup_mcp_resources)
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination

################
# Configurations
################
def get_args():
    # Experiment Settings
    parser = argparse.ArgumentParser(description="Response Generation Manager.")
    parser.add_argument("--model_path", type=str, default="openai/gpt-oss-120b",
                        help="Model path for inference")
    parser.add_argument("--input_file", type=str, default=None, help="Input dataset file name")
    parser.add_argument("--checkpoint_every", type=int, default=16, help="Save checkpoint every n completed items")
    parser.add_argument("--openrouter_url", type=str, default="https://openrouter.ai/api/v1", help="OpenRouter API URL")
    parser.add_argument("--openrouter_api_key", type=str, default="", help="OpenRouter API Key")
    parser.add_argument("--vllm_api_url", type=str, default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY", help="vLLM API Key")
    parser.add_argument("--smithery_api_key", type=str, default="", help="Smithery API Key")
    parser.add_argument("--smithery_profile", type=str, default="", help="Smithery Profile")
    parser.add_argument("--smithery_api_pool", type=str, default="smithery_api_pool.json", help="Path to Smithery API pool JSON file")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of parallel workers (default: use API pool size)")

    # Generation Parameters
    parser.add_argument('--engine', default="vllm_api", type=str, choices=["vllm_api", "openrouter_api"])
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--step", type=str, default="unknown", help="Processing step identifier.")
    parser.add_argument("--agent", type=str, default="openai_agent", help="Use agent inference for items with MCP server URLs")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for each item processing (default: 90 seconds)")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for each item processing (default: 3)")
    parser.add_argument("--fncall_prompt_type", type=str, default="nous", help="Function call prompt type (default: nous)")
    parser.add_argument("--parallel_function_calls", type=bool, default=True, help="Parallel function calls (default: True)")
    parser.add_argument("--reasoning_effort", type=str, default="high", help="Reasoning effort (default: high)")
    parser.add_argument("--enable_tool_hint", action="store_true", help="Enable tool hint (default: off)")
    parser.add_argument("--enable_irrelevant_warning", action="store_true", help="Enable irrelevant warning (default: off)")
    parser.add_argument("--max_turns", type=int, default=10, help="Maximum number of turns for agent inference (default: 10)")
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
CHECKPOINT_EVERY = args.checkpoint_every

model_abbreviation = get_model_abbreviation(args.model_path)
config_str = f"{model_abbreviation}_{args.reasoning_effort}_pfc" if args.parallel_function_calls else f"{model_abbreviation}_{args.reasoning_effort}_sfc"

base_name = INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]  # Remove "_4prepared"

if args.num_trials > 1:
    checkpoint_files = [
        f"{base_name}_{config_str}_results{i}_checkpoint.json"
        for i in range(args.num_trials)
    ]
    saved_files = [
        f"{base_name}_{config_str}_results{i}.jsonl"
        for i in range(args.num_trials)
    ]
else:
    checkpoint_file = f"{base_name}_{config_str}_results_checkpoint.json"
    saved_file = f"{base_name}_{config_str}_results.jsonl"

# API Setups
if args.engine == "openrouter_api":
    API_ENDPOINT = args.openrouter_url + "/chat/completions"
    API_HEADERS = {
        "Authorization": f"Bearer {args.openrouter_api_key}",
        "Content-Type": "application/json"
    }
    API_PARAMS = {
        "model": args.model_path,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "parallel_tool_calls": args.parallel_function_calls,
        "reasoning": {"effort": args.reasoning_effort},
    }

elif args.engine == "vllm_api":
    API_ENDPOINT = args.vllm_api_url + "/chat/completions"
    API_HEADERS = {
        "Authorization": f"Bearer {args.vllm_api_key}",
        "Content-Type": "application/json"
    }
    API_PARAMS = {
        "model": args.model_path,
        # "max_tokens": args.max_tokens # If a user does not specify a max_tokens in their request, then the minimum of max_new_tokens and (max_model_len - prompt_tokens) will be used.
        "temperature": args.temperature,
        "top_p": args.top_p,
        "parallel_tool_calls": args.parallel_function_calls,
        "reasoning": {"effort": args.reasoning_effort},
    }

# Global API pool variable
smithery_api_pool = None

def load_and_validate_smithery_api_pool(pool_file_path):
    """Load and validate Smithery API pool from JSON file, keeping only valid keys"""
    global smithery_api_pool
    
    print("=" * 50)
    print("🔍 SMITHERY API POOL VALIDATION")
    print("=" * 50)
    
    # Check if pool file exists
    if not os.path.exists(pool_file_path):
        print(f"⚠️  API pool file {pool_file_path} not found!")
        print("🔍 Testing fallback API key from arguments...")
        
        # Validate the fallback API key
        fallback_result = check_if_api_key_is_valid(args.smithery_profile, args.smithery_api_key)
        
        if not fallback_result['valid']:
            raise ValueError(f"❌ Fallback API key is also invalid: {fallback_result['message']}")
        
        print(f"✅ Fallback API key is valid: {fallback_result['message']}")
        smithery_api_pool = [{
            "profile": args.smithery_profile,
            "api_key": args.smithery_api_key,
            "source": "fallback"
        }]
        print(f"✅ Using 1 valid API key (fallback)")
        print("=" * 50)
        return smithery_api_pool
    
    # Validate the entire API pool using the test logic
    print(f"📁 Validating all entries in {pool_file_path}...")
    
    try:
        results = validate_api_pool_from_file(pool_file_path)
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            raise ValueError(f"API pool validation failed: {results['error']}")
        
        # Display detailed results like in test file
        print("=" * 30)
        print("📊 VALIDATION SUMMARY")
        print("=" * 30)
        print(f"Total entries: {results['total_entries']}")
        print(f"Valid entries: {results['valid_entries']}")
        print(f"Invalid entries: {results['invalid_entries']}")
        print(f"Success rate: {results['valid_entries']/results['total_entries']*100:.1f}%")
        
        print(f"\n📋 DETAILED RESULTS")
        print("-" * 30)
        for result in results['results']:
            status = "✅" if result['valid'] else "❌"
            print(f"{status} {result['profile']} ({result['source']}): {result['message']}")
        
        # Check if we have any valid entries
        if results['valid_entries'] == 0:
            raise ValueError("❌ No valid API keys found in the pool! All API keys failed validation.")
        
        # Load original data to get valid entries with API keys
        with open(pool_file_path, 'r') as f:
            original_data = json.load(f)
            original_pool = original_data.get('api_pool', [])
        
        # Keep only valid entries
        valid_pool = []
        for result in results['results']:
            if result['valid']:
                # Find the original entry to get the API key
                for original_entry in original_pool:
                    if original_entry['profile'] == result['profile']:
                        valid_pool.append(original_entry)
                        break
        
        smithery_api_pool = valid_pool
        
        print(f"\n✅ SUCCESS: Using {len(smithery_api_pool)} valid API keys from pool")
        print("=" * 50)
        return smithery_api_pool
        
    except Exception as e:
        print(f"❌ Error during API pool validation: {e}")
        raise ValueError(f"API pool validation failed: {str(e)}")

def get_api_key_for_worker(worker_id):
    """Get API key and profile for a specific worker"""
    if smithery_api_pool and len(smithery_api_pool) > 0:
        # Round-robin assignment
        pool_entry = smithery_api_pool[worker_id % len(smithery_api_pool)]
        return pool_entry['api_key'], pool_entry['profile']
    else:
        return args.smithery_api_key, args.smithery_profile

def construct_mcp_server_url(server_info, api_key=None, profile=None):
    """
    Construct MCP server URL from server info.
    """
    if not server_info:
        return None
        
    server_url = server_info.get('python_sdk_url', '')
    if not server_url:
        return None
    
    # Use provided api_key and profile, or fall back to args
    if api_key is None:
        api_key = args.smithery_api_key
    if profile is None:
        profile = args.smithery_profile
    
    # Get or create default config
    mcp_config = server_info.get('python_sdk_config', "")
    if mcp_config == "":
        mcp_config = {"debug": False}
    else:
        try:
            mcp_config = json.loads(mcp_config)
        except json.JSONDecodeError:
            mcp_config = {"debug": False}
    
    # Replace URL placeholders
    config_b64 = base64.b64encode(json.dumps(mcp_config).encode()).decode()
    if "{config_b64}" in server_url:
        server_url = server_url.replace("{config_b64}", config_b64)
    if "{smithery_api_key}" in server_url:
        server_url = server_url.replace("{smithery_api_key}", api_key)
    if "{smithery_profile}" in server_url:
        server_url = server_url.replace("{smithery_profile}", profile)
    elif "&profile=" not in server_url and "profile=" not in server_url:
        server_url += f"&profile={profile}"
    
    return server_url

def convert_openai_agent_result_to_messages(result, original_messages, system_prompt=None):
    """Convert OpenAI Agent result to message format compatible with Qwen Agent structure"""
    all_messages = []

    # Prepend system prompt if provided
    if system_prompt:
        all_messages.append({
            "role": "system",
            "content": system_prompt
        })
    
    # Add original user message
    all_messages.extend(original_messages)
    
    # Process conversation flow from OpenAI Agent  
    if hasattr(result, 'new_items') and result.new_items:
        current_reasoning = []  # Collect reasoning content
        
        for item_flow in result.new_items:
            if item_flow.type == "reasoning_item":
                # Collect reasoning content
                if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                    for content in item_flow.raw_item.content:
                        if hasattr(content, 'text'):
                            current_reasoning.append(content.text)
            
            elif item_flow.type == "tool_call_item":
                # Extract tool call information
                if hasattr(item_flow, 'raw_item'):
                    tool_call = {
                        "name": getattr(item_flow.raw_item, 'name', None),
                        "arguments": getattr(item_flow.raw_item, 'arguments', None),
                        "call_id": getattr(item_flow.raw_item, 'call_id', None)
                    }
                    
                    # Flush reasoning as a separate assistant message before the tool call
                    if current_reasoning:
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning)
                        })
                        current_reasoning = []  # Reset for next iteration
                    
                    # Create assistant message with tool call only
                    assistant_msg = {
                        "role": "assistant",
                        "content": "",
                        "function_call": tool_call
                    }
                    
                    all_messages.append(assistant_msg)
            
            elif item_flow.type == "tool_call_output_item":
                # Extract tool output
                if hasattr(item_flow, 'output'):
                    try:
                        # Parse the JSON output
                        output_data = json.loads(item_flow.output)
                        if output_data.get('type') == 'text':
                            # Parse the inner JSON text
                            inner_data = json.loads(output_data.get('text', '{}'))
                            tool_output = json.dumps(inner_data)
                        else:
                            tool_output = item_flow.output
                    except:
                        tool_output = item_flow.output
                    
                    # Find the corresponding tool call name from previous messages
                    tool_name = 'unknown'
                    if hasattr(item_flow, 'raw_item'):
                        raw = item_flow.raw_item
                        call_id = None
                        for attr in ['tool_call_id', 'call_id', 'id', 'toolCallId']:
                            if hasattr(raw, attr):
                                call_id = getattr(raw, attr)
                                break
                        if call_id is not None:
                            # Look for the matching tool call in previous messages
                            for prev_msg in reversed(all_messages):
                                if (prev_msg.get('role') == 'assistant' and 
                                    'function_call' in prev_msg and 
                                    prev_msg['function_call'].get('call_id') == call_id):
                                    tool_name = prev_msg['function_call'].get('name', 'unknown')
                                    break
                    
                    # Fallback: if still unknown, use the most recent assistant function_call
                    if tool_name == 'unknown':
                        for prev_msg in reversed(all_messages):
                            if prev_msg.get('role') == 'assistant' and 'function_call' in prev_msg:
                                name_candidate = prev_msg['function_call'].get('name')
                                if name_candidate:
                                    tool_name = name_candidate
                                    break
                    
                    all_messages.append({
                        "role": "function",
                        "content": tool_output,
                        "name": tool_name
                    })
            
            elif item_flow.type == "message_output_item":
                # Extract final assistant message
                if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                    message_texts = []
                    for content in item_flow.raw_item.content:
                        if hasattr(content, 'text'):
                            message_texts.append(content.text)
                    
                    # Flush any remaining reasoning content as a separate message
                    if current_reasoning:
                        all_messages.append({
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "\n".join(current_reasoning)
                        })
                        current_reasoning = []
                    
                    # Create final assistant message
                    final_content = "\n".join(message_texts)
                    if final_content.strip():
                        final_msg = {
                            "role": "assistant",
                            "content": final_content
                        }
                        all_messages.append(final_msg)
    
    # If no conversation flow items, use final output  
    new_messages_start = len(original_messages) + (1 if system_prompt else 0)
    if not any(msg.get('role') == 'assistant' and msg.get('content') for msg in all_messages[new_messages_start:]):
        final_msg = {
            "role": "assistant", 
            "content": result.final_output
        }
        
        # Try to extract reasoning from the result if available
        reasoning_content = []
        if hasattr(result, 'new_items') and result.new_items:
            for item_flow in result.new_items:
                if item_flow.type == "reasoning_item":
                    if hasattr(item_flow, 'raw_item') and hasattr(item_flow.raw_item, 'content'):
                        for content in item_flow.raw_item.content:
                            if hasattr(content, 'text'):
                                reasoning_content.append(content.text)
        
        # Emit reasoning as a separate assistant message before the final message
        if reasoning_content:
            all_messages.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": "\n".join(reasoning_content)
            })
        
        all_messages.append(final_msg)
    
    return all_messages

def create_agent_for_item(item, api_key=None, profile=None):
    """
    Create an OpenAI Agent for an item if it has MCP server information
    """
    # Check if item has MCP server info in metadata
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])
    
    if not mcp_servers or not isinstance(mcp_servers, list):
        return None
    
    # Setup OpenAI model config
    if args.engine == "openrouter_api":
        model = OpenAIResponsesModel(
            args.model_path,
            openai_client=AsyncClient(
                base_url=args.openrouter_url,
                api_key=args.openrouter_api_key,
            ),
        )
    elif args.engine == "vllm_api":
        model = OpenAIResponsesModel(
            args.model_path,
            openai_client=AsyncClient(
                base_url=args.vllm_api_url,
                api_key=args.vllm_api_key,
            ),
        )
    else:
        print(f"Unsupported engine: {args.engine}")
        return None
    
    # Setup MCP servers list
    mcp_servers_list = []
    
    for server_info in mcp_servers:
        server_name = server_info.get('server_name', 'unknown-server')
        server_details = server_info.get('server_info', {})
        
        # Construct MCP server URL with provided api_key and profile
        server_url = construct_mcp_server_url(server_details, api_key, profile)
        if not server_url:
            print(f"Failed to construct URL for server {server_name}")
            continue
        
        # Create safe server name for config
        safe_server_name = server_name.replace(' ', '-').lower()
        
        print(f"📡 Configured MCP server: {safe_server_name} -> {server_url[:100]}...")
        mcp_servers_list.append({
            "name": safe_server_name,
            "url": server_url,
            "timeout": 600.0,
            "sse_read_timeout": 600.0,
            "terminate_on_close": False
        })
    
    if not mcp_servers_list:
        print("No valid MCP servers found")
        return None
    
    # Create agent with better error handling
    try:
        print(f"🤖 Creating OpenAI agent with {len(mcp_servers_list)} MCP servers...")
        agent_config = {
            "name": "OSS-Assistant",
            "instructions": """You are a helpful assistant with access to various tools and data sources. 
            When users ask questions, you MUST use the available tools.""",
            "model": model,
            "mcp_servers_list": mcp_servers_list
        }
        print(f"✅ Agent config created successfully")
        return agent_config
    except Exception as e:
        print(f"❌ Failed to create agent with {len(mcp_servers_list)} servers: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"   Caused by: {e.__cause__}")
        return None

def qwen_compatible_system_prompt_generator(tools):
    """Generate a Qwen-compatible system prompt from tool specs.

    tools: iterable of FunctionTool-like objects with attributes:
      - name: str
      - description: str | None
      - params_json_schema: dict | None (JSON Schema for parameters)
    """
    import json as _json

    # Build function schema list expected by Qwen's NousFnCallPrompt
    function_schemas = []
    for tool in tools or []:
        name = getattr(tool, 'name', None) or ''
        description = getattr(tool, 'description', None) or ''
        params_schema = getattr(tool, 'params_json_schema', None) or {"type": "object", "properties": {}}

        function_schemas.append({
            "name": name,
            "description": description,
            "parameters": params_schema,
        })

    tool_descs_wrapped = [{"type": "function", "function": fs} for fs in function_schemas]
    tool_descs_str = "\n".join(_json.dumps(d, ensure_ascii=False) for d in tool_descs_wrapped)

    template = (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n{tool_descs}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        "{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n"
        "</tool_call>"
    )

    return template.format(tool_descs=tool_descs_str)

# Process a single item using agent inference
async def process_single_item_agent_async(item, api_key=None, profile=None):
    """Process a single item using agent inference (async version)"""
    # Get prompt ID for better error tracking
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')

    if args.enable_tool_hint:
        if "metadata" in item and "target_tools" in item["metadata"]:
            target_tools = item["metadata"].get('target_tools', "")
        else:
            target_tools = item.get("target_tools", "")
        tool_list = [tool.strip() for tool in target_tools.split(',')] 
        # remove contents before :: in tool_list
        tool_list = [tool.split('::')[1] if '::' in tool else tool for tool in tool_list]
        tool_list = [f"{tool}" for tool in tool_list]
        tool_list = ", ".join(tool_list)
        print(f"🔍 Tool list: {tool_list}")
    
    message = item["messages"]
    # remove the system prompt if it exists
    if message[0]['role'] == 'system':
        message = message[1:]
    
    # Extract the current user message (the last user message in the conversation)
    user_messages = [msg for msg in message if msg.get('role') == 'user']
    if user_messages:
        user_content = user_messages[-1]['content']
    else:
        raise ValueError("No user messages found")
    
    # Try to create agent for this item
    agent_config = None
    if args.agent:
        agent_config = create_agent_for_item(item, api_key, profile)
    
    if agent_config:
        try:
            # Use agent inference
            print(f"🚀 Running OpenAI agent inference for item {prompt_id}...")

            # Add tool hint if enabled
            if args.enable_tool_hint:
                # Get MCP server information for tool hint
                if tool_list:
                    tool_hint = f'\n\nWe need to use the following tools: {tool_list}.'
                else:
                    tool_hint = '\n\nWe need to use the provided tools.'
                user_content = user_content + tool_hint

            if args.enable_irrelevant_warning:
                user_content = user_content + '\n\nUse tools only if they are relevant. Otherwise, do not use them.'

            # Handle both single and multiple MCP servers
            server_configs = agent_config["mcp_servers_list"]
            
            # Create a list to hold all MCP server context managers
            async def create_mcp_servers():
                mcp_servers = []
                server_contexts = []
                
                for server_config in server_configs:
                    mcp_server_context = MCPServerStreamableHttp(
                        name=server_config["name"],
                        params={
                            "url": server_config["url"],
                            "headers": {
                                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
                            },
                            "timeout": server_config.get("timeout", 600.0),
                            "sse_read_timeout": server_config.get("sse_read_timeout", 600.0),
                            "terminate_on_close": server_config.get("terminate_on_close", False)
                        },
                        client_session_timeout_seconds=args.timeout,
                    )
                    
                    # Enter the context and collect both the server and its context
                    mcp_server = await mcp_server_context.__aenter__()
                    mcp_servers.append(mcp_server)
                    server_contexts.append(mcp_server_context)
                
                return mcp_servers, server_contexts
            
            # Create and manage multiple MCP servers
            try:
                mcp_servers, server_contexts = await create_mcp_servers()
                
                try:
                    # Create OpenAI Agent with multiple MCP servers
                    agent = Agent(
                        name=agent_config["name"],
                        instructions=agent_config["instructions"],
                        model=agent_config["model"],
                        mcp_servers=mcp_servers
                    )
                    run_context = RunContextWrapper(context=None)
                    
                    print(f"🔍 User Query Passed to Agent: {user_content}")
                    # If this is a multi-turn conversation, populate the session with history
                    if len(message) > 1:
                        # Create a session for conversation management
                        # Use prompt_id as session identifier to maintain conversation history
                        session = SQLiteSession(f"conversation_{prompt_id}")

                        # Clear any existing session data first
                        await session.clear_session()
                        
                        # Add conversation history to session (all messages except the last user message)
                        history_items = []
                        for msg in message[:-1]:  # All messages except the last one
                            if msg['role'] == 'user':
                                history_items.append({"role": "user", "content": msg['content']})
                            elif msg['role'] == 'assistant':
                                history_items.append({"role": "assistant", "content": msg['content']})
                            elif msg['role'] == 'function':
                                # Convert function response to assistant message mentioning the function result
                                function_name = msg.get('name', 'unknown_function')
                                history_items.append({
                                    "role": "assistant", 
                                    "content": f"[Function {function_name} returned: {msg['content']}]"
                                })
                        
                        # Add history to session
                        if history_items:
                            await session.add_items(history_items)

                        # Run agent inference with session for automatic conversation management
                        result = await Runner.run(agent, input=user_content, session=session, max_turns=args.max_turns)
                    else:
                        result = await Runner.run(agent, input=user_content, max_turns=args.max_turns)

                    available_tools = await agent.get_all_tools(run_context)
                    system_prompt = qwen_compatible_system_prompt_generator(available_tools)

                    # Convert OpenAI Agent result to message format (this is the main conversation history)
                    all_messages = convert_openai_agent_result_to_messages(result, message, system_prompt)
                                    
                    if len(all_messages) > len(message):
                        print(f"✅ OpenAI agent inference completed for item {prompt_id} with {len(mcp_servers)} MCP servers\n============================================================")
                        item['messages'] = all_messages
                    else:
                        print(f"⚠️ OpenAI agent inference returned empty response for item {prompt_id}\n============================================================")
                        # Throw exception to trigger fallback instead of returning empty content
                        raise Exception("Agent returned empty response")
                
                finally:
                    # Clean up all MCP server contexts
                    for server_context in reversed(server_contexts):
                        try:
                            await server_context.__aexit__(None, None, None)
                        except Exception as cleanup_error:
                            print(f"⚠️ Warning: Failed to cleanup MCP server context: {cleanup_error}")
            
            except Exception as server_creation_error:
                print(f"❌ Failed to create MCP servers: {server_creation_error}")
                raise
                
        except Exception as e:
            print(f"❌ OpenAI agent inference failed for item {prompt_id}: {str(e)}")
            print(f"   Error type: {type(e).__name__}")
            if "async" in str(e).lower() or "context" in str(e).lower() or "sse" in str(e).lower():
                print(f"   🔍 This appears to be an async/context/MCP streaming error")
    
            # Re-raise the exception to trigger fallback instead of returning empty content
            raise e
    else:
        # If no agent could be created, raise an exception to trigger fallback
        if args.agent:
            raise ValueError("Failed to create agent for this item")
        else:
            raise ValueError("No agent specified")
    
    return item

@timeout(args.timeout, use_signals=False)
def process_single_item_agent(item, api_key=None, profile=None):
    """Process a single item using agent inference with timeout"""
    prompt_id = item.get('metadata', {}).get('prompt_id', 'unknown')
    
    try:
        return asyncio.run(process_single_item_agent_async(item, api_key, profile))
    except Exception as e:
        print(f"Error processing item {prompt_id}: {str(e)}")
        message = item["messages"]
        item['messages'] = message + [
            {
                "role": "assistant",
                "content": f"[ERROR: {str(e)}]"
            }
        ]
        return item


# Dynamic processing with timeout resilience
class DynamicProcessor:
    """
    Dynamic processor that handles individual items with timeout resilience.
    Each item is processed independently so timeouts don't block other items.
    """
    
    def __init__(self, max_workers=None, checkpoint_every=16):
        self.max_workers = max_workers or len(smithery_api_pool) if smithery_api_pool else 1
        self.checkpoint_every = checkpoint_every
        self.processed_count = 0
        self.lock = threading.Lock()
        self.completed_items_list = []  # Thread-safe list for completed items
        
    def process_single_item_with_fallback(self, item_data):
        """Process a single item with fallback to direct API if agent fails"""
        item, item_index, api_key, profile = item_data
        prompt_id = item.get('metadata', {}).get('prompt_id', f'item_{item_index}')
        
        # Try agent processing first if available
        agent_failed = False
        agent_error = None
        
        if args.agent:
            try:
                processed_item = process_single_item_agent(item, api_key, profile)
                return processed_item, item_index, True, None  # success, no error
            except Exception as e:
                print(f"⚠️ Agent processing failed for item {prompt_id}: {str(e)}")
                agent_failed = True
                agent_error = str(e)
        else:
            print(f"ℹ️ No agent specified for item {prompt_id}, using direct API...")
            agent_failed = True
            agent_error = "No agent specified"
            
        # Fallback to direct API call if agent failed or not available
        if agent_failed:
            message = item["messages"]
            # remove the system prompt if it exists
            if message[0]['role'] == 'system':
                message = message[1:]
            # If multiple user messages, take the last user
            user_messages = [msg for msg in message if msg.get('role') == 'user']
            if user_messages:
                user_content = user_messages[-1]['content']
            else:
                raise ValueError("No user messages found")

            # Replace the last user message with the new user content
            if message[-1]['role'] == 'user':
                input_messages = message[:-1] + [{"role": "user", "content": user_content}]
            else:
                raise ValueError("Last message is not a user message?")
            
            try:
                print(f"🔄 Using direct API for item {prompt_id}...")
                api_response = make_api_request_with_retry(
                    input_messages,
                    API_PARAMS,
                    API_ENDPOINT,
                    API_HEADERS,
                )
                
                if api_response is not None:
                    response = api_response.strip()
                    item['messages'] = input_messages + [
                        {
                            "role": "assistant", 
                            "content": response
                        }
                    ]
                    return item, item_index, True, f"Direct API used: {agent_error}"
                else:
                    # API returned None - treat as failure
                    raise Exception("API request returned None after all retries")
                
            except Exception as e2:
                print(f"❌ Direct API failed for item {prompt_id}: {str(e2)}")
                item['messages'] = input_messages + [
                    {
                        "role": "assistant",
                        "content": f"[ERROR: Agent failed ({agent_error}), API failed ({str(e2)})]"
                    }
                ]
                return item, item_index, False, f"Both failed: {str(e2)}"
                
    def process_items_dynamically(self, items_to_process, processed_dataset, checkpoint_file, progress_bar):
        """
        Process items dynamically with individual timeouts and immediate checkpointing.
        Only saves completed items to checkpoint for proper resume functionality.
        """
        completed_items = {}
        
        # Prepare items with metadata for processing
        items_with_metadata = []
        for i, (item, original_index) in enumerate(items_to_process):
            api_key, profile = get_api_key_for_worker(i)
            items_with_metadata.append((item, original_index, api_key, profile))
        
        # Process items with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all items
            future_to_data = {}
            for item_data in items_with_metadata:
                future = executor.submit(self.process_single_item_with_fallback, item_data)
                future_to_data[future] = item_data
            
            # Process completions as they arrive
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    processed_item, original_index, success, error_msg = future.result()
                    completed_items[original_index] = processed_item
                    
                    # Update the main dataset immediately
                    processed_dataset[original_index] = processed_item
                    
                    # Update progress and handle checkpoint saving atomically
                    with self.lock:
                        # Add to completed items list for checkpoint (thread-safe)
                        self.completed_items_list.append(processed_item)
                        
                        self.processed_count += 1
                        progress_bar.update(1)
                        
                        # Log completion status
                        prompt_id = processed_item.get('metadata', {}).get('prompt_id', f'item_{original_index}')
                        status = "✅" if success else "❌"
                        if error_msg:
                            print(f"{status} Completed item {prompt_id} (index {original_index}) - {error_msg}")
                        else:
                            print(f"{status} Completed item {prompt_id} (index {original_index})")
                        
                        # Save checkpoint periodically - ONLY completed items
                        if self.processed_count % self.checkpoint_every == 0:
                            self._save_checkpoint_safely(checkpoint_file)
                
                except Exception as e:
                    item_data = future_to_data[future]
                    original_item, original_index, _, _ = item_data
                    prompt_id = original_item.get('metadata', {}).get('prompt_id', f'item_{original_index}')
                    print(f"❌ Unexpected error processing item {prompt_id}: {str(e)}")
                    
                    # Create error item
                    message = original_item["messages"]
                    original_item['messages'] = message + [
                        {
                            "role": "assistant",
                            "content": f"[UNEXPECTED_ERROR: {str(e)}]"
                        }
                    ]
                    processed_dataset[original_index] = original_item
                    
                    with self.lock:
                        self.completed_items_list.append(original_item)
                        self.processed_count += 1
                        progress_bar.update(1)
        
        # Final checkpoint save for any remaining completed items
        with self.lock:
            if self.completed_items_list:
                self._save_checkpoint_safely(checkpoint_file, is_final=True)
        
        return len(completed_items)
    
    def _save_checkpoint_safely(self, checkpoint_file, is_final=False):
        """
        Thread-safe checkpoint saving method.
        Must be called within self.lock context.
        """
        try:
            # Load existing checkpoint and append new completions
            existing_completed = []
            if os.path.exists(checkpoint_file):
                try:
                    existing_completed = load_dataset_from_file(checkpoint_file)
                    if not isinstance(existing_completed, list):
                        existing_completed = [existing_completed]
                except Exception as e:
                    print(f"⚠️ Warning: Could not load existing checkpoint: {e}")
                    existing_completed = []
            
            # Create combined list and sort by row_id
            all_completed = existing_completed + self.completed_items_list
            all_completed_sorted = sort_dataset_by_row_id(all_completed)
            
            # Save checkpoint safely
            safe_save_checkpoint(all_completed_sorted, checkpoint_file, convert_to_jsonl=False)
            
            checkpoint_type = "Final" if is_final else "Periodic"
            print(f"💾 {checkpoint_type} checkpoint saved: {len(all_completed_sorted)} completed items total (sorted by row_id)")
            
            # Clear the completed items list since they're now saved
            self.completed_items_list = []
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            # Don't clear the list if save failed - we'll try again next time

# Function to sort dataset by row_id from metadata
def sort_dataset_by_row_id(dataset):
    """Sort dataset by row_id from metadata, handling missing row_ids gracefully"""
    def get_sort_key(item):
        metadata = item.get('metadata', {})
        row_id = metadata.get('row_id')
        if row_id is not None:
            try:
                return int(row_id)
            except (ValueError, TypeError):
                # If row_id can't be converted to int, use as string
                return float('inf'), str(row_id)
        else:
            # Items without row_id go to the end
            return float('inf'), ''
    
    return sorted(dataset, key=get_sort_key)

# Function to add generation config to metadata
def add_generation_config_to_metadata(dataset, model_short_name, generation_params):
    """Add synthetic data generation config to each item's metadata"""
    config_entry = {
        "model": model_short_name,
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

# Generate outputs using dynamic processing with timeout resilience
def generate_and_update(dataset, checkpoint_file):
    processed_dataset = copy.deepcopy(dataset)

    # Prepare generation parameters for metadata
    generation_params = {
        "engine": args.engine,
        "model_path": args.model_path,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "num_trials": args.num_trials,
        "step": args.step,
        "agent": args.agent,
        "timeout": args.timeout,
        "max_workers": args.max_workers
    }

    # Determine which items need processing by comparing IDs/metadata
    items_to_process = []
    completed_item_ids = set()
    completed_count = 0
    
    if os.path.exists(checkpoint_file):
        try:
            checkpoint_data = load_dataset_from_file(checkpoint_file)
            if not isinstance(checkpoint_data, list):
                checkpoint_data = [checkpoint_data]
            
            print(f"Checkpoint file found with {len(checkpoint_data)} completed items.")
            
            # Extract completed item IDs from checkpoint
            for completed_item in checkpoint_data:
                # Use prompt_id from metadata if available, otherwise use a hash of the input
                metadata = completed_item.get('metadata', {})
                prompt_id = metadata.get('prompt_id')
                
                if prompt_id:
                    completed_item_ids.add(prompt_id)
                else:
                    # Fallback: use hash of the user message for identification
                    messages = completed_item.get('messages', [])
                    if messages:
                        user_msg = next((msg['content'] for msg in messages if msg.get('role') == 'user'), '')
                        if user_msg:
                            completed_item_ids.add(hash(user_msg))
            
            completed_count = len(checkpoint_data)
            
            # Update processed_dataset with completed items for those positions we can identify
            # This maintains compatibility with the old approach while being more robust
            checkpoint_index = 0
            for i, item in enumerate(processed_dataset):
                metadata = item.get('metadata', {})
                prompt_id = metadata.get('prompt_id')
                
                # Check if this item is completed
                is_completed = False
                if prompt_id and prompt_id in completed_item_ids:
                    is_completed = True
                else:
                    # Fallback check using message hash
                    messages = item.get('messages', [])
                    if messages:
                        user_msg = next((msg['content'] for msg in messages if msg.get('role') == 'user'), '')
                        if user_msg and hash(user_msg) in completed_item_ids:
                            is_completed = True
                
                if is_completed and checkpoint_index < len(checkpoint_data):
                    # Replace with completed version from checkpoint
                    processed_dataset[i] = checkpoint_data[checkpoint_index]
                    checkpoint_index += 1
                else:
                    # This item needs processing
                    items_to_process.append((item, i))
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh...")
            completed_count = 0
            # Process all items if checkpoint is corrupted
            for i in range(len(processed_dataset)):
                items_to_process.append((processed_dataset[i], i))
    else:
        print("No checkpoint found. Processing all items.")
        # Process all items
        for i in range(len(processed_dataset)):
            items_to_process.append((processed_dataset[i], i))
    
    print(f"Total items in dataset: {len(processed_dataset)}")
    print(f"Already completed: {completed_count}")
    print(f"Remaining to process: {len(items_to_process)}")
    
    if len(items_to_process) == 0:
        print("All items already processed!")
        return processed_dataset

    # Create dynamic processor
    max_workers = args.max_workers or (len(smithery_api_pool) if smithery_api_pool else 8)
    processor = DynamicProcessor(
        max_workers=max_workers, 
        checkpoint_every=CHECKPOINT_EVERY
    )
    
    print(f"🚀 Starting dynamic processing with {max_workers} workers...")
    print(f"💾 Checkpoints will be saved every {CHECKPOINT_EVERY} completed items")
    print(f"⏱️ Individual item timeout: {args.timeout} seconds")
    
    # Create progress bar for remaining items
    with tqdm(total=len(items_to_process), 
              desc="Processing items", 
              unit="item",
              initial=0,
              leave=True, 
              dynamic_ncols=True,
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as progress_bar:
        
        start_time = time()
        
        # Process items dynamically (will use agent if available, otherwise direct API)
        completed_count = processor.process_items_dynamically(
            items_to_process, 
            processed_dataset, 
            checkpoint_file, 
            progress_bar
        )
        
        end_time = time()
        
        print(f"\n🎉 Dynamic processing completed!")
        print(f"📊 Items processed: {completed_count}/{len(items_to_process)}")
        print(f"⏱️ Total time: {end_time - start_time:.2f} seconds")
        print(f"⚡ Average time per item: {(end_time - start_time)/max(completed_count, 1):.2f} seconds")

    # Add generation config to metadata and sort by row_id before returning
    processed_dataset = add_generation_config_to_metadata(processed_dataset, model_abbreviation, generation_params)
    processed_dataset_sorted = sort_dataset_by_row_id(processed_dataset)
    
    return processed_dataset_sorted

# Main function to control workflow
def main():
    # Load and validate Smithery API pool
    api_pool = load_and_validate_smithery_api_pool(args.smithery_api_pool)
    
    # Display dynamic processing info
    effective_workers = args.max_workers or len(api_pool)
    print("=" * 50)
    print("🚀 DYNAMIC PROCESSING CONFIGURATION")
    print("=" * 50)
    print(f"Processing mode: Dynamic (individual item processing)")
    print(f"Workers: {effective_workers}")
    print(f"API pool size: {len(api_pool)}")
    print(f"Timeout per item: {args.timeout} seconds")
    print(f"Checkpoint frequency: Every {args.checkpoint_every} completed items")
    
    if args.max_workers is not None:
        print(f"Worker setting: Custom ({args.max_workers} workers)")
    else:
        print(f"Worker setting: Auto-detected from API pool size")
    
    print(f"Resilience: Individual timeouts prevent blocking")
    if args.agent:
        print(f"Processing: Agent mode with direct API fallback")
    else:
        print(f"Processing: Direct API mode (no agent)")
    print(f"Checkpoint format: Only completed items (compatible with old format)")
    print(f"Sorting: All outputs sorted by row_id from metadata")
    print("=" * 50)
    
    try:
        # Load instructions from the input file
        dataset = load_dataset_from_file(INPUT_FILE_NAME)
        
        # Ensure dataset is always a list (fix for single-item JSON files)
        if not isinstance(dataset, list):
            dataset = [dataset]

        if args.num_trials == 1:
            updated_dataset = generate_and_update(dataset, checkpoint_file)
            save_dataset(updated_dataset, saved_file, convert_to_jsonl=True)

            # Optionally remove the checkpoint file after completion
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            print("Final dataset saved. Checkpoint removed.")
        else:
            for i in range(args.num_trials):
                updated_dataset = generate_and_update(dataset, checkpoint_files[i])
                save_dataset(updated_dataset, saved_files[i], convert_to_jsonl=True)

                # Optionally remove the checkpoint file after completion
                if os.path.exists(checkpoint_files[i]):
                    os.remove(checkpoint_files[i])
                print(f"Dataset for trial {i} saved. Checkpoint {i} removed.")
    
    finally:
        # Clean up MCP resources to ensure proper program exit
        if args.agent:
            try:
                print("🧹 Cleaning up MCP resources...")
                # OpenAI Agent framework handles cleanup automatically via context managers
                print("✅ MCP cleanup completed.")
            except Exception as e:
                print(f"⚠️ Warning: MCP cleanup failed: {e}")
        
        print("🎯 Program execution completed.")
        os._exit(0)  # Use os._exit to avoid atexit conflicts with multiprocessing


# Run the main function
if __name__ == "__main__":
    main()