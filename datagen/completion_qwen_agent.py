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
import nest_asyncio
import threading
import queue
import signal
import atexit
import os
from time import sleep, time
from tqdm import tqdm
from wrapt_timeout_decorator import timeout
from utils import load_dataset_from_file, save_dataset, make_api_request_with_retry, get_model_short_name, validate_api_pool_from_file, check_if_api_key_is_valid, safe_save_checkpoint, get_model_abbreviation

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Qwen Agent imports
from qwen_agent.agents import Assistant

# Check if qwen_agent is installed
try:
    import qwen_agent
except ImportError:
    print("qwen_agent is not installed. Please install it.")
    exit(1)

# Global cleanup function for MCP resources
def cleanup_mcp_resources():
    """Clean up MCP resources on exit"""
    # Only cleanup if we're using agent mode
    try:
        # Check if args is available and agent mode is enabled
        if 'args' in globals() and hasattr(args, 'agent') and args.agent:
            from qwen_agent.tools.mcp_manager import MCPManager
            if MCPManager._instance is not None:
                # print("🧹 Emergency cleanup: Shutting down MCP resources...")
                manager = MCPManager()
                manager.shutdown()
                # print("✅ Emergency MCP cleanup completed.")
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
    parser.add_argument("--model_path", type=str, default="qwen/qwen3-32b",
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
    parser.add_argument("--agent", type=str, default="qwen_agent", help="Use agent inference for items with MCP server URLs")
    parser.add_argument("--timeout", type=int, default=90, help="Timeout in seconds for each item processing (default: 90 seconds)")
    parser.add_argument("--max_retries", type=int, default=2, help="Maximum number of retries for each item processing (default: 3)")
    parser.add_argument("--fncall_prompt_type", type=str, default="nous", help="Function call prompt type (default: nous)")
    parser.add_argument("--parallel_function_calls", type=bool, default=False, help="Parallel function calls (default: False)")
    parser.add_argument("--enable_tool_hint", action="store_true", help="Enable tool hint (default: off)")
    parser.add_argument("--enable_irrelevant_warning", action="store_true", help="Enable irrelevant warning (default: off)")
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

base_name = INPUT_FILE_NAME[:INPUT_FILE_NAME.rfind('.')]
if base_name.endswith("_4prepared"):
    base_name = base_name[:-10]  # Remove "_4prepared"

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

def create_agent_for_item(item, api_key=None, profile=None):
    """
    Create a Qwen Agent for an item if it has MCP server information
    """
    # Check if item has MCP server info in metadata
    metadata = item.get('metadata', {})
    mcp_servers = metadata.get('mcp_servers', [])
    
    if not mcp_servers or not isinstance(mcp_servers, list):
        return None
    
    # Setup LLM config
    if args.engine == "openrouter_api":
        llm_cfg = {
            'model': args.model_path,
            'model_server': args.openrouter_url,
            'api_key': args.openrouter_api_key,
            'generate_cfg': {
                'max_retries': args.max_retries,
                'fncall_prompt_type': args.fncall_prompt_type,
                'parallel_function_calls': args.parallel_function_calls,
                'extra_body': {}
            },
        }
    elif args.engine == "vllm_api":
        if "devstral-small" in args.model_path.lower() or "mistral-small" in args.model_path.lower():
            print(f"Using mistral_vllm mode in Qwen Agent.")
            llm_cfg = {
                'model': args.model_path,
                'model_type': 'mistral_vllm',
                'model_server': args.vllm_api_url,
                'api_key': args.vllm_api_key,
                'generate_cfg': {
                    'max_retries': args.max_retries,
                    'fncall_prompt_type': args.fncall_prompt_type,
                    'parallel_function_calls': args.parallel_function_calls,
                    'extra_body': {},
                },
            }
        elif "kimi-k2" in args.model_path.lower():
            llm_cfg = {
                'model': args.model_path,
                'model_server': args.vllm_api_url,
                'api_key': args.vllm_api_key,
                'generate_cfg': {
                    'max_retries': args.max_retries,
                    'stop_token_ids': [163586],
                    'fncall_prompt_type': args.fncall_prompt_type,
                    'parallel_function_calls': args.parallel_function_calls,
                    'extra_body': {},
                },
            }
        elif "gpt-oss" in args.model_path.lower():
            llm_cfg = {
                'model': args.model_path,
                'model_server': args.vllm_api_url,
                'api_key': args.vllm_api_key,
                'model_type': 'oai', # Use Qwen-Compatible OAI prompt template, not oss_vllm
                'generate_cfg': {
                    'max_retries': args.max_retries,
                    'fncall_prompt_type': args.fncall_prompt_type,
                    'parallel_function_calls': args.parallel_function_calls,
                    'extra_body': {},
                },
            }
        else:
            llm_cfg = {
                'model': args.model_path,
                'model_server': args.vllm_api_url,
                'api_key': args.vllm_api_key,
                'generate_cfg': {
                    'max_retries': args.max_retries,
                    'fncall_prompt_type': args.fncall_prompt_type,
                    'parallel_function_calls': args.parallel_function_calls,
                    'extra_body': {},
                },
            }

    # If model is qwen3, add generate_cfg
    if "qwen3" in args.model_path.lower():
        llm_cfg['generate_cfg']['extra_body']['chat_template_kwargs'] = {'enable_thinking': False}
    
    # Setup tools with all MCP servers
    mcp_servers_config = {}
    
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
        
        mcp_servers_config[safe_server_name] = {
            "url": server_url,
            "type": "streamable-http",
            "sse_read_timeout": 600
        }
        
        print(f"📡 Configured MCP server: {safe_server_name} -> {server_url[:100]}...")
    
    if not mcp_servers_config:
        print("No valid MCP servers found")
        return None
    
    tools = [{
        "mcpServers": mcp_servers_config
    }]
    
    # Create agent with better error handling
    try:
        print(f"🤖 Creating agent with {len(mcp_servers_config)} MCP servers...")
        agent = Assistant(llm=llm_cfg, function_list=tools)
        print(f"✅ Agent created successfully")
        return agent
    except Exception as e:
        print(f"❌ Failed to create agent with {len(mcp_servers_config)} servers: {e}")
        print(f"   Error type: {type(e).__name__}")
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"   Caused by: {e.__cause__}")
        return None

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
    
    # Try to create agent for this item
    agent = None
    if args.agent:
        agent = create_agent_for_item(item, api_key, profile)
    
    if agent:
        try:
            # Monkey patch LLM's convert_messages_to_dicts method to capture the full messages
            original_convert_method = agent.llm.convert_messages_to_dicts

            def capture_convert_messages_to_dicts(self, messages):
                self.captured_messages = [msg.model_dump() if hasattr(msg, 'model_dump') else msg for msg in messages]
                return original_convert_method(messages)

            agent.llm.convert_messages_to_dicts = types.MethodType(capture_convert_messages_to_dicts, agent.llm)

            # Use agent inference
            print(f"🚀 Running agent inference for item {prompt_id}...")

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

            # Replace the last user message with the new user content
            if message[-1]['role'] == 'user':
                input_messages = message[:-1] + [{"role": "user", "content": user_content}]
            else:
                raise ValueError("Last message is not a user message?")

            responses = None
            for responses in agent.run(
                messages=input_messages,
            ):
                pass  # Keep iterating until we get the final response

            all_messages = []
            captured_full_messages = getattr(agent.llm, 'captured_messages', [])
            # If the captured full messages is not empty and the first message is not a system prompt, add the system prompt to the captured full messages
            if captured_full_messages and message[0]['role'] != 'system':
                captured_system_prompt = {'role': 'system', 'content': captured_full_messages[0]['content']}
                all_messages = [captured_system_prompt] + message + responses
            else:
                all_messages = message + responses
                        
            if all_messages and len(responses) > 0:
                print(f"✅ Agent inference completed for item {prompt_id}\n============================================================")
                item['messages'] = all_messages
            else:
                print(f"⚠️ Agent inference returned empty response for item {prompt_id}\n============================================================")
                # Throw exception to trigger fallback instead of returning empty content
                raise Exception("Agent returned empty response")
        except Exception as e:
            print(f"❌ Agent inference failed for item {prompt_id}: {str(e)}")
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
        # Use asyncio.run to execute the async function
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
            
            # Add tool hint if enabled (even for direct API calls)
            if args.enable_tool_hint:
                # Get tool list for tool hint
                tool_list = item.get('target_tools', [])
                # only keep tool name in server_name::tool_name format
                tool_list = [tool.split('::')[1] if '::' in tool else tool for tool in tool_list]
                tool_list = list(set(tool_list))
                tool_list = [f"{tool}" for tool in tool_list]
                tool_list = ", ".join(tool_list)
                
                if tool_list:
                    tool_hint = f"\n\nYou should use the following tools to help answer the user's question: {tool_list}. Try to solve the user's query directly without asking follow-up questions."
                else:
                    tool_hint = "\n\nYou should use the provided tools to help answer the user's question. Try to solve the user's query directly without asking follow-up questions."
                user_content = user_content + tool_hint
                print(f"🔧 Added tool hint for direct API call on item {prompt_id}: {tool_hint.strip()}")

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
                # Import and cleanup MCP manager if it was initialized
                from qwen_agent.tools.mcp_manager import MCPManager
                if MCPManager._instance is not None:
                    manager = MCPManager()
                    manager.shutdown()
                    print("✅ MCP cleanup completed.")
            except Exception as e:
                print(f"⚠️ Warning: MCP cleanup failed: {e}")
        
        print("🎯 Program execution completed.")
        os._exit(0)  # Use os._exit to avoid atexit conflicts with multiprocessing


# Run the main function
if __name__ == "__main__":
    main()