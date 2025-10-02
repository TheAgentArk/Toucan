# Copyright 2023 The Qwen team, Alibaba Group. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import logging
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

from qwen_agent.llm.base import register_llm
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.llm.schema import ASSISTANT, FunctionCall, Message
from qwen_agent.log import logger


@register_llm('mistral_vllm')
class TextChatAtMistralVllm(TextChatAtOAI):
    """
    Specialized OpenAI-compatible client for Mistral models served via vLLM.
    Handles tool calling using native API tools parameter rather than prompt-based approach.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # Track tool call ID mapping: tool_name -> call_id
        self.tool_call_id_mapping = {}

    def _chat_with_functions(
        self,
        messages: List[Message],
        functions: List[Dict],
        stream: bool,
        delta_stream: bool,
        generate_cfg: dict,
        lang: Literal['en', 'zh'],
    ) -> Union[List[Message], Iterator[List[Message]]]:
        """
        Override to handle tools parameter for Mistral models via vLLM.
        Uses native OpenAI tools format instead of prompt-based tool calling.
        """
        if delta_stream:
            raise NotImplementedError('Please use stream=True with delta_stream=False, because delta_stream=True'
                                      ' is not implemented for function calling due to some technical reasons.')
        
        # Convert functions to OpenAI tools format
        tools = [{'type': 'function', 'function': f} for f in functions]
        generate_cfg = copy.deepcopy(generate_cfg)
        generate_cfg['tools'] = tools
        generate_cfg['tool_choice'] = 'auto'
        
        # Remove function calling specific params that don't apply to native tool calling
        for k in ['parallel_function_calls', 'function_choice', 'thought_in_content']:
            if k in generate_cfg:
                del generate_cfg[k]
        
        # Call the model directly without function calling prompt processing
        if stream:
            return self._parse_tool_calls_stream(
                self._chat_stream(messages, delta_stream=False, generate_cfg=generate_cfg)
            )
        else:
            response = self._chat_no_stream(messages, generate_cfg=generate_cfg)
            return self._parse_tool_calls_response(response)
    
    def convert_messages_to_dicts(self, messages: List[Message]) -> List[dict]:
        """
        Override to convert Qwen-Agent message format to vLLM-compatible format.
        - Convert function_call to tool_calls format
        - Convert 'function' role to 'tool' role with correct tool_call_id
        """
        messages = [msg.model_dump() for msg in messages]
        converted_messages = []
        
        for msg in messages:
            new_msg = copy.deepcopy(msg)
            
            # Convert function role to tool role
            if new_msg.get('role') == 'function':
                new_msg['role'] = 'tool'
                # Use the correct tool_call_id from our mapping
                tool_name = new_msg.get('name', '')
                if tool_name in self.tool_call_id_mapping:
                    new_msg['tool_call_id'] = self.tool_call_id_mapping[tool_name]
                else:
                    # Fallback if mapping not found - generate compliant ID
                    import hashlib
                    hash_obj = hashlib.md5(tool_name.encode()).hexdigest()
                    new_msg['tool_call_id'] = hash_obj[:9]
                # Remove the 'name' field as vLLM doesn't expect it for tool messages
                new_msg.pop('name', None)
            
            # Convert function_call to tool_calls format
            if new_msg.get('function_call'):
                function_call = new_msg.pop('function_call')
                # Generate vLLM-compliant call ID: exactly 9 alphanumeric characters
                import hashlib
                hash_input = function_call['name'] + function_call.get('arguments', '')
                hash_obj = hashlib.md5(hash_input.encode()).hexdigest()
                call_id = hash_obj[:9]  # Take first 9 hex characters (0-9, a-f)
                
                # Store the mapping for later use in tool results
                self.tool_call_id_mapping[function_call['name']] = call_id
                
                new_msg['tool_calls'] = [{
                    'id': call_id,
                    'type': 'function',
                    'function': {
                        'name': function_call['name'],
                        'arguments': function_call.get('arguments', '{}')
                    }
                }]
                # Clear content when there are tool calls
                if not new_msg.get('content'):
                    new_msg['content'] = ''
            
            converted_messages.append(new_msg)

        if logger.isEnabledFor(logging.DEBUG):
            from pprint import pformat
            logger.debug(f'Mistral vLLM Input:\n{pformat(converted_messages, indent=2)}')
        
        return converted_messages
    
    def _parse_tool_calls_stream(self, response_stream: Iterator[List[Message]]) -> Iterator[List[Message]]:
        """Parse tool calls from streaming response and convert to function_call format."""
        for messages in response_stream:
            parsed_messages = []
            for msg in messages:
                parsed_msg = self._parse_single_message_tool_calls(msg)
                parsed_messages.extend(parsed_msg)
            if parsed_messages:
                yield parsed_messages
    
    def _parse_tool_calls_response(self, messages: List[Message]) -> List[Message]:
        """Parse tool calls from non-streaming response and convert to function_call format."""
        parsed_messages = []
        for msg in messages:
            parsed_msg = self._parse_single_message_tool_calls(msg)
            parsed_messages.extend(parsed_msg)
        return parsed_messages
    
    def _parse_single_message_tool_calls(self, message: Message) -> List[Message]:
        """
        Parse tool calls from a single message content.
        vLLM Mistral returns tool calls in format: [TOOL_CALLS]tool_name{args}
        Convert this to proper function_call format and track call IDs.
        """
        if not message.content or message.function_call:
            return [message]
        
        content = message.content
        if not isinstance(content, str):
            return [message]
        
        # Check if this message contains tool calls
        if '[TOOL_CALLS]' not in content:
            return [message]
        
        messages = []
        
        # Split by [TOOL_CALLS] to find tool calls
        parts = content.split('[TOOL_CALLS]')
        
        # First part (before any tool calls) is regular content
        text_content = parts[0].strip()
        if text_content:
            messages.append(Message(
                role=message.role,
                content=text_content,
                reasoning_content=message.reasoning_content,
                name=message.name,
                extra=message.extra
            ))
        
        # Process each tool call
        for part in parts[1:]:
            if not part.strip():
                continue
                
            # Parse tool call format: tool_name{args}
            tool_call_match = self._extract_tool_call(part)
            if tool_call_match:
                tool_name, tool_args = tool_call_match
                
                # Generate vLLM-compliant call ID: exactly 9 alphanumeric characters
                import hashlib
                hash_input = tool_name + tool_args
                hash_obj = hashlib.md5(hash_input.encode()).hexdigest()
                call_id = hash_obj[:9]  # Take first 9 hex characters (0-9, a-f)
                self.tool_call_id_mapping[tool_name] = call_id
                
                messages.append(Message(
                    role=ASSISTANT,
                    content='',
                    function_call=FunctionCall(
                        name=tool_name,
                        arguments=tool_args
                    ),
                    reasoning_content=message.reasoning_content,
                    name=message.name,
                    extra=message.extra
                ))
        
        return messages if messages else [message]
    
    def _extract_tool_call(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract tool name and arguments from text like: tool_name{args}
        Returns (tool_name, args_json_string) or None if parsing fails.
        """
        text = text.strip()
        
        # Find the first '{' to separate tool name from arguments
        brace_index = text.find('{')
        if brace_index == -1:
            return None
        
        tool_name = text[:brace_index].strip()
        args_text = text[brace_index:].strip()
        
        # Validate that it's proper JSON
        try:
            json.loads(args_text)  # Validate JSON
            return tool_name, args_text
        except (json.JSONDecodeError, ValueError):
            return None


# Export the class
__all__ = ['TextChatAtMistralVllm'] 