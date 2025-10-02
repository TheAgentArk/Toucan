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
import re
from typing import List, Literal, Union

import json5

from qwen_agent.llm.fncall_prompts.base_fncall_prompt import BaseFnCallPrompt
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, SYSTEM, USER, TOOL, ContentItem, FunctionCall, Message
from qwen_agent.log import logger


class KimiFnCallPrompt(BaseFnCallPrompt):

    def preprocess_fncall_messages(self,
                                   messages: List[Message],
                                   functions: List[dict],
                                   lang: Literal['en', 'zh'],
                                   parallel_function_calls: bool = True,
                                   function_choice: Union[Literal['auto'], str] = 'auto',
                                   **kwargs) -> List[Message]:
        del lang  # ignored
        del parallel_function_calls  # ignored
        if function_choice != 'auto':
            raise NotImplementedError

        ori_messages = messages

        # Convert function calls to Kimi's tool call format
        messages = []
        for msg in copy.deepcopy(ori_messages):
            role, content, reasoning_content = msg.role, msg.content, msg.reasoning_content
            if role in (SYSTEM, USER, TOOL):
                messages.append(msg)
            elif role == ASSISTANT:
                content = (content or [])
                fn_call = msg.function_call
                if fn_call:
                    # Convert function call to Kimi's tool call format
                    try:
                        arguments = json5.loads(fn_call.arguments)
                    except Exception:
                        logger.warning('Invalid json tool-calling arguments')
                        arguments = {}
                    
                    # Generate tool call ID in Kimi K2 format: functions.function_name:number
                    call_number = abs(hash(str(arguments))) % 10000
                    tool_call_id = f"functions.{fn_call.name}:{call_number}"
                    
                    # Format as Kimi tool call - only include function arguments (not function name)
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
                    tool_call_text = f"<|tool_calls_section_begin|>\n<|tool_call_begin|>{tool_call_id}<|tool_call_argument_begin|>{arguments_str}<|tool_call_end|>\n<|tool_calls_section_end|>"
                    
                    content.append(ContentItem(text=tool_call_text))
                
                if messages and messages[-1].role == ASSISTANT:
                    if messages[-1].content and messages[-1].content[-1].text and (
                            not messages[-1].content[-1].text.endswith('\n')):
                        messages[-1].content.append(ContentItem(text='\n'))
                    messages[-1].content.extend(content)
                else:
                    messages.append(Message(role=role, content=content, reasoning_content=reasoning_content))
            elif role == FUNCTION:
                assert isinstance(content, list)
                assert len(content) == 1
                # Use function name from the message if available
                function_name = getattr(msg, 'name', 'unknown_function')
                # Format tool response for Kimi using TOOL role
                fc_response = f"## Return of {tool_call_id}\n{content[0].text}"
                content = [ContentItem(text=fc_response)]
                # Create a TOOL message for proper handling
                messages.append(Message(role=TOOL, content=content, name=function_name))
            else:
                raise TypeError

        # Add tool declarations to system message if functions are provided
        if functions:
            tool_descs = []
            for f in functions:
                tool_desc = {
                    "type": "function",
                    "function": {
                        "name": f.get('name_for_model', f.get('name', '')),
                        "description": f.get('description', ''),
                        "parameters": f.get('parameters', {})
                    }
                }
                tool_descs.append(tool_desc)
            
            tool_system = f"<|im_system|>tool_declare<|im_middle|>{json.dumps(tool_descs, ensure_ascii=False)}<|im_end|>"
            
            if messages and messages[0].role == SYSTEM:
                messages[0].content.append(ContentItem(text='\n\n' + tool_system))
            else:
                messages = [Message(role=SYSTEM, content=[ContentItem(text=tool_system)])] + messages
        
        return messages

    def postprocess_fncall_messages(
        self,
        messages: List[Message],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
        thought_in_content: bool = False,
    ) -> List[Message]:
        if function_choice != 'auto':
            raise NotImplementedError
        
        new_messages = []
        for msg in messages:
            role, content, reasoning_content, extra = msg.role, msg.content, msg.reasoning_content, msg.extra
            assert isinstance(content, list)

            if role in (SYSTEM, USER, TOOL):
                new_messages.append(
                    Message(role=role, content=content, reasoning_content=reasoning_content, extra=extra))
                continue

            # Handle reasoning content
            if reasoning_content:
                new_messages.append(Message(role=role, content='', reasoning_content=reasoning_content, extra=extra))

            new_content = []
            for item in content:
                item_type, item_text = item.get_type_and_value()

                if item_type != 'text':  # multimodal
                    new_content.append(item)
                    continue

                # Look for Kimi tool call patterns
                if '<|tool_calls_section_begin|>' in item_text:
                    # Split content before and after tool calls
                    parts = item_text.split('<|tool_calls_section_begin|>')
                    pre_text = parts[0].strip()
                    if pre_text:
                        new_content.append(ContentItem(text=pre_text))
                    
                    # Process tool calls section
                    if len(parts) > 1:
                        tool_section = parts[1]
                        if '<|tool_calls_section_end|>' in tool_section:
                            tool_calls_text = tool_section.split('<|tool_calls_section_end|>')[0]
                            
                            # Extract individual tool calls
                            tool_call_pattern = r'<\|tool_call_begin\|>([^<]*)<\|tool_call_argument_begin\|>([^<]*)<\|tool_call_end\|>'
                            matches = re.findall(tool_call_pattern, tool_calls_text)
                            
                            for tool_call_id, arguments_str in matches:
                                if new_content:
                                    new_messages.append(Message(
                                        role=role,
                                        content=new_content,
                                        extra=extra,
                                    ))
                                    new_content = []
                                
                                try:
                                    # Parse function arguments directly
                                    arguments = json5.loads(arguments_str)
                                    arguments_json = json.dumps(arguments, ensure_ascii=False)
                                except Exception:
                                    logger.warning(f'Invalid json tool-calling arguments: {arguments_str}')
                                    arguments_json = arguments_str
                                
                                # Extract function name from tool_call_id (Kimi K2 format: functions.function_name:number)
                                fn_name = 'unknown_function'
                                if tool_call_id.startswith('functions.') and ':' in tool_call_id:
                                    # Extract function_name from "functions.function_name:number"
                                    fn_name = tool_call_id.split('.')[1].split(':')[0]
                                else:
                                    logger.warning(f'Invalid Kimi K2 tool_call_id format: {tool_call_id}')
                                
                                new_messages.append(
                                    Message(
                                        role=ASSISTANT,
                                        content=[],
                                        function_call=FunctionCall(
                                            name=fn_name,
                                            arguments=arguments_json,
                                        ),
                                        extra=extra,
                                    ))
                else:
                    # No tool calls, just regular content
                    if item_text.strip():
                        new_content.append(ContentItem(text=item_text))

            if new_content:
                new_messages.append(Message(role=role, content=new_content, extra=extra))
        
        return new_messages


# Kimi-specific template patterns
KIMI_TOOL_CALL_BEGIN = '<|tool_calls_section_begin|>'
KIMI_TOOL_CALL_END = '<|tool_calls_section_end|>'
KIMI_INDIVIDUAL_CALL_BEGIN = '<|tool_call_begin|>'
KIMI_INDIVIDUAL_CALL_ARG_BEGIN = '<|tool_call_argument_begin|>'
KIMI_INDIVIDUAL_CALL_END = '<|tool_call_end|>'

# Function to remove incomplete Kimi special tokens when streaming
def remove_incomplete_special_tokens(text: str) -> str:
    kimi_patterns = [
        '<|tool_calls_section_begin|>',
        '<|tool_call_begin|>',
        '<|tool_call_argument_begin|>',
        '<|tool_call_end|>',
        '<|tool_calls_section_end|>'
    ]
    
    for pattern in kimi_patterns:
        if text in pattern and text != pattern:
            text = ''
            break
    
    return text


def extract_kimi_tool_calls(text: str):
    """Extract tool calls from Kimi format text"""
    tool_calls = []
    if KIMI_TOOL_CALL_BEGIN in text and KIMI_TOOL_CALL_END in text:
        # Extract the tool calls section
        start = text.find(KIMI_TOOL_CALL_BEGIN) + len(KIMI_TOOL_CALL_BEGIN)
        end = text.find(KIMI_TOOL_CALL_END)
        tool_section = text[start:end]
        
        # Use regex to find individual tool calls
        pattern = rf'{re.escape(KIMI_INDIVIDUAL_CALL_BEGIN)}([^<]*){re.escape(KIMI_INDIVIDUAL_CALL_ARG_BEGIN)}([^<]*){re.escape(KIMI_INDIVIDUAL_CALL_END)}'
        matches = re.findall(pattern, tool_section)
        
        for tool_id, arguments in matches:
            try:
                args_dict = json5.loads(arguments)
                tool_calls.append({
                    'id': tool_id.strip(),
                    'arguments': args_dict
                })
            except Exception:
                logger.warning(f'Failed to parse tool call arguments: {arguments}')
    
    return tool_calls
