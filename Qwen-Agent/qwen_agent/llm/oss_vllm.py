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
import logging
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.oai import TextChatAtOAI
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, FunctionCall, Message
from qwen_agent.log import logger


@register_llm('oss_vllm')
class TextChatAtOSSVllm(TextChatAtOAI):
    """
    Specialized OpenAI-compatible client for GPT-OSS models served via vLLM.
    Uses native OpenAI tools format:
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given city",
                "parameters": {
                    "type": "object", 
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                },
            },
        }
    ]
    
    Completely relies on the model's native tool calling capabilities.
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # Set default model if not specified
        self.model = self.model or 'openai/gpt-oss-120b'

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
        Override to handle native tools parameter for GPT-OSS models.
        Converts Qwen-Agent function format to OpenAI tools format directly.
        """
        # Convert Qwen-Agent functions to OpenAI tools format
        # Input functions should have: name, description, parameters
        tools = []
        for func in functions:
            tool = {
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {})
                }
            }
            tools.append(tool)
        
        generate_cfg = copy.deepcopy(generate_cfg)
        generate_cfg['tools'] = tools
        generate_cfg['tool_choice'] = 'auto'
        
        # Remove prompt-based function calling parameters
        prompt_params = ['parallel_function_calls', 'function_choice', 'thought_in_content', 'fncall_prompt_type']
        for param in prompt_params:
            generate_cfg.pop(param, None)
        
        # Use parent class methods directly since we're using native format
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream, generate_cfg=generate_cfg)
        else:
            return self._chat_no_stream(messages, generate_cfg=generate_cfg)


# Export the class
__all__ = ['TextChatAtOSSVllm'] 