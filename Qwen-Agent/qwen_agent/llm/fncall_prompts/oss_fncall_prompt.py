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
from typing import List, Literal, Union

from qwen_agent.llm.fncall_prompts.base_fncall_prompt import BaseFnCallPrompt
from qwen_agent.llm.schema import Message

class OSSFnCallPrompt(BaseFnCallPrompt):
    """
    Prompt class for OSS models (e.g., GPT-OSS) using native OpenAI tools format.
    
    Similar to Mistral implementation, OSS models handle tool formatting natively
    via the OpenAI-compatible API, so we don't need to inject system prompts
    or tool descriptions manually. The model uses native tool calling capabilities.
    """
    
    def preprocess_fncall_messages(self,
                                   messages: List[Message],
                                   functions: List[dict],
                                   lang: Literal['en', 'zh'],
                                   parallel_function_calls: bool = True,
                                   function_choice: Union[Literal['auto'], str] = 'auto',
                                   **kwargs) -> List[Message]:
        del lang  # ignored - OSS models handle language automatically
        del parallel_function_calls  # ignored - handled natively by the model
        del functions  # handled by OSS model via OpenAI tools parameter
        if function_choice != 'auto':
            raise NotImplementedError("OSS models currently only support function_choice='auto'")

        # For OSS models, just pass through messages as-is
        # The model will handle tool formatting automatically
        # when tools are passed via the API's 'tools' parameter
        return copy.deepcopy(messages)

    def postprocess_fncall_messages(
        self,
        messages: List[Message],
        parallel_function_calls: bool = True,
        function_choice: Union[Literal['auto'], str] = 'auto',
        thought_in_content: bool = False,
        **kwargs
    ) -> List[Message]:
        if function_choice != 'auto':
            raise NotImplementedError("OSS models currently only support function_choice='auto'")
        
        # For OSS models, no postprocessing needed
        # Tool calls are handled natively by the model using OpenAI format
        return messages 