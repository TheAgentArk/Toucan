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
from typing import Union

from .azure import TextChatAtAzure
from .base import LLM_REGISTRY, BaseChatModel, ModelServiceError
from .mistral_vllm import TextChatAtMistralVllm
from .oss_vllm import TextChatAtOSSVllm
from .oai import TextChatAtOAI
from .openvino import OpenVINO
from .transformers_llm import Transformers
from .qwen_dashscope import QwenChatAtDS
from .qwenaudio_dashscope import QwenAudioChatAtDS
from .qwenomni_oai import QwenOmniChatAtOAI
from .qwenvl_dashscope import QwenVLChatAtDS
from .qwenvl_oai import QwenVLChatAtOAI


def get_chat_model(cfg: Union[dict, str] = 'qwen-plus') -> BaseChatModel:
    """The interface of instantiating LLM objects.

    Args:
        cfg: The LLM configuration, one example is:
          cfg = {
              # Use the model service provided by DashScope:
              'model': 'qwen-max',
              'model_server': 'dashscope',

              # Use your own model service compatible with OpenAI API:
              # 'model': 'Qwen',
              # 'model_server': 'http://127.0.0.1:7905/v1',

              # Use GPT-OSS models with native tool calling:
              # 'model': 'openai/gpt-oss-120b',
              # 'model_type': 'oss_vllm',
              # 'model_server': 'http://127.0.0.1:8000/v1',

              # Use GPT-OSS models with qwen tool calling:
              # 'model': 'openai/gpt-oss-120b',
              # 'model_type': 'oai',
              # 'model_server': 'http://127.0.0.1:8000/v1',

              # Use Mistral models with native tool calling:
              # 'model': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
              # 'model_type': 'mistral_vllm',
              # 'model_server': 'http://127.0.0.1:8000/v1',

              # (Optional) LLM hyper-parameters:
              'generate_cfg': {
                  'top_p': 0.8,
                  'max_input_tokens': 6500,
                  'max_retries': 10,
              }
          }

    Returns:
        LLM object.
    """
    if isinstance(cfg, str):
        cfg = {'model': cfg}

    if 'model_type' in cfg:
        model_type = cfg['model_type']
        if model_type in LLM_REGISTRY:
            if model_type in ('oai', 'qwenvl_oai'):
                if cfg.get('model_server', '').strip() == 'dashscope':
                    cfg = copy.deepcopy(cfg)
                    cfg['model_server'] = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
            return LLM_REGISTRY[model_type](cfg)
        else:
            raise ValueError(f'Please set model_type from {str(LLM_REGISTRY.keys())}')

    # Deduce model_type from model and model_server if model_type is not provided:

    if 'azure_endpoint' in cfg:
        model_type = 'azure'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'model_server' in cfg:
        if cfg['model_server'].strip().startswith('http'):
            # Check if it's a GPT-OSS model
            model = cfg.get('model', '')
            if 'gpt-oss' in model.lower():
                model_type = 'oss_vllm'
                cfg['model_type'] = model_type
                return LLM_REGISTRY[model_type](cfg)
            # Check if it's a Mistral model
            if any(pattern in model.lower() for pattern in ['mistral-small', 'devstral-small']):
                model_type = 'mistral_vllm'
                cfg['model_type'] = model_type
                return LLM_REGISTRY[model_type](cfg)
            else:
                model_type = 'oai'
                cfg['model_type'] = model_type
                return LLM_REGISTRY[model_type](cfg)

    model = cfg.get('model', '')

    if '-vl' in model.lower():
        model_type = 'qwenvl_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if '-audio' in model.lower():
        model_type = 'qwenaudio_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    if 'qwen' in model.lower():
        model_type = 'qwen_dashscope'
        cfg['model_type'] = model_type
        return LLM_REGISTRY[model_type](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')


__all__ = [
    'BaseChatModel',
    'QwenChatAtDS',
    'TextChatAtOAI',
    'TextChatAtOSSVllm',
    'TextChatAtMistralVllm',
    'TextChatAtAzure',
    'QwenVLChatAtDS',
    'QwenVLChatAtOAI',
    'QwenAudioChatAtDS',
    'QwenOmniChatAtOAI',
    'OpenVINO',
    'Transformers',
    'get_chat_model',
    'ModelServiceError',
]
