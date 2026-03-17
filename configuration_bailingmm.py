# coding=utf-8
# Copyright 2024 ANT Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import PretrainedConfig
from configuration_bailing_moe import BailingMoeConfig
from audio_tokenizer.configuration_audio_vae import AudioVAEconfig


class BailingMMConfig(PretrainedConfig):
    model_type = "bailingmm"

    def __init__(
        self,
        llm_config: BailingMoeConfig = None,
        audio_tokenizer_config: AudioVAEconfig = None,
        ditar_config: dict = None,
        aggregator_config: dict = None,
        model_type: str = None,
        **kwargs
    ):
        self.model_type = model_type
        if self.model_type == 'dense':
            self.llm_config = llm_config
        else:
            self.llm_config = BailingMoeConfig(**llm_config) if isinstance(llm_config, dict) else llm_config
        self.audio_tokenizer_config = AudioVAEconfig(**audio_tokenizer_config) if isinstance(audio_tokenizer_config, dict) else audio_tokenizer_config
        self.ditar_config = ditar_config
        self.aggregator_config = aggregator_config
        super().__init__(**kwargs)