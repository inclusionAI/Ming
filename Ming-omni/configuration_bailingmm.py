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
from qwen2_5_vit import Qwen2_5_VLVisionConfig
from configuration_audio import GLMAudioConfig
from configuration_bailing_moe import BailingMoeConfig
from configuration_bailing_talker import BailingTalkerConfig
from configuration_whisper_encoder import WhisperEncoderConfig


class BailingMMConfig(PretrainedConfig):
    model_type = "bailingmm"

    def __init__(
        self,
        mlp_depth=1,
        llm_config: BailingMoeConfig = None,
        vision_config: Qwen2_5_VLVisionConfig = None,
        audio_config: GLMAudioConfig = None,
        whisper_config: WhisperEncoderConfig = None,
        talker_config: BailingTalkerConfig = None,
        **kwargs
    ):
        self.audio_config = GLMAudioConfig(**audio_config) if isinstance(audio_config, dict) else audio_config
        self.vision_config = Qwen2_5_VLVisionConfig(**vision_config) if isinstance(vision_config, dict) else vision_config
        self.llm_config = BailingMoeConfig(**llm_config) if isinstance(llm_config, dict) else llm_config
        self.mlp_depth = mlp_depth
        self.talker_config = BailingTalkerConfig(**talker_config) if isinstance(talker_config, dict) else talker_config
        self.whisper_config = WhisperEncoderConfig(**whisper_config) if isinstance(whisper_config, dict) else whisper_config
        super().__init__(**kwargs)
