# coding=utf-8
# Copyright 2022 shunxing1234 and The HuggingFace Inc. team. All rights reserved.
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
""" GLMAudio model configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class WhisperEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        whisper_encoder_config: dict = None,
        ds_kernel_size=3,
        ds_stride=2,
        **kwargs
    ):
        self.whisper_encoder_config = whisper_encoder_config
        self.ds_kernel_size = ds_kernel_size
        self.ds_stride = ds_stride

        super().__init__(
            **kwargs
        )
