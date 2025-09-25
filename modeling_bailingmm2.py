#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from modeling_bailing_talker import BailingTalkerForConditionalGeneration
from modeling_whisper_encoder import WhisperAudioEncoder
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from configuration_bailingmm2 import BailingMM2Config
from modeling_bailing_moe_v2 import BailingMoeV2ForCausalLM
from modeling_utils import Transpose, encode_audio_segments, patch_continuous_features, build_modality_mask


# vision encoder
from qwen2_5_vit import Qwen2_5_VisionTransformer

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMM2Config"


class BailingMM2NativeForConditionalGeneration(PreTrainedModel):
    config_class = BailingMM2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: BailingMM2Config,
    ):
        super().__init__(config)
        self.config: BailingMM2Config = config
        self.vision = None

        self.llm_dytpe = torch.bfloat16

        if self.config.vision_config:
            self.vision = Qwen2_5_VisionTransformer(self.config.vision_config)

        if self.config.audio_config:
            self.audio = WhisperAudioEncoder(**self.config.audio_config.whisper_encoder_config)

        self.model = BailingMoeV2ForCausalLM(self.config.llm_config)

        mlp_modules_img = [nn.Linear(self.vision.image_emb_dim, self.model.config.hidden_size)]
        for _ in range(1, self.config.mlp_depth):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)

        if self.audio:
            audio_encoder_proj = torch.nn.Conv1d(
                self.audio.audio_emb_dim,
                self.model.config.hidden_size,
                kernel_size=self.config.audio_config.ds_kernel_size,
                stride=self.config.audio_config.ds_stride,
                padding=self.config.audio_config.ds_kernel_size // 2,
            )

            mlp_modules_audio = [audio_encoder_proj, Transpose(-1, -2)]
            for _ in range(1, self.config.mlp_depth):
                mlp_modules_audio.append(nn.GELU())
                mlp_modules_audio.append(nn.Linear(
                    self.model.config.hidden_size, self.model.config.hidden_size
                ))
            mlp_modules_audio.append(Transpose(-1, -2))
            self.linear_proj_audio = nn.Sequential(*mlp_modules_audio)

        # if self.config.talker_config:
        #     self.config.talker_config._name_or_path = f'{self.config._name_or_path}/talker'
        #     self.talker = BailingTalkerForConditionalGeneration(self.config.talker_config)
        self.post_init()


    def extract_image_feature(self, pixel_values, grid_thw):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
        image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds


    def extract_audio_feature(self, audio_feats, audio_feats_lengths, use_whisper_encoder=False):
        audio_embeds, _, audio_embeds_lengths = encode_audio_segments(
            encoder=self.audio,
            proj_layer=self.linear_proj_audio,
            wav_feats=audio_feats,
            wav_feats_lengths=audio_feats_lengths,
            audio_config=self.config.audio_config
        )
        if self.config.audio_config.norm_query_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=2)  # [-1, 256, 2048]
        return audio_embeds.to(audio_feats.dtype), audio_embeds_lengths

        
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        num_logits_to_keep: Optional[int] = 0,
        **generate_kwargs,
    ):
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if audio_feats is not None:
                audio_embeds, audio_embeds_lengths = self.extract_audio_feature(audio_feats, audio_feats_lengths, use_whisper_encoder=True)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.model.generate(
                input_ids=input_ids,
                query_embeds_image=image_embeds,
                query_embeds_video=video_embeds,
                query_embeds_audio=audio_embeds,
                query_embeds_audio_lengths=audio_embeds_lengths,
                placeholder_audio_loc_lens=audio_placeholder_loc_lens,
                image_grid_thw=image_grid_thw,
                image_grid_thw_video=video_grid_thw,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                num_logits_to_keep=num_logits_to_keep,
                **generate_kwargs,
            )
        return outputs


__all__ = [
    "BailingMM2NativeForConditionalGeneration"
]