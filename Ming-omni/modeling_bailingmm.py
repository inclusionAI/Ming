#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
from configuration_bailingmm import BailingMMConfig
from modeling_utils import patch_continuous_features, build_modality_mask

# audio encoder
from funasr.models.sanm.encoder import SANMEncoder
from modeling_bailing_moe import BailingMoeForCausalLM
from modeling_utils import Transpose, encode_audio_segments

# vision encoder
from qwen2_5_vit import Qwen2_5_VisionTransformer

# talker
from modeling_bailing_talker import BailingTalkerForConditionalGeneration

# whisper encoder
from modeling_whisper_encoder import WhisperAudioEncoder

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BailingMMConfig"


@dataclass
class BailingMMCausalLMOutputWithPast(ModelOutput):
    """
    Base class for BailingMM causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class BailingMMNativeForConditionalGeneration(PreTrainedModel):
    config_class = BailingMMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingAudioModel"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def __init__(
        self,
        config: BailingMMConfig,
    ):
        super().__init__(config)
        self.config: BailingMMConfig = config
        self.vision = None
        self.audio = None
        self.whisper_encoder = None
        self.talker = None

        self.llm_dytpe = torch.bfloat16

        if self.config.vision_config:
            self.vision = Qwen2_5_VisionTransformer(self.config.vision_config)

        if self.config.audio_config:
            self.audio = SANMEncoder(**self.config.audio_config.audio_encoder_config_sanm)

        if self.config.whisper_config:
            self.whisper_encoder = WhisperAudioEncoder(**self.config.whisper_config.whisper_encoder_config)

        self.model = BailingMoeForCausalLM(self.config.llm_config)

        mlp_modules_img = [nn.Linear(self.vision.image_emb_dim, self.model.config.hidden_size)]
        for _ in range(1, self.config.mlp_depth):
            mlp_modules_img.append(nn.GELU())
            mlp_modules_img.append(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size))
        self.linear_proj = nn.Sequential(*mlp_modules_img)

        if self.audio:
            audio_encoder_proj = torch.nn.Conv1d(
                self.config.audio_config.audio_encoder_output_size,
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

        if self.whisper_encoder:
            whisper_encoder_proj = torch.nn.Conv1d(
                self.whisper_encoder.audio_emb_dim,
                self.model.config.hidden_size,
                kernel_size=self.config.whisper_config.ds_kernel_size,
                stride=self.config.whisper_config.ds_stride,
                padding=self.config.whisper_config.ds_kernel_size // 2,
            )

            mlp_modules_whisper = [whisper_encoder_proj, Transpose(-1, -2)]
            for _ in range(1, self.config.mlp_depth):
                mlp_modules_whisper.append(nn.GELU())
                mlp_modules_whisper.append(nn.Linear(
                    self.model.config.hidden_size, self.model.config.hidden_size
                ))
            mlp_modules_whisper.append(Transpose(-1, -2))  # Revert to a conv-style permutation.
            self.linear_proj_whisper = nn.Sequential(*mlp_modules_whisper)

        if self.config.talker_config:
            self.config.talker_config._name_or_path = f'{self.config._name_or_path}/talker'
            self.talker = BailingTalkerForConditionalGeneration(self.config.talker_config)
        self.post_init()
        self.loaded_image_gen_modules = False

    def extract_image_feature(self, pixel_values, grid_thw):
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
            image_embeds = image_embeds.float()
            image_embeds = self.linear_proj(image_embeds)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds
    
    def extract_audio_feature(self, audio_feats, audio_feats_lengths, use_whisper_encoder=False):
        if not use_whisper_encoder:
            assert self.audio is not None
            assert self.linear_proj_audio is not None
            encoder = self.audio
            proj_layer = self.linear_proj_audio
        else:
            assert self.whisper_encoder is not None
            assert self.linear_proj_whisper is not None
            encoder = self.whisper_encoder
            proj_layer = self.linear_proj_whisper
        audio_embeds, _, audio_embeds_lengths = encode_audio_segments(
            encoder=encoder,
            proj_layer=proj_layer,
            wav_feats=audio_feats,
            wav_feats_lengths=audio_feats_lengths,
            audio_config=self.config.audio_config,
            whisper_config=self.config.whisper_config,
            use_whisper_encoder=use_whisper_encoder
        )
        if self.config.audio_config.norm_query_embeds:
            audio_embeds = F.normalize(audio_embeds, dim=2)  # [-1, 256, 2048]
        return audio_embeds.to(audio_feats.dtype), audio_embeds_lengths

    def prompt_wrap_vision(self, input_ids, inputs_embeds, vision_embeds, image_token_id=None):
        if vision_embeds is None or input_ids is None:
            return inputs_embeds

        if len(vision_embeds.shape) == 3:
            vision_embeds = vision_embeds.reshape(-1, vision_embeds.shape[-1])

        self.config.llm_config.image_patch_token = image_token_id if image_token_id is not None else self.config.llm_config.image_patch_token
        n_image_tokens = (input_ids == self.config.llm_config.image_patch_token).sum().item()
        n_image_features = vision_embeds.shape[0]

        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        image_router_mask =  (
            (input_ids == self.config.llm_config.image_patch_token)
            .unsqueeze(-1)
            .to(inputs_embeds.device)
        ) 
        image_mask = image_router_mask.expand_as(inputs_embeds)
        image_embeds = vision_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        image_router_mask = image_router_mask.squeeze(-1)
        return inputs_embeds, image_router_mask

    def prompt_wrap_audio(self, input_ids, inputs_embeds, audio_embeds, audio_embeds_lengths, placeholder_audio_loc_lens):
        inputs_embeds = patch_continuous_features(
           input_embeddings=inputs_embeds, placeholder_loc_lens=placeholder_audio_loc_lens,
           encoded_feats=audio_embeds, encoded_feat_lens=audio_embeds_lengths,
        )
        audio_router_mask = build_modality_mask(placeholder_audio_loc_lens, inputs_embeds.shape[:-1]).to(inputs_embeds.device)
        return inputs_embeds, audio_router_mask
     
    def prompt_wrap_navit(self, input_ids, query_embeds_image=None, query_embeds_video=None, query_embeds_audio=None,
        query_embeds_audio_lengths=None, placeholder_audio_loc_lens=None, target_embeds=None):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if query_embeds_image is None and query_embeds_video is None and query_embeds_audio is None and target_embeds is None:
            return inputs_embeds

        image_mask = None
        audio_mask = None
        if query_embeds_image is not None:
            inputs_embeds, image_mask = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_image)
        if query_embeds_video is not None:
            inputs_embeds, image_mask = self.prompt_wrap_vision(input_ids, inputs_embeds, query_embeds_video)
        if query_embeds_audio is not None:
            inputs_embeds, audio_mask = self.prompt_wrap_audio(
                input_ids, inputs_embeds, query_embeds_audio, query_embeds_audio_lengths, placeholder_audio_loc_lens,
            )
        return inputs_embeds, image_mask, audio_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        audio_feats: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_whisper_encoder: bool = False
    ) -> Union[Tuple, BailingMMCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if (pixel_values is not None or pixel_values_videos is not None or audio_feats is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both pixel_values/pixel_values_videos/pixel_values_audios and inputs_embeds at the same time, and must specify either one"
            )
        
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)
        if audio_feats is not None:
            audio_embeds, audio_embeds_lengths = self.extract_audio_feature(audio_feats, audio_feats_lengths, use_whisper_encoder=use_whisper_encoder)

        if (image_embeds is None and video_embeds is None and audio_embeds is None) or input_ids.size(1) == 1:
            words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
            image_mask = None
            audio_mask = None

        else:
            words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                    input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, video_embeds, audio_embeds,
                    audio_embeds_lengths, audio_placeholder_loc_lens, None,  # noqa
            )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=words_embeddings,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_mask=image_mask,
            audio_mask=audio_mask,
        )

        return BailingMMCausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def append_input_ids_with_multiscale_learnable_tokens(   
        self, 
        text_ids,
        attention_mask,
        scales,
        start_token_id,
        end_token_id,
        patch_token_id,
    ):
        assert text_ids.shape[0] == 1
        assert attention_mask.shape == text_ids.shape
        gen_mask = torch.zeros_like(attention_mask)
        for scale in scales:
            text_ids = torch.cat([
                text_ids, 
                torch.tensor([[start_token_id]]).to(text_ids.dtype).to(text_ids.device),
                torch.tensor([[patch_token_id] * (scale ** 2)]).to(text_ids.dtype).to(text_ids.device),
                torch.tensor([[end_token_id]]).to(text_ids.dtype).to(text_ids.device),
            ], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.tensor([[1] * ((scale ** 2) + 2)]).to(attention_mask.dtype).to(attention_mask.device),
            ], dim=1)
            gen_mask = torch.cat([
                gen_mask, 
                torch.tensor([[0]]).to(gen_mask.dtype).to(gen_mask.device),
                torch.tensor([[1] * (scale ** 2)]).to(gen_mask.dtype).to(gen_mask.device),
                torch.tensor([[0]]).to(gen_mask.dtype).to(gen_mask.device),
            ], dim=1)
        assert text_ids.shape == attention_mask.shape
        assert attention_mask.shape == gen_mask.shape
        return text_ids, attention_mask, gen_mask

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
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        audio_feats_lengths: Optional[torch.LongTensor] = None,
        audio_placeholder_loc_lens: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        image_gen: Optional[bool] = False,
        image_gen_steps: Optional[int] = 30,
        image_gen_seed: Optional[int] = 0,
        image_gen_cfg: Optional[float] = 3.5,
        image_gen_height: Optional[int] = 512,
        image_gen_width: Optional[int] = 512,
        **generate_kwargs,
    ):
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None
        if pixel_values is not None:
            image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
        if pixel_values_videos is not None:
            video_embeds = self.extract_image_feature(pixel_values_videos, grid_thw=video_grid_thw)

        if image_gen:
            assert self.loaded_image_gen_modules is True
            input_ids, attention_mask, gen_mask = self.append_input_ids_with_multiscale_learnable_tokens(
                input_ids,
                attention_mask,
                [4, 8, 16], #self.img_gen_scales,
                self.config.llm_config.image_patch_token + 1,
                self.config.llm_config.image_patch_token + 2,
                self.config.llm_config.image_patch_token,
            )
            query_tokens_embeds = torch.cat(
                [self.query_tokens_dict[f"{scale}x{scale}"] for scale in self.img_gen_scales], 
                dim=0,
            )
            if image_embeds is None:
                image_embeds = query_tokens_embeds
            else:
                image_embeds = torch.cat([image_embeds, query_tokens_embeds], dim=0)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                assert video_embeds is None and audio_embeds is None
                if (image_embeds is None and video_embeds is None and audio_embeds is None) or input_ids.size(1) == 1:
                    words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
                    image_mask = None
                    audio_mask = None
                else:
                    words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                            input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, video_embeds, audio_embeds,
                            audio_embeds_lengths, audio_placeholder_loc_lens, None,  # noqa
                    )
                outputs = self.model.forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=words_embeddings,
                    use_cache=use_cache,
                    image_mask=image_mask,
                    audio_mask=audio_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
                gen_mask = gen_mask.unsqueeze(-1).expand(gen_mask.shape[0], gen_mask.shape[1], hidden_states.shape[-1]).to(hidden_states.device).bool()
                hidden_states_gen = torch.masked_select(hidden_states, gen_mask).view(hidden_states.shape[0], -1, hidden_states.shape[-1])
                # 分解hidden_states为不同尺度的表示
                scale_start_idxes = [0] + self.scale_indices[:-1]
                scale_end_idxes = self.scale_indices
                assert scale_end_idxes[-1] == hidden_states_gen.shape[1]
                new_query_embeds_images = {}
                for scale, scale_start_idx, scale_end_idx in [
                    i for i in zip(self.img_gen_scales, scale_start_idxes, scale_end_idxes)
                ][-1:]:   
                    scale_name = f"{scale}x{scale}"
                    scale_hidden = hidden_states_gen[:, scale_start_idx : scale_end_idx, :]
                    
                    # 处理当前尺度的特征
                    scale_embeds = self.proj_in(scale_hidden)
                    seq_shape = scale_embeds.shape
                    #print("scale: {}, seq_shape: {}".format(scale, seq_shape))
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        scale_embeds = self.connector(
                            inputs_embeds=scale_embeds, 
                            attention_mask=torch.ones(seq_shape[0],1,seq_shape[1],seq_shape[1]).to(scale_embeds.device), 
                            output_hidden_states=True
                        ).hidden_states[-1]
                    scale_embeds = self.proj_out(scale_embeds)
                    
                    # 归一化
                    scale_embeds = torch.nn.functional.normalize(scale_embeds, dim=-1)
                    new_query_embeds_images[scale_name] = scale_embeds
                
                imgs = []
                for scale in self.img_gen_scales[-1:]:
                    imgs.append(
                        self.diffusion_loss.sample(
                            new_query_embeds_images[f"{scale}x{scale}"], 
                            steps=image_gen_steps, 
                            seed=image_gen_seed, 
                            cfg=image_gen_cfg, 
                            height=image_gen_height, 
                            width=image_gen_width
                        )
                    )
                return imgs[-1] 
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if audio_feats is not None:
                use_whisper_encoder = generate_kwargs.pop('use_whisper_encoder', False)
                audio_embeds, audio_embeds_lengths = self.extract_audio_feature(audio_feats, audio_feats_lengths,
                                                                                use_whisper_encoder=use_whisper_encoder)
            if (image_embeds is None and video_embeds is None and audio_embeds is None) or input_ids.size(1) == 1:
                words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
                image_mask = None
                audio_mask = None
            else:
                words_embeddings, image_mask, audio_mask = self.prompt_wrap_navit(
                        input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, video_embeds, audio_embeds,
                        audio_embeds_lengths, audio_placeholder_loc_lens, None,  # noqa
                )

            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=words_embeddings,
                use_cache=use_cache,
                image_mask=image_mask,
                audio_mask=audio_mask,
                **generate_kwargs,
            )
        return outputs

    def load_image_gen_modules(self, inference_model_path):
        from transformers import AutoModelForCausalLM
        from diffusion.sana_loss import SANALoss
        import os
        from safetensors.torch import load_file
        
        temp_state_dict = load_file(os.path.join(inference_model_path, 'mlp', 'model.safetensors'))
        
        self.query_tokens_dict = nn.ParameterDict()
        self.img_gen_scales = [4, 8, 16]
        for scale in self.img_gen_scales:                    
            num_tokens = scale * scale
            scale_name = f"{scale}x{scale}"
            #weights = temp_state_dict[f"query_tokens_dict.{scale_name}"]
            self.query_tokens_dict[scale_name] = nn.Parameter(
                torch.nn.functional.normalize(torch.randn(num_tokens, self.model.config.hidden_size), dim=-1)
            )
        self.query_tokens_dict.to(self.model.dtype).to(self.model.device)
        modified_state_dict_query_tokens = {
            f"{scale}x{scale}": temp_state_dict[f"query_tokens_dict.{scale}x{scale}"]
            for scale in self.img_gen_scales   
        }
        self.query_tokens_dict.load_state_dict(modified_state_dict_query_tokens, strict=True)
        # 计算各尺度的累积索引
        self.scale_indices = []
        current_idx = 0
        for scale in self.img_gen_scales:
            current_idx += scale * scale
            self.scale_indices.append(current_idx)
        
        diffusion_mlp_state_dict = {
            key[len("mlp.") :] : temp_state_dict[key]
            for key in temp_state_dict if key.startswith("mlp.")
        }
        self.diffusion_loss = SANALoss(
            model_path=inference_model_path, 
            scheduler_path=inference_model_path, 
            vision_dim=self.model.config.hidden_size, 
            #mlp_checkpoint_path=os.path.join(inference_model_path, 'mlp', 'model.safetensors'),
            mlp_state_dict=diffusion_mlp_state_dict,
            trainable_params="None",
        )
        self.diffusion_loss.to(self.model.device)
        #self.norm_query_embeds = True
        # load connector
        self.connector = AutoModelForCausalLM.from_pretrained(os.path.join(inference_model_path, 'connector'))
        for layer in self.connector.model.layers:
            layer.self_attn.is_causal = False
        self.connector.to(self.model.device)
        
        self.proj_in = nn.Linear(self.model.config.hidden_size, self.connector.config.hidden_size)
        self.proj_out = nn.Linear(self.connector.config.hidden_size, self.model.config.hidden_size)
        
        modified_state_dict_in = {
            'weight': temp_state_dict['proj_in.weight'],
            'bias': temp_state_dict['proj_in.bias']
        }
        self.proj_in.load_state_dict(modified_state_dict_in, strict=True)
        modified_state_dict_out = {
            'weight': temp_state_dict['proj_out.weight'],
            'bias': temp_state_dict['proj_out.bias']
        }
        self.proj_out.load_state_dict(modified_state_dict_out, strict=True)
        self.proj_in.to(self.model.device)
        self.proj_out.to(self.model.device)
        self.loaded_image_gen_modules = True