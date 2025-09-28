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
from bailingmm_utils import process_ratio
import os

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
        empty_load=False, 
    ):
        super().__init__(config)
        self.config: BailingMM2Config = config
        self.vision = None

        self.llm_dytpe = torch.bfloat16

        if empty_load:
            return

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
        image_gen: Optional[bool] = False,
        image_gen_pixel_values_reference: Optional[torch.FloatTensor] = None,
        image_gen_negative_input_ids: Optional[torch.LongTensor] = None,
        image_gen_negative_attention_mask: Optional[torch.Tensor] = None,
        image_gen_steps: Optional[int] = 30,
        image_gen_seed: Optional[int] = 42,
        image_gen_cfg: Optional[float] = 5.0,
        image_gen_image_cfg: Optional[float] = 2.0,
        image_gen_cfg_mode: Optional[int] = 1,
        image_gen_height: Optional[int] = 512,
        image_gen_width: Optional[int] = 512,
        image_gen_llm_hidden_states:  Optional[torch.LongTensor] = None,
        image_gen_negative_llm_hidden_states:  Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ):
        image_embeds, video_embeds, audio_embeds, audio_embeds_lengths = None, None, None, None

        if image_gen:
            if image_gen_llm_hidden_states is None:
                assert self.model is not None
                assert self.vision is not None
                if pixel_values is not None:
                    image_embeds = self.extract_image_feature(pixel_values, grid_thw=image_grid_thw)
            assert self.loaded_image_gen_modules is True, "please add `load_image_gen=True` in from_pretrained() method"
            assert position_ids is None
            condition_embeds = self.get_condition_embeds_for_image_gen(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                image_embeds=image_embeds, 
                position_ids=position_ids,
                use_cache=use_cache,
                image_grid_thw=image_grid_thw,
                llm_hidden_states=image_gen_llm_hidden_states,
            )
            negative_condition_embeds = self.get_condition_embeds_for_image_gen(
                input_ids=image_gen_negative_input_ids, 
                attention_mask=image_gen_negative_attention_mask,
                image_embeds=image_embeds, 
                position_ids=position_ids,
                use_cache=use_cache,
                image_grid_thw=image_grid_thw,
                llm_hidden_states=image_gen_negative_llm_hidden_states,
            ) if image_gen_negative_input_ids is not None else condition_embeds * 0.0
            if isinstance(image_gen_height, torch.Tensor):
                image_gen_height = int(image_gen_height.cpu().item())
            
            if isinstance(image_gen_width, torch.Tensor):
                image_gen_width = int(image_gen_width.cpu().item())
            closest_size, _ = process_ratio(ori_h=image_gen_height, ori_w=image_gen_width)
            image_gen_height, image_gen_width = closest_size
            if image_gen_seed is None or image_gen_seed < 0:
                from datetime import datetime
                image_gen_seed = datetime.now().microsecond % 1000
            sample_kwargs = {
                "encoder_hidden_states": condition_embeds,
                "steps": image_gen_steps,
                "seed": image_gen_seed,
                "cfg": image_gen_cfg,
                "height": image_gen_height,
                "width": image_gen_width,
                "negative_encoder_hidden_states": negative_condition_embeds,
                "image_cfg": image_gen_image_cfg,
                "cfg_mode": image_gen_cfg_mode,
                "ref_x": image_gen_pixel_values_reference,
            }
            print("image_gen_seed: ", image_gen_seed)
            print("image_gen_steps: ", image_gen_steps)
            print("image_gen_height: ", image_gen_height)
            print("image_gen_width: ", image_gen_width)
              
            image = self.diffusion_loss.sample(
                **sample_kwargs,
            )
            return image

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

    def load_image_gen_modules(self, inference_model_path, torch_dtype=torch.float32, dit_type="sd3"):
        device = torch.device(torch.cuda.current_device())
        if self.model is not None:
            device = self.model.device
        from transformers import AutoModelForCausalLM
        import os
        from safetensors.torch import load_file
        if os.path.exists(inference_model_path):
            temp_state_dict = load_file(os.path.join(inference_model_path, 'mlp', 'model.safetensors'))
        else:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            safetensors_path = hf_hub_download(
                repo_id=inference_model_path,
                filename="model.safetensors",
                subfolder="mlp" 
            )
            with safe_open(safetensors_path, framework="pt") as f:
                temp_state_dict = {key: f.get_tensor(key) for key in f.keys()}
        self.query_tokens_dict = nn.ParameterDict()
        #self.img_gen_scales = [4, 8, 16]
        self.img_gen_scales = [16]
        for scale in self.img_gen_scales:                    
            num_tokens = scale * scale
            scale_name = f"{scale}x{scale}"
            #weights = temp_state_dict[f"query_tokens_dict.{scale_name}"]
            self.query_tokens_dict[scale_name] = nn.Parameter(
                torch.nn.functional.normalize(torch.randn(num_tokens, self.config.llm_config.hidden_size), dim=-1)
            )
        self.query_tokens_dict.to(torch_dtype).to(device)
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
        diffusion_c_input_dim = 2048
        if "sd3" in dit_type:
            from diffusion.sd3_loss import SD3Loss
            self.diffusion_loss = SD3Loss(
                model_path=inference_model_path, 
                scheduler_path=inference_model_path, 
                vision_dim=diffusion_c_input_dim, 
                mlp_state_dict=diffusion_mlp_state_dict,
                torch_dtype=torch_dtype,
            )
        elif "sana" in dit_type:
            from diffusion.sana_loss import SANALoss
            self.diffusion_loss = SANALoss(
                model_path=inference_model_path, 
                scheduler_path=inference_model_path, 
                vision_dim=diffusion_c_input_dim, 
                mlp_state_dict=diffusion_mlp_state_dict,
                torch_dtype=torch_dtype,
            )
        else:
            raise ValueError("unsupported dit type: {}".format(dit_type))
        self.diffusion_loss.to(device)
        #self.norm_query_embeds = True
        # load connector
        self.connector = AutoModelForCausalLM.from_pretrained(inference_model_path, subfolder='connector', torch_dtype=torch_dtype)
        for layer in self.connector.model.layers:
            layer.self_attn.is_causal = False
        self.connector.to(device)
        
        
        self.proj_in = nn.Linear(self.config.llm_config.hidden_size, self.connector.config.hidden_size)
        self.proj_out = nn.Linear(self.connector.config.hidden_size, diffusion_c_input_dim)
        
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
        self.proj_in.to(device)
        self.proj_out.to(device)
        self.loaded_image_gen_modules = True
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        load_image_gen = False
        if "load_image_gen" in kwargs:
            load_image_gen = kwargs["load_image_gen"]
            del kwargs["load_image_gen"]
        dit_type = "sd3"
        if "dit_type" in kwargs:
            dit_type = kwargs["dit_type"]
            del kwargs["dit_type"]
        load_vlm = True
        if "load_vlm" in kwargs:
            load_vlm = kwargs["load_vlm"]
            del kwargs["load_vlm"]
        if load_vlm:
            model = super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                **kwargs,
            )
        else:
            model = cls(
                BailingMM2Config.from_dict(BailingMM2Config.get_config_dict(pretrained_model_name_or_path)[0]),
                empty_load=True,
            )
        if load_image_gen:
            model.load_image_gen_modules(
                pretrained_model_name_or_path, 
                torch_dtype=kwargs["torch_dtype"] if "torch_dtype" in kwargs else torch.float32,
                dit_type=dit_type,
            )
        return model
    
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
            text_ids = torch.cat(
                [
                    text_ids,
                    torch.tensor([[start_token_id]]).to(text_ids.dtype).to(text_ids.device),
                    torch.tensor([[patch_token_id] * (scale**2)])
                    .to(text_ids.dtype)
                    .to(text_ids.device),
                    torch.tensor([[end_token_id]]).to(text_ids.dtype).to(text_ids.device),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.tensor([[1] * ((scale**2) + 2)])
                    .to(attention_mask.dtype)
                    .to(attention_mask.device),
                ],
                dim=1,
            )
            gen_mask = torch.cat(
                [
                    gen_mask,
                    torch.tensor([[0]]).to(gen_mask.dtype).to(gen_mask.device),
                    torch.tensor([[1] * (scale**2)]).to(gen_mask.dtype).to(gen_mask.device),
                    torch.tensor([[0]]).to(gen_mask.dtype).to(gen_mask.device),
                ],
                dim=1,
            )
        assert text_ids.shape == attention_mask.shape
        assert attention_mask.shape == gen_mask.shape
        return text_ids, attention_mask, gen_mask
    
    def get_condition_embeds_for_image_gen(
        self,
        input_ids, 
        attention_mask,
        image_embeds, 
        position_ids,
        use_cache,
        image_grid_thw,
        llm_hidden_states,
    ):
        input_ids, attention_mask, gen_mask = self.append_input_ids_with_multiscale_learnable_tokens(
            input_ids,
            attention_mask,
            self.img_gen_scales,
            self.config.llm_config.image_patch_token + 1,
            self.config.llm_config.image_patch_token + 2,
            self.config.llm_config.image_patch_token,
        )
        if llm_hidden_states is None:
            query_tokens_embeds = torch.cat(
                [self.query_tokens_dict[f"{scale}x{scale}"] for scale in self.img_gen_scales], 
                dim=0,
            )
            if image_embeds is None:
                image_embeds = query_tokens_embeds
            else:
                image_embeds = torch.cat([image_embeds, query_tokens_embeds], dim=0)
            new_image_grid_thw = []
            for scale in self.img_gen_scales:
                new_image_grid_thw.append([1, 2, scale * scale * 2])
            new_image_grid_thw = torch.tensor(new_image_grid_thw, dtype=input_ids.dtype).to(input_ids.device)
            if image_grid_thw is None:
                image_grid_thw = new_image_grid_thw
            else:
                image_grid_thw = torch.cat([image_grid_thw, new_image_grid_thw], dim=0)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if image_embeds is None or input_ids.size(1) == 1:
                    words_embeddings = self.model.get_input_embeddings()(input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1))
                    image_mask = None
                    audio_mask = None
                else:
                    words_embeddings, image_mask, audio_mask = self.model.model.prompt_wrap_navit(
                            input_ids.clip(0, self.model.get_input_embeddings().weight.shape[0] - 1), image_embeds, None, None,
                            None, None, None,  # noqa
                    )
                # if self.config.llm_config.rope_scaling is not None and self.config.llm_config.rope_scaling["type"] == "3D": 
                #     position_ids, _ = self.get_rope_index(
                #         input_ids,
                #         image_token_id=self.config.llm_config.image_patch_token,
                #         video_token_id=self.config.llm_config.image_patch_token,
                #         image_start_token_id=self.config.llm_config.image_start_token,
                #         video_start_token_id=self.config.llm_config.video_start_token,
                #         image_grid_thw=image_grid_thw,
                #         video_grid_thw=None,
                #         attention_mask=attention_mask,
                #     )
                assert input_ids.size(1) == words_embeddings.size(1), "{} vs {}".format(
                    input_ids.size,
                    words_embeddings.size,
                )
                outputs = self.model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=words_embeddings,
                    use_cache=False,
                    image_mask=image_mask,
                    audio_mask=audio_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = llm_hidden_states
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            gen_mask = gen_mask.unsqueeze(-1).expand(gen_mask.shape[0], gen_mask.shape[1], hidden_states.shape[-1]).to(hidden_states.device).bool()
            hidden_states_gen = torch.masked_select(hidden_states, gen_mask).view(hidden_states.shape[0], -1, hidden_states.shape[-1])
            # 分解hidden_states为不同尺度的表示
            scale_start_idxes = [0] + self.scale_indices[:-1]
            scale_end_idxes = self.scale_indices
            assert scale_end_idxes[-1] == hidden_states_gen.shape[1]
            
            scale, scale_start_idx, scale_end_idx = [
                i for i in zip(self.img_gen_scales, scale_start_idxes, scale_end_idxes)
            ][-1]
            
            scale_hidden = hidden_states_gen[:, scale_start_idx : scale_end_idx, :]
            # 处理当前尺度的特征
            scale_embeds = self.proj_in(scale_hidden)
            seq_shape = scale_embeds.shape
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                scale_embeds = self.connector(
                    inputs_embeds=scale_embeds, 
                    attention_mask=torch.ones(seq_shape[0],1,seq_shape[1],seq_shape[1]).to(scale_embeds.device), 
                    output_hidden_states=True
                ).hidden_states[-1]
                
            scale_embeds = self.proj_out(scale_embeds)
            # 归一化
            scale_embeds = torch.nn.functional.normalize(scale_embeds, dim=-1)
            return scale_embeds


__all__ = [
    "BailingMM2NativeForConditionalGeneration"
]