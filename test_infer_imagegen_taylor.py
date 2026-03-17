import os
import torch
import time
import numpy as np
from bisect import bisect_left

from tqdm import tqdm

from transformers import (
    AutoProcessor,
)

from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

import warnings

warnings.filterwarnings("ignore")

from IPython import embed

import json
from PIL import Image


# def split_model():
#     device_map = {}
#     world_size = torch.cuda.device_count()
#     num_layers = 32
#     layer_per_gpu = num_layers // world_size
#     layer_per_gpu = [i * layer_per_gpu - 1 for i in range(1, world_size + 1)]
#     for i in range(num_layers):
#         device_map[f'model.model.layers.{i}'] = bisect_left(layer_per_gpu, i)

#     device_map['vision'] = 0
#     device_map['audio'] = 0
#     device_map['linear_proj'] = 0
#     device_map['linear_proj_audio'] = 0
#     device_map['model.model.word_embeddings.weight'] = 0
#     device_map['model.model.norm.weight'] = 0
#     device_map['model.lm_head.weight'] = 0
#     device_map['model.model.norm'] = 0
#     device_map[f'model.model.layers.{num_layers - 1}'] = 0
#     return device_map

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count() - 1
    print(world_size)
    num_layers = 32
    layer_per_gpu = num_layers // world_size
    layer_per_gpu = [i * layer_per_gpu - 1 for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_id = bisect_left(layer_per_gpu, i) + 1
        #print(device_id)
        if device_id > world_size:
            device_id = i % world_size + 1
        
        print(device_id)

        device_map[f'model.model.layers.{i}'] = device_id

    device_map['vision'] = 0
    device_map['audio'] = 0
    device_map['linear_proj'] = 0
    device_map['linear_proj_audio'] = 0
    device_map['model.model.word_embeddings.weight'] = 0
    device_map['model.model.norm.weight'] = 0
    device_map['model.lm_head.weight'] = 0
    device_map['model.model.norm'] = 0
    device_map[f'model.model.layers.{num_layers - 1}'] = 0
    return device_map


if __name__ == '__main__':

    model_name_or_path =  "/input/yushen.ys/checkpoints/bailing_native_moe_ming_flash_v2.0_xpo_final_20260205"
    #"/nativemm/share/cpfs/yuxuzheng.yxz/release/bailing_native_moe_ming_flash_v2.0_xpo_final_20260205_hf_metax_ais16863699"
    #"/nativemm/share/cpfs/weilong.cwl/checkpoints/megatron_flashv2.0_sft1_hf_metax/" #"."
    code_path = "."
    processor = AutoProcessor.from_pretrained(code_path, trust_remote_code=True)
    save_dir = "./generated_imgs"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    model = BailingMM2NativeForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=split_model(),
        load_image_gen=True,
    ).to(dtype=torch.bfloat16)

    model.diffusion_loss.pipelines.set_taylor_cache()

    prompt = "Draw a beautiful girl with short black hair and red dress."
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    for k in inputs.keys():
        if k in ["pixel_values", "pixel_values_videos", "audio_feats", "pixel_values_reference"]:
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)


    print(f"Instruction: {prompt}")
    # set `image_gen=True` to enable image generation
    image = model.generate(
        **inputs,
        image_gen=True,
        image_gen_seed=42,
    )
    save_path = os.path.join(save_dir, "./t2i_girl_taylor_cache_18.jpg")
    image.save(save_path)
    print(f"saved to {save_path}")


    prompt = "背景换成沙滩, 动作是拿手机自拍."
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": save_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)

    ref_image_inputs = processor.process_reference_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        image_gen_ref_images=ref_image_inputs,
    )

    inputs = inputs.to(model.device)

    for k in inputs.keys():
        if k in ["pixel_values", "pixel_values_videos", "audio_feats", "pixel_values_reference"]:
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    print(f"Instruction: {prompt}; Input image: {save_path}")
    # set `image_gen=True` to enable image generation
    image = model.generate(
        **inputs,
        image_gen=True,
        image_gen_seed=43,
    )
    save_path = os.path.join(save_dir, "./edit_girl_taylor_cache_18.jpg")
    image.save(save_path)
    print(f"saved to {save_path}")


    prompt = "A whimsical comic-style illustration of a cozy bookstore entrance on a sunny afternoon. The storefront features warm brick walls and large glass windows filled with stacked books and potted ferns. Above the wooden door hangs a hand-painted signboard with bold, stylized Chinese characters reading “理解与生成统一” accented with curling vines and tiny stars. Sunlight casts playful shadows on the cobblestone path leading to the door, where a vintage lantern in a sunbeam add charm. The linework is clean, colors vibrant yet soft, evoking a friendly, storybook atmosphere. No people or vehicles are present, emphasizing quiet serenity."
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, _, _ = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
    ).to(model.device)

    for k in inputs.keys():
        if k in ["pixel_values", "pixel_values_videos", "audio_feats", "pixel_values_reference"]:
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)


    print(f"Instruction: {prompt}")
    # set `image_gen=True` to enable image generation
    image = model.generate(
        **inputs,
        image_gen=True,
        image_gen_seed=42,
    )
    save_path = os.path.join(save_dir, "./t2i_text_taylor_cache_18.jpg")
    image.save(save_path)
    print(f"saved to {save_path}")
    