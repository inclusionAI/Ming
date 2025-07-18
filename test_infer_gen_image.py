import os
import time
import torch
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from IPython import embed
import torchvision
from PIL import Image
import re

import torch.nn as nn
from collections import defaultdict
from bailingmm_utils import process_ratio

def auto_translate(model, processor, text):
    if re.search(r'[\u4e00-\u9fff]', text):
        prefix = "Translate the Chinese phrase below into natural English. Return only the translation result without any explanations, prefixes, or formatting. Phrase to translate:"
        text = f"{prefix}{text}"
        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
        ).to(model.device)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        
        #srt_time = time.time()
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            use_cache=False,
            eos_token_id=processor.gen_terminator,
        )
        generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
        text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    return text

def generate_gen_image(
    model,
    processor,
    prompt,
    height=None,
    width=None,
    input_image_path=None,
    use_auto_translate=True,
    debug=False,
    POSITIVE_PREFIX_T2I=None,
    POSITIVE_PREFIX_I2I="high quality",
    NEGATIVE_PREFIX="worst quality, low quality, bad eyes, bad iris, twisted face, blurry, bad hand, watermark, multiple limbs, deformed fingers, bad fingers, ugly, monochrome, horror, geometry, bad anatomy, bad limbs, Blurry pupil, bad shading, error, bad composition, Extra fingers, strange fingers, Extra ears, extra leg, bad leg, disability, Blurry eyes, bad eyes, Twisted body, confusion, bad legs",
    image_gen_steps=30,
):
    # input_image_path 设置为 None, 运行 文生图，否则 图生图
    if height is None or width is None:
        image_gen_width, image_gen_height = 512 * 1, 512 * 1
    else:
        image_gen_width, image_gen_height = width, height

    closest_size, _ = process_ratio(ori_h=image_gen_height, ori_w=image_gen_width)
    image_gen_height, image_gen_width = closest_size[0] * 1, closest_size[1] * 1

    if use_auto_translate:
        prompt_ori = prompt
        prompt = auto_translate(model, processor, prompt)
        if debug:
            print("prompt: {} -> translated: {}".format(prompt_ori, prompt))
    
    if input_image_path is not None:
        if POSITIVE_PREFIX_I2I:
            prompt = "{}; Requirement: {}".format(prompt, POSITIVE_PREFIX_I2I)
    else:
        if POSITIVE_PREFIX_T2I:
            prompt = "{}; Requirement: {}".format(prompt, POSITIVE_PREFIX_T2I)
    
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": prompt},
            ] if input_image_path is None else [
                {"type": "image", "image": input_image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    if debug:
        print("messages:", messages)
    

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    if "image_gen_height" in inputs:
        image_gen_height = inputs["image_gen_height"]
        del inputs["image_gen_height"]
    
    if "image_gen_width" in inputs:
        image_gen_width = inputs["image_gen_width"]
        del inputs["image_gen_width"]

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    if NEGATIVE_PREFIX:
        negative_messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": NEGATIVE_PREFIX},
                ] if input_image_path is None else [
                    {"type": "image", "image": input_image_path},
                    {"type": "text", "text": NEGATIVE_PREFIX},
                ],
            }
        ]
        if debug:
            print("negative_messages:", negative_messages)

        negative_text = processor.apply_chat_template(negative_messages, add_generation_prompt=True)
        negative_inputs = processor(
            text=[negative_text],
            images=image_inputs,
            videos=None,
            audios=None,
            return_tensors="pt",
        ).to(model.device)
        inputs["image_gen_negative_input_ids"] = negative_inputs["input_ids"]
        inputs["image_gen_negative_attention_mask"] = negative_inputs["attention_mask"]

    image = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
        image_gen=True,
        image_gen_height=image_gen_height,
        image_gen_width=image_gen_width,
        image_gen_steps=image_gen_steps,
    )
    
    return image


if __name__ == '__main__':
    
    model_path = "inclusionAI/Ming-Lite-Omni"
    processor = AutoProcessor.from_pretrained('.', trust_remote_code=True)
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        load_image_gen=True,
    ).to("cuda")
    
    image = generate_gen_image(
        model=model, 
        processor=processor,
        prompt="a beautiful girl wearing a red dress.",
        POSITIVE_PREFIX_T2I="",
        POSITIVE_PREFIX_I2I="",
        image_gen_steps=30,
        NEGATIVE_PREFIX="",
    )
    image.save("./woman_red.jpg")

    image = generate_gen_image(
        model=model, 
        processor=processor,
        prompt="给人物戴上墨镜",
        input_image_path="./woman_red.jpg",
        POSITIVE_PREFIX_T2I="",
        POSITIVE_PREFIX_I2I="",
        image_gen_steps=30,
        NEGATIVE_PREFIX="",
        use_auto_translate=False,
    )  
    image.save("./woman_red_sunglasses.jpg")

    
    
    