import os
import time
import torch
from transformers import AutoProcessor

from modeling_bailingmm import BailingMMNativeForConditionalGeneration

def generate(messages, processor, model, save_image_path):
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
    
    srt_time = time.time()
    image = model.generate(
        **inputs,
        image_gen=True,
    )

    image.save(save_image_path)
    #print(f"Generate time: {(time.time() - srt_time):.2f}s")


if __name__ == '__main__':
    model_path = "."
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")

    model.load_image_gen_modules(model_path)
    # best pixel config for image_generation
    input_pixels = 451584
    processor.max_pixels = input_pixels
    processor.min_pixels = input_pixels

    #vision_path = "/input/zhangqinglong.zql/assets/"

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Draw a beautiful girl."},
            ],
        }
    ]

    generate(messages=messages, processor=processor, model=model, save_image_path="./generated_girl.jpg")

    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "image", "image": "./generated_girl.jpg"},
                {"type": "text", "text": "Replace the face with a boy."},
            ],
        }
    ]

    generate(messages=messages, processor=processor, model=model, save_image_path="./replaced_boy.jpg")

    