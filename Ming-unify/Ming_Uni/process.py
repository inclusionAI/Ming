import argparse
import os
import random
from io import BytesIO
from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image
from Ming_Uni.qwen2vl_processor import Qwen2VLImageProcessor

LLAVA_DEFAULT_IMAGE_TOKEN = "<image>"

from PIL import Image

from Ming_Uni.Templates_native import (
    EOT,
    SYSTEM_PREFIX,
    USER_PREFIX,
    ASSISTANT_PREFIX,
    GLM_USER_PREFIX,
    GLM_ASSISTANT_PREFIX,
    QWEN2_SYSTEM_PREFIX,
    QWEN2_USER_PREFIX,
    QWEN2_ASSISTANT_PREFIX,
    interleave_tokens,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_AU_START_TOKEN,
    DEFAULT_AU_END_TOKEN,
    DEFAULT_AUDIO_PATCH_TOKEN,
    DEFAULT_GEN_AU_START_TOKEN,
    DEFAULT_GEN_AU_END_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_END_OF_CHUNK_TOKEN,
)

additional_special_tokens_llama = [
    "[item]",
    "<html>",
    "</html>",
    "<body>",
    "</body>",
    "<table>",
    "</table>",
    "<tr>",
    "</tr>",
    "<td>",
    "</td>",
]
additional_special_tokens_qwen2 = [
    "[item]",
    "<html>",
    "</html>",
    "<body>",
    "</body>",
    "<table>",
    "</table>",
    "<tr>",
    "</tr>",
    "<td>",
    "</td>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>"
]
def init_tokenizer(llm_model, interleave_tokens=[]):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens_qwen2}
    )

    # add special_tokens to tokenizer
    if len(interleave_tokens) > 0:
        num_new_tokens = tokenizer.add_tokens(interleave_tokens, special_tokens=True)
        print("generation_num_tokens: {}".format(num_new_tokens))
        print("Tokenizer length after adding interleave tokens in dataset: ", len(tokenizer))
    return tokenizer
def center_crop(image_path, save_path, short_side=512):
    """
    按照短边裁剪为 512 像素，并对图像进行中心裁剪。
    
    :param image_path: 输入图像路径
    :param save_path: 保存裁剪后的图像路径
    :param short_side: 裁剪时短边的大小，默认值为 512
    """
    # 打开图像
    img = Image.open(image_path)
    
    # 获取原始图像的尺寸
    width, height = img.size
    
    # 计算缩放比例，根据短边调整为 short_side 的大小
    if width < height:
        scale = short_side / width
        new_width = short_side
        new_height = int(height * scale)
    else:
        scale = short_side / height
        new_height = short_side
        new_width = int(width * scale)
    
    # 缩放图像，使短边为 512
    if new_width != width or new_height != height:
        img_resized = img.resize((new_width, new_height))
    else:
        img_resized = img
        
    # 获取缩放后图像的尺寸
    resized_width, resized_height = img_resized.size
    
    # 计算中心裁剪的坐标
    left = (resized_width - short_side) // 2
    top = (resized_height - short_side) // 2
    right = left + short_side
    bottom = top + short_side
    
    # 裁剪图像
    img_cropped = img_resized.crop((left, top, right, bottom))
    
    # 保存裁剪后的图像
    img_cropped.save(save_path)
    print(f'裁剪后的图像已保存到 {save_path}')


class MyProcessor():
    def __init__(self,glm_model):
        vis_processor = Qwen2VLImageProcessor()
        # 设置最大pixels
        max_pixels = 451584
        min_pixels = 451584
        temporal_patch_size = 2
        merge_size = 2
        
        
        assert hasattr(vis_processor, "max_pixels")
        setattr(vis_processor, "max_pixels", max_pixels)
        assert hasattr(vis_processor, "min_pixels")
        setattr(vis_processor, "min_pixels", min_pixels)
        assert hasattr(vis_processor, "temporal_patch_size")
        setattr(vis_processor, "temporal_patch_size", temporal_patch_size)
        assert hasattr(vis_processor, "merge_size")
        setattr(vis_processor, "merge_size", merge_size)

        self.vis_processor = vis_processor

        self.use_qwen2_template = True

        self.llm_model_type = 'qwen2'

        self.num_query_token=2560
        self.glm_model = "Qwen/Qwen2.5-7B-Instruct"
        self.tokenizer = init_tokenizer(
            self.glm_model, 
            interleave_tokens
        )
        self._init_special_token()
    
    def _init_special_token(self):
        self.image_start_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        self.image_end_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        self.image_patch_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_PATCH_TOKEN)

        self.video_start_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_VID_START_TOKEN)
        self.video_end_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_VID_END_TOKEN)

        self.audio_start_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_AU_START_TOKEN)
        self.audio_end_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_AU_END_TOKEN)
        self.audio_patch_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_AUDIO_PATCH_TOKEN)
        self.end_of_chunk_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_END_OF_CHUNK_TOKEN)

        bos_token = None

        if self.llm_model_type in ["qwen2"]:
            bos_token = self.tokenizer.bos_token if self.tokenizer.eos_token is None else self.tokenizer.pad_token
            self.qwen2_bos_id = self.tokenizer.convert_tokens_to_ids(bos_token)
            self.qwen2_eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
            self.qwen2_pad_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)


        assert bos_token is not None
        self.llm_bos_token = bos_token
        self.llm_eos_token = self.tokenizer.eos_token
        self.llm_pad_token = self.tokenizer.pad_token

        self.img_text = DEFAULT_IM_START_TOKEN + self.num_query_token * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN

        self.usr_prefix = QWEN2_USER_PREFIX
        self.assistant_prefix = QWEN2_ASSISTANT_PREFIX

        self.img_text_id = (self.tokenizer(self.img_text, return_tensors="pt")["input_ids"][0]).tolist()
        self.system_prefix_id = (self.tokenizer(SYSTEM_PREFIX, return_tensors="pt")["input_ids"][0]).tolist()
        if self.use_qwen2_template:
            self.system_prefix_id = (self.tokenizer(QWEN2_SYSTEM_PREFIX, return_tensors="pt")["input_ids"][0]).tolist()

        self.usr_prefix_id = (self.tokenizer(self.usr_prefix, return_tensors="pt")["input_ids"][0]).tolist()
        self.assistant_prefix_id = (self.tokenizer(self.assistant_prefix, return_tensors="pt")["input_ids"][0]).tolist()

        self.EOT_id = (self.tokenizer(EOT, return_tensors="pt")["input_ids"][0]).tolist()
        self._n_id = (self.tokenizer("\n", return_tensors="pt")["input_ids"][0]).tolist()

    def preprocess_text(self, question, generate_prefix=None):
        input_text = ""
        input_ids = []
        position_ids = None

        input_text += QWEN2_SYSTEM_PREFIX
        input_ids.extend(self.system_prefix_id)

        input_text += self.usr_prefix
        input_ids.extend(self.usr_prefix_id)

        input_text += question
        question_id = (self.tokenizer(question, return_tensors="pt")["input_ids"][0]).tolist()
        input_ids.extend(question_id)

        input_text += self.assistant_prefix
        input_ids.extend(self.assistant_prefix_id)

        assert self.llm_model_type in ["qwen2"]
        #input_ids = torch.cat(
        #    [torch.tensor(input_ids), torch.tensor([self.qwen2_eos_id])]
        #)  # 后面并eos_id
        #input_text = input_text + self.llm_eos_token
        

        if generate_prefix is not None:
            input_text += generate_prefix
            generate_prefix_id = (self.tokenizer(generate_prefix, return_tensors="pt")["input_ids"][0]).tolist()
            input_ids.extend(generate_prefix_id)
        
        input_ids = torch.tensor(input_ids)
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_text=input_text,  # just for debug
        )


    def process(self, image_file, prompt, device="cpu", input_interpolate64=False, input_interpolate256=False):
        pixel_values = None
        image_grid_thw = None
        generate_prefix = "<image>"
            
        if image_file is not None:
            if isinstance(image_file, Image.Image):
                image = image_file
            elif image_file.startswith("http"):
                response = requests.get(image_file)
                response.raise_for_status()  # 检查请求是否成功
                # 将字节数据转换为BytesIO对象
                image_data = BytesIO(response.content)
                image = Image.open(image_data).convert("RGB")
            else:
                image = Image.open(image_file).convert("RGB")
                # image = fetch_image({"type": "image", "image": image_file})
            prompt = f"<image>\n {prompt}" if prompt else "<image>\n"
            image_inputs = self.vis_processor(images=image, videos=None)
            image_grid_thw = image_inputs["image_grid_thw"]  # [ 1 36 34]
            pixel_values = image_inputs["pixel_values"]  # (1224, 1176)
            # print(f"image_grid_thw: {image_grid_thw}")
            # print(f"pixel_values_size: {pixel_values.shape}")

            num_query_token = torch.prod(image_grid_thw, dim=1) // 4
            ### 64 ～～～
            #num_query_token = torch.tensor([64])
            assert num_query_token.shape[0] == 1

            assert prompt.count(LLAVA_DEFAULT_IMAGE_TOKEN) == 1

            assert not (input_interpolate64 is True and input_interpolate256 is True)
            if input_interpolate64 is True:
                img_text = DEFAULT_IM_START_TOKEN + 64 * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN
            elif input_interpolate256 is True:
                img_text = DEFAULT_IM_START_TOKEN + 256 * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN
            else:
                img_text = DEFAULT_IM_START_TOKEN + num_query_token[0] * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(LLAVA_DEFAULT_IMAGE_TOKEN, img_text).strip()

        ret = self.preprocess_text(prompt, generate_prefix)

        input_text = ret["input_text"]
        input_ids = ret["input_ids"].tolist()
        attention_mask = ret["attention_mask"]
        if attention_mask is not None:
            attention_mask = attention_mask.tolist()

        if image_file is not None:
            image_start_indices = list(torch.where(torch.tensor(input_ids) == self.image_start_token)[0])
            image_end_indices = list(torch.where(torch.tensor(input_ids) == self.image_end_token)[0])
            print(image_start_indices, image_end_indices)
            #assert len(image_start_indices) == len(image_end_indices)

            num_images = 1 if image_file is not None else 0
            #assert len(image_start_indices) == num_images
            #assert len(image_end_indices) == num_images

        assert DEFAULT_AU_START_TOKEN not in input_text and DEFAULT_AU_END_TOKEN not in input_text
        assert DEFAULT_GEN_AU_START_TOKEN not in input_text and DEFAULT_GEN_AU_END_TOKEN not in input_text
        assert DEFAULT_VID_START_TOKEN not in input_text and DEFAULT_VID_END_TOKEN not in input_text
        
        attention_mask = torch.tensor(attention_mask, dtype=torch.int32)

        assert len(input_ids) == len(attention_mask)
        if image_grid_thw is not None:
            n_image_features = int(sum(torch.prod(image_grid_thw, dim=-1) // 4))
            n_image_tokens = input_ids.count(self.image_patch_token)
            if n_image_tokens != n_image_features:
                print(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_grid_thw = image_grid_thw.tolist()

        input_image = pixel_values
        result = {
            "image": input_image.to(device) if input_image is not None else None,
            "image_grid_thw": torch.tensor(image_grid_thw).to(device) if image_grid_thw is not None else None,
            "decoder_image": torch.zeros(0, 3, 224, 224).to(device),
            "task_type": "others",
            "dataset_type": "image_text",
            "input_ids": torch.tensor(input_ids).unsqueeze(0).to(device),
            "position_ids": None,
            "generation_attention_mask": attention_mask.unsqueeze(0).to(device),
            "labels": None,
            "audio": None,
            "weights": None,
            "input_text": input_text,  # just for debug
        }
        return result
