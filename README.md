# Ming-flash-omni 2.0

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_drbxn1/afts/img/YLAgT5MSnLwAAAAAQXAAAAgADkliAQFr/original" width="100"/>
<p>

<p align="center">📑 <a href="https://arxiv.org/abs/2506.09344">Technical Report</a>｜🤗 <a href="https://huggingface.co/inclusionAI/Ming-flash-omni-2.0">Hugging Face</a>｜ 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ming-flash-omni-2.0">ModelScope</a>



## Introduction

The newly released Ming-flash-omni 2.0 leverages the [Ling-2.0](https://github.com/inclusionAI/Ling-V2) architecture—a Mixture-of-Experts (MoE) framework comprising 100B total and 6B active parameters. Representing a generational advancement over its predecessor, it establishes new State-of-the-Art (SOTA) benchmarks among open-source omni-MLLMs. Ming-flash-omni 2.0 effectively synergizes foundational abilities with specialized domain expertise. In particular, it exhibits superior performance in visual encyclopedic knowledge, immersive speech synthesis, and high-dynamic image generation and manipulation.



<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_xg7bx2/afts/img/c1qcRIb3qH4AAAAAgCAAAAgADhHHAQFr/fmt.avif" width="800"/>
<p>


## 📌 Updates
* [2026.02.11] 🔥 We release the official version of [Ming-flash-omni 2.0](https://mp.weixin.qq.com/s/hz2fsH1DGpp2zpY-Yngsog), an open-source SOTA omni-MLLM that pushes the boundaries of multimodal understanding and synthesis.
* [2025.10.27] 🔥 We release the preview version of Ming-flash-omni：[Ming-flash-omni Preview](https://github.com/inclusionAI/Ming/tree/main).
* [2025.07.15] 🔥 We release [Ming-lite-omni v1.5](https://github.com/inclusionAI/Ming/tree/v1.5) with significant improvements across all modalities.
* [2025.06.12] 🔥 Our [Technical Report](https://arxiv.org/abs/2506.09344) is in public on arxiv.
* [2025.05.28] 🔥 The official version of [Ming-lite-omni v1](https://github.com/inclusionAI/Ming/tree/v1.0) is released, with better performance and image generation support.
* [2025.05.04] 🔥 We release the test version of Ming-lite-omni：[Ming-lite-omni-Preview](https://github.com/inclusionAI/Ming/tree/Ming-Lite-Omni-Preview).


## Key Features
Compared to [Ming-flash-omni Preview](https://github.com/inclusionAI/Ming/tree/Ming-flash-omni-Preview), Ming-flash-omni 2.0 focuses on optimizing capabilities across the following key domains: 
- **Expert-level Multimodal Cognition**: It accurately identifies plants and animals, recognizing cultural references (from regional cuisines to global landmarks), and delivering expert-level analysis of artifacts, including era, form, and craftsmanship. By synergizing high-resolution visual capture with a vast knowledge graph, the model achieves "vision-to-knowledge" synthesis, enabling superior knowledge understanding.


- **Immersive and Controllable Unified Acoustic Synthesis**:  Ming-flash-omni 2.0 introduces a unified end-to-end acoustic generation pipeline that integrates Speech, Audio, and Music within a single channel. Leveraging Continuous Autoregression coupled with a Diffusion Transformer (DiT) head, the model enables zero-shot voice cloning and nuanced attribute control (e.g., emotion, timbre, and ambient atmosphere). This architecture facilitates a transition from simple text-to-speech to highly expressive, emotionally resonant, and immersive auditory experiences.


- **High-Dynamic Controllable Image Generation and Manipulation**: Ming-flash-omni 2.0 features a native multi-task architecture that unifies segmentation, generation, and editing, allowing for sophisticated spatiotemporal semantic decoupling. It excels in high-dynamic content creation, including atmospheric reconstruction, seamless scene composition, and context-aware object removal. By maintaining texture coherence and spatial depth consistency, Ming-flash-omni 2.0 achieves state-of-the-art precision in complex image manipulation tasks.



<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_xg7bx2/afts/img/WxgUSrdZVj8AAAAAdJAAAAgADhHHAQFr/original" width="800"/>
<p>


## Use Cases

### Enhanced Multimodal Cognition & Free Modality Switching  
<video src="https://github.com/user-attachments/assets/147b9594-e492-4beb-a0db-b5c810135663" controls width="50%" height="400" style="object-fit: contain; max-width: 100%;">
    Enhanced Multimodal Cognition & Free Modality Switching
</video>

### Streaming Video Conversation  
<video src="https://github.com/user-attachments/assets/b1afb34e-8877-497c-85f3-82cd7cf618db" controls width="50%" height="400" style="object-fit: contain; max-width: 100%;">
    Streaming Video Conversation
</video>

### Controllable Audio Generation
<video src="https://github.com/user-attachments/assets/6b5d504f-86a3-4121-97c9-0aa9ea9abaa4" controls="controls" width="50%" height="auto" >
    Audio Context ASR & Dialect ASR
</video>

### Image Generation & Editing
<video src="https://github.com/user-attachments/assets/8d0af9cc-e0dc-440c-9963-b589d6396917" controls="controls" width="50%" height="auto" >
    Controllable Image Generation
</video>




## Model Downloads

You can download our latest model from both Huggingface and ModelScope. For previous version model like [Ming-flash-omni-Preview](https://github.com/inclusionAI/Ming/tree/Ming-flash-omni-Preview), Please refer to this [link](https://github.com/inclusionAI/Ming/tree/Ming-flash-omni-Preview?tab=readme-ov-file#model-downloads).

<div align="center">

| **Model**               |   **Input modality**   | **Oput modality** |                                                                      **Download**                                                                      |
|:------------------------|:----------------------:| :---------------: |:------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ming-flash-omni 2.0 | Image,text,video,audio | Image,text,audio  |                           [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-flash-omni-2.0)                           |
</div>
If you're in mainland China, we strongly recommend you to download our model from 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ming-flash-omni-2.0">ModelScope</a>.

```
pip install modelscope
modelscope download --model inclusionAI/Ming-flash-omni-2.0 --local_dir inclusionAI/Ming-flash-omni-2.0  --revision master
```

Note: This download process will take several minutes to several hours, depending on your network conditions.



## Environment Preparation


### Installation with pip
```shell
pip install -r requirements.txt
pip install nvidia-cublas-cu12==12.4.5.8  # for H20 GPU
```


## Example Usage

We provide a step-by-step running example:

Step 1 - Download the source code
```
git clone https://github.com/inclusionAI/Ming.git 
cd Ming
```
Step 2 - Download the model weights and create a soft link to the source code directory

Download our model following [Model Downloads](#model-downloads)

```shell
mkdir inclusionAI 
ln -s /path/to/inclusionAI/Ming-flash-omni-2.0 inclusionAI/Ming-flash-omni-2.0
```

Step 3 - Enter the code directory, you can refer to the following codes to run the Ming-flash-omni model.
```shell
jupyter notebook cookbook.ipynb
```

We also provide a simple example on the usage of this repo. For detailed usage, please refer to [cookbook.ipynb](https://github.com/inclusionAI/Ming/blob/main/cookbook.ipynb).

```python
import os
import torch
import warnings
from bisect import bisect_left
warnings.filterwarnings("ignore")

from transformers import AutoProcessor
from modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration

def split_model():
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = 32
    layer_per_gpu = num_layers // world_size
    layer_per_gpu = [i * layer_per_gpu for i in range(1, world_size + 1)]
    for i in range(num_layers):
        device_map[f'model.model.layers.{i}'] = bisect_left(layer_per_gpu, i)
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

# Load pre-trained model with optimized settings, this will take ~10 minutes
model_path = "inclusionAI/Ming-flash-omni-2.0"
model = BailingMM2NativeForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map=split_model(),
    load_image_gen=True,
    load_talker=True,
).to(dtype=torch.bfloat16)

# Initialize processor for handling multimodal inputs
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Inference Pipeline
def generate(messages, processor, model, sys_prompt_exp=None, use_cot_system_prompt=False, max_new_tokens=512):
    text = processor.apply_chat_template(
        messages, 
        sys_prompt_exp=sys_prompt_exp,
        use_cot_system_prompt=use_cot_system_prompt
    )
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
        audio_kwargs={"use_whisper_encoder": True},
    ).to(model.device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            num_logits_to_keep=1,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output_text

# qa
messages = [
    {
        "role": "HUMAN",
        "content": [
            {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
        ],
    },
]
output_text = generate(messages, processor=processor, model=model)
print(output_text)
# Output:

# 鹦鹉是一种非常受欢迎的宠物鸟类，它们以其鲜艳的羽毛、聪明的头脑和模仿人类语言的能力而闻名。鹦鹉的生活习性非常丰富，以下是一些主要的习性：

# 1. **社交性**：鹦鹉是高度社交的鸟类，它们在野外通常生活在群体中，与同伴互动、玩耍和寻找食物。在家庭环境中，鹦鹉需要与人类或其他鹦鹉进行定期的互动，以保持其心理健康。

# 2. **智力**：鹦鹉拥有非常高的智力，它们能够学习各种技能，包括模仿人类语言、识别物体、解决问题等。这种智力使它们成为非常有趣的宠物。

# ......
```


## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex

@misc{Mingflash2025,
      title  = {Ming-Flash-Omni: A Sparse, Unified Architecture for Multimodal Perception and Generation}, 
      author = {Inclusion AI},
      year = {2025},
      eprint = {2510.24821},
      archivePrefix = {arXiv},
      url = {https://arxiv.org/abs/2510.24821}
}

@misc{Mingomni2025,
      title  = {Ming-Omni: A Unified Multimodal Model for Perception and Generation}, 
      author = {Inclusion AI},
      year = {2025},
      eprint = {2506.09344},
      archivePrefix = {arXiv},
      url = {https://arxiv.org/abs/2506.09344}
}

@article{ai2025ming,
  title={Ming-flash-omni: A sparse, unified architecture for multimodal perception and generation},
  author={Inclusion AI},
  journal={arXiv preprint arXiv:2510.24821},
  year={2025}
}
```


