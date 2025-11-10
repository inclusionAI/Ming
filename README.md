# Ming-flash-omni Preview

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_drbxn1/afts/img/YLAgT5MSnLwAAAAAQXAAAAgADkliAQFr/original" width="100"/>
<p>

<p align="center">📑 <a href="https://arxiv.org/abs/2510.24821">Technical Report</a>｜🤗 <a href="https://huggingface.co/inclusionAI/Ming-flash-omni-Preview">Hugging Face</a>｜ 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ming-flash-omni-Preview">ModelScope</a>



## Introduction

Ming-flash-omni Preview, an upgraded version of [Ming-Omni](https://arxiv.org/abs/2506.09344), built upon a sparser Mixture-of-Experts (MoE) variant of [Ling-Flash-2.0](https://github.com/inclusionAI/Ling-V2) with 100B total parameters, of which only 6B
are active per token. Compared to its predecessor, the upgraded version exhibits substantial improvements across multimodal understanding and generation. We significantly advance speech recognition capabilities, achieving state-of-the-art performance in both contextual ASR and dialect-aware ASR. In image generation, Ming-flash-omni Preview introduces high-fidelity text rendering and demonstrates marked gains in scene consistency and identity preservation during image editing. Furthermore, Ming-flash-omni Preview introduces generative segmentation, a capability that not only achieves strong standalone segmentation performance but also enhances spatial control in image generation and improves editing consistency. It demonstrates highly competitive results in various modal benchmarks compared to industry-leading models.

<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_drbxn1/afts/img/5hflRY595xwAAAAAgBAAAAgADkliAQFr/fmt.avif" width="800"/>
<p>

## 📌 Updates

* [2025.10.27] 🔥 We release the preview version of Ming-flash-omni：[Ming-flash-omni Preview](https://inclusionai.github.io/blog/ming-flash-omni-preview).
* [2025.07.15] 🔥 We release [Ming-lite-omni v1.5](https://github.com/inclusionAI/Ming/tree/v1.5) with significant improvements across all modalities.
* [2025.06.12] 🔥 Our [Technical Report](https://arxiv.org/abs/2506.09344) is in public on arxiv.
* [2025.05.28] 🔥 The official version of [Ming-lite-omni v1](https://github.com/inclusionAI/Ming/tree/v1.0) is released, with better performance and image generation support.
* [2025.05.04] 🔥 We release the test version of Ming-lite-omni：[Ming-lite-omni-Preview](https://github.com/inclusionAI/Ming/tree/Ming-Lite-Omni-Preview).


## Key Features
Compared to [Ming-lite-omni v1.5](https://github.com/inclusionAI/Ming/tree/v1.5), Ming-flash-omni Preview features key optimizations in the following 3 areas:
- **Sparse MoE Architecture for Omni-Modality**: The Sparse MoE Architecture for Omni-Modality features a 100B-A6B MoE backbone (an extension of Ling-Flash-2.0). To ensure uniform expert activation and stable training across all modalities, Ming-flash-omni Preview  employs a Dual-Balanced Routing Mechanism that combines an Auxiliary Load Balancing Loss with a Modality-Level Router Bias Update.
- **Generative Segmentation-as-Editing Paradigm**: It unifies segmentation and editing into a semantics-preserving generation task, and achieves $0.90$ on GenEval, surpassing non-RL methods in fine-grained spatial control.
- **Context-Aware and Dialectal Speech Recognition**: Ming-flash-omni Preview sets a new State-of-the-Art performance across all 12 ContextASR benchmarks, and it significantly improves recognition performance for 15 Chinese dialects.



<p align="center">
    <img src="https://mdn.alipayobjects.com/huamei_drbxn1/afts/img/MdHMSqYQCqAAAAAAVcAAAAgADkliAQFr/fmt.avif" width="800"/>
<p>


## Use Cases

### Steaming Video Conversation


https://github.com/user-attachments/assets/971ec7da-f2b9-4a07-bc32-61dd80c5c680


### Audio Context ASR & Dialect ASR

https://github.com/user-attachments/assets/86304095-3563-42e1-9aa4-192748172536


### Audio Voice Clone

https://github.com/user-attachments/assets/408b955d-1baf-44fb-a083-21c98ffe6b4a



### Image Generation & Editing

https://github.com/user-attachments/assets/71a74740-f4fc-42b8-8873-9e4378bd06bb





## Model Downloads

You can download our latest model from both Huggingface and ModelScope. For previous version model like [Ming-Lite-Omni v1.5](https://github.com/inclusionAI/Ming/tree/v1.5), Please refer to this [link](https://github.com/inclusionAI/Ming/tree/v1.5?tab=readme-ov-file#model-downloads).

<div align="center">

| **Model**               |   **Input modality**   | **Oput modality** |                                                                      **Download**                                                                      |
|:------------------------|:----------------------:| :---------------: |:------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ming-flash-omni Preview | Image,text,video,audio | Image,text,audio  |                           [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ming-flash-omni-Preview) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-flash-omni-Preview)                           |
</div>
If you're in mainland China, we strongly recommend you to download our model from 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ming-Lite-Omni-1.5">ModelScope</a>.

```
pip install modelscope
modelscope download --model inclusionAI/Ming-flash-omni-Preview --local_dir inclusionAI/Ming-flash-omni-Preview  --revision master
```

Note: This download process will take several minutes to several hours, depending on your network conditions.

##  Evaluation
Ming-flash-omni Preview shows competitive performance in vision-text understanding, image generation, audio understanding and text-to-speech capabilities. For detailed evaluation results，please refer to our [techinical report](https://arxiv.org/abs/2510.24821).  


## Environment Preparation


### Installation with pip
```shell
pip install -r requirements.txt
# for python 3.10
pip install data/matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl 
# for python 3.8 
# pip install data/matcha_tts-0.0.5.1-cp38-cp38-linux_x86_64.whl
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
ln -s /path/to/inclusionAI/Ming-flash-Omni-Preview inclusionAI/Ming-flash-Omni-Preview
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
model_path = "inclusionAI/Ming-flash-omni-Preview"
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

# 鹦鹉是一种非常聪明和社交性强的鸟类，它们的生活习性非常丰富和有趣。以下是一些关于鹦鹉生活习性的详细介绍：
# ### 1. **栖息地**
# 鹦鹉主要分布在热带和亚热带地区，包括非洲、亚洲、澳大利亚和南美洲。它们通常生活在森林、草原、沙漠和城市环境中。不同种类的鹦鹉对栖息地的要求有所不同，但大多数鹦鹉喜欢有丰富植被和水源的地方。
# ### 2. **饮食**
# 鹦鹉是杂食性动物，它们的饮食非常多样化。它们的食物包括种子、坚果、水果、蔬菜、花蜜和昆虫。鹦鹉的喙非常强壮，能够轻松地打开坚硬的果壳和坚果。一些鹦鹉还会吃泥土或沙子，以帮助消化和补充矿物质。
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
```


