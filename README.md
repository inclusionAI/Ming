# Ming-Lite-Omni

<p align="center">
    <img src="./figures/ant-bailing.png" width="100"/>
<p>

<p align="center">📑 <a href="https://arxiv.org/abs/2506.09344">Technical Report</a>｜📖<a href="https://lucaria-academy.github.io/Ming-Omni/">Project Page</a> ｜🤗 <a href="https://huggingface.co/inclusionAI/Ming-Lite-Omni-1.5">Hugging Face</a>｜ 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ming-Lite-Omni-1.5">ModelScope</a>



## Introduction

Ming-lite-omni, a light version of Ming-omni, which is derived from [Ling-lite](https://github.com/inclusionAI/Ling) and features 2.8 billion activated parameter for the MoE part. Ming-lite-omni is a unified multimodal model capable of processing images, text, audio, and video, while demonstrating strong proficiency in both speech and image generation. Ming-lite-omni employs dedicated encoders to extract tokens from different modalities, which are then processed by Ling, an MoE architecture equipped with newly proposed modality-specific routers. This design enables a single model to efficiently process and fuse multimodal inputs within a unified framework, thereby facilitating diverse tasks without requiring separate models, task-specific fine-tuning, or structural redesign. Importantly, Ming-lite-omni extends beyond conventional multimodal models by supporting audio and image generation. This is achieved through the integration of an advanced audio decoder for natural-sounding speech and Ming-Lite-Uni for high-quality image generation, which also allow the model to engage in context-aware chatting, perform text-to-speech conversion, and conduct versatile image editing. Our experimental results showcase Ming-lite-omni offers a powerful solution for unified perception and generation across all modalities. 
Notably, Ming-lite-omni is the first open-source model we are aware of to match GPT-4o in modality support, and we release all code and model weights to encourage further research and development in the community.


<p align="center">
    <img src="./figures/ming.png" width="800"/>
<p>

## 📌 Updates

* [2025.07.15] 🔥 We release [Ming-lite-omni v1.5](https://inclusionai.github.io/blog/ming-lite-omni-1_5/) with significant improvements across all modalities.
* [2025.06.12] 🔥 Our [Technical Report](https://arxiv.org/abs/2506.09344) is in public on arxiv.
* [2025.05.28] 🔥 The official version of [Ming-lite-omni v1](https://github.com/inclusionAI/Ming/tree/v1.0) is released, with better performance and image generation support.
* [2025.05.04] 🔥 We release the test version of Ming-lite-omni：[Ming-lite-omni-Preview](https://github.com/inclusionAI/Ming/tree/Ming-Lite-Omni-Preview).


## Key Features

- **Unified Omni-Modality Perception**: Ming-lite-omni, built on [Ling](https://github.com/inclusionAI/Ling), an MoE architecture LLM, resolves task conflicts and ensures coherent integration of tokens from different modalities through modality-specific routers.

- **Unified Perception and Generation**: Ming-lite-omni achieves unified understanding and generation, enabling the model to interpret multimodal instructions and user intent during generation, which helps enhance generation quality and improves usability across multiple tasks.

- **Innovative Generation Capabilities**: Ming-lite-omni can perceive all modalities and generate high-quality text, real-time speech, and vivid images simultaneously, delivering exceptional cross-modal performance across diverse tasks including image perception, audio-visual interaction, and image generation.


##  Evaluation
Ming-lite-omni v1.5 is an improved version of [Ming-lite-omni v1](https://github.com/inclusionAI/Ming/tree/v1.0), with a total of 20.3 billion parameters and 7 billion activated parameters (including 2.8 billion activated parameters for the MoE part). In various modality benchmark tests, it demonstrates highly competitive results compared to industry-leading models of similar scale.
Please refer to Ming-lite-omni v1.5 [relsease blog](https://inclusionai.github.io/blog/ming-lite-omni-1_5) for more comprehensive evaluation results. 


## Model Downloads

You can download our latest model from both Huggingface and ModelScope. For previous version model like [Ming-Lite-Omni v1](https://github.com/inclusionAI/Ming/tree/v1.0), Please refer to this [link](https://github.com/inclusionAI/Ming/tree/v1.0?tab=readme-ov-file#model-downloads).

<div align="center">

| **Model**          |   **Input modality**   | **Oput modality** |                                                                         **Download**                                                                         |
|:-------------------|:----------------------:| :---------------: |:------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Ming-Lite-Omni-1.5 | Image,text,video,audio | Image,text,audio  | [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ming-Lite-Omni-1.5) <br>[🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Lite-Omni-1.5) |
</div>
If you're in mainland China, we strongly recommend you to download our model from 🤖 <a href="https://www.modelscope.cn/models/inclusionAI/Ming-Lite-Omni-1.5">ModelScope</a>.

```
pip install modelscope
modelscope download --model inclusionAI/Ming-Lite-Omni-1.5 --local_dir inclusionAI/Ming-Lite-Omni-1.5  --revision master
```

Note: This download process will take several minutes to several hours, depending on your network conditions.





## Use Cases

Additional demonstration cases are available on our project [page](https://lucaria-academy.github.io/Ming-Omni/).


## Environment Preparation


### Installation with pip
```shell
pip install -r requirements.txt
# for python 3.10
pip install data/matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl 
# for python 3.8 
# pip install data/matcha_tts-0.0.5.1-cp38-cp38-linux_x86_64.whl
pip install diffusers==0.33.0
pip install nvidia-cublas-cu12==12.4.5.8  # for H20 GPU
```

### Installation with docker

You can also initialize the environment by building the docker image. First clone this repository:
```shell
git clone --depth 1 https://github.com/inclusionAI/Ming.git
cd Ming
```
Then build the docker image with the provided Dockerfile in `docker/docker-py310-cu121`. This step might take a while:
```shell
docker build -t ming:py310-cu121 docker/docker-py310-cu121
```
At last, start the container with the current repo directory mounted:
```shell
docker run -it --gpus all -v "$(pwd)":/workspace/Ming ming:py310-cu121 /bin/bash
```
You can run the model with python interface. You may download the huggingface model in the repo directory first (`.../Ming/`) or mount the downloaded model path when starting the container.


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
ln -s /path/to/inclusionAI/Ming-Lite-Omni-1.5 inclusionAI/Ming-Lite-Omni
```

Step 3 - Enter the code directory, you can refer to the following codes to run the Ming-Lite-Omni model.
```shell
jupyter notebook cookbook.ipynb
```

We also provide a simple example on the usage of this repo. For detailed usage, please refer to [cookbook.ipynb](https://github.com/inclusionAI/Ming/blob/main/cookbook.ipynb).

```python
import torch
from transformers import AutoProcessor, GenerationConfig
from modeling_bailingmm import BailingMMNativeForConditionalGeneration

# load model
model = BailingMMNativeForConditionalGeneration.from_pretrained(
    "inclusionAI/Ming-Lite-Omni",
    torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
    attn_implementation="flash_attention_2",
    load_image_gen=True,
    low_cpu_mem_usage=True       # Minimize CPU memory during loading
).to("cuda")  


# build processor
processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)

# qa
messages = [
    {
        "role": "HUMAN",
        "content": [
            {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
        ],
    },
]

# 1. Format inputs using chat template
text = processor.apply_chat_template(messages, add_generation_prompt=True)

# 2. Extract vision/audio data
image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

# 3. Prepare tensor inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    audios=audio_inputs,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
for k in inputs.keys():
    if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
        inputs[k] = inputs[k].to(dtype=torch.bfloat16)

# 4. Configure generation
generation_config = GenerationConfig.from_dict({'no_repeat_ngram_size': 10})
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,
    use_cache=True,
    eos_token_id=processor.gen_terminator,
    generation_config=generation_config,
)
generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

# 5. Decode output
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output_text)
# Output:

# 鹦鹉是一种非常聪明和社交性强的鸟类，它们的生活习性非常丰富和有趣。以下是一些关于鹦鹉生活习性的详细介绍：
# ### 1. **栖息地**
# 鹦鹉主要分布在热带和亚热带地区，包括非洲、亚洲、澳大利亚和南美洲。它们通常生活在森林、草原、沙漠和城市环境中。不同种类的鹦鹉对栖息地的要求有所不同，但大多数鹦鹉喜欢有丰富植被和水源的地方。
# ### 2. **饮食**
# 鹦鹉是杂食性动物，它们的饮食非常多样化。它们的食物包括种子、坚果、水果、蔬菜、花蜜和昆虫。鹦鹉的喙非常强壮，能够轻松地打开坚硬的果壳和坚果。一些鹦鹉还会吃泥土或沙子，以帮助消化和补充矿物质。
# ......
```

Note: We test the examples on hardware of NVIDIA H800-80GB/H20-96G with CUDA 12.4. Loading inclusionAI/Ming-Lite-Omni-1.5 in bfloat16 takes about 42G GPU memory.

## Gradio Demo

We provide a graphical user interface based on Gradio to facilitate the use of Ming-lite-omni.

1. Install Gradio dependencies

```shell
pip install gradio
pip install gradio_client
```

2. Start the Gradio server

```python
python gradio_demo.py 
```

## LLaMA-Factory Usage

We provide a LLaMA-Factory tools to facilitate the use of Ming-lite-omni.

1. First, make sure your environment is ready for Ming. Please clone the original code of this repository and download Ming-Lite-Omni-1.5 from [🤗 HuggingFace](https://huggingface.co/inclusionAI/Ming-Lite-Omni-1.5) or [🤖 ModelScope](https://www.modelscope.cn/models/inclusionAI/Ming-Lite-Omni-1.5) following the official installation guidance above.
    ```shell
    Ming
    ├── am.mvn
    ├── audio_detokenizer
    ├── ...
    ├── inclutionAI
    │   ├── Ming-Lite-Omni-1.5
    │   │   ├── config.json
    │   │   ├── ...
    ```

2. Download the original code of `LLaMA-Factory==0.9.3`. Apply the patch for the compatibility of Ming and install LLaMA-factory with the official installation guide.
    ```shell
    git clone https://github.com/hiyouga/LLaMA-Factory.git
    cd LLaMA-Factory
    git checkout v0.9.3

    cp <your path to Ming>/ming.patch .
    git apply ming.patch
    pip install .
    ```

3. Now you can fine-tune Ming with llama-factory. 
    ```shell
    cd <your path to Ming>
    llamafactory-cli train <your config yaml path>
    ```

For more details for llama-factory usage, please refer to the original [Github Repository](https://github.com/hiyouga/LLaMA-Factory).


## License and Legal Disclaimer

This code repository is licensed under the [MIT License](./LICENSE), and the Legal Disclaimer is located in the [LEGAL.md file](./LEGAL.md) under the project's root directory.

## Citation

If you find our work helpful, feel free to give us a cite.

```bibtex

@misc{Mingomni2025,
      title  = {Ming-Omni: A Unified Multimodal Model for Perception and Generation}, 
      author = {Inclusion AI},
      year = {2025},
      eprint = {2506.09344},
      archivePrefix = {arXiv},
      url = {https://arxiv.org/abs/2506.09344}
}
```


