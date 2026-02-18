FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel

WORKDIR /workspace

# Define environments
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=16
ENV PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV CUDNN_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_PATH/lib

# # Substitude sources with aliyun
# RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
#     sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# Install base packages like wget, git, etc.
RUN apt update && apt install -y wget git build-essential libxml2 ninja-build pkg-config && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install python packages including torch and flash_attn
RUN pip install packaging
RUN pip install flash-attn==2.6.3 --no-build-isolation
# RUN pip install flash-attn==2.7.2.post1 --no-build-isolation

# Install dependence of Ming
COPY matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl .
RUN pip install --no-cache-dir matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl \
&& rm matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl
# RUN pip install --no-cache-dir transformer_engine[pytorch]==1.11.0
RUN pip install --no-cache-dir transformers==4.51.3 accelerate==0.33.0 peft==0.15.0 diffusers==0.33.0
RUN pip install --no-cache-dir hyperpyyaml modelscope onnxruntime inflect conformer lightning triton matplotlib gdown rich ipdb
RUN pip install --no-cache-dir nvidia-cublas-cu12==12.4.5.8

RUN pip install --no-cache-dir decord==0.6.0 iopath webdataset==0.2.100 omegaconf timm sortedcontainers
RUN pip install --no-cache-dir jsonlines fairscale typing_extensions==4.12.2 six regex ftfy
RUN pip install --no-cache-dir bitsandbytes funasr==1.1.14 openai-whisper==20240930 grpcio==1.59.0 "pynvml>=11.4.1" dlrover[torch]
RUN pip install --no-cache-dir xformers==0.0.22.post7 --no-deps
RUN pip install --no-cache-dir opencv-python-headless lmdb imageio 
RUN pip install --no-cache-dir pytorch-lightning==1.8.6
# RUN pip install --no-cache-dir pytorch-lightning==1.6.5
RUN pip install --no-cache-dir "tensorboard>=2.11.0" torchscale jieba av==10.0.0
RUN pip install --no-cache-dir -U mdatasets
RUN pip install --no-cache-dir opencc-python-reimplemented==0.1.7
RUN pip install --no-cache-dir pypinyin addict
RUN pip install --no-cache-dir mmengine==0.10.7 --no-deps
RUN pip install --no-cache-dir pycryptodome==3.21.0 cryptography==43.0.0 x_transformers==2.4.12

COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]