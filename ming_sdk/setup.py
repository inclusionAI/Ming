import os
import shutil
import os.path as osp
from setuptools import setup, find_packages

__version__ = "1.0.0"  #
requirement = open("ming_sdk/requirements.txt").readlines()

__setup_name__ = "ming_sdk"


def fetch_installed_data(model_dir):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, model_dir)
    data_files = []
    # root_dir = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for f in files:
            file_path = os.path.join(root, f)
            # package c source & header files, and swig files
            if (
                f.endswith(".cc")
                or f.endswith(".h")
                or f.endswith(".i")
                or f.endswith(".yml")
                or f.endswith(".yaml")
                or f.endswith("*.md")
                or f.endswith(".so")
                or f.endswith(".py")
                or f.endswith(".dylib")
                or f.endswith(".engine")
                or f.endswith(".txt")
                or f.endswith(".jpg")
                or f.endswith(".gz")
                or f.endswith(".bk")
                or f.endswith(".json")
                or f.endswith(".bk")
                or f.endswith(".cfg")
                or f.endswith("cfg")
                or f.endswith("model")
                or f.endswith("moves")
                or f.endswith("keyrow")
                or f.endswith("tokenizer")
                or f.endswith("vectors")
                or f.endswith("patterns")
                or f.endswith(".bin")
                or f.endswith(".model")
                or f.endswith(".pt")
                or f.endswith(".wav")
                or f.endswith(".fst")
            ):
                data_files.append(file_path)
    return data_files


file_list = [
    "audio_processing_bailingmm2.py",
    "bailingmm_utils.py",
    "bailingmm_utils_video.py",
    "chat_format.py",
    "config.json",
    "configuration_audio.py",
    "configuration_bailingmm2.py",
    "configuration_bailing_moe_v2.py",
    "configuration_bailing_talker.py",
    "configuration_whisper_encoder.py",
    "cookbook.ipynb",
    "image_processing_bailingmm2.py",
    "modeling_bailingmm2.py",
    "modeling_bailing_moe_v2.py",
    "modeling_bailing_talker.py",
    "modeling_utils.py",
    "modeling_whisper_encoder.py",
    "preprocessor_config.json",
    "processing_bailingmm2.py",
    "qwen3_moe_vit.py",
    "s3bpe_tokenizer.py",
    "special_tokens_map.json",
    "tokenization_bailing.py",
    "tokenizer_config.json",
    "tokenizer.json",
]

dir_list = [
    "data/",
    "diffusion/",
    "talker_tn/",
    "talker_module",
    "AudioVAE",
    "bizgen",
    "front"
]

for i in file_list:
    shutil.copy(i, "ming_sdk")

for i in dir_list:
    shutil.copytree(i, "ming_sdk/" + i, dirs_exist_ok=True)

setup(
    name="ming_sdk",
    version=__version__,
    author="qiaozhuo",
    author_email="shoukui.xsk@antgroup.com",
    url="",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
    ],
    package_data={"ming_sdk": fetch_installed_data("")},
    description="Ming Multimodal sdk",
    keywords="Ming Multimodal sdk",
    packages=["ming_sdk"],
    python_requires=">=3.9.0",
)
