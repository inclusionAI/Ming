"""
MingUtils: Utility Functions and Classes for Ming SDK

This module provides essential utilities for the Ming SDK, including:
    - Multimedia processing (image, video, audio)
    - Prompt building and tokenization
    - File download utilities
    - Caching mechanisms
    - Streaming generation helpers

Key Components:
    - MingUtils: Main utility class for prompt processing
    - DownloadUtils: HTTP download with metadata extraction
    - SimpleCache / ThreadSafeCache: TTL-based caching with LRU eviction
    - StreamGenerator: Async-to-sync stream wrapper
"""

import os
import json
import time
import torch
import asyncio
import logging
import requests
import threading
import subprocess
from PIL import Image
from queue import Queue, Empty

from collections import OrderedDict
from typing import Iterable, TypeVar, Iterator

from transformers import AutoProcessor, AutoTokenizer
from vllm.inputs import TextPrompt as LLMInputs
from typing import Any, Dict, Optional, Tuple, Union, List, AsyncGenerator

T = TypeVar("T")
logger = logging.getLogger()
from enum import IntEnum, unique


@unique
class MingStatus(IntEnum):
    """Status codes for Ming SDK operations."""
    OK = 200
    ParametersIllegal = 401
    DonwloadFail = 402
    DonwloadTimeout = 403
    VideoSizeLimit = 404
    AudioSizeLimit = 405
    ImageSizeLimit = 406


class DownloadUtils(object):
    """HTTP download utility with metadata extraction support."""

    def __init__(self):
        pass

    def get_meta_info(self, url: str) -> Dict[str, Any]:
        """
        Extract metadata (dimensions, duration, codec, etc.) from a media URL.

        Uses ffprobe to analyze the media file without downloading it.

        Args:
            url (str): URL of the media file.

        Returns:
            Dict[str, Any]: Metadata including width, height, fps, duration,
                           codec_type, and size.
        """

        meta_info = {
            "width": 0,
            "height": 0,
            "fps": 0,
            "duration": 0,
            "codec_type": "",
            "size": 0,
        }
        if "https://" in url:
            url = url.replace("https://", "http://")

        ffprobe_cmd = f"ffprobe -v error  -show_entries format=duration,size -show_entries stream=duration,codec_type,width,height,avg_frame_rate -i '{url}' -of json"
        ret = subprocess.run(
            [ffprobe_cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if ret.returncode != 0:
            logger.warning(f"get meta info fial {url}")
            return meta_info
        output = ret.stdout.decode("utf8")
        info = json.loads(output)

        if "streams" in info:
            for stream in info["streams"]:
                if "codec_type" not in stream:
                    continue
                if stream["codec_type"] == "video":
                    meta_info["width"] = stream["width"]
                    meta_info["height"] = stream["height"]
                    meta_info["size"] = int(info["format"]["size"]) / (
                        1024 * 1024
                    )  # MB
                    if "duration" in stream:
                        meta_info["codec_type"] = "video"
                        meta_info["duration"] = float(stream["duration"])
                        meta_info["fps"] = float(stream["avg_frame_rate"])
                    else:
                        meta_info["codec_type"] = "image"
                    break
                if stream["codec_type"] == "audio":
                    meta_info["duration"] = float(stream["duration"])
                    meta_info["size"] = int(info["format"]["size"]) / (1024 * 1024)
                    meta_info["codec_type"] = "audio"
        return meta_info

    def download(
        self, url: str, target_path: str, filename, timeout: Tuple[int, int] = (10, 180)
    ) -> Dict[int, Any]:
        """
        Download a file from the given URL and save it to the specified local path.

        This function uses streaming to efficiently download large files without loading them
        entirely into memory. It ensures atomicity by writing to a temporary file first and
        then renaming it upon completion.

        Args:
            url (str): The URL of the file to download.
            target_path (str): The local directory where the file will be saved.
            filename (str): The name to save the file as.
            timeout (Tuple[int, int]): Connection and read timeout in seconds. Default is (10, 180).

        Returns:
            Tuple[int, str | None]: A tuple containing:
                - status code: 0 for success, negative values for specific failures
                - file path if successful, otherwise None
        """
        STATUS_SUCCESS = 0
        STATUS_DOWNLOAD_FAILED = -1
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            target_path = os.path.join(target_path, filename)
            with open(target_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            logger.info(f"Download success: {target_path}")
            return STATUS_SUCCESS, target_path
        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}, url {url}")
        except Exception as e:
            logger.error(f"Save file failed: {e}, url {url}")
        return STATUS_DOWNLOAD_FAILED, None

def load_json_or_str(s: str) -> object:
    """
    Attempt to parse a string as JSON; return the original value on failure.

    Args:
        s (str): Input string to parse.

    Returns:
        object: Parsed JSON object or original string.
    """
    if not isinstance(s, str):
        return s
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


class MingUtils(object):
    """
    Main utility class for prompt building and multimedia processing.

    This class handles:
        - Tokenization and prompt template application
        - Multimedia (image/video/audio) input processing
        - Message filtering based on token limits
        - History management for multi-turn conversations

    Attributes:
        processor: HuggingFace processor for multimodal inputs.
        tokenizer: HuggingFace tokenizer.
        limit_images (int): Maximum images per prompt.
        limit_videos (int): Maximum videos per prompt.
        sample_rate (int): Audio sample rate for processing.
        max_frames (int): Maximum frames for video processing.
    """
    def __init__(
        self,
        model_path: str,
        limit_mm_per_prompt={"image": 10, "video": 2},
        sample_rate=16000,
        sys_prompt=None,
    ):
        from processing_bailingmm2 import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_FRAME_PATCH_TOKEN, DEFAULT_AUDIO_PATCH_TOKEN
        self.image_path_token = DEFAULT_IMAGE_PATCH_TOKEN
        self.frame_path_token = DEFAULT_FRAME_PATCH_TOKEN
        self.audio_path_token = DEFAULT_AUDIO_PATCH_TOKEN
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.limit_mm_per_prompt = limit_mm_per_prompt
        if "image" in limit_mm_per_prompt:
            self.limit_images = limit_mm_per_prompt["image"]
        else:
            self.limit_images = None

        if "video" in limit_mm_per_prompt:
            self.limit_videos = limit_mm_per_prompt["video"]
        else:
            self.limit_videos = None
        self.sys_prompt = sys_prompt
        self.sample_rate = sample_rate
        self.max_frames = 40

    def filter_message(self, data: list, limit_images: int = 10,
                       limit_videos: int = 2, limit_audios: int = 1) -> list:
        """
        Filter conversation messages to enforce multimedia limits.

        This method ensures the total number of images, videos, and audios
        in the conversation history does not exceed specified limits.

        Args:
            data (list): List of conversation messages.
            limit_images (int): Maximum allowed images. Defaults to 10.
            limit_videos (int): Maximum allowed videos. Defaults to 2.
            limit_audios (int): Maximum allowed audios. Defaults to 1.

        Returns:
            list: Filtered list of messages respecting the limits.
        """
        total_image_count = 0
        total_video_count = 0
        total_audio_count = 0
        last_item_audios = 0

        filtered_data = []
        last_item = data[-1] if data else None

        if last_item and last_item["role"] == "HUMAN":
            last_item_images = sum(
                1 for content in last_item["content"] if content["type"] == "image"
            )
            last_item_videos = sum(
                1 for content in last_item["content"] if content["type"] == "video"
            )
            last_item_audios = sum(
                1 for content in last_item["content"] if content["type"] == "audio"
            )

            if (
                total_image_count + last_item_images <= limit_images
                and total_video_count + last_item_videos <= limit_videos
                and total_audio_count + last_item_audios <= limit_audios
            ):
                filtered_data.append(last_item)
                total_image_count += last_item_images
                total_video_count += last_item_videos
                total_audio_count += last_item_audios

        temp_human = None
        temp_assistant = None
        for entry in reversed(data[:-1]):
            if entry["role"] == "HUMAN":
                temp_human = entry

                if temp_human and temp_assistant:
                    human_images = sum(
                        1
                        for content in temp_human["content"]
                        if content["type"] == "image"
                    )
                    human_videos = sum(
                        1
                        for content in temp_human["content"]
                        if content["type"] == "video"
                    )
                    human_audios = sum(
                        1
                        for content in temp_human["content"]
                        if content["type"] == "audio"
                    )
                    assistant_images = sum(
                        1
                        for content in temp_assistant["content"]
                        if content["type"] == "image"
                    )
                    assistant_videos = sum(
                        1
                        for content in temp_assistant["content"]
                        if content["type"] == "video"
                    )
                    assistant_audios = sum(
                        1
                        for content in temp_assistant["content"]
                        if content["type"] == "audio"
                    )

                    new_image_count = (
                        total_image_count + human_images + assistant_images
                    )
                    new_video_count = (
                        total_video_count + human_videos + assistant_videos
                    )
                    new_audio_count = (
                        total_audio_count + human_audios + assistant_audios
                    )

                    if (
                        new_image_count > limit_images
                        or new_video_count > limit_videos
                        or new_audio_count > limit_audios
                    ):
                        temp_human = None
                        temp_assistant = None
                        continue
                    elif last_item_audios > 0 and human_audios + assistant_audios > 0:
                        temp_human = None
                        temp_assistant = None
                        continue
                    else:
                        filtered_data.append(temp_assistant)
                        filtered_data.append(temp_human)
                        total_image_count = new_image_count
                        total_video_count = new_video_count
                        total_audio_count = new_audio_count

                        temp_human = None
                        temp_assistant = None

            elif entry["role"] == "ASSISTANT":
                temp_assistant = entry

        return filtered_data[::-1]

    def compute_text_input_tokens(self, text: str, **kwargs) -> Optional[int]:
        """
        Compute the number of input tokens for a text prompt.

        Args:
            text (str): Input text to tokenize.
            **kwargs: Additional arguments (compute_input_tokens_flag, system_prompt).

        Returns:
            Optional[int]: Token count if compute_input_tokens_flag is True, else None.
        """
        compute_input_tokens_flag = kwargs.get("compute_input_tokens_flag", False)
        if not compute_input_tokens_flag:
            return None
        t1 = time.time()
        system_prompt = kwargs.get("system_prompt", None)
        if system_prompt and isinstance(system_prompt, str):
            text += system_prompt
        if text and len(text):
            input_text_ids = self.tokenizer.encode(text, add_special_tokens=True)
            text_token_count = len(input_text_ids)
        else:
            text_token_count = 0
        t2 = time.time()
        logger.info(f"Compute text input tokens, text_token_count: {text_token_count}, cost time: {t2-t1}s")
        return text_token_count

    def compute_image_audio_video_input_tokens(self, prompt: str, image_inputs: list,
                                                   video_inputs: list, audio_inputs: list,
                                                   **kwargs) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        Compute token counts for image, video, and audio inputs.

        This method processes multimodal inputs through the processor and
        counts the number of patch tokens for each modality.

        Args:
            prompt (str): Text prompt.
            image_inputs (list): List of image inputs.
            video_inputs (list): List of video inputs.
            audio_inputs (list): List of audio inputs.
            **kwargs: Additional arguments (compute_input_tokens_flag).

        Returns:
            Tuple[Optional[int], Optional[int], Optional[int]]: Token counts for
                (image, video, audio) if compute_input_tokens_flag is True.
        """
        compute_input_tokens_flag = kwargs.get("compute_input_tokens_flag", False)
        if not compute_input_tokens_flag:
            return None, None, None
        t1 = time.time()
        inputs_processor = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            audio_kwargs={"use_whisper_encoder": True},
            return_tensors="pt",
        )
        image_patch_id = self.tokenizer.convert_tokens_to_ids(self.image_path_token)
        image_token_count = (inputs_processor['input_ids'] == image_patch_id).sum().item()

        #DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"
        video_patch_id = self.tokenizer.convert_tokens_to_ids(self.frame_path_token)
        video_token_count = (inputs_processor['input_ids'] == video_patch_id).sum().item()

        #DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
        audio_patch_id = self.tokenizer.convert_tokens_to_ids(self.audio_path_token)
        audio_token_count = (inputs_processor['input_ids'] == audio_patch_id).sum().item()
        t2 = time.time()
        logger.info(f"Compute image/video/audio tokens, cost time: {t2-t1}s")
        return image_token_count, video_token_count, audio_token_count

    def build_prompt(
        self,
        prompt: str,
        audio: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        video: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        history: list = [],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build a prompt input for the model (common logic for text/audio/image generation).

        Args:
            prompt (str): User input text.
            audio (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Audio data (e.g., file path or binary or list).
            video (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Video data (e.g., file path or binary or list).
            image (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Image data (file path, binary, or PIL Image or list).
            history (list, optional): Conversation history. Defaults to empty list.
            **kwargs: Additional parameters for prompt building.

        Returns:
            Dict[str, Any]: A dictionary containing the built prompt.
        """

        os.environ["IMAGE_GEN_MODE"] = ""
        current_sys_prompt = self.sys_prompt
        use_cot_system_prompt = False
        for key, value in kwargs.items():
            if key == "audio":
                audio = value
            if key == "video":
                video = value
            if key == "image":
                image = value
            if key == "system_prompt":
                current_sys_prompt = value
            if key == "use_cot" and isinstance(value, bool):
                use_cot_system_prompt = value

        messages_video_and_audio = [{"role": "HUMAN", "content": []}]
        if current_sys_prompt is not None and current_sys_prompt != "":
            current_sys_prompt = current_sys_prompt
        else:
            current_sys_prompt = None

        if video is not None:
            if isinstance(video, list):
                videos = video
            else:
                videos = [video]
            logger.info("llm activate video input")

            for single_video in videos:
                messages_video_and_audio[0]["content"].append(
                    {
                        "type": "video",
                        "video": single_video,
                        "sample": "uniform",
                        "max_frames": self.max_frames,
                    }
                )

        if image is not None:
            if isinstance(image, list):
                images = image
            else:
                images = [image]
            logger.info("llm activate image input")
            messages_video_and_audio[0]["content"].append(
                {"type": "image", "image": images}
            )

        if audio is not None:
            if isinstance(audio, list):
                audios = audio
            else:
                audios = [audio]
            logger.info("llm activate audio input")
            # audio = torch.from_numpy(audio).unsqueeze(0)
            for single_audio in audios:
                messages_video_and_audio[0]["content"].append(
                    {
                        "type": "audio",
                        "audio": single_audio,
                        "sample_rate": self.sample_rate,
                    }
                )
        if prompt is not None:
            logger.info("llm activate text input")
            messages_video_and_audio[0]["content"].append(
                {"type": "text", "text": prompt}
            )
        # self.manage_history_query_message()
        if len(history) > 0:
            messages_video_and_audio = history + messages_video_and_audio

        logger.info("In ming_sdk, prompt: " + str(messages_video_and_audio))
        if self.limit_images and self.limit_videos:
            messages_video_and_audio = self.filter_message(
                messages_video_and_audio, self.limit_images, self.limit_videos
            )
        logger.info(f"After filter, prompt: {messages_video_and_audio}, current_sys_prompt: {current_sys_prompt}, use_cot_system_prompt: {use_cot_system_prompt}")

        prompt = self.processor.apply_chat_template(
            messages_video_and_audio,
            sys_prompt_exp = current_sys_prompt,
            use_cot_system_prompt = use_cot_system_prompt
        )
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages_video_and_audio
        )

        compute_input_tokens_flag = kwargs.get("compute_input_tokens_flag", False)
        logger.info(f"In build_prompt, compute_input_tokens_flag: {compute_input_tokens_flag}")
        image_token_count, video_token_count, audio_token_count = None, None, None
        if compute_input_tokens_flag:
            image_token_count, video_token_count, audio_token_count = self.compute_image_audio_video_input_tokens(prompt, image_inputs, video_inputs, audio_inputs, **kwargs)
            logger.info(f"In build_prompt, image_token_count: {image_token_count}, video_token_count: {video_token_count}, audio_token_count: {audio_token_count}")

        requests = []
        inputs = LLMInputs(
            {
                "prompt": prompt,
            }
        )
        """"
        "image": image_inputs,
        "video": video_inputs,
        "audio": audio_inputs,
        """
        if image is not None or image_inputs is not None:
            if "multi_modal_data" in inputs.keys():
                inputs["multi_modal_data"]["image"] = image_inputs
            else:
                inputs["multi_modal_data"] = {"image": image_inputs}
        if video is not None or video_inputs is not None:
            if "multi_modal_data" in inputs.keys():
                inputs["multi_modal_data"]["video"] = video_inputs
            else:
                inputs["multi_modal_data"] = {"video": video_inputs}
        if audio is not None or audio_inputs is not None:
            if "multi_modal_data" in inputs.keys():
                inputs["multi_modal_data"]["audio"] = audio_inputs
            else:
                inputs["multi_modal_data"] = {"audio": audio_inputs}
        requests.append(inputs)
        return requests, image_token_count, video_token_count, audio_token_count

    def build_img_prompt(
        self,
        prompt: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Build a prompt for image generation based on input text and optional image.

        Args:
            text (str): The text prompt for image generation.
            image (Optional[Union[str, bytes, Image.Image]]): Optional input image (for editing mode).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            List[LLMInputs]: A list of LLM input objects containing the generated prompt and image data.

        Description:
            - Constructs a message structure for the model. If no image is provided, a dummy image is used.
            - The message order depends on whether an image is provided:
            - If `image is None`: [Text, Dummy Image]
            - Else: [Image, Text]
            - Applies the chat template to generate a text prompt.
            - Processes vision-related information (e.g., image inputs).
            - Returns LLM input objects with the prompt and multi-modal data.
        """
        if image is None:
            messages = [
                {
                    "role": "HUMAN",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": Image.new("RGB", (1, 1), (0, 0, 0))},
                    ],
                }
            ]
        else:
            if isinstance(image, str):
                messages = [
                    {
                        "role": "HUMAN",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            elif isinstance(image, list):
                messages = [
                    {
                        "role": "HUMAN",
                        "content": [],
                    }
                ]
                for img in image:
                    messages[0]["content"].append({"type": "image", "image": img})
                messages[0]["content"].append({"type": "text", "text": prompt})

        logger.info("Image task, prompt: " + str(messages))
        text = self.processor.apply_chat_template(
            messages
        )

        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages
        )

        compute_input_tokens_flag = kwargs.get("compute_input_tokens_flag", False)
        logger.info(f"In build_img_prompt, compute_input_tokens_flag: {compute_input_tokens_flag}")
        image_token_count, video_token_count, audio_token_count = None, None, None
        if compute_input_tokens_flag:
            image_token_count, video_token_count, audio_token_count = self.compute_image_audio_video_input_tokens(text, image_inputs, video_inputs, audio_inputs, **kwargs)
            logger.info(f"In build_img_prompt, image_token_count: {image_token_count}, video_token_count: {video_token_count}, audio_token_count: {audio_token_count}")

        requests = [
            LLMInputs({"prompt": text, "multi_modal_data": {"image": image_inputs}}),
        ]
        return requests, image_token_count, video_token_count, audio_token_count

    def build_img_gen_prompt(
        self,
        prompt: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        device: str = "cuda:0",
        image_gen_highres = False,
        image_gen_aspect_ratio = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare input data for image generation or editing.

        Args:
            text (str): The text prompt for image generation.
            image (Optional[Union[str, bytes, Image.Image]]): Optional input image (for editing mode).
            **kwargs: Additional keyword arguments (unused in this method).

        Returns:
            Dict[str, torch.Tensor]: A dictionary of processed inputs in tensor format, including text and multi-modal data.

        Description:
            - Constructs a message structure for the model. If no image is provided, only the text is included.
            - Applies the chat template to generate a text prompt.
            - Processes vision-related information (e.g., image inputs).
            - Converts the inputs into PyTorch tensors and moves them to the GPU.
            - Converts specific tensor types (e.g., pixel values) to `torch.bfloat16` for efficient inference.
        """
        if image is None:
            messages = [
                {
                    "role": "HUMAN",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
        else:
            if isinstance(image, str):
                messages = [
                    {
                        "role": "HUMAN",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
            elif isinstance(image, list):
                messages = [
                    {
                        "role": "HUMAN",
                        "content": [],
                    }
                ]
                for img in image:
                    messages[0]["content"].append({"type": "image", "image": img})
                messages[0]["content"].append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template(messages)
        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(
            messages
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            image_gen_highres=image_gen_highres,
            image_gen_aspect_ratio=image_gen_aspect_ratio,
        ).to(device)

        for k in inputs.keys():
            if k in [
                "pixel_values",
                "pixel_values_videos",
                "audio_feats",
                "pixel_values_reference",
            ]:
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)
        return inputs

    def _check_and_download_message(self, messages) -> Dict[str, int]:
        """
        Checks each message in the input list for media content and downloads it if necessary.
        """
        pass


class StreamGenerator:
    """
    Wrapper to convert an async generator to a synchronous iterator.

    This class enables iteration over async generators from synchronous
    code by running the async operations in a separate event loop.

    Args:
        loop: The asyncio event loop to use.
        async_generator: The async generator to wrap.
    """
    def __init__(self, loop, async_generator):
        self.loop = loop
        self.async_generator = async_generator

    def __iter__(self):
        return self

    def __next__(self):
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.async_generator.__anext__(), self.loop
            )
            return future.result()
        except StopAsyncIteration:
            raise StopIteration


class SimpleCache:
    """
    A simple local cache with TTL expiration and LRU eviction.

    Features:
        - Set and get values with optional TTL
        - Automatic expiration checking on access
        - LRU eviction when max_size is reached
        - Thread-unsafe (use ThreadSafeCache for concurrent access)

    Args:
        max_size (int): Maximum number of cached items. Defaults to 128.
        default_ttl (int): Default time-to-live in seconds. Defaults to 300.
    """

    def __init__(self, max_size: int = 128, default_ttl: int = 300):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of cached items (for LRU eviction).
            default_ttl: Default time-to-live in seconds.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, tuple] = OrderedDict()  # key -> (value, expire_time)

    def _is_expired(self, expire_time: float) -> bool:
        """Check if a cache entry has expired."""
        return time.time() > expire_time

    def get(self, key: str) -> Optional[Any]:
        """
        Get a cached value, automatically cleaning up expired items.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value if exists and not expired, else None.
        """
        if key not in self._cache:
            return None

        value, expire_time = self._cache[key]
        if self._is_expired(expire_time):
            del self._cache[key]
            return None

        # Move to end (mark as recently used for LRU)
        self._cache.move_to_end(key)
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key (str): Cache key.
            value (Any): Value to cache.
            ttl (Optional[int]): Time-to-live in seconds. Uses default_ttl if None.
        """
        ttl = ttl or self.default_ttl
        expire_time = time.time() + ttl

        # If already exists, delete first (to avoid order issues)
        if key in self._cache:
            del self._cache[key]

        # Check capacity and evict the oldest item if needed
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Pop the oldest (FIFO for LRU)

        self._cache[key] = (value, expire_time)

    def delete(self, key: str) -> bool:
        """
        Delete a specific key from the cache.

        Args:
            key (str): Cache key to delete.

        Returns:
            bool: True if key existed and was deleted, False otherwise.
        """
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all items from the cache."""
        self._cache.clear()

    def size(self) -> int:
        """Return the current number of cached items."""
        return len(self._cache)

    def keys(self) -> list:
        """
        Get all valid keys (excluding expired ones).

        Returns:
            list: List of non-expired cache keys.
        """
        valid_keys = []
        for k, (_, expire_time) in self._cache.items():
            if self._is_expired(expire_time):
                continue
            valid_keys.append(k)
        # Synchronously remove expired keys
        for k in [k for k in self._cache if self._is_expired(self._cache[k][1])]:
            del self._cache[k]
        return valid_keys


class ThreadSafeCache(SimpleCache):
    """
    Thread-safe version of SimpleCache using RLock.

    This class wraps SimpleCache with thread-safe operations for
    concurrent access scenarios.

    Args:
        max_size (int): Maximum number of cached items. Defaults to 128.
        default_ttl (int): Default time-to-live in seconds. Defaults to 300.
    """
    def __init__(self, max_size: int = 128, default_ttl: int = 300):
        super().__init__(max_size, default_ttl)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            return super().get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            super().set(key, value, ttl)

    def delete(self, key: str) -> bool:
        with self._lock:
            return super().delete(key)

    def clear(self) -> None:
        with self._lock:
            super().clear()
