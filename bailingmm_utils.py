from __future__ import annotations

import base64
import logging
import math
import os
import sys
import time
import warnings
from functools import lru_cache
from io import BytesIO
from collections import Counter

import random
import numpy as np

import requests
import torch
import torchvision
from packaging import version

from PIL import Image
import torchaudio
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from typing import Union, Tuple, List

logger = logging.getLogger(__name__)

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28  # 4: 3 => 32: 24 (768) | 16:9 => 32:18 (576)
VIDEO_TOTAL_PIXELS = 96 * 128 * 28 * 28  # 9216: 24-72 frames | 7680: 10-60 frames | 6144: 8-48 frames

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 128

VideoInput = Union[
    List["Image.Image"],
    "np.ndarray",
    "torch.Tensor",
    List["np.ndarray"],
    List["torch.Tensor"],
    List[List["Image.Image"]],
    List[List["np.ndarrray"]],
    List[List["torch.Tensor"]],
]

def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def is_image(image_file):
    if isinstance(image_file, str) and (image_file.startswith("base64,") or image_file.lower().endswith(
            ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
        return True
    elif isinstance(image_file, Image.Image):
        return True
    else:
        return False

def is_video(video_file):
    if isinstance(video_file, str) and video_file.lower().endswith(
            ('.mp4', '.mkv', '.avi', '.wmv', '.iso', ".webm")):
        return True
    else:
        return False

def is_audio(audio_file):
    if isinstance(audio_file, str) and audio_file.lower().endswith(
            (".wav", ".mp3", ".aac", ".flac", ".alac", ".m4a", ".ogg", ".wma", ".aiff", ".amr", ".au")):
        return True
    else:
        return False

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image

def sample_frames(num_frames, total_frames, sample="random"):
    if sample == "sequence":
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        intervals = np.linspace(start=0, stop=total_frames, num=num_frames + 1, dtype=int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "random":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(total_frames)[:num_frames]
                frame_indices.sort()
                frame_indices = list(frame_indices)
            if len(frame_indices) < num_frames:
                padded_frame_indices = [frame_indices[-1]] * num_frames
                padded_frame_indices[:len(frame_indices)] = frame_indices
                frame_indices = padded_frame_indices
        elif sample == "uniform":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
            if len(frame_indices) < num_frames:
                frame_indices = [
                    frame_indices[int((num_frames - 1) * i / (num_frames - 1) + 0.5)] for i in range(num_frames)
                ]
        else:
            raise NotImplementedError
    return frame_indices

def get_frames(
    ele: dict,
    total_frames: int,
) -> int:
    """calculate the number of frames for video used for model inputs.
        Args:
        ele (dict): a dict contains the configuration of video.
        total_frames (int): the original total number of frames of the video.
    Returns:
        int: the number of frames for video used for model inputs.
    """
    if "nframes" in ele:
        num_frames = round_by_factor(ele["nframes"], FRAME_FACTOR)
    else:
        min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
        max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR)
        num_frames = max(min(total_frames, max_frames), min_frames)
        num_frames = floor_by_factor(num_frames, FRAME_FACTOR)

    if not (FRAME_FACTOR <= num_frames <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {num_frames}.")
    return num_frames

def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using torchvision.io.read_video
    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]

    sample_method = ele.get("sample", "sequence")
    pts_unit = "sec" if sample_method == "sequence" else "pts"
    st = time.time()
    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end", None),
        pts_unit=pts_unit,
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.info(f"torchvision:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    max_video_fps = ele.get("max_video_fps", 2.0)

    if video_fps > max_video_fps and total_frames / float(video_fps) > 4.0:
        num_frames = get_frames(ele, int(total_frames / float(video_fps) * max_video_fps))
    else:
        num_frames = get_frames(ele, total_frames)

    # num_frames = get_frames(ele, total_frames)

    frame_indices = sample_frames(
        num_frames=num_frames, total_frames=total_frames, sample=sample_method
    )
    video = video[frame_indices]
    sample_fps = num_frames / max(total_frames, 1e-6) * video_fps
    return video, sample_fps

def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float):
    """read video using decord.VideoReader

    Args:
        ele (dict): a dict contains the configuration of video.
        support keys:
            - video: the path of video. support "file://", "http://", "https://" and local path.
            - video_start: the start time of video.
            - video_end: the end time of video.
    Returns:
        torch.Tensor: the video tensor with shape (T, C, H, W).
    """
    import decord
    video_path = ele["video"]

    st = time.time()
    vr = decord.VideoReader(video_path)
    if 'video_start' in ele or 'video_end' in ele:
        raise NotImplementedError("not support start_pts and end_pts in decord for now.")
    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.info(f"decord:  {video_path=}, {total_frames=}, {video_fps=}, time={time.time() - st:.3f}s")

    sample_method = ele.get("sample", "sequence")
    max_video_fps = ele.get("max_video_fps", 2.0)

    if video_fps > max_video_fps and total_frames / float(video_fps) > 4.0:
        num_frames = get_frames(ele, int(total_frames / float(video_fps) * max_video_fps))
    else:
        num_frames = get_frames(ele, total_frames)
    frame_indices = sample_frames(
        num_frames=num_frames, total_frames=total_frames, sample=sample_method
    )

    video = vr.get_batch(frame_indices).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format
    sample_fps = num_frames / max(total_frames, 1e-6) * video_fps
    return video, sample_fps

VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_BAILINGNATIVE_VIDEO_READER = os.getenv("FORCE_BAILINGNATIVE_VIDEO_READER", None)

@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_BAILINGNATIVE_VIDEO_READER is not None:
        video_reader_backend = FORCE_BAILINGNATIVE_VIDEO_READER
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"bailing-native-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR, return_video_sample_fps: bool = False) -> torch.Tensor | \
                                                                                                       list[
                                                                                                           Image.Image]:
    if isinstance(ele["video"], str):
        if ele["video"].startswith("file://"):
            ele["video"] = ele["video"][7:]
        video_reader_backend = get_video_reader_backend()
        try:
            video, sample_fps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps = VIDEO_READER_BACKENDS["torchvision"](ele)

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            num_frames, _, height, width = video.shape
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
            # max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / num_frames * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = max(total_pixels / num_frames * FRAME_FACTOR, int(min_pixels * 1.05))

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        total_frames=len(ele['video'])
        sample_method = ele.get("sample", "sequence")
        sample_fps = process_info.pop("max_video_fps", 2.0)
 
        video = [
            torch.from_numpy(np.array(Image.open(video_element).convert("RGB")))
            for video_element in ele["video"]
        ]
        
        num_frames = get_frames(ele, total_frames)
        frame_indices = sample_frames(
            num_frames=num_frames, total_frames=total_frames, sample=sample_method
        )

        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            height, width, _ = video[1].shape
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
            # max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / num_frames * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = max(total_pixels / num_frames * FRAME_FACTOR, int(min_pixels * 1.05))

            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=28,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        # 3) gather shapes of the sampled frames
        shapes = [ video[i].shape[:2] for i in frame_indices ]  # list of (H,W)
        # pick the shape that occurs most often
        
        major_shape, _ = Counter(shapes).most_common(1)[0]
        major_h, major_w = major_shape

        # VNIAH could have one outlier shape
        for idx in frame_indices:
            h, w = video[idx].shape[:2]
            if (h, w) != (major_h, major_w):
                # re-open, resize via PIL + TF, then to tensor
                img = Image.open(ele["video"][idx]).convert("RGB")
                img = transforms.functional.resize(
                    img,
                    [major_h, major_w],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                arr = np.array(img)
                video[idx] = torch.from_numpy(arr)

        video = torch.stack(video, dim=0)
        video = video.permute(0, 3, 1, 2).float()
    logger.info("video-in:  num_frames={}, resized_height={}, resized_width={}".format(video.shape[0],resized_height,resized_width))
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    if return_video_sample_fps:
        return video, sample_fps
    return video

def fetch_audio(ele: dict[str, str | torch.Tensor], return_tensor="pt") -> Tuple[Union[torch.Tensor, np.ndarray], int]:
    if "audio" in ele:
        audio = ele["audio"]
    else:
        audio = ele["audio_url"]

    if isinstance(audio, torch.Tensor):
        waveform = audio
        sample_rate: int = ele.get("sample_rate", 16000)
    elif audio.startswith("http://") or audio.startswith("https://"):
        audio_file = BytesIO(requests.get(audio, stream=True).content)
        waveform, sample_rate = torchaudio.load(audio_file)
    elif audio.startswith("file://"):
        waveform, sample_rate = torchaudio.load(audio[7:])
    else:
        waveform, sample_rate = torchaudio.load(audio)
    if return_tensor == "pt":
        return waveform, sample_rate
    else:
        return waveform.numpy(), sample_rate

def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or "video_url" in ele
                            or "audio" in ele
                            or "audio_url" in ele
                            or ele["type"] in ["image", "image_url", "video", "video_url", "audio", "audio_url"]
                    ):
                        vision_infos.append(ele)
    return vision_infos

def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, list[
    torch.Tensor | list[np.ndarray]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images, videos or audios
    image_inputs = []
    video_inputs = []
    audio_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            if "image" in vision_info and isinstance(vision_info["image"], (tuple, list)):
                for i in range(len(vision_info["image"])):
                    image_inputs.append(fetch_image({"type": "image", "image": vision_info["image"][i]}))
            elif "image_url" in vision_info and vision_info["image_url"].get("url", None) is not None:
                vision_info["image_url"] = vision_info["image_url"].get("url")
                image_inputs.append(fetch_image(vision_info))
            else:
                image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info or "video_url" in vision_info:
            if "video" in vision_info and is_video(vision_info['video']):
                data_value = vision_info['video']
            elif "video_url" in vision_info and vision_info["video_url"].get("url", None) is not None:
                data_value = vision_info["video_url"].get("url")
            else:
                data_value = [os.path.join(vision_info['video'], frame) for frame in sorted(os.listdir(vision_info['video']))]
            vision_info['video']=data_value
            video_inputs.append(fetch_video(vision_info))
        elif "audio" in vision_info or "audio_url" in vision_info:
            if "audio" in vision_info and isinstance(vision_info["audio"], (tuple, list)):
                audio_inputs.extend(fetch_audio(info) for info in vision_info["audio"])
            elif "audio_url" in vision_info and vision_info["audio_url"].get("url", None) is not None:
                vision_info["audio_url"] = vision_info["audio_url"].get("url")
                audio_inputs.append(fetch_audio(vision_info))
            else:
                audio_inputs.append(fetch_audio(vision_info))
        else:
            raise ValueError("image, image_url, video, video_url, audio or audio_url should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    if len(audio_inputs) == 0:
        audio_inputs = None
    return image_inputs, video_inputs, audio_inputs

def get_closest_ratio(height: float, width: float, aspect_ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(aspect_ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return aspect_ratios[closest_ratio], float(closest_ratio)

def process_ratio(ori_h, ori_w):
    ASPECT_RATIO_512 = {
        "0.25": [256, 1024],
        "0.26": [256, 992],
        "0.27": [256, 960],
        "0.28": [256, 928],
        "0.32": [288, 896],
        "0.33": [288, 864],
        "0.35": [288, 832],
        "0.4": [320, 800],
        "0.42": [320, 768],
        "0.48": [352, 736],
        "0.5": [352, 704],
        "0.52": [352, 672],
        "0.57": [384, 672],
        "0.6": [384, 640],
        "0.68": [416, 608],
        "0.72": [416, 576],
        "0.78": [448, 576],
        "0.82": [448, 544],
        "0.88": [480, 544],
        "0.94": [480, 512],
        "1.0": [512, 512],
        "1.07": [512, 480],
        "1.13": [544, 480],
        "1.21": [544, 448],
        "1.29": [576, 448],
        "1.38": [576, 416],
        "1.46": [608, 416],
        "1.67": [640, 384],
        "1.75": [672, 384],
        "2.0": [704, 352],
        "2.09": [736, 352],
        "2.4": [768, 320],
        "2.5": [800, 320],
        "2.89": [832, 288],
        "3.0": [864, 288],
        "3.11": [896, 288],
        "3.62": [928, 256],
        "3.75": [960, 256],
        "3.88": [992, 256],
        "4.0": [1024, 256],
    }
    closest_size, _ = get_closest_ratio(ori_h, ori_w, aspect_ratios=ASPECT_RATIO_512)
    closest_size = list(map(lambda x: int(x), closest_size))
    if closest_size[0] / ori_h > closest_size[1] / ori_w:
        resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
    else:
        resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
    return closest_size, resize_size