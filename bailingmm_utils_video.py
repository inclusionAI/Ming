import os
import sys
import time
import math
import random
import logging
import warnings
from functools import lru_cache, partial
from collections import Counter
from typing import Optional, Callable, Union, Tuple, List
from packaging import version

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
from torchvision.transforms import v2

########

logger = logging.getLogger(__name__)

def is_torchcodec_available() -> bool:
    try:
        import torchcodec.decoders
        return True
    except Exception:
        return False

def is_decord_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("decord") is not None

def default_device() -> str:
    # Priority: cuda > mps > cpu
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def batched_resize(
        imgs: torch.Tensor, width: int, height: int,
        method=1, interp='bicubic', channel_first=False
        ) -> torch.Tensor:
    cv_interp = dict(
        bicubic=cv2.INTER_CUBIC,
        bilinear=cv2.INTER_LINEAR,
        nearest=cv2.INTER_NEAREST,
        area=cv2.INTER_AREA,
        lanczos=cv2.INTER_LANCZOS4,
    )[interp]
    v2_interp = dict(
        bicubic=v2.InterpolationMode.BICUBIC,
        bilinear=v2.InterpolationMode.BILINEAR,
        nearest=v2.InterpolationMode.NEAREST,
        area=v2.InterpolationMode.BOX,
        lanczos=v2.InterpolationMode.LANCZOS,
    )[interp]

    if method == 1: # OpenCV sequential resize
        if channel_first:
            imgs = imgs.permute(0, 2, 3, 1)  # NCHW to NHWC
        N = imgs.shape[0]
        C = imgs.shape[-1]
        imgs_np = imgs.cpu().numpy()
        imgs_np_resized = np.empty_like(imgs_np, shape=(N, height, width, C))
        for i in range(N):
            cv2.resize(
                imgs_np[i], (width, height), imgs_np_resized[i],
                interpolation=cv_interp
            )
        imgs = torch.from_numpy(imgs_np_resized)
        if channel_first:
            imgs = imgs.permute(0, 3, 1, 2)  # NHWC to NCHW
    else: # torchvision resize
        if not channel_first:
            imgs = imgs.permute(0, 3, 1, 2)  # NHWC to NCHW
        imgs = v2.functional.resize(imgs, [height, width], interpolation=v2_interp, antialias=True)
        if not channel_first:
            imgs = imgs.permute(0, 2, 3, 1)  # NCHW to NHWC
    return imgs

########
# Read video backends

def _read_video_torchcodec(
    video_path: str, sampler: Optional[Callable] = None,
) -> Tuple[torch.Tensor, float]:
    import torchcodec.decoders as tcd

    use_cuda = hasattr(tcd, 'set_cuda_backend') and torch.cuda.is_available()
    VideoDecoder = tcd.set_cuda_backend("beta")(tcd.VideoDecoder) if use_cuda else tcd.VideoDecoder

    decoder = VideoDecoder(
        source=video_path,
        dimension_order="NHWC",
        num_ffmpeg_threads=0,
        seek_mode="exact",
        device="cuda" if use_cuda else "cpu",
    )
    src_frames = decoder.metadata.num_frames
    src_fps = src_frames / decoder.metadata.duration_seconds
    if src_fps < 1.0 or src_fps > 240.0:
        print(f"warning: abnormal {src_fps=}")
        src_fps = min(max(src_fps, 1.0), 240.0)

    if sampler is None:
        smp_frames = src_frames
        frame_indices = list(range(smp_frames))
    else:
        smp_frames, frame_indices = sampler(src_fps, src_frames)
    smp_fps = smp_frames / max(src_frames, 1e-6) * src_fps
    video = decoder.get_frames_at(indices=frame_indices)
    video = video.data.cpu()

    return video, smp_fps

def _read_video_torchvision(
    video_path: str, sampler: Optional[Callable] = None,
    pts_unit: str = "sec", start_pts: float = 0.0, end_pts: float = None,
) -> Tuple[torch.Tensor, float]:
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")

    video, audio, info = torchvision.io.read_video(
        video_path, output_format="THWC",
        start_pts=start_pts, end_pts=end_pts, pts_unit=pts_unit,
    )
    src_frames, src_fps = video.size(0), info["video_fps"]

    if sampler is None:
        smp_frames = src_frames
    else:
        smp_frames, frame_indices = sampler(src_fps, src_frames)
        video = video[frame_indices]
    smp_fps = smp_frames / max(src_frames, 1e-6) * src_fps

    return video, smp_fps

def _read_video_decord(
    video_path: str, sampler: Optional[Callable] = None,
) -> Tuple[torch.Tensor, float]:
    import decord

    vr = decord.VideoReader(video_path)
    src_frames, src_fps = len(vr), vr.get_avg_fps()

    if sampler is None:
        smp_frames = src_frames
        frame_indices = list(range(smp_frames))
    else:
        smp_frames, frame_indices = sampler(src_fps, src_frames)
    smp_fps = smp_frames / max(src_frames, 1e-6) * src_fps
    video = vr.get_batch(frame_indices)
    video = torch.from_numpy(video.asnumpy()) # THWC

    return video, smp_fps

VIDEO_READER_BACKENDS = {
    "torchcodec": _read_video_torchcodec,
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}

FORCE_BAILINGNATIVE_VIDEO_READER = os.getenv("FORCE_BAILINGNATIVE_VIDEO_READER", None)

@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if FORCE_BAILINGNATIVE_VIDEO_READER is not None:
        video_reader_backend = FORCE_BAILINGNATIVE_VIDEO_READER
    elif is_torchcodec_available():
        video_reader_backend = "torchcodec"
    elif is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    print(f"bailing-native-utils using {video_reader_backend} to read video.", file=sys.stderr)
    return video_reader_backend

def load_video(video_path: str, sampler: Optional[Callable] = None) -> Tuple[torch.Tensor, float]:
    print(video_path)
    if isinstance(video_path, str):
        video_reader_backend = get_video_reader_backend()
        video_path = video_path.removeprefix("file://")
        st = time.time()
        try:
            video, smp_fps = VIDEO_READER_BACKENDS[video_reader_backend](video_path, sampler=sampler)
        except Exception as e:
            logger.warning(f"[{video_reader_backend}] error, fall back to torchvision: {e}")
            video, smp_fps = VIDEO_READER_BACKENDS["torchvision"](
                video_path, sampler=sampler
            )
        logger.info(f"[{video_reader_backend}] {video_path=}, {smp_fps=}, duration={time.time() - st:.3f}s")
    else:
        raise NotImplementedError("only support video path str input for now.")

    # output format: THWC
    return video, smp_fps

########
# V1 parameters

IMAGE_FACTOR = 32
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 16384 * 32 * 32
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 32 * 32
VIDEO_MAX_PIXELS = 768 * 32 * 32  # 4: 3 => 32: 24 (768) | 16:9 => 32:18 (576)
VIDEO_TOTAL_PIXELS = 96 * 128 * 32 * 32  # 9216: 24-72 frames | 7680: 10-60 frames | 6144: 8-48 frames

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 256

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

########
# V1 sample

def v1_sample_frames(num_frames, total_frames, sample="sequence"):
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

def v1_smart_nframes(
    ele: dict,
    total_frames: int,
    video_fps: Union[int, float],
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        ele (dict): a dict contains the configuration of video.
            support either `fps` or `nframes`:
                - nframes: the number of frames to extract for model inputs.
                - fps: the fps to extract frames for model inputs.
                    - min_frames: the minimum number of frames of the video, only used when fps is provided.
                    - max_frames: the maximum number of frames of the video, only used when fps is provided.
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
    max_frames = max(
        1,
        floor_by_factor(
            ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
        ),
    )
    if "nframes" in ele:
        nframes = min(total_frames, round_by_factor(ele["nframes"], FRAME_FACTOR), max_frames)
    else:
        fps = ele.get("max_video_fps", FPS)
        nframes = max(1, total_frames / video_fps * fps)
        if nframes > total_frames:
            logger.warning(f"smart_nframes: nframes[{nframes}] > total_frames[{total_frames}]")
        nframes = min(min(nframes, max_frames), total_frames)
    return int(nframes)


def v1_sample_video(video_fps, total_frames, ele: dict) -> List[int]:
    sample_method = ele.get("sample", "sequence")
    num_frames = v1_smart_nframes(ele, total_frames, video_fps)
    frame_indices = v1_sample_frames(
        num_frames=num_frames, total_frames=total_frames, sample=sample_method
    )

    return num_frames, frame_indices

########
# V1 pre-process

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

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

########
# V1 fetch video

def v1_fetch_video(
    ele: dict,
    image_factor: int = IMAGE_FACTOR,
    return_video_sample_fps: bool = False,
    return_video_timestamp: bool = False,
    return_metadata: bool = False,
) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        video, smp_fps = load_video(
            ele["video"], sampler=partial(v1_sample_video, ele=ele)
        )
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            num_frames, height, width, channels = video.shape
            min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
            total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
            max_pixels = max(total_pixels / num_frames * FRAME_FACTOR, int(min_pixels * 1.05))

            resized_height, resized_width = smart_resize(
                height, width,
                factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels,
            )
            print(f"fetch_video: {smp_fps=}, {num_frames=}, ({height}, {width}) => ({resized_height}, {resized_width})")
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        total_frames=len(ele['video'])
        sample_method = ele.get("sample", "sequence")
        smp_fps = process_info.pop("max_video_fps", 2.0)

        video = [
            torch.from_numpy(np.array(Image.open(video_element).convert("RGB")))
            for video_element in ele["video"]
        ]

        num_frames = v1_smart_nframes(ele, total_frames, video_fps=2.0)
        frame_indices = v1_sample_frames(
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
                height, width,
                factor=image_factor, min_pixels=min_pixels, max_pixels=max_pixels,
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
                img = v2.functional.resize(
                    img,
                    [major_h, major_w],
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                )
                arr = np.array(img)
                video[idx] = torch.from_numpy(arr)

        video = torch.stack(video, dim=0)

    logger.info(f"video-in: num_frames={video.shape[0]}, {resized_height=}, {resized_width=}")
    video = batched_resize(video, resized_width, resized_height, method=1, interp='bicubic', channel_first=False)
    video = video.permute(0, 3, 1, 2)

    resmp_fps = smp_fps
    resmp_ts = np.array([i / smp_fps for i in range(video.shape[0])])

    num_pad_frames = FRAME_FACTOR - (len(video) % FRAME_FACTOR)
    if num_pad_frames < FRAME_FACTOR:
        video = torch.cat([video] + [video[-1:]] * num_pad_frames, dim=0)
        resmp_ts = np.concatenate(
            [resmp_ts, np.array([resmp_ts[-1] * num_pad_frames])],
        )

    if return_metadata:
        metadata = {
            "resmp_ts": resmp_ts,
            "resmp_fps": resmp_fps,
        }
        return video, metadata
    if return_video_timestamp:
        return video, resmp_ts
    if return_video_sample_fps:
        return video, resmp_fps
    return video


########
# V2 parameters

IMAGE_FACTOR = 16 * 2
FRAME_FACTOR = int(os.environ.get('FRAME_FACTOR', 2))
MAX_FPS = float(os.environ.get('MAX_FPS', 2.0))
MIN_PATCHES_PER_FRAME = int(os.environ.get('MIN_PATCHES_PER_FRAME', 144))
MAX_PATCHES_PER_FRAME = int(os.environ.get('MAX_PATCHES_PER_FRAME', 3072))
MAX_TOKENS = int(os.environ.get('MAX_TOKENS', 3072))
MAX_PATCHES = MAX_TOKENS * FRAME_FACTOR
MAX_PATCHES_OS = MAX_PATCHES * int(os.environ.get('MAX_PATCHES_OS_MULTIPLIER', 4))
MAX_FRAMES_OS = MAX_PATCHES_OS // MIN_PATCHES_PER_FRAME

SUBSAMPLE_PIXELS = 16384
SAD_MEAN_ANCHOR = SUBSAMPLE_PIXELS * int(os.environ.get('SAD_MEAN_ANCHOR_MULTIPLIER', 20))
SAD_SUM_RATIO_SCALE = 1.0
SAD_SUM_RATIO_OFFSET = 0.0
MAX_SAMPLE_INTERVAL = float(os.environ.get('MAX_SAMPLE_INTERVAL', 4.0))

########
# V2 sample

def v2_sample_video(
    src_fps: float,
    src_frames: int,
) -> Tuple[int, float]:
    """calculate the number of frames and fps for video used for model inputs.
        Args:
        ele (dict): a dict contains the configuration of video.
        src_fps (float): the original fps of the video.
        src_frames (int): the original total number of frames of the video.
    Returns:
        Tuple[int, float]: the number of frames and fps for video used for model inputs.
    """
    # init sample params
    smp_fps = min(MAX_FPS, src_fps)
    smp_frames = max(1, min(MAX_FRAMES_OS, int(src_frames / src_fps * smp_fps + 0.5)))
    smp_fps = smp_frames / max(src_frames / src_fps, 1e-6)
    print(f"v2_sample_video: {src_fps=} -> {smp_fps=}, {src_frames=} -> {smp_frames=}")
    frame_indices = np.linspace(0, src_frames - 1, smp_frames, dtype=int)
    return smp_frames, frame_indices

########
# V2 pre-process

def ts2cv(imgs_ts: torch.Tensor, channel_first=False, dtype=None) -> np.ndarray:
    if channel_first:
        imgs_ts = imgs_ts.permute(0, 2, 3, 1)
    imgs_cv = imgs_ts.to(dtype=dtype, device="cpu").numpy()  # shape (N, H, W, C)
    return imgs_cv

def cv2ts(imgs_cv: np.ndarray, channel_first=False, dtype=None, device=None) -> torch.Tensor:
    if not channel_first:
        imgs_cv = imgs_cv.transpose(0, 3, 1, 2)  # shape (N, C, H, W)
    imgs_ts = torch.from_numpy(imgs_cv).to(dtype=dtype, device=device)
    return imgs_ts

def cv_cvtColor(imgs: torch.Tensor, code: int=cv2.COLOR_RGB2GRAY) -> torch.Tensor:
    # using OpenCV for RGB to Gray conversion
    gray_imgs = np.array([cv2.cvtColor(img, code) for img in imgs])  # shape (N, H, W, C)
    if len(gray_imgs.shape) == 3:
        gray_imgs = gray_imgs[:, :, :, np.newaxis]  # shape (N, H, W, 1)
    return gray_imgs

def cv_resize(imgs: np.ndarray, width: int, height: int, interp='bicubic') -> np.ndarray:
    cv_interp = dict(
        bicubic=cv2.INTER_CUBIC,
        bilinear=cv2.INTER_LINEAR,
        nearest=cv2.INTER_NEAREST,
        area=cv2.INTER_AREA,
        lanczos=cv2.INTER_LANCZOS4,
    )[interp]

    N, H, W, C = imgs.shape
    imgs_resized = np.empty_like(imgs, shape=(N, height, width, C))
    for i in range(N):
        cv2.resize(
            imgs[i], (width, height), imgs_resized[i],
            interpolation=cv_interp
        )
    return imgs_resized

def cv_frames_SAD(frames: np.ndarray) -> np.ndarray:
    """calculate the sum of absolute differences (SAD) between consecutive frames.
        Args:
        frames (np.ndarray): the input video frames, shape (T, H, W, C).
    Returns:
        np.ndarray: the SAD values between consecutive frames, shape (T-1,).
    """
    diffs = cv2.absdiff(frames[1:], frames[:-1])  # shape (T-1, H, W, C)
    sad_values = diffs.sum(axis=(1, 2, 3))  # shape (T-1,)
    return sad_values

def compute_sad_sum_ratio(x: np.ndarray, anchor, scale, offset) -> float:
    y = np.log(x + 1) - np.log(anchor)
    y = np.tanh(y) * (0.5 * scale) - 0.5
    y = np.clip(offset - y, 0, 1)
    return y

def resample_SAD(frames: torch.Tensor, smp_fps: float) -> Tuple[List[int], np.ndarray]:
    """resample video frames based on SAD values.
        Args:
        frames (torch.Tensor): the input video frames, shape (T, C, H, W).
    Returns:
        Tuple[List[int], np.ndarray]: the resampled video frame indices and SAD values.
    """
    if len(frames) <= 1:
        return list(range(len(frames))), np.array([])

    max_sample_interval = max(1, int(MAX_SAMPLE_INTERVAL * smp_fps + 0.5))

    # compute SAD values
    _frames = ts2cv(frames)
    scale_ratio = SUBSAMPLE_PIXELS / (_frames.shape[-3] * _frames.shape[-2])
    sub_width = int(_frames.shape[-2] * scale_ratio ** 0.5 + 0.5)
    sub_height = int(_frames.shape[-3] * scale_ratio ** 0.5 + 0.5)
    _frames = cv_resize(_frames, width=sub_width, height=sub_height, interp='bilinear')
    _frames = cv_cvtColor(_frames, cv2.COLOR_RGB2Lab)
    sad_values = cv_frames_SAD(_frames)

    # compute sorted SAD and its integral
    sad_sorted = np.sort(sad_values)
    sad_integral = np.cumsum(sad_sorted)
    sad_sum = sad_integral[-1]
    sad_mean = sad_sum / len(sad_values)

    # compute dynamic SAD threshold
    sad_sum_ratio = compute_sad_sum_ratio(
        sad_mean,
        anchor=SAD_MEAN_ANCHOR,
        scale=SAD_SUM_RATIO_SCALE,
        offset=SAD_SUM_RATIO_OFFSET,
    )
    print(f"{sad_mean=}, {sad_sum_ratio=}")

    # compute SAD threshold
    sad_sum_thr = sad_sum * sad_sum_ratio
    sad_idx = np.searchsorted(sad_integral, sad_sum_thr)
    sad_value_thr = sad_sorted[sad_idx]
    print(f"{sad_sum=}, {sad_sum_thr=}, {sad_idx=}, {sad_value_thr=}")

    # filter frames based on SAD threshold
    key_frame_indices = [0]
    for i, sad in enumerate(sad_values):
        if sad >= sad_value_thr:
            key_frame_indices.append(i + 1)
    if key_frame_indices[-1] <= len(sad_values) - max_sample_interval:
        key_frame_indices.append(len(sad_values))

    # ensure max sample interval
    key_frame_indices_new = [key_frame_indices[0]]
    for i in range(1, len(key_frame_indices)):
        sample_interval = key_frame_indices[i] - key_frame_indices[i - 1]
        if sample_interval >= max_sample_interval:
            num_additional_frames = sample_interval // max_sample_interval
            additional_frames_interval = sample_interval / (num_additional_frames + 1)
            for j in range(1, num_additional_frames + 1):
                new_idx = key_frame_indices[i - 1] + int(j * additional_frames_interval + 0.5)
                key_frame_indices_new.append(new_idx)
        if key_frame_indices_new[-1] != key_frame_indices[i]:
            key_frame_indices_new.append(key_frame_indices[i])
    key_frame_indices = key_frame_indices_new
    print(f"num_key_frames={len(key_frame_indices)}, {key_frame_indices=}, ")

    return key_frame_indices, sad_values

########
# V2 fetch video

def v2_fetch_video(
        ele: dict, image_factor: int = IMAGE_FACTOR,
        return_video_sample_fps: bool = False,
        return_video_timestamp: bool = False,
        return_metadata: bool = False,
        slice_frames: slice | None = None,
        ) -> torch.Tensor | list[Image.Image]:
    # load video
    video, smp_fps = load_video(ele["video"], sampler=v2_sample_video)
    if slice_frames is not None:
        video = video[slice_frames]
    smp_video = video

    # resample frames
    smp_frames = len(video)
    smp_tokens = (video.shape[-3] // image_factor) * (video.shape[-2] // image_factor) * (smp_frames // FRAME_FACTOR)
    resmp_indices, sad_values = resample_SAD(video, smp_fps)
    resmp_ts = np.array(resmp_indices) / smp_fps
    video = video[resmp_indices]
    resmp_video = video
    # duplicate last frame to make it divisible by FRAME_FACTOR
    num_pad_frames = FRAME_FACTOR - (len(video) % FRAME_FACTOR)
    if num_pad_frames < FRAME_FACTOR:
        video = torch.cat([video] + [video[-1:]] * num_pad_frames, dim=0)
        resmp_ts = np.concatenate(
            [resmp_ts, np.array([resmp_ts[-1] * num_pad_frames])],
        )
    resmp_fps = len(video) / smp_frames * smp_fps if smp_frames > 0 else smp_fps

    # resize frames
    resmp_frames, src_h, src_w, channels = video.shape
    src_patches = (src_h * src_w) / (image_factor * image_factor)
    patches_per_frame = max(MIN_PATCHES_PER_FRAME, min(MAX_PATCHES_PER_FRAME, MAX_PATCHES / resmp_frames))
    scale_ratio = (patches_per_frame / src_patches) ** 0.5
    dst_w = int(src_w * scale_ratio / image_factor + 0.5) * image_factor
    dst_h = int(src_h * scale_ratio / image_factor + 0.5) * image_factor
    resmp_tokens = (dst_h // image_factor) * (dst_w // image_factor) * (resmp_frames // FRAME_FACTOR)

    print(f"fetch_video_v2: {smp_tokens=} -> {resmp_tokens=}, {smp_fps=} -> {resmp_fps=}, "
          f"{smp_frames=} -> {resmp_frames=}, ({src_w}, {src_h}) => ({dst_w}, {dst_h})")
    video = batched_resize(video, dst_w, dst_h, method=1, interp='bicubic', channel_first=False)
    video = video.permute(0, 3, 1, 2)

    if return_metadata:
        metadata = {
            # "smp_fps": smp_fps,
            # "smp_frames": smp_frames,
            # "smp_tokens": smp_tokens,
            # "smp_video": smp_video,
            # "sad_values": sad_values,
            # "resmp_indices": resmp_indices,
            "resmp_ts": resmp_ts,
            # "resmp_video": resmp_video,
            "resmp_fps": resmp_fps,
            # "resmp_frames": resmp_frames,
            # "resmp_tokens": resmp_tokens,
        }
        return video, metadata
    if return_video_timestamp:
        return video, resmp_ts
    if return_video_sample_fps:
        return video, resmp_fps
    return video

########
