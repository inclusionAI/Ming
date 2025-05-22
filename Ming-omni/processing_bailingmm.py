# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Processor class for BailingMM."""

import numpy as np
import sys
from typing import Iterable, List, Union, Dict, Optional, Tuple

import torch
from PIL import Image

if sys.version_info >= (3, 11):
    from typing import Unpack
else:
    from typing_extensions import Unpack

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from bailingmm_utils import process_vision_info

DEFAULT_IMAGE_PATCH_TOKEN = "<imagePatch>"
DEFAULT_IM_START_TOKEN = "<image>"
DEFAULT_IM_END_TOKEN = "</image>"
DEFAULT_VID_START_TOKEN = "<video>"
DEFAULT_VID_END_TOKEN = "</video>"
DEFAULT_GEN_IMAGE_PATCH_TOKEN = "<gen_imagePatch>"
DEFAULT_GEN_IM_START_TOKEN = "<gen_image>"
DEFAULT_GEN_IM_END_TOKEN = "</gen_image>"
PLACEHOLDER_IMAGE_TOKEN_IN_TEXT = "<imageHere>"
DEFAULT_END_OF_CHUNK_TOKEN = "<end_of_chunk>"

DEFAULT_END_OF_AUDIO_TOKEN = "<end_of_audio>"
DEFAULT_AUDIO_PATCH_TOKEN = "<audioPatch>"
DEFAULT_AU_START_TOKEN = "<audio>"
DEFAULT_AU_END_TOKEN = "</audio>"
DEFAULT_GEN_AUDIO_PATCH_TOKEN = "<gen_audioPatch>"
DEFAULT_GEN_AU_START_TOKEN = "<gen_audio>"
DEFAULT_GEN_AU_END_TOKEN = "</gen_audio>"
PLACEHOLDER_AUDIO_TOKEN_IN_TEXT = "<audioHere>"
DEFAULT_FRAME_PATCH_TOKEN = "<framePatch>"
DEFAULT_TEXT_TOKEN = '<text>'
DEFAULT_ASR_TOKEN = '<asr>'
DEFAULT_TTS_TOKEN = '<tts>'

USER_PREFIX = "<role>HUMAN</role>"
ASSISTANT_PREFIX = "<role>ASSISTANT</role>"


class BailingMMProcessorKwargs(ProcessingKwargs, total=False):
    # see processing_utils.ProcessingKwargs documentation for usage.
    _defaults = {
        "text_kwargs": {"padding": False, "padding_side": "right"},
        "image_kwargs": {},
        "video_kwargs": {},
        "audio_kwargs": {"padding": "max_length", "return_tensors": True, "use_whisper_encoder": False},
    }

class BailingMMProcessor(ProcessorMixin):
    r"""
    Constructs a BailingMM processor which wraps a bailingmm image processor, bailing audio processor and a LLaMa tokenizer into a single processor.
    Args:
        image_processor ([`BailingMMImageProcessor`], *optional*):
            The image processor is a required input.
        audio_processor ([`BailingMMAudioProcessor`], *optional*):
            The audio processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        num_audio_tokens (`int`, *optional*):
            Number of audio tokens for one video that will be returned by audio model.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        audio_token (`str`, *optional*, defaults to `"<audio>"`):
            Special token used to denote audio location.
    """

    attributes = ["image_processor", "audio_processor", "tokenizer"]
    optional_attributes = ["chat_template"]

    image_processor_class = "AutoImageProcessor"
    audio_processor_class = "AutoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    valid_kwargs = [
        "chat_template",
        "num_image_tokens",
        "image_token",
        "video_token",
        "audio_tokens"
    ]

    def __init__(
        self,
        image_processor=None,
        audio_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<image>",
        video_token="<video>",
        audio_token="<audio>",
        **kwargs: Unpack[BailingMMProcessorKwargs],
    ):
        self.image_token = image_token
        self.video_token = video_token
        self.audio_token = audio_token

        if chat_template is None:
            chat_template = tokenizer.chat_template

        self.gen_terminator = [tokenizer.convert_tokens_to_ids("<|endoftext|>")]
        super().__init__(image_processor, audio_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        videos: VideoInput = None,
        audios: Union[Tuple[np.ndarray, torch.Tensor, int], List[Tuple[np.ndarray, torch.Tensor, int]]] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        LlavaNextImageProcessor's [`~LlavaNextImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or torch Tensor.
                tensor. Both channels-first and channels-last formats are supported.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or torch Tensor.
            audios (`Tuple[torch.Tensor, int]`, `List[Tuple[torch.Tensor, int]]`):
                The sequence or batch of audios to be prepared. Each audio can be a 1D torch Tensor (with its sampling rate).
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as a list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **image_num_patches** -- Patch number to be fed to a model. Returned when `images` is not `None`.
            - **image_sizes** -- Size of each image that will be used to unpad an image. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of a video input to be fed to a model. Returned when `videos` is not `None`.
            - **pixel_values_audios** -- Pixel values of an audio input to be fed to a model. Returned when `audios` is not `None`.

        """
        output_kwargs = self._merge_kwargs(
            BailingMMProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        for key in output_kwargs.keys():
            if key != 'audio_kwargs' and 'use_whisper_encoder' in output_kwargs[key]:
                output_kwargs[key].pop('use_whisper_encoder')

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        image_inputs = {}
        video_inputs = {}
        audio_inputs = {}

        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
            text = self._expand_image_tokens(text, image_grid_thw)

        if videos is not None:
            video_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs["video_grid_thw"]
            text = self._expand_video_tokens(text, video_grid_thw)

        if audios is not None:
            audio_inputs = self.audio_processor(audios, **output_kwargs["audio_kwargs"])
            text = self._expand_audio_tokens(text, audio_inputs["encoder_feats_lengths"])

        # Padding side can be in TextKwargs but is not accepted by the tokenizer
        _ = output_kwargs["text_kwargs"].pop("padding_side", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if audios is not None:
            # Capture the location and length of the audio placeholders in the inputs.
            audio_start_token = self.tokenizer.convert_tokens_to_ids(DEFAULT_AU_START_TOKEN)
            loc_lens = []
            for i, input_ids_sample in enumerate(text_inputs["input_ids"]):
                loc_lens.append([
                    (input_ids_sample.tolist().index(audio_start_token) + 1, int(audio_inputs["encoder_feats_lengths"][i].item()))
                ])
            audio_inputs["audio_placeholder_loc_lens"] = torch.tensor(loc_lens, dtype=torch.long)
            audio_inputs.pop('encoder_feats_lengths')

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs, **audio_inputs})

    def apply_system_template(self, text):
        return USER_PREFIX

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]]],
        system_template: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            system_template (`Optional[str]`, *optional*):
                The system template. If not provided, the processor's sysyetm template is used.
            **kwargs:
                Additional keyword arguments
        """
        text = ""
        for idx, message in enumerate(conversation):
            assert message["role"] in ["HUMAN", "ASSISTANT"]
            if idx == len(conversation) - 1:
                message["role"] == "HUMAN"

            if message["role"] == "ASSISTANT":
                text += ASSISTANT_PREFIX

            image_counts = str(message["content"]).count("<image>")
            video_counts = str(message["content"]).count("<video>")
            audio_counts = str(message["content"]).count("<audio>")

            for content in message["content"]:
                if content["type"] == "image":
                    num_images = 1 if isinstance(content["image"], (str, Image.Image)) else len(content["image"])
                    if image_counts < num_images:
                        image_placeholder = "<image>\n" * (num_images - image_counts)
                        text += image_placeholder.rstrip("\n")
                # only one video supported now
                elif content["type"] == "video":
                    assert video_counts <= 1, "Video count must be at most 1!"
                    if video_counts == 0:
                        text += "<video>"
                elif content["type"] == "audio":
                    num_audios = 1 if isinstance(content["audio"], str) else len(content["audio"])
                    if audio_counts < num_audios:
                        audio_placeholder = "<audio>\n" * (num_audios - audio_counts)
                        text += audio_placeholder.rstrip("\n")
                elif content["type"] == "text":
                    text += content['text']

            if message["role"] == "ASSISTANT":
                 text += USER_PREFIX
            # text += "<|eot_id|>"
            
        if kwargs.get("add_generation_prompt", True):
            text += ASSISTANT_PREFIX

        sys_prompt = system_template if system_template is not None else self.apply_system_template(text)
        text = sys_prompt + text
        return text

    def process_vision_info(
        self,
        conversations,
    ):
        return process_vision_info(conversations)

    def _expand_image_tokens(
        self,
        text: List[TextInput],
        image_grid_thw: Union[List[int], int],
        special_token: str = "<image>",
    ):
        prompt_strings = []
        image_index = 0
        num_query_token = torch.prod(image_grid_thw, dim=1) // 4
        for sample in text:
            num_images = sample.count(special_token)
            if num_images > 0:
                for i in range(image_index, num_images + image_index):
                    img_text = DEFAULT_IM_START_TOKEN + num_query_token[i] * DEFAULT_IMAGE_PATCH_TOKEN + DEFAULT_IM_END_TOKEN + "\n"
                    sample = sample.replace(special_token, img_text, 1)
            image_index += num_images
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    def _expand_video_tokens(
        self,
        text: List[TextInput],
        video_grid_thw: Union[List[int], int],
        special_token: str = "<video>",
    ):
        prompt_strings = []
        video_index = 0
        num_query_token = torch.prod(video_grid_thw, dim=1) // 4
        for sample in text:
            num_videos = sample.count(special_token)
            if num_videos > 0:
                for i in range(video_index, num_videos + video_index):
                    video_text = num_query_token[i] * DEFAULT_IMAGE_PATCH_TOKEN
                    video_text = DEFAULT_VID_START_TOKEN + video_text + DEFAULT_VID_END_TOKEN + "\n"
                    sample = sample.replace(special_token, video_text, 1)
            video_index += num_videos
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    def _expand_audio_tokens(
        self,
        text: List[TextInput],
        audio_feats_lengths: torch.Tensor,
        special_token: str = "<audio>",
    ):
        prompt_strings = []
        for sample, audio_feats_length_tensor in zip(text, audio_feats_lengths):
            audio_text = (
                DEFAULT_AU_START_TOKEN + int(audio_feats_length_tensor.item()) * DEFAULT_AUDIO_PATCH_TOKEN + DEFAULT_AU_END_TOKEN
            )
            if special_token in sample:
                sample = sample.replace(special_token, audio_text)
            else:
                sample = sample + audio_text + "\n"
            prompt_strings.append(sample)
        text = [sample for sample in prompt_strings]
        return text

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names

        return list(
            dict.fromkeys(
                tokenizer_input_names + image_processor_input_names + audio_processor_input_names))
