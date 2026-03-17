import os
import sys
import time
import queue
import shutil
import logging
import time
from PIL import Image
import threading
import multiprocessing
import queue
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

from ming_sdk.usage import Usage
from ming_sdk.ming_moe import MingMOE
from ming_sdk.ming_utils import MingUtils
from ming_sdk.ming_talker import MingTalker
from ming_sdk.ming_moe_async import MingMOEAsync
from ming_sdk.monitoring.request_metrics import (
    metrics_text,
    metrics_image,
    metrics_speech,
    metrics_tts,
    metrics_speech_text_audio
)
from ming_sdk.ming_img import MingImg, ratio_extraction_fromat, rewrite_fromat, rewrite_edit_fromat, image_gen_indent_format, auto_balance_saturation_exposure, DEFAUL_PROMPT_FOR_NO_INTENT
from ming_sdk.ming_utils import ThreadSafeCache

logger = logging.getLogger()
warnings.filterwarnings("ignore")
current_file_path = os.path.abspath(__file__)
current_dir_path = os.path.dirname(current_file_path)
sys.path.insert(0, current_dir_path)


class Ming(object):
    """
    Initialize the class with model components and configuration.

     Args:
            model_path (str):
                Path to the root model directory. Must contain:
                - `config.json`: Model architecture and tokenizer config
                - `am.mvn`: Audio normalization stats for TTS frontend
                - Subdirectories: `talker/`, `talker/vae`, and optionally diffusion checkpoints

            sys_prompt (str, optional):
                System-level instruction prepended to all conversations (e.g., "You are a helpful AI").
                If not provided, no system prompt is used. Defaults to "".

            device (str, optional):
                GPU device IDs for tensor parallelism in the LLM (vLLM backend).
                Format: comma-separated integers, e.g., "0", "0,1", "0,1,2,3".
                Determines `tensor_parallel_size`. Defaults to "0" (single GPU).

            gpu_memory_utilization (dict, optional):
                Fraction of GPU memory to allocate for specific modules.
                Helps prevent OOM errors on resource-limited devices.
                Supported keys:
                    - "moe": for LLM (default: 0.6)
                    - "talker": for TTS model (default: 0.1)
                Example: {"moe": 0.8, "talker": 0.2}

            device_map (dict, optional):
                Assign different modules to different CUDA devices for heterogeneous deployment.
                Supported keys:
                    - "talker": TTS model (default: "cuda:0")
                    - "image": Image generation model (default: "cuda:0")
                Example: {"talker": "cuda:1", "image": "cuda:0"} enables cross-GPU deployment.
            with_async (bool, optional):
                Enabling async may change the public API surface to coroutine-based methods; callers should run within an asyncio event loop.
    """

    def __init__(
        self,
        model_path: str,
        sys_prompt: str = "",
        device: str = "0",
        gpu_memory_utilization: dict = {"moe": 0.6, "talker": 0.1},
        limit_mm_per_prompt: dict = {"image": 10, "video": 2},
        device_map: dict = {"talker": ["cuda:0"], "image": "cuda:0"},
        with_async: bool = False,
        speaker: str = "DB30",
        quantization: str | None = None,
        use_talker: bool = True,
        use_image_gen: bool = False
    ):
        logger.info(
            f"gpu_memory_utilization={gpu_memory_utilization},model_path={model_path},limit_mm_per_prompt={limit_mm_per_prompt},device={device},device_map={device_map}"
        )
        tensor_parallel_size = (
            len(device.split(",")) if len(device.split(",")) > 0 else 1
        )
        shutil.copy(model_path + "/config.json", current_dir_path)
        am_path = os.path.join(model_path, "am.mvn")
        shutil.copy(am_path, ".")

        # 1. Initialize talker (TTS module)
        self.utils = MingUtils(model_path=current_dir_path, sys_prompt=sys_prompt)
        if use_talker:
            self.talker = MingTalker(
                model_path=model_path,
                device_list=device_map["talker"],
            )

        # 2. Initialize MOE (Mixture of Experts LLM)
        os.environ["VLLM_USE_V1"] = "0"
        if with_async:
            self.moe = MingMOEAsync(
                model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization["moe"],
                limit_mm_per_prompt=limit_mm_per_prompt,
                quantization=quantization, 
            )
        else:
            self.moe = MingMOE(
                model_path,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization["moe"],
                limit_mm_per_prompt=limit_mm_per_prompt,
                quantization=quantization, 
            )

        # 3. Initialize image generation module
        if use_image_gen:
            self.img = MingImg(model_path, device=device_map["image"])
        self.device_map = device_map
        self.speaker = speaker
        self.queue_manager = multiprocessing.Manager()
        self.info_cache_default_time = 1200
        self.info_cache = ThreadSafeCache(max_size=500, default_ttl=self.info_cache_default_time)

    def _generate_text(
        self,
        prompt: str,
        audio: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        video: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        history: list = [],
        **kwargs,
    ) -> Any:
        """
        Generate text output based on the input prompt.

        Args:
            prompt (str): User input text.
            audio (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Audio data (e.g., file path or binary or list).
            video (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Video data (e.g., file path or binary or list).
            image (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Image data (file path, binary, or PIL Image or list).
            history (list, optional): Conversation history. Defaults to empty list.
            **kwargs: Additional parameters for the model.
        Returns:
            str: Generated text output.
        """
        msg_request_id = kwargs.get("msg_request_id", None)
        state = metrics_text.create_state(stream_mode=False)
        text_token_count = self.utils.compute_text_input_tokens(prompt, **kwargs)
        inputs, image_token_count, video_token_count, audio_token_count = self.utils.build_prompt(
            prompt=prompt,
            audio=audio,
            video=video,
            image=image,
            history=history,
            **kwargs,
        )

        gen_text, usage = self.moe.generate(inputs, **kwargs)
        usage = Usage.update_usage_by_processor(usage, text_token_count, image_token_count, video_token_count, audio_token_count)
        state.input_token_length = usage["prompt_tokens"]
        state.output_token_length = usage["generated_tokens"]
        state.finish("success", msg_request_id=msg_request_id)

        return gen_text, usage

    def _rewrite_for_image_gen(
        self,
        prompt: str,
    ):

        logger.info(f"In _rewrite_for_image_gen, user input {prompt}")
        rewrite_input = rewrite_fromat.format(prompt)
        prompt = self._generate_text(rewrite_input)[0]
        return prompt

    def _is_image_gen_intent(
        self,
        prompt: str,
    ):

        logger.info(f"_is_image_gen_intent, user input {prompt}")
        indent_input = image_gen_indent_format.format(prompt)
        prompt = self._generate_text(indent_input)[0]
        return prompt.lower().strip() != "no"

    def _rewrite_for_image_edit(
        self,
        prompt: str,
    ):

        logger.info(f"_rewrite_for_image_edit, user input {prompt}")
        rewrite_input = rewrite_edit_fromat.format(prompt)
        prompt = self._generate_text(rewrite_input)[0]
        return prompt

    def _extract_aspect_ratio_for_image_gen(
        self,
        prompt: str,
    ):

        t1 = time.time()
        ratio_extraction_input = ratio_extraction_fromat.format(prompt)
        ratio_text = self._generate_text(ratio_extraction_input)[0]
        logger.info(f"Extract_aspect_ratio_for_image_gen, cost {time.time() - t1}s")
        ratio = None
        if ":" in ratio_text:
            split_char = ":"
            if ratio_text.count(split_char) == 1:
                try:
                    ratio = float(ratio_text.split(split_char)[0]) / float(
                        ratio_text.split(split_char)[1]
                    )
                except:
                    pass
        elif "：" in ratio_text:
            split_char = "："
            if ratio_text.count(split_char) == 1:
                try:
                    ratio = float(ratio_text.split(split_char)[0]) / float(
                        ratio_text.split(split_char)[1]
                    )
                except:
                    pass
        elif "x" in ratio_text:
            split_char = "x"
            if ratio_text.count(split_char) == 1:
                try:
                    ratio = float(ratio_text.split(split_char)[0]) / float(
                        ratio_text.split(split_char)[1]
                    )
                except:
                    pass

        return ratio

    def _generate_image(
        self,
        prompt: str,
        image: Optional[Union[str, bytes, Image.Image]] = None,
        image_gen_highres: int = 672,
        **kwargs,
    ) -> Image.Image:
        """
        Generate an image or edit an existing one based on the input prompt.

        Args:
            prompt (str): User input text.
            image (Optional[Union[str, bytes, Image.Image]]): Input image (for editing). Defaults to None.
            **kwargs: Additional parameters for image generation.

        Returns:
            Image.Image: Generated or edited image.
        """
        msg_request_id = kwargs.get("msg_request_id", None)
        is_t2i = (image is None) or (isinstance(image, list) and len(image) == 0)

        if kwargs is not None and "history" in kwargs:
            del kwargs["history"]

        request_id = self.moe.create_request_id()
        state = metrics_image.create_state(stream_mode=False, request_id=request_id)
        user_aspect_ratio = None
        user_prompt = prompt
        text_token_count = self.utils.compute_text_input_tokens(user_prompt, **kwargs)

        # Prompt rewriting for better image generation quality
        logger.info(f"In _generate_image, before input rewriten {prompt}")

        if is_t2i:
            t1 = time.time()
            if self._is_image_gen_intent(user_prompt):
                prompt = self._rewrite_for_image_gen(user_prompt)
            else:
                prompt = DEFAUL_PROMPT_FOR_NO_INTENT
            logger.info(f"Rewrite_for_image_gen, cost {time.time() - t1}s")
        else:
            t1 = time.time()

            if kwargs.get('is_segmentation', False):
                user_prompt = f"Given the following instructions: {user_prompt}; please perform referring segmentation on this image."
            prompt = self._rewrite_for_image_edit(user_prompt)
            logger.info(f"Rewrite_for_image_edit, cost {time.time() - t1}s")

        logger.info("In _generate_image, after input rewriten")
        # Extract user-specified aspect ratio from prompt
        user_aspect_ratio = self._extract_aspect_ratio_for_image_gen(user_prompt)

        inputs, image_token_count, video_token_count, audio_token_count = self.utils.build_img_prompt(prompt=prompt, image=image, **kwargs)
        kwargs["compute_input_tokens_flag"] = False
        image_gen_llm_hidden_states = None

        os.environ["IMAGE_GEN_MODE"] = "T2I" if image is None else "EDIT"
        # Generate hidden states from LLM for diffusion model
        image_gen_llm_hidden_states, usage = self.moe.generate(
            requests=inputs,
            with_hidden_status=True,
            max_new_tokens=1,
            return_hidden_states=True,
        )
        usage = Usage.update_usage_by_processor(usage, text_token_count, image_token_count, video_token_count, audio_token_count)
        image_gen_negative_llm_hidden_states = None
        inputs_img_gen_negative = None
        if is_t2i:
            negative_prompt = "mutations, deformities, tilted heads, bad fingers, bad eyes, extra limbs, excess arms, deformed limbs, deformed legs, ugly, watermarks, text, NSFW"
            inputs, _, _, _ = self.utils.build_img_prompt(
                prompt=negative_prompt, image=image, **kwargs
            )
            os.environ["IMAGE_GEN_MODE"] = "T2I" if image is None else "EDIT"
            # Generate negative prompt hidden states for classifier-free guidance
            image_gen_negative_llm_hidden_states, usage_neg = self.moe.generate(
                requests=inputs,
                with_hidden_status=True,
                max_new_tokens=1,
                return_hidden_states=True,
            )
            image_gen_negative_llm_hidden_states = (
                image_gen_negative_llm_hidden_states.unsqueeze(0)
            )
            inputs_img_gen_negative = self.utils.build_img_gen_prompt(
                prompt=negative_prompt, image=image, image_gen_highres=image_gen_highres
            )

        os.environ["IMAGE_GEN_MODE"] = ""
        # device = self.device_map["image"]
        inputs = self.utils.build_img_gen_prompt(
            prompt=prompt,
            image=image,
            image_gen_highres=image_gen_highres,
            image_gen_aspect_ratio=user_aspect_ratio,
        )
        if inputs_img_gen_negative is not None:
            inputs["image_gen_negative_input_ids"] = inputs_img_gen_negative[
                "input_ids"
            ]
            inputs["image_gen_negative_attention_mask"] = inputs_img_gen_negative[
                "attention_mask"
            ]

        logger.info("In generate_image, begin diffusion")
        inputs['input_ids'] = inputs['input_ids'].to(self.device_map['image'])
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device_map['image'])
        inputs['image_gen_height'] = inputs['image_gen_height'].to(self.device_map['image'])
        inputs['image_gen_width'] = inputs['image_gen_width'].to(self.device_map['image'])
        if "image_gen_negative_input_ids" in inputs:
            inputs['image_gen_negative_input_ids'] = inputs['image_gen_negative_input_ids'].to(self.device_map['image'])
        if "image_gen_negative_attention_mask" in inputs:
            inputs['image_gen_negative_attention_mask'] = inputs['image_gen_negative_attention_mask'].to(self.device_map['image'])
        if type(image_gen_llm_hidden_states) is tuple:
            image_gen_llm_hidden_states = image_gen_llm_hidden_states[0]
        image_gen_llm_hidden_states = image_gen_llm_hidden_states.to(self.device_map['image'])
        if image_gen_negative_llm_hidden_states is not None:
            image_gen_negative_llm_hidden_states = image_gen_negative_llm_hidden_states.to(self.device_map['image'])
        image = self.img.model_diffusion.generate(
            **inputs,
            image_gen_llm_hidden_states=image_gen_llm_hidden_states.unsqueeze(0),
            image_gen_negative_llm_hidden_states=image_gen_negative_llm_hidden_states,
            image_gen=True,
            image_gen_cfg=5.5 if image is None else 5.0,
        )
        if is_t2i:
            image = auto_balance_saturation_exposure(image)

        # Update usage statistics
        usage = Usage.update_image_usage_by_length(
            usage=usage, image_gen_highres=image_gen_highres
        )

        # Record metrics
        state.input_token_length = usage["prompt_tokens"]
        state.output_token_length = usage["generated_tokens"]
        state.finish("success", msg_request_id=msg_request_id)

        return image, usage

    def _generate_audio(
        self,
        prompt: str,
        audio: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        video: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        history: list = [],
        **kwargs,
    ) -> Union[bytes, Generator[bytes, None, None]]:
        """
        Generate audio (text-to-speech or speech-to-speech).

        Args:
            prompt (str): User input text.
            audio (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Audio data (e.g., file path or binary or list).
            video (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Video data (e.g., file path or binary or list).
            image (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Image data (file path, binary, or PIL Image or list).
            history (list, optional): Conversation history. Defaults to empty list.
            **kwargs: Additional parameters for the model.

        Returns:
            Union[bytes, Generator[bytes, None, None]]: Generated audio data or a stream generator.
        """
        msg_request_id = kwargs.get("msg_request_id", None)
        request_id = self.moe.create_request_id()
        state = metrics_speech.create_state(stream_mode=False, request_id=request_id)
        text_token_count = self.utils.compute_text_input_tokens(prompt, **kwargs)
        # Build prompt and generate text response
        inputs, image_token_count, video_token_count, audio_token_count = self.utils.build_prompt(
            prompt=prompt,
            audio=audio,
            video=video,
            image=image,
            history=history,
            **kwargs,
        )
        gen_text, usage = self.moe.generate(inputs, **kwargs)
        usage = Usage.update_usage_by_processor(usage, text_token_count, image_token_count, video_token_count, audio_token_count)
        audio, duration = self.talker.generate(text=gen_text, speaker=self.speaker, request_id=msg_request_id)

        # Update audio usage statistics
        usage = Usage.update_audio_usage_by_duration(usage=usage, duration=duration)

        # Record metrics
        state.input_token_length = usage["prompt_tokens"]
        state.output_token_length = usage["generated_tokens"]
        state.finish("success", msg_request_id=msg_request_id)

        return audio, gen_text, usage

    def _generate_tts(self, text: str, **kwargs) -> Union[bytes, Generator[bytes, None, None]]:
        # Generate TTS audio from text
        msg_request_id = kwargs.get("msg_request_id", None)
        request_id = self.moe.create_request_id()
        prompt_tokens = len(text)
        usage = Usage.create_usage_default(prompt_tokens=prompt_tokens)
        state = metrics_tts.create_state(stream_mode=False, request_id=request_id)
        audio, duration = self.talker.generate(text=text, speaker=self.speaker, request_id=msg_request_id)
        if duration == 0:
            duration = audio.shape[-1]/16000
        usage = Usage.update_audio_usage_by_duration(usage=usage, duration=duration)
        state.input_token_length = usage["prompt_tokens"]
        state.output_token_length = usage["generated_tokens"]
        state.finish("success", msg_request_id=msg_request_id)
        usage['finish_reason'] = 'stop'
        return audio, usage

    def generate(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        video: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        history: list = [],
        output_type: str = "text",
        **kwargs,
    ) -> Union[str, Image.Image, bytes, Generator]:
        """
        Generate content based on the specified output type.

        Args:
            text (Optional[str]): User input text.
            audio (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Audio data (e.g., file path or binary or list).
            video (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Video data (e.g., file path or binary or list).
            image (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Image data (file path, binary, or PIL Image or list).
            history (list, optional): Conversation history. Defaults to empty list.
            output_type (str, optional): Output type ("text", "speech", "image", "tts"). Defaults to "text".
            **kwargs: Additional parameters for the model.

        Returns:
            Union[str, Image.Image, bytes, Generator]: Generated content (text, image, or audio).

        Raises:
            ValueError: If `output_type` is not supported.
        """
        if output_type == "text":
            return self._generate_text(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )

        elif output_type == "speech":
            return self._generate_audio(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )

        elif output_type == "image":
            return self._generate_image(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )

        elif output_type == "tts":
            return self._generate_tts(text=text, **kwargs)

        else:
            raise Exception("not support output_type")

    def generate_stream(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        video: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        image: Optional[Union[str, bytes, List[Union[str, bytes]]]] = None,
        history: list = [],
        output_type: str = "text",
        **kwargs,
    ) -> Generator[Tuple[Union[bytes, str], str], None, None]:
        """
        Stream generated content (text or speech).

        Args:
            text (Optional[str]): User input text.
            audio (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Audio data (e.g., file path or binary or list).
            video (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Video data (e.g., file path or binary or list).
            image (Optional[Union[str, bytes, List[Union[str, bytes]]]]): Image data (file path, binary, or PIL Image or list).
            history (list, optional): Conversation history. Defaults to empty list.
            output_type (str, optional): Output type ("text", "speech", "TTS"). Defaults to "text".
            **kwargs: Additional parameters for the model.

        Yields:
            Tuple[Union[bytes, str], str]: Generated content (text or audio) and request ID.

        Raises:
            ValueError: If `output_type` is not supported for streaming.
        """
        msg_request_id = kwargs.get("msg_request_id", None)
        text_token_count = self.utils.compute_text_input_tokens(text, **kwargs)
        if output_type == "text":
            request_id = self.moe.create_request_id()
            self.info_cache.set(f"{msg_request_id}", request_id, self.info_cache_default_time)
            state = metrics_text.create_state(stream_mode=True, request_id=request_id)
            inputs, image_token_count, video_token_count, audio_token_count = self.utils.build_prompt(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )
            for text in self.moe.generate_stream(
                requests=inputs, request_id=request_id, **kwargs
            ):
                state.record_first_token()
                usage = self.moe.usage.get_stream_usage_by_request_id(
                    request_id=request_id
                )
                usage = Usage.update_usage_by_processor(usage, text_token_count, image_token_count, video_token_count, audio_token_count)
                state.record_input_tokens(usage["prompt_tokens"])
                state.output_token_length = usage["generated_tokens"]
                yield text, request_id, usage
            state.finish("success", msg_request_id=msg_request_id)

        elif output_type == "speech":
            request_id = self.moe.create_request_id()
            self.info_cache.set(f"{msg_request_id}", request_id, self.info_cache_default_time)
            state_text = metrics_speech.create_state(stream_mode=True, request_id=request_id)
            state_text_audio = metrics_speech_text_audio.create_state(stream_mode=True, request_id=request_id)

            inputs, image_token_count, video_token_count, audio_token_count = self.utils.build_prompt(
                prompt=text,
                audio=audio,
                video=video,
                image=image,
                history=history,
                **kwargs,
            )
            text_generator = self.moe.generate_stream(
                requests=inputs, request_id=request_id, **kwargs
            )

            def _produce_text_to_queue(text_generator, talker_input_queue, result_queue):
                def producer():
                    try:
                        for chunk in text_generator:
                            state_text.record_first_token()

                            talker_input_queue.put(chunk)
                            usage = self.moe.usage.get_stream_usage_by_request_id(
                                    request_id=request_id
                                )
                            usage = Usage.update_usage_by_processor(usage, text_token_count, image_token_count, video_token_count, audio_token_count)
                            state_text.record_input_tokens(usage["prompt_tokens"])
                            state_text.output_token_length = usage["generated_tokens"]
                            result_queue.put(("text_data", (chunk, usage)))
                        talker_input_queue.put(None, None)  # End signal
                        state_text.finish("success", msg_request_id=msg_request_id)
                    except Exception as e:
                        logger.error(f"Error in text producer: {e}")
                        talker_input_queue.put(None, None)
                
                producer_thread = threading.Thread(target=producer)
                producer_thread.daemon = True
                producer_thread.start()
                return producer_thread

            def warpper(text_queue):
                while True:
                    text = text_queue.get()
                    if text is None:
                        break
                    yield text

            def thread_talker_task(talker, moe, text_input_queue, speaker, result_queue):
                def _thread_talker_task(talker, moe, text_input_queue, speaker, result_queue):
                    try:
                        duration = 0
                        for tts_speech, sentence, meta_info in talker.generate_stream(
                            text=warpper(text_input_queue), speaker=speaker, request_id=msg_request_id
                        ):
                            state_text_audio.record_first_token()
                            # update audio usage
                            usage = moe.usage.get_stream_usage_by_request_id(
                                request_id=request_id
                            )
                            usage = Usage.update_usage_by_processor(usage, text_token_count, image_token_count, video_token_count, audio_token_count)
                            duration += meta_info["duration"]

                            usage = Usage.update_audio_usage_by_duration(usage, duration)

                            state_text_audio.record_input_tokens(usage["prompt_tokens"])
                            state_text_audio.output_token_length = usage["generated_tokens"]
                            result_queue.put(("text_audio_data", (tts_speech, sentence, meta_info, request_id, usage)))
                    finally:
                        result_queue.put((None, None))
                        state_text_audio.finish("success", msg_request_id=msg_request_id)

                talker_thread = threading.Thread(target=_thread_talker_task, args=(talker, moe, text_input_queue, speaker, result_queue))
                talker_thread.daemon = True
                talker_thread.start()
                return talker_thread
            
            talker_input_queue = self.queue_manager.Queue()
            result_queue = queue.Queue()

            producer_thread = _produce_text_to_queue(text_generator, talker_input_queue, result_queue)
            talker_thread = thread_talker_task(self.talker, self.moe, talker_input_queue, self.speaker, result_queue)
                
            while True:
                data_type, data_content = result_queue.get()
                if data_type is None:
                    break
                yield data_type, data_content

            # Ensure both threads have completed processing
            producer_thread.join()
            talker_thread.join()

        elif output_type == "tts":
            request_id = self.moe.create_request_id()
            self.info_cache.set(f"{msg_request_id}", request_id, self.info_cache_default_time)
            state = metrics_tts.create_state(stream_mode=True, request_id=request_id)
            prompt_tokens = len(text)

            duration = 0
            for tts_speech, sentence, meta_info in self.talker.generate_stream(
                text=text, speaker=self.speaker, request_id=msg_request_id,
            ):
                usage = Usage.create_usage_default(prompt_tokens=prompt_tokens)
                state.record_first_token()
                if meta_info["duration"] == 0:
                    duration += tts_speech.shape[-1]/16000
                else:
                    duration += meta_info["duration"]
                
                usage = Usage.update_audio_usage_by_duration(
                    usage=usage, duration=duration
                )
                usage = Usage.update_usage_by_processor(usage, text_token_count=text_token_count)
                state.record_input_tokens(usage["prompt_tokens"])
                state.output_token_length = usage["generated_tokens"]
                yield tts_speech, sentence, meta_info, request_id, usage
            state.finish("success", msg_request_id=msg_request_id)
        else:
            raise Exception("not support output_type")

    def generate_interrupt(self, msg_request_id: str) -> None:
        """
        Interrupt a specific request.

        Args:
            request_id (str): ID of the request to interrupt.

        Raises:
            ValueError: If `request_id` is empty.
        """

        vllm_infer_request_id = self.info_cache.get(f"{msg_request_id}")
        if vllm_infer_request_id:
            self.moe.generate_interrupt(vllm_infer_request_id)
            logger.info(f"Generate_interrupt success, msg_request_id: {msg_request_id}, vllm infer request_id: {vllm_infer_request_id}")
        else:
            logger.info(f"Generate_interrupt failed, msg_request_id: {msg_request_id} is invalid")

        self.talker.generate_interrupt(msg_request_id)
