"""
MingMOE: Mixture of Experts Language Model Module

This module provides a synchronous wrapper around vLLM for text generation
using the Ming Mixture of Experts model. It supports both streaming and
non-streaming generation modes.

Key Features:
    - Tensor parallelism for multi-GPU inference
    - Configurable GPU memory utilization
    - Rich sampling parameter support (temperature, top_p, top_k, etc.)
    - Request interruption support

Usage:
    >>> moe = MingMOE(
    ...     model_path="/path/to/model",
    ...     tensor_parallel_size=2,
    ...     gpu_memory_utilization=0.6
    ... )
    >>> output, usage = moe.generate([{"prompt": "Hello, world!"}])
"""

import sys
import uuid
import logging

from vllm import LLM, SamplingParams
from vllm.inputs import TextPrompt as LLMInputs

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
from ming_sdk.usage import Usage
from ming_sdk.ming_utils import load_json_or_str

logger = logging.getLogger()


class MingMOE(object):
    """
    Synchronous Mixture of Experts LLM wrapper using vLLM backend.

    This class provides a high-level interface for text generation using
    the Ming MOE model with vLLM as the inference engine.

    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate per request.
        llm (LLM): The vLLM engine instance.
        usage (Usage): Token usage tracker instance.
    """

    max_new_tokens = 10240

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        limit_mm_per_prompt: dict = {"image": 10, "video": 2},
        quantization: str | None = None,
    ) -> None:
        """
        Initialize the BailigMOE instance.

        Args:
            model_path (str): Path to the LLM model directory or Hugging Face repository.
            tensor_parallel_size (int, optional): Number of GPU devices for tensor parallelism. Defaults to 1.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use (0.0-1.0). Defaults to 0.6.
            sys_prompt (str, optional): System-level prompt to prepend to user inputs. Defaults to empty.
        """
        logger.info("using MingMOE")
        if quantization == "fp8":
            logger.info("using fp8")
        else:
            quantization = None
            logger.info("using bf16")
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            enforce_eager=False,
            disable_custom_all_reduce=False,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=limit_mm_per_prompt,
            quantization=quantization,
            max_seq_len_to_capture=32768,
            disable_mm_preprocessor_cache=True,
        )
        self.usage = Usage()

    def create_request_id(self) -> str:
        """
        Generate a unique request ID for tracking.

        Returns:
            str: A UUID4 string representing a unique request identifier.
        """
        return str(uuid.uuid4())

    def build_sampling_params(self, **kwargs) -> SamplingParams:
        """
        Build sampling parameters for the LLM.

        Args:
            **kwargs: Additional parameters for sampling (e.g., max_new_tokens).

        Returns:
            SamplingParams: Configured sampling parameters.
        """
        temperature, presence_penalty, repetition_penalty, return_hidden_states = (
            0.6,
            0,
            1,
            False,
        )
        max_new_tokens = self.max_new_tokens
        top_p, top_k, frequency_penalty, seed, stop = None, None, None, None, None
        min_p, stop_token_ids, ignore_eos, logprobs, prompt_logprobs = (
            None,
            None,
            False,
            None,
            None,
        )
        for key, value in kwargs.items():
            if key == "max_new_tokens" and value is not None:
                max_new_tokens = value
            if key == "temperature" and value is not None:
                temperature = value
            if key == "presence_penalty" and value is not None:
                presence_penalty = value
            if key == "repetition_penalty" and value is not None:
                repetition_penalty = value
            if key == "return_hidden_states" and value is not None:
                return_hidden_states = value
            if key == "top_p" and value is not None and isinstance(value, float):
                top_p = value
            if key == "top_k" and value is not None and isinstance(value, int):
                top_k = value
            if (
                key == "frequency_penalty"
                and value is not None
                and isinstance(value, float)
            ):
                frequency_penalty = value
            if key == "seed" and value is not None and isinstance(value, int):
                seed = value
            if key == "stop" and value is not None:
                value = load_json_or_str(value)
                if (
                    isinstance(value, list)
                    and all(isinstance(item, str) for item in value)
                ) or isinstance(value, str):
                    stop = value
            if key == "min_p" and value is not None and isinstance(value, float):
                min_p = value
            if key == "stop_token_ids" and value is not None:
                if isinstance(value, list) and all(isinstance(v, int) for v in value):
                    stop_token_ids = value
            if key == "ignore_eos" and value is not None and isinstance(value, bool):
                ignore_eos = value
            if (
                key == "logprobs"
                and value is not None
                and isinstance(value, int)
                and value > 0
            ):
                logprobs = value
            if (
                key == "prompt_logprobs"
                and value is not None
                and isinstance(value, int)
                and value > 0
            ):
                prompt_logprobs = value

        sampling_params_kwargs = {
            "temperature": temperature,
            "max_tokens": max_new_tokens,
            "presence_penalty": presence_penalty,
            "repetition_penalty": repetition_penalty,
            "return_hidden_states": return_hidden_states,
        }
        optional_params = {
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "stop": stop,
            "min_p": min_p,
            "stop_token_ids": stop_token_ids,
            "ignore_eos": ignore_eos,
            "logprobs": logprobs,
            "prompt_logprobs": prompt_logprobs,
        }
        for param, value in optional_params.items():
            if value is not None:
                sampling_params_kwargs[param] = value

        sampling_params = SamplingParams(**sampling_params_kwargs)
        return sampling_params

    def generate(
        self, requests: List[LLMInputs], with_hidden_status: bool = False, **kwargs
    ) -> Any:
        """
        Generate text responses from the LLM.

        Args:
            requests (List[LLMInputs]): List of input prompts for generation.
            with_hidden_status (bool, optional): Whether to return hidden states from the LLM. Defaults to False.
            **kwargs: Additional parameters for sampling (e.g., max_new_tokens, temperature).

        Returns:
            Any: Generated text or hidden states, depending on `with_hidden_status`.
        """
        sampling_params = self.build_sampling_params(**kwargs)
        logger.info(f"In ming_moe generate, kwargs: {kwargs}")
        logger.info(f"In vllm generate, sampling_params: {sampling_params}")
        request_id = self.create_request_id()
        logger.info(f"In vllm generate request_id: {request_id}")
        inputs = [
            (
                request_id,
                requests[0],
                sampling_params,
            )
        ]
        req_id, prompt_text, sampling_params = inputs.pop(0)
        llm_engine = self.llm.llm_engine
        llm_engine.add_request(str(req_id), prompt_text, sampling_params)
        logger.info("start to inference llm")

        output = []
        while llm_engine.has_unfinished_requests():
            try:
                output = llm_engine.step()
                if len(output) == 0:
                    continue
            except Exception as e:
                logger.error(f"[Unexpected Error] {e}")
                sys.exit(1)
        if len(output) == 0:
            raise Exception("llm inference failed")
        usage_output = output[0]
        usage = self.usage.create_usage_by_requests_id(usage_output, request_id=req_id)

        if with_hidden_status:
            return output[0].prefill_hidden_states, usage
        return output[0].outputs[0].text, usage

    def generate_stream(
        self, requests: List[LLMInputs], request_id: int = 0, **kwargs
    ) -> Generator[str, None, None]:
        """
        Args:
            requests (List[LLMInputs]): List of input prompts for generation.
            request_id (int, optional): Unique identifier for the request. Defaults to 0.
            **kwargs: Additional parameters for sampling (e.g., max_new_tokens).

        Yields:
            str: Incremental text output as it is generated.
        """
        logger.info(f"In ming_moe generate stream, kwargs: {kwargs}")
        sampling_params = self.build_sampling_params(**kwargs)
        logger.info(f"In vllm generate_stream, sampling_params: {sampling_params}")
        logger.info(f"In vllm generate_stream request_id: {request_id}")
        inputs = [
            (
                request_id,
                requests[0],
                sampling_params,
            )
        ]
        req_id, prompt_text, sampling_params = inputs.pop(0)
        llm_engine = self.llm.llm_engine
        llm_engine.add_request(str(req_id), prompt_text, sampling_params)
        logger.info("start to inference llm")

        history_sentence_index = 0
        while llm_engine.has_unfinished_requests():
            try:
                request_outputs = llm_engine.step()
                if len(request_outputs) == 0:
                    continue
                usage_output = request_outputs[0]
                self.usage.create_usage_by_requests_id(usage_output, request_id=req_id)
                new_sentence = request_outputs[0].outputs[0].text
                sentence = new_sentence[history_sentence_index:]
                history_sentence_index = len(new_sentence)
                yield sentence
            except Exception as e:
                logger.error(f"[Unexpected Error] {e}")
                sys.exit(1)

    def generate_interrupt(self, request_id: str) -> None:
        """
        Interrupt an ongoing request.

        Args:
            request_id (str): Unique identifier of the request to abort.
        """
        llm_engine = self.llm.llm_engine
        llm_engine.abort_request(str(request_id))
