"""
MingMOEAsync: Asynchronous Mixture of Experts Language Model Module

This module provides an asynchronous wrapper around vLLM for text generation
using the Ming Mixture of Experts model. It is designed for high-throughput
concurrent request scenarios.

Key Features:
    - Asynchronous inference using AsyncLLMEngine
    - Dedicated event loop running in a separate thread
    - Non-blocking streaming generation
    - Request interruption support

Usage:
    >>> moe_async = MingMOEAsync(
    ...     model_path="/path/to/model",
    ...     tensor_parallel_size=2,
    ...     gpu_memory_utilization=0.6
    ... )
    >>> output, usage = await moe_async.generate([{"prompt": "Hello!"}])
"""

import sys
import time
import uuid
import asyncio
import logging
import threading

from vllm import AsyncLLMEngine
from vllm import SamplingParams
from vllm.inputs import TextPrompt as LLMInputs
from vllm.engine.arg_utils import AsyncEngineArgs
from typing import Any, List, Optional, AsyncGenerator

from ming_sdk.usage import Usage
from ming_sdk.ming_utils import StreamGenerator, load_json_or_str


logger = logging.getLogger()


class MingMOEAsync(object):
    """
    Asynchronous Mixture of Experts LLM wrapper using vLLM AsyncLLMEngine backend.

    This class provides a high-level asynchronous interface for text generation
    using the Ming MOE model. It runs a dedicated asyncio event loop in a
    separate thread to enable non-blocking concurrent inference.

    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate per request.
        loop (asyncio.AbstractEventLoop): Dedicated event loop for async operations.
        thread (threading.Thread): Thread running the event loop.
        engine (AsyncLLMEngine): The vLLM async engine instance.
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
        Initialize the MingMOEAsync instance.

        Args:
            model_path (str): Path to the LLM model directory or Hugging Face repository.
            tensor_parallel_size (int, optional): Number of GPU devices for tensor parallelism. Defaults to 1.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use (0.0-1.0). Defaults to 0.5.
            limit_mm_per_prompt (dict, optional): Max multimedia items per prompt. Defaults to {"image": 10, "video": 2}.
            quantization (str | None, optional): Quantization method ("fp8" or None for bf16). Defaults to None.
        """
        logger.info("using MingMOEAsync")
        if quantization == "fp8":
            logger.info("using fp8")
        else:
            quantization = None
            logger.info("using bf16")
        engine_args = AsyncEngineArgs(
            model=model_path,
            trust_remote_code=True,
            enforce_eager=False,
            max_num_seqs=32,
            disable_custom_all_reduce=False,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt=limit_mm_per_prompt,
            quantization=quantization,
            max_seq_len_to_capture=32768,
            disable_mm_preprocessor_cache=True,
        )

        self.loop, self.thread = self.new_and_run_event_loop()
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.usage = Usage()

    def new_and_run_event_loop(self) -> tuple:
        """
        Create and start a new asyncio event loop in a dedicated thread.

        This method sets up an isolated event loop for async LLM inference,
        preventing conflicts with the main thread's event loop.

        Returns:
            tuple: A tuple containing (event_loop, thread) where thread is
                   running the event loop in daemon mode.
        """
        new_loop = asyncio.new_event_loop()
        thread_name = "moe_thread"
        thread = threading.Thread(
            target=new_loop.run_forever, daemon=True, name=thread_name
        )
        thread.start()

        # Wait for the event loop to start running
        while not new_loop.is_running():
            time.sleep(0.1)

        return new_loop, thread

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
        request_id = self.create_request_id()

        async def _inner():
            final = None
            async for output in self.engine.generate(
                prompt=requests[0],
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final = output
            return final

        future = asyncio.run_coroutine_threadsafe(_inner(), self.loop)
        output = None
        usage = self.usage.create_usage_default(prompt_tokens=0)
        try:
            output = future.result()
            if isinstance(output, dict) and "error" in output:
                logger.error(f"[Error] {output['type']}: {output['error']}")
        except Exception as e:
            logger.error(f"[Unexpected Error] {e}")
            sys.exit(1)
        if output is None:
            return None, usage
        if with_hidden_status:
            return output.prefill_hidden_states, self.usage.create_usage(output)
        return output.outputs[0].text, self.usage.create_usage(output)

    async def _generate_stream_async(
        self, requests: List[LLMInputs], request_id: int = 0, **kwargs
    ) -> AsyncGenerator[str, None]:
        sampling_params = self.build_sampling_params(**kwargs)
        history_sentence_index = 0
        async for request_outputs in self.engine.generate(
            prompt=requests[0],
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            self.usage.create_usage_by_requests_id(
                request_outputs, request_id=request_id
            )
            new_sentence = request_outputs.outputs[0].text
            sentence = new_sentence[history_sentence_index:]
            history_sentence_index = len(new_sentence)
            yield sentence

    def generate_stream(self, requests: List[LLMInputs], request_id: int = 0, **kwargs):
        async def _inner_stream():
            try:
                async for output in self._generate_stream_async(
                    requests, request_id, **kwargs
                ):
                    yield output
            except Exception as e:
                logger.error(f"[Unexpected Error] {e}")
                sys.exit(1)

        return StreamGenerator(self.loop, _inner_stream())

    def generate_interrupt(self, request_id: str) -> None:
        """
        Interrupt an ongoing request.

        Args:
            request_id (str): Unique identifier of the request to abort.
        """

        async def _inner():
            await self.engine.abort(str(request_id))

        asyncio.run_coroutine_threadsafe(_inner(), self.loop)
