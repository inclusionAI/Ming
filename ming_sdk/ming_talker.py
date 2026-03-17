"""
MingTalker: Text-to-Speech Module

This module provides a multiprocessing-based TTS (Text-to-Speech) system
using the BailingTalker model. It supports both streaming and non-streaming
audio generation with multi-GPU load balancing.

Key Features:
    - Process pool for parallel TTS generation across multiple GPUs
    - Streaming audio output for real-time speech synthesis
    - Chinese text detection for proper tokenization
    - Request cancellation support

Architecture:
    - Main process: Handles request queuing and result collection
    - Worker processes: Each hosts a Talker model on a specific GPU
    - Inter-process communication via multiprocessing.Queue

Usage:
    >>> talker = MingTalker(
    ...     model_path="/path/to/model",
    ...     device_list=["cuda:0", "cuda:1"]
    ... )
    >>> audio, duration = talker.generate("Hello, world!")
"""

import re
import os
import time
import torch
import logging
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Any, List, Tuple, Generator, Union
import threading

logger = logging.getLogger()
import queue
import uuid


def contains_chinese(text: str) -> bool:
    """
    Check if the input text contains Chinese characters.

    Args:
        text (str): Input text to check.

    Returns:
        bool: True if Chinese characters are present, False otherwise.
    """
    return bool(re.search(r"[\u4e00-\u9fff]", text))

class ProcessTalkerInstance:
    """
    Singleton Talker instance for each worker process.

    Each worker process maintains a single Talker model instance
    to avoid repeated model loading overhead.
    """
    def __init__(self, model_path: str, device: str):
        from modeling_bailing_talker import BailingTalker2
        from AudioVAE.modeling_audio_vae import AudioVAE

        logger.info(f"Loading talker model on {device}...")
        with torch.cuda.device(device):
            dtype = torch.bfloat16
            self.talker = BailingTalker2.from_pretrained(
                f"{model_path}/talker").to(
                dtype=dtype, 
                device=device
            )
            self.talker_vae = AudioVAE.from_pretrained(
                f"{model_path}/talker/vae").to(
                dtype=dtype, 
                device=device
            )
            self.talker.eval()
            self.talker_vae.eval()
        
        self.device = device
        logger.info(f"Talker model loaded successfully on {device}")


# Global Talker instance per process (singleton pattern)
_process_talker_instance = None

def get_process_talker(model_path: str, device: str) -> ProcessTalkerInstance:
    """
    Get or create the Talker instance for the current process (singleton).

    Args:
        model_path (str): Path to the model directory.
        device (str): GPU device to load the model on.

    Returns:
        ProcessTalkerInstance: The singleton Talker instance for this process.
    """
    global _process_talker_instance
    if _process_talker_instance is None:
        _process_talker_instance = ProcessTalkerInstance(model_path, device)

    logger.info(f'get_process_talker: {device}, actual model device: {_process_talker_instance.device}')
    return _process_talker_instance


def process_generate(args) -> torch.Tensor:
    """
    Worker function for non-streaming audio generation.

    This function runs in a worker process and performs the actual
    TTS generation using the process-local Talker instance.

    Args:
        args (tuple): Contains (text, speaker, model_path, device, cancel_flag).

    Returns:
        Tuple[torch.Tensor, float]: Generated audio waveform and duration in seconds.
    """
    text, speaker, model_path, device, cancel_flag = args
    try:
        talker_instance = get_process_talker(model_path, device)
        actual_talker_device = talker_instance.device
        talker = talker_instance.talker
        talker_vae = talker_instance.talker_vae

        is_chinese = contains_chinese(text)
        if not is_chinese:
            text = text.split()

        all_wavs = []
        duration_ = 0
        last_time = time.perf_counter()
        for (
            tts_speech,
            text_list,
            word_postion,
            duration,
        ) in talker.omni_audio_generation(
            tts_text=text,
            voice_name=speaker,
            audio_detokenizer=talker_vae,
            stream=False,
        ):
            if cancel_flag.is_set():
                break
            all_wavs.append(tts_speech)
            this_time = time.perf_counter()
            # logging.info(f"chunk time cost: {this_time - last_time:.3f}s")
            last_time = this_time
            if duration is None:
                duration = 0
            duration_ += duration
        
        waveform = torch.cat(all_wavs, dim=-1)

        # logging.info("finish process_generate func.")
        return waveform, duration_
        
    except Exception as e:
        logger.error(f"Error in process_generate on {actual_talker_device}: {e}")
        raise


def process_stream_generate(args) -> Generator[Tuple[torch.Tensor, List[str], dict], None, None]:
    """
    Worker function for streaming audio generation.

    This function runs in a worker process and generates audio chunks
    incrementally, yielding results through a multiprocessing Queue.

    Args:
        args (tuple): Contains (text_queue, speaker, model_path, device,
                      result_queue, cancel_flag).

    Yields:
        Tuple[torch.Tensor, List[str], dict]: Audio chunk, text segments, and metadata.
    """
    text_queue, speaker, model_path, device, result_queue, cancel_flag = args
    actual_talker_device = None
    try:
        talker_instance = get_process_talker(model_path, device)
        actual_talker_device = talker_instance.device
        talker = talker_instance.talker
        talker_vae = talker_instance.talker_vae

        def text_wrapper(input_queue):
            while True:
                try:
                    if cancel_flag.is_set():
                        break
                    text = input_queue.get(timeout=300)  # 5分钟超时
                    if text is None:  # 结束信号
                        break
                    yield text
                except queue.Empty:
                    logger.warning("Text queue timeout, exiting...")
                    break

        logger.info(f"Starting stream generation on {device}")
        last_time = time.perf_counter()
        for (
            tts_speech,
            text_list,
            word_postion,
            duration,
        ) in talker.omni_audio_generation(
            tts_text=text_wrapper(text_queue),
            audio_detokenizer=talker_vae,
            voice_name=speaker,
            stream=True,
        ):
            if cancel_flag.is_set():
                break
            this_time = time.perf_counter()
            # logging.info(f"chunk time cost: {this_time - last_time:.3f}s")
            last_time = this_time
            
            star_index, end_index, duration_ = 0, 0, 0
            if word_postion and len(word_postion) > 0 and word_postion[0] is not None:
                star_index = word_postion[0]
            if word_postion and len(word_postion) > 1 and word_postion[1] is not None:
                end_index = word_postion[1]
            if duration is not None and duration > 0:
                duration_ = duration
                
            meta_info = {
                "star_index": star_index,
                "end_index": end_index,
                "duration": duration_,
            }
            # Send results to the main process via queue
            result_queue.put((tts_speech, text_list, meta_info))
    except Exception as e:
        logger.error(f"Error in process_stream_generate on {actual_talker_device}: {e}")
        raise
    finally:
        # Signal end of stream to the main process
        result_queue.put(None)
        logger.info(f"Stream generation finished on {actual_talker_device}")
        logging.info(f"Stream generation finished on {actual_talker_device}")


class MingTalker(object):
    """
    Main TTS interface with multi-GPU process pool support.

    This class provides a high-level interface for text-to-speech generation
    with automatic load balancing across multiple GPUs using a process pool.

    Attributes:
        model_path (str): Path to the model directory.
        device_list (list): List of GPU devices for TTS workers.
        process_pool (ProcessPoolExecutor): Pool of worker processes.
        manager (multiprocessing.Manager): Manager for inter-process communication.
    """
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        device_list: list = ["cuda:0"],
    ) -> None:
        """
        Initialize the MingTalker instance.

        Args:
            model_path (str): Path to the model directory containing TTS components.
            tensor_parallel_size (int, optional): Number of GPU devices for tensor parallelism. Defaults to 1.
            device_list (list, optional): List of GPU devices to use. Defaults to ["cuda:0"].
        """
        super().__init__()
        self.model_path = model_path
        self.device_list = device_list
        multiprocessing.set_start_method('spawn', force=True)
        self.manager = multiprocessing.Manager()

        # Create process pool, one worker per GPU device
        self.process_pool = ProcessPoolExecutor(
            max_workers=len(device_list)
        )

        # Preload models on all devices to ensure readiness
        logging.info("Preloading models on all devices...")
        futures = []
        for device in device_list:
            future = self.process_pool.submit(process_generate,
                ("test", "DB30", model_path, device, self.manager.Event()))
            futures.append(future)

        # Wait for preload to complete (ignore test results)
        for future in as_completed(futures):
            try:
                future.result(timeout=120)  # 2-minute timeout for model loading
            except Exception as e:
                logger.warning(f"Preload may have failed (expected for test text): {e}")

        self.lock = threading.Lock()
        self.task_count = 0
        self.cancel_flag_dict = dict()
        self.futures_dict = dict()
        self.future_lock = threading.Lock()

        logging.info(f"MingTalker initialized successfully with {len(device_list)} devices")

    def logging_tasks(self, request_id, future):
        time.sleep(0.01)
        running_keys = []
        pendding_keys = []
        done_count = 0
        with self.future_lock:
            self.futures_dict[request_id] = future
            all_keys = list(self.futures_dict.keys())
            done_keys = []
            for key in all_keys:
                future = self.futures_dict[key]
                if future.done():
                    done_keys.append(key)
                elif future.running():
                    running_keys.append(key)
                else:
                    pendding_keys.append(key)

            for key in done_keys:
                self.futures_dict.pop(key)

        logging.info(f"Running Count: {len(running_keys)}, Pending Count: {len(pendding_keys)}")
        if len(pendding_keys) > 0:
            logging.info(f"Pending Keys: {pendding_keys}, Running Keys: {running_keys}")


    def generate(self, text: str, speaker: str = "DB30", request_id: str = str(uuid.uuid4()),) -> torch.Tensor:
        """
        Generate audio from text using the TTS model.

        Args:
            text (str): Input text to convert to speech.
            speaker (str, optional): Speaker identifier. Defaults to "DB30".
            request_id (str, optional): Unique request identifier. Defaults to new UUID.

        Returns:
            Tuple[torch.Tensor, float]: Generated audio waveform and duration in seconds.
        """
        # Submit task to process pool; pool scheduler assigns to available worker
        try:
            cancel_flag = self.manager.Event()
            with self.lock:
                self.task_count += 1
                logging.info(f"Running Task Count: {self.task_count}, with generate.")
                self.cancel_flag_dict[request_id] = cancel_flag

            future = self.process_pool.submit(process_generate,
                (text, speaker, self.model_path, None, cancel_flag))
            self.logging_tasks(request_id=request_id, future=future)
            return future.result()

        finally:
            with self.lock:
                self.task_count -= 1
                if request_id in self.cancel_flag_dict:
                    self.cancel_flag_dict[request_id].set()
                    self.cancel_flag_dict.pop(request_id)


    def generate_stream(
        self, text: Union[str, Generator[str, None, None]], speaker: str = "DB30", request_id: str = str(uuid.uuid4()),
    ) -> Generator[Tuple[torch.Tensor, List[str], dict], None, None]:
        """
        Stream audio generation from text incrementally.

        Args:
            text: Input text, can be string or text generator for streaming input.
            speaker (str, optional): Speaker identifier. Defaults to "DB30".
            request_id (str, optional): Unique request identifier. Defaults to new UUID.

        Yields:
            Tuple[torch.Tensor, List[str], dict]: Audio segment, text segments, and metadata.
        """
        try:
            cancel_flag = self.manager.Event()
            with self.lock:
                self.task_count += 1
                logging.info(f"Request_id: {request_id}, Running Task Count: {self.task_count}, with generate_stream.")
                self.cancel_flag_dict[request_id] = cancel_flag

            # Create inter-process communication queues
            text_queue = self.manager.Queue()
            result_queue = self.manager.Queue()

            # Start streaming generation task
            future = self.process_pool.submit(process_stream_generate,
                (text_queue, speaker, self.model_path, None, result_queue, cancel_flag))

            # Feed text into the queue
            self._produce_text_to_queue(text, text_queue)

            self.logging_tasks(request_id=request_id, future=future)

            # Consume streaming results
            for result in self._consume_stream_results(result_queue, future):
                yield result
        finally:
            with self.lock:
                self.task_count -= 1
                if request_id in self.cancel_flag_dict:
                    self.cancel_flag_dict[request_id].set()
                    self.cancel_flag_dict.pop(request_id)

    def _produce_text_to_queue(self, text: Union[str, Generator[str, None, None]],
                             text_queue: multiprocessing.Queue):
        """
        Produce text data into the queue for streaming generation.

        This method runs a background thread to feed text chunks into
        the worker process's input queue.

        Args:
            text: Input text (string or generator).
            text_queue: Multiprocessing queue for text chunks.
        """
        def producer():
            try:
                if isinstance(text, str):
                    for text_str in text:
                        text_queue.put(text_str)
                else:
                    for chunk in text:
                        text_queue.put(chunk)
                text_queue.put(None)  # End signal
            except Exception as e:
                logger.error(f"Error in text producer: {e}")
                text_queue.put(None)

        # Run producer in background thread
        import threading
        producer_thread = threading.Thread(target=producer)
        producer_thread.daemon = True
        producer_thread.start()

    def _consume_stream_results(self, result_queue: multiprocessing.Queue, future):
        """
        Consume streaming results from the worker process queue.

        Args:
            result_queue: Multiprocessing queue for audio results.
            future: Future object for the worker process task.

        Yields:
            Tuple[torch.Tensor, List[str], dict]: Audio chunk, text segments, and metadata.
        """
        try:
            while True:
                try:
                    result = result_queue.get(timeout=300)  # 5-minute timeout
                    if result is None:  # End signal
                        break

                    tts_speech, text_list, meta_info = result

                    yield tts_speech, text_list, meta_info

                except queue.Empty:
                    logger.warning("Result queue timeout")
                    break

        finally:
            # Ensure the task has completed
            try:
                future.result()
            except:
                logger.warning("Stream generation future may have timed out")

    def generate_interrupt(self, request_id: str) -> None:
        """
        Interrupt an ongoing TTS generation request.

        Args:
            request_id (str): ID of the request to cancel.
        """
        with self.lock:
            if request_id in self.cancel_flag_dict:
                cancel_flag = self.cancel_flag_dict[request_id]
                cancel_flag.set()

    def __del__(self):
        """Clean up resources and shutdown process pool."""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)