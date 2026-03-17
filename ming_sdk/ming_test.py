"""
Ming SDK Test Examples

This module demonstrates various usage patterns of the Ming SDK, including:
- Text generation (streaming and non-streaming)
- Speech synthesis (TTS)
- Speech-to-speech conversation
- Image understanding and generation
- Audio understanding (ASR)
- Video understanding

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Download model weights and configure model_path
3. Ensure sufficient GPU memory (4x A100/H20 recommended)

Author: qiaozhuo
"""

import os
import torch
import torchaudio
from ming_sdk.ming import Ming


# ============================================================================
# Text Generation Tests
# ============================================================================

def test_text_generate():
    """
    Test non-streaming text generation.
    
    Use case: Batch text processing, scenarios without real-time feedback.
    
    Returns:
        tuple: (generated_text, usage_statistics)
    """
    text, usage = ming.generate(text="Introduce Hangzhou")
    print(f"Generated text: {text}")
    print(f"Usage: {usage}")
    assert text is not None, "Text generation failed"


def test_text_generate_stream():
    """
    Test streaming text generation.
    
    Use case: Real-time dialogue systems, progressive content display.
    
    Yields:
        tuple: (text_chunk, request_id, usage) for each streaming chunk
    """
    all_text = ""
    request_id = ""
    for text, request_id, usage in ming.generate_stream(
        text="Introduce Hangzhou", 
        max_new_tokens=128
    ):
        all_text += text
        print(text, end="", flush=True)  # Real-time output
    
    print(f"\nRequest ID: {request_id}")
    print(f"Full text: {all_text}")
    print(f"Usage: {usage}")
    assert all_text, "Streaming text generation failed"


# ============================================================================
# Speech Generation Tests
# ============================================================================

def test_audio_generate():
    """
    Test non-streaming speech generation (Speech-to-Speech).
    
    Flow:
    1. Generate text response from input text
    2. Convert response text to speech
    
    Use case: Voice assistants, conversational AI systems.
    
    Returns:
        tuple: (waveform, generated_text, usage)
    """
    output_audio_path = "test_speech.wav"
    waveform, gen_text, usage = ming.generate(
        text="Introduce Hangzhou", 
        output_type="speech", 
        max_new_tokens=128
    )
    
    sr = 16000  # Sample rate: 16kHz
    torchaudio.save(output_audio_path, waveform, sr)
    
    print(f"Generated text: {gen_text}")
    print(f"Audio saved to: {output_audio_path}")
    print(f"Usage: {usage}")
    assert os.path.exists(output_audio_path), "Audio file generation failed"


def test_audio_generate_stream():
    """
    Test streaming speech generation (Speech-to-Speech).
    
    Flow:
    1. Stream generate text response
    2. Convert each text chunk to speech in real-time
    
    Features:
    - Lower first-token latency
    - Better user experience
    
    Use case: Real-time voice conversation, phone customer service.
    
    Yields:
        tuple: (data_type, data_content) where data_content varies by type
    """
    all_wavs = []
    all_text = ""
    request_id = ""
    output_audio_path = "test_speech_stream.wav"
    
    for data_type, data_content in ming.generate_stream(
        text="Introduce Hangzhou", 
        output_type="speech", 
        max_new_tokens=128
    ):
        if data_type == "text_data":
            # Pure text chunk (intermediate output)
            text, usage = data_content
        elif data_type == "text_audio_data":
            # Text + audio chunk
            tts_speech, text, meta_info, session_id, usage = data_content
            all_text += text
            all_wavs.append(tts_speech)
            print(f"Chunk: {text}")
    
    # Concatenate all audio chunks
    waveform = torch.cat(all_wavs, dim=-1)
    sr = 16000
    torchaudio.save(output_audio_path, waveform, sr)
    
    print(f"Full text: {all_text}")
    print(f"Audio saved to: {output_audio_path}")
    print(f"Usage: {usage}")
    assert os.path.exists(output_audio_path), "Streaming audio generation failed"


def test_audio_generate_stream_interrupt():
    """
    Test streaming speech generation with interruption capability.
    
    Use case: User-initiated interruption, timeout control.
    
    The generation will be interrupted when text length exceeds threshold.
    """
    all_wavs = []
    all_text = ""
    request_id = "test-interrupt-001"
    output_audio_path = "test_speech_interrupt.wav"
    
    for data_type, data_content in ming.generate_stream(
        text="Introduce Hangzhou", 
        output_type="speech", 
        max_new_tokens=128,
        msg_request_id=request_id
    ):
        if data_type == "text_data":
            text, usage = data_content
        elif data_type == "text_audio_data":
            tts_speech, text, meta_info, session_id, usage = data_content
            all_text += text
            all_wavs.append(tts_speech)
        
        # Interrupt when text length exceeds 20 characters
        if len(all_text) > 20:
            print(f"Interrupting at: {all_text}")
            ming.generate_interrupt(request_id)
            break
    
    if all_wavs:
        waveform = torch.cat(all_wavs, dim=-1)
        sr = 16000
        torchaudio.save(output_audio_path, waveform, sr)
    
    print(f"Interrupted text: {all_text}")
    print(f"Audio saved to: {output_audio_path}")
    assert os.path.exists(output_audio_path), "Interrupt test failed"


# ============================================================================
# TTS Tests
# ============================================================================

def test_tts():
    """
    Test pure text-to-speech (TTS) conversion.
    
    Use case: Text reading, voice announcement systems.
    
    Note: Unlike speech generation, TTS directly converts input text
    to speech without generating intermediate response text.
    """
    output_audio_path = "test_tts.wav"
    waveform, usage = ming.generate(
        text="I love the Forbidden City in Beijing", 
        output_type="tts"
    )
    
    sr = 16000  # Sample rate: 16kHz
    torchaudio.save(output_audio_path, waveform, sr)
    
    print(f"Audio saved to: {output_audio_path}")
    print(f"Duration: {waveform.shape[-1] / sr:.2f} seconds")
    print(f"Usage: {usage}")
    assert os.path.exists(output_audio_path), "TTS generation failed"


# ============================================================================
# Image Tests
# ============================================================================

def test_image_qa():
    """
    Test image understanding (Image QA).
    
    Use case: Image description, visual question answering.
    
    Args:
        image: Path to the image file (supports jpg, png, etc.)
    """
    image_path = "test.png"
    
    # Check if test image exists
    if not os.path.exists(image_path):
        print(f"Skipping test: Image file {image_path} not found")
        return
    
    text, usage = ming.generate(
        text="Describe this image in detail", 
        image=image_path, 
        output_type="text"
    )
    
    print(f"Image description: {text}")
    print(f"Usage: {usage}")
    assert text is not None, "Image QA failed"


# ============================================================================
# Audio Understanding Tests
# ============================================================================

def test_audio_task():
    """
    Test audio understanding (ASR/Audio QA).
    
    Supported tasks:
    - Automatic Speech Recognition (ASR)
    - Audio content understanding
    - Audio event detection
    
    Args:
        audio: Path to audio file or URL
    """
    audio_path = "test.wav"
    
    # Check if test audio exists
    if not os.path.exists(audio_path):
        print(f"Skipping test: Audio file {audio_path} not found")
        return
    
    asr_result, usage = ming.generate(
        text="Please recognize the language of this speech and transcribe it. Format: oral.",
        audio=audio_path,
    )
    
    print(f"ASR result: {asr_result}")
    print(f"Usage: {usage}")
    assert asr_result is not None, "Audio task failed"


# ============================================================================
# Video Understanding Tests
# ============================================================================

def test_video():
    """
    Test video understanding.
    
    Supported tasks:
    - Video content description
    - Video QA
    - Video summarization
    
    Args:
        video: Path to video file (supports mp4, etc.)
    """
    video_path = "test.mp4"
    
    # Check if test video exists
    if not os.path.exists(video_path):
        print(f"Skipping test: Video file {video_path} not found")
        return
    
    text, usage = ming.generate(
        text="Describe this video in detail", 
        video=video_path, 
        output_type="text"
    )
    
    print(f"Video description: {text}")
    print(f"Usage: {usage}")
    assert text is not None, "Video understanding failed"


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # =========================================================================
    # Configuration
    # =========================================================================
    
    # Model path - update this to your actual model path
    MODEL_PATH = "your_model_path"
    
    # GPU device mapping for different modules
    DEVICE_MAP = {"talker": ["cuda:0"]}
    
    # Initialize Ming SDK
    print("=" * 60)
    print("Initializing Ming SDK...")
    print("=" * 60)
    
    ming = Ming(
        model_path=MODEL_PATH,
        device="0,1,2,3",  # GPU devices for LLM (comma-separated)
        gpu_memory_utilization={"moe": 0.8, "talker": 0.17},
        device_map=DEVICE_MAP,
        speaker="DB30",  # TTS speaker ID
        with_async=True,
        use_talker=True
    )
    
    print("Ming SDK initialized successfully!\n")
    
    # =========================================================================
    # Run Tests
    # =========================================================================
    
    # Text generation tests
    print("\n" + "=" * 60)
    print("Running Text Generation Tests")
    print("=" * 60)
    test_text_generate()
    test_text_generate_stream()
    
    # Multimodal tests
    print("\n" + "=" * 60)
    print("Running Multimodal Tests")
    print("=" * 60)
    test_image_qa()
    test_audio_task()
    test_video()
    
    # Speech generation tests
    print("\n" + "=" * 60)
    print("Running Speech Generation Tests")
    print("=" * 60)
    test_audio_generate()
    test_audio_generate_stream()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)