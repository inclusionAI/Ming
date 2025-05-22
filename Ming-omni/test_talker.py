import os
import torch

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig
)

from modeling_bailingmm import BailingMMNativeForConditionalGeneration


import warnings
import argparse

from modeling_bailing_talker import AudioDetokenizer

warnings.filterwarnings("ignore")

class BailingMMInfer:
    def __init__(self,
        model_name_or_path,
        device="cuda",
        generation_config=None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.device = device

        self.model, self.tokenizer, self.processor = self.load_model_processor()
        if generation_config is None:
            generation_config = {"num_beams": 1}
        self.generation_config = GenerationConfig.from_dict(generation_config)
        self.audio_detokenizer = AudioDetokenizer(
            f'{self.model_name_or_path}/talker/audio_detokenizer.yaml',
            flow_model_path=f'{self.model_name_or_path}/talker/flow.pt',
            hifigan_model_path=f'{self.model_name_or_path}/talker/hift.pt'
        )
        self.spk_info = {
            'luna': torch.load('data/spks/luna.pt')
        }

    def load_model_processor(self):
        model = BailingMMNativeForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).eval().to(self.device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        return model, tokenizer, processor

    def generate(self, messages, max_new_tokens=512, speaker='luna', output_audio_path=None, output_audio=False, use_whisper_encoder=False):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, use_system=True
        )

        image_inputs, video_inputs, audio_inputs = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={'use_whisper_encoder': use_whisper_encoder}
        )

        inputs = inputs.to(self.device)

        for k in inputs.keys():
            if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
                inputs[k] = inputs[k].to(dtype=torch.bfloat16)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=self.processor.gen_terminator,
                generation_config=self.generation_config,
                use_whisper_encoder=use_whisper_encoder
            )

        generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if self.model.talker is not None and output_audio:
            thinker_reply_part = outputs.hidden_states[0][0] + outputs.hidden_states[0][-1]
            spk_embed = self.spk_info.get(speaker, 'luna')
            audio_tokens = self.model.talker.omni_audio_generation(output_text, vp_emb=spk_embed, thinker_reply_part=thinker_reply_part)
            waveform = self.audio_detokenizer.token2wav(audio_tokens, spk_embed, save_path=output_audio_path)
            return output_text, waveform
        return output_text


if __name__ == '__main__':
    max_new_tokens = 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name_or_path = '.'
    model = BailingMMInfer(
        model_name_or_path,
        device=device,
        generation_config={
            'output_hidden_states': True,
            'return_dict_in_generate': True,
            'no_repeat_ngram_size': 10
        }
    )

    # ASR
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "Please recognize the language of this speech and transcribe it. Format: oral."},
                {"type": "audio", "audio": 'data/wavs/BAC009S0915W0292.wav'},
            ],
        },
    ]

    outputs = model.generate(messages, max_new_tokens=max_new_tokens, use_whisper_encoder=True)
    print(outputs)

    # speech qa + tts
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "audio", "audio": 'data/wavs/BAC009S0915W0292.wav'},
            ],
        },
    ]

    outputs = model.generate(messages, max_new_tokens=max_new_tokens, speaker='luna', output_audio_path='out.wav', output_audio=True)

    print(outputs)
