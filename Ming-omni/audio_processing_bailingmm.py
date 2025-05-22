from typing import List, Tuple, Dict, Optional, Any, Union
import os
import copy

import numpy as np
import torch
import torch.utils.data
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from torch.nn.utils.rnn import pad_sequence

from transformers.utils import TensorType
from transformers.feature_extraction_utils import FeatureExtractionMixin, BatchFeature

NORM_FACTOR_FOR_DTYPE = {
    torch.int8: 2**7,
    torch.int16: 2**15,
    torch.int32: 2**31,
    torch.int64: 2**63,
    torch.float32: 1,
    torch.float64: 1,
}

# special tokens
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


class BailingMMAudioProcessor(FeatureExtractionMixin):
    def __init__(self, wav_frontend_args: Dict[str, Any]=None, whisper_frontend_args: Dict[str, Any]=None, **kwargs):
        super().__init__(**kwargs)
        self.sample_rate = 16000
        if wav_frontend_args is not None:
            self.wav_frontend = WavFrontend(**wav_frontend_args)
        if whisper_frontend_args is not None:
            self.whisper_frontend = WhisperFrontend(**whisper_frontend_args)

    def to_dict(self) -> Dict[str, Any]:
        output = copy.deepcopy(self.__dict__)
        output["wav_frontend"] = output["wav_frontend"].__dict__
        output["wav_frontend"]["cmvn"] = output["wav_frontend"]["cmvn"].tolist()
        output["wav_frontend"]["_non_persistent_buffers_set"] = list(output["wav_frontend"]["_non_persistent_buffers_set"])
        output["audio_processor_type"] = self.__class__.__name__
        if 'whisper_frontend' in output:
            output["whisper_frontend"] = output["whisper_frontend"].__dict__
        return output

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Auto-fill the cmvn file path.
        """
        result, kwargs = super().get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
        if not result["wav_frontend_args"]["cmvn_file"].startswith("/"):
            # Convert to an absolute path.
            if os.path.isdir(pretrained_model_name_or_path):
                pretrained_model_dir = pretrained_model_name_or_path
            else:
                pretrained_model_dir = os.path.dirname(pretrained_model_name_or_path)
            result["wav_frontend_args"]["cmvn_file"] = os.path.join(
                pretrained_model_dir, result["wav_frontend_args"]["cmvn_file"]
            )
        return result, kwargs

    def __call__(self, audios, **kwargs) -> BatchFeature:
        """Preprocess an audio or a batch of audios."""
        return self.preprocess(audios, **kwargs)

    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int, use_whisper_encoder: bool=False) -> torch.Tensor:
        waveform = normalize_audio_tensor(waveform, sample_rate, target_sample_rate=self.sample_rate)
        if not use_whisper_encoder:
            audio_feat = self.wav_frontend(waveform.unsqueeze(0), [len(waveform)])[0].squeeze(0)
        else:
            audio_feat = self.whisper_frontend(waveform.unsqueeze(0), [len(waveform)])[0].squeeze(0)
        return audio_feat

    def _make_batched_audios(self, audio_feat_list: List[torch.Tensor], use_whisper_encoder=False) -> Dict[str, Any]:
        audio_feats_lengths = torch.tensor([[audio_feat.shape[0]] for audio_feat in audio_feat_list], dtype=torch.long)
        if not use_whisper_encoder:
            encoder_feats_lengths = audio_feats_lengths
        else:
            # whisper + project layer has two conv
            encoder_feats_lengths = ((audio_feats_lengths-3+2*1)//2+1-3+2*1)//2+1

        max_length = max(audio_feat.shape[0] for audio_feat in audio_feat_list)
        audio_feats = torch.stack(
            [
                torch.cat(
                    (audio_feat, torch.zeros((max_length - audio_feat.shape[0], *audio_feat.shape[1:]), dtype=audio_feat.dtype)),
                    dim=0,
                ) for audio_feat in audio_feat_list
            ], dim=0,
        )
        return {"audio_feats": audio_feats.numpy(), "audio_feats_lengths": audio_feats_lengths.numpy(), "encoder_feats_lengths": encoder_feats_lengths}

    def preprocess(
        self,
        audios: Union[Tuple[torch.Tensor, int], List[Tuple[torch.Tensor, int]]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if isinstance(audios, List):
            audio_inputs = self._make_batched_audios([self._preprocess_audio(waveform, sr, use_whisper_encoder=kwargs.get('use_whisper_encoder', False)) for waveform, sr in audios], use_whisper_encoder=kwargs.get('use_whisper_encoder', False))
        else:
            waveform, sr = audios
            audio_inputs = self._make_batched_audios([self._preprocess_audio(waveform, sr, use_whisper_encoder=kwargs.get('use_whisper_encoder', False))])
        return BatchFeature(data=audio_inputs, tensor_type=return_tensors)


class WavFrontend(torch.nn.Module):
    """Conventional frontend structure for ASR.
    """

    def __init__(
            self,
            cmvn_file: Optional[str] = None,
            fs: int = 16000,
            window: str = 'hamming',
            n_mels: int = 80,
            frame_length: int = 25,
            frame_shift: int = 10,
            filter_length_min: int = -1,
            filter_length_max: int = -1,
            lfr_m: int = 1,
            lfr_n: int = 1,
            dither: float = 1.0,
            snip_edges: bool = True,
            upsacle_samples: bool = True,
    ):
        super().__init__()
        self.fs = fs
        self.window = window
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.filter_length_min = filter_length_min
        self.filter_length_max = filter_length_max
        self.lfr_m = lfr_m
        self.lfr_n = lfr_n
        self.cmvn_file = cmvn_file
        self.dither = dither
        self.snip_edges = snip_edges
        self.upsacle_samples = upsacle_samples
        self.cmvn = None if self.cmvn_file is None else load_cmvn(self.cmvn_file)

    def output_size(self) -> int:
        return self.n_mels * self.lfr_m

    def forward(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            if self.upsacle_samples:
                waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.n_mels,
                              frame_length=self.frame_length,
                              frame_shift=self.frame_shift,
                              dither=0.0, #self.dither
                              energy_floor=0.0,
                              window_type=self.window,
                              sample_frequency=self.fs,
                              snip_edges=self.snip_edges)

            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        if batch_size == 1:
            feats_pad = feats[0][None, :, :]
        else:
            feats_pad = pad_sequence(feats,
                                     batch_first=True,
                                     padding_value=0.0)
        # import ipdb;ipdb.set_trace()
        return feats_pad, feats_lens

    def forward_fbank(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            waveform_length = input_lengths[i]
            waveform = input[i][:waveform_length]
            waveform = waveform * (1 << 15)
            waveform = waveform.unsqueeze(0)
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.n_mels,
                              frame_length=self.frame_length,
                              frame_shift=self.frame_shift,
                              dither=self.dither,
                              energy_floor=0.0,
                              window_type=self.window,
                              sample_frequency=self.fs)

            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens

    def forward_lfr_cmvn(
            self,
            input: torch.Tensor,
            input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input.size(0)
        feats = []
        feats_lens = []
        for i in range(batch_size):
            mat = input[i, :input_lengths[i], :]
            if self.lfr_m != 1 or self.lfr_n != 1:
                mat = apply_lfr(mat, self.lfr_m, self.lfr_n)
            if self.cmvn is not None:
                mat = apply_cmvn(mat, self.cmvn)
            feat_length = mat.size(0)
            feats.append(mat)
            feats_lens.append(feat_length)

        feats_lens = torch.as_tensor(feats_lens)
        feats_pad = pad_sequence(feats,
                                 batch_first=True,
                                 padding_value=0.0)
        return feats_pad, feats_lens


class WhisperFrontend:
    def __init__(self, n_mels: int=128):
        self.n_mels = n_mels

    def __call__(self, input: torch.Tensor, input_lengths: List[int]):
        """
        input: [B, T]
        input_lengths: [B]
        """

        assert input.size(0) == 1

        mel = whisper.log_mel_spectrogram(input.squeeze(0), n_mels=self.n_mels).to(input.device)   # [n_mels, T]
        feats_pad = mel.transpose(0, 1).unsqueeze(0)  # [B=1, T, n_mels]
        feats_lens = torch.tensor([mel.size(1)], dtype=torch.long)  # [B=1]
        return feats_pad, feats_lens

def load_cmvn(cmvn_file):
    with open(cmvn_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if line_item[0] == '<AddShift>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                add_shift_line = line_item[3:(len(line_item) - 1)]
                means_list = list(add_shift_line)
                continue
        elif line_item[0] == '<Rescale>':
            line_item = lines[i + 1].split()
            if line_item[0] == '<LearnRateCoef>':
                rescale_line = line_item[3:(len(line_item) - 1)]
                vars_list = list(rescale_line)
                continue
    means = np.array(means_list).astype(np.float32)
    vars = np.array(vars_list).astype(np.float32)
    cmvn = np.array([means, vars])
    cmvn = torch.as_tensor(cmvn, dtype=torch.float32)
    return cmvn


def apply_cmvn(inputs, cmvn):  # noqa
    """
    Apply CMVN with mvn data
    """

    device = inputs.device
    dtype = inputs.dtype
    frame, dim = inputs.shape

    means = cmvn[0:1, :dim]
    vars = cmvn[1:2, :dim]
    inputs += means.to(device)
    inputs *= vars.to(device)

    return inputs.type(torch.float32)


def apply_lfr(inputs, lfr_m, lfr_n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / lfr_n))
    left_padding = inputs[0].repeat((lfr_m - 1) // 2, 1)
    inputs = torch.vstack((left_padding, inputs))
    T = T + (lfr_m - 1) // 2
    for i in range(T_lfr):
        if lfr_m <= T - i * lfr_n:
            LFR_inputs.append((inputs[i * lfr_n:i * lfr_n + lfr_m]).view(1, -1))
        else:  # process last LFR frame
            num_padding = lfr_m - (T - i * lfr_n)
            frame = (inputs[i * lfr_n:]).view(-1)
            for _ in range(num_padding):
                frame = torch.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)
    LFR_outputs = torch.vstack(LFR_inputs)
    return LFR_outputs.type(torch.float32)


def normalize_audio_tensor(
    waveform: torch.Tensor,
    sample_rate: int,
    device=None,
    target_sample_rate: Optional[int] = None,
):
    # Ensure dtype == float32.
    assert waveform.dtype in NORM_FACTOR_FOR_DTYPE, f"Unsupported waveform dtype: {waveform.dtype}"
    norm_factor = NORM_FACTOR_FOR_DTYPE[waveform.dtype]
    waveform = waveform.to(torch.float32) / norm_factor

    # Remove the channel dimension.
    while len(waveform.shape) > 1:
        waveform = waveform[0]

    # Move to device.
    if device is not None:
        waveform = waveform.to(device)

    # Resample.
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        if device is not None:
            resampler = resampler.to(device)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    return waveform

