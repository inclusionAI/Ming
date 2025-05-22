import whisper
from torch import Tensor
import torch.nn.functional as F
from whisper.model import AudioEncoder


class WhisperAudioEncoder(AudioEncoder):
    """
        We inherited the original Whisper encoder and modified its 30-second fixed-length padding logic to
        improve training and inference efficiency.
    """
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        self.audio_emb_dim = n_state

    def forward(self, x: Tensor):
        """
           x : torch.Tensor, shape = [B, T, d] the mel spectrogram of the audio
       """
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        positional_embedding = self.positional_embedding[:x.shape[1], :]
        assert x.shape[1:] == positional_embedding.shape, "incorrect audio shape"
        x = (x + positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        whisper_model = whisper.load_model(model_path)
        audio_encoder = cls(
            whisper_model.dims.n_mels,
            whisper_model.dims.n_audio_ctx*10,
            whisper_model.dims.n_audio_state,
            whisper_model.dims.n_audio_head,
            whisper_model.dims.n_audio_layer,
        ).to(whisper_model.device)
        state_dict = whisper_model.encoder.state_dict()
        state_dict.pop('positional_embedding')
        ret = audio_encoder.load_state_dict(state_dict, strict=False)
        logger.warning(f'whisper encoder does not load `positional_embedding`. {ret}')
        audio_encoder.audio_emb_dim = whisper_model.dims.n_audio_state
        return audio_encoder