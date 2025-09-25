from dataclasses import dataclass
from typing import Optional, Tuple, List
import os
import yaml
import re
import torch
import torch.nn as nn
import torchaudio
from hyperpyyaml import load_hyperpyyaml

from transformers import Qwen2Config, PreTrainedModel
from transformers import Qwen2ForCausalLM, AutoTokenizer
from audio_detokenizer.cli.model import AudioDetokenizerModel
from transformers.models.bailingmm.s3bpe_tokenizer import S3BpeTokenizer
from configuration_bailing_talker import BailingTalkerConfig
from transformers.utils import ModelOutput
from sentence_manager.sentence_manager import SentenceNormalizer

@dataclass
class BailingTalkerOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None


class BailingTalkerForConditionalGeneration(PreTrainedModel):
    config_class = BailingTalkerConfig
    base_model_prefix = 'model'
    def __init__(self, config:BailingTalkerConfig):
        super().__init__(config)

        self.config = config
        self.vocab_size = self.config.vocab_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.config._name_or_path)
        self.model_config = Qwen2Config.from_pretrained(self.config._name_or_path)
        self.model = Qwen2ForCausalLM(self.model_config)
        self.model.resize_token_embeddings(self.vocab_size)
        self.thinker_to_talker_proj = nn.Linear(self.config.qa_model_hidden_size, self.model_config.hidden_size)
        self.vp_head = nn.Conv1d(
            self.config.vp_feature_size,
            self.model_config.hidden_size,
            kernel_size=self.config.vp_kernel_size,
            stride=self.config.vp_stride,
            padding=self.config.vp_kernel_size // 2,
        )
        self.s3bpe_tokenizer = S3BpeTokenizer(bpe_model=f"{self.config._name_or_path}/s3_bpe/tokenizer.json", mapping_file=f"{self.config._name_or_path}/s3_bpe/char_mapping.txt")

        self.loss_function = nn.CrossEntropyLoss()
        
        default_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "sentence_manager/default_config.yaml")
        self.sentence_manager_config = yaml.safe_load(open(default_config_path))
        if "split_token" not in self.sentence_manager_config:
            self.sentence_manager_config["split_token"] = []
        assert isinstance(self.sentence_manager_config["split_token"], list)
        self.sentence_manager_config["split_token"].append(re.escape(self.tokenizer.eos_token))        
        self.normalizer = SentenceNormalizer(self.sentence_manager_config.get("text_norm", {}))

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def encode_audio_segments(
        self,
        inputs_embeds: torch.FloatTensor,
        vp_emb: torch.FloatTensor,
        vp_insert_loc: torch.LongTensor,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        thinker_reply_length: Optional[List] = None,
        thinker_prefix_insert_loc: Optional[torch.LongTensor] = None
    ):
        vp_emb_encoded = self.vp_head(vp_emb.transpose(-1, -2)).transpose(-1, -2)

        for idx in range(vp_insert_loc.shape[0]):
            inputs_embeds[idx, vp_insert_loc[idx].item():vp_insert_loc[idx].item() + 1, :] = vp_emb_encoded[idx, :, :]

        if thinker_prefix_insert_loc is not None:
            thinker_reply_part = self.thinker_to_talker_proj(thinker_reply_part)
            for idx in range(thinker_prefix_insert_loc.shape[0]):
                real_length = thinker_reply_length[idx]
                inputs_embeds[idx,
                thinker_prefix_insert_loc[idx].item():thinker_prefix_insert_loc[idx].item() + real_length,
                :] = thinker_reply_part[idx, :real_length, :]

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[dict] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        text_input_ids: Optional[torch.LongTensor] = None,
        vp_emb: Optional[torch.FloatTensor] = None,
        vp_insert_loc: Optional[torch.LongTensor] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        thinker_reply_length: Optional[torch.FloatTensor] = None,
        thinker_prefix_insert_loc: Optional[torch.LongTensor] = None,
    ):

        if inputs_embeds is None:
            audio_input_embeds = self.model.get_input_embeddings()(input_ids)
            text_input_embeds = self.model.get_input_embeddings()(text_input_ids)
            inputs_embeds = audio_input_embeds + text_input_embeds
            if past_key_values is None:
                inputs_embeds = self.encode_audio_segments(
                    inputs_embeds, vp_emb, vp_insert_loc, thinker_reply_part=thinker_reply_part,
                    thinker_reply_length=thinker_reply_length, thinker_prefix_insert_loc=thinker_prefix_insert_loc
                )

            if position_ids is None:
                position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            loss = self.loss_function(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        return BailingTalkerOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            logits=logits,
        )

    def sample(self, logits, topk=20, filter_value=-float("Inf")):
        logits = logits.reshape(1, -1)  # [1, V]
        indices_to_remove = logits < torch.topk(logits, topk)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        token_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).to(torch.long)
        return token_id


    def omni_audio_generation_func(
        self,
        tts_text,
        prompt,
        prefix_from_thinker,
        vp,
        position_ids,
        talker_audio_prefix,
        vp_insert_loc,
        thinker_length,
        vp_emb=None,
        thinker_reply_part=None,
    ):

        text_input_part = self.tokenizer.encode(tts_text)
        # audio_prefix and text_prefix for first step generation
        talker_text_prefix = (
                prompt +
                prefix_from_thinker +
                vp +
                text_input_part[:1]
        )
        # the rest of input_text
        talker_text_input_part = (
            text_input_part[1:] +
            self.tokenizer.encode("<text_eos>") +
            self.tokenizer.encode("<text_pad>")
        )

        talker_text_prefix = torch.tensor(talker_text_prefix).reshape(1, -1).to(self.device)
        
        

        audio_token = self.generate(
            talker_audio_prefix=talker_audio_prefix,
            talker_text_prefix=talker_text_prefix,
            talker_text_input_part=talker_text_input_part,
            position_ids=position_ids,
            vp_emb=vp_emb,
            vp_insert_loc=vp_insert_loc,
            thinker_reply_part=thinker_reply_part,
            thinker_reply_length=torch.tensor([thinker_length]).to(self.device),
            thinker_prefix_insert_loc=torch.tensor([len(prompt) + 1]).to(self.device) if thinker_reply_part is not None else None,
        )

        audio_token = [ele - len(self.tokenizer) for ele in audio_token]
        audio_token = self.s3bpe_tokenizer.decode(audio_token)
        audio_token = torch.tensor([audio_token], dtype=torch.int32)

        return audio_token

    def text_length(self, text):
        return len(re.findall("[\u4e00-\u4E27\u4E29-\u4E3E\u4E42-\u9fa4]", text))

    def cut_text(self, text, max_length, tail_min_length=5):
        def text_append(text_list, text, max_length):
            if len(text_list) == 0:
                text_list.append(text)
            else:
                if len(text_list[-1]) + self.text_length(text) <= max_length:
                    if text_list[-1].endswith("。") and self.text_length(text) < tail_min_length:
                        text_list.append(text.lstrip("，"))
                    else:
                        text_list[-1] += text 
                else:
                    text_list.append(text.lstrip("，"))
            return text_list 

        text = text.replace("\n", " ")
        text = self.normalizer.normalize(text)
        text = text.replace("。，", "。")
        if len(text) <= max_length:
            return [text]
        text_list = []
        text = text.replace(".", "。").replace(",", "，")

        sps1 = []
        for t in text.split("。"):
            t = t.strip()
            if len(t) > 0:
                if t[-1] not in "！？，。!?,.":
                    t += "。"
                sps1.append(t)

        for text_piece1 in sps1:
            sps2 = []
            for t in text_piece1.split("，"):
                t = t.strip()
                if len(t) > 0:
                    if t[-1] not in "！？，。!?,.":
                        t += "，"
                    sps2.append(t)
                    
            for text_piece2 in sps2:
                text_piece2 = text_piece2.replace("。，", "。")
                if self.text_length(text_piece2) > max_length:
                    for i in range(0, len(text_piece2), max_length):
                        text_list = text_append(text_list, text_piece2[i:i+max_length], max_length)
                else:
                    text_list = text_append(text_list, text_piece2, max_length)
        return text_list

    def omni_audio_generation(
        self,
        tts_text,
        vp_emb=None,
        thinker_reply_part=None,
        max_length=50,
    ):

        # thinker_reply_part: [B, T, d]
        # get text_emb and hidden_states from thinker
        thinker_length = thinker_reply_part.size(1) if thinker_reply_part is not None else 0
        prefix_from_thinker = (
            self.tokenizer.encode("<thinker_prefix>") +
            self.tokenizer.encode("<audio_pad>") * thinker_length +  # placeholder for prefix emb from thinker
            self.tokenizer.encode("</thinker_prefix>")
        )

        prompt = self.tokenizer.encode("<prompt>") + self.tokenizer.encode("</prompt>")
        vp = (
            self.tokenizer.encode("<vp>") +
            self.tokenizer.encode("<audio_pad>") +
            self.tokenizer.encode("</vp>")
        )
        talker_audio_prefix = (
            prompt +
            prefix_from_thinker +
            vp +
            self.tokenizer.encode("<audio_bos>")
        )
        attention_mask = torch.ones(len(talker_audio_prefix)).reshape(1, -1).to(self.device)
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill_((attention_mask == 0), 1)[:, -1].view(1, -1)
        talker_audio_prefix = torch.tensor(talker_audio_prefix).reshape(1, -1).to(self.device)
        vp_insert_loc = torch.tensor(len(prompt) + len(prefix_from_thinker) + 1, dtype=torch.long).reshape(1, -1)
        vp_emb = vp_emb.unsqueeze(0).to(torch.bfloat16).to(self.device)

        assert max_length > 0, f"max_length must be greater than 0, but here is {max_length}"
        text_list = self.cut_text(tts_text, max_length)

        audio_tokens = []
        for text in text_list:
            audio_tokens_piece = self.omni_audio_generation_func(
                tts_text=text,
                prompt=prompt,
                prefix_from_thinker=prefix_from_thinker,
                vp=vp,
                position_ids=position_ids,
                talker_audio_prefix=talker_audio_prefix,
                vp_insert_loc=vp_insert_loc,
                thinker_length=thinker_length,
                vp_emb=vp_emb,
                thinker_reply_part=thinker_reply_part,
            )
            audio_tokens.append(audio_tokens_piece)
        return audio_tokens


    @torch.no_grad()
    def generate(
        self,
        talker_audio_prefix: torch.LongTensor,
        talker_text_prefix: torch.LongTensor,
        talker_text_input_part: List,
        position_ids: Optional[torch.LongTensor] = None,
        vp_emb: Optional[torch.FloatTensor] = None,
        vp_insert_loc: Optional[torch.LongTensor] = None,
        thinker_reply_part: Optional[torch.FloatTensor] = None,
        thinker_reply_length: Optional[torch.FloatTensor] = None,
        thinker_prefix_insert_loc: Optional[torch.LongTensor] = None,
    ):
        result = []
        step = 0
        eos_id = self.tokenizer.encode("<audio_eos>")[0]
        while step < 1000:
            if step == 0:
                talker_audio_input_ids = talker_audio_prefix
                talker_text_input_ids = talker_text_prefix
                attention_mask = torch.ones(talker_audio_input_ids.shape).to(talker_audio_prefix.device)

            else:
                talker_audio_input_ids = next_token
                talker_text_input_ids = torch.tensor(talker_text_input_part[0], dtype=torch.long).reshape(1, -1).to(
                    talker_audio_prefix.device)
                attention_mask = torch.ones(next_token.shape[0], 1).to(talker_audio_prefix.device)
                position_ids += 1
                thinker_prefix_insert_loc = None

                if len(talker_text_input_part) > 1:
                    talker_text_input_part = talker_text_input_part[1:]
            # print(talker_audio_input_ids, self.tokenizer.decode(talker_text_input_ids.tolist()[0]), attention_mask, position_ids)
            outputs = self(
                input_ids=talker_audio_input_ids,
                text_input_ids=talker_text_input_ids,
                thinker_reply_part=thinker_reply_part,
                thinker_reply_length=thinker_reply_length,
                thinker_prefix_insert_loc=thinker_prefix_insert_loc,
                vp_emb=vp_emb,
                vp_insert_loc=vp_insert_loc,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=outputs.past_key_values if step > 0 else None
            )
            # 采样
            logits = outputs.logits[:, -1, :]

            next_token = self.sample(logits)
            if next_token.item() == eos_id:
                break
            result.append(next_token.item())
            step += 1

        return result


class AudioDetokenizer:
    def __init__(self, config_path, flow_model_path, hifigan_model_path):
        with open(config_path, 'r') as f:
            configs = load_hyperpyyaml(f)

        self.model = AudioDetokenizerModel(configs['flow'], configs['hift'])
        self.model.load(flow_model_path, hifigan_model_path)
        self.sr = 22050

    def token2wav(self, audio_tokens, flow_embedding, save_path=None):
        assert isinstance(audio_tokens, list), f"audio_tokens should be list"
        speech_list = []
        for audio_token in audio_tokens:
            model_input = {"tts_speech_token": audio_token,
                        'flow_embedding': flow_embedding}
            model_output = self.model.inference(**model_input)

            silent_dur = 0.02
            silent_tensor = torch.Tensor([0.0] * int(self.sr * silent_dur))
            model_output['tts_speech'][0][:int(self.sr * silent_dur)] = silent_tensor

            speech_list.append(model_output['tts_speech'])
        if len(speech_list) == 1:
            speech = speech_list[0]
        else:
            speech = torch.cat(speech_list, dim=1)
        if save_path is not None:
            torchaudio.save(save_path, speech, sample_rate=self.sr)            
        return speech
