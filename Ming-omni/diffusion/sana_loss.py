
import torch

import copy
from diffusers import DPMSolverMultistepScheduler
import os
from collections import OrderedDict
import logging
from safetensors.torch import load_file
from diffusers import (
    AutoencoderDC,
    FlowMatchEulerDiscreteScheduler,
    SanaTransformer2DModel
)
import torch.nn as nn
from .pipeline_sana import SanaPipeline
# from flux_encoder import tokenize_prompt, encode_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToClipMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm1 = nn.LayerNorm(2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

class ToClipMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm1 = nn.LayerNorm(2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states


class SanaModel_withMLP(nn.Module):
    def __init__(self, sana, vision_dim=1152):
        super().__init__()
        self.sana = sana
        self.dtype = torch.bfloat16
        self.mlp = ToClipMLP(vision_dim, 2304)
        # self.mlp_pool = ToClipMLP(vision_dim, 768)
        self.config = self.sana.config
    
    def forward(self, hidden_states,
                    timestep,
                    encoder_hidden_states,
                    return_dict,
                    encoder_attention_mask=None,
                     **kargs):

        encoder_hidden_states = self.mlp(encoder_hidden_states)
        hidden_states = self.sana(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                     **kargs
                )
        return hidden_states

    def enable_gradient_checkpointing(self):
        self.sana.enable_gradient_checkpointing()

def inference_load_denoising_pretrained_weights(
        net, 
        weights_path,
        names=None,
        prefix_to_remove=None,    
    ):
    # state_dict = load_file(weights_path, map_location="cpu")
    state_dict = load_file(weights_path)
    net.load_state_dict(state_dict, strict=False)
    return 


def load_denoising_pretrained_weights(
        net, 
        weights_path,
        names=None,
        prefix_to_remove=None,    
    ):
    state_dict = torch.load(weights_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "net" in state_dict:
        state_dict = state_dict["net"]

    #if torch.distributed.get_rank() == 0 and names is not None:
    #    embed()

    #torch.distributed.barrier()
    if names is not None:
        selected_state_dict = OrderedDict()
        for ori_name in names:
            name = ori_name[len(prefix_to_remove):] if prefix_to_remove is not None else ori_name
            selected_state_dict[name] = state_dict[ori_name]
    
        state_dict = selected_state_dict

    net.load_state_dict(state_dict, strict=True)
    return 


class SANALoss(torch.nn.Module):
    def __init__(
        self, 
        model_path, scheduler_path, vision_dim=3584, diffusion_type='flow_matching', convert_vpred_to_xpred=True, 
        checkpoint_path=None, 
        # checkpoint_path_withmlp=None, 
        # mlp_checkpoint_path=None, 
        mlp_state_dict=None,
        trainable_params='all', device='cpu', guidance_scale=3.5, revision=None, variant=None, repa_loss=False, mid_layer_idx=10, mid_loss_weight=1.0
    ):
        super(SANALoss, self).__init__()
        self.torch_type = torch.bfloat16
        self.base_model_path = model_path
        self.use_mid_loss = repa_loss
        self.mid_loss_weight = mid_loss_weight
        self.mid_layer_idx = mid_layer_idx
        #self.text_encoder = Gemma2Model.from_pretrained(model_path, subfolder="text_encoder")
        #self.tokenizer = AutoTokenizer.from_pretrained(model_path,subfolder="tokenizer")
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        #self.sana_pipeline = SanaPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16,)

        self.device = torch.device(torch.cuda.current_device())    
        self.scheduler_path = scheduler_path
        self.vae = AutoencoderDC.from_pretrained(
            model_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        
        # self.vae.to(self.torch_type).to(self.device)
        self.vae.requires_grad_(False)

        self.train_model = SanaTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", revision=revision, variant=variant
        )

        if checkpoint_path is not None:
            assert os.path.exists(checkpoint_path)
            load_denoising_pretrained_weights(self.train_model, checkpoint_path)
        
        # self.train_model = UNet2DConditionModel_withMLP(self.train_model, vision_dim=vision_dim)

        self.train_model = SanaModel_withMLP(self.train_model, vision_dim=vision_dim)
        # if checkpoint_path_withmlp is not None:
        #     assert os.path.exists(checkpoint_path_withmlp)
        #     load_denoising_pretrained_weights(self.train_model, checkpoint_path_withmlp)
        # elif mlp_checkpoint_path is not None:
        #     assert os.path.exists(mlp_checkpoint_path)
        #     inference_load_denoising_pretrained_weights(self.train_model, mlp_checkpoint_path)
        assert mlp_state_dict is not None
        self.train_model.mlp.load_state_dict(mlp_state_dict, strict=True)

        # 创建处理中间层特征的MLP
        hidden_dim = 2240
        self.mid_layer_mlp = None
        if self.use_mid_loss:
            self.mid_layer_mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim * 2),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim * 2, 32),
                torch.nn.LayerNorm(32)
            )

            # 初始化MLP的权重
            for m in self.mid_layer_mlp.modules():
                if isinstance(m, torch.nn.Linear):
                    # 使用Kaiming初始化权重
                    torch.nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        # 将偏置初始化为0
                        torch.nn.init.zeros_(m.bias)

        self.train_model.enable_gradient_checkpointing()
        
        self.set_trainable_params(trainable_params)


        num_parameters_trainable = 0
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                                                        self.scheduler_path, subfolder="scheduler"
                                                    )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)


        # if self.train_model.config.guidance_embeds:
        #     self.guidance = torch.tensor([guidance_scale], device=self.device)
        #     # guidance = guidance.expand(model_input.shape[0])
        # else:
        #     self.guidance = None

        logger.info("Preparation done. Starting training diffusion ...")

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        # sigmas = noise_scheduler_copy.sigmas.to(device=self.device, dtype=dtype)
        sigmas = self.noise_scheduler_copy.sigmas
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(device=timesteps.device)
        timesteps = timesteps
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_text_embeddings(self, prompt, text_encoders, tokenizers):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                [text_encoders], [tokenizers], prompt, 77
            )
            # prompt_embeds = prompt_embeds.to(local_rank)
            pooled_prompt_embeds = pooled_prompt_embeds.to(local_rank)
            # text_ids = text_ids.to(local_rank)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def set_trainable_params(self, trainable_params):
        
        self.vae.requires_grad_(False)

        if trainable_params == 'all':
            self.train_model.requires_grad_(True)
        else:
            self.train_model.requires_grad_(False)
            for name, module in self.train_model.named_modules():
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        for params in module.parameters():
                            params.requires_grad = True

        num_parameters_trainable = 0
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()

    def sample(self, encoder_hidden_states, steps=20, cfg=7.0, seed=42, height=512, width=512):
        #self.pipelines = SanaPipeline.from_pretrained(self.base_model_path)#.to(device=self.device)
        self.pipelines = SanaPipeline(vae=self.vae,
                         transformer=self.train_model,
                         text_encoder=None,
                         tokenizer=None,
                         scheduler=self.noise_scheduler,
                         ).to(self.device)
        
        prompt_attention_mask = torch.ones(encoder_hidden_states.shape[:2]).to(self.device)
        negative_attention_mask = torch.ones(encoder_hidden_states.shape[:2]).to(self.device)

        image = self.pipelines(
            prompt_embeds=encoder_hidden_states,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=encoder_hidden_states*0,
            negative_prompt_attention_mask=negative_attention_mask,
            guidance_scale=cfg,
            generator=torch.manual_seed(seed),
            num_inference_steps=steps,
            device=self.device,
            height=height,
            width=width,
            max_sequence_length=300,
        ).images[0]

        return image  
