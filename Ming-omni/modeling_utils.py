#!/usr/bin/env python
# coding=utf-8
# @Author: jiangpeijie.jpj
# @Date: Mon 4 Dec 2023 05:21:28 PM CST

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from numpy import random
from torch import nn
from torch.nn import CrossEntropyLoss
from whisper.model import AudioEncoder

from transformers.activations import ACT2CLS, ClassInstantier

try:
    from atorch.distributed.distributed import parallel_group, parallel_group_size
except Exception:
    parallel_group = None
    parallel_group_size = None


# ## Activations
class SwiGLUActivatition(nn.Module):

    def forward(self, input):
        input = torch.chunk(input, 2, dim=-1)
        return F.silu(input[0]) * input[1]


ACT2CLS["swiglu"] = SwiGLUActivatition
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
swiglu = get_activation("swiglu")


# Rotary Position Embedding Utils
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """Find dim range bounds based on rotations"""
    low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    # dim=-1 triggers a bug in earlier torch versions
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


# Comment torchscript func for accurate calculate
# @torch.jit.script
def apply_rotary_pos_emb_index(q, k, cos, sin, position_id):
    # position_id: [sq, b], q, k: [sq, b, np, hn], cos: [sq, 1, hn] -> [sq, b, 1, hn]
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2), F.embedding(
        position_id, sin.squeeze(1)
    ).unsqueeze(2)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000, precision=torch.half, learnable=False):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq 保留float精度，避免bf16损失
        # inv_freq = inv_freq.to(precision)
        self.learnable = learnable
        if learnable:
            self.inv_freq = torch.nn.Parameter(inv_freq)
            self.max_seq_len_cached = None
        else:
            self.register_buffer('inv_freq', inv_freq)
            self.max_seq_len_cached = None
            self.cos_cached = None
            self.sin_cached = None
        self.precision = precision

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        pass

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            # freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            freqs = torch.outer(t, inv_freq.to(x.device))
            assert freqs.dtype == torch.float32
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def _apply(self, fn, *args, **kwargs):
        if self.cos_cached is not None:
            self.cos_cached = fn(self.cos_cached)
        if self.sin_cached is not None:
            self.sin_cached = fn(self.sin_cached)
        return super()._apply(fn, *args, **kwargs)


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling."""

    def __init__(
        self, dim, base=10000, precision=torch.half, learnable=False, max_embedding_length=2048, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        self.max_embedding_length = max_embedding_length
        super().__init__(dim, base, precision, learnable)

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len

            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            t = t / self.scaling_factor
            # freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            freqs = torch.outer(t, inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class NTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling."""

    def __init__(
        self, dim, base=10000, precision=torch.half, learnable=False, max_embedding_length=2048, scaling_factor=1.0
    ):
        self.scaling_factor = scaling_factor
        self.max_embedding_length = max_embedding_length
        super().__init__(dim, base, precision, learnable)

    def forward(self, x, seq_dim=1, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = None if self.learnable else seq_len

            base = self.base
            if seq_len > self.max_embedding_length:
                base = self.base * (
                    (self.scaling_factor * seq_len / self.max_embedding_length) - (self.scaling_factor - 1)
                ) ** (self.dim / (self.dim - 2))

            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            # freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            freqs = torch.outer(t, inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            # [sx, 1 (b * np), hn]
            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]


class DynamicYaRNScaledRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        dim,
        base=10000,
        precision=torch.half,
        learnable=False,
        max_embedding_length=2048,
        extrapolation_factor=1,
        attn_factor=1,
        beta_fast=32,
        beta_slow=1,
    ):
        self.max_embedding_length = max_embedding_length
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        super().__init__(dim, base, precision, learnable)

    def forward(self, x, seq_dim=1, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len is None:
            seq_len = x.shape[seq_dim]

        if self.max_seq_len_cached is None or (seq_len > self.max_seq_len_cached):
            self.max_seq_len_cached = seq_len

            if seq_len > self.max_embedding_length:
                self.yarn(seq_len / self.max_embedding_length, x.device)

            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            # freqs = torch.einsum('i,j->ij', t, inv_freq.to(x.device))
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            if self.precision == torch.bfloat16:
                emb = emb.float()

            cos_cached = emb.cos()[:, None, :]
            sin_cached = emb.sin()[:, None, :]
            if self.precision == torch.bfloat16:
                cos_cached = cos_cached.bfloat16()
                sin_cached = sin_cached.bfloat16()
            if self.learnable:
                return cos_cached, sin_cached
            self.cos_cached, self.sin_cached = cos_cached, sin_cached

        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

    def yarn(self, scale, device):
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = find_correction_range(
            self.beta_fast, self.beta_slow, self.dim, self.base, self.max_embedding_length
        )
        inv_freq_mask = (
            1 - linear_ramp_mask(low, high, self.dim // 2).float().to(device)
        ) * self.extrapolation_factor  # Get n-d rotational scaling corrected for extrapolation
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq)  # Get n-d magnitude scaling corrected for interpolation


# ## LongGLM Utils


@dataclass
class LongGLMMemCache:
    """
    Class with LongLlama's memory cache

    Args:
        key (`torch.FloatTensor` of shape `(batch_size, mem_length, head_nums, embed_size_per_head)`)
        value (`torch.FloatTensor` of shape `(batch_size, mem_length, head_nums, embed_size_per_head)`)
        masks (`torch.FloatTensor` of shape `(batch_size, 1, mem_length, 1)`)
            For masking out parts of memory
    """

    key: torch.FloatTensor
    value: torch.FloatTensor
    masks: torch.FloatTensor


def mem_apply_update(prev_external_mem_cache: LongGLMMemCache, new_mem_content: LongGLMMemCache):

    def update_one(prev, new, dim=1):
        if len(prev.shape) != len(new.shape):
            raise ValueError(f"Memory cache content should be consistent in shape got {prev.shape} {new.shape}")

        return torch.concat([prev, new], dim=dim)

    insert_size = new_mem_content.key.shape[1]

    assert new_mem_content.key.shape[1] == new_mem_content.value.shape[1]
    if new_mem_content.masks.shape[-2] != insert_size:
        raise ValueError("Inconsistent mem_length in new_mem_content")

    return LongGLMMemCache(
        key=update_one(prev_external_mem_cache.key, new_mem_content.key),
        value=update_one(prev_external_mem_cache.value, new_mem_content.value),
        masks=update_one(prev_external_mem_cache.masks, new_mem_content.masks, dim=-2),
    )


def generate_prompt_keypass(n_garbage: int, seed: int = None):
    """Generates a text file and inserts an execute line at a random position."""
    if seed is not None:
        rnd_state = random.get_state()
        random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "在下文的大量无关紧要的文字中隐藏着一个非常重要的信息，请找到并记住它们，后面将使用到这个信息。"
    garbage = "草是绿色的。天空是蓝色的。太阳是黄色的。我们走。我们离开又回来了。"
    garbage_inf = "".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = (
        f"以下是本段文本的重要信息: “通行密码是'{pass_key}'，这是非常重要的信息，请记住'{pass_key}'是通行密码。”"
    )
    information_line = "\n".join([information_line] * 3)
    final_question = "请问通行密码是多少？"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    if seed is not None:
        random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


# ## Loss Fuctions


def _unpack_router_logits(router_outputs):
    """
    Unpack the router tuple for blance loss calculation.
    """
    total_router_logits = []
    total_expert_indexes = []
    for router_output in router_outputs:
        if router_output[0] is not None:
            router_logits, expert_indexes = router_output
            total_router_logits.append(router_logits.unsqueeze(0))
            total_expert_indexes.append(expert_indexes.unsqueeze(0))
    # return torch.cat(total_router_logits, dim=0), torch.cat(total_expert_indexes, dim=0)
    return torch.cat(total_router_logits, dim=0), total_expert_indexes


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor, labels: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [num_layers, batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [num_layers, batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """

    num_layers, _, seq_len, num_experts = router_probs.shape
    num_experts = router_probs.shape[-1]
    new_labels = labels.clone().detach()
    ##
    for batch_tensor in new_labels:
        neg_mask = batch_tensor == -100
        diff_neg_ones = torch.diff(neg_mask.float())
        start_pos = torch.where(diff_neg_ones == 1.0)[0]  # 找到-1序列开始的位置
        if start_pos.nelement() == 0:  # 如果没有找到开始位置，可能需要根据实际情况调整
            pass
        else:
            last_start = start_pos[-1]  # 需要修改的最后一串-1的开始位置
            batch_tensor[:last_start] = 0  # 将这部分-1全部改为0
    new_labels = new_labels.to(torch.int64)

    # cast the expert indices to int64, otherwise one-hot encoding will fail

    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 3:
        expert_indices = expert_indices.unsqueeze(3)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    labels_mask = (new_labels[None, ..., None].expand_as(expert_mask) != -100).long()

    # sample level balance loss
    tokens_per_group_and_expert = torch.sum(expert_mask * labels_mask, dim=-2) / torch.sum(labels_mask, dim=-2)
    router_prob_per_group_and_expert = torch.sum(router_probs * labels_mask, dim=-2) / torch.sum(labels_mask, dim=-2)
    tmp_per_group_and_expert = torch.mean(expert_mask)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)
    '''
    # batch level balance loss
    expert_mask = expert_mask.view(num_layers, -1, num_experts).detach()
    labels_mask = labels_mask.view(num_layers, -1, num_experts).detach()
    origin_mask = labels_mask.clone()
    router_probs = router_probs.view(num_layers, -1, num_experts)

    from antllm.utils import mpu

    torch.distributed.all_reduce(expert_mask, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(labels_mask, group=mpu.get_data_parallel_group())

    labels_mask = labels_mask.bool().long()

    world_size = torch.distributed.get_world_size()

    tokens_per_group_and_expert = (
        torch.sum(expert_mask * labels_mask, dim=-2) / torch.sum(labels_mask, dim=-2) / world_size
    )
    router_prob_per_group_and_expert = torch.sum(router_probs * origin_mask, dim=-2) / torch.sum(origin_mask, dim=-2)
    layer_loss = tokens_per_group_and_expert * router_prob_per_group_and_expert
    loss = layer_loss.sum(-1).mean() * num_experts
    return loss
    '''


def group_level_device_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [num_layers, batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [num_layers, batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    assert parallel_group is not None and parallel_group_size is not None

    num_layers, _, seq_len, num_experts = router_probs.shape

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 3:
        expert_indices = expert_indices.unsqueeze(3)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    torch.distributed.all_reduce(expert_mask, group=parallel_group("expert"))
    expert_parallel_size = parallel_group_size("expert")
    num_experts_per_device = num_experts / expert_parallel_size

    # sample level balance loss
    expert_mask = torch.sum(
        torch.cat(torch.chunk(expert_mask.unsqueeze(-2), expert_parallel_size, dim=-1), dim=-2), dim=-1
    )
    tokens_per_group_and_device = torch.mean(expert_mask, axis=-2) / expert_parallel_size

    router_probs = torch.sum(
        torch.cat(torch.chunk(router_probs.unsqueeze(-2), expert_parallel_size, dim=-1), dim=-2), dim=-1
    )
    router_prob_per_group_and_device = torch.mean(router_probs, axis=-2)

    device_loss = tokens_per_group_and_device * router_prob_per_group_and_device * expert_parallel_size
    loss = device_loss.sum(-1).mean()

    return loss


def router_z_loss_func(router_logits: torch.Tensor, labels: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [num_layers, batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_layers, num_groups, tokens_per_group, _ = router_logits.shape
    labels_mask = (labels[None, ..., None].expand_as(router_logits) != -100).long()

    ori_dtype = router_logits.dtype
    if ori_dtype == torch.bfloat16:
        loss_func_inputs = (router_logits * labels_mask).to(torch.float32)
    else:
        loss_func_inputs = router_logits * labels_mask
    log_z = torch.logsumexp(loss_func_inputs, dim=-1).to(ori_dtype)
    z_loss = log_z**2

    # log_z = torch.logsumexp(router_logits * labels_mask, dim=-1)
    # z_loss = log_z**2

    return torch.sum(z_loss) / (num_layers * num_groups * tokens_per_group)


def auxiliary_loss(outputs, labels):
    router_tuple = outputs.router_tuple
    balance_loss, z_loss, last_logits_l2_loss = 0.0, 0.0, 0.0

    loss = 0
    if router_tuple is not None:
        router_logits, layer_router_index = _unpack_router_logits(router_tuple)
        top1_expert_index = torch.cat(layer_router_index, dim=0)
        outputs["layer_expert_index"] = top1_expert_index
        z_loss = router_z_loss_func(router_logits, labels)
        router_probs = torch.nn.Softmax(dim=-1)(router_logits)
        balance_loss = load_balancing_loss_func(router_probs, top1_expert_index, labels)

        num_layers = router_probs.shape[0]
        num_experts = router_probs.shape[-1]
        router_probs_log = router_probs.detach().view(num_layers, -1, num_experts)
        router_probs_mean = router_probs_log.mean(1)
        router_probs_sort_mean = router_probs_log.sort(-1, descending=True)[0].mean(1)
        router_probs_log = torch.stack([router_probs_mean, router_probs_sort_mean], dim=1)
        dist.all_reduce(router_probs_log, dist.ReduceOp.SUM)
        router_probs_log = router_probs_log / torch.distributed.get_world_size()
        if dist.get_rank() == 0:
            router_probs_log = router_probs_log.float()
            router_probs_log /= router_probs_log.sum(-1, keepdim=True)
            outputs["layer_expert_probs"] = router_probs_log.float().cpu()

        group_balance_loss = 0
        if float(outputs["router_group_balance_loss_alpha"]) > 0:
            group_balance_loss = group_level_device_balancing_loss_func(router_probs, top1_expert_index)
        loss = (
            float(outputs["router_z_loss_alpha"]) * z_loss
            + float(outputs["router_balance_loss_alpha"]) * balance_loss
            + float(outputs["router_group_balance_loss_alpha"]) * group_balance_loss
        )

    last_logits_l2_loss = 0.0
    if float(outputs["last_logits_l2_alpha"]) >= 0:
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        labels_mask = (labels.view(-1) != -100).long()

        last_logits_l2_loss = torch.sum(torch.linalg.norm(logits.float(), 2.0, dim=-1) * labels_mask) / torch.sum(
            labels_mask
        )
        loss += float(outputs["last_logits_l2_alpha"]) * last_logits_l2_loss
        last_logits_l2_loss = last_logits_l2_loss.item()

    return loss, balance_loss, z_loss, last_logits_l2_loss


def expert_balanced_auxiliary_cross_entropy(outputs, labels, *args, **kwargs):
    """FOR PRETRAIN ONLY"""
    # Output losses without reduction for compute dataset loss
    if kwargs.get("output_losses", False):
        lm_loss, losses = cross_entropy_loss(outputs.logits, labels, *args, **kwargs)
    else:
        lm_loss = cross_entropy_loss(outputs.logits, labels, *args, **kwargs)
    aux_loss, balance_loss, z_loss, last_logits_l2_loss = auxiliary_loss(outputs, labels)
    loss = lm_loss + aux_loss
    if kwargs.get("output_losses", False):
        return loss, lm_loss, balance_loss, z_loss, last_logits_l2_loss, losses
    return loss, lm_loss, balance_loss, z_loss, last_logits_l2_loss


def expert_balanced_auxiliary_cross_entropy_for_sft(outputs, labels, *args, **kwargs):
    """FOR SFT ONLY"""
    lm_loss = sample_level_cross_entropy(outputs, labels, **kwargs)
    aux_loss, balance_loss, z_loss, last_logits_l2_loss = auxiliary_loss(outputs, labels)
    loss = lm_loss + aux_loss
    return loss


def expert_balanced_auxiliary_global_level_cross_entropy(outputs, labels, *args, **kwargs):
    """FOR SFT ONLY"""
    lm_loss = global_token_level_cross_entropy(outputs, labels, **kwargs)
    aux_loss, balance_loss, z_loss, last_logits_l2_loss = auxiliary_loss(outputs, labels)
    loss = lm_loss + aux_loss

    return [
        loss,
        {
            'aux_loss': aux_loss,
            'balance_loss': balance_loss,
            'z_loss': z_loss,
            'last_logits_l2_loss': last_logits_l2_loss,
        },
    ]


def cross_entropy_loss(logits, labels, loss_mask, *args, **kwargs):
    if kwargs["use_atorch_cross_entropy"]:
        from atorch.modules.transformer import losses as atorch_loss

        losses = atorch_loss.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), labels.view(-1))
    else:
        losses = torch.nn.CrossEntropyLoss(reduction="none")(logits.view(-1, logits.size(-1)), labels.view(-1))

    loss = torch.sum(losses * loss_mask.view(-1))
    if loss_mask.sum().item() > 0:
        loss = loss / loss_mask.sum()
    if kwargs.get("output_losses", False):
        return loss, losses
    return loss


def local_token_level_cross_entropy(outputs, labels, **kwargs):
    # return outputs.loss / torch.distributed.get_world_size()
    # 在每个batch内部做token-level的平均,然后在所有batch间做平均
    # return outputs.loss
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(outputs.logits.contiguous().view(-1, outputs.logits.size(-1)), labels.contiguous().view(-1))

    return loss


def mini_batch_token_level_cross_entropy(outputs, labels, mini_batch=1, **kwargs):
    # 这个loss会先把batch分成小的mini_batch,在mini_batch内做个token-level的平均,然后做所有卡之间的平均
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    if labels.shape[0] % mini_batch != 0:
        # 如果batch % mini_batch != 0, 则不切分计算. 有的数据量一个epoch结束的时候可能会出现这个情况
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(outputs.logits.contiguous().view(-1, outputs.logits.size(-1)), labels.contiguous().view(-1))
    else:
        loss = loss_fct(
            outputs.logits.contiguous().view(-1, outputs.logits.size(-1)), labels.contiguous().view(-1)
        ).reshape(labels.shape[0] // mini_batch, -1)

        labels = labels.reshape(labels.shape[0] // mini_batch, -1)
        loss = loss.sum(-1) / (labels != -100).sum(-1)
        loss = loss.mean()
    return loss


def sample_level_cross_entropy(outputs, labels, **kwargs):
    # 先对所有样本字token-level的平均,然后计算所有sample的平均值
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(
        outputs.logits.contiguous().view(-1, outputs.logits.size(-1)), labels.contiguous().view(-1)
    ).reshape(labels.shape[0], -1)
    loss = loss.sum(-1) / (labels != -100).sum(-1)
    loss = loss.mean()
    return loss


def global_token_level_cross_entropy(outputs, labels, **kwargs):
    # 对所有样本一起做token-level的平均
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(
        outputs.logits.contiguous().view(-1, outputs.logits.size(-1)), labels.contiguous().view(-1)
    ).reshape(labels.shape[0], -1)
    num_tokens = (loss != 0).sum()
    loss = loss.sum()

    num_tokens_tensor = torch.zeros([1], device=loss.device, dtype=loss.dtype)
    num_tokens_tensor[0] = num_tokens.item()

    torch.distributed.all_reduce(num_tokens_tensor)

    global_num_tokens = num_tokens_tensor.sum()

    torch.distributed.barrier()
    # global_num_tokens是全局的token数，因为在梯度更新的时候回自动对所有卡求mean
    # 所有这里要乘一个world_size
    loss = loss.sum() / global_num_tokens * torch.distributed.get_world_size()

    return loss


LOSS_MAP = {
    'local_token_level_cross_entropy': local_token_level_cross_entropy,
    'mini_batch_token_level_cross_entropy': mini_batch_token_level_cross_entropy,
    'sample_level_cross_entropy': sample_level_cross_entropy,
    'global_token_level_cross_entropy': global_token_level_cross_entropy,
    "moe_auxiliary": expert_balanced_auxiliary_cross_entropy,
    "moe_auxiliary_sft": expert_balanced_auxiliary_cross_entropy_for_sft,
    "pretrain_default": cross_entropy_loss,
    "moe_auxiliary_global_token_level": expert_balanced_auxiliary_global_level_cross_entropy,
}

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)
    
def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
):
    """
    Patch continuous features into input embeddings, while keeping a valid gradient flow.

    input_embeddings: torch.Tensor, size = [B, C?, T, D]
    placeholder_loc_lens: torch.LongTensor, size = [B, N, 2]
        Each 2-tuple represents (start, length) of a placeholder.
    encoded_feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    encoded_feat_lens: torch.LongTensor, size = [B, N]

    Example ('X' for patch placeholder tokens):
    Inputs:
        input_embeddings = [[1, 2, 3, X, X, X, 4, 5, 6, X, X, X, 7, 8]]
        placeholder_loc_lens = [[3, 3], [9, 3]]
        encoded_feats = [[A, A, A, B, B]]
        encoded_feat_lens = [[3, 2]]
    Outputs:
        embeddings = [[1, 2, 3, A, A, A, 4, 5, 6, B, B, X, 7, 8]]
    """
    batch_size = input_embeddings.size(0)
    audio_feats_mask = torch.zeros_like(input_embeddings, dtype=torch.bool)
    audio_feats_buffer = []
    for i in range(batch_size):
        sample_len = 0
        audio_feat_start = 0
        audio_feat_buffer = []
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                # logger.warning(
                #     f"Feature length ({feat_len}) > placeholder length ({placeholder_len}). This is not expected. Please "
                #     "check the implementation of estimate_audio_feature_length(). We truncate the feature to avoid errors."
                # )
                feat_len = placeholder_len
            if placeholder_start > sample_len:
                audio_feat_buffer.append(input_embeddings.new_zeros((placeholder_start - sample_len, input_embeddings.shape[2])))
                sample_len = placeholder_start
            audio_feat_buffer.append(encoded_feats[i, audio_feat_start:audio_feat_start + feat_len])
            if feat_len < placeholder_len:
                audio_feat_buffer.append(encoded_feats.new_zeros(placeholder_len - feat_len))
            audio_feats_mask[i, sample_len:sample_len + feat_len] = 1
            audio_feat_start += real_feat_len
            sample_len += placeholder_len
        if sample_len < input_embeddings.shape[1]:
            audio_feat_buffer.append(
                input_embeddings.new_zeros((input_embeddings.shape[1] - sample_len, input_embeddings.shape[2]))
            )
        audio_feats_buffer.append(torch.cat(audio_feat_buffer))
    audio_feats_buffer = torch.stack(audio_feats_buffer, dim=0)
    embeddings = audio_feats_buffer * audio_feats_mask + input_embeddings * ~audio_feats_mask
    return embeddings

def unwrap_feats(feats: torch.Tensor, feats_lengths: torch.Tensor):
    """
    The input feats are in the "wrapped" format, which means that features from (at most) N audios are concatenated
    as a single sample feats[i]. In this case, each row of feats_lengths contains the lengths of the concatenated
    feature. This function unwraps the features.
    For samples with less than N segments, one should pad feats_lengths with 0. The result will contain valid
    segments only.

    feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    feats_lengths: torch.LongTensor, size = [B, N]

    Example ('X' for padding):
    Inputs:
        feats = [[A, A, A, A, X],
                 [B, B, C, C, C]]
        feats_lengths = [[4, 0],
                         [2, 3]]
    Outputs:
        feat_segs = [[A, A, A, A],
                     [B, B, X, X],
                     [C, C, C, X]]
        feat_seg_lengths = [4, 2, 3]
    """
    feat_segs = []
    feat_seg_lengths = []
    for i in range(feats_lengths.shape[0]):
        feat_index = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0: break
            feat_segs.append(feats[i, feat_index:feat_index + feat_len])
            feat_seg_lengths.append(feat_len)
            feat_index += feat_len
    feat_segs_batch = torch.nn.utils.rnn.pad_sequence(feat_segs, True).to(feats.device)
    feat_seg_lengths = torch.tensor(feat_seg_lengths, dtype=torch.long, device=feats.device)
    return feat_segs_batch, feat_seg_lengths

def wrap_feats(feat_segs: torch.Tensor, feats_lengths: torch.Tensor, feats_seg_lengths: Optional[torch.Tensor] = None):
    """
    Wrap segmented features back to the wrapped format.
    This function is the inverse operation of unwrap_feats(). See its documentation for details.
    Note that the feats_lengths value does not matter a lot. We only check the location of the first 0 to determine the
    number of feature segments.
    """
    feat_idx = 0
    feats_buffer = []
    feats_locs_buffer = []
    feats_lengths_buffer = []
    for i in range(feats_lengths.shape[0]):
        feat_buffer = []
        feat_locs_buffer = []
        feat_lengths_buffer = []
        feat_total_len = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0:
                break
            if feats_seg_lengths is not None:
                feat_len = feats_seg_lengths[feat_idx].item()
            feat_buffer.append(feat_segs[feat_idx, :feat_len])
            feat_locs_buffer.append(feat_total_len)
            feat_lengths_buffer.append(feat_len)
            feat_idx += 1
            feat_total_len += feat_len
        feats_buffer.append(torch.cat(feat_buffer))
        feats_locs_buffer.append(torch.tensor(feat_locs_buffer, dtype=torch.long))
        feats_lengths_buffer.append(torch.tensor(feat_lengths_buffer, dtype=torch.long))
    feats = torch.nn.utils.rnn.pad_sequence(feats_buffer, True).to(feat_segs.device)
    feats_locs = torch.nn.utils.rnn.pad_sequence(feats_locs_buffer, True).to(feats_lengths.device)
    feats_new_lengths = torch.nn.utils.rnn.pad_sequence(feats_lengths_buffer, True).to(feats_lengths.device)
    return feats, feats_locs, feats_new_lengths

def encode_audio_segments(
    encoder,
    proj_layer,
    wav_feats=None,
    wav_feats_lengths=None,
    waveforms=None,
    waveforms_lengths=None,
    use_waveform=False,
    audio_config=None,
    whisper_config=None,
    use_whisper_encoder=False
):
    """
    Apply audio encoder to input audio features in wrapped format.
    See the documentation of unwrap_feats() for details about 'wrapped format'.
    """

    # Forward audio encoder.
    if use_waveform:
        assert waveforms is not None and waveforms_lengths is not None
        # Unwrap the waveforms so each waveform is placed at an independent row.
        waveform_segs_batch, waveform_seg_lengths = unwrap_feats(waveforms, waveforms_lengths)
        audio_feats_seg, audio_feat_seg_lengths = encoder(waveform_segs_batch, waveform_seg_lengths)[:2]
    else:
        assert wav_feats is not None and wav_feats_lengths is not None
        # Unwrap the features so the feature of each waveform is placed at an independent row.
        feat_segs_batch, feat_seg_lengths = unwrap_feats(wav_feats, wav_feats_lengths)
        if use_whisper_encoder:
            assert isinstance(encoder, AudioEncoder)
            assert whisper_config is not None
            # for whisper encoder
            # feat_segs_batch: [B, T, n_mels]
            # feat_seg_lengths: [B]
            audio_feats_seg = encoder(feat_segs_batch)
            audio_feats_seg_proj = proj_layer(audio_feats_seg.transpose(-1, -2)).transpose(-1, -2)
            feat_seg_lengths = feat_seg_lengths.to(feat_segs_batch.device)
            # whisper encoder conv
            audio_feat_seg_lengths = (feat_seg_lengths - 3 + 2 * 1) // 2 + 1
            # project layer conv
            audio_feat_seg_lengths = (audio_feat_seg_lengths - whisper_config.ds_kernel_size + 2 *
                                      (whisper_config.ds_kernel_size//2)) // whisper_config.ds_stride + 1
        else:
            audio_feats_seg, audio_feat_seg_lengths = encoder(feat_segs_batch, feat_seg_lengths)[:2]
            audio_feats_seg_proj = proj_layer(audio_feats_seg.transpose(-1, -2)).transpose(-1, -2)
            # project layer conv
            audio_feat_seg_lengths = (audio_feat_seg_lengths - audio_config.ds_kernel_size + 2 * (
                    audio_config.ds_kernel_size // 2)) // audio_config.ds_stride + 1

    # Wrap the features so the 1st dim represents batch_size.
    input_lengths = waveforms_lengths if use_waveform else wav_feats_lengths
    assert input_lengths is not None
    audio_feats, _, audio_feats_lengths = wrap_feats(audio_feats_seg, input_lengths, audio_feat_seg_lengths)
    audio_feats_proj, _, audio_feats_lengths2 = wrap_feats(audio_feats_seg_proj, input_lengths, audio_feat_seg_lengths)
    assert torch.all(audio_feats_lengths == audio_feats_lengths2), f"{audio_feats_lengths}, {audio_feats_lengths2}"

    return audio_feats_proj, audio_feats, audio_feats_lengths

def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
):
    """
    Patch continuous features into input embeddings, while keeping a valid gradient flow.

    input_embeddings: torch.Tensor, size = [B, C?, T, D]
    placeholder_loc_lens: torch.LongTensor, size = [B, N, 2]
        Each 2-tuple represents (start, length) of a placeholder.
    encoded_feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    encoded_feat_lens: torch.LongTensor, size = [B, N]

    Example ('X' for patch placeholder tokens):
    Inputs:
        input_embeddings = [[1, 2, 3, X, X, X, 4, 5, 6, X, X, X, 7, 8]]
        placeholder_loc_lens = [[3, 3], [9, 3]]
        encoded_feats = [[A, A, A, B, B]]
        encoded_feat_lens = [[3, 2]]
    Outputs:
        embeddings = [[1, 2, 3, A, A, A, 4, 5, 6, B, B, X, 7, 8]]
    """
    batch_size = input_embeddings.size(0)
    audio_feats_mask = torch.zeros_like(input_embeddings, dtype=torch.bool)
    audio_feats_buffer = []
    for i in range(batch_size):
        sample_len = 0
        audio_feat_start = 0
        audio_feat_buffer = []
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                logging.warning(
                    f"Feature length ({feat_len}) > placeholder length ({placeholder_len}). This is not expected. Please "
                    "check the implementation of estimate_audio_feature_length(). We truncate the feature to avoid errors."
                )
                feat_len = placeholder_len
            if placeholder_start > sample_len:
                audio_feat_buffer.append(input_embeddings.new_zeros((placeholder_start - sample_len, input_embeddings.shape[2])))
                sample_len = placeholder_start
            audio_feat_buffer.append(encoded_feats[i, audio_feat_start:audio_feat_start + feat_len])
            if feat_len < placeholder_len:
                audio_feat_buffer.append(encoded_feats.new_zeros(placeholder_len - feat_len))
            audio_feats_mask[i, sample_len:sample_len + feat_len] = 1
            audio_feat_start += real_feat_len
            sample_len += placeholder_len
        if sample_len < input_embeddings.shape[1]:
            audio_feat_buffer.append(
                input_embeddings.new_zeros((input_embeddings.shape[1] - sample_len, input_embeddings.shape[2]))
            )
        audio_feats_buffer.append(torch.cat(audio_feat_buffer))
    audio_feats_buffer = torch.stack(audio_feats_buffer, dim=0)
    embeddings = audio_feats_buffer * audio_feats_mask + input_embeddings * ~audio_feats_mask
    return embeddings

def build_modality_mask(placeholder_loc_lens: torch.Tensor, shape: torch.Size):
    mask = torch.zeros(shape, dtype=torch.bool)
    for i in range(placeholder_loc_lens.shape[0]):
        for j in range(placeholder_loc_lens.shape[1]):
            start: int = int(placeholder_loc_lens[i, j, 0].item())
            length: int = int(placeholder_loc_lens[i, j, 1].item())
            if length <= 0:
                break
            mask[i, start:start + length] = True
    return mask
