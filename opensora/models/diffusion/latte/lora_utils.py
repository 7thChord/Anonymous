from importlib import import_module

import numpy as np
from typing import Any, Dict, Optional, Tuple, Callable, Union, Iterable
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, is_xformers_available
from diffusers.models.lora import LoRACompatibleConv,LoRACompatibleLinear

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero
from diffusers.models.attention_processor import SpatialNorm, LORA_ATTENTION_PROCESSORS, \
    CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0, \
    AttnAddedKVProcessor, AttnAddedKVProcessor2_0, SlicedAttnAddedKVProcessor, XFormersAttnAddedKVProcessor, \
    LoRAAttnAddedKVProcessor, LoRAXFormersAttnProcessor, XFormersAttnProcessor, LoRAAttnProcessor2_0, LoRAAttnProcessor, \
    AttnProcessor, SlicedAttnProcessor, logger
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU

from dataclasses import dataclass
import einops
from einops import rearrange, repeat
from opensora.models.diffusion.latte.attention import Attention
from opensora.models.diffusion.utils.pos_embed import get_2d_sincos_pos_embed, RoPE1D, RoPE2D, LinearScalingRoPE2D, LinearScalingRoPE1D
import math

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
class LoRALinearLayer(nn.Module):
    r"""
    A linear layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        device (`torch.device`, `optional`, defaults to `None`):
            The device to use for the layer's weights.
        dtype (`torch.dtype`, `optional`, defaults to `None`):
            The dtype to use for the layer's weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        network_alpha: Optional[float] = None,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        # deprecation_message = "Use of `LoRALinearLayer` is deprecated. Please switch to PEFT backend by installing PEFT: `pip install peft`."
        # deprecate("LoRALinearLayer", "1.0.0", deprecation_message)

        self.down = nn.Linear(in_features, rank, bias=False, device=device)
        self.up = nn.Linear(rank, out_features, bias=False, device=device)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank
        self.out_features = out_features
        self.in_features = in_features

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.down.to(hidden_states.device)
        self.up.to(hidden_states.device)
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype
        # import ipdb;ipdb.set_trace()
        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRAAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, dim=1152, attention_mode='xformers', use_rope=False, rope_scaling=None, compress_kv_factor=None,out_dim=None,device=None,lora_rank=32,num_adapters=1,adapter_weights:list = [1]  ):
        super().__init__()
        self.dim = dim
        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.compress_kv_factor = compress_kv_factor
        self.num_adapters = num_adapters
        self.adapter_weights = adapter_weights
        if self.use_rope:
            self._init_rope()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        if num_adapters > 1:
            self.q_loras =torch.nn.ModuleList([])
            self.k_loras =torch.nn.ModuleList([])
            self.v_loras =torch.nn.ModuleList([])
            self.out_loras =torch.nn.ModuleList([])
            for i in range(num_adapters):
                self.q_loras.append(LoRALinearLayer(dim,dim,rank = lora_rank,device = device))
                self.k_loras.append(LoRALinearLayer(out_dim if out_dim else dim ,dim,rank = lora_rank,device = device))
                self.v_loras.append(LoRALinearLayer(out_dim if out_dim else dim,dim,rank = lora_rank, device = device))
                self.out_loras.append(LoRALinearLayer(dim,dim,rank = lora_rank,device = device))
        else:
            self.q_lora = LoRALinearLayer(dim,dim,rank = lora_rank,device = device)
            self.k_lora = LoRALinearLayer(out_dim if out_dim else dim ,dim,rank = lora_rank,device = device)
            self.v_lora = LoRALinearLayer(out_dim if out_dim else dim ,dim,rank = lora_rank,device = device)
            self.out_lora = LoRALinearLayer(dim,dim,rank = lora_rank,device = device)

    def _init_rope(self):
        if self.rope_scaling is None:
            self.rope2d = RoPE2D()
            self.rope1d = RoPE1D()
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor_2d = self.rope_scaling["factor_2d"]
            scaling_factor_1d = self.rope_scaling["factor_1d"]
            if scaling_type == "linear":
                self.rope2d = LinearScalingRoPE2D(scaling_factor=scaling_factor_2d)
                self.rope1d = LinearScalingRoPE1D(scaling_factor=scaling_factor_1d)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
            
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        position_q: Optional[torch.LongTensor] = None,
        position_k: Optional[torch.LongTensor] = None,
        last_shape: Tuple[int] = None, 
    ) -> torch.FloatTensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)



        if self.compress_kv_factor is not None:
            batch_size = hidden_states.shape[0]
            if len(last_shape) == 2:
                encoder_hidden_states = hidden_states.permute(0, 2, 1).reshape(batch_size, self.dim, *last_shape)
                encoder_hidden_states = attn.sr(encoder_hidden_states).reshape(batch_size, self.dim, -1).permute(0, 2, 1)
            elif len(last_shape) == 1:
                encoder_hidden_states = hidden_states.permute(0, 2, 1)
                if last_shape[0] % 2 == 1:
                    first_frame_pad = encoder_hidden_states[:, :, :1].repeat((1, 1, attn.kernel_size - 1))
                    encoder_hidden_states = torch.concatenate((first_frame_pad, encoder_hidden_states), dim=2)
                encoder_hidden_states = attn.sr(encoder_hidden_states).permute(0, 2, 1)
            else:
                raise NotImplementedError(f'NotImplementedError with last_shape {last_shape}')
                
            encoder_hidden_states = attn.norm(encoder_hidden_states)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)
        if self.num_adapters <= 1:
            query += self.q_lora(hidden_states)
        else:
            for i in range(self.num_adapters):
                query += self.q_loras[i](hidden_states) * self.adapter_weights[i]
            
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        if self.num_adapters <= 1:
            key +=  self.k_lora(encoder_hidden_states)
            value += self.v_lora(encoder_hidden_states)
        else:
            for i in range(self.num_adapters):
                key += self.k_loras[i](encoder_hidden_states) * self.adapter_weights[i]
                value += self.v_loras[i](encoder_hidden_states) * self.adapter_weights[i]


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.use_rope:
            # require the shape of (batch_size x nheads x ntokens x dim)
            if position_q.ndim == 3:
                query = self.rope2d(query, position_q) 
            elif position_q.ndim == 2:
                query = self.rope1d(query, position_q) 
            else:
                raise NotImplementedError
            if position_k.ndim == 3:
                key = self.rope2d(key, position_k)
            elif position_k.ndim == 2:
                key = self.rope1d(key, position_k)
            else:
                raise NotImplementedError

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.attention_mode == 'flash':
            assert attention_mask is None or torch.all(attention_mask.bool()), 'flash-attn do not support attention_mask'
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'xformers':
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
        elif self.attention_mode == 'math':
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            raise NotImplementedError(f'Found attention_mode: {self.attention_mode}')
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        prev_hidden_states = hidden_states
        hidden_states = attn.to_out[0](hidden_states, *args)
        if self.num_adapters <= 1:
            hidden_states += self.out_lora(prev_hidden_states)
        else:
            for i in range(self.num_adapters):
                hidden_states += self.out_loras[i](prev_hidden_states) * self.adapter_weights[i]        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states








class LoRASparseCausalAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size=None,
        cross_attention_dim=None,
        lora_rank:int=32,
        lora_scale:float=1.0):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        if self.lora_rank is not None and self.lora_scale is not None:
            self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        video_length = None,
    ) -> torch.FloatTensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if True else (scale,) # if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states,  *args) if self.lora_rank is None and self.lora_scale is None else self.to_q_lora(hidden_states) + attn.to_q(hidden_states, *args)


        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args) if self.lora_rank is None and self.lora_scale is None else self.to_k_lora(hidden_states) + attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args) if self.lora_rank is None and self.lora_scale is None else self.to_v_lora(hidden_states) + attn.to_v(encoder_hidden_states, *args) 
        

        # SparseCausalAttn
        # import ipdb; ipdb.set_trace()
        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = key[:,  [0] * video_length]
        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = value[:,  [0] * video_length]
        value = rearrange(value, "b f d c -> (b f) d c")

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # SparseCausalAttn
        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args) if self.lora_rank is None and self.lora_scale is None else self.to_out_lora(hidden_states) + attn.to_out[0](hidden_states, *args)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

    