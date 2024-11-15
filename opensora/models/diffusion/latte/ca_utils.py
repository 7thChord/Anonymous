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
from opensora.models.diffusion.latte.lora_utils import LoRALinearLayer
from opensora.models.diffusion.latte.attention import Attention
from opensora.models.diffusion.utils.pos_embed import get_2d_sincos_pos_embed, RoPE1D, RoPE2D, LinearScalingRoPE2D, LinearScalingRoPE1D
from .pe import PositionalEncoding, PositionalEncoding_Rotary

import math

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class LoRACAAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, dim=1152, attention_mode='xformers', use_rope=False, 
                 rope_scaling=None, compress_kv_factor=None,out_dim=None,
                 device=None,lora_rank=32,
                 num_adapters=1,
                 q_downsample:bool = False,
                 q_downsample_ratio:int = 1,
                 ca_pe_mode:str = None, # None / "naive" / "temporal" / "temporal_sine" / "temporal_rope"
                 adapter_weights:list=[1],
                 video_length:int=5,
                 use_q_lora:bool = False,
                 ):
        super().__init__()
        self.dim = dim
        self.attention_mode = attention_mode
        self.use_rope = use_rope
        self.rope_scaling = rope_scaling
        self.compress_kv_factor = compress_kv_factor
        self.num_adapters = num_adapters
        self.adapter_weights = adapter_weights
        self.lora_rank = lora_rank
        self.video_length = video_length
        if self.use_rope:
            self._init_rope()

        self.q_downsample_ratio = q_downsample_ratio
        self.q_downsample = q_downsample
        self.q_downsample_pe = False
        self.use_q_lora = use_q_lora

        self.ca_pe_mode = ca_pe_mode
        if ca_pe_mode is not None:
            self.temporal_position_encoding_max_len = video_length if 'temporal' in ca_pe_mode or 'rope' in ca_pe_mode else 2560
            PE_class = PositionalEncoding_Rotary if 'rope' in ca_pe_mode else PositionalEncoding 
            self.pos_encoder = PE_class(
                    self.dim,
                    dropout=0., 
                    max_len=self.temporal_position_encoding_max_len,
                    mode=ca_pe_mode,
                )
            print("PE for Q:", ca_pe_mode)

            if 'rope' in ca_pe_mode and 'K' in ca_pe_mode:
                self.pos_encoder_kv = PositionalEncoding_Rotary(
                    dim,
                    dropout=0.,
                    max_len=self.temporal_position_encoding_max_len,
                    mode="1d",
                )
                print("PE for K/V:", ca_pe_mode)
            else:
                self.pos_encoder_kv = None
        else:
            self.pos_encoder = None
            self.pos_encoder_kv = None

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        if isinstance(lora_rank, Iterable):
            # can be [0] with num_adapter > 1. If so, then we apply multi-CA still, without using LoRAs
            self.multi_ca_inference = True
            self.zero_lora = lora_rank[0] == 0
        else:
            self.multi_ca_inference = False
        
        if num_adapters > 1:
            self.to_k_loras = nn.ModuleList([
                    LoRALinearLayer(out_dim if out_dim else dim, dim, self.lora_rank, device = device)
                    for i in range(num_adapters)])
            self.to_v_loras = nn.ModuleList([
                    LoRALinearLayer(out_dim if out_dim else dim, dim, self.lora_rank, device = device)
                    for i in range(num_adapters)])
            self.to_out_loras = nn.ModuleList([
                    LoRALinearLayer(dim, dim, self.lora_rank, device = device)
                    for i in range(num_adapters)])
            if use_q_lora:
                self.to_q_loras = nn.ModuleList([
                    LoRALinearLayer(dim, dim, self.lora_rank, device = device)
                    for i in range(num_adapters)])
            
            # self.k_loras = torch.nn.ModuleList([])
            # self.v_loras = torch.nn.ModuleList([])
            # self.out_loras = torch.nn.ModuleList([])
            # for i in range(num_adapters):
            #     self.k_loras.append(LoRALinearLayer(out_dim if out_dim else dim ,dim, rank = self.lora_rank,device = device))
            #     self.v_loras.append(LoRALinearLayer(out_dim if out_dim else dim,dim,rank = self.lora_rank, device = device))
            #     self.out_loras.append(LoRALinearLayer(dim,dim,rank = self.lora_rank,device = device))
        else:
            self.k_lora = LoRALinearLayer(out_dim if out_dim else dim, dim, rank = self.lora_rank, device = device)
            self.v_lora = LoRALinearLayer(out_dim if out_dim else dim, dim, rank = self.lora_rank, device = device)
            self.out_lora = LoRALinearLayer(dim, dim, rank = self.lora_rank, device = device)
            if use_q_lora:
                self.q_lora = LoRALinearLayer(dim, dim, rank = self.lora_rank, device = device)

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

        q_downsample = self.q_downsample
        cross_attention = encoder_hidden_states is not None
        multi_ca = self.multi_ca_inference and self.num_adapters > 1

        args = () if USE_PEFT_BACKEND else (scale,)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        batch_size = hidden_states.shape[0] // self.video_length if q_downsample else hidden_states.shape[0]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        
        if self.compress_kv_factor is not None:
            # from class AttnProcessor2_0
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

        _, sequence_length, _ = (
            hidden_states.shape if not cross_attention else encoder_hidden_states.shape
        )


        def repeat_(t: torch.Tensor, video_length):
            # e.g. "C U" => "CCCC UUUU" for 1st dim (f=4)
            return einops.repeat(t, 'b n c -> (b f) n c', f=self.video_length)
        
        # The Q being spatiotemporal, downsampled spatially (by `q_downsample_ratio`)
        video_length = self.video_length
        height = int(math.sqrt(hidden_states.shape[1]/1.6))
        width = int(1.6 * height)
        if q_downsample:
            # hidden_states shape: [5, 640, 1152]
            # encoder_hidden_states shape: [5, 300, 1152]
            hidden_states = rearrange(hidden_states,"(b f) d c -> b c f d", f = video_length)
            hidden_states = rearrange(hidden_states, "b c f (h w) -> b c f h w", h = height, w = width)

            target_size = (video_length, height//self.q_downsample_ratio, width//self.q_downsample_ratio)
            hidden_states = torch.nn.functional.interpolate(hidden_states, target_size).to(encoder_hidden_states.device)
            
            # add PE (non-"ropeQ" style)
            if self.ca_pe_mode is not None and 'ropeQ' not in self.ca_pe_mode and '3d' in self.ca_pe_mode:
                hidden_states = rearrange(hidden_states,"b c f h w -> b f h w c")
                hidden_states = self.pos_encoder(hidden_states)
            else:
                hidden_states = rearrange(hidden_states,"b c f h w -> b (f h w) c")

                if self.ca_pe_mode is not None and 'ropeQ' not in self.ca_pe_mode and '3d' not in self.ca_pe_mode:
                    hidden_states = self.pos_encoder(hidden_states)

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args) # + self.q_lora(hidden_states)
        if self.use_q_lora:
            if multi_ca and not self.zero_lora:
                for i in range(self.num_adapters):
                    query += self.to_q_loras[i](hidden_states) * self.adapter_weights[i]
            else:
                query += self.q_lora(hidden_states)

        if q_downsample and self.ca_pe_mode and 'ropeQ' in self.ca_pe_mode and '3d' in self.ca_pe_mode:
            if '3d' in self.ca_pe_mode:
                query = rearrange(hidden_states,"b (f h w) c -> b f h w c", f=video_length, h=target_size[1], w=target_size[2])
            query = self.pos_encoder(query)
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        
        if multi_ca:
            key_c = []
            value_c = []
            do_classifier_free_guidance = encoder_hidden_states.shape[0] == self.num_adapters + 1
            for i in range(self.num_adapters):
                text_emb = encoder_hidden_states[i:i+1] # keep dim
                key = attn.to_k(text_emb, *args)
                value = attn.to_v(text_emb, *args)
                if not self.zero_lora:
                    key += self.to_k_loras[i](text_emb) * self.adapter_weights[i]
                    value += self.to_v_loras[i](text_emb) * self.adapter_weights[i]
                key_c.append(key)
                value_c.append(value)
            key_c = torch.cat(key_c, dim=1) # (1, adapter# x77, d)
            value_c = torch.cat(value_c, dim=1)

            if do_classifier_free_guidance:
                text_emb = encoder_hidden_states[-1:] # keep dim
                key_u = attn.to_k(text_emb, *args) # (1, 77, d)
                value_u = attn.to_v(text_emb, *args)

            # key_c = repeat_(key_c, video_length)
            # value_c = repeat_(value_c, video_length)
            # key_u = repeat_(key_u, video_length)
            # value_u = repeat_(value_u, video_length)

            # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # key_c = key_c.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # value_c = value_c.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # key_u = key_u.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # value_u = value_u.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)

            # hidden_states_c = F.scaled_dot_product_attention(
            #     query[:self.video_length], key_c, value_c, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )
            # hidden_states_u = F.scaled_dot_product_attention(
            #     query[video_length:], key_u, value_u, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )
            # hidden_states = torch.cat([hidden_states_u, hidden_states_c])
            # key_c, value_c shape: 5,600,1152
        
        elif self.num_adapters > 1:
            key = attn.to_k(encoder_hidden_states, *args) 
            value = attn.to_v(encoder_hidden_states, *args) 
            for i in range(self.num_adapters):
                key += self.k_loras[i](encoder_hidden_states) * self.adapter_weights[i]
                value += self.v_loras[i](encoder_hidden_states) * self.adapter_weights[i]
        
        elif self.lora_rank is not None:
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)
            key += self.k_lora(encoder_hidden_states)
            value += self.v_lora(encoder_hidden_states)
        
        else:
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if multi_ca:
            if q_downsample:
                # Qu=Qc=(1,FHW/s,d); KVu=(1,77,d); KVc=(1, adapter# * 77, d)
                equivalent_video_length = 1
                batch_size = 2 if do_classifier_free_guidance else 1
                # no need to repeat, as K V should be (1, 77, d) for STQ+multi-ca case, unless adding PE for K/V
                if self.pos_encoder_kv is not None:
                    # repeat K,V (B, 77, d) => (B, 77F, d)
                    key = einops.repeat(key, 'b n c -> b (n f) c', f=video_length)
                    value = einops.repeat(value, 'b n c -> b (n f) c', f=video_length)
                    # attach PE
                    key = self.pos_encoder_kv(key)
                    if 'V' in self.ca_pe_mode:
                        value = self.pos_encoder_kv(value)
            else:
                # Qu=Qc=(F,HW,d); KVu=(F,77,d); KVc=(F, adapter# * 77, d)
                equivalent_video_length = video_length

                key_c = repeat_(key_c, video_length)
                value_c = repeat_(value_c, video_length)
                if do_classifier_free_guidance:
                    key_u = repeat_(key_u, video_length)
                    value_u = repeat_(value_u, video_length)

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key_c = key_c.view(equivalent_video_length, -1, attn.heads, head_dim).transpose(1, 2)
            value_c = value_c.view(equivalent_video_length, -1, attn.heads, head_dim).transpose(1, 2)

            hidden_states = F.scaled_dot_product_attention(
                query[:equivalent_video_length], key_c, value_c, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            
            if do_classifier_free_guidance:
                key_u = key_u.view(equivalent_video_length, -1, attn.heads, head_dim).transpose(1, 2)
                value_u = value_u.view(equivalent_video_length, -1, attn.heads, head_dim).transpose(1, 2)
                hidden_states_u = F.scaled_dot_product_attention(
                    query[equivalent_video_length:], key_u, value_u, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = torch.cat([hidden_states_u, hidden_states]) # log(240920): bugfix - U must be ahead of C
            # query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            # key_c = key_c.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # key_u = key_u.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # value_c = value_c.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # value_u = value_u.view(video_length, -1, attn.heads, head_dim).transpose(1, 2)
            # hidden_states_c = F.scaled_dot_product_attention(
            #     query[:video_length], key_c, value_c, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )
            # hidden_states_u = F.scaled_dot_product_attention(
            #     query[video_length:], key_u, value_u, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )
            # hidden_states = torch.concat([hidden_states_c,hidden_states_u],dim=0)

        else:
            if q_downsample:
                # Default case: Q(BF, HW, d); repeat K,V (B, 300, d) => (BF, 300, d)
                # log(24.10.7): Below are AnimateDiff codes, but OpenSora should be free of repeat here
                # key = repeat_(key, video_length)
                # value = repeat_(value, video_length)

                # STQ case: Q(B, FHW/s, d); KV(B, 77, d)
                if self.pos_encoder_kv is not None:
                    # repeat K,V (B, 77, d) => (B, 77F, d)
                    key = einops.repeat(key, 'b n c -> b (n f) c', f=video_length)
                    value = einops.repeat(value, 'b n c -> b (n f) c', f=video_length)
                    if attention_mask is not None:
                        attention_mask = einops.repeat(attention_mask, 'b h x l -> b h x (l f)', f=video_length)
                    # attach PE
                    key = self.pos_encoder_kv(key)
                    if 'V' in self.ca_pe_mode:
                        value = self.pos_encoder_kv(value)
                
            # d = n_head * head_dim
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        if self.num_adapters > 1:
            for i in range(self.num_adapters):
                hidden_states += self.out_loras[i](hidden_states) * self.adapter_weights[i]
        else:
            hidden_states += self.out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if q_downsample:
            hidden_states = rearrange(hidden_states,"b (f d) c -> b c f d", f= video_length, d= (height // self.q_downsample_ratio)*(width//self.q_downsample_ratio))
            hidden_states = rearrange(hidden_states,"b c f (h w) -> (b f) c h w", w=width//self.q_downsample_ratio,h=height//self.q_downsample_ratio)
            hidden_states = torch.nn.functional.interpolate(hidden_states,(height,width))
            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length)
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) (h w) c")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
