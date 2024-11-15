import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union

from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm, Attention

from einops import rearrange, repeat
import pdb

from .lora import LoRALinearLayer


class SparseCausalAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, 
        hidden_size=None, 
        use_q_lora: bool = False, 
        use_out_lora: bool = False, 
        lora_rank:int = None, # default rank=32, scale=1.0; for multi-adapter setting, the two should be lists.
        lora_scale:float = None,
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.use_q_lora = use_q_lora
        self.use_out_lora = use_out_lora
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        if self.use_q_lora: 
            self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
        if self.use_out_lora:
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
        query = attn.to_q(hidden_states, *args) 
        if self.use_q_lora: 
            query += self.to_q_lora(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        # SparseCausalAttn
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
        hidden_states = attn.to_out[0](hidden_states, *args)
        if self.use_out_lora:
            hidden_states += self.to_out_lora(hidden_states)
        # dropout
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