from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention as CrossAttention, FeedForward

from einops import rearrange, repeat
import math
from .lora import LoRALinearLayer

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_motion_module(
    in_channels,
    motion_module_type: str, 
    motion_module_kwargs: dict,
    num_adapters: int = 1,
    adapter_weights = [1], 
    parallel_mode = 'weights',
):
    if motion_module_type == "Vanilla":
        return VanillaTemporalModule(in_channels=in_channels,
                                     num_adapters=num_adapters,
                                     adapter_weights = adapter_weights,
                                     parallel_mode = parallel_mode, 
                                     **motion_module_kwargs,)    
    else:
        raise ValueError


class VanillaTemporalModule(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads                = 8,
        num_transformer_block              = 2,
        attention_block_types              =( "Temporal_Self", "Temporal_Self" ),
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        temporal_attention_dim_div         = 1,
        zero_initialize                    = True,
        lora_rank = 32,
        lora_scale = 1.0,
        use_motion_embed = False,
        num_adapters: int = 1,
        adapter_weights = [1], 
        video_length:int = 16,
        parallel_mode = 'weights',
    ):
        super().__init__()
        
        self.temporal_transformer = TemporalTransformer3DModel(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
            num_layers=num_transformer_block,
            attention_block_types=attention_block_types,
            cross_frame_attention_mode=cross_frame_attention_mode,
            temporal_position_encoding=temporal_position_encoding,
            temporal_position_encoding_max_len=temporal_position_encoding_max_len,
            lora_rank = lora_rank,
            lora_scale = lora_scale,
            num_adapters=num_adapters,
            use_motion_embed=use_motion_embed,
            adapter_weights = adapter_weights, 
            video_length = video_length,
            parallel_mode=parallel_mode,
        )
        
        if zero_initialize:
            self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

    def forward(self, input_tensor, temb, encoder_hidden_states, attention_mask=None, anchor_frame_idx=None):
        hidden_states = input_tensor
        hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

        output = hidden_states
        return output


class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads,
        attention_head_dim,

        num_layers,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),        
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        lora_rank = None,
        lora_scale = None,
        use_motion_embed = False,
        num_adapters: int = 1,
        adapter_weights = [1], 
        video_length:int = 16,
        parallel_mode = 'weights',
    ):
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    lora_rank=lora_rank,
                    lora_scale = lora_scale,
                    use_motion_embed = use_motion_embed,
                    num_loras=num_adapters,
                    video_length = video_length,
                    adapter_weights=adapter_weights,
                    parallel_mode=parallel_mode,
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels)    
    
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)
        
        # output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        attention_block_types              = ( "Temporal_Self", "Temporal_Self", ),
        dropout                            = 0.0,
        norm_num_groups                    = 32,
        cross_attention_dim                = 768,
        activation_fn                      = "geglu",
        attention_bias                     = False,
        upcast_attention                   = False,
        cross_frame_attention_mode         = None,
        temporal_position_encoding         = False,
        temporal_position_encoding_max_len = 24,
        use_motion_embed                   = False,
        lora_rank = 32,
        lora_scale = 1.0,
        num_loras:int = 1, # No. of adapters
        video_length:int = 16,
        adapter_weights:list =[1],
        parallel_mode = 'weights',
    ):
        super().__init__()
 
        attention_blocks = []
        norms = []
        
        for block_name in attention_block_types:
            if num_loras <= 1:
                processor = VersatileAttentionProcessor(
                    dim,
                    cross_attention_dim if block_name.endswith("_Cross") else None,
                    lora_rank=lora_rank, lora_scale=lora_scale,
                    use_motion_embed=use_motion_embed,
                    video_length=video_length
                    )
            else:
                processor_class = ParallelAttentionProcessor if parallel_mode=='residual' else ParallelAttentionProcessor_legacy
                processor = processor_class(
                    dim,
                    cross_attention_dim if block_name.endswith("_Cross") else None,
                    lora_rank=lora_rank, lora_scale=lora_scale,
                    num_loras=num_loras,
                    use_motion_embed=use_motion_embed,
                    video_length=video_length,
                    adapter_weights=adapter_weights
                    )
            attention_blocks.append(
                VersatileAttention(
                    attention_mode=block_name.split("_")[0],
                    cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
                    
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
        
                    cross_frame_attention_mode=cross_frame_attention_mode,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                    processor = processor
                )
            )
            norms.append(nn.LayerNorm(dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(dim)


    def forward(self, hidden_states, encoder_hidden_states=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        d_model, 
        dropout = 0., 
        max_len = 24
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VersatileAttention(CrossAttention):
    def __init__(
            self,
            attention_mode                     = None,
            cross_frame_attention_mode         = None,
            temporal_position_encoding         = False,
            temporal_position_encoding_max_len = 24,            
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)
        assert attention_mode == "Temporal"

        self.attention_mode = attention_mode
        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if (temporal_position_encoding and attention_mode == "Temporal") else None

class VersatileAttentionProcessor(nn.Module):
    def __init__(self, hidden_size=None,
        cross_attention_dim=None,
        lora_rank:int=32,
        lora_scale:float=1.0,
        use_motion_embed:bool = False,
        video_length:int = 16,
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        self.use_motion_embed = use_motion_embed
        self.use_motion_lora = False
        # print("LoRA Rank: ", self.lora_rank)
        if self.lora_rank is not None and self.lora_scale is not None:
            self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_v_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            self.use_motion_lora = True
        if self.use_motion_embed:
            self.motion_embed = nn.Embedding(video_length,hidden_size)
        else:
            self.motion_embed = 0

    def __call__(self,
                 attn:CrossAttention,
                 hidden_states:torch.FloatTensor,
                 encoder_hidden_states: Optional[torch.FloatTensor] = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                video_length = None,):
        batch_size, sequence_length, _ = hidden_states.shape
        if attn.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if attn.pos_encoder is not None:
                hidden_states =  attn.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if self.use_motion_embed:
            hidden_states += self.motion_embed.weight 
        else:
            hidden_states += self.motion_embed
        query = attn.to_q(hidden_states) if not self.use_motion_lora else self.to_q_lora(hidden_states) + attn.to_q(hidden_states)
        # query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states) if not self.use_motion_lora else self.to_k_lora(hidden_states) + attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) if not self.use_motion_lora else self.to_v_lora(hidden_states) + attn.to_v(encoder_hidden_states) 
        # key = attn.to_k(encoder_hidden_states)
        # value = attn.to_v(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)

        # attention, what we cannot get enough of
        if False:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            #     hidden_states = self._attention(query, key, value, attention_mask)
            # else:
            #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * (attn.cross_attention_dim // attn.heads))
        # hidden_states = hidden_states.to(query.dtype)

        # linear proj
        # hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[0](hidden_states) if not self.use_motion_lora else self.to_out_lora(hidden_states) + attn.to_out[0](hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if attn.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class ParallelAttentionProcessor_legacy(nn.Module):
    def __init__(self, 
        hidden_size=None,
        cross_attention_dim=None,
        lora_rank:int=32,
        lora_scale:float=1.0,
        num_loras:int=1,
        use_motion_embed:bool = False,
        video_length:int = 16, # no mechanism yet to pass from UNet's __init__ to me
        adapter_weights:list = [1]
        ):
        super().__init__()

        self.n = num_loras
        self.lora_rank = lora_rank
        if lora_rank is None:
            raise ValueError("[ParallelAttentionProcessor]no Lora given!")
        processors = [
            VersatileAttentionProcessor(hidden_size, cross_attention_dim, lora_rank, lora_scale, use_motion_embed, video_length) for _ in range(self.n)
            ]
        self.adapter_weights = adapter_weights
        self.processors = nn.ModuleList(processors)

    def __call__(self, attn:CrossAttention,
                 hidden_states:torch.FloatTensor,
                 encoder_hidden_states: Optional[torch.FloatTensor] = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                video_length = None,):
        batch_size, sequence_length, _ = hidden_states.shape
        if attn.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if attn.pos_encoder is not None:
                hidden_states =  attn.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        for i, processor in enumerate(self.processors):
            query += processor.to_q_lora(hidden_states) * self.adapter_weights[i]
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # MotionLoRA is SA; encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        for i, processor in enumerate(self.processors):
            key += processor.to_k_lora(encoder_hidden_states) *  self.adapter_weights[i]
            value += processor.to_v_lora(encoder_hidden_states) * self.adapter_weights[i]
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)

        # attention, what we cannot get enough of
        if False:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            #     hidden_states = self._attention(query, key, value, attention_mask)
            # else:
            #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * (attn.cross_attention_dim // attn.heads))
        # hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        for i, processor in enumerate(self.processors):
            hidden_states += processor.to_out_lora(encoder_hidden_states) * self.adapter_weights[i]

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if attn.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class ParallelAttentionProcessor(nn.Module):
    def __init__(self, 
        hidden_size=None,
        cross_attention_dim=None,
        lora_rank:int=32,
        lora_scale:float=1.0,
        num_loras:int=1,
        use_motion_embed:bool = False,
        video_length:int = 16, # no mechanism yet to pass from UNet's __init__ to me
        adapter_weights:list = [1]
        ):
        super().__init__()

        self.n = num_loras
        self.lora_rank = lora_rank
        if lora_rank is None:
            raise ValueError("[ParallelAttentionProcessor]no Lora given!")
        processors = [
            VersatileAttentionProcessor(hidden_size, cross_attention_dim, lora_rank, lora_scale, use_motion_embed, video_length) for _ in range(self.n)
            ]
        self.adapter_weights = adapter_weights
        self.processors = nn.ModuleList(processors)

    def __call__(self, attn:CrossAttention,
                 hidden_states:torch.FloatTensor,
                 encoder_hidden_states: Optional[torch.FloatTensor] = None,
                 attention_mask: Optional[torch.FloatTensor] = None,
                video_length = None,):
        batch_size, sequence_length, _ = hidden_states.shape
        if attn.attention_mode == "Temporal":
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            
            if attn.pos_encoder is not None:
                hidden_states =  attn.pos_encoder(hidden_states)
            
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states
        else:
            raise NotImplementedError

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        Qs = []
        Ks = []
        Vs = []

        query = attn.to_q(hidden_states)
        for i, processor in enumerate(self.processors):
            # query += processor.to_q_lora(hidden_states) * self.adapter_weights[i]
            q = query + processor.to_q_lora(hidden_states)
            q = attn.head_to_batch_dim(q)
            Qs.append(q)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        # MotionLoRA is SA; encoder_hidden_states = hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        for i, processor in enumerate(self.processors):
            # key += processor.to_k_lora(encoder_hidden_states) *  self.adapter_weights[i]
            # value += processor.to_v_lora(encoder_hidden_states) * self.adapter_weights[i]
            k = key + processor.to_k_lora(encoder_hidden_states)
            v = value + processor.to_v_lora(encoder_hidden_states)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            Ks.append(k)
            Vs.append(v)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(attn.heads, dim=0)

        # attention, what we cannot get enough of
        sum_hidden_states = 0
        for i in range(self.n):
            hidden_states = F.scaled_dot_product_attention(Qs[i], Ks[i], Vs[i], attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

            # if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            #     hidden_states = self._attention(query, key, value, attention_mask)
            # else:
            #     hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)
            hidden_states = attn.batch_to_head_dim(hidden_states)
            # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * (attn.cross_attention_dim // attn.heads))
            # hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states) + self.processors[i].to_out_lora(hidden_states)
            sum_hidden_states += hidden_states * self.adapter_weights[i]
        hidden_states = sum_hidden_states

        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        
        if attn.attention_mode == "Temporal":
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states
