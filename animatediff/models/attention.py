# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm, Attention

from einops import rearrange, repeat
import pdb

from .sa_utils import SparseCausalAttentionProcessor
from .ca_utils import LoRACAAttentionProcessor


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        use_motion_embedding = False,
        use_attn_temp_lora = True,
        use_i2v = True,
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,

        num_adapters: int = 1,
        adapter_weights = [1], 
        use_i2v_q_lora = False, 
        use_i2v_out_lora = False,
        i2v_lora_rank = None,
        i2v_lora_scale = None,
        use_ca_lora = False,
        ca_lora_rank = None,
        ca_lora_scale = None,

        q_downsample:bool = False,
        q_downsample_ratio:int = 4,
        ca_pe_mode:str = None,
        use_q_lora:bool = False,
        use_ip_adapter:bool = False,

        remove_WO: bool = False,
        video_length:int = 16,
        parallel_mode: str = 'weights',
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_motion_embedding = use_motion_embedding,
                    use_attn_temp_lora = use_attn_temp_lora,
                    use_i2v = use_i2v,
                    unet_use_cross_frame_attention=unet_use_cross_frame_attention,
                    unet_use_temporal_attention=unet_use_temporal_attention,
                    
                    num_adapters=num_adapters,
                    adapter_weights = adapter_weights, 
                    use_i2v_q_lora = use_i2v_q_lora, 
                    use_i2v_out_lora = use_i2v_out_lora,
                    i2v_lora_rank = i2v_lora_rank,
                    i2v_lora_scale = i2v_lora_scale,
                    ca_lora_rank=ca_lora_rank,
                    ca_lora_scale=ca_lora_scale,
                    use_ca_lora=use_ca_lora,
                    q_downsample=q_downsample,
                    q_downsample_ratio=q_downsample_ratio,
                    ca_pe_mode=ca_pe_mode,
                    use_q_lora=use_q_lora,
                    use_ip_adapter=use_ip_adapter,
                    remove_WO=remove_WO,
                    video_length=video_length,
                    parallel_mode=parallel_mode,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True, attention_store = None, location = None):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        # encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)
        # e.g. "C U" => "CCCC UUUU" for 1st dim (f=4)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length, 
                attention_store = attention_store,
                location = location,
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,

        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention = None,
        use_motion_embedding = True,
        use_attn_temp_lora= True,
        use_i2v = True,
        lora_rank = 64,
        lora_scale = 1.0,
        parallel_mode = 'weights',

        num_adapters = 1,
        adapter_weights = [1],
        use_i2v_q_lora = False, 
        use_i2v_out_lora = False,
        i2v_lora_rank = None,
        i2v_lora_scale = None,
        use_ca_lora = False,
        ca_lora_rank = None,
        ca_lora_scale = None,

        q_downsample:bool = False,
        q_downsample_ratio:int = 4,
        ca_pe_mode:str = None,
        use_q_lora:bool = False,
        use_ip_adapter:bool = False,

        remove_WO: bool = False,
        video_length:int = 16,
    ):
        super().__init__()
        self.num_adapters = num_adapters
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.unet_use_cross_frame_attention = unet_use_cross_frame_attention
        self.unet_use_temporal_attention = unet_use_temporal_attention
        self.scale = 1.0
        self.adapter_weights = adapter_weights

        # SC-Attn
        assert unet_use_cross_frame_attention is not None
        if unet_use_cross_frame_attention:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn1 = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        self.num_i2v_adapters = num_adapters
        if num_adapters == 1:
            self.i2v_adapter = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                processor = SparseCausalAttentionProcessor(hidden_size = dim, use_q_lora = use_i2v_q_lora, use_out_lora = use_i2v_out_lora, lora_rank = i2v_lora_rank, lora_scale = i2v_lora_scale) 
            )
        elif num_adapters > 1:
            i2v_adapters = [
                Attention(
                    query_dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                    upcast_attention=upcast_attention,
                    processor = SparseCausalAttentionProcessor(hidden_size = dim, use_q_lora = use_i2v_q_lora, use_out_lora = use_i2v_out_lora, lora_rank = i2v_lora_rank, lora_scale = i2v_lora_scale) 
                ) for _ in range(num_adapters)
            ]
            self.i2v_adapters = nn.ModuleList(i2v_adapters)

        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        # log(270725): now always use LoRACAAttentionProcessor
        if cross_attention_dim is not None:
            if use_ca_lora:
                lora_rank = ca_lora_rank
                lora_scale = ca_lora_scale
            else:
                lora_rank = None
                lora_scale = None
            

            self.attn2 = CrossAttention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    processor = LoRACAAttentionProcessor(
                        hidden_size=dim,
                        cross_attention_dim=cross_attention_dim,
                        num_adapters=num_adapters,
                        adapter_weights=adapter_weights,
                        lora_rank=lora_rank, lora_scale=lora_scale,
                        q_downsample=q_downsample,
                        q_downsample_ratio=q_downsample_ratio,
                        ca_pe_mode=ca_pe_mode,
                        use_q_lora=use_q_lora,
                        use_ip_adapter=use_ip_adapter,
                        remove_WO=remove_WO,
                        video_length=video_length,
                        parallel_mode=parallel_mode,
                    ) 
               )
           
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        assert unet_use_temporal_attention is not None
        if unet_use_temporal_attention:
            self.attn_temp = CrossAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            self.attn_temp.set_use_memory_efficient_attention_xformers(False)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None, attention_store=None, location = None):
        # SparseCausal-Attention

        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        # Run I2V adapter
        i2v_hidden_states = 0
        if self.num_i2v_adapters == 1:
            i2v_hidden_states = self.i2v_adapter(norm_hidden_states, attention_mask=attention_mask,video_length=video_length)
        elif self.num_i2v_adapters > 1:
            for index, i2v_adapter in enumerate(self.i2v_adapters):
                i2v_hidden_states += i2v_adapter(norm_hidden_states, attention_mask=attention_mask,video_length=video_length) * self.adapter_weights[index]

        if self.unet_use_cross_frame_attention:
            hidden_states = self.attn1(norm_hidden_states,encoder_hidden_states, attention_mask=attention_mask) + i2v_hidden_states + hidden_states
        else:
            #should run here
            hidden_states = self.scale * self.attn1(norm_hidden_states, attention_mask=attention_mask) + i2v_hidden_states + hidden_states



        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, attention_store=attention_store, location=location,
                )
                + hidden_states
            )
        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        if self.unet_use_temporal_attention:
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states