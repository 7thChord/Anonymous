import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, Callable, List, Optional, Iterable
from torchvision.transforms import Resize
from diffusers.models.attention import Attention as CrossAttention, FeedForward, AdaLayerNorm, Attention
import einops
from einops import rearrange
import math
from .lora import LoRALinearLayer
from .pe import PositionalEncoding, PositionalEncoding_Rotary

def repeat_(t: torch.Tensor, video_length):
    return einops.repeat(t, 'b n c -> (b f) n c', f=video_length)


class LoRACAAttentionProcessor(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, hidden_size=None,
        cross_attention_dim=None,
        lora_rank:int = None, # default rank=32, scale=1.0; for multi-adapter setting, the two should be lists.
        lora_scale:float = None,
        num_adapters:int = 1,
        adapter_weights:list=[1],
        remove_WO: bool = False,
        video_length:int = 16,
        # STQ settings
        q_downsample:bool = False,
        q_downsample_ratio:int = 4,
        ca_pe_mode:str = None, # None ("null") / "naive" / "temporal" / "temporal_sine" / "temporal_rope"
        # Q-LoRA
        use_q_lora:bool = False,
        use_ip_adapter:bool = False,
        # Multi-CA
        parallel_mode:str = 'weights',
        ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        self.q_downsample_ratio = q_downsample_ratio
        self.lora_scale = lora_scale
        self.lora_rank = lora_rank
        self.num_adapters = num_adapters
        self.q_downsample = q_downsample
        if ca_pe_mode=='null': ca_pe_mode = None
        self.ca_pe_mode = ca_pe_mode
        self.remove_WO = remove_WO
        self.video_length = video_length
        if ca_pe_mode is not None:
            self.temporal_position_encoding_max_len = video_length if 'temporal' in ca_pe_mode or 'rope' in ca_pe_mode else 2560
            PE_class = PositionalEncoding_Rotary if 'rope' in ca_pe_mode else PositionalEncoding 
            self.pos_encoder = PE_class(
                    hidden_size,
                    dropout=0., 
                    max_len=self.temporal_position_encoding_max_len,
                    mode=ca_pe_mode,
                )
            print("PE for Q:", ca_pe_mode)

            if 'rope' in ca_pe_mode and 'K' in ca_pe_mode:
                self.pos_encoder_kv = PositionalEncoding_Rotary(
                    hidden_size,
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
        self.adapter_weights = adapter_weights
        self.use_q_lora = use_q_lora
        self.use_ip_adapter = use_ip_adapter
        self.parallel_mode = parallel_mode

        self.zero_lora = False
        if lora_rank is None or lora_rank==0:
            self.zero_lora = True
            self.multi_ca_inference = False
        elif isinstance(lora_rank, Iterable):
            # can be [0] with num_adapter > 1. If so, then we apply multi-CA still, without using LoRAs
            self.multi_ca_inference = True
            self.zero_lora = lora_rank[0] == 0
        else:
            self.multi_ca_inference = False

        if self.lora_rank is not None and self.lora_scale is not None and not self.zero_lora:
            if num_adapters == 1:
                if use_q_lora:
                    self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
                self.to_k_lora = LoRALinearLayer(cross_attention_dim if cross_attention_dim else hidden_size, hidden_size, self.lora_rank, self.lora_scale)
                self.to_v_lora = LoRALinearLayer(cross_attention_dim if cross_attention_dim else hidden_size, hidden_size, self.lora_rank, self.lora_scale)
                if not self.remove_WO:
                    self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, self.lora_rank, self.lora_scale)
            else: # >1
                if use_q_lora:
                    self.to_q_loras = nn.ModuleList([
                        LoRALinearLayer(hidden_size, hidden_size, self.lora_rank[i], self.lora_scale[i])
                        for i in range(num_adapters)]) # TODO
                self.to_k_loras = nn.ModuleList([
                    LoRALinearLayer(cross_attention_dim if cross_attention_dim else hidden_size, hidden_size, self.lora_rank[i], self.lora_scale[i])
                    for i in range(num_adapters)])
                self.to_v_loras = nn.ModuleList([
                    LoRALinearLayer(cross_attention_dim if cross_attention_dim else hidden_size, hidden_size, self.lora_rank[i], self.lora_scale[i])
                    for i in range(num_adapters)])
                if not self.remove_WO:
                    self.to_out_loras = nn.ModuleList([
                        LoRALinearLayer(hidden_size, hidden_size, self.lora_rank[i], self.lora_scale[i])
                        for i in range(num_adapters)])
                    
        if self.use_ip_adapter:
            self.to_k_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_ip = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        video_length: int= 16,
        attention_store = None,
        location = None,
    ) -> torch.FloatTensor:
        q_downsample = self.q_downsample
        use_ip_adapter = self.use_ip_adapter

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        '''
        log (240725):
        hidden_states: the latent signal
            * inference with CFG: (2F, HW, C), batch_size = 2F
            * training: (BF, HW, C)
        encoder_hidden_states: the text Emb
            * vanilla inference with CFG (2, 77, 768)
            * if multi_ca=True, it's (adapter#+1, 77, 768)
            * training: (1, 77, 768)
        '''
        concat_kv = self.parallel_mode=='weights' # if True (old version): concat all K, V in multi-adapter mode.

        cross_attention = encoder_hidden_states is not None
        multi_ca = self.multi_ca_inference and self.num_adapters>1
        
        batch_size = hidden_states.shape[0]
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            _, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2) # (BF, HW, C)
        else:
            # assume 16:10 ratio
            height = int(math.sqrt(hidden_states.shape[1]/1.6))
            width = int(1.6*height)

        # log(240806): patch for training case
        video_length = self.video_length

        _, sequence_length, _ = (
            hidden_states.shape if not cross_attention else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # log(240815): The Q being spatiotemporal, downsampled spatially (by `q_downsample_ratio`)
        if q_downsample:
            hidden_states = rearrange(hidden_states,"(b f) d c -> b c f d", f=video_length)
            hidden_states = rearrange(hidden_states, "b c f (h w) -> b c f h w", h=height, w=width)

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

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if True else (scale,) # if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states,  *args) # Q: (BF, HW, d) in default or (B, FHW/s, d) in STQ case
        if self.use_q_lora:
            if multi_ca and not self.zero_lora:
                for i in range(self.num_adapters):
                    query += self.to_q_loras[i](hidden_states) * self.adapter_weights[i]
            else:
                query += self.to_q_lora(hidden_states)

        if q_downsample and self.ca_pe_mode and 'ropeQ' in self.ca_pe_mode and '3d' in self.ca_pe_mode:
            if '3d' in self.ca_pe_mode:
                query = rearrange(hidden_states,"b (f h w) c -> b f h w c", f=video_length, h=target_size[1], w=target_size[2])
            query = self.pos_encoder(query)

        if not cross_attention:
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
                    key += self.to_k_loras[i](text_emb) * self.adapter_weights[i] if concat_kv else self.to_k_loras[i](text_emb)
                    value += self.to_v_loras[i](text_emb) * self.adapter_weights[i] if concat_kv else self.to_v_loras[i](text_emb)
                key_c.append(key)
                value_c.append(value)
            key_c = torch.cat(key_c, dim=1) # (1, adapter# x77, d)
            value_c = torch.cat(value_c, dim=1)

            if do_classifier_free_guidance:
                text_emb = encoder_hidden_states[-1:] # keep dim
                key_u = attn.to_k(text_emb, *args) # (1, 77, d)
                value_u = attn.to_v(text_emb, *args)

        elif self.num_adapters > 1:
            # non-multi-CA: multi-LoRA in parallel, att mechanism stays original
            # actually, debug only. Won't reach here in inference.
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)
            if not self.zero_lora:
                for i in range(self.num_adapters):
                    key += self.to_k_loras[i](encoder_hidden_states) * self.adapter_weights[i]
                    value += self.to_v_loras[i](encoder_hidden_states) * self.adapter_weights[i]
        elif not self.zero_lora:
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)
            key += self.to_k_lora(encoder_hidden_states) 
            value += self.to_v_lora(encoder_hidden_states)
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

            if not multi_ca or concat_kv:
                hidden_states = F.scaled_dot_product_attention(
                    query[:equivalent_video_length], key_c, value_c, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
            else:
                result_sum = 0
                for i in range(self.num_adapters):
                    result = F.scaled_dot_product_attention(
                        query[:equivalent_video_length], key_c[:, :, 77*i:77*i+77], value_c[:, :, 77*i:77*i+77], attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                    )
                    result = result.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                    result = result.to(query.dtype)
                    result = attn.to_out[0](result, *args)

                    if not self.zero_lora:
                        result += self.to_out_loras[i](result)
                    result *= self.adapter_weights[i]
                    result_sum += result
                hidden_states = result_sum
            
            if do_classifier_free_guidance:
                key_u = key_u.view(equivalent_video_length, -1, attn.heads, head_dim).transpose(1, 2)
                value_u = value_u.view(equivalent_video_length, -1, attn.heads, head_dim).transpose(1, 2)
                hidden_states_u = F.scaled_dot_product_attention(
                    query[equivalent_video_length:], key_u, value_u, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                hidden_states = torch.cat([hidden_states_u, hidden_states]) # log(240920): bugfix - U must be ahead of C

        else: # original
            if not q_downsample: # original:
                # Default case: Q(BF, HW, d); repeat K,V (B, 77, d) => (BF, 77, d)
                key = repeat_(key, video_length)
                value = repeat_(value, video_length)
            else:
                # STQ case: Q(B, FHW/s, d); KV(B, 77, d)
                batch_size = batch_size // video_length
                if self.pos_encoder_kv is not None:
                    # repeat K,V (B, 77, d) => (B, 77F, d)
                    key = einops.repeat(key, 'b n c -> b (n f) c', f=video_length)
                    value = einops.repeat(value, 'b n c -> b (n f) c', f=video_length)
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
            ) # in the range of e-01

            if attention_store is not None:
                attention_weight = attention_weight_extraction(query,key, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
                attention_store.store(attention_weight,True, location)
                # hidden_states_to_compare calculated this way is not exactly the same as the hidden_state above, but the difference is in the 
                # range of e-09 => so I am assuming this is almost the same

        if not multi_ca or concat_kv:
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if use_ip_adapter:
                end_pos = encoder_hidden_states.shape[1] - 4
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    encoder_hidden_states[:, end_pos:, :],
                )

                # for ip-adapter
                ip_key = self.to_k_ip(ip_hidden_states)
                ip_value = self.to_v_ip(ip_hidden_states)
                if not q_downsample:
                    ip_key = repeat_(ip_key, video_length=16)
                    ip_value = repeat_(ip_value, video_length=16)

                ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                ip_hidden_states = F.scaled_dot_product_attention(
                    query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                )

                # does this do anything
                with torch.no_grad():
                    self.attn_map = query @ ip_key.transpose(-2, -1).softmax(dim=-1)
                    #print(self.attn_map.shape)

                ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                ip_hidden_states = ip_hidden_states.to(query.dtype)

                hidden_states = hidden_states + 1.0 * ip_hidden_states


            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            if not self.remove_WO and not self.zero_lora:
                if multi_ca:
                    for i in range(self.num_adapters):
                        hidden_states += self.to_out_loras[i](hidden_states) * self.adapter_weights[i]
                elif self.num_adapters==1:
                    hidden_states += self.to_out_lora(hidden_states)

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        # log(240815): upsample (for STQ)
        if q_downsample:
            hidden_states = rearrange(hidden_states,"b (f d) c -> b c f d",f=video_length,d=(height//self.q_downsample_ratio)*(width//self.q_downsample_ratio))
            hidden_states = rearrange(hidden_states,"b c f (h w) -> (b f) c h w",w=width//self.q_downsample_ratio,h=height//self.q_downsample_ratio)
            hidden_states = torch.nn.functional.interpolate(hidden_states,(height,width))
            hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=video_length)
            hidden_states = rearrange(hidden_states, "b c f h w -> (b f) (h w) c")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def attention_weight_extraction(query, key, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias.to(attn_weight.device)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight
