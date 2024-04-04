import glob

import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torch
import yaml

import torchvision.transforms as T
from torchvision.io import read_video, write_video
import os
import random
import numpy as np

import logging
logger = logging.getLogger(__name__)

# Modified from tokenflow_utils.py
def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, "t", t)
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].tempo_attns[block].transformer_blocks[0].attn1.processor
            setattr(module, "t", t)


# PNP injection functions
# Modified from ResnetBlock2D.forward
# Modified from models/resnet.py
# Modified from register_conv_injection under tokenflow_utils.py
from diffusers.models.resnet import Upsample2D
from diffusers.models.resnet import Downsample2D


def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb, scale: float = 1.0):
            hidden_states = input_tensor

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm1(hidden_states, temb)
            else:
                hidden_states = self.norm1(hidden_states)

            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = (
                    self.upsample(input_tensor, scale=scale)
                    if isinstance(self.upsample, Upsample2D)
                    else self.upsample(input_tensor)
                )
                hidden_states = (
                    self.upsample(hidden_states, scale=scale)
                    if isinstance(self.upsample, Upsample2D)
                    else self.upsample(hidden_states)
                )
            elif self.downsample is not None:
                input_tensor = (
                    self.downsample(input_tensor, scale=scale)
                    if isinstance(self.downsample, Downsample2D)
                    else self.downsample(input_tensor)
                )
                hidden_states = (
                    self.downsample(hidden_states, scale=scale)
                    if isinstance(self.downsample, Downsample2D)
                    else self.downsample(hidden_states)
                )

            hidden_states = self.conv1(hidden_states, scale)

            if self.time_emb_proj is not None:
                if not self.skip_time_act:
                    temb = self.nonlinearity(temb)
                temb = self.time_emb_proj(temb, scale)[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                hidden_states = self.norm2(hidden_states, temb)
            else:
                hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states, scale)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Conv at t={self.t}")
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size : 2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size :] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor, scale)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, "injection_schedule", injection_schedule)


# Modified from AttnProcessor2_0.__call__
# Modified from models/attention.py
# Modified from register_extended_attention_pnp under tokenflow_utils.py
from typing import Optional
from diffusers.models.attention_processor import AttnProcessor2_0
from consisti2v.models.videoldm_attention import ConditionalAttention
from diffusers.models.attention_processor import Attention

def register_spatial_attention_pnp(model, injection_schedule):
    class ModifiedSpaAttnProcessor(AttnProcessor2_0):
        def __init__(self):
            if not hasattr(F, "scaled_dot_product_attention"):
                raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        def __call__(
            self,
            attn: ConditionalAttention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale: float = 1.0,
        ):
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

            # Modified here
            chunk_size = batch_size // 3  # batch_size is 3*chunk_size because concat[source, uncond, cond]

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states, scale=scale)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            key = attn.to_k(encoder_hidden_states, scale=scale)
            value = attn.to_v(encoder_hidden_states, scale=scale)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Modified here.
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Spa-Attn at t={self.t}")
                # inject source into unconditional
                query[chunk_size : 2 * chunk_size] = query[:chunk_size]
                key[chunk_size : 2 * chunk_size] = key[:chunk_size]
                # inject source into conditional
                query[2 * chunk_size :] = query[:chunk_size]
                key[2 * chunk_size :] = key[:chunk_size]

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, scale=scale)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states

    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         module.attn1.processor.__call__ = sa_processor__call__(module.attn1.processor)
    #         setattr(module.attn1.processor, "injection_schedule", [])  # Disable PNP

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            modified_processor = ModifiedSpaAttnProcessor()
            setattr(modified_processor, "injection_schedule", injection_schedule)
            module.processor = modified_processor



def register_temp_attention_pnp(model, injection_schedule):
    class ModifiedTmpAttnProcessor:
        def __init__(self):
            if not hasattr(F, "scaled_dot_product_attention"):
                raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        def __call__(
            self,
            attn: Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale: float = 1.0,
            key_pos_idx: Optional[torch.Tensor] = None,
        ):
            assert attention_mask is None
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

            # Modified here
            chunk_size = batch_size // 3  # batch_size is 3*chunk_size because concat[source, uncond, cond]

            # if attention_mask is not None:
            #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            #     # scaled_dot_product_attention expects attention_mask shape to be
            #     # (batch, heads, source_length, target_length)
            #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = attn.to_q(hidden_states, scale=scale)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
            qlen = hidden_states.shape[1]
            klen = encoder_hidden_states.shape[1]
            # currently only add bias for self attention. Relative distance doesn't make sense for cross attention.
            # if qlen == klen:
            #     time_rel_pos_bias = attn.rotary_bias(qlen, klen, device=hidden_states.device, dtype=hidden_states.dtype)
            #     attention_mask = repeat(time_rel_pos_bias, "h d1 d2 -> b h d1 d2", b=batch_size)

            key = attn.to_k(encoder_hidden_states, scale=scale)
            value = attn.to_v(encoder_hidden_states, scale=scale)

            # Modified here.
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                logger.debug(f"PnP Injecting Tmp-Attn at t={self.t}")
                # inject source into unconditional
                query[chunk_size : 2 * chunk_size] = query[:chunk_size]
                key[chunk_size : 2 * chunk_size] = key[:chunk_size]
                # inject source into conditional
                query[2 * chunk_size :] = query[:chunk_size]
                key[2 * chunk_size :] = key[:chunk_size]


            query = attn.rotary_emb.rotate_queries_or_keys(query)
            if qlen == klen:
                key = attn.rotary_emb.rotate_queries_or_keys(key)
            elif key_pos_idx is not None:
                key = attn.rotary_emb.rotate_queries_or_keys(key, seq_pos=key_pos_idx)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, scale=scale)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states
    # for _, module in model.unet.named_modules():
    #     if isinstance_str(module, "BasicTransformerBlock"):
    #         module.attn1.processor.__call__ = ta_processor__call__(module.attn1.processor)
    #         setattr(module.attn1.processor, "injection_schedule", [])  # Disable PNP

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].tempo_attns[block].transformer_blocks[0].attn1
            modified_processor = ModifiedTmpAttnProcessor()
            setattr(modified_processor, "injection_schedule", injection_schedule)
            module.processor = modified_processor