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
from torchvision.io import write_video
from einops import rearrange

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# Modified from tokenflow/utils.py
def save_video_as_frames(video_path, img_size=(512, 512)):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith(".mov"):
        video = T.functional.rotate(video, -90)
    video_name = Path(video_path).stem
    video_dir = Path(video_path).parent
    os.makedirs(f"{video_dir}/{video_name}", exist_ok=True)
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize(img_size, resample=Image.Resampling.LANCZOS)
        image_resized.save(f"{video_dir}/{video_name}/{ind}.png")


# Modified from tokenflow/utils.py
def load_video_frames(frames_path, n_frames):
    # Load paths
    print(frames_path)
    paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
    if not os.path.exists(paths[0]):
        paths = [f"{frames_path}/%05d.jpg" % i for i in range(n_frames)]
        #paths = [f"{frames_path}/frames/%05d.jpg" % i for i in range(n_frames)]
    frames = [torch.as_tensor(np.array(Image.open(path), dtype=np.uint8, copy=True)).unsqueeze(0) for path in paths]
    # if frames[0].size[0] == frames[0].size[1]:
    #     frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
    frames = torch.cat(frames, dim=0).permute(0, 3, 1, 2)  # f,c,h,w
    return paths, frames


def load_ddim_latents_at_t(t, ddim_latents_path):
    ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path)
    return ddim_latents_at_t


def load_ddim_latents_at_T(ddim_latents_path):
    noisest = max(
        [
            int(x.split("_")[-1].split(".")[0])
            for x in glob.glob(os.path.join(ddim_latents_path, f"ddim_latents_*.pt"))
        ]
    )
    ddim_latents_at_T_path = os.path.join(ddim_latents_path, f"ddim_latents_{noisest}.pt")
    ddim_latents_at_T = torch.load(ddim_latents_at_T_path) # [b, c, f, h, w] [1, 4, 16, 40, 64]
    return ddim_latents_at_T


def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


# TODO: delete this
def load_imgs(data_path, n_frames, device="cuda", pil=False):
    imgs = []
    pils = []
    for i in range(n_frames):
        img_path = os.path.join(data_path, "%05d.jpg" % i)
        if not os.path.exists(img_path):
            img_path = os.path.join(data_path, "%05d.png" % i)
        img_pil = Image.open(img_path)
        pils.append(img_pil)
        img = T.ToTensor()(img_pil).unsqueeze(0)
        imgs.append(img)
    if pil:
        return torch.cat(imgs).to(device), pils
    return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10, scaling_255=False):
    video_codec = "libx264"
    video_options = {
        "crf": "18",  # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
        "preset": "slow",
        # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    }
    if scaling_255:
        frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    else:
        frames = raw_frames.to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(save_path, frames, fps=fps, video_codec=video_codec, options=video_options)


# Modified from tokenflow_utils.py
def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, "t", t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, "t", t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn_temp
            setattr(module, "t", t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, "t", t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, "t", t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn_temp
            setattr(module, "t", t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, "t", t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, "t", t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn_temp
    setattr(module, "t", t)


# PNP injection functions
# Modified from models/resnet.py
# Modified from register_conv_injection under tokenflow_utils.py
def register_conv_injection(model, injection_schedule, d_s=0.1, d_t=0.5):
    def conv_forward(self):
        def forward(input_tensor, temb, scale=None):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                
                # inject unconditional
                hidden_states[source_batch_size : 2 * source_batch_size] = hidden_states[:source_batch_size]
                
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]


            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, "injection_schedule", injection_schedule)


# Modified from models/attention.py
# Modified from register_extended_attention_pnp under tokenflow_utils.py
def register_spatial_attention_pnp(model, injection_schedule, d_s=0.1, d_t=0.5):
    def sa_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, use_image_num=None):
            batch_size, sequence_length, _dim = hidden_states.shape
            n_frames = batch_size // 3  # batch_size is 3*n_frames because concat[source, uncond, cond]

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)  # [b (h w)] f (nd * d)

            if self.added_kv_proj_dim is not None:
                print(f"[ERROR] Run into added_kv_proj_dim, which is not supported yet. Exiting...")
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)
                encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                    query[n_frames : 2 * n_frames] = query[:n_frames]
                    key[n_frames : 2 * n_frames] = key[:n_frames]
                    # value[n_frames : 2 * n_frames] = value[:n_frames]

                    # inject source into conditional
                    query[2 * n_frames :] = query[:n_frames]
                    key[2 * n_frames :] = key[:n_frames]
                    # value[2 * n_frames :] = value[:n_frames]

                if not self.use_relative_position:
                    key = self.reshape_heads_to_batch_dim(key)
                    value = self.reshape_heads_to_batch_dim(value)

            dim = query.shape[-1]
            if not self.use_relative_position:
                query = self.reshape_heads_to_batch_dim(query)  # [b (h w) nd] f d

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask)
                else:
                    hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, "injection_schedule", [])  # Disable PNP

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}

    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, "injection_schedule", injection_schedule)



def register_cross_attention_pnp(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, use_image_num=None):
            batch_size, sequence_length, _dim = hidden_states.shape
            n_frames = batch_size // 3 # batch_size is 3*n_frames because concat[source, uncond, cond]

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states) # [b (h w)] f (nd * d)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # inject source into unconditional
                query[n_frames:2 * n_frames] = query[:n_frames]
                key[n_frames:2 * n_frames] = key[:n_frames]
                # inject source into conditional
                query[2 * n_frames:] = query[:n_frames]
                key[2 * n_frames:] = key[:n_frames]

            # print('before reshpape query shape', query.shape)
            dim = query.shape[-1]
            if not self.use_relative_position:
                query = self.reshape_heads_to_batch_dim(query) # [b (h w) nd] f d
            # print('after reshape query shape', query.shape)

            if not self.use_relative_position:
                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask)
                else:
                    hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn2.forward = sa_forward(module.attn2)
            setattr(module.attn2, 'injection_schedule', []) # Disable PNP

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_temp_attention_pnp(model, injection_schedule, d_s=0.1, d_t=0.5):
    def ta_forward(self):
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            time_rel_pos_bias = self.time_rel_pos_bias(hidden_states.shape[1], device=hidden_states.device)
            batch_size, sequence_length, _dim = hidden_states.shape

            n_frames = batch_size // 3  # batch_size is 3*n_frames because concat[source, uncond, cond]

            encoder_hidden_states = encoder_hidden_states

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)  # [b (h w)] f (nd * d)
            dim = query.shape[-1]

            if self.added_kv_proj_dim is not None:
                print(f"[ERROR] Run into added_kv_proj_dim, which is not supported yet. Exiting...")
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                key = self.reshape_heads_to_batch_dim(key)
                value = self.reshape_heads_to_batch_dim(value)
                encoder_hidden_states_key_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_key_proj)
                encoder_hidden_states_value_proj = self.reshape_heads_to_batch_dim(encoder_hidden_states_value_proj)

                key = torch.concat([encoder_hidden_states_key_proj, key], dim=1)
                value = torch.concat([encoder_hidden_states_value_proj, value], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
                key = self.to_k(encoder_hidden_states)
                value = self.to_v(encoder_hidden_states)

                if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                    # inject source into unconditional
                    query[n_frames : 2 * n_frames] = query[:n_frames]
                    key[n_frames : 2 * n_frames] = key[:n_frames]
                    # value[n_frames : 2 * n_frames] = value[:n_frames] # this will make the content following the source completely

                    # inject source into conditional
                    query[2 * n_frames :] = query[:n_frames]
                    key[2 * n_frames :] = key[:n_frames]
                    # value[2 * n_frames :] = value[:n_frames] # this will make the content following the source completely

            if attention_mask is not None:
                if attention_mask.shape[-1] != query.shape[1]:
                    target_length = query.shape[1]
                    attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                    attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

            # attention, what we cannot get enough of
            if self._use_memory_efficient_attention_xformers:
                hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
                # Some versions of xformers return output in fp32, cast it back to the dtype of the input
                hidden_states = hidden_states.to(query.dtype)
            else:
                if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                    hidden_states = self._attention(query, key, value, attention_mask, time_rel_pos_bias)
                else:
                    hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)

            # dropout
            hidden_states = self.to_out[1](hidden_states)
            return hidden_states

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn_temp.forward = ta_forward(module.attn_temp)
            setattr(module.attn_temp, "injection_schedule", [])  # Disable PNP

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn_temp
            module.forward = ta_forward(module)
            setattr(module, "injection_schedule", injection_schedule)