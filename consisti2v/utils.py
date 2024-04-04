import os
import random
import numpy as np
import torch
from torchvision.io import read_video
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from diffusers.utils import load_image
import torch.nn.functional as F
import glob

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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_ddim_latents_at_t(t, ddim_latents_path):
    ddim_latents_at_t_path = os.path.join(ddim_latents_path, f"ddim_latents_{t}.pt")
    assert os.path.exists(ddim_latents_at_t_path), f"Missing latents at t {t} path {ddim_latents_at_t_path}"
    ddim_latents_at_t = torch.load(ddim_latents_at_t_path)
    print(f"############ Loaded ddim_latents_at_t from {ddim_latents_at_t_path}")
    return ddim_latents_at_t

def load_ddim_latents_at_T(ddim_latents_path):
    noisest = max(
        [
            int(x.split("_")[-1].split(".")[0])
            for x in glob.glob(os.path.join(ddim_latents_path, f"ddim_latents_*.pt"))
        ]
    )
    ddim_latents_at_T_path = os.path.join(ddim_latents_path, f"ddim_latents_{noisest}.pt")
    ddim_latents_at_T = torch.load(ddim_latents_at_T_path)  # [b, c, f, h, w] [1, 4, 16, 40, 64]
    return ddim_latents_at_T


# Modified from tokenflow/utils.py
def convert_video_to_frames(video_path, img_size=(512, 512), save_frames=True, save_dir=None):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith(".mov"):
        video = T.functional.rotate(video, -90)
    if save_frames:
        video_name = Path(video_path).stem
        video_dir = Path(video_path).parent
        if save_dir is not None:
            video_dir = save_dir
        os.makedirs(f"{video_dir}/{video_name}", exist_ok=True)
    frames = []
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        image_resized = image.resize(img_size, resample=Image.Resampling.LANCZOS)
        print(f"image_resized.size, height, width: {image_resized.size}, {img_size[1]}, {img_size[0]}")
        if save_frames:
            image_resized.save(f"{video_dir}/{video_name}/{ind}.png")
            print(f"Saved frame {video_dir}/{video_name}/{ind}.png")
        frames.append(image_resized)
    return frames


# Modified from tokenflow/utils.py
def load_video_frames(frames_path, n_frames):
    # Load paths
    paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
    frames = [load_image(p) for p in paths]
    return paths, frames


def register_spatial_attention_pnp(model, injection_schedule):
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
                    # inject source into unconditional
                    query[n_frames: 2 * n_frames] = query[:n_frames]
                    key[n_frames: 2 * n_frames] = key[:n_frames]
                    # inject source into conditional
                    query[2 * n_frames:] = query[:n_frames]
                    key[2 * n_frames:] = key[:n_frames]

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
