"""
This script is adopted from TokenFlow/preprocess.py and SEINE/with_mask_sample.py.
"""

import os
from tqdm import tqdm, trange
import torch
import torch.nn as nn
import argparse
from torchvision.io import read_video, write_video
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import numpy as np
import random
import yaml
import einops
from einops import rearrange
from omegaconf import OmegaConf
from torchvision import transforms
import logging
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

# Project imports
from diffusion import create_diffusion
from models.unet import UNet3DConditionModel
from models.clip import TextEmbedder
from datasets import video_transforms
from pnp_utils import seed_everything, save_video_as_frames, load_video_frames
from seine_utils import mask_generation_before


def add_dict_to_yaml_file(file_path, key, value):
    data = {}

    # If the file already exists, load its contents into the data dictionary
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)

    # Add or update the key-value pair
    data[key] = value

    # Save the data back to the YAML file
    with open(file_path, "w") as file:
        yaml.dump(data, file)


def get_timesteps(scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start:]

    return timesteps, num_inference_steps - t_start


class SEINEDDIMInversionPipeline(nn.Module):
    def __init__(self, device, config):
        super().__init__()

        self.device = device

        # Create Unet
        # Modified from SEINE/models/unet.py/__init__.py
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            config.sd_path,
            subfolder="unet",
            use_concat=True,
        )

        # Load Unet
        # Modified from SEINE/with_mask_sample.py
        state_dict = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)["ema"]
        self.unet.load_state_dict(state_dict)
        self.unet.eval()
        self.unet.to(self.device)
        if config.enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()
        logger.info(f"loaded UNet from {config.ckpt_path}")

        # Load other parts
        # TODO: Use diffusion instead of scheduler: self.diffusion = create_diffusion(str(args.num_sampling_steps)
        self.vae = AutoencoderKL.from_pretrained(config.sd_path, subfolder="vae").to(self.device)
        self.text_encoder = TextEmbedder(config.sd_path, device=self.device).to(self.device)

        if config.use_fp16:
            logger.info("Using FP16")
            self.unet.to(dtype=torch.float16)
            self.vae.to(dtype=torch.float16)
            self.text_encoder.to(dtype=torch.float16)

        # Create scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            config.sd_path,
            subfolder="scheduler",
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
        )

        # Load source video frames
        self.paths, self.frames = load_video_frames(config.src_video_path, config.n_frame_to_invert)
        logger.debug(f"self.frames shape: {self.frames.shape}")

        # Preprocess video frames according to SEINE's video_transforms
        # Modified from SEINE/with_mask_sample.py
        self.transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeVideo(tuple(config.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.frames = self.transform_video(self.frames)  # [f, c, h, w]
        logger.debug(f"self.frames shape: {self.frames.shape}")

        # Encode video frames
        _frames = self.frames.to(torch.float16).to(self.device)
        self.latent_at_0 = self.vae.encode(_frames).latent_dist.sample().mul_(0.18215)
        logger.debug(f"self.latent_at_0 shape: {self.latent_at_0.shape}")
        self.latent_at_0 = rearrange(self.latent_at_0, "(b f) c h w -> b c f h w", b=1).contiguous()
        # [1, 4, 16, 40, 64]
        logger.debug(f"self.latent_at_0 shape: {self.latent_at_0.shape}")

    # Adopt from Lavie/pipline/pipeline_videogen.py
    @torch.no_grad()
    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = einops.rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = einops.rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
        video = ((video / 2 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().contiguous()
        return video

    # TODO: Use DDIMInverseScheduler instead of manually inverting the latents.
    @torch.no_grad()
    def ddim_inversion(
        self,
        cond,
        latent_frames,
        masked_video,
        mask,
        save_path,
        batch_size,
        save_latents=True,
        timesteps_to_save=None,
    ):
        timesteps = reversed(self.scheduler.timesteps)
        timesteps_to_save = timesteps_to_save if timesteps_to_save is not None else timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, latent_frames.shape[0], batch_size):
                x_batch = latent_frames[b : b + batch_size]
                mask_batch = mask[b : b + batch_size]
                masked_video_batch = masked_video[b : b + batch_size]
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]] if i > 0 else self.scheduler.final_alpha_cumprod
                )

                mu = alpha_prod_t**0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                # TODO: why in gaussian_diffusion.py, the input is [B, C, F, H, W]?
                input = torch.concat([x_batch, mask_batch, masked_video_batch], dim=1)  # [b, c, f, h, w]
                input = input.to(dtype=torch.float16)
                cond_batch = cond_batch.to(dtype=torch.float16)
                eps = self.unet(input, t, encoder_hidden_states=cond_batch).sample
                pred_x0 = (x_batch - sigma_prev * eps) / mu_prev
                latent_frames[b : b + batch_size] = mu * pred_x0 + sigma * eps

            if save_latents and t in timesteps_to_save:
                torch.save(
                    latent_frames,
                    os.path.join(save_path, "ddim_latents", f"ddim_latents_{t}.pt"),
                )
                logger.info(f"[INFO] saved noisy latents at t={t} to {save_path}/ddim_latents/ddim_latents_{t}.pt")
        # TODO: why we save t=999 latents?
        # torch.save(latent_frames, os.path.join(save_path, "latents", f"ddim_latents_{t}.pt"))
        return latent_frames

    # TODO: use DDIM scheduler to sample the latents.
    @torch.no_grad()
    def ddim_sample(self, x, cond, masked_video, mask, batch_size):
        timesteps = self.scheduler.timesteps
        for i, t in enumerate(tqdm(timesteps)):
            for b in range(0, x.shape[0], batch_size):
                x_batch = x[b : b + batch_size]
                mask_batch = mask[b : b + batch_size]
                masked_video_batch = masked_video[b : b + batch_size]
                cond_batch = cond.repeat(x_batch.shape[0], 1, 1)

                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i + 1]]
                    if i < len(timesteps) - 1
                    else self.scheduler.final_alpha_cumprod
                )
                mu = alpha_prod_t**0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                mu_prev = alpha_prod_t_prev**0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                # TODO: why in gaussian_diffusion.py, the input is [B, C, F, H, W]?
                input = torch.concat([x_batch, mask_batch, masked_video_batch], dim=1)  # [b, c, f, h, w]
                input = input.to(dtype=torch.float16)
                cond_batch = cond_batch.to(dtype=torch.float16)
                eps = self.unet(input, t, encoder_hidden_states=cond_batch).sample

                pred_x0 = (x_batch - sigma * eps) / mu
                x[b : b + batch_size] = mu_prev * pred_x0 + sigma_prev * eps
        return x

    @torch.no_grad()
    def extract_ddim_latents(self, config, timesteps_to_save, save_path):
        # Generate masked video
        # Modified from SEINE/with_mask_sample.py
        first_frame = self.frames[0].unsqueeze(0)  # [1, 3, 320, 512]
        # Note that the first frame is already normalized.
        padded_video_frames = [first_frame]
        num_zeros = len(self.frames) - 1
        for i in range(num_zeros):
            zeros = torch.zeros_like(first_frame)
            padded_video_frames.append(zeros)
        padded_video_frames = torch.cat(padded_video_frames, dim=0)  # [16, 3, 320, 512] f, c, h, w
        logger.debug(f"padded_video_frames shape: {padded_video_frames.shape}")
        video_input = padded_video_frames.to(self.device).unsqueeze(0)  # b,f,c,h,w # [1, 16, 3, 320, 512]
        mask = mask_generation_before("first1", video_input.shape, video_input.dtype, self.device)  # b,f,c,h,w
        masked_video = video_input * (mask == 0)
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
        logger.debug(f"video_input shape: {video_input.shape}")
        logger.debug(f"masked_video shape: {masked_video.shape}")
        logger.debug(f"mask shape: {mask.shape}")

        # Encode masked video
        # Modified from SEINE/with_mask_sample.py
        b, f, c, h, w = masked_video.shape
        latent_h = config.image_size[0] // 8
        latent_w = config.image_size[1] // 8
        masked_video = rearrange(masked_video, "b f c h w -> (b f) c h w").contiguous()
        masked_video = self.vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
        masked_video = rearrange(masked_video, "(b f) c h w -> b c f h w", b=b).contiguous()
        mask = torch.nn.functional.interpolate(mask[:, :, 0, :], size=(latent_h, latent_w)).unsqueeze(1)
        # [1, 1, 16, 40, 64]
        logger.debug("After encoding")
        logger.debug(f"masked_video shape: {masked_video.shape}")
        logger.debug(f"mask shape: {mask.shape}")

        # Set scheduler timesteps
        self.scheduler.set_timesteps(config.n_steps)
        logger.debug(f"self.scheduler.timesteps: {self.scheduler.timesteps}")

        # Get text embeddings
        prompt = config.inversion_prompt
        text_embeddings = self.text_encoder(text_prompts=prompt, train=False)
        logger.debug(f"text_embeddings shape: {text_embeddings.shape}")

        # Invert latents
        ddim_latents_at_T = self.ddim_inversion(
            cond=text_embeddings,
            latent_frames=self.latent_at_0,
            masked_video=masked_video,
            mask=mask,
            save_path=save_path,
            batch_size=config.batch_size,
            save_latents=True,
            timesteps_to_save=timesteps_to_save,
        )
        ddim_reconstruct_latents_at_0 = self.ddim_sample(
            x=ddim_latents_at_T,
            cond=text_embeddings,
            masked_video=masked_video,
            mask=mask,
            batch_size=config.batch_size,
        ) #
        logger.debug(f"ddim_reconstruct_latents_at_0 shape: {ddim_reconstruct_latents_at_0.shape}")
        rgb_reconstruction = self.decode_latents(ddim_reconstruct_latents_at_0)
        logger.debug(f"rgb_reconstruction shape: {rgb_reconstruction.shape}")
        return rgb_reconstruction


def main(config):
    assert config.model_name == "seine", f"model_name {config.model_name} not supported."

    # TODO: Double check this scheduler config, whether this is consistent with SEINE's scheduler.
    toy_scheduler = DDIMScheduler.from_pretrained(
        config.sd_path,
        subfolder="scheduler",
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
    )
    toy_scheduler.set_timesteps(config.n_save_steps)
    timesteps_to_save, num_inference_steps = get_timesteps(
        toy_scheduler, num_inference_steps=config.n_save_steps, strength=1.0
    )
    logger.info(f"toy_scheduler.timesteps: {toy_scheduler.timesteps}")
    logger.info(f"timesteps_to_save: {timesteps_to_save}")
    logger.info(f"num_inference_steps: {num_inference_steps}")

    save_path = os.path.join(
        config.output_dir,
        config.model_name,
        Path(config.src_video_path).stem,
        f"steps_{config.n_steps}",
        f"nframes_{config.n_frame_to_invert}",
    )
    logger.info(f"save_path: {save_path}")
    os.makedirs(os.path.join(save_path, "ddim_latents"), exist_ok=True)

    # Save inversion prompt in a yaml file
    add_dict_to_yaml_file(
        file_path=os.path.join(save_path, "inversion_prompts.yaml"),
        key=Path(config.src_video_path).stem,
        value=config.inversion_prompt,
    )

    # Save config in a yaml file
    with open(os.path.join(save_path, "config.yaml"), "w") as file:
        yaml.dump(OmegaConf.to_container(config), file)

    # Main pipeline
    seine_ddim_inversion_pipeline = SEINEDDIMInversionPipeline(device, config)
    recon_frames = seine_ddim_inversion_pipeline.extract_ddim_latents(config, timesteps_to_save, save_path)
    recon_frames = einops.rearrange(recon_frames, "1 f h w c -> f c h w")

    # Save reconstructed frames and video
    recon_frames_dir = os.path.join(save_path, "recon_frames")
    os.makedirs(recon_frames_dir, exist_ok=True)
    for i, frame in enumerate(recon_frames):
        T.ToPILImage()(frame).save(os.path.join(recon_frames_dir, f"{i:05d}.png"))
    frames = (recon_frames).to(torch.uint8).cpu().permute(0, 2, 3, 1)
    write_video(os.path.join(save_path, f"inverted.mp4"), frames, fps=8)
    logger.info(f"Saved reconstructed frames and video to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ddim_inversion.yaml")
    parser.add_argument("--video_path", type=str, required=False, help="Path to the video to invert.")
    parser.add_argument("--gpu", type=int, required=False, help="GPU number to use.")
    parser.add_argument("--width", type=int, required=False, help="")
    parser.add_argument("--height", type=int, required=False, help="")

    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    
    # Overwrite config with command line arguments
    if args.video_path is not None:
        config.src_video_path = args.video_path
    if args.gpu is not None:
        config.device = f"cuda:{args.gpu}"
    if args.width is not None and args.height is not None:
        config.image_size = [args.height, args.width]

    # Set up logging
    transformers_logging.set_verbosity_error()
    logging_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"config: {config}")

    # TODO: better to avoid save then load
    assert os.path.exists(config.src_video_path), f"src_video_path {config.src_video_path} does not exist."
    save_video_as_frames(config.src_video_path, img_size=(config.image_size[1], config.image_size[0]))  # (w, h)
    config.src_video_path = os.path.join(Path(config.src_video_path).parent, Path(config.src_video_path).stem)

    device = torch.device(config.device)
    seed_everything(config.seed)
    main(config)
