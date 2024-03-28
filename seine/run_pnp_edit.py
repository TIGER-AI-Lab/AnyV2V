import glob
import os
import numpy as np
from pathlib import Path
import logging as logger

import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
import einops
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
import logging
from torchvision import transforms
from diffusers import DDIMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import logging as transformers_logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, DDPMScheduler

# Project imports
from diffusion import create_diffusion
from models.unet import UNet3DConditionModel
from models.clip import TextEmbedder
from datasets import video_transforms
from pnp_utils import (
    seed_everything,
    save_video,
    save_video_as_frames,
    load_video_frames,
    load_ddim_latents_at_T,
    load_ddim_latents_at_t,
    register_time,
    register_conv_injection,
    register_spatial_attention_pnp,
    register_cross_attention_pnp,
    register_temp_attention_pnp,
)
from seine_utils import mask_generation_before


class SEINEPnPPipeline(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.config = config
        self.device = device

        # Create Unet
        # Modified from SEINE/models/unet.py/__init__.py
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            config.sd_path,
            subfolder="unet",
            use_concat=True,
        )
        self.latent_h = config.image_size[0] // 8
        self.latent_w = config.image_size[1] // 8
        self.latent_c = 4
        self.n_frames = config.n_frames

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
        if config.sample_method == "ddim":
            logger.info("Using DDIM Sampler")
            self.scheduler = DDIMScheduler.from_pretrained(
                config.sd_path,
                subfolder="scheduler",
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
            )
        elif config.sample_method == "ddpm":
            logger.info("Using DDPM Sampler")
            self.scheduler = DDPMScheduler.from_pretrained(
                config.sd_path,
                subfolder="scheduler",
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                beta_schedule=config.beta_schedule,
            )
        else:
            raise NotImplementedError

        # Load DDIM inversion latents and prompt
        self.ddim_latents_path = self.get_ddim_latents_path()
        ddim_latents_at_T = load_ddim_latents_at_T(self.ddim_latents_path)
        self.ddim_latents_at_T = ddim_latents_at_T.to(torch.float16).to(self.device)
        self.ddim_inversion_prompt = self.get_ddim_inversion_prompt()
        logger.info(f"ddim_inversion_prompt: {self.ddim_inversion_prompt}")

        # Load edited first frame
        self.edited_1st_frame = torch.as_tensor(
            np.array(Image.open(config.edited_first_frame_path).convert('RGB'), dtype=np.uint8, copy=True)
        ).unsqueeze(0)
        logger.debug(f"edited_1st_frame shape: {self.edited_1st_frame.shape}")

        # Load source video frames
        self.src_video_paths, self.src_video_frames = load_video_frames(
            os.path.join(Path(config.src_video_path).parent, Path(config.src_video_path).stem), config.n_frame_inverted
        )
        logger.debug(f"self.src_video_frames shape: {self.src_video_frames.shape}")

        # Preprocess video frames according to SEINE's video_transforms
        # Modified from SEINE/with_mask_sample.py
        self.transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeVideo(tuple(config.image_size)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.src_video_frames = self.transform_video(self.src_video_frames)  # [f, c, h, w]
        logger.debug(f"self.src_video_frames shape: {self.src_video_frames.shape}")

    def get_ddim_inversion_prompt(self):
        inv_prompts_path = os.path.join(str(Path(self.ddim_latents_path).parent), "inversion_prompts.yaml")
        with open(inv_prompts_path, "r") as f:
            inv_prompts = yaml.safe_load(f)
        return inv_prompts[f"{Path(config.src_video_path).stem}"]

    def get_ddim_latents_path(self):
        ddim_latents_path = os.path.join(
            config.ddim_inversion_dir,
            config.model_name,
            Path(config.src_video_path).stem,
            f"steps_{config.n_ddim_inversion_steps}",
        )
        ddim_latents_path = [x for x in glob.glob(f"{ddim_latents_path}/*") if "." not in Path(x).name]
        n_frames = [
            int([x for x in ddim_latents_path[i].split("/") if "nframes" in x][0].split("_")[1])
            for i in range(len(ddim_latents_path))
        ]
        ddim_latents_path = ddim_latents_path[np.argmax(n_frames)]
        self.config.n_frames = min(max(n_frames), config.n_frames)
        if self.config.n_frames % self.config.batch_size != 0:
            # make n_frames divisible by batch_size
            self.config.n_frames = self.config.n_frames - (self.config.n_frames % self.config.batch_size)
        return os.path.join(ddim_latents_path, "ddim_latents")

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

    @torch.no_grad()
    def denoise_step(self, x, mask, masked_video, masked_src_video, t):
        if self.config.enable_pnp:
            # register the time step and features in pnp injection modules
            if config.sample_method == "ddim":
                ddim_latents_at_t = load_ddim_latents_at_t(t, self.ddim_latents_path).to(self.device)
            elif config.sample_method == "ddpm":
                ddim_latents_at_t = load_ddim_latents_at_t(t+1, self.ddim_latents_path).to(self.device)
            else:
                raise NotImplementedError
            ddim_feature_branch_input = torch.concat(
                [ddim_latents_at_t, mask, masked_src_video], dim=1
            )  # [b, c, f, h, w]
            _input = torch.concat([x, mask, masked_video], dim=1)  # [b, c, f, h, w]
            model_input = torch.cat([ddim_feature_branch_input] + ([_input] * 2))
            text_embed_input = torch.cat([self.ddim_inversion_embeds, self.cond_embeds, self.uncond_embeds], dim=0)

            register_time(self, t.item())

            logger.debug(f"ddim_latents_at_t shape: {ddim_latents_at_t.shape}")
        else:
            _input = torch.cat([x, mask, masked_video], dim=1)
            model_input = torch.cat([_input] * 2)
            text_embed_input = torch.cat([self.cond_embeds, self.uncond_embeds], dim=0)

        logger.debug(f"Text embed input shape: {text_embed_input.shape}")
        logger.debug(f"x shape: {x.shape}")
        logger.debug(f"model input shape: {model_input.shape}")

        # apply the denoising network
        noise_pred = self.unet(model_input, t, encoder_hidden_states=text_embed_input)["sample"]

        # perform guidance
        # TODO: In SEINE's codebase, the guidance is only applied for the first 3 channels of the noise_pred
        if self.config.enable_pnp:
            _, noise_pred_cond, noise_pred_uncond = noise_pred.chunk(3)
        else:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        if self.config.cfg_scale > 1.0:
            noise_pred = noise_pred_uncond + self.config.cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            noise_pred = noise_pred_cond
            logger.debug(f"cfg is disabled")

        # compute the denoising step with the reference model
        x_tm1 = self.scheduler.step(noise_pred, t, x)["prev_sample"]
        return x_tm1

    def init_pnp(self, override_dict=None):

        if override_dict is not None:
            self.config.pnp_f_t = override_dict["conv_inject"]
            self.config.pnp_spatial_attn_t = override_dict["attn_inject"]
            self.config.pnp_cross_attn_t = override_dict["cross_inject"]
            self.config.pnp_temp_attn_t = override_dict["temp_inject"]

        conv_injection_t = int(self.config.n_steps * self.config.pnp_f_t)
        spatial_attn_qk_injection_t = int(self.config.n_steps * self.config.pnp_spatial_attn_t)
        cross_attn_qk_injection_t = int(self.config.n_steps * self.config.pnp_cross_attn_t)
        temp_attn_qk_injection_t = int(self.config.n_steps * self.config.pnp_temp_attn_t)
        logger.debug(f"conv_injection_t: {conv_injection_t}")
        logger.debug(f"spatial_attn_qk_injection_t: {spatial_attn_qk_injection_t}")
        logger.debug(f"cross_attn_qk_injection_t: {cross_attn_qk_injection_t}")
        logger.debug(f"temp_attn_qk_injection_t: {temp_attn_qk_injection_t}")
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        self.spatial_attn_qk_injection_timesteps = (
            self.scheduler.timesteps[:spatial_attn_qk_injection_t] if spatial_attn_qk_injection_t >= 0 else []
        )
        self.cross_attn_qk_injection_timesteps = (
            self.scheduler.timesteps[:cross_attn_qk_injection_t] if cross_attn_qk_injection_t >= 0 else []
        )
        self.temp_attn_qk_injection_timesteps = (
            self.scheduler.timesteps[:temp_attn_qk_injection_t] if temp_attn_qk_injection_t >= 0 else []
        )
        logger.debug(f"conv_injection_timesteps: {self.conv_injection_timesteps}")
        logger.debug(f"spatial_attn_qk_injection_timesteps: {self.spatial_attn_qk_injection_timesteps}")
        logger.debug(f"cross_attn_qk_injection_timesteps: {self.cross_attn_qk_injection_timesteps}")
        logger.debug(f"temp_attn_qk_injection_timesteps: {self.temp_attn_qk_injection_timesteps}")
        register_conv_injection(self, self.conv_injection_timesteps)
        register_spatial_attention_pnp(self, self.spatial_attn_qk_injection_timesteps)
        register_cross_attention_pnp(self, self.cross_attn_qk_injection_timesteps)
        register_temp_attention_pnp(self, self.temp_attn_qk_injection_timesteps)

    def compute_masked_video_latents_at_0(self, config, video_input):
        # Generate mask
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
        masked_video = rearrange(masked_video, "b f c h w -> (b f) c h w").contiguous()
        masked_video_latent = self.vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
        masked_video_latent = rearrange(masked_video_latent, "(b f) c h w -> b c f h w", b=b).contiguous()
        mask = torch.nn.functional.interpolate(mask[:, :, 0, :], size=(self.latent_h, self.latent_w)).unsqueeze(1)
        # [1, 1, 16, 40, 64]
        logger.debug("After encoding")
        logger.debug(f"masked_video_latent shape: {masked_video_latent.shape}")
        logger.debug(f"mask shape: {mask.shape}")
        return mask, masked_video_latent

    @torch.no_grad()
    def edit_video(self, config):
        # Generate masked 1st edited video
        # Modified from SEINE/with_mask_sample.py
        first_frame = self.edited_1st_frame
        padded_video_frames = [first_frame]
        num_zeros = len(self.src_video_frames) - 1
        for i in range(num_zeros):
            zeros = torch.zeros_like(first_frame)
            padded_video_frames.append(zeros)
        padded_video_frames = torch.cat(padded_video_frames, dim=0).permute(0, 3, 1, 2)  # [16, 3, 320, 512] f, c, h, w
        logger.debug(f"padded_video_frames shape: {padded_video_frames.shape}")

        # Apply transforms
        padded_video_frames = self.transform_video(padded_video_frames)
        video_input = padded_video_frames.to(self.device).unsqueeze(0)  # b,f,c,h,w # [1, 16, 3, 320, 512]
        mask, masked_1st_frame_edited_video = self.compute_masked_video_latents_at_0(config, video_input)

        # Generate masked src video
        first_frame = self.src_video_frames[0].unsqueeze(0)  # [1, 3, 320, 512]
        # Note that the first frame is already normalized.
        padded_video_frames = [first_frame]
        num_zeros = len(self.src_video_frames) - 1
        for i in range(num_zeros):
            zeros = torch.zeros_like(first_frame)
            padded_video_frames.append(zeros)
        padded_video_frames = torch.cat(padded_video_frames, dim=0)  # [16, 3, 320, 512] f, c, h, w
        video_input = padded_video_frames.to(self.device).unsqueeze(0)  # b,f,c,h,w # [1, 16, 3, 320, 512]
        _, masked_src_video = self.compute_masked_video_latents_at_0(config, video_input)


        # Load initial latents at T
        if config.init_with_ddim_inversion:
            x_T = self.ddim_latents_at_T.to(self.device)
            logger.info(f"Init with ddim inversion")
        else:
            x_T = torch.randn(1, self.latent_c, config.n_frames, self.latent_h, self.latent_w, dtype=torch.float16).to(self.device)
            logger.info(f"Init with random noise")

        # Compute text embeddings
        if config.enable_pnp:
            prompt_all = [self.ddim_inversion_prompt] + [config.prompt] + [config.negative_prompt]
            text_prompt = self.text_encoder(text_prompts=prompt_all, train=False)
            logger.debug(f"text_prompt shape: {text_prompt.shape}")
            self.ddim_inversion_embeds, self.cond_embeds, self.uncond_embeds = text_prompt.chunk(3, dim=0)
        else:
            text_prompt = self.text_encoder(text_prompts=[config.prompt, config.negative_prompt], train=False)
            self.cond_embeds, self.uncond_embeds = text_prompt.chunk(2, dim=0)
            logger.debug(f"text_prompt shape: {text_prompt.shape}")

        if config.use_fp16:
            x_T = x_T.to(torch.float16)
            mask = mask.to(torch.float16)
            masked_1st_frame_edited_video = masked_1st_frame_edited_video.to(torch.float16)
            masked_src_video = masked_src_video.to(torch.float16)

        x_0 = self.sample_loop(
            x_T.to(self.device),
            mask.to(self.device),
            masked_1st_frame_edited_video.to(self.device),
            masked_src_video.to(self.device),
        )  # self.ddim_encodes [1, 4, 16, 40, 64]

        edited_frames = self.decode_latents(x_0)
        return edited_frames

    def sample_loop(self, x, mask, masked_video, masked_src_video):
        batch_size = self.config.batch_size
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            denoised_latents = []
            for i, b in enumerate(range(0, len(x), batch_size)):
                denoised_latents.append(
                    self.denoise_step(
                        x[b : b + batch_size],
                        mask[b : b + batch_size],
                        masked_video[b : b + batch_size],
                        masked_src_video[b : b + batch_size],
                        t,
                    )
                )
            x = torch.cat(denoised_latents)
        return x


def main(config):
    # Set up output path
    save_path = os.path.join(
        config.output_dir,
        config.model_name,
        Path(config.src_video_path).stem,
        config.prompt.replace(" ", "_")[:240],  # Connect the prompt using _
        f"cfg{config.cfg_scale}_f{config.pnp_f_t}_spa{config.pnp_spatial_attn_t}_cro{config.pnp_cross_attn_t}_tmp{config.pnp_temp_attn_t}_stp{config.n_steps}"
        if config.enable_pnp
        else "",
    )
    config.output_path = save_path

    # Main pipeline
    logger.info(f"save_path: {save_path}")
    seine_pnp_pipeline = SEINEPnPPipeline(device, config)
    seine_pnp_pipeline.scheduler.set_timesteps(config.n_steps)
    logger.debug(f"seine_pnp_pipeline.scheduler.timesteps: {seine_pnp_pipeline.scheduler.timesteps}")
    if config.enable_pnp:
        seine_pnp_pipeline.scheduler.set_timesteps(config.n_steps)
        seine_pnp_pipeline.init_pnp()
    edited_frames = seine_pnp_pipeline.edit_video(config)
    logger.debug(f"edited_frames shape: {edited_frames.shape}")
    edited_frames = einops.rearrange(edited_frames, "1 f h w c -> f c h w")
    logger.debug(f"edited_frames shape: {edited_frames.shape}")

    # Save video
    os.makedirs(f"{config.output_path}/img_ode", exist_ok=True)
    for i in range(len(edited_frames)):
        T.ToPILImage()(edited_frames[i]).save(f"{config.output_path}/img_ode/%05d.png" % i)
    save_video(edited_frames, f"{config.output_path}/video_pnp_fps_8.mp4", fps=8)
    logger.info(f"Saved video to {config.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/pnp_edit.yaml")
    parser.add_argument("optional_args", nargs='*', default=[])
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    # Overwrite config with command line arguments
    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    # Set up logging
    transformers_logging.set_verbosity_error()
    logging_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"config: {config}")

    # Set up device and seed
    device = torch.device(config.device)
    seed_everything(config.seed)
    main(config)
