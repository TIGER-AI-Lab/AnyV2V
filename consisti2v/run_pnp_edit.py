import os
import sys
from pathlib import Path
import torch
import argparse
import logging
from omegaconf import OmegaConf
from PIL import Image

# HF imports
from diffusers import DDIMScheduler

# Project imports
from utils import (
    seed_everything,
    load_video_frames,
    convert_video_to_frames,
    load_ddim_latents_at_T,
    load_ddim_latents_at_t,
)
from consisti2v.pipelines.pipeline_video_editing import ConditionalVideoEditingPipeline
from consisti2v.utils.util import save_videos_grid
from pnp_utils import (
    register_time,
    register_conv_injection,
    register_spatial_attention_pnp,
    register_temp_attention_pnp,
)


def init_pnp(pipe, scheduler, config):
    conv_injection_t = int(config.n_steps * config.pnp_f_t)
    spatial_attn_qk_injection_t = int(config.n_steps * config.pnp_spatial_attn_t)
    temp_attn_qk_injection_t = int(config.n_steps * config.pnp_temp_attn_t)
    conv_injection_timesteps = scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
    spatial_attn_qk_injection_timesteps = (
        scheduler.timesteps[:spatial_attn_qk_injection_t] if spatial_attn_qk_injection_t >= 0 else []
    )
    temp_attn_qk_injection_timesteps = (
        scheduler.timesteps[:temp_attn_qk_injection_t] if temp_attn_qk_injection_t >= 0 else []
    )
    register_conv_injection(pipe, conv_injection_timesteps)
    register_spatial_attention_pnp(pipe, spatial_attn_qk_injection_timesteps)
    register_temp_attention_pnp(pipe, temp_attn_qk_injection_timesteps)

    logger.debug(f"conv_injection_t: {conv_injection_t}")
    logger.debug(f"spatial_attn_qk_injection_t: {spatial_attn_qk_injection_t}")
    logger.debug(f"temp_attn_qk_injection_t: {temp_attn_qk_injection_t}")
    logger.debug(f"conv_injection_timesteps: {conv_injection_timesteps}")
    logger.debug(f"spatial_attn_qk_injection_timesteps: {spatial_attn_qk_injection_timesteps}")
    logger.debug(f"temp_attn_qk_injection_timesteps: {temp_attn_qk_injection_timesteps}")


def main(config):
    # Initialize the pipeline
    pipe = ConditionalVideoEditingPipeline.from_pretrained(
        "TIGER-Lab/ConsistI2V",
        torch_dtype=torch.float16,
    )
    pipe.to(device)

    # Initialize the DDIM scheduler
    ddim_scheduler = DDIMScheduler.from_pretrained(
        "TIGER-Lab/ConsistI2V",
        subfolder="scheduler",
    )

    # Load first frame and source frames
    if config.video_path:
        frame_list = convert_video_to_frames(config.video_path, config.image_size, save_frames=True)
        frame_list = frame_list[: config.n_frames]  # 16 frames for img2vid
        logger.debug(f"len(frame_list): {len(frame_list)}")
        video_name = Path(config.video_path).stem
        video_dir = Path(config.video_path).parent
        config.video_frames_path = f"{video_dir}/{video_name}"
    elif config.video_frames_path:
        _, frame_list = load_video_frames(config.video_frames_path, config.n_frames)
    else:
        raise ValueError("Please provide either video_path or video_frames_path")
    src_frame_list = frame_list
    src_1st_frame = os.path.join(config.video_frames_path, '00000.png')

    # Load the edited first frame
    edited_1st_frame = config.edited_first_frame_path

    # Load the initial latents at t
    ddim_init_latents_t_idx = config.ddim_init_latents_t_idx
    ddim_scheduler.set_timesteps(config.n_steps)
    logger.info(f"ddim_scheduler.timesteps: {ddim_scheduler.timesteps}")
    ddim_latents_path = os.path.join(config.ddim_latents_path, config.exp_name)
    ddim_latents_at_t = load_ddim_latents_at_t(
        ddim_scheduler.timesteps[ddim_init_latents_t_idx], ddim_latents_path=ddim_latents_path
    )
    logger.debug(f"ddim_scheduler.timesteps[t_idx]: {ddim_scheduler.timesteps[ddim_init_latents_t_idx]}")
    logger.debug(f"ddim_latents_at_t.shape: {ddim_latents_at_t.shape}")

    # Blend the latents
    random_latents = torch.randn_like(ddim_latents_at_t)
    random_ratio = config.blend_ratio
    mixed_latents = random_latents * random_ratio + ddim_latents_at_t * (1 - random_ratio)

    # Init Pnp
    init_pnp(pipe, ddim_scheduler, config)

    # Edit video
    pipe.register_modules(scheduler=ddim_scheduler)
    edited_video = pipe.sample_with_pnp(
        prompt=config.editing_prompt,
        first_frame_paths=edited_1st_frame,
        height=config.image_size[1],
        width=config.image_size[0],
        video_length=config.n_frames,
        num_inference_steps=config.n_steps,
        guidance_scale_txt=config.cfg_txt,
        guidance_scale_img=config.cfg_img,
        negative_prompt=config.editing_negative_prompt,
        frame_stride=config.frame_stride,
        latents=mixed_latents,
        generator=torch.manual_seed(config.seed),
        return_dict=True,
        ddim_init_latents_t_idx=ddim_init_latents_t_idx,
        ddim_inv_latents_path=ddim_latents_path,
        ddim_inv_prompt=config.ddim_inv_prompt,
        ddim_inv_1st_frame_path=src_1st_frame,
    ).videos

    # Save video
    os.makedirs(config.output_dir, exist_ok=True)
    # Downsampling the video for space saving
    save_videos_grid(edited_video, os.path.join(config.output_dir, config.editing_prompt, "video.gif"), fps=8, format="gif")
    save_videos_grid(edited_video, os.path.join(config.output_dir, config.editing_prompt, "video.mp4"), fps=8, format="mp4")
    logger.info(f"Saved edited video to {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/pnp_edit.yaml")
    parser.add_argument("optional_args", nargs='*', default=[])
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    # Set up logging
    logging_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"config: {OmegaConf.to_yaml(config)}")

    # Set up device and seed
    device = torch.device(config.device)
    seed_everything(config.seed)
    main(config)