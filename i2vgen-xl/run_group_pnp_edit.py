import os
import sys
from pathlib import Path
import torch
import argparse
import logging
from omegaconf import OmegaConf
from PIL import Image
import json

# HF imports
from diffusers import (
    DDIMInverseScheduler,
    DDIMScheduler,
)
from diffusers.utils import load_image, export_to_video, export_to_gif

# Project imports
from utils import (
    seed_everything,
    load_video_frames,
    convert_video_to_frames,
    load_ddim_latents_at_T,
    load_ddim_latents_at_t,
)
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
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

    logger = logging.getLogger(__name__)
    logger.debug(f"conv_injection_t: {conv_injection_t}")
    logger.debug(f"spatial_attn_qk_injection_t: {spatial_attn_qk_injection_t}")
    logger.debug(f"temp_attn_qk_injection_t: {temp_attn_qk_injection_t}")
    logger.debug(f"conv_injection_timesteps: {conv_injection_timesteps}")
    logger.debug(f"spatial_attn_qk_injection_timesteps: {spatial_attn_qk_injection_timesteps}")
    logger.debug(f"temp_attn_qk_injection_timesteps: {temp_attn_qk_injection_timesteps}")


def main(template_config, configs_list):
    # Initialize the pipeline
    pipe = I2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)

    # Initialize the DDIM scheduler
    ddim_scheduler = DDIMScheduler.from_pretrained(
        "ali-vilab/i2vgen-xl",
        subfolder="scheduler",
    )

    for config_entry in configs_list:
        if config_entry["active"] == False:
            logger.info(f"Skipping config_entry: {config_entry}")
            continue
        logger.info(f"Processing config_entry: {config_entry}")

        # Override the config with the data_meta_entry
        config = OmegaConf.merge(template_config, OmegaConf.create(config_entry))

        # Update the related paths to absolute paths
        config.video_path = os.path.join(config.video_dir, config.video_name + ".mp4")
        config.video_frames_path = os.path.join(config.video_dir, config.video_name)
        config.edited_first_frame_path = os.path.join(config.data_dir, config.edited_first_frame_path)
        logger.info(f"config: {OmegaConf.to_yaml(config)}")

        # Check if there are fields contain "ReplaceMe"
        for k, v in config.items():
            if "ReplaceMe" in str(v):
                logger.error(f"Field {k} contains 'ReplaceMe'")
                continue

        # This is the same as run_pnp_edit.py
        # Load first frame and source frames
        try:
            logger.info(f"Loading frames from: {config.video_frames_path}")
            _, frame_list = load_video_frames(config.video_frames_path, config.n_frames, config.image_size)
        except:
            logger.error(f"Failed to load frames from: {config.video_frames_path}")
            logger.info(f"Converting mp4 video to frames: {config.video_path}")
            frame_list = convert_video_to_frames(config.video_path, config.image_size, save_frames=True)
            frame_list = frame_list[: config.n_frames]  # 16 frames for img2vid
            logger.debug(f"len(frame_list): {len(frame_list)}")
        src_frame_list = frame_list
        src_1st_frame = src_frame_list[0]  # Is a PIL image

        # Load the edited first frame
        edited_1st_frame = load_image(config.edited_first_frame_path)
        edited_1st_frame = edited_1st_frame.resize(config.image_size, resample=Image.Resampling.LANCZOS)

        # Load the initial latents at t
        ddim_init_latents_t_idx = config.ddim_init_latents_t_idx
        ddim_scheduler.set_timesteps(config.n_steps)
        logger.info(f"ddim_scheduler.timesteps: {ddim_scheduler.timesteps}")
        ddim_latents_at_t = load_ddim_latents_at_t(
            ddim_scheduler.timesteps[ddim_init_latents_t_idx], ddim_latents_path=config.ddim_latents_path
        )
        logger.debug(f"ddim_scheduler.timesteps[t_idx]: {ddim_scheduler.timesteps[ddim_init_latents_t_idx]}")
        logger.debug(f"ddim_latents_at_t.shape: {ddim_latents_at_t.shape}")

        # Blend the latents
        random_latents = torch.randn_like(ddim_latents_at_t)
        logger.info(f"Blending random_ratio (1 means random latent): {config.random_ratio}")
        mixed_latents = random_latents * config.random_ratio + ddim_latents_at_t * (1 - config.random_ratio)

        # Init Pnp
        init_pnp(pipe, ddim_scheduler, config)

        # Edit video
        pipe.register_modules(scheduler=ddim_scheduler)
        edited_video = pipe.sample_with_pnp(
            prompt=config.editing_prompt,
            image=edited_1st_frame,
            height=config.image_size[1],
            width=config.image_size[0],
            num_frames=config.n_frames,
            num_inference_steps=config.n_steps,
            guidance_scale=config.cfg,
            negative_prompt=config.editing_negative_prompt,
            target_fps=config.target_fps,
            latents=mixed_latents,
            generator=torch.manual_seed(config.seed),
            return_dict=True,
            ddim_init_latents_t_idx=ddim_init_latents_t_idx,
            ddim_inv_latents_path=config.ddim_latents_path,
            ddim_inv_prompt=config.ddim_inv_prompt,
            ddim_inv_1st_frame=src_1st_frame,
        ).frames[0]

        # Save video
        # Add the config to the output_dir, TODO: make this more elegant
        config_suffix = (
            "ddim_init_latents_t_idx_"
            + str(ddim_init_latents_t_idx)
            + "_nsteps_"
            + str(config.n_steps)
            + "_cfg_"
            + str(config.cfg)
            + "_pnpf"
            + str(config.pnp_f_t)
            + "_pnps"
            + str(config.pnp_spatial_attn_t)
            + "_pnpt"
            + str(config.pnp_temp_attn_t)
        )
        output_dir = os.path.join(config.output_dir, config_suffix)
        os.makedirs(output_dir, exist_ok=True)
        edited_video = [frame.resize(config.image_size, resample=Image.LANCZOS) for frame in edited_video]
        # Downsampling the video for space saving
        # edited_video = [frame.resize((512, 512), resample=Image.LANCZOS) for frame in edited_video]
        # if config.pnp_f_t == 0.0 and config.pnp_spatial_attn_t == 0.0 and config.pnp_temp_attn_t == 0.0:
        #     edited_video_file_name = "ddim_edit"
        # else:
        #     edited_video_file_name = "pnp_edit"
        edited_video_file_name = "video"
        export_to_video(edited_video, os.path.join(output_dir, f"{edited_video_file_name}.mp4"), fps=config.target_fps)
        export_to_gif(edited_video, os.path.join(output_dir, f"{edited_video_file_name}.gif"))
        logger.info(f"Saved video to: {os.path.join(output_dir, f'{edited_video_file_name}.mp4')}")
        logger.info(f"Saved gif to: {os.path.join(output_dir, f'{edited_video_file_name}.gif')}")
        for i, frame in enumerate(edited_video):
            frame.save(os.path.join(output_dir, f"{edited_video_file_name}_{i:05d}.png"))
            logger.info(f"Saved frames to: {os.path.join(output_dir, f'{edited_video_file_name}_{i:05d}.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_config", type=str, default="./configs/group_pnp_edit/template.yaml")
    parser.add_argument(
        "--configs_json", type=str, default="./configs/group_config.json"
    )  # This is going to override the template_config

    args = parser.parse_args()
    template_config = OmegaConf.load(args.template_config)

    # Set up logging
    logging_level = logging.DEBUG if template_config.debug else logging.INFO
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"template_config: {OmegaConf.to_yaml(template_config)}")

    # Load data jsonl into list
    configs_json = args.configs_json
    assert Path(configs_json).exists()
    with open(configs_json, "r") as file:
        configs_list = json.load(file)
    logger.info(f"Loaded {len(configs_list)} configs from {configs_json}")

    # Set up device and seed
    device = torch.device(template_config.device)
    torch.set_grad_enabled(False)
    seed_everything(template_config.seed)
    main(template_config, configs_list)
