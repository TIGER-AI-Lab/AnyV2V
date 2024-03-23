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


def ddim_inversion(config, first_frame, frame_list, pipe: I2VGenXLPipeline, inverse_scheduler, g):
    pipe.scheduler = inverse_scheduler
    video_latents_at_0 = pipe.encode_vae_video(
        frame_list,
        device=pipe._execution_device,
        height=config.image_size[1],
        width=config.image_size[0],
    )
    ddim_latents = pipe.invert(
        prompt=config.prompt,
        image=first_frame,
        height=config.image_size[1],
        width=config.image_size[0],
        num_frames=config.n_frames,
        num_inference_steps=config.n_steps,
        guidance_scale=config.cfg,
        negative_prompt=config.negative_prompt,
        target_fps=config.target_fps,
        latents=video_latents_at_0,
        generator=g,  # TODO: this is not correct
        return_dict=False,
        output_dir=config.output_dir,
    )  # [b, num_inference_steps, c, num_frames, h, w]
    logger.debug(f"ddim_latents.shape: {ddim_latents.shape}")
    ddim_latents = ddim_latents[0]  # [num_inference_steps, c, num_frames, h, w]
    return ddim_latents


def ddim_sampling(
    config, first_frame, ddim_latents_at_T, pipe: I2VGenXLPipeline, ddim_scheduler, ddim_init_latents_t_idx, g
):
    pipe.scheduler = ddim_scheduler
    reconstructed_video = pipe(
        prompt=config.prompt,
        image=first_frame,
        height=config.image_size[1],
        width=config.image_size[0],
        num_frames=config.n_frames,
        num_inference_steps=config.n_steps,
        guidance_scale=config.cfg,
        negative_prompt=config.negative_prompt,
        target_fps=config.target_fps,
        latents=ddim_latents_at_T,
        generator=g,  # TODO: this is not correct
        return_dict=True,
        ddim_init_latents_t_idx=ddim_init_latents_t_idx,
    ).frames[0]
    return reconstructed_video


def main(template_config, configs_list):
    # Initialize the pipeline
    pipe = I2VGenXLPipeline.from_pretrained(
        "ali-vilab/i2vgen-xl",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.to(device)
    g = torch.Generator(device=device)
    g = g.manual_seed(template_config.seed)

    # Initialize the DDIM inverse scheduler
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        "ali-vilab/i2vgen-xl",
        subfolder="scheduler",
    )
    # Initialize the DDIM scheduler
    ddim_scheduler = DDIMScheduler.from_pretrained(
        "ali-vilab/i2vgen-xl",
        subfolder="scheduler",
    )

    video_dir = template_config.video_dir
    assert os.path.exists(video_dir), f"video_dir: {video_dir} does not exist"
    # loop through the video_dir and process every mp4 file
    for config_entry in configs_list:
        if config_entry["active"] == False:
            logger.info(f"Skipping config_entry: {config_entry}")
            continue
        logger.info(f"Processing config_entry: {config_entry}")

        # Override the config with the data_meta_entry
        config = OmegaConf.merge(template_config, OmegaConf.create(config_entry))

        config.video_path = os.path.join(config.video_dir, config.video_name + ".mp4")
        config.video_frames_path = os.path.join(config.video_dir, config.video_name)

        # If already computed the latents, skip
        if os.path.exists(config.output_dir) and not config.force_recompute_latents:
            logger.info(f"### Skipping !!! {config.output_dir} already exists. ")
            continue

        logger.info(f"config: {OmegaConf.to_yaml(config)}")

        # This is the same as run_ddim_inversion.py
        try:
            logger.info(f"Loading frames from: {config.video_frames_path}")
            _, frame_list = load_video_frames(config.video_frames_path, config.n_frames, config.image_size)
        except:
            logger.error(f"Failed to load frames from: {config.video_frames_path}")
            logger.info(f"Converting mp4 video to frames: {config.video_path}")
            frame_list = convert_video_to_frames(config.video_path, config.image_size, save_frames=True)
            frame_list = frame_list[: config.n_frames]  # 16 frames for img2vid
            logger.debug(f"len(frame_list): {len(frame_list)}")
            # Save the source frames as GIF
            export_to_gif(
                frame_list,
                os.path.join(config.video_frames_path, config.video_name + ".gif")
            )
            logger.info(f"Saved source video as gif to {config.video_frames_path}")
        first_frame = frame_list[0]  # Is a PIL image

        # Produce static video
        if config.inverse_config.inverse_static_video:
            logger.info("### Inverse a static video!")
            frame_list = [frame_list[0]] * config.n_frames

        # Null image inversion
        if config.inverse_config.null_image_inversion:
            logger.info("### Inverse a null image!")
            first_frame = Image.new("RGB", (config.image_size[0], config.image_size[1]), (0, 0, 0))

        # Main pipeline
        # Inversion
        logger.info(f"config: {OmegaConf.to_yaml(config)}")
        _ddim_latents = ddim_inversion(config.inverse_config, first_frame, frame_list, pipe, inverse_scheduler, g)

        # Reconstruction
        recon_config = config.recon_config
        if recon_config.enable_recon:
            ddim_init_latents_t_idx = recon_config.ddim_init_latents_t_idx
            ddim_scheduler.set_timesteps(recon_config.n_steps)
            logger.info(f"ddim_scheduler.timesteps: {ddim_scheduler.timesteps}")
            ddim_latents_path = recon_config.ddim_latents_path
            ddim_latents_at_t = load_ddim_latents_at_t(
                ddim_scheduler.timesteps[ddim_init_latents_t_idx],
                ddim_latents_path=ddim_latents_path,
            )
            logger.debug(f"ddim_scheduler.timesteps[t_idx]: {ddim_scheduler.timesteps[ddim_init_latents_t_idx]}")
            reconstructed_video = ddim_sampling(
                recon_config,
                first_frame,
                ddim_latents_at_t,
                pipe,
                ddim_scheduler,
                ddim_init_latents_t_idx,
                g,
            )

            # Save the reconstructed video
            os.makedirs(config.output_dir, exist_ok=True)
            # Downsampling the video for space saving
            reconstructed_video = [frame.resize((512, 512), resample=Image.LANCZOS) for frame in reconstructed_video]
            export_to_video(
                reconstructed_video,
                os.path.join(config.output_dir, "ddim_reconstruction.mp4"),
                fps=10,
            )
            export_to_gif(
                reconstructed_video,
                os.path.join(config.output_dir, "ddim_reconstruction.gif"),
            )
            logger.info(f"Saved reconstructed video to {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_config", type=str, default="./configs/group_ddim_inversion/template.yaml")
    parser.add_argument("--configs_json", type=str, default="./configs/group_config.json") # This is going to override the template_config

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
    with open(configs_json, 'r') as file:
        configs_list = json.load(file)
    logger.info(f"Loaded {len(configs_list)} configs from {configs_json}")

    # Set up device and seed
    device = torch.device(template_config.device)
    torch.set_grad_enabled(False)
    seed_everything(template_config.seed)
    main(template_config, configs_list)
