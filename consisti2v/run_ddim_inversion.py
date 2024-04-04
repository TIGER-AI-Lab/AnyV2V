import os
import sys
import torch
import argparse
import logging
from omegaconf import OmegaConf
from PIL import Image
from pathlib import Path

# HF imports
from diffusers import DDIMScheduler
from ddim_inverse_scheduler import DDIMInverseScheduler

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


def ddim_inversion(config, first_frame, frame_list, pipe: ConditionalVideoEditingPipeline, inverse_scheduler, g):
    pipe.scheduler = inverse_scheduler
    video_latents_at_0 = pipe.encode_vae_video(
        frame_list,
        device=pipe._execution_device,
        height=config.image_size[1],
        width=config.image_size[0],
    )
    ddim_latents = pipe.invert(
        prompt=config.prompt,
        first_frame_paths=first_frame,
        height=config.image_size[1],
        width=config.image_size[0],
        video_length=config.n_frames,
        num_inference_steps=config.n_steps,
        guidance_scale_txt=config.cfg_txt,
        guidance_scale_img=config.cfg_img,
        negative_prompt=config.negative_prompt,
        frame_stride=config.frame_stride,
        latents=video_latents_at_0,
        generator=g,  # TODO: this is not correct
        return_dict=False,
        output_type="latent",
        output_dir=config.output_dir,
    ).videos  # [b, num_inference_steps, c, num_frames, h, w]
    logger.debug(f"ddim_latents.shape: {ddim_latents.shape}")
    ddim_latents = ddim_latents[0]  # [num_inference_steps, c, num_frames, h, w]
    return ddim_latents


def ddim_sampling(
    config, first_frame, ddim_latents_at_T, pipe: ConditionalVideoEditingPipeline, ddim_scheduler, g, ddim_init_latents_t_idx
):
    pipe.scheduler = ddim_scheduler
    reconstructed_video = pipe(
        prompt=config.prompt,
        first_frame_paths=first_frame,
        height=config.image_size[1],
        width=config.image_size[0],
        video_length=config.n_frames,
        num_inference_steps=config.n_steps,
        guidance_scale_txt=config.cfg_txt,
        guidance_scale_img=config.cfg_img,
        negative_prompt=config.negative_prompt,
        frame_stride=config.frame_stride,
        latents=ddim_latents_at_T,
        generator=g,  # TODO: this is not correct
        return_dict=True,
        ddim_init_latents_t_idx=ddim_init_latents_t_idx,
    ).videos
    return reconstructed_video


def main(config):
    seed_everything(config.seed)
    torch.set_grad_enabled(False)
    device = torch.device(config.device)

    # Initialize the pipeline
    # TODO: do we need the get_inverse_timesteps function?
    pipe = ConditionalVideoEditingPipeline.from_pretrained(
        "TIGER-Lab/ConsistI2V",
        torch_dtype=torch.float16,
    )
    # TODO: set the model to GPU and eval mode
    pipe.to(device)
    g = torch.Generator(device=device)
    g = g.manual_seed(config.seed)

    # Initialize the DDIM inverse scheduler
    inverse_scheduler = DDIMInverseScheduler.from_pretrained(
        "TIGER-Lab/ConsistI2V",
        subfolder="scheduler",
    )
    # Initialize the DDIM scheduler
    ddim_scheduler = DDIMScheduler.from_pretrained(
        "TIGER-Lab/ConsistI2V",
        subfolder="scheduler",
    )

    if config.video_path:
        frame_list = convert_video_to_frames(config.video_path, config.image_size, save_frames=config.save_frames, save_dir=config.output_dir)
        frame_list = frame_list[: config.n_frames]  # 16 frames for img2vid
        logger.debug(f"len(frame_list): {len(frame_list)}")
        video_name = Path(config.video_path).stem
        first_frame_path = os.path.join(config.output_dir, video_name, '00000.png')
    elif config.video_frames_path:
        _, frame_list = load_video_frames(config.video_frames_path, config.n_frames)
        first_frame_path = os.path.join(config.video_frames_path, '00000.png')
    else:
        raise ValueError("Please provide either video_path or video_frames_path")
    
    # Main pipeline
    ddim_latents = ddim_inversion(config.inverse_config, first_frame_path, frame_list, pipe, inverse_scheduler, g)

    recon_config = config.recon_config
    ddim_init_latents_t_idx = recon_config.ddim_init_latents_t_idx
    ddim_scheduler.set_timesteps(recon_config.n_steps)
    logger.info(f"ddim_scheduler.timesteps: {ddim_scheduler.timesteps}")
    ddim_latents_path = config.inverse_config.output_dir
    ddim_latents_at_t = load_ddim_latents_at_t(
        ddim_scheduler.timesteps[ddim_init_latents_t_idx], ddim_latents_path=ddim_latents_path
    )
    logger.debug(f"ddim_scheduler.timesteps[t_idx]: {ddim_scheduler.timesteps[ddim_init_latents_t_idx]}")

    reconstructed_video = ddim_sampling(recon_config, first_frame_path, ddim_latents_at_t, pipe, ddim_scheduler, g, ddim_init_latents_t_idx)

    # Save reconstructed frames and video
    os.makedirs(config.output_dir, exist_ok=True)
    save_videos_grid(reconstructed_video, os.path.join(config.output_dir, "ddim_reconstruction.gif"), fps=10, format="gif")
    save_videos_grid(reconstructed_video, os.path.join(config.output_dir, "ddim_reconstruction.mp4"), fps=10, format="mp4")
    logger.info(f"Saved reconstructed video to {config.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline_256/ddim_inversion_256.yaml")
    parser.add_argument("optional_args", nargs='*', default=[])
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    if args.optional_args:
        modified_config = OmegaConf.from_dotlist(args.optional_args)
        config = OmegaConf.merge(config, modified_config)

    logging_level = logging.DEBUG if config.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"config: {OmegaConf.to_yaml(config)}")

    main(config)
