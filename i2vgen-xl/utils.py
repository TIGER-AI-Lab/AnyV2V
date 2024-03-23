import os
import random
import numpy as np
import torch
from torchvision.io import read_video
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from diffusers.utils import load_image
import glob


import logging
logger = logging.getLogger(__name__)


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
    logger.debug(f"Loaded ddim_latents_at_t from {ddim_latents_at_t_path}")
    return ddim_latents_at_t


def load_ddim_latents_at_T(ddim_latents_path):
    noisest = max(
        [int(x.split("_")[-1].split(".")[0]) for x in glob.glob(os.path.join(ddim_latents_path, f"ddim_latents_*.pt"))]
    )
    ddim_latents_at_T_path = os.path.join(ddim_latents_path, f"ddim_latents_{noisest}.pt")
    ddim_latents_at_T = torch.load(ddim_latents_at_T_path)  # [b, c, f, h, w] [1, 4, 16, 40, 64]
    return ddim_latents_at_T


# Modified from tokenflow/utils.py
def convert_video_to_frames(video_path, img_size=(512, 512), save_frames=True):
    video, _, _ = read_video(video_path, output_format="TCHW")
    # rotate video -90 degree if video is .mov format. this is a weird bug in torchvision
    if video_path.endswith(".mov"):
        video = T.functional.rotate(video, -90)
    if save_frames:
        video_name = Path(video_path).stem
        video_dir = Path(video_path).parent
        os.makedirs(f"{video_dir}/{video_name}", exist_ok=True)
    frames = []
    for i in range(len(video)):
        ind = str(i).zfill(5)
        image = T.ToPILImage()(video[i])
        logger.info(f"Original video frame size: {image.size}")
        if image.size != img_size:
            image_resized = image.resize(img_size, resample=Image.Resampling.LANCZOS)
            logger.info(f"Resized video frame, height, width: {image_resized.size}, {img_size[1]}, {img_size[0]}")
        else:
            image_resized = image
        if save_frames:
            image_resized.save(f"{video_dir}/{video_name}/{ind}.png")
            print(f"Saved frame {video_dir}/{video_name}/{ind}.png")
        frames.append(image_resized)
    return frames


# Modified from tokenflow/utils.py
def load_video_frames(frames_path, n_frames, image_size=(512, 512)):
    # Load paths
    paths = [f"{frames_path}/%05d.png" % i for i in range(n_frames)]
    frames = [load_image(p) for p in paths]
    # Check if the frames are the right size
    for f in frames:
        if f.size != image_size:
            logger.error(f"Frame size {f.size} does not match config.image_size {image_size}")
            raise ValueError(f"Frame size {f.size} does not match config.image_size {image_size}")
    return paths, frames

