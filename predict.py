# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import time
import subprocess
import shutil
from PIL import Image
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip
import numpy as np
from cog import BasePredictor, Input, Path
import torch
from diffusers import DDIMInverseScheduler, DDIMScheduler
from diffusers.utils import load_image
import imageio

from black_box_image_edit import InstructPix2Pix

sys.path.insert(0, "i2vgen-xl")
from utils import load_ddim_latents_at_t
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
from run_group_ddim_inversion import ddim_inversion
from run_group_pnp_edit import init_pnp


# Weights are saved and loaded from replicate.delivery for faster booting
INSTRUCTPIX2PIX_URL = "https://weights.replicate.delivery/default/timbrooks/instruct-pix2pix.tar"  # original pipeline weights from timbrooks/instruct-pix2pix
INSTRUCTPIX2PIX_CACHE = "weights/timbrooks/instruct-pix2pix"
ALI_I2VGENXL_URL = "https://weights.replicate.delivery/default/ali-vilab/i2vgen-xl.tar"  # original pipeline weights from ali-vilab/i2vgen-xl
ALI_I2VGENXL_CACHE = "weights/ali-vilab/i2vgen-xl"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        if not os.path.exists(INSTRUCTPIX2PIX_CACHE):
            download_weights(INSTRUCTPIX2PIX_URL, INSTRUCTPIX2PIX_CACHE)
        self.black_box_image_model = InstructPix2Pix(weight=INSTRUCTPIX2PIX_CACHE)

        if not os.path.exists(ALI_I2VGENXL_CACHE):
            download_weights(ALI_I2VGENXL_URL, ALI_I2VGENXL_CACHE)
        self.pipe = I2VGenXLPipeline.from_pretrained(
            ALI_I2VGENXL_CACHE,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda:0")

        # Initialize the DDIM inverse scheduler
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            ALI_I2VGENXL_CACHE,
            subfolder="scheduler",
        )
        # Initialize the DDIM scheduler
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
            ALI_I2VGENXL_CACHE,
            subfolder="scheduler",
        )
        # Set up default inversion config file
        config = {
            # DDIM inversion
            "inverse_config": {
                "image_size": [512, 512],
                "n_frames": 16,
                "cfg": 1.0,
                "target_fps": 8,
                "prompt": "",
                "negative_prompt": "",
            },
            "pnp_config": {
                "ddim_init_latents_t_idx": 0,  # 0 for 981, 3 for 921, 9 for 801, 20 for 581 if n_steps=50
                "ddim_inv_prompt": "",
                "random_ratio": 0.0,
                "target_fps": 8,
                "pnp_f_t": 1.0,
                "pnp_spatial_attn_t": 1.0,
                "pnp_temp_attn_t": 1.0,
            },
        }
        self.config = OmegaConf.create(config)

    def predict(
        self,
        video: Path = Input(description="Input video"),
        instruct_pix2pix_prompt: str = Input(
            description="The first step invovles using timbrooks/instruct-pix2pix to edit the first frame. Specify the prompt for editing the first frame.",
            default="turn man into robot",
        ),
        editing_prompt: str = Input(
            description="Describe the input video",
            default="a man doing exercises for the body and mind",
        ),
        editing_negative_prompt: str = Input(
            description="Things not to see int the edited video",
            default="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=9.0
        ),
        ddim_inversion_steps: int = Input(
            description="Number of ddim inversion steps", default=500
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        tmp_dir = "exp_dir"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        ddim_latents_path = os.path.join(tmp_dir, "ddim_latents")

        frame_list = read_frames(str(video))

        self.config.inverse_config.image_size = list(frame_list[0].size)
        self.config.inverse_config.n_steps = ddim_inversion_steps
        self.config.inverse_config.n_frames = len(frame_list)
        self.config.inverse_config.output_dir = ddim_latents_path

        # Step 0. Black-box image editing for the first frame
        edited_first_frame_path = os.path.join(tmp_dir, "edited_first_frame.png")
        infer_video(
            self.black_box_image_model,
            str(video),
            edited_first_frame_path,
            instruct_pix2pix_prompt,
            seed=seed,
        )

        # Step 1. DDIM Inversion
        first_frame = frame_list[0]

        generator = torch.Generator(device="cuda:0")
        generator = generator.manual_seed(seed)
        _ddim_latents = ddim_inversion(
            self.config.inverse_config,
            first_frame,
            frame_list,
            self.pipe,
            self.inverse_scheduler,
            generator,
        )

        # Step 2. DDIM Sampling + PnP feature and attention injection
        # Load the edited first frame
        edited_1st_frame = load_image(edited_first_frame_path).resize(
            self.config.inverse_config.image_size, resample=Image.Resampling.LANCZOS
        )
        # Load the initial latents at t
        ddim_init_latents_t_idx = self.config.pnp_config.ddim_init_latents_t_idx
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        print(f"ddim_scheduler.timesteps: {self.ddim_scheduler.timesteps}")
        ddim_latents_at_t = load_ddim_latents_at_t(
            self.ddim_scheduler.timesteps[ddim_init_latents_t_idx],
            ddim_latents_path=ddim_latents_path,
        )
        print(
            f"ddim_scheduler.timesteps[t_idx]: {self.ddim_scheduler.timesteps[ddim_init_latents_t_idx]}"
        )
        print(f"ddim_latents_at_t.shape: {ddim_latents_at_t.shape}")

        # Blend the latents
        random_latents = torch.randn_like(ddim_latents_at_t)
        print(
            f"Blending random_ratio (1 means random latent): {self.config.pnp_config.random_ratio}"
        )
        mixed_latents = (
            random_latents * self.config.pnp_config.random_ratio
            + ddim_latents_at_t * (1 - self.config.pnp_config.random_ratio)
        )

        # Init Pnp
        self.config.pnp_config.n_steps = num_inference_steps
        init_pnp(self.pipe, self.ddim_scheduler, self.config.pnp_config)
        # Edit video
        self.pipe.register_modules(scheduler=self.ddim_scheduler)

        edited_video = self.pipe.sample_with_pnp(
            prompt=editing_prompt,
            image=edited_1st_frame,
            height=self.config.inverse_config.image_size[1],
            width=self.config.inverse_config.image_size[0],
            num_frames=self.config.inverse_config.n_frames,
            num_inference_steps=self.config.pnp_config.n_steps,
            guidance_scale=guidance_scale,
            negative_prompt=editing_negative_prompt,
            target_fps=self.config.pnp_config.target_fps,
            latents=mixed_latents,
            generator=generator,
            return_dict=True,
            ddim_init_latents_t_idx=ddim_init_latents_t_idx,
            ddim_inv_latents_path=ddim_latents_path,
            ddim_inv_prompt="",
            ddim_inv_1st_frame=first_frame,
        ).frames[0]

        edited_video = [
            frame.resize(self.config.inverse_config.image_size, resample=Image.LANCZOS)
            for frame in edited_video
        ]

        output_path = "/tmp/out.mp4"
        images_to_video(
            edited_video, output_path, fps=self.config.pnp_config.target_fps
        )

        return Path(output_path)


def infer_video(
    model, video_path, result_path, prompt, force_512=False, seed=42, negative_prompt=""
):
    # Create the output directory if it does not exist
    video_clip = VideoFileClip(video_path)

    def process_frame(image):
        pil_image = Image.fromarray(image)
        if force_512:
            pil_image = pil_image.resize((512, 512), Image.LANCZOS)
        result = model.infer_one_image(
            pil_image,
            instruct_prompt=prompt,
            seed=seed,
            negative_prompt=negative_prompt,
        )
        if force_512:
            result = result.resize(video_clip.size, Image.LANCZOS)
        return np.array(result)

    # Process only the first frame
    first_frame = video_clip.get_frame(0)  # Get the first frame
    processed_frame = process_frame(first_frame)  # Process the first frame

    Image.fromarray(processed_frame).save(result_path)
    print(f"Processed and saved the first frame: {result_path}")


def images_to_video(images, output_path, fps=24):
    writer = imageio.get_writer(output_path, fps=fps)

    for img in images:
        img_np = np.array(img)
        writer.append_data(img_np)

    writer.close()


def read_frames(video_path):
    frames = []
    with imageio.get_reader(video_path) as reader:
        for i, frame in enumerate(reader):
            pil_image = Image.fromarray(frame)
            frames.append(pil_image)
    return frames
