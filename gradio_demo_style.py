import gradio as gr

import os
import sys
import time
import subprocess
import shutil

import random
from omegaconf import OmegaConf
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
import numpy as np

from black_box_image_edit.instantstyle import InstantStyle
from black_box_image_edit.utils import crop_and_resize_video, infer_video_style

sys.path.insert(0, "i2vgen-xl")
from utils import load_ddim_latents_at_t
from pipelines.pipeline_i2vgen_xl import I2VGenXLPipeline
from run_group_ddim_inversion import ddim_inversion
from run_group_pnp_edit import init_pnp
from diffusers import DDIMInverseScheduler, DDIMScheduler
from diffusers.utils import load_image
import imageio

DEBUG_MODE = False

demo_examples = [
                    ["./demo/Man Walking.mp4", "./demo/Man Walking/edited_first_frame/turn the man into darth vader.png", "man walking", 0.1, 0.1, 1.0],
                    ["./demo/A kitten turning its head on a wooden floor.mp4", "./demo/A kitten turning its head on a wooden floor/edited_first_frame/A dog turning its head on a wooden floor.png", "A dog turning its head on a wooden floor", 0.2, 0.2, 0.5],
                    ["./demo/An Old Man Doing Exercises For The Body And Mind.mp4", "./demo/An Old Man Doing Exercises For The Body And Mind/edited_first_frame/jack ma.png", "a man doing exercises for the body and mind", 0.8, 0.8, 1.0],
                    ["./demo/Ballet.mp4", "./demo/Ballet/edited_first_frame/van gogh style.png", "girl dancing ballet, in the style of van gogh", 1.0, 1.0, 1.0],
                    ["./demo/A Couple In A Public Display Of Affection.mp4", "./demo/A Couple In A Public Display Of Affection/edited_first_frame/Snowing.png", "A couple in a public display of affection, snowing", 0.3, 0.3, 1.0]
                ]

TEMP_DIR = "_demo_temp"

class StyleEditor:
    def __init__(self) -> None:
        self.image_edit_model = InstantStyle()

    @torch.no_grad()
    def perform_edit(self, video_path, style_image, prompt, force_512=False, seed=42, negative_prompt=""):
        style_image = load_image(style_image) if isinstance(style_image, str) else style_image
        edited_image_path = infer_video_style(self.image_edit_model, 
                    video_path, 
                    output_dir=TEMP_DIR, 
                    style_image=style_image,
                    prompt=prompt, 
                    force_512=force_512, 
                    seed=seed, 
                    negative_prompt=negative_prompt,
                    overwrite=True)
        return edited_image_path

class AnyV2V_I2VGenXL:
    def __init__(self) -> None:
        # Set up default inversion config file
        config = {
            # DDIM inversion
            "inverse_config": {
                "image_size": [512, 512],
                "n_frames": 16,
                "cfg": 1.0,
                "target_fps": 8,
                "ddim_inv_prompt": "",
                "prompt": "",
                "negative_prompt": "",
            },
            "pnp_config": {
                "random_ratio": 0.0,
                "target_fps": 8,
            },
        }
        self.config = OmegaConf.create(config)

    @torch.no_grad()
    def perform_anyv2v(self, 
                       video_path, 
                       video_prompt, 
                       video_negative_prompt,
                       edited_first_frame_path, 
                       conv_inj, 
                       spatial_inj, 
                       temp_inj, 
                       num_inference_steps,
                       guidance_scale,
                       ddim_init_latents_t_idx,
                       ddim_inversion_steps,
                       seed,
                       ):

        # Initialize the I2VGenXL pipeline
        self.pipe = I2VGenXLPipeline.from_pretrained(
            "ali-vilab/i2vgen-xl",
            torch_dtype=torch.float16,
            variant="fp16",
        ).to("cuda:0")

        # Initialize the DDIM inverse scheduler
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
                "ali-vilab/i2vgen-xl",
                subfolder="scheduler",
        )
        # Initialize the DDIM scheduler
        self.ddim_scheduler = DDIMScheduler.from_pretrained(
                "ali-vilab/i2vgen-xl",
                subfolder="scheduler",
        )

        tmp_dir = os.path.join(TEMP_DIR, "AnyV2V")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

        ddim_latents_path = os.path.join(tmp_dir, "ddim_latents")

        def read_frames(video_path):
            frames = []
            with imageio.get_reader(video_path) as reader:
                for i, frame in enumerate(reader):
                    pil_image = Image.fromarray(frame)
                    frames.append(pil_image)
            return frames
        frame_list = read_frames(str(video_path))

        self.config.inverse_config.image_size = list(frame_list[0].size)
        self.config.inverse_config.n_steps = ddim_inversion_steps
        self.config.inverse_config.n_frames = len(frame_list)
        self.config.inverse_config.output_dir = ddim_latents_path
        ddim_init_latents_t_idx = min(ddim_init_latents_t_idx, num_inference_steps - 1)

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
        self.config.pnp_config.pnp_f_t = conv_inj
        self.config.pnp_config.pnp_spatial_attn_t = spatial_inj
        self.config.pnp_config.pnp_temp_attn_t = temp_inj
        self.config.pnp_config.ddim_init_latents_t_idx = ddim_init_latents_t_idx
        init_pnp(self.pipe, self.ddim_scheduler, self.config.pnp_config)
        # Edit video
        self.pipe.register_modules(scheduler=self.ddim_scheduler)

        edited_video = self.pipe.sample_with_pnp(
            prompt=video_prompt,
            image=edited_1st_frame,
            height=self.config.inverse_config.image_size[1],
            width=self.config.inverse_config.image_size[0],
            num_frames=self.config.inverse_config.n_frames,
            num_inference_steps=self.config.pnp_config.n_steps,
            guidance_scale=guidance_scale,
            negative_prompt=video_negative_prompt,
            target_fps=self.config.pnp_config.target_fps,
            latents=mixed_latents,
            generator=generator,
            return_dict=True,
            ddim_init_latents_t_idx=ddim_init_latents_t_idx,
            ddim_inv_latents_path=ddim_latents_path,
            ddim_inv_prompt=self.config.inverse_config.ddim_inv_prompt,
            ddim_inv_1st_frame=first_frame,
        ).frames[0]

        edited_video = [
            frame.resize(self.config.inverse_config.image_size, resample=Image.LANCZOS)
            for frame in edited_video
        ]

        def images_to_video(images, output_path, fps=24):
            writer = imageio.get_writer(output_path, fps=fps)

            for img in images:
                img_np = np.array(img)
                writer.append_data(img_np)

            writer.close()
        output_path = os.path.join(tmp_dir, "edited_video.mp4")
        images_to_video(
            edited_video, output_path, fps=self.config.pnp_config.target_fps
        )
        return output_path


# Init the class
#=====================================
if not DEBUG_MODE:
    Image_Editor = StyleEditor()
    AnyV2V_Editor = AnyV2V_I2VGenXL()
#=====================================

def get_first_frame_as_pil(video_path):
    with VideoFileClip(video_path) as clip:
        # Extract the first frame (at t=0) as an array
        first_frame_array = clip.get_frame(0)
        # Convert the numpy array to a PIL Image
        first_frame_image = Image.fromarray(first_frame_array)
        return first_frame_image
        
def btn_preprocess_video_fn(video_path, width, height, start_time, end_time, center_crop, x_offset, y_offset, longest_to_width):
    fps = 8
    desired_n_frames = int(end_time-start_time)*fps
    processed_video_path = crop_and_resize_video(input_video_path=video_path, 
                                                output_folder=TEMP_DIR,
                                                clip_duration=None,
                                                width=width, 
                                                height=height, 
                                                start_time=start_time, 
                                                end_time=end_time, 
                                                center_crop=center_crop, 
                                                n_frames=desired_n_frames,
                                                x_offset=x_offset, 
                                                y_offset=y_offset, 
                                                longest_to_width=longest_to_width)

    return processed_video_path

def btn_image_edit_fn(video_path, style_image, ie_force_512, ie_seed, ie_neg_prompt):
    """
    Generate an image based on the video and text input.
    This function should be replaced with your actual image generation logic.
    """
    # Placeholder logic for image generation

    if ie_seed < 0:
        ie_seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {ie_seed}")

    edited_image_path = Image_Editor.perform_edit(video_path=video_path, 
                                             style_image=style_image,
                                             force_512=ie_force_512,
                                             prompt=None,
                                             seed=ie_seed,
                                             negative_prompt=ie_neg_prompt)
    return edited_image_path


def btn_infer_fn(video_path, 
                video_prompt, 
                video_negative_prompt,
                edited_first_frame_path, 
                conv_inj, 
                spatial_inj, 
                temp_inj, 
                num_inference_steps,
                guidance_scale,
                ddim_init_latents_t_idx,
                ddim_inversion_steps,
                seed,
                ):
    if seed < 0:
        seed = int.from_bytes(os.urandom(2), "big")
    print(f"Using seed: {seed}")

    result_video_path = AnyV2V_Editor.perform_anyv2v(video_path=video_path,
                                                        video_prompt=video_prompt,
                                                        video_negative_prompt=video_negative_prompt,
                                                        edited_first_frame_path=edited_first_frame_path,
                                                        conv_inj=conv_inj,
                                                        spatial_inj=spatial_inj,
                                                        temp_inj=temp_inj,
                                                        num_inference_steps=num_inference_steps,
                                                        guidance_scale=guidance_scale,
                                                        ddim_init_latents_t_idx=ddim_init_latents_t_idx,
                                                        ddim_inversion_steps=ddim_inversion_steps,
                                                        seed=seed)

    return result_video_path

# Create the UI
#=====================================
with gr.Blocks() as demo:
    gr.Markdown("# <img src='https://tiger-ai-lab.github.io/AnyV2V/static/images/icon.png' width='30'/> AnyV2V")
    gr.Markdown("Official 🤗 Gradio demo for [AnyV2V: A Plug-and-Play Framework For Any Video-to-Video Editing Tasks](https://tiger-ai-lab.github.io/AnyV2V/)")

    with gr.Tabs():
        with gr.TabItem('AnyV2V(I2VGenXL) + InstantStyle'):
            gr.Markdown("# Preprocessing Video Stage")
            gr.Markdown("In this demo, AnyV2V only support video up to 16 seconds duration and 8 fps. If your video is not in this format, we will preprocess it for you. Click on the Preprocess video button!")
            with gr.Row():
                with gr.Column():
                    video_raw = gr.Video(label="Raw Video Input")
                    btn_pv = gr.Button("Preprocess Video")
                    
                with gr.Column():
                    video_input = gr.Video(label="Preprocessed Video Input", interactive=False)
                with gr.Column():
                    advanced_settings_pv = gr.Accordion("Advanced Settings for Video Preprocessing", open=False)
                    with advanced_settings_pv:
                        with gr.Column():
                            pv_width = gr.Number(label="Width", value=512, minimum=1, maximum=4096)
                            pv_height = gr.Number(label="Height", value=512, minimum=1, maximum=4096)
                            pv_start_time = gr.Number(label="Start Time", value=0, minimum=0)
                            pv_end_time = gr.Number(label="End Time", value=2, minimum=0)
                            pv_center_crop = gr.Checkbox(label="Center Crop", value=True)
                            pv_x_offset = gr.Number(label="Horizontal Offset (-1 to 1)", value=0, minimum=-1, maximum=1)
                            pv_y_offset = gr.Number(label="Vertical Offset (-1 to 1)", value=0, minimum=-1, maximum=1)
                            pv_longest_to_width = gr.Checkbox(label="Resize Longest Dimension to Width")
                    
            gr.Markdown("# Image Editing Stage")
            gr.Markdown("Edit the first frame of the video to your liking! Click on the Edit the first frame button after uploading the style reference. This image editing stage is powered by InstantStyle. You can try edit the image multiple times until you are happy with the result! You can also choose to download the first frame of the video and edit it with other software (e.g. Photoshop, GIMP, etc.) or use other image editing models to obtain the edited frame and upload it directly.")
            with gr.Row():
                with gr.Column():
                    src_first_frame = gr.Image(label="First Frame", type="filepath", interactive=False)
                    style_image = gr.Image(label="Style Image", type="filepath")
                    btn_image_edit = gr.Button("Edit the first frame")
                with gr.Column():
                    image_input_output = gr.Image(label="Edited Frame", type="filepath")
                with gr.Column():
                    advanced_settings_image_edit = gr.Accordion("Advanced Settings for Image Editing", open=True)
                    with advanced_settings_image_edit:
                        with gr.Column():
                            ie_neg_prompt = gr.Textbox(label="Negative Prompt", value="low res, blurry, watermark, jpeg artifacts")
                            ie_seed = gr.Number(label="Seed (-1 means random)", value=-1, minimum=-1, maximum=sys.maxsize)
                            ie_force_512 = gr.Checkbox(label="Force resize to 512x512 before feeding into the image editing model")

            gr.Markdown("# Video Editing Stage")
            gr.Markdown("Enjoy the full control of the video editing process using the edited image and the preprocessed video! Click on the Run AnyV2V button after inputting the video description prompt. Try tweak with the setting if the output does not satisfy you!")
            with gr.Row():
                with gr.Column():
                    video_prompt = gr.Textbox(label="Video description prompt")
                    settings_anyv2v = gr.Accordion("Settings for AnyV2V")
                    with settings_anyv2v:
                        with gr.Column():
                            av_pnp_f_t = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.2, label="Convolutional injection (pnp_f_t)")
                            av_pnp_spatial_attn_t = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.2, label="Spatial Attention injection (pnp_spatial_attn_t)")
                            av_pnp_temp_attn_t = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.5, label="Temporal Attention injection (pnp_temp_attn_t)")
                    btn_infer = gr.Button("Run Video Editing")
                with gr.Column():
                    video_output = gr.Video(label="Video Output")
                with gr.Column():
                    advanced_settings_anyv2v = gr.Accordion("Advanced Settings for AnyV2V", open=False)
                    with advanced_settings_anyv2v:
                        with gr.Column():
                            av_ddim_init_latents_t_idx = gr.Number(label="DDIM Initial Latents t Index", value=0, minimum=0)
                            av_ddim_inversion_steps = gr.Number(label="DDIM Inversion Steps", value=100, minimum=1)
                            av_num_inference_steps = gr.Number(label="Number of Inference Steps", value=50, minimum=1)
                            av_guidance_scale = gr.Number(label="Guidance Scale", value=9, minimum=0)
                            av_seed = gr.Number(label="Seed (-1 means random)", value=42, minimum=-1, maximum=sys.maxsize)
                            av_neg_prompt = gr.Textbox(label="Negative Prompt", value="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms")

    examples = gr.Examples(examples=demo_examples, 
                           label="Examples (Just click on Video Editing button after loading them into the UI)",
                            inputs=[video_input, image_input_output, video_prompt, av_pnp_f_t, av_pnp_spatial_attn_t, av_pnp_temp_attn_t])

    btn_pv.click(
        btn_preprocess_video_fn,
        inputs=[video_raw, pv_width, pv_height, pv_start_time, pv_end_time, pv_center_crop, pv_x_offset, pv_y_offset, pv_longest_to_width],
        outputs=video_input
    )

    btn_image_edit.click(
        btn_image_edit_fn,
        inputs=[video_input, style_image, ie_force_512, ie_seed, ie_neg_prompt],
        outputs=image_input_output
    )
    
    btn_infer.click(
        btn_infer_fn,
        inputs=[video_input, 
                video_prompt, 
                av_neg_prompt,
                image_input_output, 
                av_pnp_f_t, 
                av_pnp_spatial_attn_t, 
                av_pnp_temp_attn_t,
                av_num_inference_steps,
                av_guidance_scale,
                av_ddim_init_latents_t_idx,
                av_ddim_inversion_steps,
                av_seed],
        outputs=video_output
    )

    video_input.change(fn=get_first_frame_as_pil, inputs=video_input, outputs=src_first_frame)

#=====================================

# Minimizing usage of GPU Resources
torch.set_grad_enabled(False)


demo.launch()