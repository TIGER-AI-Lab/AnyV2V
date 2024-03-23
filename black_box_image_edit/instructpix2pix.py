import torch
import PIL

from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

class InstructPix2Pix():
    """
    A wrapper around the StableDiffusionInstructPix2PixPipeline for guided image transformation.

    This class uses the Pix2Pix pipeline to transform an image based on an instruction prompt.
    Reference: https://huggingface.co/docs/diffusers/api/pipelines/pix2pix
    """
    def __init__(self, device="cuda", weight="timbrooks/instruct-pix2pix"):
        """
        Attributes:
            pipe (StableDiffusionInstructPix2PixPipeline): The Pix2Pix pipeline for image transformation.

        Args:
            device (str, optional): Device on which the pipeline runs. Defaults to "cuda".
            weight (str, optional): Pretrained weights for the model. Defaults to "timbrooks/instruct-pix2pix".
        """
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            weight,
            torch_dtype=torch.float16,
            safety_checker=None,
        ).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42, negative_prompt=""):
        """
        Modifies the source image based on the provided instruction prompt.

        Args:
            src_image (PIL.Image.Image): Source image in RGB format.
            instruct_prompt (str): Caption for editing the image.
            seed (int, optional): Seed for random generator. Defaults to 42.

        Returns:
            PIL.Image.Image: The transformed image.
        """
        src_image = src_image.convert('RGB') # force it to RGB format
        generator = torch.manual_seed(seed)

        # configs from https://github.com/timothybrooks/instruct-pix2pix/blob/main/edit_cli.py
        image = self.pipe(instruct_prompt, image=src_image,
                          num_inference_steps=100,
                          image_guidance_scale=1.5,
                          guidance_scale=7.5,
                          negative_prompt=negative_prompt,
                          generator=generator
                          ).images[0]
        return image

class MagicBrush(InstructPix2Pix):
    def __init__(self, device="cuda", weight="vinesmsuic/magicbrush-jul7"):
        """
        A class for MagicBrush.

        Args:
            device (str, optional): The device on which the model should run. Default is "cuda".
            weight (str, optional): The pretrained model weights for MagicBrush. Default is "vinesmsuic/magicbrush-jul7".
        """
        super().__init__(device=device, weight=weight)

    def infer_one_image(self, src_image: PIL.Image.Image = None, src_prompt: str = None, target_prompt: str = None, instruct_prompt: str = None, seed: int = 42, negative_prompt=""):
        return super().infer_one_image(src_image, src_prompt, target_prompt, instruct_prompt, seed, negative_prompt)