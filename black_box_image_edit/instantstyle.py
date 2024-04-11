from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import cv2
import torch
import PIL
import numpy as np
import os

class InstantStyle():
    def __init__(self, 
                 device="cuda", 
                 weight="stabilityai/stable-diffusion-xl-base-1.0", 
                 control_weight="diffusers/controlnet-canny-sdxl-1.0",
                 custom_sdxl_models_folder="sdxl_models"):
        from .ip_adapter import IPAdapterXL

        controlnet = ControlNetModel.from_pretrained(control_weight, 
                                                    use_safetensors=False, 
                                                    torch_dtype=torch.float16).to(device)
        # load SDXL pipeline
        sdxl_control_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            weight,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        sdxl_control_pipe.enable_vae_tiling()
        self.ip_model = IPAdapterXL(sdxl_control_pipe, 
                            os.path.join(custom_sdxl_models_folder, "image_encoder"),
                            os.path.join(custom_sdxl_models_folder, "ip-adapter_sdxl.bin"),
                            device, 
                            target_blocks=["up_blocks.0.attentions.1"])


    def infer_one_image(self, src_image: PIL.Image.Image = None, 
                        style_image: PIL.Image.Image = None,
                        prompt: str = "masterpiece, best quality, high quality", 
                        seed: int = 42, 
                        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"):

        src_image = src_image.convert('RGB') # force it to RGB format
        style_image = style_image.convert('RGB') # force it to RGB format

        def pil_to_cv2(image_pil):
            image_np = np.array(image_pil)
            image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return image_cv2
        # control image
        input_image = pil_to_cv2(src_image)
        detected_map = cv2.Canny(input_image, 50, 200)
        canny_map = PIL.Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

        # generate image
        if prompt is None:
            prompt = "masterpiece, best quality, high quality"
        image = self.ip_model.generate(pil_image=style_image,
                                prompt=prompt,
                                negative_prompt=negative_prompt,
                                scale=1.0,
                                guidance_scale=5,
                                num_samples=1,
                                num_inference_steps=30, 
                                seed=seed,
                                image=canny_map,
                                controlnet_conditioning_scale=0.6,
                                )[0]
        return image