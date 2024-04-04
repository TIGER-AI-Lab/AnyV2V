# Adapted from https://github.com/showlab/Tune-A-Video/blob/main/tuneavideo/pipelines/pipeline_tuneavideo.py

import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass

import math
import numpy as np
import torch
from tqdm import tqdm

from torchvision import transforms as T
from PIL import Image

from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange, repeat

from ..models.unet import UNet3DConditionModel
from ..utils.frameinit_utils import freq_mix_3d, get_freq_filter


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# copied from https://github.com/huggingface/diffusers/blob/v0.23.0/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L59C1-L70C21
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AutoregressiveAnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.freq_filter = None

    @torch.no_grad()
    def init_filter(self, video_length, height, width, filter_params):
        # initialize frequency filter for noise reinitialization
        batch_size = 1
        num_channels_latents = self.unet.config.in_channels
        filter_shape = [
            batch_size, 
            num_channels_latents, 
            video_length, 
            height // self.vae_scale_factor, 
            width // self.vae_scale_factor
        ]
        # self.freq_filter = get_freq_filter(filter_shape, device=self._execution_device, params=filter_params)
        self.freq_filter = get_freq_filter(
            filter_shape, 
            device=self._execution_device, 
            filter_type=filter_params.method,
            n=filter_params.n if filter_params.method=="butterworth" else None,
            d_s=filter_params.d_s,
            d_t=filter_params.d_t
        )

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)


    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance is not None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance == "text":
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            elif do_classifier_free_guidance == "both":
                text_embeddings = torch.cat([uncond_embeddings, uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents, first_frames=None):
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0]), **self._progress_bar_config):
            video.append(self.vae.decode(latents[frame_idx:frame_idx+1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)

        if first_frames is not None:
            first_frames = first_frames.unsqueeze(2)
            video = torch.cat([first_frames, video], dim=2)

        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps, first_frame_paths=None):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if first_frame_paths is not None and (not isinstance(prompt, str) and not isinstance(first_frame_paths, list)):
            raise ValueError(f"`first_frame_paths` has to be of type `str` or `list` but is {type(first_frame_paths)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None, noise_sampling_method="vanilla", noise_alpha=1.0):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                # shape = shape
                shape = (1,) + shape[1:]
                if noise_sampling_method == "vanilla":
                    latents = [
                        torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                        for i in range(batch_size)
                    ]
                elif noise_sampling_method == "pyoco_mixed":
                    base_shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    latents = []
                    noise_alpha_squared = noise_alpha ** 2
                    for i in range(batch_size):
                        base_latent = torch.randn(base_shape, generator=generator[i], device=rand_device, dtype=dtype) * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared))
                        ind_latent = torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype) * math.sqrt(1 / (1 + noise_alpha_squared))
                        latents.append(base_latent + ind_latent)
                elif noise_sampling_method == "pyoco_progressive":
                    latents = []
                    noise_alpha_squared = noise_alpha ** 2
                    for i in range(batch_size):
                        latent = torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                        ind_latent = torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype) * math.sqrt(1 / (1 + noise_alpha_squared))
                        for j in range(1, video_length):
                            latent[:, :, j, :, :] = latent[:, :, j - 1, :, :] * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared)) + ind_latent[:, :, j, :, :]
                        latents.append(latent)
                latents = torch.cat(latents, dim=0).to(device)
            else:
                if noise_sampling_method == "vanilla":
                    latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
                elif noise_sampling_method == "pyoco_mixed":
                    noise_alpha_squared = noise_alpha ** 2
                    base_shape = (batch_size, num_channels_latents, 1, height // self.vae_scale_factor, width // self.vae_scale_factor)
                    base_latents = torch.randn(base_shape, generator=generator, device=rand_device, dtype=dtype) * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared))
                    ind_latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype) * math.sqrt(1 / (1 + noise_alpha_squared))
                    latents = base_latents + ind_latents
                elif noise_sampling_method == "pyoco_progressive":
                    noise_alpha_squared = noise_alpha ** 2
                    latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype)
                    ind_latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype) * math.sqrt(1 / (1 + noise_alpha_squared))
                    for j in range(1, video_length):
                        latents[:, :, j, :, :] = latents[:, :, j - 1, :, :] * math.sqrt((noise_alpha_squared) / (1 + noise_alpha_squared)) + ind_latents[:, :, j, :, :]
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        video_length: Optional[int],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale_txt: float = 7.5,
        guidance_scale_img: float = 2.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        # additional
        first_frame_paths: Optional[Union[str, List[str]]] = None,
        first_frames: Optional[torch.FloatTensor] = None,
        noise_sampling_method: str = "vanilla",
        noise_alpha: float = 1.0,
        guidance_rescale: float = 0.0,
        frame_stride: Optional[int] = None,
        autoregress_steps: int = 3,
        use_frameinit: bool = False,
        frameinit_noise_level: int = 999,
        **kwargs,
    ):
        if first_frame_paths is not None and first_frames is not None:
            raise ValueError("Only one of `first_frame_paths` and `first_frames` can be passed.")
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, callback_steps, first_frame_paths)

        # Define call parameters
        # batch_size = 1 if isinstance(prompt, str) else len(prompt)
        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)
            first_frame_input = first_frame_paths if first_frame_paths is not None else first_frames
            if first_frame_input is not None:
                assert len(prompt) == len(first_frame_input), "prompt and first_frame_paths should have the same length"

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = None
        # two guidance mode: text and text+image
        if guidance_scale_txt > 1.0:
            do_classifier_free_guidance = "text"
        if guidance_scale_img > 1.0:
            do_classifier_free_guidance = "both"

        # Encode input prompt
        prompt = prompt if isinstance(prompt, list) else [prompt] * batch_size
        if negative_prompt is not None:
            negative_prompt = negative_prompt if isinstance(negative_prompt, list) else [negative_prompt] * batch_size 
        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        # Encode input first frame
        first_frame_latents = None
        if first_frame_paths is not None:
            first_frame_paths = first_frame_paths if isinstance(first_frame_paths, list) else [first_frame_paths] * batch_size
            img_transform = T.Compose([
                T.ToTensor(),
                T.Resize(height, antialias=None),
                T.CenterCrop((height, width)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
            first_frames = []
            for first_frame_path in first_frame_paths:
                first_frame = Image.open(first_frame_path).convert('RGB')
                first_frame = img_transform(first_frame).unsqueeze(0)
                first_frames.append(first_frame)
            first_frames = torch.cat(first_frames, dim=0)
        if first_frames is not None:
            first_frames = first_frames.to(device, dtype=self.vae.dtype)
            first_frame_latents = self.vae.encode(first_frames).latent_dist
            first_frame_latents = first_frame_latents.sample()
            first_frame_latents = first_frame_latents * self.vae.config.scaling_factor # b, c, h, w
            first_frame_latents = repeat(first_frame_latents, "b c h w -> (b n) c h w", n=num_videos_per_prompt)
            first_frames = repeat(first_frames, "b c h w -> (b n) c h w", n=num_videos_per_prompt)

        full_video_latent = torch.zeros(batch_size * num_videos_per_prompt, self.unet.config.in_channels, video_length * autoregress_steps - autoregress_steps + 1, height // self.vae_scale_factor, width // self.vae_scale_factor, device=device, dtype=self.vae.dtype)

        start_idx = 0
        for ar_step in range(autoregress_steps):
            # Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            latents = self.prepare_latents(
                batch_size * num_videos_per_prompt,
                num_channels_latents,
                video_length,
                height,
                width,
                text_embeddings.dtype,
                device,
                generator,
                latents,
                noise_sampling_method,
                noise_alpha,
            )
            latents_dtype = latents.dtype
            
            if use_frameinit:
                current_diffuse_timestep = frameinit_noise_level # diffuse to noise level
                diffuse_timesteps = torch.full((batch_size,),int(current_diffuse_timestep))
                diffuse_timesteps = diffuse_timesteps.long()
                first_frames_static_vid = repeat(first_frame_latents, "b c h w -> b c t h w", t=video_length)
                z_T = self.scheduler.add_noise(
                    original_samples=first_frames_static_vid.to(device), 
                    noise=latents.to(device), 
                    timesteps=diffuse_timesteps.to(device)
                )
                latents = freq_mix_3d(z_T.to(dtype=torch.float32), latents, LPF=self.freq_filter)
                latents = latents.to(dtype=latents_dtype)
            
            if first_frame_latents is not None:
                first_frame_noisy_latent = latents[:, :, 0, :, :]
                latents = latents[:, :, 1:, :, :]

            # Prepare extra step kwargs.
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    if do_classifier_free_guidance is None:
                        latent_model_input = latents
                    elif do_classifier_free_guidance == "text":
                        latent_model_input = torch.cat([latents] * 2)
                    elif do_classifier_free_guidance == "both":
                        latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    if first_frame_latents is not None:
                        if do_classifier_free_guidance is None:
                            first_frame_latents_input = first_frame_latents
                        elif do_classifier_free_guidance == "text":
                            first_frame_latents_input = torch.cat([first_frame_latents] * 2)
                        elif do_classifier_free_guidance == "both":
                            first_frame_latents_input = torch.cat([first_frame_noisy_latent, first_frame_latents, first_frame_latents])

                        first_frame_latents_input = first_frame_latents_input.unsqueeze(2)

                        # predict the noise residual
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings, first_frame_latents=first_frame_latents_input, frame_stride=frame_stride).sample.to(dtype=latents_dtype)
                    else:
                        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample.to(dtype=latents_dtype)
                    # noise_pred = []
                    # import pdb
                    # pdb.set_trace()
                    # for batch_idx in range(latent_model_input.shape[0]):
                    #     noise_pred_single = self.unet(latent_model_input[batch_idx:batch_idx+1], t, encoder_hidden_states=text_embeddings[batch_idx:batch_idx+1]).sample.to(dtype=latents_dtype)
                    #     noise_pred.append(noise_pred_single)
                    # noise_pred = torch.cat(noise_pred)

                    # perform guidance
                    if do_classifier_free_guidance:
                        if do_classifier_free_guidance == "text":
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale_txt * (noise_pred_text - noise_pred_uncond)
                        elif do_classifier_free_guidance == "both":
                            noise_pred_uncond, noise_pred_img, noise_pred_both = noise_pred.chunk(3)
                            noise_pred = noise_pred_uncond + guidance_scale_img * (noise_pred_img - noise_pred_uncond) + guidance_scale_txt * (noise_pred_both - noise_pred_img)
                    
                    if do_classifier_free_guidance and guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        # currently only support text guidance
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

            # Post-processing
            
            latents = torch.cat([first_frame_latents.unsqueeze(2), latents], dim=2)
            first_frame_latents = latents[:, :, -1, :, :]
            full_video_latent[:, :, start_idx:start_idx + video_length, :, :] = latents

            latents = None
            start_idx += (video_length - 1)

        # video = self.decode_latents(latents, first_frames)
        video = self.decode_latents(full_video_latent)

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)
