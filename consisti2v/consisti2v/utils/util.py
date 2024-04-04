import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision
import torch.distributed as dist
import wandb

from tqdm import tqdm
from einops import rearrange

from torchmetrics.image.fid import _compute_fid


def zero_rank_print(s):
    if (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0): print("### " + s)


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8, wandb=False, global_step=0, format="gif"):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    if wandb:
        wandb_video = wandb.Video(outputs, fps=fps)
        wandb.log({"val_videos": wandb_video}, step=global_step)
        
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if format == "gif":
        imageio.mimsave(path, outputs, fps=fps)
    elif format == "mp4":
        torchvision.io.write_video(path, np.array(outputs), fps=fps, video_codec='h264', options={'crf': '10'})

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, first_frame_latents, frame_stride, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context, first_frame_latents=first_frame_latents, frame_stride=frame_stride).sample
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt, first_frame_latents, frame_stride):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, first_frame_latents, frame_stride, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt="", first_frame_latents=None, frame_stride=3):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt, first_frame_latents, frame_stride)
    return ddim_latents


def compute_fid(real_features, fake_features, num_features, device):
    orig_dtype = real_features.dtype

    mx_num_feats = (num_features, num_features)
    real_features_sum = torch.zeros(num_features).double().to(device)
    real_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
    real_features_num_samples = torch.tensor(0).long().to(device)

    fake_features_sum = torch.zeros(num_features).double().to(device)
    fake_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
    fake_features_num_samples = torch.tensor(0).long().to(device)

    real_features = real_features.double()
    fake_features = fake_features.double()

    real_features_sum += real_features.sum(dim=0)
    real_features_cov_sum += real_features.t().mm(real_features)
    real_features_num_samples += real_features.shape[0]

    fake_features_sum += fake_features.sum(dim=0)
    fake_features_cov_sum += fake_features.t().mm(fake_features)
    fake_features_num_samples += fake_features.shape[0]

    """Calculate FID score based on accumulated extracted features from the two distributions."""
    if real_features_num_samples < 2 or fake_features_num_samples < 2:
        raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
    mean_real = (real_features_sum / real_features_num_samples).unsqueeze(0)
    mean_fake = (fake_features_sum / fake_features_num_samples).unsqueeze(0)

    cov_real_num = real_features_cov_sum - real_features_num_samples * mean_real.t().mm(mean_real)
    cov_real = cov_real_num / (real_features_num_samples - 1)
    cov_fake_num = fake_features_cov_sum - fake_features_num_samples * mean_fake.t().mm(mean_fake)
    cov_fake = cov_fake_num / (fake_features_num_samples - 1)
    return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(orig_dtype)


def compute_inception_score(gen_probs, num_splits=10):
    num_gen = gen_probs.shape[0]
    gen_probs = gen_probs.detach().cpu().numpy()
    scores = []
    np.random.RandomState(42).shuffle(gen_probs)
    for i in range(num_splits):
        part = gen_probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))
    # idx = torch.randperm(features.shape[0])
    # features = features[idx]
    # # calculate probs and logits
    # prob = features.softmax(dim=1)
    # log_prob = features.log_softmax(dim=1)

    # # split into groups
    # prob = prob.chunk(splits, dim=0)
    # log_prob = log_prob.chunk(splits, dim=0)

    # # calculate score per split
    # mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
    # kl_ = [p * (log_p - m_p.log()) for p, log_p, m_p in zip(prob, log_prob, mean_prob)]
    # kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
    # kl = torch.stack(kl_)

    # return mean and std
    # return kl.mean(), kl.std()