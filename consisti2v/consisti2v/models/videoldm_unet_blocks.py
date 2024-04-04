from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from diffusers.utils import logging
from diffusers.models.unet_2d_blocks import (
    DownBlock2D,
    UpBlock2D
)
from diffusers.models.resnet import (
    ResnetBlock2D,
    Downsample2D,
    Upsample2D,
)
from diffusers.models.transformer_2d import Transformer2DModelOutput
from diffusers.models.dual_transformer_2d import DualTransformer2DModel
from diffusers.models.activations import get_activation
from diffusers.utils import logging, is_torch_version
from diffusers.utils.import_utils import is_xformers_available
from .videoldm_transformer_blocks import Transformer2DConditionModel

logger = logging.get_logger(__name__)

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    transformer_layers_per_block=1,
    num_attention_heads=None,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=None,
    downsample_type=None,
    dropout=0.0,
    # additional
    use_temporal=True,
    augment_temporal_attention=False,
    n_frames=8,
    n_temp_heads=8,
    first_frame_condition_mode="none",
    latent_channels=4,
    rotary_emb=False,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads
        
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return VideoLDMDownBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            # additional
            use_temporal=use_temporal,
            n_frames=n_frames,
            first_frame_condition_mode=first_frame_condition_mode,
            latent_channels=latent_channels
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        return VideoLDMCrossAttnDownBlock(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            # additional
            use_temporal=use_temporal,
            augment_temporal_attention=augment_temporal_attention,
            n_frames=n_frames,
            n_temp_heads=n_temp_heads,
            first_frame_condition_mode=first_frame_condition_mode,
            latent_channels=latent_channels,
            rotary_emb=rotary_emb,
        )

    raise ValueError(f'{down_block_type} does not exist.')


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    transformer_layers_per_block=1,
    num_attention_heads=None,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    attention_type="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=None,
    upsample_type=None,
    dropout=0.0,
    # additional
    use_temporal=True,
    augment_temporal_attention=False,
    n_frames=8,
    n_temp_heads=8,
    first_frame_condition_mode="none",
    latent_channels=4,
    rotary_emb=None,
):
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return VideoLDMUpBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            # additional
            use_temporal=use_temporal,
            n_frames=n_frames,
            first_frame_condition_mode=first_frame_condition_mode,
            latent_channels=latent_channels
        )
    elif up_block_type == 'CrossAttnUpBlock2D':
        return VideoLDMCrossAttnUpBlock(
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_type=attention_type,
            # additional
            use_temporal=use_temporal,
            augment_temporal_attention=augment_temporal_attention,
            n_frames=n_frames,
            n_temp_heads=n_temp_heads,
            first_frame_condition_mode=first_frame_condition_mode,
            latent_channels=latent_channels,
            rotary_emb=rotary_emb,
        )

    raise ValueError(f'{up_block_type} does not exist.')


class TemporalResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        # additional
        n_frames=8,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = Conv3DLayer(in_channels, out_channels, n_frames=n_frames)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv3DLayer(out_channels, out_channels, n_frames=n_frames)

        self.nonlinearity = get_activation(non_linearity)

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        # weighted sum between spatial and temporal features
        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        output_tensor = self.alpha * input_tensor + (1 - self.alpha) * output_tensor

        return output_tensor


class Conv3DLayer(nn.Conv3d):
    def __init__(self, in_dim, out_dim, n_frames):
        k, p = (3, 1, 1), (1, 0, 0)
        super().__init__(in_channels=in_dim, out_channels=out_dim, kernel_size=k, stride=1, padding=p)

        self.to_3d = Rearrange('(b t) c h w -> b c t h w', t=n_frames)
        self.to_2d = Rearrange('b c t h w -> (b t) c h w')

    def forward(self, x):
        h = self.to_3d(x)
        h = super().forward(h)
        out = self.to_2d(h)
        return out


class IdentityLayer(nn.Identity):
    def __init__(self, return_trans2d_output, *args, **kwargs):
        super().__init__()
        self.return_trans2d_output = return_trans2d_output

    def forward(self, x, *args, **kwargs):
        if self.return_trans2d_output:
            return Transformer2DModelOutput(sample=x)
        else:
            return x


class VideoLDMCrossAttnDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type="default",
        # additional
        use_temporal=True,
        augment_temporal_attention=False,
        n_frames=8,
        n_temp_heads=8,
        first_frame_condition_mode="none",
        latent_channels=4,
        rotary_emb=False,
    ):
        super().__init__()

        self.use_temporal = use_temporal

        self.n_frames = n_frames
        self.first_frame_condition_mode = first_frame_condition_mode
        if self.first_frame_condition_mode == "conv2d":
            self.first_frame_conv = nn.Conv2d(latent_channels, in_channels, kernel_size=1)

        resnets = []
        attentions = []

        self.n_frames = n_frames
        self.n_temp_heads = n_temp_heads

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DConditionModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        # additional
                        n_frames=n_frames,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

        # >>> Temporal Layers >>>
        conv3ds = []
        tempo_attns = []

        for i in range(num_layers):
            if self.use_temporal:
                conv3ds.append(
                    TemporalResnetBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        n_frames=n_frames,
                    )
                )

                tempo_attns.append(
                    Transformer2DConditionModel(
                        n_temp_heads,
                        out_channels // n_temp_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        # additional
                        n_frames=n_frames,
                        is_temporal=True,
                        augment_temporal_attention=augment_temporal_attention,
                        rotary_emb=rotary_emb
                    )
                )
            else:
                conv3ds.append(IdentityLayer(return_trans2d_output=False))
                tempo_attns.append(IdentityLayer(return_trans2d_output=True))

        self.conv3ds = nn.ModuleList(conv3ds)
        self.tempo_attns = nn.ModuleList(tempo_attns)
        # <<< Temporal Layers <<<

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # additional
        first_frame_latents=None,
    ):
        condition_on_first_frame = (self.first_frame_condition_mode != "none" and self.first_frame_condition_mode != "input_only")
        # input shape: hidden_states = (b f) c h w, first_frame_latents = b c 1 h w
        if self.first_frame_condition_mode == "conv2d":
            hidden_states = rearrange(hidden_states, '(b t) c h w -> b c t h w', t=self.n_frames)
            hidden_height = hidden_states.shape[3]
            first_frame_height = first_frame_latents.shape[3]
            downsample_ratio = hidden_height / first_frame_height
            first_frame_latents = F.interpolate(first_frame_latents.squeeze(2), scale_factor=downsample_ratio, mode="nearest")
            first_frame_latents = self.first_frame_conv(first_frame_latents).unsqueeze(2)
            hidden_states[:, :, 0:1, :, :] = first_frame_latents
            hidden_states = rearrange(hidden_states, 'b c t h w -> (b t) c h w', t=self.n_frames)

        output_states = ()

        for resnet, conv3d, attn, tempo_attn in zip(self.resnets, self.conv3ds, self.attentions, self.tempo_attns):

            hidden_states = resnet(hidden_states, temb)
            hidden_states = conv3d(hidden_states)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                condition_on_first_frame=condition_on_first_frame,
            ).sample
            hidden_states = tempo_attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                condition_on_first_frame=False,
            ).sample

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class VideoLDMCrossAttnUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        attention_type="default",
        # additional
        use_temporal=True,
        augment_temporal_attention=False,
        n_frames=8,
        n_temp_heads=8,
        first_frame_condition_mode="none",
        latent_channels=4,
        rotary_emb=False,
    ):
        super().__init__()

        self.use_temporal = use_temporal

        self.n_frames = n_frames
        self.first_frame_condition_mode = first_frame_condition_mode
        if self.first_frame_condition_mode == "conv2d":
            self.first_frame_conv = nn.Conv2d(latent_channels, prev_output_channel, kernel_size=1)

        resnets = []
        attentions = []

        self.n_frames = n_frames
        self.n_temp_heads = n_temp_heads

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DConditionModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        # additional
                        n_frames=n_frames,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

        # >>> Temporal Layers >>>
        conv3ds = []
        tempo_attns = []

        for i in range(num_layers):
            if self.use_temporal:
                conv3ds.append(
                    TemporalResnetBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        n_frames=n_frames,
                    )
                )

                tempo_attns.append(
                    Transformer2DConditionModel(
                        n_temp_heads,
                        out_channels // n_temp_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        # additional
                        n_frames=n_frames,
                        augment_temporal_attention=augment_temporal_attention,
                        is_temporal=True,
                        rotary_emb=rotary_emb,
                    )
                )
            else:
                conv3ds.append(IdentityLayer(return_trans2d_output=False))
                tempo_attns.append(IdentityLayer(return_trans2d_output=True))

        self.conv3ds = nn.ModuleList(conv3ds)
        self.tempo_attns = nn.ModuleList(tempo_attns)
        # <<< Temporal Layers <<<

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # additional
        first_frame_latents=None,
    ):
        condition_on_first_frame = (self.first_frame_condition_mode != "none" and self.first_frame_condition_mode != "input_only")
        # input shape: hidden_states = (b f) c h w, first_frame_latents = b c 1 h w
        if self.first_frame_condition_mode == "conv2d":
            hidden_states = rearrange(hidden_states, '(b t) c h w -> b c t h w', t=self.n_frames)
            hidden_height = hidden_states.shape[3]
            first_frame_height = first_frame_latents.shape[3]
            downsample_ratio = hidden_height / first_frame_height
            first_frame_latents = F.interpolate(first_frame_latents.squeeze(2), scale_factor=downsample_ratio, mode="nearest")
            first_frame_latents = self.first_frame_conv(first_frame_latents).unsqueeze(2)
            hidden_states[:, :, 0:1, :, :] = first_frame_latents
            hidden_states = rearrange(hidden_states, 'b c t h w -> (b t) c h w', t=self.n_frames)

        for resnet, conv3d, attn, tempo_attn in zip(self.resnets, self.conv3ds, self.attentions, self.tempo_attns):

            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = conv3d(hidden_states)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                condition_on_first_frame=condition_on_first_frame,
            ).sample
            hidden_states = tempo_attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                condition_on_first_frame=False,
            ).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states
    

class VideoLDMUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers_per_block: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        num_attention_heads=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
        attention_type="default",
        # additional
        use_temporal=True,
        n_frames: int = 8,
        first_frame_condition_mode="none",
        latent_channels=4,
    ):
        super().__init__()

        self.use_temporal = use_temporal

        self.n_frames = n_frames
        self.first_frame_condition_mode = first_frame_condition_mode
        if self.first_frame_condition_mode == "conv2d":
            self.first_frame_conv = nn.Conv2d(latent_channels, in_channels, kernel_size=1)

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        if self.use_temporal:
            conv3ds = [
                TemporalResnetBlock(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    n_frames=n_frames,
                )
            ]
        else:
            conv3ds = [IdentityLayer(return_trans2d_output=False)]

        attentions = []

        for _ in range(num_layers):
            if not dual_cross_attention:
                attentions.append(
                    Transformer2DConditionModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                        # additional
                        n_frames=n_frames,
                    )
                )
            else:
                attentions.append(
                    DualTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=1,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                    )
                )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if self.use_temporal:
                conv3ds.append(
                    TemporalResnetBlock(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        n_frames=n_frames,
                    )
                )
            else:
                conv3ds.append(IdentityLayer(return_trans2d_output=False))

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.conv3ds = nn.ModuleList(conv3ds)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # additional
        first_frame_latents=None,
    ) -> torch.FloatTensor:
        condition_on_first_frame = (self.first_frame_condition_mode != "none" and self.first_frame_condition_mode != "input_only")
        # input shape: hidden_states = (b f) c h w, first_frame_latents = b c 1 h w
        if self.first_frame_condition_mode == "conv2d":
            hidden_states = rearrange(hidden_states, '(b t) c h w -> b c t h w', t=self.n_frames)
            hidden_height = hidden_states.shape[3]
            first_frame_height = first_frame_latents.shape[3]
            downsample_ratio = hidden_height / first_frame_height
            first_frame_latents = F.interpolate(first_frame_latents.squeeze(2), scale_factor=downsample_ratio, mode="nearest")
            first_frame_latents = self.first_frame_conv(first_frame_latents).unsqueeze(2)
            hidden_states[:, :, 0:1, :, :] = first_frame_latents
            hidden_states = rearrange(hidden_states, 'b c t h w -> (b t) c h w', t=self.n_frames)

        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0
        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        hidden_states = self.conv3ds[0](hidden_states)
        for attn, resnet, conv3d in zip(self.attentions, self.resnets[1:], self.conv3ds[1:]):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    # additional
                    condition_on_first_frame=condition_on_first_frame,
                )[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(resnet),
                    hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
                hidden_states = conv3d(hidden_states)
            else:
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                    # additional
                    condition_on_first_frame=condition_on_first_frame,
                )[0]
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)
                hidden_states = conv3d(hidden_states)

        return hidden_states


class VideoLDMDownBlock(DownBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
        # additional
        use_temporal=True,
        n_frames: int = 8,
        first_frame_condition_mode="none",
        latent_channels=4,
    ):
        super().__init__(
            in_channels,
            out_channels,
            temb_channels,
            dropout,
            num_layers,
            resnet_eps,
            resnet_time_scale_shift,
            resnet_act_fn,
            resnet_groups,
            resnet_pre_norm,
            output_scale_factor,
            add_downsample,
            downsample_padding,)

        self.use_temporal = use_temporal

        self.n_frames = n_frames
        self.first_frame_condition_mode = first_frame_condition_mode
        if self.first_frame_condition_mode == "conv2d":
            self.first_frame_conv = nn.Conv2d(latent_channels, in_channels, kernel_size=1)

        # >>> Temporal Layers >>>
        conv3ds = []
        for i in range(num_layers):
            if self.use_temporal:
                conv3ds.append(
                    TemporalResnetBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        n_frames=n_frames,
                    )
                )
            else:
                conv3ds.append(IdentityLayer(return_trans2d_output=False))
        self.conv3ds = nn.ModuleList(conv3ds)
        # <<< Temporal Layers <<<

    def forward(self, hidden_states, temb=None, scale: float = 1, first_frame_latents=None):
        # input shape: hidden_states = (b f) c h w, first_frame_latents = b c 1 h w
        if self.first_frame_condition_mode == "conv2d":
            hidden_states = rearrange(hidden_states, '(b t) c h w -> b c t h w', t=self.n_frames)
            hidden_height = hidden_states.shape[3]
            first_frame_height = first_frame_latents.shape[3]
            downsample_ratio = hidden_height / first_frame_height
            first_frame_latents = F.interpolate(first_frame_latents.squeeze(2), scale_factor=downsample_ratio, mode="nearest")
            first_frame_latents = self.first_frame_conv(first_frame_latents).unsqueeze(2)
            hidden_states[:, :, 0:1, :, :] = first_frame_latents
            hidden_states = rearrange(hidden_states, 'b c t h w -> (b t) c h w', t=self.n_frames)

        output_states = ()

        for resnet, conv3d in zip(self.resnets, self.conv3ds):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)

            hidden_states = conv3d(hidden_states)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale=scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states
    

class VideoLDMUpBlock(UpBlock2D):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
        # additional
        use_temporal=True,
        n_frames: int = 8,
        first_frame_condition_mode="none",
        latent_channels=4,
    ):
        super().__init__(
            in_channels,
            prev_output_channel,
            out_channels,
            temb_channels,
            dropout,
            num_layers,
            resnet_eps,
            resnet_time_scale_shift,
            resnet_act_fn,
            resnet_groups,
            resnet_pre_norm,
            output_scale_factor,
            add_upsample,
        )

        self.use_temporal = use_temporal

        self.n_frames = n_frames
        self.first_frame_condition_mode = first_frame_condition_mode
        if self.first_frame_condition_mode == "conv2d":
            self.first_frame_conv = nn.Conv2d(latent_channels, prev_output_channel, kernel_size=1)

        # >>> Temporal Layers >>>
        conv3ds = []
        for i in range(num_layers):
            if self.use_temporal:
                conv3ds.append(
                    TemporalResnetBlock(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        n_frames=n_frames,
                    )
                )
            else:
                conv3ds.append(IdentityLayer(return_trans2d_output=False))

        self.conv3ds = nn.ModuleList(conv3ds)
        # <<< Temporal Layers <<<

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, scale: float = 1, first_frame_latents=None):
        # input shape: hidden_states = (b f) c h w, first_frame_latents = b c 1 h w
        if self.first_frame_condition_mode == "conv2d":
            hidden_states = rearrange(hidden_states, '(b t) c h w -> b c t h w', t=self.n_frames)
            hidden_height = hidden_states.shape[3]
            first_frame_height = first_frame_latents.shape[3]
            downsample_ratio = hidden_height / first_frame_height
            first_frame_latents = F.interpolate(first_frame_latents.squeeze(2), scale_factor=downsample_ratio, mode="nearest")
            first_frame_latents = self.first_frame_conv(first_frame_latents).unsqueeze(2)
            hidden_states[:, :, 0:1, :, :] = first_frame_latents
            hidden_states = rearrange(hidden_states, 'b c t h w -> (b t) c h w', t=self.n_frames)

        for resnet, conv3d in zip(self.resnets, self.conv3ds):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)
            
            hidden_states = conv3d(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size, scale=scale)

        return hidden_states