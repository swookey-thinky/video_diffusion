"""Utility layers used in defining a DDPM Unets with Dropout."""

import torch
from typing import Dict

from video_diffusion.layers.mlp import Mlp
from video_diffusion.layers.utils import conv_nd, avg_pool_nd, zero_module, ContextBlock


class ResnetBlockDDPM3D(ContextBlock):
    """ResNet block based on WideResNet architecture.

    From DDPM, uses GroupNorm instead of weight normalization and Swish activation.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        time_emb_dim=None,
        use_scale_shift_norm=False,
        dropout=0.0,
        mlp_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self._use_scale_shift_norm = use_scale_shift_norm

        # From https://arxiv.org/abs/2204.03458, they found
        # that multiple MLP layers helped.
        self.timestep_proj = (
            torch.nn.Sequential(
                *[
                    Mlp(
                        in_features=(
                            time_emb_dim
                            if idx == 0
                            else (2 * dim_out if use_scale_shift_norm else dim_out)
                        ),
                        out_features=(2 * dim_out if use_scale_shift_norm else dim_out),
                        act_layer=torch.nn.SiLU,
                    )
                    for idx in range(mlp_layers)
                ]
            )
            if time_emb_dim is not None
            else None
        )

        self.block1 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_in),
            torch.nn.SiLU(),
            torch.nn.Conv3d(dim_in, dim_out, (1, 3, 3), padding=(0, 1, 1)),
        )

        # In the DDPM implementation, dropout was added to the second
        # resnet layer in the block, in front of the final convolution.
        self.block2 = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_out),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout) if dropout > 0.0 else torch.nn.Identity(),
            torch.nn.Conv3d(dim_out, dim_out, (1, 3, 3), padding=(0, 1, 1)),
        )

        self.residual_proj = torch.nn.Linear(dim_in, dim_out)
        self.dim_out = dim_out

    def forward(self, x, context: Dict):
        B, C, H, W = x.shape
        h = self.block1(x)

        # Add in the timstep embedding between blocks 1 and 2
        if "timestep_embedding" in context and self.timestep_proj is not None:
            time_emb = context["timestep_embedding"]
            emb_out = self.timestep_proj(time_emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]

            # Scale/Shift of the norm here is from the IDDPM paper,
            # as one of their improvements.
            if self._use_scale_shift_norm:
                out_norm, out_rest = self.block2[0], self.block2[1:]
                scale, shift = torch.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h += emb_out
                h = self.block2(h)

        # Project the residual channel to the output dimensions
        if C != self.dim_out:
            x = self.residual_proj(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return h + x


class ResnetBlockBigGAN3D(ContextBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        dim_in,
        time_emb_dim,
        dropout,
        dim_out=None,
        use_conv=False,
        use_scale_shift_norm=False,
        up=False,
        down=False,
        mlp_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.input_channels = dim_in
        self.emb_channels = time_emb_dim
        self.dropout = dropout
        self.out_channels = dim_out or dim_in
        self.use_conv = use_conv

        # 3D convolutions
        convolution_dims = 3

        # Scale/shift norm is called Adaptive Group Normalization
        # in the paper.
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=dim_in),
            torch.nn.SiLU(),
            conv_nd(
                convolution_dims,
                dim_in,
                self.out_channels,
                (1, 3, 3),
                padding=(0, 1, 1),
            ),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(dim_in, False, convolution_dims)
            self.x_upd = Upsample(dim_in, False, convolution_dims)
        elif down:
            self.h_upd = Downsample(dim_in, False, convolution_dims)
            self.x_upd = Downsample(dim_in, False, convolution_dims)
        else:
            self.h_upd = self.x_upd = torch.nn.Identity()

        # From https://arxiv.org/abs/2204.03458, they found
        # that multiple MLP layers helped.
        self.emb_layers = torch.nn.Sequential(
            *[
                Mlp(
                    in_features=(
                        time_emb_dim
                        if idx == 0
                        else (
                            2 * self.out_channels
                            if use_scale_shift_norm
                            else self.out_channels
                        )
                    ),
                    out_features=(
                        2 * self.out_channels
                        if use_scale_shift_norm
                        else self.out_channels
                    ),
                    act_layer=torch.nn.SiLU,
                )
                for idx in range(mlp_layers)
            ]
        )

        self.out_layers = torch.nn.Sequential(
            torch.nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout),
            zero_module(
                conv_nd(
                    convolution_dims,
                    self.out_channels,
                    self.out_channels,
                    (1, 3, 3),
                    padding=(0, 1, 1),
                )
            ),
        )

        if self.out_channels == dim_in:
            self.skip_connection = torch.nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                convolution_dims,
                dim_in,
                self.out_channels,
                (1, 3, 3),
                padding=(0, 1, 1),
            )
        else:
            self.skip_connection = conv_nd(
                convolution_dims, dim_in, self.out_channels, 1
            )

    def forward(self, x, context: Dict):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb = context["timestep_embedding"]
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class Downsample(torch.nn.Module):
    """A downsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        kernel_size = 3 if dims != 3 else (1, 3, 3)
        padding = 1 if dims != 3 else (0, 1, 1)
        if use_conv:
            self.op = conv_nd(
                dims, channels, channels, kernel_size, stride=stride, padding=padding
            )
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class Upsample(torch.nn.Module):
    """An upsampling layer with an optional convolution.

    Args:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
            upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims

        kernel_size = 3 if dims != 3 else (1, 3, 3)
        padding = 1 if dims != 3 else (0, 1, 1)

        if use_conv:
            self.conv = conv_nd(dims, channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = torch.nn.functional.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x
