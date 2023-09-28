# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from functools import partial
import numpy as np

import os

SPECIALIZED = int(os.environ["SPECIALIZED"]) if "SPECIALIZED" in os.environ else 0

if not SPECIALIZED:
    print("Using specialized kernels: NO")
    from dwoconv1d import DepthwiseOrientedConv1d
else:
    print("Using specialized kernels: YES")
    from dwoconv1d_specialized import DepthwiseOrientedConv1d


# LayerNorm
## Taken from ConvNeXt: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Layer-wise angle utilities
def channel_cycle_offset(c, C, N):
    r"""Given N directions and C channels, output the angle for the current channel c.
    We split the angles to form N groups of ~C/N channels. The channels of group n share the same angle theta = n / N * 180 degrees
    """
    n = (N * (c + 1) - 1) // C
    return n / N


def layer_wise_rotation_offset(k, K, enable_layer_cycle=0):
    r"""Layer-wise rotation. Add a 45 degree offset for every layer at odd depth. Enabled / disabled by enable_layer_cycle"""
    if enable_layer_cycle:
        return (k % 2) / 2
    return 0


def theta_offset(x):
    r"""Convert the offset to an angle theta. Kernels shifted by 180 degrees have same angles so only keep angles [0, 180[."""
    return np.pi * (x % 1)


def get_convolution_1d(
    in_channels,
    out_channels,
    *,
    kernel_size,
    padding,
    groups,
    stride=1,
    N=8,
    layer_offset=0,
    nhwc=False,
    resid=False,
    bias=True,
    arbitrary_size=False,
):
    r"""1D oriented convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the depthwise convolution.
        padding (int): Padding for the depthwise convolution.
        groups (int): Number of groups for the depthwise convolution.
        stride (int): Stride for the depthwise convolution. Default: 1.
        N (int): Number of directions for oriented kernel. Default: 8.
        layer_offset (float): layer-wise rotation offset. Default: 0.
        nhwc (bool): Use fused implementation for NHWC input. Default: False.
        resid (bool): Use fused implementation for residual connection. Default: False.
        bias (bool): Use bias. Default: True.
    """
    assert in_channels == out_channels == groups
    theta = [
        theta_offset(channel_cycle_offset(k, in_channels, N) + layer_offset)
        for k in range(in_channels)
    ]
    conv = DepthwiseOrientedConv1d(
        in_channels=in_channels,
        out_channels=in_channels,
        groups=in_channels,
        stride=stride,
        nhwc=nhwc,
        resid=resid,
        kernel_size=kernel_size,
        padding=padding,
        bias=bias,
        angle=theta,
        arbitrary_size=arbitrary_size,
    )
    return conv


# Stems
class Stem1D(nn.Module):
    r"""1D Stem for ConvNeXt-1D and ConvNeXt-1D++.
    Args:
        dim0 (int): Number of input channels. Default: 3.
        dim1 (int): Number of intermediate channels. Default: 64.
        dim2 (int): Number of output channels. Default: 96.
        kernel_size (int): Kernel size for the depthwise convolutions. Default: 31.
        enable_layer_cycle (int): Enable layer-wise rotation. Default: 0.
    """

    def __init__(
        self, dim0, dim1, dim2, kernel_size, enable_layer_cycle=0, arbitrary_size=False
    ):
        super().__init__()

        self.pw1 = nn.Linear(dim0, dim1)
        self.pw2 = nn.Linear(dim1, dim2)
        stride = 2
        padding = (kernel_size - stride) // 2
        self.dw1 = get_convolution_1d(
            dim1,
            dim1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim1,
            nhwc=True,
            resid=False,
            layer_offset=layer_wise_rotation_offset(0, 2, enable_layer_cycle),
            arbitrary_size=arbitrary_size,
        )
        self.dw2 = get_convolution_1d(
            dim1,
            dim1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim1,
            nhwc=True,
            resid=False,
            layer_offset=layer_wise_rotation_offset(1, 2, enable_layer_cycle),
            arbitrary_size=arbitrary_size,
        )
        self.norm1 = LayerNorm(dim1, eps=1e-6, data_format="channels_last")
        self.norm2 = LayerNorm(dim1, eps=1e-6, data_format="channels_last")
        self.norm3 = LayerNorm(dim2, eps=1e-6, data_format="channels_last")

    def forward(self, x):
        x = x.permute((0, 2, 3, 1)).contiguous()
        x = self.pw1(x)
        x = self.dw1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dw2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = self.norm3(x)
        x = x.permute((0, 3, 1, 2)).contiguous()
        return x


# Blocks
class Block2D(nn.Module):
    r"""2D Block for ConvNeXt-2D.

    Args:
        dim (int): Number of input/output channels.
        kernel_sizes (list[int]): List of kernel sizes for the depthwise convolution. Default: [7].
        N (int): Number of directions for oriented kernels. Default: 80.
        drop_path (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        layer_offset (float): Layer-wise rotation offset. Default: 0.
    """

    def __init__(
        self,
        dim,
        kernel_sizes=[7],
        N=0,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        layer_offset=0,
        arbitrary_size=False,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            groups=dim,
        )

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Block1D(nn.Module):
    r"""1D Block for ConvNeXt-1D.

    Args:
        dim (int): Number of input/output channels.
        kernel_sizes (list[int]): List of kernel sizes for the depthwise convolution. Default: [31].
        N (int): Number of directions for oriented kernels. Default: 8.
        drop_path (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        layer_offset (float): Layer-wise rotation offset. Default: 0.
    """

    def __init__(
        self,
        dim,
        kernel_sizes=[31],
        N=8,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        layer_offset=0,
        arbitrary_size=False,
    ):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.dwconv = get_convolution_1d(
            dim,
            dim,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            groups=dim,
            nhwc=False,
            resid=False,
            N=N,
            layer_offset=layer_offset,
            bias=True,
            arbitrary_size=arbitrary_size,
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Block1DPP(nn.Module):
    r"""1D++ Block for ConvNeXt-1D++.

    Args:
        dim (int): Number of input/output channels.
        kernel_sizes (tuple(int)): Kernel sizes for the two depthwise convolutions. Default: [15, 31].
        N (int): Number of directions for oriented kernels. Default: 8.
        drop_path (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        layer_offset (float): Layer-wise rotation offset. Default: 0.
    """

    def __init__(
        self,
        dim,
        kernel_sizes=[15, 31],
        N=8,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        layer_offset=0,
        arbitrary_size=False,
    ):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.dwconv1 = get_convolution_1d(
            dim,
            dim,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            groups=dim,
            nhwc=False,
            resid=False,
            N=N,
            layer_offset=layer_offset,
            arbitrary_size=arbitrary_size,
        )
        # use residual connection and nhwc format for dwconv2
        self.dwconv2 = get_convolution_1d(
            4 * dim,
            4 * dim,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
            groups=4 * dim,
            nhwc=True,
            resid=True,
            N=N,
            layer_offset=layer_offset,
            arbitrary_size=arbitrary_size,
        )

    def forward(self, x):
        input = x
        x = self.dwconv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.dwconv2(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Block2DPP(nn.Module):
    r"""2D++ Block for ConvNeXt-2D++.

    Args:
        dim (int): Number of input/output channels.
        kernel_sizes (tuple(int)): Kernel sizes for the two depthwise convolutions. Default: [7, 31].
        N (int): Number of directions for oriented kernels. Default: 8.
        drop_path (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        layer_offset (float): Layer-wise rotation offset. Default: 0.
    """

    def __init__(
        self,
        dim,
        kernel_sizes=[7, 31],
        N=8,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        layer_offset=0,
        arbitrary_size=False,
    ):
        super().__init__()
        assert len(kernel_sizes) == 2

        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.dwconv1 = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            groups=dim,
        )
        # use residual connection and nhwc format for dwconv2
        self.dwconv2 = get_convolution_1d(
            4 * dim,
            4 * dim,
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2,
            groups=4 * dim,
            nhwc=True,
            resid=True,
            N=N,
            layer_offset=layer_offset,
            arbitrary_size=arbitrary_size,
        )

    def forward(self, x):
        input = x
        x = self.dwconv1(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.dwconv2(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


# Define network parameters
Blocks = {
    "2d": Block2D,
    "2dpp": Block2DPP,
    "1d": Block1D,
    "1dpp": Block1DPP,
}
block_sizes = {
    "2d": [[7], [7], [7], [7]],
    "2dpp": [[7, 31], [7, 31], [7, 31], [7, 31]],
    "1d": [[31], [31], [31], [31]],
    "1dpp": [[15, 31], [15, 31], [15, 31], [15, 31]],
}

stem_sizes = {"2d": 4, "2dpp": 4, "1d": 31, "1dpp": 15}


class ConvNeXt(nn.Module):
    r"""ConvNeXt-1D/2D/1D++/2D++ models.

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        variant (str): ConvNeXt-1D/2D/1D++/2D++ variant. Default: '2d'.
        N (int): Number of directions for oriented kernels. Default: 8.
        stem_dim1 (int): Intermediate stem dimension for 1D/1D++ stems. Default: 64.
    """

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        variant="2d",
        N=8,
        stem_dim1=64,
        arbitrary_size=False,
        **kwargs,
    ):
        super().__init__()
        print(f"variant = {variant}")
        assert variant in ["2d", "2dpp", "1d", "1dpp"]

        enable_layer_cycle = int(variant in ["1d", "1dpp", "2dpp"])
        if variant in ["2d", "2dpp"]:
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        elif variant in ["1d", "1dpp"]:
            kernel_size = stem_sizes[variant]
            stem = Stem1D(
                dim0=in_chans,
                dim1=stem_dim1,
                dim2=dims[0],
                kernel_size=kernel_size,
                enable_layer_cycle=enable_layer_cycle,
                arbitrary_size=arbitrary_size,
            )

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        Block = Blocks[variant]
        kernel_sizes = block_sizes[variant]
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        kernel_sizes=kernel_sizes[i],
                        layer_offset=layer_wise_rotation_offset(
                            j, depths[i], enable_layer_cycle
                        ),
                        arbitrary_size=arbitrary_size,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.norm(
            x.mean([-2, -1])
        )  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return x


@register_model
def convnext_tiny(pretrained=False, variant="2d", drop_path_rate=None, **kwargs):
    drop_path_rate = drop_path_rate if drop_path_rate is not None else 0.1
    model = ConvNeXt(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=drop_path_rate,
        variant=variant,
        **kwargs,
    )
    if pretrained:
        url = f"pretrained/image_classification/convnext_tiny_{variant}_best_ema.pth"
        checkpoint = torch.load(url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_base(pretrained=False, variant="2d", drop_path_rate=None, **kwargs):
    drop_path_rate = drop_path_rate if drop_path_rate is not None else 0.5
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=drop_path_rate,
        variant=variant,
        **kwargs,
    )
    if pretrained:
        url = f"pretrained/image_classification/convnext_base_{variant}_best_ema.pth"
        checkpoint = torch.load(url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
