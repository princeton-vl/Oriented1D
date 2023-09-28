# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

import os
import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from dwoconv1d_reference_kernel import get_dwoconv1d_kernel

__all__ = ["DepthwiseOrientedConv1d"]


def oriented_dwconv1d_get_module_from_params(ftype):
    module = get_dwoconv1d_kernel(ftype)
    return module.dwoconv1d


def wrap_nil_tensor(x):
    return x if x is not None else torch.zeros(0)


class DepthwiseOrientedConv1d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        *,
        groups=1,
        stride=1,
        padding=1,
        bias=True,
        angle=0,
        nhwc=False,
        resid=False,
        arbitrary_size=False
    ):
        # Init correct shapes
        super().__init__(
            in_channels,
            out_channels,
            (1, kernel_size),
            groups=groups,
            stride=stride,
            padding=(0, padding),
            bias=bias,
        )
        # code only supports odd-kernel size depthwise convolutions
        assert (
            isinstance(kernel_size, int)
            and in_channels == out_channels
            and groups == in_channels
            and kernel_size % 2 == 1
        )
        # code only supports same strides for H and W (convenience)
        assert self.stride[0] == self.stride[1]

        if isinstance(angle, float) or isinstance(angle, int):
            angle = torch.zeros((in_channels, 1)) + angle

        self.theta = torch.nn.Parameter(torch.tensor(angle))
        self.nhwc = nhwc
        self.pad = kernel_size // 2

        # Residual connection
        if resid:
            dw = torch.zeros_like(self.weight)
            dw[:, :, :, pad : (pad + 1)] = 1
            self.resid = dw.cuda()
        else:
            self.resid = None

        # Compile kernels
        self.module_fp32 = oriented_dwconv1d_get_module_from_params("float")

    def forward(self, x):
        return self.module_fp32(
            x,
            self.weight,
            self.theta,
            wrap_nil_tensor(self.bias),
            wrap_nil_tensor(self.resid),
            self.bias is not None,
            self.resid is not None,
            self.nhwc,
            self.kernel_size[1],
            self.stride[0],
            self.pad,
        )
