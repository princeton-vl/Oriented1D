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

from dwoconv1d_specialized_kernel import get_dwoconv1d_kernel

__all__ = ["DepthwiseOrientedConv1d"]


class _DepthwiseOrientedConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, theta, bias, module, nhwc=False, resid=None):
        fprop, dgrad, wgrad, nhwc_to_nchw, nchw_to_nhwc = module
        if resid is not None:
            w = w + resid
        x = x.contiguous()
        if nhwc:
            x = nhwc_to_nchw(x)

        ctx.set_materialize_grads(False)
        ctx.module = module
        ctx.nhwc = nhwc
        ctx.save_for_backward(x, w, theta, bias)

        y = fprop(x, w, theta)
        if bias is not None:
            y = y + bias.to(y).view(1, -1, 1, 1)
        if nhwc:
            y = nchw_to_nhwc(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        if dy is None:
            return None, None, None, None, None, None, None

        fprop, dgrad, wgrad, nhwc_to_nchw, nchw_to_nhwc = ctx.module
        x, w, theta, bias = ctx.saved_tensors

        # output
        dy = dy.contiguous()  # avoid non-contiguous input bug
        if ctx.nhwc:
            dy = nhwc_to_nchw(dy)

        dx, dw, dt, db = None, None, None, None

        # input
        if ctx.needs_input_grad[0]:
            dx = dgrad(dy, w, theta)
            if ctx.nhwc:
                dx = nchw_to_nhwc(dx)
        # weight
        if ctx.needs_input_grad[1]:
            dw = wgrad(dy, x, theta)
        # theta
        dt = None
        # bias
        if bias is not None and ctx.needs_input_grad[3]:
            db = dy.sum(axis=[0, 2, 3])

        return dx, dw, dt, db, None, None, None


def oriented_dwconv1d_get_module_from_params(ftype, kernel_size, stride, pad, h, w):
    module = get_dwoconv1d_kernel(ftype, kernel_size, stride, pad, h, w)
    return (
        module.fprop,
        module.dgrad,
        module.wgrad,
        module.nhwc_to_nchw,
        module.nchw_to_nhwc,
    )


def get_tensor_hw(x, nhwc):
    return x.shape[1:3] if nhwc else x.shape[2:4]


# https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-or-equal-to-n-in-python
def next_power_of_2(x):
    return 1 if x == 0 else 1 << (x - 1).bit_length()


def prev_power_of_2(x):
    return 1 if x == 0 else 1 << (x.bit_length() - 1)


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
        arbitrary_size=False,
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
        pad = kernel_size // 2

        # Residual connection
        if resid:
            dw = torch.zeros_like(self.weight)
            dw[:, :, :, pad : (pad + 1)] = 1
            self.resid = dw.cuda()
        else:
            self.resid = None

        self.params = ("float", kernel_size, self.stride[1], pad)
        self.module = None

        if arbitrary_size:
            self.forward = self.forward_safe

    def forward_safe(self, x):
        if self.nhwc:
            n, h, w, c = x.shape
        else:
            n, c, h, w = x.shape

        stride = self.stride[0]
        hpow2, wpow2 = min(next_power_of_2(h) // 2, 128), min(
            next_power_of_2(w) // 2, 128
        )
        pad_h = (hpow2 - (h % hpow2)) % hpow2
        pad_w = (wpow2 - (w % wpow2)) % wpow2

        if self.nhwc:
            x = x.permute((0, 3, 1, 2))

        x = F.pad(x, (0, pad_w, 0, pad_h), "constant", 0).contiguous()

        module = oriented_dwconv1d_get_module_from_params(
            *self.params, *get_tensor_hw(x, False)
        )
        x = _DepthwiseOrientedConv1d.apply(
            x, self.weight, self.theta, self.bias, module, False, self.resid
        )

        x = x[:, :, : x.shape[2] - pad_h // stride, : x.shape[3] - pad_w // stride]
        if self.nhwc:
            x = x.permute((0, 2, 3, 1))

        return x.contiguous()

    def forward(self, x):
        if self.module is None:
            self.module = oriented_dwconv1d_get_module_from_params(
                *self.params, *get_tensor_hw(x, self.nhwc)
            )
        return _DepthwiseOrientedConv1d.apply(
            x, self.weight, self.theta, self.bias, self.module, self.nhwc, self.resid
        )


import time


def benchmark_method(
    name, model, x0, n_iter=100, n_warmup=10, resid=False, nchw_to_nhwc=False
):
    fprop_time, bprop_time, tot_time = 0, 0, 0
    x = x0
    if nchw_to_nhwc:
        x = x.permute((0, 3, 1, 2)).contiguous()
    for i in range(n_iter + n_warmup):
        start_time = time.perf_counter()
        if resid:
            y = model(x) + x
        else:
            y = model(x)
        if nchw_to_nhwc:
            y = y.permute((0, 2, 3, 1))
        torch.cuda.synchronize()
        fprop_end_time = time.perf_counter()

        loss = y.mean() * 1024
        loss.backward()
        for p in model.parameters():
            p.grad = None
        x0.grad = None
        torch.cuda.synchronize()
        bprop_end_time = time.perf_counter()

        if resid:
            y = model(x) + x
        else:
            y = model(x)
        if nchw_to_nhwc:
            y = y.permute((0, 2, 3, 1))
        loss = y.mean() * 1024
        loss.backward()
        for p in model.parameters():
            p.grad = None
        x0.grad = None
        torch.cuda.synchronize()

        tot_end_time = time.perf_counter()

        if i >= n_warmup:
            fprop_time += (fprop_end_time - start_time) * 1000
            bprop_time += (bprop_end_time - fprop_end_time) * 1000
            tot_time += (tot_end_time - bprop_end_time) * 1000
    if n_iter == 0:
        y = torch.zeros(1)
        fprop_time, bprop_time, tot_time = 1, 1, 1
        n_iter = 1
    n_ops = y.shape.numel() * n_iter * 1e-9 * 1000
    print(
        f"{name}\t{fprop_time/n_iter:.3f}ms     \t{bprop_time/n_iter:.3f}ms     \t{tot_time/n_iter:.3f}ms   \tresid={resid}\tshape={y.shape}"
    )
    print(
        f"{name}\t{n_ops/fprop_time:.2f}GFLOPs  \t{n_ops/bprop_time:.2f}GFLOPs  \t{n_ops/tot_time:.3f}GFLOPs\tresid={resid}\tshape={y.shape}"
    )

    x0.grad = None
    x0.requires_grad = True
    if resid:
        y = model(x) + x
    else:
        y = model(x)
    if nchw_to_nhwc:
        y = y.permute((0, 2, 3, 1))
    loss = y.mean() * 1024
    loss.backward()

    if isinstance(model, nn.Conv2d):
        return (
            y.clone(),
            x0.grad.clone(),
            model.weight.grad.clone(),
            model.bias.grad.clone() if model.bias is not None else torch.Tensor(),
        )
    return None, None, None


def error_abs(x, y):
    return (torch.abs(x - y).mean()).item()


if __name__ == "__main__":
    import torch.backends.cudnn as cudnn
    import itertools
    from dwoconv1d_reference import (
        DepthwiseOrientedConv1d as DepthwiseOrientedConv1dReference,
    )

    cudnn.benchmark = True

    torch.random.manual_seed(0)
    if torch.cuda.is_available():
        for stride, D in zip([1, 1, 1, 1, 2], [7, 14, 28, 56, 224]):
            N, C, H, W, S = 32, 512 if D <= 64 else 64, D, D, 7
            half = False
            angle = np.pi / 180 * 0
            bias, resid, nhwc = False, False, False
            n_iter = int(1e11 / (N * C * H * W * S))
            print(f"n_iter: {n_iter}")

            if stride != 1:
                resid = False
            if H % stride != 0:
                continue

            print(f"Creating input: {(N, C, H, W)}")
            x = (
                torch.randn((N, H, W, C)).cuda()
                if nhwc
                else torch.randn((N, C, H, W)).cuda()
            )
            x.requires_grad = True
            print("Creating networks")

            mof1d = DepthwiseOrientedConv1d(
                C,
                C,
                S,
                padding=S // 2,
                angle=angle,
                stride=stride,
                bias=bias,
                resid=resid,
                nhwc=nhwc,
                groups=C,
                arbitrary_size=False,
            ).cuda()
            mof1dr = DepthwiseOrientedConv1dReference(
                C,
                C,
                S,
                padding=S // 2,
                angle=angle,
                stride=stride,
                bias=bias,
                resid=resid,
                nhwc=nhwc,
                groups=C,
            ).cuda()
            mpt1d = nn.Conv2d(
                C, C, (1, S), padding=(0, S // 2), stride=stride, bias=bias, groups=C
            ).cuda()
            mpt2d7 = nn.Conv2d(
                C, C, 7, padding=3, stride=stride, bias=bias, groups=C
            ).cuda()

            if half:
                x = x.half()
                mof1d = mof1d.half()
                mof1dr = mof1dr.half()
                mpt1d = mpt1d.half()
                mpt2d7 = mpt2d7.half()

            mof1dr.load_state_dict({k: v for k, v in mof1d.state_dict().items()})
            print(f"N,C,H,W,STR={N},{C},{H},{W},{stride}")

            y_of1d, dx_of1d, dw_of1d, db_of1d = benchmark_method(
                f"ort_{S} ", mof1d, x, n_iter
            )
            benchmark_method(
                f"1d_{S}  ", mpt1d, x, n_iter, resid=resid, nchw_to_nhwc=nhwc
            )
            benchmark_method("2d_7  ", mpt2d7, x, n_iter, resid=resid)
            y_of1dr, dx_of1dr, dw_of1dr, db_of1dr = benchmark_method(
                f"ref_{S} ", mof1dr, x, 0, n_warmup=0
            )

            print(
                f"y  diff: {error_abs(y_of1d, y_of1dr):.6f}/{torch.abs(y_of1dr).mean().item():.6f}"
            )
            print(
                f"dx diff: {error_abs(dx_of1d, dx_of1dr):.6f}/{torch.abs(dx_of1dr).mean().item():.6f}"
            )
            print(
                f"dw diff: {error_abs(dw_of1d, dw_of1dr):.6f}/{torch.abs(dw_of1dr).mean().item():.6f}"
            )
            print(
                f"db diff: {error_abs(db_of1d, db_of1dr):.6f}/{torch.abs(db_of1dr).mean().item():.6f}"
            )
            print()
            print()
