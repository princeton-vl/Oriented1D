# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

import torch
import torch.utils.cpp_extension
import os
import random


def compile_dwoconv1d_kernel(name, H, W, S, stride, pad, ftype, verbose=False):
    build_dir = torch.utils.cpp_extension._get_build_directory(name, verbose=verbose)
    if verbose:
        print(build_dir)
    try:
        module = torch.utils.cpp_extension._import_module_from_library(
            name, build_dir, True
        )
        return module
    except:
        pass

    # order matters
    path = os.path.dirname(os.path.abspath(__file__))
    sources = ["kernels.h", "nhwc_utils.h", "kernels.cu"]
    sources = [f"{path}/kernels/{fn}" for fn in sources]
    kernel_header = r"""
torch::Tensor fprop(torch::Tensor x, torch::Tensor w, torch::Tensor theta);
torch::Tensor dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta);
torch::Tensor wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta);

torch::Tensor nhwc_to_nchw(torch::Tensor x);
torch::Tensor nchw_to_nhwc(torch::Tensor x);
"""

    kernel_impl = r"""
torch::Tensor fprop(torch::Tensor x,  torch::Tensor w, torch::Tensor theta) { return template_fprop_fp32<%(H)s, %(W)s, %(S)s, %(stride)s, %(pad)s>(x, w, theta); }
torch::Tensor dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta) { return template_dgrad_fp32<%(H)s, %(W)s, %(S)s, %(stride)s, %(pad)s>(dy, w, theta); }
torch::Tensor wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta) { return template_wgrad_fp32<%(H)s, %(W)s, %(S)s, %(stride)s, %(pad)s>(dy, x, theta); }
torch::Tensor nhwc_to_nchw(torch::Tensor x) { return template_nhwc_to_nchw<%(ftype)s>(x); }
torch::Tensor nchw_to_nhwc(torch::Tensor x) { return template_nchw_to_nhwc<%(ftype)s>(x); }

""" % dict(
        H=H, W=W, S=S, stride=stride, pad=pad, ftype=ftype
    )

    module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=[kernel_header],
        cuda_sources=[open(fn, "r").read() for fn in sources] + [kernel_impl],
        functions=["fprop", "dgrad", "wgrad", "nhwc_to_nchw", "nchw_to_nhwc"],
        with_cuda=True,
        extra_cuda_cflags=["-std=c++17"],
        build_directory=build_dir,
        verbose=True,
    )
    return module


modules = {}


def tensor2int(x):
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def get_dwoconv1d_kernel(ftype, kernel_size, stride, pad, H, W, verbose=False):
    H, W, kernel_size, stride, pad = (
        tensor2int(H),
        tensor2int(W),
        tensor2int(kernel_size),
        tensor2int(stride),
        tensor2int(pad),
    )
    name = f"specialized_dwoconv1d_{ftype}_K{kernel_size}_S{stride}_P{pad}_H{H}_W{W}"
    global modules
    if name not in modules:
        modules[name] = compile_dwoconv1d_kernel(
            name, H, W, kernel_size, stride, pad, ftype, verbose=verbose
        )
        print(f"modules: {len(modules)} H{H} W{W} K{kernel_size} S{stride}")
    return modules[name]
