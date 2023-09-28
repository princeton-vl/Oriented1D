# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

import torch
import torch.utils.cpp_extension
import os
import random


def compile_dwoconv1d_kernel(name, S, Sb, stride, pad, ftype, verbose=False):
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
    # sources = ['half.h', 'kernels.h', 'nhwc_utils.h', 'kernels.cu']
    sources = [f"{path}/kernels/{fn}" for fn in sources]
    kernel_header = r"""
torch::Tensor fprop(torch::Tensor x, torch::Tensor w, torch::Tensor theta);
torch::Tensor dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta);
torch::Tensor wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta);

torch::Tensor nhwc_to_nchw(torch::Tensor x);
torch::Tensor nchw_to_nhwc(torch::Tensor x);
""" % dict(
        type=ftype
    )

    kernel_impl = r"""
torch::Tensor fprop(torch::Tensor x,  torch::Tensor w, torch::Tensor theta) { return template_fprop<%(type)s, %(S)s, %(Sb)s, %(stride)s, %(pad)s>(x, w, theta); }
torch::Tensor dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta) { return template_dgrad<%(type)s, %(S)s, %(Sb)s, %(stride)s, %(pad)s>(dy, w, theta); }
torch::Tensor wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta) { return template_wgrad<%(type)s, %(S)s, %(Sb)s, %(stride)s, %(pad)s>(dy, x, theta); }
torch::Tensor nhwc_to_nchw(torch::Tensor x) { return template_nhwc_to_nchw<%(type)s>(x); }
torch::Tensor nchw_to_nhwc(torch::Tensor x) { return template_nchw_to_nhwc<%(type)s>(x); }
""" % dict(
        S=S, Sb=Sb, stride=stride, pad=pad, type=ftype
    )

    module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=[kernel_header],
        cuda_sources=[open(fn, "r").read() for fn in sources] + [kernel_impl],
        functions=[
            "fprop",
            "dgrad",
            "wgrad",
            "nhwc_to_nchw",
            "nchw_to_nhwc",
        ],
        with_cuda=True,
        extra_cuda_cflags=["-std=c++17"],
        build_directory=build_dir,
        verbose=True,
    )
    return module


modules = {}


def tensor2int(x):
    return int(x.item()) if isinstance(x, torch.Tensor) else int(x)


def get_dwoconv1d_kernel(ftype, kernel_size, stride, pad, Sb=None, verbose=False):
    kernel_size, stride, pad = (
        tensor2int(kernel_size),
        tensor2int(stride),
        tensor2int(pad),
    )
    name = f"general_dwoconv1d_{ftype}_K{kernel_size}_Sb{Sb}_S{stride}_P{pad}"
    if Sb is None:
        Sb = 1
    global modules
    if name not in modules:
        modules[name] = compile_dwoconv1d_kernel(
            name, kernel_size, Sb, stride, pad, ftype, verbose=verbose
        )
    return modules[name]
