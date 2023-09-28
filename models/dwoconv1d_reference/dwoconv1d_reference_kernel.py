# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

import torch
import torch.utils.cpp_extension
import os
import random


def compile_dwoconv1d_kernel(name, ftype, verbose=False):
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
    sources = ["kernels.cpp"]
    sources = [f"{path}/kernels/{fn}" for fn in sources]
    kernel_header = r"""
torch::Tensor fprop(torch::Tensor x, torch::Tensor w, torch::Tensor theta, int S, int stride, int pad);
torch::Tensor dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta, int S, int stride, int pad);
torch::Tensor wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta, int S, int stride, int pad);

torch::Tensor nhwc_to_nchw(torch::Tensor x);
torch::Tensor nchw_to_nhwc(torch::Tensor x);
torch::Tensor dwoconv1d(torch::Tensor x, torch::Tensor w, torch::Tensor theta, torch::Tensor bias, torch::Tensor resid, bool with_bias, bool with_resid, bool nhwc, int S, int stride, int pad);
""" % dict(
        type=ftype
    )

    kernel_impl = r"""
torch::Tensor fprop(torch::Tensor x,  torch::Tensor w, torch::Tensor theta, int S, int stride, int pad) { return template_fprop<%(type)s>(x, w, theta, S, stride, pad); }
torch::Tensor dgrad(torch::Tensor dy, torch::Tensor w, torch::Tensor theta, int S, int stride, int pad) { return template_dgrad<%(type)s>(dy, w, theta, S, stride, pad); }
torch::Tensor wgrad(torch::Tensor dy, torch::Tensor x, torch::Tensor theta, int S, int stride, int pad) { return template_wgrad<%(type)s>(dy, x, theta, S, stride, pad); }
torch::Tensor nhwc_to_nchw(torch::Tensor x) { return template_nhwc_to_nchw<%(type)s>(x); }
torch::Tensor nchw_to_nhwc(torch::Tensor x) { return template_nchw_to_nhwc<%(type)s>(x); }
torch::Tensor dwoconv1d(torch::Tensor x, torch::Tensor w, torch::Tensor theta, torch::Tensor bias, torch::Tensor resid, bool with_bias, bool with_resid, bool nhwc, int S, int stride, int pad) { return template_dwoconv1d<%(type)s>(x, w, theta, bias, resid, with_bias, with_resid, nhwc, S, stride, pad); }
""" % dict(
        type=ftype
    )

    module = torch.utils.cpp_extension.load_inline(
        name=name,
        cpp_sources=[kernel_header]
        + [open(fn, "r").read() for fn in sources]
        + [kernel_impl],
        functions=[
            "fprop",
            "dgrad",
            "wgrad",
            "nhwc_to_nchw",
            "nchw_to_nhwc",
            "dwoconv1d",
        ],
        with_cuda=True,
        extra_cuda_cflags=["-std=c++17"],
        build_directory=build_dir,
        verbose=True,
    )
    return module


modules = {}


def get_dwoconv1d_kernel(ftype, verbose=False):
    name = f"reference_dwoconv1d_{ftype}"
    global modules
    if name not in modules:
        modules[name] = compile_dwoconv1d_kernel(name, ftype, verbose=verbose)
    return modules[name]
