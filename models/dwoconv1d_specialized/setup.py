# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

from distutils.core import setup

setup(
    name="dwoconv1d_specialized",
    version="1.0.0",
    py_modules=["dwoconv1d_specialized", "dwoconv1d_specialized_kernel"],
)
