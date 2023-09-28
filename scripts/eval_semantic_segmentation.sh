# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

cd semantic_segmentation

PORT=$((29500+$RANDOM%1000)) \
bash tools/dist_test.sh \
    "configs/convnext/${CONFIG:=upernet_convnext_tiny_1d_512_160k_ade20k_ms}.py" \
    ${CHECKPOINT:=https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth} ${NGPUS:=8} --eval mIoU --aug-test