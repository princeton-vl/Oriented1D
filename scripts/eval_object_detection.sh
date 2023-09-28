# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

cd object_detection

PORT=$((29500+$RANDOM%1000)) \
bash tools/dist_test.sh \
    "configs/convnext/${CONFIG:=cascade_mask_rcnn_convnext_tiny_1d_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k}.py" \
    ${CHECKPOINT:=https://dl.fbaipublicfiles.com/convnext/ade20k/upernet_convnext_tiny_1k_512x512.pth} ${NGPUS:=8} --eval bbox segm