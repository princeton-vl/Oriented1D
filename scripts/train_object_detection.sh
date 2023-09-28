# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

cd object_detection

PORT=$((29500+$RANDOM%1000)) \
bash tools/dist_train.sh \
    "configs/convnext/${CONFIG:=mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k}.py" ${NGPUS:=8} \
    --work-dir ${OUTPUT_DIR:="output-detection/$EXPERIMENT_NAME"} --seed ${SEED:=0} --deterministic \
    --cfg-options model.pretrained=${PRETRAINED:=https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth}