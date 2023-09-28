# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

cd semantic_segmentation

PORT=$((29500+$RANDOM%1000)) \
bash tools/dist_train.sh \
    "configs/convnext/${CONFIG:=upernet_convnext_tiny_1d_512_160k_ade20k_ms}.py" ${NGPUS:=8} \
    --work-dir ${OUTPUT_DIR:="output-segmentation/$EXPERIMENT_NAME"} --seed ${SEED:=0} --deterministic \
    --options model.pretrained=${PRETRAINED:=https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224.pth}