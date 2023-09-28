# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

DATA_PATH="$PWD/data/imagenet"

python -m torch.distributed.launch --master_port=$((29500+$RANDOM%1000)) --nproc_per_node=${NGPUS} main.py \
    --model ${MODEL:=convnext_tiny} --variant ${VARIANT:=2d} --input_size ${INPUT_SIZE:=224} \
    --batch_size ${BATCH_SIZE:=32} --update_freq $((4096/${BATCH_SIZE:=32}/${NGPUS})) \
    --data_set IMNET \
    --data_path ${DATA_PATH} \
    --lr ${LEARNING_RATE:=4e-3} \
    --num_workers ${NCPUS:=4} \
    --output_dir ${OUTPUT_DIR:="output/$EXPERIMENT_NAME"} \
    --seed ${SEED:=0} \
    --model_ema 1 \
    ${EXTRA:=""}