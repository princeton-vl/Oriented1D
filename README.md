# Convolutional Networks with Oriented 1D Kernels

This repository contains the source code for our paper:

[Convolutional Networks with Oriented 1D Kernels](https://arxiv.org/abs/2309.15812) <br/>
ICCV 2023<br/>
Alexandre Kirchmeyer, Jia Deng<br/>

We provide results, installation instructions, dataset preparation, and training / evaluation scripts in the following sections:

- [Image classification](#image-classification)
- [Object detection](#object-detection)
- [Semantic Segmentation](#semantic-segmentation)
- [Analysis](#analysis)


If you find this repository helpful, please consider citing:

```
@article{kirchmeyer2023convolutional,
  title={Convolutional Networks with Oriented 1D Kernels},
  author={Kirchmeyer, Alexandre and Deng, Jia},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

# Image Classification

## Pre-trained models

The link for all our pre-trained models can be found at: https://drive.google.com/drive/folders/160d6dPKXIgdmBRPTBv7-nk2jh8cGsaa2?usp=sharing

Download them in the `pretrained` folder. 

## Results

| name | resolution | EMA acc@1 | acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-T (rep.) | 224x224 | 82.0 | 81.8 | 28.6M | 4.5G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-T-1D | 224x224 | 82.2 | 82.2 | 28.5M | 4.4G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-T-1D++ | 224x224 | 82.7 | 82.4 | 29.2M | 4.7G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-T-2D++ | 224x224 | 82.5 | 82.3 | 29.2M | 4.8G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-B (rep.) | 224x224 | 83.7 | 83.6 | 89M | 15.4G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-B-1D | 224x224 | 83.8 | 83.6 | 88M | 15.3G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-B-1D++ | 224x224 | 83.8 | 83.5 | 90M | 15.8G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |
| ConvNeXt-B-2D++ | 224x224 | 84.0 | 83.6 | 91M | 15.9G | [model](https://drive.google.com/drive/u/1/folders/1JUUyz3vFbclvFHyi3AXwue5ejlY6t90q) |

## Installation 

To run our models, please create a new conda environment and install the following dependencies.

```
conda create -n convnext python=3.8
```

Install [Pytorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/stable/index.html) following official instructions. For example:
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Then install the following dependencies: 
```
pip install ninja # for cuda kernel compilation
pip install timm tensorboardX six # for convnext
```

And install our cuda kernels:
```
# Install cuda kernels
cd models/dwoconv1d
pip install -e .
cd ../../models/dwoconv1d_specialized
pip install -e .
cd ../../models/dwoconv1d_reference
pip install -e .
cd ../..
```	

The installation instructions were adapted from: https://github.com/facebookresearch/ConvNeXt/blob/main/INSTALL.md

## Dataset preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and add a reference of the dataset in the `data` folder:
```
mkdir -p data
ln -s path/to/imagenet/dataset data/imagenet
```

## Training

To train a ConvNeXt-1D/2D/1D++/2D++ image classifier, run: 
```
# single-gpu
SPECIALIZED=<1/0> NGPUS=1 VARIANT=<1d/2d/1dpp/2dpp> BATCH_SIZE=<batch_size> MODEL=<convnext_tiny/convnext_base> EXPERIMENT_NAME=<exp name> ./scripts/train_image_classification.sh
# multi-gpu
SPECIALIZED=<1/0> NGPUS=8 VARIANT=<1d/2d/1dpp/2dpp> BATCH_SIZE=<batch_size> MODEL=<convnext_tiny/convnext_base> EXPERIMENT_NAME=<exp name> ./scripts/train_image_classification.sh
```

For example, to train a ConvNeXt-1D-T image classifier on 8 GPUs with a batch size of 64, run:
```
SPECIALIZED=0 NGPUS=8 VARIANT=1d BATCH_SIZE=64 MODEL=convnext_tiny EXPERIMENT_NAME=convnext_tiny_1d ./scripts/train_image_classification.sh
```
## Inference

To evaluate a pre-trained ConvNeXt-1D/2D/1D++/2D++ image classifier, run: 
```
# non-EMA single-gpu
SPECIALIZED=<1/0> NGPUS=1 USE_EMA=0 VARIANT=<1d/2d/1dpp/2dpp> BATCH_SIZE=<batch_size> MODEL=<convnext_tiny/convnext_base> CHECKPOINT=<CHECKPOINT> ./scripts/eval_image_classification.sh
# EMA multi-gpu
SPECIALIZED=<1/0> NGPUS=8 USE_EMA=1 VARIANT=<1d/2d/1dpp/2dpp> BATCH_SIZE=<batch_size> MODEL=<convnext_tiny/convnext_base> CHECKPOINT=<CHECKPOINT> ./scripts/eval_image_classification.sh
```

For example, to evaluate the non-EMA and EMA accuracy of a pre-trained ConvNeXt-1D-T image classifier on 1 GPU, run:
```
# non-EMA
SPECIALIZED=0 NGPUS=1 USE_EMA=0 VARIANT=1d BATCH_SIZE=64 MODEL=convnext_tiny CHECKPOINT=$PWD/pretrained/image_classification/convnext_tiny_1d_best.pth ./scripts/eval_image_classification.sh
# EMA
SPECIALIZED=0 NGPUS=1 USE_EMA=1 VARIANT=1d BATCH_SIZE=64 MODEL=convnext_tiny CHECKPOINT=$PWD/pretrained/image_classification/convnext_tiny_1d_best_ema.pth ./scripts/eval_image_classification.sh
```

# Object detection

## Results

| name | Method | AP_box | AP_mask | #params | FLOPs | model |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-T (rep.) | Cascade Mask R-CNN | 50.2 | 43.6 | 86M | 741G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-T-1D | Cascade Mask R-CNN | 50.3 | 43.6 | 86M | 739G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-T-1D++ | Cascade Mask R-CNN | 51.3 | 44.5 | 86M | 739G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-T-2D++ | Cascade Mask R-CNN | 51.2 | 44.4 | 87M | 741G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-B (rep.) | Cascade Mask R-CNN | 52.4 | 45.2 | 146M | 964G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-B-1D | Cascade Mask R-CNN | 52.8 | 45.7 | 145M | 960G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-B-1D++ | Cascade Mask R-CNN | 52.9 | 46.0 | 147M | 960G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |
| ConvNeXt-B-2D++ | Cascade Mask R-CNN | 52.9 | 45.8 | 148M | 964G | [model](https://drive.google.com/drive/u/1/folders/1QZscbGIxky2otdB9f8tDopWyvN8u9K81) |

## Conversion

In order to use our pre-trained classifiers as backbones for downstream tasks, we first need to convert the weights to backbone weights as our backbones use slightly smaller kernel sizes. 
This is because image size is fixed during ImageNet training but they can be arbitrarily big in semantic segmentation / object detection. Because we train with kernel sizes of up to 1x31, which can represent more than 2x the size of the input, this means that some of these weights will be uninitialized. 
This essentially makes predictions random for larger image sizes. 
As a result, we "crop" kernel sizes for downstream tasks and provide a conversion script to convert from image classifier weights to object detection weights and use those for our object detection and semantic segmentation backbones.

To convert the weights, run:
```
python -m scripts.convert_to_backbone_weights --model convnext_tiny --variant 1d --checkpoint pretrained/image_classification/convnext_tiny_1d_best.pth --output_dir pretrained/backbones
```

## Installation:

To run our models on object detection and semantic segmentation, create the following conda environment.

Note that you can use the same environment for semantic segmentation.

```
conda create -n mmlab python=3.7 -y                     
conda activate mmlab

conda install pytorch=1.7.0 torchvision=0.8.0 torchaudio=0.7.0 cudatoolkit=11.0 -c pytorch -y

pip install cython==0.29.33

pip install mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install mmdet==2.20.0
pip install scipy mmsegmentation==0.11.0 timm==0.9.7 fvcore

# Install kernels
cd models/dwoconv1d
pip install -e .
cd ../../models/dwoconv1d_specialized
pip install -e .
cd ../../models/dwoconv1d_reference
pip install -e .
cd ../..
```

Then clone the repository [Swin Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection/tree/6a979e2164e3fb0de0ca2546545013a4d71b2f7d) (commit `6a979e2`) and copy our files from `object_detection` into the folder.

**Note**: This requires a CUDA runtime version < 12 to run (our codebase which is based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) uses MMCV version <= 1.3.0 which only supports CUDA <= 11)

**Note** that we have only tested object detection on an NVidia RTX 3090. It may not work on older generation GPUs.

**Note**: When running our models, if you run into the CUDA error `invalid argument`, this means that your GPU does not have enough shared memory. It can help to increase `Sb` in `dwoconv1d.py` to `Sb = 8` for example.

## Data preparation

To run training and inference, download the [COCO](https://cocodataset.org/) dataset and add a reference to the dataset in the `object_detection/data` folder:
```
mkdir -p object_detection/data
ln -s path/to/coco/dataset object_detection/data/coco
```

For more details, check: https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md

## Training

To train an object detector with a ConvNeXt-1D/2D/1D++/2D++ backbone, run: 
```
# single-gpu training
NGPUS=1 CONFIG=<CONFIG_FILE> PRETRAINED=<PRETRAIN MODEL> EXPERIMENT_NAME=<RUN NAME> ./scripts/train_object_detection.sh 

# multi-gpu training
NGPUS=8 CONFIG=<CONFIG_FILE> PRETRAINED=<PRETRAIN MODEL> EXPERIMENT_NAME=<RUN NAME> ./scripts/train_object_detection.sh 
```

For example, to train a Cascade Mask R-CNN model with a ConvNeXt-1D-T backbone and 8 gpus, run:
```
NGPUS=8 CONFIG=cascade_mask_rcnn_convnext_tiny_1d_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k PRETRAINED=$PWD/pretrained/backbones/convnext_tiny_1d_best.pth EXPERIMENT_NAME=cascade_mask_rcnn_convnext_tiny_1d ./scripts/train_object_detection.sh 
```

## Inference

To evaluate an object detector with a ConvNeXt-1D/2D/1D++/2D++ backbone, run: 
```
# single-gpu evaluation
NGPUS=1 CONFIG=<CONFIG_FILE> CHECKPOINT=<CHECKPOINT MODEL> ./scripts/eval_semantic_segmentation.sh 
# multi-gpu evaluation
NGPUS=8 CONFIG=<CONFIG_FILE> CHECKPOINT=<CHECKPOINT MODEL> ./scripts/eval_semantic_segmentation.sh 
```

For example, to evaluate a Cascade Mask R-CNN model with a ConvNeXt-1D-T backbone and 8 gpus, run:
```
NGPUS=8 CONFIG=cascade_mask_rcnn_convnext_tiny_1d_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k CHECKPOINT=$PWD/pretrained/object_detection/cascade_mask_rcnn_convnext_tiny_1d_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco_in1k_seed0_epoch_36.pth ./scripts/eval_object_detection.sh
```

# Semantic segmentation

## Results

| name | Method | mIoU (ms) | #params | FLOPs | model |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ConvNeXt-T (rep.) | UPerNet | 46.6 | 60M | 939G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-T-1D | UPerNet | 45.2 | 60M | 927G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-T-1D++ | UPerNet | 47.4 | 60M | 927G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-T-2D++ | UPerNet | 48.1 | 61M | 939G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-B (rep.) | UPerNet | 49.4 | 146M | 964G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-B-1D | UPerNet | 49.4 | 145M | 960G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-B-1D++ | UPerNet | 50.7 | 147M | 960G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |
| ConvNeXt-B-2D++ | UPerNet | 50.2 | 148M | 964G | [model](https://drive.google.com/drive/u/1/folders/1Kru_YvASMneH8tTpB1-fbJz-U01UqY7L) |

## Installation
Follow the conda environment instructions in Object Detection section.

Then clone the repository [Beit](https://github.com/microsoft/unilm/tree/f8f3df80c65eb5e5fc6d6d3c9bd3137621795d1e/beit/semantic_segmentation) (commit `8b57ed1`) and copy our files from `semantic_segmentation` into the folder. 


## Dataset preparation

To prepare dataset,
```
cd <root>/semantic_segmentation
mkdir -p data/ade
cd data/ade
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```

For more details, check: https://github.com/open-mmlab/mmsegmentation/blob/v0.11.0/docs/dataset_prepare.md

## Training

To train a segmentation model with a ConvNeXt-1D/2D/1D++/2D++ backbone, run: 
```
# single-gpu training
NGPUS=1 CONFIG=<CONFIG_FILE> PRETRAINED=<PRETRAIN MODEL> EXPERIMENT_NAME=<RUN NAME> ./scripts/train_semantic_segmentation.sh 

# multi-gpu training
NGPUS=8 CONFIG=<CONFIG_FILE> PRETRAINED=<PRETRAIN MODEL> EXPERIMENT_NAME=<RUN NAME> ./scripts/train_semantic_segmentation.sh 
```

For example, to train a UperNet model with a ConvNeXt-1D-T backbone and 8 gpus, run:
```
NGPUS=8 CONFIG=upernet_convnext_tiny_1d_512_160k_ade20k_ms PRETRAINED=$PWD/pretrained/backbones/convnext_tiny_1d_best.pth EXPERIMENT_NAME=upernet_convnext_tiny_1d ./scripts/train_semantic_segmentation.sh 
```

**Note**: Change `data=dict(samples_per_gpu=8)` in the configuration file to reflect the number of gpus you are using. By default, we used 4 GPUS for our training.


## Inference

To evaluate a segmentation model with a ConvNeXt-1D/2D/1D++/2D++ backbone, run: 
```
# multi-gpu evaluation
NGPUS=4 CONFIG=<CONFIG_FILE> CHECKPOINT=<CHECKPOINT MODEL> ./scripts/eval_semantic_segmentation.sh 
```

For example, to evaluate a UperNet model with a ConvNeXt-1D-T backbone and 4 gpus, run:
```
NGPUS=4 CONFIG=upernet_convnext_tiny_1d_512_160k_ade20k_ms CHECKPOINT=$PWD/pretrained/semantic_segmentation/upernet_convnext_tiny_1d_512_160k_ade20k_ms_seed0_iter_160000.pth ./scripts/eval_semantic_segmentation.sh 
```

# Analysis

## Benchmarks

We provide PyTorch-level benchmarks of our oriented 1D kernels.

### Instructions

To run the benchmarks, run:
```
# Faster specialized implementation
python -m models.dwoconv1d_specialized.dwoconv1d_specialized

# Slower general implementation
python -m models.dwoconv1d.dwoconv1d
```

To reproduce table 5 of our paper, run:

```
python -m analysis.benchmark
```

**Note**: Our oriented 1D kernels are written in CUDA and will be compiled automatically on the first run of any model. They are cached in `$HOME/.cache/torch_extensions` by default. The installation may take a few minutes to complete.

### Results
On 1 NVIDIA RTX3090, you should get the following runtimes on the `analysis.benchmark` task:

| Method | $H,W$ | $K = 7$ | $K = 31$ |
|:---:|:---:|:---:|:---:|
| PyTorch | $14^2$ | 0.4 | 1.1 |
| Ours | $14^2$ | 0.3 | 0.8 |
| PyTorch | $28^2$ | 1.4 |3.8 |
| Ours | $28^2$ | 0.9 | 2.6 |
| PyTorch | $56^2$ | 5.5 | 15.0 |
| Ours | $56^2$ | 3.2 | 9.9 |
## ERF

To compute the ERF on ConvNeXt-B-1D, run the following command: 
```
python -m analysis.visualize_erf  --model convnext_base --variant 1d  --data_path data/imagenet --save_path erf_convnext_base_1d.npy --num_images 1000
```

To visualize the ERF, run the following command: 
```
python -m analysis.analyze_erf --source erf_convnext_base_1d.npy 
```

The code was adapted from [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch). We thank the authors for their work.

# Limitations

## Non-specialized vs specialized kernels

We provide two implementations of oriented 1D kernels: specialized vs non-specialized kernels.

We use the specialized implementation to benchmark our kernels and use the non-specialized implementation for network-level inference.

The non-specialized implementation is a bit slower than the specialized implementation, but can take arbitrary sizes as input (provided there is enough GPU shared memory capacity).

The specialized implementation compiles one kernel for every input size, which is not feasible for object detection or arbitrary images. 

## Object detection

Note also that our implementations do not support inputs which are not divisible by the stride (e.g. 65x31 for stride=2) which is why we use a slower forward pass `forward_safe` for object detection and semantic segmentation. `forward_safe` also handles the alignment issues we encountered during development.

## Arbitrary size

Note that our kernels were optimized for specific input sizes (56x56 mainly). Tweaking kernel-level constants may be needed to achieve optimal performance for other input sizes.

## Strided convolutions

Strided convolutions are currently a bit slower than PyTorch strided convolutions, which explains part of the slowdown encountered by ConvNeXt-1D/1D++ over ConvNeXt-2D/2D++.

# Acknowledgments
This repository is built on top of the [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) codebase, [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch), and [NVIDIA Cutlass](https://github.com/NVIDIA/cutlass). We thank the authors for their work.

# License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
