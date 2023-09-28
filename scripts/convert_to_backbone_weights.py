# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------


import torch
import copy
import os
import argparse

import sys
sys.path.append('semantic_segmentation')
from semantic_segmentation.backbone.convnext import ConvNeXt

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--variant')
parser.add_argument('--checkpoint')
parser.add_argument('--output_dir')
args = parser.parse_args()


old_state_dict = torch.load(args.checkpoint)

def convnext_tiny(variant='2d', **kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], variant=variant, **kwargs)

def convnext_base( variant='2d',  **kwargs):
    return ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], variant=variant, **kwargs)

models = {
    'convnext_tiny': convnext_tiny,
    'convnext_base': convnext_base,
}

target_model = models[args.model](variant=args.variant)
target_state_dict = copy.deepcopy(target_model.state_dict())

state_dict = copy.deepcopy(old_state_dict)
new_state_dict = {}
for key in list(state_dict.keys()):
    new_key = key
    if 'dw' in key:
        new_key = new_key.replace('.weight', '.conv.weight')
        new_key = new_key.replace('.bias',   '.conv.bias')
        new_key = new_key.replace('.theta',  '.conv.theta')

    if 'dw' in key and 'weight' in key:
        source, target = state_dict[key], target_state_dict[new_key] 
        if len(source.shape) == 4 and source.shape[2] == 1: 
            new_state_dict[key] = reshape_kernel(source, target.shape[-1]) 
            print(f'key\t{key}\told\t{source.shape}\tnew\t{target.shape}')
            continue
    new_state_dict[new_key] = state_dict.pop(key)

dir, filename = os.path.dirname(args.checkpoint), os.path.basename(args.checkpoint)
torch.save(new_state_dict, f'{args.output_dir}/{filename}')
print('done')