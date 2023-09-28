# Based on the RepLKNet codebase: https://github.com/DingXiaoH/RepLKNet-pytorch

# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

import os
import argparse
import numpy as np
import torch
from timm.models import create_model
import models.convnext
from timm.utils import AverageMeter
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torch import optim as optim
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Script for visualizing the ERF", add_help=False)
    parser.add_argument("--model", default="convnext_tiny", type=str, help="model")
    parser.add_argument("--variant", default="2d", type=str, help="2d/1d/2dpp/1dpp")
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="path to weights file. None loads from pretrained folder",
    )
    parser.add_argument(
        "--data_path", default="path_to_imagenet", type=str, help="dataset path"
    )
    parser.add_argument(
        "--save_path",
        default="temp.npy",
        type=str,
        help="path to save the ERF matrix (.npy file)",
    )
    parser.add_argument(
        "--num_images", default=1000, type=int, help="num of images to use"
    )
    args = parser.parse_args()
    return args


def get_input_grad(model, samples):
    outputs = model.forward_features(samples)
    out_size = outputs.size()
    central_point = torch.nn.functional.relu(
        outputs[:, :, out_size[2] // 2, out_size[3] // 2]
    ).sum()
    grad = torch.autograd.grad(central_point, samples)
    grad = grad[0]
    grad = torch.nn.functional.relu(grad)
    aggregated = grad.sum((0, 1))
    grad_map = aggregated.cpu().numpy()
    return grad_map


def main(args):
    #   ================================= transform: resize to 1024x1024
    t = [
        transforms.Resize((1024, 1024), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
    transform = transforms.Compose(t)

    model = create_model(args.model, variant=args.variant, arbitrary_size=True)

    print("reading from datapath", args.data_path)
    root = os.path.join(args.data_path, "val")
    dataset = datasets.ImageFolder(root, transform=transform)

    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_val = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_val,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    if args.weights is None:
        args.weights = (
            f"pretrained/image_classification/{args.model}_{args.variant}_best.pth"
        )
    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)

    model.cuda()
    model.eval()  #   fix BN and droppath

    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

    meter = AverageMeter()
    optimizer.zero_grad()

    for _, (samples, _) in tqdm(enumerate(data_loader_val)):
        if meter.count == args.num_images:
            np.save(args.save_path, meter.avg)
            exit()

        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        contribution_scores = get_input_grad(model, samples)

        if np.isnan(np.sum(contribution_scores)):
            print("got NAN, next image")
            continue
        else:
            meter.update(contribution_scores)


if __name__ == "__main__":
    args = parse_args()
    main(args)
