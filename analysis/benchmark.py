# Convolutional Networks with Oriented 1D Kernels (https://arxiv.org/abs/2309.15812)
# Licensed under The MIT License [see LICENSE for details]
# Based on the ConvNeXt code base: https://github.com/facebookresearch/ConvNeXt
# --------------------------------------------------------

import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import itertools
import time

from dwoconv1d_specialized import DepthwiseOrientedConv1d

torch.random.manual_seed(0)
cudnn.benchmark = True


def benchmark_method(name, model, x, n_iter=100, K=1, n_warmup=10, verbose=True):
    fprop_times, bprop_times, tot_times = [], [], []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem1 = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

    for i in range(n_iter + n_warmup):
        start_time = time.perf_counter()
        y = model(x)
        torch.cuda.synchronize()
        fprop_end_time = time.perf_counter()

        loss = y.mean() * 1024
        loss.backward()
        for p in model.parameters():
            p.grad = None
        x.grad = None
        torch.cuda.synchronize()
        bprop_end_time = time.perf_counter()

        y = model(x)
        loss = y.mean() * 1024
        loss.backward()
        for p in model.parameters():
            p.grad = None
        x.grad = None
        torch.cuda.synchronize()
        tot_end_time = time.perf_counter()

        if i >= n_warmup:
            fprop_times.append((fprop_end_time - start_time) * 1000)
            bprop_times.append((bprop_end_time - fprop_end_time) * 1000)
            tot_times.append((tot_end_time - bprop_end_time) * 1000)
    fprop_times = np.array(fprop_times)
    bprop_times = np.array(bprop_times)
    tot_times = np.array(tot_times)
    n_ops = y.shape.numel() * K * n_iter * 1e-9 * 1000
    torch.cuda.synchronize()

    mem2 = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    if verbose:
        print(
            f"{name}\tmean\t{fprop_times.mean():.3f}ms    \t{bprop_times.mean():.3f}ms    \t{tot_times.mean():.3f}ms  \tshape={y.shape}"
        )
        print(
            f"         \tstd\t{fprop_times.std():.3f}ms    \t{bprop_times.std():.3f}ms    \t{tot_times.std():.3f}ms  \tshape={y.shape}"
        )
        print(f"torch.cuda.memory_allocated: {mem2-mem1:.3f}G {mem1:.3f}G {mem2:.3f}G")
    return fprop_times, bprop_times, tot_times


def error_abs(x, y):
    return (torch.abs(x - y).mean()).item()


methods = [
    ("conv1d ", nn.Conv2d, 0),
    # ('rep1d ', DepthWiseConv1dGEMM, 0),
    # ('ortrep', DepthWiseOrientedConv1dGEMM, 1),
    # ('rot   ', DepthwiseRotatedConv1d, 1),
    # ('rotcmp', DepthWiseRotatedCompressedConv1d, 1),
    ("ort1d  ", DepthwiseOrientedConv1d, 1),
]


def table5():
    n_iter = 100
    N, C = 64, 512
    Ss = [7, 31]
    angles = [0, 22.5, 45, 66.75, 90, 112.5, 135, 157.5]
    verbose = False
    for H in [14, 28, 56]:
        W = H
        x = torch.randn(N, C, H, W).cuda()
        for name, method, is1d_rot in methods:
            for S in Ss:
                times = []
                for angle in angles:
                    if is1d_rot: 
                        nn = method(C, C, S, angle=angle, padding=S//2, stride=1, bias=False, groups=C).cuda()
                    else: 
                        nn = method(C, C, (1, S), padding=(0, S//2), stride=1, bias=False, groups=C).cuda()
                    f, b, t = benchmark_method(f'{name}   \tangle={angle:.1f}\t  ', nn,  x, n_iter, K=S, verbose=verbose)
                    times.append(t.mean())
                print(f"{H}x{W} S={S}\t{name}   \t{np.round(times, 1)}")
        print()


table5()
