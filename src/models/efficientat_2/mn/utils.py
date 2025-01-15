"""
Taken from https://github.com/fschmid56/EfficientAT/blob/main/models/mn/utils.py 
"""

import math
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch import Tensor


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def cnn_out_size(in_size, padding, dilation, kernel, stride):
    s = in_size + 2 * padding - dilation * (kernel - 1) - 1
    return math.floor(s / stride + 1)


def collapse_dim(
    x: Tensor,
    dim: int,
    mode: str = "pool",
    pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
    combine_dim: int = None,
):
    """
    Collapses dimension of multi-dimensional tensor by pooling or combining dimensions
    :param x: input Tensor
    :param dim: dimension to collapse
    :param mode: 'pool' or 'combine'
    :param pool_fn: function to be applied in case of pooling
    :param combine_dim: dimension to join 'dim' to
    :return: collapsed tensor
    """
    if mode == "pool":
        return pool_fn(x, dim)
    elif mode == "combine":
        s = list(x.size())
        s[combine_dim] *= dim
        s[dim] //= dim
        return x.view(s)


class CollapseDim(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str = "pool",
        pool_fn: Callable[[Tensor, int], Tensor] = torch.mean,
        combine_dim: int = None,
    ):
        super(CollapseDim, self).__init__()
        self.dim = dim
        self.mode = mode
        self.pool_fn = pool_fn
        self.combine_dim = combine_dim

    def forward(self, x):
        return collapse_dim(
            x, dim=self.dim, mode=self.mode, pool_fn=self.pool_fn, combine_dim=self.combine_dim
        )


# taken from https://github.com/fschmid56/EfficientAT/blob/main/helpers/utils.py


def NAME_TO_WIDTH(name):
    mn_map = {
        "mn01": 0.1,
        "mn02": 0.2,
        "mn04": 0.4,
        "mn05": 0.5,
        "mn06": 0.6,
        "mn08": 0.8,
        "mn10": 1.0,
        "mn12": 1.2,
        "mn14": 1.4,
        "mn16": 1.6,
        "mn20": 2.0,
        "mn30": 3.0,
        "mn40": 4.0,
    }

    dymn_map = {"dymn04": 0.4, "dymn10": 1.0, "dymn20": 2.0}

    try:
        if name.startswith("dymn"):
            w = dymn_map[name[:6]]
        else:
            w = mn_map[name[:4]]
    except:
        w = 1.0

    return w


# import csv

# # Load label
# with open("metadata/class_labels_indices.csv", "r") as f:
#     reader = csv.reader(f, delimiter=",")
#     lines = list(reader)

# labels = []
# ids = []  # Each label has a unique id such as "/m/068hy"
# for i1 in range(1, len(lines)):
#     id = lines[i1][1]
#     label = lines[i1][2]
#     ids.append(id)
#     labels.append(label)

# classes_num = len(labels)


import numpy as np


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)

    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)

    return wrapper


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    def wrapper(epoch):
        if epoch <= start:
            return 1.0
        elif epoch - start < rampdown_length:
            return last_value + (1.0 - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value

    return wrapper


import torch


def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    return rn_indices, lam


from torch.distributions.beta import Beta


def mixstyle(x, p=0.4, alpha=0.4, eps=1e-6, mix_labels=False):
    if np.random.rand() > p:
        return x
    batch_size = x.size(0)

    # changed from dim=[2,3] to dim=[1,3] - from channel-wise statistics to frequency-wise statistics
    f_mu = x.mean(dim=[1, 3], keepdim=True)
    f_var = x.var(dim=[1, 3], keepdim=True)

    f_sig = (f_var + eps).sqrt()  # compute instance standard deviation
    f_mu, f_sig = f_mu.detach(), f_sig.detach()  # block gradients
    x_normed = (x - f_mu) / f_sig  # normalize input
    lmda = (
        Beta(alpha, alpha).sample((batch_size, 1, 1, 1)).to(x.device)
    )  # sample instance-wise convex weights
    perm = torch.randperm(batch_size).to(x.device)  # generate shuffling indices
    f_mu_perm, f_sig_perm = f_mu[perm], f_sig[perm]  # shuffling
    mu_mix = f_mu * lmda + f_mu_perm * (1 - lmda)  # generate mixed mean
    sig_mix = f_sig * lmda + f_sig_perm * (1 - lmda)  # generate mixed standard deviation
    x = x_normed * sig_mix + mu_mix  # denormalize input using the mixed statistics
    if mix_labels:
        return x, perm, lmda
    return x
