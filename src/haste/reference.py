'''Readable reference implementation of the HASTE linear extension.'''

import torch
from torch import Tensor

from haste.hashing import codes


def merge(x: Tensor, weight: Tensor, labels: Tensor, bias: Tensor | None = None) -> Tensor:
    '''Apply Equation 9: average input channels and sum corresponding weights.

    Occupied buckets are compacted; unused hash codes allocate no storage.
    This oracle prioritizes clarity over runtime and supports CPU validation.
    '''
    buckets, groups, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    size = counts.numel()
    values = x.new_zeros((x.shape[0], size), dtype=torch.float32)
    filters = weight.new_zeros((weight.shape[0], size), dtype=torch.float32)
    values.index_add_(1, groups, x.float())
    filters.index_add_(1, groups, weight.float())
    result = (values / counts) @ filters.T
    if bias is not None:
        result += bias.float()
    return result.to(x.dtype)


def linear(
    x: Tensor, weight: Tensor, bias: Tensor | None, planes: Tensor, stats: Tensor | None = None
) -> Tensor:
    '''Compress independently within each sample and contiguous token window.'''
    if x.ndim != 3 or x.shape[-1] != weight.shape[-1]:
        raise ValueError('Expected input [batch, tokens, channels] matching the linear weight')
    output = x.new_empty((*x.shape[:2], weight.shape[0]))
    window = planes.shape[0]
    for batch in range(x.shape[0]):
        for start in range(0, x.shape[1], window):
            part = x[batch, start : start + window]
            labels = codes(part, planes)
            output[batch, start : start + window] = merge(part, weight, labels, bias)
            if stats is not None:
                stats[0].add_(labels.unique().numel())
                stats[1].add_(1)
                stats[2].add_(part.shape[-1])
    return output
