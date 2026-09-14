'''Random projection hashing from HASTE Equations 3 through 5.'''

import torch
from torch import Tensor

from haste.config import Haste


def projections(config: Haste, salt: int = 0) -> Tensor:
    '''Fix hyperplanes per layer; ternary sampling follows HASTE Section 3.4.'''
    rng = torch.Generator(device='cpu').manual_seed(config.seed + salt)
    shape = (config.window, config.bits)
    if config.projection == 'gaussian':
        return torch.randn(shape, generator=rng)
    sample = torch.rand(shape, generator=rng)
    tail = (1 - config.sparsity) / 2
    return (sample < tail).float() - (sample >= 1 - tail).float()


def codes(x: Tensor, planes: Tensor) -> Tensor:
    '''Hash centered channels across tokens; merge values remain uncentered.

    Algorithm 1 centers across channels. Token windows replace CNN patches
    in this extension, following the direction proposed in Section 5.2.
    '''
    centered = x.float() - x.float().mean(dim=1, keepdim=True)
    scores = centered.T @ planes[: x.shape[0]].float()
    powers = 1 << torch.arange(planes.shape[1], device=x.device, dtype=torch.int64)
    return ((scores > 0).long() * powers).sum(dim=1)
