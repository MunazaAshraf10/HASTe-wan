'''Reversible linear layer adaptation with immutable pretrained parameters.'''

import torch
from torch import Tensor, nn

from haste import reference
from haste.config import Haste
from haste.hashing import projections


class CompressedLinear(nn.Module):
    '''Windowed channel compression extending HASTE Equation 9 to linear layers.'''

    planes: Tensor
    counts: Tensor

    def __init__(self, source: nn.Linear, config: Haste, salt: int = 0):
        super().__init__()
        self.source = source
        self.config = config
        self.enabled = True
        self.collect = False
        self.register_buffer(
            'planes', projections(config, salt).to(source.weight.device), persistent=False
        )
        self.register_buffer(
            'counts',
            torch.zeros(3, device=source.weight.device, dtype=torch.int64),
            persistent=False,
        )

    @torch.compiler.disable
    def forward(self, x: Tensor) -> Tensor:
        '''Keep dynamic bucket execution outside compiled transformer graphs.'''
        if not self.enabled:
            return self.source(x)
        if torch.is_grad_enabled():
            raise RuntimeError('HASTE supports inference only; use torch.inference_mode()')
        if self.config.backend == 'reference':
            return reference.linear(
                x,
                self.source.weight,
                self.source.bias,
                self.planes,
                self.counts if self.collect else None,
            )
        if not x.is_cuda:
            raise RuntimeError('The CUDA backend requires an NVIDIA GPU')
        from haste.kernels import linear

        return linear(
            x,
            self.source.weight,
            self.source.bias,
            self.planes,
            self.config.tile,
            self.counts if self.collect else None,
            self.config.projection == 'ternary',
        )
