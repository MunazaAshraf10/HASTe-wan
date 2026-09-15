from functools import partial
from pathlib import Path

import torch
from torch import nn

from haste.backend import cuda
from haste.config import Config
from haste.data import environment, write_json
from haste.hashing import projections
from haste.linear import CompressedLinear
from haste.timing import measure


@torch.inference_mode()
def kernels(config: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('Kernel benchmarks require CUDA')
    if cuda is None:
        raise RuntimeError('Kernel benchmarks require Triton; run uv sync on Linux')

    device = config.run.devices[0]
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    rng = torch.Generator(device='cuda').manual_seed(42)
    records = []
    planes = projections(config.haste).cuda()
    for channels, outputs in ((5120, 13824), (13824, 5120)):
        x = torch.randn(
            (1, config.haste.window, channels), device='cuda', dtype=torch.bfloat16, generator=rng
        )
        layer = nn.Linear(channels, outputs, device='cuda', dtype=torch.bfloat16).eval()
        adapted = CompressedLinear(layer, config.haste)
        part = x[0]
        cap = min(channels, 1 << config.haste.bits)
        ternary = config.haste.projection == 'ternary'
        labels = cuda.hashes(part, planes, ternary)
        order, offsets, count = cuda.groups(labels)
        value = cuda.values(part, order, offsets, count, cap, config.haste.window)
        rows = min(config.haste.tile, outputs)
        filt = cuda.filters(layer.weight, order, offsets, count, cap, 0, rows)
        y = torch.empty((part.shape[0], outputs), device='cuda', dtype=x.dtype)
        calls = {
            'hashing': partial(cuda.hashes, part, planes, ternary),
            'sort_compact': partial(cuda.groups, labels),
            'activation_merge': partial(
                cuda.values, part, order, offsets, count, cap, config.haste.window
            ),
            'weight_merge_tile': partial(
                cuda.filters, layer.weight, order, offsets, count, cap, 0, rows
            ),
            'product_tile': partial(cuda.product, value, filt, count, layer.bias, y, 0),
            'compressed_linear': partial(adapted, x),
            'dense_linear': partial(layer, x),
        }
        timing = {
            name: measure(call, config.run.warmup, config.run.repeats)
            for name, call in calls.items()
        }
        records.append(
            dict(
                channels=channels,
                outputs=outputs,
                tokens=part.shape[0],
                occupied=count.item(),
                tile_rows=rows,
                timing=timing,
            )
        )
    x = torch.randn(
        (1, config.haste.window, 5120), device='cuda', dtype=torch.bfloat16, generator=rng
    )
    first = nn.Linear(5120, 13824, device='cuda', dtype=torch.bfloat16).eval()
    last = nn.Linear(13824, 5120, device='cuda', dtype=torch.bfloat16).eval()
    gelu = nn.GELU(approximate='tanh')
    dense = nn.Sequential(first, gelu, last)
    compressed = nn.Sequential(
        CompressedLinear(first, config.haste), gelu, CompressedLinear(last, config.haste, salt=1)
    )
    whole = {
        'dense': measure(lambda: dense(x), config.run.warmup, config.run.repeats),
        'haste': measure(lambda: compressed(x), config.run.warmup, config.run.repeats),
    }
    write_json(
        Path(config.run.output) / 'kernels.json',
        dict(
            config=config.record(),
            environment=environment(device),
            linear=records,
            feedforward=whole,
        ),
    )
