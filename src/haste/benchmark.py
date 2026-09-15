import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from haste.attention import SparseAttention
from haste.config import Config
from haste.data import environment, write_json
from haste.reference import Geometry
from haste.timing import measure


@torch.inference_mode()
def kernels(config: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('Attention benchmarks require CUDA')
    device = config.run.devices[0]
    torch.cuda.set_device(device)
    rng = torch.Generator(device='cuda').manual_seed(42)
    records = []
    for frames, area in ((3, 64), (5, 256), (22, 600)):
        nq, nk = frames * area, (2 * frames - 1) * area
        q = torch.randn((40, nq, 128), device='cuda', dtype=torch.bfloat16, generator=rng)
        k = torch.randn((40, nk, 128), device='cuda', dtype=torch.bfloat16, generator=rng)
        v = torch.randn(k.shape, device='cuda', dtype=torch.bfloat16, generator=rng)
        geometry = Geometry(nq, nk, area, area)
        setup = time.perf_counter()

        def visibility(batch, head, query, key, geometry=geometry):
            return geometry.allowed(query, key)

        mask = torch.compile(create_block_mask)(visibility, None, None, nq, nk, device='cuda')
        dense = torch.compile(flex_attention)
        torch.cuda.synchronize()
        setup_ms = (time.perf_counter() - setup) * 1000
        baseline = measure(
            partial(dense, q[None], k[None], v[None], block_mask=mask),
            config.run.warmup,
            config.run.repeats,
        )
        records.append(
            dict(
                backend='upstream_flex',
                mode='dense',
                queries=nq,
                keys=nk,
                setup_ms=setup_ms,
                timing=baseline,
            )
        )
        for backend in ('xattention', 'svg2'):
            for mode in ('sparse', 'tmr'):
                attention = SparseAttention(
                    replace(config.haste, backend=backend, mode=mode, engine='cuda')
                )
                cold = measure(partial(attention, q, k, v, geometry), 0, 2)
                timing = measure(
                    partial(attention, q, k, v, geometry), config.run.warmup, config.run.repeats
                )
                # Diagnostics run separately from the repeated latency measurements.
                attention.collect = True
                attention.reset()
                attention(q, k, v, geometry)
                attention(q, k, v, geometry)
                records.append(
                    dict(
                        backend=backend,
                        mode=mode,
                        queries=nq,
                        keys=nk,
                        compilation_calls_ms=cold['samples_ms'],
                        timing=timing,
                        attention=attention.record(),
                    )
                )
    write_json(
        Path(config.run.output) / 'kernels.json',
        dict(config=config.record(), environment=environment(device), attention=records),
    )
