import copy
import json
import multiprocessing as mp
from dataclasses import replace
from inspect import signature
from pathlib import Path

import numpy as np
import torch
from diffusers.models.transformers.transformer_wan_animate_2 import (
    WanAnimate2KVCache,
    WanAnimate2Transformer3DModel,
)

from haste.calibration import artifact, error_value, key
from haste.config import Config
from haste.data import environment, manifest, write_json
from haste.runner import infer, inputs, load
from haste.wan import install


def move(value, device):
    '''Copy replay tensors recursively; CPU grid metadata stays on CPU.'''
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {
            name: move(
                item,
                'cpu'
                if name in ('grid_sizes', 'reference_grid_sizes', 'offset_grid_sizes')
                else device,
            )
            for name, item in value.items()
        }
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return copy.deepcopy(value)


def velocity(output) -> list[torch.Tensor]:
    return output.sample if hasattr(output, 'sample') else output[0]


class Capture:
    '''Capture sampled dense calls and one shared reference cache per segment/branch.'''

    def __init__(self, transformer, folder: Path, steps: set[int]):
        self.folder = folder
        self.steps = steps
        self.indices = {}
        self.segment = -1
        self.current = None
        self.records = []
        self.handles = [
            transformer.register_forward_pre_hook(self.start, with_kwargs=True),
            transformer.register_forward_hook(self.stop, with_kwargs=True),
        ]

    def start(self, module, args, kwargs):
        kwargs = dict(signature(module.forward).bind(*args, **kwargs).arguments)
        self.current = None
        branch = 'uncond' if kwargs.get('is_uncondtion', False) else 'cond'
        if kwargs['kv_cache_mode'] == 'extract':
            self.segment += 1
            self.indices.clear()
            return
        step = self.indices.get(branch, 0)
        self.indices[branch] = step + 1
        # One segment per example bounds offline storage and is recorded explicitly.
        if step not in self.steps or self.segment != 0:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        cache_path = self.folder / f'{branch}_reference.pt'
        if not cache_path.exists():
            cache = kwargs['kv_cache']
            tensors = [
                move(layer.get(), 'cpu') if layer.key is not None else None
                for layer in cache.layer_caches
            ]
            torch.save(tensors, cache_path)
        values = {name: value for name, value in kwargs.items() if name != 'kv_cache'}
        name = f'{branch}_{step}'
        torch.save(move(values, 'cpu'), self.folder / f'{name}_input.pt')
        self.current = dict(branch=branch, step=step, name=name, segment=0)

    def stop(self, module, args, kwargs, output):
        if self.current is not None:
            torch.save(
                move(velocity(output), 'cpu'), self.folder / f'{self.current['name']}_output.pt'
            )
            self.records.append(self.current)
            self.current = None

    def close(self):
        for handle in self.handles:
            handle.remove()


def replay(folder: Path, row: dict, device: torch.device) -> tuple[dict, list[torch.Tensor]]:
    values = torch.load(folder / f'{row['name']}_input.pt', map_location='cpu', weights_only=True)
    values = move(values, device)
    tensors = torch.load(
        folder / f'{row['branch']}_reference.pt', map_location=device, weights_only=True
    )
    cache = WanAnimate2KVCache(len(tensors))
    for layer, pair in zip(cache.layer_caches, tensors, strict=True):
        if pair is not None:
            layer.store(*pair)
    values['kv_cache'] = cache
    dense = torch.load(folder / f'{row['name']}_output.pt', map_location=device, weights_only=True)
    return values, dense


@torch.inference_mode()
def worker(device: int, config: Config, jobs: list[tuple], samples: list, root: Path) -> None:
    if not jobs:
        return
    pipe = load(config, device)
    cuda_device = torch.device('cuda', device)
    folder = root / f'gpu{device}'
    folder.mkdir(parents=True, exist_ok=False)
    captured = {}
    for sample_index in sorted({job[2] for job in jobs}):
        sample = samples[sample_index]
        steps = {step for job in jobs if job[2] == sample_index for step in job[3]}
        capture = Capture(pipe.transformer, folder / sample.name, steps)
        try:
            infer(pipe, inputs(config, sample), sample.seed, device)
        finally:
            capture.close()
        captured[sample_index] = capture.records
    rows = []
    settings = replace(config.haste, mode='sparse')
    with install(pipe.transformer, settings) as patch:
        for block, head, sample_index, steps in jobs:
            sample = samples[sample_index]
            for branch in ('cond', 'uncond'):
                if branch == 'uncond' and block == 9:
                    continue
                calls = [
                    row
                    for row in captured[sample_index]
                    if row['branch'] == branch and row['step'] in steps
                ]
                if not calls:
                    if branch == 'uncond' and config.model.guidance <= 1:
                        continue
                    raise RuntimeError(
                        f'Missing dense calibration calls for {sample.name} {branch}'
                    )
                if len(calls) != len(steps):
                    raise RuntimeError(
                        'Dense trajectory does not cover all sampled calibration steps'
                    )
                errors, sparsities, bands = [], [], []
                for threshold in config.calibration.thresholds:
                    measured, densities, spectral_bands = [], [], []
                    for call in calls:
                        values, dense = replay(folder / sample.name, call, cuda_device)
                        patch.isolate(block, head, threshold)
                        patch.collect()
                        candidate = velocity(pipe.transformer(**values))
                        error, energy = error_value(candidate, dense, config.calibration)
                        stats = patch.counts()
                        counter = next(row for row in stats if row['block'] == block)['heads'][0]
                        measured.append(error)
                        spectral_bands.append(energy)
                        densities.append(counter[4] / counter[5])
                        patch.collect(False)
                        del values, dense, candidate
                    errors.append(float(np.mean(measured)))
                    sparsities.append(1 - float(np.mean(densities)))
                    bands.append(np.mean(spectral_bands, axis=0).tolist())
                rows.append(
                    dict(
                        block=block,
                        head=head,
                        branch=branch,
                        sample=sample.record(),
                        steps=steps,
                        segment=0,
                        errors=errors,
                        sparsity=sparsities,
                        bands=bands,
                    )
                )
                write_json(
                    folder / 'measurements.json', dict(rows=rows, environment=environment(device))
                )
    write_json(folder / 'measurements.json', dict(rows=rows, environment=environment(device)))


def launch(config: Config) -> None:
    if config.run.split != 'dev':
        raise ValueError('Calibration is restricted to the development split')
    if config.calibration.intervals > config.model.steps:
        raise ValueError('Calibration intervals cannot exceed denoising steps')
    if not torch.cuda.is_available() or max(config.run.devices) >= torch.cuda.device_count():
        raise RuntimeError('Configured CUDA devices are unavailable')
    samples = manifest(config.run.manifest, 'dev')
    root = Path(config.run.output) / 'calibration' / key(config)
    root.mkdir(parents=True, exist_ok=False)
    model = WanAnimate2Transformer3DModel.load_config(
        config.model.name, subfolder='transformer', revision=config.model.revision
    )
    if not isinstance(model, dict):
        raise TypeError('Expected a public transformer configuration dictionary')
    shape = dict(layers=model['num_layers'], heads=model['num_heads'])
    rng = np.random.default_rng(config.calibration.seed)
    intervals = np.array_split(np.arange(config.model.steps), config.calibration.intervals)
    jobs = []
    for block in range(shape['layers']):
        group = min(2, block * 3 // shape['layers'])
        if config.haste.layers != 'all' and group != ('early', 'middle', 'late').index(
            config.haste.layers
        ):
            continue
        for head in range(shape['heads']):
            jobs.append(
                (
                    block,
                    head,
                    int(rng.integers(len(samples))),
                    [int(rng.choice(interval)) for interval in intervals],
                )
            )
    devices = config.run.devices
    context = mp.get_context('spawn')
    processes = []
    for index, device in enumerate(devices):
        assigned = jobs[index :: len(devices)]
        if len(devices) == 1:
            worker(device, config, assigned, samples, root)
        else:
            process = context.Process(target=worker, args=(device, config, assigned, samples, root))
            process.start()
            processes.append(process)
    for process in processes:
        process.join()
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError('Calibration worker failed; partial measurements remain available')
    rows, environments = [], []
    for path in sorted(root.glob('gpu*/measurements.json')):
        data = json.loads(path.read_text())
        rows.extend(data['rows'])
        environments.append(data['environment'])
    artifact(
        root / 'table.json',
        config,
        rows,
        dict(
            environments=environments,
            inputs=[sample.record() for sample in samples],
            seeds=dict(calibration=config.calibration.seed, clustering=config.haste.seed),
        ),
    )
