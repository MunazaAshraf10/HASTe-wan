import gc
import hashlib
import itertools
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import ClassifierFreeGuidance, WanAnimate2ModularPipeline
from diffusers.utils import export_to_video, load_image, load_video

from haste.config import Config
from haste.data import Sample, environment, manifest, write_json
from haste.metrics import compare
from haste.timing import Trace, measure
from haste.wan import install


def load(config: Config, device: int):
    '''Load the public base preset and compile upstream attention blocks.'''

    if not torch.cuda.is_available():
        raise RuntimeError('Wan generation requires an NVIDIA CUDA device')
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('highest')
    pipe = WanAnimate2ModularPipeline.from_pretrained(
        config.model.name, revision=config.model.revision
    )
    pipe.load_components(dtype=torch.bfloat16, revision=config.model.revision)
    pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=config.model.guidance))
    pipe.to(torch.device('cuda', device))
    pipe.transformer.eval()
    pipe.transformer.compile_repeated_blocks(fullgraph=False)
    return pipe


def inputs(config: Config, sample: Sample) -> dict:

    video, fps = load_video(sample.video, return_fps=True)
    return dict(
        image=load_image(sample.image),
        driving_video=video,
        driving_video_fps=fps,
        prompt=sample.prompt,
        height=config.model.height,
        width=config.model.width,
        segment_frame_length=config.model.frames,
        num_inference_steps=config.model.steps,
        fps=config.model.fps,
        output='videos',
    )


@torch.inference_mode()
def infer(pipe, args: dict, seed: int, device: int) -> np.ndarray:
    generator = torch.Generator(device=torch.device('cuda', device)).manual_seed(seed)
    videos = pipe(**args, generator=generator)
    frames = videos[0]
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f'Unexpected pipeline output shape: {frames.shape}')
    if frames.dtype != np.uint8:
        frames = frames.astype(np.float32)
    return frames


def save_frames(path: Path, frames: np.ndarray, fps: int) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path.with_suffix('.npy'), frames, allow_pickle=False)
    export_to_video(list(frames), str(path.with_suffix('.mp4')), fps=fps)


def identity(config: Config) -> str:
    encoded = json.dumps(config.record(), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_mode(
    pipe,
    config: Config,
    sample: Sample,
    args: dict,
    device: int,
    folder: Path,
    mode: str,
    paired: bool,
) -> tuple[np.ndarray, dict]:
    frames = None

    def call():
        nonlocal frames
        frames = infer(pipe, args, sample.seed, device)

    timing = measure(call, config.run.warmup if paired else 0, config.run.repeats if paired else 1)
    memory = timing.pop('peak_bytes')
    if frames is None:
        raise RuntimeError('No frames produced')
    save_frames(folder / mode, frames, config.model.fps)
    transformer = {}
    if paired:
        trace = Trace(pipe.transformer, pipe.scheduler)
        try:
            infer(pipe, args, sample.seed, device)
            transformer = trace.result()
        finally:
            trace.close()
    return frames, dict(timing=timing, peak_bytes=memory, transformer_ms=transformer)


def destination(config: Config, sample: Sample, paired: bool, baseline: bool) -> Path:
    mode = 'benchmark' if paired else 'baseline' if baseline else 'generate'
    return Path(config.run.output) / mode / identity(config) / sample.name


def pair(pipe, config: Config, sample: Sample, device: int, paired: bool, baseline: bool) -> None:
    folder = destination(config, sample, paired, baseline)
    folder.mkdir(parents=True, exist_ok=False)
    record: dict[str, Any] = dict(
        config=config.record(),
        sample=sample.record(),
        environment=environment(device),
        status='running',
        modes={},
    )
    write_json(folder / 'record.json', record)
    try:
        args = inputs(config, sample)
        reference_frames = None
        if paired or baseline:
            reference_frames, info = run_mode(
                pipe, config, sample, args, device, folder, 'baseline', paired
            )
            record['modes']['baseline'] = info
        if paired or not baseline:
            with install(pipe.transformer, config.haste) as patch:
                candidate, info = run_mode(
                    pipe, config, sample, args, device, folder, 'haste', paired
                )
                record['modes']['haste'] = info
                if paired:
                    patch.collect()
                    infer(pipe, args, sample.seed, device)
                    record['buckets'] = patch.counts()
                    patch.collect(False)
            if reference_frames is not None:
                # Release cached CUDA allocations before loading the metric network.
                torch.cuda.empty_cache()
                record['metrics'] = compare(
                    reference_frames, candidate, torch.device('cuda', device)
                )
                record['speedup'] = (
                    record['modes']['baseline']['timing']['median_ms'] / info['timing']['median_ms']
                )
        record['status'] = 'complete'
    except Exception as error:
        record['status'] = 'failed'
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write_json(folder / 'record.json', record)
        gc.collect()
        torch.cuda.empty_cache()


def variants(config: Config) -> list[Config]:
    '''Development grid plus layer group ablations at the default method settings.'''
    if config.run.split != 'dev':
        raise ValueError('Sweeps are restricted to dev; freeze a configuration before evaluation')
    grid = []
    for window, bits, projection, linear in itertools.product(
        (32, 64, 128), (8, 12, 16, 24), ('gaussian', 'ternary'), ('input', 'output', 'both')
    ):
        grid.append(
            replace(
                config,
                haste=replace(
                    config.haste, window=window, bits=bits, projection=projection, linear=linear
                ),
            )
        )
    for group in ('early', 'middle', 'late'):
        grid.append(replace(config, haste=replace(config.haste, layers=group)))
    return grid


def worker(device: int, jobs: list[tuple[Config, Sample]], paired: bool, baseline: bool) -> None:
    if not jobs:
        return
    start = time.perf_counter()
    pipe = load(jobs[0][0], device)
    load_seconds = time.perf_counter() - start
    for config, sample in jobs:
        print(
            f'GPU {device}: {sample.name}, configuration {identity(config)}, '
            f'load {load_seconds:.1f}s',
            flush=True,
        )
        pair(pipe, config, sample, device, paired, baseline)


def launch(
    config: Config, paired: bool = False, baseline: bool = False, sweep: bool = False
) -> None:
    samples = manifest(config.run.manifest, config.run.split)
    configs = variants(config) if sweep else [config]
    jobs = list(itertools.product(configs, samples))
    for settings, sample in jobs:
        path = destination(settings, sample, paired, baseline)
        if path.exists():
            raise FileExistsError(f'Results already exist at {path}; choose a new output directory')
    devices = config.run.devices
    if not torch.cuda.is_available() or max(devices) >= torch.cuda.device_count():
        raise RuntimeError('Configured CUDA devices are unavailable')
    if len(devices) == 1:
        worker(devices[0], jobs, paired, baseline)
        return
    context = mp.get_context('spawn')
    processes = [
        context.Process(target=worker, args=(device, jobs[index :: len(devices)], paired, baseline))
        for index, device in enumerate(devices)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    if failures := [process.exitcode for process in processes if process.exitcode != 0]:
        raise RuntimeError(f'Experiment workers failed with exit codes {failures}')
