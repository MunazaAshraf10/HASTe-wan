'''Synchronized CUDA timings with separate compilation and warmup samples.'''

import statistics
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import torch


def measure(call: Callable[[], Any], warmup: int, repeats: int) -> dict:
    cold = []
    samples = []
    memory = []
    for index in range(warmup + repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        (cold if index < warmup else samples).append(elapsed)
        if index >= warmup:
            memory.append(
                dict(
                    allocated=torch.cuda.max_memory_allocated(),
                    reserved=torch.cuda.max_memory_reserved(),
                )
            )
    return dict(
        first_call_ms=cold[0] if cold else None,
        first_call_includes_compilation=bool(cold),
        warmup_ms=cold,
        samples_ms=samples,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        peak_bytes={name: max(row[name] for row in memory) for name in ('allocated', 'reserved')},
    )


class Trace:
    '''Record CUDA event intervals for transformer invocations without synchronizing hooks.'''

    def __init__(self, transformer, scheduler=None):
        self.events = []
        self.pending = []
        self.segments = []
        self.segment_start = None
        self.segment_end = None
        self.scheduler = scheduler
        self.step = None if scheduler is None else scheduler.step
        if scheduler is not None:

            @wraps(scheduler.step)
            def advance(*args, **kwargs):
                return self.advance(*args, **kwargs)

            scheduler.step = advance
        self.handles = [
            transformer.register_forward_pre_hook(self.start, with_kwargs=True),
            transformer.register_forward_hook(self.stop, with_kwargs=True),
        ]

    def start(self, module, args, kwargs):
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        mode = kwargs.get('kv_cache_mode', 'unknown')
        if mode == 'extract':
            self.finish_segment()
        elif mode == 'cached' and self.segment_start is None:
            self.segment_start = event
        self.pending.append((event, mode))

    def advance(self, *args, **kwargs):
        if self.step is None:
            raise RuntimeError('No scheduler was attached to this trace')
        output = self.step(*args, **kwargs)
        self.segment_end = torch.cuda.Event(enable_timing=True)
        self.segment_end.record()
        return output

    def finish_segment(self) -> None:
        if self.segment_start is not None and self.segment_end is not None:
            self.segments.append((self.segment_start, self.segment_end))
        self.segment_start = None
        self.segment_end = None

    def stop(self, module, args, kwargs, output):
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        start, mode = self.pending.pop()
        self.events.append((start, end, mode))

    def result(self) -> dict:
        self.finish_segment()
        torch.cuda.synchronize()
        result = {
            mode: sum(start.elapsed_time(end) for start, end, kind in self.events if kind == mode)
            for mode in {entry[2] for entry in self.events}
        }
        result['denoising'] = sum(start.elapsed_time(end) for start, end in self.segments)
        return result

    def close(self):
        if self.scheduler is not None:
            self.scheduler.step = self.step
        for handle in self.handles:
            handle.remove()
