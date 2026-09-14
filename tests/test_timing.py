'''Instrumentation must preserve the scheduler interface and remove its hooks.'''

import inspect

from torch import nn

from haste.timing import Trace


class Scheduler:
    def step(self, sample, timestep, *, generator=None):
        return sample


def test_trace_preserves_scheduler_signature():
    scheduler = Scheduler()
    original = scheduler.step
    layer = nn.Linear(2, 2)
    trace = Trace(layer, scheduler)
    assert inspect.signature(scheduler.step) == inspect.signature(original)
    trace.close()
    assert scheduler.step == original
