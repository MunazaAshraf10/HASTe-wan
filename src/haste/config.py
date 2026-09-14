'''Validated experiment settings and stable configuration serialization.'''

import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Haste:
    window: int = 64
    bits: int = 16
    projection: str = 'ternary'
    sparsity: float = 2 / 3
    seed: int = 17
    layers: str = 'all'
    linear: str = 'both'
    backend: str = 'cuda'
    tile: int = 128

    def __post_init__(self):
        if self.window not in (32, 64, 128):
            raise ValueError('window must be 32, 64, or 128')
        if not 1 <= self.bits <= 30:
            raise ValueError('bits must be between 1 and 30')
        if self.projection not in ('gaussian', 'ternary'):
            raise ValueError('projection must be gaussian or ternary')
        if not 0 <= self.sparsity < 1:
            raise ValueError('sparsity must be in [0, 1)')
        if self.layers not in ('all', 'early', 'middle', 'late'):
            raise ValueError('layers must be all, early, middle, or late')
        if self.linear not in ('both', 'input', 'output'):
            raise ValueError('linear must be both, input, or output')
        if self.backend not in ('cuda', 'reference'):
            raise ValueError('backend must be cuda or reference')
        if self.tile < 32 or self.tile % 32:
            raise ValueError('tile must be a positive multiple of 32')


@dataclass(frozen=True)
class Model:
    name: str = 'Wan-AI/Wan2.2-Animate-2-14B-Diffusers'
    revision: str = '7d48412d7b903ff3a89f4f5a960d99e1899605a1'
    height: int = 320
    width: int = 480
    steps: int = 40
    frames: int = 81
    fps: int = 24
    guidance: float = 3.0

    def __post_init__(self):
        if min(self.height, self.width, self.steps, self.frames, self.fps) <= 0:
            raise ValueError('model dimensions and sampling counts must be positive')
        if self.height % 16 or self.width % 16:
            raise ValueError('height and width must be divisible by 16')
        if self.frames < 5 or self.frames % 4 != 1:
            raise ValueError('frames must be at least 5 and congruent to 1 modulo 4')


@dataclass(frozen=True)
class Run:
    manifest: str = 'experiments/manifest.jsonl'
    output: str = 'results'
    devices: tuple[int, ...] = (0,)
    warmup: int = 1
    repeats: int = 3
    split: str = 'dev'

    def __post_init__(self):
        object.__setattr__(self, 'devices', tuple(self.devices))
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError('devices must be a nonempty list of distinct GPU indices')
        if min(self.devices) < 0 or self.warmup < 1 or self.repeats < 1:
            raise ValueError('GPU indices must be nonnegative; warmup and repeats must be positive')
        if self.split not in ('dev', 'eval'):
            raise ValueError('split must be dev or eval')


@dataclass(frozen=True)
class Config:
    haste: Haste = field(default_factory=Haste)
    model: Model = field(default_factory=Model)
    run: Run = field(default_factory=Run)

    @classmethod
    def read(cls, path: str | Path) -> Config:
        with Path(path).open('rb') as stream:
            data = tomllib.load(stream)
        if extra := data.keys() - {'haste', 'model', 'run'}:
            raise ValueError(f'Unknown configuration sections: {sorted(extra)}')
        return cls(
            Haste(**data.get('haste', {})),
            Model(**data.get('model', {})),
            Run(**data.get('run', {})),
        )

    def record(self) -> dict:
        return asdict(self)
