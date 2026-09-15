import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Haste:
    backend: str = 'xattention'
    engine: str = 'cuda'
    mode: str = 'tmr'
    threshold: float = 0.9
    drift: float | None = None
    gate: tuple[float, float] | None = None
    block: int = 128
    stride: int = 8
    queries: int = 300
    keys: int = 1000
    iterations: int = 50
    updates: int = 2
    minimum: float = 0.1
    seed: int = 17
    layers: str = 'all'
    table: str = ''

    def __post_init__(self):
        if self.backend not in ('xattention', 'svg2'):
            raise ValueError('backend must be xattention or svg2')
        if self.engine not in ('cuda', 'reference'):
            raise ValueError('engine must be cuda or reference')
        if self.mode not in ('sparse', 'tmr', 'ebc', 'full'):
            raise ValueError('mode must be sparse, tmr, ebc, or full')
        if not 0 < self.threshold <= 1 or not 0 <= self.minimum <= 1:
            raise ValueError('threshold must be in (0, 1]; minimum must be in [0, 1]')
        if self.drift is not None and (not math.isfinite(self.drift) or self.drift < 0):
            raise ValueError('drift must be nonnegative')
        if self.gate is not None:
            object.__setattr__(self, 'gate', tuple(self.gate))
            if len(self.gate) != 2 or not 0 <= self.gate[0] <= self.gate[1] <= 1:
                raise ValueError('gate must be a [lower, upper] refresh fraction pair in [0, 1]')
        if self.block not in (32, 64, 128) or self.stride not in (4, 8, 16):
            raise ValueError('block must be 32, 64, or 128; stride must be 4, 8, or 16')
        if min(self.queries, self.keys, self.iterations, self.updates) < 1:
            raise ValueError('cluster counts and iterations must be positive')
        if self.seed < 0:
            raise ValueError('clustering seed must be nonnegative')
        if self.layers not in ('all', 'early', 'middle', 'late'):
            raise ValueError('layers must be all, early, middle, or late')

    @property
    def delta(self) -> float:
        return (
            self.drift if self.drift is not None else 30.0 if self.backend == 'xattention' else 8.0
        )

    @property
    def reuse(self) -> bool:
        return self.mode in ('tmr', 'full')


@dataclass(frozen=True)
class Calibration:
    thresholds: tuple[float, ...] = (0.85, 0.9, 0.95)
    intervals: int = 4
    seed: int = 23
    weights: tuple[float, ...] = (1.0, 0.5, 0.01, 0.01)
    cutoff: float = 0.25
    epsilon: float = 1e-12
    limit: float = 600.0

    def __post_init__(self):
        object.__setattr__(self, 'thresholds', tuple(self.thresholds))
        object.__setattr__(self, 'weights', tuple(self.weights))
        if not self.thresholds or any(not 0 < t <= 1 for t in self.thresholds):
            raise ValueError('calibration thresholds must be in (0, 1]')
        if len(set(self.thresholds)) != len(self.thresholds) or 0.9 not in self.thresholds:
            raise ValueError('thresholds must be distinct and include the 0.9 budget baseline')
        if (
            len(self.weights) != 4
            or any(not math.isfinite(w) for w in self.weights)
            or min(self.weights) < 0
            or sum(self.weights) == 0
        ):
            raise ValueError('four nonnegative spectral weights with positive total are required')
        if self.intervals < 1 or not 0 < self.cutoff < 0.5:
            raise ValueError('intervals must be positive and cutoff must be in (0, 0.5)')
        if (
            self.epsilon <= 0
            or self.limit <= 0
            or self.seed < 0
            or not math.isfinite(self.epsilon)
            or not math.isfinite(self.limit)
        ):
            raise ValueError('epsilon and solver limit must be positive')


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
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError('guidance must be finite and nonnegative')
        if len(self.revision) != 40 or any(c not in '0123456789abcdef' for c in self.revision):
            raise ValueError('model revision must be an immutable commit SHA')
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
    diagnostics: bool = True

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
    calibration: Calibration = field(default_factory=Calibration)

    @classmethod
    def read(cls, path: str | Path) -> Config:
        with Path(path).open('rb') as stream:
            data = tomllib.load(stream)
        if extra := data.keys() - {'haste', 'model', 'run', 'calibration'}:
            raise ValueError(f'Unknown configuration sections: {sorted(extra)}')
        obsolete = {'window', 'bits', 'projection', 'sparsity', 'linear', 'tile'}
        if obsolete & data.get('haste', {}).keys():
            raise ValueError(
                'Channel compression settings are obsolete; use sparse attention settings'
            )
        return cls(
            Haste(**data.get('haste', {})),
            Model(**data.get('model', {})),
            Run(**data.get('run', {})),
            Calibration(**data.get('calibration', {})),
        )

    def record(self) -> dict:
        return asdict(self)
