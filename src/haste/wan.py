from dataclasses import dataclass, field
from types import TracebackType

from torch import nn

from haste.config import Haste
from haste.linear import CompressedLinear


@dataclass
class Entry:
    parent: nn.Sequential
    position: int
    source: nn.Linear
    layer: CompressedLinear
    block: int


@dataclass
class Patch:
    entries: list[Entry] = field(default_factory=list)

    def enable(self, enabled: bool = True) -> None:
        for entry in self.entries:
            entry.layer.enabled = enabled

    def collect(self, enabled: bool = True) -> None:
        for entry in self.entries:
            entry.layer.collect = enabled
            entry.layer.counts.zero_()

    def counts(self) -> list[dict]:
        '''Read aggregate device counters after a diagnostic run has completed.'''
        records = []
        for entry in self.entries:
            occupied, windows, channels = entry.layer.counts.tolist()
            records.append(
                dict(
                    block=entry.block,
                    projection='input' if entry.position == 0 else 'output',
                    occupied=occupied,
                    windows=windows,
                    channels=channels,
                )
            )
        return records

    def remove(self) -> None:
        for entry in self.entries:
            if entry.parent[entry.position] is not entry.layer:
                raise RuntimeError('A patched feedforward layer was modified by another caller')
        for entry in reversed(self.entries):
            entry.parent[entry.position] = entry.source
        self.entries.clear()

    def __enter__(self) -> Patch:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.remove()


def install(transformer: nn.Module, config: Haste) -> Patch:
    '''Validate the Wan feedforward layout before changing any module.

    Only transformer.blocks[*].ffn linear projections are eligible. Text,
    image, attention, modulation, output head, and VAE layers are preserved.
    '''
    blocks = getattr(transformer, 'blocks', None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError('Expected a Wan Animate 2 transformer with a nonempty blocks ModuleList')
    candidates = []
    for index, block in enumerate(blocks):
        ffn = getattr(block, 'ffn', None)
        if (
            not isinstance(ffn, nn.Sequential)
            or len(ffn) != 3
            or not isinstance(ffn[0], nn.Linear)
            or not isinstance(ffn[1], nn.GELU)
            or not isinstance(ffn[2], nn.Linear)
        ):
            raise ValueError(f'Unsupported feedforward structure in block {index}')
        group = min(2, index * 3 // len(blocks))
        if config.layers != 'all' and group != ('early', 'middle', 'late').index(config.layers):
            continue
        positions = {'both': (0, 2), 'input': (0,), 'output': (2,)}[config.linear]
        for pos in positions:
            source = ffn[pos]
            if not isinstance(source, nn.Linear):
                raise TypeError('Expected an unmodified linear projection')
            candidates.append(
                Entry(
                    ffn, pos, source, CompressedLinear(source, config, index * 2 + pos // 2), index
                )
            )
    if not candidates:
        raise ValueError('Layer selection is empty')
    patch = Patch(candidates)
    for entry in patch.entries:
        entry.parent[entry.position] = entry.layer
    return patch
