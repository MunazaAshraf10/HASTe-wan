from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any, cast

import torch
import torch.nn.functional as functional
from diffusers.models.transformers.transformer_wan_animate_2 import (
    WanAnimate2Attention,
    WanAnimate2AttnProcessor,
    rope_apply,
)
from torch import nn

from haste.attention import SparseAttention
from haste.config import Haste
from haste.reference import Geometry


class Processor:
    '''Animate 2 self-attention adaptation; reference extraction stays upstream.

    QK normalization, RoPE, and output projections follow Diffusers 0.40.0.
    Reference visibility uses the upstream reference-frame offset of one.
    '''

    def __init__(self, source: WanAnimate2AttnProcessor, config: Haste, block: int):
        self.source = source
        self.dense = torch.compile(source, fullgraph=False) if config.engine == 'cuda' else source
        self.config = config
        self.block = block
        self.enabled = True
        self.branch = 'cond'
        self.states = {}
        self.references = {}
        self.thresholds = {}
        self.target = None
        self.collect = False
        self.archives = []

    def discard(self, branch: str) -> None:
        state = self.states.pop(branch, None)
        if state is not None:
            state.reset()
            if self.collect:
                self.archives.append((branch, state))
        self.references.pop(branch, None)

    def reset(self) -> None:
        for branch in list(self.states):
            self.discard(branch)
        self.references.clear()

    def state(self, branch: str) -> SparseAttention:
        if branch not in self.states:
            cfg = self.config
            if self.target is not None:
                cfg = replace(cfg, mode='sparse', threshold=self.target[1])
            self.states[branch] = SparseAttention(cfg)
            self.states[branch].collect = self.collect
        return self.states[branch]

    @torch.compiler.disable
    def __call__(
        self,
        attn,
        hidden_states,
        rotary_emb,
        grid_sizes,
        kv_cache,
        kv_cache_mode,
        rope_stride=1,
        reference_rotary_emb=None,
        reference_grid_sizes=None,
        reference_rope_stride=1,
        attention_mask=None,
        origin_latent_frames=None,
        origin_latent_hw=None,
    ):
        args: dict[str, Any] = dict(
            rotary_emb=rotary_emb,
            grid_sizes=grid_sizes,
            kv_cache=kv_cache,
            kv_cache_mode=kv_cache_mode,
            rope_stride=rope_stride,
            reference_rotary_emb=reference_rotary_emb,
            reference_grid_sizes=reference_grid_sizes,
            reference_rope_stride=reference_rope_stride,
            attention_mask=attention_mask,
            origin_latent_frames=origin_latent_frames,
            origin_latent_hw=origin_latent_hw,
        )
        if not self.enabled or kv_cache_mode == 'extract':
            return self.dense(attn, hidden_states, **args)
        if kv_cache_mode != 'cached':
            raise ValueError('Unsupported reference cache mode')
        if reference_grid_sizes is None:
            raise ValueError('Reference geometry is required')
        if grid_sizes.device.type != 'cpu' or reference_grid_sizes.device.type != 'cpu':
            raise ValueError('Wan grid metadata must remain on CPU')
        if origin_latent_frames is None or origin_latent_hw is None:
            raise ValueError('Original latent geometry is required')
        grids, refs = grid_sizes.tolist(), reference_grid_sizes.tolist()
        if any(grid != grids[0] for grid in grids) or any(grid != refs[0] for grid in refs):
            raise ValueError('Batched samples must have matching token geometry')
        frames, height, width = grids[0]
        rf, rh, rw = refs[0]
        nq, nr = frames * height * width, rf * rh * rw
        if (frames, height * width, rf, rh * rw) != (
            origin_latent_frames + 1,
            origin_latent_hw,
            origin_latent_frames,
            origin_latent_hw,
        ):
            raise ValueError('Animate 2 sparse attention requires the complete segment grid')
        if hidden_states.shape[1] != nq:
            raise ValueError('Padded hidden states are unsupported by this Wan checkpoint')
        if getattr(attn, 'fused_projections', False):
            q, k, v = attn.to_qkv(hidden_states).chunk(3, -1)
        else:
            q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        q = attn.norm_q(q).unflatten(2, (attn.heads, -1))
        k = attn.norm_k(k).unflatten(2, (attn.heads, -1))
        v = v.unflatten(2, (attn.heads, -1))
        q = rope_apply(q, grid_sizes, rotary_emb, rope_stride).to(v.dtype)
        k = rope_apply(k, grid_sizes, rotary_emb, rope_stride).to(v.dtype)
        kr, vr = kv_cache.get()
        reference = (id(kr), kr.data_ptr(), kr.shape, kr.device, kr.dtype)
        if self.references.get(self.branch) != reference:
            self.discard(self.branch)
            self.references[self.branch] = reference
        kr = rope_apply(kr, reference_grid_sizes, reference_rotary_emb, reference_rope_stride).to(
            v.dtype
        )
        k = torch.cat((k, kr[:, :nr]), 1)
        v = torch.cat((v, vr[:, :nr]), 1)
        batch, _, heads, dim = q.shape
        q = q.permute(0, 2, 1, 3).reshape(batch * heads, nq, dim)
        k = k.permute(0, 2, 1, 3).reshape(batch * heads, nq + nr, dim)
        v = v.permute(0, 2, 1, 3).reshape(batch * heads, nq + nr, dim)
        geometry = Geometry(nq, nq + nr, height * width, rh * rw)
        state = self.state(self.branch)
        if self.target is not None:
            if batch != 1:
                raise ValueError('Isolated-head calibration requires one sample per invocation')
            head, threshold = self.target
            hs = slice(head, head + 1)
            # Replace only one head contribution; all other heads stay on the upstream path.
            original = self.dense(attn, hidden_states, **args)
            sparse = state(q[hs], k[hs], v[hs], geometry)
            dense = state.apply(q[hs], k[hs], v[hs], geometry, dense=True)
            delta = (sparse - dense).reshape(1, nq, dim)
            return original + functional.linear(
                delta, attn.to_out[0].weight[:, head * dim : (head + 1) * dim]
            )
        thresholds = self.thresholds.get(self.branch)
        if thresholds is not None:
            thresholds = thresholds.to(device=q.device).repeat(batch)
        out = state(q, k, v, geometry, thresholds, heads)
        out = out.reshape(batch, heads, nq, dim).permute(0, 2, 1, 3).flatten(2)
        return attn.to_out[1](attn.to_out[0](out))


@dataclass
class Entry:
    module: WanAnimate2Attention
    source: WanAnimate2AttnProcessor
    processor: Processor
    block: int


@dataclass
class Patch:
    entries: list[Entry] = field(default_factory=list)
    handles: list = field(default_factory=list)

    def context(self, module, args, kwargs) -> None:
        branch = 'uncond' if kwargs.get('is_uncondtion', False) else 'cond'
        for entry in self.entries:
            entry.processor.branch = branch
        if kwargs.get('kv_cache_mode') == 'extract':
            self.reset()

    def reset(self) -> None:
        for entry in self.entries:
            entry.processor.reset()

    def enable(self, enabled: bool = True) -> None:
        self.reset()
        for entry in self.entries:
            entry.processor.enabled = enabled

    def isolate(self, block: int, head: int, threshold: float) -> None:
        self.reset()
        for entry in self.entries:
            entry.processor.enabled = entry.block == block
            entry.processor.target = (head, threshold) if entry.block == block else None

    def collect(self, enabled: bool = True) -> None:
        for entry in self.entries:
            processor = entry.processor
            processor.collect = enabled
            processor.archives.clear()
            for state in processor.states.values():
                state.collect = enabled
                state.stats = None
                state.events.clear()

    def counts(self) -> list[dict]:
        records = []
        for entry in self.entries:
            processor = entry.processor
            for branch, state in [*processor.archives, *processor.states.items()]:
                records.append(dict(block=entry.block, branch=branch, **state.record()))
        return records

    def remove(self) -> None:
        for entry in self.entries:
            if entry.module.processor is not entry.processor:
                raise RuntimeError('An installed attention processor was changed by another caller')
        for entry in self.entries:
            cast(Any, entry.module).set_processor(entry.source)
            entry.processor.reset()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
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
    '''Validate all target processors before mutation; restore exact processor objects.'''
    blocks = getattr(transformer, 'blocks', None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError('Expected a Wan Animate 2 transformer with a nonempty blocks ModuleList')
    entries = []
    for index, block in enumerate(blocks):
        attn = getattr(block, 'self_attn', None)
        if (
            not isinstance(attn, WanAnimate2Attention)
            or type(attn.processor) is not WanAnimate2AttnProcessor
        ):
            raise ValueError(f'Unsupported self-attention structure in block {index}')
        if attn.is_cross_attention or attn.to_out[1].p != 0:
            raise ValueError('Expected inference self-attention without dropout')
        group = min(2, index * 3 // len(blocks))
        if config.layers != 'all' and group != ('early', 'middle', 'late').index(config.layers):
            continue
        entries.append(Entry(attn, attn.processor, Processor(attn.processor, config, index), index))
    if not entries:
        raise ValueError('Layer selection is empty')
    patch = Patch(entries)
    for entry in entries:
        entry.module.set_processor(entry.processor)
    patch.handles.append(transformer.register_forward_pre_hook(patch.context, with_kwargs=True))
    return patch
