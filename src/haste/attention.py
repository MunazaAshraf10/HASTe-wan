import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from haste import reference
from haste.backend import cuda
from haste.config import Haste
from haste.reference import Geometry


@dataclass
class State:
    shape: tuple
    qc: int
    kc: int
    bins: int
    thresholds: torch.Tensor
    aq: torch.Tensor
    ak: torch.Tensor
    refresh: torch.Tensor
    mask: torch.Tensor
    scores: torch.Tensor
    pairs: torch.Tensor
    generation: torch.Tensor
    qlabels: torch.Tensor
    klabels: torch.Tensor
    qorder: torch.Tensor
    korder: torch.Tensor
    qoffset: torch.Tensor
    koffset: torch.Tensor
    qhist: torch.Tensor
    khist: torch.Tensor
    qcenters: torch.Tensor | None = None
    kcenters: torch.Tensor | None = None


class SparseAttention:
    '''HASTE Algorithm 1 state for one layer and guidance branch.

    Tensor leading dimensions flatten independent samples and heads. Masks,
    permutations, and anchors are updated together only for refreshed heads.
    '''

    def __init__(self, config: Haste):
        self.config = config
        self.state: State | None = None
        self.events = []
        self.collect = False
        self.stats = None

    @contextmanager
    def phase(self, name: str):
        if self.collect and self.config.engine == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self.events.append((name, start, end))
        else:
            yield

    def reset(self) -> None:
        self.state = None

    def prepare(self, q: torch.Tensor, k: torch.Tensor, geometry: Geometry) -> bool:
        shape = (q.shape, k.shape, q.device, q.dtype, geometry, self.config)
        if self.state is not None and self.state.shape == shape:
            return False
        h, nq, d = q.shape
        nk = k.shape[1]
        cfg = self.config
        qc = math.ceil(nq / cfg.block) if cfg.backend == 'xattention' else min(cfg.queries, nq)
        kc = math.ceil(nk / cfg.block) if cfg.backend == 'xattention' else min(cfg.keys, nk)
        frames = math.ceil(nq / geometry.area)
        bins = max(frames + 1, math.ceil((nk - nq) / geometry.reference_area) + 2)
        device = q.device
        self.state = State(
            shape=shape,
            qc=qc,
            kc=kc,
            bins=bins,
            thresholds=torch.full((h,), -1.0, device=device),
            aq=torch.zeros((h, d), device=device),
            ak=torch.zeros((h, d), device=device),
            refresh=torch.ones(h, device=device, dtype=torch.bool),
            mask=torch.zeros((h, qc, kc), device=device, dtype=torch.bool),
            scores=torch.zeros((h, qc, kc), device=device),
            pairs=torch.zeros((h, qc, kc), device=device),
            generation=torch.zeros((h, kc), device=device),
            qlabels=torch.zeros((h, nq), device=device, dtype=torch.int64),
            klabels=torch.zeros((h, nk), device=device, dtype=torch.int64),
            qorder=torch.zeros((h, nq), device=device, dtype=torch.int64),
            korder=torch.zeros((h, nk), device=device, dtype=torch.int64),
            qoffset=torch.zeros((h, qc + 1), device=device, dtype=torch.int64),
            koffset=torch.zeros((h, kc + 1), device=device, dtype=torch.int64),
            qhist=torch.zeros((h, qc, bins), device=device),
            khist=torch.zeros((h, kc, bins), device=device),
        )
        s = self.state
        if cfg.backend == 'xattention':
            s.qlabels[:] = torch.arange(nq, device=device) // cfg.block
            s.klabels[:] = torch.arange(nk, device=device) // cfg.block
            s.qorder, s.qoffset = reference.layout(s.qlabels, qc)
            s.korder, s.koffset = reference.layout(s.klabels, kc)
            s.pairs, s.generation = reference.pairs(s.qlabels, s.klabels, qc, kc, geometry)
        else:
            rng = torch.Generator(device=device).manual_seed(cfg.seed)
            s.qcenters = q[:, torch.randperm(nq, device=device, generator=rng)[:qc]].float().clone()
            s.kcenters = k[:, torch.randperm(nk, device=device, generator=rng)[:kc]].float().clone()
        return True

    def refresh(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        first: bool,
        threshold: torch.Tensor,
        heads: int,
    ) -> None:
        '''Algorithm 1 refresh decision; anchors move only for refreshed heads.'''
        s = self.state
        if s is None:
            raise RuntimeError('Attention state is not initialized')
        qm, km = reference.pooled(q), reference.pooled(k)
        if first or not self.config.reuse:
            s.refresh = torch.ones_like(s.refresh)
        else:
            s.refresh = reference.distance(qm, km, s.aq, s.ak) > self.config.delta
            if self.config.gate is not None:
                s.refresh = reference.gate(s.refresh, heads, self.config.gate)
        s.refresh |= threshold != s.thresholds
        s.thresholds.copy_(threshold)
        s.aq = torch.where(s.refresh[:, None], qm, s.aq)
        s.ak = torch.where(s.refresh[:, None], km, s.ak)

    def predict(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        geometry: Geometry,
        threshold: torch.Tensor,
        first: bool,
    ) -> None:
        s = self.state
        if s is None:
            raise RuntimeError('Attention state is not initialized')
        cfg = self.config
        if cfg.engine == 'reference':
            for head in range(q.shape[0]):
                if not s.refresh[head]:
                    continue
                hs = slice(head, head + 1)
                if cfg.backend == 'xattention':
                    scores = reference.xscore(q[hs], k[hs], geometry, cfg.block, cfg.stride)
                else:
                    for name, x, count in (('q', q, s.qc), ('k', k, s.kc)):
                        labels, centers = reference.clusters(
                            x[hs],
                            count,
                            cfg.iterations if first else cfg.updates,
                            cfg.seed,
                            getattr(s, name + 'centers')[hs],
                        )
                        getattr(s, name + 'labels')[hs] = labels
                        getattr(s, name + 'centers')[hs] = centers
                        order, offsets = reference.layout(labels, count)
                        getattr(s, name + 'order')[hs] = order
                        getattr(s, name + 'offset')[hs] = offsets
                    permitted, gen = reference.pairs(
                        s.qlabels[hs], s.klabels[hs], s.qc, s.kc, geometry
                    )
                    s.pairs[hs], s.generation[hs] = permitted, gen
                    qsize = (s.qoffset[hs, 1:] - s.qoffset[hs, :-1]).clamp_min(1)
                    weight = permitted / qsize[..., None]
                    if s.qcenters is None or s.kcenters is None:
                        raise RuntimeError('Missing semantic cluster centers')
                    logits = (
                        s.qcenters[hs] @ s.kcenters[hs].transpose(-1, -2) / math.sqrt(q.shape[-1])
                    )
                    logits += weight.clamp_min(1e-30).log()
                    scores = logits.masked_fill(weight == 0, -torch.inf).softmax(-1).nan_to_num()
                s.scores[hs] = scores
                s.mask[hs] = reference.select(
                    scores,
                    threshold[hs],
                    cfg.minimum if cfg.backend == 'svg2' else 0,
                    s.generation[hs],
                )
            return
        self.predict_cuda(q, k, geometry, threshold, first)

    def arrange(self, name: str, n: int, count: int, heads: int) -> None:
        s = self.state
        if s is None or cuda is None:
            raise RuntimeError('CUDA layout state is unavailable')
        labels = getattr(s, name + 'labels')
        counts = torch.empty((heads, count), device=labels.device, dtype=torch.int32)
        offsets = getattr(s, name + 'offset')
        cuda.sizes[(heads, count)](labels, counts, s.refresh, n, count, 512)
        cuda.offsets[(heads,)](counts, offsets, s.refresh, count, 1 << (count - 1).bit_length())
        cuda.scatter[(heads, count)](
            labels, getattr(s, name + 'order'), offsets, s.refresh, n, count, 512
        )

    def predict_cuda(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        geometry: Geometry,
        threshold: torch.Tensor,
        first: bool,
    ) -> None:
        s = self.state
        if s is None or cuda is None:
            raise RuntimeError('CUDA attention state is unavailable')
        cfg = self.config
        h, nq, d = q.shape
        nk = k.shape[1]
        qc, kc, bins = s.qc, s.kc, s.bins
        c = 1 << (d - 1).bit_length()
        if cfg.backend == 'svg2':
            with self.phase('clustering'):
                for name, x, count in (('q', q, qc), ('k', k, kc)):
                    n = x.shape[1]
                    centers = getattr(s, name + 'centers')
                    labels = getattr(s, name + 'labels')
                    for step in range((cfg.iterations if first else cfg.updates) + 1):
                        cuda.assign[(h, math.ceil(n / 32))](
                            x, centers, labels, s.refresh, n, d, count, 32, c
                        )
                        if step < (cfg.iterations if first else cfg.updates):
                            # Clone preserves inactive heads without device-to-host decisions.
                            self.arrange(name, n, count, h)
                            updated = centers.clone()
                            cuda.centroid[(h, count)](
                                x,
                                getattr(s, name + 'order'),
                                getattr(s, name + 'offset'),
                                centers,
                                updated,
                                s.refresh,
                                n,
                                d,
                                count,
                                128,
                                c,
                            )
                            centers = updated
                    setattr(s, name + 'centers', centers)
            with self.phase('permutation'):
                for name, n, count in (('q', nq, qc), ('k', nk, kc)):
                    self.arrange(name, n, count, h)
            with self.phase('scoring'):
                for name, n, count in (('q', nq, qc), ('k', nk, kc)):
                    cuda.histogram[(h, count, bins)](
                        getattr(s, name + 'labels'),
                        getattr(s, name + 'hist'),
                        s.refresh,
                        n,
                        count,
                        bins,
                        nq,
                        geometry.area,
                        geometry.reference_area,
                        name == 'q',
                        512,
                    )
                logits = torch.empty_like(s.scores)
                cuda.semantic[(h, qc, math.ceil(kc / 32))](
                    s.qcenters,
                    s.kcenters,
                    s.qhist,
                    s.khist,
                    logits,
                    s.pairs,
                    s.generation,
                    s.refresh,
                    qc,
                    kc,
                    d,
                    bins,
                    math.ceil(nq / geometry.area),
                    c,
                    32,
                )
                cuda.normalize[(h, qc)](
                    logits, s.scores, s.refresh, qc, kc, 1 << (kc - 1).bit_length()
                )
        else:
            with self.phase('scoring'):
                cols = math.ceil(nk / cfg.stride)
                # Query chunks cap the reduced score buffer independently of video length.
                for start in range(0, qc, 4):
                    blocks = min(4, qc - start)
                    rows = min(
                        blocks * cfg.block // cfg.stride,
                        math.ceil(nq / cfg.stride) - start * cfg.block // cfg.stride,
                    )
                    logits = torch.empty((h, rows, cols), device=q.device)
                    probabilities = torch.empty_like(logits)
                    cuda.xestimate[(h, rows, math.ceil(cols / 32))](
                        q,
                        k,
                        logits,
                        s.refresh,
                        nq,
                        nk,
                        d,
                        geometry.area,
                        geometry.reference_area,
                        cfg.block,
                        cfg.stride,
                        start * cfg.block // cfg.stride,
                        rows,
                        c,
                        32,
                    )
                    cuda.normalize[(h, rows)](
                        logits, probabilities, s.refresh, rows, cols, 1 << (cols - 1).bit_length()
                    )
                    cuda.xreduce[(h, blocks, kc)](
                        probabilities,
                        s.scores,
                        s.refresh,
                        nq,
                        nk,
                        cfg.block,
                        cfg.stride,
                        start,
                        rows,
                        cfg.block // cfg.stride,
                    )
        with self.phase('selection'):
            cuda.choose[(h, qc)](
                s.scores,
                s.mask,
                s.generation,
                threshold,
                s.refresh,
                qc,
                kc,
                cfg.minimum if cfg.backend == 'svg2' else 0,
                1 << (kc - 1).bit_length(),
            )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        geometry: Geometry,
        threshold: torch.Tensor | None = None,
        heads: int | None = None,
    ) -> torch.Tensor:
        if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
            raise ValueError('QKV must have shape (samples times heads, tokens, channels)')
        if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
            raise ValueError('QKV head and channel counts must match')
        if q.shape[1] != geometry.queries or k.shape[1] != geometry.keys:
            raise ValueError('QKV lengths must match the attention geometry')
        if self.config.engine == 'cuda' and (q.device.type != 'cuda' or q.dtype != torch.bfloat16):
            raise ValueError('CUDA attention requires BF16 CUDA tensors')
        if k.device != q.device or v.device != q.device or k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError('QKV must share dtype and device')
        if self.config.engine == 'reference' and max(q.shape[1], k.shape[1]) > 4096:
            raise ValueError('The mathematical reference is limited to 4096 tokens')
        if self.config.engine == 'cuda' and q.shape[2] not in (16, 32, 64, 128, 256):
            raise ValueError('CUDA head dimensions must be 16, 32, 64, 128, or 256')
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        first = self.prepare(q, k, geometry)
        s = self.state
        if s is None:
            raise RuntimeError('Attention state is not initialized')
        if threshold is None:
            threshold = torch.full((q.shape[0],), self.config.threshold, device=q.device)
        if threshold.shape != (q.shape[0],) or threshold.device != q.device:
            raise ValueError('Thresholds require one value per head on the QKV device')
        # Without a head count, the flattened leading dimension is one sample.
        with self.phase('drift'):
            self.refresh(q, k, first, threshold, q.shape[0] if heads is None else heads)
        self.predict(q, k, geometry, threshold, first)
        s.mask = torch.where(threshold[:, None, None] >= 1, s.pairs > 0, s.mask & (s.pairs > 0))
        with self.phase('attention'):
            out = self.apply(q, k, v, geometry)
        if self.collect:
            stats = torch.stack(
                (
                    s.refresh.float(),
                    torch.ones_like(s.refresh, dtype=torch.float32),
                    s.mask.sum((1, 2)).float(),
                    (s.pairs > 0).sum((1, 2)).float(),
                    (s.mask * s.pairs).sum((1, 2)),
                    s.pairs.sum((1, 2)),
                ),
                -1,
            )
            self.stats = stats.double() if self.stats is None else self.stats + stats.double()
        return out

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        geometry: Geometry,
        dense: bool = False,
    ) -> torch.Tensor:
        s = self.state
        if s is None:
            raise RuntimeError('Attention state is not initialized')
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        selected = s.pairs > 0 if dense else s.mask
        if self.config.engine == 'reference':
            mask = reference.expand(selected, s.qlabels, s.klabels, geometry)
            out = reference.attention(q, k, v, mask)
        else:
            if cuda is None:
                raise RuntimeError('Triton is unavailable')
            out = torch.empty_like(q)
            cuda.attend[(q.shape[0], math.ceil(q.shape[1] / 32))](
                q,
                k,
                v,
                s.qorder,
                s.korder,
                s.qlabels,
                s.qoffset,
                s.koffset,
                selected,
                out,
                q.shape[1],
                k.shape[1],
                q.shape[2],
                s.qc,
                s.kc,
                geometry.area,
                geometry.reference_area,
                1 << (q.shape[2] - 1).bit_length(),
                32,
                32,
            )
        return out

    def record(self) -> dict:
        timing = {}
        if self.events:
            torch.cuda.synchronize()
            for name, start, end in self.events:
                timing[name] = timing.get(name, 0.0) + start.elapsed_time(end)
        return dict(
            columns=[
                'refreshes',
                'calls',
                'retained_blocks',
                'permitted_blocks',
                'retained_pairs',
                'permitted_pairs',
            ],
            heads=[] if self.stats is None else self.stats.tolist(),
            phases_ms=timing,
        )
