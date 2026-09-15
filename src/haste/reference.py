import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Geometry:
    queries: int
    keys: int
    area: int
    reference_area: int

    def __post_init__(self):
        if min(self.queries, self.keys, self.area, self.reference_area) < 1:
            raise ValueError('attention geometry must be positive')
        if self.keys < self.queries:
            raise ValueError('keys must contain the generation sequence')

    def allowed(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        '''Animate 2 permits generation keys and reference frame q_frame minus one.'''
        return (
            (q >= 0)
            & (q < self.queries)
            & (k >= 0)
            & (k < self.keys)
            & (
                (k < self.queries)
                | (q // self.area == (k - self.queries) // self.reference_area + 1)
            )
        )


def top_p(
    scores: torch.Tensor, threshold: torch.Tensor | float, minimum: float = 0
) -> torch.Tensor:
    '''Retain the shortest descending prefix reaching normalized probability mass p.'''
    order = scores.argsort(dim=-1, descending=True, stable=True)
    ranked = scores.gather(-1, order)
    total = ranked.sum(-1, keepdim=True)
    target = torch.as_tensor(threshold, device=scores.device)
    while target.ndim < scores.ndim:
        target = target.unsqueeze(-1)
    before = ranked.cumsum(-1) - ranked
    keep = (before < total * target) & (ranked > 0)
    rank = torch.arange(scores.shape[-1], device=scores.device)
    keep |= (rank < math.floor(scores.shape[-1] * minimum)) & (ranked > 0)
    keep |= (target >= 1) & (ranked > 0)
    return torch.zeros_like(keep).scatter(-1, order, keep)


def pooled(x: torch.Tensor) -> torch.Tensor:
    '''Token-averaged head features in FP32 without materializing an FP32 copy.'''
    return x.mean(-2, dtype=torch.float32)


def distance(
    qm: torch.Tensor, km: torch.Tensor, aq: torch.Tensor, ak: torch.Tensor
) -> torch.Tensor:
    '''HASTE Equation 5 on pooled features: L1 drift from the last refresh anchor.'''
    return (qm - aq).abs().sum(-1) + (km - ak).abs().sum(-1)


def drift(q: torch.Tensor, k: torch.Tensor, aq: torch.Tensor, ak: torch.Tensor) -> torch.Tensor:
    return distance(pooled(q), pooled(k), aq, ak)


def gate(refresh: torch.Tensor, heads: int, bounds: tuple[float, float]) -> torch.Tensor:
    '''HASTE Section 4.2 layer gating: whole-layer reuse or refresh at the fraction bounds.

    Heads are grouped per sample; decisions strictly inside the bounds are unchanged.
    '''
    if refresh.shape[0] % heads:
        raise ValueError('Gated heads must divide the flattened sample and head dimension')
    low, high = bounds
    groups = refresh.view(-1, heads)
    fraction = groups.float().mean(-1, keepdim=True)
    gated = torch.where(fraction < low, False, torch.where(fraction > high, True, groups))
    return gated.reshape(-1)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    '''Independent small-tensor oracle: softmax(QK transpose / sqrt(d)) V.'''
    logits = q.double() @ k.double().transpose(-1, -2) / math.sqrt(q.shape[-1])
    logits = logits.masked_fill(~mask, -torch.inf)
    weights = logits.softmax(-1).nan_to_num()
    return (weights @ v.double()).to(q.dtype)


def clusters(
    x: torch.Tensor, count: int, iterations: int, seed: int, centers: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Euclidean Lloyd iterations; empty clusters retain their previous centers.'''
    count = min(count, x.shape[-2])
    if centers is None:
        rng = torch.Generator(device=x.device).manual_seed(seed)
        indices = torch.randperm(x.shape[-2], generator=rng, device=x.device)[:count]
        centers = x[:, indices].float().clone()
    for step in range(iterations + 1):
        distance = (
            x.float().square().sum(-1, keepdim=True)
            + centers.square().sum(-1).unsqueeze(-2)
            - 2 * x.float() @ centers.transpose(-1, -2)
        )
        labels = distance.argmin(-1)
        if step == iterations:
            break
        sums = torch.zeros_like(centers)
        sums.scatter_add_(1, labels[..., None].expand_as(x), x.float())
        sizes = torch.zeros(centers.shape[:2], device=x.device)
        sizes.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
        centers = torch.where(sizes[..., None] > 0, sums / sizes[..., None].clamp_min(1), centers)
    return labels, centers


def layout(labels: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    order = labels.argsort(dim=-1, stable=True)
    sizes = torch.zeros((labels.shape[0], count), device=labels.device, dtype=torch.int64)
    sizes.scatter_add_(1, labels, torch.ones_like(labels))
    offsets = torch.cat((torch.zeros_like(sizes[:, :1]), sizes.cumsum(-1)), -1)
    return order, offsets


def expand(
    mask: torch.Tensor, qlabels: torch.Tensor, klabels: torch.Tensor, geometry: Geometry
) -> torch.Tensor:
    heads = torch.arange(mask.shape[0], device=mask.device)[:, None, None]
    selected = mask[heads, qlabels[..., None], klabels[:, None, :]]
    qids = torch.arange(geometry.queries, device=mask.device)[:, None]
    kids = torch.arange(geometry.keys, device=mask.device)[None, :]
    return selected & geometry.allowed(qids, kids)


def pairs(
    qlabels: torch.Tensor, klabels: torch.Tensor, qc: int, kc: int, geometry: Geometry
) -> tuple[torch.Tensor, torch.Tensor]:
    '''Count permitted token pairs per cluster without a token attention matrix.'''
    frames = math.ceil(geometry.queries / geometry.area)
    qframe = torch.arange(geometry.queries, device=qlabels.device) // geometry.area
    kid = torch.arange(geometry.keys, device=qlabels.device)
    # Extra bins cover references outside the current query frame range.
    bins = max(
        frames + 1, math.ceil((geometry.keys - geometry.queries) / geometry.reference_area) + 2
    )
    kframe = torch.where(
        kid < geometry.queries, bins - 1, (kid - geometry.queries) // geometry.reference_area + 1
    )
    qhist = torch.zeros((qlabels.shape[0], qc * bins), device=qlabels.device, dtype=torch.float32)
    khist = torch.zeros((klabels.shape[0], kc * bins), device=klabels.device, dtype=torch.float32)
    qhist.scatter_add_(1, qlabels * bins + qframe, torch.ones_like(qlabels, dtype=torch.float32))
    khist.scatter_add_(1, klabels * bins + kframe, torch.ones_like(klabels, dtype=torch.float32))
    qhist = qhist.reshape(-1, qc, bins)
    khist = khist.reshape(-1, kc, bins)
    generation = khist[..., bins - 1]
    permitted = qhist[..., :frames] @ khist[..., :frames].transpose(-1, -2)
    permitted += qhist.sum(-1)[..., None] * generation[:, None, :]
    return permitted, generation


def select(
    scores: torch.Tensor, threshold: torch.Tensor, minimum: float, generation: torch.Tensor
) -> torch.Tensor:
    '''Top-p with a generation-key fallback for structurally empty query rows.'''
    mask = top_p(scores, threshold, minimum)
    valid = generation[:, None, :] > 0
    fallback = scores.masked_fill(~valid, -torch.inf).argmax(-1, keepdim=True)
    missing = ~(mask & valid).any(-1, keepdim=True)
    added = torch.zeros_like(mask).scatter(-1, fallback, missing)
    return mask | added


def xscore(
    q: torch.Tensor, k: torch.Tensor, geometry: Geometry, block: int, stride: int
) -> torch.Tensor:
    '''XAttention inverse-stride antidiagonal estimator, with valid-pair normalization.

    The Animate 2 extension excludes forbidden antidiagonal pairs before averaging.
    Storage is bounded to one query block against the reduced key sequence.
    '''
    heads, nq, dim = q.shape
    nk = k.shape[1]
    result = q.new_zeros((heads, math.ceil(nq / block), math.ceil(nk / block)), dtype=torch.float32)
    kg = torch.arange(math.ceil(nk / stride), device=q.device)
    for start in range(0, nq, block):
        qg = torch.arange(
            start // stride, math.ceil(min(start + block, nq) / stride), device=q.device
        )
        logits = q.new_zeros((heads, len(qg), len(kg)), dtype=torch.float32)
        counts = torch.zeros_like(logits)
        for offset in range(stride):
            qi = qg * stride + stride - 1 - offset
            ki = kg * stride + offset
            valid = geometry.allowed(qi[:, None], ki[None, :])
            score = q[:, qi.clamp_max(nq - 1)].float() @ k[
                :, ki.clamp_max(nk - 1)
            ].float().transpose(-1, -2)
            logits += score * valid
            counts += valid
        logits = (logits / counts.clamp_min(1) / math.sqrt(dim)).masked_fill(
            counts == 0, -torch.inf
        )
        mass = logits.softmax(-1).nan_to_num().sum(-2)
        result[:, start // block].scatter_add_(1, (kg * stride // block).expand(heads, -1), mass)
    return result
