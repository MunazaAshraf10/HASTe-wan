from dataclasses import replace
from typing import Any

import pytest
import torch

from haste.attention import SparseAttention
from haste.config import Haste
from haste.reference import (
    Geometry,
    attention,
    clusters,
    drift,
    expand,
    layout,
    pairs,
    top_p,
    xscore,
)


def test_top_p_minimal_prefix_and_stable_ties():
    scores = torch.tensor([[0.4, 0.3, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]])
    assert top_p(scores, 0.5).tolist() == [[True, True, False, False], [True, False, False, False]]
    assert top_p(scores, 1).tolist() == [[True] * 4, [True, True, False, False]]
    assert not top_p(torch.zeros_like(scores), 0.9).any()


def test_rectangular_geometry_and_cluster_pair_counts():
    geometry = Geometry(9, 17, 3, 2)
    q = torch.tensor([[0, 1, 0, 1, 0, 1, 2, 2, 2]])
    k = torch.tensor([[0, 1, 0, 1, 0, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1]])
    counts, generation = pairs(q, k, 4, 4, geometry)
    oracle = torch.zeros_like(counts)
    allowed = geometry.allowed(torch.arange(9)[:, None], torch.arange(17)[None, :])
    for qi in range(9):
        for ki in range(17):
            oracle[0, q[0, qi], k[0, ki]] += allowed[qi, ki]
    torch.testing.assert_close(counts, oracle)
    assert generation.sum() == 9
    assert not allowed[:3, 9:].any()
    assert allowed[3:6, 9:11].all()
    assert not allowed[:, 13:].any()


@pytest.mark.parametrize('backend', ['xattention', 'svg2'])
def test_full_attention_matches_independent_oracle(backend):
    rng = torch.Generator().manual_seed(2)
    q = torch.randn(2, 35, 16, generator=rng)
    k = torch.randn(2, 58, 16, generator=rng)
    v = torch.randn(2, 58, 16, generator=rng)
    geometry = Geometry(35, 58, 7, 5)
    cfg = Haste(
        backend=backend, engine='reference', threshold=1, block=32, queries=5, keys=7, iterations=2
    )
    actual = SparseAttention(cfg)(q, k, v, geometry)
    allowed = geometry.allowed(torch.arange(35)[:, None], torch.arange(58)[None, :])
    torch.testing.assert_close(actual, attention(q, k, v, allowed), rtol=0, atol=0)


@pytest.mark.parametrize('backend', ['xattention', 'svg2'])
def test_anchor_does_not_follow_reused_steps(backend):
    q = torch.zeros(2, 8, 4)
    k = torch.zeros(2, 12, 4)
    v = torch.randn_like(k)
    cfg = Haste(
        backend=backend, engine='reference', queries=2, keys=3, iterations=1, drift=1.0, updates=1
    )
    state = SparseAttention(cfg)
    geometry = Geometry(8, 12, 4, 4)
    state(q, k, v, geometry)
    assert state.state is not None
    anchor = state.state.aq.clone()
    order = state.state.qorder.clone()
    q[0] += 0.2
    state(q, k, v, geometry)
    assert state.state.refresh.tolist() == [False, False]
    torch.testing.assert_close(state.state.aq, anchor)
    torch.testing.assert_close(state.state.qorder, order)
    q[0] += 0.2
    state(q, k, v, geometry)
    assert state.state.refresh.tolist() == [True, False]
    torch.testing.assert_close(state.state.aq[0], torch.full((4,), 0.4))
    state.reset()
    assert state.state is None


def test_boundary_and_no_reuse_mode():
    q = torch.ones(1, 8, 4)
    k = torch.ones(1, 12, 4)
    geometry = Geometry(8, 12, 4, 4)
    state = SparseAttention(Haste(engine='reference', drift=4.0))
    state(q, k, k, geometry)
    state(q + 1, k, k, geometry)
    assert state.state is not None
    assert not state.state.refresh.any()
    dense = SparseAttention(replace(state.config, mode='sparse'))
    dense(q, k, k, geometry)
    dense(q, k, k, geometry)
    assert dense.state is not None
    assert dense.state.refresh.all()


def test_zero_vectors_empty_clusters_and_permutation():
    x = torch.zeros(2, 11, 4)
    labels, centers = clusters(x, 5, 2, 17)
    assert (labels == 0).all()
    assert (centers == 0).all()
    order, offsets = layout(labels, 5)
    torch.testing.assert_close(order, torch.arange(11).expand(2, -1))
    assert offsets.tolist() == [[0, 11, 11, 11, 11, 11]] * 2
    x = torch.randn(2, 11, 4)
    labels, centers = clusters(x, 5, 2, 17)
    order, offsets = layout(labels, 5)
    permuted = x.gather(1, order[..., None].expand_as(x))
    restored = torch.empty_like(x).scatter(1, order[..., None].expand_as(x), permuted)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)


def test_drift_uses_separate_token_means():
    q = torch.full((1, 3, 2), 2.0)
    k = torch.full((1, 7, 2), 3.0)
    assert drift(q, k, torch.zeros(1, 2), torch.zeros(1, 2)).item() == 10


def test_noncontiguous_inputs_and_sparse_mask_oracle():
    q = torch.randn(2, 16, 17).transpose(1, 2)
    k = torch.randn(2, 16, 25).transpose(1, 2)
    v = torch.randn_like(k)
    geometry = Geometry(17, 25, 5, 3)
    state = SparseAttention(
        Haste(engine='reference', backend='svg2', queries=4, keys=6, iterations=2)
    )
    out = state(q, k, v, geometry)
    assert state.state is not None
    mask = expand(state.state.mask, state.state.qlabels, state.state.klabels, geometry)
    assert mask.any(-1).all()
    torch.testing.assert_close(out, attention(q, k, v, mask), rtol=0, atol=0)


def test_antidiagonal_estimator_matches_explicit_repacking():
    q, k = torch.randn(1, 64, 8), torch.randn(1, 64, 8)
    geometry = Geometry(64, 64, 64, 64)
    packed_q = torch.cat([q[:, offset::8] for offset in reversed(range(8))], -1)
    packed_k = torch.cat([k[:, offset::8] for offset in range(8)], -1)
    probabilities = (packed_q @ packed_k.transpose(-1, -2) / (8 * 8**0.5)).softmax(-1)
    expected = probabilities.reshape(1, 2, 4, 2, 4).sum((2, 4))
    torch.testing.assert_close(xscore(q, k, geometry, 32, 8), expected)


def test_reused_heads_skip_scoring(monkeypatch):
    q, k = torch.zeros(2, 32, 8), torch.zeros(2, 64, 8)
    state = SparseAttention(Haste(engine='reference', drift=1))
    geometry = Geometry(32, 64, 8, 8)
    calls = []
    original = xscore

    def count(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr('haste.reference.xscore', count)
    state(q, k, k, geometry)
    state(q, k, k, geometry)
    assert len(calls) == 2
    q[1] += 1
    state(q, k, k, geometry)
    assert len(calls) == 3
    state(q, k, k, geometry, torch.tensor([0.8, 0.9]))
    assert len(calls) == 4


def test_layer_gating_forces_whole_layer_decisions():
    q = torch.zeros(2 * 4, 8, 4)
    k = torch.zeros(2 * 4, 12, 4)
    geometry = Geometry(8, 12, 4, 4)
    state = SparseAttention(Haste(engine='reference', drift=1.0, gate=(0.5, 0.75)))
    state(q, k, k, geometry, heads=4)
    assert state.state is not None
    anchor = state.state.aq.clone()
    # Sample 0: one of four heads drifts (25% < lower bound) so the layer reuses.
    # Sample 1: three of four drift (75% is inside the bounds) so per-head decisions stand.
    q[0] += 1
    q[4:7] += 1
    state(q, k, k, geometry, heads=4)
    assert state.state.refresh.tolist() == [False] * 4 + [True] * 3 + [False]
    torch.testing.assert_close(state.state.aq[:4], anchor[:4])
    torch.testing.assert_close(state.state.aq[4:7], torch.ones(3, 4))
    # Sample 1: four of four drift (100% > upper bound); sample 0 stays reused.
    q[4:] += 1
    state(q, k, k, geometry, heads=4)
    assert state.state.refresh.tolist() == [False] * 4 + [True] * 4
    uneven = SparseAttention(Haste(engine='reference', gate=(0.2, 0.8)))
    uneven(q, k, k, geometry, heads=3)
    with pytest.raises(ValueError, match='divide'):
        uneven(q, k, k, geometry, heads=3)


def test_gate_bounds_are_validated():
    with pytest.raises(ValueError, match='gate'):
        Haste(gate=(0.9, 0.1))
    # TOML arrays arrive as lists and are frozen into tuples.
    settings: dict[str, Any] = {'gate': [0.5]}
    with pytest.raises(ValueError, match='gate'):
        Haste(**settings)
    settings['gate'] = [0.1, 0.9]
    assert Haste(**settings).gate == (0.1, 0.9)
