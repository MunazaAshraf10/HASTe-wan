from dataclasses import replace

import pytest
import torch

from haste.attention import SparseAttention
from haste.config import Haste
from haste.reference import Geometry, attention, expand

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'),
]


@pytest.mark.parametrize('backend', ['xattention', 'svg2'])
@pytest.mark.parametrize('shape', [(35, 58, 16), (129, 241, 128), (512, 896, 128)])
def test_cuda_matches_mathematical_reference(backend, shape):
    nq, nk, dim = shape
    torch.manual_seed(11)
    q = torch.randn(2, nq, dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, nk, dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    geometry = Geometry(nq, nk, 32, 32)
    cfg = Haste(backend=backend, queries=5, keys=7, iterations=2, updates=1, block=32)
    actual = SparseAttention(cfg)
    expected = SparseAttention(replace(cfg, engine='reference'))
    out = actual(q, k, v, geometry)
    expected(q, k, v, geometry)
    assert actual.state is not None and expected.state is not None
    torch.testing.assert_close(actual.state.scores, expected.state.scores, atol=3e-4, rtol=3e-3)
    mask = expand(actual.state.mask, actual.state.qlabels, actual.state.klabels, geometry)
    torch.testing.assert_close(out, attention(q, k, v, mask), atol=0.025, rtol=0.025)
    anchors = actual.state.aq.clone()
    order = actual.state.korder.clone()
    actual(q, k, v, geometry)
    assert not actual.state.refresh.any()
    torch.testing.assert_close(actual.state.aq, anchors, rtol=0, atol=0)
    torch.testing.assert_close(actual.state.korder, order, rtol=0, atol=0)
    changed = q.clone()
    changed[0] += 4
    actual(changed, k, v, geometry)
    assert actual.state.refresh.tolist() == [True, False]


@pytest.mark.parametrize('backend', ['xattention', 'svg2'])
def test_cuda_full_retention_and_zero_vectors(backend):
    q = torch.zeros(2, 65, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.zeros(2, 97, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    geometry = Geometry(65, 97, 16, 16)
    state = SparseAttention(Haste(backend=backend, threshold=1, queries=5, keys=7, iterations=1))
    actual = state(q, k, v, geometry)
    allowed = geometry.allowed(
        torch.arange(65, device='cuda')[:, None], torch.arange(97, device='cuda')[None, :]
    )
    torch.testing.assert_close(actual, attention(q, k, v, allowed), atol=0.01, rtol=0.01)
