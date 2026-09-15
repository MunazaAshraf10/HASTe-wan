import pytest
import torch
from torch import nn

from haste.config import Haste
from haste.hashing import codes, projections
from haste.linear import CompressedLinear
from haste.reference import linear, merge


def oracle(x, weight, labels, bias=None):
    # The dense channel mixing matrix averages precisely the equivalence classes.
    same = (labels[:, None] == labels[None, :]).double()
    mixing = same / same.sum(dim=0, keepdim=True)
    output = x.double() @ mixing @ weight.double().T
    return output if bias is None else output + bias.double()


@pytest.mark.parametrize(
    'labels',
    [
        torch.arange(9),
        torch.zeros(9, dtype=torch.long),
        torch.tensor([7, 7, 2, 2, 99, 7, 2, 12, 99]),
    ],
)
@pytest.mark.parametrize('with_bias', [False, True])
def test_arbitrary_buckets_match_operator(labels, with_bias):
    rng = torch.Generator().manual_seed(19)
    x = torch.randn(5, 18, generator=rng)[:, ::2]
    weight = torch.randn(7, 18, generator=rng)[:, ::2]
    bias = torch.randn(7, generator=rng) if with_bias else None
    actual = merge(x, weight, labels, bias)
    torch.testing.assert_close(
        actual.double(), oracle(x, weight, labels, bias), atol=2e-6, rtol=2e-6
    )


def test_identical_channels_are_exact():
    rng = torch.Generator().manual_seed(29)
    x = torch.randn(11, 1, generator=rng).expand(11, 8)
    weight = torch.randn(6, 8, generator=rng)
    actual = merge(x, weight, torch.zeros(8, dtype=torch.long))
    torch.testing.assert_close(actual, x @ weight.T, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('kind', ['gaussian', 'ternary'])
def test_centering_and_projection_reproducibility(kind):
    config = Haste(projection=kind)
    planes = projections(config)
    torch.testing.assert_close(planes, projections(config), rtol=0, atol=0)
    assert not torch.equal(planes, projections(config, salt=1))
    x = torch.arange(64 * 7).reshape(64, 7).float()
    shift = torch.arange(64).float()[:, None] * 10
    assert torch.equal(codes(x, planes), codes(x + shift, planes))
    assert torch.equal(codes(torch.zeros_like(x), planes), torch.zeros(7, dtype=torch.long))
    if kind == 'ternary':
        assert set(planes.unique().tolist()) <= {-1, 0, 1}


def test_windows_and_samples_are_independent():
    config = Haste(window=32, bits=3)
    planes = projections(config)
    x = torch.randn(2, 71, 12)
    weight = torch.randn(7, 12)
    actual = linear(x, weight, None, planes)
    expected = torch.empty_like(actual)
    for b in range(2):
        for start in (0, 32, 64):
            part = x[b, start : start + 32]
            expected[b, start : start + 32] = oracle(part, weight, codes(part, planes)).float()
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    changed = x.clone()
    changed[1] += 100
    torch.testing.assert_close(linear(changed, weight, None, planes)[0], actual[0])


def test_layer_preserves_parameters_and_dense_mode():
    source = nn.Linear(9, 5)
    before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    layer = CompressedLinear(source, Haste(backend='reference'))
    x = torch.randn(2, 35, 9)
    with pytest.raises(RuntimeError, match='inference only'):
        layer(x)
    with torch.inference_mode():
        layer(x)
        layer.enabled = False
        torch.testing.assert_close(layer(x), source(x), atol=0, rtol=0)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
