import pytest
import torch

from haste.backend import cuda
from haste.config import Haste
from haste.hashing import codes, projections
from haste.reference import linear as reference

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'),
]


@pytest.mark.parametrize(
    'channels,outputs,tokens', [(65, 37, 35), (5120, 13824, 32), (13824, 5120, 33)]
)
@pytest.mark.parametrize('kind', ['gaussian', 'ternary'])
def test_cuda_matches_reference(channels, outputs, tokens, kind):
    assert cuda is not None, 'CUDA tests require Triton'

    config = Haste(window=32, bits=8, projection=kind)
    planes = projections(config).cuda()
    rng = torch.Generator(device='cuda').manual_seed(71)
    x = torch.randn(2, tokens, channels * 2, device='cuda', dtype=torch.bfloat16, generator=rng)[
        ..., ::2
    ]
    weight = (
        torch.randn(outputs, channels, device='cuda', dtype=torch.bfloat16, generator=rng)
        / channels**0.5
    )
    bias = torch.randn(outputs, device='cuda', dtype=torch.bfloat16, generator=rng)
    order, offsets, count = cuda.partition(x[0, :32], planes, ternary=kind == 'ternary')
    labels = codes(x[0, :32], planes)
    groups = count.item()
    bounds = offsets[: groups + 1].cpu().tolist()
    sorted_labels = labels[order]
    assert groups == labels.unique().numel()
    assert bounds[0] == 0 and bounds[-1] == channels
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        assert sorted_labels[start:end].unique().numel() == 1
    with torch.inference_mode():
        expected = reference(x, weight, bias, planes)
        actual = cuda.linear(x, weight, bias, planes, tile=128, ternary=kind == 'ternary')
    torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize('bits', [1, 24])
def test_zero_and_identical_channels(bits):
    assert cuda is not None, 'CUDA tests require Triton'

    planes = projections(Haste(window=32, bits=bits)).cuda()
    weight = torch.randn(35, 67, device='cuda')
    for x in (
        torch.zeros(1, 17, 67, device='cuda'),
        torch.randn(1, 17, 1, device='cuda').expand(1, 17, 67),
    ):
        with torch.inference_mode():
            result = cuda.linear(x, weight, None, planes)
        torch.testing.assert_close(result, x @ weight.T, atol=3e-5, rtol=3e-5)


def test_device_only_execution():
    assert cuda is not None, 'CUDA tests require Triton'

    x = torch.randn(1, 35, 129, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(67, 129, device='cuda', dtype=torch.bfloat16)
    planes = projections(Haste(window=32, bits=8)).cuda()
    with torch.inference_mode():
        cuda.linear(x, weight, None, planes)
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode('error')
            output = cuda.linear(x, weight, None, planes)
        finally:
            torch.cuda.set_sync_debug_mode(previous)
    assert output.shape == (1, 35, 67)
