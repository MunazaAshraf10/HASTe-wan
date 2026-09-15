from pathlib import Path

import numpy as np
import pytest
import torch

from haste.metrics import compare, frames_tensor


def test_frame_conversion_preserves_scale():
    frames = np.array([0, 127, 255], dtype=np.uint8).reshape(1, 1, 1, 3)
    actual = frames_tensor(frames, torch.device('cpu'))
    torch.testing.assert_close(actual.flatten(), torch.tensor([0, 127 / 255, 1]))
    with pytest.raises(ValueError, match='finite'):
        frames_tensor(np.full((1, 2, 2, 3), np.nan), torch.device('cpu'))


@pytest.mark.metrics
@pytest.mark.skipif(
    not (Path(torch.hub.get_dir()) / 'checkpoints' / 'alexnet-owt-7be5be79.pth').exists(),
    reason='Public AlexNet weights are not cached',
)
def test_paired_metrics_match_known_residual():
    rng = np.random.default_rng(4)
    baseline = rng.uniform(0.2, 0.7, (2, 64, 64, 3)).astype(np.float32)
    same = compare(baseline, baseline, torch.device('cpu'))
    assert same['identical'] and same['psnr'] is None
    assert same['ssim'] == pytest.approx(1, abs=1e-6)
    assert same['lpips'] == pytest.approx(0, abs=1e-6)
    shifted = compare(baseline, baseline + 0.1, torch.device('cpu'))
    assert shifted['psnr'] == pytest.approx(20, abs=1e-4)
    assert shifted['temporal_error'] == pytest.approx(0, abs=1e-6)
    assert 0 < shifted['ssim'] < 1
    assert shifted['lpips'] > 0
