from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from haste.images import comparison, picks, save_frames, to_uint8
from haste.metrics import compare_files, load_frames


def test_picks_cover_first_and_last_frames():
    assert picks(1) == [0]
    assert picks(2) == [0, 1]
    assert picks(81) == [0, 27, 53, 80]
    assert picks(3, columns=8) == [0, 1, 2]


def test_uint8_conversion_rounds_unit_floats():
    frames = np.array([0.0, 0.5, 1.0], dtype=np.float32).reshape(1, 1, 1, 3)
    assert to_uint8(frames).flatten().tolist() == [0, 128, 255]
    raw = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    assert to_uint8(raw) is raw


def test_frames_and_comparison_sheets(tmp_path: Path):
    rng = np.random.default_rng(1)
    baseline = rng.uniform(0, 1, (9, 16, 24, 3)).astype(np.float32)
    candidate = np.clip(baseline + 0.05, 0, 1)
    written = save_frames(tmp_path, 'baseline', baseline)
    assert [path.name for path in written] == [
        'baseline_frame_000.png',
        'baseline_frame_003.png',
        'baseline_frame_005.png',
        'baseline_frame_008.png',
        'baseline_frames.png',
    ]
    frame = np.asarray(Image.open(written[0]))
    np.testing.assert_array_equal(frame, to_uint8(baseline)[0])
    sheet = comparison(baseline, candidate, [0.5] * 9)
    assert sheet.size == (90 + 4 * (24 + 6) + 6, 18 + 3 * (16 + 6) + 6)
    with pytest.raises(ValueError, match='identical dimensions'):
        comparison(baseline, candidate[:4])


@pytest.mark.metrics
@pytest.mark.skipif(
    not (Path(torch.hub.get_dir()) / 'checkpoints' / 'alexnet-owt-7be5be79.pth').exists(),
    reason='Public AlexNet weights are not cached',
)
def test_compare_files_reads_arrays_and_writes_sheet(tmp_path: Path):
    rng = np.random.default_rng(2)
    baseline = rng.uniform(0, 1, (3, 32, 32, 3)).astype(np.float32)
    np.save(tmp_path / 'baseline.npy', baseline)
    np.save(tmp_path / 'haste.npy', np.clip(baseline + 0.1, 0, 1))
    assert load_frames(tmp_path / 'baseline.npy').shape == (3, 32, 32, 3)
    result = compare_files(tmp_path / 'baseline.npy', tmp_path / 'haste.npy', tmp_path / 'out')
    assert (tmp_path / 'out' / 'comparison.png').exists()
    assert len(result['ssim_frames']) == 3 and 0 < result['ssim'] < 1
