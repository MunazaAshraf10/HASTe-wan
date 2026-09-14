'''Pareto comparisons must use the same data and hardware cohort.'''

from copy import deepcopy

from haste.config import Config
from haste.data import write_json
from haste.report import summarize


def record(bits, speedup, video='a'):
    config = Config().record()
    config['haste']['bits'] = bits
    return dict(
        status='complete',
        config=config,
        speedup=speedup,
        sample=dict(image_sha256='image', video_sha256=video, prompt='A person', seed=42),
        environment=dict(
            gpu='H100', memory=80, packages={'torch': '2.14.0'}, source_sha256='source'
        ),
        metrics=dict(identical=False, psnr=30.0, ssim=0.95, lpips=0.1, temporal_error=0.02),
    )


def test_frontier_does_not_compare_different_videos(tmp_path):
    first = record(8, 1.1)
    second = record(16, 2.0, video='different')
    third = record(24, 1.5)
    for index, row in enumerate((first, second, third)):
        write_json(tmp_path / str(index) / 'record.json', row)
    failed = deepcopy(first)
    failed['status'] = 'failed'
    write_json(tmp_path / 'failed' / 'record.json', failed)
    summary = summarize(tmp_path)
    assert len(summary['experiments']) == 3
    assert sorted(row['speedup'] for row in summary['dev_frontier']) == [1.5, 2.0]
    assert (tmp_path / 'summary.csv').exists()
