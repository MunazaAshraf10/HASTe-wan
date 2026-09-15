import json
from dataclasses import replace

import pytest

from haste.config import Config, Haste
from haste.data import manifest
from haste.runner import identity, variants


def test_grid_is_unique_and_eval_cannot_sweep():
    config = Config()
    grid = variants(config)
    assert len(grid) == 8
    assert len({identity(item) for item in grid}) == len(grid)
    with pytest.raises(ValueError, match='restricted to dev'):
        variants(replace(config, run=replace(config.run, split='eval')))


def test_manifest_rejects_content_leakage(tmp_path):
    (tmp_path / 'image.png').write_bytes(b'image')
    (tmp_path / 'first.mp4').write_bytes(b'same video')
    (tmp_path / 'second.mp4').write_bytes(b'same video')
    rows = [
        dict(name=split, image='image.png', video=video, prompt='A person.', seed=1, split=split)
        for split, video in [('dev', 'first.mp4'), ('eval', 'second.mp4')]
    ]
    path = tmp_path / 'manifest.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match='both dev and eval'):
        manifest(str(path), 'dev')


@pytest.mark.parametrize('kwargs', [{'threshold': 0}, {'block': 16}, {'drift': -1}, {'queries': 0}])
def test_invalid_kernel_settings(kwargs):
    with pytest.raises(ValueError):
        Haste(**kwargs)
