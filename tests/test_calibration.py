import itertools
import json
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch
from diffusers.models.transformers.transformer_wan_animate_2 import WanAnimate2KVCache

from haste.calibrate import Capture, replay
from haste.calibration import artifact, load_table, solve, spectral
from haste.config import Calibration, Config


def test_milp_matches_exhaustive_search():
    errors = np.array([[0.3, 0.1, 0.0], [0.8, 0.2, 0.0], [0.2, 0.1, 0.0]])
    sparsity = np.array([[0.8, 0.5, 0.1], [0.9, 0.6, 0.2], [0.7, 0.4, 0.0]])
    budget = 0.55
    feasible = [
        (sum(errors[h, c] for h, c in enumerate(choice)), choice)
        for choice in itertools.product(range(3), repeat=3)
        if np.mean([sparsity[h, c] for h, c in enumerate(choice)]) >= budget
    ]
    result = solve(errors, sparsity, budget)
    assert result['optimal']
    assert result['objective'] == pytest.approx(min(feasible)[0])
    with pytest.raises(RuntimeError, match='no feasible'):
        solve(errors, sparsity, 1.0)


@pytest.mark.parametrize('temporal,spatial,band', [(0, 0, 0), (0, 3, 1), (3, 0, 2), (3, 3, 3)])
def test_fft_bands_and_parseval(temporal, spatial, band):
    t = torch.arange(8)[:, None, None]
    h = torch.arange(8)[None, :, None]
    signal = torch.cos(2 * torch.pi * (temporal * t + spatial * h) / 8).expand(8, 8, 8)
    dense = torch.ones_like(signal)
    energies = spectral(signal, dense, Calibration())
    assert energies[band] == pytest.approx(float(signal.square().mean()), abs=1e-6)
    assert energies.sum() == pytest.approx(float(signal.square().mean()), abs=1e-6)
    assert torch.isfinite(spectral(signal, torch.zeros_like(signal), Calibration())).all()


def test_artifact_compatibility(tmp_path):
    config = Config()
    path = tmp_path / 'table.json'
    rows = [dict(branch='cond', block=0, head=0, errors=[3, 2, 1], sparsity=[0.8, 0.6, 0.4])]
    artifact(path, config, rows, {})
    result = load_table(str(path), config)
    assert result['table'][0]['threshold'] == 0.9
    with pytest.raises(ValueError, match='incompatible'):
        load_table(str(path), replace(config, model=replace(config.model, width=640)))
    data = json.loads(path.read_text())
    data['split'] = 'eval'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='development'):
        load_table(str(path), config)


def test_capture_replay_preserves_call_state(tmp_path):
    class Model(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            timestep,
            kv_cache,
            kv_cache_mode,
            reference_grid_sizes,
            is_uncondtion=False,
        ):
            if kv_cache_mode == 'extract':
                kv_cache.get(0).store(torch.ones(1, 2, 1, 4), torch.full((1, 2, 1, 4), 2.0))
            return ([hidden_states[0] + timestep],)

    model = Model()
    cache = WanAnimate2KVCache(1)
    capture = Capture(model, tmp_path, {0})
    args: dict[str, Any] = dict(
        hidden_states=[torch.ones(1, 2, 2, 2)],
        timestep=torch.tensor(2),
        kv_cache=cache,
        reference_grid_sizes=torch.tensor([[1, 1, 2]]),
    )
    try:
        model(**args, kv_cache_mode='extract')
        positional = {name: value for name, value in args.items() if name != 'hidden_states'}
        model(args['hidden_states'], **positional, kv_cache_mode='cached')
        model(**args, kv_cache_mode='cached', is_uncondtion=True)
        args['hidden_states'][0].zero_()
        key_tensor, value_tensor = cache.get(0).get()
        key_tensor.zero_()
        model(**args, kv_cache_mode='extract')
        model(**args, kv_cache_mode='cached', is_uncondtion=True)
    finally:
        capture.close()
    assert len(capture.records) == 2
    for row in capture.records:
        restored, dense = replay(tmp_path, row, torch.device('cpu'))
        assert restored['hidden_states'][0].eq(1).all()
        assert restored['kv_cache'].get(0).key.eq(1).all()
        torch.testing.assert_close(model(**restored)[0][0], dense[0])
    assert not model._forward_hooks and not model._forward_pre_hooks
