import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from haste.config import Calibration, Config
from haste.data import digest, write_json


def spectral(error: torch.Tensor, dense: torch.Tensor, config: Calibration) -> torch.Tensor:
    '''HASTE Equations 9–11: normalized four-band velocity error.

    Last dimensions are latent time, height, width. Low frequencies satisfy
    abs(fftfreq) < cutoff; spatial low requires both spatial axes to be low.
    These explicit band boundaries are an adaptation of the unspecified split.
    '''
    if error.shape != dense.shape or error.ndim < 3:
        raise ValueError('Velocity tensors must share temporal and spatial dimensions')
    axes = (-3, -2, -1)
    power = torch.fft.fftn(error.float(), dim=axes, norm='ortho').abs().square()
    energy = torch.fft.fftn(dense.float(), dim=axes, norm='ortho').abs().square().sum()
    t, h, w = dense.shape[-3:]
    low_t = torch.fft.fftfreq(t, device=dense.device).abs() < config.cutoff
    low_h = torch.fft.fftfreq(h, device=dense.device).abs() < config.cutoff
    low_w = torch.fft.fftfreq(w, device=dense.device).abs() < config.cutoff
    temporal = low_t[:, None, None]
    spatial = low_h[None, :, None] & low_w[None, None, :]
    bands = (temporal & spatial, temporal & ~spatial, ~temporal & spatial, ~temporal & ~spatial)
    return torch.stack([power[..., band].sum() for band in bands]) / (energy + config.epsilon)


def error_value(
    sparse: list[torch.Tensor], dense: list[torch.Tensor], config: Calibration
) -> tuple[float, list[float]]:
    if len(sparse) != len(dense) or not dense:
        raise ValueError('Sparse and dense outputs must contain matching samples')
    bands = torch.stack(
        [spectral(a - b, b, config) for a, b in zip(sparse, dense, strict=True)]
    ).mean(0)
    weights = torch.tensor(config.weights, device=bands.device)
    return float((bands * weights).sum()), bands.tolist()


def solve(errors: np.ndarray, sparsity: np.ndarray, budget: float, limit: float = 600) -> dict:
    '''HASTE Equation 7: one threshold per head under an average sparsity budget.'''
    errors, sparsity = np.asarray(errors, dtype=np.float64), np.asarray(sparsity, dtype=np.float64)
    if errors.ndim != 2 or errors.shape != sparsity.shape or min(errors.shape) == 0:
        raise ValueError(
            'Error and sparsity tables must have matching nonempty head/candidate axes'
        )
    if not np.isfinite(errors).all() or not np.isfinite(sparsity).all():
        raise ValueError('Calibration measurements must be finite')
    if (errors < 0).any() or (sparsity < 0).any() or (sparsity > 1).any() or not 0 <= budget <= 1:
        raise ValueError('Invalid error or sparsity measurement')
    heads, choices = errors.shape
    matrix = lil_matrix((heads + 1, heads * choices), dtype=np.float64)
    for head in range(heads):
        matrix[head, head * choices : (head + 1) * choices] = 1
    matrix[-1] = sparsity.reshape(1, -1) / heads
    constraints = LinearConstraint(
        matrix.tocsr(), np.r_[np.ones(heads), budget], np.r_[np.ones(heads), np.inf]
    )
    # Rescaling changes neither the minimizer nor the feasible set.
    scale = max(float(errors.max()), 1e-30)
    result = milp(
        errors.ravel() / scale,
        integrality=np.ones(errors.size),
        bounds=Bounds(0, 1),
        constraints=constraints,
        options=dict(time_limit=limit, mip_rel_gap=0),
    )
    if result.x is None:
        raise RuntimeError(f'Calibration solver found no feasible table: {result.message}')
    indices = result.x.reshape(heads, choices).argmax(-1)
    realized = float(sparsity[np.arange(heads), indices].mean())
    integral = np.zeros_like(result.x).reshape(heads, choices)
    integral[np.arange(heads), indices] = 1
    if not np.allclose(result.x, integral.ravel(), atol=1e-6) or realized + 1e-8 < budget:
        raise RuntimeError('Calibration solver did not return a feasible integral assignment')
    return dict(
        indices=indices.tolist(),
        budget=budget,
        sparsity=realized,
        objective=float(errors[np.arange(heads), indices].sum()),
        optimal=result.status == 0,
        gap=float(result.mip_gap),
        message=result.message,
    )


def signature(config: Config) -> dict:
    settings = asdict(config.haste)
    for key in ('mode', 'drift', 'gate', 'threshold', 'table', 'engine'):
        settings.pop(key)
    return dict(
        model=asdict(config.model), attention=settings, calibration=asdict(config.calibration)
    )


def load_table(path: str, config: Config) -> dict:
    if not path:
        raise ValueError('EBC and full modes require a calibration table')
    data = json.loads(Path(path).read_text())
    expected = json.loads(json.dumps(signature(config)))
    if data.get('schema') != 1 or data.get('signature') != expected:
        raise ValueError(
            'Calibration table is incompatible with this model or attention configuration'
        )
    if data.get('split') != 'dev' or data.get('status') != 'complete':
        raise ValueError('Calibration requires a complete development artifact')
    rows = data.get('table', [])
    identities = [(row['branch'], row['block'], row['head']) for row in rows]
    if len(identities) != len(set(identities)) or any(
        not 0 < row['threshold'] <= 1 for row in rows
    ):
        raise ValueError('Calibration table has duplicate heads or invalid thresholds')
    return data


def apply_table(patch, data: dict) -> None:
    rows = {(row['branch'], row['block'], row['head']): row['threshold'] for row in data['table']}
    branches = ('cond', 'uncond') if data['signature']['model']['guidance'] > 1 else ('cond',)
    for entry in patch.entries:
        for branch in branches:
            if branch == 'uncond' and entry.block == 9:
                continue
            keys = [(branch, entry.block, head) for head in range(entry.module.heads)]
            if any(key not in rows for key in keys):
                raise ValueError(f'Missing calibrated heads for {branch} block {entry.block}')
            entry.processor.thresholds[branch] = torch.tensor(
                [rows[key] for key in keys], device=entry.module.to_q.weight.device
            )


def provenance(path: str) -> dict | None:
    if not path:
        return None
    return dict(path=str(Path(path).resolve()), sha256=digest(Path(path)))


def artifact(path: Path, config: Config, rows: list[dict], metadata: dict) -> None:
    table, solutions = [], {}
    for branch in sorted({row['branch'] for row in rows}):
        subset = sorted(
            (row for row in rows if row['branch'] == branch),
            key=lambda row: (row['block'], row['head']),
        )
        errors = np.asarray([row['errors'] for row in subset])
        sparsity = np.asarray([row['sparsity'] for row in subset])
        baseline = config.calibration.thresholds.index(0.9)
        solution = solve(
            errors, sparsity, float(sparsity[:, baseline].mean()), config.calibration.limit
        )
        solutions[branch] = solution
        for row, choice in zip(subset, solution['indices'], strict=True):
            table.append(
                dict(
                    branch=branch,
                    block=row['block'],
                    head=row['head'],
                    threshold=config.calibration.thresholds[choice],
                )
            )
    write_json(
        path,
        dict(
            schema=1,
            status='complete',
            split='dev',
            signature=signature(config),
            measurements=rows,
            solutions=solutions,
            table=table,
            **metadata,
        ),
    )


def key(config: Config) -> str:
    return hashlib.sha256(json.dumps(signature(config), sort_keys=True).encode()).hexdigest()[:16]
