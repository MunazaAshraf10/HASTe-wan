import math
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import load_video
from torch import Tensor
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from haste.images import comparison


def frames_tensor(frames: np.ndarray, device: torch.device) -> Tensor:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError('Expected video frames [frames, height, width, 3]')
    tensor = torch.as_tensor(array.copy(), device=device).permute(0, 3, 1, 2).float()
    if array.dtype == np.uint8:
        tensor /= 255
    if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise ValueError('Video values must be finite and in [0, 1]')
    return tensor


@torch.inference_mode()
def compare(baseline: np.ndarray, candidate: np.ndarray, device: torch.device) -> dict:
    '''Compute frame averaged SSIM/LPIPS and temporal residual MAE.

    PSNR uses global video MSE and unit data range. Exact equality is encoded
    explicitly because JSON does not define positive infinity.
    '''

    if baseline.shape != candidate.shape:
        raise ValueError('Paired videos must have identical dimensions')
    if baseline.ndim != 4 or baseline.shape[-1] != 3 or len(baseline) == 0:
        raise ValueError('Expected a nonempty RGB video')
    net = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device).eval()
    errors, ssim, perceptual, temporal = [], [], [], []
    previous: tuple[Tensor, Tensor] | None = None
    for index in range(len(baseline)):
        a = frames_tensor(baseline[index : index + 1], device)
        b = frames_tensor(candidate[index : index + 1], device)
        errors.append((a - b).square().mean())
        ssim.append(structural_similarity_index_measure(b, a, data_range=1.0))
        perceptual.append(net(a, b))
        net.reset()
        if previous is not None:
            temporal.append(((b - previous[1]) - (a - previous[0])).abs().mean())
        previous = (a, b)
    mse = torch.stack(errors).mean().item()
    return dict(
        mse=mse,
        psnr=None if mse == 0 else -10 * math.log10(mse),
        identical=mse == 0,
        ssim=torch.stack(ssim).mean().item(),
        lpips=torch.stack(perceptual).mean().item(),
        temporal_error=torch.stack(temporal).mean().item() if temporal else None,
        ssim_frames=[value.item() for value in ssim],
    )


def load_frames(path: Path) -> np.ndarray:
    '''Read a lossless NumPy video or decode a video file into RGB frames.'''
    if path.suffix == '.npy':
        return np.load(path, allow_pickle=False)
    frames = load_video(str(path))
    if not isinstance(frames, list):
        raise ValueError(f'Expected decoded frames from {path}')
    return np.stack([np.asarray(frame.convert('RGB')) for frame in frames])


def compare_files(baseline: Path, candidate: Path, output: Path | None = None) -> dict:
    '''Paired quality metrics and a comparison sheet for two saved generations.'''
    a, b = load_frames(baseline), load_frames(candidate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = compare(a, b, device)
    folder = candidate.parent if output is None else output
    folder.mkdir(parents=True, exist_ok=True)
    comparison(a, b, metrics['ssim_frames']).save(folder / 'comparison.png')
    return dict(baseline=str(baseline), candidate=str(candidate), **metrics)
