from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MARGIN = 6
HEADER = 18


def to_uint8(frames: np.ndarray) -> np.ndarray:
    array = np.asarray(frames)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError('Expected video frames [frames, height, width, 3]')
    if array.dtype == np.uint8:
        return array
    return (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)


def picks(count: int, columns: int = 4) -> list[int]:
    '''Evenly spaced frame indices, always including the first and last frame.'''
    if count < 1:
        raise ValueError('A video needs at least one frame')
    columns = min(columns, count)
    return sorted({round(i * (count - 1) / max(columns - 1, 1)) for i in range(columns)})


def sheet(rows: list[tuple[str, list[np.ndarray]]], captions: list[str]) -> Image.Image:
    '''Labelled grid of equally sized RGB frames; one caption per column.'''
    height, width = rows[0][1][0].shape[:2]
    columns = len(captions)
    label = 90
    image = Image.new(
        'RGB',
        (
            label + columns * (width + MARGIN) + MARGIN,
            HEADER + len(rows) * (height + MARGIN) + MARGIN,
        ),
        (24, 24, 24),
    )
    draw = ImageDraw.Draw(image)
    for column, caption in enumerate(captions):
        draw.text((label + MARGIN + column * (width + MARGIN), 3), caption, fill=(230, 230, 230))
    for row, (name, frames) in enumerate(rows):
        top = HEADER + MARGIN + row * (height + MARGIN)
        draw.text((MARGIN, top + height // 2 - 5), name, fill=(230, 230, 230))
        for column, frame in enumerate(frames):
            image.paste(Image.fromarray(frame), (label + MARGIN + column * (width + MARGIN), top))
    return image


def save_frames(folder: Path, name: str, frames: np.ndarray) -> list[Path]:
    '''Write a contact sheet and the individual picked frames as PNG files.'''
    folder.mkdir(parents=True, exist_ok=True)
    video = to_uint8(frames)
    indices = picks(len(video))
    written = []
    for index in indices:
        path = folder / f'{name}_frame_{index:03d}.png'
        Image.fromarray(video[index]).save(path)
        written.append(path)
    path = folder / f'{name}_frames.png'
    sheet([(name, [video[i] for i in indices])], [f'frame {i}' for i in indices]).save(path)
    written.append(path)
    return written


def comparison(
    baseline: np.ndarray, candidate: np.ndarray, ssim: list[float] | None = None, gain: int = 4
) -> Image.Image:
    '''Baseline over candidate over amplified absolute difference for picked frames.'''
    a, b = to_uint8(baseline), to_uint8(candidate)
    if a.shape != b.shape:
        raise ValueError('Paired videos must have identical dimensions')
    indices = picks(len(a))
    difference = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * gain, 0, 255).astype(
        np.uint8
    )
    captions = [
        f'frame {i}' + (f'  SSIM {ssim[i]:.3f}' if ssim is not None else '') for i in indices
    ]
    return sheet(
        [
            ('baseline', [a[i] for i in indices]),
            ('haste', [b[i] for i in indices]),
            (f'|diff| x{gain}', [difference[i] for i in indices]),
        ],
        captions,
    )
