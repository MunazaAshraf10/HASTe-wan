'''Input manifests and reproducible artifact metadata.'''

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Sample:
    name: str
    image: str
    video: str
    prompt: str
    seed: int
    split: str

    def __post_init__(self):
        if not self.name or any(
            c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.'
            for c in self.name
        ):
            raise ValueError(
                'Sample names must contain only letters, digits, periods, or underscores'
            )
        if self.name in ('.', '..') or self.split not in ('dev', 'eval'):
            raise ValueError('Invalid sample name or split')
        if not self.prompt.strip() or self.seed < 0:
            raise ValueError('A nonempty prompt and nonnegative seed are required')

    def record(self) -> dict:
        return {
            **asdict(self),
            'image_sha256': digest(Path(self.image)),
            'video_sha256': digest(Path(self.video)),
        }


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest(path: str, split: str) -> list[Sample]:
    source = Path(path).resolve()
    rows = []
    names = set()
    videos: dict[str, str] = {}
    for number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            for key in ('image', 'video'):
                item[key] = str((source.parent / item[key]).resolve())
                if not Path(item[key]).is_file():
                    raise ValueError(f'Missing input: {item[key]}')
            sample = Sample(**item)
            if sample.name in names:
                raise ValueError(f'Duplicate sample name: {sample.name}')
            names.add(sample.name)
            video_hash = digest(Path(sample.video))
            if video_hash in videos and videos[video_hash] != sample.split:
                raise ValueError('The same driving video occurs in both dev and eval splits')
            videos[video_hash] = sample.split
            if sample.split == split:
                rows.append(sample)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'Invalid manifest row {number}: {error}') from error
    if not rows:
        raise ValueError(f'No samples found for split {split}')
    return rows


def environment(device: int) -> dict:
    names = (
        'torch',
        'triton',
        'diffusers',
        'transformers',
        'accelerate',
        'numpy',
        'torchmetrics',
        'torchvision',
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    info = torch.cuda.get_device_properties(device)
    package = Path(__file__).resolve().parent
    source_hash = hashlib.sha256()
    for path in sorted(package.glob('*.py')):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    root = package.parents[1]
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
        cuda=torch.version.cuda,
        gpu=info.name,
        memory=info.total_memory,
        capability=[info.major, info.minor],
        device=device,
        lock_sha256=digest(root / 'uv.lock') if (root / 'uv.lock').exists() else None,
        source_sha256=source_hash.hexdigest(),
        paper_sha256=digest(root / 'HASTE.pdf') if (root / 'HASTE.pdf').exists() else None,
    )


def write_json(path: Path, data: dict) -> None:
    '''Publish a complete record atomically within the artifact directory.'''
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
