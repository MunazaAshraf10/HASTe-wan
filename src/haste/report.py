import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from haste.data import write_json


def cohort(records: list[dict]) -> str:
    '''Compare frontiers only across matching inputs, sampling, and runtime stacks.'''
    signatures = []
    for record in records:
        sample = record['sample']
        env = record['environment']
        signatures.append(
            dict(
                sample={
                    key: sample[key] for key in ('image_sha256', 'video_sha256', 'prompt', 'seed')
                },
                model=record['config']['model'],
                runtime={key: env[key] for key in ('gpu', 'memory', 'packages', 'source_sha256')},
            )
        )
    encoded = json.dumps(
        sorted(signatures, key=lambda row: json.dumps(row, sort_keys=True)), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def summarize(root: Path) -> dict:
    grouped = defaultdict(list)
    for path in root.rglob('record.json'):
        record = json.loads(path.read_text())
        if record['status'] == 'complete' and 'metrics' in record:
            config = record['config']
            key = json.dumps(config, sort_keys=True)
            grouped[key].append(record)
    rows = []
    for key, records in grouped.items():
        config = json.loads(key)
        row: dict[str, Any] = dict(
            config=config,
            split=config['run']['split'],
            samples=len(records),
            cohort=cohort(records),
            identical=sum(r['metrics']['identical'] for r in records),
        )
        for metric in ('ssim', 'lpips', 'temporal_error', 'psnr'):
            values = [r['metrics'][metric] for r in records if r['metrics'][metric] is not None]
            row[metric] = statistics.mean(values) if values else None
        row['speedup'] = statistics.geometric_mean(r['speedup'] for r in records)
        rows.append(row)
    frontier = [
        row
        for row in rows
        if row['split'] == 'dev'
        and not any(
            other['split'] == 'dev'
            and other['cohort'] == row['cohort']
            and other['speedup'] >= row['speedup']
            and other['lpips'] <= row['lpips']
            and (other['speedup'] > row['speedup'] or other['lpips'] < row['lpips'])
            for other in rows
        )
    ]
    result = dict(
        experiments=rows,
        dev_frontier=frontier,
        note='Select a development configuration explicitly before held out evaluation.',
    )
    write_json(root / 'summary.json', result)
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                'split',
                'samples',
                'cohort',
                'identical',
                'speedup',
                'ssim',
                'lpips',
                'temporal_error',
                'psnr',
                'config',
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'config': json.dumps(row['config'], sort_keys=True)})
    return result
