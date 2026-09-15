import argparse
from pathlib import Path

from haste.benchmark import kernels
from haste.calibrate import launch as calibrate
from haste.config import Config
from haste.data import write_json
from haste.metrics import compare_files
from haste.report import summarize
from haste.runner import launch


def main() -> None:
    parser = argparse.ArgumentParser(description='HASTE sparse attention for Wan Animate 2')
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='Generate with HASTE or the dense baseline')
    generate.add_argument('config', type=Path)
    generate.add_argument('--baseline', action='store_true')
    benchmark = commands.add_parser(
        'benchmark', help='Run paired generation and fidelity evaluation'
    )
    benchmark.add_argument('config', type=Path)
    benchmark.add_argument('--sweep', action='store_true')
    kernel = commands.add_parser('kernels', help='Measure CUDA attention components')
    kernel.add_argument('config', type=Path)
    calibration = commands.add_parser('calibrate', help='Measure and solve head thresholds')
    calibration.add_argument('config', type=Path)
    report = commands.add_parser('summarize', help='Aggregate experiment records')
    report.add_argument('directory', type=Path)
    compare = commands.add_parser('compare', help='Score two saved generations (.npy or .mp4)')
    compare.add_argument('baseline', type=Path)
    compare.add_argument('candidate', type=Path)
    compare.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.command == 'compare':
        result = compare_files(args.baseline, args.candidate, args.output)
        write_json((args.output or args.candidate.parent) / 'metrics.json', result)
        psnr = 'identical' if result['identical'] else f'{result["psnr"]:.2f} dB'
        print(f'SSIM {result["ssim"]:.4f}  PSNR {psnr}  LPIPS {result["lpips"]:.4f}')
        return
    if args.command == 'summarize':
        result = summarize(args.directory)
        count = len(result['experiments'])
        print(f'Summarized {count} configurations')
        return
    config = Config.read(args.config)
    if args.command == 'calibrate':
        calibrate(config)
        return
    if args.command == 'kernels':
        kernels(config)
        return

    launch(
        config,
        paired=args.command == 'benchmark',
        baseline=getattr(args, 'baseline', False),
        sweep=getattr(args, 'sweep', False),
    )


if __name__ == '__main__':
    main()
