'''Command line entry points for generation and controlled research experiments.'''

import argparse
from pathlib import Path

from haste.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description='HASTE channel compression for Wan Animate 2')
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='Generate with HASTE or the dense baseline')
    generate.add_argument('config', type=Path)
    generate.add_argument('--baseline', action='store_true')
    benchmark = commands.add_parser(
        'benchmark', help='Run paired generation and fidelity evaluation'
    )
    benchmark.add_argument('config', type=Path)
    benchmark.add_argument('--sweep', action='store_true')
    kernel = commands.add_parser(
        'kernels', help='Measure CUDA components and complete feedforward layers'
    )
    kernel.add_argument('config', type=Path)
    report = commands.add_parser('summarize', help='Aggregate experiment records')
    report.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'summarize':
        from haste.report import summarize

        result = summarize(args.directory)
        count = len(result['experiments'])
        print(f'Summarized {count} configurations')
        return
    config = Config.read(args.config)
    if args.command == 'kernels':
        from haste.benchmark import kernels

        kernels(config)
        return
    from haste.runner import launch

    launch(
        config,
        paired=args.command == 'benchmark',
        baseline=getattr(args, 'baseline', False),
        sweep=getattr(args, 'sweep', False),
    )


if __name__ == '__main__':
    main()
