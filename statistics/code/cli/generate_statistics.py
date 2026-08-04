"""Command-line entry point for the statistics pipeline."""

import argparse
from pathlib import Path

from mave_statistics.pipeline import generate_statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--nodouble-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-splits", type=int, default=50)
    parser.add_argument("--minimum-completeness", type=float, default=0.95)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = generate_statistics(
        args.results_dir,
        args.nodouble_results_dir,
        args.output_dir,
        expected_splits=args.expected_splits,
        minimum_completeness=args.minimum_completeness,
    )
    print(f"Generated and validated {len(paths)} statistics CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
