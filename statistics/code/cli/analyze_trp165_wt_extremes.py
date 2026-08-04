"""Command-line entry point for the audited WT Trp165 analysis."""

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from mave_statistics.trp165 import (
    DEFAULT_RATES,
    run_trp165_analysis,
    validate_requested_rates,
    validate_requested_splits,
    validate_source_checksums,
)


_POSITIVE_INTEGER = re.compile(r"[1-9][0-9]*")
_POSITIVE_RANGE = re.compile(r"([1-9][0-9]*)-([1-9][0-9]*)")
_MAX_REQUESTED_VALUE = max(DEFAULT_RATES)


def parse_positive_int_ranges(values: Sequence[str]) -> tuple[int, ...]:
    """Parse tokens such as '1', '10', and '1-50' into unique sorted ints."""
    if not values:
        raise ValueError("at least one positive integer or range is required")
    parsed: set[int] = set()
    for token in values:
        range_match = _POSITIVE_RANGE.fullmatch(token)
        if range_match is not None:
            start, stop = (int(value) for value in range_match.groups())
            if start > stop:
                raise ValueError(f"range start exceeds range end: {token}")
            if stop > _MAX_REQUESTED_VALUE:
                raise ValueError(f"range exceeds supported request bounds: {token}")
            expanded = range(start, stop + 1)
        elif _POSITIVE_INTEGER.fullmatch(token) is not None:
            expanded = (int(token),)
        else:
            raise ValueError(f"invalid positive integer or range: {token}")
        for value in expanded:
            if value in parsed:
                raise ValueError(f"duplicate integer requested: {value}")
            parsed.add(value)
    return tuple(sorted(parsed))


def parse_source_checksums(values: Sequence[str]) -> dict[str, str]:
    """Parse unique safe relative-path=64-hex-digest records."""
    parsed: dict[str, str] = {}
    for record in values:
        path, separator, digest = record.rpartition("=")
        if not separator:
            raise ValueError(
                f"source checksum must use RELATIVE_PATH=SHA256: {record}"
            )
        validated = validate_source_checksums({path: digest})
        normalized_path, normalized_digest = next(iter(validated.items()))
        if normalized_path in parsed:
            raise ValueError(f"duplicate source checksum path: {normalized_path}")
        parsed[normalized_path] = normalized_digest
    return dict(sorted(parsed.items()))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze WT Trp165 B1 transfer between wt12 and wt200."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rates",
        nargs="+",
        default=[str(value) for value in DEFAULT_RATES],
    )
    parser.add_argument("--splits", nargs="+", default=["1-50"])
    parser.add_argument("--repo-commit")
    parser.add_argument(
        "--source-checksum",
        action="append",
        default=[],
        metavar="RELATIVE_PATH=SHA256",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rates = validate_requested_rates(parse_positive_int_ranges(args.rates))
    splits = validate_requested_splits(parse_positive_int_ranges(args.splits))
    paths = run_trp165_analysis(
        args.run_root,
        args.output_dir,
        rates=rates,
        splits=splits,
        repo_commit=args.repo_commit,
        source_checksums=parse_source_checksums(args.source_checksum),
    )
    print(paths["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
