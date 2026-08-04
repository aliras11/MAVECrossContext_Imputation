"""Summarize pair/map and pooled variability across randomized splits."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from decomposition.summarize_decomposition import (
    add_b0_all,
    aggregate_moments,
    derive_metrics,
)


PAIR_SPLIT_KEYS = [
    "model_family",
    "model_variant",
    "is_primary_variant",
    "rate",
    "split",
    "task",
    "b0_subtype",
    "source",
    "target",
]

POOLED_SPLIT_KEYS = [
    "model_family",
    "model_variant",
    "is_primary_variant",
    "rate",
    "split",
    "task",
    "b0_subtype",
]

PAIR_SUMMARY_KEYS = [
    column for column in PAIR_SPLIT_KEYS if column != "split"
]
POOLED_SUMMARY_KEYS = [
    column for column in POOLED_SPLIT_KEYS if column != "split"
]

SPLIT_METRICS = [
    "MSE",
    "RMSE",
    "mean_sigma2",
    "tau2_plus_b2_raw",
    "VR_raw",
    "rho_raw",
    "mean_error",
    "calibration_lambda",
]


def build_pair_map_by_split(stats: pd.DataFrame) -> pd.DataFrame:
    """Derive statistics separately for every pair/map and split."""
    aggregated = aggregate_moments(stats, PAIR_SPLIT_KEYS)
    with_b0_all = add_b0_all(
        aggregated,
        group_columns=[
            column for column in PAIR_SPLIT_KEYS if column != "b0_subtype"
        ],
    )
    return derive_metrics(with_b0_all, pooled_across_pairs=False)


def build_pooled_by_split(stats: pd.DataFrame) -> pd.DataFrame:
    """Cell-pool over pairs separately within every split."""
    aggregated = aggregate_moments(stats, POOLED_SPLIT_KEYS)
    with_b0_all = add_b0_all(
        aggregated,
        group_columns=[
            column for column in POOLED_SPLIT_KEYS if column != "b0_subtype"
        ],
    )
    return derive_metrics(with_b0_all, pooled_across_pairs=True)


def summarize_across_splits(
    by_split: pd.DataFrame,
    *,
    group_columns: list[str],
    expected_splits: int,
    metric_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize the realized statistic across equivalently generated splits."""
    if expected_splits <= 0:
        raise ValueError("expected_splits must be positive")
    metrics = metric_columns or SPLIT_METRICS
    missing_metrics = [metric for metric in metrics if metric not in by_split]
    if missing_metrics:
        raise ValueError(f"missing split metric columns: {missing_metrics}")

    rows = []
    grouped = by_split.groupby(group_columns, dropna=False, sort=True)
    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        identity = dict(zip(group_columns, group_key))
        split_count = int(group["split"].nunique())
        if split_count != expected_splits or len(group) != expected_splits:
            raise ValueError(
                f"expected {expected_splits} splits for {identity}, "
                f"found {split_count}"
            )
        row = {**identity, "split_count": split_count}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            valid_count = int(values.notna().sum())
            standard_deviation = values.std(ddof=1)
            row.update(
                {
                    f"{metric}_valid_count": valid_count,
                    f"{metric}_mean": values.mean(),
                    f"{metric}_sd": standard_deviation,
                    f"{metric}_variance": values.var(ddof=1),
                    f"{metric}_MCSE": (
                        standard_deviation / math.sqrt(valid_count)
                        if valid_count
                        else float("nan")
                    ),
                    f"{metric}_min": values.min(),
                    f"{metric}_q025": values.quantile(0.025),
                    f"{metric}_q25": values.quantile(0.25),
                    f"{metric}_median": values.quantile(0.5),
                    f"{metric}_q75": values.quantile(0.75),
                    f"{metric}_q975": values.quantile(0.975),
                    f"{metric}_max": values.max(),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        group_columns, kind="stable"
    ).reset_index(drop=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize_split_variability(
    *,
    stats_path: Path,
    output_root: Path,
    expected_splits: int,
) -> dict[str, Path]:
    """Write the four requested by-split and across-split artifacts."""
    stats = pd.read_csv(stats_path)
    pair_map_by_split = build_pair_map_by_split(stats)
    pooled_by_split = build_pooled_by_split(stats)
    pair_map_summary = summarize_across_splits(
        pair_map_by_split,
        group_columns=PAIR_SUMMARY_KEYS,
        expected_splits=expected_splits,
    )
    pooled_summary = summarize_across_splits(
        pooled_by_split,
        group_columns=POOLED_SUMMARY_KEYS,
        expected_splits=expected_splits,
    )

    outputs = {
        "pair_map_by_split": (
            output_root / "decomposition_pair_map_by_split.csv"
        ),
        "pair_map_split_summary": (
            output_root / "decomposition_pair_map_split_summary.csv"
        ),
        "pooled_by_split": (
            output_root / "decomposition_pooled_by_split.csv"
        ),
        "pooled_split_summary": (
            output_root / "decomposition_pooled_split_summary.csv"
        ),
    }
    _atomic_write_csv(pair_map_by_split, outputs["pair_map_by_split"])
    _atomic_write_csv(pair_map_summary, outputs["pair_map_split_summary"])
    _atomic_write_csv(pooled_by_split, outputs["pooled_by_split"])
    _atomic_write_csv(pooled_summary, outputs["pooled_split_summary"])
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize variability across decomposition splits"
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-splits", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = summarize_split_variability(
        stats_path=args.stats,
        output_root=args.output_root,
        expected_splits=args.expected_splits,
    )
    print(outputs["pooled_split_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
