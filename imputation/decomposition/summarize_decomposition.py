"""Derive pair/map and cell-pooled decomposition summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from decomposition.core import ADDITIVE_COLUMNS


PAIR_MAP_KEYS = [
    "model_family",
    "model_variant",
    "is_primary_variant",
    "rate",
    "task",
    "b0_subtype",
    "source",
    "target",
]

PRIMARY_KEYS = [
    "model_family",
    "model_variant",
    "is_primary_variant",
    "rate",
    "task",
    "b0_subtype",
]

VALIDATION_COUNT_KEYS = [
    "model_family",
    "model_variant",
    "is_primary_variant",
    "rate",
    "split",
    "source",
    "target",
]


def aggregate_moments(
    stats: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Sum additive moments before deriving any nonlinear statistic."""
    aggregated = (
        stats.groupby(group_columns, as_index=False, dropna=False)[
            list(ADDITIVE_COLUMNS)
        ]
        .sum()
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )
    aggregated["N"] = aggregated["N"].astype(int)
    return aggregated


def add_b0_all(
    stats: pd.DataFrame,
    *,
    group_columns: list[str],
) -> pd.DataFrame:
    """Derive B0_all from the two disjoint source-missing subtypes."""
    if stats["b0_subtype"].eq("all").any():
        raise ValueError("B0_all is already present")
    b0_parts = stats.loc[
        stats["task"].eq("B0")
        & stats["b0_subtype"].isin({"injected_source", "natural_source"})
    ]
    if b0_parts.empty:
        return stats.copy()
    b0_all = aggregate_moments(b0_parts, group_columns)
    b0_all["b0_subtype"] = "all"
    combined = pd.concat([stats, b0_all], ignore_index=True)
    sort_columns = [
        column
        for column in [*group_columns, "b0_subtype"]
        if column in combined.columns
    ]
    return combined.sort_values(sort_columns, kind="stable").reset_index(
        drop=True
    )


def derive_metrics(
    stats: pd.DataFrame,
    *,
    pooled_across_pairs: bool,
) -> pd.DataFrame:
    """Derive nonlinear statistics from already-pooled additive moments."""
    result = stats.copy()
    n = result["N"].astype(float)
    result["MSE"] = result["SSE"] / n
    result["RMSE"] = np.sqrt(result["MSE"])
    result["mean_sigma2"] = result["Q"] / n
    result["tau2_plus_b2_raw"] = result["MSE"] - result["mean_sigma2"]

    result["VR_raw"] = np.where(
        result["mean_sigma2"].ne(0),
        result["tau2_plus_b2_raw"] / result["mean_sigma2"],
        np.nan,
    )
    result["rho_raw"] = np.where(
        result["MSE"].ne(0),
        result["tau2_plus_b2_raw"] / result["MSE"],
        np.nan,
    )
    result["ratio_interpretable"] = (
        result["tau2_plus_b2_raw"].ge(0)
        & np.isfinite(result["VR_raw"])
        & np.isfinite(result["rho_raw"])
    )
    result["VR"] = result["VR_raw"].where(result["ratio_interpretable"])
    result["rho"] = result["rho_raw"].where(result["ratio_interpretable"])

    result["mean_error"] = result["sum_error"] / n
    result["squared_mean_error"] = result["mean_error"] ** 2
    result["mean_truth"] = result["sum_truth"] / n
    result["mean_prediction"] = result["sum_prediction"] / n
    centered_truth_ss = (
        result["sum_truth2"] - result["sum_truth"] ** 2 / n
    )
    centered_cross = (
        result["sum_truth_prediction"]
        - result["sum_truth"] * result["sum_prediction"] / n
    )
    result["calibration_lambda"] = np.where(
        centered_truth_ss.gt(0),
        centered_cross / centered_truth_ss,
        np.nan,
    )
    result["calibration_alpha"] = (
        result["mean_prediction"]
        - result["calibration_lambda"] * result["mean_truth"]
    )
    slope_deviation = result["calibration_lambda"] - 1
    result["mean_squared_linear_calibration_deviation"] = (
        result["calibration_alpha"] ** 2
        + 2
        * result["calibration_alpha"]
        * slope_deviation
        * result["mean_truth"]
        + slope_deviation**2 * (result["sum_truth2"] / n)
    )

    result["estimand_label"] = np.select(
        [
            result["task"].eq("W"),
            result["task"].eq("B1") & pooled_across_pairs,
            result["task"].eq("B1"),
        ],
        [
            "tau2_plus_b2",
            "tau2_plus_b2_naive_mixed",
            "tau2_plus_b2_naive",
        ],
        default="tau2_plus_b2_naive",
    )
    return result


def build_pair_map_pooled(stats: pd.DataFrame) -> pd.DataFrame:
    """Pool over splits/cells while retaining source→target or W target map."""
    aggregated = aggregate_moments(stats, PAIR_MAP_KEYS)
    with_b0_all = add_b0_all(
        aggregated,
        group_columns=[
            column for column in PAIR_MAP_KEYS if column != "b0_subtype"
        ],
    )
    return derive_metrics(with_b0_all, pooled_across_pairs=False)


def build_primary_pooled(stats: pd.DataFrame) -> pd.DataFrame:
    """Cell-pool over all eligible prediction events, including pairs."""
    aggregated = aggregate_moments(stats, PRIMARY_KEYS)
    with_b0_all = add_b0_all(
        aggregated,
        group_columns=[
            column for column in PRIMARY_KEYS if column != "b0_subtype"
        ],
    )
    return derive_metrics(with_b0_all, pooled_across_pairs=True)


def build_reconciliation(
    *,
    stats: pd.DataFrame,
    pair_map: pd.DataFrame,
    primary: pd.DataFrame,
    validation: pd.DataFrame,
    tolerance: float = 1e-10,
    relative_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Reconcile pair pooling and Stage-1 eligible-event counts."""
    pair_aggregated = aggregate_moments(pair_map, PRIMARY_KEYS)
    pair_check = primary[PRIMARY_KEYS + list(ADDITIVE_COLUMNS)].merge(
        pair_aggregated[PRIMARY_KEYS + list(ADDITIVE_COLUMNS)],
        on=PRIMARY_KEYS,
        how="outer",
        suffixes=("_expected", "_observed"),
        validate="one_to_one",
    )
    deltas = []
    moment_checks = []
    for column in ADDITIVE_COLUMNS:
        expected = pair_check[f"{column}_expected"].fillna(0)
        observed = pair_check[f"{column}_observed"].fillna(0)
        pair_check[f"delta_{column}"] = observed - expected
        deltas.append(pair_check[f"delta_{column}"].abs())
        if column == "N":
            moment_checks.append(observed.eq(expected))
        else:
            moment_checks.append(
                pd.Series(
                    np.isclose(
                        observed,
                        expected,
                        rtol=relative_tolerance,
                        atol=tolerance,
                    ),
                    index=pair_check.index,
                )
            )
    pair_check["check_type"] = "pair_to_primary"
    pair_check["expected_N"] = pair_check["N_expected"].fillna(0)
    pair_check["observed_N"] = pair_check["N_observed"].fillna(0)
    pair_check["delta_N"] = (
        pair_check["observed_N"] - pair_check["expected_N"]
    )
    pair_check["max_abs_moment_delta"] = pd.concat(deltas, axis=1).max(
        axis=1
    )
    pair_check["validation_status"] = np.where(
        pd.concat(moment_checks, axis=1).all(axis=1),
        "ok",
        "failed",
    )

    stats_counts = (
        stats.groupby(
            VALIDATION_COUNT_KEYS,
            as_index=False,
            dropna=False,
        )["N"]
        .sum()
        .rename(columns={"N": "observed_N"})
    )
    validation_counts = (
        validation.groupby(
            VALIDATION_COUNT_KEYS,
            as_index=False,
            dropna=False,
        )["eligible_scored"]
        .sum()
        .rename(columns={"eligible_scored": "expected_N"})
    )
    count_check = validation_counts.merge(
        stats_counts,
        on=VALIDATION_COUNT_KEYS,
        how="outer",
        validate="one_to_one",
    )
    count_check[["expected_N", "observed_N"]] = count_check[
        ["expected_N", "observed_N"]
    ].fillna(0)
    count_check["delta_N"] = (
        count_check["observed_N"] - count_check["expected_N"]
    )
    count_check["max_abs_moment_delta"] = count_check["delta_N"].abs()
    count_check["check_type"] = "stats_to_validation"
    count_check["validation_status"] = np.where(
        count_check["delta_N"].eq(0),
        "ok",
        "failed",
    )

    reconciliation = pd.concat(
        [pair_check, count_check],
        ignore_index=True,
        sort=False,
    )
    sort_columns = [
        column
        for column in [
            "check_type",
            "model_family",
            "model_variant",
            "rate",
            "split",
            "task",
            "b0_subtype",
            "source",
            "target",
        ]
        if column in reconciliation.columns
    ]
    return reconciliation.sort_values(
        sort_columns, kind="stable", na_position="last"
    ).reset_index(drop=True)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def summarize_decomposition(
    *,
    stats_path: Path,
    validation_path: Path,
    output_root: Path,
) -> dict[str, Path]:
    """Write pair/map, primary pooled, and reconciliation outputs."""
    stats = pd.read_csv(stats_path)
    validation = pd.read_csv(validation_path)
    if validation["validation_status"].ne("ok").any():
        raise ValueError("Stage-1 file validation contains failed rows")

    pair_map = build_pair_map_pooled(stats)
    primary = build_primary_pooled(stats)
    reconciliation = build_reconciliation(
        stats=stats,
        pair_map=pair_map,
        primary=primary,
        validation=validation,
    )
    outputs = {
        "primary_pooled": output_root / "decomposition_primary_pooled.csv",
        "pair_map_pooled": output_root / "decomposition_pair_map_pooled.csv",
        "reconciliation": output_root / "decomposition_reconciliation.csv",
    }
    _atomic_write_csv(reconciliation, outputs["reconciliation"])
    failed = reconciliation["validation_status"].ne("ok")
    if failed.any():
        raise ValueError(
            f"decomposition reconciliation failed for {int(failed.sum())} row(s)"
        )
    _atomic_write_csv(primary, outputs["primary_pooled"])
    _atomic_write_csv(pair_map, outputs["pair_map_pooled"])
    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize validated decomposition moments"
    )
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = summarize_decomposition(
        stats_path=args.stats,
        validation_path=args.validation,
        output_root=args.output_root,
    )
    print(outputs["primary_pooled"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
