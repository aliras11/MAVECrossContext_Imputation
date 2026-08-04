"""Summarize pooled RMSE values and safely publish statistics tables."""

from collections.abc import Mapping
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from .constants import EXPECTED_STATISTICS_CSVS, MODEL_DISPLAY_NAMES
from .inference import BASELINE_COLUMNS, CONTEXT_COLUMNS, HEADLINE_COLUMNS
from .pooling import AUDIT_KEYS


SUMMARY_COLUMNS = [
    "dataset",
    "loss_type",
    "model",
    "method",
    "rate",
    "n_splits",
    "mean_rmse",
    "se_rmse",
    "ci95_half_width",
    "ci95_low",
    "ci95_high",
    "formatted_mean_se",
]

AUDIT_COLUMNS = [
    *AUDIT_KEYS,
    "observed_rows",
    "expected_rows",
    "completeness_fraction",
]


def summarize_pooled(pooled: pd.DataFrame) -> pd.DataFrame:
    """Return split-level mean, standard error, and normal 95% intervals."""
    group_columns = ["dataset", "loss_type", "model", "rate"]
    required = {*group_columns, "split", "rmse"}
    missing = sorted(required - set(pooled.columns))
    if missing:
        raise ValueError(f"pooled data missing required columns: {missing}")
    if pooled.duplicated([*group_columns, "split"], keep=False).any():
        raise ValueError("duplicate model × split observations in pooled data")

    rows = []
    for keys, group in pooled.groupby(group_columns, sort=True):
        values = group["rmse"]
        n_splits = int(values.count())
        mean_rmse = float(values.mean())
        se_rmse = float(values.std(ddof=1) / np.sqrt(n_splits))
        ci95_half_width = 1.96 * se_rmse
        dataset, loss_type, model, rate = keys
        rows.append({
            "dataset": dataset,
            "loss_type": loss_type,
            "model": model,
            "method": MODEL_DISPLAY_NAMES.get(model, model),
            "rate": rate,
            "n_splits": n_splits,
            "mean_rmse": mean_rmse,
            "se_rmse": se_rmse,
            "ci95_half_width": ci95_half_width,
            "ci95_low": mean_rmse - ci95_half_width,
            "ci95_high": mean_rmse + ci95_half_width,
            "formatted_mean_se": f"{mean_rmse:.4f} ± {se_rmse:.4f}",
        })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _required_columns(filename: str) -> list[str]:
    if filename == "nodouble_model_rate_completeness.csv":
        return AUDIT_COLUMNS
    if filename.startswith("rmse_summary_"):
        return SUMMARY_COLUMNS
    if filename.startswith("vs_colmean_"):
        return BASELINE_COLUMNS
    if filename.startswith("pairwise_mwu_by_context_"):
        return CONTEXT_COLUMNS
    return HEADLINE_COLUMNS


def _validate_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    actual = set(tables)
    expected = set(EXPECTED_STATISTICS_CSVS)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "statistics table inventory mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for filename in sorted(expected):
        frame = tables[filename]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{filename} is not a pandas DataFrame")
        if frame.empty:
            raise ValueError(f"{filename} must not be empty")
        missing_columns = sorted(
            set(_required_columns(filename)) - set(frame.columns)
        )
        if missing_columns:
            raise ValueError(
                f"{filename} lacks required columns: {missing_columns}"
            )


def write_statistics_tables(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, Path]:
    """Stage, read-validate, and publish the exact statistics inventory."""
    _validate_tables(tables)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(
        prefix=f".{output_dir.name}.statistics-staging-",
        dir=output_dir.parent,
    ))
    try:
        for filename in sorted(EXPECTED_STATISTICS_CSVS):
            tables[filename].to_csv(staging_dir / filename, index=False)
        for filename in sorted(EXPECTED_STATISTICS_CSVS):
            staged = pd.read_csv(staging_dir / filename)
            if staged.empty:
                raise ValueError(f"staged CSV is empty: {filename}")

        output_dir.mkdir(parents=True, exist_ok=True)
        for path in output_dir.glob("*.csv"):
            if path.is_file():
                path.unlink()
        for filename in sorted(EXPECTED_STATISTICS_CSVS):
            (staging_dir / filename).replace(output_dir / filename)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    final_names = {
        path.name for path in output_dir.glob("*.csv") if path.is_file()
    }
    if final_names != set(EXPECTED_STATISTICS_CSVS):
        raise RuntimeError(
            "published statistics inventory mismatch: "
            f"{sorted(final_names)}"
        )
    return {
        filename: output_dir / filename
        for filename in sorted(EXPECTED_STATISTICS_CSVS)
    }
