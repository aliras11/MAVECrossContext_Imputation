"""Load heterogeneous result CSVs into one validated schema."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import (
    NODOUBLE_RESULT_FILES,
    NORMALIZED_COLUMNS,
    REGULAR_RESULT_FILES,
)

LOGICAL_KEY = [
    "dataset", "model", "rate", "split", "src", "tgt", "loss_type"
]


def _require_files(directory: Path, names: tuple[str, ...]) -> dict[str, Path]:
    directory = Path(directory)
    missing = [directory / name for name in names if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "missing required result files: " + ", ".join(map(str, missing))
        )
    return {name: directory / name for name in names}


def _shift_type(src: str, tgt: str) -> str:
    if src == tgt:
        return "self"
    src_family = "wt" if src.startswith("wt") else "av"
    tgt_family = "wt" if tgt.startswith("wt") else "av"
    return f"{src_family}→{tgt_family}"


def validate_normalized_results(df: pd.DataFrame) -> None:
    missing_columns = sorted(set(NORMALIZED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"missing normalized columns: {missing_columns}")
    if df.duplicated(LOGICAL_KEY).any():
        raise ValueError("duplicate logical result rows")
    rmse = df["rmse"].to_numpy(dtype=float)
    if not np.isfinite(rmse).all():
        raise ValueError("normalized results require finite rmse")
    if (rmse < 0).any():
        raise ValueError("normalized results require non-negative rmse")
    n_points = df["n_points"].to_numpy(dtype=float)
    if not np.isfinite(n_points).all() or (n_points <= 0).any():
        raise ValueError("normalized results require positive n_points")


def _base_row(
    row: pd.Series, model: str, delimiter: str
) -> dict[str, Any]:
    src, tgt = row["src-tgt"].split(delimiter)
    return {
        "model": model,
        "rate": int(row["rate"]),
        "split": int(row["split"]),
        "src": src,
        "tgt": tgt,
        "shift_type": _shift_type(src, tgt),
    }


def _append_score(
    rows: list[dict[str, Any]],
    base: dict[str, Any],
    row: pd.Series,
    loss_type: str,
    score_column: str,
    count_column: str,
) -> None:
    score = row.get(score_column)
    count = row.get(count_column)
    if pd.isna(score):
        return
    if pd.isna(count):
        raise ValueError(
            "raw score requires a finite positive integer count"
        )
    try:
        numeric_count = float(count)
    except (TypeError, ValueError):
        raise ValueError(
            "raw score requires a finite positive integer count"
        ) from None
    if (
        not np.isfinite(numeric_count)
        or numeric_count <= 0
        or not numeric_count.is_integer()
    ):
        raise ValueError(
            "raw score requires a finite positive integer count"
        )
    rows.append({
        **base,
        "loss_type": loss_type,
        "rmse": float(score),
        "n_points": int(numeric_count),
    })


def _parse_ae(
    path: Path,
    model: str,
    *,
    include_within: bool,
    include_double: bool,
    regression_for_self: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in pd.read_csv(path).iterrows():
        base = _base_row(row, model, "-to-")
        if include_within and base["src"] == base["tgt"]:
            _append_score(rows, base, row, "within_map", "test", "n_test")
        if base["src"] != base["tgt"] or regression_for_self:
            _append_score(
                rows, base, row, "regression_test",
                "regression_test_loss", "n_regression_test_loss",
            )
            if include_double:
                _append_score(
                    rows, base, row, "double_missing",
                    "double_missing", "n_double_missing",
                )
    return rows


def _parse_mice(
    path: Path,
    model: str,
    *,
    include_within: bool,
    include_double: bool,
    regression_for_self: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in pd.read_csv(path).iterrows():
        base = _base_row(row, model, "->")
        if include_within and base["src"] == base["tgt"]:
            _append_score(
                rows, base, row, "within_map", "rmse_within_map_tgt", "n_test"
            )
        if base["src"] != base["tgt"] or regression_for_self:
            _append_score(
                rows, base, row, "regression_test",
                "rmse_regression", "n_regression_test_loss",
            )
            if include_double:
                _append_score(
                    rows, base, row, "double_missing",
                    "rmse_double_missing", "n_double_missing",
                )
    return rows


def _parse_linear(
    path: Path, *, include_double: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in pd.read_csv(path).iterrows():
        base = _base_row(row, str(row["model"]), "->")
        _append_score(
            rows, base, row, "regression_test",
            "rmse_test", "n_regression_test_loss",
        )
        if include_double:
            _append_score(
                rows, base, row, "double_missing",
                "rmse_double_missing", "n_double_missing",
            )
    return rows


def _parse_column_loss(
    path: Path, model: str, suffix: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in pd.read_csv(path).iterrows():
        column = row.get("column")
        if not isinstance(column, str) or not column.endswith(suffix):
            continue
        context = column.removesuffix(suffix)
        base = {
            "model": model,
            "rate": int(row["test_fraction"]),
            "split": int(row["split"]),
            "src": context,
            "tgt": context,
            "shift_type": "self",
        }
        _append_score(rows, base, row, "within_map", "loss", "n_test")
    return rows


def _parse_pca(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in pd.read_csv(path).iterrows():
        base = _base_row(row, f"pca_k{int(row['n_components'])}", "-to-")
        _append_score(rows, base, row, "within_map", "test", "n_test")
    return rows


def _normalized_frame(
    rows: list[dict[str, Any]], dataset: str
) -> pd.DataFrame:
    result = pd.DataFrame(
        [{**row, "dataset": dataset} for row in rows],
        columns=NORMALIZED_COLUMNS,
    )
    for column in ("rate", "split", "n_points"):
        result[column] = result[column].astype(int)
    validate_normalized_results(result)
    return result


def load_main_results(results_dir: Path) -> pd.DataFrame:
    """Load all required regular result sources into normalized rows."""
    paths = _require_files(results_dir, REGULAR_RESULT_FILES)
    rows: list[dict[str, Any]] = []
    rows.extend(_parse_ae(
        paths["single_AE3_rmse_results.csv"], "single_ae",
        include_within=True, include_double=True,
    ))
    rows.extend(_parse_ae(
        paths["dual_AE3_rmse_results.csv"], "dual_ae",
        include_within=False, include_double=True,
    ))
    rows.extend(_parse_mice(
        paths["mice_loss_measurements_all_splits_rates2.csv"], "mice",
        include_within=False, include_double=True,
    ))
    rows.extend(_parse_mice(
        paths["mice_loss_measurements_all_splits_ratesrf2.csv"], "mice_rf",
        include_within=True, include_double=True,
    ))
    rows.extend(_parse_linear(
        paths["linear_model_loss_measurements_all_splits_rates2.csv"],
        include_double=True,
    ))
    rows.extend(_parse_column_loss(
        paths["blosum_knn_direct_rmse_all_splits.csv"],
        "knn", "_score_imputed",
    ))
    rows.extend(_parse_column_loss(
        paths["col_mean_imputed_results.csv"],
        "col_mean", "_score",
    ))
    rows.extend(_parse_pca(paths["pca_rmse_results_all.csv"]))
    return _normalized_frame(rows, "regular")


def load_nodouble_results(results_dir: Path) -> pd.DataFrame:
    """Load all required no-double sources into regression-test rows."""
    paths = _require_files(results_dir, NODOUBLE_RESULT_FILES)
    rows: list[dict[str, Any]] = []
    rows.extend(_parse_ae(
        paths["single_AE3_rmse_no_double_missing.csv"], "single_ae",
        include_within=False, include_double=False, regression_for_self=True,
    ))
    rows.extend(_parse_ae(
        paths["dual_AE3_rmse_no_double_missing.csv"], "dual_ae",
        include_within=False, include_double=False, regression_for_self=True,
    ))
    rows.extend(_parse_mice(
        paths["mice_loss_no_double_missing.csv"], "mice",
        include_within=False, include_double=False, regression_for_self=True,
    ))
    rows.extend(_parse_mice(
        paths["mice_rf_loss_no_double_missing.csv"], "mice_rf",
        include_within=False, include_double=False, regression_for_self=True,
    ))
    rows.extend(_parse_linear(
        paths["linear_model_loss_no_double_missing.csv"],
        include_double=False,
    ))
    return _normalized_frame(rows, "no_double")
