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


def _parse_task_column_losses(
    path: Path, dataset: str, *, expected_splits: int = 50
) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = [
        "dataset", "model", "rate", "split", "src", "tgt", "shift_type",
        "loss_type", "rmse", "n_points", "sse", "prediction_file",
        "train_file", "mask_file",
    ]
    if list(frame.columns) != required:
        raise ValueError("task-matched Column Mean losses must have exactly the Task 1 columns in order")
    for column in ("prediction_file", "train_file", "mask_file"):
        if frame[column].isna().any():
            raise ValueError("task-matched Column Mean provenance paths must be nonblank run-root-relative paths")
        paths = frame[column].astype(str)
        stripped = paths.str.strip()
        if (paths != stripped).any() or stripped.eq("").any() or stripped.eq(".").any() or stripped.map(lambda value: Path(value).is_absolute() or ".." in Path(value).parts).any():
            raise ValueError("task-matched Column Mean provenance paths must be nonblank run-root-relative paths")
    if set(frame["dataset"].astype(str)) != {dataset}:
        raise ValueError(f"task-matched Column Mean losses must use dataset={dataset!r}")
    if set(frame["model"].astype(str)) != {"col_mean"}:
        raise ValueError("task-matched Column Mean losses must use model='col_mean'")
    numeric = frame[["rmse", "n_points", "sse"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all() or (numeric["rmse"] < 0).any() or (numeric["sse"] < 0).any():
        raise ValueError("task-matched Column Mean losses require finite non-negative RMSE and SSE")
    if (numeric["n_points"] <= 0).any() or not np.equal(numeric["n_points"], np.floor(numeric["n_points"])).all():
        raise ValueError("task-matched Column Mean losses require positive integer n_points")
    if not np.isclose(numeric["sse"], numeric["n_points"] * numeric["rmse"] ** 2, rtol=1e-6, atol=1e-10).all():
        raise ValueError("task-matched Column Mean losses require sse == n_points * rmse^2")
    allowed = {"within_map", "regression_test", "double_missing"}
    if not set(frame["loss_type"]).issubset(allowed):
        raise ValueError("task-matched Column Mean losses have unknown loss_type")
    self_rows = frame["src"].eq(frame["tgt"])
    if not frame.loc[self_rows, "loss_type"].eq("within_map").all() or not frame.loc[~self_rows, "loss_type"].isin({"regression_test", "double_missing"}).all():
        raise ValueError("task-matched Column Mean loss_type/context mismatch")
    if frame.duplicated(LOGICAL_KEY).any():
        raise ValueError("duplicate logical task-matched Column Mean loss rows")
    contexts = ("av12", "av25", "av100", "av200", "wt12", "wt25", "wt100", "wt200")
    rates = (10, 20, 40, 60, 80, 90) if dataset == "regular" else (10, 40, 80, 99, 999)
    expected_rows = len(rates) * expected_splits * (120 if dataset == "regular" else 56)
    if len(frame) != expected_rows or set(frame["rate"]) != set(rates) or set(frame["split"]) != set(range(1, expected_splits + 1)):
        raise ValueError("incomplete task-matched Column Mean loss grid")
    expected = set()
    for rate in rates:
        for split in range(1, expected_splits + 1):
            if dataset == "regular":
                expected.update((rate, split, context, context, "within_map") for context in contexts)
            for source in contexts:
                for target in contexts:
                    if source != target:
                        expected.add((rate, split, source, target, "regression_test"))
                        if dataset == "regular":
                            expected.add((rate, split, source, target, "double_missing"))
    observed = set(frame[["rate", "split", "src", "tgt", "loss_type"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("task-matched Column Mean loss grid has missing or unexpected logical keys")
    expected_shift = frame.apply(lambda row: _shift_type(str(row["src"]), str(row["tgt"])), axis=1)
    if not frame["shift_type"].eq(expected_shift).all():
        raise ValueError("task-matched Column Mean losses have inconsistent shift_type")
    return frame.loc[:, NORMALIZED_COLUMNS].to_dict("records")


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


def load_main_results(results_dir: Path, *, expected_splits: int = 50) -> pd.DataFrame:
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
    rows.extend(_parse_task_column_losses(
        paths["column_mean_task_losses_regular.csv"], "regular", expected_splits=expected_splits
    ))
    rows.extend(_parse_pca(paths["pca_rmse_results_all.csv"]))
    return _normalized_frame(rows, "regular")


def load_nodouble_results(results_dir: Path, *, expected_splits: int = 50) -> pd.DataFrame:
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
    rows.extend(_parse_task_column_losses(
        paths["column_mean_task_losses_no_double.csv"], "no_double", expected_splits=expected_splits
    ))
    return _normalized_frame(rows, "no_double")
