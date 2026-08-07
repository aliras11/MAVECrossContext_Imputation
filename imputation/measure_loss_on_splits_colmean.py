"""Produce task-matched Column Mean losses for regular splits.

Column Mean produces one target-only prediction per assay context.  This
scorer reuses that prediction for the within-map task and for every ordered
non-self source-to-target task, selecting evaluation cells from the target
mask and source availability in the training split.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path
import re

import numpy as np
import pandas as pd

from runtime_paths import (
    FULL_DATA_CSV,
    LOSS_RESULTS_DIR,
    REGULAR_SPLITS_DIR,
    RUN_ROOT,
)


CONTEXTS = (
    "av12",
    "av25",
    "av100",
    "av200",
    "wt12",
    "wt25",
    "wt100",
    "wt200",
)
REGULAR_RATES = (10, 20, 40, 60, 80, 90)
DEFAULT_EXPECTED_SPLITS = 50
REGULAR_ROWS_PER_SPLIT = len(CONTEXTS) + 2 * len(CONTEXTS) * (len(CONTEXTS) - 1)
OUTPUT_COLUMNS = (
    "dataset",
    "model",
    "rate",
    "split",
    "src",
    "tgt",
    "shift_type",
    "loss_type",
    "rmse",
    "n_points",
    "sse",
    "prediction_file",
    "train_file",
    "mask_file",
)
LOGICAL_KEY = (
    "dataset",
    "model",
    "rate",
    "split",
    "src",
    "tgt",
    "loss_type",
)
TRAIN_FILE_PATTERN = re.compile(r"^train_split_r(?P<rate>\d+)_s(?P<split>\d+)\.csv$")
TASK_TO_LOSS_TYPE = {
    "W": "within_map",
    "B1": "regression_test",
    "B0": "double_missing",
}


def _validate_unique_hgvs(frame: pd.DataFrame, label: str) -> None:
    if "hgvs_pro" not in frame.columns:
        raise ValueError(f"{label} is missing hgvs_pro")
    if frame["hgvs_pro"].isna().any():
        raise ValueError(f"{label} contains null hgvs_pro values")
    duplicates = int(frame["hgvs_pro"].duplicated(keep=False).sum())
    if duplicates:
        raise ValueError(
            f"{label} contains {duplicates} rows with duplicate hgvs_pro values"
        )


def align_frames(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align four complete split artifacts one-to-one by ``hgvs_pro``."""
    frames = (
        (full_df, "full data"),
        (train_df, "train split"),
        (mask_df, "mask"),
        (prediction_df, "prediction"),
    )
    for frame, label in frames:
        _validate_unique_hgvs(frame, label)

    reference_keys = set(full_df["hgvs_pro"])
    for frame, label in frames[1:]:
        candidate_keys = set(frame["hgvs_pro"])
        missing = reference_keys - candidate_keys
        unexpected = candidate_keys - reference_keys
        if missing or unexpected:
            raise ValueError(
                f"{label} hgvs_pro keys do not match full data "
                f"(missing={len(missing)}, unexpected={len(unexpected)})"
            )

    order = pd.Index(full_df["hgvs_pro"], name="hgvs_pro")
    return tuple(
        frame.set_index("hgvs_pro", drop=False).loc[order].reset_index(drop=True)
        for frame, _label in frames
    )


def _numeric_values(values: pd.Series, *, column: str, label: str) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & converted.isna()
    if invalid.any():
        examples = values.loc[invalid].astype(str).drop_duplicates().head(3).tolist()
        raise ValueError(
            f"{label} column {column!r} contains {int(invalid.sum())} "
            f"nonnumeric nonmissing value(s); examples={examples}"
        )
    return converted


def _numeric(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"{label} is missing required column {column!r}")
    return _numeric_values(frame[column], column=column, label=label)


def _finite(values: pd.Series) -> pd.Series:
    return pd.Series(
        np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)),
        index=values.index,
    )


def _validate_target_mask(mask: pd.Series, *, column: str) -> None:
    if mask.isna().any() or not mask.isin({0, 1}).all():
        raise ValueError(f"mask column {column} must contain only 0 or 1")


def _same_including_missing(
    left: pd.Series,
    right: pd.Series,
    *,
    column: str,
    left_label: str,
    right_label: str,
) -> bool:
    left_numeric = _numeric_values(left, column=column, label=left_label)
    right_numeric = _numeric_values(right, column=column, label=right_label)
    both_missing = left_numeric.isna() & right_numeric.isna()
    both_finite = left_numeric.notna() & right_numeric.notna()
    # Permit only ordinary CSV round-trip and group-mean floating-point noise.
    return bool(
        (both_missing | both_finite).all()
        and np.isclose(left_numeric.loc[both_finite], right_numeric.loc[both_finite], rtol=1e-8, atol=1e-10).all()
    )


def _validate_target_artifacts(
    full: pd.DataFrame, train: pd.DataFrame, mask: pd.DataFrame,
    prediction: pd.DataFrame, *, target: str, no_double: bool,
) -> None:
    target_masks: dict[str, pd.Series] = {}
    for suffix in ("score", "se"):
        column = f"{target}_{suffix}"
        full_values, train_values, mask_values = (
            _numeric(full, column, "full data"), _numeric(train, column, "train split"),
            _numeric(mask, column, "mask"),
        )
        _validate_target_mask(mask_values, column=column)
        held = mask_values.eq(1)
        if not _finite(full_values.loc[held]).all() or _finite(train_values.loc[held]).any():
            raise ValueError(f"target artifact contradiction for {target}: mask-1 {suffix} must be finite in full and missing in train")
        if not _same_including_missing(
            train_values.loc[~held],
            full_values.loc[~held],
            column=column,
            left_label="train split",
            right_label="full data",
        ):
            raise ValueError(f"target artifact contradiction for {target}: mask-0 {suffix} differs from full data")
        target_masks[suffix] = mask_values
    if not target_masks["score"].eq(target_masks["se"]).all():
        raise ValueError(f"target artifact contradiction for {target}: score and SE masks differ")
    score = f"{target}_score"
    train_score = _numeric(train, score, "train split")
    if "pos_aminoacid" not in train:
        raise ValueError("train split is missing pos_aminoacid for Column Mean reconstruction")
    expected = train_score.fillna(train.groupby("pos_aminoacid")[score].transform("mean")).fillna(train_score.mean())
    actual = _numeric(prediction, score, "prediction")
    if not _same_including_missing(
        actual,
        expected,
        column=score,
        left_label="prediction",
        right_label="deterministic Column Mean reconstruction",
    ):
        raise ValueError(f"prediction artifact for {target}_score does not match deterministic Column Mean reconstruction")
    if no_double:
        for context in CONTEXTS:
            for suffix in ("score", "se"):
                column = f"{context}_{suffix}"
                mask_values = _numeric(mask, column, "mask")
                if mask_values.isna().any() or not mask_values.isin({0, 1}).all():
                    raise ValueError(f"no-double mask column {column} must contain only 0 or 1")
                if context != target:
                    if not mask_values.eq(0).all():
                        raise ValueError(
                            "no-double split violates non-target artifact contract "
                            f"for {column}: mask must be zero for every row"
                        )
                    if not _same_including_missing(
                        train[column],
                        full[column],
                        column=column,
                        left_label="no-double train split",
                        right_label="full data",
                    ):
                        raise ValueError(
                            "no-double split violates non-target artifact contract "
                            f"for {column}: train split differs from full data"
                        )


def _shift_type(source: str, target: str) -> str:
    if source == target:
        return "self"
    source_family = "wt" if source.startswith("wt") else "av"
    target_family = "wt" if target.startswith("wt") else "av"
    return f"{source_family}→{target_family}"


def _relative_to_run_root(path: Path, run_root: Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_root = Path(run_root).resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise ValueError(
            f"artifact path must be inside run root {resolved_root}: {resolved_path}"
        ) from error


def _score_row(
    *,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    dataset: str,
    rate: int,
    split: int,
    source: str,
    target: str,
    task: str,
    prediction_file: str,
    train_file: str,
    mask_file: str,
) -> dict[str, object]:
    target_column = f"{target}_score"
    truth = _numeric(full_df, target_column, "full data")
    prediction = _numeric(prediction_df, target_column, "prediction")
    target_mask = _numeric(mask_df, target_column, "mask")
    _validate_target_mask(target_mask, column=target_column)

    selected = target_mask.eq(1)
    if task == "B1" or task == "B0":
        source_train = _numeric(train_df, f"{source}_score", "train split")
        source_available = _finite(source_train)
        selected &= source_available if task == "B1" else ~source_available
    elif task != "W":
        raise ValueError(f"unsupported Column Mean task: {task}")

    eligible = selected & _finite(truth) & _finite(prediction)
    n_points = int(eligible.sum())
    if n_points == 0:
        raise ValueError(
            "zero eligible points for "
            f"{dataset} rate={rate} split={split} {source}->{target} "
            f"{TASK_TO_LOSS_TYPE[task]}"
        )

    error = (
        prediction.loc[eligible].to_numpy(dtype=float)
        - truth.loc[eligible].to_numpy(dtype=float)
    )
    sse = float(np.dot(error, error))
    rmse = float(np.sqrt(sse / n_points))
    if not np.isfinite(sse) or not np.isfinite(rmse):
        raise ValueError(
            f"non-finite loss for {dataset} rate={rate} split={split} "
            f"{source}->{target} {TASK_TO_LOSS_TYPE[task]}"
        )

    return {
        "dataset": dataset,
        "model": "col_mean",
        "rate": int(rate),
        "split": int(split),
        "src": source,
        "tgt": target,
        "shift_type": _shift_type(source, target),
        "loss_type": TASK_TO_LOSS_TYPE[task],
        "rmse": rmse,
        "n_points": n_points,
        "sse": sse,
        "prediction_file": prediction_file,
        "train_file": train_file,
        "mask_file": mask_file,
    }


def build_task_loss_rows(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    dataset: str,
    rate: int,
    split: int,
    prediction_path: Path,
    train_path: Path,
    mask_path: Path,
    run_root: Path,
    targets: Sequence[str],
    include_within: bool,
    include_b0: bool,
) -> list[dict[str, object]]:
    """Build task-matched rows for one wide Column Mean prediction file."""
    full, train, mask, prediction = align_frames(
        full_df, train_df, mask_df, prediction_df
    )
    paths = {
        "prediction_file": _relative_to_run_root(prediction_path, run_root),
        "train_file": _relative_to_run_root(train_path, run_root),
        "mask_file": _relative_to_run_root(mask_path, run_root),
    }
    rows: list[dict[str, object]] = []
    for target in targets:
        if target not in CONTEXTS:
            raise ValueError(f"unknown target context: {target}")
        _validate_target_artifacts(
            full, train, mask, prediction, target=target,
            no_double=dataset == "no_double",
        )
        if include_within:
            rows.append(
                _score_row(
                    full_df=full,
                    train_df=train,
                    mask_df=mask,
                    prediction_df=prediction,
                    dataset=dataset,
                    rate=rate,
                    split=split,
                    source=target,
                    target=target,
                    task="W",
                    **paths,
                )
            )
        for source in CONTEXTS:
            if source == target:
                continue
            rows.append(
                _score_row(
                    full_df=full,
                    train_df=train,
                    mask_df=mask,
                    prediction_df=prediction,
                    dataset=dataset,
                    rate=rate,
                    split=split,
                    source=source,
                    target=target,
                    task="B1",
                    **paths,
                )
            )
            if include_b0:
                rows.append(
                    _score_row(
                        full_df=full,
                        train_df=train,
                        mask_df=mask,
                        prediction_df=prediction,
                        dataset=dataset,
                        rate=rate,
                        split=split,
                        source=source,
                        target=target,
                        task="B0",
                        **paths,
                    )
                )
    return rows


def iter_split_inputs(
    *,
    split_root: Path,
    prediction_root: Path,
    rate: int,
    expected_splits: int,
    scope: str,
) -> Iterable[tuple[int, Path, Path, Path]]:
    if expected_splits < 1:
        raise ValueError("expected_splits must be a positive integer")
    input_dir = split_root / f"test_frac_{rate}"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {input_dir}")
    train_paths = sorted(input_dir.glob(f"train_split_r{rate}_s*.csv"))
    if not train_paths:
        raise FileNotFoundError(f"no train splits found in {input_dir}")
    split_paths: dict[int, Path] = {}
    for train_path in train_paths:
        match = TRAIN_FILE_PATTERN.fullmatch(train_path.name)
        if match is None or int(match.group("rate")) != int(rate):
            raise ValueError(f"unexpected train filename: {train_path}")
        split = int(match.group("split"))
        if split in split_paths:
            raise ValueError(f"duplicate train split for rate={rate}, split={split}")
        split_paths[split] = train_path

    expected_ids = set(range(1, expected_splits + 1))
    observed_ids = set(split_paths)
    missing = sorted(expected_ids - observed_ids)
    unexpected = sorted(observed_ids - expected_ids)
    if missing or unexpected:
        raise ValueError(
            f"{scope} split IDs must equal 1..{expected_splits}; "
            f"missing={missing}, unexpected={unexpected}"
        )

    for split in sorted(split_paths):
        train_path = split_paths[split]
        mask_path = train_path.with_name(f"mask_r{rate}_s{split}.csv")
        prediction_path = (
            prediction_root
            / f"mean_imputed_{rate}"
            / f"split_{split}"
            / f"mean_imputed_split{split}.csv"
        )
        for path, label in ((mask_path, "mask"), (prediction_path, "prediction")):
            if not path.is_file():
                raise FileNotFoundError(f"{label} file not found: {path}")
        yield split, train_path, mask_path, prediction_path


def validate_output(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        raise ValueError("no Column Mean task-loss rows were generated")
    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    duplicated = result.duplicated(list(LOGICAL_KEY), keep=False)
    if duplicated.any():
        example = result.loc[duplicated, list(LOGICAL_KEY)].iloc[0].to_dict()
        raise ValueError(f"duplicate logical Column Mean loss key: {example}")
    if (result["n_points"] <= 0).any():
        raise ValueError("Column Mean task losses contain a zero-point row")
    for column in ("rmse", "sse"):
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Column Mean task losses contain invalid {column}")
    return result


def validate_expected_row_count(
    result: pd.DataFrame,
    *,
    expected_rows: int,
    scope: str,
) -> pd.DataFrame:
    if len(result) != expected_rows:
        raise ValueError(
            f"{scope} expected {expected_rows} task-loss rows, found {len(result)}"
        )
    return result


def produce_regular_task_losses(
    *,
    full_data_path: Path,
    base_dir: Path,
    run_root: Path,
    rates: Sequence[int],
    expected_splits: int = DEFAULT_EXPECTED_SPLITS,
) -> pd.DataFrame:
    full_df = pd.read_csv(full_data_path)
    rows: list[dict[str, object]] = []
    for rate in rates:
        for split, train_path, mask_path, prediction_path in iter_split_inputs(
            split_root=base_dir,
            prediction_root=base_dir,
            rate=int(rate),
            expected_splits=expected_splits,
            scope=f"regular rate={int(rate)}",
        ):
            rows.extend(
                build_task_loss_rows(
                    full_df,
                    pd.read_csv(train_path),
                    pd.read_csv(mask_path),
                    pd.read_csv(prediction_path),
                    dataset="regular",
                    rate=int(rate),
                    split=split,
                    prediction_path=prediction_path,
                    train_path=train_path,
                    mask_path=mask_path,
                    run_root=run_root,
                    targets=CONTEXTS,
                    include_within=True,
                    include_b0=True,
                )
            )
    result = validate_output(rows)
    return validate_expected_row_count(
        result,
        expected_rows=len(rates) * expected_splits * REGULAR_ROWS_PER_SPLIT,
        scope="regular Column Mean",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce W, B1, and B0 task-matched Column Mean losses."
    )
    parser.add_argument("--full-data", type=Path, default=FULL_DATA_CSV)
    parser.add_argument("--base-dir", type=Path, default=REGULAR_SPLITS_DIR)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=LOSS_RESULTS_DIR / "column_mean_task_losses_regular.csv",
    )
    parser.add_argument(
        "--rates", nargs="+", type=int, default=list(REGULAR_RATES)
    )
    parser.add_argument(
        "--expected-splits",
        type=int,
        default=DEFAULT_EXPECTED_SPLITS,
        help="Require exact split IDs 1..N for every requested rate (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = produce_regular_task_losses(
        full_data_path=args.full_data,
        base_dir=args.base_dir,
        run_root=args.run_root,
        rates=args.rates,
        expected_splits=args.expected_splits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} task-matched Column Mean rows to {args.output}")


if __name__ == "__main__":
    main()
