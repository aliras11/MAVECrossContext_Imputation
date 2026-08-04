"""Scientific event definitions and additive statistics."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import pandas as pd


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

FIX_COMMIT = "aa9ad790d3cd1001f785f16cf3856c3e1ea67755"

ADDITIVE_COLUMNS = (
    "N",
    "sum_error",
    "SSE",
    "Q",
    "sum_truth",
    "sum_prediction",
    "sum_truth2",
    "sum_prediction2",
    "sum_truth_prediction",
)


def validate_unique_keys(df: pd.DataFrame, key: str, label: str) -> None:
    """Require a present, non-null, unique identifier column."""
    if key not in df.columns:
        raise ValueError(f"{label} is missing key column {key!r}")
    if df[key].isna().any():
        raise ValueError(f"{label} contains null {key!r} values")
    duplicate_count = int(df[key].duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(
            f"{label} contains {duplicate_count} rows with duplicate {key!r} values"
        )


def classify_variant(hgvs_pro: pd.Series) -> pd.Series:
    """Classify HGVS protein identifiers into analysis variant classes."""
    values = hgvs_pro.astype("string")
    return pd.Series(
        np.select(
            [values.str.endswith("=", na=False), values.str.endswith("Ter", na=False)],
            ["synonymous", "nonsense"],
            default="missense",
        ),
        index=hgvs_pro.index,
        dtype="string",
    )


def _require_same_keys(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    key: str,
    label: str,
) -> None:
    reference_keys = set(reference[key])
    candidate_keys = set(candidate[key])
    unexpected = candidate_keys - reference_keys
    missing = reference_keys - candidate_keys
    if unexpected:
        raise ValueError(f"{label} contains {len(unexpected)} unexpected {key!r} keys")
    if missing:
        raise ValueError(f"{label} is missing {len(missing)} expected {key!r} keys")


def build_prediction_events(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    mask_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    *,
    source: str,
    target: str,
    supported_tasks: Collection[str],
) -> tuple[pd.DataFrame, dict]:
    """Build eligible, classified prediction events for one logical output unit."""
    key = "hgvs_pro"
    for frame, label in (
        (full_df, "full data"),
        (train_df, "train split"),
        (mask_df, "mask"),
        (prediction_df, "prediction"),
    ):
        validate_unique_keys(frame, key, label)
    for frame, label in (
        (train_df, "train split"),
        (mask_df, "mask"),
        (prediction_df, "prediction"),
    ):
        _require_same_keys(full_df, frame, key=key, label=label)

    target_score = f"{target}_score"
    target_se = f"{target}_se"
    source_score = f"{source}_score"

    full_columns = [key, target_score, target_se]
    train_columns = [key, target_score]
    mask_columns = [key, target_score]
    if source != target:
        full_columns.append(source_score)
        train_columns.append(source_score)
        mask_columns.append(source_score)

    full_base = full_df[full_columns].rename(
        columns={target_score: "truth", target_se: "target_se"}
    )
    train_base = train_df[train_columns].rename(
        columns={target_score: "target_train"}
    )
    mask_base = mask_df[mask_columns].rename(
        columns={target_score: "target_mask"}
    )
    if source == target:
        full_base["source_full"] = full_base["truth"]
        train_base["source_train"] = train_base["target_train"]
        mask_base["source_mask"] = mask_base["target_mask"]
    else:
        full_base = full_base.rename(columns={source_score: "source_full"})
        train_base = train_base.rename(columns={source_score: "source_train"})
        mask_base = mask_base.rename(columns={source_score: "source_mask"})

    merged = (
        full_base.merge(
            train_base, on=key, how="left", validate="one_to_one"
        )
        .merge(mask_base, on=key, how="left", validate="one_to_one")
        .merge(
            prediction_df[[key, "prediction"]],
            on=key,
            how="left",
            validate="one_to_one",
        )
    )

    truth_missing = merged["truth"].isna()
    target_se_missing = merged["target_se"].isna()
    source_full_missing = merged["source_full"].isna()
    source_train_missing = merged["source_train"].isna()
    target_mask_missing = merged["target_mask"].isna()
    source_mask_missing = merged["source_mask"].isna()

    for column in (
        "truth",
        "target_se",
        "source_full",
        "source_train",
        "target_mask",
        "source_mask",
    ):
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    def finite(column: str) -> pd.Series:
        return pd.Series(
            np.isfinite(
                merged[column].to_numpy(dtype=float, na_value=np.nan)
            ),
            index=merged.index,
        )

    truth_finite = finite("truth")
    target_se_finite = finite("target_se")
    source_full_finite = finite("source_full")
    source_train_finite = finite("source_train")
    target_held_out = merged["target_mask"].eq(1)
    source_observed = merged["source_train"].notna()
    source_injected = merged["source_mask"].eq(1)
    source_natural = merged["source_full"].isna()
    invalid_target_mask = (
        target_mask_missing | ~merged["target_mask"].isin({0, 1})
    )
    invalid_source_mask = pd.Series(False, index=merged.index)
    inconsistent_source_state = pd.Series(False, index=merged.index)
    if source != target:
        invalid_source_mask = (
            source_mask_missing | ~merged["source_mask"].isin({0, 1})
        )
        inconsistent_source_state = target_held_out & (
            (source_observed & source_injected)
            | (source_observed & source_natural)
            | (~source_observed & source_injected & source_natural)
            | (
                ~source_observed
                & merged["source_mask"].eq(0)
                & ~source_natural
            )
        )
    nonfinite_target = target_held_out & (
        (~truth_missing & ~truth_finite)
        | (~target_se_missing & ~target_se_finite)
    )
    nonfinite_source = pd.Series(False, index=merged.index)
    if source != target:
        nonfinite_source = target_held_out & (
            (~source_full_missing & ~source_full_finite)
            | (~source_train_missing & ~source_train_finite)
        )

    if source == target:
        merged["task"] = np.where(target_held_out, "W", "not_scored")
        merged["b0_subtype"] = "not_applicable"
    else:
        merged["task"] = np.select(
            [
                target_held_out & source_observed,
                target_held_out & ~source_observed & source_injected,
                target_held_out & ~source_observed & source_natural,
            ],
            ["B1", "B0", "B0"],
            default="not_scored",
        )
        merged["b0_subtype"] = np.select(
            [
                merged["task"].eq("B1"),
                target_held_out & ~source_observed & source_injected,
                target_held_out & ~source_observed & source_natural,
            ],
            ["not_applicable", "injected_source", "natural_source"],
            default="not_applicable",
        )
    merged["variant_class"] = classify_variant(merged[key])

    numeric_prediction = pd.to_numeric(merged["prediction"], errors="coerce")
    prediction_finite = pd.Series(
        np.isfinite(numeric_prediction.to_numpy(dtype=float, na_value=np.nan)),
        index=merged.index,
    )
    task_supported = merged["task"].isin(supported_tasks)
    valid_task = merged["task"].isin({"B1", "B0", "W"})
    no_truth = target_held_out & truth_missing
    no_target_se = target_held_out & truth_finite & target_se_missing
    base_eligible = (
        target_held_out & truth_finite & target_se_finite
    )
    unclassified_source_state = base_eligible & ~valid_task
    unsupported_task = base_eligible & valid_task & ~task_supported
    missing_supported_predictions = base_eligible & task_supported & ~prediction_finite
    invalid_scientific_input = (
        nonfinite_target
        | nonfinite_source
        | invalid_target_mask
        | invalid_source_mask
        | inconsistent_source_state
    )
    eligible = (
        base_eligible
        & task_supported
        & prediction_finite
        & ~invalid_scientific_input
    )

    events = merged.loc[
        eligible,
        [
            key,
            "task",
            "b0_subtype",
            "variant_class",
            "truth",
            "target_se",
            "prediction",
        ],
    ].copy()
    events["error"] = events["prediction"] - events["truth"]
    events["error2"] = events["error"] ** 2
    events["target_sigma2"] = events["target_se"] ** 2
    events["truth2"] = events["truth"] ** 2
    events["prediction2"] = events["prediction"] ** 2
    events["truth_prediction"] = events["truth"] * events["prediction"]

    validation = {
        "rows": len(prediction_df),
        "unique_keys": int(prediction_df[key].nunique()),
        "duplicate_keys": 0,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "expected_target_masked": int(target_held_out.sum()),
        "prediction_present": int(prediction_finite.sum()),
        "eligible_scored": int(eligible.sum()),
        "excluded_no_truth": int(no_truth.sum()),
        "excluded_no_target_se": int(no_target_se.sum()),
        "excluded_nonfinite_target": int(nonfinite_target.sum()),
        "excluded_no_prediction": int(missing_supported_predictions.sum()),
        "unsupported_task_events": int(unsupported_task.sum()),
        "unclassified_source_state": int(unclassified_source_state.sum()),
        "inconsistent_source_state": int(
            inconsistent_source_state.sum()
        ),
        "nonfinite_source_state": int(nonfinite_source.sum()),
        "invalid_mask_values": int(
            invalid_target_mask.sum() + invalid_source_mask.sum()
        ),
        "validation_status": (
            "failed"
            if (
                missing_supported_predictions.any()
                or unclassified_source_state.any()
                or nonfinite_target.any()
                or nonfinite_source.any()
                or invalid_target_mask.any()
                or invalid_source_mask.any()
                or inconsistent_source_state.any()
            )
            else "ok"
        ),
    }
    return events.reset_index(drop=True), validation


def reduce_events(
    events: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """Reduce eligible events to additive sufficient statistics."""
    additive = events.assign(
        N=1,
        sum_error=events["error"],
        SSE=events["error2"],
        Q=events["target_sigma2"],
        sum_truth=events["truth"],
        sum_prediction=events["prediction"],
        sum_truth2=events["truth2"],
        sum_prediction2=events["prediction2"],
        sum_truth_prediction=events["truth_prediction"],
    )
    reduced = (
        additive.groupby(group_columns, as_index=False, dropna=False)[
            list(ADDITIVE_COLUMNS)
        ]
        .sum()
        .sort_values(group_columns, kind="stable")
        .reset_index(drop=True)
    )
    reduced["N"] = reduced["N"].astype(int)
    return reduced
