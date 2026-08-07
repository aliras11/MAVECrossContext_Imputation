"""Unpaired split-level inference for headline and baseline comparisons."""

from collections.abc import Collection
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from mave_statistics.constants import MODEL_DISPLAY_NAMES


HEADLINE_COLUMNS = [
    "dataset", "loss_type", "rate",
    "model_a", "method_a", "model_b", "method_b",
    "n_splits_a", "n_splits_b",
    "mean_rmse_a", "mean_rmse_b", "mean_diff_a_minus_b",
    "median_rmse_a", "median_rmse_b", "median_diff_a_minus_b",
    "U", "p_raw", "p_bonferroni", "significant_0_05",
    "better_model_by_mean", "analysis_unit", "correction_method",
    "correction_family", "family_size",
]

BASELINE_COLUMNS = [
    "dataset", "loss_type", "rate",
    "model", "method", "baseline_model", "baseline_method",
    "n_splits_model", "n_splits_baseline",
    "mean_rmse_model", "mean_rmse_baseline",
    "mean_diff_model_minus_baseline",
    "median_rmse_model", "median_rmse_baseline",
    "median_diff_model_minus_baseline", "pct_improvement",
    "U", "p_raw", "p_bonferroni", "significant_0_05",
    "better_than_baseline", "analysis_unit", "correction_method",
    "correction_family", "family_size",
]

CONTEXT_COLUMNS = [
    "dataset", "loss_type", "rate",
    "context_type", "src", "tgt", "context",
    "model_a", "method_a", "model_b", "method_b",
    "expected_splits", "n_splits_a", "n_splits_b",
    "split_completeness_a", "split_completeness_b",
    "mean_rmse_a", "mean_rmse_b", "mean_diff_a_minus_b",
    "median_rmse_a", "median_rmse_b", "median_diff_a_minus_b",
    "U", "p_raw",
    "p_bonferroni_within_context", "p_bonferroni_ratewide",
    "significant_within_context_0_05",
    "significant_ratewide_0_05",
    "better_model_by_mean", "correction_method",
    "within_context_family_size", "ratewide_family_size",
    "test_status",
]

ANALYSIS_UNIT = "cell_pooled_rmse_per_split"
CORRECTION_METHOD = "bonferroni"


def bonferroni(raw_p: pd.Series, family_size: int) -> pd.Series:
    """Adjust p-values by an explicit Bonferroni family size."""
    adjusted = np.minimum(1.0, raw_p.astype(float) * family_size)
    return pd.Series(adjusted, index=raw_p.index, dtype=float)


def _filtered(
    pooled: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    rate: int,
) -> pd.DataFrame:
    return pooled.loc[
        (pooled["dataset"] == dataset)
        & (pooled["loss_type"] == loss_type)
        & (pooled["rate"] == rate)
    ].copy()


def _reject_duplicate_model_splits(frame: pd.DataFrame) -> None:
    if frame.duplicated(["model", "split"], keep=False).any():
        raise ValueError("duplicate model × split rows in filtered pooled data")


def _rmse_array(frame: pd.DataFrame, model: str) -> np.ndarray:
    values = frame.loc[frame["model"] == model, "rmse"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite rmse values for model {model!r}")
    return values


def _finish_pvalues(result: pd.DataFrame, family_size: int) -> None:
    result["p_bonferroni"] = bonferroni(result["p_raw"], family_size)
    result["significant_0_05"] = result["p_bonferroni"] < 0.05

    raw = result["p_raw"].to_numpy(dtype=float)
    corrected = result["p_bonferroni"].to_numpy(dtype=float)
    assert np.isfinite(raw).all()
    assert np.isfinite(corrected).all()
    assert ((0.0 <= raw) & (raw <= 1.0)).all()
    assert ((0.0 <= corrected) & (corrected <= 1.0)).all()
    assert (corrected >= raw).all()


def headline_pairwise_mwu(
    pooled: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    rate: int,
    excluded_models: Collection[str] = (),
) -> pd.DataFrame:
    """Compare every eligible model pair using unpaired split RMSE arrays."""
    selected = _filtered(
        pooled, dataset=dataset, loss_type=loss_type, rate=rate
    )
    selected = selected.loc[
        ~selected["model"].isin(set(excluded_models))
    ].copy()
    _reject_duplicate_model_splits(selected)

    models = sorted(selected["model"].unique())
    family_size = len(models) * (len(models) - 1) // 2
    correction_family = (
        f"{dataset}:{loss_type}:rate={rate}:headline_model_pairs"
    )
    rows = []
    for model_a, model_b in combinations(models, 2):
        x = _rmse_array(selected, model_a)
        y = _rmse_array(selected, model_b)
        test = mannwhitneyu(x, y, alternative="two-sided")
        mean_a = float(np.mean(x))
        mean_b = float(np.mean(y))
        median_a = float(np.median(x))
        median_b = float(np.median(y))
        mean_diff = mean_a - mean_b
        median_diff = median_a - median_b
        better = (
            model_a
            if mean_diff < 0
            else model_b
            if mean_diff > 0
            else "tie"
        )
        rows.append({
            "dataset": dataset,
            "loss_type": loss_type,
            "rate": rate,
            "model_a": model_a,
            "method_a": MODEL_DISPLAY_NAMES.get(model_a, model_a),
            "model_b": model_b,
            "method_b": MODEL_DISPLAY_NAMES.get(model_b, model_b),
            "n_splits_a": len(x),
            "n_splits_b": len(y),
            "mean_rmse_a": mean_a,
            "mean_rmse_b": mean_b,
            "mean_diff_a_minus_b": mean_diff,
            "median_rmse_a": median_a,
            "median_rmse_b": median_b,
            "median_diff_a_minus_b": median_diff,
            "U": float(test.statistic),
            "p_raw": float(test.pvalue),
            "better_model_by_mean": better,
            "analysis_unit": ANALYSIS_UNIT,
            "correction_method": CORRECTION_METHOD,
            "correction_family": correction_family,
            "family_size": family_size,
        })

    if not rows:
        return pd.DataFrame(columns=HEADLINE_COLUMNS)
    result = pd.DataFrame(rows)
    _finish_pvalues(result, family_size)
    return result[HEADLINE_COLUMNS]


def baseline_mwu(
    model_pooled: pd.DataFrame,
    baseline_pooled: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    rate: int,
    baseline_model: str = "col_mean",
) -> pd.DataFrame:
    """Compare selected models with regular within-map Column Mean splits."""
    selected = _filtered(
        model_pooled, dataset=dataset, loss_type=loss_type, rate=rate
    )
    selected = selected.loc[selected["model"] != baseline_model].copy()
    reference = _filtered(
        baseline_pooled,
        dataset="regular",
        loss_type="within_map",
        rate=rate,
    )
    reference = reference.loc[
        reference["model"] == baseline_model
    ].copy()
    _reject_duplicate_model_splits(selected)
    _reject_duplicate_model_splits(reference)

    baseline_values = _rmse_array(reference, baseline_model)
    if len(baseline_values) == 0:
        raise ValueError(
            "baseline pooled data has no regular within_map reference rows"
        )

    models = sorted(selected["model"].unique())
    family_size = len(models)
    correction_family = (
        f"{dataset}:{loss_type}:rate={rate}:models_vs_{baseline_model}"
    )
    baseline_mean = float(np.mean(baseline_values))
    baseline_median = float(np.median(baseline_values))
    rows = []
    for model in models:
        values = _rmse_array(selected, model)
        test = mannwhitneyu(
            values, baseline_values, alternative="two-sided"
        )
        model_mean = float(np.mean(values))
        model_median = float(np.median(values))
        mean_diff = model_mean - baseline_mean
        median_diff = model_median - baseline_median
        pct_improvement = (
            100.0 * (baseline_median - model_median) / baseline_median
        )
        rows.append({
            "dataset": dataset,
            "loss_type": loss_type,
            "rate": rate,
            "model": model,
            "method": MODEL_DISPLAY_NAMES.get(model, model),
            "baseline_model": baseline_model,
            "baseline_method": MODEL_DISPLAY_NAMES.get(
                baseline_model, baseline_model
            ),
            "n_splits_model": len(values),
            "n_splits_baseline": len(baseline_values),
            "mean_rmse_model": model_mean,
            "mean_rmse_baseline": baseline_mean,
            "mean_diff_model_minus_baseline": mean_diff,
            "median_rmse_model": model_median,
            "median_rmse_baseline": baseline_median,
            "median_diff_model_minus_baseline": median_diff,
            "pct_improvement": pct_improvement,
            "U": float(test.statistic),
            "p_raw": float(test.pvalue),
            "better_than_baseline": median_diff < 0,
            "analysis_unit": ANALYSIS_UNIT,
            "correction_method": CORRECTION_METHOD,
            "correction_family": correction_family,
            "family_size": family_size,
        })

    if not rows:
        return pd.DataFrame(columns=BASELINE_COLUMNS)
    result = pd.DataFrame(rows)
    _finish_pvalues(result, family_size)
    return result[BASELINE_COLUMNS]


def baseline_view_from_headline(
    pairwise: pd.DataFrame, *, baseline_model: str = "col_mean"
) -> pd.DataFrame:
    """Project unified all-method comparisons onto Column Mean comparisons."""
    rows = []
    for row in pairwise.itertuples(index=False):
        if baseline_model not in (row.model_a, row.model_b):
            continue
        model_is_a = row.model_a != baseline_model
        model = row.model_a if model_is_a else row.model_b
        method = row.method_a if model_is_a else row.method_b
        baseline_method = row.method_b if model_is_a else row.method_a
        mean_model = row.mean_rmse_a if model_is_a else row.mean_rmse_b
        mean_baseline = row.mean_rmse_b if model_is_a else row.mean_rmse_a
        median_model = row.median_rmse_a if model_is_a else row.median_rmse_b
        median_baseline = row.median_rmse_b if model_is_a else row.median_rmse_a
        rows.append({
            "dataset": row.dataset, "loss_type": row.loss_type, "rate": row.rate,
            "model": model, "method": method, "baseline_model": baseline_model,
            "baseline_method": baseline_method,
            "n_splits_model": row.n_splits_a if model_is_a else row.n_splits_b,
            "n_splits_baseline": row.n_splits_b if model_is_a else row.n_splits_a,
            "mean_rmse_model": mean_model, "mean_rmse_baseline": mean_baseline,
            "mean_diff_model_minus_baseline": mean_model - mean_baseline,
            "median_rmse_model": median_model, "median_rmse_baseline": median_baseline,
            "median_diff_model_minus_baseline": median_model - median_baseline,
            "pct_improvement": 100 * (median_baseline - median_model) / median_baseline,
            "U": row.U, "p_raw": row.p_raw, "p_bonferroni": row.p_bonferroni,
            "significant_0_05": row.significant_0_05,
            "better_than_baseline": median_model < median_baseline,
            "analysis_unit": row.analysis_unit, "correction_method": row.correction_method,
            "correction_family": row.correction_family, "family_size": row.family_size,
        })
    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


def _reject_duplicate_context_model_splits(
    frame: pd.DataFrame,
    *,
    loss_type: str,
) -> None:
    context_columns = ["tgt"] if loss_type == "within_map" else ["src", "tgt"]
    duplicate_columns = ["model", *context_columns, "split"]
    if frame.duplicated(duplicate_columns, keep=False).any():
        raise ValueError("duplicate model × split rows in context data")


def _context_descriptors(
    frame: pd.DataFrame,
    *,
    loss_type: str,
) -> list[dict[str, object]]:
    if loss_type == "within_map":
        targets = sorted(frame["tgt"].drop_duplicates())
        return [
            {
                "context_type": "target_map",
                "src": target,
                "tgt": target,
                "context": target,
            }
            for target in targets
        ]

    contexts = (
        frame[["src", "tgt"]]
        .drop_duplicates()
        .sort_values(["src", "tgt"])
    )
    return [
        {
            "context_type": "source_target",
            "src": row.src,
            "tgt": row.tgt,
            "context": f"{row.src}->{row.tgt}",
        }
        for row in contexts.itertuples(index=False)
    ]


def _context_sample(
    frame: pd.DataFrame,
    *,
    descriptor: dict[str, object],
    model: str,
) -> pd.DataFrame:
    selected = frame.loc[frame["model"] == model]
    if descriptor["context_type"] == "target_map":
        return selected.loc[selected["tgt"] == descriptor["tgt"]]
    return selected.loc[
        (selected["src"] == descriptor["src"])
        & (selected["tgt"] == descriptor["tgt"])
    ]


def _context_rmse_array(sample: pd.DataFrame, model: str) -> np.ndarray:
    values = sample["rmse"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite rmse values for model {model!r}")
    return values


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else np.nan


def _median_or_nan(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else np.nan


def _better_model(
    model_a: str,
    model_b: str,
    mean_diff: float,
) -> object:
    if not np.isfinite(mean_diff):
        return pd.NA
    if mean_diff < 0:
        return model_a
    if mean_diff > 0:
        return model_b
    return "tie"


def _finish_context_pvalues(result: pd.DataFrame) -> None:
    result["p_bonferroni_within_context"] = np.nan
    result["p_bonferroni_ratewide"] = np.nan
    result["within_context_family_size"] = 0

    context_columns = ["context_type", "src", "tgt", "context"]
    for _, group in result.groupby(context_columns, sort=False, dropna=False):
        valid_index = group.index[group["test_status"] == "ok"]
        family_size = len(valid_index)
        result.loc[group.index, "within_context_family_size"] = family_size
        if family_size:
            result.loc[
                valid_index, "p_bonferroni_within_context"
            ] = bonferroni(
                result.loc[valid_index, "p_raw"],
                family_size,
            )

    valid_index = result.index[result["test_status"] == "ok"]
    ratewide_family_size = len(valid_index)
    result["ratewide_family_size"] = ratewide_family_size
    if ratewide_family_size:
        result.loc[valid_index, "p_bonferroni_ratewide"] = bonferroni(
            result.loc[valid_index, "p_raw"],
            ratewide_family_size,
        )

    result["significant_within_context_0_05"] = (
        result["p_bonferroni_within_context"] < 0.05
    )
    result["significant_ratewide_0_05"] = (
        result["p_bonferroni_ratewide"] < 0.05
    )

    valid_pvalues = result.loc[
        valid_index,
        [
            "p_raw",
            "p_bonferroni_within_context",
            "p_bonferroni_ratewide",
        ],
    ]
    values = valid_pvalues.to_numpy(dtype=float)
    assert np.isfinite(values).all()
    assert ((0.0 <= values) & (values <= 1.0)).all()
    assert (
        valid_pvalues["p_bonferroni_within_context"]
        >= valid_pvalues["p_raw"]
    ).all()
    assert (
        valid_pvalues["p_bonferroni_ratewide"]
        >= valid_pvalues["p_raw"]
    ).all()


def context_pairwise_mwu(
    raw: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    rate: int,
    expected_splits: int,
    requested_models: Collection[str],
    excluded_models: Collection[str] = (),
    minimum_completeness: float = 0.95,
) -> pd.DataFrame:
    """Compare requested model pairs within every observed context."""
    excluded = set(excluded_models)
    models = sorted(set(requested_models) - excluded)
    selected = _filtered(
        raw, dataset=dataset, loss_type=loss_type, rate=rate
    )
    selected = selected.loc[
        selected["model"].isin(models)
    ].copy()
    _reject_duplicate_context_model_splits(
        selected, loss_type=loss_type
    )

    descriptors = _context_descriptors(selected, loss_type=loss_type)
    rows = []
    for descriptor in descriptors:
        for model_a, model_b in combinations(models, 2):
            sample_a = _context_sample(
                selected, descriptor=descriptor, model=model_a
            )
            sample_b = _context_sample(
                selected, descriptor=descriptor, model=model_b
            )
            x = _context_rmse_array(sample_a, model_a)
            y = _context_rmse_array(sample_b, model_b)
            n_a = int(sample_a["split"].nunique())
            n_b = int(sample_b["split"].nunique())
            completeness_a = n_a / expected_splits
            completeness_b = n_b / expected_splits
            mean_a = _mean_or_nan(x)
            mean_b = _mean_or_nan(y)
            median_a = _median_or_nan(x)
            median_b = _median_or_nan(y)
            mean_diff = mean_a - mean_b
            median_diff = median_a - median_b
            testable = (
                completeness_a >= minimum_completeness
                and completeness_b >= minimum_completeness
            )
            if testable:
                test = mannwhitneyu(x, y, alternative="two-sided")
                statistic = float(test.statistic)
                p_raw = float(test.pvalue)
                test_status = "ok"
            else:
                statistic = np.nan
                p_raw = np.nan
                test_status = "insufficient_context_completeness"

            rows.append({
                "dataset": dataset,
                "loss_type": loss_type,
                "rate": rate,
                **descriptor,
                "model_a": model_a,
                "method_a": MODEL_DISPLAY_NAMES.get(model_a, model_a),
                "model_b": model_b,
                "method_b": MODEL_DISPLAY_NAMES.get(model_b, model_b),
                "expected_splits": expected_splits,
                "n_splits_a": n_a,
                "n_splits_b": n_b,
                "split_completeness_a": completeness_a,
                "split_completeness_b": completeness_b,
                "mean_rmse_a": mean_a,
                "mean_rmse_b": mean_b,
                "mean_diff_a_minus_b": mean_diff,
                "median_rmse_a": median_a,
                "median_rmse_b": median_b,
                "median_diff_a_minus_b": median_diff,
                "U": statistic,
                "p_raw": p_raw,
                "better_model_by_mean": _better_model(
                    model_a, model_b, mean_diff
                ),
                "correction_method": CORRECTION_METHOD,
                "test_status": test_status,
            })

    if not rows:
        return pd.DataFrame(columns=CONTEXT_COLUMNS)
    result = pd.DataFrame(rows)
    _finish_context_pvalues(result)
    return result[CONTEXT_COLUMNS]
