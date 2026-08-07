"""Orchestrate normalized inputs into the complete statistics inventory."""

from collections.abc import Collection
from pathlib import Path

import pandas as pd

from .constants import B0_MODELS, B1_MODELS, NODOUBLE_RATES, W_MODELS
from .inference import (
    baseline_view_from_headline,
    context_pairwise_mwu,
    headline_pairwise_mwu,
)
from .loaders import load_main_results, load_nodouble_results
from .pooling import filter_incomplete_model_rates, pool_rmse_by_split
from .reporting import summarize_pooled, write_statistics_tables


def _drop_extra_pca_models(frame: pd.DataFrame) -> pd.DataFrame:
    model = frame["model"].astype(str)
    keep = ~model.str.startswith("pca_k") | model.eq("pca_k1")
    return frame.loc[keep].copy()


def _family(
    frame: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    models: tuple[str, ...],
) -> pd.DataFrame:
    return frame.loc[
        (frame["dataset"] == dataset)
        & (frame["loss_type"] == loss_type)
        & frame["model"].isin(models)
    ].copy()


def _rates(frame: pd.DataFrame) -> list[int]:
    return sorted(int(rate) for rate in frame["rate"].unique())


def _validate_output_location(
    main_results_dir: Path,
    nodouble_results_dir: Path,
    output_dir: Path,
) -> None:
    output = Path(output_dir).resolve()
    raw_directories = (
        Path(main_results_dir).resolve(),
        Path(nodouble_results_dir).resolve(),
    )
    if any(
        output == raw_directory or output in raw_directory.parents
        for raw_directory in raw_directories
    ):
        raise ValueError(
            "output directory cannot be a raw results directory or its parent"
        )


def _headline_tables(
    pooled: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    excluded_models: tuple[str, ...] = (),
) -> pd.DataFrame:
    return pd.concat([
        headline_pairwise_mwu(
            pooled,
            dataset=dataset,
            loss_type=loss_type,
            rate=rate,
            excluded_models=excluded_models,
        )
        for rate in _rates(pooled)
    ], ignore_index=True)


def _baseline_tables(
    pooled: pd.DataFrame,
    baseline_pooled: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
) -> pd.DataFrame:
    return pd.concat([
        baseline_mwu(
            pooled,
            baseline_pooled,
            dataset=dataset,
            loss_type=loss_type,
            rate=rate,
        )
        for rate in _rates(pooled)
    ], ignore_index=True)


def _context_tables(
    raw: pd.DataFrame,
    *,
    dataset: str,
    loss_type: str,
    expected_splits: int,
    minimum_completeness: float,
    requested_models: Collection[str],
    excluded_models: tuple[str, ...] = (),
) -> pd.DataFrame:
    return pd.concat([
        context_pairwise_mwu(
            raw,
            dataset=dataset,
            loss_type=loss_type,
            rate=rate,
            expected_splits=expected_splits,
            requested_models=requested_models,
            excluded_models=excluded_models,
            minimum_completeness=minimum_completeness,
        )
        for rate in _rates(raw)
    ], ignore_index=True)


def _nodouble_context_tables(
    raw: pd.DataFrame,
    *,
    expected_splits: int,
    minimum_completeness: float,
) -> pd.DataFrame:
    rates = _rates(raw)
    return pd.concat([
        context_pairwise_mwu(
            raw,
            dataset="no_double",
            loss_type="regression_test",
            rate=rate,
            expected_splits=expected_splits,
            requested_models=tuple(sorted(
                raw.loc[raw["rate"] == rate, "model"].unique()
            )),
            minimum_completeness=minimum_completeness,
        )
        for rate in rates
    ], ignore_index=True)


def build_statistics_tables(
    main_df: pd.DataFrame,
    no_double_df: pd.DataFrame,
    *,
    expected_splits: int = 50,
    minimum_completeness: float = 0.95,
) -> dict[str, pd.DataFrame]:
    """Build every summary, headline, baseline, context, and audit table."""
    main = _drop_extra_pca_models(main_df)
    no_double = _drop_extra_pca_models(no_double_df)

    b1_raw = _family(
        main,
        dataset="regular",
        loss_type="regression_test",
        models=B1_MODELS,
    )
    b0_raw = _family(
        main,
        dataset="regular",
        loss_type="double_missing",
        models=B0_MODELS,
    )
    w_raw = _family(
        main,
        dataset="regular",
        loss_type="within_map",
        models=W_MODELS,
    )
    no_double_raw = _family(
        no_double,
        dataset="no_double",
        loss_type="regression_test",
        models=B1_MODELS,
    )
    filtered_no_double_raw, no_double_audit = filter_incomplete_model_rates(
        no_double_raw,
        rates=NODOUBLE_RATES,
        models=B1_MODELS,
        minimum_fraction=minimum_completeness,
    )

    b1_pooled = pool_rmse_by_split(b1_raw)
    b0_pooled = pool_rmse_by_split(b0_raw)
    w_pooled = pool_rmse_by_split(w_raw)
    w_headline_pooled = w_pooled
    no_double_pooled = pool_rmse_by_split(filtered_no_double_raw)

    tables = {
        "nodouble_model_rate_completeness.csv": no_double_audit,
        "pairwise_mwu_regression_test.csv": _headline_tables(
            b1_pooled,
            dataset="regular",
            loss_type="regression_test",
        ),
        "pairwise_mwu_double_missing.csv": _headline_tables(
            b0_pooled,
            dataset="regular",
            loss_type="double_missing",
        ),
        "pairwise_mwu_within_map.csv": _headline_tables(
            w_headline_pooled,
            dataset="regular",
            loss_type="within_map",
        ),
        "pairwise_mwu_nodouble_regression_test.csv": _headline_tables(
            no_double_pooled,
            dataset="no_double",
            loss_type="regression_test",
        ),
        "rmse_summary_regression_test.csv": summarize_pooled(b1_pooled),
        "rmse_summary_double_missing.csv": summarize_pooled(b0_pooled),
        "rmse_summary_within_map.csv": summarize_pooled(w_pooled),
        "rmse_summary_nodouble_regression_test.csv": summarize_pooled(
            no_double_pooled
        ),
        "pairwise_mwu_by_context_regression_test.csv": _context_tables(
            b1_raw,
            dataset="regular",
            loss_type="regression_test",
            expected_splits=expected_splits,
            minimum_completeness=minimum_completeness,
            requested_models=B1_MODELS,
        ),
        "pairwise_mwu_by_context_double_missing.csv": _context_tables(
            b0_raw,
            dataset="regular",
            loss_type="double_missing",
            expected_splits=expected_splits,
            minimum_completeness=minimum_completeness,
            requested_models=B0_MODELS,
        ),
        "pairwise_mwu_by_context_within_map.csv": _context_tables(
            w_raw,
            dataset="regular",
            loss_type="within_map",
            expected_splits=expected_splits,
            minimum_completeness=minimum_completeness,
            requested_models=W_MODELS,
        ),
        "pairwise_mwu_by_context_nodouble_regression_test.csv": (
            _nodouble_context_tables(
                filtered_no_double_raw,
                expected_splits=expected_splits,
                minimum_completeness=minimum_completeness,
            )
        ),
    }
    tables["vs_colmean_regression_test.csv"] = baseline_view_from_headline(tables["pairwise_mwu_regression_test.csv"])
    tables["vs_colmean_double_missing.csv"] = baseline_view_from_headline(tables["pairwise_mwu_double_missing.csv"])
    tables["vs_colmean_within_map.csv"] = baseline_view_from_headline(tables["pairwise_mwu_within_map.csv"])
    return tables


def generate_statistics(
    main_results_dir: Path,
    nodouble_results_dir: Path,
    output_dir: Path,
    *,
    expected_splits: int = 50,
    minimum_completeness: float = 0.95,
) -> dict[str, Path]:
    """Load result sources, build statistics, and publish validated CSVs."""
    _validate_output_location(
        main_results_dir,
        nodouble_results_dir,
        output_dir,
    )
    tables = build_statistics_tables(
        load_main_results(main_results_dir),
        load_nodouble_results(nodouble_results_dir),
        expected_splits=expected_splits,
        minimum_completeness=minimum_completeness,
    )
    return write_statistics_tables(tables, output_dir)
