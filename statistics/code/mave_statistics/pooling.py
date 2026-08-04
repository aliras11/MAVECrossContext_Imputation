"""Pool normalized RMSE results and audit model-rate completeness."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


POOL_KEYS = ["dataset", "loss_type", "model", "rate", "split"]
AUDIT_KEYS = ["dataset", "loss_type", "model", "rate"]


def pool_rmse_by_split(df: pd.DataFrame) -> pd.DataFrame:
    """Pool RMSE over all prediction cells within each model-rate-split."""
    work = df.copy()
    if not np.isfinite(work["rmse"]).all():
        raise ValueError("cannot pool non-finite rmse")
    if not np.isfinite(work["n_points"]).all():
        raise ValueError("cannot pool non-finite n_points")
    if (work["n_points"] <= 0).any():
        raise ValueError("cannot pool non-positive n_points")
    work["sse"] = work["n_points"] * work["rmse"].pow(2)
    pooled = (
        work.groupby(POOL_KEYS, as_index=False, sort=True)
        .agg(sse=("sse", "sum"), n_points=("n_points", "sum"))
    )
    pooled["rmse"] = np.sqrt(pooled["sse"] / pooled["n_points"])
    return pooled[POOL_KEYS + ["rmse", "n_points", "sse"]]


def model_rate_completeness(
    df: pd.DataFrame,
    *,
    rates: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Audit observed context-split rows against the fullest model at each rate."""
    selected_rates = list(df["rate"].unique() if rates is None else rates)
    selected_models = list(df["model"].unique() if models is None else models)
    strata = df[["dataset", "loss_type"]].drop_duplicates()
    requested = (
        strata.assign(_key=1)
        .merge(pd.DataFrame({"model": selected_models, "_key": 1}), on="_key")
        .merge(pd.DataFrame({"rate": selected_rates, "_key": 1}), on="_key")
        .drop(columns="_key")
    )
    observed = (
        df.groupby(AUDIT_KEYS, as_index=False, sort=True)
        .size()
        .rename(columns={"size": "observed_rows"})
    )
    audit = requested.merge(observed, on=AUDIT_KEYS, how="left")
    audit["observed_rows"] = audit["observed_rows"].fillna(0).astype(int)
    expected = (
        observed.groupby(["dataset", "loss_type", "rate"], as_index=False)
        .agg(expected_rows=("observed_rows", "max"))
    )
    audit = audit.merge(
        expected, on=["dataset", "loss_type", "rate"], how="left"
    )
    audit["expected_rows"] = audit["expected_rows"].fillna(0).astype(int)
    audit["completeness_fraction"] = np.divide(
        audit["observed_rows"],
        audit["expected_rows"],
        out=np.zeros(len(audit), dtype=float),
        where=audit["expected_rows"].to_numpy() > 0,
    )
    return audit[
        AUDIT_KEYS + ["observed_rows", "expected_rows", "completeness_fraction"]
    ]


def filter_incomplete_model_rates(
    df: pd.DataFrame,
    *,
    minimum_fraction: float = 0.95,
    rates: Sequence[int] | None = None,
    models: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return original rows only for model-rate strata meeting completeness."""
    audit = model_rate_completeness(df, rates=rates, models=models)
    included = audit.loc[
        audit["completeness_fraction"] >= minimum_fraction, AUDIT_KEYS
    ]
    return df.merge(included, on=AUDIT_KEYS, how="inner"), audit
