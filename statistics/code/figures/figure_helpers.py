"""Presentation helpers shared by the manuscript figure modules.

Scientific loading, completeness checks, and RMSE pooling are owned by the
``mave_statistics`` package.  This module re-exports those canonical
operations and keeps only figure-facing labels, styles, display summaries,
and output helpers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mave_statistics.constants import (
    MODEL_DISPLAY_NAMES,
    NODOUBLE_RATES as CANONICAL_NODOUBLE_RATES,
    REGULAR_RATES,
)
from mave_statistics.loaders import load_main_results, load_nodouble_results
from mave_statistics.pooling import (
    filter_incomplete_model_rates,
    model_rate_completeness,
    pool_rmse_by_split,
)


MODEL_COLORS = {
    "SingleAE": "#4477AA",
    "DualAE": "#0077BB",
    "MICE-PMM": "#EE6677",
    "MICE-RF": "#CC3311",
    "Basic Linear": "#CCBB44",
    "1-Param Nonlinear": "#EE7733",
    "Linear + Domain": "#228833",
    "Mixed (rand. int.)": "#AA3377",
    "Mixed (rand. slope)": "#66CCEE",
    "Column Mean": "#BBBBBB",
    "$k$NN-BLOSUM": "#000000",
    "PCA (k = 1)": "#33BBEE",
}

MODEL_MARKERS = {
    "SingleAE": "o",
    "DualAE": "s",
    "MICE-PMM": "^",
    "MICE-RF": "v",
    "Basic Linear": "d",
    "1-Param Nonlinear": "D",
    "Linear + Domain": "P",
    "Mixed (rand. int.)": "p",
    "Mixed (rand. slope)": "h",
    "Column Mean": "X",
    "$k$NN-BLOSUM": "*",
    "PCA (k = 1)": "<",
}

LOSS_TYPE_LABELS = {
    "regression_test": r"Between-Map: Source-Informed ($B_1$)",
    "double_missing": r"Between-Map: Missing-Source ($B_0$)",
    "within_map": "Within-Map (W)",
}

LOSS_TYPE_LABELS_SHORT = {
    "regression_test": r"$B_1$",
    "double_missing": r"$B_0$",
    "within_map": "W",
}

RATES = list(REGULAR_RATES)
NODOUBLE_RATES = list(CANONICAL_NODOUBLE_RATES)

BAR_SUMMARY_COLUMNS = {
    "model",
    "method",
    "rate",
    "mean_rmse",
    "se_rmse",
}
BAR_PAIRWISE_COLUMNS = {
    "rate",
    "model_a",
    "model_b",
    "p_bonferroni",
    "significant_0_05",
}
PERCENT_PANEL_COLUMNS = {
    "model",
    "method",
    "rate",
    "pct_improvement",
    "p_bonferroni",
    "significant_0_05",
}


def _load_statistics_csv(
    statistics_dir: Path,
    filename: str,
    required_columns: set[str],
) -> pd.DataFrame:
    path = Path(statistics_dir) / filename
    frame = pd.read_csv(path)
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )
    return frame


def load_summary_panel_data(
    statistics_dir: Path,
    summary_filename: str,
) -> pd.DataFrame:
    """Load one generated long-form RMSE summary for figure presentation."""
    return _load_statistics_csv(
        statistics_dir,
        summary_filename,
        BAR_SUMMARY_COLUMNS,
    )


def load_bar_panel_data(
    statistics_dir: Path,
    summary_filename: str,
    pairwise_filename: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load generated bar heights, errors, and corrected comparisons."""
    summary = load_summary_panel_data(statistics_dir, summary_filename)
    pairwise = _load_statistics_csv(
        statistics_dir,
        pairwise_filename,
        BAR_PAIRWISE_COLUMNS,
    )
    return summary, pairwise


def best_model_is_significant(
    pairwise: pd.DataFrame,
    *,
    rate: int,
    best_model: str,
    displayed_models: set[str],
) -> bool:
    """Return whether the best bar differs from every displayed opponent."""
    rate_rows = pairwise.loc[pairwise["rate"] == rate]
    all_models = set(rate_rows["model_a"]) | set(rate_rows["model_b"])
    if all_models != set(displayed_models):
        return False
    expected_opponents = all_models - {best_model}
    selected = rate_rows.loc[
        (rate_rows["model_a"] == best_model)
        | (rate_rows["model_b"] == best_model)
    ]
    observed_opponents = set(
        selected["model_a"].where(
            selected["model_a"] != best_model,
            selected["model_b"],
        )
    )
    return bool(
        observed_opponents == expected_opponents
        and len(expected_opponents) > 0
        and selected["p_bonferroni"].notna().all()
        and (selected["p_bonferroni"] < 0.05).all()
    )


def load_percent_panel_data(
    statistics_dir: Path,
    baseline_filename: str,
) -> pd.DataFrame:
    """Load generated percent effects and corrected baseline comparisons."""
    return _load_statistics_csv(
        statistics_dir,
        baseline_filename,
        PERCENT_PANEL_COLUMNS,
    )


def apply_style() -> None:
    """Set the established matplotlib/seaborn manuscript style."""
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "axes.labelweight": "bold",
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


def save_figure(
    fig,
    stem: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Save ``fig`` as nonempty SVG and PNG files under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_path = output_dir / f"{stem}.svg"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def get_display_name(model_key: str) -> str:
    """Map an internal model key to its established manuscript label."""
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def get_task_display_name(model_key: str, loss_type: str) -> str:
    """Return the manuscript label, distinguishing the two SingleAE uses."""
    if model_key == "single_ae":
        if loss_type == "within_map":
            return "SingleAE (within-map)"
        if loss_type in {"regression_test", "double_missing"}:
            return "SingleAE (cross-map)"
    return get_display_name(model_key)


def get_color(model_key: str) -> str:
    """Return the established model color, with a neutral fallback."""
    return MODEL_COLORS.get(get_display_name(model_key), "#333333")


def get_marker(model_key: str) -> str:
    """Return the established model marker, with a circle fallback."""
    return MODEL_MARKERS.get(get_display_name(model_key), "o")


def weighted_aggregate_over_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Use the canonical cell-weighted pool for model-rate-split rows."""
    return pool_rmse_by_split(df)


def summary_stats(
    df: pd.DataFrame,
    aggregate_pairs: bool = True,
) -> pd.DataFrame:
    """Build figure-ready mean, SEM, and 95% CI summaries across splits."""
    work = pool_rmse_by_split(df) if aggregate_pairs else df
    stats = (
        work.groupby(["model", "rate", "loss_type"], as_index=False)
        .agg(
            mean=("rmse", "mean"),
            sem=("rmse", lambda values: values.std() / np.sqrt(len(values))),
            n_splits=("rmse", "count"),
        )
    )
    stats["ci95"] = stats["sem"] * 1.96
    return stats


def weighted_average_rmse(df: pd.DataFrame) -> pd.DataFrame:
    """Pool all selected rows into one RMSE per model-rate-split."""
    selected = df.copy()
    selected["loss_type"] = "selected_rows"
    pooled = pool_rmse_by_split(selected)
    return pooled.rename(
        columns={"rmse": "weighted_rmse", "n_points": "total_n_points"}
    )[
        ["model", "rate", "split", "weighted_rmse", "total_n_points"]
    ]


def pareto_frontier(
    points: list[tuple[str, float, float]],
) -> list[tuple[str, float, float]]:
    """Return points that maximize coverage while minimizing risk."""
    nondominated = []
    for index, (model, coverage, risk) in enumerate(points):
        dominated = any(
            other_index != index
            and other_coverage >= coverage
            and other_risk <= risk
            and (other_coverage > coverage or other_risk < risk)
            for other_index, (
                _other_model,
                other_coverage,
                other_risk,
            ) in enumerate(points)
        )
        if not dominated:
            nondominated.append((model, coverage, risk))
    return sorted(nondominated, key=lambda point: (point[1], point[2]))
