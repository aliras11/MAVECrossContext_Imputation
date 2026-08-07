"""fig5c: Risk-Coverage Curves (Selective Prediction View)

A 2x3 small-multiples grid (one panel per saturation level) showing the
coverage-accuracy tradeoff for between-map imputation models. LMMs
structurally cannot predict B_0 cells, so they "abstain" on the
B_0 fraction of the test set; AE, MICE, and Column Mean predict every between-map cell.

Headline: LMMs aren't lower curves than full-coverage methods — they're shorter ones
(truncated coverage). Reference: Geifman & El-Yaniv (2017),
Traub et al. (NeurIPS 2024).
"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def main(
    results_dir: Path,
    nodouble_results_dir: Path,
    statistics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate this module's established figure outputs."""
    fh.apply_style()

    # --- Model family membership ---
    LMM_MODELS = {
        'basic_linear', 'oneparam_linear', 'full_interaction_linear',
        'full_interaction_mixed', 'mixed_random',
    }
    FULL_COVERAGE_MODELS = {'single_ae', 'dual_ae', 'mice', 'mice_rf', 'col_mean'}
    ALL_MODELS = LMM_MODELS | FULL_COVERAGE_MODELS

    # --- Load and filter ---
    df = fh.load_main_results(results_dir)
    df = df[~df['model'].str.startswith('pca_k')]
    between = df[df['loss_type'].isin(['regression_test', 'double_missing'])]

    # --- Compute model-agnostic n_B1 and n_B0 per rate from dual_ae rows ---
    # dual_ae has both loss types; n_points reflects test-cell counts that are
    # determined by the split, not by the model. Sum across pairs and average
    # across splits for a single representative coverage value per rate.
    def compute_b1_share_per_rate(between_df):
        """Return dict: rate -> B_1 share = n_B1 / (n_B1 + n_B0)."""
        ref = between_df[between_df['model'] == 'dual_ae']
        shares = {}
        for rate in fh.RATES:
            sub = ref[ref['rate'] == rate]
            n_splits = sub['split'].nunique()
            n_b1 = sub[sub['loss_type'] == 'regression_test']['n_points'].sum() / n_splits
            n_b0 = sub[sub['loss_type'] == 'double_missing']['n_points'].sum() / n_splits
            shares[rate] = n_b1 / (n_b1 + n_b0)
        return shares

    b1_share = compute_b1_share_per_rate(between)

    # --- Per-(model, rate) coverage and RMSE ---
    def compute_panel_points(between_df, rate, b1_share_at_rate):
        """For one saturation rate, return list of (model, coverage, rmse) tuples."""
        points = []
        for model in sorted(ALL_MODELS):
            if model in LMM_MODELS:
                sub = between_df[
                    (between_df['model'] == model)
                    & (between_df['rate'] == rate)
                    & (between_df['loss_type'] == 'regression_test')
                ]
                coverage = b1_share_at_rate
            else:  # AE / MICE
                sub = between_df[
                    (between_df['model'] == model)
                    & (between_df['rate'] == rate)
                ]
                coverage = 1.0
            if len(sub) == 0:
                continue
            pooled = fh.weighted_average_rmse(sub)
            rmse = pooled['weighted_rmse'].mean()
            points.append((model, coverage, rmse))
        return points

    panel_points = {rate: compute_panel_points(between, rate, b1_share[rate])
                    for rate in fh.RATES}

    # --- Global y-limits with 5% padding ---
    all_rmse = [r for pts in panel_points.values() for (_, _, r) in pts]
    y_min = min(all_rmse)
    y_max = max(all_rmse)
    y_pad = 0.05 * (y_max - y_min)
    y_lim = (y_min - y_pad, y_max + y_pad)

    # --- Plot ---
    PANEL_LAYOUT = [
        [10, 20, 40],
        [60, 80, 90],
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)

    for i, row in enumerate(PANEL_LAYOUT):
        for j, rate in enumerate(row):
            ax = axes[i, j]
            pts = panel_points[rate]

            # Pareto frontier (faint gray)
            frontier = fh.pareto_frontier(pts)
            if len(frontier) >= 2:
                fx = [c for (_, c, _) in frontier]
                fy = [r for (_, _, r) in frontier]
                ax.plot(fx, fy, '-', color='gray', alpha=0.4, linewidth=1.2,
                        zorder=3)

            # Per-model markers
            for (model, cov, rmse) in pts:
                ax.plot(cov, rmse,
                        marker=fh.get_marker(model),
                        color=fh.get_color(model),
                        markersize=13,
                        markeredgecolor='white',
                        markeredgewidth=1.0,
                        linestyle='',
                        zorder=5)

            # LMM truncation arrow: short rightward arrow above LMM cluster
            lmm_pts = [p for p in pts if p[0] in LMM_MODELS]
            if lmm_pts:
                lmm_x = lmm_pts[0][1]  # all LMMs share the same coverage
                lmm_ys = [r for (_, _, r) in lmm_pts]
                lmm_y = float(np.median(lmm_ys))
                arrow_y = lmm_y - 0.04 * (y_lim[1] - y_lim[0])
                arrow_dx = 0.05
                ax.annotate(
                    '',
                    xy=(lmm_x + arrow_dx, arrow_y),
                    xytext=(lmm_x, arrow_y),
                    arrowprops=dict(arrowstyle='->', color='gray', alpha=0.4,
                                    lw=1.5),
                    zorder=2,
                )

            # Per-panel formatting
            ax.set_title(f'Saturation = {100 - rate}%', fontsize=14)
            ax.set_xlim(0, 1.05)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_ylim(*y_lim)
            ax.grid(True, alpha=0.3, linestyle=':')

            if i == len(PANEL_LAYOUT) - 1:
                ax.set_xlabel('Coverage (fraction of test cells)')
            if j == 0:
                ax.set_ylabel('RMSE')

    # --- Legends in right margin ---
    legend_models = sorted(ALL_MODELS, key=lambda m: fh.get_display_name(m))
    model_handles = [
        Line2D([0], [0], marker=fh.get_marker(m), color='w',
               markerfacecolor=fh.get_color(m), markersize=11,
               markeredgecolor='white', markeredgewidth=1.0,
               label=fh.get_display_name(m), linestyle='')
        for m in legend_models
    ]
    fig.legend(
        handles=model_handles, title='Model',
        loc='upper left', bbox_to_anchor=(0.86, 0.85),
        frameon=True, edgecolor='black',
        fontsize=12, title_fontsize=14,
    )

    frontier_handle = [
        Line2D([0], [0], color='gray', alpha=0.4, linewidth=1.2,
               label='Pareto frontier')
    ]
    fig.legend(
        handles=frontier_handle,
        loc='upper left', bbox_to_anchor=(0.86, 0.45),
        frameon=True, edgecolor='black',
        fontsize=12,
    )

    # Suptitle
    fig.suptitle(
        'Risk–Coverage Tradeoff per Saturation: Selective Prediction View',
        fontsize=18, fontweight='bold', y=0.99,
    )

    plt.subplots_adjust(left=0.06, right=0.84, top=0.93, bottom=0.07,
                        wspace=0.08, hspace=0.22)

    fh.save_figure(fig, 'risk_coverage_curves', output_dir)
