"""fig5b: Coverage-vs-Accuracy, Decomposed

Three panels (1x3): single-missing (B_1), double-missing (B_0), and the existing
pooled view from fig5. Same data flow and pooling formula as
fig_coverage_vs_accuracy_trajectory.py — applied per panel over different
loss_type slices.
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

    # --- Load and filter ---
    df = fh.load_main_results(results_dir)
    df = df[~df['model'].str.startswith('pca_k')]
    between = df[df['loss_type'].isin(['regression_test', 'double_missing'])]

    panel_data = {
        'B_1':    between[between['loss_type'] == 'regression_test'],
        'B_0':    between[between['loss_type'] == 'double_missing'],
        'Pooled': between,
    }

    PANEL_ORDER = ['B_1', 'B_0', 'Pooled']
    PANEL_TITLES = {
        'B_1':    'Single-Missing (B_1)',
        'B_0':    'Double-Missing (B_0)',
        'Pooled': 'Pooled (B_1 + B_0)',
    }

    # --- Per-panel aggregation ---
    # weighted_average_rmse groups by (model, rate, split) and applies n_points-weighted MSE -> sqrt.
    # For B_1 / B_0 panels (single loss_type), this collapses across src->tgt pairs only.
    # For the Pooled panel (both loss_types), it pools across pairs AND loss types — same
    # as fig_coverage_vs_accuracy_trajectory.py.
    panel_summaries = {}
    for name, pdat in panel_data.items():
        pooled = fh.weighted_average_rmse(pdat)
        summary = pooled.groupby(['model', 'rate']).agg(
            rmse_mean=('weighted_rmse', 'mean'),
            rmse_ci95=('weighted_rmse', lambda x: x.std() / np.sqrt(len(x)) * 1.96),
            n_points_mean=('total_n_points', 'mean'),
        ).reset_index()
        panel_summaries[name] = summary

    # --- Plot ---
    RATE_MARKERS = {10: 'o', 20: 's', 40: '^', 60: 'D', 80: 'p', 90: 'h'}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True, sharex=False)

    for ax, name in zip(axes, PANEL_ORDER):
        summary = panel_summaries[name]
        for model in sorted(summary['model'].unique()):
            ms = summary[summary['model'] == model].sort_values('rate')
            color = fh.get_color(model)

            x = ms['n_points_mean'].values
            y = ms['rmse_mean'].values
            yerr = ms['rmse_ci95'].values
            rates = ms['rate'].values

            # Trajectory line (semi-transparent, behind markers)
            ax.plot(x, y, '-', color=color, linewidth=1.5, alpha=0.5, zorder=3)

            # Per-rate markers with errorbars
            for xi, yi, ye, rate in zip(x, y, yerr, rates):
                ax.errorbar(xi, yi, yerr=ye,
                            fmt=RATE_MARKERS[rate], color=color, markersize=7,
                            capsize=2, capthick=0.8, linewidth=0,
                            markeredgecolor='white', markeredgewidth=0.6,
                            alpha=0.9, zorder=5)

    # --- Shared figure-level legends on the right (Model above, Saturation below) ---
    all_models = sorted(panel_summaries['B_1']['model'].unique(),
                        key=lambda m: fh.get_display_name(m))
    model_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=fh.get_color(m), markersize=9,
               markeredgecolor='white', markeredgewidth=0.6,
               label=fh.get_display_name(m))
        for m in all_models
    ]
    rate_handles = [
        Line2D([0], [0], marker=RATE_MARKERS[r], color='gray',
               markerfacecolor='gray', markersize=8, linestyle='',
               markeredgecolor='white', markeredgewidth=0.6,
               label=f'{100 - r}%')
        for r in fh.RATES
    ]
    fig.legend(
        handles=model_handles, title='Model',
        loc='upper left', bbox_to_anchor=(0.86, 0.93),
        frameon=True, edgecolor='black',
        fontsize=12, title_fontsize=14,
    )
    fig.legend(
        handles=rate_handles, title='Saturation',
        loc='upper left', bbox_to_anchor=(0.86, 0.42),
        frameon=True, edgecolor='black',
        fontsize=12, title_fontsize=14,
    )

    # --- B_0 panel footnote: explain why LMMs are absent ---
    b0_ax = axes[PANEL_ORDER.index('B_0')]
    b0_ax.text(
        0.5, -0.18,
        'Linear and mixed-effects models cannot predict B$_0$ cells '
        '(no source value to regress on); only AE and MICE families shown.',
        transform=b0_ax.transAxes,
        ha='center', va='top', fontsize=10, style='italic', color='#444444',
    )

    # --- Titles and axis labels ---
    for ax, name in zip(axes, PANEL_ORDER):
        ax.set_title(PANEL_TITLES[name], fontsize=16, fontweight='bold')
        ax.set_xlabel('Total Points Predicted')
        ax.grid(True, alpha=0.3, linestyle=':')

    axes[0].set_ylabel('RMSE (MSE-weighted within panel)')

    fig.suptitle(
        'Between-Map Imputation: Accuracy vs Coverage, Decomposed',
        fontsize=20, fontweight='bold', y=0.98,
    )

    # Tight inter-panel spacing; leave room on the right for the shared legends.
    plt.subplots_adjust(left=0.05, right=0.84, top=0.90, bottom=0.16, wspace=0.06)
    fh.save_figure(fig, 'between_map_accuracy_vs_coverage_panels', output_dir)
