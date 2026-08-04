"""Degradation by Missingness Level — How much worse each model gets relative to its 10% baseline.

Shows % RMSE increase at each rate (20, 40, 60, 80, 90) relative to the model's own 10% RMSE.
One panel per loss type.
"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
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
    df = fh.load_main_results(results_dir)
    df = df[~df['model'].str.startswith('pca_k') | (df['model'] == 'pca_k1')]

    stats = fh.summary_stats(df)

    panels = [
        ('regression_test', r'Between-Map: Source-Informed ($B_1$)'),
        ('within_map', 'Within-Map (W)'),
        ('double_missing', r'Between-Map: Missing-Source ($B_0$)'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    compare_rates = [20, 40, 60, 80, 90]
    x_vals = np.arange(len(compare_rates))

    for ax, (lt, title) in zip(axes, panels):
        lt_stats = stats[stats['loss_type'] == lt]
        models = lt_stats['model'].unique()

        # Sort by degradation at 90% (or last available rate)
        deg_at_90 = []
        for model in models:
            ms = lt_stats[lt_stats['model'] == model]
            r10 = ms[ms['rate'] == 10]['mean'].values
            r90 = ms[ms['rate'] == 90]['mean'].values
            if len(r10) > 0 and len(r90) > 0:
                deg_at_90.append((model, ((r90[0] - r10[0]) / r10[0]) * 100))
        deg_at_90.sort(key=lambda x: x[1])
        ordered_models = [m for m, _ in deg_at_90]

        for model in ordered_models:
            ms = lt_stats[lt_stats['model'] == model].sort_values('rate')
            r10 = ms[ms['rate'] == 10]['mean'].values[0]

            pct_degs = []
            for rate in compare_rates:
                r_val = ms[ms['rate'] == rate]['mean'].values
                if len(r_val) > 0:
                    pct_degs.append(((r_val[0] - r10) / r10) * 100)
                else:
                    pct_degs.append(np.nan)

            display = fh.get_task_display_name(model, lt)
            ax.plot(x_vals, pct_degs,
                    f'{fh.get_marker(model)}-',
                    color=fh.get_color(model),
                    label=display,
                    markersize=7, linewidth=1.8, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.6)

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f'{100-r}%' for r in compare_rates])
        ax.set_xlabel('Saturation (%)')
        ax.set_title(title)
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(True, alpha=0.3, linestyle=':')

    axes[0].set_ylabel('% RMSE Change from 90% Saturation Baseline')

    fig.suptitle('Performance Degradation Relative to 90% Saturation',
                 fontsize=20, fontweight='bold')

    # Shared legend below all panels — collect unique handles from first panel (most models)
    handles, labels = axes[0].get_legend_handles_labels()
    # Add any models unique to other panels
    for ax_extra in axes[1:]:
        h2, l2 = ax_extra.get_legend_handles_labels()
        for h, l in zip(h2, l2):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    fig.legend(handles=handles, labels=labels,
               loc='lower center', bbox_to_anchor=(0.5, -0.08),
               ncol=4, frameon=True, edgecolor='black',
               fontsize=11, handlelength=1.5, columnspacing=1.0)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    fh.save_figure(fig, 'degradation_by_rate', output_dir)
