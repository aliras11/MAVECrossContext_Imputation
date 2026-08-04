"""Point Composition Pie Charts — Between-Map Test Set Breakdown

Shows regression_test vs double_missing composition for wt12→av12
at each missingness rate, with counts and totals.
"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


def main(
    results_dir: Path,
    nodouble_results_dir: Path,
    statistics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate this module's established figure outputs."""
    fh.apply_style()
    df = fh.load_main_results(results_dir)
    df = df[~df['model'].str.startswith('pca_k')]

    # Only between-map loss types
    between = df[df['loss_type'].isin(['regression_test', 'double_missing'])]

    # Pie chart data: wt12→av12 point composition (average across splits)
    pair_data = between[(between['src'] == 'wt12') & (between['tgt'] == 'av12') &
                         (between['model'] == 'single_ae')]  # counts are same for all models
    pie_stats = pair_data.groupby(['rate', 'loss_type'])['n_points'].mean().unstack(fill_value=0)

    fig, axes = plt.subplots(1, 6, figsize=(16, 5))
    pie_colors = ['#2166AC', '#D6604D']  # RdBu diverging: blue=B_1, red=B_0

    for ax, rate in zip(axes, fh.RATES):
        n_reg = pie_stats.loc[rate, 'regression_test'] if 'regression_test' in pie_stats.columns else 0
        n_dm = pie_stats.loc[rate, 'double_missing'] if 'double_missing' in pie_stats.columns else 0
        total = n_reg + n_dm

        wedges, texts, autotexts = ax.pie(
            [n_reg, n_dm], colors=pie_colors, autopct='%1.0f%%',
            startangle=90, textprops={'fontsize': 12},
            wedgeprops={'edgecolor': 'white', 'linewidth': 1})

        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')

        ax.set_title(f'{100-rate}% saturation', fontsize=13, fontweight='bold')
        ax.text(0, -1.35, f'$B_1$: {n_reg:,.0f}\n$B_0$: {n_dm:,.0f}\nTest pts: {total:,.0f}',
                ha='center', va='top', fontsize=10, linespacing=1.3)

    # Legend
    pie_handles = [
        Patch(facecolor='#2166AC', edgecolor='white', label=r'Source-Informed ($B_1$)'),
        Patch(facecolor='#D6604D', edgecolor='white', label=r'Missing-Source ($B_0$)'),
    ]
    fig.suptitle('Between-Map Test Set Composition by Saturation Level',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()

    fig.legend(handles=pie_handles, loc='lower center', ncol=2,
               frameon=True, edgecolor='black', fontsize=13,
               title='Point composition for wt12 \u2192 av12', title_fontsize=14,
               bbox_to_anchor=(0.5, -0.08))
    fh.save_figure(fig, 'between_map_point_composition', output_dir)
