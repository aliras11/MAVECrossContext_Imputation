"""fig5d: Accuracy + Composition paired panel.

Three-panel layout: top-left B_1 RMSE vs saturation, top-right B_0 RMSE vs
saturation (only B_0-capable models), bottom a stacked composition bar showing
the share of B_1 vs B_0 cells in the between-map test set as a function of
saturation. Headline: at high missingness, the bulk of the imputable problem
is B_0; methods that cannot predict B_0 are tackling a vanishing slice of the
true problem.
"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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

    # --- Per-model, per-rate mean RMSE for each loss type ---
    # weighted_average_rmse on a single loss_type slice collapses across (src->tgt) pairs only.
    b1_pooled = fh.weighted_average_rmse(between[between['loss_type'] == 'regression_test'])
    b0_pooled = fh.weighted_average_rmse(between[between['loss_type'] == 'double_missing'])

    b1_summary = b1_pooled.groupby(['model', 'rate']).agg(
        rmse_mean=('weighted_rmse', 'mean'),
    ).reset_index()
    b0_summary = b0_pooled.groupby(['model', 'rate']).agg(
        rmse_mean=('weighted_rmse', 'mean'),
    ).reset_index()

    # Saturation values (linearly spaced numeric x-positions)
    SATURATIONS = [10, 20, 40, 60, 80, 90]
    RATE_TO_SAT = dict(zip(fh.RATES, SATURATIONS))  # missingness rate -> saturation
    # But fh.RATES = [10, 20, 40, 60, 80, 90] is the missingness rate; saturation = 100 - rate.
    # We will plot vs saturation = 100 - rate.

    # --- Composition counts (model-agnostic; pick any model that has both loss types) ---
    # Use dual_ae as the reference (per spec).
    ref_model = 'dual_ae'
    ref = between[between['model'] == ref_model]

    # Sum n_points across pairs and splits per (rate, loss_type) to get representative
    # total counts of test cells in each bucket.
    ref_counts = ref.groupby(['rate', 'loss_type'])['n_points'].sum().unstack('loss_type').fillna(0)
    # Per-split mean count is more interpretable but the share is invariant to that scaling.
    ref_counts['B1_share'] = ref_counts['regression_test'] / (
        ref_counts['regression_test'] + ref_counts['double_missing']
    )
    ref_counts['B0_share'] = ref_counts['double_missing'] / (
        ref_counts['regression_test'] + ref_counts['double_missing']
    )
    ref_counts = ref_counts.reset_index()

    # --- B_1 / B_0 model rosters ---
    B1_MODELS_DISPLAY_ORDER = [
        'single_ae', 'dual_ae', 'mice', 'mice_rf',
        'basic_linear', 'oneparam_linear', 'full_interaction_linear',
        'mixed_random', 'full_interaction_mixed',
    ]
    B0_MODELS_DISPLAY_ORDER = ['single_ae', 'dual_ae', 'mice', 'mice_rf']

    # --- Bar colors (off-palette but visually compatible) ---
    B1_BAR_COLOR = '#88AAEE'
    B0_BAR_COLOR = '#EE9988'

    # --- Figure layout ---
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1], hspace=0.30, wspace=0.18)
    ax_b1 = fig.add_subplot(gs[0, 0])
    ax_b0 = fig.add_subplot(gs[0, 1])
    ax_comp = fig.add_subplot(gs[1, :])


    def _plot_panel(ax, summary, models, title, annotate_lmm=False):
        for model in models:
            ms = summary[summary['model'] == model].sort_values('rate')
            if ms.empty:
                continue
            x = [100 - r for r in ms['rate'].values]
            y = ms['rmse_mean'].values
            # Sort by saturation (ascending) so the line connects in saturation order
            order = np.argsort(x)
            x = np.array(x)[order]
            y = y[order]
            color = fh.get_color(model)
            ax.plot(x, y,
                    marker='o', markersize=7, linewidth=1.6,
                    color=color,
                    markeredgecolor='white', markeredgewidth=0.6,
                    label=fh.get_display_name(model))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Saturation (%)')
        ax.set_ylabel('RMSE')
        ax.set_xticks(SATURATIONS)
        ax.set_xticklabels([f'{s}%' for s in SATURATIONS])
        ax.grid(True, linestyle=':', alpha=0.3)
        if annotate_lmm:
            ax.text(
                0.98, 0.02,
                'LMM family absent: structurally cannot predict $B_0$ cells',
                transform=ax.transAxes,
                ha='right', va='bottom',
                fontsize=10, style='italic', color='#555555',
            )


    _plot_panel(ax_b1, b1_summary, B1_MODELS_DISPLAY_ORDER,
                f"{fh.LOSS_TYPE_LABELS['regression_test']} RMSE", annotate_lmm=False)
    _plot_panel(ax_b0, b0_summary, B0_MODELS_DISPLAY_ORDER,
                f"{fh.LOSS_TYPE_LABELS['double_missing']} RMSE", annotate_lmm=True)

    # --- Composition stacked bar ---
    comp_x = [100 - int(r) for r in ref_counts['rate'].values]
    b1_share = ref_counts['B1_share'].values
    b0_share = ref_counts['B0_share'].values
    order = np.argsort(comp_x)
    comp_x = np.array(comp_x)[order]
    b1_share = b1_share[order]
    b0_share = b0_share[order]

    bar_width = 6
    ax_comp.bar(comp_x, b1_share, width=bar_width, color=B1_BAR_COLOR,
                edgecolor='black', linewidth=0.5, label=r'$B_1$ share')
    ax_comp.bar(comp_x, b0_share, width=bar_width, bottom=b1_share,
                color=B0_BAR_COLOR, edgecolor='black', linewidth=0.5,
                label=r'$B_0$ share')

    # Annotate B_0 share when >15%
    for xi, b1s, b0s in zip(comp_x, b1_share, b0_share):
        if b0s > 0.15:
            ax_comp.text(xi, b1s + b0s / 2,
                         f'{b0s * 100:.0f}% $B_0$',
                         ha='center', va='center',
                         color='white', fontsize=11, fontweight='bold')

    ax_comp.axhline(0.5, color='black', alpha=0.2, linewidth=1)
    ax_comp.set_xticks(SATURATIONS)
    ax_comp.set_xticklabels([f'{s}%' for s in SATURATIONS])
    ax_comp.set_xlabel('Saturation (%)')
    ax_comp.set_ylabel('Fraction of between-map test cells')
    ax_comp.set_ylim(0, 1)
    ax_comp.set_title('Test-set composition: $B_0$ share grows with missingness',
                      fontsize=13)
    ax_comp.grid(False)

    # --- Legends in the right margin (outside all panels) ---
    model_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=fh.get_color(m), markersize=9,
               markeredgecolor='white', markeredgewidth=0.6,
               label=fh.get_display_name(m))
        for m in B1_MODELS_DISPLAY_ORDER
    ]
    fig.legend(
        handles=model_handles, title='Model',
        bbox_to_anchor=(0.86, 0.92), loc='upper left',
        frameon=True, edgecolor='black',
        fontsize=12, title_fontsize=14,
    )

    comp_handles = [
        Patch(facecolor=B1_BAR_COLOR, edgecolor='black', linewidth=0.5,
              label=r'$B_1$ share'),
        Patch(facecolor=B0_BAR_COLOR, edgecolor='black', linewidth=0.5,
              label=r'$B_0$ share'),
    ]
    fig.legend(
        handles=comp_handles, title='Composition',
        bbox_to_anchor=(0.86, 0.32), loc='upper left',
        frameon=True, edgecolor='black',
        fontsize=12, title_fontsize=14,
    )

    fig.suptitle(
        'Imputation Accuracy by Task and the Shifting Composition of the Test Set',
        fontsize=18, fontweight='bold', y=0.99,
    )

    plt.subplots_adjust(left=0.06, right=0.84, top=0.93, bottom=0.07)

    fh.save_figure(fig, 'accuracy_composition_panels', output_dir)
