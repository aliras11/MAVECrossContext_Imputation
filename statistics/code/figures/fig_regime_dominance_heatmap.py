"""fig5e: Regime-Dominance Heatmap

For each (saturation, task) cell, identify the best imputation method and
visualize the within-column RMSE rank for each model. Headline claim: there is
a best method per regime; which one depends predictably on saturation and task.
MICE RF is the all-rounder; Basic Linear shines at extreme sparsity.

Layout: a 10-row x 12-column heatmap (5 LMM + 4 AE/MICE + Column Mean rows; B_1 saturations
left block, B_0 saturations right block, both ordered 90% -> 10%) with a
winner-strip beneath highlighting the best method per regime.
"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, Rectangle
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

    # ---------------------------------------------------------------------------
    # Data prep
    # ---------------------------------------------------------------------------
    df = fh.load_main_results(results_dir)
    df = df[~df['model'].str.startswith('pca_k')]
    between = df[df['loss_type'].isin(['regression_test', 'double_missing'])].copy()

    MODELS = [
        'basic_linear',
        'oneparam_linear',
        'full_interaction_linear',
        'full_interaction_mixed',
        'mixed_random',
        'single_ae',
        'dual_ae',
        'mice',
        'mice_rf',
        'col_mean',
    ]
    LMM_MODELS = {
        'basic_linear', 'oneparam_linear',
        'full_interaction_linear', 'full_interaction_mixed', 'mixed_random',
    }

    # Compute mean RMSE per (model, rate, loss_type) by pooling over src->tgt pairs
    # within each loss_type slice (same trick as fig5b for per-task summaries).
    def task_rmse_table(loss_type):
        """Return DataFrame with index=model, columns=rate, values=mean pooled RMSE."""
        sub = between[between['loss_type'] == loss_type]
        pooled = fh.weighted_average_rmse(sub)
        summary = pooled.groupby(['model', 'rate'])['weighted_rmse'].mean().reset_index()
        table = summary.pivot(index='model', columns='rate', values='weighted_rmse')
        return table

    b1_table = task_rmse_table('regression_test')   # all 10 models present
    b0_table = task_rmse_table('double_missing')    # AE/MICE plus Column Mean

    # Column ordering: rates 10, 20, 40, 60, 80, 90 left-to-right means saturations
    # 90% -> 10% (descending saturation, easy regime on left, extreme sparsity on right).
    SAT_RATES = [10, 20, 40, 60, 80, 90]
    B1_COLS = [('regression_test', r) for r in SAT_RATES]
    B0_COLS = [('double_missing', r) for r in SAT_RATES]
    ALL_COLS = B1_COLS + B0_COLS  # 12 columns

    # Build the all-method x 12 RMSE matrix (NaN for LMM x B_0).
    rmse_matrix = np.full((len(MODELS), len(ALL_COLS)), np.nan)
    for i, model in enumerate(MODELS):
        for j, (loss_type, rate) in enumerate(ALL_COLS):
            if loss_type == 'double_missing' and model in LMM_MODELS:
                continue
            table = b1_table if loss_type == 'regression_test' else b0_table
            if model in table.index and rate in table.columns:
                rmse_matrix[i, j] = table.loc[model, rate]

    # Within-column ranks (1 = best). NaNs stay NaN. The denominator (number of
    # models present in the column) varies by column.
    rank_matrix = np.full_like(rmse_matrix, np.nan, dtype=float)
    for j in range(rmse_matrix.shape[1]):
        col = rmse_matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        order = np.argsort(col[valid])  # ascending RMSE
        ranks = np.empty(valid.sum(), dtype=float)
        ranks[order] = np.arange(1, valid.sum() + 1)
        full = np.full_like(col, np.nan, dtype=float)
        full[valid] = ranks
        rank_matrix[:, j] = full

    # Sort rows by mean rank ascending (best on top).
    mean_ranks = np.nanmean(rank_matrix, axis=1)
    row_order = np.argsort(mean_ranks)
    MODELS_SORTED = [MODELS[i] for i in row_order]
    rmse_matrix = rmse_matrix[row_order]
    rank_matrix = rank_matrix[row_order]

    # ---------------------------------------------------------------------------
    # Winner per column: model with lowest mean RMSE in that (saturation, task)
    # ---------------------------------------------------------------------------
    winners = []
    for j in range(rmse_matrix.shape[1]):
        col = rmse_matrix[:, j]
        if np.all(np.isnan(col)):
            winners.append(None)
            continue
        best_idx = np.nanargmin(col)
        winners.append(MODELS_SORTED[best_idx])

    # ---------------------------------------------------------------------------
    # Figure layout
    # ---------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[10, 1],
        hspace=0.40,
        left=0.10, right=0.84, top=0.86, bottom=0.08,
    )
    ax_main = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1])

    # Main heatmap: rank-colored cells.
    cmap = plt.get_cmap('RdYlGn_r')
    n_rows, n_cols = rank_matrix.shape

    # Normalize ranks to [0, 1] using the number of models in each task.
    # Empty (NaN) cells are masked.
    masked_ranks = np.ma.masked_invalid(rank_matrix)
    im = ax_main.imshow(
        masked_ranks,
        cmap=cmap,
        aspect='auto',
        vmin=1, vmax=len(MODELS),
        interpolation='nearest',
    )
    # Set NaN color to a neutral gray (under cmap).
    cmap.set_bad(color='#cccccc')

    # Overlay diagonal hatching on NaN (LMM x B_0) cells.
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(rank_matrix[i, j]):
                ax_main.add_patch(Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor='#cccccc', edgecolor='white',
                    hatch='///', linewidth=0.5, zorder=2,
                ))

    # Annotate non-NaN cells with RMSE values.
    for i in range(n_rows):
        for j in range(n_cols):
            val = rmse_matrix[i, j]
            if np.isnan(val):
                continue
            rank = rank_matrix[i, j]
            # Contrast-aware text color: top-3 (best, dark green) and bottom-3 (worst,
            # dark red) get white text; middle ranks (lighter colors) get black.
            if rank <= 3 or rank >= len(MODELS) - 2:
                text_color = 'white'
            else:
                text_color = 'black'
            ax_main.text(
                j, i, f'{val:.3f}',
                ha='center', va='center',
                fontsize=9, color=text_color, zorder=3,
            )

    # Vertical separator between B_1 and B_0 blocks.
    ax_main.axvline(x=5.5, color='black', linewidth=2.5, zorder=4)

    # Block headers above the heatmap.
    ax_main.text(
        2.5, -0.95, fh.LOSS_TYPE_LABELS["regression_test"],
        ha='center', va='bottom', fontsize=14, fontweight='bold',
        transform=ax_main.transData,
    )
    ax_main.text(
        8.5, -0.95, fh.LOSS_TYPE_LABELS["double_missing"],
        ha='center', va='bottom', fontsize=14, fontweight='bold',
        transform=ax_main.transData,
    )

    # Tick labels.
    sat_labels = [f'{100 - r}%' for r in SAT_RATES]
    xticklabels = sat_labels + sat_labels
    ax_main.set_xticks(np.arange(n_cols))
    ax_main.set_xticklabels(xticklabels, fontsize=11)
    ax_main.set_yticks(np.arange(n_rows))
    ax_main.set_yticklabels(
        [fh.get_display_name(m) for m in MODELS_SORTED],
        fontsize=12,
    )
    ax_main.set_xlabel('')
    ax_main.tick_params(axis='x', which='both', length=0, pad=4)
    ax_main.tick_params(axis='y', which='both', length=0, pad=4)

    # Cell grid: thin white separators.
    ax_main.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax_main.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax_main.grid(which='minor', color='white', linewidth=1.0)
    ax_main.tick_params(which='minor', length=0)
    ax_main.set_xlim(-0.5, n_cols - 0.5)
    ax_main.set_ylim(n_rows - 0.5, -0.5)

    # ---------------------------------------------------------------------------
    # Winner strip (12 chips, one per column).
    # ---------------------------------------------------------------------------
    for j, winner in enumerate(winners):
        if winner is None:
            continue
        color = fh.get_color(winner)
        ax_strip.add_patch(Rectangle(
            (j - 0.5, -0.5), 1, 1,
            facecolor=color, edgecolor='white', linewidth=1.0,
        ))
        # Choose contrast-aware text color (luminance-based).
        rgb = plt.matplotlib.colors.to_rgb(color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = 'white' if luminance < 0.55 else 'black'
        label = fh.get_display_name(winner)
        # If too long, abbreviate.
        if len(label) > 12:
            # E.g. "Mixed (rand. int.)" -> "Mxd r.int."
            label = label.replace('Mixed (rand. int.)', 'Mxd r.int.') \
                         .replace('Mixed (rand. slope)', 'Mxd r.slp.') \
                         .replace('1-Param Nonlinear', '1P Nonlin.') \
                         .replace('Linear + Domain', 'Lin+Dom') \
                         .replace('Basic Linear', 'Bas. Lin.')
        ax_strip.text(
            j, 0, label,
            ha='center', va='center',
            fontsize=9, color=text_color, fontweight='bold',
        )

    # Vertical separator on strip too.
    ax_strip.axvline(x=5.5, color='black', linewidth=2.5, zorder=4)

    ax_strip.set_xlim(-0.5, n_cols - 0.5)
    ax_strip.set_ylim(-0.5, 0.5)
    ax_strip.set_xticks(np.arange(n_cols))
    ax_strip.set_xticklabels(xticklabels, fontsize=11)
    ax_strip.set_yticks([])
    ax_strip.tick_params(axis='x', which='both', length=0, pad=4)
    ax_strip.set_xlabel('Saturation (%)', fontsize=14, fontweight='bold')
    for spine in ax_strip.spines.values():
        spine.set_visible(False)
    ax_strip.set_title(
        'Best method per (saturation, task) regime',
        fontsize=12, fontweight='bold', pad=6,
    )

    # ---------------------------------------------------------------------------
    # Right-margin: rank colorbar, hatch legend, winning-model legend.
    # ---------------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.86, 0.40, 0.015, 0.40])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Within-column RMSE rank (1 = best)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_ticks(np.arange(1, len(MODELS) + 1))

    # Hatch / gray legend just below the colorbar.
    hatch_handle = Patch(
        facecolor='#cccccc', edgecolor='black', hatch='///',
        label=r'Structurally cannot predict $B_0$',
    )
    fig.legend(
        handles=[hatch_handle],
        bbox_to_anchor=(0.86, 0.36),
        loc='upper left',
        frameon=False,
        fontsize=11,
    )

    # Winning-model legend: deduplicated set of winners.
    unique_winners = []
    for w in winners:
        if w is not None and w not in unique_winners:
            unique_winners.append(w)
    winner_handles = [
        Patch(facecolor=fh.get_color(m), edgecolor='white',
              label=fh.get_display_name(m))
        for m in unique_winners
    ]
    fig.legend(
        handles=winner_handles,
        title='Winning model',
        bbox_to_anchor=(0.86, 0.30),
        loc='upper left',
        frameon=True, edgecolor='black',
        fontsize=11, title_fontsize=12,
    )

    # ---------------------------------------------------------------------------
    # Suptitle and save.
    # ---------------------------------------------------------------------------
    fig.suptitle(
        r'Regime-Dominance: RMSE Ranks Across Saturation $\times$ Task',
        fontsize=18, fontweight='bold', y=0.99,
    )

    fh.save_figure(fig, 'regime_dominance_heatmap', output_dir)
