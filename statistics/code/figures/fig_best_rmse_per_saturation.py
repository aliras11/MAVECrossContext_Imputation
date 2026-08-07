"""fig8: Best Achievable RMSE per Saturation

Three series on one panel:
  * Best B_1            — per-rate min over all task-matched B_1 models
  * Best B_0            — per-rate min over all task-matched B_0 models
  * Task-routed best    — per-rate hybrid pooling best_B1 + best_B0 weighted by
                          n_B1 and n_B0 (per-task oracle lower bound)

"Best" is defined as lowest mean RMSE across the 50 splits. No statistical-
significance gating. Error bars are intentionally omitted because the per-rate
"best" is a selected statistic and naïve CIs are biased downward (see design doc
docs/plans/2026-05-05-fig8-best-rmse-per-saturation-design.md).
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

    # === Data load and filter (same as fig5b's setup) ===
    df = fh.load_main_results(results_dir)
    df = df[~df['model'].str.startswith('pca_k')]
    between = df[df['loss_type'].isin(['regression_test', 'double_missing'])]


    def per_model_summary(loss_type):
        """Per-(model, rate) RMSE summary for a single loss type.

        Uses figure_helpers.weighted_average_rmse to collapse across (split, src->tgt pair) with
        the n-weighted-MSE formula, then averages weighted_rmse across splits.
        """
        pdat = between[between['loss_type'] == loss_type]
        pooled = fh.weighted_average_rmse(pdat)
        return pooled.groupby(['model', 'rate']).agg(
            rmse_mean=('weighted_rmse', 'mean'),
            n_points_mean=('total_n_points', 'mean'),
        ).reset_index()


    summary_B1 = per_model_summary('regression_test')
    summary_B0 = per_model_summary('double_missing')


    # === Per-rate winners (lowest mean across splits; mean-first selection) ===
    def winners(summary):
        rows = []
        for rate in fh.RATES:
            sub = summary[summary['rate'] == rate].reset_index(drop=True)
            i = sub['rmse_mean'].idxmin()
            rows.append({
                'rate':       rate,
                'model':      sub.loc[i, 'model'],
                'rmse_mean':  sub.loc[i, 'rmse_mean'],
            })
        return pd.DataFrame(rows)


    win_B1 = winners(summary_B1)
    win_B0 = winners(summary_B0)

    # === Cell counts per rate (model-agnostic; verified consistent across models) ===
    # Pick any one model that has both regression_test and double_missing rows.
    # dual_ae satisfies this for every (rate, split, src, tgt).
    one = between[between['model'] == 'dual_ae']
    counts = (one.groupby(['rate', 'split', 'loss_type'])['n_points'].sum()
              .reset_index()
              .groupby(['rate', 'loss_type'])['n_points'].mean()
              .unstack('loss_type')
              .rename(columns={'regression_test': 'n_B1',
                               'double_missing':  'n_B0'})
              .reset_index())

    # === Task-routed best: per-task oracle lower bound ===
    # At each rate, weighted-RMS-pool the per-task winners' RMSEs by their cell counts.
    trb_rows = []
    for rate in fh.RATES:
        n_B1 = float(counts.loc[counts['rate'] == rate, 'n_B1'].iloc[0])
        n_B0 = float(counts.loc[counts['rate'] == rate, 'n_B0'].iloc[0])
        rmse_B1 = float(win_B1.loc[win_B1['rate'] == rate, 'rmse_mean'].iloc[0])
        rmse_B0 = float(win_B0.loc[win_B0['rate'] == rate, 'rmse_mean'].iloc[0])
        pooled = np.sqrt((rmse_B1**2 * n_B1 + rmse_B0**2 * n_B0) / (n_B1 + n_B0))
        trb_rows.append({'rate': rate, 'rmse_mean': pooled})
    task_routed_best = pd.DataFrame(trb_rows)

    # Sanity: the weighted RMS of two scalars must lie in their convex hull.
    # Equivalently, task-routed best RMSE must lie between best_B1 and best_B0 RMSE.
    for rate in fh.RATES:
        rb = float(task_routed_best.loc[task_routed_best['rate'] == rate, 'rmse_mean'].iloc[0])
        b1 = float(win_B1.loc[win_B1['rate']           == rate, 'rmse_mean'].iloc[0])
        b0 = float(win_B0.loc[win_B0['rate']           == rate, 'rmse_mean'].iloc[0])
        lo, hi = min(b1, b0), max(b1, b0)
        assert lo - 1e-9 <= rb <= hi + 1e-9, (
            f"Task-routed best at rate={rate} (={rb:.4f}) is outside "
            f"[best_B1, best_B0] = [{lo:.4f}, {hi:.4f}]. "
            "Sanity invariant violated — investigate."
        )
    print('Sanity check passed: task-routed best RMSE lies between best_B1 and best_B0 at every rate.')

    # === Plot setup ===
    # X-axis is saturation (%); plot rates in saturation-ascending order.
    sat_sorted_rates = sorted(fh.RATES, key=lambda r: 100 - r)  # rates that yield ascending saturation
    sat_x = [100 - r for r in sat_sorted_rates]                  # x-values: 10, 20, 40, 60, 80, 90

    fig, ax = plt.subplots(figsize=(10, 7))


    def plot_series(ax, x, y, marker_colors, line_color, line_style, line_width):
        """Plot one series: a single connecting line plus per-point colored markers."""
        ax.plot(x, y, linestyle=line_style, color=line_color,
                linewidth=line_width, zorder=3)
        for xi, yi, mc in zip(x, y, marker_colors):
            ax.plot(xi, yi, marker='o', markersize=11,
                    markerfacecolor=mc, markeredgecolor='white', markeredgewidth=1.0,
                    linestyle='', zorder=5)


    # Helper: pull the per-rate value out of the per-task winners DataFrame in
    # saturation-ascending order.
    def _series_y(df_winners):
        return [float(df_winners.loc[df_winners['rate'] == r, 'rmse_mean'].iloc[0])
                for r in sat_sorted_rates]


    def _series_marker_colors(df_winners):
        return [fh.get_color(df_winners.loc[df_winners['rate'] == r, 'model'].iloc[0])
                for r in sat_sorted_rates]


    # B_0 first (drawn behind the others)
    plot_series(ax, sat_x, _series_y(win_B0), _series_marker_colors(win_B0),
                line_color='#7a7a7a', line_style=':', line_width=1.6)

    # B_1 next
    plot_series(ax, sat_x, _series_y(win_B1), _series_marker_colors(win_B1),
                line_color='#4a4a4a', line_style='--', line_width=1.6)

    # Task-routed best last (visually dominant)
    trb_y = [float(task_routed_best.loc[task_routed_best['rate'] == r, 'rmse_mean'].iloc[0])
             for r in sat_sorted_rates]
    trb_marker_colors = ['#1a1a1a'] * len(sat_sorted_rates)  # uniform black per design
    plot_series(ax, sat_x, trb_y, trb_marker_colors,
                line_color='#1a1a1a', line_style='-', line_width=2.2)

    # === Legends ===
    # Series legend: line styles only (no markers — those are the per-point model colors).
    series_handles = [
        Line2D([0], [0], color='#1a1a1a', linewidth=2.2, linestyle='-',
               label='Task-routed best'),
        Line2D([0], [0], color='#4a4a4a', linewidth=1.6, linestyle='--',
               label='Best $B_1$'),
        Line2D([0], [0], color='#7a7a7a', linewidth=1.6, linestyle=':',
               label='Best $B_0$'),
    ]
    # Winning-models legend: only models that win at least one (rate, series) point.
    winning_models = sorted(
        set(win_B1['model'].unique()) | set(win_B0['model'].unique()),
        key=lambda m: fh.get_display_name(m),
    )
    model_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=fh.get_color(m), markersize=10,
               markeredgecolor='white', markeredgewidth=1.0,
               label=fh.get_display_name(m))
        for m in winning_models
    ]

    # Both legends placed outside the axes, stacked on the right of the figure.
    series_legend = fig.legend(
        handles=series_handles, title='Series',
        loc='upper left', bbox_to_anchor=(0.80, 0.92),
        frameon=True, edgecolor='black',
        fontsize=11, title_fontsize=12,
    )
    fig.legend(
        handles=model_handles, title='Winning model',
        loc='upper left', bbox_to_anchor=(0.80, 0.62),
        frameon=True, edgecolor='black',
        fontsize=11, title_fontsize=12,
    )

    # === Axes, title, grid ===
    ax.set_xlabel('Saturation (%)')
    ax.set_ylabel('RMSE')
    ax.set_title(
        'Post Hoc Task-Routed Oracle Across Saturation',
        fontsize=18, fontweight='bold',
    )
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xticks(sat_x)
    ax.set_xticklabels([f'{s}%' for s in sat_x])

    # Tight panel + room on the right for stacked external legends.
    plt.subplots_adjust(left=0.09, right=0.78, top=0.90, bottom=0.10)
    fh.save_figure(fig, 'best_rmse_per_saturation', output_dir)
