"""Coverage vs Accuracy Trajectories — Between-Map Models Only

Each model traces a path across missingness rates (10% → 90%).
X = total points predicted, Y = pooled RMSE.
Marker shapes encode missingness level, colors encode model.
"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def trajectory_summary(between: pd.DataFrame) -> pd.DataFrame:
    """Summarize canonical cell-weighted split pools for plotting."""
    pooled = fh.weighted_average_rmse(between)
    return (
        pooled.groupby(["model", "rate"], as_index=False)
        .agg(
            pooled_rmse_mean=("weighted_rmse", "mean"),
            pooled_rmse_ci95=(
                "weighted_rmse",
                lambda values: values.std() / np.sqrt(len(values)) * 1.96,
            ),
            total_n_points_mean=("total_n_points", "mean"),
        )
    )


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
    between_models = sorted(between['model'].unique())

    summary = trajectory_summary(between)

    # --- Marker shapes by missingness rate ---
    RATE_MARKERS = {10: 'o', 20: 's', 40: '^', 60: 'D', 80: 'p', 90: 'h'}

    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect endpoint positions for label spreading
    endpoints = []

    for model in between_models:
        ms = summary[summary['model'] == model].sort_values('rate')
        display = fh.get_display_name(model)
        color = fh.get_color(model)

        x = ms['total_n_points_mean'].values
        y = ms['pooled_rmse_mean'].values
        yerr = ms['pooled_rmse_ci95'].values
        rates = ms['rate'].values

        # Trajectory line
        ax.plot(x, y, '-', color=color, linewidth=1.5, alpha=0.5, zorder=3)

        # Points with rate-specific markers
        for xi, yi, ye, rate in zip(x, y, yerr, rates):
            ax.errorbar(xi, yi, yerr=ye,
                        fmt=RATE_MARKERS[rate], color=color, markersize=7,
                        capsize=2, capthick=0.8, linewidth=0,
                        markeredgecolor='white', markeredgewidth=0.6,
                        alpha=0.9, zorder=5)

        # Arrow target: rightmost point (max x), which is the 90% endpoint for
        # AE/MICE but the ~40% peak for LMMs whose trajectories loop back
        max_x_idx = np.argmax(x)
        endpoints.append({'model': model, 'display': display, 'color': color,
                          'x': x[max_x_idx], 'y': y[max_x_idx]})

    # --- Place all labels in one column on the right ---
    endpoints.sort(key=lambda e: e['y'])
    min_gap = 0.022
    spread_ys = [endpoints[0]['y']]
    for i in range(1, len(endpoints)):
        spread_ys.append(max(endpoints[i]['y'], spread_ys[-1] + min_gap))

    label_x = max(ep['x'] for ep in endpoints) + 20000

    for ep, label_y in zip(endpoints, spread_ys):
        ax.annotate(ep['display'],
                    xy=(ep['x'], ep['y']),
                    xytext=(label_x, label_y),
                    fontsize=13, ha='left', va='center', color=ep['color'],
                    fontweight='bold', zorder=10,
                    arrowprops=dict(arrowstyle='->', color=ep['color'],
                                    alpha=0.5, lw=0.8, shrinkA=0, shrinkB=4))

    # Rate shape legend
    from matplotlib.lines import Line2D
    rate_handles = [Line2D([0], [0], marker=RATE_MARKERS[r], color='gray',
                           markerfacecolor='gray', markersize=7, linestyle='',
                           markeredgecolor='white', markeredgewidth=0.6,
                           label=f'{100-r}%') for r in fh.RATES]
    ax.legend(handles=rate_handles, title='Saturation',
              loc='upper left', frameon=True, edgecolor='black',
              fontsize=12, title_fontsize=13)

    ax.set_xlabel('Total Points Predicted')
    ax.set_ylabel('Pooled RMSE (MSE-weighted)')
    ax.set_title('Between-Map Imputation: Accuracy vs Coverage',
                 fontsize=20, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')

    # Extend x-axis to make room for labels
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 1.35)

    plt.tight_layout()
    fh.save_figure(fig, 'between_map_accuracy_vs_coverage', output_dir)
