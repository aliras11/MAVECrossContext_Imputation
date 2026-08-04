"""Within-Map Imputation — Scatter/Line Plot"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
import numpy as np


def main(
    results_dir: Path,
    nodouble_results_dir: Path,
    statistics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate this module's established figure outputs."""
    fh.apply_style()
    df = fh.load_main_results(results_dir)

    # Filter to within_map, PCA k=1 only
    wm = df[(df['loss_type'] == 'within_map') &
            (~df['model'].str.startswith('pca_k') | (df['model'] == 'pca_k1'))]
    stats = fh.summary_stats(wm)

    order = stats[stats['rate'] == 40].sort_values('mean')['model'].tolist()

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = (100 - np.array(fh.RATES)) / 100.0

    for model in order:
        ms = stats[stats['model'] == model].sort_values('rate')
        display = fh.get_display_name(model)
        ax.errorbar(x_vals, ms['mean'].values, yerr=ms['ci95'].values,
                    fmt=f'{fh.get_marker(model)}-',
                    color=fh.get_color(model),
                    label=display,
                    markersize=8, capsize=4, capthick=1.2,
                    linewidth=2, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xlabel('Saturation (%)')
    ax.set_ylabel('Mean RMSE (\u00b1 95% CI)')
    ax.set_title('Within-Map (W) Imputation Performance')
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f'{100-r}%' for r in fh.RATES])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3,
              frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    fh.save_figure(fig, 'within_map_scatter', output_dir)
