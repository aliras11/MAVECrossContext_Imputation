"""PCA Component Sensitivity — Within-Map RMSE by number of components"""
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

    # Filter PCA models only
    pca = df[df['model'].str.startswith('pca_k')]
    stats = fh.summary_stats(pca)

    # Extract k values and sort
    k_models = sorted(stats['model'].unique(), key=lambda x: int(x.replace('pca_k', '')))
    k_values = [int(m.replace('pca_k', '') ) for m in k_models]

    # Use a sequential colormap for k progression
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0.15, 0.9, len(k_models)))

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = (100 - np.array(fh.RATES)) / 100.0

    for i, (model, k) in enumerate(zip(k_models, k_values)):
        ms = stats[stats['model'] == model].sort_values('rate')
        ax.errorbar(x_vals, ms['mean'].values, yerr=ms['ci95'].values,
                    fmt='o-', color=colors[i],
                    label=f'k = {k}',
                    markersize=5, capsize=4, capthick=1.2,
                    linewidth=2, alpha=0.85,
                    markeredgecolor='white', markeredgewidth=0.8)

    ax.set_xlabel('Saturation')
    ax.set_ylabel('Mean RMSE (\u00b1 95% CI)')
    ax.set_title('PCA Imputation: Component Sensitivity')
    ax.set_xticks(x_vals)
    ax.set_xticklabels([f'{100-r}%' for r in fh.RATES])
    ax.legend(title='Components', loc='upper right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    fh.save_figure(fig, 'pca_k_sensitivity', output_dir)
