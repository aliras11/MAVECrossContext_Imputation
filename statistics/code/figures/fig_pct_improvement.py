"""Median RMSE differences from Column Mean — one 2x3 grid per loss type."""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt


def main(
    results_dir: Path,
    nodouble_results_dir: Path,
    statistics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate this module's established figure outputs."""
    fh.apply_style()

    loss_types = [
        (
            'regression_test',
            r'Between-Map: Source-Informed ($B_1$)',
            'vs_colmean_regression_test.csv',
        ),
        (
            'double_missing',
            r'Between-Map: Missing-Source ($B_0$)',
            'vs_colmean_double_missing.csv',
        ),
        (
            'within_map',
            'Within-Map (W)',
            'vs_colmean_within_map.csv',
        ),
    ]

    for loss_type, loss_label, baseline_filename in loss_types:
        baseline = fh.load_percent_panel_data(
            statistics_dir,
            baseline_filename,
        )
        if baseline.empty:
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        for ax, rate in zip(axes.flat, fh.RATES):
            bl = baseline.loc[baseline["rate"] == rate].copy()
            if bl.empty:
                continue

            bl = bl.sort_values('pct_improvement', ascending=True)

            colors = [fh.get_color(m) for m in bl['model']]
            y_pos = range(len(bl))
            ax.barh(y_pos, bl['pct_improvement'].values, color=colors,
                    edgecolor='white', linewidth=0.5, alpha=0.85, height=0.7)
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(
                [
                    fh.get_task_display_name(model, loss_type)
                    for model in bl["model"]
                ],
                fontsize=12,
            )
            ax.axvline(x=0, color='black', linewidth=0.8)
            ax.set_title(f'{100-rate}% Saturation', fontsize=16)
            ax.grid(True, alpha=0.3, linestyle=':', axis='x')

            # Significance stars
            for i, (_, r) in enumerate(bl.iterrows()):
                if r['significant_0_05']:
                    star_color = '#228833' if r['pct_improvement'] > 0 else '#CC3311'
                    x_pos = r['pct_improvement']
                    nudge = 1 if x_pos >= 0 else -1
                    ax.text(x_pos + nudge, i, '*', va='center', fontsize=16,
                            fontweight='bold', color=star_color)

        fig.suptitle(
            f'Median RMSE Relative to Column Mean Baseline ({loss_label})',
            fontsize=20,
            fontweight='bold',
        )
        fig.supxlabel(
            'Reduction in median RMSE relative to column-mean baseline (%)',
            fontsize=15,
            fontweight='bold',
            y=0.015,
        )
        plt.tight_layout(rect=[0, 0.055, 0.97, 0.96])

        fname = f"pct_improvement_{loss_type}"
        fh.save_figure(fig, fname, output_dir)
