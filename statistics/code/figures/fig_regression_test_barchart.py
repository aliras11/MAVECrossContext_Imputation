"""Regression Test — Grouped Bar Chart with best-model significance marker"""
from pathlib import Path
from figures import figure_helpers as fh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def main(
    results_dir: Path,
    nodouble_results_dir: Path,
    statistics_dir: Path,
    output_dir: Path,
) -> None:
    """Generate this module's established figure outputs."""
    plot_rates = fh.RATES
    suffix = ""
    fh.apply_style()
    summary, pairwise = fh.load_bar_panel_data(
        statistics_dir,
        "rmse_summary_regression_test.csv",
        "pairwise_mwu_regression_test.csv",
    )

    # Order models by mean RMSE at rate 40
    order = (
        summary.loc[summary["rate"] == 40]
        .sort_values("mean_rmse")["model"]
        .tolist()
    )
    order = [model for model in order if model != "col_mean"]

    n_methods = len(order)
    n_rates = len(plot_rates)

    means = (
        summary.pivot(index="model", columns="rate", values="mean_rmse")
        .reindex(index=order, columns=plot_rates)
        .to_numpy()
    )
    sems = (
        summary.pivot(index="model", columns="rate", values="se_rmse")
        .reindex(index=order, columns=plot_rates)
        .to_numpy()
    )
    cm_means = (
        summary.loc[summary["model"] == "col_mean"]
        .set_index("rate")["mean_rmse"]
        .reindex(plot_rates)
        .to_numpy()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(n_rates)
    width = 0.8 / n_methods

    bar_containers = {}
    for i, model in enumerate(order):
        offset = (i - n_methods / 2 + 0.5) * width
        color = fh.get_color(model)
        display = fh.get_task_display_name(model, "regression_test")

        ax.bar(x + offset, means[i, :], width,
               label=display, color=color, alpha=0.85,
               yerr=sems[i, :] * 1.96, capsize=2,
               error_kw={'elinewidth': 0.8, 'capthick': 0.8})
        bar_containers[model] = (offset, means[i, :], sems[i, :])

    # Mark best model at each rate (if sig from all others)
    for j, rate in enumerate(plot_rates):
        finite = np.flatnonzero(np.isfinite(means[:, j]))
        if len(finite) == 0:
            continue
        best_idx = finite[np.argmin(means[finite, j])]
        best_model = order[best_idx]
        if np.isfinite(cm_means[j]) and cm_means[j] < means[best_idx, j]:
            best_model = "col_mean"
        if fh.best_model_is_significant(
            pairwise,
            rate=rate,
            best_model=best_model,
            displayed_models={order[i] for i in finite} | {"col_mean"},
        ):
            if best_model == "col_mean":
                bar_x, marker_y = x[j], cm_means[j] + 0.006
            else:
                offset, m, s = bar_containers[best_model]
                bar_x = x[j] + offset
                marker_y = m[j] + s[j] * 1.96 + 0.006
            ax.plot(bar_x, marker_y, '*', markersize=12, color='gold',
                    markeredgecolor='black', markeredgewidth=0.5, zorder=10)

    # Column mean baseline horizontal lines
    for j, rate in enumerate(plot_rates):
        if not np.isnan(cm_means[j]):
            ax.hlines(cm_means[j], x[j] - 0.42, x[j] + 0.42,
                      colors='red', linestyles='--', linewidth=1.8, alpha=0.7,
                      label='Column Mean baseline' if j == 0 else None)

    ax.set_xlabel('Saturation (%)')
    ax.set_ylabel('Mean RMSE (\u00b1 95% CI)')
    ax.set_title(r'Between-Map: Source-Informed ($B_1$)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{100-r}%' for r in plot_rates])
    ax.grid(True, alpha=0.3, linestyle=':', axis='y')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                           markeredgecolor='black', markersize=12,
                           label='Lowest mean; significant vs all'))
    labels.append('Lowest mean; significant vs all')
    ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left',
              fontsize=12, frameon=True, edgecolor='black')

    plt.tight_layout()
    fh.save_figure(fig, f'regression_test_barchart{suffix}', output_dir)
