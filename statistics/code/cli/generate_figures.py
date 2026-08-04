"""Generate, validate, and publish every active manuscript figure."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile

from figures.figure_generation import (
    FIGURE_EXTENSIONS,
    publish_staged_outputs,
)


# (source module, raw output stem) -> paper-facing output stem
FIGURE_MAP = {
    ("fig_within_map_barchart.py", "within_map_barchart"):
        "fig2_within_map_barchart",
    ("fig_regression_test_barchart.py", "regression_test_barchart"):
        "fig3_regression_test_barchart",
    ("fig_double_missing_barchart.py", "double_missing_barchart"):
        "fig4_double_missing_barchart",
    (
        "fig_nodouble_regression_test_barchart.py",
        "nodouble_regression_test_barchart",
    ): "nodouble_regression_test_barchart",
    (
        "fig_coverage_vs_accuracy_trajectory.py",
        "between_map_accuracy_vs_coverage",
    ): "fig5_coverage_vs_accuracy",
    (
        "fig_coverage_vs_accuracy_panels.py",
        "between_map_accuracy_vs_coverage_panels",
    ): "fig5b_coverage_vs_accuracy_panels",
    ("fig_risk_coverage_curves.py", "risk_coverage_curves"):
        "fig5c_risk_coverage_curves",
    ("fig_accuracy_composition_panels.py", "accuracy_composition_panels"):
        "fig5d_accuracy_composition_panels",
    ("fig_regime_dominance_heatmap.py", "regime_dominance_heatmap"):
        "fig5e_regime_dominance_heatmap",
    ("fig_point_composition_pies.py", "between_map_point_composition"):
        "fig6_point_composition",
    ("fig_degradation_chart.py", "degradation_by_rate"):
        "fig7_degradation_by_rate",
    ("fig_best_rmse_per_saturation.py", "best_rmse_per_saturation"):
        "fig8_best_rmse_per_saturation",
    ("fig_pca_k_sensitivity.py", "pca_k_sensitivity"):
        "figS1_pca_k_sensitivity",
    ("fig_pct_improvement.py", "pct_improvement_regression_test"):
        "figS2a_pct_improvement_regression_test",
    ("fig_pct_improvement.py", "pct_improvement_double_missing"):
        "figS2b_pct_improvement_double_missing",
    ("fig_pct_improvement.py", "pct_improvement_within_map"):
        "figS2c_pct_improvement_within_map",
    ("fig_regression_test_scatter.py", "regression_test_scatter"):
        "figS3a_regression_test_scatter",
    ("fig_double_missing_scatter.py", "double_missing_scatter"):
        "figS3b_double_missing_scatter",
    ("fig_within_map_scatter.py", "within_map_scatter"):
        "figS3c_within_map_scatter",
}


def _module_name(script: str) -> str:
    module_name = script.removesuffix(".py")
    return module_name if "." in module_name else f"figures.{module_name}"


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_inventory(
    directories: tuple[tuple[str, Path], ...],
) -> list[dict[str, object]]:
    inventory = []
    for role, directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for file_path in directory.glob("*.csv"):
            inventory.append({
                "role": role,
                "filename": file_path.name,
                "relative_path": file_path.relative_to(directory).as_posix(),
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            })
    return sorted(
        inventory,
        key=lambda item: (
            str(item["filename"]),
            str(item["role"]),
            str(item["relative_path"]),
        ),
    )


def _output_inventory(
    output_dir: Path,
) -> list[dict[str, object]]:
    outputs = []
    for (_, raw_stem), published_stem in sorted(FIGURE_MAP.items()):
        for extension in FIGURE_EXTENSIONS:
            file_path = output_dir / f"{published_stem}{extension}"
            outputs.append({
                "raw_stem": raw_stem,
                "published_stem": published_stem,
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            })
    return outputs


def generate_all_figures(
    results_dir: Path,
    nodouble_results_dir: Path,
    statistics_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate into a temporary stage, validate, then publish."""
    results_dir = Path(results_dir)
    nodouble_results_dir = Path(nodouble_results_dir)
    statistics_dir = Path(statistics_dir)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    with tempfile.TemporaryDirectory(
        prefix=".mave-figures-",
        dir=output_dir.parent,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        module_names = sorted(
            {_module_name(script) for script, _ in FIGURE_MAP}
        )
        for module_name in module_names:
            module = importlib.import_module(module_name)
            module.main(
                results_dir,
                nodouble_results_dir,
                statistics_dir,
                staging_dir,
            )

        figure_paths = publish_staged_outputs(
            staging_dir=staging_dir,
            figure_map=FIGURE_MAP,
            destinations=[output_dir],
        )

    completed = datetime.now(timezone.utc)
    manifest = {
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "python": sys.executable,
        "automated_figure_count": len(FIGURE_MAP),
        "raw_stems": sorted(raw_stem for _, raw_stem in FIGURE_MAP),
        "published_stems": sorted(FIGURE_MAP.values()),
        "raw_to_published": {
            raw_stem: published_stem
            for (_, raw_stem), published_stem in sorted(FIGURE_MAP.items())
        },
        "input_files": _input_inventory((
            ("nodouble_results", nodouble_results_dir),
            ("results", results_dir),
            ("statistics", statistics_dir),
        )),
        "outputs": _output_inventory(output_dir),
    }
    manifest_path = output_dir / "figure_manifest.json"
    temporary_manifest = output_dir / ".figure_manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)

    published = {file_path.name: file_path for file_path in figure_paths}
    published[manifest_path.name] = manifest_path
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--nodouble-results-dir", type=Path, required=True)
    parser.add_argument("--statistics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    published = generate_all_figures(
        args.results_dir,
        args.nodouble_results_dir,
        args.statistics_dir,
        args.output_dir,
    )
    print(f"Published {len(published) - 1} figure files and one manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
