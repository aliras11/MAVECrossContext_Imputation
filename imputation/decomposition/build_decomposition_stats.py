"""Build validated sufficient-statistics shards from raw model outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from decomposition.core import (
    ACCEPTED_CORRECTED_ROOTS,
    build_prediction_events,
    reduce_events,
)
from decomposition.layouts import (
    InputUnit,
    ModelLayout,
    discover_input_units,
    load_layouts,
    read_normalized_predictions,
)


STAT_GROUP_COLUMNS = [
    "model_family",
    "model_variant",
    "is_primary_variant",
    "rate",
    "split",
    "task",
    "b0_subtype",
    "source",
    "target",
    "variant_class",
]

METHOD_COUNT_COLUMNS = [
    *STAT_GROUP_COLUMNS,
    "prediction_method",
    "N",
]

AFFECTED_FAMILIES = frozenset({"single_ae", "dual_ae", "pca"})

FINAL_ARTIFACT_NAMES = (
    "decomposition_sufficient_stats.csv",
    "decomposition_file_validation.csv",
    "decomposition_method_counts.csv",
    "decomposition_primary_pooled.csv",
    "decomposition_pair_map_pooled.csv",
    "decomposition_reconciliation.csv",
    "decomposition_pair_map_by_split.csv",
    "decomposition_pair_map_split_summary.csv",
    "decomposition_pooled_by_split.csv",
    "decomposition_pooled_split_summary.csv",
)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False)
    temporary.replace(path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _header_signature(path: Path) -> str:
    columns = pd.read_csv(path, nrows=0).columns
    return hashlib.sha256(",".join(columns).encode("utf-8")).hexdigest()


def _is_fix_or_descendant(commit: str) -> bool:
    if commit in ACCEPTED_CORRECTED_ROOTS:
        return True
    repository_root = Path(__file__).resolve().parents[1]
    for root in ACCEPTED_CORRECTED_ROOTS:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", root, commit],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return True
    return False


def _add_unit_metadata(events: pd.DataFrame, unit: InputUnit) -> pd.DataFrame:
    events = events.copy()
    events["model_family"] = unit.layout.model_family
    events["model_variant"] = unit.layout.model_variant
    events["is_primary_variant"] = unit.layout.is_primary_variant
    events["rate"] = unit.rate
    events["split"] = unit.split
    events["source"] = (
        pd.NA if unit.source == unit.target else unit.source
    )
    events["target"] = unit.target
    return events


def _method_counts(
    events: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    if predictions["prediction_method"].isna().all() or events.empty:
        return pd.DataFrame(columns=METHOD_COUNT_COLUMNS)
    eligible_methods = events[
        ["hgvs_pro", *STAT_GROUP_COLUMNS]
    ].merge(
        predictions[["hgvs_pro", "prediction_method"]],
        on="hgvs_pro",
        how="left",
        validate="one_to_one",
    )
    counts = (
        eligible_methods.groupby(
            [*STAT_GROUP_COLUMNS, "prediction_method"],
            as_index=False,
            dropna=False,
        )
        .size()
        .rename(columns={"size": "N"})
    )
    return counts[METHOD_COUNT_COLUMNS]


def build_shard(
    *,
    run_root: Path,
    output_root: Path,
    layout: ModelLayout,
    rate: int,
    split: int,
) -> dict[str, Path]:
    """Build one model-family/variant/rate/split shard."""
    full_path = (
        run_root
        / "full_data"
        / "mthfr_crossAllcontext_domainannotation.csv"
    )
    split_input_dir = run_root / "data_splits" / f"test_frac_{rate}"
    train_path = split_input_dir / f"train_split_r{rate}_s{split}.csv"
    mask_path = split_input_dir / f"mask_r{rate}_s{split}.csv"

    full_df = pd.read_csv(full_path)
    train_df = pd.read_csv(train_path)
    mask_df = pd.read_csv(mask_path)
    units = discover_input_units(
        layout,
        run_root / "data_splits",
        rate=rate,
        split=split,
    )

    stats_frames: list[pd.DataFrame] = []
    validation_rows: list[dict] = []
    method_frames: list[pd.DataFrame] = []
    wide_file_cache: dict[Path, pd.DataFrame] = {}
    for unit in units:
        predictions = read_normalized_predictions(
            unit,
            physical_file_cache=(
                wide_file_cache
                if unit.layout.file_kind == "wide"
                else None
            ),
        )
        events, validation = build_prediction_events(
            full_df,
            train_df,
            mask_df,
            predictions,
            source=unit.source,
            target=unit.target,
            supported_tasks=layout.supported_tasks,
        )
        events = _add_unit_metadata(events, unit)
        if not events.empty:
            stats_frames.append(reduce_events(events, STAT_GROUP_COLUMNS))
        method_frames.append(_method_counts(events, predictions))
        validation_rows.append(
            {
                "model_family": layout.model_family,
                "model_variant": layout.model_variant,
                "is_primary_variant": layout.is_primary_variant,
                "rate": rate,
                "split": split,
                "source": (
                    pd.NA if unit.source == unit.target else unit.source
                ),
                "target": unit.target,
                "source_file": str(unit.path),
                "file_size": unit.path.stat().st_size,
                "header_signature": _header_signature(unit.path),
                **validation,
            }
        )

    stats = (
        pd.concat(stats_frames, ignore_index=True)
        if stats_frames
        else pd.DataFrame(
            columns=[*STAT_GROUP_COLUMNS, *ADDITIVE_COLUMNS]
        )
    )
    validation = pd.DataFrame(validation_rows)
    method_counts = (
        pd.concat(method_frames, ignore_index=True)
        if method_frames
        else pd.DataFrame(columns=METHOD_COUNT_COLUMNS)
    )

    shard_stem = (
        f"{layout.model_family}__{layout.model_variant}__r{rate}__s{split}"
    )
    shard_dir = output_root / "shards"
    outputs = {
        "stats": shard_dir / f"{shard_stem}.stats.csv",
        "validation": shard_dir / f"{shard_stem}.validation.csv",
        "method_counts": shard_dir / f"{shard_stem}.method_counts.csv",
    }
    _atomic_write_csv(stats, outputs["stats"])
    _atomic_write_csv(validation, outputs["validation"])
    _atomic_write_csv(method_counts, outputs["method_counts"])
    return outputs


def plan_tasks(
    *,
    layouts: dict[tuple[str, str], ModelLayout],
    selections: list[tuple[str, str]],
    rates: list[int],
    splits: list[int],
    generation_commit: str | None,
    development_only: bool,
) -> pd.DataFrame:
    """Expand requested model/rate/split coordinates into unique shard tasks."""
    if len(selections) != len(set(selections)):
        raise ValueError("model selections must be unique")
    unknown = [selection for selection in selections if selection not in layouts]
    if unknown:
        raise ValueError(f"unknown model layout selections: {unknown}")
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("rates must contain positive integers")
    if len(rates) != len(set(rates)):
        raise ValueError("rates must be unique")
    if not splits or any(split <= 0 for split in splits):
        raise ValueError("splits must contain positive integers")
    if len(splits) != len(set(splits)):
        raise ValueError("splits must be unique")

    uses_affected_outputs = any(
        family in AFFECTED_FAMILIES for family, _variant in selections
    )
    if uses_affected_outputs and not development_only:
        if generation_commit is None:
            raise ValueError(
                "generation_commit is required for SingleAE, DualAE, or PCA"
            )
        if not _is_fix_or_descendant(generation_commit):
            raise ValueError(
                "generation_commit must be an approved corrected provenance "
                "root or descendant: " + ", ".join(ACCEPTED_CORRECTED_ROOTS)
            )

    rows = []
    for family, variant in selections:
        layout = layouts[(family, variant)]
        for rate in rates:
            for split in splits:
                rows.append(
                    {
                        "model_family": family,
                        "model_variant": variant,
                        "is_primary_variant": layout.is_primary_variant,
                        "rate": rate,
                        "split": split,
                        "generation_commit": generation_commit,
                        "development_only": development_only,
                        "shard_stem": (
                            f"{family}__{variant}__r{rate}__s{split}"
                        ),
                    }
                )
    tasks = pd.DataFrame(rows)
    tasks.insert(0, "task_index", range(len(tasks)))
    return tasks


def finalize_shards(
    *,
    output_root: Path,
    task_plan: pd.DataFrame,
    run_root: Path,
    run_id: str,
    generation_commit: str | None,
    repository_commit: str,
    layout_manifest_path: Path,
) -> dict[str, Path]:
    """Validate and concatenate every expected shard into Stage-1 artifacts."""
    task_key_columns = [
        "model_family",
        "model_variant",
        "rate",
        "split",
    ]
    required_task_columns = {
        "task_index",
        "shard_stem",
        "development_only",
        *task_key_columns,
    }
    missing_task_columns = required_task_columns - set(task_plan.columns)
    if missing_task_columns:
        raise ValueError(
            "task plan is missing columns: "
            + ", ".join(sorted(missing_task_columns))
        )
    if task_plan.empty:
        raise ValueError("task plan is empty")
    if (
        task_plan.duplicated(task_key_columns).any()
        or task_plan["task_index"].duplicated().any()
        or task_plan["shard_stem"].duplicated().any()
    ):
        raise ValueError("task plan contains duplicate task coordinates")

    shard_dir = output_root / "shards"
    shard_paths = []
    missing = []
    for row in task_plan.itertuples(index=False):
        stem = row.shard_stem
        paths = {
            "stats": shard_dir / f"{stem}.stats.csv",
            "validation": shard_dir / f"{stem}.validation.csv",
            "method_counts": shard_dir / f"{stem}.method_counts.csv",
        }
        absent = [path for path in paths.values() if not path.is_file()]
        if absent:
            missing.append(stem)
        else:
            shard_paths.append(paths)
    if missing:
        raise ValueError(
            f"missing shard artifacts for {len(missing)} task(s): "
            + ", ".join(missing)
        )

    stats_frames = [pd.read_csv(paths["stats"]) for paths in shard_paths]
    validation_frames = [
        pd.read_csv(paths["validation"]) for paths in shard_paths
    ]
    method_frames = [
        pd.read_csv(paths["method_counts"]) for paths in shard_paths
    ]
    validation = pd.concat(validation_frames, ignore_index=True)
    if "validation_status" not in validation.columns:
        raise ValueError("shard validation is missing validation_status")
    failed = validation.loc[validation["validation_status"].ne("ok")]
    if not failed.empty:
        raise ValueError(
            f"{len(failed)} logical input(s) have failed validation"
        )

    stats = pd.concat(stats_frames, ignore_index=True)
    method_counts = pd.concat(method_frames, ignore_index=True)
    stats_sort = [column for column in STAT_GROUP_COLUMNS if column in stats]
    validation_sort = [
        column
        for column in (
            "model_family",
            "model_variant",
            "rate",
            "split",
            "source",
            "target",
        )
        if column in validation
    ]
    method_sort = [
        column
        for column in METHOD_COUNT_COLUMNS
        if column in method_counts.columns and column != "N"
    ]
    if stats_sort:
        stats = stats.sort_values(stats_sort, kind="stable").reset_index(
            drop=True
        )
    if validation_sort:
        validation = validation.sort_values(
            validation_sort, kind="stable"
        ).reset_index(drop=True)
    if method_sort and not method_counts.empty:
        method_counts = method_counts.sort_values(
            method_sort, kind="stable"
        ).reset_index(drop=True)

    outputs = {
        "sufficient_stats": output_root / "decomposition_sufficient_stats.csv",
        "file_validation": output_root / "decomposition_file_validation.csv",
        "method_counts": output_root / "decomposition_method_counts.csv",
        "run_manifest": output_root / "decomposition_run_manifest.json",
    }
    _atomic_write_csv(stats, outputs["sufficient_stats"])
    _atomic_write_csv(validation, outputs["file_validation"])
    _atomic_write_csv(method_counts, outputs["method_counts"])

    requested_models = (
        task_plan[["model_family", "model_variant"]]
        .drop_duplicates()
        .sort_values(["model_family", "model_variant"])
        .to_dict("records")
    )
    manifest_bytes = layout_manifest_path.read_bytes()
    run_manifest = {
        "run_id": run_id,
        "run_root": str(run_root.resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit,
        "generation_commit": generation_commit,
        "layout_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "rates": sorted(int(value) for value in task_plan["rate"].unique()),
        "splits": sorted(int(value) for value in task_plan["split"].unique()),
        "requested_models": requested_models,
        "development_only": bool(task_plan["development_only"].all()),
        "expected_shards": len(task_plan),
        "completed_shards": len(shard_paths),
        "status": "stage1_complete",
    }
    _atomic_write_json(run_manifest, outputs["run_manifest"])
    return outputs


def mark_run_complete(output_root: Path) -> Path:
    """Mark a run complete only after all scientific artifacts validate."""
    manifest_path = output_root / "decomposition_run_manifest.json"
    required_paths = [
        manifest_path,
        *(output_root / name for name in FINAL_ARTIFACT_NAMES),
    ]
    missing = [path.name for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(
            "missing final artifact(s): " + ", ".join(sorted(missing))
        )

    validation = pd.read_csv(
        output_root / "decomposition_file_validation.csv"
    )
    reconciliation = pd.read_csv(
        output_root / "decomposition_reconciliation.csv"
    )
    if validation["validation_status"].ne("ok").any():
        raise ValueError("cannot complete a run with failed file validation")
    if reconciliation["validation_status"].ne("ok").any():
        raise ValueError("cannot complete a run with failed reconciliation")

    run_manifest = json.loads(manifest_path.read_text())
    if run_manifest.get("status") not in {"stage1_complete", "complete"}:
        raise ValueError(
            "run manifest must have stage1_complete status before completion"
        )
    run_manifest["status"] = "complete"
    run_manifest["completed_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    _atomic_write_json(run_manifest, manifest_path)
    return manifest_path


def _parse_model_selection(value: str) -> tuple[str, str]:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError(
            "model selection must have FAMILY:VARIANT form"
        )
    return parts[0], parts[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build validated variance-decomposition statistics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--output-root", type=Path, required=True)
    plan_parser.add_argument(
        "--models",
        type=_parse_model_selection,
        nargs="+",
        required=True,
    )
    plan_parser.add_argument("--rates", type=int, nargs="+", required=True)
    plan_parser.add_argument("--splits", type=int, nargs="+", required=True)
    plan_parser.add_argument("--generation-commit")
    plan_parser.add_argument("--development-only", action="store_true")

    shard_parser = subparsers.add_parser("shard")
    shard_parser.add_argument("--manifest", type=Path, required=True)
    shard_parser.add_argument("--run-root", type=Path, required=True)
    shard_parser.add_argument("--output-root", type=Path, required=True)
    shard_parser.add_argument("--task-manifest", type=Path, required=True)
    shard_parser.add_argument("--task-index", type=int, required=True)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--manifest", type=Path, required=True)
    finalize_parser.add_argument("--run-root", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--task-manifest", type=Path, required=True)
    finalize_parser.add_argument("--run-id", required=True)
    finalize_parser.add_argument("--generation-commit")
    finalize_parser.add_argument("--repository-commit", required=True)

    complete_parser = subparsers.add_parser("complete")
    complete_parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "complete":
        print(mark_run_complete(args.output_root))
        return 0

    layouts = load_layouts(args.manifest)

    if args.command == "plan":
        tasks = plan_tasks(
            layouts=layouts,
            selections=args.models,
            rates=args.rates,
            splits=args.splits,
            generation_commit=args.generation_commit,
            development_only=args.development_only,
        )
        task_manifest = args.output_root / "decomposition_task_manifest.tsv"
        _atomic_write_tsv(tasks, task_manifest)
        print(task_manifest)
        return 0

    task_plan = pd.read_csv(args.task_manifest, sep="\t")
    if args.command == "shard":
        selected = task_plan.loc[
            task_plan["task_index"].eq(args.task_index)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"task index {args.task_index} selects {len(selected)} rows"
            )
        task = selected.iloc[0]
        layout_key = (task["model_family"], task["model_variant"])
        outputs = build_shard(
            run_root=args.run_root,
            output_root=args.output_root,
            layout=layouts[layout_key],
            rate=int(task["rate"]),
            split=int(task["split"]),
        )
        print(outputs["stats"])
        return 0

    outputs = finalize_shards(
        output_root=args.output_root,
        task_plan=task_plan,
        run_root=args.run_root,
        run_id=args.run_id,
        generation_commit=args.generation_commit,
        repository_commit=args.repository_commit,
        layout_manifest_path=args.manifest,
    )
    print(outputs["run_manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
