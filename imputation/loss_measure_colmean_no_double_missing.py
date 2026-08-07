"""Produce B1 Column Mean losses for no-double-missing splits."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from measure_loss_on_splits_colmean import (
    CONTEXTS,
    build_task_loss_rows,
    iter_split_inputs,
    validate_output,
)
from runtime_paths import (
    FULL_DATA_CSV,
    LOSS_RESULTS_DIR,
    NODOUBLE_SPLITS_DIR,
    RUN_ROOT,
)


NODOUBLE_RATES = (10, 40, 80, 99, 999)


def produce_nodouble_task_losses(
    *,
    full_data_path: Path,
    base_dir: Path,
    run_root: Path,
    targets: Sequence[str],
    rates: Sequence[int],
) -> pd.DataFrame:
    full_df = pd.read_csv(full_data_path)
    rows: list[dict[str, object]] = []
    for target in targets:
        if target not in CONTEXTS:
            raise ValueError(f"unknown target context: {target}")
        target_root = base_dir / f"tgt_{target}"
        for rate in rates:
            for split, train_path, mask_path, prediction_path in iter_split_inputs(
                split_root=target_root,
                prediction_root=target_root,
                rate=int(rate),
            ):
                rows.extend(
                    build_task_loss_rows(
                        full_df,
                        pd.read_csv(train_path),
                        pd.read_csv(mask_path),
                        pd.read_csv(prediction_path),
                        dataset="no_double",
                        rate=int(rate),
                        split=split,
                        prediction_path=prediction_path,
                        train_path=train_path,
                        mask_path=mask_path,
                        run_root=run_root,
                        targets=(target,),
                        include_within=False,
                        include_b0=False,
                    )
                )
    return validate_output(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Produce B1 task-matched Column Mean losses for no-double splits."
    )
    parser.add_argument("--full-data", type=Path, default=FULL_DATA_CSV)
    parser.add_argument("--base-dir", type=Path, default=NODOUBLE_SPLITS_DIR)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=LOSS_RESULTS_DIR / "column_mean_task_losses_no_double.csv",
    )
    parser.add_argument(
        "--target", dest="targets", nargs="+", choices=CONTEXTS, default=list(CONTEXTS)
    )
    parser.add_argument(
        "--rates", nargs="+", type=int, default=list(NODOUBLE_RATES)
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = produce_nodouble_task_losses(
        full_data_path=args.full_data,
        base_dir=args.base_dir,
        run_root=args.run_root,
        targets=args.targets,
        rates=args.rates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} no-double Column Mean B1 rows to {args.output}")


if __name__ == "__main__":
    main()
