"""Canonical paths for the consolidated imputation runtime."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPOSITORY_ROOT.parent
FULL_DATA_CSV = REPOSITORY_ROOT / "data" / "mthfr_crossAllcontext_domainannotation.csv"
REGULAR_SPLITS_DIR = RUN_ROOT / "data_splits"
NODOUBLE_SPLITS_DIR = RUN_ROOT / "data_splits_no_double_missing"
LOSS_RESULTS_DIR = RUN_ROOT / "splits_results_0506"
