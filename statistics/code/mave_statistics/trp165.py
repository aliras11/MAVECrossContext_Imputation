"""Validation, input adaptation, and additive pooling for WT Trp165."""

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np
import pandas as pd

from mave_statistics.constants import B1_MODELS, MODEL_DISPLAY_NAMES


DEFAULT_RATES = (10, 20, 40, 60, 80, 90)
DEFAULT_SPLITS = tuple(range(1, 51))
DEFAULT_DIRECTIONS = (("wt200", "wt12"), ("wt12", "wt200"))
SOURCE_METHOD_IDS = B1_MODELS
REFERENCE_METHOD_IDS = ("col_mean",)
METHOD_IDS = (*SOURCE_METHOD_IDS, *REFERENCE_METHOD_IDS)
EVENT_COLUMNS = (
    "method_id", "method", "rate", "saturation", "direction",
    "source", "target", "split", "hgvs_pro", "prediction", "truth",
    "residual", "squared_error", "source_file",
)
COLUMN_MEAN_DIAGNOSTIC_COLUMNS = (
    "rate", "split", "target", "observed_target_count", "split_position_mean",
    "whole_map_mean", "whole_map_fallback_used", "reference_value",
)
TRP165_IDENTIFIERS = frozenset({
    "p.Trp165Ala", "p.Trp165Arg", "p.Trp165Asn", "p.Trp165Asp",
    "p.Trp165Cys", "p.Trp165Gln", "p.Trp165Glu", "p.Trp165Gly",
    "p.Trp165His", "p.Trp165Ile", "p.Trp165Leu", "p.Trp165Lys",
    "p.Trp165Met", "p.Trp165Phe", "p.Trp165Pro", "p.Trp165Ser",
    "p.Trp165Thr", "p.Trp165Tyr", "p.Trp165Val", "p.Trp165Ter",
})


@dataclass(frozen=True)
class MethodLayout:
    """The exact Pitt output layout for one canonical method."""

    method_id: str
    output_id: str
    family: Literal["single_ae", "dual_ae", "mice_pmm", "mice_rf", "linear"]
    identifier_column: Literal["hgvs_pro", "hgvs"]
    prediction_kind: Literal["imputed", "completed_target"]


METHOD_LAYOUTS: Mapping[str, MethodLayout] = MappingProxyType({
    "single_ae": MethodLayout(
        "single_ae", "single_AE3", "single_ae", "hgvs_pro", "imputed"
    ),
    "dual_ae": MethodLayout(
        "dual_ae", "Dual_AE3", "dual_ae", "hgvs_pro", "imputed"
    ),
    "mice": MethodLayout(
        "mice", "mice", "mice_pmm", "hgvs_pro", "completed_target"
    ),
    "mice_rf": MethodLayout(
        "mice_rf", "mice_rf", "mice_rf", "hgvs", "completed_target"
    ),
    "basic_linear": MethodLayout(
        "basic_linear", "basic_linear", "linear", "hgvs_pro", "completed_target"
    ),
    "oneparam_linear": MethodLayout(
        "oneparam_linear", "oneparam_linear", "linear", "hgvs_pro", "completed_target"
    ),
    "full_interaction_linear": MethodLayout(
        "full_interaction_linear", "full_interaction_linear", "linear", "hgvs_pro",
        "completed_target",
    ),
    "full_interaction_mixed": MethodLayout(
        "full_interaction_mixed", "full_interaction_mixed", "linear", "hgvs_pro",
        "completed_target",
    ),
    "mixed_random": MethodLayout(
        "mixed_random", "mixed_random", "linear", "hgvs_pro", "completed_target"
    ),
})


@dataclass(frozen=True)
class InputRecord:
    """Provenance captured for one source file used by collection."""

    role: str
    method: str | None
    rate: int | None
    split: int | None
    source: str | None
    target: str | None
    path: str
    size_bytes: int
    mtime_ns: int

BY_SPLIT_COLUMNS = (
    "method_id", "method", "rate", "saturation", "direction",
    "source", "target", "split", "N", "SSE", "MSE", "RMSE", "status",
)
PRIMARY_COLUMNS = (
    "method_id", "method", "rate", "saturation", "direction",
    "source", "target", "N", "SSE", "MSE", "RMSE", "rank", "status",
)
_EVENT_COLUMNS = (
    "method_id", "rate", "split", "source", "target", "hgvs_pro",
    "prediction", "truth", "residual", "squared_error",
)
_EVENT_KEY = ("method_id", "rate", "split", "source", "target", "hgvs_pro")
_BY_SPLIT_KEY = ("method_id", "rate", "split", "source", "target")
_PRIMARY_KEY = ("method_id", "rate", "source", "target")
_COLUMN_MEAN_DIAGNOSTIC_KEY = ("rate", "split", "target")
_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}")


def prediction_path(
    run_root: Path,
    layout: MethodLayout,
    *,
    rate: int,
    split: int,
    source: str,
    target: str,
) -> Path:
    """Return the one expected Pitt prediction file for a method/event stratum."""
    root = Path(run_root) / "data_splits"
    if layout.family == "single_ae":
        return root / f"single_AE3_testfrac{rate}" / f"split_{split}" / (
            f"{source}_to_{target}_singleAE.csv"
        )
    if layout.family == "dual_ae":
        return root / f"Dual_AE3_testfrac{rate}" / f"split_{split}" / (
            f"{source}_to_{target}_DualAE.csv"
        )
    if layout.family == "mice_pmm":
        return root / f"mice_test_frac_{rate}" / f"split_{split}" / (
            f"mice_imputed_{source}_to_{target}_split{split}_rate{rate}.csv"
        )
    if layout.family == "mice_rf":
        return root / f"mice_test_rf2_frac_{rate}" / f"split_{split}" / (
            f"miceRF_imputed_{source}_to_{target}_split{split}_rate{rate}.csv"
        )
    return root / f"linear_model_output_{rate}" / f"split_{split}" / (
        f"{layout.output_id}_{source}_score_to_{target}_score_s{split}_r{rate}.csv"
    )


def column_mean_prediction_path(
    run_root: Path, *, rate: int, split: int
) -> Path:
    """Return the one completed ColumnMean file shared by both directions."""
    return (
        Path(run_root)
        / "data_splits"
        / f"mean_imputed_{rate}"
        / f"split_{split}"
        / f"mean_imputed_split{split}.csv"
    )


def _input_record(
    path: Path,
    *,
    role: str,
    method: str | None = None,
    rate: int | None = None,
    split: int | None = None,
    source: str | None = None,
    target: str | None = None,
    stat: os.stat_result | None = None,
) -> InputRecord:
    stat = path.stat() if stat is None else stat
    return InputRecord(
        role=role, method=method, rate=rate, split=split, source=source,
        target=target, path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
    )


def _read_csv_stably(
    path: Path,
    **kwargs: object,
) -> tuple[pd.DataFrame, os.stat_result]:
    before = path.stat()
    frame = pd.read_csv(path, **kwargs)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"input changed during read: {path}")
    return frame, after


def _strict_boolean(values: pd.Series, label: str) -> pd.Series:
    """Parse supported boolean serializations, rejecting ambiguous flag values."""
    if values.isna().any():
        raise ValueError(f"{label} contains null boolean values")
    parsed = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.lower() in (
            "true", "false", "0", "1", "0.0", "1.0"
        ):
            parsed.append(value.lower() in ("true", "1", "1.0"))
        else:
            raise ValueError(f"{label} contains invalid boolean values")
    return pd.Series(parsed, index=values.index, dtype=bool)


def _prediction_columns(
    layout: MethodLayout, source: str, target: str
) -> tuple[str, ...]:
    prediction_column = (
        f"{target}_imputed" if layout.prediction_kind == "imputed"
        else f"{target}_score"
    )
    columns = [layout.identifier_column, prediction_column]
    if layout.family in ("single_ae", "dual_ae"):
        columns.extend(("seen_in_training_src", "seen_in_training_tgt"))
    return tuple(columns)


def _read_prediction(
    path: Path, layout: MethodLayout, source: str, target: str
) -> tuple[pd.DataFrame, os.stat_result]:
    columns = _prediction_columns(layout, source, target)
    boolean_types = {
        column: "string"
        for column in ("seen_in_training_src", "seen_in_training_tgt")
        if column in columns
    }
    frame, stat = _read_csv_stably(
        path,
        usecols=lambda column: column in columns,
        dtype=boolean_types or None,
    )
    _require_columns(frame, columns, f"prediction file {path}")
    frame = frame.rename(columns={layout.identifier_column: "hgvs_pro"})
    _validate_identifiers(frame, f"prediction file {path}")
    return frame, stat


def _validate_ae_flags(
    prediction: pd.DataFrame,
    *,
    truth: pd.DataFrame,
    mask: pd.DataFrame,
    source: str,
    target: str,
    label: str,
) -> None:
    truth_by_id = truth.set_index("hgvs_pro")
    mask_by_id = mask.set_index("hgvs_pro")
    identifiers = truth_by_id.index
    missing = identifiers.difference(prediction.set_index("hgvs_pro").index)
    if not missing.empty:
        raise ValueError(f"{label} is missing Trp165 identifiers for AE flags")
    prediction_by_id = prediction.set_index("hgvs_pro").loc[identifiers]
    expected_source_seen = (
        mask_by_id.loc[identifiers, f"{source}_score"].eq(0)
        & np.isfinite(truth_by_id.loc[identifiers, f"{source}_score"])
    )
    expected_target_seen = (
        mask_by_id.loc[identifiers, f"{target}_score"].eq(0)
        & np.isfinite(truth_by_id.loc[identifiers, f"{target}_score"])
    )
    for column, expected in (
        ("seen_in_training_src", expected_source_seen),
        ("seen_in_training_tgt", expected_target_seen),
    ):
        actual = _strict_boolean(prediction_by_id[column], f"{label} {column}")
        if not actual.eq(expected.astype(bool)).all():
            raise ValueError(f"{label} {column} does not match mask observedness")


def _event_rows(
    prediction: pd.DataFrame,
    *,
    eligible_ids: tuple[str, ...],
    truth: pd.DataFrame,
    layout: MethodLayout,
    rate: int,
    split: int,
    source: str,
    target: str,
    path: Path,
) -> pd.DataFrame:
    prediction_by_id = prediction.set_index("hgvs_pro")
    missing = pd.Index(eligible_ids).difference(prediction_by_id.index)
    if not missing.empty:
        raise ValueError(f"prediction file {path} is missing eligible identifiers")
    prediction_column = (
        f"{target}_imputed" if layout.prediction_kind == "imputed"
        else f"{target}_score"
    )
    predicted = pd.to_numeric(
        prediction_by_id.loc[list(eligible_ids), prediction_column], errors="coerce"
    )
    if not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError(f"prediction file {path} has non-finite eligible predictions")
    truth_by_id = truth.set_index("hgvs_pro")
    observed = pd.to_numeric(
        truth_by_id.loc[list(eligible_ids), f"{target}_score"], errors="coerce"
    )
    events = pd.DataFrame({
        "method_id": layout.method_id,
        "method": MODEL_DISPLAY_NAMES[layout.method_id],
        "rate": rate,
        "split": split,
        "source": source,
        "target": target,
        "direction": f"{source}_to_{target}",
        "hgvs_pro": eligible_ids,
        "prediction": predicted.to_numpy(dtype=float),
        "truth": observed.to_numpy(dtype=float),
        "source_file": str(path),
    })
    events["residual"] = events["prediction"] - events["truth"]
    events["squared_error"] = events["residual"] ** 2
    return events


def _column_mean_event_rows(
    completed: pd.DataFrame,
    *,
    eligible_ids: tuple[str, ...],
    truth: pd.DataFrame,
    rate: int,
    split: int,
    source: str,
    target: str,
    path: Path,
) -> pd.DataFrame:
    completed_by_id = completed.set_index("hgvs_pro")
    missing = pd.Index(eligible_ids).difference(completed_by_id.index)
    if not missing.empty:
        raise ValueError(f"ColumnMean file {path} is missing eligible identifiers")
    predicted = pd.to_numeric(
        completed_by_id.loc[list(eligible_ids), f"{target}_score"],
        errors="coerce",
    )
    if not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError(
            f"ColumnMean file {path} has non-finite eligible predictions"
        )
    truth_by_id = truth.set_index("hgvs_pro")
    observed = pd.to_numeric(
        truth_by_id.loc[list(eligible_ids), f"{target}_score"], errors="coerce"
    )
    events = pd.DataFrame({
        "method_id": "col_mean",
        "method": MODEL_DISPLAY_NAMES["col_mean"],
        "rate": rate,
        "split": split,
        "source": source,
        "target": target,
        "direction": f"{source}_to_{target}",
        "hgvs_pro": eligible_ids,
        "prediction": predicted.to_numpy(dtype=float),
        "truth": observed.to_numpy(dtype=float),
        "source_file": str(path),
    })
    events["residual"] = events["prediction"] - events["truth"]
    events["squared_error"] = events["residual"] ** 2
    return events


def _column_mean_diagnostic(
    full_truth: pd.DataFrame,
    trp165_truth: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    rate: int,
    split: int,
    target: str,
) -> dict[str, object]:
    """Return the observed-position or whole-map ColumnMean reference audit."""
    target_column = f"{target}_score"
    _require_columns(full_truth, ("hgvs_pro", target_column), "truth")
    _require_columns(trp165_truth, ("hgvs_pro", target_column), "Trp165 truth")
    _require_columns(mask, ("hgvs_pro", target_column), "mask")
    _validate_identifiers(full_truth, "truth")
    _validate_identifiers(trp165_truth, "Trp165 truth")
    _validate_identifiers(mask, "mask")

    full_truth_by_id = full_truth.set_index("hgvs_pro")
    trp165_truth_by_id = trp165_truth.set_index("hgvs_pro")
    mask_by_id = mask.set_index("hgvs_pro")
    missing = full_truth_by_id.index.difference(mask_by_id.index)
    if not missing.empty:
        raise ValueError("mask is missing full truth identifiers")
    target_mask = _binary_mask_values(mask_by_id, (target_column,))[target_column]

    target_truth = pd.to_numeric(
        trp165_truth_by_id[target_column], errors="coerce"
    )
    finite_target_truth = pd.Series(
        np.isfinite(target_truth.to_numpy(dtype=float)),
        index=target_truth.index,
    )
    observed = (
        finite_target_truth
        & target_mask.loc[trp165_truth_by_id.index].eq(0)
    )
    position_mean = target_truth.loc[observed].mean()

    full_target_truth = pd.to_numeric(
        full_truth_by_id[target_column], errors="coerce"
    )
    finite_full_truth = pd.Series(
        np.isfinite(full_target_truth.to_numpy(dtype=float)),
        index=full_target_truth.index,
    )
    whole_map_observed = (
        finite_full_truth
        & target_mask.loc[full_truth_by_id.index].eq(0)
    )
    whole_map_mean = full_target_truth.loc[whole_map_observed].mean()
    fallback_used = not bool(observed.any())
    reference_value = whole_map_mean if fallback_used else position_mean

    return {
        "rate": rate,
        "split": split,
        "target": target,
        "observed_target_count": int(observed.sum()),
        "split_position_mean": float(position_mean),
        "whole_map_mean": float(whole_map_mean),
        "whole_map_fallback_used": fallback_used,
        "reference_value": float(reference_value),
    }


def _validate_column_mean_reference(
    completed: pd.DataFrame,
    trp165_truth: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    target: str,
    reference_value: float,
    path: Path,
) -> None:
    if not np.isfinite(reference_value):
        raise ValueError(f"ColumnMean reference value is non-finite: {path}")
    target_column = f"{target}_score"
    completed_by_id = completed.set_index("hgvs_pro")
    trp165_ids = pd.Index(trp165_truth["hgvs_pro"])
    mask_by_id = mask.set_index("hgvs_pro")
    target_mask = _binary_mask_values(mask_by_id, (target_column,))[target_column]
    masked_ids = trp165_ids[target_mask.loc[trp165_ids].eq(1).to_numpy()]
    missing = masked_ids.difference(completed_by_id.index)
    if not missing.empty:
        raise ValueError(
            f"ColumnMean file {path} is missing target-masked Trp165 identifiers"
        )
    completed_values = pd.to_numeric(
        completed_by_id.loc[masked_ids, target_column], errors="coerce"
    )
    finite = np.isfinite(completed_values.to_numpy(dtype=float))
    if finite.any() and not np.allclose(
        completed_values.to_numpy(dtype=float)[finite],
        reference_value,
    ):
        raise ValueError(
            f"ColumnMean file {path} does not match the expected reference value"
        )


def _reconcile_conceptual_keys(
    events: pd.DataFrame, method_ids: Sequence[str]
) -> None:
    for _, stratum in events.groupby(["rate", "split", "source", "target"], sort=False):
        keys = stratum.groupby("method_id", sort=False)["hgvs_pro"].agg(tuple)
        if len(keys) != len(method_ids) or keys.nunique() != 1:
            raise ValueError("methods do not share the same conceptual event keys")


def _event_reconciliation_audit(
    events: pd.DataFrame,
    method_ids: Sequence[str],
    expected_strata: Sequence[tuple[int, int, str, str, int]],
) -> dict[str, object]:
    """Return evidence that every method shares each requested event stratum."""
    strata = []
    for rate, split, source, target, expected_count in sorted(
        expected_strata, key=lambda values: (values[0], f"{values[2]}_to_{values[3]}", values[1])
    ):
        stratum = events.loc[
            (events["rate"] == rate)
            & (events["split"] == split)
            & (events["source"] == source)
            & (events["target"] == target)
        ]
        event_keys = {
            method_id: tuple(stratum.loc[
                stratum["method_id"] == method_id, "hgvs_pro"
            ])
            for method_id in method_ids
        }
        observed_counts = {
            method_id: len(event_keys[method_id]) for method_id in method_ids
        }
        matching_event_keys = len(set(event_keys.values())) == 1
        observed_method_ids = set(stratum["method_id"])
        observed_methods = [
            method_id for method_id in method_ids if method_id in observed_method_ids
        ]
        methods_match = (
            observed_method_ids == set(method_ids)
            if expected_count > 0
            else not observed_method_ids
        )
        if (
            not methods_match
            or any(count != expected_count for count in observed_counts.values())
            or not matching_event_keys
        ):
            raise ValueError("methods do not reconcile with the expected event universe")
        strata.append({
            "rate": rate,
            "split": split,
            "direction": f"{source}_to_{target}",
            "source": source,
            "target": target,
            "expected_count": expected_count,
            "observed_counts": observed_counts,
            "method_agreement": {
                "status": "ok",
                "expected_methods": list(method_ids),
                "requested_methods": list(method_ids),
                "observed_methods": observed_methods,
                "matching_event_keys": matching_event_keys,
            },
        })
    return {"status": "ok", "strata": strata}


def collect_trp165_events(
    run_root: Path,
    *,
    rates: Sequence[int] = DEFAULT_RATES,
    splits: Sequence[int] = DEFAULT_SPLITS,
    directions: Sequence[tuple[str, str]] = DEFAULT_DIRECTIONS,
    method_ids: Sequence[str] = METHOD_IDS,
) -> tuple[pd.DataFrame, list[InputRecord], dict[str, object]]:
    """Load and validate every requested Trp165 event and prediction file."""
    method_ids, rates, splits, directions = _validate_requested_values(
        method_ids=method_ids, rates=rates, splits=splits, directions=directions
    )
    root = Path(run_root)
    truth_path = root / "full_data/mthfr_crossAllcontext_domainannotation.csv"
    truth, truth_stat = _read_csv_stably(truth_path)
    _require_columns(truth, ("hgvs_pro", "aa_pos", "wt200_score", "wt12_score"), "truth")
    _validate_identifiers(truth, "truth")
    full_truth = truth
    trp165_truth = full_truth.loc[full_truth["aa_pos"] == 165].copy()
    if len(trp165_truth) != 20:
        raise ValueError(
            f"expected 20 Trp165 truth rows, found {len(trp165_truth)}"
        )
    if frozenset(trp165_truth["hgvs_pro"]) != TRP165_IDENTIFIERS:
        raise ValueError("Trp165 truth identities do not match the required 20 outcomes")
    inputs = [_input_record(truth_path, role="truth", stat=truth_stat)]
    frames = []
    expected_strata = []
    diagnostics = []
    include_column_mean = "col_mean" in method_ids
    source_method_ids = tuple(
        method_id for method_id in method_ids if method_id in SOURCE_METHOD_IDS
    )
    for rate in rates:
        for split in splits:
            mask_path = root / "data_splits" / f"test_frac_{rate}" / f"mask_r{rate}_s{split}.csv"
            mask, mask_stat = _read_csv_stably(mask_path)
            inputs.append(_input_record(
                mask_path, role="mask", rate=rate, split=split, stat=mask_stat
            ))
            column_mean_path = column_mean_prediction_path(
                root, rate=rate, split=split
            )
            completed = None
            if include_column_mean:
                if not column_mean_path.is_file():
                    raise FileNotFoundError(
                        f"required ColumnMean prediction file is missing: "
                        f"{column_mean_path}"
                    )
                completed, completed_stat = _read_csv_stably(column_mean_path)
                _require_columns(
                    completed,
                    ("hgvs_pro", "wt200_score", "wt12_score"),
                    f"ColumnMean file {column_mean_path}",
                )
                _validate_identifiers(
                    completed, f"ColumnMean file {column_mean_path}"
                )
                if frozenset(completed["hgvs_pro"]) != frozenset(
                    full_truth["hgvs_pro"]
                ):
                    raise ValueError(
                        f"ColumnMean file {column_mean_path} has an incomplete "
                        "or unexpected identifier set"
                    )
                inputs.append(_input_record(
                    column_mean_path,
                    role="prediction",
                    method="col_mean",
                    rate=rate,
                    split=split,
                    stat=completed_stat,
                ))
            for source, target in directions:
                eligible_ids = eligible_b1_ids(
                    trp165_truth,
                    mask,
                    source=source,
                    target=target,
                    expected_position_count=20,
                )
                expected_strata.append((rate, split, source, target, len(eligible_ids)))
                if completed is not None:
                    diagnostic = _column_mean_diagnostic(
                        full_truth,
                        trp165_truth,
                        mask,
                        rate=rate,
                        split=split,
                        target=target,
                    )
                    diagnostics.append(diagnostic)
                    _validate_column_mean_reference(
                        completed,
                        trp165_truth,
                        mask,
                        target=target,
                        reference_value=float(diagnostic["reference_value"]),
                        path=column_mean_path,
                    )
                for method_id in source_method_ids:
                    layout = METHOD_LAYOUTS[method_id]
                    path = prediction_path(
                        root, layout, rate=rate, split=split, source=source, target=target
                    )
                    if not path.is_file():
                        raise FileNotFoundError(f"required prediction file is missing: {path}")
                    prediction, prediction_stat = _read_prediction(
                        path, layout, source, target
                    )
                    if layout.family in ("single_ae", "dual_ae"):
                        _validate_ae_flags(
                            prediction,
                            truth=trp165_truth,
                            mask=mask,
                            source=source,
                            target=target,
                            label=f"prediction file {path}",
                        )
                    frames.append(_event_rows(
                        prediction,
                        eligible_ids=eligible_ids,
                        truth=trp165_truth,
                        layout=layout,
                        rate=rate,
                        split=split,
                        source=source,
                        target=target,
                        path=path,
                    ))
                    inputs.append(_input_record(
                        path, role="prediction", method=method_id, rate=rate, split=split,
                        source=source, target=target, stat=prediction_stat,
                    ))
                if completed is not None:
                    frames.append(_column_mean_event_rows(
                        completed,
                        eligible_ids=eligible_ids,
                        truth=trp165_truth,
                        rate=rate,
                        split=split,
                        source=source,
                        target=target,
                        path=column_mean_path,
                    ))
    events = pd.concat(frames, ignore_index=True)
    _validate_events(events)
    _reconcile_conceptual_keys(events, method_ids)
    events = events.sort_values(
        ["rate", "direction", "split", "hgvs_pro", "method_id"], kind="stable"
    ).reset_index(drop=True)
    reconciliation = _event_reconciliation_audit(events, method_ids, expected_strata)
    return events, inputs, {
        "trp165_position_count": len(trp165_truth),
        "rates": list(rates),
        "splits": list(splits),
        "directions": [list(direction) for direction in directions],
        "method_ids": list(method_ids),
        "column_mean_diagnostics": pd.DataFrame(
            diagnostics, columns=COLUMN_MEAN_DIAGNOSTIC_COLUMNS
        ),
        "event_reconciliation": reconciliation,
    }


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def _validate_identifiers(frame: pd.DataFrame, label: str) -> None:
    identifiers = frame["hgvs_pro"]
    if identifiers.isna().any():
        raise ValueError(f"{label} has null hgvs_pro identifiers")
    if identifiers.duplicated().any():
        raise ValueError(f"{label} has duplicate hgvs_pro identifiers")


def _binary_mask_values(mask: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    converted = mask.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = converted.to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.isin(numeric, (0, 1)).all():
        raise ValueError("mask values must be numeric binary values in {0, 1}")
    return converted


def _validate_requested_integer_subset(
    values: Sequence[int],
    *,
    label: str,
    allowed: Sequence[int],
) -> tuple[int, ...]:
    selected = tuple(values)
    if not selected:
        raise ValueError(f"at least one {label} is required")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in selected):
        raise ValueError(f"{label} values must be integers")
    if len(set(selected)) != len(selected):
        raise ValueError(f"duplicate {label} requested")
    if not set(selected).issubset(allowed):
        raise ValueError(f"unknown {label} requested")
    return selected


def validate_requested_rates(rates: Sequence[int]) -> tuple[int, ...]:
    """Return requested missingness rates after exact supported-subset validation."""
    return _validate_requested_integer_subset(
        rates, label="rate", allowed=DEFAULT_RATES
    )


def validate_requested_splits(splits: Sequence[int]) -> tuple[int, ...]:
    """Return requested split numbers after exact supported-subset validation."""
    return _validate_requested_integer_subset(
        splits, label="split", allowed=DEFAULT_SPLITS
    )


def _normalize_safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source checksum path must be a nonempty relative path")
    windows_path = PureWindowsPath(value)
    portable_value = value.replace("\\", "/")
    path = PurePosixPath(portable_value)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in path.parts
        or path == PurePosixPath(".")
    ):
        raise ValueError(f"unsafe source checksum path: {value}")
    return path.as_posix()


def validate_source_checksums(
    source_checksums: Mapping[str, str] | None,
) -> dict[str, str]:
    """Return canonical safe relative source paths mapped to 64-hex digests."""
    validated: dict[str, str] = {}
    for raw_path, digest in (source_checksums or {}).items():
        path = _normalize_safe_relative_path(raw_path)
        if path in validated:
            raise ValueError(f"duplicate source checksum path: {path}")
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"invalid SHA-256 digest for source path: {path}")
        validated[path] = digest.lower()
    return dict(sorted(validated.items()))


def eligible_b1_ids(
    truth: pd.DataFrame,
    mask: pd.DataFrame,
    *,
    source: str,
    target: str,
    expected_position_count: int | None = 20,
) -> tuple[str, ...]:
    """Return sorted Trp165 B1 IDs using mask and finite truth only."""
    if (source, target) not in DEFAULT_DIRECTIONS:
        raise ValueError("unsupported WT direction")
    source_column = f"{source}_score"
    target_column = f"{target}_score"
    _require_columns(
        truth, ("hgvs_pro", "aa_pos", source_column, target_column), "truth"
    )
    _require_columns(mask, ("hgvs_pro", source_column, target_column), "mask")
    _validate_identifiers(truth, "truth")
    _validate_identifiers(mask, "mask")

    trp165 = truth.loc[truth["aa_pos"] == 165].copy()
    if expected_position_count is not None and len(trp165) != expected_position_count:
        raise ValueError(
            f"expected {expected_position_count} Trp165 truth rows, found {len(trp165)}"
        )

    mask_by_id = mask.set_index("hgvs_pro")
    missing = trp165.loc[~trp165["hgvs_pro"].isin(mask_by_id.index), "hgvs_pro"]
    if not missing.empty:
        raise ValueError("mask is missing Trp165 truth identifiers")
    mask_values = _binary_mask_values(mask_by_id, (source_column, target_column))
    aligned_mask = mask_values.loc[trp165["hgvs_pro"]].reset_index(drop=True)

    source_truth = pd.to_numeric(trp165[source_column], errors="coerce").to_numpy()
    target_truth = pd.to_numeric(trp165[target_column], errors="coerce").to_numpy()
    eligible = (
        (aligned_mask[target_column].to_numpy() == 1)
        & (aligned_mask[source_column].to_numpy() == 0)
        & np.isfinite(source_truth)
        & np.isfinite(target_truth)
    )
    return tuple(sorted(trp165.loc[eligible, "hgvs_pro"].tolist()))


def _validate_requested_values(
    *,
    method_ids: Sequence[str],
    rates: Sequence[int],
    splits: Sequence[int],
    directions: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], tuple[tuple[str, str], ...]]:
    selected_methods = tuple(method_ids)
    selected_rates = validate_requested_rates(rates)
    selected_splits = validate_requested_splits(splits)
    selected_directions = tuple(tuple(direction) for direction in directions)
    selections = (
        ("method", selected_methods, METHOD_IDS),
        ("direction", selected_directions, DEFAULT_DIRECTIONS),
    )
    for label, selected, allowed in selections:
        if not selected:
            raise ValueError(f"at least one {label} is required")
        if len(set(selected)) != len(selected):
            raise ValueError(f"duplicate {label} requested")
        if not set(selected).issubset(allowed):
            raise ValueError(f"unknown {label} requested")
    return selected_methods, selected_rates, selected_splits, selected_directions


def _validate_events(events: pd.DataFrame) -> None:
    _require_columns(events, _EVENT_COLUMNS, "events")
    if events.loc[:, _EVENT_KEY].isna().any().any():
        raise ValueError("events have null unique-key values")
    if events.duplicated(list(_EVENT_KEY)).any():
        raise ValueError("events have duplicate unique keys")
    numeric_columns = ("prediction", "truth", "residual", "squared_error")
    numeric = events.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("events require finite prediction, truth, residual, and squared_error")
    prediction = numeric["prediction"].to_numpy(dtype=float)
    truth = numeric["truth"].to_numpy(dtype=float)
    residual = numeric["residual"].to_numpy(dtype=float)
    squared_error = numeric["squared_error"].to_numpy(dtype=float)
    if not np.allclose(residual, prediction - truth, rtol=1e-10, atol=1e-12):
        raise ValueError("events have inconsistent residual values")
    if not np.allclose(squared_error, residual ** 2, rtol=1e-10, atol=1e-12):
        raise ValueError("events have inconsistent squared_error values")


def aggregate_event_outputs(
    events: pd.DataFrame,
    *,
    method_ids: Sequence[str] = METHOD_IDS,
    rates: Sequence[int] = DEFAULT_RATES,
    splits: Sequence[int] = DEFAULT_SPLITS,
    directions: Sequence[tuple[str, str]] = DEFAULT_DIRECTIONS,
    rank_method_ids: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete split grid and additive primary RMSE table."""
    method_ids, rates, splits, directions = _validate_requested_values(
        method_ids=method_ids, rates=rates, splits=splits, directions=directions
    )
    ranked_method_ids = (
        method_ids if rank_method_ids is None else tuple(rank_method_ids)
    )
    if len(set(ranked_method_ids)) != len(ranked_method_ids):
        raise ValueError("duplicate rank method requested")
    if not set(ranked_method_ids).issubset(method_ids):
        raise ValueError("rank method must be in the requested method grid")
    _validate_events(events)

    direction_frame = pd.DataFrame(directions, columns=["source", "target"])
    grid = (
        pd.MultiIndex.from_product(
            [method_ids, rates, splits], names=["method_id", "rate", "split"]
        )
        .to_frame(index=False)
        .merge(direction_frame, how="cross")
    )
    grid["method"] = grid["method_id"].map(MODEL_DISPLAY_NAMES)
    grid["saturation"] = 100 - grid["rate"]
    grid["direction"] = grid["source"] + "_to_" + grid["target"]

    allowed = pd.MultiIndex.from_frame(grid[["method_id", "rate", "split", "source", "target"]])
    observed = pd.MultiIndex.from_frame(events[["method_id", "rate", "split", "source", "target"]])
    if not observed.isin(allowed).all():
        raise ValueError("events fall outside the requested method/rate/split/direction grid")

    pooled = (
        events.groupby(["method_id", "rate", "split", "source", "target"], as_index=False, sort=True)
        .agg(N=("hgvs_pro", "size"), SSE=("squared_error", "sum"))
    )
    by_split = grid.merge(
        pooled,
        on=["method_id", "rate", "split", "source", "target"],
        how="left",
    )
    by_split["N"] = by_split["N"].fillna(0).astype(int)
    positive_split = by_split["N"] > 0
    by_split["SSE"] = pd.to_numeric(by_split["SSE"], errors="coerce").where(
        positive_split, np.nan
    )
    by_split["MSE"] = pd.Series(np.nan, index=by_split.index, dtype=float)
    by_split.loc[positive_split, "MSE"] = (
        by_split.loc[positive_split, "SSE"] / by_split.loc[positive_split, "N"]
    )
    by_split["RMSE"] = np.sqrt(by_split["MSE"])
    by_split["status"] = np.where(positive_split, "ok", "no_eligible_event")
    by_split = by_split.sort_values(
        ["method_id", "rate", "source", "target", "split"], kind="stable"
    ).reset_index(drop=True)

    primary = (
        by_split.assign(_sse_for_sum=by_split["SSE"].fillna(0.0))
        .groupby(["method_id", "method", "rate", "saturation", "direction", "source", "target"], as_index=False, sort=True)
        .agg(N=("N", "sum"), SSE=("_sse_for_sum", "sum"))
    )
    positive_primary = primary["N"] > 0
    primary["SSE"] = pd.to_numeric(primary["SSE"], errors="coerce").where(
        positive_primary, np.nan
    )
    primary["MSE"] = pd.Series(np.nan, index=primary.index, dtype=float)
    primary.loc[positive_primary, "MSE"] = (
        primary.loc[positive_primary, "SSE"] / primary.loc[positive_primary, "N"]
    )
    primary["RMSE"] = np.sqrt(primary["MSE"])
    primary["status"] = np.where(positive_primary, "ok", "no_eligible_event")
    primary["rank"] = pd.Series(pd.NA, index=primary.index, dtype="Int64")
    rankable = positive_primary & primary["method_id"].isin(ranked_method_ids)
    ranked = (
        primary.loc[rankable]
        .groupby(["rate", "direction"])["RMSE"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )
    primary.loc[ranked.index, "rank"] = ranked
    primary = primary.sort_values(
        ["method_id", "rate", "source", "target"], kind="stable"
    ).reset_index(drop=True)
    return by_split.loc[:, BY_SPLIT_COLUMNS], primary.loc[:, PRIMARY_COLUMNS]


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_snapshot(path: Path) -> dict[str, int | str]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"input changed during checksum: {path}")
    return {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest,
    }


def _manifest_input_record(
    record: InputRecord,
    *,
    include_sha256: bool,
    expected_snapshot: Mapping[str, int | str] | None = None,
) -> dict[str, object]:
    path = Path(record.path)
    if expected_snapshot is not None:
        current_snapshot = _input_snapshot(path)
        if current_snapshot != expected_snapshot:
            raise RuntimeError(f"input changed during analysis: {path}")
        size_bytes = int(expected_snapshot["size_bytes"])
        mtime_ns = int(expected_snapshot["mtime_ns"])
    else:
        stat = path.stat()
        size_bytes = stat.st_size
        mtime_ns = stat.st_mtime_ns
    if size_bytes != record.size_bytes or mtime_ns != record.mtime_ns:
        raise RuntimeError(f"input changed during analysis: {path}")
    payload = asdict(record)
    if include_sha256:
        if expected_snapshot is None:
            raise ValueError("checksummed inputs require a pre-analysis snapshot")
        payload["sha256"] = str(expected_snapshot["sha256"])
    return payload


def _validate_input_coverage(
    inputs: Sequence[InputRecord],
    *,
    rates: Sequence[int],
    splits: Sequence[int],
) -> tuple[InputRecord, list[InputRecord], list[InputRecord], dict[str, int]]:
    full_data = [record for record in inputs if record.role == "truth"]
    masks = [record for record in inputs if record.role == "mask"]
    predictions = [record for record in inputs if record.role == "prediction"]
    expected_masks = len(rates) * len(splits)
    expected_predictions = (
        (len(SOURCE_METHOD_IDS) * len(DEFAULT_DIRECTIONS) + 1)
        * len(rates)
        * len(splits)
    )
    coverage = {
        "expected_full_data_files": 1,
        "observed_full_data_files": len(full_data),
        "expected_mask_files": expected_masks,
        "observed_mask_files": len(masks),
        "expected_prediction_files": expected_predictions,
        "observed_prediction_files": len(predictions),
    }
    if (
        coverage["observed_full_data_files"] != coverage["expected_full_data_files"]
        or coverage["observed_mask_files"] != coverage["expected_mask_files"]
        or coverage["observed_prediction_files"]
        != coverage["expected_prediction_files"]
    ):
        raise ValueError("input coverage does not match the requested analysis grid")
    for label, records in (
        ("mask", masks),
        ("prediction", predictions),
    ):
        paths = [record.path for record in records]
        if len(set(paths)) != len(paths):
            raise ValueError(f"duplicate {label} input paths")
    return full_data[0], masks, predictions, coverage


def _write_and_validate_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    columns: Sequence[str],
    key: Sequence[str],
) -> None:
    if tuple(frame.columns) != tuple(columns):
        raise ValueError(f"{path.name} does not have the required schema")
    if frame.duplicated(list(key)).any():
        raise ValueError(f"{path.name} has duplicate key rows")
    frame.to_csv(path, index=False)
    round_trip = pd.read_csv(path)
    if tuple(round_trip.columns) != tuple(columns):
        raise ValueError(f"{path.name} schema changed during CSV round trip")
    if len(round_trip) != len(frame):
        raise ValueError(f"{path.name} row count changed during CSV round trip")
    if round_trip.duplicated(list(key)).any():
        raise ValueError(f"{path.name} key uniqueness changed during CSV round trip")


def run_trp165_analysis(
    run_root: Path,
    output_dir: Path,
    *,
    rates: Sequence[int] = DEFAULT_RATES,
    splits: Sequence[int] = DEFAULT_SPLITS,
    repo_commit: str | None = None,
    source_checksums: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Run the requested analysis and atomically publish one immutable bundle."""
    root = Path(run_root)
    destination = Path(output_dir)
    if os.path.lexists(destination):
        raise FileExistsError(f"output directory already exists: {destination}")
    selected_rates = validate_requested_rates(rates)
    selected_splits = validate_requested_splits(splits)
    validated_source_checksums = validate_source_checksums(source_checksums)
    started_at = datetime.now(timezone.utc).isoformat()
    audited_input_paths = [
        root / "full_data/mthfr_crossAllcontext_domainannotation.csv",
        *[
            root / "data_splits" / f"test_frac_{rate}" / f"mask_r{rate}_s{split}.csv"
            for rate in selected_rates
            for split in selected_splits
        ],
    ]
    input_snapshots = {
        str(path): _input_snapshot(path) for path in audited_input_paths
    }

    events, inputs, audit = collect_trp165_events(
        root,
        rates=selected_rates,
        splits=selected_splits,
        directions=DEFAULT_DIRECTIONS,
        method_ids=METHOD_IDS,
    )
    by_split, primary = aggregate_event_outputs(
        events,
        rates=selected_rates,
        splits=selected_splits,
        directions=DEFAULT_DIRECTIONS,
        method_ids=METHOD_IDS,
        rank_method_ids=SOURCE_METHOD_IDS,
    )
    diagnostics = audit["column_mean_diagnostics"]
    event_output = events.assign(saturation=100 - events["rate"]).loc[
        :, EVENT_COLUMNS
    ]
    full_data, masks, predictions, coverage = _validate_input_coverage(
        inputs,
        rates=selected_rates,
        splits=selected_splits,
    )

    filenames = {
        "events": "trp165_wt_extreme_events.csv",
        "by_split": "trp165_wt_extreme_by_split.csv",
        "primary": "trp165_wt_extreme_primary.csv",
        "diagnostics": "trp165_wt_extreme_column_mean_diagnostics.csv",
        "manifest": "trp165_wt_extreme_manifest.json",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-",
        dir=destination.parent,
    ) as temporary_directory:
        staging_dir = Path(temporary_directory)
        staged_paths = {
            name: staging_dir / filename for name, filename in filenames.items()
        }
        for name, frame, columns, key in (
            ("events", event_output, EVENT_COLUMNS, _EVENT_KEY),
            ("by_split", by_split, BY_SPLIT_COLUMNS, _BY_SPLIT_KEY),
            ("primary", primary, PRIMARY_COLUMNS, _PRIMARY_KEY),
            (
                "diagnostics",
                diagnostics,
                COLUMN_MEAN_DIAGNOSTIC_COLUMNS,
                _COLUMN_MEAN_DIAGNOSTIC_KEY,
            ),
        ):
            _write_and_validate_csv(
                frame,
                staged_paths[name],
                columns=columns,
                key=key,
            )

        artifact_sha256 = {
            staged_paths[name].name: sha256_file(staged_paths[name])
            for name in ("events", "by_split", "primary", "diagnostics")
        }
        manifest = {
            "schema_version": 2,
            "status": "complete",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_root": str(root),
            "output_dir": str(destination),
            "arguments": {
                "rates": list(selected_rates),
                "splits": list(selected_splits),
                "directions": [list(direction) for direction in DEFAULT_DIRECTIONS],
                "method_ids": list(METHOD_IDS),
            },
            "repo_commit": repo_commit,
            "source_checksums": validated_source_checksums,
            "full_data_input": _manifest_input_record(
                full_data,
                include_sha256=True,
                expected_snapshot=input_snapshots[full_data.path],
            ),
            "mask_inputs": [
                _manifest_input_record(
                    record,
                    include_sha256=True,
                    expected_snapshot=input_snapshots[record.path],
                )
                for record in masks
            ],
            "prediction_inputs": [
                _manifest_input_record(record, include_sha256=False)
                for record in predictions
            ],
            "coverage": coverage,
            "event_reconciliation": audit["event_reconciliation"],
            "artifact_sha256": artifact_sha256,
        }
        staged_paths["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

        if os.path.lexists(destination):
            raise FileExistsError(f"output directory already exists: {destination}")
        os.replace(staging_dir, destination)

    return {
        name: destination / filename for name, filename in filenames.items()
    }
