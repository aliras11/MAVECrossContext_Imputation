"""Checked descriptions of heterogeneous raw model outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import pandas as pd

from decomposition.core import CONTEXTS


REQUIRED_FIELDS = frozenset(
    {
        "model_family",
        "model_variant",
        "file_kind",
        "directory_template",
        "filename_regex",
        "key_column",
        "prediction_column_rule",
        "source_target_rule",
        "supported_tasks",
        "is_primary_variant",
    }
)
VALID_FILE_KINDS = frozenset({"pair", "wide"})
VALID_TASKS = frozenset({"B1", "B0", "W"})
VALID_SOURCE_TARGET_RULES = frozenset({"filename", "wide_contexts"})


@dataclass(frozen=True)
class ModelLayout:
    model_family: str
    model_variant: str
    file_kind: str
    directory_template: str
    filename_regex: str
    key_column: str
    prediction_column_rule: str
    source_target_rule: str
    supported_tasks: frozenset[str]
    is_primary_variant: bool
    method_column_rule: str | None = None


@dataclass(frozen=True)
class InputUnit:
    """One logical source→target or within-map prediction input."""

    layout: ModelLayout
    path: Path
    source: str
    target: str
    rate: int
    split: int


def load_layouts(path: Path) -> dict[tuple[str, str], ModelLayout]:
    """Load and validate a model-layout manifest."""
    payload = json.loads(path.read_text())
    entries = payload.get("layouts")
    if not isinstance(entries, list):
        raise ValueError("layout manifest must contain a 'layouts' list")

    layouts: dict[tuple[str, str], ModelLayout] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"layout {index} must be a JSON object")
        missing = REQUIRED_FIELDS - set(entry)
        if missing:
            raise ValueError(
                f"layout {index} is missing required fields: {sorted(missing)}"
            )
        if entry["file_kind"] not in VALID_FILE_KINDS:
            raise ValueError(f"layout {index} has invalid file_kind")
        if entry["source_target_rule"] not in VALID_SOURCE_TARGET_RULES:
            raise ValueError(f"layout {index} has invalid source_target_rule")
        if not isinstance(entry["is_primary_variant"], bool):
            raise ValueError(f"layout {index} is_primary_variant must be boolean")
        supported_tasks = frozenset(entry["supported_tasks"])
        if not supported_tasks or not supported_tasks <= VALID_TASKS:
            raise ValueError(f"layout {index} has invalid supported_tasks")

        layout = ModelLayout(
            model_family=entry["model_family"],
            model_variant=entry["model_variant"],
            file_kind=entry["file_kind"],
            directory_template=entry["directory_template"],
            filename_regex=entry["filename_regex"],
            key_column=entry["key_column"],
            prediction_column_rule=entry["prediction_column_rule"],
            method_column_rule=entry.get("method_column_rule"),
            source_target_rule=entry["source_target_rule"],
            supported_tasks=supported_tasks,
            is_primary_variant=entry["is_primary_variant"],
        )
        key = (layout.model_family, layout.model_variant)
        if key in layouts:
            raise ValueError(f"duplicate model layout for {key}")
        layouts[key] = layout
    return layouts


def discover_input_units(
    layout: ModelLayout,
    data_splits: Path,
    *,
    rate: int,
    split: int,
) -> list[InputUnit]:
    """Discover logical prediction units for one model/rate/split task."""
    directory_name = layout.directory_template.format(
        rate=rate,
        model_variant=layout.model_variant,
    )
    split_dir = data_splits / directory_name / f"split_{split}"
    if not split_dir.is_dir():
        raise FileNotFoundError(f"prediction split directory not found: {split_dir}")

    pattern = re.compile(layout.filename_regex)
    matched: list[tuple[Path, re.Match[str]]] = []
    for path in sorted(split_dir.glob("*.csv")):
        match = pattern.fullmatch(path.name)
        if match is not None:
            matched.append((path, match))
    if not matched:
        raise FileNotFoundError(
            f"no {layout.model_family}/{layout.model_variant} files found in {split_dir}"
        )

    for path, match in matched:
        groups = match.groupdict()
        if groups.get("rate") is not None and int(groups["rate"]) != rate:
            raise ValueError(
                f"filename rate does not match requested rate {rate}: {path}"
            )
        if groups.get("split") is not None and int(groups["split"]) != split:
            raise ValueError(
                f"filename split does not match requested split {split}: {path}"
            )

    if layout.file_kind == "pair":
        observed_pairs = [
            (match.group("source"), match.group("target"))
            for _path, match in matched
        ]
        if len(observed_pairs) != len(set(observed_pairs)):
            raise ValueError(
                f"{layout.model_family}/{layout.model_variant} contains "
                "duplicate source-target pairs"
            )
        include_self_pairs = "W" in layout.supported_tasks
        expected_pairs = {
            (source, target)
            for source in CONTEXTS
            for target in CONTEXTS
            if include_self_pairs or source != target
        }
        observed_pair_set = set(observed_pairs)
        missing_pairs = expected_pairs - observed_pair_set
        unexpected_pairs = observed_pair_set - expected_pairs
        if missing_pairs:
            raise ValueError(
                f"{layout.model_family}/{layout.model_variant} is missing "
                f"{len(missing_pairs)} expected source-target pair(s) in "
                f"{split_dir}"
            )
        if unexpected_pairs:
            raise ValueError(
                f"{layout.model_family}/{layout.model_variant} contains "
                f"{len(unexpected_pairs)} unexpected source-target pair(s) in "
                f"{split_dir}"
            )
        units = []
        for path, match in matched:
            groups = match.groupdict()
            units.append(
                InputUnit(
                    layout=layout,
                    path=path,
                    source=groups["source"],
                    target=groups["target"],
                    rate=rate,
                    split=split,
                )
            )
        return units

    if len(matched) != 1:
        raise ValueError(
            f"wide layout expected exactly one file in {split_dir}, found {len(matched)}"
        )
    path = matched[0][0]
    return [
        InputUnit(
            layout=layout,
            path=path,
            source=context,
            target=context,
            rate=rate,
            split=split,
        )
        for context in CONTEXTS
    ]


def read_normalized_predictions(
    unit: InputUnit,
    *,
    physical_file_cache: dict[Path, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Read one logical input as key, prediction, and optional method."""
    layout = unit.layout
    match = re.fullmatch(layout.filename_regex, unit.path.name)
    if match is not None and layout.file_kind == "pair":
        groups = match.groupdict()
        if (groups["source"], groups["target"]) != (unit.source, unit.target):
            raise ValueError(
                f"declared pair {unit.source}->{unit.target} disagrees with {unit.path}"
            )

    prediction_column = layout.prediction_column_rule.format(
        source=unit.source,
        target=unit.target,
    )
    method_column = (
        layout.method_column_rule.format(source=unit.source, target=unit.target)
        if layout.method_column_rule
        else None
    )
    required_columns = [layout.key_column, prediction_column]
    if method_column:
        required_columns.append(method_column)

    if physical_file_cache is not None:
        if unit.path not in physical_file_cache:
            physical_file_cache[unit.path] = pd.read_csv(unit.path)
        physical_frame = physical_file_cache[unit.path]
        missing = [
            column
            for column in required_columns
            if column not in physical_frame.columns
        ]
        raw = physical_frame[required_columns].copy()
    else:
        header = pd.read_csv(unit.path, nrows=0).columns
        missing = [
            column for column in required_columns if column not in header
        ]
        raw = (
            pd.read_csv(unit.path, usecols=required_columns)
            if not missing
            else None
        )
    if missing:
        raise ValueError(
            f"missing prediction column(s) {missing} in {unit.path}"
        )
    assert raw is not None
    normalized = raw.rename(
        columns={
            layout.key_column: "hgvs_pro",
            prediction_column: "prediction",
        }
    )
    normalized["prediction"] = pd.to_numeric(
        normalized["prediction"], errors="coerce"
    )
    if method_column:
        normalized = normalized.rename(
            columns={method_column: "prediction_method"}
        )
    else:
        normalized["prediction_method"] = pd.Series(
            [pd.NA] * len(normalized),
            dtype="string",
        )
    return normalized[["hgvs_pro", "prediction", "prediction_method"]]
