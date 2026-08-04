"""Validation and publication helpers for staged manuscript figures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil


FIGURE_EXTENSIONS = (".svg", ".png")
FigureMap = Mapping[tuple[str, str], str]


def validate_staged_outputs(
    staging_dir: Path,
    figure_map: FigureMap,
) -> list[Path]:
    """Return all expected staged paths, or raise before publication."""
    staging_dir = Path(staging_dir)
    expected = []
    for _, raw_stem in figure_map:
        for extension in FIGURE_EXTENSIONS:
            path = staging_dir / f"{raw_stem}{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    f"missing or empty staged figure: {path}"
                )
            expected.append(path)
    return expected


def publish_staged_outputs(
    *,
    staging_dir: Path,
    figure_map: FigureMap,
    destinations: Sequence[Path],
) -> list[Path]:
    """Validate the complete set, then publish renamed files."""
    staging_dir = Path(staging_dir)
    validate_staged_outputs(staging_dir, figure_map)
    published = []
    for destination in destinations:
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for (_, raw_stem), published_stem in figure_map.items():
            for extension in FIGURE_EXTENSIONS:
                source = staging_dir / f"{raw_stem}{extension}"
                target = destination / f"{published_stem}{extension}"
                temporary = destination / f".{target.name}.tmp"
                shutil.copy2(source, temporary)
                temporary.replace(target)
                published.append(target)
    return published
