"""Deterministic resolution of physical input files used by validation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


_MOORING_NAME_MARKERS = ("stiffness", "刚度")
_NON_MOORING_NAME_MARKERS = (
    "frequency-matrix",
    "frequency_matrix",
    "thrust force",
    "风机叶片",
    "轮毂",
)


def _as_existing_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Mooring stiffness file not found: {resolved}")
    if resolved.suffix.lower() not in {".xlsx", ".xls"}:
        raise ValueError(f"Mooring stiffness input must be an Excel file: {resolved}")
    return resolved


def _is_mooring_candidate(path: Path) -> bool:
    name = path.name.casefold()
    if name.startswith("~$"):
        return False
    if any(marker in name for marker in _NON_MOORING_NAME_MARKERS):
        return False
    return any(marker in name for marker in _MOORING_NAME_MARKERS)


def resolve_mooring_stiffness_file(
    *,
    explicit_path: str | Path | None = None,
    search_dirs: Iterable[str | Path] = (),
) -> Path:
    """Resolve one unambiguous mooring stiffness workbook.

    Explicit input always wins. Automatic discovery accepts only filenames that
    identify stiffness data and rejects wind-frequency and thrust workbooks.
    """

    if explicit_path not in (None, ""):
        return _as_existing_file(explicit_path)

    candidates: set[Path] = set()
    for directory in search_dirs:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            continue
        for pattern in ("*.xlsx", "*.xls"):
            candidates.update(
                path.resolve()
                for path in root.rglob(pattern)
                if _is_mooring_candidate(path)
            )

    ordered = sorted(candidates, key=lambda path: str(path).casefold())
    if not ordered:
        locations = ", ".join(str(Path(item)) for item in search_dirs) or "<none>"
        raise FileNotFoundError(
            "No mooring stiffness workbook was found. "
            f"Pass --stiffness-file or place one in: {locations}"
        )
    if len(ordered) > 1:
        listed = "\n".join(f"  - {path}" for path in ordered)
        raise RuntimeError(
            "Multiple mooring stiffness workbooks were found. "
            "Select one with --stiffness-file:\n"
            f"{listed}"
        )
    return ordered[0]
