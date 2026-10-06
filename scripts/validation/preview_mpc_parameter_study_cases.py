"""Load the frozen wind-only case registry for preview-MPC studies.

The registry predates parameter selection.  This adapter only verifies and
converts its records; it does not inspect controller results or select cases.
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from preview_mpc_parameter_study_protocol import ParameterStudyCase


_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_STAGE_FILES = {
    "stage10": "stage10_cases.csv",
    "stage20": "stage20_cases.csv",
    "stage30": "stage30_cases.csv",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("selection manifest must be a JSON object")
    return payload


def _read_stage_rows(path: Path, stage_name: str) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path.name} must contain cases")
    for row in rows:
        if row.get("stage") != stage_name:
            raise ValueError(f"{path.name} contains a non-{stage_name} row")
        if "selection_inputs=wind_only" not in str(row.get("label", "")):
            raise ValueError(f"{path.name} is not marked as wind-only selection")
        try:
            start = datetime.strptime(str(row["timestamp"]), _TIMESTAMP_FORMAT)
            end = datetime.strptime(str(row["interval_end"]), _TIMESTAMP_FORMAT)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path.name} has an invalid time interval") from exc
        if (end - start).total_seconds() != 21600.0:
            raise ValueError(f"{path.name} case duration must be six hours")
        selection_hash = str(row.get("selection_hash", ""))
        if len(selection_hash) != 64 or any(
            character not in "0123456789abcdef" for character in selection_hash
        ):
            raise ValueError(f"{path.name} has an invalid selection hash")
    return rows


def load_frozen_staged_cases(
    *,
    repository_root: Path,
) -> tuple[dict[str, tuple[ParameterStudyCase, ...]], dict[str, Any]]:
    """Return frozen stage cases and their source-bound registry identity."""

    directory = repository_root / "configs" / "validation_cases_v2"
    manifest_path = directory / "selection_manifest.json"
    combined_path = directory / "all_staged_cases.csv"
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported staged-case manifest schema")
    if manifest.get("case_duration_s") != 21600:
        raise ValueError("staged-case manifest must specify six-hour cases")
    if manifest.get("controller_outputs_used") is not False:
        raise ValueError("staged-case registry must not use controller outputs")
    if _sha256_file(combined_path) != manifest.get("combined_csv_sha256"):
        raise ValueError("all_staged_cases.csv does not match the frozen manifest")

    cases_by_stage: dict[str, tuple[ParameterStudyCase, ...]] = {}
    stage_file_hashes: dict[str, str] = {}
    staged_rows: list[dict[str, str]] = []
    for stage_name, filename in _STAGE_FILES.items():
        path = directory / filename
        rows = _read_stage_rows(path, stage_name)
        expected_count = int(manifest["stage_counts"][stage_name])
        if len(rows) != expected_count:
            raise ValueError(f"{stage_name} case count differs from manifest")
        category_counts = Counter(str(row["category"]) for row in rows)
        if dict(sorted(category_counts.items())) != dict(
            sorted(manifest["category_counts"][stage_name].items())
        ):
            raise ValueError(f"{stage_name} category counts differ from manifest")
        staged_rows.extend(rows)
        cases_by_stage[stage_name] = tuple(
            ParameterStudyCase(
                case_id=str(row["case_id"]),
                origin_time=str(row["timestamp"]),
                source_event_id=(
                    f"wind_only_selection_{row['selection_hash']}"
                ),
                stratum=str(row["category"]),
                duration_h=6.0,
                selection_basis=(
                    "frozen_wind_only_staged_registry:"
                    f"subtype={row['subtype']};"
                    f"selection_hash={row['selection_hash']}"
                ),
            )
            for row in rows
        )
        stage_file_hashes[stage_name] = _sha256_file(path)

    with combined_path.open(newline="", encoding="utf-8") as stream:
        combined_rows = list(csv.DictReader(stream))

    def row_identity(row: dict[str, str]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((str(key), str(value)) for key, value in row.items()))

    if sorted(map(row_identity, combined_rows)) != sorted(map(row_identity, staged_rows)):
        raise ValueError("stage files do not reproduce all_staged_cases.csv")
    start_times = sorted(
        datetime.strptime(str(row["timestamp"]), _TIMESTAMP_FORMAT)
        for row in staged_rows
    )
    minimum_separation_s = min(
        (right - left).total_seconds()
        for left, right in zip(start_times, start_times[1:])
    )
    if minimum_separation_s < int(manifest["minimum_start_separation_s"]):
        raise ValueError("staged-case windows violate manifest time separation")

    identity = {
        "registry_directory": str(directory.relative_to(repository_root)),
        "selection_manifest_sha256": _sha256_file(manifest_path),
        "all_staged_cases_sha256": _sha256_file(combined_path),
        "stage_file_sha256": stage_file_hashes,
        "source_observations_sha256": manifest["source_observations_sha256"],
        "controller_outputs_used": False,
        "historical_overlap_count": manifest["historical_overlap_count"],
        "within_and_cross_stage_overlap_count": manifest[
            "within_and_cross_stage_overlap_count"
        ],
        "minimum_actual_start_separation_s": minimum_separation_s,
    }
    return cases_by_stage, identity


__all__ = ["load_frozen_staged_cases"]
