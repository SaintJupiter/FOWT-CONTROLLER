"""Source-bound local radiation damping from a frozen WAMIT ``.1`` file."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any
import zipfile

import numpy as np


_DOF_COUNT = 6


@dataclass(frozen=True)
class LocalRadiationDamping:
    """One constant damping approximation evaluated at a named period.

    The matrix is suitable only as a local low-order approximation to the
    frequency-dependent radiation term.  It is not a replacement for a
    radiation-memory model.
    """

    damping_matrix: Any
    requested_period_s: float
    lower_source_period_s: float
    upper_source_period_s: float
    interpolation_fraction_in_angular_frequency: float
    seawater_density_kg_m3: float
    wamit_characteristic_length_m: float
    archive_sha256: str
    archive_member: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.damping_matrix, dtype=float)
        if matrix.shape != (_DOF_COUNT, _DOF_COUNT):
            raise ValueError("damping_matrix must have shape (6, 6)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("damping_matrix must contain finite values")
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
            raise ValueError("damping_matrix must be symmetric")
        eigenvalues = np.linalg.eigvalsh(matrix)
        tolerance = 1e-10 * max(1.0, float(np.max(np.abs(eigenvalues))))
        if float(np.min(eigenvalues)) < -tolerance:
            raise ValueError("damping_matrix must be positive semidefinite")
        result = np.array(matrix, dtype=float, copy=True)
        result.setflags(write=False)
        object.__setattr__(self, "damping_matrix", result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _period_damping_tables(text: str) -> dict[float, np.ndarray]:
    tables: dict[float, np.ndarray] = {}
    for raw_line in text.splitlines():
        fields = raw_line.split()
        if len(fields) != 5:
            continue
        try:
            period = float(fields[0])
            row = int(fields[1]) - 1
            column = int(fields[2]) - 1
            nondimensional_damping = float(fields[4])
        except ValueError as exc:
            raise ValueError("invalid WAMIT .1 damping record") from exc
        if period <= 0.0:
            continue
        if not 0 <= row < _DOF_COUNT or not 0 <= column < _DOF_COUNT:
            raise ValueError("WAMIT .1 rigid-body index must be in [1, 6]")
        tables.setdefault(period, np.zeros((_DOF_COUNT, _DOF_COUNT)))[
            row, column
        ] = nondimensional_damping
    if len(tables) < 2:
        raise ValueError("WAMIT .1 file must contain at least two finite periods")
    return tables


def _named_numeric_value(text: str, field_name: str) -> float:
    for raw_line in text.splitlines():
        fields = raw_line.split()
        if field_name in fields:
            try:
                return float(fields[0].strip('"'))
            except (IndexError, ValueError) as exc:
                raise ValueError(f"invalid {field_name} value") from exc
    raise ValueError(f"{field_name} not found in source file")


def _dimensionalize_damping(
    nondimensional: np.ndarray,
    *,
    angular_frequency_rad_s: float,
    seawater_density_kg_m3: float,
    characteristic_length_m: float,
) -> np.ndarray:
    result = np.empty((_DOF_COUNT, _DOF_COUNT), dtype=float)
    for row in range(_DOF_COUNT):
        for column in range(_DOF_COUNT):
            rotational_indices = int(row >= 3) + int(column >= 3)
            length_power = 3 + rotational_indices
            result[row, column] = (
                nondimensional[row, column]
                * seawater_density_kg_m3
                * characteristic_length_m**length_power
                * angular_frequency_rad_s
            )
    return result


def load_local_radiation_damping(
    manifest_path: str | Path,
    *,
    requested_period_s: float,
    verify_archive_hash: bool = True,
) -> LocalRadiationDamping:
    """Interpolate and dimensionalize a WAMIT radiation-damping matrix.

    WAMIT defines ``Bbar_ij = B_ij / (rho * L**k * omega)``, where ``k`` is
    three for translation/translation terms, four for mixed terms and five
    for rotation/rotation terms.  Interpolation is performed in angular
    frequency, then the source matrix is symmetrized after dimensionalization.
    """

    period = float(requested_period_s)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("requested_period_s must be positive and finite")

    manifest = Path(manifest_path).resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    repository_root = manifest.parents[2]
    archive = repository_root / payload["sources"]["model_zip"]["path"]
    expected_hash = str(payload["sources"]["model_zip"]["sha256"])
    actual_hash = _sha256(archive)
    if verify_archive_hash and actual_hash != expected_hash:
        raise ValueError("reference model archive SHA-256 does not match manifest")

    member = str(payload["archive_members"]["wamit_added_mass"])
    hydrodyn_member = str(payload["archive_members"]["hydrodyn"])
    with zipfile.ZipFile(archive) as model_archive:
        table_text = model_archive.read(member).decode("utf-8")
        hydrodyn_text = model_archive.read(hydrodyn_member).decode("utf-8")
    tables = _period_damping_tables(table_text)
    periods = sorted(tables)
    if period < periods[0] or period > periods[-1]:
        raise ValueError("requested period lies outside the finite WAMIT table")
    lower = max(item for item in periods if item <= period)
    upper = min(item for item in periods if item >= period)
    requested_omega = 2.0 * math.pi / period
    if lower == upper:
        fraction = 0.0
        nondimensional = tables[lower]
    else:
        lower_omega = 2.0 * math.pi / lower
        upper_omega = 2.0 * math.pi / upper
        fraction = (requested_omega - lower_omega) / (upper_omega - lower_omega)
        nondimensional = tables[lower] + fraction * (
            tables[upper] - tables[lower]
        )

    density = float(
        payload["report_static_balance_evidence"]["seawater_density_kg_m3"]
    )
    characteristic_length = _named_numeric_value(hydrodyn_text, "WAMITULEN")
    dimensional = _dimensionalize_damping(
        nondimensional,
        angular_frequency_rad_s=requested_omega,
        seawater_density_kg_m3=density,
        characteristic_length_m=characteristic_length,
    )
    symmetric = 0.5 * (dimensional + dimensional.T)
    return LocalRadiationDamping(
        damping_matrix=symmetric,
        requested_period_s=period,
        lower_source_period_s=lower,
        upper_source_period_s=upper,
        interpolation_fraction_in_angular_frequency=fraction,
        seawater_density_kg_m3=density,
        wamit_characteristic_length_m=characteristic_length,
        archive_sha256=actual_hash,
        archive_member=member,
    )


__all__ = ["LocalRadiationDamping", "load_local_radiation_damping"]
