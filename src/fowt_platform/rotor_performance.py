"""Strict readers for public rotor-performance tables.

The module deliberately stops at tabulated aerodynamic coefficients.  It does
not turn ``C_T`` into thrust or decide a load direction; those choices belong
to a later, explicitly configured wind-load layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class RotorPerformanceTable:
    """ROSCO rotor coefficients indexed by pitch angle and tip-speed ratio."""

    pitch_deg: np.ndarray
    tip_speed_ratio: np.ndarray
    wind_speed_mps: np.ndarray
    cp: np.ndarray
    ct: np.ndarray
    cq: np.ndarray
    source_name: str
    source_sha256: str
    source_archive_sha256: str | None = None

    def ct_at(self, *, pitch_deg: float, tip_speed_ratio: float) -> float:
        """Return bilinearly interpolated ``C_T`` within the published grid.

        Extrapolation is intentionally rejected. A force model must decide how
        to handle operating points that fall outside this table.
        """

        return _bilinear_lookup(
            values=self.ct,
            x_axis=self.pitch_deg,
            y_axis=self.tip_speed_ratio,
            x=float(pitch_deg),
            y=float(tip_speed_ratio),
            x_name="pitch_deg",
            y_name="tip_speed_ratio",
        )


def parse_rosco_rotor_performance_table(
    text: str,
    *,
    source_name: str = "<text>",
    source_archive_sha256: str | None = None,
) -> RotorPerformanceTable:
    """Parse ROSCO's Cp/Ct/Cq text layout without relying on line numbers."""

    lines = text.splitlines()
    pitch_deg = _numbers_after_header(lines, "# Pitch angle vector")
    tip_speed_ratio = _numbers_after_header(lines, "# TSR vector")
    wind_speed_mps = _numbers_after_header(lines, "# Wind speed vector")
    shape = (len(tip_speed_ratio), len(pitch_deg))
    if not shape[0] or not shape[1] or not len(wind_speed_mps):
        raise ValueError("ROSCO table axes must all contain at least one value")
    if len(wind_speed_mps) != 1:
        raise ValueError(
            "Only single-wind-speed ROSCO tables are supported; "
            "multi-layer tables require an explicit wind-speed lookup"
        )

    cp = _matrix_after_header(lines, "# Power coefficient", shape)
    ct = _matrix_after_header(lines, "# Thrust coefficient", shape)
    cq = _matrix_after_header(lines, "# Torque coefficient", shape)
    _validate_axis(pitch_deg, "pitch_deg", minimum_length=2)
    _validate_axis(tip_speed_ratio, "tip_speed_ratio", minimum_length=2)
    _validate_axis(wind_speed_mps, "wind_speed_mps")

    return RotorPerformanceTable(
        pitch_deg=pitch_deg,
        tip_speed_ratio=tip_speed_ratio,
        wind_speed_mps=wind_speed_mps,
        cp=cp,
        ct=ct,
        cq=cq,
        source_name=source_name,
        source_sha256=sha256(text.encode("utf-8")).hexdigest(),
        source_archive_sha256=source_archive_sha256,
    )


def load_rosco_rotor_performance_table(path: str | Path) -> RotorPerformanceTable:
    """Load a ROSCO table stored as a normal text file."""

    path = Path(path)
    return parse_rosco_rotor_performance_table(
        path.read_text(encoding="utf-8"), source_name=str(path)
    )


def load_rosco_rotor_performance_table_from_zip(
    archive_path: str | Path,
    member_name: str,
    *,
    expected_archive_sha256: str,
) -> RotorPerformanceTable:
    """Load a table from a frozen archive after verifying its identity."""

    archive_path = Path(archive_path)
    actual_archive_sha256 = _file_sha256(archive_path)
    if actual_archive_sha256.lower() != expected_archive_sha256.lower():
        raise ValueError(
            "Rotor-performance archive SHA-256 does not match the expected source"
        )
    try:
        with ZipFile(archive_path) as archive:
            raw = archive.read(member_name)
    except KeyError as error:
        raise ValueError(f"Rotor-performance member not found: {member_name}") from error
    return parse_rosco_rotor_performance_table(
        raw.decode("utf-8"),
        source_name=f"{archive_path}!{member_name}",
        source_archive_sha256=actual_archive_sha256,
    )


def _numbers_after_header(lines: list[str], header: str) -> np.ndarray:
    index = _header_index(lines, header)
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            raise ValueError(f"No numeric row after {header!r}")
        values = np.fromstring(stripped, sep=" ", dtype=float)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"Invalid numeric row after {header!r}")
        return values
    raise ValueError(f"No numeric row after {header!r}")


def _matrix_after_header(
    lines: list[str], header: str, shape: tuple[int, int]
) -> np.ndarray:
    index = _header_index(lines, header)
    rows: list[np.ndarray] = []
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("#"):
            if rows:
                break
            continue
        row = np.fromstring(stripped, sep=" ", dtype=float)
        if len(row) != shape[1] or not np.isfinite(row).all():
            raise ValueError(f"Invalid matrix row in {header!r}")
        rows.append(row)
        if len(rows) > shape[0]:
            raise ValueError(
                f"{header!r} has more rows than expected from its axes"
            )
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != shape:
        raise ValueError(
            f"{header!r} has shape {matrix.shape}, expected {shape} from its axes"
        )
    return matrix


def _header_index(lines: list[str], header: str) -> int:
    normalized_header = " ".join(header.split()).lower()
    for index, line in enumerate(lines):
        if " ".join(line.split()).lower().startswith(normalized_header):
            return index
    raise ValueError(f"ROSCO table header not found: {header!r}")


def _validate_axis(axis: np.ndarray, name: str, *, minimum_length: int = 1) -> None:
    if len(axis) < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} entries")
    if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bilinear_lookup(
    *,
    values: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x: float,
    y: float,
    x_name: str,
    y_name: str,
) -> float:
    if not np.isfinite([x, y]).all():
        raise ValueError(f"{x_name} and {y_name} must be finite")
    if len(x_axis) < 2 or len(y_axis) < 2:
        raise ValueError("Bilinear lookup requires at least two entries per axis")
    if not x_axis[0] <= x <= x_axis[-1]:
        raise ValueError(f"{x_name}={x} lies outside [{x_axis[0]}, {x_axis[-1]}]")
    if not y_axis[0] <= y <= y_axis[-1]:
        raise ValueError(f"{y_name}={y} lies outside [{y_axis[0]}, {y_axis[-1]}]")

    x0, x1, fx = _bracket(x_axis, x)
    y0, y1, fy = _bracket(y_axis, y)
    return float(
        (1.0 - fx) * (1.0 - fy) * values[y0, x0]
        + fx * (1.0 - fy) * values[y0, x1]
        + (1.0 - fx) * fy * values[y1, x0]
        + fx * fy * values[y1, x1]
    )


def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    upper = int(np.searchsorted(axis, value, side="right"))
    upper = min(max(upper, 1), len(axis) - 1)
    lower = upper - 1
    return lower, upper, (value - axis[lower]) / (axis[upper] - axis[lower])
