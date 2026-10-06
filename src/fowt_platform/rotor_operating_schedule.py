"""Narrow, source-bound nominal rotor-operation schedules.

The public IEA 15 MW ROSCO tuning file supplies a few below-rated operating
setpoints and the accompanying performance table supplies ``C_T(beta, lambda)``.
This module combines only those published inputs.  It is intentionally *not*
a ROSCO implementation: it has no pitch/torque dynamics, wind estimator, yaw
controller, generator dynamics, or above-rated pitch schedule.

Accordingly, callers may use the resulting coefficients only where the input
normal inflow remains within the declared below-rated range and the published
two-dimensional performance table can be queried without extrapolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any
from zipfile import ZipFile

import numpy as np

from .rotor_performance import RotorPerformanceOperatingPoint, RotorPerformanceTable


NOMINAL_BELOW_RATED_GENERATING = "nominal_below_rated_generating"


def _finite_scalar(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_text(name: str, value: Any) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _positive_inflow_sequence(name: str, value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty finite sequence") from exc
    if result.ndim != 1 or len(result) == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite sequence")
    if np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    copied = np.array(result, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class NominalBelowRatedRotorSchedule:
    """Published setpoints used by one quasi-steady below-rated approximation.

    ``cut_in_wind_speed_mps`` and ``rated_wind_speed_mps`` define the only
    wind-speed range where this compact schedule is permitted.  The routine
    tracks ``operational_tip_speed_ratio`` subject to the source-declared
    minimum and rated rotor speeds, then uses the source-declared minimum
    pitch.  It does not state that the real ROSCO controller will occupy the
    same operating point during a transient.
    """

    cut_in_wind_speed_mps: float
    rated_wind_speed_mps: float
    operational_tip_speed_ratio: float
    minimum_rotor_speed_rad_s: float
    rated_rotor_speed_rad_s: float
    minimum_pitch_rad: float
    source_name: str
    source_sha256: str
    source_archive_sha256: str | None = None

    def __post_init__(self) -> None:
        cut_in = _finite_scalar("cut_in_wind_speed_mps", self.cut_in_wind_speed_mps)
        rated = _finite_scalar("rated_wind_speed_mps", self.rated_wind_speed_mps)
        tsr = _finite_scalar(
            "operational_tip_speed_ratio", self.operational_tip_speed_ratio
        )
        minimum_speed = _finite_scalar(
            "minimum_rotor_speed_rad_s", self.minimum_rotor_speed_rad_s
        )
        rated_speed = _finite_scalar(
            "rated_rotor_speed_rad_s", self.rated_rotor_speed_rad_s
        )
        pitch = _finite_scalar("minimum_pitch_rad", self.minimum_pitch_rad)
        if cut_in <= 0.0:
            raise ValueError("cut_in_wind_speed_mps must be positive")
        if rated <= cut_in:
            raise ValueError("rated_wind_speed_mps must exceed cut_in_wind_speed_mps")
        if tsr <= 0.0:
            raise ValueError("operational_tip_speed_ratio must be positive")
        if minimum_speed < 0.0:
            raise ValueError("minimum_rotor_speed_rad_s must be non-negative")
        if rated_speed <= 0.0 or rated_speed < minimum_speed:
            raise ValueError(
                "rated_rotor_speed_rad_s must be positive and no smaller than "
                "minimum_rotor_speed_rad_s"
            )
        object.__setattr__(self, "cut_in_wind_speed_mps", cut_in)
        object.__setattr__(self, "rated_wind_speed_mps", rated)
        object.__setattr__(self, "operational_tip_speed_ratio", tsr)
        object.__setattr__(self, "minimum_rotor_speed_rad_s", minimum_speed)
        object.__setattr__(self, "rated_rotor_speed_rad_s", rated_speed)
        object.__setattr__(self, "minimum_pitch_rad", pitch)
        object.__setattr__(self, "source_name", _nonempty_text("source_name", self.source_name))
        object.__setattr__(self, "source_sha256", _nonempty_text("source_sha256", self.source_sha256))
        if self.source_archive_sha256 is not None:
            object.__setattr__(
                self,
                "source_archive_sha256",
                _nonempty_text("source_archive_sha256", self.source_archive_sha256),
            )

    @property
    def minimum_pitch_deg(self) -> float:
        """Return the source-declared fine-pitch setting in table units."""

        return float(np.degrees(self.minimum_pitch_rad))

    def rotor_speed_rad_s_at(
        self,
        *,
        normal_inflow_speed_mps: Any,
        rotor_radius_m: Any,
    ) -> float:
        """Return the bounded nominal rotor speed for one supported inflow.

        Below cut-in and above rated wind are rejected rather than converted to
        a parked, region-3, or extrapolated operating point.  The public
        ROSCO file contains the full dynamic controller, but this narrow
        schedule intentionally does not reproduce it.
        """

        inflow = _finite_scalar("normal_inflow_speed_mps", normal_inflow_speed_mps)
        radius = _finite_scalar("rotor_radius_m", rotor_radius_m)
        if inflow <= 0.0:
            raise ValueError("normal_inflow_speed_mps must be positive")
        if radius <= 0.0:
            raise ValueError("rotor_radius_m must be positive")
        if inflow <= self.cut_in_wind_speed_mps:
            raise ValueError(
                "normal_inflow_speed_mps is at or below the declared "
                "below-rated schedule cut-in boundary"
            )
        if inflow >= self.rated_wind_speed_mps:
            raise ValueError(
                "normal_inflow_speed_mps is at or above the declared "
                "below-rated schedule rated boundary"
            )
        commanded = self.operational_tip_speed_ratio * inflow / radius
        return float(
            min(
                max(commanded, self.minimum_rotor_speed_rad_s),
                self.rated_rotor_speed_rad_s,
            )
        )


@dataclass(frozen=True)
class NominalBelowRatedRotorOperatingPoint:
    """One table-resolved nominal operating point with its explicit input."""

    normal_inflow_speed_mps: float
    pitch_deg: float
    rotor_speed_rad_s: float
    rotor_speed_rpm: float
    tip_speed_ratio: float
    thrust_coefficient: float

    def __post_init__(self) -> None:
        inflow = _finite_scalar("normal_inflow_speed_mps", self.normal_inflow_speed_mps)
        pitch = _finite_scalar("pitch_deg", self.pitch_deg)
        speed = _finite_scalar("rotor_speed_rad_s", self.rotor_speed_rad_s)
        rpm = _finite_scalar("rotor_speed_rpm", self.rotor_speed_rpm)
        tsr = _finite_scalar("tip_speed_ratio", self.tip_speed_ratio)
        coefficient = _finite_scalar("thrust_coefficient", self.thrust_coefficient)
        if inflow <= 0.0 or speed < 0.0 or rpm < 0.0 or tsr <= 0.0:
            raise ValueError("nominal operating-point speeds and tip-speed ratio are invalid")
        if coefficient < 0.0:
            raise ValueError(
                "nominal below-rated schedule selected a negative thrust coefficient"
            )
        object.__setattr__(self, "normal_inflow_speed_mps", inflow)
        object.__setattr__(self, "pitch_deg", pitch)
        object.__setattr__(self, "rotor_speed_rad_s", speed)
        object.__setattr__(self, "rotor_speed_rpm", rpm)
        object.__setattr__(self, "tip_speed_ratio", tsr)
        object.__setattr__(self, "thrust_coefficient", coefficient)


@dataclass(frozen=True)
class NominalBelowRatedRotorOperatingForecast:
    """One current operating point and one point per future forecast lead."""

    schedule: NominalBelowRatedRotorSchedule
    performance_table: RotorPerformanceTable
    current: NominalBelowRatedRotorOperatingPoint
    future: tuple[NominalBelowRatedRotorOperatingPoint, ...]
    operating_mode: str
    operating_mode_source: str

    def __post_init__(self) -> None:
        if not isinstance(self.schedule, NominalBelowRatedRotorSchedule):
            raise TypeError("schedule must be a NominalBelowRatedRotorSchedule")
        if not isinstance(self.performance_table, RotorPerformanceTable):
            raise TypeError("performance_table must be a RotorPerformanceTable")
        if not isinstance(self.current, NominalBelowRatedRotorOperatingPoint):
            raise TypeError("current must be a NominalBelowRatedRotorOperatingPoint")
        if not self.future or not all(
            isinstance(point, NominalBelowRatedRotorOperatingPoint)
            for point in self.future
        ):
            raise ValueError("future must contain at least one nominal operating point")
        mode = _nonempty_text("operating_mode", self.operating_mode)
        if mode != NOMINAL_BELOW_RATED_GENERATING:
            raise ValueError(
                "this schedule requires operating_mode="
                f"{NOMINAL_BELOW_RATED_GENERATING!r}"
            )
        object.__setattr__(self, "operating_mode", mode)
        object.__setattr__(
            self,
            "operating_mode_source",
            _nonempty_text("operating_mode_source", self.operating_mode_source),
        )
        table_archive = self.performance_table.source_archive_sha256
        schedule_archive = self.schedule.source_archive_sha256
        if (table_archive is None) != (schedule_archive is None):
            raise ValueError(
                "performance table and nominal schedule must both bind the same "
                "source archive or both explicitly omit an archive binding"
            )
        if table_archive is not None and table_archive != schedule_archive:
            raise ValueError(
                "performance table and nominal schedule must come from the same archive"
            )

    @property
    def current_thrust_coefficient(self) -> float:
        return self.current.thrust_coefficient

    @property
    def future_thrust_coefficients(self) -> np.ndarray:
        result = np.asarray(
            [point.thrust_coefficient for point in self.future], dtype=float
        )
        result.setflags(write=False)
        return result


def load_rosco_nominal_below_rated_schedule_from_zip(
    archive_path: str | Path,
    member_name: str,
    *,
    expected_archive_sha256: str,
) -> NominalBelowRatedRotorSchedule:
    """Read the small fixed ROSCO-tuning subset used by this module.

    This is deliberately a narrow reader for the known ROSCO YAML text
    structure.  It is not a general YAML parser and rejects missing or
    duplicated required fields instead of guessing a default.
    """

    archive_path = Path(archive_path)
    actual_archive_sha256 = _file_sha256(archive_path)
    if actual_archive_sha256.lower() != expected_archive_sha256.lower():
        raise ValueError("ROSCO schedule archive SHA-256 does not match the expected source")
    try:
        with ZipFile(archive_path) as archive:
            raw = archive.read(member_name)
    except KeyError as exc:
        raise ValueError(f"ROSCO schedule member not found: {member_name}") from exc
    text = raw.decode("utf-8")
    return NominalBelowRatedRotorSchedule(
        cut_in_wind_speed_mps=_rosco_scalar(text, "turbine_params", "v_min"),
        rated_wind_speed_mps=_rosco_scalar(text, "turbine_params", "v_rated"),
        operational_tip_speed_ratio=_rosco_scalar(
            text, "turbine_params", "TSR_operational"
        ),
        minimum_rotor_speed_rad_s=_rosco_scalar(
            text, "controller_params", "vs_minspd"
        ),
        rated_rotor_speed_rad_s=_rosco_scalar(
            text, "turbine_params", "rated_rotor_speed"
        ),
        minimum_pitch_rad=_rosco_scalar(text, "controller_params", "min_pitch"),
        source_name=f"{archive_path}!{member_name}",
        source_sha256=sha256(text.encode("utf-8")).hexdigest(),
        source_archive_sha256=actual_archive_sha256,
    )


def assemble_nominal_below_rated_operating_forecast(
    *,
    schedule: NominalBelowRatedRotorSchedule,
    performance_table: RotorPerformanceTable,
    rotor_radius_m: Any,
    current_normal_inflow_speed_mps: Any,
    future_normal_inflow_speeds_mps: Any,
    operating_mode: str,
    operating_mode_source: str,
) -> NominalBelowRatedRotorOperatingForecast:
    """Resolve explicit current and future normal inflows to table ``C_T``.

    The function makes no reference to event probabilities, confidence values,
    ballast actions, or target water.  A caller must separately bind the
    normal inflow sequence to a forecast and provide compatible platform and
    rotor kinematics.
    """

    if not isinstance(schedule, NominalBelowRatedRotorSchedule):
        raise TypeError("schedule must be a NominalBelowRatedRotorSchedule")
    if not isinstance(performance_table, RotorPerformanceTable):
        raise TypeError("performance_table must be a RotorPerformanceTable")
    radius = _finite_scalar("rotor_radius_m", rotor_radius_m)
    if radius <= 0.0:
        raise ValueError("rotor_radius_m must be positive")
    current_inflow = _finite_scalar(
        "current_normal_inflow_speed_mps", current_normal_inflow_speed_mps
    )
    if current_inflow <= 0.0:
        raise ValueError("current_normal_inflow_speed_mps must be positive")
    future_inflows = _positive_inflow_sequence(
        "future_normal_inflow_speeds_mps", future_normal_inflow_speeds_mps
    )
    return NominalBelowRatedRotorOperatingForecast(
        schedule=schedule,
        performance_table=performance_table,
        current=_resolve_point(
            schedule=schedule,
            performance_table=performance_table,
            rotor_radius_m=radius,
            normal_inflow_speed_mps=current_inflow,
        ),
        future=tuple(
            _resolve_point(
                schedule=schedule,
                performance_table=performance_table,
                rotor_radius_m=radius,
                normal_inflow_speed_mps=float(inflow),
            )
            for inflow in future_inflows
        ),
        operating_mode=operating_mode,
        operating_mode_source=operating_mode_source,
    )


def inspect_nominal_below_rated_operating_inputs(
    *,
    schedule: NominalBelowRatedRotorSchedule,
    performance_table: RotorPerformanceTable,
    rotor_radius_m: Any,
    current_normal_inflow_speed_mps: Any,
    future_normal_inflow_speeds_mps: Any,
) -> dict[str, Any]:
    """Report whether declared normal inflows fit the compact rotor model.

    This is an input-domain check only. It resolves the same rotor speed and
    table coordinate that the operating forecast would use, but neither turns
    ``C_T`` into a load nor introduces a fallback state. Unsupported points
    remain unsupported rather than being clipped or extrapolated.
    """

    if not isinstance(schedule, NominalBelowRatedRotorSchedule):
        raise TypeError("schedule must be a NominalBelowRatedRotorSchedule")
    if not isinstance(performance_table, RotorPerformanceTable):
        raise TypeError("performance_table must be a RotorPerformanceTable")
    radius = _finite_scalar("rotor_radius_m", rotor_radius_m)
    if radius <= 0.0:
        raise ValueError("rotor_radius_m must be positive")
    current = _finite_scalar(
        "current_normal_inflow_speed_mps", current_normal_inflow_speed_mps
    )
    try:
        future = np.asarray(future_normal_inflow_speeds_mps, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "future_normal_inflow_speeds_mps must be a non-empty finite sequence"
        ) from exc
    if future.ndim != 1 or len(future) == 0 or not np.all(np.isfinite(future)):
        raise ValueError(
            "future_normal_inflow_speeds_mps must be a non-empty finite sequence"
        )

    def inspect(label: str, inflow: float) -> dict[str, Any]:
        record: dict[str, Any] = {
            "label": label,
            "normal_inflow_speed_mps": float(inflow),
            "supported": False,
        }
        try:
            point = _resolve_point(
                schedule=schedule,
                performance_table=performance_table,
                rotor_radius_m=radius,
                normal_inflow_speed_mps=float(inflow),
            )
        except ValueError as exc:
            record["reason"] = str(exc)
        else:
            record.update(
                {
                    "supported": True,
                    "rotor_speed_rad_s": float(point.rotor_speed_rad_s),
                    "tip_speed_ratio": float(point.tip_speed_ratio),
                }
            )
        return record

    records = [inspect("current", current)]
    records.extend(
        inspect(f"future_{index + 1}", float(inflow))
        for index, inflow in enumerate(future)
    )
    lower_from_tsr = (
        schedule.minimum_rotor_speed_rad_s
        * radius
        / float(performance_table.tip_speed_ratio[-1])
    )
    upper_from_tsr = (
        schedule.rated_rotor_speed_rad_s
        * radius
        / float(performance_table.tip_speed_ratio[0])
    )
    return {
        "operating_model": "source_bound_nominal_below_rated_generating",
        "supported": all(bool(record["supported"]) for record in records),
        "records": records,
        "declared_normal_inflow_range_mps": [
            float(schedule.cut_in_wind_speed_mps),
            float(schedule.rated_wind_speed_mps),
        ],
        "table_tip_speed_ratio_range": [
            float(performance_table.tip_speed_ratio[0]),
            float(performance_table.tip_speed_ratio[-1]),
        ],
        "derived_nominal_inflow_range_mps": [
            float(max(schedule.cut_in_wind_speed_mps, lower_from_tsr)),
            float(min(schedule.rated_wind_speed_mps, upper_from_tsr)),
        ],
        "unsupported_record_count": int(
            sum(not bool(record["supported"]) for record in records)
        ),
    }


def _resolve_point(
    *,
    schedule: NominalBelowRatedRotorSchedule,
    performance_table: RotorPerformanceTable,
    rotor_radius_m: float,
    normal_inflow_speed_mps: float,
) -> NominalBelowRatedRotorOperatingPoint:
    rotor_speed_rad_s = schedule.rotor_speed_rad_s_at(
        normal_inflow_speed_mps=normal_inflow_speed_mps,
        rotor_radius_m=rotor_radius_m,
    )
    rotor_speed_rpm = rotor_speed_rad_s * 60.0 / (2.0 * np.pi)
    table_point = RotorPerformanceOperatingPoint(
        pitch_deg=schedule.minimum_pitch_deg,
        rotor_speed_rpm=rotor_speed_rpm,
        rotor_radius_m=rotor_radius_m,
        normal_inflow_speed_mps=normal_inflow_speed_mps,
    )
    thrust_coefficient = performance_table.ct_at_operating_point(table_point)
    return NominalBelowRatedRotorOperatingPoint(
        normal_inflow_speed_mps=normal_inflow_speed_mps,
        pitch_deg=schedule.minimum_pitch_deg,
        rotor_speed_rad_s=rotor_speed_rad_s,
        rotor_speed_rpm=rotor_speed_rpm,
        tip_speed_ratio=table_point.tip_speed_ratio,
        thrust_coefficient=thrust_coefficient,
    )


def _rosco_scalar(text: str, section_name: str, field_name: str) -> float:
    """Read one scalar from the simple section/key ROSCO YAML layout."""

    active_section: str | None = None
    matches: list[str] = []
    section_pattern = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*$")
    field_pattern = re.compile(
        rf"^\s+{re.escape(field_name)}\s*:\s*([^#\s]+)"
    )
    for raw_line in text.splitlines():
        source_line = raw_line.split("#", 1)[0].rstrip()
        if not source_line.strip():
            continue
        section_match = section_pattern.match(source_line)
        if section_match is not None:
            active_section = section_match.group(1)
            continue
        if active_section != section_name:
            continue
        field_match = field_pattern.match(source_line)
        if field_match is not None:
            matches.append(field_match.group(1))
    if len(matches) != 1:
        raise ValueError(
            f"ROSCO YAML must contain exactly one {section_name}.{field_name} scalar"
        )
    return _finite_scalar(f"ROSCO {section_name}.{field_name}", matches[0])


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "NOMINAL_BELOW_RATED_GENERATING",
    "NominalBelowRatedRotorOperatingForecast",
    "NominalBelowRatedRotorOperatingPoint",
    "NominalBelowRatedRotorSchedule",
    "assemble_nominal_below_rated_operating_forecast",
    "inspect_nominal_below_rated_operating_inputs",
    "load_rosco_nominal_below_rated_schedule_from_zip",
]
