#!/usr/bin/env python3
"""Place the static moment probe in explicit FINO1 wind-load scales.

The existing ``10 MNm`` OpenFAST test is an incremental static-moment probe.
This entry does not reinterpret it as the absolute load at one wind speed.  It
instead reports two intentionally separate diagnostics for the FINO1 test data:

* the legacy paper's fallback thrust/arm relation, retained only to identify
  the scale present in the published-controller data path; and
* a fixed, aligned IEA 15 MW rotor state (``1 deg``, ``5 rpm``) evaluated with
  the frozen public ``C_T`` table.  This is a load-scale diagnostic, not a
  reconstruction of the turbine's real variable-speed and variable-pitch
  operation.

The second diagnostic reports both absolute moment magnitude and consecutive
10-minute moment increments.  The latter is the relevant comparison for an
incremental static preload such as ``10 MNm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
SRC_DIRECTORY = ROOT / "src"
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fowt_platform import (
    generalized_load_from_point_force,
    load_rosco_rotor_performance_table_from_zip,
    openfast_reference_vector_to_frozen_equilibrium_axes,
)
from run_rotor_normal_load_input_audit import (
    MODEL_ARCHIVE_SHA256,
    MODEL_ZIP,
    REFERENCE_STATE_JSON,
    ROTOR_TABLE_MEMBER,
    _load_candidate_reference_state,
    _source_values,
)


CANONICAL_DATA = ROOT / (
    "data/processed/wind_ml_10min/fino1_platform_10min/"
    "canonical_observations_10min.csv.gz"
)
PAPER_CASES = ROOT / "paper_snapshot/2026-08-11/evidence/cases/frozen_h_holdout150.csv"
LEGACY_AIR_DENSITY_KG_M3 = 1.225
LEGACY_ROTOR_RADIUS_M = 63.0
LEGACY_THRUST_COEFFICIENT = 0.80
LEGACY_AERODYNAMIC_ARM_M = 127.0
DEFAULT_STATIC_PROBE_MOMENT_NM = 1.0e7
FIXED_PITCH_DEG = 1.0
FIXED_ROTOR_SPEED_RPM = 5.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(name: str, values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


def _quantiles(values: Any) -> dict[str, float]:
    array = _finite_vector("values", values)
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def _fraction_above(values: np.ndarray, threshold: float) -> float:
    return float(100.0 * np.mean(np.asarray(values, dtype=float) > threshold))


def legacy_wind_moment_components_mnm(
    wind_speed_mps: Any,
    wind_direction_deg: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the legacy fallback resultant, roll and pitch moment components.

    This exactly mirrors the fallback relation retained in the historic paper
    plotting path.  It is deliberately not promoted to the new IEA 15 MW
    physical load model.
    """

    speed = np.asarray(wind_speed_mps, dtype=float)
    direction = np.asarray(wind_direction_deg, dtype=float)
    if speed.shape != direction.shape or speed.ndim != 1:
        raise ValueError("wind speed and direction must be one-dimensional and aligned")
    if not np.all(np.isfinite(speed)) or not np.all(np.isfinite(direction)):
        raise ValueError("legacy wind inputs must be finite")
    thrust_n = (
        0.5
        * LEGACY_AIR_DENSITY_KG_M3
        * math.pi
        * LEGACY_ROTOR_RADIUS_M**2
        * LEGACY_THRUST_COEFFICIENT
        * np.square(speed)
    )
    resultant_mnm = thrust_n * LEGACY_AERODYNAMIC_ARM_M / 1.0e6
    direction_rad = np.deg2rad(direction)
    roll_mnm = -resultant_mnm * np.sin(direction_rad)
    pitch_mnm = resultant_mnm * np.cos(direction_rad)
    return resultant_mnm, roll_mnm, pitch_mnm


def _fixed_operating_state_moment_mnm(wind_speed_mps: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map speed to an aligned quasi-steady pitch-moment diagnostic.

    The returned validity mask enforces the published ``C_T`` table bounds.
    No extrapolation is performed.
    """

    speed = _finite_vector("wind_speed_mps", wind_speed_mps)
    table = load_rosco_rotor_performance_table_from_zip(
        MODEL_ZIP,
        ROTOR_TABLE_MEMBER,
        expected_archive_sha256=MODEL_ARCHIVE_SHA256,
    )
    source = _source_values(MODEL_ZIP)
    reference = _load_candidate_reference_state(
        REFERENCE_STATE_JSON,
        model_zip=MODEL_ZIP,
    )
    rotor_radius_m = float(source["rotor_radius_m"])
    angular_speed_rad_s = FIXED_ROTOR_SPEED_RPM * 2.0 * math.pi / 60.0
    tip_speed_ratio = angular_speed_rad_s * rotor_radius_m / speed
    valid = (
        (speed > 0.0)
        & (FIXED_PITCH_DEG >= float(table.pitch_deg[0]))
        & (FIXED_PITCH_DEG <= float(table.pitch_deg[-1]))
        & (tip_speed_ratio >= float(table.tip_speed_ratio[0]))
        & (tip_speed_ratio <= float(table.tip_speed_ratio[-1]))
    )
    result = np.full(speed.shape, np.nan, dtype=float)
    if not np.any(valid):
        return result, valid

    normal = openfast_reference_vector_to_frozen_equilibrium_axes(
        source["downwind_rotor_normal_openfast_reference"],
        equilibrium_platform_roll_deg=0.0,
        equilibrium_platform_pitch_deg=reference["pitch_deg"],
        equilibrium_platform_yaw_deg=0.0,
    )
    hub = openfast_reference_vector_to_frozen_equilibrium_axes(
        source["hub_from_platform_reference_openfast_reference_m"],
        equilibrium_platform_roll_deg=0.0,
        equilibrium_platform_pitch_deg=reference["pitch_deg"],
        equilibrium_platform_yaw_deg=0.0,
    )
    unique_speeds, inverse = np.unique(speed[valid], return_inverse=True)
    unique_moments = np.empty(unique_speeds.shape, dtype=float)
    for index, selected_speed in enumerate(unique_speeds):
        selected_tsr = angular_speed_rad_s * rotor_radius_m / selected_speed
        ct = table.ct_at(pitch_deg=FIXED_PITCH_DEG, tip_speed_ratio=selected_tsr)
        thrust_n = (
            0.5
            * float(source["air_density_kg_m3"])
            * math.pi
            * rotor_radius_m**2
            * ct
            * selected_speed**2
        )
        generalized_load = generalized_load_from_point_force(thrust_n * normal, hub)
        unique_moments[index] = abs(float(generalized_load[4])) / 1.0e6
    result[valid] = unique_moments[inverse]
    return result, valid


def _contiguous_ten_minute_changes(
    timestamp: pd.Series,
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    time = pd.to_datetime(timestamp).to_numpy(dtype="datetime64[ns]")
    if time.size < 2:
        return np.empty(0, dtype=float)
    changes = np.abs(np.diff(values))
    ten_minutes = np.timedelta64(10, "m")
    pair_valid = valid[:-1] & valid[1:] & (np.diff(time) == ten_minutes)
    return changes[pair_valid]


def _coverage_mask(
    timestamps: pd.Series,
    case_starts: pd.Series,
    duration_hours: float = 6.0,
) -> np.ndarray:
    values = pd.to_datetime(timestamps).to_numpy(dtype="datetime64[ns]")
    mask = np.zeros(values.shape, dtype=bool)
    duration = np.timedelta64(int(round(duration_hours * 60)), "m")
    for start in pd.to_datetime(case_starts).to_numpy(dtype="datetime64[ns]"):
        mask |= (values >= start) & (values < start + duration)
    return mask


def _scope_summary(frame: pd.DataFrame, *, scope_name: str, static_probe_mnm: float) -> dict[str, Any]:
    speed = frame["wind_speed_ms"].to_numpy(dtype=float)
    direction = frame["wind_dir_deg"].to_numpy(dtype=float)
    resultant, roll, pitch = legacy_wind_moment_components_mnm(speed, direction)
    fixed_moment, fixed_valid = _fixed_operating_state_moment_mnm(speed)
    fixed_values = fixed_moment[fixed_valid]
    fixed_change = _contiguous_ten_minute_changes(
        frame["timestamp"], fixed_moment, fixed_valid
    )
    return {
        "scope": scope_name,
        "sample_count": int(len(frame)),
        "wind_speed_mps": _quantiles(speed),
        "legacy_paper_fallback": {
            "definition": (
                "T=0.5*rho*pi*63^2*0.8*U^2; M_resultant=T*127 m; "
                "components use the historic wind-direction decomposition"
            ),
            "resultant_moment_mnm": _quantiles(resultant),
            "absolute_pitch_component_mnm": _quantiles(np.abs(pitch)),
            "absolute_roll_component_mnm": _quantiles(np.abs(roll)),
            "fraction_resultant_above_static_probe_percent": _fraction_above(
                resultant, static_probe_mnm
            ),
        },
        "fixed_iea15_operating_state": {
            "definition": (
                "aligned quasi-steady source-bound C_T lookup at fixed 1 deg pitch "
                "and 5 rpm; not a reconstruction of real turbine operation or yaw"
            ),
            "validity": {
                "valid_sample_count": int(np.sum(fixed_valid)),
                "valid_fraction_percent": float(100.0 * np.mean(fixed_valid)),
                "excluded_for_ct_table_bounds_count": int(np.sum(~fixed_valid)),
            },
            "absolute_pitch_moment_mnm": _quantiles(fixed_values),
            "consecutive_10min_absolute_pitch_moment_increment_mnm": _quantiles(
                fixed_change
            ),
            "fraction_increment_above_static_probe_percent": _fraction_above(
                fixed_change, static_probe_mnm
            ),
        },
    }


def run_audit(*, static_probe_moment_nm: float = DEFAULT_STATIC_PROBE_MOMENT_NM) -> dict[str, Any]:
    if static_probe_moment_nm <= 0.0 or not np.isfinite(static_probe_moment_nm):
        raise ValueError("static_probe_moment_nm must be finite and positive")
    if not CANONICAL_DATA.is_file() or not PAPER_CASES.is_file():
        raise FileNotFoundError("expected FINO1 canonical data and frozen 150-case list")

    frame = pd.read_csv(
        CANONICAL_DATA,
        compression="gzip",
        usecols=["timestamp", "wind_speed_ms", "wind_dir_deg", "split"],
        parse_dates=["timestamp"],
    )
    test = frame.loc[frame["split"] == "test"].copy().sort_values("timestamp")
    if test.empty:
        raise ValueError("canonical FINO1 data contains no test split")
    if test[["wind_speed_ms", "wind_dir_deg"]].isna().any().any():
        raise ValueError("test split contains missing wind speed or direction")
    cases = pd.read_csv(PAPER_CASES, usecols=["case_id", "timestamp"])
    coverage = _coverage_mask(test["timestamp"], cases["timestamp"])
    selected = test.loc[coverage].copy()
    if selected.empty:
        raise ValueError("the frozen 150-case list does not overlap the FINO1 test split")
    static_probe_mnm = static_probe_moment_nm / 1.0e6
    return {
        "evidence_level": "FINO1 wind-load-scale diagnostic",
        "boundaries": {
            "is_controller_performance_validation": False,
            "is_real_turbine_operating_load_reconstruction": False,
            "is_openfast_dynamic_response_validation": False,
            "static_probe_definition": (
                "incremental static moment around the OpenFAST candidate reference, "
                "not an absolute wind load"
            ),
            "purpose": (
                "place the static probe and continuous 10-minute wind-load increments "
                "within explicit project-data load scales"
            ),
        },
        "sources": {
            "canonical_fino1_csv": str(CANONICAL_DATA.relative_to(ROOT)),
            "canonical_fino1_csv_sha256": _sha256(CANONICAL_DATA),
            "frozen_150_case_list": str(PAPER_CASES.relative_to(ROOT)),
            "frozen_150_case_list_sha256": _sha256(PAPER_CASES),
            "static_probe_mnm": static_probe_mnm,
            "fixed_rotor_state": {
                "pitch_deg": FIXED_PITCH_DEG,
                "rotor_speed_rpm": FIXED_ROTOR_SPEED_RPM,
                "model_zip": str(MODEL_ZIP.relative_to(ROOT)),
                "model_zip_sha256": MODEL_ARCHIVE_SHA256,
                "ct_table_member": ROTOR_TABLE_MEMBER,
            },
        },
        "test_split": _scope_summary(
            test, scope_name="all canonical rows with split=test", static_probe_mnm=static_probe_mnm
        ),
        "frozen_150_case_coverage": _scope_summary(
            selected,
            scope_name=(
                "unique 10-minute test rows covered by at least one frozen 6-hour case"
            ),
            static_probe_mnm=static_probe_mnm,
        ),
        "frozen_case_count": int(len(cases)),
        "covered_unique_10min_rows": int(len(selected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-probe-moment-nm", type=float, default=DEFAULT_STATIC_PROBE_MOMENT_NM)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(static_probe_moment_nm=args.static_probe_moment_nm)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
