#!/usr/bin/env python3
"""Check one source-bound rotor load through the low-order wind-load channel.

This entry joins the frozen rotor-performance audit to the incremental
platform input interface at one explicit rotor condition.  It is deliberately
not a platform-response validation: the current low-order damping model is not
calibrated, and the resulting static offset is retained only as an internal
balance check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from fowt_platform import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    load_volturnus_reference_components,
    solve_incremental_static_offset,
)


VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from run_rotor_normal_load_input_audit import run_audit as run_rotor_load_audit


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
LOAD_NAMES = ("surge_n", "sway_n", "heave_n", "roll_nm", "pitch_nm", "yaw_nm")
STATE_NAMES = ("surge_m", "sway_m", "heave_m", "roll_rad", "pitch_rad", "yaw_rad")
ACCELERATION_NAMES = (
    "surge_mps2",
    "sway_mps2",
    "heave_mps2",
    "roll_radps2",
    "pitch_radps2",
    "yaw_radps2",
)


def _load_vector(rotor_audit: dict[str, object]) -> np.ndarray:
    named_load = rotor_audit["generalized_load_about_platform_reference"]
    if not isinstance(named_load, dict):
        raise ValueError("rotor audit is missing its generalized load")
    load = np.asarray([named_load[name] for name in LOAD_NAMES], dtype=float)
    if load.shape != (6,) or not np.all(np.isfinite(load)):
        raise ValueError("rotor audit generalized load must be a finite six-vector")
    load.setflags(write=False)
    return load


def _named_load(load: np.ndarray) -> dict[str, float]:
    return {name: float(load[index]) for index, name in enumerate(LOAD_NAMES)}


def _named_state(position: np.ndarray) -> dict[str, float]:
    return {name: float(position[index]) for index, name in enumerate(STATE_NAMES)}


def _named_acceleration(acceleration: np.ndarray) -> dict[str, float]:
    return {
        name: float(acceleration[index])
        for index, name in enumerate(ACCELERATION_NAMES)
    }


def run_smoke(
    *,
    wind_speed_mps: float = 5.0,
    blade_pitch_deg: float = 1.0,
    rotor_speed_rpm: float = 5.0,
) -> dict[str, object]:
    """Pass one explicit prescribed-rotor load into ``IncrementalLoads.wind``."""

    rotor_audit = run_rotor_load_audit(
        pitch_deg=blade_pitch_deg,
        rotor_speed_rpm=rotor_speed_rpm,
        wind_speed_mps=wind_speed_mps,
    )
    rotor_checks = rotor_audit.get("checks")
    if not isinstance(rotor_checks, dict) or not rotor_checks.get("all_checks_pass"):
        raise ValueError("source-bound rotor load audit did not pass its checks")
    wind_load = _load_vector(rotor_audit)
    zero = np.zeros(6)
    loads = IncrementalLoads(
        wind=wind_load,
        wave=zero,
        ballast=zero,
        other=zero,
    )
    reference = load_volturnus_reference_components(REFERENCE_MANIFEST)
    matrices = reference.assemble_whole_system_candidate(damping=np.zeros((6, 6)))
    model = IncrementalPlatformModel(matrices)
    static_offset = solve_incremental_static_offset(model.matrices, loads.total)
    balanced_state = IncrementalState(
        position=static_offset.position,
        velocity=np.zeros(6),
    )
    balanced_acceleration = model.derivative(balanced_state, loads).velocity_rate
    force_residual_max = float(np.max(np.abs(static_offset.residual_load[:3])))
    moment_residual_max = float(np.max(np.abs(static_offset.residual_load[3:])))
    translation_acceleration_max = float(np.max(np.abs(balanced_acceleration[:3])))
    rotation_acceleration_max = float(np.max(np.abs(balanced_acceleration[3:])))
    channel_isolation = bool(
        np.array_equal(loads.wind, wind_load)
        and np.array_equal(loads.wave, zero)
        and np.array_equal(loads.ballast, zero)
        and np.array_equal(loads.other, zero)
    )
    return {
        "evidence_level": "internal_source_bound_rotor_load_chain_smoke_check",
        "boundaries": {
            "is_rosco_operating_case": False,
            "is_platform_dynamic_response_validation": False,
            "is_static_offset_validation": False,
            "is_physical_damping_calibration": False,
            "purpose": "verify that one frozen performance-table load enters the low-order wind channel without mixing with wave, ballast or other loads",
        },
        "prescribed_rotor_condition": {
            "wind_speed_mps": float(wind_speed_mps),
            "blade_pitch_deg": float(blade_pitch_deg),
            "rotor_speed_rpm": float(rotor_speed_rpm),
            "tip_speed_ratio": rotor_audit["table_lookup_node"]["tip_speed_ratio"],
            "thrust_coefficient": rotor_audit["table_lookup_node"]["thrust_coefficient"],
        },
        "source_bound_rotor_audit": {
            "source": rotor_audit["source"],
            "checks": rotor_checks,
        },
        "source_bound_wind_load": _named_load(wind_load),
        "internal_static_balance_only": {
            "offset": _named_state(static_offset.position),
            "residual": _named_load(static_offset.residual_load),
            "residual_max_abs": {
                "translation_n": force_residual_max,
                "rotation_nm": moment_residual_max,
            },
            "balanced_acceleration": _named_acceleration(balanced_acceleration),
            "balanced_acceleration_max_abs": {
                "translation_mps2": translation_acceleration_max,
                "rotation_radps2": rotation_acceleration_max,
            },
        },
        "checks": {
            "wind_load_isolated_from_other_load_channels": channel_isolation,
            "static_force_residual_is_negligible": force_residual_max < 1.0e-6,
            "static_moment_residual_is_negligible": moment_residual_max < 1.0e-3,
            "balanced_translation_acceleration_is_negligible": (
                translation_acceleration_max < 1.0e-12
            ),
            "balanced_rotation_acceleration_is_negligible": (
                rotation_acceleration_max < 1.0e-12
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wind-speed-mps", type=float, default=5.0)
    parser.add_argument("--blade-pitch-deg", type=float, default=1.0)
    parser.add_argument("--rotor-speed-rpm", type=float, default=5.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_smoke(
        wind_speed_mps=args.wind_speed_mps,
        blade_pitch_deg=args.blade_pitch_deg,
        rotor_speed_rpm=args.rotor_speed_rpm,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
