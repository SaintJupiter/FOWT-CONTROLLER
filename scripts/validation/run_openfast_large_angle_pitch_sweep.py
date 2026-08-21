#!/usr/bin/env python3
"""Locate the static-pitch range where the frozen low-order model departs from OpenFAST.

The sweep deliberately applies a static, frozen-axis *pure pitch moment* through
HydroDyn ``AddF0``.  For each requested pitch target, the pitch compliance of
the frozen low-order restoring matrix first determines the one pitch moment that
would yield that target.  No translational force, roll moment, or other
constraint component is introduced in the frozen axes.  The same moment is then
expressed at the OpenFAST reference point and applied to the public VolturnUS-S
model.  This makes the result a one-sided range check of the low-order static
approximation, rather than a wind-load or controller-performance simulation.

No restoring, damping, or OpenFAST input parameter is tuned from one target to
the next.  Each target uses an independent temporary relaxation only to locate
its candidate static pose, followed by an original-damping release at that same
final load.  The release tail is the reported OpenFAST reading and the stability
decision.  HydroDyn ``AddF0`` is a static input, so this audit reports final
static readings, not a continuous loading-path simulation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from fowt_platform.incremental import (
    _openfast_platform_to_inertial_rotation,
    solve_incremental_static_offset,
)
from fowt_platform import assemble_volturnus_static_restoring_aligned_runtime_assembly
from run_openfast_moment_angle_consistency import (
    REFERENCE_MANIFEST,
    REFERENCE_STATE_JSON,
    RELAXATION_ADD_BLIN_DIAGONAL,
    _case_result,
    _frozen_equilibrium_load,
    _load_candidate_reference_state,
    _named_six_vector,
    _platform_reference_state,
    _release_tail_stability,
    _run_phase,
    _static_low_order_solution,
    _tail_mean_state,
)


DEFAULT_TARGET_PITCH_DEG = (2.0, 3.0, 5.0, 7.0, 10.0, 12.0)


def _six_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite six-component vector")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _target_pitch_sequence(values: Iterable[Any]) -> tuple[float, ...]:
    targets = tuple(float(value) for value in values)
    if not targets:
        raise ValueError("at least one target pitch angle is required")
    if any(not np.isfinite(value) or value <= 0.0 for value in targets):
        raise ValueError("target pitch angles must be finite and positive")
    if any(later <= earlier for earlier, later in zip(targets, targets[1:])):
        raise ValueError("target pitch angles must be strictly increasing")
    return targets


def _frozen_equilibrium_load_to_openfast_reference(
    frozen_load: Any,
    *,
    nominal_reference: dict[str, float],
) -> np.ndarray:
    """Invert the static audit's OpenFAST-reference to frozen-axis rotation."""

    frozen = _six_vector("frozen-equilibrium generalized load", frozen_load)
    rotation = _openfast_platform_to_inertial_rotation(
        platform_roll_deg=0.0,
        platform_pitch_deg=nominal_reference["pitch_deg"],
        platform_yaw_deg=0.0,
    )
    result = np.concatenate((rotation @ frozen[:3], rotation @ frozen[3:]))
    return _six_vector("OpenFAST-reference generalized load", result)


def _target_case(
    target_pitch_deg: float,
    *,
    nominal_reference: dict[str, float],
) -> dict[str, object]:
    """Construct one pure frozen-axis pitch moment for a low-order target angle."""

    target = float(target_pitch_deg)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("target_pitch_deg must be finite and positive")
    runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    matrices = runtime_assembly.base_matrices
    unit_pitch_moment = np.zeros(6)
    unit_pitch_moment[4] = 1.0
    pitch_compliance_rad_per_nm = float(
        solve_incremental_static_offset(matrices, unit_pitch_moment).position[4]
    )
    if not np.isfinite(pitch_compliance_rad_per_nm) or abs(pitch_compliance_rad_per_nm) <= 1.0e-18:
        raise ValueError("frozen low-order pitch compliance must be finite and nonzero")
    frozen_load = np.zeros(6)
    frozen_load[4] = np.deg2rad(target) / pitch_compliance_rad_per_nm
    openfast_load = _frozen_equilibrium_load_to_openfast_reference(
        frozen_load,
        nominal_reference=nominal_reference,
    )
    return {
        "target_low_order_pitch_increment_deg": target,
        "low_order_runtime_provenance": runtime_assembly.provenance,
        "frozen_axis_pitch_compliance_rad_per_nm": pitch_compliance_rad_per_nm,
        "frozen_equilibrium_load_si": _named_six_vector(frozen_load),
        "openfast_add_f0_about_wrp_si": _named_six_vector(openfast_load),
        "openfast_add_f0_pitch_component_mnm": float(openfast_load[4] / 1.0e6),
        "openfast_preload": openfast_load,
    }


def _case_id(target_pitch_deg: float) -> str:
    return f"pitch_target_{target_pitch_deg:g}deg".replace(".", "p")


def _case_summary(result: dict[str, object], target: dict[str, object]) -> dict[str, object]:
    low_order = result["low_order"]
    openfast = result["openfast"]
    if not isinstance(low_order, dict) or not isinstance(openfast, dict):
        raise ValueError("static comparison result is incomplete")
    static_offset = low_order["static_offset"]
    openfast_increment = openfast[
        "increment_from_matched_zero_preload_release_tail_mean_deg"
    ]
    if not isinstance(static_offset, dict) or not isinstance(openfast_increment, dict):
        raise ValueError("static comparison result is missing pitch increments")
    target_pitch = float(target["target_low_order_pitch_increment_deg"])
    low_pitch = float(static_offset["pitch_deg"])
    openfast_pitch = float(openfast_increment["pitch_deg"])
    absolute_error = low_pitch - openfast_pitch
    return {
        "target_low_order_pitch_increment_deg": target_pitch,
        "low_order_pitch_increment_deg": low_pitch,
        "openfast_pitch_increment_deg": openfast_pitch,
        "low_order_minus_openfast_pitch_deg": absolute_error,
        "openfast_over_low_order_pitch_ratio": (
            openfast_pitch / low_pitch if abs(low_pitch) > 1.0e-12 else None
        ),
        "relative_low_order_minus_openfast_pitch_percent": (
            100.0 * absolute_error / low_pitch if abs(low_pitch) > 1.0e-12 else None
        ),
        "openfast_roll_increment_deg": float(openfast_increment["roll_deg"]),
        "original_damping_release_tail_range_deg": result[
            "original_damping_release_tail_range_deg"
        ],
        "original_damping_release_stability": result[
            "original_damping_release_stability"
        ],
    }


def run_sweep(
    *,
    work_dir: Path,
    openfast_binary: Path,
    target_pitch_deg: Iterable[Any] = DEFAULT_TARGET_PITCH_DEG,
    relaxation_duration_s: float = 300.0,
    release_duration_s: float = 600.0,
    output_step_s: float = 1.0,
    tail_window_s: float = 150.0,
    nominal_reference_state_path: Path = REFERENCE_STATE_JSON,
) -> dict[str, object]:
    """Run a positive-pitch end-state sweep without fitting any parameter."""

    targets = _target_pitch_sequence(target_pitch_deg)
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")
    if relaxation_duration_s <= 0.0 or release_duration_s <= 0.0 or output_step_s <= 0.0:
        raise ValueError("durations and output_step_s must be positive")
    if not 0.0 < tail_window_s <= release_duration_s:
        raise ValueError("tail_window_s must be positive and no greater than release_duration_s")

    nominal_reference = _load_candidate_reference_state(nominal_reference_state_path)
    reference_state = _platform_reference_state(nominal_reference)
    zero = np.zeros(6)
    baseline_relaxation = _run_phase(
        phase_root=work_dir / "zero_preload" / "relaxation",
        openfast_binary=openfast_binary,
        duration_s=relaxation_duration_s,
        output_step_s=output_step_s,
        initial_state=reference_state,
        preload=zero,
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=RELAXATION_ADD_BLIN_DIAGONAL,
    )
    baseline_release = _run_phase(
        phase_root=work_dir / "zero_preload" / "release",
        openfast_binary=openfast_binary,
        duration_s=release_duration_s,
        output_step_s=output_step_s,
        initial_state=_tail_mean_state(baseline_relaxation),
        preload=zero,
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=None,
    )

    cases: list[dict[str, object]] = []
    for target_pitch in targets:
        target = _target_case(target_pitch, nominal_reference=nominal_reference)
        openfast_preload = target.pop("openfast_preload")
        if not isinstance(openfast_preload, np.ndarray):
            raise RuntimeError("target case did not retain its OpenFAST preload")
        case: dict[str, object] = {
            "case_id": _case_id(target_pitch),
            "target": target,
        }
        try:
            comparison = _case_result(
                case_id=_case_id(target_pitch),
                preload=openfast_preload,
                baseline_release=baseline_release,
                nominal_reference=nominal_reference,
                work_dir=work_dir,
                openfast_binary=openfast_binary,
                relaxation_duration_s=relaxation_duration_s,
                release_duration_s=release_duration_s,
                output_step_s=output_step_s,
                tail_window_s=tail_window_s,
            )
        except Exception as error:  # Keep later targets observable if one leaves the local regime.
            case["status"] = "failed_before_release_comparison"
            case["error"] = f"{type(error).__name__}: {error}"
        else:
            case["status"] = "completed"
            case["comparison"] = comparison
            case["summary"] = _case_summary(comparison, target)
        cases.append(case)

    completed = [case for case in cases if case["status"] == "completed"]
    stable = [
        case
        for case in completed
        if isinstance(case.get("summary"), dict)
        and isinstance(case["summary"].get("original_damping_release_stability"), dict)
        and bool(case["summary"]["original_damping_release_stability"].get("passes"))
    ]
    return {
        "evidence_level": "OpenFAST/low-order one-sided static pitch end-state sweep",
        "scope": {
            "purpose": "compare final pitch increments under the same frozen-axis pitch moment and identify where the frozen small-angle low-order reading departs from OpenFAST",
            "is_controller_validation": False,
            "is_wind_load_replay": False,
            "is_field_operating_load_envelope": False,
            "positive_pitch_side_only": True,
            "environment": "AeroDyn, InflowWind, ServoDyn, waves, and current disabled; HydroDyn and MoorDyn retained",
            "load_definition": "HydroDyn AddF0 static preload at the WAMIT/OpenFAST reference point, obtained by rotating a pure frozen-axis pitch moment",
            "readout_definition": "matched zero-preload original-damping release tail mean",
            "stability_definition": "original-damping release tail range and consecutive-tail mean drift only",
            "loading_path_boundary": "HydroDyn AddF0 is a static input. This endpoint comparison does not represent a time-continuous load ramp or a path-dependent mooring response.",
            "parameter_fitting": "none",
        },
        "reference": {
            "reference_manifest": str(REFERENCE_MANIFEST),
            "nominal_reference_state_path": str(nominal_reference_state_path),
            "nominal_reference_state": nominal_reference,
            "openfast_binary": str(openfast_binary),
            "relaxation_duration_s": relaxation_duration_s,
            "release_duration_s": release_duration_s,
            "output_step_s": output_step_s,
            "tail_window_s": tail_window_s,
            "target_pitch_deg": list(targets),
            "low_order_runtime_provenance": (
                "static_restoring_aligned_with_aux_frc_inertia_mooring"
            ),
        },
        "zero_preload": {
            "temporary_relaxation": baseline_relaxation,
            "original_damping_release": baseline_release,
            "original_damping_release_stability": _release_tail_stability(baseline_release),
        },
        "summary": {
            "requested_case_count": len(cases),
            "completed_case_count": len(completed),
            "stable_release_case_count": len(stable),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--openfast",
        type=Path,
        default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast"),
    )
    parser.add_argument(
        "--target-pitch-deg",
        type=float,
        nargs="+",
        default=DEFAULT_TARGET_PITCH_DEG,
    )
    parser.add_argument("--relaxation-duration-s", type=float, default=300.0)
    parser.add_argument("--release-duration-s", type=float, default=600.0)
    parser.add_argument("--output-step-s", type=float, default=1.0)
    parser.add_argument("--tail-window-s", type=float, default=150.0)
    parser.add_argument(
        "--nominal-reference-state-json",
        type=Path,
        default=REFERENCE_STATE_JSON,
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = run_sweep(
        work_dir=args.work_dir,
        openfast_binary=args.openfast,
        target_pitch_deg=args.target_pitch_deg,
        relaxation_duration_s=args.relaxation_duration_s,
        release_duration_s=args.release_duration_s,
        output_step_s=args.output_step_s,
        tail_window_s=args.tail_window_s,
        nominal_reference_state_path=args.nominal_reference_state_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
