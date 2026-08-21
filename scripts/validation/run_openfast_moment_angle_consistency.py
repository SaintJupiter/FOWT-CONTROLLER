#!/usr/bin/env python3
"""Compare low-order roll/pitch offsets with OpenFAST under one static preload.

This is deliberately an open-loop, no-wind and no-wave audit.  HydroDyn's
``AddF0`` input applies the prescribed six-component static preload at the
WAMIT/OpenFAST reference point.  The same load, rotated into the frozen
equilibrium platform axes, is passed to the low-order static solver.

It is not a controller validation, a damping calibration, or a proof that the
low-order model reproduces every OpenFAST response.  It addresses the narrower
question that matters before reconnecting the controller: do both models give
comparable roll and pitch offsets for the same explicitly prescribed moment?
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from fowt_platform.incremental import (
    openfast_reference_vector_to_frozen_equilibrium_axes,
    solve_incremental_static_offset,
)
from fowt_platform import assemble_volturnus_static_restoring_aligned_runtime_assembly
from run_openfast_prescribed_wind_node import (
    ED_NAME,
    MODEL_ZIP,
    REFERENCE_STATE_JSON,
    _assert_output_reaches_time_horizon,
    _configure_case,
    _extract_clean_model,
    _load_candidate_reference_state,
    _read_openfast_output,
    _replace_field,
    _sha256,
)
from run_openfast_prescribed_wind_relaxation_release_audit import (
    HYDRODYN_NAME,
    PLATFORM_STATE_FIELDS,
    RELAXATION_ADD_BLIN_DIAGONAL,
    _add_hydrodyn_linear_damping,
    _late_window_statistics,
    _normalise_platform_initial_state,
    _set_platform_initial_state,
)


REFERENCE_MANIFEST = (
    ROOT / "configs" / "reference_platforms" / "volturnus_s_openfast_v1_1_16.json"
)
LOAD_NAMES = ("surge_n", "sway_n", "heave_n", "roll_nm", "pitch_nm", "yaw_nm")
MAX_RELEASE_TAIL_RANGE_DEG = 0.01
MAX_RELEASE_TAIL_MEAN_DRIFT_DEG = 0.005


def _six_vector(name: str, value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite six-component vector")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _named_six_vector(vector: np.ndarray) -> dict[str, float]:
    values = _six_vector("vector", vector)
    return {name: float(values[index]) for index, name in enumerate(LOAD_NAMES)}


def _platform_reference_state(nominal: dict[str, float]) -> dict[str, float]:
    return _normalise_platform_initial_state(
        {
            "surge_m": nominal["surge_m"],
            "sway_m": 0.0,
            "heave_m": nominal["heave_m"],
            "roll_deg": 0.0,
            "pitch_deg": nominal["pitch_deg"],
            "yaw_deg": 0.0,
        }
    )


def _disable_aerodynamic_modules(fst_text: str) -> str:
    """Leave only ElastoDyn, HydroDyn, and MoorDyn in this static-load audit."""

    updated = fst_text
    for field, value in (("CompInflow", "0"), ("CompAero", "0"), ("CompServo", "0")):
        updated = _replace_field(updated, field, value)
    return updated


def _set_hydrodyn_add_f0(hydrodyn_text: str, preload: Any) -> str:
    """Replace exactly one six-DOF HydroDyn ``AddF0`` preload vector."""

    values = _six_vector("HydroDyn AddF0 preload", preload)
    lines = hydrodyn_text.splitlines(keepends=True)
    try:
        start = next(index for index, line in enumerate(lines) if "AddF0" in line)
    except StopIteration as error:
        raise ValueError("HydroDyn input is missing AddF0") from error
    if start + 6 > len(lines):
        raise ValueError("HydroDyn AddF0 vector is incomplete")

    for index, value in enumerate(values):
        suffix = ""
        newline = "\n" if lines[start + index].endswith("\n") else ""
        if index == 0:
            tokens = lines[start].rstrip("\n").split(maxsplit=1)
            suffix = f" {tokens[1]}" if len(tokens) == 2 else ""
        lines[start + index] = f"{value:.12g}{suffix}{newline}"
    return "".join(lines)


def _read_hydrodyn_add_f0(hydrodyn_text: str) -> np.ndarray:
    """Read the configured preload back from an input copy for audit evidence."""

    lines = hydrodyn_text.splitlines()
    try:
        start = next(index for index, line in enumerate(lines) if "AddF0" in line)
    except StopIteration as error:
        raise ValueError("HydroDyn input is missing AddF0") from error
    try:
        values = [float(lines[start + index].split()[0]) for index in range(6)]
    except (IndexError, ValueError) as error:
        raise ValueError("HydroDyn AddF0 vector cannot be read back") from error
    return _six_vector("configured HydroDyn AddF0 preload", values)


def _frozen_equilibrium_load(openfast_preload: Any, *, nominal_reference: dict[str, float]) -> np.ndarray:
    """Rotate a WRP load into the low-order model's frozen equilibrium axes."""

    preload = _six_vector("OpenFAST AddF0 preload", openfast_preload)
    force = openfast_reference_vector_to_frozen_equilibrium_axes(
        preload[:3],
        equilibrium_platform_roll_deg=0.0,
        equilibrium_platform_pitch_deg=nominal_reference["pitch_deg"],
        equilibrium_platform_yaw_deg=0.0,
    )
    moment = openfast_reference_vector_to_frozen_equilibrium_axes(
        preload[3:],
        equilibrium_platform_roll_deg=0.0,
        equilibrium_platform_pitch_deg=nominal_reference["pitch_deg"],
        equilibrium_platform_yaw_deg=0.0,
    )
    return _six_vector("frozen-equilibrium preload", np.concatenate((force, moment)))


def _configure_no_environment_case(
    semi_dir: Path,
    *,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
    preload: np.ndarray,
    temporary_add_blin_diagonal: np.ndarray | None,
) -> tuple[Path, np.ndarray]:
    """Build one OpenFAST copy with only a known HydroDyn static preload."""

    fst_path = _configure_case(
        semi_dir,
        wind_speed_mps=0.1,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=initial_state["surge_m"],
        equilibrium_heave_m=initial_state["heave_m"],
        equilibrium_pitch_deg=initial_state["pitch_deg"],
    )
    fst_path.write_text(
        _disable_aerodynamic_modules(fst_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    ed_path = semi_dir / ED_NAME
    ed_path.write_text(
        _set_platform_initial_state(ed_path.read_text(encoding="utf-8"), initial_state),
        encoding="utf-8",
    )
    hydrodyn_path = semi_dir / HYDRODYN_NAME
    hydrodyn_text = _set_hydrodyn_add_f0(
        hydrodyn_path.read_text(encoding="utf-8"), preload
    )
    if temporary_add_blin_diagonal is not None:
        hydrodyn_text = _add_hydrodyn_linear_damping(
            hydrodyn_text, temporary_add_blin_diagonal
        )
    hydrodyn_path.write_text(hydrodyn_text, encoding="utf-8")
    return fst_path, _read_hydrodyn_add_f0(hydrodyn_text)


def _run_phase(
    *,
    phase_root: Path,
    openfast_binary: Path,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
    preload: np.ndarray,
    tail_window_s: float,
    temporary_add_blin_diagonal: np.ndarray | None,
) -> dict[str, object]:
    semi_dir = _extract_clean_model(phase_root)
    fst_path, configured_preload = _configure_no_environment_case(
        semi_dir,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=initial_state,
        preload=preload,
        temporary_add_blin_diagonal=temporary_add_blin_diagonal,
    )
    completed = subprocess.run(
        [str(openfast_binary), fst_path.name],
        cwd=semi_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenFAST moment/angle phase failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    columns, _ = _read_openfast_output(semi_dir / f"{fst_path.stem}.out")
    _assert_output_reaches_time_horizon(
        columns, duration_s=duration_s, output_step_s=output_step_s
    )
    return {
        "input_tree": str(semi_dir),
        "initial_state": _normalise_platform_initial_state(initial_state),
        "configured_add_f0_about_wrp_si": _named_six_vector(configured_preload),
        "hydrodyn_damping_mode": (
            "original_public_input"
            if temporary_add_blin_diagonal is None
            else "temporary_numerical_relaxation_add_blin"
        ),
        "sample_count": int(columns["Time"].size),
        "late_window_statistics": _late_window_statistics(
            columns, duration_s=duration_s, tail_window_s=tail_window_s
        ),
    }


def _tail_mean_state(phase: dict[str, object]) -> dict[str, float]:
    audit = phase["late_window_statistics"]
    if not isinstance(audit, dict):
        raise ValueError("OpenFAST phase is missing tail statistics")
    values = audit["second_window_statistics"]
    if not isinstance(values, dict):
        raise ValueError("OpenFAST phase is missing second-window statistics")
    state = {
        state_name: float(values[state_name]["mean"])
        for state_name, _ in PLATFORM_STATE_FIELDS
    }
    return _normalise_platform_initial_state(state)


def _tail_range(phase: dict[str, object], state_name: str) -> float:
    audit = phase["late_window_statistics"]
    if not isinstance(audit, dict):
        raise ValueError("OpenFAST phase is missing tail statistics")
    values = audit["second_window_statistics"]
    if not isinstance(values, dict) or state_name not in values:
        raise ValueError(f"OpenFAST phase is missing {state_name} tail range")
    return float(values[state_name]["range"])


def _release_tail_stability(phase: dict[str, object]) -> dict[str, object]:
    """Check whether the release tail is quiet relative to the test increment."""

    audit = phase["late_window_statistics"]
    if not isinstance(audit, dict):
        raise ValueError("OpenFAST phase is missing tail statistics")
    drift = audit["second_minus_first_mean"]
    if not isinstance(drift, dict):
        raise ValueError("OpenFAST phase is missing tail-window mean differences")
    components = {
        state_name: {
            "tail_range_deg": _tail_range(phase, state_name),
            "second_minus_first_tail_mean_deg": float(drift[state_name]),
        }
        for state_name in ("roll_deg", "pitch_deg")
    }
    is_stable = all(
        values["tail_range_deg"] <= MAX_RELEASE_TAIL_RANGE_DEG
        and abs(values["second_minus_first_tail_mean_deg"])
        <= MAX_RELEASE_TAIL_MEAN_DRIFT_DEG
        for values in components.values()
    )
    return {
        "criteria": {
            "maximum_tail_range_deg": MAX_RELEASE_TAIL_RANGE_DEG,
            "maximum_abs_second_minus_first_tail_mean_deg": MAX_RELEASE_TAIL_MEAN_DRIFT_DEG,
        },
        "components": components,
        "passes": is_stable,
    }


def _static_low_order_solution(
    frozen_preload: np.ndarray) -> dict[str, object]:
    runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    matrices = runtime_assembly.base_matrices
    offset = solve_incremental_static_offset(matrices, frozen_preload)
    position = offset.position
    return {
        "runtime_provenance": runtime_assembly.provenance,
        "input_load_frozen_equilibrium_si": _named_six_vector(frozen_preload),
        "static_offset": {
            "translation_m": [float(value) for value in position[:3]],
            "rotation_rad": [float(value) for value in position[3:]],
            "roll_deg": float(np.rad2deg(position[3])),
            "pitch_deg": float(np.rad2deg(position[4])),
        },
        "static_residual_max_abs": float(np.max(np.abs(offset.residual_load))),
    }


def _case_result(
    *,
    case_id: str,
    preload: np.ndarray,
    baseline_release: dict[str, object],
    nominal_reference: dict[str, float],
    work_dir: Path,
    openfast_binary: Path,
    relaxation_duration_s: float,
    release_duration_s: float,
    output_step_s: float,
    tail_window_s: float,
) -> dict[str, object]:
    reference_state = _platform_reference_state(nominal_reference)
    relaxation = _run_phase(
        phase_root=work_dir / case_id / "relaxation",
        openfast_binary=openfast_binary,
        duration_s=relaxation_duration_s,
        output_step_s=output_step_s,
        initial_state=reference_state,
        preload=preload,
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=RELAXATION_ADD_BLIN_DIAGONAL,
    )
    release_initial_state = _tail_mean_state(relaxation)
    release = _run_phase(
        phase_root=work_dir / case_id / "release",
        openfast_binary=openfast_binary,
        duration_s=release_duration_s,
        output_step_s=output_step_s,
        initial_state=release_initial_state,
        preload=preload,
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=None,
    )
    baseline_state = _tail_mean_state(baseline_release)
    release_state = _tail_mean_state(release)
    frozen_preload = _frozen_equilibrium_load(preload, nominal_reference=nominal_reference)
    low_order = _static_low_order_solution(frozen_preload)
    openfast_roll_increment = release_state["roll_deg"] - baseline_state["roll_deg"]
    openfast_pitch_increment = release_state["pitch_deg"] - baseline_state["pitch_deg"]
    low_order_roll = float(low_order["static_offset"]["roll_deg"])
    low_order_pitch = float(low_order["static_offset"]["pitch_deg"])
    return {
        "case_id": case_id,
        "openfast_add_f0_about_wrp_si": _named_six_vector(preload),
        "low_order": low_order,
        "openfast": {
            "temporary_relaxation": relaxation,
            "original_damping_release": release,
            "increment_from_matched_zero_preload_release_tail_mean_deg": {
                "roll_deg": float(openfast_roll_increment),
                "pitch_deg": float(openfast_pitch_increment),
            },
        },
        "angle_difference_low_order_minus_openfast_deg": {
            "roll_deg": float(low_order_roll - openfast_roll_increment),
            "pitch_deg": float(low_order_pitch - openfast_pitch_increment),
        },
        "original_damping_release_tail_range_deg": {
            "roll_deg": _tail_range(release, "roll_deg"),
            "pitch_deg": _tail_range(release, "pitch_deg"),
        },
        "original_damping_release_stability": _release_tail_stability(release),
        "state_transfer": {
            "release_initial_state_source": "temporary_relaxation.second_window_mean_state_for_release_only",
            "release_initial_state": release_initial_state,
        },
    }


def run_audit(
    *,
    work_dir: Path,
    openfast_binary: Path,
    moment_magnitude_nm: float = 1.0e7,
    relaxation_duration_s: float = 300.0,
    release_duration_s: float = 600.0,
    output_step_s: float = 1.0,
    tail_window_s: float = 150.0,
    nominal_reference_state_path: Path = REFERENCE_STATE_JSON,
) -> dict[str, object]:
    """Run pure-roll, pure-pitch, and coupled static-preload comparisons."""

    if not np.isfinite(moment_magnitude_nm) or moment_magnitude_nm <= 0.0:
        raise ValueError("moment_magnitude_nm must be finite and positive")
    if relaxation_duration_s <= 0.0 or release_duration_s <= 0.0 or output_step_s <= 0.0:
        raise ValueError("durations and output_step_s must be positive")
    if not 0.0 < tail_window_s <= release_duration_s:
        raise ValueError("tail_window_s must be positive and no greater than release_duration_s")
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")

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
    magnitude = float(moment_magnitude_nm)
    cases = (
        ("pure_roll_positive", np.array([0.0, 0.0, 0.0, magnitude, 0.0, 0.0])),
        ("pure_roll_negative", np.array([0.0, 0.0, 0.0, -magnitude, 0.0, 0.0])),
        ("pure_pitch_positive", np.array([0.0, 0.0, 0.0, 0.0, magnitude, 0.0])),
        ("pure_pitch_negative", np.array([0.0, 0.0, 0.0, 0.0, -magnitude, 0.0])),
        ("coupled_roll_pitch_positive", np.array([0.0, 0.0, 0.0, magnitude, magnitude, 0.0])),
    )
    results = [
        _case_result(
            case_id=case_id,
            preload=preload,
            baseline_release=baseline_release,
            nominal_reference=nominal_reference,
            work_dir=work_dir,
            openfast_binary=openfast_binary,
            relaxation_duration_s=relaxation_duration_s,
            release_duration_s=release_duration_s,
            output_step_s=output_step_s,
            tail_window_s=tail_window_s,
        )
        for case_id, preload in cases
    ]
    return {
        "evidence_level": "OpenFAST/low-order same-static-preload roll-pitch consistency audit",
        "scope": {
            "purpose": "compare equilibrium-near roll and pitch increments for the same user-prescribed HydroDyn AddF0 moment",
            "is_controller_validation": False,
            "is_damping_calibration": False,
            "is_full_six_dof_time_history_validation": False,
            "is_higher_wind_or_wave_response_validation": False,
            "environment": "AeroDyn, InflowWind, ServoDyn, waves, and current disabled; HydroDyn and MoorDyn retained",
            "load_definition": "HydroDyn AddF0 static preload at the WAMIT/OpenFAST reference point",
            "comparison_reference": "matched zero-preload original-damping release tail mean",
            "low_order_load_mapping": "OpenFAST WRP load is rotated into frozen equilibrium platform axes before static low-order solution",
        },
        "reference": {
            "model_zip": str(MODEL_ZIP),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "openfast_binary_sha256": _sha256(openfast_binary),
            "nominal_reference_state_path": str(nominal_reference_state_path),
            "nominal_reference_state": nominal_reference,
            "moment_magnitude_nm": magnitude,
        },
        "zero_preload": {
            "temporary_relaxation": baseline_relaxation,
            "original_damping_release": baseline_release,
            "original_damping_release_stability": _release_tail_stability(
                baseline_release
            ),
            "state_transfer": {
                "release_initial_state_source": "temporary_relaxation.second_window_mean_state_for_release_only",
                "release_initial_state": _tail_mean_state(baseline_relaxation),
            },
        },
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--openfast",
        type=Path,
        default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast"),
    )
    parser.add_argument("--moment-magnitude-nm", type=float, default=1.0e7)
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
    result = run_audit(
        work_dir=args.work_dir,
        openfast_binary=args.openfast,
        moment_magnitude_nm=args.moment_magnitude_nm,
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
