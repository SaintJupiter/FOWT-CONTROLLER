#!/usr/bin/env python3
"""Compare one local pitch free response after static-restoring alignment.

The audit releases both models from the frozen no-wind reference pose plus a
small pitch displacement.  OpenFAST retains its original HydroDyn and MoorDyn
dissipation, while the low-order candidate deliberately uses zero damping.
Accordingly, the comparison is restricted to the pitch-response direction and
the local oscillation period.  It is not a damping fit, a full six-DOF
validation, or a controller-performance test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from fowt_platform import (
    IncrementalState,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    simulate_linear_free_response,
)
from run_openfast_moment_angle_consistency import _disable_aerodynamic_modules
from run_openfast_prescribed_wind_node import (
    ED_NAME,
    MODEL_ZIP,
    REFERENCE_STATE_JSON,
    _assert_output_reaches_time_horizon,
    _configure_case,
    _extract_clean_model,
    _load_candidate_reference_state,
    _read_openfast_output,
    _sha256,
)
from run_openfast_prescribed_wind_relaxation_release_audit import (
    _initial_state_from_nominal_reference,
    _normalise_platform_initial_state,
    _set_platform_initial_state,
)


REFERENCE_MANIFEST = (
    ROOT / "configs" / "reference_platforms" / "volturnus_s_openfast_v1_1_16.json"
)
PITCH_INDEX = 4


def _positive_peak_times(
    time_s: Any,
    values: Any,
    *,
    minimum_amplitude: float,
) -> np.ndarray:
    """Return positive interior peaks above an explicit noise threshold."""

    time = np.asarray(time_s, dtype=float)
    signal = np.asarray(values, dtype=float)
    if time.ndim != 1 or signal.shape != time.shape or time.size < 3:
        raise ValueError("time_s and values must be matching one-dimensional arrays")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(signal)):
        raise ValueError("time_s and values must contain only finite values")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time_s must be strictly increasing")
    if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0.0:
        raise ValueError("minimum_amplitude must be finite and non-negative")
    indices = np.flatnonzero(
        (signal[1:-1] > signal[:-2])
        & (signal[1:-1] >= signal[2:])
        & (signal[1:-1] >= minimum_amplitude)
    ) + 1
    return np.asarray(time[indices], dtype=float)


def _dominant_positive_period_s(
    time_s: Any,
    values: Any,
    *,
    minimum_amplitude: float,
) -> tuple[float, np.ndarray]:
    """Summarise a decaying or undamped local response by positive peak spacing."""

    peaks = _positive_peak_times(
        time_s,
        values,
        minimum_amplitude=minimum_amplitude,
    )
    if peaks.size < 3:
        raise ValueError("at least three positive peaks are required for a period")
    return float(np.median(np.diff(peaks))), peaks


def _release_initial_state(
    nominal_reference: dict[str, float],
    *,
    pitch_increment_deg: float,
) -> dict[str, float]:
    if not np.isfinite(pitch_increment_deg) or pitch_increment_deg <= 0.0:
        raise ValueError("pitch_increment_deg must be finite and positive")
    state = _initial_state_from_nominal_reference(nominal_reference)
    state["pitch_deg"] += float(pitch_increment_deg)
    return _normalise_platform_initial_state(state)


def _configure_release_case(
    semi_dir: Path,
    *,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
) -> Path:
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
    return fst_path


def _run_openfast_release(
    *,
    work_dir: Path,
    openfast_binary: Path,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
) -> dict[str, Any]:
    semi_dir = _extract_clean_model(work_dir)
    fst_path = _configure_release_case(
        semi_dir,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=initial_state,
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
            "OpenFAST aligned pitch free-response run failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    columns, _ = _read_openfast_output(semi_dir / f"{fst_path.stem}.out")
    _assert_output_reaches_time_horizon(
        columns,
        duration_s=duration_s,
        output_step_s=output_step_s,
    )
    return {
        "input_tree": str(semi_dir),
        "initial_state": initial_state,
        "time_s": columns["Time"],
        "pitch_deg": columns["PtfmPitch"],
    }


def run_audit(
    *,
    work_dir: Path,
    output_json: Path,
    openfast_binary: Path,
    duration_s: float = 150.0,
    output_step_s: float = 0.1,
    pitch_increment_deg: float = 1.0,
    minimum_peak_amplitude_deg: float = 0.05,
) -> dict[str, object]:
    """Run one explicit OpenFAST/low-order local pitch release comparison."""

    if min(duration_s, output_step_s, pitch_increment_deg) <= 0.0:
        raise ValueError("duration_s, output_step_s, and pitch_increment_deg must be positive")
    if not np.isfinite(minimum_peak_amplitude_deg) or minimum_peak_amplitude_deg < 0.0:
        raise ValueError("minimum_peak_amplitude_deg must be finite and non-negative")
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")

    nominal_reference = _load_candidate_reference_state(REFERENCE_STATE_JSON)
    initial_state = _release_initial_state(
        nominal_reference,
        pitch_increment_deg=pitch_increment_deg,
    )
    openfast = _run_openfast_release(
        work_dir=work_dir,
        openfast_binary=openfast_binary,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=initial_state,
    )
    openfast_pitch_increment_deg = (
        np.asarray(openfast["pitch_deg"], dtype=float) - nominal_reference["pitch_deg"]
    )
    openfast_period_s, openfast_peak_times_s = _dominant_positive_period_s(
        openfast["time_s"],
        openfast_pitch_increment_deg,
        minimum_amplitude=minimum_peak_amplitude_deg,
    )

    assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    low_order_time_s = np.asarray(openfast["time_s"], dtype=float)
    initial_position = np.zeros(6)
    initial_position[PITCH_INDEX] = np.deg2rad(pitch_increment_deg)
    low_order_response = simulate_linear_free_response(
        matrices=assembly.base_matrices,
        initial_state=IncrementalState(
            position=initial_position,
            velocity=np.zeros(6),
        ),
        time_s=low_order_time_s,
    )
    low_order_pitch_increment_deg = np.rad2deg(
        low_order_response.position[:, PITCH_INDEX]
    )
    low_order_period_s, low_order_peak_times_s = _dominant_positive_period_s(
        low_order_time_s,
        low_order_pitch_increment_deg,
        minimum_amplitude=minimum_peak_amplitude_deg,
    )
    period_error_percent = 100.0 * (low_order_period_s - openfast_period_s) / openfast_period_s

    result: dict[str, object] = {
        "evidence_level": "one local no-environment pitch free-response comparison",
        "boundaries": {
            "is_damping_calibration": False,
            "is_full_six_dof_validation": False,
            "is_controller_validation": False,
            "openfast_dissipation": "original HydroDyn and MoorDyn input retained",
            "low_order_dissipation": "zero damping by construction",
            "comparison_scope": "pitch response direction and local period only",
        },
        "reference": {
            "model_zip": str(MODEL_ZIP),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "openfast_binary_sha256": _sha256(openfast_binary),
            "reference_state_path": str(REFERENCE_STATE_JSON),
            "reference_state": nominal_reference,
            "duration_s": duration_s,
            "output_step_s": output_step_s,
            "pitch_increment_deg": pitch_increment_deg,
        },
        "openfast": {
            "initial_state": initial_state,
            "positive_peak_times_s": [float(value) for value in openfast_peak_times_s],
            "dominant_positive_period_s": openfast_period_s,
            "pitch_increment_range_deg": [
                float(np.min(openfast_pitch_increment_deg)),
                float(np.max(openfast_pitch_increment_deg)),
            ],
        },
        "low_order": {
            "runtime_provenance": assembly.provenance,
            "positive_peak_times_s": [float(value) for value in low_order_peak_times_s],
            "dominant_positive_period_s": low_order_period_s,
            "pitch_increment_range_deg": [
                float(np.min(low_order_pitch_increment_deg)),
                float(np.max(low_order_pitch_increment_deg)),
            ],
        },
        "comparison": {
            "low_order_minus_openfast_period_s": low_order_period_s - openfast_period_s,
            "low_order_relative_period_error_percent": period_error_percent,
            "pitch_response_direction_matches": bool(
                np.sign(openfast_pitch_increment_deg[0])
                == np.sign(low_order_pitch_increment_deg[0])
            ),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--openfast",
        type=Path,
        default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast"),
    )
    parser.add_argument("--duration-s", type=float, default=150.0)
    parser.add_argument("--output-step-s", type=float, default=0.1)
    parser.add_argument("--pitch-increment-deg", type=float, default=1.0)
    parser.add_argument("--minimum-peak-amplitude-deg", type=float, default=0.05)
    args = parser.parse_args()
    result = run_audit(
        work_dir=args.work_dir,
        output_json=args.output_json,
        openfast_binary=args.openfast,
        duration_s=args.duration_s,
        output_step_s=args.output_step_s,
        pitch_increment_deg=args.pitch_increment_deg,
        minimum_peak_amplitude_deg=args.minimum_peak_amplitude_deg,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
