#!/usr/bin/env python3
"""Locate and check a prescribed-wind OpenFAST reference candidate.

This is a narrow OpenFAST-side audit for one prescribed rotor condition. A
temporary diagonal HydroDyn damping matrix is used only to locate a candidate
platform pose. A fresh copy of the public input deck is then released with its
original HydroDyn damping. The script records both stages but never creates a
reference-state artifact or claims a physical operating equilibrium.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from run_openfast_prescribed_wind_node import (
    ED_NAME,
    MODEL_ZIP,
    PRESCRIBED_BLADE_PITCH_DEG,
    PRESCRIBED_ROTOR_SPEED_RPM,
    REFERENCE_STATE_JSON,
    _assert_output_reaches_time_horizon,
    _assert_prescribed_rotor_output,
    _configure_case,
    _extract_clean_model,
    _load_candidate_reference_state,
    _read_openfast_output,
    _replace_field,
    _sha256,
)


HYDRODYN_NAME = "IEA-15-240-RWT-UMaineSemi_HydroDyn.dat"
PLATFORM_STATE_FIELDS = (
    ("surge_m", "PtfmSurge"),
    ("sway_m", "PtfmSway"),
    ("heave_m", "PtfmHeave"),
    ("roll_deg", "PtfmRoll"),
    ("pitch_deg", "PtfmPitch"),
    ("yaw_deg", "PtfmYaw"),
)

# These values are the temporary numerical-relaxation diagonal documented for
# the no-wind audit. They are not a physical HydroDyn damping calibration.
RELAXATION_ADD_BLIN_DIAGONAL = np.array(
    [2.925e6, 2.925e6, 2.848e7, 2.512e10, 2.512e10, 6.650e9],
    dtype=float,
)


def _normalise_platform_initial_state(state: dict[str, float]) -> dict[str, float]:
    expected = {name for name, _ in PLATFORM_STATE_FIELDS}
    if set(state) != expected:
        missing = sorted(expected.difference(state))
        unexpected = sorted(set(state).difference(expected))
        raise ValueError(
            "platform initial state must define exactly six fields; "
            f"missing={missing}, unexpected={unexpected}"
        )
    values = {name: float(value) for name, value in state.items()}
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("platform initial state values must be finite")
    return values


def _initial_state_from_nominal_reference(reference: dict[str, float]) -> dict[str, float]:
    return _normalise_platform_initial_state(
        {
            "surge_m": reference["surge_m"],
            "sway_m": 0.0,
            "heave_m": reference["heave_m"],
            "roll_deg": 0.0,
            "pitch_deg": reference["pitch_deg"],
            "yaw_deg": 0.0,
        }
    )


def _set_platform_initial_state(ed_text: str, state: dict[str, float]) -> str:
    values = _normalise_platform_initial_state(state)
    updated = ed_text
    for name, openfast_field in PLATFORM_STATE_FIELDS:
        updated = _replace_field(updated, openfast_field, f"{values[name]:.9g}")
    return updated


def _parse_six_numeric_values(line: str, *, context: str) -> tuple[np.ndarray, str, str]:
    newline = "\n" if line.endswith("\n") else ""
    tokens = line.rstrip("\n").split(maxsplit=6)
    if len(tokens) < 6:
        raise ValueError(f"{context} row does not contain six numeric entries")
    try:
        values = np.asarray(
            [float(token.replace("D", "E").replace("d", "e")) for token in tokens[:6]],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError(f"{context} row contains a nonnumeric matrix entry") from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{context} row contains a non-finite matrix entry")
    suffix = tokens[6] if len(tokens) == 7 else ""
    return values, suffix, newline


def _add_hydrodyn_linear_damping(
    hydrodyn_text: str, diagonal: np.ndarray
) -> str:
    """Add a six-DOF diagonal numerical damping term to HydroDyn ``AddBLin``."""
    diagonal = np.asarray(diagonal, dtype=float)
    if diagonal.shape != (6,) or not np.all(np.isfinite(diagonal)):
        raise ValueError("temporary HydroDyn damping must be a finite six-value diagonal")

    lines = hydrodyn_text.splitlines(keepends=True)
    try:
        start = next(index for index, line in enumerate(lines) if "AddBLin" in line)
    except StopIteration as error:
        raise ValueError("HydroDyn input is missing the AddBLin matrix") from error
    if start + 6 > len(lines):
        raise ValueError("HydroDyn AddBLin matrix is incomplete")

    for row_index in range(6):
        values, suffix, newline = _parse_six_numeric_values(
            lines[start + row_index], context="HydroDyn AddBLin"
        )
        values[row_index] += diagonal[row_index]
        rendered = " ".join(f"{value:.9g}" for value in values)
        if suffix:
            rendered = f"{rendered} {suffix}"
        lines[start + row_index] = f"{rendered}{newline}"
    return "".join(lines)


def _window_statistics(columns: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, dict[str, float]]:
    if not np.any(mask):
        raise ValueError("OpenFAST output has no samples in the requested tail window")
    statistics: dict[str, dict[str, float]] = {}
    for state_name, channel_name in PLATFORM_STATE_FIELDS:
        if channel_name not in columns:
            raise ValueError(f"OpenFAST output is missing platform channel {channel_name}")
        values = columns[channel_name][mask]
        statistics[state_name] = {
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }
    return statistics


def _late_window_statistics(
    columns: dict[str, np.ndarray], *, duration_s: float, tail_window_s: float
) -> dict[str, object]:
    if not 0.0 < tail_window_s <= duration_s:
        raise ValueError("tail_window_s must be positive and no greater than duration_s")
    time = columns["Time"]
    split_s = duration_s - 0.5 * tail_window_s
    first_mask = (time >= duration_s - tail_window_s) & (time < split_s)
    second_mask = time >= split_s
    first = _window_statistics(columns, first_mask)
    second = _window_statistics(columns, second_mask)
    late_mean_state = _normalise_platform_initial_state(
        {state_name: values["mean"] for state_name, values in second.items()}
    )
    return {
        "tail_window_s": tail_window_s,
        "first_window_s": [duration_s - tail_window_s, split_s],
        "second_window_s": [split_s, duration_s],
        "first_window_statistics": first,
        "second_window_statistics": second,
        "second_window_mean_state_for_release_only": late_mean_state,
        "second_minus_first_mean": {
            state_name: float(second[state_name]["mean"] - first[state_name]["mean"])
            for state_name, _ in PLATFORM_STATE_FIELDS
        },
    }


def _prescribed_rotor_tail_output(
    columns: dict[str, np.ndarray],
    *,
    duration_s: float,
    tail_window_s: float,
    prescribed_blade_pitch_deg: float = PRESCRIBED_BLADE_PITCH_DEG,
    prescribed_rotor_speed_rpm: float = PRESCRIBED_ROTOR_SPEED_RPM,
) -> dict[str, object]:
    mask = columns["Time"] >= duration_s - tail_window_s
    if not np.any(mask):
        raise ValueError("OpenFAST output has no samples in the rotor-state tail window")
    blade_channels = ("BldPitch1", "BldPitch2", "BldPitch3")
    required = (*blade_channels, "RotSpeed")
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(
            "OpenFAST output is missing prescribed-rotor channels: "
            f"{', '.join(missing)}"
        )
    for name in blade_channels:
        if not np.allclose(
            columns[name][mask], prescribed_blade_pitch_deg, atol=1.0e-6
        ):
            raise ValueError(
                "OpenFAST output does not retain the prescribed blade pitch: "
                f"{name} varies within the tail window"
            )
    if not np.allclose(
        columns["RotSpeed"][mask], prescribed_rotor_speed_rpm, atol=1.0e-6
    ):
        raise ValueError(
            "OpenFAST output does not retain the prescribed rotor speed "
            "within the tail window"
        )
    blade_pitch_deg = [float(np.mean(columns[name][mask])) for name in blade_channels]
    rotor_speed_rpm = float(np.mean(columns["RotSpeed"][mask]))
    _assert_prescribed_rotor_output(
        blade_pitch_deg=blade_pitch_deg,
        rotor_speed_rpm=rotor_speed_rpm,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    return {
        "tail_window_s": tail_window_s,
        "blade_pitch_deg": blade_pitch_deg,
        "rotor_speed_rpm": rotor_speed_rpm,
    }


def _run_phase(
    *,
    phase_root: Path,
    openfast_binary: Path,
    wind_speed_mps: float,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
    tail_window_s: float,
    temporary_add_blin_diagonal: np.ndarray | None,
    prescribed_blade_pitch_deg: float,
    prescribed_rotor_speed_rpm: float,
) -> dict[str, object]:
    semi_dir = _extract_clean_model(phase_root)
    fst_path = _configure_case(
        semi_dir,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=initial_state["surge_m"],
        equilibrium_heave_m=initial_state["heave_m"],
        equilibrium_pitch_deg=initial_state["pitch_deg"],
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    ed_path = semi_dir / ED_NAME
    ed_path.write_text(
        _set_platform_initial_state(ed_path.read_text(encoding="utf-8"), initial_state),
        encoding="utf-8",
    )

    damping_mode = "original_public_input"
    if temporary_add_blin_diagonal is not None:
        hydrodyn_path = semi_dir / HYDRODYN_NAME
        hydrodyn_path.write_text(
            _add_hydrodyn_linear_damping(
                hydrodyn_path.read_text(encoding="utf-8"),
                temporary_add_blin_diagonal,
            ),
            encoding="utf-8",
        )
        damping_mode = "temporary_numerical_relaxation_add_blin"

    completed = subprocess.run(
        [str(openfast_binary), fst_path.name],
        cwd=semi_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "OpenFAST relaxation/release phase failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    columns, _ = _read_openfast_output(semi_dir / f"{fst_path.stem}.out")
    _assert_output_reaches_time_horizon(
        columns, duration_s=duration_s, output_step_s=output_step_s
    )
    return {
        "input_tree": str(semi_dir),
        "initial_state": _normalise_platform_initial_state(initial_state),
        "hydrodyn_damping_mode": damping_mode,
        "temporary_add_blin_diagonal": (
            None
            if temporary_add_blin_diagonal is None
            else [float(value) for value in temporary_add_blin_diagonal]
        ),
        "sample_count": int(columns["Time"].size),
        "prescribed_rotor_tail_output": _prescribed_rotor_tail_output(
            columns,
            duration_s=duration_s,
            tail_window_s=tail_window_s,
            prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
            prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
        ),
        "late_window_statistics": _late_window_statistics(
            columns, duration_s=duration_s, tail_window_s=tail_window_s
        ),
    }


def run_audit(
    *,
    work_dir: Path,
    openfast_binary: Path,
    duration_s: float,
    output_step_s: float,
    wind_speed_mps: float,
    tail_window_s: float,
    nominal_reference_state_path: Path = REFERENCE_STATE_JSON,
    prescribed_blade_pitch_deg: float = PRESCRIBED_BLADE_PITCH_DEG,
    prescribed_rotor_speed_rpm: float = PRESCRIBED_ROTOR_SPEED_RPM,
) -> dict[str, object]:
    if duration_s <= 0.0 or output_step_s <= 0.0:
        raise ValueError("duration_s and output_step_s must be positive")
    if wind_speed_mps <= 0.0:
        raise ValueError("wind_speed_mps must be positive")
    if not np.isfinite(prescribed_blade_pitch_deg):
        raise ValueError("prescribed_blade_pitch_deg must be finite")
    if not np.isfinite(prescribed_rotor_speed_rpm) or prescribed_rotor_speed_rpm < 0.0:
        raise ValueError("prescribed_rotor_speed_rpm must be finite and non-negative")
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")

    nominal_reference = _load_candidate_reference_state(
        nominal_reference_state_path
    )
    relaxation = _run_phase(
        phase_root=work_dir / "relaxation",
        openfast_binary=openfast_binary,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=_initial_state_from_nominal_reference(nominal_reference),
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=RELAXATION_ADD_BLIN_DIAGONAL,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    release_initial_state = relaxation["late_window_statistics"][
        "second_window_mean_state_for_release_only"
    ]
    assert isinstance(release_initial_state, dict)
    release = _run_phase(
        phase_root=work_dir / "release",
        openfast_binary=openfast_binary,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=_normalise_platform_initial_state(release_initial_state),
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=None,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    return {
        "evidence_level": "OpenFAST prescribed-wind relaxation and release audit",
        "boundaries": {
            "is_rosco_controlled_operating_case": False,
            "is_physical_damping_calibration": False,
            "is_static_equilibrium_solver": False,
            "automatically_creates_reference_state_artifact": False,
            "purpose": "locate a candidate pose and inspect its original-damping release",
        },
        "reference": {
            "model_zip": str(MODEL_ZIP),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "openfast_binary_sha256": _sha256(openfast_binary),
            "nominal_reference_state_path": str(nominal_reference_state_path),
            "nominal_reference_state": nominal_reference,
            "wind_speed_mps": wind_speed_mps,
            "prescribed_rotor_speed_rpm": prescribed_rotor_speed_rpm,
            "prescribed_blade_pitch_deg": prescribed_blade_pitch_deg,
            "wave_and_current": "disabled",
        },
        "temporary_relaxation": relaxation,
        "original_damping_release": release,
        "state_transfer": {
            "release_initial_state_source": "temporary_relaxation.second_window_mean_state_for_release_only",
            "release_initial_state": release_initial_state,
            "release_uses_fresh_public_input_copy": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--openfast",
        type=Path,
        default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast"),
    )
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--output-step-s", type=float, default=1.0)
    parser.add_argument("--wind-speed-mps", type=float, default=10.74)
    parser.add_argument(
        "--prescribed-blade-pitch-deg",
        type=float,
        default=PRESCRIBED_BLADE_PITCH_DEG,
    )
    parser.add_argument(
        "--prescribed-rotor-speed-rpm",
        type=float,
        default=PRESCRIBED_ROTOR_SPEED_RPM,
    )
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
        duration_s=args.duration_s,
        output_step_s=args.output_step_s,
        wind_speed_mps=args.wind_speed_mps,
        tail_window_s=args.tail_window_s,
        nominal_reference_state_path=args.nominal_reference_state_json,
        prescribed_blade_pitch_deg=args.prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=args.prescribed_rotor_speed_rpm,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
