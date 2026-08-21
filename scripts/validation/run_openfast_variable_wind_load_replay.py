#!/usr/bin/env python3
"""Compare a low-order response with an OpenFAST aerodynamic-load replay.

The comparison is deliberately conditional.  OpenFAST is run twice from the
same prescribed-wind candidate state: one constant 5 m/s reference and one
smooth 4.5--5.5 m/s uniform-wind history.  The two output histories are
subtracted sample by sample.  The low-order model then receives the resulting
six-component aerodynamic-load difference in its frozen reference axes.

This does not validate a controller, a physical damping model, or a fully
coupled OpenFAST replacement.  It only checks whether the current small-angle
model follows the low-frequency incremental pitch response under the same
recorded aerodynamic load history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fowt_platform import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    PlatformMatrices,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    generalized_load_from_openfast_hub_wrench,
)
from run_openfast_prescribed_wind_node import (
    ED_NAME,
    FST_NAME,
    INFLOW_NAME,
    MODEL_ZIP,
    _assert_output_reaches_time_horizon,
    _assert_prescribed_rotor_output,
    _configure_case,
    _extract_clean_model,
    _hub_wrench_unit_scales,
    _read_openfast_hub_geometry,
    _read_openfast_output,
    _replace_field,
    _sha256,
)
from run_openfast_prescribed_wind_relaxation_release_audit import (
    _normalise_platform_initial_state,
    _set_platform_initial_state,
    run_audit as run_relaxation_release_audit,
)


REFERENCE_MANIFEST = ROOT / "configs/reference_platforms/volturnus_s_openfast_v1_1_16.json"
DEFAULT_OPENFAST = Path("/tmp/fowt-openfast-4.1.2/bin/openfast")
BASE_WIND_SPEED_MPS = 5.0
WIND_AMPLITUDE_MPS = 0.5
INITIAL_HOLD_S = 60.0
RAMP_S = 120.0
WIND_PERIOD_S = 300.0
PRESCRIBED_BLADE_PITCH_DEG = 1.0
PRESCRIBED_ROTOR_SPEED_RPM = 5.0


def controlled_wind_speed_mps(
    time_s: Any,
    *,
    base_wind_speed_mps: float = BASE_WIND_SPEED_MPS,
    amplitude_mps: float = WIND_AMPLITUDE_MPS,
    initial_hold_s: float = INITIAL_HOLD_S,
    ramp_s: float = RAMP_S,
    period_s: float = WIND_PERIOD_S,
) -> np.ndarray:
    """Return a smooth, bounded uniform-wind history without filtering data."""

    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or time.size == 0 or not np.all(np.isfinite(time)):
        raise ValueError("time_s must be a non-empty finite one-dimensional series")
    if np.any(time < 0.0):
        raise ValueError("time_s must be non-negative")
    if base_wind_speed_mps <= 0.0 or amplitude_mps <= 0.0:
        raise ValueError("base_wind_speed_mps and amplitude_mps must be positive")
    if initial_hold_s < 0.0 or ramp_s <= 0.0 or period_s <= 0.0:
        raise ValueError("initial_hold_s must be non-negative and ramp_s/period_s positive")

    elapsed = np.maximum(time - initial_hold_s, 0.0)
    envelope = np.sin(0.5 * np.pi * np.minimum(elapsed / ramp_s, 1.0)) ** 2
    variation = amplitude_mps * envelope * np.sin(2.0 * np.pi * elapsed / period_s)
    return base_wind_speed_mps + variation


def _write_uniform_wind_file(path: Path, time_s: np.ndarray, speed_mps: np.ndarray) -> None:
    if time_s.shape != speed_mps.shape or time_s.ndim != 1:
        raise ValueError("uniform-wind time and speed arrays must be aligned vectors")
    if np.any(np.diff(time_s) <= 0.0) or np.any(speed_mps <= 0.0):
        raise ValueError("uniform-wind time must increase and speed must remain positive")
    lines = [
        "! Uniform wind history generated for a conditional OpenFAST load replay.\n",
        "!Time WindSpeed WindDir VertSpeed HorzShear VertShear LinVShear GustSpeed\n",
    ]
    lines.extend(
        f"{time_value:.9g} {speed_value:.9g} 0.0 0.0 0.0 0.0 0.0 0.0\n"
        for time_value, speed_value in zip(time_s, speed_mps, strict=True)
    )
    path.write_text("".join(lines), encoding="utf-8")


def _configure_variable_uniform_wind(
    semi_dir: Path,
    *,
    time_s: np.ndarray,
    speed_mps: np.ndarray,
) -> Path:
    inflow_path = semi_dir.parent / "IEA-15-240-RWT" / INFLOW_NAME
    wind_path = inflow_path.parent / "Wind" / "controlled_5m_sine_replay.wnd"
    _write_uniform_wind_file(wind_path, time_s, speed_mps)
    inflow = inflow_path.read_text(encoding="utf-8")
    inflow = _replace_field(inflow, "WindType", "2")
    inflow = _replace_field(inflow, "Filename_Uni", '"Wind/controlled_5m_sine_replay.wnd"')
    inflow = _replace_field(inflow, "VelInterpCubic", "False")
    inflow_path.write_text(inflow, encoding="utf-8")
    return wind_path


def _required_output_channels(columns: dict[str, np.ndarray]) -> None:
    required = (
        "Time",
        "PtfmRoll",
        "PtfmPitch",
        "PtfmYaw",
        "Azimuth",
        "NacYaw",
        "BldPitch1",
        "BldPitch2",
        "BldPitch3",
        "RotSpeed",
        "RtFldFxh",
        "RtFldFyh",
        "RtFldFzh",
        "RtFldMxh",
        "RtFldMyh",
        "RtFldMzh",
    )
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(f"OpenFAST output is missing channels: {', '.join(missing)}")


def _run_openfast_case(
    *,
    case_root: Path,
    openfast_binary: Path,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
    variable_wind: bool,
) -> dict[str, Any]:
    semi_dir = _extract_clean_model(case_root)
    fst_path = _configure_case(
        semi_dir,
        wind_speed_mps=BASE_WIND_SPEED_MPS,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=initial_state["surge_m"],
        equilibrium_heave_m=initial_state["heave_m"],
        equilibrium_pitch_deg=initial_state["pitch_deg"],
        prescribed_blade_pitch_deg=PRESCRIBED_BLADE_PITCH_DEG,
        prescribed_rotor_speed_rpm=PRESCRIBED_ROTOR_SPEED_RPM,
    )
    ed_path = semi_dir / ED_NAME
    ed_path.write_text(
        _set_platform_initial_state(ed_path.read_text(encoding="utf-8"), initial_state),
        encoding="utf-8",
    )
    wind_file = None
    if variable_wind:
        time_s = np.arange(0.0, duration_s + 0.5 * output_step_s, output_step_s)
        speed_mps = controlled_wind_speed_mps(time_s)
        wind_file = _configure_variable_uniform_wind(
            semi_dir, time_s=time_s, speed_mps=speed_mps
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
            "OpenFAST variable-wind replay case failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    output_path = semi_dir / f"{fst_path.stem}.out"
    columns, units = _read_openfast_output(output_path)
    _assert_output_reaches_time_horizon(
        columns, duration_s=duration_s, output_step_s=output_step_s
    )
    _required_output_channels(columns)
    _assert_prescribed_rotor_output(
        blade_pitch_deg=[
            columns["BldPitch1"], columns["BldPitch2"], columns["BldPitch3"]
        ],
        rotor_speed_rpm=columns["RotSpeed"],
        prescribed_blade_pitch_deg=PRESCRIBED_BLADE_PITCH_DEG,
        prescribed_rotor_speed_rpm=PRESCRIBED_ROTOR_SPEED_RPM,
    )
    return {
        "input_tree": str(semi_dir),
        "output_path": str(output_path),
        "wind_file": None if wind_file is None else str(wind_file),
        "columns": columns,
        "units": units,
        "hub_geometry": _read_openfast_hub_geometry(ed_path),
    }


def _generalized_hub_load_series(
    *,
    columns: dict[str, np.ndarray],
    units: dict[str, str],
    hub_geometry: dict[str, object],
    equilibrium_pitch_deg: float,
) -> np.ndarray:
    force_scale, moment_scale = _hub_wrench_unit_scales(units)
    hub_from_reference = np.asarray(
        hub_geometry["hub_from_platform_reference_m"], dtype=float
    )
    loads = np.empty((columns["Time"].size, 6), dtype=float)
    for index in range(loads.shape[0]):
        loads[index] = generalized_load_from_openfast_hub_wrench(
            force_hub_n=force_scale
            * np.array(
                [
                    columns["RtFldFxh"][index],
                    columns["RtFldFyh"][index],
                    columns["RtFldFzh"][index],
                ]
            ),
            moment_hub_nm=moment_scale
            * np.array(
                [
                    columns["RtFldMxh"][index],
                    columns["RtFldMyh"][index],
                    columns["RtFldMzh"][index],
                ]
            ),
            hub_from_platform_reference_m=hub_from_reference,
            shaft_tilt_deg=float(hub_geometry["shaft_tilt_deg"]),
            azimuth_deg=columns["Azimuth"][index],
            platform_roll_deg=columns["PtfmRoll"][index],
            platform_pitch_deg=columns["PtfmPitch"][index],
            platform_yaw_deg=columns["PtfmYaw"][index],
            nacelle_yaw_deg=columns["NacYaw"][index],
            azimuth_blade1_up_deg=float(hub_geometry["azimuth_blade1_up_deg"]),
            equilibrium_platform_pitch_deg=equilibrium_pitch_deg,
        )
    return loads


def _assert_common_time_axis(base_time_s: np.ndarray, variable_time_s: np.ndarray) -> None:
    if base_time_s.shape != variable_time_s.shape or not np.allclose(
        base_time_s, variable_time_s, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError("constant and variable OpenFAST runs do not share one time axis")


def _advance_replay(
    time_s: np.ndarray,
    incremental_wind_load: np.ndarray,
    *,
    matrices: PlatformMatrices,
) -> np.ndarray:
    if incremental_wind_load.shape != (time_s.size, 6):
        raise ValueError("incremental wind load must have one six-DOF row per time")
    if not isinstance(matrices, PlatformMatrices):
        raise TypeError("matrices must be PlatformMatrices")
    model = IncrementalPlatformModel(matrices)
    state = IncrementalState.zeros()
    response = np.zeros((time_s.size, 6), dtype=float)
    for index, duration_s in enumerate(np.diff(time_s)):
        state = model.advance_frozen_step(
            state,
            IncrementalLoads(
                wind=incremental_wind_load[index],
                wave=np.zeros(6),
                ballast=np.zeros(6),
                other=np.zeros(6),
            ),
            float(duration_s),
        )
        response[index + 1] = state.position
    return response


def _p2p(values: np.ndarray) -> float:
    return float(np.max(values) - np.min(values))


def _fundamental_phase_and_amplitude(
    time_s: np.ndarray,
    values: np.ndarray,
    *,
    period_s: float,
) -> tuple[float, float]:
    """Project a complete-period response onto its fundamental sine and cosine terms."""

    if time_s.ndim != 1 or values.ndim != 1 or time_s.shape != values.shape:
        raise ValueError("time_s and values must be aligned one-dimensional vectors")
    if time_s.size < 3 or period_s <= 0.0:
        raise ValueError("at least three samples and a positive period are required")
    phase_argument = 2.0 * np.pi * time_s / period_s
    design = np.column_stack((np.sin(phase_argument), np.cos(phase_argument)))
    sine_coefficient, cosine_coefficient = np.linalg.lstsq(
        design, values - np.mean(values), rcond=None
    )[0]
    phase_deg = float(np.rad2deg(np.arctan2(cosine_coefficient, sine_coefficient)))
    amplitude = float(np.hypot(sine_coefficient, cosine_coefficient))
    return phase_deg, amplitude


def _wrapped_phase_difference_deg(low_order_deg: float, openfast_deg: float) -> float:
    return float((low_order_deg - openfast_deg + 180.0) % 360.0 - 180.0)


def _response_metrics(
    openfast_deg: np.ndarray,
    low_order_deg: np.ndarray,
    baseline_deg: np.ndarray,
    analysis_mask: np.ndarray,
    *,
    time_s: np.ndarray | None = None,
    fundamental_period_s: float | None = None,
) -> dict[str, Any]:
    observed = np.asarray(openfast_deg[analysis_mask], dtype=float)
    modeled = np.asarray(low_order_deg[analysis_mask], dtype=float)
    baseline = np.asarray(baseline_deg[analysis_mask], dtype=float)
    if observed.size < 3:
        raise ValueError("analysis interval must contain at least three samples")
    observed_p2p = _p2p(observed)
    baseline_p2p = _p2p(baseline)
    rmse = float(np.sqrt(np.mean(np.square(modeled - observed))))
    correlation = (
        None
        if np.std(observed) <= 1.0e-12 or np.std(modeled) <= 1.0e-12
        else float(np.corrcoef(observed, modeled)[0, 1])
    )
    discernible = bool(observed_p2p > max(5.0 * baseline_p2p, 0.002))
    phase_metrics: dict[str, float | None] = {
        "fundamental_period_s": None,
        "openfast_fundamental_amplitude_deg": None,
        "low_order_fundamental_amplitude_deg": None,
        "fundamental_phase_difference_low_order_minus_openfast_deg": None,
    }
    if (time_s is None) != (fundamental_period_s is None):
        raise ValueError("time_s and fundamental_period_s must be provided together")
    if time_s is not None and fundamental_period_s is not None and discernible:
        selected_time = np.asarray(time_s[analysis_mask], dtype=float)
        openfast_phase_deg, openfast_amplitude_deg = _fundamental_phase_and_amplitude(
            selected_time, observed, period_s=fundamental_period_s
        )
        low_order_phase_deg, low_order_amplitude_deg = _fundamental_phase_and_amplitude(
            selected_time, modeled, period_s=fundamental_period_s
        )
        phase_metrics = {
            "fundamental_period_s": float(fundamental_period_s),
            "openfast_fundamental_amplitude_deg": openfast_amplitude_deg,
            "low_order_fundamental_amplitude_deg": low_order_amplitude_deg,
            "fundamental_phase_difference_low_order_minus_openfast_deg": (
                _wrapped_phase_difference_deg(low_order_phase_deg, openfast_phase_deg)
            ),
        }
    return {
        "openfast_delta_mean_deg": float(np.mean(observed)),
        "low_order_mean_deg": float(np.mean(modeled)),
        "openfast_delta_peak_to_peak_deg": observed_p2p,
        "low_order_peak_to_peak_deg": _p2p(modeled),
        "baseline_peak_to_peak_deg": baseline_p2p,
        "amplitude_ratio_low_order_over_openfast": (
            None if observed_p2p <= 1.0e-12 else _p2p(modeled) / observed_p2p
        ),
        "rmse_deg": rmse,
        "normalized_rmse_over_openfast_peak_to_peak": (
            None if observed_p2p <= 1.0e-12 else rmse / observed_p2p
        ),
        "zero_lag_correlation": correlation,
        "discernible_against_constant_wind_tail": discernible,
        **phase_metrics,
    }


def _load_summary(loads: np.ndarray) -> dict[str, Any]:
    names = ("surge_n", "sway_n", "heave_n", "roll_nm", "pitch_nm", "yaw_nm")
    return {
        name: {
            "mean": float(np.mean(loads[:, index])),
            "peak_to_peak": _p2p(loads[:, index]),
            "maximum_absolute": float(np.max(np.abs(loads[:, index]))),
        }
        for index, name in enumerate(names)
    }


def _full_period_analysis_interval_s(
    *,
    duration_s: float,
    settled_after_s: float = INITIAL_HOLD_S + RAMP_S,
    period_s: float = WIND_PERIOD_S,
) -> tuple[float, float, int]:
    """Select the last whole wind periods after the smooth ramp is complete."""

    available_s = duration_s - settled_after_s
    period_count = int(math.floor(available_s / period_s))
    if period_count < 1:
        raise ValueError("duration_s leaves no complete wind period after the ramp")
    start_s = duration_s - period_count * period_s
    if start_s < settled_after_s - 1.0e-12:
        raise AssertionError("analysis interval starts before the wind ramp has settled")
    return float(start_s), float(duration_s), period_count


def run_audit(
    *,
    work_dir: Path,
    openfast_binary: Path = DEFAULT_OPENFAST,
    duration_s: float = 900.0,
    output_step_s: float = 0.5,
) -> dict[str, Any]:
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")
    if duration_s <= INITIAL_HOLD_S + RAMP_S + WIND_PERIOD_S:
        raise ValueError("duration_s must include a full settled wind period after the ramp")
    if output_step_s <= 0.0:
        raise ValueError("output_step_s must be positive")

    relaxation = run_relaxation_release_audit(
        work_dir=work_dir / "candidate_state",
        openfast_binary=openfast_binary,
        duration_s=600.0,
        output_step_s=output_step_s,
        wind_speed_mps=BASE_WIND_SPEED_MPS,
        tail_window_s=150.0,
        prescribed_blade_pitch_deg=PRESCRIBED_BLADE_PITCH_DEG,
        prescribed_rotor_speed_rpm=PRESCRIBED_ROTOR_SPEED_RPM,
    )
    candidate = _normalise_platform_initial_state(
        relaxation["state_transfer"]["release_initial_state"]
    )
    constant = _run_openfast_case(
        case_root=work_dir / "constant_wind",
        openfast_binary=openfast_binary,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=candidate,
        variable_wind=False,
    )
    variable = _run_openfast_case(
        case_root=work_dir / "variable_wind",
        openfast_binary=openfast_binary,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=candidate,
        variable_wind=True,
    )
    constant_columns = constant["columns"]
    variable_columns = variable["columns"]
    assert isinstance(constant_columns, dict) and isinstance(variable_columns, dict)
    time_s = constant_columns["Time"]
    _assert_common_time_axis(time_s, variable_columns["Time"])
    constant_load = _generalized_hub_load_series(
        columns=constant_columns,
        units=constant["units"],
        hub_geometry=constant["hub_geometry"],
        equilibrium_pitch_deg=candidate["pitch_deg"],
    )
    variable_load = _generalized_hub_load_series(
        columns=variable_columns,
        units=variable["units"],
        hub_geometry=variable["hub_geometry"],
        equilibrium_pitch_deg=candidate["pitch_deg"],
    )
    load_delta = variable_load - constant_load
    runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6), dtype=float),
    )
    low_order_rad = _advance_replay(
        time_s,
        load_delta,
        matrices=runtime_assembly.base_matrices,
    )
    analysis_start_s, analysis_end_s, analysis_period_count = _full_period_analysis_interval_s(
        duration_s=duration_s
    )
    analysis_mask = (time_s >= analysis_start_s) & (time_s <= analysis_end_s)
    pitch_openfast_delta = variable_columns["PtfmPitch"] - constant_columns["PtfmPitch"]
    roll_openfast_delta = variable_columns["PtfmRoll"] - constant_columns["PtfmRoll"]
    pitch_low_order_deg = np.rad2deg(low_order_rad[:, 4])
    roll_low_order_deg = np.rad2deg(low_order_rad[:, 3])
    return {
        "evidence_level": "conditional OpenFAST aerodynamic-load replay",
        "boundaries": {
            "is_controller_validation": False,
            "is_full_openfast_replacement": False,
            "is_physical_damping_calibration": False,
            "is_wave_or_current_validation": False,
            "is_two_way_aerodynamic_coupling_validation": False,
            "low_order_damping": "explicitly zero; no damping was fitted to this history",
            "input_definition": (
                "time-matched variable-wind minus constant-wind aerodynamic hub wrench, "
                "mapped as a full six-DOF generalized load about the platform reference"
            ),
        },
        "low_order_runtime": {
            "provenance": runtime_assembly.provenance,
            "damping": "zero matrix",
        },
        "reference": {
            "model_zip": str(MODEL_ZIP.relative_to(ROOT)),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "openfast_binary_sha256": _sha256(openfast_binary),
            "candidate_state_from": "5 m/s relaxation followed by original-damping release",
            "candidate_state": candidate,
            "prescribed_blade_pitch_deg": PRESCRIBED_BLADE_PITCH_DEG,
            "prescribed_rotor_speed_rpm": PRESCRIBED_ROTOR_SPEED_RPM,
            "wave_and_current": "disabled",
        },
        "wind_history": {
            "base_wind_speed_mps": BASE_WIND_SPEED_MPS,
            "amplitude_mps": WIND_AMPLITUDE_MPS,
            "range_mps": [BASE_WIND_SPEED_MPS - WIND_AMPLITUDE_MPS, BASE_WIND_SPEED_MPS + WIND_AMPLITUDE_MPS],
            "initial_hold_s": INITIAL_HOLD_S,
            "smooth_ramp_s": RAMP_S,
            "period_s": WIND_PERIOD_S,
            "output_step_s": output_step_s,
            "duration_s": duration_s,
            "interpolation": "OpenFAST uniform-wind linear interpolation; low-order replay uses unfiltered zero-order hold at the recorded output samples",
        },
        "analysis_interval_s": [analysis_start_s, analysis_end_s],
        "analysis_wind_period_count": analysis_period_count,
        "absolute_load_summary": {
            "constant_wind": _load_summary(constant_load),
            "variable_wind": _load_summary(variable_load),
        },
        "load_delta_summary": _load_summary(load_delta),
        "response_metrics": {
            "pitch": _response_metrics(
                pitch_openfast_delta,
                pitch_low_order_deg,
                constant_columns["PtfmPitch"],
                analysis_mask,
                time_s=time_s,
                fundamental_period_s=WIND_PERIOD_S,
            ),
            "roll": _response_metrics(
                roll_openfast_delta,
                roll_low_order_deg,
                constant_columns["PtfmRoll"],
                analysis_mask,
                time_s=time_s,
                fundamental_period_s=WIND_PERIOD_S,
            ),
        },
        "artifacts": {
            "constant_openfast_output": constant["output_path"],
            "variable_openfast_output": variable["output_path"],
            "variable_uniform_wind_file": variable["wind_file"],
        },
        "relaxation_release_summary": relaxation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--openfast", type=Path, default=DEFAULT_OPENFAST)
    parser.add_argument("--duration-s", type=float, default=900.0)
    parser.add_argument("--output-step-s", type=float, default=0.5)
    args = parser.parse_args()
    result = run_audit(
        work_dir=args.work_dir,
        openfast_binary=args.openfast,
        duration_s=args.duration_s,
        output_step_s=args.output_step_s,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
