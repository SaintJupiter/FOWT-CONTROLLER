#!/usr/bin/env python3
"""Probe OpenFAST local platform linearization without building a new plant.

The probe starts from the frozen rigid-platform, still-water reference pose
used by the existing narrow OpenFAST audits.  It keeps HydroDyn and MoorDyn,
turns off aerodynamic and structural-flexibility paths, and asks OpenFAST for
one restricted-configuration full-state linearization at a specified elapsed
time.  It deliberately
does not read a matrix into the controller or fit mass, damping, or mooring
coefficients.  Its only job is to determine whether a reproducible local
pitch/roll linearization artifact is available for later review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from run_openfast_moment_angle_consistency import _disable_aerodynamic_modules
from run_openfast_prescribed_wind_node import (
    ED_NAME,
    MODEL_ZIP,
    REFERENCE_STATE_JSON,
    SEA_NAME,
    _assert_output_reaches_time_horizon,
    _configure_case,
    _extract_clean_model,
    _load_candidate_reference_state,
    _read_openfast_output,
    _replace_field,
    _sha256,
)
from run_openfast_prescribed_wind_relaxation_release_audit import (
    _initial_state_from_nominal_reference,
    _late_window_statistics,
    _set_platform_initial_state,
)


HYDRODYN_NAME = "IEA-15-240-RWT-UMaineSemi_HydroDyn.dat"


def _configure_linearization_settings(
    fst_text: str,
    *,
    linearization_time_s: float,
) -> str:
    """Set a single absolute-time OpenFAST linearization request."""

    if not np.isfinite(linearization_time_s) or linearization_time_s <= 0.0:
        raise ValueError("linearization_time_s must be finite and positive")
    updated = fst_text
    for field, value in (
        ("Linearize", "True"),
        ("CalcSteady", "False"),
        ("NLinTimes", "1"),
        ("LinTimes", f"{linearization_time_s:.9g}"),
        ("LinInputs", "0"),
        ("LinOutputs", "1"),
        ("LinOutJac", "False"),
        ("LinOutMod", "False"),
    ):
        updated = _replace_linearization_field(updated, field=field, value=value)
    return updated


def _replace_linearization_field(text: str, *, field: str, value: str) -> str:
    """Replace one FST linearization row, including list-valued ``LinTimes``."""

    lines: list[str] = []
    replaced = False
    for line in text.splitlines(keepends=True):
        tokens = line.split()
        if field not in tokens:
            lines.append(line)
            continue
        if replaced:
            raise ValueError(f"OpenFAST input has multiple {field} rows")
        field_index = line.index(field)
        suffix = line[field_index + len(field) :]
        lines.append(f"{value:<22} {field}{suffix}")
        replaced = True
    if not replaced:
        raise ValueError(f"missing OpenFAST field {field}")
    return "".join(lines)


def _configure_stationary_rigid_elastodyn(ed_text: str) -> str:
    """Remove rotating and flexible states that are outside this local probe."""

    updated = ed_text
    for field in (
        "FlapDOF1",
        "FlapDOF2",
        "EdgeDOF",
        "DrTrDOF",
        "GenDOF",
        "YawDOF",
        "TwFADOF1",
        "TwFADOF2",
        "TwSSDOF1",
        "TwSSDOF2",
    ):
        updated = _replace_field(updated, field, "False")
    return _replace_field(updated, "RotSpeed", "0.0")


def _disable_second_order_wave_terms_for_linearization(sea_text: str) -> str:
    """Disable SeaState QTF paths unsupported by the OpenFAST linearizer."""

    updated = _replace_field(sea_text, "WvDiffQTF", "False")
    return _replace_field(updated, "WvSumQTF", "False")


def _configure_hydrodyn_for_local_linearization(hydrodyn_text: str) -> str:
    """Select HydroDyn paths that OpenFAST 4.1.2 permits in this first probe.

    The initial probe keeps the hydrostatic and mooring portions of the public
    configuration but disables wave excitation, convolution radiation, and the
    second-order difference-frequency forcing path.  It is consequently an
    availability check for local platform states, not a radiation/damping
    identification run.
    """

    updated = _replace_field(hydrodyn_text, "ExctnMod", "0")
    updated = _replace_field(updated, "RdtnMod", "0")
    return _replace_field(updated, "DiffQTF", "0")


def _platform_linearization_mentions(lin_path: Path) -> list[str]:
    """Return source lines that make platform pitch/roll labels inspectable."""

    mentions: list[str] = []
    for line in lin_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "PtfmPitch" in line or "PtfmRoll" in line:
            mentions.append(line.strip())
    return mentions


def _linearization_summary(lin_path: Path) -> dict[str, object]:
    """Extract only the high-level dimensions and platform labels for review."""

    text = lin_path.read_text(encoding="utf-8", errors="replace")
    counts: dict[str, int] = {}
    for label, key in (
        ("Number of continuous states", "continuous_states"),
        ("Number of discrete states", "discrete_states"),
        ("Number of inputs", "inputs"),
        ("Number of outputs", "outputs"),
    ):
        match = re.search(rf"{re.escape(label)}:\s*(\d+)", text)
        if match is None:
            raise ValueError(f"linearization output is missing {label}")
        counts[key] = int(match.group(1))
    platform_state_lines = [
        line.strip()
        for line in text.splitlines()
        if "ED Platform " in line and "DOF" in line
    ]
    return {
        **counts,
        "platform_state_lines": platform_state_lines,
        "platform_pitch_roll_output_lines": _platform_linearization_mentions(lin_path),
    }


def _configure_probe_case(
    semi_dir: Path,
    *,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
) -> Path:
    """Build one no-aero, rigid-platform OpenFAST input copy."""

    fst_path = _configure_case(
        semi_dir,
        wind_speed_mps=0.1,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=initial_state["surge_m"],
        equilibrium_heave_m=initial_state["heave_m"],
        equilibrium_pitch_deg=initial_state["pitch_deg"],
    )
    fst = _disable_aerodynamic_modules(fst_path.read_text(encoding="utf-8"))
    fst_path.write_text(
        _configure_linearization_settings(
            fst,
            linearization_time_s=duration_s,
        ),
        encoding="utf-8",
    )
    ed_path = semi_dir / ED_NAME
    ed = _set_platform_initial_state(ed_path.read_text(encoding="utf-8"), initial_state)
    ed_path.write_text(
        _configure_stationary_rigid_elastodyn(ed),
        encoding="utf-8",
    )
    sea_path = semi_dir / SEA_NAME
    sea_path.write_text(
        _disable_second_order_wave_terms_for_linearization(
            sea_path.read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    hydrodyn_path = semi_dir / HYDRODYN_NAME
    hydrodyn_path.write_text(
        _configure_hydrodyn_for_local_linearization(
            hydrodyn_path.read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    return fst_path


def run_probe(
    *,
    work_dir: Path,
    output_json: Path,
    openfast_binary: Path,
    duration_s: float = 60.0,
    output_step_s: float = 1.0,
    tail_window_s: float = 20.0,
) -> dict[str, object]:
    """Run a one-point linearization probe and retain reviewable evidence."""

    if min(duration_s, output_step_s, tail_window_s) <= 0.0:
        raise ValueError("duration_s, output_step_s, and tail_window_s must be positive")
    if tail_window_s > duration_s:
        raise ValueError("tail_window_s must not exceed duration_s")
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")

    nominal = _load_candidate_reference_state(REFERENCE_STATE_JSON)
    initial_state = _initial_state_from_nominal_reference(nominal)
    semi_dir = _extract_clean_model(work_dir)
    fst_path = _configure_probe_case(
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
            "OpenFAST platform linearization probe failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    columns, _ = _read_openfast_output(semi_dir / f"{fst_path.stem}.out")
    _assert_output_reaches_time_horizon(
        columns,
        duration_s=duration_s,
        output_step_s=output_step_s,
    )
    lin_files = sorted(semi_dir.glob(f"{fst_path.stem}.*.lin"))
    if not lin_files:
        lin_files = sorted(semi_dir.glob("*.lin"))
    if not lin_files:
        raise RuntimeError("OpenFAST completed without producing a .lin file")

    result: dict[str, object] = {
        "evidence_level": "OpenFAST local-linearization availability probe",
        "scope": {
            "purpose": "check whether a frozen rigid-platform reference node produces an inspectable local pitch/roll linearization artifact",
            "environment": "AeroDyn, InflowWind, ServoDyn, waves, and current disabled; HydroDyn and MoorDyn retained",
            "hydrodyn_linearization_mode": "hydrostatic-plus-mooring local probe with wave excitation and radiation-memory paths disabled",
            "not_a_dynamic_coefficient_fit": True,
            "not_a_source_consistent_runtime_assembly": True,
            "not_a_controller_validation": True,
        },
        "source": {
            "model_zip": str(MODEL_ZIP),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "openfast_binary_sha256": _sha256(openfast_binary),
            "initial_state_artifact": str(REFERENCE_STATE_JSON),
            "initial_state": initial_state,
        },
        "linearization_request": {
            "linearization_time_s": duration_s,
            "linearize": True,
            "calculate_steady_state": False,
            "linearization_times": 1,
            "input_selection": "none",
            "output_selection": "standard OpenFAST output channels",
            "stationary_rigid_turbine": True,
        },
        "terminal_platform_state_statistics": _late_window_statistics(
            columns,
            duration_s=duration_s,
            tail_window_s=tail_window_s,
        ),
        "linearization_files": [str(path) for path in lin_files],
        "linearization_summaries": {
            str(path): _linearization_summary(path) for path in lin_files
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
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--output-step-s", type=float, default=1.0)
    parser.add_argument("--tail-window-s", type=float, default=20.0)
    args = parser.parse_args()
    result = run_probe(
        work_dir=args.work_dir,
        output_json=args.output_json,
        openfast_binary=args.openfast,
        duration_s=args.duration_s,
        output_step_s=args.output_step_s,
        tail_window_s=args.tail_window_s,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
