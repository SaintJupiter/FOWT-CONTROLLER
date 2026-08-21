#!/usr/bin/env python3
"""Check the source-consistent static pitch scale at three OpenFAST endpoints.

This is intentionally a narrow no-MoorDyn check.  It uses the same frozen
ElastoDyn/WAMIT static assembly as the low-order prediction and compares its
2, 5 and 10 degree pitch increments against OpenFAST runs with the mooring
module disabled.  It neither identifies a mooring model nor validates the
dynamic six-degree-of-freedom platform response.
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

from fowt_platform import load_volturnus_source_consistent_static_reference
from run_openfast_moment_angle_consistency import (
    REFERENCE_STATE_JSON,
    RELAXATION_ADD_BLIN_DIAGONAL,
    _load_candidate_reference_state,
    _platform_reference_state,
    _release_tail_stability,
    _tail_mean_state,
)
from run_openfast_pitch_restoring_component_audit import _run_phase, _tail_mean_pitch


REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
DEFAULT_TARGET_PITCH_DEG = (2.0, 5.0, 10.0)


def _targets(values: Iterable[Any]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not np.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("target pitch angles must be finite and positive")
    if any(later <= earlier for earlier, later in zip(result, result[1:])):
        raise ValueError("target pitch angles must be strictly increasing")
    return result


def run_sweep(
    *,
    work_dir: Path,
    openfast_binary: Path,
    target_pitch_deg: Iterable[Any] = DEFAULT_TARGET_PITCH_DEG,
    relaxation_duration_s: float = 300.0,
    release_duration_s: float = 600.0,
    output_step_s: float = 1.0,
    tail_window_s: float = 150.0,
) -> dict[str, object]:
    """Run the deliberately limited source-consistent no-MoorDyn static check."""

    targets = _targets(target_pitch_deg)
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")
    if min(relaxation_duration_s, release_duration_s, output_step_s, tail_window_s) <= 0.0:
        raise ValueError("durations and output step must be positive")
    if tail_window_s > release_duration_s:
        raise ValueError("tail_window_s must not exceed release_duration_s")

    reference = load_volturnus_source_consistent_static_reference(REFERENCE_MANIFEST)
    initial_state = _platform_reference_state(
        _load_candidate_reference_state(REFERENCE_STATE_JSON)
    )
    zero = np.zeros(6)
    baseline_relaxation = _run_phase(
        phase_root=work_dir / "zero_preload" / "relaxation",
        openfast_binary=openfast_binary,
        duration_s=relaxation_duration_s,
        output_step_s=output_step_s,
        initial_state=initial_state,
        preload=zero,
        tail_window_s=tail_window_s,
        moordyn_enabled=False,
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
        moordyn_enabled=False,
        temporary_add_blin_diagonal=None,
    )
    baseline_pitch_deg = _tail_mean_pitch(baseline_release)

    cases: list[dict[str, object]] = []
    stiffness = reference.pitch_restoring_stiffness_without_mooring_nm_per_rad
    for target in targets:
        preload = np.zeros(6)
        preload[4] = stiffness * np.deg2rad(target)
        relaxation = _run_phase(
            phase_root=work_dir / f"pitch_{target:g}deg" / "relaxation",
            openfast_binary=openfast_binary,
            duration_s=relaxation_duration_s,
            output_step_s=output_step_s,
            initial_state=initial_state,
            preload=preload,
            tail_window_s=tail_window_s,
            moordyn_enabled=False,
            temporary_add_blin_diagonal=RELAXATION_ADD_BLIN_DIAGONAL,
        )
        release = _run_phase(
            phase_root=work_dir / f"pitch_{target:g}deg" / "release",
            openfast_binary=openfast_binary,
            duration_s=release_duration_s,
            output_step_s=output_step_s,
            initial_state=_tail_mean_state(relaxation),
            preload=preload,
            tail_window_s=tail_window_s,
            moordyn_enabled=False,
            temporary_add_blin_diagonal=None,
        )
        openfast_increment_deg = _tail_mean_pitch(release) - baseline_pitch_deg
        stability = _release_tail_stability(release)
        cases.append(
            {
                "target_source_consistent_pitch_increment_deg": target,
                "applied_openfast_add_f0_pitch_moment_nm": float(preload[4]),
                "openfast_pitch_increment_deg": openfast_increment_deg,
                "difference_deg": target - openfast_increment_deg,
                "relative_difference_percent": 100.0
                * (target - openfast_increment_deg)
                / target,
                "original_damping_release_stability": stability,
            }
        )
    return {
        "evidence_level": "OpenFAST/source-consistent static no-MoorDyn endpoint check",
        "scope": {
            "purpose": "check the source-consistent hydrostatic-plus-gravity pitch scale at three static endpoints",
            "environment": "AeroDyn, InflowWind, ServoDyn, waves, current, and MoorDyn disabled; ElastoDyn and HydroDyn retained",
            "load_definition": "pure pitch HydroDyn AddF0 about the WAMIT/OpenFAST reference point",
            "readout_definition": "matched zero-preload original-damping release tail mean",
            "stability_definition": "roll and pitch release-tail range and drift only; no six-degree-of-freedom convergence claim",
            "not_a_mooring_validation": True,
            "not_a_dynamic_six_dof_validation": True,
            "not_a_controller_validation": True,
        },
        "source_consistent_static_reference": {
            "total_mass_kg": reference.total_mass_kg,
            "vertical_center_of_mass_m": reference.vertical_center_of_mass_m,
            "pitch_restoring_stiffness_without_mooring_nm_per_rad": stiffness,
        },
        "zero_preload_release_stability": _release_tail_stability(baseline_release),
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
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
