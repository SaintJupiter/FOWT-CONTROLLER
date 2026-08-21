#!/usr/bin/env python3
"""Measure the conditional static pitch response with MoorDyn toggled.

The current low-order candidate combines a first-order HydroDyn/WAMIT matrix
with an auxiliary second-order WAMIT force-control representation.  Before
changing any source value, this audit measures equal positive and negative
``AddF0`` pitch preloads with MoorDyn enabled and disabled in otherwise
identical OpenFAST copies.

The reported difference is a *system-level conditional response difference*
after the remaining platform degrees of freedom settle.  It is deliberately
not interpreted as a raw MoorDyn K55 entry, a standalone gravity stiffness, or
an additive source decomposition.
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

from fowt_platform.reference import load_volturnus_reference_components
from run_openfast_moment_angle_consistency import (
    REFERENCE_MANIFEST,
    REFERENCE_STATE_JSON,
    RELAXATION_ADD_BLIN_DIAGONAL,
    _configure_no_environment_case,
    _frozen_equilibrium_load,
    _load_candidate_reference_state,
    _named_six_vector,
    _platform_reference_state,
    _release_tail_stability,
    _tail_mean_state,
)
from run_openfast_prescribed_wind_node import (
    _assert_output_reaches_time_horizon,
    _extract_clean_model,
    _read_openfast_output,
    _replace_field,
)
from run_openfast_prescribed_wind_relaxation_release_audit import _late_window_statistics


DEFAULT_PITCH_MOMENT_NM = 10_000_000.0
DEFAULT_FULL_REFERENCE_JSON = (
    ROOT / "artifacts" / "openfast_moment_angle_consistency_20260821.json"
)


def _set_moordyn_enabled(fst_text: str, *, enabled: bool) -> str:
    """Select the public MoorDyn file or remove the mooring module entirely."""

    return _replace_field(fst_text, "CompMooring", "3" if enabled else "0")


def _run_phase(
    *,
    phase_root: Path,
    openfast_binary: Path,
    duration_s: float,
    output_step_s: float,
    initial_state: dict[str, float],
    preload: np.ndarray,
    tail_window_s: float,
    moordyn_enabled: bool,
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
    fst_path.write_text(
        _set_moordyn_enabled(
            fst_path.read_text(encoding="utf-8"), enabled=moordyn_enabled
        ),
        encoding="utf-8",
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
            "OpenFAST component phase failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    columns, _ = _read_openfast_output(semi_dir / f"{fst_path.stem}.out")
    _assert_output_reaches_time_horizon(
        columns, duration_s=duration_s, output_step_s=output_step_s
    )
    return {
        "input_tree": str(semi_dir),
        "moordyn_enabled": moordyn_enabled,
        "initial_state": initial_state,
        "configured_add_f0_about_wrp_si": _named_six_vector(configured_preload),
        "hydrodyn_damping_mode": (
            "original_public_input"
            if temporary_add_blin_diagonal is None
            else "temporary_numerical_relaxation_add_blin"
        ),
        "late_window_statistics": _late_window_statistics(
            columns, duration_s=duration_s, tail_window_s=tail_window_s
        ),
    }


def _tail_mean_pitch(phase: dict[str, object]) -> float:
    statistics = phase["late_window_statistics"]
    if not isinstance(statistics, dict):
        raise ValueError("phase is missing tail statistics")
    values = statistics["second_window_statistics"]
    if not isinstance(values, dict):
        raise ValueError("phase is missing second-window statistics")
    pitch = values["pitch_deg"]
    if not isinstance(pitch, dict):
        raise ValueError("phase is missing tail pitch statistics")
    return float(pitch["mean"])


def _run_configuration(
    *,
    label: str,
    moordyn_enabled: bool,
    pitch_moment_nm: float,
    nominal_state: dict[str, float],
    work_dir: Path,
    openfast_binary: Path,
    relaxation_duration_s: float,
    release_duration_s: float,
    output_step_s: float,
    tail_window_s: float,
) -> dict[str, object]:
    zero = np.zeros(6)
    reference_state = _platform_reference_state(nominal_state)
    baseline_relaxation = _run_phase(
        phase_root=work_dir / label / "zero_preload" / "relaxation",
        openfast_binary=openfast_binary,
        duration_s=relaxation_duration_s,
        output_step_s=output_step_s,
        initial_state=reference_state,
        preload=zero,
        tail_window_s=tail_window_s,
        moordyn_enabled=moordyn_enabled,
        temporary_add_blin_diagonal=RELAXATION_ADD_BLIN_DIAGONAL,
    )
    baseline_release = _run_phase(
        phase_root=work_dir / label / "zero_preload" / "release",
        openfast_binary=openfast_binary,
        duration_s=release_duration_s,
        output_step_s=output_step_s,
        initial_state=_tail_mean_state(baseline_relaxation),
        preload=zero,
        tail_window_s=tail_window_s,
        moordyn_enabled=moordyn_enabled,
        temporary_add_blin_diagonal=None,
    )
    def run_loaded_case(sign_label: str, sign: float) -> dict[str, object]:
        preload = np.zeros(6)
        preload[4] = sign * pitch_moment_nm
        relaxation = _run_phase(
            phase_root=work_dir / label / sign_label / "relaxation",
            openfast_binary=openfast_binary,
            duration_s=relaxation_duration_s,
            output_step_s=output_step_s,
            initial_state=reference_state,
            preload=preload,
            tail_window_s=tail_window_s,
            moordyn_enabled=moordyn_enabled,
            temporary_add_blin_diagonal=RELAXATION_ADD_BLIN_DIAGONAL,
        )
        release = _run_phase(
            phase_root=work_dir / label / sign_label / "release",
            openfast_binary=openfast_binary,
            duration_s=release_duration_s,
            output_step_s=output_step_s,
            initial_state=_tail_mean_state(relaxation),
            preload=preload,
            tail_window_s=tail_window_s,
            moordyn_enabled=moordyn_enabled,
            temporary_add_blin_diagonal=None,
        )
        return {
            "temporary_relaxation": relaxation,
            "original_damping_release": release,
            "original_damping_release_stability": _release_tail_stability(release),
            "pitch_increment_from_matched_baseline_deg": (
                _tail_mean_pitch(release) - _tail_mean_pitch(baseline_release)
            ),
        }

    positive = run_loaded_case("positive_pitch_preload", 1.0)
    negative = run_loaded_case("negative_pitch_preload", -1.0)
    positive_increment_deg = float(positive["pitch_increment_from_matched_baseline_deg"])
    negative_increment_deg = float(negative["pitch_increment_from_matched_baseline_deg"])
    odd_increment_rad = float(np.deg2rad((positive_increment_deg - negative_increment_deg) / 2.0))
    if abs(odd_increment_rad) <= 1.0e-12:
        raise ValueError(f"{label} produced a zero central pitch increment")
    return {
        "label": label,
        "moordyn_enabled": moordyn_enabled,
        "baseline": {
            "temporary_relaxation": baseline_relaxation,
            "original_damping_release": baseline_release,
            "original_damping_release_stability": _release_tail_stability(
                baseline_release
            ),
        },
        "positive_pitch_preload": positive,
        "negative_pitch_preload": negative,
        "central_odd_pitch_increment_deg": (
            (positive_increment_deg - negative_increment_deg) / 2.0
        ),
        "central_even_pitch_increment_deg": (
            (positive_increment_deg + negative_increment_deg) / 2.0
        ),
        "effective_pitch_stiffness_gnm_per_rad": pitch_moment_nm / odd_increment_rad / 1.0e9,
    }


def _existing_full_configuration(
    *, reference_result_path: Path, pitch_moment_nm: float
) -> dict[str, object]:
    """Reuse the already-completed full-system ±10 MNm audit exactly as run."""

    payload = json.loads(reference_result_path.read_text(encoding="utf-8"))
    reference = payload.get("reference")
    cases = payload.get("cases")
    zero_preload = payload.get("zero_preload")
    if not isinstance(reference, dict) or not isinstance(cases, list) or not isinstance(zero_preload, dict):
        raise ValueError("full-system reference result has an unexpected structure")
    if not np.isclose(float(reference.get("moment_magnitude_nm", float("nan"))), pitch_moment_nm):
        raise ValueError("full-system reference result uses a different pitch-moment magnitude")
    by_id = {str(case.get("case_id")): case for case in cases if isinstance(case, dict)}
    try:
        positive = by_id["pure_pitch_positive"]
        negative = by_id["pure_pitch_negative"]
        positive_increment = float(
            positive["openfast"]["increment_from_same_run_zero_preload_tail_mean_deg"]["pitch_deg"]
        )
        negative_increment = float(
            negative["openfast"]["increment_from_same_run_zero_preload_tail_mean_deg"]["pitch_deg"]
        )
        positive_stability = positive["original_damping_release_stability"]
        negative_stability = negative["original_damping_release_stability"]
    except (KeyError, TypeError) as error:
        raise ValueError("full-system reference result is missing pitch cases") from error
    odd_increment_rad = float(np.deg2rad((positive_increment - negative_increment) / 2.0))
    if abs(odd_increment_rad) <= 1.0e-12:
        raise ValueError("full-system reference result has a zero central pitch increment")
    return {
        "label": "moordyn_enabled",
        "moordyn_enabled": True,
        "source": {"kind": "reused_completed_audit", "path": str(reference_result_path)},
        "baseline": {
            "original_damping_release_stability": zero_preload.get(
                "original_damping_release_stability"
            )
        },
        "positive_pitch_preload": {
            "original_damping_release_stability": positive_stability,
            "pitch_increment_from_matched_baseline_deg": positive_increment,
        },
        "negative_pitch_preload": {
            "original_damping_release_stability": negative_stability,
            "pitch_increment_from_matched_baseline_deg": negative_increment,
        },
        "central_odd_pitch_increment_deg": (
            (positive_increment - negative_increment) / 2.0
        ),
        "central_even_pitch_increment_deg": (
            (positive_increment + negative_increment) / 2.0
        ),
        "effective_pitch_stiffness_gnm_per_rad": pitch_moment_nm / odd_increment_rad / 1.0e9,
    }


def _candidate_effective_pitch_stiffness(*, include_mooring: bool) -> float:
    reference = load_volturnus_reference_components(REFERENCE_MANIFEST)
    full = reference.assemble_whole_system_candidate(np.zeros((6, 6)))
    restoring = full.hydrostatic_stiffness + full.weight_stiffness
    if include_mooring:
        restoring = restoring + full.mooring_stiffness

    # The no-MoorDyn matrix has free surge and yaw modes.  In this symmetric
    # pure-pitch preload, surge receives no direct load and is uncoupled from
    # heave/pitch in the HydroDyn-plus-weight block.  Remove only these free
    # coordinates rather than supplying an artificial constraint.
    pitch_coordinates = (0, 2, 4) if include_mooring else (2, 4)
    reduced = restoring[np.ix_(pitch_coordinates, pitch_coordinates)]
    unit_pitch_moment = np.zeros(len(pitch_coordinates))
    unit_pitch_moment[-1] = 1.0
    compliance = float(np.linalg.solve(reduced, unit_pitch_moment)[-1])
    if compliance <= 0.0 or not np.isfinite(compliance):
        raise ValueError("candidate pitch compliance must be positive and finite")
    return 1.0 / compliance / 1.0e9


def run_audit(
    *,
    work_dir: Path,
    openfast_binary: Path,
    pitch_moment_nm: float = DEFAULT_PITCH_MOMENT_NM,
    relaxation_duration_s: float = 300.0,
    release_duration_s: float = 600.0,
    output_step_s: float = 1.0,
    tail_window_s: float = 150.0,
    nominal_reference_state_path: Path = REFERENCE_STATE_JSON,
    full_reference_result_path: Path | None = DEFAULT_FULL_REFERENCE_JSON,
) -> dict[str, object]:
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")
    if not np.isfinite(pitch_moment_nm) or pitch_moment_nm <= 0.0:
        raise ValueError("pitch_moment_nm must be finite and positive")
    if min(relaxation_duration_s, release_duration_s, output_step_s, tail_window_s) <= 0.0:
        raise ValueError("durations and output_step_s must be positive")
    if tail_window_s > release_duration_s:
        raise ValueError("tail_window_s must not exceed release_duration_s")

    nominal_state = _load_candidate_reference_state(nominal_reference_state_path)
    openfast_preload = np.zeros(6)
    openfast_preload[4] = pitch_moment_nm
    frozen_preload = _frozen_equilibrium_load(openfast_preload, nominal_reference=nominal_state)
    full = (
        _existing_full_configuration(
            reference_result_path=full_reference_result_path,
            pitch_moment_nm=pitch_moment_nm,
        )
        if full_reference_result_path is not None
        else _run_configuration(
            label="hydrodyn_plus_moordyn",
            moordyn_enabled=True,
            pitch_moment_nm=pitch_moment_nm,
            nominal_state=nominal_state,
            work_dir=work_dir,
            openfast_binary=openfast_binary,
            relaxation_duration_s=relaxation_duration_s,
            release_duration_s=release_duration_s,
            output_step_s=output_step_s,
            tail_window_s=tail_window_s,
        )
    )
    moordyn_disabled = _run_configuration(
        label="moordyn_disabled",
        moordyn_enabled=False,
        pitch_moment_nm=pitch_moment_nm,
        nominal_state=nominal_state,
        work_dir=work_dir,
        openfast_binary=openfast_binary,
        relaxation_duration_s=relaxation_duration_s,
        release_duration_s=release_duration_s,
        output_step_s=output_step_s,
        tail_window_s=tail_window_s,
    )

    full_k = float(full["effective_pitch_stiffness_gnm_per_rad"])
    moordyn_disabled_k = float(
        moordyn_disabled["effective_pitch_stiffness_gnm_per_rad"]
    )
    candidate_full_k = _candidate_effective_pitch_stiffness(include_mooring=True)
    candidate_hydro_k = _candidate_effective_pitch_stiffness(include_mooring=False)
    return {
        "evidence_level": "OpenFAST component-ablation static pitch audit",
        "scope": {
            "purpose": "measure the conditional static pitch response with MoorDyn enabled and disabled before changing low-order source values",
            "environment": "AeroDyn, InflowWind, ServoDyn, waves, and current disabled; ElastoDyn and HydroDyn retained in both configurations",
            "same_between_configurations": "OpenFAST model identity, six platform DOFs, initial-state procedure, equal positive/negative HydroDyn AddF0 preloads, temporary relaxation procedure, original-damping release, and tail readout",
            "changed_between_configurations": "CompMooring is 3 for the full configuration and 0 for the HydroDyn-only configuration",
            "interpretation_boundary": "The comparison is a conditional terminal-pitch readout after the other platform coordinates settle. Disabling MoorDyn changes mooring pretension, the zero-load equilibrium, and the available static restraints. The result is not a raw line stiffness, a standalone gravity stiffness, an additive matrix decomposition, or a dynamic validation.",
        },
        "input": {
            "positive_openfast_add_f0_about_wrp_si": _named_six_vector(openfast_preload),
            "negative_openfast_add_f0_about_wrp_si": _named_six_vector(-openfast_preload),
            "positive_frozen_equilibrium_load_si": _named_six_vector(frozen_preload),
            "negative_frozen_equilibrium_load_si": _named_six_vector(-frozen_preload),
            "nominal_reference_state": nominal_state,
            "relaxation_duration_s": relaxation_duration_s,
            "release_duration_s": release_duration_s,
            "tail_window_s": tail_window_s,
        },
        "openfast": {
            "moordyn_enabled": full,
            "moordyn_disabled": moordyn_disabled,
            "moordyn_toggle_conditional_pitch_slope_difference_gnm_per_rad": (
                full_k - moordyn_disabled_k
            ),
            "readout_qualification": "The MoorDyn-disabled result is a conditional pitch terminal slope only. Its surge, heave, and yaw tail states remain in the stored phase records and must be reviewed with the pitch readout.",
        },
        "current_low_order_candidate": {
            "effective_pitch_stiffness_with_auxiliary_frc_mooring_gnm_per_rad": candidate_full_k,
            "without_auxiliary_frc_mooring_matrix_sensitivity_gnm_per_rad": candidate_hydro_k,
            "matrix_sensitivity_boundary": "The no-auxiliary-mooring matrix reduction is not directly comparable with the MoorDyn-disabled OpenFAST condition because it removes free coordinates by construction while OpenFAST is allowed to settle those coordinates.",
        },
        "comparison": {
            "current_low_order_candidate_minus_moordyn_enabled_openfast_gnm_per_rad": (
                candidate_full_k - full_k
            ),
            "comparison_boundary": "This audit quantifies the OpenFAST module-toggle response but cannot assign the current full-system mismatch to MoorDyn, hydrostatics, gravity, or inertia individually. A source-consistent low-order assembly must be rebuilt and checked against the MoorDyn-enabled configuration before any coefficient is changed.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--openfast", type=Path, default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast")
    )
    parser.add_argument("--pitch-moment-mnm", type=float, default=DEFAULT_PITCH_MOMENT_NM / 1.0e6)
    parser.add_argument("--relaxation-duration-s", type=float, default=300.0)
    parser.add_argument("--release-duration-s", type=float, default=600.0)
    parser.add_argument("--output-step-s", type=float, default=1.0)
    parser.add_argument("--tail-window-s", type=float, default=150.0)
    parser.add_argument(
        "--nominal-reference-state-json", type=Path, default=REFERENCE_STATE_JSON
    )
    parser.add_argument(
        "--full-reference-json", type=Path, default=DEFAULT_FULL_REFERENCE_JSON,
        help="reuse a completed full HydroDyn+MoorDyn ± pitch audit; pass --rerun-full to disable reuse",
    )
    parser.add_argument("--rerun-full", action="store_true")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        work_dir=args.work_dir,
        openfast_binary=args.openfast,
        pitch_moment_nm=args.pitch_moment_mnm * 1.0e6,
        relaxation_duration_s=args.relaxation_duration_s,
        release_duration_s=args.release_duration_s,
        output_step_s=args.output_step_s,
        tail_window_s=args.tail_window_s,
        nominal_reference_state_path=args.nominal_reference_state_json,
        full_reference_result_path=(None if args.rerun_full else args.full_reference_json),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
