#!/usr/bin/env python3
"""Check lateral residual motion around a prescribed-wind candidate pose.

The input candidate is read from a prior relaxation-and-release audit.  This
script starts two independent original-HydroDyn runs from that same pose: one
unperturbed and one with a small sway or yaw offset.  It reports the relative
tail-state change only.  It neither creates a reference-state artifact nor
identifies a physical damping or equilibrium model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


VALIDATION_DIRECTORY = Path(__file__).resolve().parent
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from run_openfast_prescribed_wind_relaxation_release_audit import (
    PLATFORM_STATE_FIELDS,
    _normalise_platform_initial_state,
    _run_phase,
)
from run_openfast_prescribed_wind_node import (
    MODEL_ZIP,
    _sha256,
)


CANDIDATE_STATE_KEYS = (
    "temporary_relaxation",
    "late_window_statistics",
    "second_window_mean_state_for_release_only",
)
LATERAL_PERTURBATION_FIELDS = ("sway_m", "yaw_deg")


def _load_candidate_audit(
    relaxation_audit_path: Path,
) -> tuple[dict[str, float], dict[str, object]]:
    """Load a candidate pose and the operating condition that produced it."""
    try:
        payload: object = json.loads(relaxation_audit_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"relaxation-release audit JSON not found: {relaxation_audit_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid relaxation-release audit JSON: {relaxation_audit_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("relaxation-release audit JSON must be an object")

    reference = payload.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("relaxation-release audit JSON is missing reference metadata")

    candidate: object = payload
    for key in CANDIDATE_STATE_KEYS:
        if not isinstance(candidate, dict) or key not in candidate:
            dotted_path = ".".join(CANDIDATE_STATE_KEYS)
            raise ValueError(f"audit JSON is missing candidate state at {dotted_path}")
        candidate = candidate[key]
    if not isinstance(candidate, dict):
        raise ValueError("audit candidate state must be an object")
    return _normalise_platform_initial_state(candidate), reference


def _load_candidate_state(relaxation_audit_path: Path) -> dict[str, float]:
    """Load only the explicitly non-reference candidate pose from an audit."""
    state, _ = _load_candidate_audit(relaxation_audit_path)
    return state


def _validate_candidate_operating_condition(
    reference: dict[str, object], *, wind_speed_mps: float
) -> tuple[float, float]:
    """Reject a candidate pose created under a different prescribed-wind node."""
    expected_values: dict[str, object] = {
        "model_zip_sha256": _sha256(MODEL_ZIP),
        "wave_and_current": "disabled",
    }
    for field, expected in expected_values.items():
        if reference.get(field) != expected:
            raise ValueError(
                "candidate audit operating condition does not match this release: "
                f"{field}={reference.get(field)!r}, expected {expected!r}"
            )
    recorded_wind = reference.get("wind_speed_mps")
    if (
        isinstance(recorded_wind, bool)
        or not isinstance(recorded_wind, (int, float))
        or not np.isfinite(recorded_wind)
        or not np.isclose(float(recorded_wind), wind_speed_mps, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError(
            "candidate audit operating condition does not match this release: "
            f"wind_speed_mps={recorded_wind!r}, expected {wind_speed_mps!r}"
        )
    pitch = reference.get("prescribed_blade_pitch_deg")
    rotor_speed = reference.get("prescribed_rotor_speed_rpm")
    if (
        isinstance(pitch, bool)
        or not isinstance(pitch, (int, float))
        or not np.isfinite(float(pitch))
    ):
        raise ValueError(
            "candidate audit operating condition is missing a finite "
            "prescribed_blade_pitch_deg"
        )
    if (
        isinstance(rotor_speed, bool)
        or not isinstance(rotor_speed, (int, float))
        or not np.isfinite(float(rotor_speed))
        or float(rotor_speed) < 0.0
    ):
        raise ValueError(
            "candidate audit operating condition is missing a non-negative "
            "prescribed_rotor_speed_rpm"
        )
    return float(pitch), float(rotor_speed)


def _perturb_state(
    candidate_state: dict[str, float], *, field: str, value: float
) -> dict[str, float]:
    if field not in LATERAL_PERTURBATION_FIELDS:
        raise ValueError(
            f"field must be one of {LATERAL_PERTURBATION_FIELDS}, got {field!r}"
        )
    if not np.isfinite(value) or value == 0.0:
        raise ValueError("perturbation value must be finite and nonzero")
    perturbed = _normalise_platform_initial_state(candidate_state)
    perturbed[field] += float(value)
    return _normalise_platform_initial_state(perturbed)


def _mean_state(
    phase: dict[str, object], *, window_key: str
) -> dict[str, float]:
    statistics = phase.get("late_window_statistics")
    if not isinstance(statistics, dict):
        raise ValueError("OpenFAST phase result is missing late-window statistics")
    window = statistics.get(window_key)
    if not isinstance(window, dict):
        raise ValueError(f"OpenFAST phase result is missing {window_key}")
    return _normalise_platform_initial_state(
        {
            state_name: values["mean"]
            for state_name, _ in PLATFORM_STATE_FIELDS
            if isinstance(values := window.get(state_name), dict) and "mean" in values
        }
    )


def _state_difference(
    left: dict[str, float], right: dict[str, float]
) -> dict[str, float]:
    left = _normalise_platform_initial_state(left)
    right = _normalise_platform_initial_state(right)
    return {state_name: float(left[state_name] - right[state_name]) for state_name, _ in PLATFORM_STATE_FIELDS}


def _relative_tail_state_change(
    *, unperturbed: dict[str, object], perturbed: dict[str, object]
) -> dict[str, dict[str, float]]:
    first = _state_difference(
        _mean_state(perturbed, window_key="first_window_statistics"),
        _mean_state(unperturbed, window_key="first_window_statistics"),
    )
    second = _state_difference(
        _mean_state(perturbed, window_key="second_window_statistics"),
        _mean_state(unperturbed, window_key="second_window_statistics"),
    )
    return {
        "first_window_perturbed_minus_unperturbed": first,
        "second_window_perturbed_minus_unperturbed": second,
        "second_minus_first_relative_state": _state_difference(second, first),
    }


def run_audit(
    *,
    work_dir: Path,
    openfast_binary: Path,
    relaxation_audit_path: Path,
    duration_s: float,
    output_step_s: float,
    wind_speed_mps: float,
    tail_window_s: float,
    perturbation_field: str,
    perturbation_value: float,
) -> dict[str, object]:
    if duration_s <= 0.0 or output_step_s <= 0.0:
        raise ValueError("duration_s and output_step_s must be positive")
    if wind_speed_mps <= 0.0:
        raise ValueError("wind_speed_mps must be positive")
    if not 0.0 < tail_window_s <= duration_s:
        raise ValueError("tail_window_s must be positive and no greater than duration_s")
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")

    candidate_state, candidate_reference = _load_candidate_audit(relaxation_audit_path)
    prescribed_blade_pitch_deg, prescribed_rotor_speed_rpm = (
        _validate_candidate_operating_condition(
        candidate_reference, wind_speed_mps=wind_speed_mps
        )
    )
    perturbed_state = _perturb_state(
        candidate_state, field=perturbation_field, value=perturbation_value
    )
    unperturbed = _run_phase(
        phase_root=work_dir / "unperturbed",
        openfast_binary=openfast_binary,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=candidate_state,
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=None,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    perturbed = _run_phase(
        phase_root=work_dir / "perturbed",
        openfast_binary=openfast_binary,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        initial_state=perturbed_state,
        tail_window_s=tail_window_s,
        temporary_add_blin_diagonal=None,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    return {
        "evidence_level": "OpenFAST prescribed-wind lateral perturbation release audit",
        "boundaries": {
            "is_rosco_controlled_operating_case": False,
            "is_physical_damping_calibration": False,
            "is_static_equilibrium_solver": False,
            "automatically_creates_reference_state_artifact": False,
            "purpose": "compare an unperturbed and a lateral-perturbed release around a candidate pose",
        },
        "reference": {
            "model_zip": str(MODEL_ZIP),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "wind_speed_mps": wind_speed_mps,
            "prescribed_blade_pitch_deg": prescribed_blade_pitch_deg,
            "prescribed_rotor_speed_rpm": prescribed_rotor_speed_rpm,
            "relaxation_audit_path": str(relaxation_audit_path),
            "candidate_state_source": ".".join(CANDIDATE_STATE_KEYS),
            "candidate_state": candidate_state,
        },
        "perturbation": {
            "field": perturbation_field,
            "value": float(perturbation_value),
            "perturbed_initial_state": perturbed_state,
        },
        "unperturbed_release": unperturbed,
        "perturbed_release": perturbed,
        "relative_tail_state_change": _relative_tail_state_change(
            unperturbed=unperturbed, perturbed=perturbed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--relaxation-audit-json", type=Path, required=True)
    parser.add_argument(
        "--openfast", type=Path, default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast")
    )
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--output-step-s", type=float, default=1.0)
    parser.add_argument("--wind-speed-mps", type=float, default=10.74)
    parser.add_argument("--tail-window-s", type=float, default=150.0)
    parser.add_argument("--perturbation-field", choices=LATERAL_PERTURBATION_FIELDS, default="sway_m")
    parser.add_argument("--perturbation-value", type=float, default=0.1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = run_audit(
        work_dir=args.work_dir,
        openfast_binary=args.openfast,
        relaxation_audit_path=args.relaxation_audit_json,
        duration_s=args.duration_s,
        output_step_s=args.output_step_s,
        wind_speed_mps=args.wind_speed_mps,
        tail_window_s=args.tail_window_s,
        perturbation_field=args.perturbation_field,
        perturbation_value=args.perturbation_value,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
