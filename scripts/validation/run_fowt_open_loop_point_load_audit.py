#!/usr/bin/env python3
"""Audit point-load mapping and open-loop response of the low-order platform.

This entry intentionally accepts an explicit platform-coordinate force.  It
does not choose an aerodynamic thrust law or invent a wave-excitation model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fowt_platform import (
    generalized_load_from_point_force,
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    load_volturnus_reference_components,
    solve_incremental_static_offset,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _named(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(DOF_NAMES)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_loop_case(
    *,
    model: IncrementalPlatformModel,
    force_platform_n: np.ndarray,
    application_point_m: np.ndarray,
    duration_s: float,
) -> dict[str, object]:
    generalized_force = generalized_load_from_point_force(
        force_platform_n,
        application_point_m,
    )
    loads = IncrementalLoads(
        wind=np.zeros(6),
        wave=np.zeros(6),
        ballast=np.zeros(6),
        other=generalized_force,
    )
    static_offset = solve_incremental_static_offset(model.matrices, loads.total)
    balanced_state = IncrementalState(
        position=static_offset.position,
        velocity=np.zeros(6),
    )
    balanced_acceleration = model.derivative(
        balanced_state,
        loads,
    ).velocity_rate
    response = model.advance_frozen_step(
        IncrementalState.zeros(),
        loads,
        duration_s=duration_s,
    )
    return {
        "force_platform_n": force_platform_n.tolist(),
        "application_point_from_reference_m": application_point_m.tolist(),
        "generalized_load": _named(generalized_force),
        "static_offset": _named(static_offset.position),
        "static_residual": _named(static_offset.residual_load),
        "balanced_acceleration": _named(balanced_acceleration),
        "response_from_reference_at_duration": _named(response.position),
    }


def _restoring_component_check(model: IncrementalPlatformModel) -> dict[str, object]:
    perturbation = np.array([1.0, 1.0, 1.0, np.deg2rad(1.0), np.deg2rad(1.0), np.deg2rad(1.0)])
    components = {
        "hydrostatic": model.matrices.hydrostatic_stiffness,
        "mooring": model.matrices.mooring_stiffness,
        "weight": model.matrices.weight_stiffness,
    }
    checks = {
        name: float(perturbation @ (-matrix @ perturbation)) <= 1e-8
        for name, matrix in components.items()
    }
    return {
        name: {
            "restoring_load": _named(-matrix @ perturbation),
            "virtual_work_j": float(perturbation @ (-matrix @ perturbation)),
        }
        for name, matrix in components.items()
    } | {"virtual_work_nonpositive": checks}


def run_audit(*, force_n: float, height_m: float, duration_s: float) -> dict[str, object]:
    if force_n <= 0.0 or height_m < 0.0 or duration_s <= 0.0:
        raise ValueError("force_n and duration_s must be positive; height_m must be non-negative")

    reference = load_volturnus_reference_components(REFERENCE_MANIFEST)
    matrices = reference.assemble_whole_system_candidate(damping=np.zeros((6, 6)))
    model = IncrementalPlatformModel(matrices)
    application_point = np.array([0.0, 0.0, height_m])
    directions = {
        "platform_forward": np.array([force_n, 0.0, 0.0]),
        "platform_port": np.array([0.0, force_n, 0.0]),
        "platform_aft": np.array([-force_n, 0.0, 0.0]),
        "platform_starboard": np.array([0.0, -force_n, 0.0]),
    }
    cases = {
        name: _open_loop_case(
            model=model,
            force_platform_n=force,
            application_point_m=application_point,
            duration_s=duration_s,
        )
        for name, force in directions.items()
    }
    moment_checks = {
        "forward_force_has_positive_pitch_moment": (
            cases["platform_forward"]["generalized_load"]["pitch"] > 0.0
        ),
        "port_force_has_negative_roll_moment": (
            cases["platform_port"]["generalized_load"]["roll"] < 0.0
        ),
        "aft_force_reverses_pitch_moment": (
            cases["platform_aft"]["generalized_load"]["pitch"] < 0.0
        ),
        "starboard_force_reverses_roll_moment": (
            cases["platform_starboard"]["generalized_load"]["roll"] > 0.0
        ),
    }
    static_checks = {
        name: bool(
            np.max(
                np.abs(
                    np.asarray(list(case["balanced_acceleration"].values()), dtype=float)
                )
            )
            < 1e-12
        )
        for name, case in cases.items()
    }
    restoring = _restoring_component_check(model)
    return {
        "evidence_level": "internal_open_loop_load_mapping_and_response_check",
        "reference_model": reference.model_version,
        "reference_point": reference.reference_point,
        "reference_manifest": str(REFERENCE_MANIFEST.relative_to(ROOT)),
        "reference_manifest_sha256": _sha256(REFERENCE_MANIFEST),
        "matrix_assembly_scope": (
            "P2 whole-system rigid-body candidate plus infinite-frequency "
            "added mass, hydrostatic, local mooring and weight stiffness"
        ),
        "force_input_scope": "explicit platform-coordinate point force",
        "not_aerodynamic_validation": True,
        "not_wave_validation": True,
        "not_physical_damping_validation": True,
        "damping_matrix": "zero; no local physical damping value has been assigned",
        "force_n": force_n,
        "application_height_m": height_m,
        "duration_s": duration_s,
        "restoring_component_check": restoring,
        "cases": cases,
        "checks": {
            "point_force_moment_signs": moment_checks,
            "balanced_static_offsets_have_negligible_acceleration": static_checks,
            "all_checks_pass": bool(
                all(moment_checks.values())
                and all(static_checks.values())
                and all(restoring["virtual_work_nonpositive"].values())
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-n", type=float, default=1.0e6)
    parser.add_argument("--height-m", type=float, default=150.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = run_audit(
        force_n=args.force_n,
        height_m=args.height_m,
        duration_s=args.duration_s,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
