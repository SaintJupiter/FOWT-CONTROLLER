#!/usr/bin/env python3
"""Audit one source-bound quasi-steady rotor-normal load input node.

The audit combines only public, frozen IEA 15 MW input facts that are needed
to evaluate one tabulated rotor-performance lookup point.  It does not estimate
relative wind from platform motion, pick a controller state, infer yaw
alignment, or advance the low-order platform model.  The result is therefore
an auditable generalized load input, not a dynamic-response validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from fowt_platform import (
    downwind_normal_relative_wind_component,
    generalized_load_from_point_force,
    load_rosco_rotor_performance_table_from_zip,
    openfast_reference_vector_to_frozen_equilibrium_axes,
    quasi_steady_rotor_normal_load_from_relative_air,
    relative_air_velocity_at_platform_point,
)


MODEL_ZIP = ROOT / (
    "references/fowt_reference_models/3_核心必用_官方模型文件包_IEAWindSystems_"
    "v1.1.16/IEAWindSystems_IEA-15-240-RWT_v1.1.16.zip"
)
MODEL_ARCHIVE_SHA256 = (
    "c96c6be345abf9440170764e2d9a2c201b5da6191192cef66537c5bc16805b80"
)
ARCHIVE_ROOT = "IEA-15-240-RWT-1.1.16/OpenFAST"
SHARED_DIRECTORY = "IEA-15-240-RWT"
SEMI_DIRECTORY = "IEA-15-240-RWT-UMaineSemi"
FST_MEMBER = f"{ARCHIVE_ROOT}/{SEMI_DIRECTORY}/{SEMI_DIRECTORY}.fst"
ELASTODYN_MEMBER = f"{ARCHIVE_ROOT}/{SEMI_DIRECTORY}/{SEMI_DIRECTORY}_ElastoDyn.dat"
ROTOR_TABLE_MEMBER = f"{ARCHIVE_ROOT}/{SHARED_DIRECTORY}/Cp_Ct_Cq.IEA15MW.txt"
REFERENCE_STATE_JSON = ROOT / (
    "configs/reference_platforms/volturnus_s_openfast_nominal_reference_state_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_openfast_scalar_field(text: str, field: str) -> float:
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and tokens[1] == field:
            try:
                value = float(tokens[0])
            except ValueError as error:
                raise ValueError(f"OpenFAST field {field} is not numeric") from error
            if not np.isfinite(value):
                raise ValueError(f"OpenFAST field {field} must be finite")
            return value
    raise ValueError(f"OpenFAST field not found: {field}")


def _source_values(model_zip: Path) -> dict[str, float | np.ndarray]:
    with zipfile.ZipFile(model_zip) as archive:
        fst = archive.read(FST_MEMBER).decode("utf-8")
        elastodyn = archive.read(ELASTODYN_MEMBER).decode("utf-8")

    air_density = _read_openfast_scalar_field(fst, "AirDens")
    rotor_radius = _read_openfast_scalar_field(elastodyn, "TipRad")
    shaft_tilt_deg = _read_openfast_scalar_field(elastodyn, "ShftTilt")
    initial_blade_pitch_deg = np.array(
        [
            _read_openfast_scalar_field(elastodyn, "BlPitch(1)"),
            _read_openfast_scalar_field(elastodyn, "BlPitch(2)"),
            _read_openfast_scalar_field(elastodyn, "BlPitch(3)"),
        ],
        dtype=float,
    )
    if not np.allclose(initial_blade_pitch_deg, initial_blade_pitch_deg[0], atol=1e-12):
        raise ValueError(
            "reference ElastoDyn input does not define one common initial blade pitch"
        )
    initial_rotor_speed_rpm = _read_openfast_scalar_field(elastodyn, "RotSpeed")
    tower_height = _read_openfast_scalar_field(elastodyn, "TowerHt")
    tower_to_shaft = _read_openfast_scalar_field(elastodyn, "Twr2Shft")
    overhang = _read_openfast_scalar_field(elastodyn, "OverHang")
    platform_reference_z = _read_openfast_scalar_field(elastodyn, "PtfmRefzt")

    shaft_tilt_rad = math.radians(shaft_tilt_deg)
    downwind_normal_openfast_reference = np.array(
        [math.cos(shaft_tilt_rad), 0.0, math.sin(shaft_tilt_rad)],
        dtype=float,
    )
    hub_from_reference_openfast_reference = np.array(
        [
            overhang * math.cos(shaft_tilt_rad),
            0.0,
            tower_height
            - platform_reference_z
            + tower_to_shaft
            + overhang * math.sin(shaft_tilt_rad),
        ],
        dtype=float,
    )
    return {
        "air_density_kg_m3": air_density,
        "rotor_radius_m": rotor_radius,
        "shaft_tilt_deg": shaft_tilt_deg,
        "initial_blade_pitch_deg": float(initial_blade_pitch_deg[0]),
        "initial_rotor_speed_rpm": initial_rotor_speed_rpm,
        "downwind_rotor_normal_openfast_reference": downwind_normal_openfast_reference,
        "hub_from_platform_reference_openfast_reference_m": hub_from_reference_openfast_reference,
    }


def _load_candidate_reference_state(
    reference_state_path: Path,
    *,
    model_zip: Path,
) -> dict[str, float]:
    """Load the narrow OpenFAST reference-state candidate for axis conversion."""

    try:
        payload = json.loads(reference_state_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"OpenFAST reference-state artifact not found: {reference_state_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid OpenFAST reference-state JSON: {reference_state_path}"
        ) from error
    source_model = payload.get("source_model")
    if not isinstance(source_model, dict) or source_model.get("archive_sha256") != _sha256(
        model_zip
    ):
        raise ValueError(
            "reference-state archive SHA-256 does not match the frozen model archive"
        )
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("reference-state artifact must define state")
    values: dict[str, float] = {}
    for field in ("surge_m", "heave_m", "pitch_deg"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"reference-state field {field} must be numeric")
        if not np.isfinite(value):
            raise ValueError(f"reference-state field {field} must be finite")
        values[field] = float(value)
    return values


def _named_generalized_load(load: np.ndarray) -> dict[str, float]:
    names = ("surge_n", "sway_n", "heave_n", "roll_nm", "pitch_nm", "yaw_nm")
    return {name: float(load[index]) for index, name in enumerate(names)}


def run_audit(
    *,
    pitch_deg: float | None = None,
    rotor_speed_rpm: float | None = None,
    wind_speed_mps: float,
    model_zip: Path = MODEL_ZIP,
    reference_state_path: Path = REFERENCE_STATE_JSON,
) -> dict[str, Any]:
    """Build one source-bound rotor-normal load without advancing the plant."""

    if not model_zip.is_file():
        raise FileNotFoundError(f"frozen model archive not found: {model_zip}")
    archive_sha256 = _sha256(model_zip)
    if archive_sha256 != MODEL_ARCHIVE_SHA256:
        raise ValueError("frozen model archive SHA-256 does not match the audit source")

    table = load_rosco_rotor_performance_table_from_zip(
        model_zip,
        ROTOR_TABLE_MEMBER,
        expected_archive_sha256=MODEL_ARCHIVE_SHA256,
    )
    source = _source_values(model_zip)
    reference_state = _load_candidate_reference_state(
        reference_state_path,
        model_zip=model_zip,
    )
    frozen_normal = openfast_reference_vector_to_frozen_equilibrium_axes(
        source["downwind_rotor_normal_openfast_reference"],
        equilibrium_platform_roll_deg=0.0,
        equilibrium_platform_pitch_deg=reference_state["pitch_deg"],
        equilibrium_platform_yaw_deg=0.0,
    )
    frozen_hub_from_reference = openfast_reference_vector_to_frozen_equilibrium_axes(
        source["hub_from_platform_reference_openfast_reference_m"],
        equilibrium_platform_roll_deg=0.0,
        equilibrium_platform_pitch_deg=reference_state["pitch_deg"],
        equilibrium_platform_yaw_deg=0.0,
    )
    table_wind_speed_metadata = float(table.wind_speed_metadata_mps[0])
    selected_wind_speed = float(wind_speed_mps)
    if not np.isfinite(selected_wind_speed) or selected_wind_speed <= 0.0:
        raise ValueError("wind_speed_mps must be finite and positive")
    selected_pitch_deg = (
        float(source["initial_blade_pitch_deg"])
        if pitch_deg is None
        else float(pitch_deg)
    )
    selected_rotor_speed_rpm = (
        float(source["initial_rotor_speed_rpm"])
        if rotor_speed_rpm is None
        else float(rotor_speed_rpm)
    )
    if not np.isfinite(selected_rotor_speed_rpm) or selected_rotor_speed_rpm < 0.0:
        raise ValueError("rotor_speed_rpm must be finite and non-negative")
    if not np.isfinite(selected_pitch_deg):
        raise ValueError("pitch_deg must be finite")

    rotor_radius = float(source["rotor_radius_m"])
    tip_speed_ratio = (
        selected_rotor_speed_rpm
        * 2.0
        * math.pi
        / 60.0
        * rotor_radius
        / selected_wind_speed
    )
    thrust_coefficient = table.ct_at(
        pitch_deg=selected_pitch_deg,
        tip_speed_ratio=tip_speed_ratio,
    )
    # The audit intentionally uses a stationary, aligned reference condition.
    # It exercises the new kinematic chain without claiming a dynamic inflow
    # or yaw model for the public OpenFAST operating node.
    ambient_air_velocity_platform = selected_wind_speed * frozen_normal
    relative_air_velocity_at_hub = relative_air_velocity_at_platform_point(
        ambient_air_velocity_platform_mps=ambient_air_velocity_platform,
        platform_reference_velocity_platform_mps=[0.0, 0.0, 0.0],
        platform_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
        point_from_platform_reference_m=frozen_hub_from_reference,
    )
    relative_normal_wind_speed = downwind_normal_relative_wind_component(
        relative_air_velocity_at_hub,
        frozen_normal,
    )
    rotor_load = quasi_steady_rotor_normal_load_from_relative_air(
        air_density_kg_m3=source["air_density_kg_m3"],
        rotor_radius_m=rotor_radius,
        thrust_coefficient=thrust_coefficient,
        relative_air_velocity_platform_mps=relative_air_velocity_at_hub,
        downwind_rotor_normal_platform=frozen_normal,
    )
    generalized_load = generalized_load_from_point_force(
        rotor_load.force_platform_n,
        frozen_hub_from_reference,
    )
    normal_openfast_reference = np.asarray(
        source["downwind_rotor_normal_openfast_reference"], dtype=float
    )
    checks = {
        "archive_identity_matches": archive_sha256 == MODEL_ARCHIVE_SHA256,
        "wind_speed_is_positive": bool(selected_wind_speed > 0.0),
        "rotor_normal_is_unit_length": bool(
            np.isclose(np.linalg.norm(frozen_normal), 1.0, atol=1e-12)
        ),
        "relative_normal_wind_speed_matches_selected_ambient_wind": bool(
            np.isclose(
                relative_normal_wind_speed,
                selected_wind_speed,
                atol=1e-12,
            )
        ),
    }
    checks["all_checks_pass"] = all(checks.values())

    return {
        "evidence_level": "source_bound_quasi_steady_rotor_normal_load_input",
        "boundaries": {
            "is_controller_operating_case": False,
            "is_platform_relative_wind_model": False,
            "uses_fixed_reference_relative_wind_kinematics": True,
            "is_yaw_or_misalignment_model": False,
            "is_wave_load_model": False,
            "is_low_order_dynamic_response_validation": False,
            "is_openfast_static_offset_comparison": False,
            "purpose": (
                "audit one published rotor-table operating node, an aligned "
                "fixed-reference relative-air vector, and its generalized "
                "load mapping in frozen equilibrium platform axes"
            ),
        },
        "source": {
            "model_zip": str(model_zip.relative_to(ROOT)),
            "model_zip_sha256": MODEL_ARCHIVE_SHA256,
            "rotor_performance_member": ROTOR_TABLE_MEMBER,
            "rotor_performance_text_sha256": table.source_sha256,
            "air_density_source_member": FST_MEMBER,
            "rotor_geometry_source_member": ELASTODYN_MEMBER,
            "candidate_reference_state": {
                "path": str(reference_state_path.relative_to(ROOT)),
                "sha256": _sha256(reference_state_path),
                "state": reference_state,
                "axis_conversion_attitude_deg": {
                    "roll_deg": 0.0,
                    "pitch_deg": reference_state["pitch_deg"],
                    "yaw_deg": 0.0,
                },
            },
        },
        "table_lookup_node": {
            "pitch_deg": selected_pitch_deg,
            "rotor_speed_rpm": selected_rotor_speed_rpm,
            "prescribed_aligned_ambient_wind_speed_mps": selected_wind_speed,
            "ambient_wind_source": "explicit audit input",
            "performance_table_wind_speed_metadata_mps": table_wind_speed_metadata,
            "coefficient_lookup_axes": ["pitch_deg", "tip_speed_ratio"],
            "tip_speed_ratio": tip_speed_ratio,
            "thrust_coefficient": thrust_coefficient,
            "pitch_source": (
                "ElastoDyn initial value" if pitch_deg is None else "explicit audit override"
            ),
            "rotor_speed_source": (
                "ElastoDyn initial value"
                if rotor_speed_rpm is None
                else "explicit audit override"
            ),
        },
        "fixed_reference_relative_air": {
            "ambient_air_velocity_platform_mps": [
                float(value) for value in ambient_air_velocity_platform
            ],
            "platform_reference_velocity_platform_mps": [0.0, 0.0, 0.0],
            "platform_angular_velocity_platform_radps": [0.0, 0.0, 0.0],
            "relative_air_velocity_at_hub_platform_mps": [
                float(value) for value in relative_air_velocity_at_hub
            ],
            "relative_normal_wind_speed_mps": relative_normal_wind_speed,
        },
        "source_parameters": {
            "air_density_kg_m3": float(source["air_density_kg_m3"]),
            "rotor_radius_m": rotor_radius,
            "shaft_tilt_deg": float(source["shaft_tilt_deg"]),
            "initial_blade_pitch_deg": float(source["initial_blade_pitch_deg"]),
            "initial_rotor_speed_rpm": float(source["initial_rotor_speed_rpm"]),
            "downwind_rotor_normal_openfast_reference": [
                float(value) for value in normal_openfast_reference
            ],
            "hub_from_platform_reference_openfast_reference_m": [
                float(value)
                for value in np.asarray(
                    source["hub_from_platform_reference_openfast_reference_m"], dtype=float
                )
            ],
            "downwind_rotor_normal_frozen_equilibrium": [
                float(value) for value in frozen_normal
            ],
            "hub_from_platform_reference_frozen_equilibrium_m": [
                float(value) for value in frozen_hub_from_reference
            ],
        },
        "rotor_normal_load": {
            "thrust_n": rotor_load.thrust_n,
            "force_platform_n": [float(value) for value in rotor_load.force_platform_n],
        },
        "generalized_load_about_platform_reference": _named_generalized_load(
            generalized_load
        ),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pitch-deg",
        type=float,
        help="explicit override; default is the common initial ElastoDyn blade pitch",
    )
    parser.add_argument(
        "--rotor-speed-rpm",
        type=float,
        help="explicit override; default is the initial ElastoDyn rotor speed",
    )
    parser.add_argument(
        "--wind-speed-mps",
        type=float,
        required=True,
        help=(
            "aligned ambient wind speed; the frozen C_T table is indexed by "
            "pitch and tip-speed ratio"
        ),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    result = run_audit(
        pitch_deg=args.pitch_deg,
        rotor_speed_rpm=args.rotor_speed_rpm,
        wind_speed_mps=args.wind_speed_mps,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
