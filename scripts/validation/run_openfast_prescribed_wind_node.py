#!/usr/bin/env python3
"""Run a narrow OpenFAST reference node for the low-order wind-load audit.

The generated OpenFAST copies use the public IEA 15 MW / UMaine VolturnUS-S
input deck with wave and current disabled. Rotor speed and blade pitch are
prescribed, so this is deliberately *not* a ROSCO-controlled operating case.
Both cases keep InflowWind and AeroDyn enabled and differ only in the imposed
uniform wind speed. OpenFAST requires a positive steady wind speed, so the
baseline is a documented low-wind reference rather than artificial still air.
The entry records an OpenFAST response node and AeroDyn hub-load channels; it
does not claim a static-equilibrium or full-model match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from fowt_platform import generalized_load_from_openfast_hub_wrench


MODEL_ZIP = ROOT / (
    "references/fowt_reference_models/3_核心必用_官方模型文件包_IEAWindSystems_"
    "v1.1.16/IEAWindSystems_IEA-15-240-RWT_v1.1.16.zip"
)
ARCHIVE_ROOT = "IEA-15-240-RWT-1.1.16/OpenFAST"
SEMI_DIRECTORY = "IEA-15-240-RWT-UMaineSemi"
FST_NAME = "IEA-15-240-RWT-UMaineSemi.fst"
ED_NAME = "IEA-15-240-RWT-UMaineSemi_ElastoDyn.dat"
SEA_NAME = "IEA-15-240-RWT-UMaineSemi_SeaState.dat"
INFLOW_NAME = "IEA-15-240-RWT_InflowFile.dat"
PRESCRIBED_ROTOR_SPEED_RPM = 7.55
PRESCRIBED_BLADE_PITCH_DEG = 1.0
REFERENCE_STATE_JSON = ROOT / (
    "configs/reference_platforms/volturnus_s_openfast_nominal_reference_state_v1.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidate_reference_state(
    reference_state_path: Path,
    *,
    model_zip: Path = MODEL_ZIP,
) -> dict[str, float]:
    """Load the audited OpenFAST initial state and bind it to its source archive."""
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

    if payload.get("schema_version") != 1:
        raise ValueError("unsupported OpenFAST reference-state schema version")
    source_model = payload.get("source_model")
    if not isinstance(source_model, dict):
        raise ValueError("reference-state artifact must define source_model")
    expected_archive_sha256 = source_model.get("archive_sha256")
    if not isinstance(expected_archive_sha256, str) or len(expected_archive_sha256) != 64:
        raise ValueError("reference-state artifact must define a SHA-256 archive identity")
    actual_archive_sha256 = _sha256(model_zip)
    if actual_archive_sha256 != expected_archive_sha256:
        raise ValueError(
            "reference-state archive SHA-256 does not match the OpenFAST model archive"
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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _replace_field(text: str, field: str, value: str) -> str:
    lines = []
    replaced = False
    for line in text.splitlines(keepends=True):
        fields = line.split()
        if len(fields) >= 2 and fields[1] == field:
            suffix = line[line.index(field) + len(field) :]
            lines.append(f"{value:<22} {field}{suffix}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise ValueError(f"missing OpenFAST field {field}")
    return "".join(lines)


def _extract_clean_model(work_dir: Path) -> Path:
    extraction_root = work_dir / "source"
    shared_prefix = f"{ARCHIVE_ROOT}/IEA-15-240-RWT/"
    semi_prefix = f"{ARCHIVE_ROOT}/{SEMI_DIRECTORY}/"
    with zipfile.ZipFile(MODEL_ZIP) as archive:
        members = [
            member
            for member in archive.namelist()
            if member.startswith(shared_prefix) or member.startswith(semi_prefix)
        ]
        archive.extractall(extraction_root, members)
    semi_dir = extraction_root / ARCHIVE_ROOT / SEMI_DIRECTORY
    if not (semi_dir / FST_NAME).is_file():
        raise FileNotFoundError("OpenFAST UMaineSemi input was not extracted")
    return semi_dir


def _configure_case(
    semi_dir: Path,
    *,
    wind_speed_mps: float,
    duration_s: float,
    output_step_s: float,
    equilibrium_surge_m: float,
    equilibrium_heave_m: float,
    equilibrium_pitch_deg: float,
    prescribed_blade_pitch_deg: float = PRESCRIBED_BLADE_PITCH_DEG,
    prescribed_rotor_speed_rpm: float = PRESCRIBED_ROTOR_SPEED_RPM,
) -> Path:
    fst_path = semi_dir / FST_NAME
    fst = fst_path.read_text(encoding="utf-8")
    for field, value in (
        ("TMax", f"{duration_s:.6g}"),
        ("CompInflow", "1"),
        ("CompAero", "2"),
        ("CompServo", "0"),
        ("OutFileFmt", "1"),
        ("DT_Out", f"{output_step_s:.6g}"),
    ):
        fst = _replace_field(fst, field, value)
    fst_path.write_text(fst, encoding="utf-8")

    ed_path = semi_dir / ED_NAME
    ed = ed_path.read_text(encoding="utf-8")
    for field in (
        "FlapDOF1",
        "FlapDOF2",
        "EdgeDOF",
        "GenDOF",
        "TwFADOF1",
        "TwFADOF2",
        "TwSSDOF1",
        "TwSSDOF2",
    ):
        ed = _replace_field(ed, field, "False")
    for field, value in (
        ("BlPitch(1)", f"{prescribed_blade_pitch_deg:.9g}"),
        ("BlPitch(2)", f"{prescribed_blade_pitch_deg:.9g}"),
        ("BlPitch(3)", f"{prescribed_blade_pitch_deg:.9g}"),
        ("RotSpeed", f"{prescribed_rotor_speed_rpm:.9g}"),
        ("PtfmSurge", f"{equilibrium_surge_m:.9g}"),
        ("PtfmHeave", f"{equilibrium_heave_m:.9g}"),
        ("PtfmPitch", f"{equilibrium_pitch_deg:.9g}"),
    ):
        ed = _replace_field(ed, field, value)
    ed_path.write_text(ed, encoding="utf-8")

    sea_path = semi_dir / SEA_NAME
    sea = sea_path.read_text(encoding="utf-8")
    sea = _replace_field(sea, "WaveMod", "0")
    sea = _replace_field(sea, "CurrMod", "0")
    sea_path.write_text(sea, encoding="utf-8")

    inflow_path = semi_dir.parent / "IEA-15-240-RWT" / INFLOW_NAME
    inflow = inflow_path.read_text(encoding="utf-8")
    inflow = _replace_field(inflow, "WindType", "1")
    inflow = _replace_field(inflow, "HWindSpeed", f"{wind_speed_mps:.9g}")
    inflow = _replace_field(inflow, "PropagationDir", "0.0")
    inflow_path.write_text(inflow, encoding="utf-8")
    return fst_path


def _read_openfast_columns(output_path: Path) -> dict[str, np.ndarray]:
    columns, _ = _read_openfast_output(output_path)
    return columns


def _read_openfast_output(
    output_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    lines = output_path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index
        for index, line in enumerate(lines)
        if line.split() and line.split()[0] == "Time"
    )
    names = lines[header_index].split()
    units = lines[header_index + 1].split()
    if len(units) != len(names):
        raise ValueError(
            "OpenFAST output unit count does not match its channel header: "
            f"{len(units)} units for {len(names)} channels"
        )
    data = np.loadtxt(lines[header_index + 2 :])
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] != len(names):
        raise ValueError(
            "OpenFAST output column count does not match its channel header: "
            f"{data.shape[1]} values for {len(names)} channels"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("OpenFAST output contains non-finite numeric values")
    return (
        {name: data[:, index] for index, name in enumerate(names)},
        {name: units[index] for index, name in enumerate(names)},
    )


def _assert_output_reaches_time_horizon(
    columns: dict[str, np.ndarray], *, duration_s: float, output_step_s: float
) -> None:
    """Reject a partial OpenFAST output before it enters an audit summary."""
    time = columns.get("Time")
    if time is None or time.size == 0:
        raise ValueError("OpenFAST output has no Time samples")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("OpenFAST output Time samples must be strictly increasing")
    tolerance_s = max(1.0e-9, output_step_s + 1.0e-9)
    if abs(float(time[-1]) - duration_s) > tolerance_s:
        raise ValueError(
            "OpenFAST output does not reach the requested time horizon: "
            f"expected {duration_s:g} s, got {float(time[-1]):g} s"
        )


def _read_openfast_scalar_field(path: Path, field: str) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        tokens = line.split()
        if len(tokens) >= 2 and tokens[1] == field:
            try:
                return float(tokens[0])
            except ValueError as exc:
                raise ValueError(
                    f"OpenFAST field {field} in {path} is not numeric"
                ) from exc
    raise ValueError(f"missing numeric OpenFAST field {field} in {path}")


def _read_openfast_hub_geometry(ed_path: Path) -> dict[str, object]:
    tower_height_m = _read_openfast_scalar_field(ed_path, "TowerHt")
    tower_to_shaft_m = _read_openfast_scalar_field(ed_path, "Twr2Shft")
    overhang_m = _read_openfast_scalar_field(ed_path, "OverHang")
    shaft_tilt_deg = _read_openfast_scalar_field(ed_path, "ShftTilt")
    platform_reference_z_m = _read_openfast_scalar_field(ed_path, "PtfmRefzt")
    azimuth_blade1_up_deg = _read_openfast_scalar_field(ed_path, "AzimB1Up")
    shaft_tilt_rad = np.deg2rad(shaft_tilt_deg)
    hub_from_reference_m = np.array(
        [
            overhang_m * np.cos(shaft_tilt_rad),
            0.0,
            tower_height_m
            - platform_reference_z_m
            + tower_to_shaft_m
            + overhang_m * np.sin(shaft_tilt_rad),
        ],
        dtype=float,
    )
    return {
        "tower_height_m": tower_height_m,
        "tower_to_shaft_m": tower_to_shaft_m,
        "overhang_m": overhang_m,
        "shaft_tilt_deg": shaft_tilt_deg,
        "platform_reference_z_m": platform_reference_z_m,
        "azimuth_blade1_up_deg": azimuth_blade1_up_deg,
        "hub_from_platform_reference_m": hub_from_reference_m,
    }


def _hub_wrench_unit_scales(units: dict[str, str]) -> tuple[float, float]:
    force_channels = ("RtFldFxh", "RtFldFyh", "RtFldFzh")
    moment_channels = ("RtFldMxh", "RtFldMyh", "RtFldMzh")
    force_units = {units[name] for name in force_channels}
    moment_units = {units[name] for name in moment_channels}
    force_scales = {"(N)": 1.0, "(kN)": 1_000.0}
    moment_scales = {"(N-m)": 1.0, "(kN-m)": 1_000.0}
    if len(force_units) != 1 or not force_units <= force_scales.keys():
        raise ValueError(
            "unexpected AeroDyn hub-force units; expected '(N)' or '(kN)', got "
            f"{sorted(force_units)}"
        )
    if len(moment_units) != 1 or not moment_units <= moment_scales.keys():
        raise ValueError(
            "unexpected AeroDyn hub-moment units; expected '(N-m)' or '(kN-m)', got "
            f"{sorted(moment_units)}"
        )
    return force_scales[force_units.pop()], moment_scales[moment_units.pop()]


def _assert_hub_conversion_angle_units(units: dict[str, str]) -> None:
    required = ("Azimuth", "PtfmRoll", "PtfmPitch", "PtfmYaw", "NacYaw")
    missing = [name for name in required if name not in units]
    if missing:
        raise ValueError(
            "OpenFAST output is missing hub-conversion angle units: "
            f"{', '.join(missing)}"
        )
    unexpected = {name: units[name] for name in required if units[name] != "(deg)"}
    if unexpected:
        raise ValueError(
            "OpenFAST hub-conversion angles must use '(deg)', got "
            f"{unexpected}"
        )


def _mean_hub_load_about_reference(
    *,
    columns: dict[str, np.ndarray],
    units: dict[str, str],
    second_half: np.ndarray,
    hub_geometry: dict[str, object],
    equilibrium_platform_pitch_deg: float,
) -> np.ndarray:
    force_scale, moment_scale = _hub_wrench_unit_scales(units)
    _assert_hub_conversion_angle_units(units)
    hub_from_reference = np.asarray(
        hub_geometry["hub_from_platform_reference_m"], dtype=float
    )
    transformed = []
    for index in np.flatnonzero(second_half):
        transformed.append(
            generalized_load_from_openfast_hub_wrench(
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
                azimuth_blade1_up_deg=float(
                    hub_geometry["azimuth_blade1_up_deg"]
                ),
                equilibrium_platform_pitch_deg=equilibrium_platform_pitch_deg,
            )
        )
    if not transformed:
        raise ValueError("OpenFAST output has no samples in the second-half window")
    return np.mean(np.asarray(transformed), axis=0)


def _assert_prescribed_rotor_output(
    *,
    blade_pitch_deg: list[float] | list[np.ndarray],
    rotor_speed_rpm: float | np.ndarray,
    prescribed_blade_pitch_deg: float = PRESCRIBED_BLADE_PITCH_DEG,
    prescribed_rotor_speed_rpm: float = PRESCRIBED_ROTOR_SPEED_RPM,
) -> None:
    if not np.allclose(blade_pitch_deg, prescribed_blade_pitch_deg, atol=1e-6):
        raise ValueError(
            "OpenFAST output does not retain the prescribed blade pitch: "
            f"{blade_pitch_deg} deg"
        )
    if not np.allclose(rotor_speed_rpm, prescribed_rotor_speed_rpm, atol=1e-6):
        raise ValueError(
            "OpenFAST output does not retain the prescribed rotor speed: "
            f"{rotor_speed_rpm} rpm"
        )


def _run_case(
    *,
    case_root: Path,
    openfast_binary: Path,
    wind_speed_mps: float,
    duration_s: float,
    output_step_s: float,
    equilibrium_surge_m: float,
    equilibrium_heave_m: float,
    equilibrium_pitch_deg: float,
    prescribed_blade_pitch_deg: float = PRESCRIBED_BLADE_PITCH_DEG,
    prescribed_rotor_speed_rpm: float = PRESCRIBED_ROTOR_SPEED_RPM,
) -> dict[str, object]:
    semi_dir = _extract_clean_model(case_root)
    fst_path = _configure_case(
        semi_dir,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=equilibrium_surge_m,
        equilibrium_heave_m=equilibrium_heave_m,
        equilibrium_pitch_deg=equilibrium_pitch_deg,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
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
            "OpenFAST prescribed-wind node failed:\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    columns, units = _read_openfast_output(semi_dir / f"{fst_path.stem}.out")
    _assert_output_reaches_time_horizon(
        columns, duration_s=duration_s, output_step_s=output_step_s
    )
    required = (
        "Time",
        "PtfmSurge",
        "PtfmRoll",
        "PtfmHeave",
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
    second_half = columns["Time"] >= duration_s / 2.0
    blade_pitch_samples = [
        columns[name][second_half]
        for name in ("BldPitch1", "BldPitch2", "BldPitch3")
    ]
    rotor_speed_samples = columns["RotSpeed"][second_half]
    _assert_prescribed_rotor_output(
        blade_pitch_deg=blade_pitch_samples,
        rotor_speed_rpm=rotor_speed_samples,
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    blade_pitch_deg = [
        float(np.mean(columns[name][second_half]))
        for name in ("BldPitch1", "BldPitch2", "BldPitch3")
    ]
    rotor_speed_rpm = float(np.mean(columns["RotSpeed"][second_half]))
    hub_geometry = _read_openfast_hub_geometry(semi_dir / ED_NAME)
    reference_load = _mean_hub_load_about_reference(
        columns=columns,
        units=units,
        second_half=second_half,
        hub_geometry=hub_geometry,
        equilibrium_platform_pitch_deg=equilibrium_pitch_deg,
    )
    return {
        "wind_speed_mps": wind_speed_mps,
        "sample_count": int(columns["Time"].size),
        "second_half_window_s": [duration_s / 2.0, duration_s],
        "second_half_mean": {
            "surge_m": float(np.mean(columns["PtfmSurge"][second_half])),
            "heave_m": float(np.mean(columns["PtfmHeave"][second_half])),
            "pitch_deg": float(np.mean(columns["PtfmPitch"][second_half])),
            "blade_pitch_deg": blade_pitch_deg,
            "rotor_speed_rpm": rotor_speed_rpm,
        },
        "second_half_range": {
            "surge_m": [
                float(np.min(columns["PtfmSurge"][second_half])),
                float(np.max(columns["PtfmSurge"][second_half])),
            ],
            "pitch_deg": [
                float(np.min(columns["PtfmPitch"][second_half])),
                float(np.max(columns["PtfmPitch"][second_half])),
            ],
        },
        "aerodynamic_hub_load_about_platform_reference_second_half_mean": {
            "generalized_load_si": {
                "force_n": [float(value) for value in reference_load[:3]],
                "moment_nm": [float(value) for value in reference_load[3:]],
            },
            "source_channels": {
                "force": ["RtFldFxh", "RtFldFyh", "RtFldFzh"],
                "moment": ["RtFldMxh", "RtFldMyh", "RtFldMzh"],
                "force_units": units["RtFldFxh"],
                "moment_units": units["RtFldMxh"],
            },
            "conversion": {
                "method": "per-output-step OpenFAST h-to-reference rotation before averaging",
                "reference_frame": "frozen equilibrium platform axes",
                "hub_from_platform_reference_m": [
                    float(value)
                    for value in hub_geometry["hub_from_platform_reference_m"]
                ],
                "shaft_tilt_deg": float(hub_geometry["shaft_tilt_deg"]),
                "azimuth_blade1_up_deg": float(
                    hub_geometry["azimuth_blade1_up_deg"]
                ),
                "equilibrium_platform_pitch_deg": equilibrium_pitch_deg,
                "nacelle_yaw_requirement": "NacYaw must be zero",
            },
        },
        "input_tree": str(semi_dir),
    }


def run_audit(
    *,
    work_dir: Path,
    openfast_binary: Path,
    duration_s: float,
    output_step_s: float,
    wind_speed_mps: float,
    baseline_wind_speed_mps: float = 0.1,
    reference_state_path: Path = REFERENCE_STATE_JSON,
    prescribed_blade_pitch_deg: float = PRESCRIBED_BLADE_PITCH_DEG,
    prescribed_rotor_speed_rpm: float = PRESCRIBED_ROTOR_SPEED_RPM,
) -> dict[str, object]:
    if duration_s <= 0.0 or output_step_s <= 0.0:
        raise ValueError("duration_s and output_step_s must be positive")
    if baseline_wind_speed_mps <= 0.0 or wind_speed_mps <= baseline_wind_speed_mps:
        raise ValueError("wind_speed_mps must exceed the positive low-wind reference")
    if not np.isfinite(prescribed_blade_pitch_deg):
        raise ValueError("prescribed_blade_pitch_deg must be finite")
    if not np.isfinite(prescribed_rotor_speed_rpm) or prescribed_rotor_speed_rpm < 0.0:
        raise ValueError("prescribed_rotor_speed_rpm must be finite and non-negative")
    if work_dir.exists():
        raise FileExistsError(f"work_dir already exists: {work_dir}")
    if not openfast_binary.is_file():
        raise FileNotFoundError(f"OpenFAST binary not found: {openfast_binary}")

    equilibrium = _load_candidate_reference_state(reference_state_path)
    still = _run_case(
        case_root=work_dir / "low_wind_reference",
        openfast_binary=openfast_binary,
        wind_speed_mps=baseline_wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=equilibrium["surge_m"],
        equilibrium_heave_m=equilibrium["heave_m"],
        equilibrium_pitch_deg=equilibrium["pitch_deg"],
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    wind = _run_case(
        case_root=work_dir / "steady_wind",
        openfast_binary=openfast_binary,
        wind_speed_mps=wind_speed_mps,
        duration_s=duration_s,
        output_step_s=output_step_s,
        equilibrium_surge_m=equilibrium["surge_m"],
        equilibrium_heave_m=equilibrium["heave_m"],
        equilibrium_pitch_deg=equilibrium["pitch_deg"],
        prescribed_blade_pitch_deg=prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=prescribed_rotor_speed_rpm,
    )
    still_mean = still["second_half_mean"]
    wind_mean = wind["second_half_mean"]
    still_aero = still[
        "aerodynamic_hub_load_about_platform_reference_second_half_mean"
    ]["generalized_load_si"]
    wind_aero = wind[
        "aerodynamic_hub_load_about_platform_reference_second_half_mean"
    ]["generalized_load_si"]
    return {
        "evidence_level": "limited OpenFAST prescribed-rotor wind-response node",
        "boundaries": {
            "is_rosco_controlled_operating_case": False,
            "is_wave_response_validation": False,
            "is_physical_damping_calibration": False,
            "is_static_offset_comparison": False,
            "is_small_angle_quantitative_validation": False,
            "purpose": "wind-response direction and AeroDyn hub-load channel check",
            "low_wind_reference_mps": baseline_wind_speed_mps,
        },
        "reference": {
            "model_zip": str(MODEL_ZIP.relative_to(ROOT)),
            "model_zip_sha256": _sha256(MODEL_ZIP),
            "openfast_binary": str(openfast_binary),
            "openfast_binary_sha256": _sha256(openfast_binary),
            "reference_state_artifact": {
                "path": _display_path(reference_state_path),
                "sha256": _sha256(reference_state_path),
            },
            "candidate_reference_initialization": equilibrium,
            "other_platform_initial_conditions": "source-file defaults (zero)",
            "prescribed_rotor_speed_rpm": prescribed_rotor_speed_rpm,
            "prescribed_blade_pitch_deg": prescribed_blade_pitch_deg,
            "wave_and_current": "disabled",
        },
        "low_wind_reference": still,
        "steady_wind": wind,
        "wind_induced_second_half_mean_difference": {
            "surge_m": float(wind_mean["surge_m"] - still_mean["surge_m"]),
            "heave_m": float(wind_mean["heave_m"] - still_mean["heave_m"]),
            "pitch_deg": float(wind_mean["pitch_deg"] - still_mean["pitch_deg"]),
            "aerodynamic_load_about_platform_reference_force_n": [
                float(wind_value - still_value)
                for wind_value, still_value in zip(
                    wind_aero["force_n"], still_aero["force_n"], strict=True
                )
            ],
            "aerodynamic_load_about_platform_reference_moment_nm": [
                float(wind_value - still_value)
                for wind_value, still_value in zip(
                    wind_aero["moment_nm"], still_aero["moment_nm"], strict=True
                )
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--openfast",
        type=Path,
        default=Path("/tmp/fowt-openfast-4.1.2/bin/openfast"),
        help=(
            "OpenFAST v4.1.2 is the compatible solver retained from the P2 "
            "reference-state audit; the local v5 binary rejects this v1.1.16 "
            "primary-input syntax before simulation."
        ),
    )
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--output-step-s", type=float, default=0.5)
    parser.add_argument("--wind-speed-mps", type=float, default=5.0)
    parser.add_argument("--baseline-wind-speed-mps", type=float, default=0.1)
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
    parser.add_argument(
        "--reference-state-json",
        type=Path,
        default=REFERENCE_STATE_JSON,
        help="audited candidate initial state, bound to the frozen model archive by SHA-256",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    result = run_audit(
        work_dir=args.work_dir,
        openfast_binary=args.openfast,
        duration_s=args.duration_s,
        output_step_s=args.output_step_s,
        wind_speed_mps=args.wind_speed_mps,
        baseline_wind_speed_mps=args.baseline_wind_speed_mps,
        reference_state_path=args.reference_state_json,
        prescribed_blade_pitch_deg=args.prescribed_blade_pitch_deg,
        prescribed_rotor_speed_rpm=args.prescribed_rotor_speed_rpm,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
