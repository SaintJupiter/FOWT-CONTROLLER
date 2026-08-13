from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = ROOT / "archive" / "legacy_fowt_control"
SRC_DIR = ROOT / "src"
for path in (LEGACY_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core_model import FloatingPlatform  # noqa: E402
from defaults import DEFAULT_PLATFORM_PROFILES  # noqa: E402
from wind_env import load_thrust_curve  # noqa: E402
from wind_prediction.input_files import resolve_mooring_stiffness_file  # noqa: E402


DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DEFAULT_THRUST_CURVE = (
    LEGACY_DIR / "data" / "Thrust force at hub height 风机叶片及轮毂相关数据.xlsx"
)


@dataclass(frozen=True)
class AuditSettings:
    dt_s: float = 0.1
    settle_duration_s: float = 600.0
    decay_duration_s: float = 180.0
    decay_angle_deg: float = 2.0
    audit_wind_speed_ms: float = 12.0
    sample_interval_s: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "dt_s": self.dt_s,
            "settle_duration_s": self.settle_duration_s,
            "decay_duration_s": self.decay_duration_s,
            "decay_angle_deg": self.decay_angle_deg,
            "sample_interval_s": self.sample_interval_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(self.audit_wind_speed_ms) or self.audit_wind_speed_ms < 0.0:
            raise ValueError("audit_wind_speed_ms must be finite and non-negative")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError(f"cannot write empty audit table: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_hashes(paths: Iterable[Path]) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted({item.resolve() for item in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            name = str(path.relative_to(ROOT))
        except ValueError:
            name = str(path)
        records[name] = {
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        }
    return records


def _make_plant(stiffness_file: Path, platform_profile: str) -> FloatingPlatform:
    plant = FloatingPlatform(
        stiffness_file,
        platform_profile=platform_profile,
        allow_linear_mooring_fallback=False,
        platform_profile_purpose="audit",
    )
    plant.wave_components = []
    return plant


def _load_row(
    *,
    scenario: str,
    component: str,
    values: np.ndarray,
) -> dict[str, object]:
    row: dict[str, object] = {
        "scenario": scenario,
        "component": component,
    }
    row.update({name: float(values[index]) for index, name in enumerate(DOF_NAMES)})
    return row


def _scan_load_channels(
    plant: FloatingPlatform,
    incremental_reference_state: np.ndarray,
    thrust_n: float,
) -> tuple[list[dict[str, object]], dict[str, bool]]:
    scenarios: list[tuple[str, np.ndarray, float, float]] = []
    for direction in (0.0, 90.0, 180.0, 270.0):
        scenarios.append(
            (
                f"wind_{int(direction)}deg",
                incremental_reference_state.copy(),
                thrust_n,
                direction,
            )
        )
    perturbations = {
        "surge_plus_5m": (0, 5.0),
        "surge_minus_5m": (0, -5.0),
        "sway_plus_5m": (1, 5.0),
        "sway_minus_5m": (1, -5.0),
        "heave_plus_1m": (2, 1.0),
        "heave_minus_1m": (2, -1.0),
        "roll_plus_1deg": (3, math.radians(1.0)),
        "roll_minus_1deg": (3, math.radians(-1.0)),
        "pitch_plus_1deg": (4, math.radians(1.0)),
        "pitch_minus_1deg": (4, math.radians(-1.0)),
        "yaw_plus_1deg": (5, math.radians(1.0)),
        "yaw_minus_1deg": (5, math.radians(-1.0)),
    }
    for name, (index, delta) in perturbations.items():
        state = incremental_reference_state.copy()
        state[index] += delta
        scenarios.append((name, state, 0.0, 0.0))

    rows: list[dict[str, object]] = []
    for scenario, state, scenario_thrust, direction in scenarios:
        loads = plant.generalized_load_components(
            0.0,
            state,
            scenario_thrust,
            direction,
        )
        for component, values in loads.items():
            rows.append(
                _load_row(
                    scenario=scenario,
                    component=component,
                    values=values,
                )
            )

    checks = {
        "wind_cardinal_direction_signs": True,
        "mooring_restores_surge_sway": True,
        "hydrostatic_restores_heave_roll_pitch": True,
        "yaw_restoring_opposes_yaw": True,
        "load_components_sum_to_total": True,
    }
    tolerance = max(abs(thrust_n), 1.0) * 1e-9
    cardinal_expected = {
        0.0: np.array([thrust_n, 0.0]),
        90.0: np.array([0.0, thrust_n]),
        180.0: np.array([-thrust_n, 0.0]),
        270.0: np.array([0.0, -thrust_n]),
    }
    for direction, expected in cardinal_expected.items():
        wind = plant.generalized_load_components(
            0.0,
            incremental_reference_state,
            thrust_n,
            direction,
        )["wind"]
        checks["wind_cardinal_direction_signs"] &= bool(
            np.allclose(wind[:2], expected, atol=tolerance, rtol=0.0)
        )

    base_loads = plant.generalized_load_components(
        0.0,
        incremental_reference_state,
        0.0,
        0.0,
    )
    for index in (0, 1):
        for delta in (-5.0, 5.0):
            state = incremental_reference_state.copy()
            state[index] += delta
            mooring = plant.generalized_load_components(0.0, state, 0.0, 0.0)[
                "mooring"
            ]
            incremental_load = mooring[index] - base_loads["mooring"][index]
            checks["mooring_restores_surge_sway"] &= bool(
                incremental_load * delta < 0.0
            )
    for index, delta in (
        (2, 1.0),
        (3, math.radians(1.0)),
        (4, math.radians(1.0)),
    ):
        for sign in (-1.0, 1.0):
            state = incremental_reference_state.copy()
            state[index] += sign * delta
            hydro = plant.generalized_load_components(0.0, state, 0.0, 0.0)[
                "hydrostatic"
            ]
            incremental_load = hydro[index] - base_loads["hydrostatic"][index]
            checks["hydrostatic_restores_heave_roll_pitch"] &= bool(
                incremental_load * sign * delta < 0.0
            )
    yaw_delta = math.radians(1.0)
    for sign in (-1.0, 1.0):
        state = incremental_reference_state.copy()
        state[5] += sign * yaw_delta
        yaw_load = plant.generalized_load_components(0.0, state, 0.0, 0.0)[
            "yaw_restoring"
        ]
        incremental_load = yaw_load[5] - base_loads["yaw_restoring"][5]
        checks["yaw_restoring_opposes_yaw"] &= bool(
            incremental_load * sign * yaw_delta < 0.0
        )

    sample_state = incremental_reference_state.copy()
    sample_state[0:6] += np.array([2.0, -3.0, 0.5, 0.02, -0.03, 0.01])
    sample_state[6:12] = np.array([0.3, -0.2, 0.1, 0.01, -0.02, 0.005])
    sample_loads = plant.generalized_load_components(
        7.0,
        sample_state,
        thrust_n,
        35.0,
    )
    recombined = sum(
        values for name, values in sample_loads.items() if name != "total"
    )
    checks["load_components_sum_to_total"] = bool(
        np.allclose(sample_loads["total"], recombined, rtol=1e-12, atol=1e-6)
    )
    return rows, checks


def _scan_ballast_increment_chain(
    *,
    stiffness_file: Path,
    platform_profile: str,
) -> tuple[list[dict[str, object]], dict[str, bool], bool]:
    reference_plant = _make_plant(stiffness_file, platform_profile)
    if reference_plant.load_reference_mode != "reference_incremental":
        return [], {}, False

    increment_kg = 1000.0
    scenarios = {
        "tank1_inflow": np.array([increment_kg, 0.0, 0.0]),
        "tank1_outflow": np.array([-increment_kg, 0.0, 0.0]),
        "tank1_outflow_tank2_inflow": np.array(
            [-increment_kg, increment_kg, 0.0]
        ),
        "common_inflow": np.array([increment_kg, increment_kg, increment_kg]),
    }
    rows: list[dict[str, object]] = []
    loads_by_name: dict[str, np.ndarray] = {}
    checks = {
        "ballast_signed_mass_and_first_moment": True,
        "ballast_signed_inertia": True,
        "ballast_incremental_gravity_load": True,
        "ballast_fixed_added_mass_is_not_rescaled": True,
        "ballast_inflow_outflow_are_opposite": True,
        "ballast_balanced_exchange_preserves_total_mass": True,
        "ballast_balanced_exchange_moment_direction": True,
    }

    for scenario, deltas in scenarios.items():
        plant = _make_plant(stiffness_file, platform_profile)
        plant.force_ballast_mass(*(plant.reference_ballast_mass + deltas))
        loads = plant.generalized_load_components(
            0.0,
            np.zeros(12, dtype=float),
            0.0,
            0.0,
        )
        ballast_load = np.asarray(loads["ballast_gravity"], dtype=float)
        loads_by_name[scenario] = ballast_load

        reference = plant.reference_mass_properties
        expected_mass = reference.total_mass_kg + float(np.sum(deltas))
        expected_first_moment = (
            reference.total_mass_kg * reference.center_of_mass_m
            + np.sum(deltas[:, None] * plant.tank_pos, axis=0)
        )
        expected_center = expected_first_moment / expected_mass
        radius_squared = np.einsum("ij,ij->i", plant.tank_pos, plant.tank_pos)
        expected_inertia = reference.inertia_about_reference_kg_m2 + np.sum(
            deltas[:, None, None]
            * (
                radius_squared[:, None, None] * np.eye(3)
                - plant.tank_pos[:, :, None] * plant.tank_pos[:, None, :]
            ),
            axis=0,
        )
        expected_gravity = np.zeros(6, dtype=float)
        expected_gravity[2] = -float(np.sum(deltas)) * plant.g
        for delta_mass, tank_position in zip(deltas, plant.tank_pos):
            expected_gravity[3:6] += np.cross(
                tank_position,
                [0.0, 0.0, -float(delta_mass) * plant.g],
            )
        rigid_mass = plant._rigid_body_mass_matrix(plant.mass_properties)

        checks["ballast_signed_mass_and_first_moment"] &= bool(
            math.isclose(
                plant.mass_properties.total_mass_kg,
                expected_mass,
                abs_tol=1e-9,
            )
            and np.allclose(
                plant.mass_properties.center_of_mass_m,
                expected_center,
                rtol=1e-12,
                atol=1e-12,
            )
        )
        checks["ballast_signed_inertia"] &= bool(
            np.allclose(
                plant.mass_properties.inertia_about_reference_kg_m2,
                expected_inertia,
                rtol=1e-12,
                atol=1e-6,
            )
        )
        checks["ballast_incremental_gravity_load"] &= bool(
            np.allclose(ballast_load, expected_gravity, rtol=1e-12, atol=1e-6)
        )
        checks["ballast_fixed_added_mass_is_not_rescaled"] &= bool(
            np.allclose(
                plant.M_total - rigid_mass,
                plant.reference_added_mass_matrix,
                rtol=1e-12,
                atol=1e-6,
            )
        )
        rows.append(
            {
                "scenario": scenario,
                "tank1_delta_kg": float(deltas[0]),
                "tank2_delta_kg": float(deltas[1]),
                "tank3_delta_kg": float(deltas[2]),
                "platform_total_mass_kg": float(
                    plant.mass_properties.total_mass_kg
                ),
                "center_x_m": float(plant.mass_properties.center_of_mass_m[0]),
                "center_y_m": float(plant.mass_properties.center_of_mass_m[1]),
                "center_z_m": float(plant.mass_properties.center_of_mass_m[2]),
                "gravity_heave_n": float(ballast_load[2]),
                "gravity_roll_nm": float(ballast_load[3]),
                "gravity_pitch_nm": float(ballast_load[4]),
                "inertia_about_reference_kg_m2": json.dumps(
                    plant.mass_properties.inertia_about_reference_kg_m2.tolist(),
                    separators=(",", ":"),
                ),
                "rigid_body_mass_matrix": json.dumps(
                    rigid_mass.tolist(),
                    separators=(",", ":"),
                ),
                "fixed_added_mass_matrix": json.dumps(
                    plant.reference_added_mass_matrix.tolist(),
                    separators=(",", ":"),
                ),
                "effective_mass_matrix": json.dumps(
                    plant.M_total.tolist(),
                    separators=(",", ":"),
                ),
            }
        )

    checks["ballast_inflow_outflow_are_opposite"] = bool(
        np.allclose(
            loads_by_name["tank1_inflow"],
            -loads_by_name["tank1_outflow"],
            rtol=1e-12,
            atol=1e-6,
        )
    )
    exchange_name = "tank1_outflow_tank2_inflow"
    exchange_row = next(
        row for row in rows if row["scenario"] == exchange_name
    )
    checks["ballast_balanced_exchange_preserves_total_mass"] = bool(
        math.isclose(
            exchange_row["platform_total_mass_kg"],
            reference_plant.reference_total_mass,
            abs_tol=1e-9,
        )
        and abs(float(exchange_row["gravity_heave_n"])) <= 1e-9
    )
    expected_exchange_moment = np.zeros(3, dtype=float)
    for delta_mass, tank_position in zip(
        scenarios[exchange_name],
        reference_plant.tank_pos,
    ):
        expected_exchange_moment += np.cross(
            tank_position,
            [0.0, 0.0, -float(delta_mass) * reference_plant.g],
        )
    checks["ballast_balanced_exchange_moment_direction"] = bool(
        np.allclose(
            loads_by_name[exchange_name][3:6],
            expected_exchange_moment,
            rtol=1e-12,
            atol=1e-6,
        )
    )
    return rows, checks, True


def _sample_state(time_s: float, plant: FloatingPlatform) -> dict[str, object]:
    state = np.asarray(plant.state, dtype=float)
    return {
        "time_s": float(time_s),
        "surge_m": float(state[0]),
        "sway_m": float(state[1]),
        "heave_m": float(state[2]),
        "roll_deg": float(np.degrees(state[3])),
        "pitch_deg": float(np.degrees(state[4])),
        "yaw_deg": float(np.degrees(state[5])),
        "surge_velocity_m_s": float(state[6]),
        "sway_velocity_m_s": float(state[7]),
        "heave_velocity_m_s": float(state[8]),
        "roll_rate_deg_s": float(np.degrees(state[9])),
        "pitch_rate_deg_s": float(np.degrees(state[10])),
        "yaw_rate_deg_s": float(np.degrees(state[11])),
    }


def _run_incremental_reference_settle(
    plant: FloatingPlatform,
    settings: AuditSettings,
) -> list[dict[str, object]]:
    steps = int(round(settings.settle_duration_s / settings.dt_s))
    sample_stride = max(1, int(round(settings.sample_interval_s / settings.dt_s)))
    rows = [_sample_state(0.0, plant)]
    for index in range(steps):
        time_s = index * settings.dt_s
        plant.step(0.0, 0.0, settings.dt_s, time_s)
        if (index + 1) % sample_stride == 0 or index + 1 == steps:
            rows.append(_sample_state((index + 1) * settings.dt_s, plant))
    return rows


def _run_free_decay(
    *,
    plant: FloatingPlatform,
    incremental_reference_state: np.ndarray,
    axis_index: int,
    settings: AuditSettings,
) -> list[dict[str, object]]:
    plant.state = np.asarray(incremental_reference_state, dtype=float).copy()
    plant.state[axis_index] += math.radians(settings.decay_angle_deg)
    plant.state[6:12] = 0.0
    steps = int(round(settings.decay_duration_s / settings.dt_s))
    sample_stride = max(1, int(round(settings.sample_interval_s / settings.dt_s)))
    rows = [_sample_state(0.0, plant)]
    for index in range(steps):
        time_s = index * settings.dt_s
        plant.step(0.0, 0.0, settings.dt_s, time_s)
        if (index + 1) % sample_stride == 0 or index + 1 == steps:
            rows.append(_sample_state((index + 1) * settings.dt_s, plant))
    return rows


def _estimate_period_s(rows: list[dict[str, object]], key: str) -> float | None:
    values = np.asarray([row[key] for row in rows], dtype=float)
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    centered = values - float(values[-1])
    peaks = np.flatnonzero(
        (centered[1:-1] > centered[:-2]) & (centered[1:-1] >= centered[2:])
    ) + 1
    if peaks.size < 2:
        return None
    return float(np.median(np.diff(times[peaks])))


def run_audit(
    *,
    output_dir: Path,
    stiffness_file: Path,
    thrust_curve_file: Path,
    platform_profile: str,
    settings: AuditSettings,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thrust_curve = load_thrust_curve(thrust_curve_file)
    if thrust_curve is None:
        raise FileNotFoundError(f"thrust curve not found: {thrust_curve_file}")
    audit_thrust_n = float(thrust_curve.thrust(settings.audit_wind_speed_ms))

    settle_plant = _make_plant(stiffness_file, platform_profile)
    settle_rows = _run_incremental_reference_settle(settle_plant, settings)
    incremental_reference_state = settle_plant.state.copy()
    incremental_reference_state[6:12] = 0.0
    incremental_reference_loads = settle_plant.generalized_load_components(
        settings.settle_duration_s,
        incremental_reference_state,
        0.0,
        0.0,
    )

    load_rows, directional_checks = _scan_load_channels(
        settle_plant,
        incremental_reference_state,
        audit_thrust_n,
    )
    ballast_rows, ballast_checks, ballast_audit_applicable = (
        _scan_ballast_increment_chain(
            stiffness_file=stiffness_file,
            platform_profile=platform_profile,
        )
    )
    directional_checks.update(ballast_checks)
    pitch_rows = _run_free_decay(
        plant=_make_plant(stiffness_file, platform_profile),
        incremental_reference_state=incremental_reference_state,
        axis_index=4,
        settings=settings,
    )
    roll_rows = _run_free_decay(
        plant=_make_plant(stiffness_file, platform_profile),
        incremental_reference_state=incremental_reference_state,
        axis_index=3,
        settings=settings,
    )

    _write_csv(output_dir / "load_channel_scan.csv", load_rows)
    if ballast_rows:
        _write_csv(output_dir / "ballast_increment_scan.csv", ballast_rows)
    _write_csv(output_dir / "zero_load_settle.csv", settle_rows)
    _write_csv(output_dir / "free_decay_pitch.csv", pitch_rows)
    _write_csv(output_dir / "free_decay_roll.csv", roll_rows)

    resolved_platform = settle_plant.resolved_platform_identity()
    input_artifacts = _artifact_hashes((stiffness_file, thrust_curve_file))
    runtime_sources = _artifact_hashes(
        (
            Path(__file__),
            LEGACY_DIR / "core_model.py",
            LEGACY_DIR / "defaults.py",
            LEGACY_DIR / "wind_env.py",
            SRC_DIR / "wind_prediction" / "ballast_mass_properties.py",
            SRC_DIR / "wind_prediction" / "input_files.py",
        )
    )
    generated_output_paths = [
        output_dir / "load_channel_scan.csv",
        output_dir / "zero_load_settle.csv",
        output_dir / "free_decay_pitch.csv",
        output_dir / "free_decay_roll.csv",
    ]
    if ballast_rows:
        generated_output_paths.append(output_dir / "ballast_increment_scan.csv")
    output_artifacts = _artifact_hashes(generated_output_paths)

    final_velocity = np.asarray(settle_plant.state[6:12], dtype=float)
    summary = {
        "evidence_level": (
            "internal_mathematical_consistency_not_physical_validation"
        ),
        "absolute_static_equilibrium_solved": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform_profile": platform_profile,
        "resolved_platform": resolved_platform,
        "resolved_platform_sha256": _json_sha256(resolved_platform),
        "stiffness_file": str(stiffness_file.resolve()),
        "mooring_mode": settle_plant.mooring_mode,
        "mooring_reference_mode": settle_plant.mooring_reference_mode,
        "mooring_reference_force_raw_n": settle_plant.mooring_reference_force_raw,
        "mooring_source_path": settle_plant.mooring_source_path,
        "thrust_curve_file": str(thrust_curve_file.resolve()),
        "thrust_curve_source": thrust_curve.source,
        "settings": settings.__dict__,
        "audit_wind_thrust_n": audit_thrust_n,
        "incremental_reference_state": incremental_reference_state,
        "incremental_reference_total_load": incremental_reference_loads["total"],
        "incremental_reference_force_norm_n": float(
            np.linalg.norm(incremental_reference_loads["total"][:3])
        ),
        "incremental_reference_moment_norm_nm": float(
            np.linalg.norm(incremental_reference_loads["total"][3:])
        ),
        "settle_final_velocity_norm": float(np.linalg.norm(final_velocity)),
        "pitch_free_decay_period_s": _estimate_period_s(pitch_rows, "pitch_deg"),
        "roll_free_decay_period_s": _estimate_period_s(roll_rows, "roll_deg"),
        "checks": directional_checks,
        "ballast_increment_audit_applicable": ballast_audit_applicable,
        "known_model_scope": {
            "wind": "hub-height thrust curve projected to surge/sway and roll/pitch moments",
            "mooring": (
                "one horizontal force-displacement curve reused under horizontal "
                "symmetry; the research profile removes its zero-displacement offset"
            ),
            "waves": "disabled in this baseline audit; empirical JONSWAP/RAO path remains separate",
            "mass_properties": (
                "signed tank increments update total mass, center of mass and "
                "inertia; the research profile combines a coupled rigid-body "
                "matrix with fixed low-order added mass"
            ),
            "ballast_exchange": (
                "each tank exchanges water independently with the sea; equal and "
                "opposite tank changes are a balanced two-port exchange, not an "
                "internal tank-to-tank transfer"
            ),
            "deferred_dynamics": (
                "the current small-angle framework does not yet include exact "
                "attitude-dependent gravity generalized forces or variable-mass "
                "momentum-flux terms"
            ),
        },
        "provenance": {
            "input_artifacts": input_artifacts,
            "input_bundle_sha256": _json_sha256(input_artifacts),
            "runtime_sources": runtime_sources,
            "runtime_source_bundle_sha256": _json_sha256(runtime_sources),
            "output_artifacts": output_artifacts,
            "output_bundle_sha256": _json_sha256(output_artifacts),
        },
    }
    _write_json(output_dir / "audit_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit open-loop loads and the static/dynamic baseline of the 6-DOF plant.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stiffness-file", type=Path, default=None)
    parser.add_argument("--thrust-curve-file", type=Path, default=DEFAULT_THRUST_CURVE)
    parser.add_argument(
        "--platform-profile",
        choices=sorted(DEFAULT_PLATFORM_PROFILES),
        required=True,
    )
    parser.add_argument("--dt-s", type=float, default=0.1)
    parser.add_argument("--settle-duration-s", type=float, default=600.0)
    parser.add_argument("--decay-duration-s", type=float, default=180.0)
    parser.add_argument("--decay-angle-deg", type=float, default=2.0)
    parser.add_argument("--audit-wind-speed-ms", type=float, default=12.0)
    parser.add_argument("--sample-interval-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stiffness_file = resolve_mooring_stiffness_file(
        explicit_path=args.stiffness_file,
        search_dirs=(LEGACY_DIR / "data",),
    )
    settings = AuditSettings(
        dt_s=args.dt_s,
        settle_duration_s=args.settle_duration_s,
        decay_duration_s=args.decay_duration_s,
        decay_angle_deg=args.decay_angle_deg,
        audit_wind_speed_ms=args.audit_wind_speed_ms,
        sample_interval_s=args.sample_interval_s,
    )
    summary = run_audit(
        output_dir=args.output_dir,
        stiffness_file=stiffness_file,
        thrust_curve_file=args.thrust_curve_file,
        platform_profile=args.platform_profile,
        settings=settings,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value))


if __name__ == "__main__":
    main()
