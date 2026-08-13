"""Minimal end-to-end runner for the compact controller and a plant model."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .controller_plant_adapter import CompactControllerPlantAdapter


class BallastPlant(Protocol):
    state: np.ndarray
    current_ballast_mass: np.ndarray
    target_ballast_mass: np.ndarray
    tank_capacity: float

    def resolved_platform_identity(self) -> Mapping[str, Any]: ...

    def set_ballast_target(self, *masses_kg: float) -> None: ...

    def step(
        self,
        thrust_n: float,
        wind_direction_deg: float,
        dt_s: float,
        current_time_s: float,
    ) -> tuple[np.ndarray, Mapping[str, Any]]: ...


ThrustModel = Callable[[float], float]


@dataclass(frozen=True)
class ChainRunSummary:
    duration_s: float
    step_count: int
    decision_count: int
    forecast_decision_count: int
    target_revision_count: int
    pump_volume_m3: float
    pump_switch_count: int
    max_abs_pitch_deg: float
    max_abs_roll_deg: float
    max_command_transfer_error_kg: float
    min_tank_mass_kg: float
    max_tank_mass_kg: float
    max_abs_tank_mass_change_kg: float
    platform_mass_telemetry_available: bool
    initial_platform_total_mass_kg: float | None
    final_platform_total_mass_kg: float | None
    max_abs_platform_mass_change_kg: float | None
    max_platform_mass_balance_error_kg: float | None
    max_platform_center_of_mass_shift_m: float | None
    max_abs_platform_inertia_change_kg_m2: float | None
    finite_state: bool
    capacity_respected: bool
    completed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChainRunResult:
    summary: ChainRunSummary
    sampled_timeseries: tuple[dict[str, Any], ...]
    decisions: tuple[dict[str, Any], ...]


def _wind_arrays(wind_trace: Mapping[str, Any], step_count: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        speed = np.asarray(wind_trace["ws"], dtype=float).reshape(-1)
        direction = np.asarray(wind_trace["wd"], dtype=float).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("wind_trace must contain numeric ws and wd arrays") from exc
    if speed.size < step_count or direction.size < step_count:
        raise ValueError("wind_trace is shorter than the requested chain run")
    if not np.all(np.isfinite(speed[:step_count])) or np.any(speed[:step_count] < 0.0):
        raise ValueError("wind_trace speed must be finite and non-negative")
    if not np.all(np.isfinite(direction[:step_count])):
        raise ValueError("wind_trace direction must be finite")
    return speed, direction


def _optional_array(
    values: Mapping[str, Any],
    key: str,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if key not in values:
        return None
    array = np.asarray(values[key], dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise RuntimeError(f"plant feedback field {key!r} must have shape {shape}")
    return array


def _initial_mass_telemetry(
    plant: BallastPlant,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    properties = getattr(plant, "mass_properties", None)
    if properties is None:
        return None, None, None
    total_mass = float(properties.total_mass_kg)
    center = np.asarray(properties.center_of_mass_m, dtype=float)
    inertia = np.asarray(properties.inertia_about_reference_kg_m2, dtype=float)
    if not math.isfinite(total_mass) or center.shape != (3,) or inertia.shape != (3, 3):
        raise RuntimeError("plant exposes invalid initial mass properties")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(inertia)):
        raise RuntimeError("plant exposes non-finite initial mass properties")
    return total_mass, center.copy(), inertia.copy()


def _validated_platform_identity(plant: BallastPlant) -> Mapping[str, Any]:
    identity_source = getattr(plant, "resolved_platform_identity", None)
    if not callable(identity_source):
        raise ValueError(
            "control chain plants must expose a resolved platform identity"
        )
    identity = identity_source()
    if not isinstance(identity, Mapping):
        raise ValueError("resolved platform identity must be a mapping")
    if identity.get("schema_version") != "floating_platform_identity.v1":
        raise ValueError(
            "resolved platform identity must use floating_platform_identity.v1"
        )
    profile = identity.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("resolved platform identity must contain profile metadata")
    for field in ("requested_name", "base_name", "status", "purpose"):
        if not isinstance(profile.get(field), str) or not str(profile[field]).strip():
            raise ValueError(
                f"resolved platform identity profile must contain {field!r}"
            )
    status = str(profile.get("status", "")).strip()
    purpose = str(profile.get("purpose", "")).strip()
    if status == "audit_only_not_for_control_validation" or purpose == "audit":
        raise ValueError("audit-only platform models cannot enter a control chain run")
    allowed_pairs = {
        ("runtime", "control"),
        ("framework_only_not_for_performance_validation", "framework_smoke"),
    }
    if (status, purpose) not in allowed_pairs:
        raise ValueError(
            "platform status and purpose are not permitted for a control chain run"
        )
    for section in (
        "model_modes",
        "constants",
        "structure",
        "ballast_system",
        "hydrodynamics",
        "mooring",
        "reference_mass_properties",
    ):
        content = identity.get(section)
        if not isinstance(content, Mapping) or not content:
            raise ValueError(
                f"resolved platform identity must contain non-empty {section!r}"
            )
    return identity


def _validated_pump_schedule(values: Any, *, source: str) -> tuple[tuple[float, float], ...]:
    try:
        schedule = tuple(
            sorted((float(error), float(rate)) for error, rate in values)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} pump schedule is invalid") from exc
    if not schedule or not all(
        math.isfinite(value) for pair in schedule for value in pair
    ):
        raise ValueError(f"{source} pump schedule is invalid")
    return schedule


def _validate_execution_plant_compatibility(
    identity: Mapping[str, Any],
    controller: CompactControllerPlantAdapter,
) -> None:
    """Reject connected runs whose actuator assumptions do not match the plant."""

    execution = controller.controller.config.execution
    constants = identity["constants"]
    ballast = identity["ballast_system"]
    scalar_pairs = {
        "seawater_density_kg_m3": (
            execution.water_density_kg_m3,
            constants.get("seawater_density_kg_m3"),
        ),
        "tank_capacity_kg": (
            execution.tank_capacity_kg,
            ballast.get("tank_capacity_kg"),
        ),
        "pump_stop_error_kg": (
            execution.stop_error_kg,
            ballast.get("pump_stop_error_kg"),
        ),
        "pump_restart_error_kg": (
            execution.restart_error_kg,
            ballast.get("pump_restart_error_kg"),
        ),
        "pump_minimum_on_s": (
            execution.min_on_s,
            ballast.get("pump_minimum_on_s"),
        ),
        "pump_minimum_off_s": (
            execution.min_off_s,
            ballast.get("pump_minimum_off_s"),
        ),
        "pump_hold_before_stop_s": (
            execution.near_target_hold_s,
            ballast.get("pump_hold_before_stop_s"),
        ),
        "pump_ramp_up_m3_min_per_s": (
            execution.ramp_up_m3_min_per_s,
            ballast.get("pump_ramp_up_m3_min_per_s"),
        ),
        "pump_ramp_down_m3_min_per_s": (
            execution.ramp_down_m3_min_per_s,
            ballast.get("pump_ramp_down_m3_min_per_s"),
        ),
    }
    for name, (controller_value, plant_value) in scalar_pairs.items():
        try:
            plant_number = float(plant_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"resolved platform identity does not define numeric {name!r}"
            ) from exc
        if not math.isclose(
            float(controller_value),
            plant_number,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"controller/plant execution mismatch for {name}: "
                f"controller={float(controller_value)!r}, plant={plant_number!r}"
            )

    controller_schedule = _validated_pump_schedule(
        execution.pump_rate_schedule_m3_min,
        source="controller",
    )
    plant_schedule = _validated_pump_schedule(
        ballast.get("pump_rate_schedule_m3_min"),
        source="platform identity",
    )
    if controller_schedule != plant_schedule:
        raise ValueError("controller/plant execution mismatch for pump rate schedule")
    plant_max_rate = max(rate for _, rate in plant_schedule)
    if not math.isclose(
        float(execution.max_pump_rate_m3_min),
        plant_max_rate,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "controller/plant execution mismatch for maximum pump rate: "
            f"controller={float(execution.max_pump_rate_m3_min)!r}, "
            f"plant={plant_max_rate!r}"
        )


def run_controller_plant_chain(
    *,
    plant: BallastPlant,
    controller: CompactControllerPlantAdapter,
    wind_trace: Mapping[str, Any],
    thrust_model: ThrustModel,
    duration_s: float,
    dt_s: float,
    sample_interval_s: float = 10.0,
) -> ChainRunResult:
    """Run one chain with measured plant feedback as the only persistent state."""

    platform_identity = _validated_platform_identity(plant)
    _validate_execution_plant_compatibility(platform_identity, controller)

    duration_s = float(duration_s)
    dt_s = float(dt_s)
    sample_interval_s = float(sample_interval_s)
    if duration_s <= 0.0 or dt_s <= 0.0 or sample_interval_s <= 0.0:
        raise ValueError("duration_s, dt_s and sample_interval_s must be positive")
    step_count = int(round(duration_s / dt_s))
    if not math.isclose(step_count * dt_s, duration_s, abs_tol=1e-9):
        raise ValueError("duration_s must be an integer multiple of dt_s")
    sample_stride = max(1, int(round(sample_interval_s / dt_s)))
    wind_speed, wind_direction = _wind_arrays(wind_trace, step_count)

    initial_masses = (
        np.asarray(plant.current_ballast_mass, dtype=float).reshape(-1).copy()
    )
    if initial_masses.size != 3 or not np.all(np.isfinite(initial_masses)):
        raise ValueError("plant must expose exactly three finite ballast masses")
    last_info: dict[str, Any] = {
        "tank_masses": initial_masses.copy(),
        "target_ballast_mass": initial_masses.copy(),
        "primary_target_kg": initial_masses.copy(),
        "pump_rate_smoothed_m3_min": np.zeros(3, dtype=float),
        "pump_latched": np.zeros(3, dtype=bool),
    }
    previous_latched = np.zeros(3, dtype=bool)
    sampled: list[dict[str, Any]] = []
    pump_volume_m3 = 0.0
    pump_switch_count = 0
    max_pitch = 0.0
    max_roll = 0.0
    max_transfer_error = 0.0
    min_mass = float(np.min(initial_masses))
    max_mass = float(np.max(initial_masses))
    max_tank_mass_change = 0.0
    initial_platform_mass, initial_platform_center, initial_platform_inertia = (
        _initial_mass_telemetry(plant)
    )
    final_platform_mass = initial_platform_mass
    max_platform_mass_change = 0.0 if initial_platform_mass is not None else None
    max_platform_mass_balance_error = 0.0 if initial_platform_mass is not None else None
    max_platform_center_shift = 0.0 if initial_platform_center is not None else None
    max_platform_inertia_change = 0.0 if initial_platform_inertia is not None else None
    mass_telemetry_available = initial_platform_mass is not None
    finite_state = True

    for index in range(step_count):
        time_s = index * dt_s
        ws = float(wind_speed[index])
        wd = float(wind_direction[index])
        wind_obs = {"ws": ws, "wd_deg": wd}
        command = controller.compute(
            state=plant.state,
            wind_obs=wind_obs,
            plant_info_prev=last_info,
            current_time=time_s,
        )
        target = np.asarray(command["preview_primary_target_kg"], dtype=float).reshape(-1)
        if target.size != 3 or not np.all(np.isfinite(target)):
            raise RuntimeError("controller emitted an invalid three-tank target")
        plant.set_ballast_target(*target)
        target_applied = np.asarray(plant.target_ballast_mass, dtype=float).reshape(-1)
        transfer_error = float(np.max(np.abs(target_applied - target)))
        max_transfer_error = max(max_transfer_error, transfer_error)

        _, raw_info = plant.step(float(thrust_model(ws)), wd, dt_s, time_s)
        info = dict(raw_info)
        masses = np.asarray(info["tank_masses"], dtype=float).reshape(-1)
        rates = np.asarray(info["pump_net_rate_m3_min"], dtype=float).reshape(-1)
        latched = np.asarray(info["pump_latched"], dtype=bool).reshape(-1)
        state = np.asarray(plant.state, dtype=float).reshape(-1)
        if masses.size != 3 or rates.size != 3 or latched.size != 3:
            raise RuntimeError("plant feedback does not contain three pump/tank channels")
        step_finite = bool(
            np.all(np.isfinite(masses))
            and np.all(np.isfinite(rates))
            and np.all(np.isfinite(state))
        )
        finite_state = finite_state and step_finite
        if not step_finite:
            raise RuntimeError(f"non-finite plant state at t={time_s:.3f}s")

        pump_volume_m3 += float(np.sum(np.abs(rates))) * dt_s / 60.0
        pump_switch_count += int(np.sum(latched != previous_latched))
        previous_latched = latched.copy()
        pitch_deg = float(np.degrees(state[4]))
        roll_deg = float(np.degrees(state[3]))
        max_pitch = max(max_pitch, abs(pitch_deg))
        max_roll = max(max_roll, abs(roll_deg))
        min_mass = min(min_mass, float(np.min(masses)))
        max_mass = max(max_mass, float(np.max(masses)))
        max_tank_mass_change = max(
            max_tank_mass_change,
            float(np.max(np.abs(masses - initial_masses))),
        )

        platform_mass_value = info.get("platform_total_mass_kg")
        platform_center = _optional_array(info, "platform_center_of_mass_m", (3,))
        platform_inertia = _optional_array(
            info,
            "platform_inertia_about_reference_kg_m2",
            (3, 3),
        )
        platform_effective_mass = _optional_array(
            info,
            "platform_effective_mass_matrix",
            (6, 6),
        )
        present = (
            platform_mass_value is not None,
            platform_center is not None,
            platform_inertia is not None,
            platform_effective_mass is not None,
        )
        if any(present) and not all(present):
            raise RuntimeError("plant mass-property telemetry must be complete when provided")
        if mass_telemetry_available and not all(present):
            raise RuntimeError("plant stopped reporting required mass-property telemetry")
        if all(present):
            platform_mass = float(platform_mass_value)
            if not math.isfinite(platform_mass):
                raise RuntimeError("plant feedback contains non-finite platform mass")
            if initial_platform_mass is None:
                initial_platform_mass = platform_mass
                initial_platform_center = platform_center.copy()
                initial_platform_inertia = platform_inertia.copy()
                max_platform_mass_change = 0.0
                max_platform_mass_balance_error = 0.0
                max_platform_center_shift = 0.0
                max_platform_inertia_change = 0.0
            final_platform_mass = platform_mass
            mass_telemetry_available = True
            max_platform_mass_change = max(
                float(max_platform_mass_change),
                abs(platform_mass - float(initial_platform_mass)),
            )
            expected_mass_delta = float(np.sum(masses - initial_masses))
            observed_mass_delta = platform_mass - float(initial_platform_mass)
            max_platform_mass_balance_error = max(
                float(max_platform_mass_balance_error),
                abs(observed_mass_delta - expected_mass_delta),
            )
            max_platform_center_shift = max(
                float(max_platform_center_shift),
                float(np.linalg.norm(platform_center - initial_platform_center)),
            )
            max_platform_inertia_change = max(
                float(max_platform_inertia_change),
                float(np.max(np.abs(platform_inertia - initial_platform_inertia))),
            )
        last_info = info

        if index % sample_stride == 0 or index == step_count - 1:
            sample = {
                "time_s": time_s,
                "wind_speed_ms": ws,
                "wind_direction_deg": wd,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "tank_masses_kg": masses.tolist(),
                "target_masses_kg": target.tolist(),
                "pump_rates_m3_min": rates.tolist(),
                "pump_latched": latched.astype(int).tolist(),
            }
            if all(present):
                sample.update(
                    {
                        "platform_total_mass_kg": platform_mass,
                        "platform_center_of_mass_m": platform_center.tolist(),
                        "platform_inertia_about_reference_kg_m2": platform_inertia.tolist(),
                        "platform_effective_mass_matrix": platform_effective_mass.tolist(),
                    }
                )
            sampled.append(sample)

    decisions = tuple(dict(record) for record in controller.records)
    forecast_decisions = sum(int(row.get("forecast_available", 0)) for row in decisions)
    capacity = float(plant.tank_capacity)
    summary = ChainRunSummary(
        duration_s=duration_s,
        step_count=step_count,
        decision_count=len(decisions),
        forecast_decision_count=forecast_decisions,
        target_revision_count=controller.target_revision,
        pump_volume_m3=float(pump_volume_m3),
        pump_switch_count=int(pump_switch_count),
        max_abs_pitch_deg=float(max_pitch),
        max_abs_roll_deg=float(max_roll),
        max_command_transfer_error_kg=float(max_transfer_error),
        min_tank_mass_kg=float(min_mass),
        max_tank_mass_kg=float(max_mass),
        max_abs_tank_mass_change_kg=float(max_tank_mass_change),
        platform_mass_telemetry_available=bool(mass_telemetry_available),
        initial_platform_total_mass_kg=(
            None if initial_platform_mass is None else float(initial_platform_mass)
        ),
        final_platform_total_mass_kg=(
            None if final_platform_mass is None else float(final_platform_mass)
        ),
        max_abs_platform_mass_change_kg=(
            None if max_platform_mass_change is None else float(max_platform_mass_change)
        ),
        max_platform_mass_balance_error_kg=(
            None
            if max_platform_mass_balance_error is None
            else float(max_platform_mass_balance_error)
        ),
        max_platform_center_of_mass_shift_m=(
            None if max_platform_center_shift is None else float(max_platform_center_shift)
        ),
        max_abs_platform_inertia_change_kg_m2=(
            None if max_platform_inertia_change is None else float(max_platform_inertia_change)
        ),
        finite_state=bool(finite_state),
        capacity_respected=bool(min_mass >= -1e-6 and max_mass <= capacity + 1e-6),
        completed=True,
    )
    return ChainRunResult(
        summary=summary,
        sampled_timeseries=tuple(sampled),
        decisions=decisions,
    )


__all__ = [
    "BallastPlant",
    "ChainRunResult",
    "ChainRunSummary",
    "run_controller_plant_chain",
]
