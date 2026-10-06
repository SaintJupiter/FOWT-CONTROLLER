"""Execution-aware pump model for candidate-sequence evaluation.

The rollout mirrors the target limiter and the three independent ballast-pump
state machines used by the legacy plant.  It intentionally stops at actuator
execution: no rigid-body or six-degree-of-freedom dynamics live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

import numpy as np


_EPS = 1e-12
_DEFAULT_RATE_SCHEDULE_M3_MIN = (
    (0.0, 0.0),
    (200.0, 4.0),
    (300.0, 6.0),
    (500.0, 8.0),
    (700.0, 10.0),
    (1000.0, 12.0),
    (2000.0, 14.0),
    (3000.0, 15.0),
)


def _within_stop_error(abs_error: Any, stop_error: float) -> np.ndarray:
    """Return the single completion predicate used by the pump state machine."""

    values = np.asarray(abs_error, dtype=float)
    return values <= _EPS if stop_error == 0.0 else values < stop_error


class ExecutionTargetOperation(str, Enum):
    """Target operation understood by the actuator rollout."""

    TRACK = "track"
    RELEASE_TO_CURRENT = "release_to_current"


@dataclass(frozen=True)
class ExecutionRolloutRequest:
    """Immutable target request passed from planning to actuator execution.

    Prediction and posture thresholds belong to the caller.  The execution
    model only distinguishes normal target tracking from an explicit release
    of the target to the currently measured tank masses.
    """

    operation: ExecutionTargetOperation
    target_masses_kg: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        operation = ExecutionTargetOperation(self.operation)
        object.__setattr__(self, "operation", operation)

        if operation is ExecutionTargetOperation.RELEASE_TO_CURRENT:
            if self.target_masses_kg is not None:
                raise ValueError(
                    "release_to_current derives its target from the rollout state"
                )
            return

        if self.target_masses_kg is None:
            raise ValueError("track requests require target_masses_kg")
        values = np.asarray(self.target_masses_kg, dtype=float).reshape(-1)
        if values.size != 3 or not np.all(np.isfinite(values)):
            raise ValueError("target_masses_kg must contain three finite values")
        object.__setattr__(
            self,
            "target_masses_kg",
            tuple(float(value) for value in values),
        )

    @classmethod
    def track(cls, target_masses_kg: Any) -> "ExecutionRolloutRequest":
        return cls(
            operation=ExecutionTargetOperation.TRACK,
            target_masses_kg=target_masses_kg,
        )

    @classmethod
    def release_to_current(cls) -> "ExecutionRolloutRequest":
        return cls(operation=ExecutionTargetOperation.RELEASE_TO_CURRENT)


def _three(values: Any, *, default: float = 0.0) -> np.ndarray:
    array = np.asarray(values if values is not None else [], dtype=float).reshape(-1)[:3]
    if array.size < 3:
        array = np.pad(array, (0, 3 - array.size), constant_values=default)
    return array


def _readonly_three(values: Any, *, default: float = 0.0, dtype=float) -> np.ndarray:
    """Copy one normalized three-tank vector into an immutable fact record."""

    result = np.array(_three(values, default=default), dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _readonly_count_three(name: str, values: Any) -> np.ndarray:
    """Normalize a non-negative integral three-pump counter vector."""

    try:
        raw = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three non-negative integers") from exc
    if (
        raw.shape != (3,)
        or not np.all(np.isfinite(raw))
        or np.any(raw < 0.0)
        or not np.allclose(raw, np.rint(raw), rtol=0.0, atol=0.0)
    ):
        raise ValueError(f"{name} must contain three non-negative integers")
    result = np.array(raw, dtype=int, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ExecutionRolloutConfig:
    block_duration_s: float = 1200.0
    water_density_kg_m3: float = 1025.0
    max_pump_rate_m3_min: float = 15.0
    target_slew_enabled: bool = False
    target_slew_rate_m3_min: float = 10.0
    stop_error_kg: float = 300.0
    tank_capacity_kg: float = 1850.0 * 1025.0
    internal_step_s: float = 1.0
    restart_error_kg: float = 500.0
    min_on_s: float = 20.0
    min_off_s: float = 12.0
    near_target_hold_s: float = 10.0
    ramp_up_m3_min_per_s: float = 2.0
    ramp_down_m3_min_per_s: float = 3.0
    pump_rate_schedule_m3_min: tuple[tuple[float, float], ...] = (
        _DEFAULT_RATE_SCHEDULE_M3_MIN
    )

    def __post_init__(self) -> None:
        if type(self.target_slew_enabled) is not bool:
            raise ValueError("target_slew_enabled must be a boolean")
        positive = {
            "block_duration_s": self.block_duration_s,
            "water_density_kg_m3": self.water_density_kg_m3,
            "max_pump_rate_m3_min": self.max_pump_rate_m3_min,
            "tank_capacity_kg": self.tank_capacity_kg,
            "internal_step_s": self.internal_step_s,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        non_negative = {
            "target_slew_rate_m3_min": self.target_slew_rate_m3_min,
            "stop_error_kg": self.stop_error_kg,
            "restart_error_kg": self.restart_error_kg,
            "min_on_s": self.min_on_s,
            "min_off_s": self.min_off_s,
            "near_target_hold_s": self.near_target_hold_s,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        for name, value in {
            "ramp_up_m3_min_per_s": self.ramp_up_m3_min_per_s,
            "ramp_down_m3_min_per_s": self.ramp_down_m3_min_per_s,
        }.items():
            rate = float(value)
            if math.isnan(rate) or rate < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.target_slew_enabled and self.target_slew_rate_m3_min <= 0.0:
            raise ValueError(
                "target_slew_rate_m3_min must be positive when target slew is enabled"
            )
        if self.restart_error_kg < self.stop_error_kg:
            raise ValueError("restart_error_kg must not be smaller than stop_error_kg")

        try:
            schedule = tuple(
                (float(error_kg), float(rate_m3_min))
                for error_kg, rate_m3_min in self.pump_rate_schedule_m3_min
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "pump_rate_schedule_m3_min must contain (error_kg, rate_m3_min) pairs"
            ) from exc
        if not schedule:
            raise ValueError("pump_rate_schedule_m3_min must not be empty")
        if not all(math.isfinite(value) for pair in schedule for value in pair):
            raise ValueError("pump_rate_schedule_m3_min must contain finite values")
        errors = tuple(pair[0] for pair in schedule)
        rates = tuple(pair[1] for pair in schedule)
        if errors[0] < 0.0 or any(b <= a for a, b in zip(errors, errors[1:])):
            raise ValueError("pump schedule errors must be non-negative and increasing")
        if any(rate < 0.0 for rate in rates) or any(
            b < a for a, b in zip(rates, rates[1:])
        ):
            raise ValueError("pump schedule rates must be non-negative and non-decreasing")
        if max(rates) > float(self.max_pump_rate_m3_min):
            raise ValueError("pump schedule rate exceeds max_pump_rate_m3_min")
        object.__setattr__(self, "pump_rate_schedule_m3_min", schedule)


@dataclass(frozen=True)
class ExecutionRolloutState:
    # The original four fields stay first so existing constructors remain valid.
    masses_kg: np.ndarray
    target_masses_kg: np.ndarray
    pump_rates_m3_min: np.ndarray
    pump_latched: np.ndarray
    primary_target_kg: np.ndarray | None = None
    pump_on_elapsed_s: np.ndarray | None = None
    pump_off_elapsed_s: np.ndarray | None = None
    pump_near_target_s: np.ndarray | None = None
    pump_command_rates_m3_min: np.ndarray | None = None
    last_flow_directions: np.ndarray | None = None
    previous_block_tank_mass_delta_kg: np.ndarray | None = None

    def __post_init__(self) -> None:
        masses = _three(self.masses_kg)
        target = _three(self.target_masses_kg)
        rates = _three(self.pump_rates_m3_min)
        primary = target if self.primary_target_kg is None else _three(self.primary_target_kg)
        commands = (
            np.abs(rates)
            if self.pump_command_rates_m3_min is None
            else np.abs(_three(self.pump_command_rates_m3_min))
        )
        last_directions = (
            np.sign(rates)
            if self.last_flow_directions is None
            else np.sign(_three(self.last_flow_directions))
        )
        previous_block_delta = _three(
            self.previous_block_tank_mass_delta_kg
        )

        object.__setattr__(self, "masses_kg", _readonly_three(masses))
        object.__setattr__(self, "target_masses_kg", _readonly_three(target))
        object.__setattr__(self, "pump_rates_m3_min", _readonly_three(rates))
        object.__setattr__(
            self,
            "pump_latched",
            _readonly_three(self.pump_latched, dtype=bool),
        )
        object.__setattr__(self, "primary_target_kg", _readonly_three(primary))
        object.__setattr__(
            self,
            "pump_on_elapsed_s",
            _readonly_three(np.maximum(_three(self.pump_on_elapsed_s), 0.0)),
        )
        object.__setattr__(
            self,
            "pump_off_elapsed_s",
            _readonly_three(
                np.maximum(_three(self.pump_off_elapsed_s, default=np.inf), 0.0)
            ),
        )
        object.__setattr__(
            self,
            "pump_near_target_s",
            _readonly_three(np.maximum(_three(self.pump_near_target_s), 0.0)),
        )
        object.__setattr__(
            self,
            "pump_command_rates_m3_min",
            _readonly_three(commands),
        )
        object.__setattr__(
            self,
            "last_flow_directions",
            _readonly_three(last_directions),
        )
        object.__setattr__(
            self,
            "previous_block_tank_mass_delta_kg",
            _readonly_three(previous_block_delta),
        )

    @property
    def actual_masses_kg(self) -> np.ndarray:
        return self.masses_kg

    @property
    def primary_target_masses_kg(self) -> np.ndarray:
        return self.primary_target_kg

    @property
    def rate_limited_target_kg(self) -> np.ndarray:
        return self.target_masses_kg

    @property
    def signed_flow_m3_min(self) -> np.ndarray:
        return self.pump_rates_m3_min

    @classmethod
    def from_plant_info(
        cls,
        plant_info: dict[str, Any],
        config: ExecutionRolloutConfig,
    ) -> "ExecutionRolloutState":
        masses = _three(plant_info.get("tank_masses"))
        target = _three(plant_info.get("target_ballast_mass", masses))
        primary_raw = plant_info.get("primary_target_kg")
        if primary_raw is None:
            primary_raw = plant_info.get("preview_primary_target_kg")
        if primary_raw is None:
            primary_raw = plant_info.get("target_final_kg", target)
        primary = _three(primary_raw)

        command_raw = plant_info.get(
            "pump_rate_smoothed_m3_min",
            plant_info.get("pump_rate_cmd_m3_min"),
        )
        net_raw = plant_info.get("pump_net_rate_m3_min")
        if net_raw is not None:
            signed_rates = _three(net_raw)
        elif command_raw is not None:
            signed_rates = np.sign(target - masses) * np.abs(_three(command_raw))
        else:
            signed_rates = np.zeros(3, dtype=float)
        command_rates = np.abs(
            _three(command_raw) if command_raw is not None else signed_rates
        )
        latched_raw = plant_info.get("pump_latched", command_rates > _EPS)

        capacity = max(float(config.tank_capacity_kg), 0.0)
        return cls(
            masses_kg=np.clip(masses, 0.0, capacity),
            target_masses_kg=np.clip(target, 0.0, capacity),
            pump_rates_m3_min=signed_rates,
            pump_latched=_three(latched_raw).astype(bool),
            primary_target_kg=np.clip(primary, 0.0, capacity),
            pump_on_elapsed_s=plant_info.get("pump_on_elapsed_s"),
            pump_off_elapsed_s=plant_info.get(
                "pump_off_elapsed_s",
                np.full(3, max(float(config.min_off_s), 0.0)),
            ),
            pump_near_target_s=plant_info.get("pump_near_target_s"),
            pump_command_rates_m3_min=command_rates,
            last_flow_directions=np.sign(signed_rates),
            previous_block_tank_mass_delta_kg=plant_info.get(
                "previous_block_tank_mass_delta_kg"
            ),
        )


@dataclass(frozen=True)
class ExecutionRolloutStep:
    state: ExecutionRolloutState
    requested_target_kg: np.ndarray
    shaped_target_kg: np.ndarray
    mass_delta_kg: np.ndarray
    transferred_volume_m3: float
    active_time_s: float
    starts: int
    direction_switches: int
    target_reached: bool
    stops: int = 0
    pump_volume_m3: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    pump_runtime_s: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    start_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    stop_counts: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=int))
    direction_switch_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=int)
    )
    target_operation: ExecutionTargetOperation = ExecutionTargetOperation.TRACK
    target_slew_reset: bool = False
    minimum_total_mass_delta_from_initial_kg: float = 0.0
    maximum_total_mass_delta_from_initial_kg: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.state, ExecutionRolloutState):
            raise TypeError("state must be ExecutionRolloutState")
        operation = ExecutionTargetOperation(self.target_operation)
        object.__setattr__(self, "target_operation", operation)
        if type(self.target_reached) is not bool:
            raise ValueError("target_reached must be a boolean")
        if type(self.target_slew_reset) is not bool:
            raise ValueError("target_slew_reset must be a boolean")
        for name in (
            "requested_target_kg",
            "shaped_target_kg",
            "mass_delta_kg",
            "pump_volume_m3",
            "pump_runtime_s",
        ):
            object.__setattr__(self, name, _readonly_three(getattr(self, name)))
        for name in (
            "start_counts",
            "stop_counts",
            "direction_switch_counts",
        ):
            object.__setattr__(self, name, _readonly_count_three(name, getattr(self, name)))
        for name in ("transferred_volume_m3", "active_time_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative scalar")
            object.__setattr__(self, name, value)
        minimum_total_delta = float(self.minimum_total_mass_delta_from_initial_kg)
        maximum_total_delta = float(self.maximum_total_mass_delta_from_initial_kg)
        if (
            not math.isfinite(minimum_total_delta)
            or not math.isfinite(maximum_total_delta)
            or minimum_total_delta > maximum_total_delta
        ):
            raise ValueError("total-mass delta extrema must be finite and ordered")
        net_total_delta = float(np.sum(self.mass_delta_kg))
        if not minimum_total_delta - _EPS <= net_total_delta <= maximum_total_delta + _EPS:
            raise ValueError("net total-mass delta must lie within recorded extrema")
        object.__setattr__(
            self,
            "minimum_total_mass_delta_from_initial_kg",
            minimum_total_delta,
        )
        object.__setattr__(
            self,
            "maximum_total_mass_delta_from_initial_kg",
            maximum_total_delta,
        )
        for name in ("starts", "stops", "direction_switches"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        if self.starts != int(np.sum(self.start_counts)):
            raise ValueError("starts must equal the summed start_counts")
        if self.stops != int(np.sum(self.stop_counts)):
            raise ValueError("stops must equal the summed stop_counts")
        if self.direction_switches != int(np.sum(self.direction_switch_counts)):
            raise ValueError(
                "direction_switches must equal the summed direction_switch_counts"
            )

    @property
    def per_tank_mass_delta_kg(self) -> np.ndarray:
        return self.mass_delta_kg


def _scheduled_rate_m3_min(
    abs_error_kg: float,
    config: ExecutionRolloutConfig,
) -> float:
    points = sorted(
        (
            (max(float(error), 0.0), max(float(rate), 0.0))
            for error, rate in config.pump_rate_schedule_m3_min
        ),
        key=lambda point: point[0],
    )
    if not points:
        return 0.0

    error = max(float(abs_error_kg), 0.0)
    if error <= points[0][0]:
        rate = points[0][1]
    elif error >= points[-1][0]:
        rate = points[-1][1]
    else:
        rate = points[-1][1]
        for (error0, rate0), (error1, rate1) in zip(points, points[1:]):
            if error0 <= error <= error1:
                if abs(error1 - error0) <= _EPS:
                    rate = rate1
                else:
                    fraction = (error - error0) / (error1 - error0)
                    rate = rate0 + fraction * (rate1 - rate0)
                break
    return float(min(rate, max(float(config.max_pump_rate_m3_min), 0.0)))


def _ramp_towards_rate(
    current_rate: float,
    target_rate: float,
    ramp_up_m3_min_per_s: float,
    ramp_down_m3_min_per_s: float,
    dt: float,
) -> float:
    current = max(float(current_rate), 0.0)
    target = max(float(target_rate), 0.0)
    if target >= current:
        max_delta = (
            max(float(ramp_up_m3_min_per_s), 0.0) * dt
            if np.isfinite(float(ramp_up_m3_min_per_s))
            else np.inf
        )
        return float(min(target, current + max_delta))
    max_delta = (
        max(float(ramp_down_m3_min_per_s), 0.0) * dt
        if np.isfinite(float(ramp_down_m3_min_per_s))
        else np.inf
    )
    return float(max(target, current - max_delta))


def simulate_execution_step(
    state: ExecutionRolloutState,
    requested_target_kg: np.ndarray | ExecutionRolloutRequest,
    config: ExecutionRolloutConfig,
) -> ExecutionRolloutStep:
    """Advance target shaping and all three pump states by one planning block.

    Passing a target array preserves the original tracking API.  An explicit
    release request resets both target-command states to the current masses;
    pump latches and timers are deliberately left intact so shutdown follows
    the same dwell and ramp-down rules as normal plant execution.
    """

    duration = max(float(config.block_duration_s), 0.0)
    density = max(float(config.water_density_kg_m3), _EPS)
    capacity = max(float(config.tank_capacity_kg), 0.0)
    internal_step = max(float(config.internal_step_s), _EPS)
    stop_error = max(float(config.stop_error_kg), 0.0)
    restart_error = max(float(config.restart_error_kg), stop_error)
    min_on = max(float(config.min_on_s), 0.0)
    min_off = max(float(config.min_off_s), 0.0)
    near_target_hold = max(float(config.near_target_hold_s), 0.0)

    initial_masses = np.clip(_three(state.masses_kg), 0.0, capacity)
    masses = initial_masses.copy()

    if isinstance(requested_target_kg, ExecutionRolloutRequest):
        request = requested_target_kg
        operation = request.operation
        explicit_target = request.target_masses_kg
    else:
        # Keep the original array-like call path byte-for-byte compatible in
        # its normalization and clipping behavior.
        operation = ExecutionTargetOperation.TRACK
        explicit_target = requested_target_kg

    release_to_current = operation is ExecutionTargetOperation.RELEASE_TO_CURRENT
    if release_to_current:
        requested = initial_masses.copy()
    else:
        requested = np.clip(_three(explicit_target), 0.0, capacity)

    shaped_target = np.clip(_three(state.target_masses_kg), 0.0, capacity)
    if release_to_current or not bool(config.target_slew_enabled):
        shaped_target = requested.copy()
    latched = _three(state.pump_latched).astype(bool)
    on_elapsed = np.maximum(_three(state.pump_on_elapsed_s), 0.0)
    off_elapsed = np.maximum(
        _three(state.pump_off_elapsed_s, default=np.inf),
        0.0,
    )
    near_target = np.maximum(_three(state.pump_near_target_s), 0.0)
    command_rates = np.minimum(
        np.abs(_three(state.pump_command_rates_m3_min)),
        max(float(config.max_pump_rate_m3_min), 0.0),
    )
    signed_flow = _three(state.pump_rates_m3_min)
    last_directions = np.sign(_three(state.last_flow_directions))

    pump_volume = np.zeros(3, dtype=float)
    pump_runtime = np.zeros(3, dtype=float)
    start_counts = np.zeros(3, dtype=int)
    stop_counts = np.zeros(3, dtype=int)
    direction_switch_counts = np.zeros(3, dtype=int)
    minimum_total_mass_delta = 0.0
    maximum_total_mass_delta = 0.0

    elapsed = 0.0
    while elapsed < duration - _EPS:
        dt = min(internal_step, duration - elapsed)
        if bool(config.target_slew_enabled):
            max_target_delta = (
                max(float(config.target_slew_rate_m3_min), 0.0)
                * density
                * dt
                / 60.0
            )
            shaped_target = np.clip(
                shaped_target
                + np.clip(
                    requested - shaped_target,
                    -max_target_delta,
                    max_target_delta,
                ),
                0.0,
                capacity,
            )

        error = shaped_target - masses
        abs_error = np.abs(error)
        # Match the physical plant's strict positive stop threshold.  A zero
        # threshold remains a useful exact-target mode for deterministic tests
        # and release requests, where equality must still allow shutdown.
        within_stop_error = _within_stop_error(abs_error, stop_error)
        near_target = np.where(within_stop_error, near_target + dt, 0.0)
        on_elapsed = np.where(latched, on_elapsed + dt, 0.0)
        off_elapsed = np.where(latched, 0.0, off_elapsed + dt)

        stop_ready = (
            latched
            & within_stop_error
            & (on_elapsed >= min_on)
            & (near_target >= near_target_hold)
        )
        if np.any(stop_ready):
            latched[stop_ready] = False
            on_elapsed[stop_ready] = 0.0
            off_elapsed[stop_ready] = 0.0
            near_target[stop_ready] = 0.0
            stop_counts[stop_ready] += 1

        start_ready = (
            ~latched
            & ~stop_ready
            & ~within_stop_error
            & (abs_error > restart_error)
            & (off_elapsed >= min_off)
        )
        if np.any(start_ready):
            latched[start_ready] = True
            on_elapsed[start_ready] = 0.0
            off_elapsed[start_ready] = 0.0
            near_target[start_ready] = 0.0
            start_counts[start_ready] += 1

        for pump_idx in range(3):
            target_rate = (
                _scheduled_rate_m3_min(abs_error[pump_idx], config)
                if latched[pump_idx]
                else 0.0
            )
            command_rates[pump_idx] = _ramp_towards_rate(
                command_rates[pump_idx],
                target_rate,
                config.ramp_up_m3_min_per_s,
                config.ramp_down_m3_min_per_s,
                dt,
            )

        active = command_rates > _EPS
        pump_runtime[active] += dt
        max_mass_delta = command_rates * density * dt / 60.0
        mass_delta = np.clip(error, -max_mass_delta, max_mass_delta)
        next_masses = np.clip(masses + mass_delta, 0.0, capacity)
        mass_delta = next_masses - masses
        signed_flow = mass_delta * 60.0 / (density * dt)
        pump_volume += np.abs(mass_delta) / density

        step_directions = np.sign(mass_delta)
        switched = (
            (step_directions != 0.0)
            & (last_directions != 0.0)
            & (step_directions != last_directions)
        )
        direction_switch_counts[switched] += 1
        last_directions = np.where(
            step_directions != 0.0,
            step_directions,
            last_directions,
        )
        masses = next_masses
        total_mass_delta = float(np.sum(masses - initial_masses))
        minimum_total_mass_delta = min(minimum_total_mass_delta, total_mass_delta)
        maximum_total_mass_delta = max(maximum_total_mass_delta, total_mass_delta)
        elapsed += dt

    next_state = ExecutionRolloutState(
        masses_kg=masses,
        target_masses_kg=shaped_target,
        pump_rates_m3_min=signed_flow,
        pump_latched=latched,
        primary_target_kg=requested,
        pump_on_elapsed_s=on_elapsed,
        pump_off_elapsed_s=off_elapsed,
        pump_near_target_s=near_target,
        pump_command_rates_m3_min=command_rates,
        last_flow_directions=last_directions,
        previous_block_tank_mass_delta_kg=masses - initial_masses,
    )
    net_mass_delta = masses - initial_masses
    return ExecutionRolloutStep(
        state=next_state,
        requested_target_kg=requested,
        shaped_target_kg=shaped_target.copy(),
        mass_delta_kg=net_mass_delta,
        transferred_volume_m3=float(np.sum(pump_volume)),
        active_time_s=float(np.sum(pump_runtime)),
        starts=int(np.sum(start_counts)),
        direction_switches=int(np.sum(direction_switch_counts)),
        target_reached=bool(
            np.all(_within_stop_error(np.abs(requested - masses), stop_error))
        ),
        stops=int(np.sum(stop_counts)),
        pump_volume_m3=pump_volume,
        pump_runtime_s=pump_runtime,
        start_counts=start_counts,
        stop_counts=stop_counts,
        direction_switch_counts=direction_switch_counts,
        target_operation=operation,
        target_slew_reset=release_to_current,
        minimum_total_mass_delta_from_initial_kg=minimum_total_mass_delta,
        maximum_total_mass_delta_from_initial_kg=maximum_total_mass_delta,
    )
