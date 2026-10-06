"""Fixed-cadence controller-to-platform chain checks.

This module is intentionally narrower than a simulation runner.  Its callers
provide a pre-frozen sequence of control-cycle inputs.  It then verifies the
causal handoff from a controller decision to a committed pump request and the
next low-order platform state.  It does not read datasets, choose cases,
interpolate forecasts, compare algorithms, or report a performance metric.

Forecast evidence is supplied to :func:`decide_control_cycle` only.  The
physical platform receives a rotor load built from the current observed wind at
the beginning of each control interval and that load is held fixed for the
interval.  This keeps the future forecast from being silently applied as a
future plant load.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import BallastRuntimeAssembly
from fowt_platform.incremental import IncrementalState
from fowt_platform.rotor_input import (
    RotorGeneralizedLoad,
    quasi_steady_rotor_generalized_load_from_enu_wind,
)

from .controller_core import ControlCoreConfig, ControlDecision, decide_control_cycle
from .controller_platform_handoff import (
    ControllerPlatformPath,
    advance_controller_platform_path,
    control_observation_from_incremental_state,
)
from .execution_rollout import (
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from .forecast_evidence import ForecastEvidence, validate_forecast_evidence
from .forecast_physical_load import ForecastRotorLoadParameters
from .forecast_stage_grid import build_forecast_stage_grid


_ZERO_LOAD = np.zeros(6, dtype=float)
_CURRENT_WIND_LOAD_RULE = (
    "current_observation_and_platform_motion_at_cycle_start_frozen_over_control_block"
)


def _finite_vector(name: str, value: Any, *, length: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite values") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nonempty_string(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


@dataclass(frozen=True)
class FixedChainCycleInput:
    """One externally frozen decision-cycle input.

    ``forecast`` must already have been generated at this cycle's origin.  No
    fallback continuation is created here if a forecast tail is incomplete.
    """

    start_time_s: float
    current_wind_uv_ms: Any
    current_wind_source: str
    current_wind_observation_time: str
    forecast: ForecastEvidence

    def __post_init__(self) -> None:
        try:
            start_time_s = float(self.start_time_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_time_s must be finite and non-negative") from exc
        if not math.isfinite(start_time_s) or start_time_s < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not isinstance(self.forecast, ForecastEvidence):
            raise TypeError("forecast must be a ForecastEvidence")
        validate_forecast_evidence(self.forecast)
        if not self.forecast.provides_future_preview:
            raise ValueError("fixed chain checks require a future forecast preview")
        if self.forecast.origin_time is None:
            raise ValueError("fixed chain checks require forecast.origin_time")
        object.__setattr__(self, "start_time_s", start_time_s)
        object.__setattr__(
            self,
            "current_wind_uv_ms",
            _finite_vector("current_wind_uv_ms", self.current_wind_uv_ms, length=2),
        )
        object.__setattr__(
            self,
            "current_wind_source",
            _nonempty_string("current_wind_source", self.current_wind_source),
        )
        observation_time = _nonempty_string(
            "current_wind_observation_time", self.current_wind_observation_time
        )
        if observation_time != self.forecast.origin_time:
            raise ValueError(
                "current_wind_observation_time must equal forecast.origin_time"
            )
        object.__setattr__(self, "current_wind_observation_time", observation_time)


@dataclass(frozen=True)
class FrozenCurrentRotorLoadAssumptions:
    """Explicit current-wind rotor-load assumptions for a fixed chain check.

    The thrust coefficient is a caller-supplied operating-state input.  This
    class records its source instead of inventing a rotor-speed or pitch rule.
    It is deliberately independent of future forecast coefficients.
    """

    parameters: ForecastRotorLoadParameters
    current_thrust_coefficient: float
    thrust_coefficient_source: str

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, ForecastRotorLoadParameters):
            raise TypeError("parameters must be ForecastRotorLoadParameters")
        try:
            coefficient = float(self.current_thrust_coefficient)
        except (TypeError, ValueError) as exc:
            raise ValueError("current_thrust_coefficient must be finite") from exc
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("current_thrust_coefficient must be non-negative and finite")
        object.__setattr__(self, "current_thrust_coefficient", coefficient)
        object.__setattr__(
            self,
            "thrust_coefficient_source",
            _nonempty_string("thrust_coefficient_source", self.thrust_coefficient_source),
        )


@dataclass(frozen=True)
class FixedChainCheckSetup:
    """Platform and actuator inputs shared by one pre-frozen case sequence."""

    controller_config: ControlCoreConfig
    runtime_assembly: BallastRuntimeAssembly
    reference_tank_masses_kg: Any
    tank_capacities_kg: Any
    tank_coordinates_m: Any
    current_rotor_load: FrozenCurrentRotorLoadAssumptions

    def __post_init__(self) -> None:
        if not isinstance(self.controller_config, ControlCoreConfig):
            raise TypeError("controller_config must be ControlCoreConfig")
        if not isinstance(self.runtime_assembly, BallastRuntimeAssembly):
            raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
        if not isinstance(self.current_rotor_load, FrozenCurrentRotorLoadAssumptions):
            raise TypeError("current_rotor_load must be FrozenCurrentRotorLoadAssumptions")
        reference = _finite_vector(
            "reference_tank_masses_kg", self.reference_tank_masses_kg, length=3
        )
        capacities = _finite_vector(
            "tank_capacities_kg", self.tank_capacities_kg, length=3
        )
        coordinates = np.asarray(self.tank_coordinates_m, dtype=float)
        if coordinates.shape != (3, 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("tank_coordinates_m must have shape (3, 3) and be finite")
        if np.any(reference < 0.0) or np.any(capacities <= 0.0):
            raise ValueError("tank masses must be non-negative and capacities positive")
        if np.any(reference > capacities):
            raise ValueError("reference_tank_masses_kg must not exceed tank capacities")
        if not np.allclose(
            capacities,
            self.controller_config.execution.tank_capacity_kg,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "tank_capacities_kg must match controller execution tank capacity"
            )
        if not math.isclose(
            self.controller_config.execution.internal_step_s,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "fixed chain checks require a 1 s execution internal step"
            )
        object.__setattr__(self, "reference_tank_masses_kg", reference)
        object.__setattr__(self, "tank_capacities_kg", capacities)
        stored_coordinates = np.array(coordinates, dtype=float, copy=True)
        stored_coordinates.setflags(write=False)
        object.__setattr__(self, "tank_coordinates_m", stored_coordinates)


@dataclass(frozen=True)
class FixedChainCycleRecord:
    """Trace facts for one attempted control interval, without a score claim."""

    start_time_s: float
    forecast_source: str
    forecast_model_version: str
    forecast_origin_time: str
    forecast_horizon_steps: int
    selected_action: str
    target_operation: str
    execution_operation: str
    status: str
    current_wind_source: str
    current_wind_observation_time: str
    current_wind_uv_ms: tuple[float, float]
    current_wind_load_rule: str
    current_thrust_coefficient: float
    thrust_coefficient_source: str
    wind_generalized_load: tuple[float, float, float, float, float, float]
    wave_load_source: str
    other_load_source: str
    future_forecast_load_applied_to_platform: bool
    start_actual_tank_masses_kg: tuple[float, float, float]
    end_actual_tank_masses_kg: tuple[float, float, float]
    start_platform_position: tuple[float, float, float, float, float, float]
    end_platform_position: tuple[float, float, float, float, float, float]
    committed_transferred_volume_m3: float
    committed_active_time_s: float
    committed_starts: int
    committed_stops: int
    committed_direction_switches: int
    actual_p4_path_advanced: bool
    decision: ControlDecision

    def as_trace(self) -> dict[str, object]:
        """Return the cycle facts together with the original decision audit.

        The decision remains the output of the unchanged controller.  The
        committed metrics are separately accumulated from the actual P4
        substeps, so candidate-preview execution must not be confused with
        what was physically advanced during this control block.
        """

        return {
            "start_time_s": self.start_time_s,
            "status": self.status,
            "forecast": {
                "source": self.forecast_source,
                "model_version": self.forecast_model_version,
                "origin_time": self.forecast_origin_time,
                "horizon_steps": self.forecast_horizon_steps,
            },
            "current_wind": {
                "source": self.current_wind_source,
                "observation_time": self.current_wind_observation_time,
                "uv_ms": list(self.current_wind_uv_ms),
                "load_rule": self.current_wind_load_rule,
            },
            "rotor_load": {
                "thrust_coefficient": self.current_thrust_coefficient,
                "thrust_coefficient_source": self.thrust_coefficient_source,
                "generalized_load": list(self.wind_generalized_load),
                "future_forecast_applied_to_platform": (
                    self.future_forecast_load_applied_to_platform
                ),
            },
            "selected_action": self.selected_action,
            "target_operation": self.target_operation,
            "execution_operation": self.execution_operation,
            "committed_execution": {
                "role": (
                    "actual_p4_path_aggregate"
                    if self.actual_p4_path_advanced
                    else "no_actual_p4_path"
                ),
                "advanced": self.actual_p4_path_advanced,
                "transferred_volume_m3": self.committed_transferred_volume_m3,
                "active_time_s": self.committed_active_time_s,
                "active_time_definition": (
                    "aggregate_pump_seconds"
                    if self.actual_p4_path_advanced
                    else None
                ),
                "starts": self.committed_starts,
                "stops": self.committed_stops,
                "direction_switches": self.committed_direction_switches,
            },
            "start_actual_tank_masses_kg": list(self.start_actual_tank_masses_kg),
            "end_actual_tank_masses_kg": list(self.end_actual_tank_masses_kg),
            "start_platform_position": list(self.start_platform_position),
            "end_platform_position": list(self.end_platform_position),
            "decision": self.decision.as_trace(),
        }


@dataclass(frozen=True)
class FixedChainCheckResult:
    """Completion state of one fixed input sequence.
    """

    case_name: str
    status: str
    planned_cycle_count: int
    records: tuple[FixedChainCycleRecord, ...]
    final_platform_state: IncrementalState
    final_execution_state: ExecutionRolloutState

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def attempted_cycle_count(self) -> int:
        """Return cycles for which a decision was actually constructed."""

        return len(self.records)

    @property
    def completed_cycle_count(self) -> int:
        """Return only cycles whose pump-platform path was advanced."""

        return sum(record.status == "completed" for record in self.records)

    def as_trace(self) -> dict[str, object]:
        """Return a serializable audit record for a fixed chain check."""

        return {
            "case_name": self.case_name,
            "status": self.status,
            "completed": self.completed,
            "planned_cycle_count": self.planned_cycle_count,
            "attempted_cycle_count": self.attempted_cycle_count,
            "completed_cycle_count": self.completed_cycle_count,
            "records": [record.as_trace() for record in self.records],
            "final_platform_position": list(self.final_platform_state.position),
            "final_platform_velocity": list(self.final_platform_state.velocity),
            "final_actual_tank_masses_kg": list(
                self.final_execution_state.actual_masses_kg
            ),
        }


def _required_forecast_points(
    *,
    config: ControlCoreConfig,
    forecast: ForecastEvidence,
) -> int:
    return build_forecast_stage_grid(
        sample_period_s=forecast.sample_period_s,
        stage_duration_s=config.stage_duration_s,
        stage_count=config.stage_count,
    ).required_point_count


def _validate_fixed_cycle_schedule(
    *,
    cycles: tuple[FixedChainCycleInput, ...],
    config: ControlCoreConfig,
) -> None:
    if not cycles:
        raise ValueError("cycles must contain at least one fixed control input")
    for index, cycle in enumerate(cycles):
        expected_start = index * config.stage_duration_s
        if not math.isclose(
            cycle.start_time_s,
            expected_start,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "fixed cycle start times must begin at zero and follow the control cadence"
            )
        required = _required_forecast_points(config=config, forecast=cycle.forecast)
        if cycle.forecast.horizon_steps < required:
            raise ValueError(
                "forecast evidence does not cover the configured control horizon "
                f"at cycle {index}: need {required} points, got "
                f"{cycle.forecast.horizon_steps}"
            )


def _current_rotor_load(
    *,
    cycle: FixedChainCycleInput,
    platform_state: IncrementalState,
    assumptions: FrozenCurrentRotorLoadAssumptions,
) -> RotorGeneralizedLoad:
    """Build one current observation load without consuming a future lead."""

    parameters = assumptions.parameters
    rotor_plane_wind = (
        np.asarray(cycle.current_wind_uv_ms, dtype=float)
        * parameters.rotor_plane_speed_scale
    )
    return quasi_steady_rotor_generalized_load_from_enu_wind(
        air_density_kg_m3=parameters.air_density_kg_m3,
        rotor_radius_m=parameters.rotor_radius_m,
        thrust_coefficient=assumptions.current_thrust_coefficient,
        enu_downwind_air_velocity_mps=rotor_plane_wind,
        frozen_equilibrium_heading_rad=parameters.frozen_equilibrium_heading_rad,
        platform_reference_velocity_platform_mps=platform_state.velocity[:3],
        platform_angular_velocity_platform_radps=platform_state.velocity[3:],
        rotor_force_application_point_from_reference_m=(
            parameters.rotor_force_application_point_from_reference_m
        ),
        downwind_rotor_normal_platform=parameters.downwind_rotor_normal_platform,
    )


def _validate_actual_tank_masses(
    execution_state: ExecutionRolloutState,
    capacities: np.ndarray,
) -> None:
    actual = np.asarray(execution_state.actual_masses_kg, dtype=float).reshape(-1)
    if actual.shape != (3,) or not np.all(np.isfinite(actual)):
        raise ValueError("actual tank masses must contain three finite values")
    if np.any(actual < 0.0) or np.any(actual > capacities):
        raise ValueError("actual tank masses must lie within tank capacities")


def _record(
    *,
    cycle: FixedChainCycleInput,
    decision: ControlDecision,
    status: str,
    setup: FixedChainCheckSetup,
    rotor_load: RotorGeneralizedLoad,
    start_platform_state: IncrementalState,
    end_platform_state: IncrementalState,
    start_execution_state: ExecutionRolloutState,
    end_execution_state: ExecutionRolloutState,
    committed_path: ControllerPlatformPath | None,
) -> FixedChainCycleRecord:
    if committed_path is None:
        transferred_volume_m3 = 0.0
        active_time_s = 0.0
        starts = 0
        stops = 0
        direction_switches = 0
    else:
        transferred_volume_m3 = float(
            sum(
                step.execution_step.transferred_volume_m3
                for step in committed_path.substeps
            )
        )
        active_time_s = float(
            sum(step.execution_step.active_time_s for step in committed_path.substeps)
        )
        starts = int(sum(step.execution_step.starts for step in committed_path.substeps))
        stops = int(sum(step.execution_step.stops for step in committed_path.substeps))
        direction_switches = int(
            sum(
                step.execution_step.direction_switches
                for step in committed_path.substeps
            )
        )
    return FixedChainCycleRecord(
        start_time_s=cycle.start_time_s,
        forecast_source=cycle.forecast.source,
        forecast_model_version=cycle.forecast.model_version,
        forecast_origin_time=str(cycle.forecast.origin_time),
        forecast_horizon_steps=cycle.forecast.horizon_steps,
        selected_action=decision.action.value,
        target_operation=decision.target_operation.value,
        execution_operation=decision.execution_request.operation.value,
        status=status,
        current_wind_source=cycle.current_wind_source,
        current_wind_observation_time=cycle.current_wind_observation_time,
        current_wind_uv_ms=tuple(float(value) for value in cycle.current_wind_uv_ms),
        current_wind_load_rule=_CURRENT_WIND_LOAD_RULE,
        current_thrust_coefficient=(
            setup.current_rotor_load.current_thrust_coefficient
        ),
        thrust_coefficient_source=(
            setup.current_rotor_load.thrust_coefficient_source
        ),
        wind_generalized_load=tuple(
            float(value) for value in rotor_load.generalized_load_platform
        ),
        wave_load_source="explicit_zero",
        other_load_source="explicit_zero",
        future_forecast_load_applied_to_platform=False,
        start_actual_tank_masses_kg=tuple(
            float(value) for value in start_execution_state.actual_masses_kg
        ),
        end_actual_tank_masses_kg=tuple(
            float(value) for value in end_execution_state.actual_masses_kg
        ),
        start_platform_position=tuple(
            float(value) for value in start_platform_state.position
        ),
        end_platform_position=tuple(
            float(value) for value in end_platform_state.position
        ),
        committed_transferred_volume_m3=transferred_volume_m3,
        committed_active_time_s=active_time_s,
        committed_starts=starts,
        committed_stops=stops,
        committed_direction_switches=direction_switches,
        actual_p4_path_advanced=committed_path is not None,
        decision=decision,
    )


def run_fixed_case_chain_check(
    *,
    case_name: str,
    setup: FixedChainCheckSetup,
    cycles: tuple[FixedChainCycleInput, ...],
    initial_platform_state: IncrementalState,
    initial_execution_state: ExecutionRolloutState,
) -> FixedChainCheckResult:
    """Run a fixed-cycle causal handoff check without performance aggregation.

    Both tracking and target-release requests advance through the same P4
    substep order.  A release remains an actuator lifecycle operation, not a
    substituted tracking target.
    """

    normalized_name = _nonempty_string("case_name", case_name)
    if not isinstance(setup, FixedChainCheckSetup):
        raise TypeError("setup must be FixedChainCheckSetup")
    if not isinstance(initial_platform_state, IncrementalState):
        raise TypeError("initial_platform_state must be IncrementalState")
    if not isinstance(initial_execution_state, ExecutionRolloutState):
        raise TypeError("initial_execution_state must be ExecutionRolloutState")
    frozen_cycles = tuple(cycles)
    if not all(isinstance(cycle, FixedChainCycleInput) for cycle in frozen_cycles):
        raise TypeError("cycles must contain FixedChainCycleInput values")
    _validate_fixed_cycle_schedule(cycles=frozen_cycles, config=setup.controller_config)

    platform_state = initial_platform_state
    execution_state = initial_execution_state
    records: list[FixedChainCycleRecord] = []
    for cycle in frozen_cycles:
        _validate_actual_tank_masses(execution_state, setup.tank_capacities_kg)
        observation = control_observation_from_incremental_state(
            time_s=cycle.start_time_s,
            platform_state=platform_state,
            execution_state=execution_state,
            current_wind_uv_ms=tuple(float(value) for value in cycle.current_wind_uv_ms),
        )
        decision = decide_control_cycle(
            observation,
            cycle.forecast,
            setup.controller_config,
        )
        rotor_load = _current_rotor_load(
            cycle=cycle,
            platform_state=platform_state,
            assumptions=setup.current_rotor_load,
        )
        path: ControllerPlatformPath = advance_controller_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=decision.execution_request,
            execution_config=setup.controller_config.execution,
            runtime_assembly=setup.runtime_assembly,
            reference_tank_masses_kg=setup.reference_tank_masses_kg,
            tank_capacities_kg=setup.tank_capacities_kg,
            tank_coordinates_m=setup.tank_coordinates_m,
            rotor_load=rotor_load,
            wave_load=_ZERO_LOAD,
            other_load=_ZERO_LOAD,
            duration_s=setup.controller_config.stage_duration_s,
        )
        next_platform_state = path.final_platform_state
        next_execution_state = path.final_execution_state
        records.append(
            _record(
                cycle=cycle,
                decision=decision,
                status="completed",
                setup=setup,
                rotor_load=rotor_load,
                start_platform_state=platform_state,
                end_platform_state=next_platform_state,
                start_execution_state=execution_state,
                end_execution_state=next_execution_state,
                committed_path=path,
            )
        )
        platform_state = next_platform_state
        execution_state = next_execution_state

    return FixedChainCheckResult(
        case_name=normalized_name,
        status="completed",
        planned_cycle_count=len(frozen_cycles),
        records=tuple(records),
        final_platform_state=platform_state,
        final_execution_state=execution_state,
    )


__all__ = [
    "FixedChainCheckResult",
    "FixedChainCheckSetup",
    "FixedChainCycleInput",
    "FixedChainCycleRecord",
    "FrozenCurrentRotorLoadAssumptions",
    "run_fixed_case_chain_check",
]
