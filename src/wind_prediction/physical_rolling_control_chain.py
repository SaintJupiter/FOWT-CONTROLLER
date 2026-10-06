"""Roll one physical current-cycle decision into the next modeled state.

The physical decision layer evaluates same-origin forecast responses, but its
selected response is only a request for the *current* control block.  This
module supplies the missing execution boundary for small, fixed case checks:
it executes that one request against the current observed wind load, then
uses the resulting modeled platform and pump state as the next cycle's model
origin.

It deliberately does not interpolate a forecast into the plant, turn held
forecast responses into a command schedule, create a fallback after
abstention, or aggregate a performance result.  Future wind information is
used only by the physical selector.  The committed low-order plant model receives the
current observed wind and caller-supplied current non-rotor load for each
committed block.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import (
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from fowt_platform.incremental import IncrementalState
from fowt_platform.rotor_input import RotorGeneralizedLoad

from .execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from .forecast_evidence import ForecastEvidence, validate_forecast_evidence
from .forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
)
from .forecast_platform_trajectory import ForecastTrajectoryEnvironment
from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .physical_current_cycle_decision import (
    PhysicalCurrentCycleDecision,
    PhysicalCurrentCycleDecisionPolicy,
    PhysicalCurrentCycleForecastTrendRoutePolicy,
    assemble_physical_current_cycle_decision_from_input,
)
from .physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    advance_physical_execution_platform_path,
)
from .physical_forecast_admission import (
    PhysicalForecastLeadReliabilityScope,
    bind_physical_forecast_admission,
    bind_physical_forecast_lead_reliability_scope,
)
from .physical_forecast_cycle_facts import assemble_first_interval_physical_cycle_input
from .physical_forecast_rhs_ballast_trend import (
    PhysicalForecastRhsBallastTrendPolicy,
    diagnose_physical_forecast_rhs_ballast_trend,
)
from .physical_lifecycle_selection import PhysicalPostureLimits


_ZERO_LOAD = np.zeros(6, dtype=float)


def _nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a non-empty string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _finite_vector(name: str, value: Any, *, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape} and finite values") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    copied = np.array(result, dtype=float, copy=True)
    copied.setflags(write=False)
    return copied


def _finite_positive(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


@dataclass(frozen=True)
class PhysicalRollingCycleInput:
    """One externally frozen observation and forecast for a current block.

    ``forecast`` and ``current_enu_downwind_air_velocity_mps`` are supplied by
    the same caller-owned source at the cycle origin.  The future thrust
    coefficients remain declared operating-state inputs rather than being
    inferred from a control action.  ``environment`` carries both current and
    future non-rotor loads; only its current values are applied to the actual
    committed plant step.
    """

    start_time_s: float
    forecast: ForecastEvidence
    current_enu_downwind_air_velocity_mps: Any
    current_wind_source: str
    current_wind_observation_time: str
    current_thrust_coefficient: float
    current_nacelle_yaw_relative_platform_rad: float
    current_nacelle_yaw_source: str
    future_thrust_coefficients: Any
    thrust_coefficient_source: str
    forecast_admitted: bool
    forecast_admission_basis: str
    environment: ForecastTrajectoryEnvironment

    def __post_init__(self) -> None:
        try:
            start = float(self.start_time_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_time_s must be finite and non-negative") from exc
        if not math.isfinite(start) or start < 0.0:
            raise ValueError("start_time_s must be finite and non-negative")
        if not isinstance(self.forecast, ForecastEvidence):
            raise TypeError("forecast must be ForecastEvidence")
        validate_forecast_evidence(self.forecast)
        if not self.forecast.provides_future_preview:
            raise ValueError("physical rolling chain requires a future forecast preview")
        origin = self.forecast.origin_time
        if origin is None:
            raise ValueError("physical rolling chain forecast requires origin_time")
        if not isinstance(self.environment, ForecastTrajectoryEnvironment):
            raise TypeError("environment must be ForecastTrajectoryEnvironment")
        if self.environment.forecast_origin_time != origin:
            raise ValueError("environment forecast origin must match forecast.origin_time")
        if int(self.environment.interval_count) != int(self.forecast.horizon_steps):
            raise ValueError("environment interval count must match forecast horizon")
        current_time = _nonempty_text(
            "current_wind_observation_time", self.current_wind_observation_time
        )
        if current_time != origin:
            raise ValueError("current_wind_observation_time must match forecast.origin_time")
        current = _finite_vector(
            "current_enu_downwind_air_velocity_mps",
            self.current_enu_downwind_air_velocity_mps,
            shape=(2,),
        )
        future = _finite_vector(
            "future_thrust_coefficients",
            self.future_thrust_coefficients,
            shape=(int(self.forecast.horizon_steps),),
        )
        if np.any(future < 0.0):
            raise ValueError("future_thrust_coefficients must be non-negative")
        current_coefficient = float(self.current_thrust_coefficient)
        if not math.isfinite(current_coefficient) or current_coefficient < 0.0:
            raise ValueError("current_thrust_coefficient must be non-negative and finite")
        current_yaw = float(self.current_nacelle_yaw_relative_platform_rad)
        if not math.isfinite(current_yaw):
            raise ValueError(
                "current_nacelle_yaw_relative_platform_rad must be finite"
            )
        if not isinstance(self.forecast_admitted, bool):
            raise TypeError("forecast_admitted must be bool")
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "current_enu_downwind_air_velocity_mps", current)
        object.__setattr__(self, "current_wind_source", _nonempty_text("current_wind_source", self.current_wind_source))
        object.__setattr__(self, "current_wind_observation_time", current_time)
        object.__setattr__(self, "current_thrust_coefficient", current_coefficient)
        object.__setattr__(self, "current_nacelle_yaw_relative_platform_rad", current_yaw)
        object.__setattr__(
            self,
            "current_nacelle_yaw_source",
            _nonempty_text("current_nacelle_yaw_source", self.current_nacelle_yaw_source),
        )
        object.__setattr__(self, "future_thrust_coefficients", future)
        object.__setattr__(self, "thrust_coefficient_source", _nonempty_text("thrust_coefficient_source", self.thrust_coefficient_source))
        object.__setattr__(self, "forecast_admission_basis", _nonempty_text("forecast_admission_basis", self.forecast_admission_basis))


@dataclass(frozen=True)
class PhysicalRollingChainSetup:
    """Static assumptions retained across a short rolling chain check."""

    runtime_assembly: BallastRuntimeAssembly
    reference_tank_masses_kg: Any
    tank_capacities_kg: Any
    tank_coordinates_m: Any
    rotor_load_parameters: ForecastRotorLoadParameters
    execution_config: ExecutionRolloutConfig
    posture_limits: PhysicalPostureLimits
    decision_policy: PhysicalCurrentCycleDecisionPolicy
    minimum_lead_reliability: float
    reliability_scope_basis: str
    future_platform_kinematics_source: str
    forecast_rhs_ballast_trend_policy: PhysicalForecastRhsBallastTrendPolicy | None = None
    forecast_trend_route_policy: PhysicalCurrentCycleForecastTrendRoutePolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.runtime_assembly, BallastRuntimeAssembly):
            raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
        if not isinstance(self.rotor_load_parameters, ForecastRotorLoadParameters):
            raise TypeError("rotor_load_parameters must be ForecastRotorLoadParameters")
        if not isinstance(self.execution_config, ExecutionRolloutConfig):
            raise TypeError("execution_config must be ExecutionRolloutConfig")
        if not isinstance(self.posture_limits, PhysicalPostureLimits):
            raise TypeError("posture_limits must be PhysicalPostureLimits")
        if not isinstance(self.decision_policy, PhysicalCurrentCycleDecisionPolicy):
            raise TypeError("decision_policy must be PhysicalCurrentCycleDecisionPolicy")
        reference = _finite_vector(
            "reference_tank_masses_kg", self.reference_tank_masses_kg, shape=(3,)
        )
        capacities = _finite_vector(
            "tank_capacities_kg", self.tank_capacities_kg, shape=(3,)
        )
        coordinates = _finite_vector(
            "tank_coordinates_m", self.tank_coordinates_m, shape=(3, 3)
        )
        if np.any(reference < 0.0) or np.any(capacities <= 0.0):
            raise ValueError("tank reference masses must be non-negative and capacities positive")
        if np.any(reference > capacities):
            raise ValueError("reference_tank_masses_kg must lie within capacity")
        if not np.allclose(
            capacities,
            float(self.execution_config.tank_capacity_kg),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("tank capacities must match execution_config.tank_capacity_kg")
        minimum = float(self.minimum_lead_reliability)
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError("minimum_lead_reliability must lie in [0, 1]")
        trend_policy = self.forecast_rhs_ballast_trend_policy
        route_policy = self.forecast_trend_route_policy
        if (trend_policy is None) != (route_policy is None):
            raise ValueError("trend and trend-route policies must be supplied together")
        if trend_policy is not None and not isinstance(
            trend_policy, PhysicalForecastRhsBallastTrendPolicy
        ):
            raise TypeError("forecast_rhs_ballast_trend_policy must be PhysicalForecastRhsBallastTrendPolicy or None")
        if route_policy is not None and not isinstance(
            route_policy, PhysicalCurrentCycleForecastTrendRoutePolicy
        ):
            raise TypeError("forecast_trend_route_policy must be PhysicalCurrentCycleForecastTrendRoutePolicy or None")
        object.__setattr__(self, "reference_tank_masses_kg", reference)
        object.__setattr__(self, "tank_capacities_kg", capacities)
        object.__setattr__(self, "tank_coordinates_m", coordinates)
        object.__setattr__(self, "minimum_lead_reliability", minimum)
        object.__setattr__(self, "reliability_scope_basis", _nonempty_text("reliability_scope_basis", self.reliability_scope_basis))
        object.__setattr__(self, "future_platform_kinematics_source", _nonempty_text("future_platform_kinematics_source", self.future_platform_kinematics_source))


@dataclass(frozen=True)
class PhysicalRollingCycleRecord:
    """One committed decision and its modeled current-block state transition."""

    cycle: PhysicalRollingCycleInput
    decision: PhysicalCurrentCycleDecision
    actual_rotor_load: RotorGeneralizedLoad
    start_platform_state: IncrementalState
    end_platform_state: IncrementalState | None
    start_execution_state: ExecutionRolloutState
    end_execution_state: ExecutionRolloutState | None
    actual_path: PhysicalExecutionPlatformPath | None
    status: str

    def __post_init__(self) -> None:
        if not isinstance(self.cycle, PhysicalRollingCycleInput):
            raise TypeError("cycle must be PhysicalRollingCycleInput")
        if not isinstance(self.decision, PhysicalCurrentCycleDecision):
            raise TypeError("decision must be PhysicalCurrentCycleDecision")
        if not isinstance(self.actual_rotor_load, RotorGeneralizedLoad):
            raise TypeError("actual_rotor_load must be RotorGeneralizedLoad")
        if not isinstance(self.start_platform_state, IncrementalState):
            raise TypeError("start_platform_state must be IncrementalState")
        if not isinstance(self.start_execution_state, ExecutionRolloutState):
            raise TypeError("start_execution_state must be ExecutionRolloutState")
        if self.status not in {"completed", "abstained"}:
            raise ValueError("status must be completed or abstained")
        if self.status == "completed":
            if not isinstance(self.actual_path, PhysicalExecutionPlatformPath):
                raise ValueError("completed record must retain an actual physical path")
            if self.end_platform_state is not self.actual_path.final_platform_state:
                raise ValueError("completed record must retain the actual path platform state")
            if self.end_execution_state is not self.actual_path.final_execution_state:
                raise ValueError("completed record must retain the actual path execution state")
            request = self.decision.require_current_execution_request()
            if (
                request.operation
                is not self.actual_path.substeps[0].execution_step.target_operation
            ):
                raise ValueError(
                    "committed path target operation must match the selected request"
                )
        elif any(value is not None for value in (self.end_platform_state, self.end_execution_state, self.actual_path)):
            raise ValueError("abstained record must not advance a plant state")

    def as_trace(self) -> dict[str, object]:
        """Return decision and committed-state facts without a score claim."""

        path = self.actual_path
        candidate_evaluations = [
            {
                "route": candidate.route.value,
                "forecast_lead_index": candidate.forecast_lead_index,
                "target_operation": candidate.execution_request.operation.value,
                "within_posture_limits": candidate.within_posture_limits,
                "max_normalized_posture": candidate.max_normalized_posture,
                "current_block_transferred_volume_m3": (
                    candidate.current_block_transferred_volume_m3
                ),
                "current_block_pump_active_time_s": (
                    candidate.current_block_pump_active_time_s
                ),
                "current_block_pump_start_count": (
                    candidate.current_block_pump_start_count
                ),
                "current_block_pump_stop_count": (
                    candidate.current_block_pump_stop_count
                ),
                "current_block_pump_direction_switch_count": (
                    candidate.current_block_pump_direction_switch_count
                ),
            }
            for candidate in self.decision.candidates
        ]
        return {
            "start_time_s": self.cycle.start_time_s,
            "forecast_origin_time": self.cycle.forecast.origin_time,
            "forecast_source": self.cycle.forecast.source,
            "forecast_model_version": self.cycle.forecast.model_version,
            "selected_route": None if self.decision.selected_route is None else self.decision.selected_route.value,
            "selected_forecast_lead_index": self.decision.selected_forecast_lead_index,
            "selection_basis": self.decision.selection_basis.value,
            "decision_reason": self.decision.reason.value,
            "forecast_trend_kind": None if self.decision.forecast_rhs_ballast_trend is None else self.decision.forecast_rhs_ballast_trend.kind.value,
            "forecast_trend_route_preference_applied": self.decision.forecast_trend_route_preference_applied,
            "candidate_evaluations": candidate_evaluations,
            "status": self.status,
            "current_wind_source": self.cycle.current_wind_source,
            "current_wind_uv_ms": [float(value) for value in self.cycle.current_enu_downwind_air_velocity_mps],
            "current_thrust_coefficient": self.cycle.current_thrust_coefficient,
            "thrust_coefficient_source": self.cycle.thrust_coefficient_source,
            "current_nacelle_yaw_relative_platform_rad": self.cycle.current_nacelle_yaw_relative_platform_rad,
            "current_nacelle_yaw_source": self.cycle.current_nacelle_yaw_source,
            "actual_rotor_generalized_load": [float(value) for value in self.actual_rotor_load.generalized_load_platform],
            "actual_platform_rotor_load_basis": "same_origin_current_load_assembled_for_decision",
            "start_actual_tank_masses_kg": [float(value) for value in self.start_execution_state.actual_masses_kg],
            "end_actual_tank_masses_kg": None if self.end_execution_state is None else [float(value) for value in self.end_execution_state.actual_masses_kg],
            "start_platform_position": [float(value) for value in self.start_platform_state.position],
            "end_platform_position": None if self.end_platform_state is None else [float(value) for value in self.end_platform_state.position],
            "start_platform_velocity": [float(value) for value in self.start_platform_state.velocity],
            "end_platform_velocity": None if self.end_platform_state is None else [float(value) for value in self.end_platform_state.velocity],
            "committed_execution": None if path is None else {
                "target_operation": path.substeps[0].execution_step.target_operation.value,
                "transferred_volume_m3": path.transferred_volume_m3,
                "active_time_s": path.aggregate_pump_active_time_s,
                "starts": path.pump_start_count,
                "stops": path.pump_stop_count,
                "direction_switches": path.pump_direction_switch_count,
            },
        }


@dataclass(frozen=True)
class PhysicalRollingChainResult:
    """Result of a short rolling chain check, without performance aggregation."""

    status: str
    planned_cycle_count: int
    records: tuple[PhysicalRollingCycleRecord, ...]
    final_platform_state: IncrementalState
    final_execution_state: ExecutionRolloutState

    def __post_init__(self) -> None:
        if self.status not in {"completed", "abstained"}:
            raise ValueError("status must be completed or abstained")
        if not isinstance(self.planned_cycle_count, int) or self.planned_cycle_count <= 0:
            raise ValueError("planned_cycle_count must be a positive integer")
        if not all(isinstance(record, PhysicalRollingCycleRecord) for record in self.records):
            raise TypeError("records must contain PhysicalRollingCycleRecord values")
        if self.status == "completed" and len(self.records) != self.planned_cycle_count:
            raise ValueError("completed result must contain every planned cycle")
        if not isinstance(self.final_platform_state, IncrementalState):
            raise TypeError("final_platform_state must be IncrementalState")
        if not isinstance(self.final_execution_state, ExecutionRolloutState):
            raise TypeError("final_execution_state must be ExecutionRolloutState")

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def as_trace(self) -> dict[str, object]:
        return {
            "purpose": "physical_current_cycle_decision_to_modeled_state_handoff_only",
            "status": self.status,
            "completed": self.completed,
            "planned_cycle_count": self.planned_cycle_count,
            "completed_cycle_count": sum(record.status == "completed" for record in self.records),
            "records": [record.as_trace() for record in self.records],
            "final_platform_position": [float(value) for value in self.final_platform_state.position],
            "final_actual_tank_masses_kg": [float(value) for value in self.final_execution_state.actual_masses_kg],
        }


def _validate_cycle_block_duration(
    *, cycle: PhysicalRollingCycleInput, setup: PhysicalRollingChainSetup
) -> None:
    """Require one forecast lead interval to match one committed control block."""

    if not math.isclose(
        cycle.forecast.sample_period_s,
        setup.execution_config.block_duration_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(
            "forecast sample period must equal the current execution block duration"
        )


def _validate_cycle_schedule(
    *, cycles: tuple[PhysicalRollingCycleInput, ...], setup: PhysicalRollingChainSetup
) -> None:
    if not cycles:
        raise ValueError("cycles must contain at least one input")
    duration = float(setup.execution_config.block_duration_s)
    try:
        first_origin = datetime.fromisoformat(
            str(cycles[0].forecast.origin_time).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("forecast origins must use ISO-8601 timestamps") from exc
    for index, cycle in enumerate(cycles):
        expected_start = index * duration
        if not math.isclose(cycle.start_time_s, expected_start, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("cycle start times must follow the configured control cadence")
        _validate_cycle_block_duration(cycle=cycle, setup=setup)
        try:
            origin = datetime.fromisoformat(
                str(cycle.forecast.origin_time).replace("Z", "+00:00")
            )
            origin_elapsed_s = (origin - first_origin).total_seconds()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "forecast origins must use one consistent ISO-8601 time basis"
            ) from exc
        if not math.isclose(origin_elapsed_s, cycle.start_time_s, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "forecast origins must advance with the configured control cadence"
            )


def _assemble_cycle_facts_and_decision(
    *,
    cycle: PhysicalRollingCycleInput,
    setup: PhysicalRollingChainSetup,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
) -> tuple[FirstIntervalPhysicalDecisionInput, PhysicalCurrentCycleDecision]:
    forecast = cycle.forecast
    horizon = int(forecast.horizon_steps)
    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=setup.runtime_assembly,
        actual_tank_masses_kg=execution_state.actual_masses_kg,
        reference_tank_masses_kg=setup.reference_tank_masses_kg,
        tank_capacities_kg=setup.tank_capacities_kg,
        tank_coordinates_m=setup.tank_coordinates_m,
    )
    operating_state = ForecastRotorOperatingState(
        current_thrust_coefficient=cycle.current_thrust_coefficient,
        future_thrust_coefficients=cycle.future_thrust_coefficients,
        current_platform_reference_velocity_platform_mps=platform_state.velocity[:3],
        current_platform_angular_velocity_platform_radps=platform_state.velocity[3:],
        future_platform_reference_velocities_platform_mps=np.zeros((horizon, 3)),
        future_platform_angular_velocities_platform_radps=np.zeros((horizon, 3)),
        future_platform_kinematics_mode="frozen_zero",
        future_platform_kinematics_source=setup.future_platform_kinematics_source,
        current_nacelle_yaw_relative_platform_rad=(
            cycle.current_nacelle_yaw_relative_platform_rad
        ),
        future_rotor_orientation_source=cycle.current_nacelle_yaw_source,
    )
    facts = assemble_first_interval_physical_cycle_input(
        forecast=forecast,
        current_enu_downwind_air_velocity_mps=(
            cycle.current_enu_downwind_air_velocity_mps
        ),
        current_wind_source=cycle.current_wind_source,
        current_wind_observation_time=cycle.current_wind_observation_time,
        rotor_load_parameters=setup.rotor_load_parameters,
        rotor_operating_state=operating_state,
        platform_snapshot=snapshot,
        initial_platform_state=platform_state,
        initial_platform_state_time=str(forecast.origin_time),
        environment=cycle.environment,
        runtime_assembly=setup.runtime_assembly,
        execution_state=execution_state,
        execution_config=setup.execution_config,
        execution_state_time=str(forecast.origin_time),
    )
    admission = bind_physical_forecast_admission(
        decision_input=facts,
        forecast=forecast,
        admitted=cycle.forecast_admitted,
        basis=cycle.forecast_admission_basis,
    )
    scope = bind_physical_forecast_lead_reliability_scope(
        decision_input=facts,
        forecast=forecast,
        minimum_lead_reliability=setup.minimum_lead_reliability,
        basis=setup.reliability_scope_basis,
    )
    trend = (
        None
        if setup.forecast_rhs_ballast_trend_policy is None
        else diagnose_physical_forecast_rhs_ballast_trend(
            decision_input=facts,
            policy=setup.forecast_rhs_ballast_trend_policy,
            reliability_scope=scope,
        )
    )
    decision = assemble_physical_current_cycle_decision_from_input(
        decision_input=facts,
        forecast_admission=admission,
        posture_limits=setup.posture_limits,
        policy=setup.decision_policy,
        reliability_scope=scope,
        forecast_rhs_ballast_trend=trend,
        forecast_trend_route_policy=setup.forecast_trend_route_policy,
    )
    return facts, decision


def advance_physical_rolling_control_cycle(
    *,
    setup: PhysicalRollingChainSetup,
    cycle: PhysicalRollingCycleInput,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
) -> PhysicalRollingCycleRecord:
    """Select and execute exactly one current block from its entering state.

    The caller owns the cadence and obtains a fresh observation and forecast
    before calling this function again.  This entry point deliberately does
    not accept a future request sequence or advance a later control cycle.
    ``run_physical_rolling_control_chain`` uses the same implementation after
    it has checked a preassembled fixed schedule.
    """

    if not isinstance(setup, PhysicalRollingChainSetup):
        raise TypeError("setup must be PhysicalRollingChainSetup")
    if not isinstance(cycle, PhysicalRollingCycleInput):
        raise TypeError("cycle must be PhysicalRollingCycleInput")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be IncrementalState")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    _validate_cycle_block_duration(cycle=cycle, setup=setup)

    facts, decision = _assemble_cycle_facts_and_decision(
        cycle=cycle,
        setup=setup,
        platform_state=platform_state,
        execution_state=execution_state,
    )
    actual_rotor_load = facts.trajectory.load_assembly.current_rotor_load
    request = decision.current_execution_request
    if request is None:
        return PhysicalRollingCycleRecord(
            cycle=cycle,
            decision=decision,
            actual_rotor_load=actual_rotor_load,
            start_platform_state=platform_state,
            end_platform_state=None,
            start_execution_state=execution_state,
            end_execution_state=None,
            actual_path=None,
            status="abstained",
        )

    path = advance_physical_execution_platform_path(
        platform_state=platform_state,
        execution_state=execution_state,
        execution_request=request,
        execution_config=setup.execution_config,
        runtime_assembly=setup.runtime_assembly,
        reference_tank_masses_kg=setup.reference_tank_masses_kg,
        tank_capacities_kg=setup.tank_capacities_kg,
        tank_coordinates_m=setup.tank_coordinates_m,
        rotor_load=actual_rotor_load,
        wave_load=cycle.environment.current_wave_load,
        other_load=cycle.environment.current_other_load,
        duration_s=setup.execution_config.block_duration_s,
    )
    return PhysicalRollingCycleRecord(
        cycle=cycle,
        decision=decision,
        actual_rotor_load=actual_rotor_load,
        start_platform_state=platform_state,
        end_platform_state=path.final_platform_state,
        start_execution_state=execution_state,
        end_execution_state=path.final_execution_state,
        actual_path=path,
        status="completed",
    )


def run_physical_rolling_control_chain(
    *,
    setup: PhysicalRollingChainSetup,
    cycles: tuple[PhysicalRollingCycleInput, ...],
    initial_platform_state: IncrementalState,
    initial_execution_state: ExecutionRolloutState,
) -> PhysicalRollingChainResult:
    """Commit one selected physical request per block and roll modeled state.

    No implicit fallback is applied when the selector abstains.  The result
    stops at that block with the last modeled state untouched, making a missing
    decision visible to its caller rather than silently reverting to legacy
    control behavior.
    """

    if not isinstance(setup, PhysicalRollingChainSetup):
        raise TypeError("setup must be PhysicalRollingChainSetup")
    if not isinstance(initial_platform_state, IncrementalState):
        raise TypeError("initial_platform_state must be IncrementalState")
    if not isinstance(initial_execution_state, ExecutionRolloutState):
        raise TypeError("initial_execution_state must be ExecutionRolloutState")
    frozen_cycles = tuple(cycles)
    if not all(isinstance(cycle, PhysicalRollingCycleInput) for cycle in frozen_cycles):
        raise TypeError("cycles must contain PhysicalRollingCycleInput values")
    _validate_cycle_schedule(cycles=frozen_cycles, setup=setup)

    platform_state = initial_platform_state
    execution_state = initial_execution_state
    records: list[PhysicalRollingCycleRecord] = []
    for cycle in frozen_cycles:
        record = advance_physical_rolling_control_cycle(
            cycle=cycle,
            setup=setup,
            platform_state=platform_state,
            execution_state=execution_state,
        )
        records.append(record)
        if record.status == "abstained":
            return PhysicalRollingChainResult(
                status="abstained",
                planned_cycle_count=len(frozen_cycles),
                records=tuple(records),
                final_platform_state=platform_state,
                final_execution_state=execution_state,
            )
        assert record.end_platform_state is not None
        assert record.end_execution_state is not None
        platform_state = record.end_platform_state
        execution_state = record.end_execution_state

    return PhysicalRollingChainResult(
        status="completed",
        planned_cycle_count=len(frozen_cycles),
        records=tuple(records),
        final_platform_state=platform_state,
        final_execution_state=execution_state,
    )


__all__ = [
    "advance_physical_rolling_control_cycle",
    "PhysicalRollingChainResult",
    "PhysicalRollingChainSetup",
    "PhysicalRollingCycleInput",
    "PhysicalRollingCycleRecord",
    "run_physical_rolling_control_chain",
]
