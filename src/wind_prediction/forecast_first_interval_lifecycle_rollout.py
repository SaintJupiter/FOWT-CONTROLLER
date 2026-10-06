"""First-forecast-interval pump execution against frozen forecast loads.

This module joins one already defined target lifecycle to the first interval
of a frozen forecast trajectory.  It preserves the trajectory's source-bound
rotor, wave and other loads while the actual pump state changes the ballast
snapshot at each physical substep.  The result is a narrow counterfactual
record, not a candidate comparison, safety result, policy decision or a
multi-interval closed-loop forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from fowt_platform.incremental import IncrementalState

from .execution_rollout import (
    ExecutionRolloutStep,
    ExecutionRolloutState,
    ExecutionTargetOperation,
    simulate_execution_step,
)
from .forecast_platform_trajectory import ForecastPlatformTrajectory
from .physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    advance_physical_execution_platform_path,
)
from .physical_endpoint_preview_binding import validate_endpoint_preview_at_trajectory_lead
from .physical_target_lifecycle import PhysicalTargetLifecycleTrace


_LOAD_TOLERANCE = 1.0e-7
_MASS_TOLERANCE = 1.0e-8
_STATE_TOLERANCE = 1.0e-10
_VOLUME_TOLERANCE = 1.0e-12
_TIME_TOLERANCE = 1.0e-9


def _readonly_six(name: str, value: Any) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (6,) and finite values") from exc
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must have shape (6,) and finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _execution_states_match(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
    for attribute in (
        "actual_masses_kg",
        "rate_limited_target_kg",
        "primary_target_masses_kg",
        "signed_flow_m3_min",
        "pump_latched",
        "pump_on_elapsed_s",
        "pump_off_elapsed_s",
        "pump_near_target_s",
        "pump_command_rates_m3_min",
        "last_flow_directions",
    ):
        if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
            return False
    return True


def _execution_steps_match(
    left: ExecutionRolloutStep,
    right: ExecutionRolloutStep,
) -> bool:
    """Return whether one substep retains the execution model's full outcome."""

    if (
        left.target_operation is not right.target_operation
        or left.target_reached is not right.target_reached
        or left.target_slew_reset is not right.target_slew_reset
        or left.starts != right.starts
        or left.stops != right.stops
        or left.direction_switches != right.direction_switches
        or not _execution_states_match(left.state, right.state)
    ):
        return False
    for attribute in (
        "requested_target_kg",
        "shaped_target_kg",
        "mass_delta_kg",
        "pump_volume_m3",
        "pump_runtime_s",
    ):
        if not np.allclose(
            getattr(left, attribute),
            getattr(right, attribute),
            rtol=0.0,
            atol=_MASS_TOLERANCE,
        ):
            return False
    for attribute in (
        "start_counts",
        "stop_counts",
        "direction_switch_counts",
    ):
        if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
            return False
    return bool(
        np.isclose(
            left.transferred_volume_m3,
            right.transferred_volume_m3,
            rtol=0.0,
            atol=_VOLUME_TOLERANCE,
        )
        and np.isclose(
            left.active_time_s,
            right.active_time_s,
            rtol=0.0,
            atol=_TIME_TOLERANCE,
        )
    )


def _snapshots_have_same_dynamics(
    left: BallastModelSnapshot,
    right: BallastModelSnapshot,
) -> bool:
    if not np.allclose(
        left.actual_tank_masses_kg,
        right.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ) or not np.allclose(
        left.reference_tank_masses_kg,
        right.reference_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ) or not np.allclose(
        left.tank_capacities_kg,
        right.tank_capacities_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ) or not np.allclose(
        left.tank_coordinates_m,
        right.tank_coordinates_m,
        rtol=0.0,
        atol=1.0e-12,
    ) or not np.isclose(
        left.gravity_m_s2,
        right.gravity_m_s2,
        rtol=0.0,
        atol=1.0e-12,
    ) or left.runtime_provenance != right.runtime_provenance:
        return False
    if not np.allclose(
        left.incremental_ballast_load,
        right.incremental_ballast_load,
        rtol=0.0,
        atol=_LOAD_TOLERANCE,
    ):
        return False
    for name in (
        "mass",
        "damping",
        "hydrostatic_stiffness",
        "mooring_stiffness",
        "weight_stiffness",
    ):
        if not np.allclose(
            getattr(left.matrices, name),
            getattr(right.matrices, name),
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            return False
    return True


def _first_interval_duration(trajectory: ForecastPlatformTrajectory) -> float:
    first_step = trajectory.steps[0]
    duration = float(first_step.end_time_s - first_step.start_time_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("first forecast interval must have a positive finite duration")
    return duration


def _configured_substep_durations(
    *,
    duration_s: float,
    internal_step_s: float,
) -> tuple[float, ...]:
    """Return the fixed split used by the physical path constructor."""

    remaining = float(duration_s)
    step_limit = float(internal_step_s)
    durations: list[float] = []
    while remaining > 1.0e-12:
        duration = min(step_limit, remaining)
        durations.append(duration)
        remaining -= duration
    return tuple(durations)


def _validate_lifecycle_at_first_interval(
    *,
    trajectory: ForecastPlatformTrajectory,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
    runtime_assembly: BallastRuntimeAssembly,
) -> float:
    if not isinstance(trajectory, ForecastPlatformTrajectory):
        raise TypeError("trajectory must be ForecastPlatformTrajectory")
    if not isinstance(lifecycle_trace, PhysicalTargetLifecycleTrace):
        raise TypeError("lifecycle_trace must be PhysicalTargetLifecycleTrace")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")

    duration = _first_interval_duration(trajectory)
    snapshot = trajectory.platform_snapshot
    if lifecycle_trace.execution_start_time != trajectory.initial_state_time:
        raise ValueError(
            "lifecycle execution start time must match the forecast trajectory origin"
        )
    if not np.isclose(
        lifecycle_trace.execution_duration_s,
        duration,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(
            "lifecycle execution duration must match the first forecast interval"
        )
    if duration > float(lifecycle_trace.execution_config.block_duration_s) + 1e-12:
        raise ValueError(
            "first forecast interval must not exceed the lifecycle execution block"
        )
    if not np.allclose(
        lifecycle_trace.execution_start_state.actual_masses_kg,
        snapshot.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ):
        raise ValueError(
            "lifecycle execution start masses must match the frozen-tank snapshot"
        )
    if not np.allclose(
        snapshot.tank_capacities_kg,
        float(lifecycle_trace.execution_config.tank_capacity_kg),
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ):
        raise ValueError(
            "lifecycle execution tank capacity must match the frozen-tank snapshot"
        )
    if runtime_assembly.provenance != snapshot.runtime_provenance:
        raise ValueError(
            "runtime_assembly provenance must match the frozen-tank snapshot"
        )
    reconstructed_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=snapshot.actual_tank_masses_kg,
        reference_tank_masses_kg=snapshot.reference_tank_masses_kg,
        tank_capacities_kg=snapshot.tank_capacities_kg,
        tank_coordinates_m=snapshot.tank_coordinates_m,
    )
    if not _snapshots_have_same_dynamics(snapshot, reconstructed_snapshot):
        raise ValueError(
            "runtime_assembly must reconstruct the frozen-tank trajectory snapshot"
        )

    preview = lifecycle_trace.source_preview
    if preview is not None:
        validate_endpoint_preview_at_trajectory_lead(
            preview=preview,
            trajectory=trajectory,
            lead_index=0,
            lead_time_s=duration,
            first_lead_only=True,
        )
    return duration


def _validate_path_facts(
    *,
    trajectory: ForecastPlatformTrajectory,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
    runtime_assembly: BallastRuntimeAssembly,
    physical_path: PhysicalExecutionPlatformPath,
) -> None:
    """Check the path's causal handoff against the configured P4 split."""

    first_substep = physical_path.substeps[0]
    if not np.allclose(
        first_substep.start_platform_state.position,
        trajectory.initial_state.position,
        rtol=0.0,
        atol=_STATE_TOLERANCE,
    ) or not np.allclose(
        first_substep.start_platform_state.velocity,
        trajectory.initial_state.velocity,
        rtol=0.0,
        atol=_STATE_TOLERANCE,
    ):
        raise ValueError(
            "physical_path initial platform state must match the forecast trajectory"
        )
    if not _execution_states_match(
        first_substep.start_execution_state,
        lifecycle_trace.execution_start_state,
    ):
        raise ValueError(
            "physical_path initial actuator state must match the lifecycle trace"
        )

    expected_durations = _configured_substep_durations(
        duration_s=physical_path.duration_s,
        internal_step_s=lifecycle_trace.execution_config.internal_step_s,
    )
    if len(physical_path.substeps) != len(expected_durations):
        raise ValueError("physical_path must match configured physical substeps")

    expected_state = lifecycle_trace.execution_start_state
    first_loads = trajectory.steps[0].platform_step.loads
    for substep, expected_duration in zip(
        physical_path.substeps,
        expected_durations,
    ):
        if not np.isclose(
            substep.duration_s,
            expected_duration,
            rtol=0.0,
            atol=_TIME_TOLERANCE,
        ):
            raise ValueError("physical_path must match configured physical substeps")
        if not _execution_states_match(substep.start_execution_state, expected_state):
            raise ValueError(
                "physical_path actuator states must follow each physical substep"
            )
        if not np.allclose(
            substep.ballast_snapshot.actual_tank_masses_kg,
            expected_state.actual_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE,
        ):
            raise ValueError(
                "physical_path tank snapshots must follow the actuator state at each substep"
            )
        expected_snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=expected_state.actual_masses_kg,
            reference_tank_masses_kg=(
                trajectory.platform_snapshot.reference_tank_masses_kg
            ),
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
        )
        if not _snapshots_have_same_dynamics(
            substep.ballast_snapshot,
            expected_snapshot,
        ):
            raise ValueError(
                "physical_path tank snapshot must reconstruct from the configured "
                "runtime assembly and forecast tank geometry"
            )
        for channel in ("wind", "wave", "other"):
            if not np.allclose(
                getattr(substep.platform_step.loads, channel),
                getattr(first_loads, channel),
                rtol=0.0,
                atol=_LOAD_TOLERANCE,
            ):
                raise ValueError(
                    "physical_path must retain the first forecast interval external loads"
                )
        if (
            substep.execution_step.target_operation
            is not lifecycle_trace.execution_request.operation
        ):
            raise ValueError(
                "physical_path target operation must match the lifecycle trace"
            )
        if lifecycle_trace.execution_request.operation is ExecutionTargetOperation.TRACK:
            if not np.allclose(
                substep.execution_step.requested_target_kg,
                lifecycle_trace.execution_step.requested_target_kg,
                rtol=0.0,
                atol=_MASS_TOLERANCE,
            ):
                raise ValueError(
                    "physical_path track target must match the lifecycle trace"
                )
        elif not np.allclose(
            substep.execution_step.requested_target_kg,
            substep.start_execution_state.actual_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE,
        ):
            raise ValueError(
                "physical_path release target must use the substep start masses"
            )
        expected_state = substep.execution_step.state

    if not _execution_states_match(
        physical_path.final_execution_state,
        lifecycle_trace.execution_step.state,
    ):
        raise ValueError(
            "physical_path final actuator state must match the lifecycle trace"
        )

    execution_step = lifecycle_trace.execution_step
    if not np.allclose(
        physical_path.actual_tank_mass_delta_kg,
        execution_step.mass_delta_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE,
    ):
        raise ValueError(
            "physical_path tank-mass change must match the lifecycle actuator result"
        )
    if not np.isclose(
        physical_path.transferred_volume_m3,
        execution_step.transferred_volume_m3,
        rtol=0.0,
        atol=_VOLUME_TOLERANCE,
    ):
        raise ValueError(
            "physical_path transferred volume must match the lifecycle actuator result"
        )
    if not np.isclose(
        physical_path.aggregate_pump_active_time_s,
        execution_step.active_time_s,
        rtol=0.0,
        atol=_TIME_TOLERANCE,
    ):
        raise ValueError(
            "physical_path active pump time must match the lifecycle actuator result"
        )
    if physical_path.pump_start_count != execution_step.starts:
        raise ValueError("physical_path pump starts must match the lifecycle actuator result")
    if physical_path.pump_stop_count != execution_step.stops:
        raise ValueError("physical_path pump stops must match the lifecycle actuator result")
    if physical_path.pump_direction_switch_count != execution_step.direction_switches:
        raise ValueError(
            "physical_path pump direction switches must match the lifecycle actuator result"
        )

    for substep in physical_path.substeps:
        expected_execution_step = simulate_execution_step(
            substep.start_execution_state,
            lifecycle_trace.execution_request,
            replace(
                lifecycle_trace.execution_config,
                block_duration_s=substep.duration_s,
            ),
        )
        if not _execution_steps_match(substep.execution_step, expected_execution_step):
            raise ValueError(
                "physical_path actuator outcome must follow the execution state and request"
            )


@dataclass(frozen=True)
class FirstForecastIntervalLifecycleRollout:
    """Executed first interval beside the matching frozen-tank baseline.

    The baseline uses the existing right-endpoint held forecast load with no
    tank change.  ``physical_path`` uses that same external load while tank
    masses change through the actuator trace.  The difference is retained as
    state facts only; it carries no improvement, risk or action-quality label.
    """

    trajectory: ForecastPlatformTrajectory
    lifecycle_trace: PhysicalTargetLifecycleTrace
    runtime_assembly: BallastRuntimeAssembly
    physical_path: PhysicalExecutionPlatformPath
    frozen_tank_baseline_end_state: IncrementalState
    position_delta_from_frozen_tank_baseline: Any
    velocity_delta_from_frozen_tank_baseline: Any

    def __post_init__(self) -> None:
        duration = _validate_lifecycle_at_first_interval(
            trajectory=self.trajectory,
            lifecycle_trace=self.lifecycle_trace,
            runtime_assembly=self.runtime_assembly,
        )
        if not isinstance(self.physical_path, PhysicalExecutionPlatformPath):
            raise TypeError("physical_path must be PhysicalExecutionPlatformPath")
        baseline_end = self.trajectory.steps[0].end_state
        if self.frozen_tank_baseline_end_state is not baseline_end:
            raise ValueError(
                "frozen_tank_baseline_end_state must be the first trajectory endpoint"
            )
        if not np.isclose(
            self.physical_path.duration_s,
            duration,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("physical_path duration must match the first forecast interval")

        _validate_path_facts(
            trajectory=self.trajectory,
            lifecycle_trace=self.lifecycle_trace,
            runtime_assembly=self.runtime_assembly,
            physical_path=self.physical_path,
        )

        position_delta = _readonly_six(
            "position_delta_from_frozen_tank_baseline",
            self.position_delta_from_frozen_tank_baseline,
        )
        velocity_delta = _readonly_six(
            "velocity_delta_from_frozen_tank_baseline",
            self.velocity_delta_from_frozen_tank_baseline,
        )
        expected_position_delta = (
            self.physical_path.final_platform_state.position - baseline_end.position
        )
        expected_velocity_delta = (
            self.physical_path.final_platform_state.velocity - baseline_end.velocity
        )
        if not np.allclose(
            position_delta,
            expected_position_delta,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ) or not np.allclose(
            velocity_delta,
            expected_velocity_delta,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ):
            raise ValueError(
                "state deltas must use the matching frozen-tank baseline endpoint"
            )
        object.__setattr__(
            self,
            "position_delta_from_frozen_tank_baseline",
            position_delta,
        )
        object.__setattr__(
            self,
            "velocity_delta_from_frozen_tank_baseline",
            velocity_delta,
        )

    @property
    def execution_coupled_end_state(self) -> IncrementalState:
        """Platform state after the executed first-interval pump path."""

        return self.physical_path.final_platform_state

    @property
    def reached_final_tank_masses_kg(self) -> np.ndarray:
        """Tank masses reached by the execution model at interval end."""

        return np.array(self.physical_path.final_execution_state.actual_masses_kg, copy=True)

    @property
    def lifecycle(self):
        """Return the supplied target-lifecycle meaning without selecting it."""

        return self.lifecycle_trace.lifecycle

    @property
    def requested_target_masses_kg(self) -> np.ndarray:
        """Return the target water masses given to the actuator for this replay."""

        return self.lifecycle_trace.requested_target_masses_kg

    @property
    def remaining_mass_to_target_kg(self) -> np.ndarray:
        """Return the actuator's remaining target error at interval end."""

        return self.lifecycle_trace.remaining_mass_to_target_kg

    @property
    def target_reached(self) -> bool:
        """Return whether the actuator reached its requested target in this interval."""

        return bool(self.lifecycle_trace.execution_step.target_reached)

    @property
    def actual_tank_mass_delta_kg(self) -> np.ndarray:
        """Return actual tank-mass change across the first forecast interval."""

        return self.physical_path.actual_tank_mass_delta_kg

    @property
    def transferred_volume_m3(self) -> float:
        """Return cumulative absolute pump volume over the first forecast interval."""

        return self.physical_path.transferred_volume_m3

    @property
    def aggregate_pump_active_time_s(self) -> float:
        """Return cumulative three-pump active time, not elapsed wall-clock time."""

        return self.physical_path.aggregate_pump_active_time_s

    @property
    def pump_start_count(self) -> int:
        """Return pump starts across the first forecast interval."""

        return self.physical_path.pump_start_count

    @property
    def pump_stop_count(self) -> int:
        """Return pump stops across the first forecast interval."""

        return self.physical_path.pump_stop_count

    @property
    def pump_direction_switch_count(self) -> int:
        """Return pump direction changes across the first forecast interval."""

        return self.physical_path.pump_direction_switch_count


def rollout_first_forecast_interval_lifecycle(
    *,
    trajectory: ForecastPlatformTrajectory,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
    runtime_assembly: BallastRuntimeAssembly,
) -> FirstForecastIntervalLifecycleRollout:
    """Replay one target lifecycle over the first frozen forecast interval.

    The function takes no score, candidate set, safety thresholds or policy
    flags.  It only applies the supplied lifecycle trace to the source-bound
    first interval and retains the matching fixed-tank trajectory endpoint.
    """

    duration = _validate_lifecycle_at_first_interval(
        trajectory=trajectory,
        lifecycle_trace=lifecycle_trace,
        runtime_assembly=runtime_assembly,
    )
    first_step = trajectory.steps[0]
    physical_path = advance_physical_execution_platform_path(
        platform_state=trajectory.initial_state,
        execution_state=lifecycle_trace.execution_start_state,
        execution_request=lifecycle_trace.execution_request,
        execution_config=lifecycle_trace.execution_config,
        runtime_assembly=runtime_assembly,
        reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
        tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
        tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
        rotor_load=first_step.rotor_load,
        wave_load=first_step.wave_load,
        other_load=first_step.other_load,
        duration_s=duration,
    )
    baseline_end = first_step.end_state
    return FirstForecastIntervalLifecycleRollout(
        trajectory=trajectory,
        lifecycle_trace=lifecycle_trace,
        runtime_assembly=runtime_assembly,
        physical_path=physical_path,
        frozen_tank_baseline_end_state=baseline_end,
        position_delta_from_frozen_tank_baseline=(
            physical_path.final_platform_state.position - baseline_end.position
        ),
        velocity_delta_from_frozen_tank_baseline=(
            physical_path.final_platform_state.velocity - baseline_end.velocity
        ),
    )


__all__ = [
    "FirstForecastIntervalLifecycleRollout",
    "rollout_first_forecast_interval_lifecycle",
]
