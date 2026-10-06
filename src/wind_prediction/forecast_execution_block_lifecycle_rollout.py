"""Replay declared lifecycle requests through bounded forecast counterfactuals.

The controller may receive several discrete forecast-load intervals before its
next target update.  This module holds one already declared lifecycle request
only for that currently configured execution block, advancing the low-order
platform and the actual pump state interval by interval.  It neither creates a
new target from later forecast points nor selects between lifecycle operations.

The current-block result is consequently a bounded open-loop counterfactual.
This module also offers a separately named held-request forecast-horizon trace:
it extends an already completed current-block prefix with the same declared
request.  That trace is only a conditional forecast fact.  It is not a
multi-cycle controller, does not create later targets, and cannot be used as
the state handoff for the next real control period.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from fowt_platform.ballast_snapshot import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from fowt_platform.incremental import IncrementalState

from .execution_rollout import (
    ExecutionRolloutState,
    ExecutionRolloutStep,
    ExecutionTargetOperation,
    simulate_execution_step,
)
from .forecast_first_interval_lifecycle_rollout import (
    FirstForecastIntervalLifecycleRollout,
    rollout_first_forecast_interval_lifecycle,
)
from .forecast_platform_trajectory import (
    ForecastPlatformTrajectory,
    ForecastPlatformTrajectoryStep,
)
from .physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    advance_physical_execution_platform_path,
)
from .physical_endpoint_preview_binding import (
    validate_endpoint_preview_at_trajectory_lead,
)
from .physical_target_lifecycle import PhysicalTargetLifecycleTrace


_STATE_TOLERANCE = 1.0e-10
_MASS_TOLERANCE_KG = 1.0e-8
_TIME_TOLERANCE_S = 1.0e-9
_LOAD_TOLERANCE = 1.0e-7
_VOLUME_TOLERANCE_M3 = 1.0e-10
_SUBSTEP_SPLIT_TOLERANCE_S = 1.0e-12
_PITCH_ROLL_INDICES = (4, 3)


def _readonly_vector(value: object) -> np.ndarray:
    """Return a detached, read-only floating-point vector."""

    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _states_match(left: IncrementalState, right: IncrementalState) -> bool:
    return bool(
        np.allclose(
            left.position,
            right.position,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        )
        and np.allclose(
            left.velocity,
            right.velocity,
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        )
    )


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


def _path_uses_trajectory_loads(
    path: PhysicalExecutionPlatformPath,
    step: ForecastPlatformTrajectoryStep,
) -> bool:
    return all(
        np.allclose(
            substep.platform_step.loads.wind,
            step.rotor_load.generalized_load_platform,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        )
        and np.allclose(
            substep.platform_step.loads.wave,
            step.wave_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        )
        and np.allclose(
            substep.platform_step.loads.other,
            step.other_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        )
        for substep in path.substeps
    )


def _execution_steps_match(
    left: ExecutionRolloutStep,
    right: ExecutionRolloutStep,
) -> bool:
    """Return whether one stored actuator step follows the declared request."""

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
            atol=_MASS_TOLERANCE_KG,
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
            atol=_VOLUME_TOLERANCE_M3,
        )
        and np.isclose(
            left.active_time_s,
            right.active_time_s,
            rtol=0.0,
            atol=_TIME_TOLERANCE_S,
        )
    )


def _snapshots_have_same_dynamics(
    left: BallastModelSnapshot,
    right: BallastModelSnapshot,
) -> bool:
    """Compare the complete frozen model reconstructed from one pump state."""

    for attribute, tolerance in (
        ("actual_tank_masses_kg", _MASS_TOLERANCE_KG),
        ("reference_tank_masses_kg", _MASS_TOLERANCE_KG),
        ("tank_capacities_kg", _MASS_TOLERANCE_KG),
        ("tank_coordinates_m", _STATE_TOLERANCE),
        ("incremental_ballast_load", _LOAD_TOLERANCE),
    ):
        if not np.allclose(
            getattr(left, attribute),
            getattr(right, attribute),
            rtol=0.0,
            atol=tolerance,
        ):
            return False
    if (
        not math.isclose(
            left.gravity_m_s2,
            right.gravity_m_s2,
            rel_tol=0.0,
            abs_tol=_STATE_TOLERANCE,
        )
        or left.runtime_provenance != right.runtime_provenance
    ):
        return False
    for attribute in (
        "mass",
        "damping",
        "hydrostatic_stiffness",
        "mooring_stiffness",
        "weight_stiffness",
    ):
        if not np.allclose(
            getattr(left.matrices, attribute),
            getattr(right.matrices, attribute),
            rtol=0.0,
            atol=_LOAD_TOLERANCE,
        ):
            return False
    return True


def _configured_substep_durations(
    *,
    duration_s: float,
    internal_step_s: float,
) -> tuple[float, ...]:
    """Return the execution split required by the shared pump model."""

    remaining = float(duration_s)
    durations: list[float] = []
    # This must match the physical-path generator exactly.  The record
    # validator must not reject a numerically valid final substep it generated.
    while remaining > _SUBSTEP_SPLIT_TOLERANCE_S:
        duration = min(float(internal_step_s), remaining)
        durations.append(duration)
        remaining -= duration
    return tuple(durations)


def _validate_interval_lifecycle_path(
    *,
    interval: "ForecastExecutionBlockInterval",
    trajectory: ForecastPlatformTrajectory,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
    runtime_assembly: BallastRuntimeAssembly,
) -> None:
    """Bind every stored substep to the declared request and runtime assembly."""

    path = interval.physical_path
    expected_durations = _configured_substep_durations(
        duration_s=path.duration_s,
        internal_step_s=lifecycle_trace.execution_config.internal_step_s,
    )
    if len(path.substeps) != len(expected_durations):
        raise ValueError("execution-block path must retain the configured physical substeps")
    if not _path_uses_trajectory_loads(path, interval.trajectory_step):
        raise ValueError("execution-block path must retain its forecast interval loads")

    expected_state = path.substeps[0].start_execution_state
    snapshot_source = trajectory.platform_snapshot
    for substep, expected_duration in zip(path.substeps, expected_durations):
        if not math.isclose(
            substep.duration_s,
            expected_duration,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError("execution-block path must retain the configured physical substeps")
        if not _execution_states_match(substep.start_execution_state, expected_state):
            raise ValueError("execution-block pump states must remain continuous")
        expected_snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=expected_state.actual_masses_kg,
            reference_tank_masses_kg=snapshot_source.reference_tank_masses_kg,
            tank_capacities_kg=snapshot_source.tank_capacities_kg,
            tank_coordinates_m=snapshot_source.tank_coordinates_m,
        )
        if not _snapshots_have_same_dynamics(
            substep.ballast_snapshot,
            expected_snapshot,
        ):
            raise ValueError(
                "execution-block tank snapshots must reconstruct from the supplied runtime assembly"
            )
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
                "execution-block actuator steps must retain the declared lifecycle request"
            )
        expected_state = substep.execution_step.state


@dataclass(frozen=True)
class ForecastExecutionBlockInterval:
    """One held-request path over one forecast-load interval or its prefix."""

    trajectory_step: ForecastPlatformTrajectoryStep
    start_time_s: float
    end_time_s: float
    physical_path: PhysicalExecutionPlatformPath

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory_step, ForecastPlatformTrajectoryStep):
            raise TypeError("trajectory_step must be ForecastPlatformTrajectoryStep")
        if not isinstance(self.physical_path, PhysicalExecutionPlatformPath):
            raise TypeError("physical_path must be PhysicalExecutionPlatformPath")
        start = float(self.start_time_s)
        end = float(self.end_time_s)
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("interval times must form a positive finite duration")
        step = self.trajectory_step
        if (
            start < step.start_time_s - _TIME_TOLERANCE_S
            or end > step.end_time_s + _TIME_TOLERANCE_S
        ):
            raise ValueError("execution-block interval must remain within one forecast step")
        if not math.isclose(
            self.physical_path.duration_s,
            end - start,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError("physical_path duration must match the execution-block interval")
        if not _path_uses_trajectory_loads(self.physical_path, step):
            raise ValueError(
                "physical_path external loads must retain the matching forecast step"
            )
        object.__setattr__(self, "start_time_s", start)
        object.__setattr__(self, "end_time_s", end)

    @property
    def lead_index(self) -> int:
        """Return the source-bound forecast-load interval index."""

        return int(self.trajectory_step.lead_index)

    @property
    def final_platform_state(self) -> IncrementalState:
        """Return the controlled low-order state at this interval end."""

        return self.physical_path.final_platform_state

    @property
    def final_execution_state(self) -> ExecutionRolloutState:
        """Return the actual pump state at this interval end."""

        return self.physical_path.final_execution_state


@dataclass(frozen=True)
class ForecastExecutionBlockLifecycleRollout:
    """One lifecycle request held through the current execution block only."""

    trajectory: ForecastPlatformTrajectory
    lifecycle_trace: PhysicalTargetLifecycleTrace
    runtime_assembly: BallastRuntimeAssembly
    intervals: tuple[ForecastExecutionBlockInterval, ...]
    duration_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ForecastPlatformTrajectory):
            raise TypeError("trajectory must be ForecastPlatformTrajectory")
        if not isinstance(self.lifecycle_trace, PhysicalTargetLifecycleTrace):
            raise TypeError("lifecycle_trace must be PhysicalTargetLifecycleTrace")
        if not isinstance(self.runtime_assembly, BallastRuntimeAssembly):
            raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
        intervals = tuple(self.intervals)
        if not intervals:
            raise ValueError("intervals must contain the first forecast interval")
        if not all(isinstance(interval, ForecastExecutionBlockInterval) for interval in intervals):
            raise TypeError("intervals must contain ForecastExecutionBlockInterval values")

        expected_duration = min(
            float(self.lifecycle_trace.execution_config.block_duration_s),
            float(self.trajectory.steps[-1].end_time_s),
        )
        duration = float(self.duration_s)
        if not math.isclose(
            duration,
            expected_duration,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError(
                "duration_s must cover the current execution block or available forecast horizon"
            )
        first = intervals[0]
        if first.trajectory_step is not self.trajectory.steps[0]:
            raise ValueError("the first execution-block interval must use the first forecast step")
        if not math.isclose(first.start_time_s, 0.0, rel_tol=0.0, abs_tol=_TIME_TOLERANCE_S):
            raise ValueError("the first execution-block interval must begin at the forecast origin")
        if not math.isclose(
            first.end_time_s,
            self.trajectory.steps[0].end_time_s,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError("the first execution-block interval must cover the first forecast lead")
        if not _states_match(
            first.physical_path.substeps[0].start_platform_state,
            self.trajectory.initial_state,
        ):
            raise ValueError("the first execution-block interval must start at the trajectory origin state")
        if not _execution_states_match(
            first.physical_path.substeps[0].start_execution_state,
            self.lifecycle_trace.execution_start_state,
        ):
            raise ValueError("the first execution-block interval must start at the lifecycle pump state")
        if not _execution_states_match(
            first.final_execution_state,
            self.lifecycle_trace.execution_step.state,
        ):
            raise ValueError(
                "the first execution-block interval must reproduce the lifecycle pump result"
            )

        for interval in intervals:
            _validate_interval_lifecycle_path(
                interval=interval,
                trajectory=self.trajectory,
                lifecycle_trace=self.lifecycle_trace,
                runtime_assembly=self.runtime_assembly,
            )

        previous = first
        for index, interval in enumerate(intervals[1:], start=1):
            if interval.trajectory_step is not self.trajectory.steps[index]:
                raise ValueError("execution-block intervals must use contiguous forecast steps")
            if not math.isclose(
                interval.start_time_s,
                previous.end_time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ):
                raise ValueError("execution-block intervals must remain time-contiguous")
            if not _states_match(
                interval.physical_path.substeps[0].start_platform_state,
                previous.final_platform_state,
            ):
                raise ValueError("controlled platform states must remain continuous across intervals")
            if not _execution_states_match(
                interval.physical_path.substeps[0].start_execution_state,
                previous.final_execution_state,
            ):
                raise ValueError("pump states must remain continuous across intervals")
            previous = interval

        if not math.isclose(
            intervals[-1].end_time_s,
            duration,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError("execution-block intervals must cover the declared duration")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "duration_s", duration)

    @property
    def final_platform_state(self) -> IncrementalState:
        """Return the state after the currently held control block."""

        return self.intervals[-1].final_platform_state

    @property
    def start_platform_state(self) -> IncrementalState:
        """Return the common platform state at the control-block origin."""

        return self.intervals[0].physical_path.substeps[0].start_platform_state

    @property
    def final_execution_state(self) -> ExecutionRolloutState:
        """Return the pump state after the currently held control block."""

        return self.intervals[-1].final_execution_state

    @property
    def start_execution_state(self) -> ExecutionRolloutState:
        """Return the complete pump state at the control-block origin."""

        return self.intervals[0].physical_path.substeps[0].start_execution_state

    @property
    def reached_final_tank_masses_kg(self) -> np.ndarray:
        """Return the actual tank masses at the end of the held block."""

        return _readonly_vector(self.final_execution_state.actual_masses_kg)

    @property
    def block_end_requested_target_masses_kg(self) -> np.ndarray:
        """Return the final physical substep's request target.

        ``track`` operations retain one declared target across the block, while
        ``release_to_current`` reconstructs its request from each substep's
        modeled actual tank state. The block-end request is therefore the only
        target representation that has the same meaning for every lifecycle.
        """

        return _readonly_vector(
            self.intervals[-1].physical_path.substeps[-1].execution_step.requested_target_kg
        )

    @property
    def block_end_remaining_mass_to_target_kg(self) -> np.ndarray:
        """Return the remaining difference to the final physical substep request."""

        return _readonly_vector(
            self.intervals[-1].physical_path.substeps[-1].execution_step.requested_target_kg
            - self.final_execution_state.actual_masses_kg
        )

    @property
    def actual_tank_mass_delta_kg(self) -> np.ndarray:
        """Return actual three-tank mass changes across the held block."""

        return _readonly_vector(
            self.final_execution_state.actual_masses_kg
            - self.start_execution_state.actual_masses_kg
        )

    @property
    def post_origin_peak_abs_pitch_roll_rad(self) -> np.ndarray:
        """Return post-origin pitch/roll angle peaks in fixed ``(pitch, roll)`` order.

        This is a path fact under the frozen execution-block counterfactual.  It
        is not a safety result or a candidate score.
        """

        positions = np.asarray(
            [
                substep.next_platform_state.position[list(_PITCH_ROLL_INDICES)]
                for interval in self.intervals
                for substep in interval.physical_path.substeps
            ],
            dtype=float,
        )
        return _readonly_vector(np.max(np.abs(positions), axis=0))

    @property
    def post_origin_peak_abs_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return post-origin pitch/roll rate peaks in fixed ``(pitch, roll)`` order.

        This is a path fact under the frozen execution-block counterfactual.  It
        is not a safety result or a candidate score.
        """

        velocities = np.asarray(
            [
                substep.next_platform_state.velocity[list(_PITCH_ROLL_INDICES)]
                for interval in self.intervals
                for substep in interval.physical_path.substeps
            ],
            dtype=float,
        )
        return _readonly_vector(np.max(np.abs(velocities), axis=0))

    @property
    def lifecycle(self):
        """Return the declared operation without selecting or re-planning it."""

        return self.lifecycle_trace.lifecycle


@dataclass(frozen=True)
class HeldLifecycleForecastHorizonRollout:
    """Hold one already declared request through the available forecast horizon.

    ``prefix`` is the completed real-current-block counterfactual.  The
    continuation intervals begin at that exact block boundary and hold the same
    declared request under the remaining source-bound forecast loads.  This
    record is deliberately not a rolling plan: it does not create later target
    updates, choose an operation, or provide a state handoff for the next real
    control period.
    """

    prefix: ForecastExecutionBlockLifecycleRollout
    continuation_intervals: tuple[ForecastExecutionBlockInterval, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, ForecastExecutionBlockLifecycleRollout):
            raise TypeError("prefix must be ForecastExecutionBlockLifecycleRollout")
        continuation = tuple(self.continuation_intervals)
        if not all(
            isinstance(interval, ForecastExecutionBlockInterval)
            for interval in continuation
        ):
            raise TypeError(
                "continuation_intervals must contain ForecastExecutionBlockInterval values"
            )

        trajectory = self.prefix.trajectory
        prefix_end_index = next(
            (
                index
                for index, step in enumerate(trajectory.steps)
                if math.isclose(
                    step.end_time_s,
                    self.prefix.duration_s,
                    rel_tol=0.0,
                    abs_tol=_TIME_TOLERANCE_S,
                )
            ),
            None,
        )
        if prefix_end_index is None:
            raise ValueError("prefix must end at a forecast-lead boundary")
        if self.prefix.intervals[-1].trajectory_step is not trajectory.steps[prefix_end_index]:
            raise ValueError("prefix must end at its matching forecast-lead boundary")

        expected_steps = trajectory.steps[prefix_end_index + 1 :]
        if len(continuation) != len(expected_steps):
            raise ValueError(
                "continuation_intervals must cover every remaining forecast lead"
            )

        previous_platform_state = self.prefix.final_platform_state
        previous_execution_state = self.prefix.final_execution_state
        previous_end_time_s = float(self.prefix.duration_s)
        execution_config = self.prefix.lifecycle_trace.execution_config
        for step, interval in zip(expected_steps, continuation):
            if interval.trajectory_step is not step:
                raise ValueError(
                    "continuation intervals must retain contiguous trajectory steps"
                )
            if not math.isclose(
                interval.start_time_s,
                previous_end_time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ) or not math.isclose(
                interval.end_time_s,
                step.end_time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ):
                raise ValueError(
                    "continuation intervals must cover complete, contiguous forecast leads"
                )
            if step.duration_s > float(execution_config.block_duration_s) + _TIME_TOLERANCE_S:
                raise ValueError(
                    "each remaining forecast lead must fit within one execution block"
                )
            if not _states_match(
                interval.physical_path.substeps[0].start_platform_state,
                previous_platform_state,
            ):
                raise ValueError(
                    "held-request platform states must remain continuous"
                )
            if not _execution_states_match(
                interval.physical_path.substeps[0].start_execution_state,
                previous_execution_state,
            ):
                raise ValueError(
                    "held-request pump states must remain continuous"
                )
            _validate_interval_lifecycle_path(
                interval=interval,
                trajectory=trajectory,
                lifecycle_trace=self.prefix.lifecycle_trace,
                runtime_assembly=self.prefix.runtime_assembly,
            )
            previous_platform_state = interval.final_platform_state
            previous_execution_state = interval.final_execution_state
            previous_end_time_s = interval.end_time_s

        if not math.isclose(
            previous_end_time_s,
            trajectory.steps[-1].end_time_s,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError("held-request rollout must end at the forecast horizon")
        object.__setattr__(self, "continuation_intervals", continuation)

    @property
    def intervals(self) -> tuple[ForecastExecutionBlockInterval, ...]:
        """Return the current-block prefix followed by held-request intervals."""

        return (*self.prefix.intervals, *self.continuation_intervals)

    @property
    def duration_s(self) -> float:
        """Return the full source-bound forecast duration covered by this trace."""

        return float(self.intervals[-1].end_time_s)

    @property
    def final_platform_state(self) -> IncrementalState:
        """Return the modeled platform state at the forecast-horizon end."""

        return self.intervals[-1].final_platform_state

    @property
    def final_execution_state(self) -> ExecutionRolloutState:
        """Return the modeled pump state at the forecast-horizon end."""

        return self.intervals[-1].final_execution_state

    @property
    def lifecycle(self):
        """Return the one request held by this counterfactual, without choosing it."""

        return self.prefix.lifecycle

    @property
    def post_origin_peak_abs_pitch_roll_rad(self) -> np.ndarray:
        """Return modeled pitch/roll peaks across the held forecast horizon."""

        positions = np.asarray(
            [
                substep.next_platform_state.position[list(_PITCH_ROLL_INDICES)]
                for interval in self.intervals
                for substep in interval.physical_path.substeps
            ],
            dtype=float,
        )
        return _readonly_vector(np.max(np.abs(positions), axis=0))

    @property
    def post_origin_peak_abs_pitch_roll_rate_rad_s(self) -> np.ndarray:
        """Return modeled pitch/roll-rate peaks across the held horizon."""

        velocities = np.asarray(
            [
                substep.next_platform_state.velocity[list(_PITCH_ROLL_INDICES)]
                for interval in self.intervals
                for substep in interval.physical_path.substeps
            ],
            dtype=float,
        )
        return _readonly_vector(np.max(np.abs(velocities), axis=0))


def rollout_current_execution_block_lifecycle(
    *,
    trajectory: ForecastPlatformTrajectory,
    lifecycle_trace: PhysicalTargetLifecycleTrace,
    runtime_assembly: BallastRuntimeAssembly,
) -> ForecastExecutionBlockLifecycleRollout:
    """Hold one lifecycle request only until the next configured control block.

    A first-lead endpoint uses the established first-interval replay.  A later
    forecast endpoint may also be projected explicitly to the current control
    origin, but it still uses the *first* interval's external loads while the
    supplied target is executed now.  Subsequent intervals only continue that
    already declared request inside the same control block; they do not issue
    a new target update.
    """

    source_preview = lifecycle_trace.source_preview
    first_step = trajectory.steps[0]
    if source_preview is None or source_preview.lead_index == 0:
        first_path = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=lifecycle_trace,
            runtime_assembly=runtime_assembly,
        ).physical_path
    else:
        # A later endpoint is a declared current-block counterfactual, not an
        # instruction to begin pumping at that future time.  Bind it to its
        # own source lead, then execute its one current request under the
        # current block's first forecast-load interval.
        validate_endpoint_preview_at_trajectory_lead(
            preview=source_preview,
            trajectory=trajectory,
            lead_index=source_preview.lead_index,
            lead_time_s=source_preview.lead_time_s,
            first_lead_only=False,
        )
        if not math.isclose(
            lifecycle_trace.execution_duration_s,
            first_step.duration_s,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError(
                "projected endpoint execution duration must match the first forecast interval"
            )
        first_path = advance_physical_execution_platform_path(
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
            duration_s=first_step.duration_s,
        )
    duration_limit = min(
        float(lifecycle_trace.execution_config.block_duration_s),
        float(trajectory.steps[-1].end_time_s),
    )
    intervals: list[ForecastExecutionBlockInterval] = [
        ForecastExecutionBlockInterval(
            trajectory_step=first_step,
            start_time_s=0.0,
            end_time_s=first_step.end_time_s,
            physical_path=first_path,
        )
    ]
    platform_state = first_path.final_platform_state
    execution_state = first_path.final_execution_state
    current_time_s = float(first_step.end_time_s)

    for step in trajectory.steps[1:]:
        if current_time_s >= duration_limit - _TIME_TOLERANCE_S:
            break
        end_time_s = min(float(step.end_time_s), duration_limit)
        interval_duration = end_time_s - current_time_s
        if interval_duration <= _TIME_TOLERANCE_S:
            continue
        path = advance_physical_execution_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=lifecycle_trace.execution_request,
            execution_config=lifecycle_trace.execution_config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=(
                trajectory.platform_snapshot.reference_tank_masses_kg
            ),
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=step.rotor_load,
            wave_load=step.wave_load,
            other_load=step.other_load,
            duration_s=interval_duration,
        )
        interval = ForecastExecutionBlockInterval(
            trajectory_step=step,
            start_time_s=current_time_s,
            end_time_s=end_time_s,
            physical_path=path,
        )
        intervals.append(interval)
        platform_state = interval.final_platform_state
        execution_state = interval.final_execution_state
        current_time_s = end_time_s

    return ForecastExecutionBlockLifecycleRollout(
        trajectory=trajectory,
        lifecycle_trace=lifecycle_trace,
        runtime_assembly=runtime_assembly,
        intervals=tuple(intervals),
        duration_s=duration_limit,
    )


def rollout_held_lifecycle_to_forecast_horizon(
    *,
    prefix: ForecastExecutionBlockLifecycleRollout,
) -> HeldLifecycleForecastHorizonRollout:
    """Extend a completed current-block prefix with the same declared request.

    The returned trace remains a held-request counterfactual.  It intentionally
    does not issue a target at later forecast leads and must not be used as the
    state source for actual rolling control.
    """

    if not isinstance(prefix, ForecastExecutionBlockLifecycleRollout):
        raise TypeError("prefix must be ForecastExecutionBlockLifecycleRollout")
    trajectory = prefix.trajectory
    prefix_end_index = next(
        (
            index
            for index, step in enumerate(trajectory.steps)
            if math.isclose(
                step.end_time_s,
                prefix.duration_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            )
        ),
        None,
    )
    if prefix_end_index is None:
        raise ValueError("prefix must end at a forecast-lead boundary")

    continuation: list[ForecastExecutionBlockInterval] = []
    platform_state = prefix.final_platform_state
    execution_state = prefix.final_execution_state
    current_time_s = float(prefix.duration_s)
    lifecycle_trace = prefix.lifecycle_trace
    for step in trajectory.steps[prefix_end_index + 1 :]:
        interval_duration = float(step.end_time_s - current_time_s)
        if interval_duration <= _TIME_TOLERANCE_S:
            raise ValueError("remaining forecast leads must have positive durations")
        if interval_duration > float(lifecycle_trace.execution_config.block_duration_s) + _TIME_TOLERANCE_S:
            raise ValueError(
                "each remaining forecast lead must fit within one execution block"
            )
        path = advance_physical_execution_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=lifecycle_trace.execution_request,
            execution_config=lifecycle_trace.execution_config,
            runtime_assembly=prefix.runtime_assembly,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=step.rotor_load,
            wave_load=step.wave_load,
            other_load=step.other_load,
            duration_s=interval_duration,
        )
        interval = ForecastExecutionBlockInterval(
            trajectory_step=step,
            start_time_s=current_time_s,
            end_time_s=step.end_time_s,
            physical_path=path,
        )
        continuation.append(interval)
        platform_state = interval.final_platform_state
        execution_state = interval.final_execution_state
        current_time_s = interval.end_time_s

    return HeldLifecycleForecastHorizonRollout(
        prefix=prefix,
        continuation_intervals=tuple(continuation),
    )


__all__ = [
    "ForecastExecutionBlockInterval",
    "ForecastExecutionBlockLifecycleRollout",
    "HeldLifecycleForecastHorizonRollout",
    "rollout_current_execution_block_lifecycle",
    "rollout_held_lifecycle_to_forecast_horizon",
]
