"""Causally ordered pump execution and low-order platform substeps.

This module advances one already committed actuator request through the
low-order platform model.  It does not generate a target, evaluate a
candidate, infer safety, or use forecast information.  The platform first
uses the actual tank state at a substep start, then the pumps advance over the
same duration.  This is a frozen splitting approximation, not a continuous
variable-mass model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from fowt_platform.incremental import IncrementalPlatformModel, IncrementalState
from fowt_platform.open_loop_step import (
    OpenLoopPlatformStep,
    advance_frozen_open_loop_step,
)
from fowt_platform.rotor_input import RotorGeneralizedLoad

from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionRolloutStep,
    simulate_execution_step,
)


@dataclass(frozen=True)
class PhysicalExecutionPlatformSubstep:
    """One platform step followed by the matching pump-execution step."""

    duration_s: float
    ballast_snapshot: BallastModelSnapshot
    start_platform_state: IncrementalState
    start_execution_state: ExecutionRolloutState
    platform_step: OpenLoopPlatformStep
    execution_step: ExecutionRolloutStep

    def __post_init__(self) -> None:
        duration = _validated_duration(self.duration_s)
        if not isinstance(self.ballast_snapshot, BallastModelSnapshot):
            raise TypeError("ballast_snapshot must be BallastModelSnapshot")
        if not isinstance(self.start_platform_state, IncrementalState):
            raise TypeError("start_platform_state must be IncrementalState")
        if not isinstance(self.start_execution_state, ExecutionRolloutState):
            raise TypeError("start_execution_state must be ExecutionRolloutState")
        if not isinstance(self.platform_step, OpenLoopPlatformStep):
            raise TypeError("platform_step must be OpenLoopPlatformStep")
        if not isinstance(self.execution_step, ExecutionRolloutStep):
            raise TypeError("execution_step must be ExecutionRolloutStep")
        if not np.allclose(
            self.start_execution_state.actual_masses_kg,
            self.ballast_snapshot.actual_tank_masses_kg,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "substep pump start masses must match the ballast snapshot"
            )
        if not np.allclose(
            self.execution_step.state.actual_masses_kg
            - self.start_execution_state.actual_masses_kg,
            self.execution_step.mass_delta_kg,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError(
                "substep actuator outcome must start from the supplied pump state"
            )
        if not np.allclose(
            self.platform_step.loads.ballast,
            self.ballast_snapshot.incremental_ballast_load,
            rtol=0.0,
            atol=1e-7,
        ):
            raise ValueError(
                "platform step ballast load must match the substep tank snapshot"
            )
        expected_next_state = IncrementalPlatformModel(
            self.ballast_snapshot.matrices
        ).advance_frozen_step(
            self.start_platform_state,
            self.platform_step.loads,
            duration,
        )
        if not np.allclose(
            self.platform_step.next_state.position,
            expected_next_state.position,
            rtol=0.0,
            atol=1e-12,
        ) or not np.allclose(
            self.platform_step.next_state.velocity,
            expected_next_state.velocity,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "platform step must satisfy the frozen substep state equation"
            )
        object.__setattr__(self, "duration_s", duration)

    @property
    def next_platform_state(self) -> IncrementalState:
        """Platform state at the end of this physical substep."""

        return self.platform_step.next_state

    @property
    def next_execution_state(self) -> ExecutionRolloutState:
        """Actual pump state at the end of this physical substep."""

        return self.execution_step.state


@dataclass(frozen=True)
class PhysicalExecutionPlatformPath:
    """A short path assembled from causally ordered physical substeps.

    Every substep advances the platform from the actual tank masses at the
    beginning of that substep, then advances the committed pump request by the
    same duration.  It accepts one supplied request and deliberately does not
    select or revise a target.
    """

    substeps: tuple[PhysicalExecutionPlatformSubstep, ...]
    duration_s: float
    _final_execution_state: ExecutionRolloutState = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        try:
            duration = float(self.duration_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("duration_s must be positive and finite") from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be positive and finite")
        if not self.substeps:
            raise ValueError("substeps must contain at least one physical substep")
        if not all(
            isinstance(step, PhysicalExecutionPlatformSubstep)
            for step in self.substeps
        ):
            raise TypeError("substeps must contain PhysicalExecutionPlatformSubstep values")
        if not math.isclose(
            sum(step.duration_s for step in self.substeps),
            duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("substep durations must sum to duration_s")

        first_operation = self.substeps[0].execution_step.target_operation
        for previous, current in zip(self.substeps, self.substeps[1:]):
            if not np.allclose(
                current.start_platform_state.position,
                previous.next_platform_state.position,
                rtol=0.0,
                atol=1e-12,
            ) or not np.allclose(
                current.start_platform_state.velocity,
                previous.next_platform_state.velocity,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "platform states must form a continuous physical substep path"
                )
            if not np.allclose(
                current.ballast_snapshot.actual_tank_masses_kg,
                previous.next_execution_state.actual_masses_kg,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "tank snapshots must follow the preceding actuator state"
                )
            if not _execution_states_match(
                current.start_execution_state,
                previous.next_execution_state,
            ):
                raise ValueError(
                    "actuator states must form a continuous physical substep path"
                )
            if current.execution_step.target_operation is not first_operation:
                raise ValueError(
                    "physical substeps must retain one committed target operation"
                )
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "substeps", tuple(self.substeps))
        final_state = self.substeps[-1].next_execution_state
        object.__setattr__(
            self,
            "_final_execution_state",
            replace(
                final_state,
                previous_block_tank_mass_delta_kg=(
                    final_state.actual_masses_kg
                    - self.substeps[0].start_execution_state.actual_masses_kg
                ),
            ),
        )

    @property
    def final_platform_state(self) -> IncrementalState:
        """Low-order platform state after the final physical substep."""

        return self.substeps[-1].next_platform_state

    @property
    def final_execution_state(self) -> ExecutionRolloutState:
        """Actual actuator state with history measured over the whole path.

        Individual substeps retain their own mass delta so adjacent actuator
        states remain directly comparable.  The state exported to the next
        control block instead records the mass change from the beginning of
        this complete executed path.
        """

        return self._final_execution_state

    @property
    def start_actual_tank_masses_kg(self) -> np.ndarray:
        """Return the actual tank masses at the start of this executed path."""

        result = np.array(
            self.substeps[0].ballast_snapshot.actual_tank_masses_kg,
            dtype=float,
            copy=True,
        )
        result.setflags(write=False)
        return result

    @property
    def actual_tank_mass_delta_kg(self) -> np.ndarray:
        """Return actual three-tank mass change across the executed path."""

        result = np.array(
            self.substeps[-1].next_execution_state.actual_masses_kg
            - self.substeps[0].ballast_snapshot.actual_tank_masses_kg,
            dtype=float,
            copy=True,
        )
        result.setflags(write=False)
        return result

    @property
    def transferred_volume_m3(self) -> float:
        """Return cumulative absolute pump volume across all substeps and pumps."""

        return float(
            sum(step.execution_step.transferred_volume_m3 for step in self.substeps)
        )

    @property
    def aggregate_pump_active_time_s(self) -> float:
        """Return the sum of active seconds over all three pumps, not wall time."""

        return float(sum(step.execution_step.active_time_s for step in self.substeps))

    @property
    def pump_start_count(self) -> int:
        """Return starts observed while executing the supplied request."""

        return int(sum(step.execution_step.starts for step in self.substeps))

    @property
    def pump_stop_count(self) -> int:
        """Return stops observed while executing the supplied request."""

        return int(sum(step.execution_step.stops for step in self.substeps))

    @property
    def pump_direction_switch_count(self) -> int:
        """Return pump direction changes observed while executing the request."""

        return int(
            sum(step.execution_step.direction_switches for step in self.substeps)
        )


def _validated_duration(value: Any) -> float:
    try:
        duration = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_s must be positive and finite") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    return duration


def _execution_states_match(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
    """Return whether two pump states retain every execution-relevant value."""

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


def _validated_capacities(
    tank_capacities_kg: Any,
    execution_config: ExecutionRolloutConfig,
) -> np.ndarray:
    try:
        capacities = np.asarray(tank_capacities_kg, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tank_capacities_kg must contain exactly three finite positive values"
        ) from exc
    if (
        capacities.size != 3
        or not np.all(np.isfinite(capacities))
        or np.any(capacities <= 0.0)
    ):
        raise ValueError(
            "tank_capacities_kg must contain exactly three finite positive values"
        )
    if not np.allclose(
        capacities,
        float(execution_config.tank_capacity_kg),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError(
            "tank_capacities_kg must match execution_config.tank_capacity_kg "
            "for all three tanks"
        )
    return capacities


def advance_physical_execution_platform_substep(
    *,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    execution_request: ExecutionRolloutRequest,
    execution_config: ExecutionRolloutConfig,
    runtime_assembly: BallastRuntimeAssembly,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    rotor_load: RotorGeneralizedLoad,
    wave_load: Any,
    other_load: Any,
    duration_s: float,
) -> PhysicalExecutionPlatformSubstep:
    """Advance one frozen platform interval and one pump interval.

    The platform uses the start-of-step tank snapshot.  The actuator then
    advances the supplied request over the same interval.  The shared
    execution model has one capacity parameter, so all three platform tank
    capacities must match it.
    """

    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be an IncrementalState")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be an ExecutionRolloutState")
    if not isinstance(execution_request, ExecutionRolloutRequest):
        raise TypeError("execution_request must be an ExecutionRolloutRequest")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be an ExecutionRolloutConfig")
    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
    if not isinstance(rotor_load, RotorGeneralizedLoad):
        raise TypeError("rotor_load must be a RotorGeneralizedLoad")
    duration = _validated_duration(duration_s)
    capacities = _validated_capacities(tank_capacities_kg, execution_config)

    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=execution_state.actual_masses_kg,
        reference_tank_masses_kg=reference_tank_masses_kg,
        tank_capacities_kg=capacities,
        tank_coordinates_m=tank_coordinates_m,
    )
    platform_step = advance_frozen_open_loop_step(
        snapshot=snapshot,
        state=platform_state,
        rotor_load=rotor_load,
        wave_load=wave_load,
        other_load=other_load,
        duration_s=duration,
    )
    execution_step = simulate_execution_step(
        execution_state,
        execution_request,
        replace(execution_config, block_duration_s=duration),
    )
    return PhysicalExecutionPlatformSubstep(
        duration_s=duration,
        ballast_snapshot=snapshot,
        start_platform_state=platform_state,
        start_execution_state=execution_state,
        platform_step=platform_step,
        execution_step=execution_step,
    )


def advance_physical_execution_platform_path(
    *,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    execution_request: ExecutionRolloutRequest,
    execution_config: ExecutionRolloutConfig,
    runtime_assembly: BallastRuntimeAssembly,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    rotor_load: RotorGeneralizedLoad,
    wave_load: Any,
    other_load: Any,
    duration_s: float,
    rotor_load_substeps: tuple[RotorGeneralizedLoad, ...] | None = None,
) -> PhysicalExecutionPlatformPath:
    """Advance one committed request through the configured physical substeps.

    ``rotor_load_substeps`` may provide one already converted realised rotor
    load per physical substep. Wave and other loads remain constant. This
    helper does not interpolate wind or select a control action.
    """

    if not isinstance(execution_request, ExecutionRolloutRequest):
        raise TypeError("execution_request must be an ExecutionRolloutRequest")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be an ExecutionRolloutConfig")
    duration = _validated_duration(duration_s)
    if duration > float(execution_config.block_duration_s) + 1e-12:
        raise ValueError(
            "duration_s must not exceed execution_config.block_duration_s; "
            "the caller must obtain a new control decision for the next block"
        )

    substep_durations: list[float] = []
    remaining = duration
    while remaining > 1e-12:
        substep_duration = min(float(execution_config.internal_step_s), remaining)
        substep_durations.append(substep_duration)
        remaining -= substep_duration
    if rotor_load_substeps is not None:
        rotor_load_substeps = tuple(rotor_load_substeps)
        if len(rotor_load_substeps) != len(substep_durations):
            raise ValueError(
                "rotor_load_substeps must contain one load per physical substep"
            )
        if not all(
            isinstance(item, RotorGeneralizedLoad)
            for item in rotor_load_substeps
        ):
            raise TypeError(
                "rotor_load_substeps must contain RotorGeneralizedLoad values"
            )

    current_platform_state = platform_state
    current_execution_state = execution_state
    substeps: list[PhysicalExecutionPlatformSubstep] = []
    for substep_index, substep_duration in enumerate(substep_durations):
        substep_rotor_load = (
            rotor_load
            if rotor_load_substeps is None
            else rotor_load_substeps[substep_index]
        )
        substep = advance_physical_execution_platform_substep(
            platform_state=current_platform_state,
            execution_state=current_execution_state,
            execution_request=execution_request,
            execution_config=execution_config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=reference_tank_masses_kg,
            tank_capacities_kg=tank_capacities_kg,
            tank_coordinates_m=tank_coordinates_m,
            rotor_load=substep_rotor_load,
            wave_load=wave_load,
            other_load=other_load,
            duration_s=substep_duration,
        )
        substeps.append(substep)
        current_platform_state = substep.next_platform_state
        current_execution_state = substep.next_execution_state

    return PhysicalExecutionPlatformPath(substeps=tuple(substeps), duration_s=duration)


__all__ = [
    "PhysicalExecutionPlatformPath",
    "PhysicalExecutionPlatformSubstep",
    "advance_physical_execution_platform_path",
    "advance_physical_execution_platform_substep",
]
