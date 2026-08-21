"""One causally ordered physical substep for the compact controller chain.

This module deliberately joins only the committed actuator request to the
low-order platform step.  Controller decision-making and long-horizon
simulation remain outside this handoff so the time ordering stays explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from fowt_platform.ballast_snapshot import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    assemble_ballast_model_snapshot,
)
from fowt_platform.incremental import IncrementalState
from fowt_platform.open_loop_step import (
    OpenLoopPlatformStep,
    advance_frozen_open_loop_step,
)
from fowt_platform.rotor_input import RotorGeneralizedLoad

from .controller_core import ControlObservation
from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionRolloutStep,
    simulate_execution_step,
)


@dataclass(frozen=True)
class ControllerPlatformSubstep:
    """Results of one frozen platform step followed by one actuator step.

    ``ballast_snapshot`` and ``platform_step`` use the actual tank masses at
    the beginning of the physical substep.  ``execution_step`` contains the
    pump state after that substep, whose masses become the next substep's
    snapshot input.
    """

    ballast_snapshot: BallastModelSnapshot
    platform_step: OpenLoopPlatformStep
    execution_step: ExecutionRolloutStep

    @property
    def next_platform_state(self) -> IncrementalState:
        """Platform state at the end of this physical substep."""

        return self.platform_step.next_state

    @property
    def next_execution_state(self) -> ExecutionRolloutState:
        """Actuator state at the end of this physical substep."""

        return self.execution_step.state


def control_observation_from_incremental_state(
    *,
    time_s: float,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    current_wind_uv_ms: Any,
) -> ControlObservation:
    """Build the next controller observation from the P4 low-order state.

    The incremental state follows ``[surge, sway, heave, roll, pitch, yaw]``.
    The controller convention remains ``(pitch, roll)`` in degrees, matching
    the existing decision core. ``current_wind_uv_ms`` remains the caller's
    ENU downwind vector ``[east, north]`` in m/s; this handoff does not rotate
    it into the frozen platform axes. The actuator state is passed through as
    the actual state produced by the preceding physical substep.
    """

    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be an IncrementalState")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be an ExecutionRolloutState")
    return ControlObservation(
        time_s=time_s,
        posture_deg=(
            math.degrees(float(platform_state.position[4])),
            math.degrees(float(platform_state.position[3])),
        ),
        posture_rate_deg_s=(
            math.degrees(float(platform_state.velocity[4])),
            math.degrees(float(platform_state.velocity[3])),
        ),
        current_wind_uv_ms=current_wind_uv_ms,
        execution_state=execution_state,
    )


def advance_controller_platform_substep(
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
) -> ControllerPlatformSubstep:
    """Advance platform and actuator once using the P4 frozen-step order.

    The platform is advanced from the start-of-step tank snapshot before the
    pump state changes.  The same committed request then advances the
    actuator over the same duration.  This is intentionally a one-step
    frozen splitting approximation, not a continuous variable-mass model.
    The current actuator model has one shared tank-capacity limit, so this
    handoff accepts only three matching platform capacity values.
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
    try:
        duration = float(duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("duration_s must be positive and finite") from exc
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration_s must be positive and finite")
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
    return ControllerPlatformSubstep(
        ballast_snapshot=snapshot,
        platform_step=platform_step,
        execution_step=execution_step,
    )


__all__ = [
    "ControllerPlatformSubstep",
    "advance_controller_platform_substep",
    "control_observation_from_incremental_state",
]
