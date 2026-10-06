"""Compatibility exports for the legacy controller-to-platform handoff.

The generic pump-and-platform propagation now lives in
``physical_execution_platform_path``.  This module retains the old public
names and the only controller-specific conversion: translating a low-order
platform state into the legacy controller observation convention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from fowt_platform.incremental import IncrementalState

from .execution_rollout import ExecutionRolloutState
from .physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    PhysicalExecutionPlatformSubstep,
    advance_physical_execution_platform_path,
    advance_physical_execution_platform_substep,
)


if TYPE_CHECKING:
    from .controller_core import ControlObservation


# Direct aliases preserve the published P4 interface without maintaining a
# second physical implementation under the legacy controller module.
ControllerPlatformSubstep = PhysicalExecutionPlatformSubstep
ControllerPlatformPath = PhysicalExecutionPlatformPath
advance_controller_platform_substep = advance_physical_execution_platform_substep
advance_controller_platform_path = advance_physical_execution_platform_path


def control_observation_from_incremental_state(
    *,
    time_s: float,
    platform_state: IncrementalState,
    execution_state: ExecutionRolloutState,
    current_wind_uv_ms: Any,
) -> "ControlObservation":
    """Build the legacy controller observation from one P4 low-order state.

    The incremental state uses ``[surge, sway, heave, roll, pitch, yaw]``.
    The legacy controller still expects ``(pitch, roll)`` in degrees.  The
    input wind remains an ENU downwind vector and is not rotated here.
    """

    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be an IncrementalState")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be an ExecutionRolloutState")
    from .controller_core import ControlObservation

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


__all__ = [
    "ControllerPlatformPath",
    "ControllerPlatformSubstep",
    "advance_controller_platform_path",
    "advance_controller_platform_substep",
    "control_observation_from_incremental_state",
]
