"""Current-posture ballast-feedback baseline on the active physical path.

The baseline deliberately has no wind forecast, preview horizon, candidate
enumeration, or MPC objective.  It converts the measured pitch-roll posture
outside an explicit deadband into an additional restoring moment using the
same frozen low-order matrices seen by the active plant.  The requested moment
is then allocated to the same three differential ballast tanks used by the
preview controller.

This module chooses one target only.  Pump dynamics and platform propagation
remain owned by the shared physical execution path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

from fowt_platform.ballast_moment_allocation import (
    BallastMomentAllocation,
    allocate_pitch_roll_moment_to_tanks,
)
from fowt_platform.ballast_snapshot import BallastModelSnapshot
from fowt_platform.incremental import IncrementalState

from .execution_rollout import ExecutionRolloutRequest


_PITCH_ROLL_STATE_INDICES = np.array([4, 3], dtype=int)


def _positive_pair(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must contain two finite positive values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nonnegative_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return scalar


def _readonly_pair(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PostureFeedbackBaselineDesign:
    """Explicit stage design for a non-preview posture-feedback baseline.

    Pair-valued quantities use ``(pitch, roll)`` order.  The stiffness and
    damping fractions are dimensionless multipliers on the corresponding
    frozen plant matrices.  They are controller design choices, not platform
    parameters or calibrated optima.
    """

    identity: str
    status: str
    deadband_pitch_roll_rad: Any
    stiffness_feedback_fraction: float
    damping_feedback_fraction: float

    def __post_init__(self) -> None:
        identity = str(self.identity).strip()
        status = str(self.status).strip()
        if not identity or not status:
            raise ValueError("identity and status must be non-empty")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "deadband_pitch_roll_rad",
            _positive_pair(
                "deadband_pitch_roll_rad",
                self.deadband_pitch_roll_rad,
            ),
        )
        object.__setattr__(
            self,
            "stiffness_feedback_fraction",
            _nonnegative_scalar(
                "stiffness_feedback_fraction",
                self.stiffness_feedback_fraction,
            ),
        )
        object.__setattr__(
            self,
            "damping_feedback_fraction",
            _nonnegative_scalar(
                "damping_feedback_fraction",
                self.damping_feedback_fraction,
            ),
        )
        if self.stiffness_feedback_fraction == 0.0 and self.damping_feedback_fraction == 0.0:
            raise ValueError("at least one feedback fraction must be positive")

    def as_dict(self) -> dict[str, Any]:
        parameters = {
            "deadband_pitch_roll_rad": self.deadband_pitch_roll_rad.tolist(),
            "stiffness_feedback_fraction": self.stiffness_feedback_fraction,
            "damping_feedback_fraction": self.damping_feedback_fraction,
        }
        digest = hashlib.sha256(
            json.dumps(parameters, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "identity": self.identity,
            "status": self.status,
            "parameter_sha256": digest,
            "deadband_pitch_roll_deg": np.rad2deg(
                self.deadband_pitch_roll_rad
            ).tolist(),
            **parameters,
        }


def research_posture_feedback_baseline_v1() -> PostureFeedbackBaselineDesign:
    """Return the transparent initial baseline design, not a tuned optimum."""

    return PostureFeedbackBaselineDesign(
        identity="research_posture_feedback_baseline_v1",
        status="stage_working_point_not_calibrated_or_optimal",
        deadband_pitch_roll_rad=np.deg2rad([1.0, 0.8]),
        stiffness_feedback_fraction=1.0,
        damping_feedback_fraction=1.0,
    )


def research_posture_feedback_baseline_v2() -> PostureFeedbackBaselineDesign:
    """Return the explicit P-only baseline used with the damped plant model."""

    return PostureFeedbackBaselineDesign(
        identity="research_posture_feedback_baseline_v2",
        status="p_only_stage_working_point_not_calibrated_or_optimal",
        deadband_pitch_roll_rad=np.deg2rad([1.0, 0.8]),
        stiffness_feedback_fraction=1.0,
        damping_feedback_fraction=0.0,
    )


@dataclass(frozen=True)
class PostureFeedbackBaselineDecision:
    """One source-free feedback target before shared physical execution."""

    design: PostureFeedbackBaselineDesign
    measured_pitch_roll_rad: Any
    measured_pitch_roll_rate_rad_s: Any
    effective_pitch_roll_error_rad: Any
    active_axes: tuple[bool, bool]
    requested_ballast_pitch_roll_moment_nm: Any
    allocation: BallastMomentAllocation
    block_reachability_scale: float
    reachable_target_tank_masses_kg: Any
    execution_request: ExecutionRolloutRequest

    def __post_init__(self) -> None:
        if not isinstance(self.design, PostureFeedbackBaselineDesign):
            raise TypeError("design must be PostureFeedbackBaselineDesign")
        for name in (
            "measured_pitch_roll_rad",
            "measured_pitch_roll_rate_rad_s",
            "effective_pitch_roll_error_rad",
            "requested_ballast_pitch_roll_moment_nm",
        ):
            object.__setattr__(self, name, _readonly_pair(name, getattr(self, name)))
        reachable_target = np.asarray(
            self.reachable_target_tank_masses_kg,
            dtype=float,
        )
        if reachable_target.shape != (3,) or not np.all(np.isfinite(reachable_target)):
            raise ValueError(
                "reachable_target_tank_masses_kg must contain three finite values"
            )
        stored_target = np.array(reachable_target, dtype=float, copy=True)
        stored_target.setflags(write=False)
        object.__setattr__(self, "reachable_target_tank_masses_kg", stored_target)
        active = tuple(bool(value) for value in self.active_axes)
        if len(active) != 2:
            raise ValueError("active_axes must contain pitch and roll flags")
        object.__setattr__(self, "active_axes", active)
        if not isinstance(self.allocation, BallastMomentAllocation):
            raise TypeError("allocation must be BallastMomentAllocation")
        scale = float(self.block_reachability_scale)
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("block_reachability_scale must lie in [0, 1]")
        object.__setattr__(self, "block_reachability_scale", scale)
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")


def decide_posture_feedback_target(
    *,
    platform_snapshot: BallastModelSnapshot,
    platform_state: IncrementalState,
    design: PostureFeedbackBaselineDesign,
    maximum_abs_tank_mass_change_per_block_kg: Any | None = None,
) -> PostureFeedbackBaselineDecision:
    """Map current pitch-roll posture to one absolute three-tank target.

    The target is referenced to the common equilibrium tank masses.  This
    prevents repeated control cycles from accumulating an incremental target
    merely because a posture error persists.  When both axes lie inside the
    deadband, the request releases to the measured tank state and stops adding
    a new feedback correction.
    """

    if not isinstance(platform_snapshot, BallastModelSnapshot):
        raise TypeError("platform_snapshot must be a BallastModelSnapshot")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be an IncrementalState")
    if not isinstance(design, PostureFeedbackBaselineDesign):
        raise TypeError("design must be a PostureFeedbackBaselineDesign")

    posture = np.asarray(
        platform_state.position[_PITCH_ROLL_STATE_INDICES],
        dtype=float,
    )
    posture_rate = np.asarray(
        platform_state.velocity[_PITCH_ROLL_STATE_INDICES],
        dtype=float,
    )
    deadband = np.asarray(design.deadband_pitch_roll_rad, dtype=float)
    effective_error = np.sign(posture) * np.maximum(
        np.abs(posture) - deadband,
        0.0,
    )
    active = np.abs(effective_error) > 0.0
    active_rate = np.where(active, posture_rate, 0.0)

    matrices = platform_snapshot.matrices
    stiffness = matrices.restoring_stiffness[
        np.ix_(_PITCH_ROLL_STATE_INDICES, _PITCH_ROLL_STATE_INDICES)
    ]
    damping = matrices.damping[
        np.ix_(_PITCH_ROLL_STATE_INDICES, _PITCH_ROLL_STATE_INDICES)
    ]
    requested_moment = -(
        design.stiffness_feedback_fraction * (stiffness @ effective_error)
        + design.damping_feedback_fraction * (damping @ active_rate)
    )
    allocation = allocate_pitch_roll_moment_to_tanks(
        requested_pitch_roll_moment_nm=requested_moment,
        actual_tank_masses_kg=platform_snapshot.reference_tank_masses_kg,
        tank_capacities_kg=platform_snapshot.tank_capacities_kg,
        tank_coordinates_m=platform_snapshot.tank_coordinates_m,
        gravity_m_s2=platform_snapshot.gravity_m_s2,
    )
    reachable_target = np.asarray(allocation.target_tank_masses_kg, dtype=float)
    reachability_scale = 1.0
    if np.any(active) and maximum_abs_tank_mass_change_per_block_kg is not None:
        limits = np.asarray(
            maximum_abs_tank_mass_change_per_block_kg,
            dtype=float,
        )
        if limits.ndim == 0:
            limits = np.full(3, float(limits))
        if (
            limits.shape != (3,)
            or not np.all(np.isfinite(limits))
            or np.any(limits <= 0.0)
        ):
            raise ValueError(
                "maximum_abs_tank_mass_change_per_block_kg must contain "
                "three finite positive values"
            )
        requested_delta = (
            reachable_target - platform_snapshot.actual_tank_masses_kg
        )
        nonzero = np.abs(requested_delta) > 1.0e-12
        if np.any(nonzero):
            reachability_scale = float(
                min(1.0, np.min(limits[nonzero] / np.abs(requested_delta[nonzero])))
            )
            reachable_target = (
                platform_snapshot.actual_tank_masses_kg
                + reachability_scale * requested_delta
            )
    request = (
        ExecutionRolloutRequest.track(reachable_target)
        if np.any(active)
        else ExecutionRolloutRequest.release_to_current()
    )
    return PostureFeedbackBaselineDecision(
        design=design,
        measured_pitch_roll_rad=posture,
        measured_pitch_roll_rate_rad_s=posture_rate,
        effective_pitch_roll_error_rad=effective_error,
        active_axes=tuple(bool(value) for value in active),
        requested_ballast_pitch_roll_moment_nm=requested_moment,
        allocation=allocation,
        block_reachability_scale=reachability_scale,
        reachable_target_tank_masses_kg=reachable_target,
        execution_request=request,
    )


__all__ = [
    "PostureFeedbackBaselineDecision",
    "PostureFeedbackBaselineDesign",
    "decide_posture_feedback_target",
    "research_posture_feedback_baseline_v1",
    "research_posture_feedback_baseline_v2",
]
