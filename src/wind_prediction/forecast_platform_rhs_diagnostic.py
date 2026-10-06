"""Origin and same-time state-equation diagnostics for a frozen forecast trajectory.

The trajectory module already fixes the forecast origin, the right-endpoint
load hold and the frozen tank snapshot. This module records the corresponding
six-degree-of-freedom state-equation terms at the current origin and at each
*future endpoint*:

``M q_ddot = tau_external - C q_dot - K q``.

It is intentionally a diagnostic record, not a demand or decision layer.  In
particular, a zero right-hand side only means zero instantaneous acceleration
under the frozen model and says nothing about a new equilibrium, stability,
candidate quality or platform safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform.incremental import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    IncrementalStateDerivative,
)
from fowt_platform.current_posture_feedback_diagnostic import (
    CurrentPosturePassiveLoadDiagnostic,
    diagnose_current_posture_passive_load,
)

from .forecast_platform_trajectory import ForecastPlatformTrajectory


_STATE_TOLERANCE = 1.0e-10


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


@dataclass(frozen=True)
class ForecastPlatformRhsPointDiagnostic:
    """One future endpoint expressed by the frozen six-DOF state equation.

    ``platform_state`` and ``loads`` are deliberately retained in full
    ``[X, Y, Z, K, M, N]`` form.  Callers must not reconstruct this record from
    separately supplied values because the time and snapshot identity come
    from ``trajectory.steps[lead_index]``.
    """

    trajectory: ForecastPlatformTrajectory
    lead_index: int
    lead_time_s: float
    platform_state: IncrementalState
    loads: IncrementalLoads
    passive_generalized_load: Any
    dynamic_rhs_generalized_load: Any
    state_derivative: IncrementalStateDerivative

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ForecastPlatformTrajectory):
            raise TypeError("trajectory must be ForecastPlatformTrajectory")
        if not isinstance(self.lead_index, int) or isinstance(self.lead_index, bool):
            raise TypeError("lead_index must be an integer")
        if not 0 <= self.lead_index < len(self.trajectory.steps):
            raise ValueError("lead_index must refer to a trajectory step")
        step = self.trajectory.steps[self.lead_index]
        lead_time_s = float(self.lead_time_s)
        if not np.isfinite(lead_time_s) or not np.isclose(
            lead_time_s,
            step.end_time_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("lead_time_s must match the trajectory endpoint time")
        if not isinstance(self.platform_state, IncrementalState):
            raise TypeError("platform_state must be IncrementalState")
        if self.platform_state is not step.end_state:
            raise ValueError("platform_state must be the matching trajectory endpoint")
        if not isinstance(self.loads, IncrementalLoads):
            raise TypeError("loads must be IncrementalLoads")
        if self.loads is not step.platform_step.loads:
            raise ValueError("loads must be the matching trajectory endpoint loads")
        if not isinstance(self.state_derivative, IncrementalStateDerivative):
            raise TypeError("state_derivative must be IncrementalStateDerivative")

        passive = _readonly_six(
            "passive_generalized_load", self.passive_generalized_load
        )
        dynamic_rhs = _readonly_six(
            "dynamic_rhs_generalized_load", self.dynamic_rhs_generalized_load
        )
        matrices = self.trajectory.platform_snapshot.matrices
        expected_passive = (
            -matrices.damping @ self.platform_state.velocity
            - matrices.restoring_stiffness @ self.platform_state.position
        )
        expected_rhs = self.loads.total + expected_passive
        expected_derivative = IncrementalPlatformModel(matrices).derivative(
            self.platform_state,
            self.loads,
        )
        if not np.allclose(passive, expected_passive, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                "passive_generalized_load must match the frozen damping and restoring terms"
            )
        if not np.allclose(dynamic_rhs, expected_rhs, rtol=0.0, atol=1.0e-8):
            raise ValueError(
                "dynamic_rhs_generalized_load must equal external plus passive load"
            )
        if not np.allclose(
            self.state_derivative.position_rate,
            expected_derivative.position_rate,
            rtol=0.0,
            atol=1.0e-10,
        ) or not np.allclose(
            self.state_derivative.velocity_rate,
            expected_derivative.velocity_rate,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("state_derivative must match the frozen platform model")
        if not np.allclose(
            matrices.mass @ self.state_derivative.velocity_rate,
            dynamic_rhs,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError(
                "state_derivative acceleration must satisfy the frozen state equation"
            )
        object.__setattr__(self, "lead_time_s", lead_time_s)
        object.__setattr__(self, "passive_generalized_load", passive)
        object.__setattr__(self, "dynamic_rhs_generalized_load", dynamic_rhs)


@dataclass(frozen=True)
class CurrentPlatformRhsDiagnostic:
    """Full frozen-model RHS at the forecast origin.

    The record uses the source-bound current rotor load, explicit current wave
    and other loads, the current tank snapshot and the trajectory's initial
    platform state. It records the physical state equation at one instant. It
    is not a ballast demand, target, score or action choice.
    """

    trajectory: ForecastPlatformTrajectory
    posture_passive_load: CurrentPosturePassiveLoadDiagnostic
    loads: IncrementalLoads
    passive_generalized_load: Any
    dynamic_rhs_generalized_load: Any
    state_derivative: IncrementalStateDerivative

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, ForecastPlatformTrajectory):
            raise TypeError("trajectory must be ForecastPlatformTrajectory")
        if not isinstance(self.posture_passive_load, CurrentPosturePassiveLoadDiagnostic):
            raise TypeError(
                "posture_passive_load must be CurrentPosturePassiveLoadDiagnostic"
            )
        if self.posture_passive_load.platform_snapshot is not self.trajectory.platform_snapshot:
            raise ValueError("posture_passive_load must use the trajectory snapshot")
        if self.posture_passive_load.platform_state is not self.trajectory.initial_state:
            raise ValueError("posture_passive_load must use the trajectory initial state")
        if not isinstance(self.loads, IncrementalLoads):
            raise TypeError("loads must be IncrementalLoads")
        if not isinstance(self.state_derivative, IncrementalStateDerivative):
            raise TypeError("state_derivative must be IncrementalStateDerivative")

        snapshot = self.trajectory.platform_snapshot
        environment = self.trajectory.environment
        if environment.forecast_origin_time != self.trajectory.initial_state_time:
            raise ValueError(
                "current rhs requires environment forecast_origin_time to match "
                "the trajectory initial_state_time"
            )
        operating_state = self.trajectory.load_assembly.operating_state
        state_velocity = self.trajectory.initial_state.velocity
        if not np.allclose(
            operating_state.current_platform_reference_velocity_platform_mps,
            state_velocity[:3],
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ) or not np.allclose(
            operating_state.current_platform_angular_velocity_platform_radps,
            state_velocity[3:],
            rtol=0.0,
            atol=_STATE_TOLERANCE,
        ):
            raise ValueError(
                "current rhs rotor operating-state kinematics must match the "
                "trajectory initial state"
            )
        expected_wind = self.trajectory.load_assembly.current_rotor_load.generalized_load_platform
        if not np.allclose(self.loads.wind, expected_wind, rtol=0.0, atol=1.0e-8):
            raise ValueError("current rhs wind load must use the source-bound current rotor load")
        if not np.allclose(
            self.loads.wave,
            environment.current_wave_load,
            rtol=0.0,
            atol=1.0e-8,
        ) or not np.allclose(
            self.loads.other,
            environment.current_other_load,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("current rhs non-rotor loads must use the explicit origin environment")
        if not np.allclose(
            self.loads.ballast,
            snapshot.incremental_ballast_load,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("current rhs ballast load must use the trajectory snapshot")

        passive = _readonly_six("passive_generalized_load", self.passive_generalized_load)
        dynamic_rhs = _readonly_six(
            "dynamic_rhs_generalized_load", self.dynamic_rhs_generalized_load
        )
        matrices = snapshot.matrices
        state = self.trajectory.initial_state
        expected_passive = (
            -matrices.damping @ state.velocity
            - matrices.restoring_stiffness @ state.position
        )
        expected_dynamic_rhs = self.loads.total + expected_passive
        expected_derivative = IncrementalPlatformModel(matrices).derivative(state, self.loads)
        if not np.allclose(passive, expected_passive, rtol=0.0, atol=1.0e-8):
            raise ValueError("current passive generalized load must match the frozen state")
        if not np.allclose(
            passive[[4, 3]],
            self.posture_passive_load.current_passive_pitch_roll_load_nm,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError("current posture passive load must match current RHS pitch-roll terms")
        if not np.allclose(dynamic_rhs, expected_dynamic_rhs, rtol=0.0, atol=1.0e-8):
            raise ValueError("current dynamic RHS must equal external plus passive load")
        if not np.allclose(
            self.state_derivative.position_rate,
            expected_derivative.position_rate,
            rtol=0.0,
            atol=1.0e-10,
        ) or not np.allclose(
            self.state_derivative.velocity_rate,
            expected_derivative.velocity_rate,
            rtol=0.0,
            atol=1.0e-10,
        ):
            raise ValueError("current state derivative must match the frozen platform model")
        if not np.allclose(
            matrices.mass @ self.state_derivative.velocity_rate,
            dynamic_rhs,
            rtol=0.0,
            atol=1.0e-7,
        ):
            raise ValueError("current acceleration must satisfy the frozen state equation")
        object.__setattr__(self, "passive_generalized_load", passive)
        object.__setattr__(self, "dynamic_rhs_generalized_load", dynamic_rhs)


def diagnose_current_platform_rhs(
    *,
    trajectory: ForecastPlatformTrajectory,
) -> CurrentPlatformRhsDiagnostic:
    """Record the full frozen-model RHS at the explicit forecast origin."""

    if not isinstance(trajectory, ForecastPlatformTrajectory):
        raise TypeError("trajectory must be ForecastPlatformTrajectory")
    snapshot = trajectory.platform_snapshot
    state = trajectory.initial_state
    posture_passive = diagnose_current_posture_passive_load(
        platform_snapshot=snapshot,
        platform_state=state,
    )
    loads = IncrementalLoads(
        wind=trajectory.load_assembly.current_rotor_load.generalized_load_platform,
        wave=trajectory.environment.current_wave_load,
        ballast=snapshot.incremental_ballast_load,
        other=trajectory.environment.current_other_load,
    )
    passive = -snapshot.matrices.damping @ state.velocity - snapshot.matrices.restoring_stiffness @ state.position
    derivative = IncrementalPlatformModel(snapshot.matrices).derivative(state, loads)
    return CurrentPlatformRhsDiagnostic(
        trajectory=trajectory,
        posture_passive_load=posture_passive,
        loads=loads,
        passive_generalized_load=passive,
        dynamic_rhs_generalized_load=loads.total + passive,
        state_derivative=derivative,
    )


def diagnose_forecast_platform_rhs(
    *,
    trajectory: ForecastPlatformTrajectory,
) -> tuple[ForecastPlatformRhsPointDiagnostic, ...]:
    """Record frozen-model state-equation terms at every forecast endpoint.

    Each record uses the endpoint state and the load held on the interval that
    ends at that same lead time.  The function neither changes the trajectory
    nor substitutes a load or state from another instant.
    """

    if not isinstance(trajectory, ForecastPlatformTrajectory):
        raise TypeError("trajectory must be ForecastPlatformTrajectory")
    model = IncrementalPlatformModel(trajectory.platform_snapshot.matrices)
    diagnostics: list[ForecastPlatformRhsPointDiagnostic] = []
    for step in trajectory.steps:
        state = step.end_state
        loads = step.platform_step.loads
        derivative = model.derivative(state, loads)
        passive = (
            -trajectory.platform_snapshot.matrices.damping @ state.velocity
            - trajectory.platform_snapshot.matrices.restoring_stiffness @ state.position
        )
        diagnostics.append(
            ForecastPlatformRhsPointDiagnostic(
                trajectory=trajectory,
                lead_index=step.lead_index,
                lead_time_s=step.end_time_s,
                platform_state=state,
                loads=loads,
                passive_generalized_load=passive,
                dynamic_rhs_generalized_load=loads.total + passive,
                state_derivative=derivative,
            )
        )
    return tuple(diagnostics)


__all__ = [
    "CurrentPlatformRhsDiagnostic",
    "ForecastPlatformRhsPointDiagnostic",
    "diagnose_current_platform_rhs",
    "diagnose_forecast_platform_rhs",
]
