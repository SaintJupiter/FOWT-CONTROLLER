"""Read the present small-angle posture as a passive pitch-roll load fact.

The frozen low-order platform model evolves according to

``M q_ddot + C q_dot + K q = tau_external``.

At one measured incremental state, the right-hand-side contribution from the
passive damping and restoring terms is ``-(C q_dot + K q)``.  This module
extracts its pitch and roll components in the same ``(pitch, roll) = (M, K)``
order used by the forecast-ballast diagnostics.

It is deliberately only a state diagnostic.  It does not choose a feedback
gain, construct a ballast target, combine the present-state term with a future
load forecast, name an action, or update controller state.  Those decisions
remain outside the physical evidence layer until their objective and safety
semantics are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast_snapshot import BallastModelSnapshot
from .incremental import IncrementalState


_PITCH_ROLL_INDICES = (4, 3)


def _readonly_pair(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite values") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class CurrentPosturePassiveLoadDiagnostic:
    """Passive pitch-roll contribution at one frozen platform state.

    All vector fields use ``(pitch, roll) = (M, K)`` order and unit ``N m``.
    ``current_restoring_pitch_roll_load_nm`` and
    ``current_damping_pitch_roll_load_nm`` are the corresponding
    right-hand-side contributions in the frozen linear equation.  Their sum is
    retained as ``current_passive_pitch_roll_load_nm`` so a later control
    layer can inspect the current-state physical fact without inheriting a
    hidden feedback gain.
    """

    platform_snapshot: BallastModelSnapshot
    platform_state: IncrementalState
    current_restoring_pitch_roll_load_nm: Any
    current_damping_pitch_roll_load_nm: Any
    current_passive_pitch_roll_load_nm: Any

    def __post_init__(self) -> None:
        if not isinstance(self.platform_snapshot, BallastModelSnapshot):
            raise TypeError("platform_snapshot must be a BallastModelSnapshot")
        if not isinstance(self.platform_state, IncrementalState):
            raise TypeError("platform_state must be an IncrementalState")
        for name in (
            "current_restoring_pitch_roll_load_nm",
            "current_damping_pitch_roll_load_nm",
            "current_passive_pitch_roll_load_nm",
        ):
            object.__setattr__(self, name, _readonly_pair(name, getattr(self, name)))

        matrices = self.platform_snapshot.matrices
        expected_restoring = -(
            matrices.restoring_stiffness @ self.platform_state.position
        )[list(_PITCH_ROLL_INDICES)]
        expected_damping = -(
            matrices.damping @ self.platform_state.velocity
        )[list(_PITCH_ROLL_INDICES)]
        expected_passive = expected_restoring + expected_damping
        if not np.allclose(
            self.current_restoring_pitch_roll_load_nm,
            expected_restoring,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "current_restoring_pitch_roll_load_nm must match the frozen restoring matrix and state"
            )
        if not np.allclose(
            self.current_damping_pitch_roll_load_nm,
            expected_damping,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "current_damping_pitch_roll_load_nm must match the frozen damping matrix and state"
            )
        if not np.allclose(
            self.current_passive_pitch_roll_load_nm,
            expected_passive,
            rtol=0.0,
            atol=1.0e-8,
        ):
            raise ValueError(
                "current_passive_pitch_roll_load_nm must equal restoring plus damping contributions"
            )


def diagnose_current_posture_passive_load(
    *,
    platform_snapshot: BallastModelSnapshot,
    platform_state: IncrementalState,
) -> CurrentPosturePassiveLoadDiagnostic:
    """Return passive restoring and damping moments at the current state.

    ``platform_state`` must already be expressed about the same fixed
    equilibrium, reference point and axes as ``platform_snapshot.matrices``.
    The function does not infer this state from legacy posture angles because
    those angles omit the remaining generalized coordinates and velocities.
    """

    if not isinstance(platform_snapshot, BallastModelSnapshot):
        raise TypeError("platform_snapshot must be a BallastModelSnapshot")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be an IncrementalState")

    matrices = platform_snapshot.matrices
    restoring = -(
        matrices.restoring_stiffness @ platform_state.position
    )[list(_PITCH_ROLL_INDICES)]
    damping = -(matrices.damping @ platform_state.velocity)[list(_PITCH_ROLL_INDICES)]
    return CurrentPosturePassiveLoadDiagnostic(
        platform_snapshot=platform_snapshot,
        platform_state=platform_state,
        current_restoring_pitch_roll_load_nm=restoring,
        current_damping_pitch_roll_load_nm=damping,
        current_passive_pitch_roll_load_nm=restoring + damping,
    )


__all__ = [
    "CurrentPosturePassiveLoadDiagnostic",
    "diagnose_current_posture_passive_load",
]
