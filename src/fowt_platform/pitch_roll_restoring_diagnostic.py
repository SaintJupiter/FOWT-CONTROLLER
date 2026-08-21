"""Constrained local pitch-roll demand diagnostics for explicit load changes.

The controller currently uses an empirical two-axis demand proxy.  This module
does not replace that proxy or enter candidate selection.  It only expresses
an explicit forecast-to-current generalized-load difference as a local
pitch-roll quantity with physical units.

The calculation holds surge, sway, heave and yaw fixed at the reference state
and uses the pitch-roll principal block of the frozen restoring matrix.  It is
therefore a local static diagnostic, not a free six-degree-of-freedom
equilibrium or a time-domain platform prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .incremental import DOF_COUNT, PlatformMatrices


_PITCH_ROLL_INDICES = (4, 3)


def _finite_six_vector(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (DOF_COUNT,):
        raise ValueError(f"{name} must have shape (6,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_pair(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _finite_symmetric_pair_matrix(name: str, value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError(f"{name} must have shape (2, 2), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    result = np.array(matrix, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PitchRollRestoringDiagnostic:
    """Local response to a future generalized-load increment.

    All vector fields use the fixed order ``(pitch, roll)``.  Moment fields
    have unit ``N m`` and ``constrained_pitch_roll_response_rad`` has unit
    ``rad``.  The stiffness matrix is also ordered ``(pitch, roll)`` and has
    unit ``N m/rad``.
    """

    relative_pitch_roll_load_nm: Any
    counteracting_pitch_roll_moment_nm: Any
    constrained_pitch_roll_response_rad: Any
    pitch_roll_stiffness_nm_per_rad: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_pitch_roll_load_nm",
            _finite_pair(
                "relative_pitch_roll_load_nm",
                self.relative_pitch_roll_load_nm,
            ),
        )
        object.__setattr__(
            self,
            "counteracting_pitch_roll_moment_nm",
            _finite_pair(
                "counteracting_pitch_roll_moment_nm",
                self.counteracting_pitch_roll_moment_nm,
            ),
        )
        object.__setattr__(
            self,
            "constrained_pitch_roll_response_rad",
            _finite_pair(
                "constrained_pitch_roll_response_rad",
                self.constrained_pitch_roll_response_rad,
            ),
        )
        object.__setattr__(
            self,
            "pitch_roll_stiffness_nm_per_rad",
            _finite_symmetric_pair_matrix(
                "pitch_roll_stiffness_nm_per_rad",
                self.pitch_roll_stiffness_nm_per_rad,
            ),
        )


def diagnose_pitch_roll_restoring_demand(
    *,
    matrices: PlatformMatrices,
    current_generalized_load: Any,
    future_generalized_load: Any,
) -> PitchRollRestoringDiagnostic:
    """Diagnose the local pitch-roll effect of a forecast load change.

    Both generalized loads must use the same frozen platform axes, reference
    point and ``[X, Y, Z, K, M, N]`` convention.  Only their relative pitch
    and roll moments are used.  ``surge``, ``sway``, ``heave`` and ``yaw`` are
    held fixed, so the returned angle is a constrained local static indicator
    rather than a six-degree-of-freedom equilibrium.

    The returned counteracting moment is the external ``(pitch, roll)``
    moment that would oppose the relative forecast load.  It is not a ballast
    mass target, pump command or candidate-ranking quantity.
    """

    if not isinstance(matrices, PlatformMatrices):
        raise TypeError("matrices must be PlatformMatrices")
    current = _finite_six_vector("current_generalized_load", current_generalized_load)
    future = _finite_six_vector("future_generalized_load", future_generalized_load)

    stiffness = matrices.restoring_stiffness[
        np.ix_(_PITCH_ROLL_INDICES, _PITCH_ROLL_INDICES)
    ]
    eigenvalues = np.linalg.eigvalsh(stiffness)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError(
            "pitch-roll restoring stiffness must be positive definite for the constrained local diagnostic"
        )

    relative_load = (future - current)[list(_PITCH_ROLL_INDICES)]
    constrained_response = np.linalg.solve(stiffness, relative_load)
    counteracting_moment = -relative_load
    return PitchRollRestoringDiagnostic(
        relative_pitch_roll_load_nm=relative_load,
        counteracting_pitch_roll_moment_nm=counteracting_moment,
        constrained_pitch_roll_response_rad=constrained_response,
        pitch_roll_stiffness_nm_per_rad=stiffness,
    )


__all__ = [
    "PitchRollRestoringDiagnostic",
    "diagnose_pitch_roll_restoring_demand",
]
