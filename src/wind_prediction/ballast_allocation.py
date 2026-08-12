"""Pure three-tank ballast allocation mappings.

The compensation vector uses the planner convention: positive pitch or roll
components request ballast motion that opposes the corresponding physical
attitude.  The default matrix therefore matches the compensation allocation
used by the existing planner.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


DEFAULT_COMPENSATION_ALLOCATION: NDArray[np.float64] = np.array(
    [
        [-1.0, 0.0],
        [0.5, 1.0],
        [0.5, -1.0],
    ],
    dtype=float,
)
DEFAULT_COMPENSATION_ALLOCATION.setflags(write=False)

__all__ = [
    "DEFAULT_COMPENSATION_ALLOCATION",
    "compensation_to_mass_delta_kg",
    "mass_delta_to_compensation_vec",
]


def _finite_vector(value: ArrayLike, *, name: str, length: int) -> NDArray[np.float64]:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector of length {length}") from exc
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _positive_scalar(value: float, *, name: str) -> float:
    try:
        scalar_array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if scalar_array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    scalar = float(scalar_array)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return scalar


def _allocation_matrix(value: ArrayLike) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("allocation_matrix must be a numeric 3-by-2 matrix") from exc
    if matrix.shape != (3, 2):
        raise ValueError(
            f"allocation_matrix must have shape (3, 2), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("allocation_matrix must contain only finite values")
    if np.linalg.matrix_rank(matrix) != 2:
        raise ValueError("allocation_matrix must have full column rank")
    return matrix


def _validated_scales(
    deadband_deg: ArrayLike,
    action_mass_quantum_kg: float,
) -> tuple[NDArray[np.float64], float]:
    deadband = _finite_vector(deadband_deg, name="deadband_deg", length=2)
    if np.any(deadband <= 0.0):
        raise ValueError("deadband_deg values must be greater than zero")
    mass_quantum = _positive_scalar(
        action_mass_quantum_kg,
        name="action_mass_quantum_kg",
    )
    return deadband, mass_quantum


def compensation_to_mass_delta_kg(
    compensation_vec: ArrayLike,
    *,
    deadband_deg: ArrayLike,
    action_mass_quantum_kg: float,
    allocation_matrix: ArrayLike = DEFAULT_COMPENSATION_ALLOCATION,
) -> NDArray[np.float64]:
    """Map pitch/roll compensation to three tank mass increments in kilograms."""

    compensation = _finite_vector(
        compensation_vec,
        name="compensation_vec",
        length=2,
    )
    deadband, mass_quantum = _validated_scales(
        deadband_deg,
        action_mass_quantum_kg,
    )
    allocation = _allocation_matrix(allocation_matrix)
    return (allocation @ (compensation / deadband)) * mass_quantum


def mass_delta_to_compensation_vec(
    mass_delta_kg: ArrayLike,
    *,
    deadband_deg: ArrayLike,
    action_mass_quantum_kg: float,
    allocation_matrix: ArrayLike = DEFAULT_COMPENSATION_ALLOCATION,
) -> NDArray[np.float64]:
    """Map three tank mass increments back to pitch/roll compensation.

    The pseudoinverse removes common-mode intake or discharge because equal
    mass changes in all three tanks do not create pitch or roll compensation.
    """

    mass_delta = _finite_vector(
        mass_delta_kg,
        name="mass_delta_kg",
        length=3,
    )
    deadband, mass_quantum = _validated_scales(
        deadband_deg,
        action_mass_quantum_kg,
    )
    allocation = _allocation_matrix(allocation_matrix)
    normalized_compensation = np.linalg.pinv(allocation) @ (
        mass_delta / mass_quantum
    )
    return np.asarray(normalized_compensation, dtype=float) * deadband
