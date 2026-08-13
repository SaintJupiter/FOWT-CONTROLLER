"""Undamped modal checks for low-order platform matrix candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _symmetric(name: str, value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (6, 6):
        raise ValueError(f"{name} must have shape (6, 6), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    return np.array(matrix, dtype=float, copy=True)


@dataclass(frozen=True)
class UndampedModes:
    angular_frequency_rad_s: np.ndarray
    frequency_hz: np.ndarray
    period_s: np.ndarray
    mode_shapes: np.ndarray


def analyze_undamped_modes(mass: Any, stiffness: Any) -> UndampedModes:
    """Solve the symmetric generalized eigenproblem K phi = w^2 M phi."""

    mass_matrix = _symmetric("mass", mass)
    stiffness_matrix = _symmetric("stiffness", stiffness)
    try:
        cholesky = np.linalg.cholesky(mass_matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("mass must be positive definite") from error

    stiffness_eigenvalues = np.linalg.eigvalsh(stiffness_matrix)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(stiffness_eigenvalues))))
    if float(np.min(stiffness_eigenvalues)) < -tolerance:
        raise ValueError("stiffness must be positive semidefinite")

    left_solved = np.linalg.solve(cholesky, stiffness_matrix)
    transformed = np.linalg.solve(cholesky, left_solved.T).T
    transformed = 0.5 * (transformed + transformed.T)
    eigenvalues, transformed_modes = np.linalg.eigh(transformed)
    eigenvalue_tolerance = (
        np.finfo(float).eps
        * max(1.0, float(np.max(np.abs(eigenvalues))))
        * mass_matrix.shape[0]
    )
    if np.any(eigenvalues < -eigenvalue_tolerance):
        raise ValueError("generalized stiffness contains an unstable mode")
    eigenvalues[eigenvalues < 0.0] = 0.0

    mode_shapes = np.linalg.solve(cholesky.T, transformed_modes)
    angular_frequency = np.sqrt(eigenvalues)
    frequency = angular_frequency / (2.0 * np.pi)
    period = np.full_like(angular_frequency, np.inf)
    nonzero = angular_frequency > 0.0
    period[nonzero] = 2.0 * np.pi / angular_frequency[nonzero]

    for array in (angular_frequency, frequency, period, mode_shapes):
        array.setflags(write=False)
    return UndampedModes(
        angular_frequency_rad_s=angular_frequency,
        frequency_hz=frequency,
        period_s=period,
        mode_shapes=mode_shapes,
    )
