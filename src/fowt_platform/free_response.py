"""Exact free response of a constant-coefficient incremental platform model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm

from .incremental import IncrementalState, PlatformMatrices


def _time_samples(value: Any) -> np.ndarray:
    samples = np.asarray(value, dtype=float)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("time_s must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(samples)):
        raise ValueError("time_s must contain only finite values")
    if samples[0] < 0.0 or np.any(np.diff(samples) <= 0.0):
        raise ValueError("time_s must be nonnegative and strictly increasing")
    result = np.array(samples, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class LinearFreeResponse:
    """Sampled response and mechanical energy about a fixed equilibrium."""

    time_s: Any
    position: Any
    velocity: Any
    mechanical_energy: Any

    def __post_init__(self) -> None:
        time_s = _time_samples(self.time_s)
        sample_count = time_s.size
        position = np.asarray(self.position, dtype=float)
        velocity = np.asarray(self.velocity, dtype=float)
        energy = np.asarray(self.mechanical_energy, dtype=float)
        if position.shape != (sample_count, 6):
            raise ValueError("position must have shape (sample_count, 6)")
        if velocity.shape != (sample_count, 6):
            raise ValueError("velocity must have shape (sample_count, 6)")
        if energy.shape != (sample_count,):
            raise ValueError("mechanical_energy must have shape (sample_count,)")
        if not all(
            np.all(np.isfinite(array)) for array in (position, velocity, energy)
        ):
            raise ValueError("response arrays must contain only finite values")
        for name, value in (
            ("time_s", time_s),
            ("position", position),
            ("velocity", velocity),
            ("mechanical_energy", energy),
        ):
            result = np.array(value, dtype=float, copy=True)
            result.setflags(write=False)
            object.__setattr__(self, name, result)


def simulate_linear_free_response(
    *,
    matrices: PlatformMatrices,
    initial_state: IncrementalState,
    time_s: Any,
) -> LinearFreeResponse:
    """Evaluate the exact LTI response with no external incremental loads.

    The matrices remain fixed for the complete response. This function therefore
    checks one frozen operating point and does not represent changing ballast,
    nonlinear drag, or amplitude-dependent hydrodynamic damping.
    """

    if not isinstance(matrices, PlatformMatrices):
        raise TypeError("matrices must be PlatformMatrices")
    if not isinstance(initial_state, IncrementalState):
        raise TypeError("initial_state must be IncrementalState")
    samples = _time_samples(time_s)

    mass_inverse_stiffness = np.linalg.solve(
        matrices.mass,
        matrices.restoring_stiffness,
    )
    mass_inverse_damping = np.linalg.solve(matrices.mass, matrices.damping)
    system_matrix = np.block(
        [
            [np.zeros((6, 6)), np.eye(6)],
            [-mass_inverse_stiffness, -mass_inverse_damping],
        ]
    )
    initial_vector = np.concatenate(
        (initial_state.position, initial_state.velocity)
    )
    state_history = np.vstack(
        [expm(system_matrix * sample) @ initial_vector for sample in samples]
    )
    position = state_history[:, :6]
    velocity = state_history[:, 6:]
    kinetic_energy = 0.5 * np.einsum(
        "ni,ij,nj->n",
        velocity,
        matrices.mass,
        velocity,
    )
    potential_energy = 0.5 * np.einsum(
        "ni,ij,nj->n",
        position,
        matrices.restoring_stiffness,
        position,
    )
    return LinearFreeResponse(
        time_s=samples,
        position=position,
        velocity=velocity,
        mechanical_energy=kinetic_energy + potential_energy,
    )
