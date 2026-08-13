"""Mass properties of a platform with three point-mass ballast tanks.

This module contains no controller or plant state.  The caller supplies all
quantities relative to one reference point, and the returned inertia tensor is
about that same point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BallastMassProperties:
    """Combined mass properties about the caller's reference point."""

    total_mass_kg: float
    center_of_mass_m: np.ndarray
    inertia_about_reference_kg_m2: np.ndarray


def _finite_array(name: str, values: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array with shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_mass(name: str, value: Any) -> float:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative scalar") from exc
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    mass = float(array)
    if not np.isfinite(mass) or mass < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return mass


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=float,
    )


def rigid_body_mass_matrix_about_reference(
    *,
    total_mass_kg: float,
    center_of_mass_m: Any,
    inertia_about_reference_kg_m2: Any,
) -> np.ndarray:
    """Build the six-DOF rigid-body mass matrix about one reference point."""

    mass = _finite_mass("total_mass_kg", total_mass_kg)
    if mass <= 0.0:
        raise ValueError("total_mass_kg must be positive")
    center = _finite_array("center_of_mass_m", center_of_mass_m, (3,))
    inertia = _finite_array(
        "inertia_about_reference_kg_m2",
        inertia_about_reference_kg_m2,
        (3, 3),
    )
    if not np.allclose(inertia, inertia.T, rtol=1e-12, atol=1e-12):
        raise ValueError("inertia_about_reference_kg_m2 must be symmetric")

    coupling = -mass * _skew(center)
    matrix = np.block(
        [
            [mass * np.eye(3), coupling],
            [coupling.T, inertia],
        ]
    )
    matrix = 0.5 * (matrix + matrix.T)
    if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
        raise ValueError("rigid-body mass matrix must be positive definite")
    matrix.setflags(write=False)
    return matrix


def ballast_gravity_load_about_reference(
    *,
    tank_mass_deltas_kg: Any,
    tank_coordinates_m: Any,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Return the incremental weight load from signed tank-mass changes."""

    mass_deltas = _finite_array(
        "tank_mass_deltas_kg",
        tank_mass_deltas_kg,
        (3,),
    )
    coordinates = _finite_array(
        "tank_coordinates_m",
        tank_coordinates_m,
        (3, 3),
    )
    try:
        gravity = float(gravity_m_s2)
    except (TypeError, ValueError) as exc:
        raise ValueError("gravity_m_s2 must be finite and positive") from exc
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")

    forces = np.zeros((3, 3), dtype=float)
    forces[:, 2] = -gravity * mass_deltas
    moments = np.cross(coordinates, forces)
    load = np.concatenate((np.sum(forces, axis=0), np.sum(moments, axis=0)))
    load.setflags(write=False)
    return load


def weight_stiffness_about_reference(
    *,
    total_mass_kg: float,
    center_of_mass_m: Any,
    gravity_m_s2: float = 9.81,
) -> np.ndarray:
    """Linear roll/pitch weight stiffness about a level fixed reference."""

    mass = _finite_mass("total_mass_kg", total_mass_kg)
    if mass <= 0.0:
        raise ValueError("total_mass_kg must be positive")
    center = _finite_array("center_of_mass_m", center_of_mass_m, (3,))
    try:
        gravity = float(gravity_m_s2)
    except (TypeError, ValueError) as exc:
        raise ValueError("gravity_m_s2 must be finite and positive") from exc
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")

    stiffness = np.zeros((6, 6), dtype=float)
    rotational_stiffness = -mass * gravity * center[2]
    stiffness[3, 3] = rotational_stiffness
    stiffness[4, 4] = rotational_stiffness
    stiffness.setflags(write=False)
    return stiffness


def compute_ballast_mass_properties(
    *,
    dry_mass_kg: float,
    dry_center_of_mass_m: Any,
    dry_inertia_about_reference_kg_m2: Any,
    tank_masses_kg: Any,
    tank_coordinates_m: Any,
) -> BallastMassProperties:
    """Combine dry-platform and three-tank mass properties.

    Each tank is represented as a point mass at ``tank_coordinates_m``.  Its
    inertia contribution about the supplied reference point is
    ``m * (||r||^2 I - r r.T)``.  The dry-platform inertia must already be
    expressed about that same reference point.
    """

    dry_mass = _finite_mass("dry_mass_kg", dry_mass_kg)
    dry_center = _finite_array(
        "dry_center_of_mass_m",
        dry_center_of_mass_m,
        (3,),
    )
    dry_inertia = _finite_array(
        "dry_inertia_about_reference_kg_m2",
        dry_inertia_about_reference_kg_m2,
        (3, 3),
    )
    if not np.allclose(dry_inertia, dry_inertia.T, rtol=1e-12, atol=1e-12):
        raise ValueError("dry_inertia_about_reference_kg_m2 must be symmetric")

    tank_masses = _finite_array("tank_masses_kg", tank_masses_kg, (3,))
    if np.any(tank_masses < 0.0):
        raise ValueError("tank_masses_kg must be non-negative")
    tank_coordinates = _finite_array(
        "tank_coordinates_m",
        tank_coordinates_m,
        (3, 3),
    )

    total_mass = dry_mass + float(np.sum(tank_masses))
    if total_mass <= 0.0:
        raise ValueError("total mass must be positive to define a center of mass")

    first_moment = dry_mass * dry_center + np.sum(
        tank_masses[:, None] * tank_coordinates,
        axis=0,
    )
    center_of_mass = first_moment / total_mass

    radius_squared = np.einsum(
        "ij,ij->i",
        tank_coordinates,
        tank_coordinates,
    )
    identity = np.eye(3, dtype=float)
    point_mass_inertias = tank_masses[:, None, None] * (
        radius_squared[:, None, None] * identity
        - tank_coordinates[:, :, None] * tank_coordinates[:, None, :]
    )
    total_inertia = dry_inertia + np.sum(point_mass_inertias, axis=0)
    total_inertia = 0.5 * (total_inertia + total_inertia.T)
    if np.min(np.linalg.eigvalsh(total_inertia)) <= 0.0:
        raise ValueError("combined inertia must be positive definite")

    center_of_mass = np.array(center_of_mass, dtype=float, copy=True)
    total_inertia = np.array(total_inertia, dtype=float, copy=True)
    center_of_mass.setflags(write=False)
    total_inertia.setflags(write=False)

    return BallastMassProperties(
        total_mass_kg=float(total_mass),
        center_of_mass_m=center_of_mass,
        inertia_about_reference_kg_m2=total_inertia,
    )


def compute_incremental_ballast_mass_properties(
    *,
    reference_mass_kg: float,
    reference_center_of_mass_m: Any,
    reference_inertia_about_reference_kg_m2: Any,
    tank_mass_deltas_kg: Any,
    tank_coordinates_m: Any,
) -> BallastMassProperties:
    """Apply signed tank-mass changes to complete reference properties.

    ``reference_*`` describes the complete system at the working ballast
    reference, including its baseline ballast. Only the signed tank changes
    are added here. This prevents the reference ballast from being counted a
    second time when a public platform mass or inertia already includes it.
    """

    reference_mass = _finite_mass("reference_mass_kg", reference_mass_kg)
    reference_center = _finite_array(
        "reference_center_of_mass_m",
        reference_center_of_mass_m,
        (3,),
    )
    reference_inertia = _finite_array(
        "reference_inertia_about_reference_kg_m2",
        reference_inertia_about_reference_kg_m2,
        (3, 3),
    )
    if not np.allclose(
        reference_inertia,
        reference_inertia.T,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "reference_inertia_about_reference_kg_m2 must be symmetric"
        )

    mass_deltas = _finite_array(
        "tank_mass_deltas_kg",
        tank_mass_deltas_kg,
        (3,),
    )
    tank_coordinates = _finite_array(
        "tank_coordinates_m",
        tank_coordinates_m,
        (3, 3),
    )

    total_mass = reference_mass + float(np.sum(mass_deltas))
    if total_mass <= 0.0:
        raise ValueError("total mass must remain positive after ballast changes")

    first_moment = reference_mass * reference_center + np.sum(
        mass_deltas[:, None] * tank_coordinates,
        axis=0,
    )
    center_of_mass = first_moment / total_mass

    radius_squared = np.einsum(
        "ij,ij->i",
        tank_coordinates,
        tank_coordinates,
    )
    identity = np.eye(3, dtype=float)
    inertia_deltas = mass_deltas[:, None, None] * (
        radius_squared[:, None, None] * identity
        - tank_coordinates[:, :, None] * tank_coordinates[:, None, :]
    )
    total_inertia = reference_inertia + np.sum(inertia_deltas, axis=0)
    total_inertia = 0.5 * (total_inertia + total_inertia.T)
    if np.min(np.linalg.eigvalsh(total_inertia)) <= 0.0:
        raise ValueError(
            "inertia must remain positive definite after ballast changes"
        )

    center_of_mass = np.array(center_of_mass, dtype=float, copy=True)
    total_inertia = np.array(total_inertia, dtype=float, copy=True)
    center_of_mass.setflags(write=False)
    total_inertia.setflags(write=False)
    return BallastMassProperties(
        total_mass_kg=float(total_mass),
        center_of_mass_m=center_of_mass,
        inertia_about_reference_kg_m2=total_inertia,
    )


__all__ = [
    "ballast_gravity_load_about_reference",
    "BallastMassProperties",
    "compute_ballast_mass_properties",
    "compute_incremental_ballast_mass_properties",
    "rigid_body_mass_matrix_about_reference",
    "weight_stiffness_about_reference",
]
