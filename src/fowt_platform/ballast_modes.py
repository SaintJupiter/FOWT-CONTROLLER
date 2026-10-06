"""Geometry-derived differential coordinates for three ballast tanks.

The controller may plan only zero-net redistribution even though the three
physical pumps exchange water with the sea independently.  This module keeps
that modeling boundary explicit: it converts two differential coordinates to
three tank-mass changes, but it neither simulates pumps nor asserts that the
instantaneous physical tank state has zero total-mass error.

The two basis vectors are derived from tank geometry.  One produces a pure
pitch gravity moment and the other a pure roll gravity moment.  Each vector
has unit Euclidean norm, so both modal coordinates retain units of kilograms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast import ballast_gravity_load_about_reference


_PITCH_ROLL_LOAD_INDICES = np.array([4, 3], dtype=int)


def _finite_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    return np.array(array, dtype=float, copy=True)


def _positive_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DifferentialModeProjection:
    """Decomposition of three tank-mass changes into differential and common parts."""

    modal_masses_kg: Any
    differential_tank_mass_deltas_kg: Any
    common_tank_mass_delta_kg: float
    residual_tank_mass_deltas_kg: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "modal_masses_kg",
            _readonly(_finite_array("modal_masses_kg", self.modal_masses_kg, (2,))),
        )
        for name in (
            "differential_tank_mass_deltas_kg",
            "residual_tank_mass_deltas_kg",
        ):
            object.__setattr__(
                self,
                name,
                _readonly(_finite_array(name, getattr(self, name), (3,))),
            )
        common = float(self.common_tank_mass_delta_kg)
        if not np.isfinite(common):
            raise ValueError("common_tank_mass_delta_kg must be finite")
        object.__setattr__(self, "common_tank_mass_delta_kg", common)


@dataclass(frozen=True)
class ThreeTankDifferentialModes:
    """Two geometry-derived mass modes spanning zero-net three-tank changes.

    Column zero of ``tank_mass_basis`` produces pitch moment with negligible
    roll moment.  Column one produces roll moment with negligible pitch
    moment.  Both columns sum to zero and have unit Euclidean norm.
    """

    tank_coordinates_m: Any
    gravity_m_s2: float = 9.81

    def __post_init__(self) -> None:
        coordinates = _finite_array(
            "tank_coordinates_m", self.tank_coordinates_m, (3, 3)
        )
        gravity = _positive_scalar("gravity_m_s2", self.gravity_m_s2)
        unit_tank_loads = np.column_stack(
            [
                ballast_gravity_load_about_reference(
                    tank_mass_deltas_kg=np.eye(3)[index],
                    tank_coordinates_m=coordinates,
                    gravity_m_s2=gravity,
                )
                for index in range(3)
            ]
        )
        pitch_roll_per_tank = unit_tank_loads[_PITCH_ROLL_LOAD_INDICES]
        augmented = np.vstack((pitch_roll_per_tank, np.ones(3)))
        if np.linalg.matrix_rank(augmented, tol=1.0e-10) != 3:
            raise ValueError(
                "tank geometry cannot form independent pure pitch and roll "
                "moments under zero-net redistribution"
            )

        columns: list[np.ndarray] = []
        for target in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            column = np.linalg.solve(augmented, target)
            norm = float(np.linalg.norm(column))
            if norm <= np.finfo(float).eps:
                raise ValueError("tank geometry produced a degenerate ballast mode")
            column /= norm
            columns.append(column)
        basis = np.column_stack(columns)
        generalized_load_per_mode = unit_tank_loads @ basis
        pitch_roll_per_mode = generalized_load_per_mode[_PITCH_ROLL_LOAD_INDICES]

        scale = max(1.0, float(np.max(np.abs(pitch_roll_per_mode))))
        if not np.allclose(
            pitch_roll_per_mode - np.diag(np.diag(pitch_roll_per_mode)),
            np.zeros((2, 2)),
            rtol=0.0,
            atol=1.0e-10 * scale,
        ):
            raise ValueError("derived ballast modes are not pitch-roll decoupled")
        if np.any(np.diag(pitch_roll_per_mode) <= 0.0):
            raise ValueError("derived ballast mode signs must produce positive moments")

        object.__setattr__(self, "tank_coordinates_m", _readonly(coordinates))
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "_tank_mass_basis", _readonly(basis))
        object.__setattr__(
            self,
            "_generalized_load_per_mode_kg",
            _readonly(generalized_load_per_mode),
        )

    @property
    def tank_mass_basis(self) -> np.ndarray:
        """Return the shape-(3, 2) tank-mass map in kg per modal kg."""

        return self._tank_mass_basis

    @property
    def generalized_load_per_mode_kg(self) -> np.ndarray:
        """Return the shape-(6, 2) gravity-load map in N or N m per modal kg."""

        return self._generalized_load_per_mode_kg

    @property
    def pitch_roll_moment_per_mode_kg(self) -> np.ndarray:
        """Return the diagonal pitch-roll moment map in N m per modal kg."""

        result = self._generalized_load_per_mode_kg[_PITCH_ROLL_LOAD_INDICES]
        result.setflags(write=False)
        return result

    def tank_mass_deltas_from_modes(self, modal_masses_kg: Any) -> np.ndarray:
        modal = _finite_array("modal_masses_kg", modal_masses_kg, (2,))
        return _readonly(self._tank_mass_basis @ modal)

    def generalized_load_from_modes(self, modal_masses_kg: Any) -> np.ndarray:
        modal = _finite_array("modal_masses_kg", modal_masses_kg, (2,))
        return _readonly(self._generalized_load_per_mode_kg @ modal)

    def project_tank_mass_deltas(self, tank_mass_deltas_kg: Any) -> DifferentialModeProjection:
        """Project a measured tank change without hiding its common component."""

        deltas = _finite_array("tank_mass_deltas_kg", tank_mass_deltas_kg, (3,))
        common = float(np.mean(deltas))
        zero_sum = deltas - common
        modal, _, _, _ = np.linalg.lstsq(self._tank_mass_basis, zero_sum, rcond=None)
        differential = self._tank_mass_basis @ modal
        residual = deltas - differential - common
        return DifferentialModeProjection(
            modal_masses_kg=modal,
            differential_tank_mass_deltas_kg=differential,
            common_tank_mass_delta_kg=common,
            residual_tank_mass_deltas_kg=residual,
        )

    def target_tank_masses(
        self,
        *,
        reference_tank_masses_kg: Any,
        modal_masses_kg: Any,
    ) -> np.ndarray:
        reference = _finite_array(
            "reference_tank_masses_kg", reference_tank_masses_kg, (3,)
        )
        return _readonly(reference + self.tank_mass_deltas_from_modes(modal_masses_kg))

    def modal_constraint_inequalities(
        self,
        *,
        reference_tank_masses_kg: Any,
        tank_capacities_kg: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``A, b`` for the capacity constraint ``A z <= b``."""

        reference = _finite_array(
            "reference_tank_masses_kg", reference_tank_masses_kg, (3,)
        )
        capacities = _finite_array("tank_capacities_kg", tank_capacities_kg, (3,))
        if np.any(capacities <= 0.0):
            raise ValueError("tank_capacities_kg values must be positive")
        if np.any(reference < 0.0) or np.any(reference > capacities):
            raise ValueError(
                "reference_tank_masses_kg must remain within tank capacities"
            )
        matrix = np.vstack((self._tank_mass_basis, -self._tank_mass_basis))
        bound = np.concatenate((capacities - reference, reference))
        return _readonly(matrix), _readonly(bound)


__all__ = ["DifferentialModeProjection", "ThreeTankDifferentialModes"]
