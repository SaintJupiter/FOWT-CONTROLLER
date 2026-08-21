"""Build a frozen low-order model snapshot from measured tank masses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast import (
    BallastMassProperties,
    ballast_gravity_load_about_reference,
    compute_incremental_ballast_mass_properties,
    rigid_body_mass_matrix_about_reference,
    weight_stiffness_about_reference,
)
from .ballast_state import tank_mass_deltas_from_actual_masses
from .incremental import PlatformMatrices


def _readonly_three(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _readonly_matrix(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite matrix with shape (3, 3)")
    if not np.allclose(array, array.T, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError(f"{name} must be symmetric")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class BallastModelSnapshot:
    """Matrices and incremental load derived from one frozen tank state.

    The snapshot is not a newly solved static equilibrium. A nonzero ballast
    load must still be balanced by the platform response or a separate
    equilibrium calculation.
    """

    actual_tank_masses_kg: Any
    reference_tank_masses_kg: Any
    tank_mass_deltas_kg: Any
    matrices: PlatformMatrices
    incremental_ballast_load: Any
    runtime_provenance: str

    def __post_init__(self) -> None:
        for name in (
            "actual_tank_masses_kg",
            "reference_tank_masses_kg",
            "tank_mass_deltas_kg",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_three(name, getattr(self, name)),
            )
        load = np.asarray(self.incremental_ballast_load, dtype=float)
        if load.shape != (6,) or not np.all(np.isfinite(load)):
            raise ValueError(
                "incremental_ballast_load must be a finite shape-(6,) vector"
            )
        load = np.array(load, dtype=float, copy=True)
        load.setflags(write=False)
        object.__setattr__(self, "incremental_ballast_load", load)
        if not isinstance(self.matrices, PlatformMatrices):
            raise TypeError("matrices must be PlatformMatrices")
        provenance = str(self.runtime_provenance).strip()
        if not provenance:
            raise ValueError("runtime_provenance must be a non-empty label")
        object.__setattr__(self, "runtime_provenance", provenance)


@dataclass(frozen=True)
class BallastRuntimeAssembly:
    """Explicit dynamic baseline used to update one measured tank state.

    ``base_matrices`` represents the complete low-order plant at the reference
    tank state.  ``reference_mass_properties`` describes the rigid-body part
    of that same reference state.  Tank changes update only the rigid-body
    mass and gravity-stiffness increments; damping, hull hydrostatics and
    mooring stiffness remain those supplied in ``base_matrices``.

    The assembly deliberately carries a short provenance label.  It makes the
    dynamic source visible at the controller boundary without claiming that
    every historical baseline is source-consistent.
    """

    base_matrices: PlatformMatrices
    reference_mass_properties: BallastMassProperties
    gravity_m_s2: float
    provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.base_matrices, PlatformMatrices):
            raise TypeError("base_matrices must be PlatformMatrices")
        if not isinstance(self.reference_mass_properties, BallastMassProperties):
            raise TypeError(
                "reference_mass_properties must be BallastMassProperties"
            )
        mass_properties = self.reference_mass_properties
        copied_properties = BallastMassProperties(
            total_mass_kg=float(mass_properties.total_mass_kg),
            center_of_mass_m=_readonly_three(
                "reference_mass_properties.center_of_mass_m",
                mass_properties.center_of_mass_m,
            ),
            inertia_about_reference_kg_m2=_readonly_matrix(
                "reference_mass_properties.inertia_about_reference_kg_m2",
                mass_properties.inertia_about_reference_kg_m2,
            ),
        )
        try:
            gravity = float(self.gravity_m_s2)
        except (TypeError, ValueError) as exc:
            raise ValueError("gravity_m_s2 must be finite and positive") from exc
        if not np.isfinite(gravity) or gravity <= 0.0:
            raise ValueError("gravity_m_s2 must be finite and positive")
        expected_weight_stiffness = weight_stiffness_about_reference(
            total_mass_kg=copied_properties.total_mass_kg,
            center_of_mass_m=copied_properties.center_of_mass_m,
            gravity_m_s2=gravity,
        )
        if not np.allclose(
            self.base_matrices.weight_stiffness,
            expected_weight_stiffness,
            rtol=1.0e-10,
            atol=1.0e-6,
        ):
            raise ValueError(
                "base_matrices.weight_stiffness must match the supplied "
                "reference mass properties"
            )
        provenance = str(self.provenance).strip()
        if not provenance:
            raise ValueError("provenance must be a non-empty label")
        object.__setattr__(self, "reference_mass_properties", copied_properties)
        object.__setattr__(self, "gravity_m_s2", gravity)
        object.__setattr__(self, "provenance", provenance)


def assemble_ballast_model_snapshot(
    *,
    runtime_assembly: BallastRuntimeAssembly,
    actual_tank_masses_kg: Any,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
) -> BallastModelSnapshot:
    """Assemble a bounded fixed-parameter snapshot from measured tank masses."""

    if not isinstance(runtime_assembly, BallastRuntimeAssembly):
        raise TypeError("runtime_assembly must be BallastRuntimeAssembly")
    actual = _readonly_three("actual_tank_masses_kg", actual_tank_masses_kg)
    baseline = _readonly_three(
        "reference_tank_masses_kg",
        reference_tank_masses_kg,
    )
    deltas = tank_mass_deltas_from_actual_masses(
        actual_tank_masses_kg=actual,
        reference_tank_masses_kg=baseline,
        tank_capacities_kg=tank_capacities_kg,
    )
    reference_properties = runtime_assembly.reference_mass_properties
    updated_properties = compute_incremental_ballast_mass_properties(
        reference_mass_kg=reference_properties.total_mass_kg,
        reference_center_of_mass_m=reference_properties.center_of_mass_m,
        reference_inertia_about_reference_kg_m2=(
            reference_properties.inertia_about_reference_kg_m2
        ),
        tank_mass_deltas_kg=deltas,
        tank_coordinates_m=tank_coordinates_m,
    )
    reference_rigid_body_mass = rigid_body_mass_matrix_about_reference(
        total_mass_kg=reference_properties.total_mass_kg,
        center_of_mass_m=reference_properties.center_of_mass_m,
        inertia_about_reference_kg_m2=(
            reference_properties.inertia_about_reference_kg_m2
        ),
    )
    updated_rigid_body_mass = rigid_body_mass_matrix_about_reference(
        total_mass_kg=updated_properties.total_mass_kg,
        center_of_mass_m=updated_properties.center_of_mass_m,
        inertia_about_reference_kg_m2=(
            updated_properties.inertia_about_reference_kg_m2
        ),
    )
    base = runtime_assembly.base_matrices
    reference_weight_stiffness = weight_stiffness_about_reference(
        total_mass_kg=reference_properties.total_mass_kg,
        center_of_mass_m=reference_properties.center_of_mass_m,
        gravity_m_s2=runtime_assembly.gravity_m_s2,
    )
    updated_weight_stiffness = weight_stiffness_about_reference(
        total_mass_kg=updated_properties.total_mass_kg,
        center_of_mass_m=updated_properties.center_of_mass_m,
        gravity_m_s2=runtime_assembly.gravity_m_s2,
    )
    matrices = PlatformMatrices(
        mass=base.mass + (updated_rigid_body_mass - reference_rigid_body_mass),
        damping=base.damping,
        hydrostatic_stiffness=base.hydrostatic_stiffness,
        mooring_stiffness=base.mooring_stiffness,
        weight_stiffness=(
            base.weight_stiffness
            + (updated_weight_stiffness - reference_weight_stiffness)
        ),
    )
    load = ballast_gravity_load_about_reference(
        tank_mass_deltas_kg=deltas,
        tank_coordinates_m=tank_coordinates_m,
        gravity_m_s2=runtime_assembly.gravity_m_s2,
    )
    return BallastModelSnapshot(
        actual_tank_masses_kg=actual,
        reference_tank_masses_kg=baseline,
        tank_mass_deltas_kg=deltas,
        matrices=matrices,
        incremental_ballast_load=load,
        runtime_provenance=runtime_assembly.provenance,
    )
