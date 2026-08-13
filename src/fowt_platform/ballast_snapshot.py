"""Build a frozen low-order model snapshot from measured tank masses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast import ballast_gravity_load_about_reference
from .ballast_state import tank_mass_deltas_from_actual_masses
from .incremental import PlatformMatrices
from .reference import VolturnusReferenceComponents


def _readonly_three(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
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


def assemble_ballast_model_snapshot(
    *,
    reference: VolturnusReferenceComponents,
    damping: Any,
    actual_tank_masses_kg: Any,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
) -> BallastModelSnapshot:
    """Assemble a bounded fixed-parameter snapshot from measured tank masses."""

    if not isinstance(reference, VolturnusReferenceComponents):
        raise TypeError("reference must be VolturnusReferenceComponents")
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
    matrices = reference.assemble_ballast_adjusted_candidate(
        damping=damping,
        tank_mass_deltas_kg=deltas,
        tank_coordinates_m=tank_coordinates_m,
    )
    load = ballast_gravity_load_about_reference(
        tank_mass_deltas_kg=deltas,
        tank_coordinates_m=tank_coordinates_m,
    )
    return BallastModelSnapshot(
        actual_tank_masses_kg=actual,
        reference_tank_masses_kg=baseline,
        tank_mass_deltas_kg=deltas,
        matrices=matrices,
        incremental_ballast_load=load,
    )
