"""Boundary between executable tank states and incremental platform physics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _three_finite(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def tank_mass_deltas_from_actual_masses(
    *,
    actual_tank_masses_kg: Any,
    reference_tank_masses_kg: Any,
    tank_capacities_kg: Any,
) -> np.ndarray:
    """Convert bounded actual tank masses to signed reference increments."""

    actual = _three_finite("actual_tank_masses_kg", actual_tank_masses_kg)
    reference = _three_finite(
        "reference_tank_masses_kg",
        reference_tank_masses_kg,
    )
    capacities = _three_finite("tank_capacities_kg", tank_capacities_kg)
    if np.any(capacities <= 0.0):
        raise ValueError("each tank capacity must be positive")
    if np.any(actual < 0.0) or np.any(actual > capacities):
        raise ValueError("actual tank masses must remain within each tank capacity")
    if np.any(reference < 0.0) or np.any(reference > capacities):
        raise ValueError("reference tank masses must remain within each tank capacity")

    deltas = np.array(actual - reference, dtype=float, copy=True)
    deltas.setflags(write=False)
    return deltas


__all__ = ["tank_mass_deltas_from_actual_masses"]
