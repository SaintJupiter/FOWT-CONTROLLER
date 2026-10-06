"""Independent three-tank diagnostics for discrete physical load forecasts.

The entry point in this module applies the existing one-lead forecast ballast
diagnostic to each already-physical future generalized-load point.  Every lead
uses the same current generalized load and the same current tank state.

It is intentionally not a time-domain ballast trajectory: no earlier
diagnostic endpoint is carried into a later lead, and the module does not
interpolate loads, schedule pumps, select actions, or rank results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .forecast_ballast_diagnostic import (
    ForecastBallastDemandDiagnostic,
    diagnose_forecast_ballast_redistribution,
)
from .ballast_moment_allocation import allocate_pitch_roll_moment_to_tanks
from .generalized_load_forecast import GeneralizedLoadForecast
from .incremental import PlatformMatrices


def _actual_tank_masses(value: Any) -> np.ndarray:
    """Copy one three-tank diagnostic start state into an immutable record."""

    try:
        masses = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "actual_tank_masses_kg must contain three finite values"
        ) from exc
    if masses.shape != (3,) or not np.all(np.isfinite(masses)):
        raise ValueError("actual_tank_masses_kg must contain three finite values")
    result = np.array(masses, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _tank_coordinates(value: Any) -> np.ndarray:
    try:
        coordinates = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "tank_coordinates_m must contain nine finite values with shape (3, 3)"
        ) from exc
    if coordinates.shape != (3, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "tank_coordinates_m must contain nine finite values with shape (3, 3)"
        )
    result = np.array(coordinates, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positive_gravity(value: Any) -> float:
    try:
        gravity = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gravity_m_s2 must be finite and positive") from exc
    if not np.isfinite(gravity) or gravity <= 0.0:
        raise ValueError("gravity_m_s2 must be finite and positive")
    return gravity


@dataclass(frozen=True)
class TimedForecastBallastDemandDiagnostic:
    """One independent three-tank diagnostic at a forecast lead time.

    ``lead_time_s`` identifies the discrete physical-load point used to form
    ``diagnostic``.  It does not specify an execution duration or a pump
    schedule.
    """

    lead_time_s: float
    diagnostic: ForecastBallastDemandDiagnostic
    actual_tank_masses_kg: Any
    tank_capacities_kg: Any
    tank_coordinates_m: Any
    gravity_m_s2: float

    def __post_init__(self) -> None:
        lead_time_s = float(self.lead_time_s)
        if not np.isfinite(lead_time_s) or lead_time_s <= 0.0:
            raise ValueError("lead_time_s must be finite and positive")
        if not isinstance(self.diagnostic, ForecastBallastDemandDiagnostic):
            raise TypeError("diagnostic must be ForecastBallastDemandDiagnostic")
        actual = _actual_tank_masses(self.actual_tank_masses_kg)
        capacities = _actual_tank_masses(self.tank_capacities_kg)
        if np.any(capacities <= 0.0):
            raise ValueError("tank_capacities_kg values must be positive")
        coordinates = _tank_coordinates(self.tank_coordinates_m)
        gravity = _positive_gravity(self.gravity_m_s2)
        if np.any(actual < 0.0) or np.any(actual > capacities):
            raise ValueError("actual_tank_masses_kg must remain within tank capacities")

        allocation = self.diagnostic.allocation
        expected_allocation = allocate_pitch_roll_moment_to_tanks(
            requested_pitch_roll_moment_nm=(
                allocation.requested_pitch_roll_moment_nm
            ),
            actual_tank_masses_kg=actual,
            tank_capacities_kg=capacities,
            tank_coordinates_m=coordinates,
            gravity_m_s2=gravity,
        )
        if not np.allclose(
            expected_allocation.target_tank_masses_kg,
            allocation.target_tank_masses_kg,
            rtol=0.0,
            atol=1e-8,
        ) or not np.allclose(
            expected_allocation.achieved_pitch_roll_moment_nm,
            allocation.achieved_pitch_roll_moment_nm,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError(
                "diagnostic allocation must match the frozen tank geometry, capacities and gravity"
            )
        object.__setattr__(self, "lead_time_s", lead_time_s)
        object.__setattr__(self, "actual_tank_masses_kg", actual)
        object.__setattr__(self, "tank_capacities_kg", capacities)
        object.__setattr__(self, "tank_coordinates_m", coordinates)
        object.__setattr__(self, "gravity_m_s2", gravity)


def diagnose_generalized_load_forecast_ballast_redistribution(
    *,
    forecast: GeneralizedLoadForecast,
    matrices: PlatformMatrices,
    actual_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    gravity_m_s2: Any = 9.81,
) -> tuple[TimedForecastBallastDemandDiagnostic, ...]:
    """Return one independent three-tank diagnostic for every forecast point.

    ``forecast`` contains discrete loads already expressed in the frozen
    platform convention ``[X, Y, Z, K, M, N]``.  Each future point is compared
    to the same current load and current actual tank masses.  The returned
    tuple is consequently a profile of independent endpoint diagnostics, not a
    sequence of mass states that a pump system can follow without further
    reachability and scheduling assumptions.
    """

    if not isinstance(forecast, GeneralizedLoadForecast):
        raise TypeError("forecast must be GeneralizedLoadForecast")
    actual = _actual_tank_masses(actual_tank_masses_kg)
    capacities = _actual_tank_masses(tank_capacities_kg)
    coordinates = _tank_coordinates(tank_coordinates_m)
    gravity = _positive_gravity(gravity_m_s2)

    return tuple(
        TimedForecastBallastDemandDiagnostic(
            lead_time_s=forecast.lead_time_at(index),
            diagnostic=diagnose_forecast_ballast_redistribution(
                matrices=matrices,
                current_generalized_load=forecast.current_generalized_load,
                future_generalized_load=forecast.future_load_at(index),
                actual_tank_masses_kg=actual,
                tank_capacities_kg=capacities,
                tank_coordinates_m=coordinates,
                gravity_m_s2=gravity,
            ),
            actual_tank_masses_kg=actual,
            tank_capacities_kg=capacities,
            tank_coordinates_m=coordinates,
            gravity_m_s2=gravity,
        )
        for index in range(forecast.horizon_steps)
    )


__all__ = [
    "TimedForecastBallastDemandDiagnostic",
    "diagnose_generalized_load_forecast_ballast_redistribution",
]
