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
from .generalized_load_forecast import GeneralizedLoadForecast
from .incremental import PlatformMatrices


@dataclass(frozen=True)
class TimedForecastBallastDemandDiagnostic:
    """One independent three-tank diagnostic at a forecast lead time.

    ``lead_time_s`` identifies the discrete physical-load point used to form
    ``diagnostic``.  It does not specify an execution duration or a pump
    schedule.
    """

    lead_time_s: float
    diagnostic: ForecastBallastDemandDiagnostic

    def __post_init__(self) -> None:
        lead_time_s = float(self.lead_time_s)
        if not np.isfinite(lead_time_s) or lead_time_s <= 0.0:
            raise ValueError("lead_time_s must be finite and positive")
        if not isinstance(self.diagnostic, ForecastBallastDemandDiagnostic):
            raise TypeError("diagnostic must be ForecastBallastDemandDiagnostic")
        object.__setattr__(self, "lead_time_s", lead_time_s)


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

    return tuple(
        TimedForecastBallastDemandDiagnostic(
            lead_time_s=forecast.lead_time_at(index),
            diagnostic=diagnose_forecast_ballast_redistribution(
                matrices=matrices,
                current_generalized_load=forecast.current_generalized_load,
                future_generalized_load=forecast.future_load_at(index),
                actual_tank_masses_kg=actual_tank_masses_kg,
                tank_capacities_kg=tank_capacities_kg,
                tank_coordinates_m=tank_coordinates_m,
                gravity_m_s2=gravity_m_s2,
            ),
        )
        for index in range(forecast.horizon_steps)
    )


__all__ = [
    "TimedForecastBallastDemandDiagnostic",
    "diagnose_generalized_load_forecast_ballast_redistribution",
]
