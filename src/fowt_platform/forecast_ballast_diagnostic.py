"""Trace an explicit forecast load change through a three-tank diagnostic.

This module deliberately composes two bounded physical diagnostics without
entering controller ranking.  It maps an already-defined future generalized
load change to the ballast redistribution that would counteract its local
pitch-roll component, subject only to the present three-tank capacity bounds.

It does not create loads from wind forecasts, select a candidate action,
model a pump, or claim that the returned tank masses are reachable in one
control period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .ballast_moment_allocation import (
    BallastMomentAllocation,
    allocate_pitch_roll_moment_to_tanks,
)
from .incremental import PlatformMatrices
from .pitch_roll_restoring_diagnostic import (
    PitchRollRestoringDiagnostic,
    diagnose_pitch_roll_restoring_demand,
)


@dataclass(frozen=True)
class ForecastBallastDemandDiagnostic:
    """Bounded physical diagnostic from relative load to tank distribution.

    ``restoring_diagnostic`` expresses the forecast-minus-current external
    load in ``(pitch, roll)`` order.  ``allocation`` then requests the
    opposite ballast-generated moment under a zero-net-mass three-tank
    redistribution assumption.

    The result is not a controller target.  In particular, it omits pump
    flow limits, control-period duration, target lifecycle and candidate
    ranking.  All input generalized loads must use one frozen platform axis
    system, reference point and ``[X, Y, Z, K, M, N]`` convention.
    """

    restoring_diagnostic: PitchRollRestoringDiagnostic
    allocation: BallastMomentAllocation

    def __post_init__(self) -> None:
        if not isinstance(self.restoring_diagnostic, PitchRollRestoringDiagnostic):
            raise TypeError("restoring_diagnostic must be PitchRollRestoringDiagnostic")
        if not isinstance(self.allocation, BallastMomentAllocation):
            raise TypeError("allocation must be BallastMomentAllocation")

    @property
    def remaining_relative_pitch_roll_load_nm(self) -> np.ndarray:
        """Return the load increment remaining after the achieved ballast load.

        It is defined as ``relative_external_load + achieved_ballast_load``.
        A fully feasible allocation therefore returns zero.  This quantity is
        the negative of ``allocation.residual_pitch_roll_moment_nm`` because
        the allocation request is the counteracting ballast moment.
        """

        result = np.array(
            self.restoring_diagnostic.relative_pitch_roll_load_nm
            + self.allocation.achieved_pitch_roll_moment_nm,
            dtype=float,
            copy=True,
        )
        result.setflags(write=False)
        return result


def diagnose_forecast_ballast_redistribution(
    *,
    matrices: PlatformMatrices,
    current_generalized_load: Any,
    future_generalized_load: Any,
    actual_tank_masses_kg: Any,
    tank_capacities_kg: Any,
    tank_coordinates_m: Any,
    gravity_m_s2: Any = 9.81,
) -> ForecastBallastDemandDiagnostic:
    """Diagnose a capacity-bounded three-tank response to an explicit load change.

    ``current_generalized_load`` and ``future_generalized_load`` are already
    expressed at the same platform reference point and in the same frozen
    axes.  The future load is not inferred from a wind prediction here.  The
    returned allocation preserves total ballast mass and should only be read
    as an instantaneous redistribution diagnostic.
    """

    restoring = diagnose_pitch_roll_restoring_demand(
        matrices=matrices,
        current_generalized_load=current_generalized_load,
        future_generalized_load=future_generalized_load,
    )
    allocation = allocate_pitch_roll_moment_to_tanks(
        requested_pitch_roll_moment_nm=restoring.counteracting_pitch_roll_moment_nm,
        actual_tank_masses_kg=actual_tank_masses_kg,
        tank_capacities_kg=tank_capacities_kg,
        tank_coordinates_m=tank_coordinates_m,
        gravity_m_s2=gravity_m_s2,
    )
    return ForecastBallastDemandDiagnostic(
        restoring_diagnostic=restoring,
        allocation=allocation,
    )


__all__ = [
    "ForecastBallastDemandDiagnostic",
    "diagnose_forecast_ballast_redistribution",
]
