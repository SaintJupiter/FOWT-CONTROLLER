"""State-conditioned three-tank ballast diagnostics at forecast endpoints.

The forecast trajectory already binds every future endpoint to one physical
wind load, one frozen tank snapshot and one low-order platform state.  This
module maps the pitch-roll entries of that *same-time* state-equation right
hand side to a capacity-bounded, zero-net-mass ballast redistribution.

It is deliberately a frozen-endpoint diagnostic.  It does not solve a new
equilibrium, update tank-dependent matrices after the hypothetical
redistribution, schedule a pump, select a lifecycle, or claim that the
resulting tank state is the preferred control target.  Those questions remain
downstream work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform.ballast import ballast_gravity_load_about_reference
from fowt_platform.ballast_moment_allocation import (
    BallastMomentAllocation,
    allocate_pitch_roll_moment_to_tanks,
)

from .forecast_platform_rhs_diagnostic import (
    ForecastPlatformRhsPointDiagnostic,
    diagnose_forecast_platform_rhs,
)
from .forecast_platform_trajectory import ForecastPlatformTrajectory


_PITCH_ROLL_INDICES = np.array([4, 3], dtype=int)
_LOAD_TOLERANCE_NM = 1.0e-6
_MASS_TOLERANCE_KG = 1.0e-8


def _readonly_vector(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape} and finite values") from exc
    if vector.shape != shape or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    result = np.array(vector, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ForecastRhsBallastDiagnostic:
    """One frozen future RHS mapped to a bounded ballast redistribution.

    The request is the negative of the endpoint's dynamic pitch-roll right
    hand side in the fixed order ``(pitch, roll) = (M, K)``.  The reported
    residual adds only the direct incremental ballast gravity load to that
    frozen-matrix RHS.  It does not reassemble mass or weight-stiffness terms
    for the hypothetical tank state.  Due to six-degree-of-freedom mass
    coupling, neither a zero residual nor a capacity-feasible allocation
    implies zero pitch or roll acceleration in a reassembled future platform
    state.
    """

    rhs_point: ForecastPlatformRhsPointDiagnostic
    requested_ballast_pitch_roll_load_nm: Any
    allocation: BallastMomentAllocation
    incremental_ballast_generalized_load: Any
    frozen_matrix_direct_load_residual_pitch_roll_nm: Any

    def __post_init__(self) -> None:
        if not isinstance(self.rhs_point, ForecastPlatformRhsPointDiagnostic):
            raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
        if not isinstance(self.allocation, BallastMomentAllocation):
            raise TypeError("allocation must be BallastMomentAllocation")

        requested = _readonly_vector(
            "requested_ballast_pitch_roll_load_nm",
            self.requested_ballast_pitch_roll_load_nm,
            (2,),
        )
        incremental_load = _readonly_vector(
            "incremental_ballast_generalized_load",
            self.incremental_ballast_generalized_load,
            (6,),
        )
        residual = _readonly_vector(
            "frozen_matrix_direct_load_residual_pitch_roll_nm",
            self.frozen_matrix_direct_load_residual_pitch_roll_nm,
            (2,),
        )

        snapshot = self.rhs_point.trajectory.platform_snapshot
        expected_request = -self.rhs_point.dynamic_rhs_generalized_load[
            _PITCH_ROLL_INDICES
        ]
        if not np.allclose(
            requested,
            expected_request,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "requested ballast pitch-roll load must counter the matching dynamic RHS"
            )

        expected_allocation = allocate_pitch_roll_moment_to_tanks(
            requested_pitch_roll_moment_nm=requested,
            actual_tank_masses_kg=snapshot.actual_tank_masses_kg,
            tank_capacities_kg=snapshot.tank_capacities_kg,
            tank_coordinates_m=snapshot.tank_coordinates_m,
            gravity_m_s2=snapshot.gravity_m_s2,
        )
        for name, tolerance in (
            ("tank_mass_deltas_kg", _MASS_TOLERANCE_KG),
            ("target_tank_masses_kg", _MASS_TOLERANCE_KG),
            ("achieved_pitch_roll_moment_nm", _LOAD_TOLERANCE_NM),
            ("residual_pitch_roll_moment_nm", _LOAD_TOLERANCE_NM),
        ):
            if not np.allclose(
                getattr(self.allocation, name),
                getattr(expected_allocation, name),
                rtol=0.0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"allocation {name} must match the endpoint snapshot"
                )
        if not np.isclose(
            self.allocation.feasible_scale,
            expected_allocation.feasible_scale,
            rtol=0.0,
            atol=1.0e-12,
        ) or self.allocation.capacity_limited is not expected_allocation.capacity_limited:
            raise ValueError("allocation feasibility must match the endpoint snapshot")

        expected_incremental_load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=self.allocation.tank_mass_deltas_kg,
            tank_coordinates_m=snapshot.tank_coordinates_m,
            gravity_m_s2=snapshot.gravity_m_s2,
        )
        if not np.allclose(
            incremental_load,
            expected_incremental_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "incremental ballast load must match the allocated tank redistribution"
            )

        expected_residual = (
            self.rhs_point.dynamic_rhs_generalized_load[_PITCH_ROLL_INDICES]
            + incremental_load[_PITCH_ROLL_INDICES]
        )
        if not np.allclose(
            residual,
            expected_residual,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "frozen-matrix direct-load residual must use the same endpoint state and allocation"
            )
        if not np.allclose(
            residual,
            -self.allocation.residual_pitch_roll_moment_nm,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "frozen-matrix direct-load residual must agree with the allocation residual"
            )

        object.__setattr__(self, "requested_ballast_pitch_roll_load_nm", requested)
        object.__setattr__(
            self,
            "incremental_ballast_generalized_load",
            incremental_load,
        )
        object.__setattr__(
            self,
            "frozen_matrix_direct_load_residual_pitch_roll_nm",
            residual,
        )

    @property
    def lead_index(self) -> int:
        """Return the source-bound forecast lead index."""

        return int(self.rhs_point.lead_index)

    @property
    def lead_time_s(self) -> float:
        """Return the source-bound endpoint time after the forecast origin."""

        return float(self.rhs_point.lead_time_s)

    @property
    def hypothetical_tank_masses_kg(self) -> np.ndarray:
        """Return the zero-net redistribution endpoint for this diagnostic."""

        return np.array(self.allocation.target_tank_masses_kg, copy=True)


def diagnose_forecast_rhs_ballast_redistribution(
    *,
    rhs_point: ForecastPlatformRhsPointDiagnostic,
) -> ForecastRhsBallastDiagnostic:
    """Map one same-time future RHS to a bounded three-tank diagnostic.

    ``rhs_point`` already carries the matched future platform state, frozen
    snapshot and physical generalized loads.  This function deliberately
    accepts no raw forecast, event score, reliability value, controller state,
    candidate fraction or pump parameter.
    """

    if not isinstance(rhs_point, ForecastPlatformRhsPointDiagnostic):
        raise TypeError("rhs_point must be ForecastPlatformRhsPointDiagnostic")
    snapshot = rhs_point.trajectory.platform_snapshot
    requested = -rhs_point.dynamic_rhs_generalized_load[_PITCH_ROLL_INDICES]
    allocation = allocate_pitch_roll_moment_to_tanks(
        requested_pitch_roll_moment_nm=requested,
        actual_tank_masses_kg=snapshot.actual_tank_masses_kg,
        tank_capacities_kg=snapshot.tank_capacities_kg,
        tank_coordinates_m=snapshot.tank_coordinates_m,
        gravity_m_s2=snapshot.gravity_m_s2,
    )
    incremental_load = ballast_gravity_load_about_reference(
        tank_mass_deltas_kg=allocation.tank_mass_deltas_kg,
        tank_coordinates_m=snapshot.tank_coordinates_m,
        gravity_m_s2=snapshot.gravity_m_s2,
    )
    residual = (
        rhs_point.dynamic_rhs_generalized_load[_PITCH_ROLL_INDICES]
        + incremental_load[_PITCH_ROLL_INDICES]
    )
    return ForecastRhsBallastDiagnostic(
        rhs_point=rhs_point,
        requested_ballast_pitch_roll_load_nm=requested,
        allocation=allocation,
        incremental_ballast_generalized_load=incremental_load,
        frozen_matrix_direct_load_residual_pitch_roll_nm=residual,
    )


def diagnose_forecast_horizon_rhs_ballast_redistributions(
    *,
    trajectory: ForecastPlatformTrajectory,
) -> tuple[ForecastRhsBallastDiagnostic, ...]:
    """Map every discrete forecast endpoint to a same-time ballast fact.

    The returned records retain one zero-net, capacity-bounded redistribution
    for each source-bound forecast lead.  They are deliberately not action
    candidates, a target sequence, or a horizon-level demand.  In particular,
    the function does not decide which endpoint should supply the next target
    or execute any pump preview.  It only prevents the later decision layer
    from silently reducing a multi-step forecast to its first point.
    """

    if not isinstance(trajectory, ForecastPlatformTrajectory):
        raise TypeError("trajectory must be ForecastPlatformTrajectory")
    return tuple(
        diagnose_forecast_rhs_ballast_redistribution(rhs_point=rhs_point)
        for rhs_point in diagnose_forecast_platform_rhs(trajectory=trajectory)
    )


__all__ = [
    "ForecastRhsBallastDiagnostic",
    "diagnose_forecast_horizon_rhs_ballast_redistributions",
    "diagnose_forecast_rhs_ballast_redistribution",
]
