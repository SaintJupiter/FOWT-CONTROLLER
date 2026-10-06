"""Describe the ballast compensation actually completed by one pump preview.

The physical forecast chain first forms a bounded zero-net-mass endpoint and
then asks the pump model what can actually be reached before a selected
forecast lead.  This module closes that factual gap: it converts the pump
preview's *actual* tank-mass change into a pitch-roll ballast load and reports
the remaining relative external load.

It is not a controller.  It does not name, rank, authorize or commit an
action, and it does not interpret the residual as a posture, safety or
performance result.  Independent exchange with the sea is kept explicit via
the executed net ballast-mass change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform.ballast_moment_allocation import (
    pitch_roll_moment_from_tank_mass_deltas,
)
from fowt_platform.ballast_snapshot import BallastModelSnapshot

from .physical_forecast_endpoint_preview import PhysicalForecastEndpointPreview


_MASS_TOLERANCE_KG = 1.0e-8
# Endpoint samples are formed by subtracting tank masses near the 1e8 kg
# baseline used in the current low-order fixtures.  The resulting roundoff is
# a few micro-Nm after conversion to a gravity moment, so retain a narrow
# absolute tolerance for equivalent endpoint constructions.
_LOAD_TOLERANCE_NM = 1.0e-5


def _readonly_array(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain finite values with shape {shape}"
        ) from exc
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysicalExecutionCompensationDiagnostic:
    """One source-bound physical endpoint after its actual pump execution.

    All pitch-roll vectors use the frozen ``(pitch, roll) = (M, K)`` order.
    ``relative_external_pitch_roll_load_nm`` is the future-minus-current
    external load at the selected forecast lead.  The post-execution residual
    is defined only as that relative load plus the ballast load produced by
    the actual tank-mass change during the preview.
    """

    preview: PhysicalForecastEndpointPreview
    platform_snapshot: BallastModelSnapshot
    execution_start_tank_masses_kg: Any
    execution_final_tank_masses_kg: Any
    endpoint_ballast_pitch_roll_moment_nm: Any
    executed_ballast_pitch_roll_moment_nm: Any
    remaining_endpoint_ballast_pitch_roll_moment_nm: Any
    relative_external_pitch_roll_load_nm: Any
    relative_load_residual_after_execution_nm: Any
    executed_net_mass_delta_kg: float

    def __post_init__(self) -> None:
        if not isinstance(self.preview, PhysicalForecastEndpointPreview):
            raise TypeError("preview must be PhysicalForecastEndpointPreview")
        if not isinstance(self.platform_snapshot, BallastModelSnapshot):
            raise TypeError("platform_snapshot must be BallastModelSnapshot")

        for name in (
            "execution_start_tank_masses_kg",
            "execution_final_tank_masses_kg",
        ):
            object.__setattr__(self, name, _readonly_array(name, getattr(self, name), (3,)))
        for name in (
            "endpoint_ballast_pitch_roll_moment_nm",
            "executed_ballast_pitch_roll_moment_nm",
            "remaining_endpoint_ballast_pitch_roll_moment_nm",
            "relative_external_pitch_roll_load_nm",
            "relative_load_residual_after_execution_nm",
        ):
            object.__setattr__(self, name, _readonly_array(name, getattr(self, name), (2,)))

        start = self.execution_start_tank_masses_kg
        final = self.execution_final_tank_masses_kg
        snapshot = self.platform_snapshot
        endpoint = self.preview.endpoint_sample.hypothetical_tank_masses_kg
        endpoint_delta = self.preview.endpoint_sample.mass_delta_from_actual_kg
        step = self.preview.reachability.execution_step

        inferred_start = endpoint - endpoint_delta
        if not np.allclose(start, inferred_start, rtol=0.0, atol=_MASS_TOLERANCE_KG):
            raise ValueError(
                "execution_start_tank_masses_kg must equal the endpoint preview start state"
            )
        if not np.allclose(
            start,
            self.preview.timed_diagnostic.actual_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "execution start must match the physical diagnostic actual tank state"
            )
        if not np.allclose(
            start,
            snapshot.actual_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "platform_snapshot actual tank masses must match the execution start"
            )
        if not np.allclose(
            snapshot.tank_capacities_kg,
            self.preview.timed_diagnostic.tank_capacities_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "platform_snapshot tank capacities must match the physical diagnostic"
            )
        if not np.allclose(
            snapshot.tank_coordinates_m,
            self.preview.timed_diagnostic.tank_coordinates_m,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "platform_snapshot tank coordinates must match the physical diagnostic"
            )
        if not np.isclose(
            snapshot.gravity_m_s2,
            self.preview.timed_diagnostic.gravity_m_s2,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "platform_snapshot gravity must match the physical diagnostic"
            )
        if np.any(endpoint < 0.0) or np.any(endpoint > snapshot.tank_capacities_kg):
            raise ValueError("physical endpoint must remain within snapshot capacities")
        if not np.allclose(
            final,
            step.state.actual_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "execution_final_tank_masses_kg must match the pump preview result"
            )
        if np.any(final < 0.0) or np.any(final > snapshot.tank_capacities_kg):
            raise ValueError("execution final tank masses must remain within snapshot capacities")

        executed_delta = final - start
        if not np.allclose(
            executed_delta,
            step.mass_delta_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "pump preview mass_delta_kg must equal final minus execution start masses"
            )
        expected_endpoint_moment = pitch_roll_moment_from_tank_mass_deltas(
            tank_mass_deltas_kg=endpoint_delta,
            tank_coordinates_m=snapshot.tank_coordinates_m,
            gravity_m_s2=snapshot.gravity_m_s2,
        )
        expected_executed_moment = pitch_roll_moment_from_tank_mass_deltas(
            tank_mass_deltas_kg=executed_delta,
            tank_coordinates_m=snapshot.tank_coordinates_m,
            gravity_m_s2=snapshot.gravity_m_s2,
        )
        expected_remaining_moment = expected_endpoint_moment - expected_executed_moment
        expected_fractional_endpoint_moment = (
            self.preview.endpoint_sample.fraction
            * self.preview.timed_diagnostic.diagnostic.allocation.achieved_pitch_roll_moment_nm
        )
        expected_relative_load = (
            self.preview.timed_diagnostic.diagnostic.restoring_diagnostic.relative_pitch_roll_load_nm
        )
        expected_residual = expected_relative_load + expected_executed_moment
        if not np.allclose(
            self.endpoint_ballast_pitch_roll_moment_nm,
            expected_endpoint_moment,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "endpoint_ballast_pitch_roll_moment_nm must use the physical endpoint"
            )
        if not np.allclose(
            expected_endpoint_moment,
            expected_fractional_endpoint_moment,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "physical endpoint moment must match the selected allocation fraction"
            )
        if not np.allclose(
            self.executed_ballast_pitch_roll_moment_nm,
            expected_executed_moment,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "executed_ballast_pitch_roll_moment_nm must use the actual pump result"
            )
        if not np.allclose(
            self.remaining_endpoint_ballast_pitch_roll_moment_nm,
            expected_remaining_moment,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "remaining endpoint ballast moment must equal endpoint minus actual execution"
            )
        if not np.allclose(
            self.relative_external_pitch_roll_load_nm,
            expected_relative_load,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError("relative external load must match the selected forecast lead")
        if not np.allclose(
            self.relative_load_residual_after_execution_nm,
            expected_residual,
            rtol=0.0,
            atol=_LOAD_TOLERANCE_NM,
        ):
            raise ValueError(
                "relative load residual must equal external relative load plus actual ballast load"
            )

        net = float(self.executed_net_mass_delta_kg)
        if not np.isfinite(net) or not np.isclose(
            net,
            float(np.sum(executed_delta)),
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "executed_net_mass_delta_kg must equal the actual summed tank-mass change"
            )
        if not np.isclose(
            net,
            self.preview.reachability.executed_net_mass_delta_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "executed net mass must match the physical reachability preview"
            )
        object.__setattr__(self, "executed_net_mass_delta_kg", net)


def diagnose_actual_execution_compensation(
    *,
    preview: PhysicalForecastEndpointPreview,
    platform_snapshot: BallastModelSnapshot,
) -> PhysicalExecutionCompensationDiagnostic:
    """Describe the physical compensation actually produced in one preview.

    The supplied snapshot freezes the geometry, gravity, capacity bounds and
    actual start state used to interpret the execution result.  The returned
    diagnostic remains local to one selected endpoint and lead time.
    """

    if not isinstance(preview, PhysicalForecastEndpointPreview):
        raise TypeError("preview must be PhysicalForecastEndpointPreview")
    if not isinstance(platform_snapshot, BallastModelSnapshot):
        raise TypeError("platform_snapshot must be BallastModelSnapshot")

    endpoint = preview.endpoint_sample.hypothetical_tank_masses_kg
    start = endpoint - preview.endpoint_sample.mass_delta_from_actual_kg
    final = preview.reachability.execution_step.state.actual_masses_kg
    endpoint_moment = pitch_roll_moment_from_tank_mass_deltas(
        tank_mass_deltas_kg=endpoint - start,
        tank_coordinates_m=platform_snapshot.tank_coordinates_m,
        gravity_m_s2=platform_snapshot.gravity_m_s2,
    )
    executed_moment = pitch_roll_moment_from_tank_mass_deltas(
        tank_mass_deltas_kg=final - start,
        tank_coordinates_m=platform_snapshot.tank_coordinates_m,
        gravity_m_s2=platform_snapshot.gravity_m_s2,
    )
    relative_load = (
        preview.timed_diagnostic.diagnostic.restoring_diagnostic.relative_pitch_roll_load_nm
    )
    return PhysicalExecutionCompensationDiagnostic(
        preview=preview,
        platform_snapshot=platform_snapshot,
        execution_start_tank_masses_kg=start,
        execution_final_tank_masses_kg=final,
        endpoint_ballast_pitch_roll_moment_nm=endpoint_moment,
        executed_ballast_pitch_roll_moment_nm=executed_moment,
        remaining_endpoint_ballast_pitch_roll_moment_nm=(
            endpoint_moment - executed_moment
        ),
        relative_external_pitch_roll_load_nm=relative_load,
        relative_load_residual_after_execution_nm=relative_load + executed_moment,
        executed_net_mass_delta_kg=float(np.sum(final - start)),
    )


__all__ = [
    "PhysicalExecutionCompensationDiagnostic",
    "diagnose_actual_execution_compensation",
]
