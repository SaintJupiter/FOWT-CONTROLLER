"""Make source-bound RHS ballast endpoints pump-inspectable.

``ForecastRhsBallastDiagnostic`` derives one capacity-bounded, zero-net
redistribution from the frozen state equation at an already bound forecast
endpoint.  This module carries one such full endpoint, or every endpoint in a
forecast horizon, through the existing pump reachability model.  Horizon
previews are independent counterfactuals from the same measured pump state;
they are not a target sequence or a selection rule.  The module does not
expose fractional endpoint paths, choose a target lifecycle, compare options,
or apply forecast reliability or event information.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fowt_platform.ballast_endpoint_path import (
    BallastEndpointPathSample,
    sample_ballast_endpoint_path,
)

from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from .forecast_rhs_ballast_diagnostic import ForecastRhsBallastDiagnostic
from .physical_endpoint_reachability import (
    PhysicalEndpointReachability,
    evaluate_physical_endpoint_reachability,
)


_MASS_TOLERANCE_KG = 1.0e-8
_TIME_TOLERANCE_S = 1.0e-9


def _execution_start_time(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("execution_start_time must be a non-empty string")
    start_time = value.strip()
    if not start_time:
        raise ValueError("execution_start_time must be a non-empty string")
    return start_time


def _same_execution_state(
    left: ExecutionRolloutState,
    right: ExecutionRolloutState,
) -> bool:
    """Compare the complete persisted pump state without reducing it to mass."""

    for attribute in (
        "actual_masses_kg",
        "rate_limited_target_kg",
        "primary_target_masses_kg",
        "signed_flow_m3_min",
        "pump_latched",
        "pump_on_elapsed_s",
        "pump_off_elapsed_s",
        "pump_near_target_s",
        "pump_command_rates_m3_min",
        "last_flow_directions",
    ):
        if not np.array_equal(getattr(left, attribute), getattr(right, attribute)):
            return False
    return True


@dataclass(frozen=True)
class ForecastRhsBallastEndpointPreview:
    """One full RHS-derived endpoint and its actual pump preview.

    ``endpoint_sample`` is an internal compatibility object for the existing
    reachability routine and is always the full diagnostic endpoint
    (``fraction == 1``).  Public consumers should use
    ``target_tank_masses_kg`` instead of assigning fraction semantics to this
    record.
    """

    rhs_diagnostic: ForecastRhsBallastDiagnostic
    endpoint_sample: BallastEndpointPathSample
    reachability: PhysicalEndpointReachability
    execution_start_time: str

    def __post_init__(self) -> None:
        if not isinstance(self.rhs_diagnostic, ForecastRhsBallastDiagnostic):
            raise TypeError("rhs_diagnostic must be ForecastRhsBallastDiagnostic")
        if not isinstance(self.endpoint_sample, BallastEndpointPathSample):
            raise TypeError("endpoint_sample must be BallastEndpointPathSample")
        if not isinstance(self.reachability, PhysicalEndpointReachability):
            raise TypeError("reachability must be PhysicalEndpointReachability")

        rhs_point = self.rhs_diagnostic.rhs_point
        snapshot = rhs_point.trajectory.platform_snapshot
        start_time = _execution_start_time(self.execution_start_time)
        if start_time != rhs_point.trajectory.initial_state_time:
            raise ValueError(
                "execution_start_time must match the rhs trajectory origin"
            )
        if not math.isclose(
            self.endpoint_sample.fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("rhs endpoint preview must use the full diagnostic endpoint")
        if not self.endpoint_sample.within_capacity or not np.isclose(
            self.endpoint_sample.net_mass_delta_from_actual_kg,
            0.0,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "rhs endpoint must be a capacity-bounded zero-net redistribution"
            )

        target = self.rhs_diagnostic.hypothetical_tank_masses_kg
        if not np.allclose(
            self.endpoint_sample.hypothetical_tank_masses_kg,
            target,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("endpoint_sample must equal the rhs diagnostic target")
        if not np.allclose(
            self.endpoint_sample.mass_delta_from_actual_kg,
            target - snapshot.actual_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("endpoint_sample must start from the rhs snapshot masses")
        if np.any(target < 0.0) or np.any(target > snapshot.tank_capacities_kg):
            raise ValueError("rhs diagnostic target must remain within snapshot capacity")
        if not np.allclose(
            self.reachability.execution_start_state.actual_masses_kg,
            snapshot.actual_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("reachability must start from the rhs snapshot masses")
        if not np.allclose(
            snapshot.tank_capacities_kg,
            float(self.reachability.execution_config.tank_capacity_kg),
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("reachability capacity must match the rhs snapshot")
        if not math.isclose(
            self.reachability.lead_time_s,
            self.lead_time_s,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError("reachability lead must match the rhs forecast lead")
        if not np.allclose(
            self.reachability.endpoint_sample.hypothetical_tank_masses_kg,
            target,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("reachability must preview the rhs diagnostic target")
        if not math.isclose(
            self.reachability.endpoint_sample.fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                "reachability must retain the full rhs diagnostic endpoint"
            )
        if not np.allclose(
            self.reachability.endpoint_sample.mass_delta_from_actual_kg,
            target - snapshot.actual_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "reachability endpoint must retain the rhs snapshot mass delta"
            )
        if (
            self.reachability.execution_step.target_operation
            is not ExecutionTargetOperation.TRACK
        ):
            raise ValueError("rhs endpoint preview must use a track operation")
        if not np.allclose(
            self.reachability.execution_step.requested_target_kg,
            target,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError("reachability request must track the rhs diagnostic target")

        object.__setattr__(self, "execution_start_time", start_time)

    @property
    def lead_index(self) -> int:
        """Return the source-bound forecast lead index."""

        return int(self.rhs_diagnostic.lead_index)

    @property
    def lead_time_s(self) -> float:
        """Return the source-bound forecast lead time."""

        return float(self.rhs_diagnostic.lead_time_s)

    @property
    def target_tank_masses_kg(self) -> np.ndarray:
        """Return the full RHS-derived endpoint without path semantics."""

        return np.array(self.rhs_diagnostic.hypothetical_tank_masses_kg, copy=True)


@dataclass(frozen=True)
class ForecastRhsBallastHorizonPreviews:
    """Independent full-endpoint pump previews over one forecast horizon.

    Every preview starts from the same current execution state and uses the
    first forecast interval as its execution-preview duration.  Later records
    therefore answer a narrow, comparable question: what would the pump be
    able to do *now* if its full target were supplied by that later forecast
    endpoint?  The records are not applied sequentially and do not create a
    multi-period target plan.
    """

    previews: tuple[ForecastRhsBallastEndpointPreview, ...]
    execution_preview_duration_s: float
    execution_start_time: str

    def __post_init__(self) -> None:
        previews = tuple(self.previews)
        if not previews:
            raise ValueError("previews must contain at least one forecast endpoint")
        for preview in previews:
            if not isinstance(preview, ForecastRhsBallastEndpointPreview):
                raise TypeError(
                    "previews must contain ForecastRhsBallastEndpointPreview values"
                )

        first = previews[0]
        trajectory = first.rhs_diagnostic.rhs_point.trajectory
        start_time = _execution_start_time(self.execution_start_time)
        if start_time != trajectory.initial_state_time:
            raise ValueError(
                "execution_start_time must match the shared forecast trajectory origin"
            )
        try:
            duration = float(self.execution_preview_duration_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "execution_preview_duration_s must be a positive finite scalar"
            ) from exc
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError(
                "execution_preview_duration_s must be a positive finite scalar"
            )
        if not math.isclose(
            duration,
            first.lead_time_s,
            rel_tol=0.0,
            abs_tol=_TIME_TOLERANCE_S,
        ):
            raise ValueError(
                "execution_preview_duration_s must equal the first forecast lead"
            )

        first_state = first.reachability.execution_start_state
        first_config = first.reachability.execution_config
        if len(previews) != len(trajectory.steps):
            raise ValueError("previews must contain one endpoint per forecast lead")
        for index, preview in enumerate(previews):
            if preview.rhs_diagnostic.rhs_point.trajectory is not trajectory:
                raise ValueError("previews must use one shared forecast trajectory")
            if preview.lead_index != index:
                raise ValueError("previews must retain contiguous forecast leads")
            if not math.isclose(
                preview.lead_time_s,
                trajectory.steps[index].end_time_s,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ):
                raise ValueError("previews must retain the forecast lead times")
            if preview.execution_start_time != start_time:
                raise ValueError("previews must share the execution start time")
            if not math.isclose(
                preview.reachability.execution_duration_s,
                duration,
                rel_tol=0.0,
                abs_tol=_TIME_TOLERANCE_S,
            ):
                raise ValueError("previews must share one execution-preview duration")
            if not _same_execution_state(
                preview.reachability.execution_start_state,
                first_state,
            ):
                raise ValueError("previews must share the complete execution start state")
            if preview.reachability.execution_config != first_config:
                raise ValueError("previews must share one execution configuration")

        object.__setattr__(self, "previews", previews)
        object.__setattr__(self, "execution_preview_duration_s", duration)
        object.__setattr__(self, "execution_start_time", start_time)

    @property
    def trajectory(self):
        """Return the source-bound forecast trajectory shared by all previews."""

        return self.previews[0].rhs_diagnostic.rhs_point.trajectory

    @property
    def first_preview(self) -> ForecastRhsBallastEndpointPreview:
        """Return the only endpoint used by the current first-interval path."""

        return self.previews[0]

    def preview_for_lead(self, lead_index: int) -> ForecastRhsBallastEndpointPreview:
        """Return one source-bound horizon endpoint without selecting it."""

        if not isinstance(lead_index, int) or isinstance(lead_index, bool):
            raise TypeError("lead_index must be an integer")
        try:
            return self.previews[lead_index]
        except IndexError as exc:
            raise IndexError("lead_index is outside the forecast horizon") from exc


def preview_forecast_rhs_ballast_endpoint(
    *,
    rhs_diagnostic: ForecastRhsBallastDiagnostic,
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_duration_s: Any,
    execution_start_time: Any,
) -> ForecastRhsBallastEndpointPreview:
    """Preview the complete RHS-derived endpoint through one pump interval.

    The physical target, start masses, tank capacities and lead time are all
    derived from ``rhs_diagnostic``.  Callers only supply the current execution
    state and the requested execution interval; no endpoint fraction, event
    probability, reliability, candidate or score enters this function.
    """

    if not isinstance(rhs_diagnostic, ForecastRhsBallastDiagnostic):
        raise TypeError("rhs_diagnostic must be ForecastRhsBallastDiagnostic")
    if not isinstance(execution_state, ExecutionRolloutState):
        raise TypeError("execution_state must be ExecutionRolloutState")
    if not isinstance(execution_config, ExecutionRolloutConfig):
        raise TypeError("execution_config must be ExecutionRolloutConfig")
    snapshot = rhs_diagnostic.rhs_point.trajectory.platform_snapshot
    target = rhs_diagnostic.hypothetical_tank_masses_kg
    endpoint_sample = sample_ballast_endpoint_path(
        actual_tank_masses_kg=snapshot.actual_tank_masses_kg,
        diagnostic_target_tank_masses_kg=target,
        tank_capacities_kg=snapshot.tank_capacities_kg,
        fractions=(1.0,),
    )[0]
    reachability = evaluate_physical_endpoint_reachability(
        lead_time_s=rhs_diagnostic.lead_time_s,
        endpoint_sample=endpoint_sample,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=execution_duration_s,
    )
    return ForecastRhsBallastEndpointPreview(
        rhs_diagnostic=rhs_diagnostic,
        endpoint_sample=endpoint_sample,
        reachability=reachability,
        execution_start_time=execution_start_time,
    )


def preview_forecast_rhs_ballast_horizon(
    *,
    rhs_diagnostics: tuple[ForecastRhsBallastDiagnostic, ...],
    execution_state: ExecutionRolloutState,
    execution_config: ExecutionRolloutConfig,
    execution_start_time: Any,
) -> ForecastRhsBallastHorizonPreviews:
    """Preview every RHS endpoint from one current actuator state.

    The first forecast lead fixes the duration of the current actuator preview.
    Each later endpoint is deliberately previewed from that same state over
    the same duration, rather than treating independent endpoint diagnostics
    as a sequential control schedule.
    """

    diagnostics = tuple(rhs_diagnostics)
    if not diagnostics:
        raise ValueError("rhs_diagnostics must contain at least one forecast endpoint")
    first = diagnostics[0]
    if not isinstance(first, ForecastRhsBallastDiagnostic):
        raise TypeError("rhs_diagnostics must contain ForecastRhsBallastDiagnostic values")
    duration = first.lead_time_s
    previews = tuple(
        preview_forecast_rhs_ballast_endpoint(
            rhs_diagnostic=diagnostic,
            execution_state=execution_state,
            execution_config=execution_config,
            execution_duration_s=duration,
            execution_start_time=execution_start_time,
        )
        for diagnostic in diagnostics
    )
    return ForecastRhsBallastHorizonPreviews(
        previews=previews,
        execution_preview_duration_s=duration,
        execution_start_time=execution_start_time,
    )


__all__ = [
    "ForecastRhsBallastEndpointPreview",
    "ForecastRhsBallastHorizonPreviews",
    "preview_forecast_rhs_ballast_endpoint",
    "preview_forecast_rhs_ballast_horizon",
]
