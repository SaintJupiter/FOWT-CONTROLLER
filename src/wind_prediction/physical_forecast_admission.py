"""Bind caller-owned forecast availability facts to physical decision inputs.

The physical chain already turns a concrete ``ForecastEvidence`` record into
source-bound rotor loads, a frozen platform trajectory and one forecast-derived
ballast endpoint.  This module records whether a supervisory rule permits that
future endpoint to enter the current-block lifecycle comparison.  It does not
derive a reliability threshold, read event probabilities, rescale loads or
select a target.

The records close a narrow provenance gap: an admission result or lead-time
availability scope cannot be paired with a different forecast than the one
used to construct the physical facts.  ``ForecastEvidence`` is accepted only
while constructing a record; it is not retained because its arrays and
mappings are caller-mutable.  Event probabilities never enter this module.
Lead reliability can only limit which future leads are considered by a later
decision rule.  It never rescales a physical load, pump request, or response.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .forecast_evidence import ForecastEvidence, validate_forecast_evidence


_WIND_TOLERANCE_MPS = 1.0e-10


def _nonempty_text(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a non-empty string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _unit_interval_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite value in [0, 1]") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return result


def _validate_forecast_matches_physical_input(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    forecast: ForecastEvidence,
) -> None:
    """Verify one mutable forecast before discarding it at this boundary."""

    if not isinstance(decision_input, FirstIntervalPhysicalDecisionInput):
        raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
    if not isinstance(forecast, ForecastEvidence):
        raise TypeError("forecast must be ForecastEvidence")
    validate_forecast_evidence(forecast)
    if not forecast.provides_future_preview:
        raise ValueError("physical forecast admission requires a future preview")

    trajectory = decision_input.trajectory
    assembly = trajectory.load_assembly
    identity = assembly.input_identity
    if forecast.origin_time is None:
        raise ValueError("physical forecast admission requires forecast.origin_time")
    if forecast.origin_time != trajectory.initial_state_time:
        raise ValueError(
            "forecast.origin_time must match the physical decision input origin"
        )
    if forecast.origin_time != identity.forecast_origin_time:
        raise ValueError(
            "forecast.origin_time must match the source-bound load origin"
        )
    if forecast.source != identity.forecast_source:
        raise ValueError("forecast.source must match the source-bound load identity")
    if forecast.model_version != identity.forecast_model_version:
        raise ValueError(
            "forecast.model_version must match the source-bound load identity"
        )
    if not np.isclose(
        forecast.sample_period_s,
        identity.forecast_sample_period_s,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "forecast.sample_period_s must match the source-bound load identity"
        )
    if forecast.horizon_steps != identity.forecast_horizon_steps:
        raise ValueError(
            "forecast horizon must match the source-bound load identity"
        )

    expected_rotor_plane_wind = (
        np.asarray(forecast.uv_ms, dtype=float)
        * assembly.parameters.rotor_plane_speed_scale
    )
    if not np.allclose(
        expected_rotor_plane_wind,
        assembly.future_rotor_plane_enu_downwind_mps,
        rtol=0.0,
        atol=_WIND_TOLERANCE_MPS,
    ):
        raise ValueError(
            "forecast wind vectors must match the source-bound physical load input"
        )


@dataclass(frozen=True, init=False)
class PhysicalForecastAdmission:
    """One source-bound supervisory admission for a physical decision block.

    ``admitted`` and ``basis`` are supplied by the supervisory layer.  The
    constructor validates a supplied ``ForecastEvidence`` against the exact
    physical load input, then discards the mutable evidence object.  The
    record therefore retains only immutable physical facts and does not carry
    reliability or event metadata into selection.  It intentionally does not
    claim that the caller's admission criterion is a safety, accuracy or
    economic criterion.
    """

    decision_input: FirstIntervalPhysicalDecisionInput
    admitted: bool
    basis: str

    def __init__(
        self,
        *,
        decision_input: FirstIntervalPhysicalDecisionInput,
        forecast: ForecastEvidence,
        admitted: bool,
        basis: str,
    ) -> None:
        _validate_forecast_matches_physical_input(
            decision_input=decision_input,
            forecast=forecast,
        )
        if not isinstance(admitted, bool):
            raise TypeError("admitted must be bool")
        object.__setattr__(self, "decision_input", decision_input)
        object.__setattr__(self, "admitted", admitted)
        object.__setattr__(self, "basis", _nonempty_text("basis", basis))

    @property
    def forecast_identity(self):
        """Return the immutable load identity that remains after validation."""

        return self.decision_input.trajectory.load_assembly.input_identity


@dataclass(frozen=True, init=False)
class PhysicalForecastLeadReliabilityScope:
    """Caller-defined horizon scope derived from source-bound lead reliability.

    ``minimum_lead_reliability`` is deliberately supplied by the caller and
    has no default.  The values in ``ForecastEvidence.lead_reliability`` are
    held-out lead diagnostics, not a per-cycle probability of correctness.
    Consequently this record only exposes a conservative contiguous prefix of
    forecast leads.  It does not certify forecast accuracy or select an
    action.
    """

    decision_input: FirstIntervalPhysicalDecisionInput
    minimum_lead_reliability: float
    basis: str
    lead_reliability: tuple[float, ...]
    available_lead_indices: tuple[int, ...]

    def __init__(
        self,
        *,
        decision_input: FirstIntervalPhysicalDecisionInput,
        forecast: ForecastEvidence,
        minimum_lead_reliability: float,
        basis: str,
    ) -> None:
        _validate_forecast_matches_physical_input(
            decision_input=decision_input,
            forecast=forecast,
        )
        minimum = _unit_interval_float(
            "minimum_lead_reliability",
            minimum_lead_reliability,
        )
        reliability = tuple(
            float(value)
            for value in np.asarray(forecast.lead_reliability, dtype=float).reshape(-1)
        )
        available: list[int] = []
        for index, value in enumerate(reliability):
            if value < minimum:
                break
            available.append(index)

        object.__setattr__(self, "decision_input", decision_input)
        object.__setattr__(self, "minimum_lead_reliability", minimum)
        object.__setattr__(self, "basis", _nonempty_text("basis", basis))
        object.__setattr__(self, "lead_reliability", reliability)
        object.__setattr__(self, "available_lead_indices", tuple(available))

    @property
    def forecast_identity(self):
        """Return the immutable identity retained after source validation."""

        return self.decision_input.trajectory.load_assembly.input_identity


def bind_physical_forecast_admission(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    forecast: ForecastEvidence,
    admitted: bool,
    basis: str,
) -> PhysicalForecastAdmission:
    """Create one validated admission fact without interpreting its criterion."""

    return PhysicalForecastAdmission(
        decision_input=decision_input,
        forecast=forecast,
        admitted=admitted,
        basis=basis,
    )


def bind_physical_forecast_lead_reliability_scope(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    forecast: ForecastEvidence,
    minimum_lead_reliability: float,
    basis: str,
) -> PhysicalForecastLeadReliabilityScope:
    """Bind a caller-owned reliability cutoff without interpreting its value."""

    return PhysicalForecastLeadReliabilityScope(
        decision_input=decision_input,
        forecast=forecast,
        minimum_lead_reliability=minimum_lead_reliability,
        basis=basis,
    )


__all__ = [
    "PhysicalForecastAdmission",
    "PhysicalForecastLeadReliabilityScope",
    "bind_physical_forecast_admission",
    "bind_physical_forecast_lead_reliability_scope",
]
