"""Describe trends in frozen forecast RHS ballast requests.

Each forecast endpoint already carries a source-bound pitch-roll ballast
request derived from the frozen platform right-hand side.  This module only
describes the sequence of those requests over the admitted forecast prefix.
It does not read raw wind data or event probabilities, rerun the platform,
form a tank target, or issue a pump command.

The resulting facts can later help a current-cycle selector distinguish a
brief request peak from a persistent build-up.  They are not posture or safety
claims: all candidate requests still require their own physical rollout and
posture check downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .physical_forecast_admission import PhysicalForecastLeadReliabilityScope


class PhysicalForecastRhsBallastAxis(str, Enum):
    """The fixed components of a pitch-roll ballast request."""

    PITCH = "pitch"
    ROLL = "roll"


class PhysicalForecastRhsBallastAxisTrendKind(str, Enum):
    """One component's observable trend over the admitted lead prefix."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    TRANSIENT_PEAK = "transient_peak"
    DIRECTION_REVERSAL = "direction_reversal"
    PERSISTENT_GROWTH = "persistent_growth"
    MIXED_OR_STEADY = "mixed_or_steady"


class PhysicalForecastRhsBallastTrendKind(str, Enum):
    """A conservative combined interpretation of pitch and roll requests."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    TRANSIENT_PEAK = "transient_peak"
    DIRECTION_REVERSAL = "direction_reversal"
    PERSISTENT_GROWTH = "persistent_growth"
    MIXED = "mixed"


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


def _readonly_matrix(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape} and finite values") from exc
    if matrix.shape != shape or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    result = np.array(matrix, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PhysicalForecastRhsBallastTrendPolicy:
    """Caller-owned resolutions for interpreting frozen RHS request changes.

    ``axis_resolution_nm`` is used uniformly for sign, peak return and growth
    checks.  ``minimum_evidence_duration_s`` avoids classifying a pattern from
    an arbitrarily short forecast fragment.  Neither parameter weights a load,
    target, or pump action.
    """

    axis_resolution_nm: Any
    minimum_evidence_duration_s: float

    def __post_init__(self) -> None:
        resolution = _readonly_vector(
            "axis_resolution_nm",
            self.axis_resolution_nm,
            (2,),
        )
        if np.any(resolution <= 0.0):
            raise ValueError("axis_resolution_nm must contain positive values")
        duration = float(self.minimum_evidence_duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise ValueError("minimum_evidence_duration_s must be positive and finite")
        object.__setattr__(self, "axis_resolution_nm", resolution)
        object.__setattr__(self, "minimum_evidence_duration_s", duration)


@dataclass(frozen=True)
class PhysicalForecastRhsBallastAxisTrend:
    """One pitch or roll request-component trend with its source evidence."""

    axis: PhysicalForecastRhsBallastAxis
    kind: PhysicalForecastRhsBallastAxisTrendKind
    peak_lead_index: int | None
    initial_sign: int
    terminal_sign: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis", PhysicalForecastRhsBallastAxis(self.axis))
        object.__setattr__(
            self,
            "kind",
            PhysicalForecastRhsBallastAxisTrendKind(self.kind),
        )
        if self.peak_lead_index is not None and (
            not isinstance(self.peak_lead_index, int)
            or isinstance(self.peak_lead_index, bool)
            or self.peak_lead_index < 0
        ):
            raise ValueError("peak_lead_index must be a non-negative integer or None")
        for name in ("initial_sign", "terminal_sign"):
            value = getattr(self, name)
            if value not in (-1, 0, 1):
                raise ValueError(f"{name} must be -1, 0, or 1")


def _resolved_sign(value: float, resolution_nm: float) -> int:
    if value >= resolution_nm:
        return 1
    if value <= -resolution_nm:
        return -1
    return 0


def _classify_axis_request(
    *,
    axis: PhysicalForecastRhsBallastAxis,
    values_nm: np.ndarray,
    lead_indices: tuple[int, ...],
    lead_times_s: np.ndarray,
    resolution_nm: float,
    minimum_evidence_duration_s: float,
) -> PhysicalForecastRhsBallastAxisTrend:
    """Classify one frozen RHS request component without choosing an action."""

    values = np.asarray(values_nm, dtype=float)
    if values.ndim != 1 or values.shape != lead_times_s.shape:
        raise ValueError("axis request values must match the supplied lead times")
    if len(lead_indices) != len(values):
        raise ValueError("lead indices must match the supplied axis request values")
    if len(values) < 3 or lead_times_s[-1] - lead_times_s[0] < minimum_evidence_duration_s:
        return PhysicalForecastRhsBallastAxisTrend(
            axis=axis,
            kind=PhysicalForecastRhsBallastAxisTrendKind.INSUFFICIENT_EVIDENCE,
            peak_lead_index=None,
            initial_sign=0,
            terminal_sign=0,
        )

    signs = tuple(_resolved_sign(float(value), resolution_nm) for value in values)
    nonzero_signs = tuple(sign for sign in signs if sign != 0)
    if not nonzero_signs:
        return PhysicalForecastRhsBallastAxisTrend(
            axis=axis,
            kind=PhysicalForecastRhsBallastAxisTrendKind.INSUFFICIENT_EVIDENCE,
            peak_lead_index=None,
            initial_sign=0,
            terminal_sign=0,
        )

    peak_position = int(np.argmax(np.abs(values)))
    peak_lead_index = int(lead_indices[peak_position])
    initial_sign = nonzero_signs[0]
    terminal_sign = nonzero_signs[-1]
    if any(sign != initial_sign for sign in nonzero_signs[1:]):
        kind = PhysicalForecastRhsBallastAxisTrendKind.DIRECTION_REVERSAL
    else:
        magnitude = np.abs(values)
        if (
            0 < peak_position < len(values) - 1
            and magnitude[peak_position] - magnitude[0] >= resolution_nm
            and magnitude[peak_position] - magnitude[-1] >= resolution_nm
        ):
            kind = PhysicalForecastRhsBallastAxisTrendKind.TRANSIENT_PEAK
        elif (
            magnitude[-1] - magnitude[0] >= resolution_nm
            and np.all(np.diff(magnitude) >= -resolution_nm)
        ):
            kind = PhysicalForecastRhsBallastAxisTrendKind.PERSISTENT_GROWTH
        else:
            kind = PhysicalForecastRhsBallastAxisTrendKind.MIXED_OR_STEADY

    return PhysicalForecastRhsBallastAxisTrend(
        axis=axis,
        kind=kind,
        peak_lead_index=peak_lead_index,
        initial_sign=initial_sign,
        terminal_sign=terminal_sign,
    )


def _combine_axis_trends(
    axis_trends: tuple[PhysicalForecastRhsBallastAxisTrend, ...],
) -> PhysicalForecastRhsBallastTrendKind:
    resolved = tuple(
        trend.kind
        for trend in axis_trends
        if trend.kind is not PhysicalForecastRhsBallastAxisTrendKind.INSUFFICIENT_EVIDENCE
    )
    if not resolved:
        return PhysicalForecastRhsBallastTrendKind.INSUFFICIENT_EVIDENCE
    if all(
        kind is PhysicalForecastRhsBallastAxisTrendKind.DIRECTION_REVERSAL
        for kind in resolved
    ):
        return PhysicalForecastRhsBallastTrendKind.DIRECTION_REVERSAL
    if all(kind is PhysicalForecastRhsBallastAxisTrendKind.TRANSIENT_PEAK for kind in resolved):
        return PhysicalForecastRhsBallastTrendKind.TRANSIENT_PEAK
    if all(kind is PhysicalForecastRhsBallastAxisTrendKind.PERSISTENT_GROWTH for kind in resolved):
        return PhysicalForecastRhsBallastTrendKind.PERSISTENT_GROWTH
    return PhysicalForecastRhsBallastTrendKind.MIXED


@dataclass(frozen=True, init=False)
class PhysicalForecastRhsBallastTrend:
    """Immutable trend facts for one source-bound forecast RHS request series."""

    decision_input: FirstIntervalPhysicalDecisionInput
    policy: PhysicalForecastRhsBallastTrendPolicy
    reliability_scope: PhysicalForecastLeadReliabilityScope | None
    lead_indices: tuple[int, ...]
    lead_times_s: Any
    requested_pitch_roll_loads_nm: Any
    axis_trends: tuple[PhysicalForecastRhsBallastAxisTrend, ...]
    kind: PhysicalForecastRhsBallastTrendKind

    def __init__(
        self,
        *,
        decision_input: FirstIntervalPhysicalDecisionInput,
        policy: PhysicalForecastRhsBallastTrendPolicy,
        reliability_scope: PhysicalForecastLeadReliabilityScope | None = None,
    ) -> None:
        if not isinstance(decision_input, FirstIntervalPhysicalDecisionInput):
            raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
        if not isinstance(policy, PhysicalForecastRhsBallastTrendPolicy):
            raise TypeError("policy must be PhysicalForecastRhsBallastTrendPolicy")
        if reliability_scope is not None:
            if not isinstance(reliability_scope, PhysicalForecastLeadReliabilityScope):
                raise TypeError(
                    "reliability_scope must be PhysicalForecastLeadReliabilityScope or None"
                )
            if reliability_scope.decision_input is not decision_input:
                raise ValueError("reliability_scope must retain the decision_input")
            lead_indices = reliability_scope.available_lead_indices
        else:
            lead_indices = tuple(range(len(decision_input.horizon_rhs_ballast_diagnostics)))

        expected_prefix = tuple(range(len(lead_indices)))
        if lead_indices != expected_prefix:
            raise ValueError("trend lead indices must form a contiguous prefix")
        lead_times = _readonly_vector(
            "lead_times_s",
            tuple(
                decision_input.horizon_rhs_ballast_diagnostics[index].lead_time_s
                for index in lead_indices
            ),
            (len(lead_indices),),
        )
        request_rows = tuple(
            decision_input.horizon_rhs_ballast_diagnostics[
                index
            ].requested_ballast_pitch_roll_load_nm
            for index in lead_indices
        )
        requests = _readonly_matrix(
            "requested_pitch_roll_loads_nm",
            np.empty((0, 2), dtype=float) if not request_rows else request_rows,
            (len(lead_indices), 2),
        )
        axis_trends = tuple(
            _classify_axis_request(
                axis=axis,
                values_nm=requests[:, component_index],
                lead_indices=lead_indices,
                lead_times_s=lead_times,
                resolution_nm=float(policy.axis_resolution_nm[component_index]),
                minimum_evidence_duration_s=policy.minimum_evidence_duration_s,
            )
            for component_index, axis in enumerate(
                (
                    PhysicalForecastRhsBallastAxis.PITCH,
                    PhysicalForecastRhsBallastAxis.ROLL,
                )
            )
        )

        object.__setattr__(self, "decision_input", decision_input)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "reliability_scope", reliability_scope)
        object.__setattr__(self, "lead_indices", lead_indices)
        object.__setattr__(self, "lead_times_s", lead_times)
        object.__setattr__(self, "requested_pitch_roll_loads_nm", requests)
        object.__setattr__(self, "axis_trends", axis_trends)
        object.__setattr__(self, "kind", _combine_axis_trends(axis_trends))

    @property
    def forecast_identity(self):
        """Return the immutable source identity retained by the request series."""

        return self.decision_input.trajectory.load_assembly.input_identity

def diagnose_physical_forecast_rhs_ballast_trend(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    policy: PhysicalForecastRhsBallastTrendPolicy,
    reliability_scope: PhysicalForecastLeadReliabilityScope | None = None,
) -> PhysicalForecastRhsBallastTrend:
    """Return source-bound frozen-RHS trend facts without selecting a route."""

    return PhysicalForecastRhsBallastTrend(
        decision_input=decision_input,
        policy=policy,
        reliability_scope=reliability_scope,
    )


__all__ = [
    "PhysicalForecastRhsBallastAxis",
    "PhysicalForecastRhsBallastAxisTrend",
    "PhysicalForecastRhsBallastAxisTrendKind",
    "PhysicalForecastRhsBallastTrend",
    "PhysicalForecastRhsBallastTrendKind",
    "PhysicalForecastRhsBallastTrendPolicy",
    "diagnose_physical_forecast_rhs_ballast_trend",
]
