"""Choose the nearest usable forecast endpoint under caller-owned posture bounds.

This is a deliberately narrow bridge between the source-bound endpoint response
facts and a later control layer.  It does not generate endpoints, assign
weights, rank pump cost, or send a command to the plant.  When continuing the
current target falls outside the declared held-horizon bounds, the earliest
forecast endpoint whose held response remains inside them is retained.  Lead
order is the only preference: it keeps the current decision closest to the
forecast origin without introducing a score or a future command sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .forecast_endpoint_horizon_response import (
    ForecastEndpointHeldResponse,
    ForecastEndpointHeldResponseSet,
)
from .physical_forecast_admission import PhysicalForecastLeadReliabilityScope
from .physical_lifecycle_comparison import PhysicalLifecycleHorizonComparison
from .physical_lifecycle_selection import PhysicalPostureLimits


_PITCH_ROLL_INDICES = (4, 3)


class ForecastEndpointPostureAuthorizationReason(str, Enum):
    """Why a forecast endpoint was or was not authorized for this block."""

    FORECAST_NOT_ADMITTED = "forecast_not_admitted"
    INITIAL_STATE_OUTSIDE_SCOPE = "initial_state_outside_scope"
    CONTINUATION_WITHIN_LIMITS = "continuation_within_limits"
    NO_RELIABILITY_ELIGIBLE_ENDPOINT = "no_reliability_eligible_endpoint"
    EARLIEST_ENDPOINT_WITHIN_LIMITS = "earliest_endpoint_within_limits"
    NO_ENDPOINT_WITHIN_LIMITS = "no_endpoint_within_limits"


def _origin_within_limits(
    *,
    horizon_comparison: PhysicalLifecycleHorizonComparison,
    posture_limits: PhysicalPostureLimits,
) -> bool:
    state = horizon_comparison.continue_existing.current_block_outcome.rollout.start_platform_state
    position = np.abs(np.asarray(state.position[list(_PITCH_ROLL_INDICES)], dtype=float))
    velocity = np.abs(np.asarray(state.velocity[list(_PITCH_ROLL_INDICES)], dtype=float))
    return bool(
        np.all(position <= posture_limits.max_abs_pitch_roll_rad)
        and np.all(velocity <= posture_limits.max_abs_pitch_roll_rate_rad_s)
    )


def _response_within_limits(
    *,
    response: ForecastEndpointHeldResponse,
    posture_limits: PhysicalPostureLimits,
) -> bool:
    return bool(
        np.all(
            response.horizon_peak_abs_pitch_roll_rad
            <= posture_limits.max_abs_pitch_roll_rad
        )
        and np.all(
            response.horizon_peak_abs_pitch_roll_rate_rad_s
            <= posture_limits.max_abs_pitch_roll_rate_rad_s
        )
    )


def _continuation_within_limits(
    *,
    horizon_comparison: PhysicalLifecycleHorizonComparison,
    posture_limits: PhysicalPostureLimits,
) -> bool:
    continuation = horizon_comparison.continue_existing
    return bool(
        np.all(
            continuation.peak_abs_pitch_roll_rad
            <= posture_limits.max_abs_pitch_roll_rad
        )
        and np.all(
            continuation.peak_abs_pitch_roll_rate_rad_s
            <= posture_limits.max_abs_pitch_roll_rate_rad_s
        )
    )


def _derive_authorization(
    *,
    forecast_admitted: bool,
    origin_within_limits: bool,
    continuation_within_limits: bool,
    available_lead_indices: tuple[int, ...],
    eligible_lead_indices: tuple[int, ...],
) -> tuple[int | None, ForecastEndpointPostureAuthorizationReason]:
    """Derive the abstaining authorization outcome from already computed facts."""

    if not forecast_admitted:
        return None, ForecastEndpointPostureAuthorizationReason.FORECAST_NOT_ADMITTED
    if not origin_within_limits:
        return None, ForecastEndpointPostureAuthorizationReason.INITIAL_STATE_OUTSIDE_SCOPE
    if continuation_within_limits:
        return None, ForecastEndpointPostureAuthorizationReason.CONTINUATION_WITHIN_LIMITS
    if not available_lead_indices:
        return (
            None,
            ForecastEndpointPostureAuthorizationReason.NO_RELIABILITY_ELIGIBLE_ENDPOINT,
        )
    if eligible_lead_indices:
        return (
            eligible_lead_indices[0],
            ForecastEndpointPostureAuthorizationReason.EARLIEST_ENDPOINT_WITHIN_LIMITS,
        )
    return None, ForecastEndpointPostureAuthorizationReason.NO_ENDPOINT_WITHIN_LIMITS


@dataclass(frozen=True)
class ForecastEndpointPostureAuthorization:
    """One abstaining current-block forecast-endpoint authorization record.

    ``authorized_lead_index`` identifies the earliest source-bound endpoint
    within the supplied posture bounds.  It exposes that endpoint's already-
    declared *current-block* request only.  It never exposes a later command
    schedule, and ``None`` must not be interpreted as an implicit continue,
    release, or legacy-controller action.
    """

    response_set: ForecastEndpointHeldResponseSet
    horizon_comparison: PhysicalLifecycleHorizonComparison
    posture_limits: PhysicalPostureLimits
    reliability_scope: PhysicalForecastLeadReliabilityScope | None = None
    available_lead_indices: tuple[int, ...] = field(init=False)
    eligible_lead_indices: tuple[int, ...] = field(init=False)
    authorized_lead_index: int | None = field(init=False)
    reason: ForecastEndpointPostureAuthorizationReason = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.response_set, ForecastEndpointHeldResponseSet):
            raise TypeError("response_set must be ForecastEndpointHeldResponseSet")
        if not isinstance(self.horizon_comparison, PhysicalLifecycleHorizonComparison):
            raise TypeError("horizon_comparison must be PhysicalLifecycleHorizonComparison")
        if not isinstance(self.posture_limits, PhysicalPostureLimits):
            raise TypeError("posture_limits must be PhysicalPostureLimits")
        if (
            self.horizon_comparison.current_block_comparison
            is not self.response_set.current_block_comparison
        ):
            raise ValueError(
                "horizon comparison must retain the response set current-block comparison"
            )

        comparison = self.response_set.current_block_comparison
        scope = self.reliability_scope
        if scope is None:
            available = tuple(response.lead_index for response in self.response_set.responses)
        else:
            if not isinstance(scope, PhysicalForecastLeadReliabilityScope):
                raise TypeError(
                    "reliability_scope must be PhysicalForecastLeadReliabilityScope or None"
                )
            if scope.decision_input is not comparison.decision_input:
                raise ValueError(
                    "reliability scope must retain the response-set decision input"
                )
            available = scope.available_lead_indices
        admitted = bool(comparison.forecast_admission.admitted)
        eligible = tuple(
            response.lead_index
            for response in self.response_set.responses
            if response.lead_index in available and _response_within_limits(
                response=response,
                posture_limits=self.posture_limits,
            )
        )
        selected, reason = _derive_authorization(
            forecast_admitted=admitted,
            origin_within_limits=_origin_within_limits(
                horizon_comparison=self.horizon_comparison,
                posture_limits=self.posture_limits,
            ),
            continuation_within_limits=_continuation_within_limits(
                horizon_comparison=self.horizon_comparison,
                posture_limits=self.posture_limits,
            ),
            available_lead_indices=available,
            eligible_lead_indices=eligible,
        )
        if selected is not None:
            self.response_set.response_for_lead(selected)
        object.__setattr__(self, "available_lead_indices", available)
        object.__setattr__(self, "eligible_lead_indices", eligible)
        object.__setattr__(self, "authorized_lead_index", selected)
        object.__setattr__(self, "reason", reason)

    @property
    def authorized_response(self) -> ForecastEndpointHeldResponse | None:
        """Return the retained forecast endpoint response, if any."""

        if self.authorized_lead_index is None:
            return None
        return self.response_set.response_for_lead(self.authorized_lead_index)

    @property
    def authorized_current_execution_request(self):
        """Return only the current-block request carried by the endpoint trace."""

        response = self.authorized_response
        return None if response is None else response.lifecycle_trace.execution_request


def authorize_forecast_endpoint_from_posture_response(
    *,
    response_set: ForecastEndpointHeldResponseSet,
    horizon_comparison: PhysicalLifecycleHorizonComparison,
    posture_limits: PhysicalPostureLimits,
    reliability_scope: PhysicalForecastLeadReliabilityScope | None = None,
) -> ForecastEndpointPostureAuthorization:
    """Evaluate an explicit posture-only authorization without plant writeback."""

    return ForecastEndpointPostureAuthorization(
        response_set=response_set,
        horizon_comparison=horizon_comparison,
        posture_limits=posture_limits,
        reliability_scope=reliability_scope,
    )


__all__ = [
    "ForecastEndpointPostureAuthorization",
    "ForecastEndpointPostureAuthorizationReason",
    "authorize_forecast_endpoint_from_posture_response",
]
