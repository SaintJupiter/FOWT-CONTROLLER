"""Choose one current-block pump request from physical forecast responses.

The surrounding physical chain supplies same-origin held responses for three
meanings of a target: continuing the stored target, releasing it to the
current tank state, and tracking every source-bound forecast endpoint.  This
module is their single command-facing selection point.  It does not create a
future command schedule.  It chooses only the request to be executed in the
current block.

Candidate responses first have to remain within caller-owned posture bounds.
Among the remaining responses, materially lower normalized posture demand is
preferred.  When responses are close on that measure, the current-block pump
burden breaks the tie in the order volume, active time, starts and direction
switches.  This is a constrained lexicographic rule, not a weighted score.
Forecast reliability can restrict which source leads are considered, but never
scales a load, a target or an actuator request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math

import numpy as np

from .execution_rollout import ExecutionRolloutRequest
from .forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockLifecycleRollout,
)
from .forecast_endpoint_horizon_response import (
    ForecastEndpointHeldResponse,
    ForecastEndpointHeldResponseSet,
    assemble_forecast_endpoint_held_responses,
)
from .first_interval_physical_decision_input import FirstIntervalPhysicalDecisionInput
from .physical_forecast_admission import (
    PhysicalForecastAdmission,
    PhysicalForecastLeadReliabilityScope,
)
from .physical_forecast_rhs_ballast_trend import (
    PhysicalForecastRhsBallastTrend,
    PhysicalForecastRhsBallastTrendKind,
)
from .physical_lifecycle_comparison import (
    PhysicalLifecycleHorizonComparison,
    PhysicalLifecycleHorizonOutcome,
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from .physical_lifecycle_selection import PhysicalPostureLimits


_PITCH_ROLL_INDICES = (4, 3)
_ROUTE_TIE_ORDER = {
    "continue_existing": 0,
    "forecast_endpoint": 1,
    "release_to_current": 2,
}


class PhysicalCurrentCycleRoute(str, Enum):
    """The request origins available to the current-block decision."""

    CONTINUE_EXISTING = "continue_existing"
    FORECAST_ENDPOINT = "forecast_endpoint"
    RELEASE_TO_CURRENT = "release_to_current"


class PhysicalCurrentCycleDecisionReason(str, Enum):
    """Why the current-block decision selected a request or abstained."""

    CONTINUE_EXISTING_SELECTED = "continue_existing_selected"
    FORECAST_ENDPOINT_SELECTED = "forecast_endpoint_selected"
    RELEASE_TO_CURRENT_SELECTED = "release_to_current_selected"
    NO_REQUEST_WITHIN_LIMITS = "no_request_within_limits"


class PhysicalCurrentCycleSelectionBasis(str, Enum):
    """The comparison rule that resolved the selected current request."""

    LOWER_POSTURE_DEMAND = "lower_posture_demand"
    LOWER_EXECUTION_BURDEN = "lower_execution_burden_within_posture_margin"
    FORECAST_TREND_ROUTE_PREFERENCE = "forecast_trend_route_preference"
    NO_FEASIBLE_REQUEST = "no_feasible_request"


@dataclass(frozen=True)
class PhysicalCurrentCycleDecisionPolicy:
    """One explicit equivalence tolerance for the lexicographic rule.

    ``posture_equivalence_margin`` is expressed on normalized posture demand:
    ``1.0`` corresponds to one of the caller-owned angle or rate limits.  It
    is not a measured-posture deadband.  It only says when two already
    simulated candidate responses are close enough for execution burden to
    decide between them.
    """

    posture_equivalence_margin: float

    def __post_init__(self) -> None:
        margin = float(self.posture_equivalence_margin)
        if not math.isfinite(margin) or margin < 0.0 or margin > 1.0:
            raise ValueError(
                "posture_equivalence_margin must be finite and lie in [0, 1]"
            )
        object.__setattr__(self, "posture_equivalence_margin", margin)


@dataclass(frozen=True)
class PhysicalCurrentCycleForecastTrendRoutePolicy:
    """Explicit mapping from a frozen-RHS trend fact to target release.

    This policy does not interpret wind, event probabilities, posture, or pump
    cost.  It only permits a transient or reversing RHS request to prefer the
    already posture-equivalent ``release_to_current`` route.  Release makes
    the target equal the current actual tank state.  An idle pump therefore
    remains idle, and an active pump receives a zero request at the next
    actuator substep unless a minimum-on or ramp constraint delays that stop.
    The caller must construct the policy deliberately; passing ``None`` leaves
    the physical selector unchanged.
    """

    release_preferred_trend_kinds: tuple[PhysicalForecastRhsBallastTrendKind, ...]

    def __post_init__(self) -> None:
        try:
            kinds = tuple(
                PhysicalForecastRhsBallastTrendKind(value)
                for value in self.release_preferred_trend_kinds
            )
        except TypeError as exc:
            raise TypeError(
                "release_preferred_trend_kinds must be an iterable of trend kinds"
            ) from exc
        if not kinds:
            raise ValueError("release_preferred_trend_kinds must not be empty")
        if len(set(kinds)) != len(kinds):
            raise ValueError("release_preferred_trend_kinds must not repeat kinds")
        permitted = {
            PhysicalForecastRhsBallastTrendKind.TRANSIENT_PEAK,
            PhysicalForecastRhsBallastTrendKind.DIRECTION_REVERSAL,
        }
        if not set(kinds).issubset(permitted):
            raise ValueError(
                "only transient_peak or direction_reversal may prefer release_to_current"
            )
        object.__setattr__(self, "release_preferred_trend_kinds", kinds)


@dataclass(frozen=True)
class PhysicalCurrentCycleCandidate:
    """One current-block request plus its modeled response and pump burden."""

    route: PhysicalCurrentCycleRoute
    execution_request: ExecutionRolloutRequest
    current_block_rollout: ForecastExecutionBlockLifecycleRollout
    max_normalized_posture: float
    current_block_transferred_volume_m3: float
    current_block_pump_active_time_s: float
    current_block_pump_start_count: int
    current_block_pump_direction_switch_count: int
    current_block_pump_stop_count: int = 0
    forecast_lead_index: int | None = None

    def __post_init__(self) -> None:
        route = PhysicalCurrentCycleRoute(self.route)
        object.__setattr__(self, "route", route)
        if not isinstance(self.execution_request, ExecutionRolloutRequest):
            raise TypeError("execution_request must be ExecutionRolloutRequest")
        if not isinstance(
            self.current_block_rollout,
            ForecastExecutionBlockLifecycleRollout,
        ):
            raise TypeError(
                "current_block_rollout must be ForecastExecutionBlockLifecycleRollout"
            )
        if (
            self.current_block_rollout.lifecycle_trace.execution_request
            is not self.execution_request
        ):
            raise ValueError(
                "candidate request must be the current-block rollout request"
            )
        if route is PhysicalCurrentCycleRoute.FORECAST_ENDPOINT:
            if not isinstance(self.forecast_lead_index, int) or isinstance(
                self.forecast_lead_index,
                bool,
            ):
                raise ValueError(
                    "forecast_endpoint candidates require an integer lead index"
                )
            if self.forecast_lead_index < 0:
                raise ValueError("forecast_lead_index must be non-negative")
        elif self.forecast_lead_index is not None:
            raise ValueError(
                "only forecast_endpoint candidates may retain a lead index"
            )

        quantities = {
            "max_normalized_posture": self.max_normalized_posture,
            "current_block_transferred_volume_m3": self.current_block_transferred_volume_m3,
            "current_block_pump_active_time_s": self.current_block_pump_active_time_s,
        }
        for name, value in quantities.items():
            number = float(value)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
            object.__setattr__(self, name, number)
        for name in (
            "current_block_pump_start_count",
            "current_block_pump_stop_count",
            "current_block_pump_direction_switch_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def within_posture_limits(self) -> bool:
        """Return whether the held response stays within declared limits."""

        return bool(self.max_normalized_posture <= 1.0)


def _origin_within_limits(
    *,
    horizon_comparison: PhysicalLifecycleHorizonComparison,
    posture_limits: PhysicalPostureLimits,
) -> bool:
    state = (
        horizon_comparison.continue_existing.current_block_outcome.rollout.start_platform_state
    )
    position = np.abs(
        np.asarray(state.position[list(_PITCH_ROLL_INDICES)], dtype=float)
    )
    velocity = np.abs(
        np.asarray(state.velocity[list(_PITCH_ROLL_INDICES)], dtype=float)
    )
    return bool(
        np.all(position <= posture_limits.max_abs_pitch_roll_rad)
        and np.all(velocity <= posture_limits.max_abs_pitch_roll_rate_rad_s)
    )


def _normalized_posture_demand(
    *,
    peak_abs_pitch_roll_rad: np.ndarray,
    peak_abs_pitch_roll_rate_rad_s: np.ndarray,
    posture_limits: PhysicalPostureLimits,
) -> float:
    angle_ratio = np.asarray(peak_abs_pitch_roll_rad, dtype=float) / np.asarray(
        posture_limits.max_abs_pitch_roll_rad,
        dtype=float,
    )
    rate_ratio = np.asarray(
        peak_abs_pitch_roll_rate_rad_s,
        dtype=float,
    ) / np.asarray(
        posture_limits.max_abs_pitch_roll_rate_rad_s,
        dtype=float,
    )
    return float(np.max(np.concatenate((angle_ratio, rate_ratio))))


def _rollout_execution_burden(rollout) -> tuple[float, float, int, int, int]:
    """Return only the actual-pump burden of one already replayed block."""

    return (
        float(
            sum(
                interval.physical_path.transferred_volume_m3
                for interval in rollout.intervals
            )
        ),
        float(
            sum(
                interval.physical_path.aggregate_pump_active_time_s
                for interval in rollout.intervals
            )
        ),
        int(
            sum(
                interval.physical_path.pump_start_count
                for interval in rollout.intervals
            )
        ),
        int(
            sum(
                interval.physical_path.pump_stop_count
                for interval in rollout.intervals
            )
        ),
        int(
            sum(
                interval.physical_path.pump_direction_switch_count
                for interval in rollout.intervals
            )
        ),
    )


def _candidate_from_horizon_outcome(
    *,
    route: PhysicalCurrentCycleRoute,
    outcome: PhysicalLifecycleHorizonOutcome,
    posture_limits: PhysicalPostureLimits,
) -> PhysicalCurrentCycleCandidate:
    current = outcome.current_block_outcome
    volume, active_time, starts, stops, switches = _rollout_execution_burden(current.rollout)
    return PhysicalCurrentCycleCandidate(
        route=route,
        execution_request=current.rollout.lifecycle_trace.execution_request,
        current_block_rollout=current.rollout,
        max_normalized_posture=_normalized_posture_demand(
            peak_abs_pitch_roll_rad=outcome.peak_abs_pitch_roll_rad,
            peak_abs_pitch_roll_rate_rad_s=outcome.peak_abs_pitch_roll_rate_rad_s,
            posture_limits=posture_limits,
        ),
        current_block_transferred_volume_m3=volume,
        current_block_pump_active_time_s=active_time,
        current_block_pump_start_count=starts,
        current_block_pump_stop_count=stops,
        current_block_pump_direction_switch_count=switches,
    )


def _candidate_from_forecast_response(
    *,
    response: ForecastEndpointHeldResponse,
    posture_limits: PhysicalPostureLimits,
) -> PhysicalCurrentCycleCandidate:
    volume, active_time, starts, stops, switches = _rollout_execution_burden(
        response.current_block_rollout
    )
    return PhysicalCurrentCycleCandidate(
        route=PhysicalCurrentCycleRoute.FORECAST_ENDPOINT,
        execution_request=response.lifecycle_trace.execution_request,
        current_block_rollout=response.current_block_rollout,
        max_normalized_posture=_normalized_posture_demand(
            peak_abs_pitch_roll_rad=response.horizon_peak_abs_pitch_roll_rad,
            peak_abs_pitch_roll_rate_rad_s=response.horizon_peak_abs_pitch_roll_rate_rad_s,
            posture_limits=posture_limits,
        ),
        current_block_transferred_volume_m3=volume,
        current_block_pump_active_time_s=active_time,
        current_block_pump_start_count=starts,
        current_block_pump_stop_count=stops,
        current_block_pump_direction_switch_count=switches,
        forecast_lead_index=response.lead_index,
    )


def _available_forecast_lead_indices(
    *,
    response_set: ForecastEndpointHeldResponseSet,
    reliability_scope: PhysicalForecastLeadReliabilityScope | None,
) -> tuple[int, ...]:
    if reliability_scope is None:
        return tuple(response.lead_index for response in response_set.responses)
    if not isinstance(reliability_scope, PhysicalForecastLeadReliabilityScope):
        raise TypeError(
            "reliability_scope must be PhysicalForecastLeadReliabilityScope or None"
        )
    if reliability_scope.decision_input is not response_set.current_block_comparison.decision_input:
        raise ValueError(
            "reliability scope must retain the response-set decision input"
        )
    return reliability_scope.available_lead_indices


def _candidate_execution_key(candidate: PhysicalCurrentCycleCandidate) -> tuple:
    """Order equivalent posture responses by current-block execution burden."""

    return (
        candidate.current_block_transferred_volume_m3,
        candidate.current_block_pump_active_time_s,
        candidate.current_block_pump_start_count,
        candidate.current_block_pump_direction_switch_count,
        candidate.max_normalized_posture,
        _ROUTE_TIE_ORDER[candidate.route.value],
        -1 if candidate.forecast_lead_index is None else candidate.forecast_lead_index,
    )


def _select_current_cycle_candidate(
    *,
    candidates: tuple[PhysicalCurrentCycleCandidate, ...],
    policy: PhysicalCurrentCycleDecisionPolicy,
) -> PhysicalCurrentCycleCandidate | None:
    """Choose one safe candidate without a weighted score or hidden fallback."""

    return _select_current_cycle_candidate_with_basis(
        candidates=candidates,
        policy=policy,
    )[0]


def _select_current_cycle_candidate_with_basis(
    *,
    candidates: tuple[PhysicalCurrentCycleCandidate, ...],
    policy: PhysicalCurrentCycleDecisionPolicy,
) -> tuple[PhysicalCurrentCycleCandidate | None, PhysicalCurrentCycleSelectionBasis]:
    """Choose one candidate and retain whether posture or burden resolved it."""

    feasible = tuple(
        candidate for candidate in candidates if candidate.within_posture_limits
    )
    if not feasible:
        return None, PhysicalCurrentCycleSelectionBasis.NO_FEASIBLE_REQUEST
    best_posture = min(candidate.max_normalized_posture for candidate in feasible)
    equivalent = tuple(
        candidate
        for candidate in feasible
        if candidate.max_normalized_posture
        <= best_posture + policy.posture_equivalence_margin
    )
    selected = min(equivalent, key=_candidate_execution_key)
    basis = (
        PhysicalCurrentCycleSelectionBasis.LOWER_POSTURE_DEMAND
        if len(equivalent) == 1
        else PhysicalCurrentCycleSelectionBasis.LOWER_EXECUTION_BURDEN
    )
    return selected, basis


def _select_current_cycle_candidate_with_forecast_trend(
    *,
    candidates: tuple[PhysicalCurrentCycleCandidate, ...],
    policy: PhysicalCurrentCycleDecisionPolicy,
    release_preference_enabled: bool,
) -> tuple[
    PhysicalCurrentCycleCandidate | None,
    PhysicalCurrentCycleSelectionBasis,
    bool,
]:
    """Apply a narrow route preference after hard posture feasibility.

    The caller supplies the trend-to-route mapping explicitly.  When that
    mapping permits it, an already-safe release response can be preferred only
    inside the existing posture-equivalence margin.  The function never
    restores an infeasible response, changes a posture limit, or synthesizes
    a later command.
    """

    base_selected, base_basis = _select_current_cycle_candidate_with_basis(
        candidates=candidates,
        policy=policy,
    )
    if not release_preference_enabled:
        return base_selected, base_basis, False

    feasible = tuple(candidate for candidate in candidates if candidate.within_posture_limits)
    if not feasible:
        return (
            None,
            PhysicalCurrentCycleSelectionBasis.NO_FEASIBLE_REQUEST,
            False,
        )
    if not any(
        candidate.route is PhysicalCurrentCycleRoute.FORECAST_ENDPOINT
        for candidate in feasible
    ):
        return base_selected, base_basis, False
    best_posture = min(candidate.max_normalized_posture for candidate in feasible)
    release_equivalent = tuple(
        candidate
        for candidate in feasible
        if candidate.route is PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT
        and candidate.max_normalized_posture
        <= best_posture + policy.posture_equivalence_margin
    )
    if not release_equivalent:
        return base_selected, base_basis, False

    selected = min(release_equivalent, key=_candidate_execution_key)
    if selected is base_selected:
        return base_selected, base_basis, False
    return (
        selected,
        PhysicalCurrentCycleSelectionBasis.FORECAST_TREND_ROUTE_PREFERENCE,
        True,
    )


def _reason_for(
    candidate: PhysicalCurrentCycleCandidate | None,
) -> PhysicalCurrentCycleDecisionReason:
    if candidate is None:
        return PhysicalCurrentCycleDecisionReason.NO_REQUEST_WITHIN_LIMITS
    if candidate.route is PhysicalCurrentCycleRoute.CONTINUE_EXISTING:
        return PhysicalCurrentCycleDecisionReason.CONTINUE_EXISTING_SELECTED
    if candidate.route is PhysicalCurrentCycleRoute.FORECAST_ENDPOINT:
        return PhysicalCurrentCycleDecisionReason.FORECAST_ENDPOINT_SELECTED
    return PhysicalCurrentCycleDecisionReason.RELEASE_TO_CURRENT_SELECTED


@dataclass(frozen=True)
class PhysicalCurrentCycleDecision:
    """One source-bound, non-writing decision for the current execution block."""

    response_set: ForecastEndpointHeldResponseSet
    horizon_comparison: PhysicalLifecycleHorizonComparison
    posture_limits: PhysicalPostureLimits
    policy: PhysicalCurrentCycleDecisionPolicy
    reliability_scope: PhysicalForecastLeadReliabilityScope | None = None
    forecast_rhs_ballast_trend: PhysicalForecastRhsBallastTrend | None = None
    forecast_trend_route_policy: PhysicalCurrentCycleForecastTrendRoutePolicy | None = None
    origin_within_posture_limits: bool = field(init=False)
    candidates: tuple[PhysicalCurrentCycleCandidate, ...] = field(init=False)
    selected_candidate: PhysicalCurrentCycleCandidate | None = field(init=False)
    selected_route: PhysicalCurrentCycleRoute | None = field(init=False)
    selected_forecast_lead_index: int | None = field(init=False)
    reason: PhysicalCurrentCycleDecisionReason = field(init=False)
    selection_basis: PhysicalCurrentCycleSelectionBasis = field(init=False)
    forecast_trend_route_preference_applied: bool = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.response_set, ForecastEndpointHeldResponseSet):
            raise TypeError("response_set must be ForecastEndpointHeldResponseSet")
        if not isinstance(self.horizon_comparison, PhysicalLifecycleHorizonComparison):
            raise TypeError("horizon_comparison must be PhysicalLifecycleHorizonComparison")
        if not isinstance(self.posture_limits, PhysicalPostureLimits):
            raise TypeError("posture_limits must be PhysicalPostureLimits")
        if not isinstance(self.policy, PhysicalCurrentCycleDecisionPolicy):
            raise TypeError("policy must be PhysicalCurrentCycleDecisionPolicy")
        if (
            self.horizon_comparison.current_block_comparison
            is not self.response_set.current_block_comparison
        ):
            raise ValueError(
                "horizon comparison must retain the response-set current-block comparison"
            )
        if self.forecast_rhs_ballast_trend is not None:
            if not isinstance(
                self.forecast_rhs_ballast_trend,
                PhysicalForecastRhsBallastTrend,
            ):
                raise TypeError(
                    "forecast_rhs_ballast_trend must be PhysicalForecastRhsBallastTrend or None"
                )
            if (
                self.forecast_rhs_ballast_trend.decision_input
                is not self.response_set.current_block_comparison.decision_input
            ):
                raise ValueError(
                    "forecast_rhs_ballast_trend must retain the response-set decision input"
                )
            if (
                self.forecast_rhs_ballast_trend.reliability_scope
                is not self.reliability_scope
            ):
                raise ValueError(
                    "forecast_rhs_ballast_trend must use the decision reliability scope"
                )
        if self.forecast_trend_route_policy is not None:
            if not isinstance(
                self.forecast_trend_route_policy,
                PhysicalCurrentCycleForecastTrendRoutePolicy,
            ):
                raise TypeError(
                    "forecast_trend_route_policy must be PhysicalCurrentCycleForecastTrendRoutePolicy or None"
                )
            if self.forecast_rhs_ballast_trend is None:
                raise ValueError(
                    "forecast_trend_route_policy requires a forecast_rhs_ballast_trend"
                )
            if self.reliability_scope is None:
                raise ValueError(
                    "forecast trend route preference requires a reliability scope"
                )

        comparison = self.response_set.current_block_comparison
        available_leads = set(
            _available_forecast_lead_indices(
                response_set=self.response_set,
                reliability_scope=self.reliability_scope,
            )
        )
        candidates = [
            _candidate_from_horizon_outcome(
                route=PhysicalCurrentCycleRoute.CONTINUE_EXISTING,
                outcome=self.horizon_comparison.continue_existing,
                posture_limits=self.posture_limits,
            ),
            _candidate_from_horizon_outcome(
                route=PhysicalCurrentCycleRoute.RELEASE_TO_CURRENT,
                outcome=self.horizon_comparison.release_to_current,
                posture_limits=self.posture_limits,
            ),
        ]
        candidates.extend(
            _candidate_from_forecast_response(
                response=response,
                posture_limits=self.posture_limits,
            )
            for response in self.response_set.responses
            if response.lead_index in available_leads
        )
        candidate_tuple = tuple(candidates)
        trend_route_preference_enabled = bool(
            self.forecast_trend_route_policy is not None
            and self.forecast_rhs_ballast_trend is not None
            and self.forecast_rhs_ballast_trend.kind
            in self.forecast_trend_route_policy.release_preferred_trend_kinds
        )
        selected, selection_basis, trend_preference_applied = (
            _select_current_cycle_candidate_with_forecast_trend(
                candidates=candidate_tuple,
                policy=self.policy,
                release_preference_enabled=trend_route_preference_enabled,
            )
        )
        object.__setattr__(
            self,
            "origin_within_posture_limits",
            _origin_within_limits(
                horizon_comparison=self.horizon_comparison,
                posture_limits=self.posture_limits,
            ),
        )
        object.__setattr__(self, "candidates", candidate_tuple)
        object.__setattr__(self, "selected_candidate", selected)
        object.__setattr__(
            self,
            "selected_route",
            None if selected is None else selected.route,
        )
        object.__setattr__(
            self,
            "selected_forecast_lead_index",
            None if selected is None else selected.forecast_lead_index,
        )
        object.__setattr__(self, "reason", _reason_for(selected))
        object.__setattr__(self, "selection_basis", selection_basis)
        object.__setattr__(
            self,
            "forecast_trend_route_preference_applied",
            trend_preference_applied,
        )

        if comparison.forecast_admission.admitted is False and any(
            candidate.route is PhysicalCurrentCycleRoute.FORECAST_ENDPOINT
            for candidate in candidate_tuple
        ):
            raise ValueError(
                "non-admitted forecasts must not contribute endpoint candidates"
            )

    @property
    def current_execution_request(self) -> ExecutionRolloutRequest | None:
        """Return the sole request selected for this block, without executing it."""

        return (
            None
            if self.selected_candidate is None
            else self.selected_candidate.execution_request
        )

    @property
    def selected_current_block_rollout(
        self,
    ) -> ForecastExecutionBlockLifecycleRollout | None:
        """Return the exact modeled execution path for the selected request."""

        return (
            None
            if self.selected_candidate is None
            else self.selected_candidate.current_block_rollout
        )

    def require_current_execution_request(self) -> ExecutionRolloutRequest:
        """Return the current request or require the caller to handle abstention."""

        request = self.current_execution_request
        if request is None:
            raise ValueError(
                "current-cycle decision abstained; caller must handle the missing request"
            )
        return request


def assemble_physical_current_cycle_decision(
    *,
    response_set: ForecastEndpointHeldResponseSet,
    horizon_comparison: PhysicalLifecycleHorizonComparison,
    posture_limits: PhysicalPostureLimits,
    reliability_scope: PhysicalForecastLeadReliabilityScope | None = None,
    forecast_rhs_ballast_trend: PhysicalForecastRhsBallastTrend | None = None,
    forecast_trend_route_policy: PhysicalCurrentCycleForecastTrendRoutePolicy | None = None,
    policy: PhysicalCurrentCycleDecisionPolicy,
) -> PhysicalCurrentCycleDecision:
    """Compose one current-block request from held physical response facts."""

    return PhysicalCurrentCycleDecision(
        response_set=response_set,
        horizon_comparison=horizon_comparison,
        posture_limits=posture_limits,
        reliability_scope=reliability_scope,
        forecast_rhs_ballast_trend=forecast_rhs_ballast_trend,
        forecast_trend_route_policy=forecast_trend_route_policy,
        policy=policy,
    )


def assemble_physical_current_cycle_decision_from_input(
    *,
    decision_input: FirstIntervalPhysicalDecisionInput,
    forecast_admission: PhysicalForecastAdmission,
    posture_limits: PhysicalPostureLimits,
    policy: PhysicalCurrentCycleDecisionPolicy,
    reliability_scope: PhysicalForecastLeadReliabilityScope | None = None,
    forecast_rhs_ballast_trend: PhysicalForecastRhsBallastTrend | None = None,
    forecast_trend_route_policy: PhysicalCurrentCycleForecastTrendRoutePolicy | None = None,
) -> PhysicalCurrentCycleDecision:
    """Assemble one current request from fresh, source-bound physical facts.

    This is the narrow integration point for one control block.  It replays
    the same-origin lifecycle and forecast-endpoint responses once, selects
    one request, and exposes the corresponding actual execution rollout.
    It neither writes the selected rollout back into a later cycle nor treats
    its final simulated state as a future measurement.
    """

    if not isinstance(decision_input, FirstIntervalPhysicalDecisionInput):
        raise TypeError("decision_input must be FirstIntervalPhysicalDecisionInput")
    if not isinstance(forecast_admission, PhysicalForecastAdmission):
        raise TypeError("forecast_admission must be PhysicalForecastAdmission")
    if forecast_admission.decision_input is not decision_input:
        raise ValueError("forecast_admission must retain the decision_input")
    if (
        reliability_scope is not None
        and reliability_scope.decision_input is not decision_input
    ):
        raise ValueError("reliability_scope must retain the decision_input")
    if (
        forecast_rhs_ballast_trend is not None
        and forecast_rhs_ballast_trend.decision_input is not decision_input
    ):
        raise ValueError("forecast_rhs_ballast_trend must retain the decision_input")
    if (
        forecast_rhs_ballast_trend is not None
        and forecast_rhs_ballast_trend.reliability_scope is not reliability_scope
    ):
        raise ValueError(
            "forecast_rhs_ballast_trend must use the decision reliability scope"
        )
    if (
        forecast_trend_route_policy is not None
        and forecast_rhs_ballast_trend is None
    ):
        raise ValueError(
            "forecast_trend_route_policy requires a forecast_rhs_ballast_trend"
        )

    comparison = assemble_physical_lifecycle_comparison_for_current_block(
        decision_input=decision_input,
        forecast_admission=forecast_admission,
    )
    return assemble_physical_current_cycle_decision(
        response_set=assemble_forecast_endpoint_held_responses(
            current_block_comparison=comparison,
        ),
        horizon_comparison=assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=comparison,
        ),
        posture_limits=posture_limits,
        reliability_scope=reliability_scope,
        forecast_rhs_ballast_trend=forecast_rhs_ballast_trend,
        forecast_trend_route_policy=forecast_trend_route_policy,
        policy=policy,
    )


__all__ = [
    "PhysicalCurrentCycleCandidate",
    "PhysicalCurrentCycleDecision",
    "PhysicalCurrentCycleDecisionPolicy",
    "PhysicalCurrentCycleForecastTrendRoutePolicy",
    "PhysicalCurrentCycleDecisionReason",
    "PhysicalCurrentCycleRoute",
    "PhysicalCurrentCycleSelectionBasis",
    "assemble_physical_current_cycle_decision",
    "assemble_physical_current_cycle_decision_from_input",
]
