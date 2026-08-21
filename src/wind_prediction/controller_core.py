"""Simulator-independent forecast-assisted ballast control cycle.

The core consumes measured posture, current actuator state, and validated
forecast evidence. It returns one evaluated target transaction. Rigid-body
dynamics and platform-state propagation deliberately remain outside this
module so controller logic can be tested before the plant model is calibrated.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .action_plan import (
    TargetOperation,
    build_candidate_plan,
    commit_action_plan,
    record_evaluation,
)
from .action_target import (
    ACTION_ORDER,
    ActionSemantics,
    ActionVectorSource,
    ControlAction,
    action_semantics_for,
    action_vector_for,
    execution_request_for_action as _execution_request_for_action,
)
from .ballast_allocation import mass_delta_to_compensation_vec
from .decision_demand import (
    LegacyWindDemandProxyConfig,
    form_stage_demand,
    posture_feedback_demand_deg,
)
from .execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionRolloutStep,
    simulate_execution_step,
)
from .forecast_action_policy import (
    ActionAuthorization,
    ForecastActionPolicyConfig,
    ForecastActionPolicyResult,
    evaluate_forecast_action_policy,
)
from .forecast_evidence import ForecastEvidence, validate_forecast_evidence


_EPS = 1e-12


@dataclass(frozen=True)
class ControlCoreConfig:
    """Parameters that define the compact controller, not the plant model."""

    deadband_deg: tuple[float, float] = (1.0, 0.8)
    legacy_demand_axis_scale_deg: tuple[float, float] | None = None
    posture_priority_envelope_deg: tuple[float, float] = (4.0, 4.0)
    stage_duration_s: float = 1200.0
    stage_count: int = 3
    stage_discounts: tuple[float, ...] = (1.0, 0.85, 0.70)
    posture_decay: float = 0.85
    forecast_gain: float = 1.0
    wind_reference_ms: float = 12.0
    wind_effect_cap: float = 1.5
    pressure_sign_multiplier: float = -1.0
    action_mass_quantum_kg: float = 180_000.0
    reduced_ratio: float = 0.08
    normal_ratio: float = 0.15
    strengthen_ratio: float = 0.35
    reverse_ratio: float = 0.12
    minimum_action_demand_ratio: float = 0.20
    compensation_retention: float = 0.90
    w_residual: float = 10.0
    w_terminal_residual: float = 20.0
    w_pump_volume: float = 1.0
    w_pump_runtime: float = 0.5
    w_starts: float = 0.2
    w_direction_switches: float = 0.3
    execution: ExecutionRolloutConfig = field(default_factory=ExecutionRolloutConfig)
    forecast_policy: ForecastActionPolicyConfig = field(
        default_factory=lambda: ForecastActionPolicyConfig(enabled=True)
    )

    def __post_init__(self) -> None:
        deadband = _finite_pair("deadband_deg", self.deadband_deg)
        envelope = _finite_pair(
            "posture_priority_envelope_deg",
            self.posture_priority_envelope_deg,
        )
        if min(deadband) <= 0.0:
            raise ValueError("deadband_deg values must be positive")
        demand_scale = (
            deadband
            if self.legacy_demand_axis_scale_deg is None
            else _finite_pair(
                "legacy_demand_axis_scale_deg",
                self.legacy_demand_axis_scale_deg,
            )
        )
        if min(demand_scale) <= 0.0:
            raise ValueError("legacy_demand_axis_scale_deg values must be positive")
        if min(envelope) <= 0.0:
            raise ValueError("posture_priority_envelope_deg values must be positive")
        if any(deadband_value >= envelope_value for deadband_value, envelope_value in zip(deadband, envelope)):
            raise ValueError(
                "deadband_deg must remain below posture_priority_envelope_deg"
            )
        if not isinstance(self.stage_count, int) or self.stage_count <= 0:
            raise ValueError("stage_count must be a positive integer")
        if self.stage_count > 6:
            raise ValueError("stage_count must not exceed 6")
        if len(self.stage_discounts) != self.stage_count:
            raise ValueError("stage_discounts must contain one value per stage")
        discounts = _finite_tuple("stage_discounts", self.stage_discounts)
        if any(value < 0.0 or value > 1.0 for value in discounts):
            raise ValueError("stage_discounts values must lie in [0, 1]")
        if any(later > earlier for earlier, later in zip(discounts, discounts[1:])):
            raise ValueError("stage_discounts must be non-increasing")
        positive = {
            "stage_duration_s": self.stage_duration_s,
            "wind_reference_ms": self.wind_reference_ms,
            "action_mass_quantum_kg": self.action_mass_quantum_kg,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        bounded = {
            "posture_decay": self.posture_decay,
            "compensation_retention": self.compensation_retention,
        }
        for name, value in bounded.items():
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        non_negative = {
            "forecast_gain": self.forecast_gain,
            "wind_effect_cap": self.wind_effect_cap,
            "reduced_ratio": self.reduced_ratio,
            "normal_ratio": self.normal_ratio,
            "strengthen_ratio": self.strengthen_ratio,
            "reverse_ratio": self.reverse_ratio,
            "minimum_action_demand_ratio": self.minimum_action_demand_ratio,
            "w_residual": self.w_residual,
            "w_terminal_residual": self.w_terminal_residual,
            "w_pump_volume": self.w_pump_volume,
            "w_pump_runtime": self.w_pump_runtime,
            "w_starts": self.w_starts,
            "w_direction_switches": self.w_direction_switches,
        }
        for name, value in non_negative.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")
        if not self.reduced_ratio <= self.normal_ratio <= self.strengthen_ratio:
            raise ValueError(
                "action ratios must satisfy reduced <= normal <= strengthen"
            )
        if not 0.0 <= float(self.minimum_action_demand_ratio) <= 1.0:
            raise ValueError("minimum_action_demand_ratio must lie in [0, 1]")
        if float(self.pressure_sign_multiplier) not in {-1.0, 1.0}:
            raise ValueError("pressure_sign_multiplier must be -1 or 1")
        if not math.isclose(
            float(self.stage_duration_s),
            float(self.execution.block_duration_s),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "stage_duration_s must equal execution.block_duration_s so "
                "candidate preview and the next decision use the same interval"
            )
        if self.forecast_policy.enabled and not math.isclose(
            float(self.stage_duration_s),
            float(self.forecast_policy.stage_duration_s),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "forecast_policy.stage_duration_s must equal stage_duration_s"
            )
        if self.forecast_policy.enabled and len(
            self.forecast_policy.stage_event_keys
        ) != self.stage_count:
            raise ValueError(
                "forecast_policy.stage_event_keys must contain one key per stage"
            )
        if not any(
            value > 0.0
            for value in (
                self.w_residual,
                self.w_terminal_residual,
                self.w_pump_volume,
                self.w_pump_runtime,
                self.w_starts,
                self.w_direction_switches,
            )
        ):
            raise ValueError("at least one candidate cost weight must be positive")

    @property
    def resolved_legacy_demand_axis_scale_deg(self) -> tuple[float, float]:
        """Return the legacy demand scale, applying the compatibility fallback."""

        values = (
            self.deadband_deg
            if self.legacy_demand_axis_scale_deg is None
            else self.legacy_demand_axis_scale_deg
        )
        return tuple(float(value) for value in values)


@dataclass(frozen=True)
class ControlObservation:
    """All measured values needed for one controller decision.

    ``current_wind_uv_ms`` is the downwind air-velocity vector in geographic
    ENU axes, ordered ``[east, north]`` and expressed in m/s. It is not
    resolved into platform forward/port axes here. A future physical-demand
    mapping must perform that conversion explicitly with its chosen heading.
    """

    time_s: float
    posture_deg: tuple[float, float]
    posture_rate_deg_s: tuple[float, float]
    current_wind_uv_ms: tuple[float, float]
    execution_state: ExecutionRolloutState

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.time_s)):
            raise ValueError("time_s must be finite")
        object.__setattr__(self, "posture_deg", _finite_pair("posture_deg", self.posture_deg))
        object.__setattr__(
            self,
            "posture_rate_deg_s",
            _finite_pair("posture_rate_deg_s", self.posture_rate_deg_s),
        )
        object.__setattr__(
            self,
            "current_wind_uv_ms",
            _finite_pair("current_wind_uv_ms", self.current_wind_uv_ms),
        )


@dataclass(frozen=True)
class ControlStage:
    index: int
    lead_start_s: float
    lead_end_s: float
    feedback_demand_deg: tuple[float, float]
    forecast_increment_deg: tuple[float, float]
    combined_demand_deg: tuple[float, float]
    mean_reliability: float
    forecast_demand_reason: str
    forecast_direction_consistency: float | None
    forecast_direction_consistent: bool | None
    authorizations: Mapping[ControlAction, ActionAuthorization]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorizations",
            MappingProxyType(dict(self.authorizations)),
        )


@dataclass(frozen=True)
class ControlContext:
    feedback_demand_deg: tuple[float, float]
    forecast_available: bool
    forecast_source: str
    forecast_policy: ForecastActionPolicyResult
    posture_priority: bool
    stages: tuple[ControlStage, ...]


@dataclass(frozen=True)
class CandidateEvaluation:
    sequence: tuple[ControlAction, ...]
    feasible: bool
    reject_reason: str
    score: float
    residual_cost: float
    terminal_residual_cost: float
    transferred_volume_m3: float
    active_time_s: float
    starts: int
    direction_switches: int
    first_action_vector_deg: tuple[float, float]
    first_target_masses_kg: tuple[float, float, float]
    first_execution_request: ExecutionRolloutRequest | None = field(
        default=None,
        compare=False,
        repr=False,
    )
    first_execution: ExecutionRolloutStep | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class CandidateScoreTerms:
    """Weighted terms that form one feasible candidate's ranking score."""

    residual: float
    terminal_residual: float
    pump_volume: float
    pump_runtime: float
    starts: float
    direction_switches: float

    @property
    def total(self) -> float:
        return float(
            self.residual
            + self.terminal_residual
            + self.pump_volume
            + self.pump_runtime
            + self.starts
            + self.direction_switches
        )

    def as_trace(self) -> dict[str, float]:
        return {
            "residual": self.residual,
            "terminal_residual": self.terminal_residual,
            "pump_volume": self.pump_volume,
            "pump_runtime": self.pump_runtime,
            "starts": self.starts,
            "direction_switches": self.direction_switches,
            "total": self.total,
        }


@dataclass(frozen=True)
class ControlDecision:
    action: ControlAction
    target_operation: TargetOperation
    action_vector_deg: tuple[float, float]
    target_masses_kg: tuple[float, float, float]
    execution_request: ExecutionRolloutRequest = field(repr=False, compare=False)
    score: float
    context: ControlContext
    ranked_candidates: tuple[CandidateEvaluation, ...]
    rejected_candidate_count: int
    first_execution: ExecutionRolloutStep
    action_plan: Mapping[str, Any]
    config: ControlCoreConfig = field(repr=False, compare=False)

    def as_trace(self) -> dict[str, Any]:
        first = self.ranked_candidates[0]
        first_stage = self.context.stages[0]
        selected_authorization = first_stage.authorizations[self.action]
        selected_semantics = action_semantics_for(self.action)
        selected_score_terms = _candidate_score_terms(
            residual_cost=first.residual_cost,
            terminal_residual_cost=first.terminal_residual_cost,
            transferred_volume_m3=first.transferred_volume_m3,
            active_time_s=first.active_time_s,
            starts=first.starts,
            direction_switches=first.direction_switches,
            config=self.config,
        )
        return {
            "selected_action": self.action.value,
            "target_operation": self.target_operation.value,
            "action_vector_deg": list(self.action_vector_deg),
            "target_masses_kg": list(self.target_masses_kg),
            "execution_request": {
                "operation": self.execution_request.operation.value,
                "target_masses_kg": (
                    None
                    if self.execution_request.target_masses_kg is None
                    else list(self.execution_request.target_masses_kg)
                ),
            },
            "score": self.score,
            "candidate_count": len(self.ranked_candidates),
            "rejected_candidate_count": self.rejected_candidate_count,
            "selection": {
                "selected_sequence": [
                    action.value for action in first.sequence
                ],
                "feasible_sequence_count": len(self.ranked_candidates),
                "selected_action_authorization": {
                    "allowed": selected_authorization.allowed,
                    "reason": selected_authorization.reason,
                },
                "selected_action_semantics": {
                    "target_operation": selected_semantics.target_operation.value,
                    "vector_source": selected_semantics.vector_source.value,
                    "requires_nonzero_demand": (
                        selected_semantics.requires_nonzero_demand
                    ),
                },
                "score_terms": selected_score_terms.as_trace(),
                "score_weights": {
                    "residual": self.config.w_residual,
                    "terminal_residual": self.config.w_terminal_residual,
                    "pump_volume": self.config.w_pump_volume,
                    "pump_runtime": self.config.w_pump_runtime,
                    "starts": self.config.w_starts,
                    "direction_switches": self.config.w_direction_switches,
                },
            },
            "forecast": {
                "available": self.context.forecast_available,
                "source": self.context.forecast_source,
                "policy_reason": self.context.forecast_policy.reason,
            },
            "posture": {
                "feedback_demand_deg": list(self.context.feedback_demand_deg),
                "priority_active": self.context.posture_priority,
            },
            "stages": [
                {
                    "index": stage.index,
                    "lead_start_s": stage.lead_start_s,
                    "lead_end_s": stage.lead_end_s,
                    "feedback_demand_deg": list(stage.feedback_demand_deg),
                    "forecast_increment_deg": list(stage.forecast_increment_deg),
                    "combined_demand_deg": list(stage.combined_demand_deg),
                    "mean_reliability": stage.mean_reliability,
                    "forecast_demand_reason": stage.forecast_demand_reason,
                    "forecast_direction_consistency": (
                        stage.forecast_direction_consistency
                    ),
                    "forecast_direction_consistent": (
                        stage.forecast_direction_consistent
                    ),
                    "forecast_policy_evidence": _forecast_stage_evidence_trace(
                        self.context.forecast_policy,
                        stage.index,
                        forecast_available=self.context.forecast_available,
                    ),
                    "action_authorizations": {
                        action.value: {
                            "allowed": authorization.allowed,
                            "reason": authorization.reason,
                        }
                        for action, authorization in stage.authorizations.items()
                    },
                }
                for stage in self.context.stages
            ],
            "cost": {
                "residual": first.residual_cost,
                "terminal_residual": first.terminal_residual_cost,
                "pump_volume_m3": first.transferred_volume_m3,
                "active_time_s": first.active_time_s,
                "starts": first.starts,
                "direction_switches": first.direction_switches,
            },
            "candidate_ranking_top": [
                {
                    "rank": rank,
                    "sequence": [action.value for action in candidate.sequence],
                    "score": candidate.score,
                    "score_terms": _candidate_score_terms(
                        residual_cost=candidate.residual_cost,
                        terminal_residual_cost=candidate.terminal_residual_cost,
                        transferred_volume_m3=candidate.transferred_volume_m3,
                        active_time_s=candidate.active_time_s,
                        starts=candidate.starts,
                        direction_switches=candidate.direction_switches,
                        config=self.config,
                    ).as_trace(),
                    "residual_cost": candidate.residual_cost,
                    "terminal_residual_cost": candidate.terminal_residual_cost,
                    "pump_volume_m3": candidate.transferred_volume_m3,
                    "active_time_s": candidate.active_time_s,
                    "starts": candidate.starts,
                    "direction_switches": candidate.direction_switches,
                    "first_action_vector_deg": list(
                        candidate.first_action_vector_deg
                    ),
                    "first_target_masses_kg": list(
                        candidate.first_target_masses_kg
                    ),
                }
                for rank, candidate in enumerate(
                    self.ranked_candidates[:10],
                    start=1,
                )
            ],
            "first_execution": {
                "requested_target_kg": self.first_execution.requested_target_kg.tolist(),
                "mass_delta_kg": self.first_execution.mass_delta_kg.tolist(),
                "transferred_volume_m3": self.first_execution.transferred_volume_m3,
                "active_time_s": self.first_execution.active_time_s,
                "starts": self.first_execution.starts,
                "stops": self.first_execution.stops,
                "direction_switches": self.first_execution.direction_switches,
                "target_reached": self.first_execution.target_reached,
            },
            "action_plan": dict(self.action_plan),
        }


def _finite_pair(name: str, values: Any) -> tuple[float, float]:
    result = _finite_tuple(name, values)
    if len(result) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return float(result[0]), float(result[1])


def _finite_tuple(name: str, values: Any) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    return result


def _disabled_policy(reason: str, stage_duration_s: float) -> ForecastActionPolicyResult:
    return ForecastActionPolicyResult(
        enabled=False,
        reason=str(reason),
        stage_duration_s=float(stage_duration_s),
    )


def _forecast_stage_evidence_trace(
    policy: ForecastActionPolicyResult,
    stage_index: int,
    *,
    forecast_available: bool,
) -> dict[str, Any] | None:
    """Project policy evidence without hiding a usable forecast input."""

    if not forecast_available:
        return None
    if not policy.enabled:
        return {
            "available": True,
            "policy_evaluated": False,
            "reason": policy.reason,
        }
    if stage_index >= len(policy.stages):
        return {
            "available": True,
            "policy_evaluated": False,
            "reason": "forecast_policy_stage_unavailable",
        }
    stage = policy.stages[stage_index]
    return {
        "available": True,
        "policy_evaluated": True,
        "reason": policy.reason,
        "lead_start_s": stage.lead_start_s,
        "lead_end_s": stage.lead_end_s,
        "mean_speed_ms": stage.mean_speed_ms,
        "speed_change_ms": stage.speed_change_ms,
        "relative_speed_change": stage.relative_speed_change,
        "direction_consistency": stage.direction_consistency,
        "direction_consistent": stage.direction_consistent,
        "mean_reliability": stage.mean_reliability,
        "event_probability": stage.event_probability,
        "event_threshold": stage.event_threshold,
        "event_threshold_source": stage.event_threshold_source,
        "trend": {
            "strengthening": stage.strengthening,
            "sustained": stage.sustained,
            "declining": stage.declining,
            "continuous_reversal": stage.continuous_reversal,
            "reversal_run_length": stage.reversal_run_length,
            "high_impact_event_supported": stage.high_impact_event_supported,
        },
    }


def _stage_authorizations(
    policy: ForecastActionPolicyResult,
    stage_index: int,
    *,
    posture_priority: bool,
) -> Mapping[ControlAction, ActionAuthorization]:
    available = policy.enabled and stage_index < len(policy.stages)
    policy_stage = policy.stages[stage_index] if available else None
    authorizations: dict[ControlAction, ActionAuthorization] = {
        ControlAction.NORMAL: ActionAuthorization(True, "feedback_action_available"),
        ControlAction.REDUCED: ActionAuthorization(True, "feedback_action_available"),
        ControlAction.CONTINUE_TARGET: ActionAuthorization(True, "target_can_continue"),
        ControlAction.STRENGTHEN: ActionAuthorization(
            bool(policy_stage and policy_stage.strengthen.allowed),
            (
                policy_stage.strengthen.reason
                if policy_stage is not None
                else "future_evidence_unavailable"
            ),
        ),
        ControlAction.RELEASE_TARGET: ActionAuthorization(
            bool(policy_stage and policy_stage.release.allowed),
            (
                policy_stage.release.reason
                if policy_stage is not None
                else "future_relief_unavailable"
            ),
        ),
        ControlAction.REVERSE: ActionAuthorization(
            bool(policy_stage and policy_stage.reverse.allowed),
            (
                policy_stage.reverse.reason
                if policy_stage is not None
                else "future_reversal_unavailable"
            ),
        ),
    }
    if posture_priority:
        authorizations[ControlAction.STRENGTHEN] = ActionAuthorization(
            True,
            "measured_posture_priority",
        )
        for action in (
            ControlAction.REDUCED,
            ControlAction.CONTINUE_TARGET,
            ControlAction.RELEASE_TARGET,
            ControlAction.REVERSE,
        ):
            authorizations[action] = ActionAuthorization(
                False,
                "measured_posture_priority_requires_correction",
            )
    return authorizations


def _forecast_demand_metadata(
    policy: ForecastActionPolicyResult,
    stage_index: int,
    *,
    forecast_available: bool,
) -> tuple[str, float | None, bool | None]:
    if not forecast_available:
        return "forecast_unavailable", None, None
    if not policy.enabled:
        return "forecast_policy_disabled", None, None
    if stage_index >= len(policy.stages):
        return "forecast_policy_stage_unavailable", None, None
    stage = policy.stages[stage_index]
    if stage.direction_consistent:
        reason = "forecast_direction_consistent"
    else:
        reason = "forecast_direction_inconsistent_recorded"
    return reason, stage.direction_consistency, stage.direction_consistent


def build_control_context(
    observation: ControlObservation,
    evidence: ForecastEvidence | None,
    config: ControlCoreConfig,
) -> ControlContext:
    """Fuse measured posture with forecast increments without plant dynamics."""

    deadband = np.asarray(config.deadband_deg, dtype=float)
    posture = np.asarray(observation.posture_deg, dtype=float)
    feedback = posture_feedback_demand_deg(posture, deadband_deg=deadband)
    priority_envelope = np.asarray(config.posture_priority_envelope_deg, dtype=float)
    posture_priority = bool(np.any(np.abs(posture) >= priority_envelope))

    forecast_available = bool(
        evidence is not None and evidence.provides_future_preview
    )
    if evidence is None:
        vectors = np.tile(
            np.asarray(observation.current_wind_uv_ms, dtype=float),
            (config.stage_count, 1),
        )
        reliability = np.ones(config.stage_count, dtype=float)
        sample_period_s = float(config.stage_duration_s)
        source = "none"
        policy = _disabled_policy("forecast_unavailable", config.stage_duration_s)
    else:
        validate_forecast_evidence(evidence)
        vectors = np.asarray(evidence.uv_ms, dtype=float)
        reliability = np.asarray(evidence.lead_reliability, dtype=float)
        sample_period_s = float(evidence.sample_period_s)
        source = str(evidence.source)
        if forecast_available:
            policy = evaluate_forecast_action_policy(
                evidence,
                current_uv_ms=observation.current_wind_uv_ms,
                config=config.forecast_policy,
            )
        else:
            policy = _disabled_policy("no_future_preview", config.stage_duration_s)

    points_per_stage_float = float(config.stage_duration_s) / sample_period_s
    points_per_stage = int(round(points_per_stage_float))
    if points_per_stage <= 0 or not math.isclose(
        points_per_stage_float,
        points_per_stage,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("stage_duration_s must be an integer multiple of forecast sample period")
    required_points = points_per_stage * config.stage_count
    if vectors.shape[0] < required_points:
        if evidence is not None:
            raise ValueError("forecast evidence does not cover the configured control horizon")
        vectors = np.tile(vectors[-1], (required_points, 1))
        reliability = np.ones(required_points, dtype=float)

    proxy_config = LegacyWindDemandProxyConfig(
        demand_axis_scale_deg=config.resolved_legacy_demand_axis_scale_deg,
        reference_speed_ms=config.wind_reference_ms,
        effect_cap=config.wind_effect_cap,
        sign_multiplier=config.pressure_sign_multiplier,
    )
    stages: list[ControlStage] = []
    for index in range(config.stage_count):
        start = index * points_per_stage
        end = start + points_per_stage
        if forecast_available:
            # Forecast points are timestamped after the origin.  Keep the
            # trace aligned with those point leads rather than the policy's
            # enclosing planning interval.
            stage_lead_start_s = float((start + 1) * sample_period_s)
            stage_lead_end_s = float(end * sample_period_s)
        else:
            stage_lead_start_s = float(index * config.stage_duration_s)
            stage_lead_end_s = float((index + 1) * config.stage_duration_s)
        stage_feedback = feedback * float(config.posture_decay) ** index
        mean_reliability = float(np.mean(reliability[start:end]))
        reason, direction_consistency, direction_consistent = _forecast_demand_metadata(
            policy,
            index,
            forecast_available=forecast_available,
        )
        stage_demand = form_stage_demand(
            feedback_demand_deg=stage_feedback,
            current_wind_uv_ms=observation.current_wind_uv_ms,
            future_wind_uv_ms=vectors[start:end] if forecast_available else None,
            forecast_available=forecast_available,
            stage_discount=float(config.forecast_gain)
            * float(config.stage_discounts[index]),
            proxy_config=proxy_config,
            forecast_reason=reason,
            direction_consistency=direction_consistency,
            direction_consistent=direction_consistent,
        )
        stages.append(
            ControlStage(
                index=index,
                lead_start_s=stage_lead_start_s,
                lead_end_s=stage_lead_end_s,
                feedback_demand_deg=stage_demand.feedback_demand_deg,
                forecast_increment_deg=stage_demand.forecast_increment_deg,
                combined_demand_deg=stage_demand.combined_demand_deg,
                mean_reliability=mean_reliability,
                forecast_demand_reason=stage_demand.forecast_reason,
                forecast_direction_consistency=(
                    stage_demand.direction_consistency
                ),
                forecast_direction_consistent=stage_demand.direction_consistent,
                authorizations=_stage_authorizations(
                    policy,
                    index,
                    posture_priority=bool(posture_priority and index == 0),
                ),
            )
        )
    return ControlContext(
        feedback_demand_deg=tuple(float(value) for value in feedback),
        forecast_available=forecast_available,
        forecast_source=source,
        forecast_policy=policy,
        posture_priority=posture_priority,
        stages=tuple(stages),
    )


def _action_vector(
    action: ControlAction,
    demand_deg: np.ndarray,
    previous_action_vector_deg: np.ndarray,
    config: ControlCoreConfig,
) -> np.ndarray:
    """Compatibility wrapper for the extracted action-target calculation."""

    return action_vector_for(
        action,
        demand_deg=demand_deg,
        previous_action_vector_deg=previous_action_vector_deg,
        demand_axis_scale_deg=config.resolved_legacy_demand_axis_scale_deg,
        reduced_ratio=config.reduced_ratio,
        normal_ratio=config.normal_ratio,
        strengthen_ratio=config.strengthen_ratio,
        reverse_ratio=config.reverse_ratio,
    )


def execution_request_for_action(
    action: ControlAction,
    *,
    state: ExecutionRolloutState,
    action_vector_deg: np.ndarray,
    config: ControlCoreConfig,
) -> ExecutionRolloutRequest:
    """Compatibility wrapper for the extracted execution-target request."""

    return _execution_request_for_action(
        action,
        state=state,
        action_vector_deg=action_vector_deg,
        demand_axis_scale_deg=config.resolved_legacy_demand_axis_scale_deg,
        action_mass_quantum_kg=config.action_mass_quantum_kg,
        tank_capacity_kg=config.execution.tank_capacity_kg,
    )


def _candidate_sequences(stage_count: int) -> tuple[tuple[ControlAction, ...], ...]:
    return tuple(itertools.product(ACTION_ORDER, repeat=stage_count))


def _candidate_score_terms(
    *,
    residual_cost: float,
    terminal_residual_cost: float,
    transferred_volume_m3: float,
    active_time_s: float,
    starts: int,
    direction_switches: int,
    config: ControlCoreConfig,
) -> CandidateScoreTerms:
    """Evaluate the existing score formula as individually traceable terms."""

    reference_volume = max(
        float(config.action_mass_quantum_kg)
        / max(float(config.execution.water_density_kg_m3), _EPS),
        _EPS,
    )
    runtime_reference = max(
        float(config.stage_count) * 3.0 * float(config.stage_duration_s),
        _EPS,
    )
    action_reference = max(3.0 * config.stage_count, 1.0)
    return CandidateScoreTerms(
        residual=float(config.w_residual) * float(residual_cost),
        terminal_residual=(
            float(config.w_terminal_residual) * float(terminal_residual_cost)
        ),
        pump_volume=(
            float(config.w_pump_volume)
            * float(transferred_volume_m3)
            / reference_volume
        ),
        pump_runtime=(
            float(config.w_pump_runtime) * float(active_time_s) / runtime_reference
        ),
        starts=float(config.w_starts) * int(starts) / action_reference,
        direction_switches=(
            float(config.w_direction_switches)
            * int(direction_switches)
            / action_reference
        ),
    )


def _evaluate_candidate(
    sequence: tuple[ControlAction, ...],
    observation: ControlObservation,
    context: ControlContext,
    config: ControlCoreConfig,
    initial_previous_vector_deg: np.ndarray,
) -> CandidateEvaluation:
    for action, stage in zip(sequence, context.stages):
        authorization = stage.authorizations[action]
        if not authorization.allowed:
            return _rejected_candidate(sequence, authorization.reason)

    state = observation.execution_state
    compensation = np.zeros(2, dtype=float)
    previous_vector = np.asarray(initial_previous_vector_deg, dtype=float).copy()
    demand_scale = np.asarray(
        config.resolved_legacy_demand_axis_scale_deg,
        dtype=float,
    )
    residual_cost = 0.0
    transferred_volume = 0.0
    active_time = 0.0
    starts = 0
    switches = 0
    first_vector = np.zeros(2, dtype=float)
    first_target = state.actual_masses_kg.copy()
    first_request: ExecutionRolloutRequest | None = None
    first_step: ExecutionRolloutStep | None = None
    final_residual = np.zeros(2, dtype=float)

    for index, (action, stage) in enumerate(zip(sequence, context.stages)):
        semantics = action_semantics_for(action)
        demand = np.asarray(stage.combined_demand_deg, dtype=float)
        residual_before = demand - compensation
        residual_demand_ratio = float(
            np.linalg.norm(residual_before / demand_scale)
        )
        if semantics.requires_nonzero_demand and residual_demand_ratio < float(
            config.minimum_action_demand_ratio
        ):
            return _rejected_candidate(sequence, "demand_below_action_threshold")
        action_vector = _action_vector(
            action,
            residual_before,
            previous_vector,
            config,
        )
        if (
            semantics.requires_nonzero_demand
            and np.linalg.norm(action_vector / demand_scale) <= _EPS
        ):
            return _rejected_candidate(sequence, "active_action_has_zero_effect")

        request = execution_request_for_action(
            action,
            state=state,
            action_vector_deg=action_vector,
            config=config,
        )
        step = simulate_execution_step(state, request, config.execution)
        executed_compensation = mass_delta_to_compensation_vec(
            step.mass_delta_kg,
            deadband_deg=config.resolved_legacy_demand_axis_scale_deg,
            action_mass_quantum_kg=config.action_mass_quantum_kg,
        )
        compensation = (
            compensation * float(config.compensation_retention)
            + executed_compensation
        )
        final_residual = demand - compensation
        residual_norm = float(np.linalg.norm(final_residual / demand_scale))
        residual_cost += float(config.stage_discounts[index]) * residual_norm**2
        transferred_volume += float(step.transferred_volume_m3)
        active_time += float(step.active_time_s)
        starts += int(step.starts)
        switches += int(step.direction_switches)

        if index == 0:
            first_vector = action_vector.copy()
            first_target = step.requested_target_kg.copy()
            first_request = request
            first_step = step
        if semantics.target_operation is TargetOperation.RELEASE:
            previous_vector = np.zeros(2, dtype=float)
        elif semantics.vector_source is not ActionVectorSource.NONE:
            previous_vector = action_vector.copy()
        state = step.state

    if first_step is None or first_request is None:
        return _rejected_candidate(sequence, "empty_sequence")
    terminal_norm = float(np.linalg.norm(final_residual / demand_scale))
    score_terms = _candidate_score_terms(
        residual_cost=residual_cost,
        terminal_residual_cost=terminal_norm**2,
        transferred_volume_m3=transferred_volume,
        active_time_s=active_time,
        starts=starts,
        direction_switches=switches,
        config=config,
    )
    return CandidateEvaluation(
        sequence=sequence,
        feasible=True,
        reject_reason="",
        score=score_terms.total,
        residual_cost=float(residual_cost),
        terminal_residual_cost=float(terminal_norm**2),
        transferred_volume_m3=float(transferred_volume),
        active_time_s=float(active_time),
        starts=int(starts),
        direction_switches=int(switches),
        first_action_vector_deg=tuple(float(value) for value in first_vector),
        first_target_masses_kg=tuple(float(value) for value in first_target),
        first_execution_request=first_request,
        first_execution=first_step,
    )


def _rejected_candidate(
    sequence: tuple[ControlAction, ...],
    reason: str,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        sequence=sequence,
        feasible=False,
        reject_reason=str(reason),
        score=float("inf"),
        residual_cost=float("inf"),
        terminal_residual_cost=float("inf"),
        transferred_volume_m3=0.0,
        active_time_s=0.0,
        starts=0,
        direction_switches=0,
        first_action_vector_deg=(0.0, 0.0),
        first_target_masses_kg=(0.0, 0.0, 0.0),
    )


def decide_control_cycle(
    observation: ControlObservation,
    evidence: ForecastEvidence | None,
    config: ControlCoreConfig = ControlCoreConfig(),
) -> ControlDecision:
    """Evaluate all authorized sequences and commit the best first action."""

    context = build_control_context(observation, evidence, config)
    initial_previous = mass_delta_to_compensation_vec(
        observation.execution_state.primary_target_masses_kg
        - observation.execution_state.actual_masses_kg,
        deadband_deg=config.resolved_legacy_demand_axis_scale_deg,
        action_mass_quantum_kg=config.action_mass_quantum_kg,
    )
    evaluated = tuple(
        _evaluate_candidate(
            sequence,
            observation,
            context,
            config,
            initial_previous,
        )
        for sequence in _candidate_sequences(config.stage_count)
    )
    feasible = sorted(
        (candidate for candidate in evaluated if candidate.feasible),
        key=lambda candidate: (
            candidate.score,
            tuple(ACTION_ORDER.index(action) for action in candidate.sequence),
        ),
    )
    if not feasible:
        reasons: dict[str, int] = {}
        for candidate in evaluated:
            reasons[candidate.reject_reason] = reasons.get(candidate.reject_reason, 0) + 1
        raise RuntimeError(f"no feasible controller candidate: {reasons}")

    best = feasible[0]
    action = best.sequence[0]
    operation = action_semantics_for(action).target_operation
    plan = build_candidate_plan(
        action.value,
        operation,
        action_vector=best.first_action_vector_deg,
        target_masses_kg=best.first_target_masses_kg,
    )
    plan = record_evaluation(
        plan,
        action.value,
        safety_passed=True,
        action_vector=best.first_action_vector_deg,
        target_masses_kg=best.first_target_masses_kg,
        score=best.score,
        summary="simulator-independent candidate and execution evaluation",
    )
    snapshot = commit_action_plan(plan)
    assert best.first_execution is not None
    assert best.first_execution_request is not None
    return ControlDecision(
        action=action,
        target_operation=operation,
        action_vector_deg=best.first_action_vector_deg,
        target_masses_kg=best.first_target_masses_kg,
        execution_request=best.first_execution_request,
        score=best.score,
        context=context,
        ranked_candidates=tuple(feasible),
        rejected_candidate_count=len(evaluated) - len(feasible),
        first_execution=best.first_execution,
        action_plan=MappingProxyType(snapshot),
        config=config,
    )


__all__ = [
    "ActionSemantics",
    "ActionVectorSource",
    "CandidateEvaluation",
    "CandidateScoreTerms",
    "ControlAction",
    "ControlContext",
    "ControlCoreConfig",
    "ControlDecision",
    "ControlObservation",
    "ControlStage",
    "build_control_context",
    "decide_control_cycle",
    "execution_request_for_action",
    "action_semantics_for",
]
