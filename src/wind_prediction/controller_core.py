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
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .action_plan import (
    TargetOperation,
    build_candidate_plan,
    commit_action_plan,
    record_evaluation,
)
from .ballast_allocation import (
    compensation_to_mass_delta_kg,
    mass_delta_to_compensation_vec,
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


class ControlAction(str, Enum):
    """Controller decisions with explicit target-lifecycle meaning."""

    STRENGTHEN = "strengthen"
    NORMAL = "normal"
    REDUCED = "reduced"
    CONTINUE_TARGET = "continue_target"
    RELEASE_TARGET = "release_target"
    REVERSE = "reverse"


_ACTION_ORDER = (
    ControlAction.CONTINUE_TARGET,
    ControlAction.RELEASE_TARGET,
    ControlAction.REDUCED,
    ControlAction.NORMAL,
    ControlAction.STRENGTHEN,
    ControlAction.REVERSE,
)


@dataclass(frozen=True)
class ControlCoreConfig:
    """Parameters that define the compact controller, not the plant model."""

    deadband_deg: tuple[float, float] = (1.0, 0.8)
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


@dataclass(frozen=True)
class ControlObservation:
    """All measured values needed for one controller decision."""

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
    first_execution: ExecutionRolloutStep | None = field(
        default=None,
        compare=False,
        repr=False,
    )


@dataclass(frozen=True)
class ControlDecision:
    action: ControlAction
    target_operation: TargetOperation
    action_vector_deg: tuple[float, float]
    target_masses_kg: tuple[float, float, float]
    score: float
    context: ControlContext
    ranked_candidates: tuple[CandidateEvaluation, ...]
    rejected_candidate_count: int
    first_execution: ExecutionRolloutStep
    action_plan: Mapping[str, Any]
    config: ControlCoreConfig = field(repr=False, compare=False)

    def as_trace(self) -> dict[str, Any]:
        first = self.ranked_candidates[0]
        return {
            "selected_action": self.action.value,
            "target_operation": self.target_operation.value,
            "action_vector_deg": list(self.action_vector_deg),
            "target_masses_kg": list(self.target_masses_kg),
            "score": self.score,
            "candidate_count": len(self.ranked_candidates),
            "rejected_candidate_count": self.rejected_candidate_count,
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


def _signed_deadzone(values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - thresholds, 0.0)


def _wind_effect_deg(uv_ms: np.ndarray, config: ControlCoreConfig) -> np.ndarray:
    vectors = np.asarray(uv_ms, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] == 0:
        raise ValueError("wind vectors must have shape (N, 2)")
    speeds = np.linalg.norm(vectors, axis=1)
    mean_speed = float(np.mean(speeds))
    mean_vector = np.mean(vectors, axis=0)
    direction_speed = float(np.linalg.norm(mean_vector))
    if mean_speed <= _EPS or direction_speed <= _EPS:
        return np.zeros(2, dtype=float)
    magnitude = min(
        (mean_speed / float(config.wind_reference_ms)) ** 2,
        float(config.wind_effect_cap),
    )
    u, v = (float(mean_vector[0]), float(mean_vector[1]))
    wind_direction = math.atan2(-u, -v)
    deadband = np.asarray(config.deadband_deg, dtype=float)
    raw = np.array(
        [
            -deadband[0] * magnitude * math.cos(wind_direction),
            deadband[1] * magnitude * math.sin(wind_direction),
        ],
        dtype=float,
    )
    return float(config.pressure_sign_multiplier) * raw


def _disabled_policy(reason: str, stage_duration_s: float) -> ForecastActionPolicyResult:
    return ForecastActionPolicyResult(
        enabled=False,
        reason=str(reason),
        stage_duration_s=float(stage_duration_s),
    )


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


def build_control_context(
    observation: ControlObservation,
    evidence: ForecastEvidence | None,
    config: ControlCoreConfig,
) -> ControlContext:
    """Fuse measured posture with forecast increments without plant dynamics."""

    posture = np.asarray(observation.posture_deg, dtype=float)
    deadband = np.asarray(config.deadband_deg, dtype=float)
    feedback = _signed_deadzone(posture, deadband)
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

    current_effect = _wind_effect_deg(
        np.asarray([observation.current_wind_uv_ms], dtype=float),
        config,
    )
    stages: list[ControlStage] = []
    for index in range(config.stage_count):
        start = index * points_per_stage
        end = start + points_per_stage
        stage_feedback = feedback * float(config.posture_decay) ** index
        mean_reliability = float(np.mean(reliability[start:end]))
        if forecast_available:
            future_effect = _wind_effect_deg(vectors[start:end], config)
            forecast_increment = (
                (future_effect - current_effect)
                * float(config.forecast_gain)
                * float(config.stage_discounts[index])
                * mean_reliability
            )
        else:
            forecast_increment = np.zeros(2, dtype=float)
        combined = stage_feedback + forecast_increment
        stages.append(
            ControlStage(
                index=index,
                lead_start_s=float(index * config.stage_duration_s),
                lead_end_s=float((index + 1) * config.stage_duration_s),
                feedback_demand_deg=tuple(float(value) for value in stage_feedback),
                forecast_increment_deg=tuple(
                    float(value) for value in forecast_increment
                ),
                combined_demand_deg=tuple(float(value) for value in combined),
                mean_reliability=mean_reliability,
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


def _normalized_direction(demand_deg: np.ndarray, deadband_deg: np.ndarray) -> np.ndarray:
    normalized = np.asarray(demand_deg, dtype=float) / deadband_deg
    magnitude = float(np.linalg.norm(normalized))
    if magnitude <= _EPS:
        return np.zeros(2, dtype=float)
    return normalized / magnitude * deadband_deg


def _action_vector(
    action: ControlAction,
    demand_deg: np.ndarray,
    previous_action_vector_deg: np.ndarray,
    config: ControlCoreConfig,
) -> np.ndarray:
    if action in {ControlAction.CONTINUE_TARGET, ControlAction.RELEASE_TARGET}:
        return np.zeros(2, dtype=float)
    deadband = np.asarray(config.deadband_deg, dtype=float)
    if action is ControlAction.REVERSE:
        reference = np.asarray(previous_action_vector_deg, dtype=float)
        if np.linalg.norm(reference / deadband) <= _EPS:
            return np.zeros(2, dtype=float)
        return -_normalized_direction(reference, deadband) * float(config.reverse_ratio)
    direction = _normalized_direction(demand_deg, deadband)
    ratio = {
        ControlAction.REDUCED: config.reduced_ratio,
        ControlAction.NORMAL: config.normal_ratio,
        ControlAction.STRENGTHEN: config.strengthen_ratio,
    }[action]
    return direction * float(ratio)


def execution_request_for_action(
    action: ControlAction,
    *,
    state: ExecutionRolloutState,
    action_vector_deg: np.ndarray,
    config: ControlCoreConfig,
) -> ExecutionRolloutRequest:
    """Translate one decision into the exact target evaluated by the pumps."""

    if action is ControlAction.CONTINUE_TARGET:
        return ExecutionRolloutRequest.track(state.primary_target_masses_kg)
    if action is ControlAction.RELEASE_TARGET:
        return ExecutionRolloutRequest.release_to_current()
    delta = compensation_to_mass_delta_kg(
        action_vector_deg,
        deadband_deg=config.deadband_deg,
        action_mass_quantum_kg=config.action_mass_quantum_kg,
    )
    target = np.clip(
        state.actual_masses_kg + delta,
        0.0,
        float(config.execution.tank_capacity_kg),
    )
    return ExecutionRolloutRequest.track(target)


def _target_operation(action: ControlAction) -> TargetOperation:
    if action is ControlAction.CONTINUE_TARGET:
        return TargetOperation.CONTINUE
    if action is ControlAction.RELEASE_TARGET:
        return TargetOperation.RELEASE
    return TargetOperation.SET_DELTA


def _candidate_sequences(stage_count: int) -> tuple[tuple[ControlAction, ...], ...]:
    return tuple(itertools.product(_ACTION_ORDER, repeat=stage_count))


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
    deadband = np.asarray(config.deadband_deg, dtype=float)
    residual_cost = 0.0
    transferred_volume = 0.0
    active_time = 0.0
    starts = 0
    switches = 0
    first_vector = np.zeros(2, dtype=float)
    first_target = state.actual_masses_kg.copy()
    first_step: ExecutionRolloutStep | None = None
    final_residual = np.zeros(2, dtype=float)

    for index, (action, stage) in enumerate(zip(sequence, context.stages)):
        demand = np.asarray(stage.combined_demand_deg, dtype=float)
        residual_before = demand - compensation
        residual_demand_ratio = float(np.linalg.norm(residual_before / deadband))
        if (
            action
            in {
                ControlAction.STRENGTHEN,
                ControlAction.NORMAL,
                ControlAction.REDUCED,
                ControlAction.REVERSE,
            }
            and residual_demand_ratio < float(config.minimum_action_demand_ratio)
        ):
            return _rejected_candidate(sequence, "demand_below_action_threshold")
        action_vector = _action_vector(
            action,
            residual_before,
            previous_vector,
            config,
        )
        if action in {
            ControlAction.STRENGTHEN,
            ControlAction.NORMAL,
            ControlAction.REDUCED,
            ControlAction.REVERSE,
        } and np.linalg.norm(action_vector / deadband) <= _EPS:
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
            deadband_deg=config.deadband_deg,
            action_mass_quantum_kg=config.action_mass_quantum_kg,
        )
        compensation = (
            compensation * float(config.compensation_retention)
            + executed_compensation
        )
        final_residual = demand - compensation
        residual_norm = float(np.linalg.norm(final_residual / deadband))
        residual_cost += float(config.stage_discounts[index]) * residual_norm**2
        transferred_volume += float(step.transferred_volume_m3)
        active_time += float(step.active_time_s)
        starts += int(step.starts)
        switches += int(step.direction_switches)

        if index == 0:
            first_vector = action_vector.copy()
            first_target = step.requested_target_kg.copy()
            first_step = step
        if action is ControlAction.RELEASE_TARGET:
            previous_vector = np.zeros(2, dtype=float)
        elif action not in {ControlAction.CONTINUE_TARGET}:
            previous_vector = action_vector.copy()
        state = step.state

    if first_step is None:
        return _rejected_candidate(sequence, "empty_sequence")
    terminal_norm = float(np.linalg.norm(final_residual / deadband))
    reference_volume = max(
        float(config.action_mass_quantum_kg)
        / max(float(config.execution.water_density_kg_m3), _EPS),
        _EPS,
    )
    runtime_reference = max(
        float(config.stage_count) * 3.0 * float(config.stage_duration_s),
        _EPS,
    )
    score = (
        float(config.w_residual) * residual_cost
        + float(config.w_terminal_residual) * terminal_norm**2
        + float(config.w_pump_volume) * transferred_volume / reference_volume
        + float(config.w_pump_runtime) * active_time / runtime_reference
        + float(config.w_starts) * starts / max(3.0 * config.stage_count, 1.0)
        + float(config.w_direction_switches)
        * switches
        / max(3.0 * config.stage_count, 1.0)
    )
    return CandidateEvaluation(
        sequence=sequence,
        feasible=True,
        reject_reason="",
        score=float(score),
        residual_cost=float(residual_cost),
        terminal_residual_cost=float(terminal_norm**2),
        transferred_volume_m3=float(transferred_volume),
        active_time_s=float(active_time),
        starts=int(starts),
        direction_switches=int(switches),
        first_action_vector_deg=tuple(float(value) for value in first_vector),
        first_target_masses_kg=tuple(float(value) for value in first_target),
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
        deadband_deg=config.deadband_deg,
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
            tuple(_ACTION_ORDER.index(action) for action in candidate.sequence),
        ),
    )
    if not feasible:
        reasons: dict[str, int] = {}
        for candidate in evaluated:
            reasons[candidate.reject_reason] = reasons.get(candidate.reject_reason, 0) + 1
        raise RuntimeError(f"no feasible controller candidate: {reasons}")

    best = feasible[0]
    action = best.sequence[0]
    operation = _target_operation(action)
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
    return ControlDecision(
        action=action,
        target_operation=operation,
        action_vector_deg=best.first_action_vector_deg,
        target_masses_kg=best.first_target_masses_kg,
        score=best.score,
        context=context,
        ranked_candidates=tuple(feasible),
        rejected_candidate_count=len(evaluated) - len(feasible),
        first_execution=best.first_execution,
        action_plan=MappingProxyType(snapshot),
        config=config,
    )


__all__ = [
    "CandidateEvaluation",
    "ControlAction",
    "ControlContext",
    "ControlCoreConfig",
    "ControlDecision",
    "ControlObservation",
    "ControlStage",
    "build_control_context",
    "decide_control_cycle",
    "execution_request_for_action",
]
