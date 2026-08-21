"""Independent forecast-to-action authorization for the v2 controller.

This module interprets forecast evidence but does not select an action or alter
planner state.  In particular, lead reliability is reserved for authorizing
high-impact actions; it never rescales the ordinary wind-vector trajectory.
Event probability has one responsibility: corroborating high-impact actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .forecast_evidence import ForecastEvidence, validate_forecast_evidence


DEFAULT_STAGE_EVENT_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)


@dataclass(frozen=True)
class ForecastActionPolicyConfig:
    """Thresholds for forecast trend interpretation and action authorization."""

    enabled: bool = False
    stage_duration_s: float = 1200.0
    relative_speed_change_threshold: float = 0.08
    minimum_speed_change_ms: float = 0.5
    direction_consistency_min: float = 0.80
    reversal_angle_deg: float = 120.0
    continuous_reversal_points: int = 2
    high_impact_reliability_min: float = 0.65
    high_impact_event_probability_min: float = 0.60
    stage_event_keys: tuple[str, ...] = DEFAULT_STAGE_EVENT_KEYS

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        _validate_config(self)
        keys = tuple(str(key).strip() for key in self.stage_event_keys)
        if not keys or any(not key for key in keys):
            raise ValueError("stage_event_keys must contain non-empty names")
        if len(set(keys)) != len(keys):
            raise ValueError("stage_event_keys must not contain duplicates")
        object.__setattr__(self, "stage_event_keys", keys)


@dataclass(frozen=True)
class ActionAuthorization:
    """Whether one action family may enter downstream candidate evaluation."""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class ForecastStageTrend:
    """Trend evidence, action permissions and compatibility advice for one stage.

    ``hold`` remains a trend-level advisory for the V1 compatibility path. It
    is not the V2 ``continue_target`` lifecycle action: continuing an existing
    target may keep a pump running, whereas a forecast hold advisory does not
    issue an execution request by itself.
    """

    stage_index: int
    lead_start_s: float
    lead_end_s: float
    point_indices: tuple[int, ...]
    mean_speed_ms: float
    speed_change_ms: float
    relative_speed_change: float
    direction_consistency: float
    direction_consistent: bool
    mean_reliability: float
    event_probability: float | None
    event_threshold: float
    event_threshold_source: str
    high_impact_event_supported: bool
    strengthening: bool
    sustained: bool
    declining: bool
    continuous_reversal: bool
    reversal_run_length: int
    strengthen: ActionAuthorization
    hold: ActionAuthorization
    release: ActionAuthorization
    reverse: ActionAuthorization


@dataclass(frozen=True)
class ForecastActionPolicyResult:
    """Pure output of the independent v2 forecast action policy."""

    enabled: bool
    reason: str
    stage_duration_s: float = 1200.0
    stages: tuple[ForecastStageTrend, ...] = field(default_factory=tuple)


def build_forecast_action_policy_snapshot(
    result: ForecastActionPolicyResult,
    *,
    mode: str,
) -> dict[str, object] | None:
    """Convert a policy result to the stable decision-audit representation."""

    normalized_mode = str(mode)
    if normalized_mode == "off" or not result.enabled:
        return None
    if normalized_mode not in {"shadow", "enforce"}:
        raise ValueError("mode must be 'off', 'shadow', or 'enforce'")

    stage_labels: list[str] = []
    authorized: dict[str, list[str]] = {}
    blocked: dict[str, list[str]] = {}
    threshold_sources: dict[str, str] = {}
    reasons: dict[str, dict[str, str]] = {}
    for stage in result.stages:
        start_min = int(stage.stage_index * result.stage_duration_s / 60.0)
        end_min = int((stage.stage_index + 1) * result.stage_duration_s / 60.0)
        label = f"{start_min}-{end_min} min"
        stage_labels.append(label)
        action_authorizations = {
            "strengthen": stage.strengthen,
            "hold": stage.hold,
            "release": stage.release,
            "reverse": stage.reverse,
        }
        authorized[label] = [
            name for name, authorization in action_authorizations.items()
            if authorization.allowed
        ]
        blocked[label] = [
            name for name, authorization in action_authorizations.items()
            if not authorization.allowed
        ]
        threshold_sources[label] = str(stage.event_threshold_source)
        reasons[label] = {
            name: str(authorization.reason)
            for name, authorization in action_authorizations.items()
        }
    return {
        "mode": normalized_mode,
        "stage_labels": stage_labels,
        "authorized_actions": authorized,
        "blocked_actions": blocked,
        "threshold_source": threshold_sources,
        "reasons": reasons,
    }


def _validate_config(config: ForecastActionPolicyConfig) -> None:
    if config.stage_duration_s <= 0.0:
        raise ValueError("stage_duration_s must be positive")
    if config.relative_speed_change_threshold < 0.0:
        raise ValueError("relative_speed_change_threshold must be non-negative")
    if config.minimum_speed_change_ms < 0.0:
        raise ValueError("minimum_speed_change_ms must be non-negative")
    if not 0.0 <= config.direction_consistency_min <= 1.0:
        raise ValueError("direction_consistency_min must lie in [0, 1]")
    if not 0.0 < config.reversal_angle_deg <= 180.0:
        raise ValueError("reversal_angle_deg must lie in (0, 180]")
    if config.continuous_reversal_points < 2:
        raise ValueError("continuous_reversal_points must be at least 2")
    if not 0.0 <= config.high_impact_reliability_min <= 1.0:
        raise ValueError("high_impact_reliability_min must lie in [0, 1]")
    if not 0.0 <= config.high_impact_event_probability_min <= 1.0:
        raise ValueError("high_impact_event_probability_min must lie in [0, 1]")


def _coerce_inputs(
    evidence: ForecastEvidence | None,
    *,
    uv_ms: Sequence[Sequence[float]] | np.ndarray | None,
    lead_reliability: Sequence[float] | np.ndarray | None,
    lead_times_s: Sequence[float] | np.ndarray | None,
    event_probs: Mapping[str, float] | None,
    sample_period_s: float | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, float],
    Mapping[str, object],
]:
    if evidence is not None:
        raw_inputs = (uv_ms, lead_reliability, event_probs, sample_period_s)
        if any(value is not None for value in raw_inputs):
            raise ValueError("pass ForecastEvidence or raw forecast inputs, not both")
        validate_forecast_evidence(evidence)
        vectors = np.asarray(evidence.uv_ms, dtype=float)
        reliability = np.asarray(evidence.lead_reliability, dtype=float).reshape(-1)
        probabilities = {str(k): float(v) for k, v in evidence.event_probs.items()}
        period_s = float(evidence.sample_period_s)
        metadata = evidence.metadata
    else:
        if uv_ms is None:
            raise ValueError("uv_ms is required when ForecastEvidence is not provided")
        vectors = np.asarray(uv_ms, dtype=float)
        if vectors.ndim != 2 or vectors.shape[1] != 2 or vectors.shape[0] == 0:
            raise ValueError("uv_ms must have shape (H, 2)")
        if not np.all(np.isfinite(vectors)):
            raise ValueError("uv_ms must be finite")
        reliability = (
            np.ones(vectors.shape[0], dtype=float)
            if lead_reliability is None
            else np.asarray(lead_reliability, dtype=float).reshape(-1)
        )
        if reliability.shape != (vectors.shape[0],):
            raise ValueError("lead_reliability must contain one value per lead")
        reliability_invalid = np.any(
            (reliability < 0.0) | (reliability > 1.0)
        )
        if not np.all(np.isfinite(reliability)) or reliability_invalid:
            raise ValueError("lead_reliability must lie in [0, 1]")
        probabilities = {str(k): float(v) for k, v in (event_probs or {}).items()}
        if any(not np.isfinite(v) or v < 0.0 or v > 1.0 for v in probabilities.values()):
            raise ValueError("event probabilities must lie in [0, 1]")
        period_s = 600.0 if sample_period_s is None else float(sample_period_s)
        if not np.isfinite(period_s) or period_s <= 0.0:
            raise ValueError("sample_period_s must be positive")
        metadata = {}

    if lead_times_s is None:
        lead_times = period_s * np.arange(1, vectors.shape[0] + 1, dtype=float)
    else:
        lead_times = np.asarray(lead_times_s, dtype=float).reshape(-1)
        if lead_times.shape != (vectors.shape[0],):
            raise ValueError("lead_times_s must contain one value per lead")
        if not np.all(np.isfinite(lead_times)) or np.any(lead_times <= 0.0):
            raise ValueError("lead_times_s must be positive and finite")
        if np.any(np.diff(lead_times) <= 0.0):
            raise ValueError("lead_times_s must be strictly increasing")
    return vectors, reliability, lead_times, probabilities, metadata


def _event_thresholds_from_metadata(
    metadata: Mapping[str, object],
) -> dict[str, float]:
    """Read per-event model thresholds from a ForecastEvidence metadata blob.

    The preferred shape mirrors ``lstm_event_thresholds.json``::

        {"event_thresholds": {"thresholds": {"event_name": 0.8}}}

    A direct ``event_thresholds`` mapping is also accepted.  For callers that
    place the model threshold file itself in metadata, a top-level
    ``thresholds`` mapping is recognized as well.  Unrelated metadata is
    ignored, while malformed values in a recognized threshold map fail fast.
    """

    raw = metadata.get("event_thresholds")
    if raw is None:
        raw = metadata.get("model_event_thresholds")
    if raw is None:
        raw = metadata.get("lstm_event_thresholds")
    if raw is None and "thresholds" in metadata:
        raw = {"thresholds": metadata["thresholds"]}
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("ForecastEvidence.metadata['event_thresholds'] must be a mapping")
    nested = raw.get("thresholds")
    threshold_map = nested if nested is not None else raw
    if not isinstance(threshold_map, Mapping):
        raise ValueError("model event thresholds must be a mapping")

    thresholds: dict[str, float] = {}
    for name, raw_value in threshold_map.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model event threshold {name!r} must be numeric") from exc
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"model event threshold {name!r} must lie in [0, 1]")
        thresholds[str(name)] = value
    return thresholds


def _resolve_event_threshold(
    event_key: str | None,
    model_thresholds: Mapping[str, float],
    fallback: float,
) -> tuple[float, str]:
    if event_key is not None and event_key in model_thresholds:
        return float(model_thresholds[event_key]), "model_metadata"
    return float(fallback), "config_fallback"


def _direction_consistency(vectors: np.ndarray) -> float:
    magnitudes = np.linalg.norm(vectors, axis=1)
    denominator = float(np.sum(magnitudes))
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip(np.linalg.norm(np.sum(vectors, axis=0)) / denominator, 0.0, 1.0))


def _authorization(allowed: bool, allowed_reason: str, blocked_reason: str) -> ActionAuthorization:
    return ActionAuthorization(bool(allowed), allowed_reason if allowed else blocked_reason)


def _coerce_current_uv(
    current_uv_ms: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    if current_uv_ms is None:
        return None
    current = np.asarray(current_uv_ms, dtype=float).reshape(-1)
    if current.shape != (2,):
        raise ValueError("current_uv_ms must contain exactly two components")
    if not np.all(np.isfinite(current)):
        raise ValueError("current_uv_ms must be finite")
    return current


def _high_impact_gate(
    *,
    trend_present: bool,
    reliability: float,
    event_probability: float | None,
    event_threshold: float,
    config: ForecastActionPolicyConfig,
    action: str,
) -> ActionAuthorization:
    if not trend_present:
        return ActionAuthorization(False, f"no_{action}_trend")
    if reliability < config.high_impact_reliability_min:
        return ActionAuthorization(False, "low_forecast_reliability")
    if event_probability is None:
        return ActionAuthorization(False, "event_support_unavailable")
    if event_probability < event_threshold:
        return ActionAuthorization(False, "event_support_below_threshold")
    return ActionAuthorization(True, f"{action}_trend_reliable_and_event_supported")


def evaluate_forecast_action_policy(
    evidence: ForecastEvidence | None = None,
    *,
    uv_ms: Sequence[Sequence[float]] | np.ndarray | None = None,
    current_uv_ms: Sequence[float] | np.ndarray | None = None,
    lead_reliability: Sequence[float] | np.ndarray | None = None,
    lead_times_s: Sequence[float] | np.ndarray | None = None,
    event_probs: Mapping[str, float] | None = None,
    sample_period_s: float | None = None,
    config: ForecastActionPolicyConfig = ForecastActionPolicyConfig(),
) -> ForecastActionPolicyResult:
    """Interpret forecast evidence without changing any controller state.

    Six 10-minute point leads naturally form three 20-minute policy stages
    under the default configuration.  The first stage contains the ``+10`` and
    ``+20 min`` records; its ``0-20 min`` label is a policy/event window, not a
    claim that a forecast vector exists at ``t=0``.  When ``current_uv_ms`` is
    provided, the first-stage speed trend is measured from the current
    observation to the mean forecast speed in that stage, and direction
    reversal is referenced to the current wind vector.  Omitting it preserves
    the original forecast-only interpretation.
    Reliability and event probability gate only strengthening and reverse
    authorization; the ordinary trend flags use unscaled wind vectors.
    """

    _validate_config(config)
    if not config.enabled:
        return ForecastActionPolicyResult(enabled=False, reason="policy_disabled")

    vectors, reliability, lead_times, probabilities, metadata = _coerce_inputs(
        evidence,
        uv_ms=uv_ms,
        lead_reliability=lead_reliability,
        lead_times_s=lead_times_s,
        event_probs=event_probs,
        sample_period_s=sample_period_s,
    )
    current_vector = _coerce_current_uv(current_uv_ms)
    model_event_thresholds = _event_thresholds_from_metadata(metadata)
    speeds = np.linalg.norm(vectors, axis=1)
    # Prediction stages are right-closed: 10/20 min belong to stage 0,
    # 30/40 min to stage 1, and 50/60 min to stage 2.
    stage_ids = np.ceil(lead_times / config.stage_duration_s).astype(int) - 1

    first_stage_id = int(np.min(stage_ids))
    first_stage_indices = np.flatnonzero(stage_ids == first_stage_id)
    reference = (
        current_vector
        if current_vector is not None
        else np.sum(vectors[first_stage_indices], axis=0)
    )
    reference_norm = float(np.linalg.norm(reference))
    reversal_cosine = float(np.cos(np.deg2rad(config.reversal_angle_deg)))
    reversal_flags = np.zeros(vectors.shape[0], dtype=bool)
    if reference_norm > 1e-12:
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 1e-12
        dots = np.zeros(vectors.shape[0], dtype=float)
        dots[valid] = (vectors[valid] @ reference) / (norms[valid] * reference_norm)
        reversal_flags = valid & (dots <= reversal_cosine)

    reversal_runs = np.zeros(vectors.shape[0], dtype=int)
    run = 0
    for index, reversed_direction in enumerate(reversal_flags):
        run = run + 1 if bool(reversed_direction) else 0
        reversal_runs[index] = run

    stages: list[ForecastStageTrend] = []
    previous_stage_speed: float | None = None
    for stage_index in np.unique(stage_ids):
        indices = np.flatnonzero(stage_ids == stage_index)
        stage_vectors = vectors[indices]
        mean_speed = float(np.mean(speeds[indices]))
        if previous_stage_speed is None:
            if current_vector is None:
                speed_change = float(speeds[indices[-1]] - speeds[indices[0]])
                comparison_speed = float(max(speeds[indices[0]], 1e-12))
                consistency_vectors = stage_vectors
            else:
                current_speed = float(np.linalg.norm(current_vector))
                speed_change = float(mean_speed - current_speed)
                comparison_speed = float(max(current_speed, 1e-12))
                consistency_vectors = np.vstack((current_vector, stage_vectors))
        else:
            speed_change = float(mean_speed - previous_stage_speed)
            comparison_speed = float(max(previous_stage_speed, 1e-12))
            consistency_vectors = stage_vectors
        relative_change = float(speed_change / comparison_speed)
        change_threshold = max(
            config.minimum_speed_change_ms,
            config.relative_speed_change_threshold * comparison_speed,
        )
        consistency = _direction_consistency(consistency_vectors)
        direction_consistent = consistency >= config.direction_consistency_min
        strengthening = bool(speed_change >= change_threshold and direction_consistent)
        declining = bool(speed_change <= -change_threshold and direction_consistent)
        continuous_reversal = bool(
            np.max(reversal_runs[indices], initial=0) >= config.continuous_reversal_points
        )
        sustained = bool(
            not strengthening
            and not declining
            and not continuous_reversal
            and consistency >= config.direction_consistency_min
        )
        stage_reliability = float(np.mean(reliability[indices]))
        event_key = (
            config.stage_event_keys[int(stage_index)]
            if int(stage_index) < len(config.stage_event_keys)
            else None
        )
        event_probability = probabilities.get(event_key) if event_key is not None else None
        event_threshold, event_threshold_source = _resolve_event_threshold(
            event_key,
            model_event_thresholds,
            config.high_impact_event_probability_min,
        )
        event_supported = bool(
            event_probability is not None
            and event_probability >= event_threshold
        )

        strengthen_auth = _high_impact_gate(
            trend_present=strengthening,
            reliability=stage_reliability,
            event_probability=event_probability,
            event_threshold=event_threshold,
            config=config,
            action="strengthening",
        )
        reverse_auth = _high_impact_gate(
            trend_present=continuous_reversal,
            reliability=stage_reliability,
            event_probability=event_probability,
            event_threshold=event_threshold,
            config=config,
            action="continuous_reversal",
        )
        hold_auth = _authorization(
            sustained or declining,
            "stable_or_declining_forecast_supports_hold",
            "forecast_does_not_support_hold",
        )
        release_auth = _authorization(
            declining,
            "sustained_decline_supports_release",
            "no_sustained_decline",
        )
        stages.append(
            ForecastStageTrend(
                stage_index=int(stage_index),
                lead_start_s=float(lead_times[indices[0]]),
                lead_end_s=float(lead_times[indices[-1]]),
                point_indices=tuple(int(i) for i in indices),
                mean_speed_ms=mean_speed,
                speed_change_ms=speed_change,
                relative_speed_change=relative_change,
                direction_consistency=consistency,
                direction_consistent=direction_consistent,
                mean_reliability=stage_reliability,
                event_probability=(
                    None if event_probability is None else float(event_probability)
                ),
                event_threshold=event_threshold,
                event_threshold_source=event_threshold_source,
                high_impact_event_supported=event_supported,
                strengthening=strengthening,
                sustained=sustained,
                declining=declining,
                continuous_reversal=continuous_reversal,
                reversal_run_length=int(np.max(reversal_runs[indices], initial=0)),
                strengthen=strengthen_auth,
                hold=hold_auth,
                release=release_auth,
                reverse=reverse_auth,
            )
        )
        previous_stage_speed = mean_speed

    return ForecastActionPolicyResult(
        enabled=True,
        reason="evaluated",
        stage_duration_s=float(config.stage_duration_s),
        stages=tuple(stages),
    )
