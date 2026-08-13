"""Forecast-event admission used by control-facing planner logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


EVENT_RISK_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)
TRUSTED_EVENT_ATTENTION_KEYS = (
    "ballast_attention_event",
    *EVENT_RISK_KEYS,
)
TRUSTED_EVENT_DYNAMIC_KEYS = (
    "speed_ramp_ge_3ms",
    "direction_shift_ge_45deg",
    "vector_change_ge_train_p90",
)
TRUSTED_EVENT_HIGHWIND_KEY = "future_speed_ge_train_p95"


@dataclass(frozen=True)
class TrustedEventGateConfig:
    enabled: bool
    attention_threshold: float
    highwind_threshold: float
    dynamic_threshold: float
    attention_min_heads: int
    dynamic_min_heads: int
    dynamic_enabled: bool


def trusted_event_gate(
    event_probabilities: Mapping[str, float],
    config: TrustedEventGateConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Admit trusted event probabilities and fail closed otherwise."""

    raw = {str(key): float(value) for key, value in event_probabilities.items()}
    attention_hits = sum(
        int(raw.get(key, 0.0) >= config.attention_threshold)
        for key in TRUSTED_EVENT_ATTENTION_KEYS
    )
    dynamic_hits = sum(
        int(raw.get(key, 0.0) >= config.dynamic_threshold)
        for key in TRUSTED_EVENT_DYNAMIC_KEYS
    )
    highwind_hit = (
        raw.get(TRUSTED_EVENT_HIGHWIND_KEY, 0.0) >= config.highwind_threshold
    )
    attention_pass = attention_hits >= config.attention_min_heads
    dynamic_pass = config.dynamic_enabled and dynamic_hits >= config.dynamic_min_heads
    trusted = bool(highwind_hit or attention_pass or dynamic_pass)

    effective = dict(raw)
    if not config.enabled:
        reason = "disabled"
    elif trusted:
        reasons = []
        if highwind_hit:
            reasons.append("highwind")
        if attention_pass:
            reasons.append("attention_multihead")
        if dynamic_pass:
            reasons.append("dynamic_multihead")
        reason = "+".join(reasons) if reasons else "trusted"
    else:
        reason = "untrusted_event_signal"
        for key in EVENT_RISK_KEYS:
            effective[key] = 0.0
        effective["ballast_attention_event"] = 0.0

    diagnostics = {
        "enabled": int(config.enabled),
        "trusted": int(trusted) if config.enabled else 1,
        "reason": reason,
        "highwind_hit": int(highwind_hit),
        "attention_hits": int(attention_hits),
        "dynamic_hits": int(dynamic_hits),
        "highwind_prob": float(raw.get(TRUSTED_EVENT_HIGHWIND_KEY, 0.0)),
        "attention_threshold": float(config.attention_threshold),
        "highwind_threshold": float(config.highwind_threshold),
        "dynamic_threshold": float(config.dynamic_threshold),
    }
    return effective, diagnostics
