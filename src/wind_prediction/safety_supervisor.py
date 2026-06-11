"""Safety supervisor helpers for prediction-primary control.

The provider still owns the full supervisor decision flow. This module keeps
the safety state defaults and posture metrics in one place so the safety path
can be extracted behind a smaller interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReactiveFloorPostureMetrics:
    posture_vec_deg: np.ndarray
    pitch_abs_deg: float
    roll_abs_deg: float
    max_axis_deg: float
    theta_total_deg: float
    metric_value_deg: float
    theta_only_now: bool
    trigger_class: str
    enter: bool
    exit_ready: bool


@dataclass(frozen=True)
class ReactiveFloorThetaGateDecision:
    pass_gate: bool
    reason: str


@dataclass(frozen=True)
class ReactiveFloorVetoDecision:
    veto: bool
    reason: str


def posture_vector_deg(plant_info: dict[str, Any] | None) -> np.ndarray:
    """Return a two-axis posture vector in degrees."""

    posture = np.asarray(
        (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
        dtype=float,
    ).reshape(-1)
    if posture.size < 2:
        posture = np.pad(posture, (0, 2 - posture.size))
    return posture[:2]


def theta_total_deg(posture: np.ndarray) -> float:
    """Compute total tilt angle from pitch/roll degrees."""

    pitch = math.radians(float(posture[0]))
    roll = math.radians(float(posture[1]))
    return math.degrees(
        math.atan(math.sqrt(math.tan(pitch) ** 2 + math.tan(roll) ** 2))
    )


def reactive_floor_posture_metrics(
    plant_info: dict[str, Any] | None,
    *,
    high_posture_metric: str,
    high_posture_enter: float,
    high_posture_exit: float,
) -> ReactiveFloorPostureMetrics:
    """Compute posture metrics used by reactive-floor safety logic."""

    posture = posture_vector_deg(plant_info)
    pitch_abs = abs(float(posture[0]))
    roll_abs = abs(float(posture[1]))
    max_axis = max(pitch_abs, roll_abs)
    theta_total = theta_total_deg(posture)
    hybrid = str(high_posture_metric) == "hybrid"
    theta_only_now = (
        hybrid
        and max_axis <= float(high_posture_enter)
        and theta_total > float(high_posture_enter) + 0.5
    )
    trigger_class = "none"
    if max_axis > float(high_posture_enter):
        trigger_class = "max_axis"
    elif theta_only_now:
        trigger_class = "theta_total_only"
    metric_value = (
        max_axis
        if not hybrid
        else max(max_axis, theta_total - 0.5)
    )
    enter = (
        max_axis > float(high_posture_enter)
        if not hybrid
        else (
            max_axis > float(high_posture_enter)
            or theta_total > float(high_posture_enter) + 0.5
        )
    )
    exit_ready = (
        max_axis < float(high_posture_exit)
        if not hybrid
        else (
            max_axis < float(high_posture_exit)
            and theta_total < float(high_posture_exit) + 0.5
        )
    )
    return ReactiveFloorPostureMetrics(
        posture_vec_deg=posture,
        pitch_abs_deg=pitch_abs,
        roll_abs_deg=roll_abs,
        max_axis_deg=max_axis,
        theta_total_deg=theta_total,
        metric_value_deg=metric_value,
        theta_only_now=theta_only_now,
        trigger_class=trigger_class,
        enter=enter,
        exit_ready=exit_ready,
    )


def reactive_floor_theta_gate_decision(
    *,
    mode: str,
    theta_only_latched: bool,
    theta_total_deg: float,
    theta_elapsed_s: float,
    theta_only_enter_deg: float,
    theta_only_min_duration_s: float,
    posture_not_worsening: bool,
    target_stale: bool,
    pump_idle: bool,
    forecast_relief_clear: bool,
) -> ReactiveFloorThetaGateDecision:
    """Decide whether a theta-total-only floor episode should refresh now."""

    theta_mode = str(mode) in (
        "theta_only_gated",
        "relaxed_veto_on_theta_only",
    )
    if not (theta_mode and theta_only_latched):
        return ReactiveFloorThetaGateDecision(
            pass_gate=True,
            reason="not_theta_total_only",
        )

    theta_high = float(theta_total_deg) > float(theta_only_enter_deg)
    theta_sustained = (
        float(theta_elapsed_s) >= float(theta_only_min_duration_s)
    )
    theta_posture_worsening = not bool(posture_not_worsening)
    theta_stale_idle = bool(target_stale) and bool(pump_idle)
    theta_no_relief = not bool(forecast_relief_clear)
    pass_gate = bool(
        theta_high
        or theta_sustained
        or theta_posture_worsening
        or theta_stale_idle
        or theta_no_relief
    )
    if not pass_gate:
        return ReactiveFloorThetaGateDecision(
            pass_gate=False,
            reason="watch_theta_only",
        )

    passed = []
    if theta_high:
        passed.append("theta_high")
    if theta_sustained:
        passed.append("theta_sustained")
    if theta_posture_worsening:
        passed.append("posture_worsening")
    if theta_stale_idle:
        passed.append("stale_idle")
    if theta_no_relief:
        passed.append("forecast_no_relief")
    return ReactiveFloorThetaGateDecision(
        pass_gate=True,
        reason="pass_" + "_".join(passed),
    )


def reactive_floor_veto_decision(
    *,
    mode: str,
    theta_only_latched: bool,
    forecast_relief_clear: bool,
    posture_not_worsening: bool,
    target_stale: bool,
    short_risk_low: bool,
) -> ReactiveFloorVetoDecision:
    """Decide whether forecast evidence can veto a reactive-floor refresh."""

    if str(mode) == "relaxed_veto_on_theta_only" and bool(theta_only_latched):
        veto = bool(
            forecast_relief_clear
            and posture_not_worsening
            and not target_stale
        )
    else:
        veto = bool(
            forecast_relief_clear
            and posture_not_worsening
            and not target_stale
            and short_risk_low
        )
    reason = (
        "forecast_relief_current_stable_target_ok"
        if veto
        else "not_vetoed"
    )
    return ReactiveFloorVetoDecision(veto=veto, reason=reason)


def reactive_floor_state_defaults(reactive_floor_action: str) -> dict[str, Any]:
    """Return provider attributes for a fresh reactive-floor state."""

    return {
        "_reactive_floor_latched": False,
        "_reactive_floor_enter_start_s": None,
        "_reactive_floor_active": False,
        "_reactive_floor_veto_active": False,
        "_reactive_floor_reason": "disabled",
        "_reactive_floor_veto_reason": "disabled",
        "_reactive_floor_posture_metric_deg": 0.0,
        "_reactive_floor_max_axis_deg": 0.0,
        "_reactive_floor_theta_total_deg": 0.0,
        "_reactive_floor_current_response_deg": float("nan"),
        "_reactive_floor_target_stale": False,
        "_reactive_floor_target_err_mean_kg": 0.0,
        "_reactive_floor_target_age_s": 0.0,
        "_reactive_floor_pump_idle": False,
        "_reactive_floor_elapsed_s": 0.0,
        "_reactive_floor_resolved_action": str(reactive_floor_action),
        "_reactive_floor_medium_delay_active": False,
        "_reactive_floor_forecast_relief_clear": False,
        "_reactive_floor_short_risk_low": False,
        "_reactive_floor_delta_mean_kg": 0.0,
        "_reactive_floor_count": 0,
        "_reactive_floor_veto_count": 0,
        "_reactive_floor_trigger_class": "none",
        "_reactive_floor_theta_only_elapsed_s": 0.0,
        "_reactive_floor_theta_only_gate_pass": False,
        "_reactive_floor_theta_only_gate_reason": "disabled",
        "_reactive_floor_last_refresh_s": -1.0e12,
        "_reactive_floor_post_exit_active": False,
        "_reactive_floor_post_exit_reason": "disabled",
        "_reactive_floor_post_exit_released": False,
        "_reactive_floor_post_exit_release_count": 0,
        "_reactive_floor_post_exit_cap_count": 0,
        "_reactive_floor_post_exit_delta_before_kg": 0.0,
        "_reactive_floor_post_exit_delta_after_kg": 0.0,
        "_reactive_floor_post_exit_posture_metric_deg": 0.0,
        "_reactive_floor_post_exit_current_response_deg": float("nan"),
        "_reactive_floor_post_exit_episode_active": False,
        "_reactive_floor_post_exit_prev_metric_deg": float("nan"),
    }


def apply_reactive_floor_state_defaults(
    target: Any,
    *,
    reactive_floor_action: str,
) -> None:
    """Apply fresh reactive-floor attributes to a provider-like object."""

    for name, value in reactive_floor_state_defaults(
        reactive_floor_action,
    ).items():
        setattr(target, name, value)


def reactive_floor_log_fields(source: Any, prefix: str = "") -> dict[str, Any]:
    """Return telemetry fields for the current reactive-floor state."""

    if not bool(getattr(source, "reactive_floor_predictive_veto_enabled", False)):
        return {}
    return {
        f"{prefix}reactive_floor_enabled": int(
            source.reactive_floor_predictive_veto_enabled
        ),
        f"{prefix}reactive_floor_mode": source.reactive_floor_predictive_veto_mode,
        f"{prefix}reactive_floor_metric": source.high_posture_metric,
        f"{prefix}reactive_floor_action": source.reactive_floor_action,
        f"{prefix}reactive_floor_resolved_action": (
            source._reactive_floor_resolved_action
        ),
        f"{prefix}reactive_floor_medium_delay_s": float(
            source.reactive_floor_medium_delay_s
        ),
        f"{prefix}reactive_floor_medium_delay_active": int(
            source._reactive_floor_medium_delay_active
        ),
        f"{prefix}reactive_floor_elapsed_s": float(source._reactive_floor_elapsed_s),
        f"{prefix}reactive_floor_active": int(source._reactive_floor_active),
        f"{prefix}reactive_floor_veto_active": int(source._reactive_floor_veto_active),
        f"{prefix}reactive_floor_reason": source._reactive_floor_reason,
        f"{prefix}reactive_floor_veto_reason": source._reactive_floor_veto_reason,
        f"{prefix}reactive_floor_trigger_class": (
            source._reactive_floor_trigger_class
        ),
        f"{prefix}reactive_floor_theta_only_elapsed_s": float(
            source._reactive_floor_theta_only_elapsed_s
        ),
        f"{prefix}reactive_floor_theta_only_gate_pass": int(
            source._reactive_floor_theta_only_gate_pass
        ),
        f"{prefix}reactive_floor_theta_only_gate_reason": (
            source._reactive_floor_theta_only_gate_reason
        ),
        f"{prefix}reactive_floor_posture_metric_deg": float(
            source._reactive_floor_posture_metric_deg
        ),
        f"{prefix}reactive_floor_max_axis_deg": float(
            source._reactive_floor_max_axis_deg
        ),
        f"{prefix}reactive_floor_theta_total_deg": float(
            source._reactive_floor_theta_total_deg
        ),
        f"{prefix}reactive_floor_current_response_deg": float(
            source._reactive_floor_current_response_deg
        ),
        f"{prefix}reactive_floor_target_stale": int(
            source._reactive_floor_target_stale
        ),
        f"{prefix}reactive_floor_target_err_mean_kg": float(
            source._reactive_floor_target_err_mean_kg
        ),
        f"{prefix}reactive_floor_target_age_s": float(
            source._reactive_floor_target_age_s
        ),
        f"{prefix}reactive_floor_pump_idle": int(source._reactive_floor_pump_idle),
        f"{prefix}reactive_floor_forecast_relief_clear": int(
            source._reactive_floor_forecast_relief_clear
        ),
        f"{prefix}reactive_floor_short_risk_low": int(
            source._reactive_floor_short_risk_low
        ),
        f"{prefix}reactive_floor_delta_mean_kg": float(
            source._reactive_floor_delta_mean_kg
        ),
        f"{prefix}reactive_floor_count": int(source._reactive_floor_count),
        f"{prefix}reactive_floor_veto_count": int(source._reactive_floor_veto_count),
        f"{prefix}reactive_floor_post_exit_mode": source.reactive_floor_post_exit_mode,
        f"{prefix}reactive_floor_post_exit_active": int(
            source._reactive_floor_post_exit_active
        ),
        f"{prefix}reactive_floor_post_exit_reason": (
            source._reactive_floor_post_exit_reason
        ),
        f"{prefix}reactive_floor_post_exit_released": int(
            source._reactive_floor_post_exit_released
        ),
        f"{prefix}reactive_floor_post_exit_release_count": int(
            source._reactive_floor_post_exit_release_count
        ),
        f"{prefix}reactive_floor_post_exit_cap_count": int(
            source._reactive_floor_post_exit_cap_count
        ),
        f"{prefix}reactive_floor_post_exit_delta_before_kg": float(
            source._reactive_floor_post_exit_delta_before_kg
        ),
        f"{prefix}reactive_floor_post_exit_delta_after_kg": float(
            source._reactive_floor_post_exit_delta_after_kg
        ),
        f"{prefix}reactive_floor_post_exit_posture_metric_deg": float(
            source._reactive_floor_post_exit_posture_metric_deg
        ),
        f"{prefix}reactive_floor_post_exit_current_response_deg": float(
            source._reactive_floor_post_exit_current_response_deg
        ),
        f"{prefix}reactive_floor_post_exit_episode_active": int(
            source._reactive_floor_post_exit_episode_active
        ),
    }
