"""Compatibility telemetry defaults for frozen control overlays.

These helpers keep historical CSV/report columns stable after an overlay's
runtime path has been removed. They must stay behavior-neutral: constants only,
no planner state and no control decisions.
"""

from __future__ import annotations

from typing import Any


def _prefixed(prefix: str, fields: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in fields.items()}


def frozen_overlay_record_defaults(reason: str = "disabled") -> dict[str, Any]:
    """Historical per-record fields for overlays with removed runtime paths."""
    reason_text = str(reason)
    return {
        "event_risk_pressure_boost_enabled": 0,
        "event_risk_pressure_floor_enabled": 0,
        "event_risk_floor_active_0_20m": 0,
        "event_risk_floor_active_20_40m": 0,
        "event_risk_floor_active_40_60m": 0,
        "event_risk_floor_norm_0_20m": 0.0,
        "event_risk_floor_norm_20_40m": 0.0,
        "event_risk_floor_norm_40_60m": 0.0,
        "preview_lead_action_enabled": 0,
        "preview_lead_action_active": 0,
        "preview_lead_action_reason": reason_text,
        "preview_lead_action_block_index": -1,
        "preview_lead_action_name": "",
        "preview_sequence_lead_action_active": 0,
        "preview_sequence_lead_action_reason": reason_text,
        "preview_sequence_lead_action_block_index": -1,
        "preview_sequence_lead_action_name": "",
        "hold_risk_micro_action_enabled": 0,
        "hold_risk_micro_action_active": 0,
        "hold_risk_micro_action_reason": reason_text,
        "hold_risk_micro_action_name": "",
        "quiet_posture_action_active": 0,
        "quiet_posture_action_reason": reason_text,
        "no_unexplained_hold_action_enabled": 0,
        "no_unexplained_hold_action_active": 0,
        "no_unexplained_hold_action_reason": reason_text,
        "no_unexplained_hold_action_before": "",
        "no_unexplained_hold_action_after": "",
        "no_unexplained_hold_threshold_deg": 0.0,
        "no_unexplained_hold_trigger_buckets": 0,
        "no_unexplained_hold_consecutive_high_buckets": 0,
        "no_unexplained_hold_clean_buckets": 0,
        "no_unexplained_hold_pitch_abs_deg": 0.0,
        "no_unexplained_hold_roll_abs_deg": 0.0,
        "no_unexplained_hold_target_err_mean_kg": 0.0,
        "no_unexplained_hold_pump_rate_m3_min": 0.0,
        "no_unexplained_hold_fullspeed_proxy": 0,
        "medium_escalation_enabled": 0,
        "medium_escalation_active": 0,
        "medium_escalation_reason": reason_text,
        "medium_escalation_candidate_action": "",
        "medium_escalation_pitch_abs_deg": 0.0,
        "medium_escalation_roll_abs_deg": 0.0,
        "medium_escalation_pitch_rate_deg_s": 0.0,
        "medium_escalation_pressure_norm": 0.0,
        "medium_escalation_future_pressure_norm": 0.0,
        "medium_escalation_small_action_norm": 0.0,
        "medium_escalation_medium_action_norm": 0.0,
        "medium_escalation_small_pitch_residual_deg": 0.0,
        "medium_escalation_medium_pitch_residual_deg": 0.0,
        "medium_escalation_pitch_improve_deg": 0.0,
        "medium_escalation_fallback_dominated": 0,
        "medium_escalation_lowrisk_posture": 0,
    }


def frozen_overlay_preview_defaults(reason: str = "disabled") -> dict[str, Any]:
    """Preview-prefixed fields for overlays with removed runtime paths."""
    record_defaults = frozen_overlay_record_defaults(reason)
    preview_like = {
        key: value
        for key, value in record_defaults.items()
        if key.startswith("preview_")
    }
    unprefixed = {
        key: value
        for key, value in record_defaults.items()
        if not key.startswith("preview_")
    }
    return {**preview_like, **_prefixed("preview_", unprefixed)}


def frozen_active_intent_record_defaults(prefix: str = "") -> dict[str, Any]:
    """Telemetry compatibility for the removed active-intent reproposal path."""
    return _prefixed(
        prefix,
        {
            "primary_active_intent_reproposal": 0,
            "primary_active_intent_reproposal_reason": "disabled",
            "primary_active_intent_reproposal_delta_mean_kg": 0.0,
            "primary_active_intent_reproposal_eligible": 0,
            "primary_active_intent_reproposal_eligibility_reason": "disabled",
            "primary_active_intent_reproposal_target_err_mean_kg": 0.0,
            "primary_active_intent_reproposal_pump_rate_m3_min": 0.0,
        },
    )


def frozen_recovery_mode_record_defaults(prefix: str = "") -> dict[str, Any]:
    """Telemetry compatibility for the removed recovery-mode observer."""
    return _prefixed(
        prefix,
        {
            "recovery_mode_enabled": 0,
            "recovery_mode_active": 0,
            "recovery_mode_entry_reason": "",
            "recovery_mode_exit_reason": "",
            "recovery_mode_duration_buckets": 0,
            "recovery_mode_consecutive_high_pitch": 0,
            "recovery_mode_consecutive_high_roll": 0,
            "recovery_mode_attitude_trend": 0.0,
            "recovery_mode_pressure_support": 0.0,
            "recovery_mode_fullspeed_proxy": 0,
            "recovery_mode_lowrisk_rejected": 0,
            "recovery_mode_suggested_level": "hold",
            "recovery_mode_action_effective": 0,
            "recovery_mode_pump_budget_used_m3": 0.0,
        },
    )


def frozen_pump_suppression_record_defaults(reason: str = "disabled") -> dict[str, Any]:
    """Telemetry compatibility for the frozen preview pump-suppression path."""
    reason_text = str(reason)
    return {
        "pump_suppression_active": 0,
        "pump_suppression_restart_err_kg": 0.0,
        "pump_suppression_reason": reason_text,
        "pump_suppression_plateau_enabled": 0,
        "pump_suppression_plateau_only": 0,
        "pump_suppression_plateau_candidate": 0,
        "pump_suppression_plateau_range_norm": 0.0,
        "pump_suppression_plateau_rise_norm": 0.0,
        "pump_suppression_plateau_near_min_norm": 0.0,
        "pump_suppression_plateau_boundary_veto": 0,
        "pump_suppression_plateau_latched_active": 0,
        "pump_suppression_plateau_latched_until_s": 0.0,
        "pump_suppression_event_risk_guard_enabled": 0,
        "pump_suppression_event_risk_guard_threshold": 0.0,
    }


def frozen_pump_suppression_preview_defaults(reason: str = "disabled") -> dict[str, Any]:
    """Preview-prefixed compatibility fields for frozen pump suppression."""
    record_defaults = frozen_pump_suppression_record_defaults(reason)
    preview = _prefixed("preview_", record_defaults)
    preview["preview_pump_restart_err_kg"] = preview.pop(
        "preview_pump_suppression_restart_err_kg"
    )
    return preview


def frozen_hold_comfort_record_defaults(reason: str = "disabled") -> dict[str, Any]:
    """Telemetry compatibility for the removed hold-comfort output overlay."""
    reason_text = str(reason)
    return {
        "hold_comfort_release_enabled": 0,
        "hold_comfort_no_preview_only": 0,
        "hold_comfort_release_active": 0,
        "hold_comfort_release_reason": reason_text,
        "hold_comfort_forecast_veto_enabled": 0,
        "hold_comfort_forecast_veto_active": 0,
        "hold_comfort_forecast_veto_reason": reason_text,
        "hold_comfort_veto_posture_relief": 0.0,
        "hold_comfort_veto_posture_specific_only": 0,
        "hold_comfort_pressure_max_norm": 0.0,
        "hold_comfort_current_pressure_max_norm": 0.0,
        "hold_comfort_release_pitch_abs_deg": 0.0,
        "hold_comfort_release_roll_abs_deg": 0.0,
        "hold_comfort_action_name": "",
    }


def frozen_hold_comfort_preview_defaults(reason: str = "disabled") -> dict[str, Any]:
    """Preview-prefixed compatibility fields for removed hold comfort."""
    return _prefixed("preview_", frozen_hold_comfort_record_defaults(reason))
