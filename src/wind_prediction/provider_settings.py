"""Validation and normalization of forecast-assisted provider settings."""

from __future__ import annotations

from typing import Any

from .forecast_adapter import ForecastModelAdapter
from .forecast_action_policy import ForecastActionPolicyConfig
from .provider_defaults import ProviderOptions


def apply_provider_settings(
    provider: Any,
    settings: ProviderOptions,
    forecast_adapter: ForecastModelAdapter | None,
) -> None:
    """Install validated immutable settings on a provider instance."""
    provider.scale = float(settings.scale)
    if settings.bias_shape not in ("hold", "event_decay"):
        raise ValueError(f"unsupported bias_shape={settings.bias_shape!r}")
    if settings.objective_mode not in ("lex", "economic"):
        raise ValueError(f"unsupported objective_mode={settings.objective_mode!r}")
    if settings.event_reset_mode not in ("action", "target_change", "active_bucket"):
        raise ValueError(f"unsupported event_reset_mode={settings.event_reset_mode!r}")
    if settings.target_lifecycle_mode not in ("legacy", "planner_authoritative"):
        raise ValueError(
            "unsupported target_lifecycle_mode="
            f"{settings.target_lifecycle_mode!r}"
        )
    provider.bias_shape = str(settings.bias_shape)
    provider.decay_tau_s = float(settings.decay_tau_s)
    provider.objective_mode = str(settings.objective_mode)
    provider.event_reset_mode = str(settings.event_reset_mode)
    provider.target_lifecycle_mode = str(settings.target_lifecycle_mode)
    if settings.forecast_action_policy_mode not in ("off", "shadow", "enforce"):
        raise ValueError(
            "unsupported forecast_action_policy_mode="
            f"{settings.forecast_action_policy_mode!r}"
        )
    policy_config = settings.forecast_action_policy_config
    if policy_config is None:
        policy_config = ForecastActionPolicyConfig()
    if not isinstance(policy_config, ForecastActionPolicyConfig):
        raise TypeError(
            "forecast_action_policy_config must be ForecastActionPolicyConfig"
        )
    if (settings.forecast_action_policy_mode == "off") != (not policy_config.enabled):
        raise ValueError(
            "forecast_action_policy_mode and forecast_action_policy_config.enabled "
            "must agree"
        )
    provider.forecast_action_policy_mode = str(
        settings.forecast_action_policy_mode
    )
    provider.forecast_action_policy_config = policy_config
    provider._forecast_action_policy_result = None
    provider._forecast_action_policy_snapshot = None
    provider.target_reset_tol_deg = float(settings.target_reset_tol_deg)
    provider.forecast_adapter = forecast_adapter
    # Latest blend-forecast diagnostic, populated by _forecast_uv each
    # planner step when the forecast_adapter is a BlendedForecastAdapter.
    # Empty for plain/oracle/persistence/current_only sources.
    provider._last_forecast_blend_diagnostic: dict[str, Any] = {}
    if abs(float(settings.setpoint_bias_sign)) < 1e-9:
        raise ValueError("setpoint_bias_sign must be nonzero")
    provider.setpoint_bias_sign = float(settings.setpoint_bias_sign)
    # Feedforward channel: emit per-tank mass delta (kg) directly into the
    # closed-loop command stream BEFORE the rate limiter, AFTER the heave
    # balancer. This bypasses the PI deadband, which is the structural
    # ceiling of the setpoint-shift channel (verified 2026-05-04).
    # The FF channel uses the PLANNER FRAME (compensation_direction) sign
    # of first_avec, NOT the setpoint frame -- because we are directly
    # commanding ballast motion, not shifting a PI target.
    provider.ff_channel_enabled = bool(settings.ff_channel_enabled)
    provider.ff_scale = float(settings.ff_scale)
    provider.setpoint_channel_enabled = bool(settings.setpoint_channel_enabled)
    # Prediction-primary prototype: instead of nudging the PI setpoint, emit
    # an absolute three-tank mass target that ClosedLoopPolicy can apply
    # directly before the plant pump latch/rate layer. Default-off preserves
    # all existing production behavior.
    provider.prediction_primary_enabled = bool(settings.prediction_primary_enabled)
    provider.prediction_primary_scale = float(settings.prediction_primary_scale)
    provider.prediction_primary_disable_setpoint = bool(settings.prediction_primary_disable_setpoint)
    if settings.primary_hold_target_mode not in ("current", "pause", "pi_release", "forecast_pause"):
        raise ValueError(
            f"unsupported primary_hold_target_mode={settings.primary_hold_target_mode!r}"
    )
    provider.primary_hold_target_mode = str(settings.primary_hold_target_mode)
    provider.trusted_event_gate_enabled = bool(settings.trusted_event_gate_enabled)
    provider.trusted_event_highwind_threshold = float(settings.trusted_event_highwind_threshold)
    provider.trusted_event_attention_threshold = float(settings.trusted_event_attention_threshold)
    provider.trusted_event_attention_min_heads = max(1, int(settings.trusted_event_attention_min_heads))
    provider.trusted_event_dynamic_enabled = bool(settings.trusted_event_dynamic_enabled)
    provider.trusted_event_dynamic_threshold = float(settings.trusted_event_dynamic_threshold)
    provider.trusted_event_dynamic_min_heads = max(1, int(settings.trusted_event_dynamic_min_heads))
    provider.far_horizon_enabled = bool(settings.far_horizon_enabled)
    provider.far_horizon_relief_gate_enabled = bool(settings.far_horizon_relief_gate_enabled)
    provider.far_horizon_weights = (
        max(float(settings.far_horizon_weight_60_80), 0.0),
        max(float(settings.far_horizon_weight_80_100), 0.0),
        max(float(settings.far_horizon_weight_100_120), 0.0),
    )
    provider.far_horizon_relief_pitch_abs_max_deg = max(
        float(settings.far_horizon_relief_pitch_abs_max_deg),
        0.0,
    )
    provider.far_horizon_relief_roll_abs_max_deg = max(
        float(settings.far_horizon_relief_roll_abs_max_deg),
        0.0,
    )
    provider.relief_medium_cap_enabled = bool(settings.relief_medium_cap_enabled)
    provider.relief_medium_cap_event_threshold = float(settings.relief_medium_cap_event_threshold)
    provider.relief_medium_cap_ratio = float(settings.relief_medium_cap_ratio)
    provider.relief_medium_cap_adaptive = bool(settings.relief_medium_cap_adaptive)
    provider.relief_medium_cap_pitch_abs_max_deg = max(
        float(settings.relief_medium_cap_pitch_abs_max_deg),
        0.0,
    )
    provider.relief_medium_cap_roll_abs_max_deg = max(
        float(settings.relief_medium_cap_roll_abs_max_deg),
        0.0,
    )
    provider.active_posture_refresh_enabled = bool(settings.active_posture_refresh_enabled)
    provider.active_posture_refresh_enter_deg = max(
        float(settings.active_posture_refresh_enter_deg),
        0.0,
    )
    provider.active_posture_refresh_exit_deg = max(
        float(settings.active_posture_refresh_exit_deg),
        0.0,
    )
    provider.active_posture_refresh_update_interval_s = max(
        float(settings.active_posture_refresh_update_interval_s),
        1.0,
    )
    provider.active_posture_refresh_pressure_max_norm = max(
        float(settings.active_posture_refresh_pressure_max_norm),
        0.0,
    )
    provider.active_posture_refresh_min_target_age_s = max(
        float(settings.active_posture_refresh_min_target_age_s),
        0.0,
    )
    if settings.active_posture_refresh_axis_mode not in ("max", "axis_debt"):
        raise ValueError(
            "unsupported active_posture_refresh_axis_mode="
            f"{settings.active_posture_refresh_axis_mode!r}"
        )
    provider.active_posture_refresh_axis_mode = str(settings.active_posture_refresh_axis_mode)
    provider.active_posture_refresh_worsening_eps_deg_s = max(
        float(settings.active_posture_refresh_worsening_eps_deg_s),
        0.0,
    )
    provider.active_posture_refresh_debt_deg = max(
        float(settings.active_posture_refresh_debt_deg),
        0.0,
    )
    provider.active_posture_refresh_include_hold = bool(
        settings.active_posture_refresh_include_hold
    )
    if settings.active_posture_refresh_action_name not in (
        "pump_saving",
        "active_small",
        "active_medium",
    ):
        raise ValueError(
            "unsupported active_posture_refresh_action_name="
            f"{settings.active_posture_refresh_action_name!r}"
        )
    provider.active_posture_refresh_action_name = str(
        settings.active_posture_refresh_action_name
    )
    provider.fallback_risk_active_release_enabled = bool(
        settings.fallback_risk_active_release_enabled
    )
    provider.fallback_risk_active_release_enter_deg = max(
        float(settings.fallback_risk_active_release_enter_deg),
        0.0,
    )
    provider.fallback_risk_active_release_exit_deg = max(
        float(settings.fallback_risk_active_release_exit_deg),
        0.0,
    )
    provider.fallback_risk_active_release_worsening_eps_deg_s = max(
        float(settings.fallback_risk_active_release_worsening_eps_deg_s),
        0.0,
    )
    provider.fallback_risk_active_release_pump_rate_m3_min = max(
        float(settings.fallback_risk_active_release_pump_rate_m3_min),
        0.0,
    )
    provider.fallback_risk_active_release_backlog_kg = max(
        float(settings.fallback_risk_active_release_backlog_kg),
        0.0,
    )
    provider.no_preview_myopic_horizon_enabled = bool(settings.no_preview_myopic_horizon_enabled)
    provider.primary_stall_refresh_enabled = bool(settings.primary_stall_refresh_enabled)
    provider.primary_stall_pitch_deg = float(settings.primary_stall_pitch_deg)
    provider.primary_stall_roll_deg = float(settings.primary_stall_roll_deg)
    provider.primary_stall_target_err_kg = float(settings.primary_stall_target_err_kg)
    provider.primary_stall_pump_rate_m3_min = float(settings.primary_stall_pump_rate_m3_min)
    provider.primary_stall_min_age_s = float(settings.primary_stall_min_age_s)
    provider.primary_stall_pitch_axis_bias = bool(settings.primary_stall_pitch_axis_bias)
    provider.primary_stall_pitch_axis_roll_max_deg = float(
        settings.primary_stall_pitch_axis_roll_max_deg
    )
    provider.active_effectiveness_refresh_enabled = bool(
        settings.active_effectiveness_refresh_enabled
    )
    provider.active_effectiveness_refresh_min_active_buckets = max(
        int(settings.active_effectiveness_refresh_min_active_buckets),
        1,
    )
    provider.active_effectiveness_refresh_pitch_deg = max(
        float(settings.active_effectiveness_refresh_pitch_deg),
        0.0,
    )
    provider.active_effectiveness_refresh_roll_deg = max(
        float(settings.active_effectiveness_refresh_roll_deg),
        0.0,
    )
    provider.active_effectiveness_refresh_target_err_kg = max(
        float(settings.active_effectiveness_refresh_target_err_kg),
        0.0,
    )
    provider.active_effectiveness_refresh_pump_rate_m3_min = max(
        float(settings.active_effectiveness_refresh_pump_rate_m3_min),
        0.0,
    )
    provider.active_effectiveness_refresh_min_target_age_s = max(
        float(settings.active_effectiveness_refresh_min_target_age_s),
        0.0,
    )
    provider.active_effectiveness_refresh_min_pressure_norm = max(
        float(settings.active_effectiveness_refresh_min_pressure_norm),
        0.0,
    )
    provider.active_effectiveness_refresh_response_eps_deg = max(
        float(settings.active_effectiveness_refresh_response_eps_deg),
        0.0,
    )
    provider.active_effectiveness_refresh_required_worsening_deg = max(
        float(settings.active_effectiveness_refresh_required_worsening_deg),
        0.0,
    )
    provider.active_effectiveness_refresh_min_delta_kg = max(
        float(settings.active_effectiveness_refresh_min_delta_kg),
        0.0,
    )
    provider.active_effectiveness_refresh_episode_budget_m3 = max(
        float(settings.active_effectiveness_refresh_episode_budget_m3),
        0.0,
    )
    provider.active_effectiveness_refresh_lowrisk_pressure_norm = max(
        float(settings.active_effectiveness_refresh_lowrisk_pressure_norm),
        0.0,
    )
    provider.gusty_oscillation_enabled = bool(settings.gusty_oscillation_enabled)
    provider.gusty_oscillation_min_target_age_s = max(
        float(settings.gusty_oscillation_min_target_age_s),
        0.0,
    )
    provider.gusty_oscillation_target_scale = min(
        max(float(settings.gusty_oscillation_target_scale), 0.0),
        1.0,
    )
    provider.gusty_oscillation_headroom_scaled = bool(
        settings.gusty_oscillation_headroom_scaled
    )
    provider.gusty_oscillation_low_posture_deg = max(
        float(settings.gusty_oscillation_low_posture_deg),
        0.0,
    )
    provider.gusty_oscillation_high_posture_deg = max(
        float(settings.gusty_oscillation_high_posture_deg),
        provider.gusty_oscillation_low_posture_deg,
    )
    provider.gusty_oscillation_low_posture_scale = min(
        max(float(settings.gusty_oscillation_low_posture_scale), 0.0),
        1.0,
    )
    provider.gusty_oscillation_high_posture_scale = min(
        max(float(settings.gusty_oscillation_high_posture_scale), 0.0),
        1.0,
    )
    provider.gusty_oscillation_short_gate_enabled = bool(
        settings.gusty_oscillation_short_gate_enabled
    )
    provider.gusty_oscillation_short_min_range_ms = max(
        float(settings.gusty_oscillation_short_min_range_ms),
        0.0,
    )
    provider.gusty_oscillation_short_min_pressure_norm = max(
        float(settings.gusty_oscillation_short_min_pressure_norm),
        0.0,
    )
    provider.gusty_oscillation_short_max_dir_shift_deg = max(
        float(settings.gusty_oscillation_short_max_dir_shift_deg),
        0.0,
    )
    provider.gusty_oscillation_short_min_turns = max(
        int(settings.gusty_oscillation_short_min_turns),
        0,
    )
    if settings.gusty_hold_current_gate_mode not in ("off", "gate_on", "gate_off"):
        raise ValueError(
            "gusty_hold_current_gate_mode must be off, gate_on, or gate_off"
        )
    provider.gusty_hold_current_gate_mode = str(settings.gusty_hold_current_gate_mode)
    provider.gusty_hold_current_isolated_primary_enabled = bool(
        settings.gusty_hold_current_isolated_primary_enabled
    )
    if settings.direction_reversal_hold_current_gate_mode not in (
        "off",
        "gate_on",
        "gate_off",
    ):
        raise ValueError(
            "direction_reversal_hold_current_gate_mode must be off, gate_on, or gate_off"
        )
    provider.direction_reversal_hold_current_gate_mode = str(
        settings.direction_reversal_hold_current_gate_mode
    )
    provider.forecast_control_trust_gate_enabled = bool(
        settings.forecast_control_trust_gate_enabled
    )
    provider.forecast_control_trust_event_threshold = min(
        max(float(settings.forecast_control_trust_event_threshold), 0.0),
        1.0,
    )
    provider.forecast_control_trust_current_stability_required = bool(
        settings.forecast_control_trust_current_stability_required
    )
    provider.forecast_control_trust_max_speed_range_ms = max(
        float(settings.forecast_control_trust_max_speed_range_ms),
        0.0,
    )
    provider.forecast_control_trust_max_dir_shift_deg = max(
        float(settings.forecast_control_trust_max_dir_shift_deg),
        0.0,
    )
    provider.forecast_control_trust_suppression_rise_norm = max(
        float(settings.forecast_control_trust_suppression_rise_norm),
        0.0,
    )
    provider.forecast_control_trust_min_pressure_norm = max(
        float(settings.forecast_control_trust_min_pressure_norm),
        0.0,
    )
    provider.forecast_pressure_trust_gate_enabled = bool(
        settings.forecast_pressure_trust_gate_enabled
    )
    provider.forecast_pressure_trust_current_norm_max = max(
        float(settings.forecast_pressure_trust_current_norm_max),
        0.0,
    )
    provider.forecast_pressure_trust_future_norm_min = max(
        float(settings.forecast_pressure_trust_future_norm_min),
        0.0,
    )
    provider.forecast_pressure_trust_event_threshold = min(
        max(float(settings.forecast_pressure_trust_event_threshold), 0.0),
        1.0,
    )
    provider.forecast_pressure_trust_untrusted_scale = min(
        max(float(settings.forecast_pressure_trust_untrusted_scale), 0.0),
        1.0,
    )
    provider.forecast_pressure_trust_min_direction_dot = float(
        settings.forecast_pressure_trust_min_direction_dot
    )
    provider.forecast_pressure_trust_spike_margin_norm = max(
        float(settings.forecast_pressure_trust_spike_margin_norm),
        0.0,
    )
    provider.gusty_hold_current_runtime_release_enabled = bool(
        settings.gusty_hold_current_runtime_release_enabled
    )
    provider.gusty_hold_current_runtime_release_deg = max(
        float(settings.gusty_hold_current_runtime_release_deg),
        0.0,
    )
    provider.gusty_hold_current_runtime_release_cooldown_s = max(
        float(settings.gusty_hold_current_runtime_release_cooldown_s),
        0.0,
    )
    provider.gusty_hold_current_debt_guard_enabled = bool(
        settings.gusty_hold_current_debt_guard_enabled
    )
    provider.gusty_hold_current_debt_guard_target_motion_kg_s = max(
        float(settings.gusty_hold_current_debt_guard_target_motion_kg_s),
        0.0,
    )
    provider.gusty_hold_current_debt_guard_target_err_kg = max(
        float(settings.gusty_hold_current_debt_guard_target_err_kg),
        0.0,
    )
    provider.gusty_hold_current_debt_guard_min_age_s = max(
        float(settings.gusty_hold_current_debt_guard_min_age_s),
        0.0,
    )
    provider.gusty_hold_current_day_budget_enabled = bool(
        settings.gusty_hold_current_day_budget_enabled
    )
    provider.gusty_hold_current_day_budget_max_buckets = max(
        int(settings.gusty_hold_current_day_budget_max_buckets),
        0,
    )
    provider.gusty_hold_current_day_budget_window_s = max(
        float(settings.gusty_hold_current_day_budget_window_s),
        1.0,
    )
    provider.forecast_advised_economy_enabled = bool(
        settings.forecast_advised_economy_enabled
    )
    provider.forecast_advised_economy_min_headroom_deg = max(
        float(settings.forecast_advised_economy_min_headroom_deg),
        0.0,
    )
    provider.forecast_advised_economy_rise_norm = max(
        float(settings.forecast_advised_economy_rise_norm),
        0.0,
    )
    provider.forecast_advised_economy_max_speed_range_ms = max(
        float(settings.forecast_advised_economy_max_speed_range_ms),
        0.0,
    )
    provider.forecast_advised_economy_max_dir_shift_deg = max(
        float(settings.forecast_advised_economy_max_dir_shift_deg),
        0.0,
    )
    provider.forecast_advised_economy_confirm_buckets = max(
        int(settings.forecast_advised_economy_confirm_buckets),
        1,
    )
    provider.forecast_advised_economy_hold_confirm_buckets = max(
        int(settings.forecast_advised_economy_hold_confirm_buckets),
        0,
    )
    provider.forecast_advised_economy_min_axis_deg = max(
        float(settings.forecast_advised_economy_min_axis_deg),
        0.0,
    )
    provider.forecast_advised_economy_min_pressure_norm = max(
        float(settings.forecast_advised_economy_min_pressure_norm),
        0.0,
    )
    provider.forecast_advised_economy_posture_release_deg = max(
        float(settings.forecast_advised_economy_posture_release_deg),
        0.0,
    )
    provider.forecast_advised_economy_release_cooldown_s = max(
        float(settings.forecast_advised_economy_release_cooldown_s),
        0.0,
    )
    provider.forecast_advised_economy_latch_s = max(
        float(settings.forecast_advised_economy_latch_s),
        0.0,
    )
    provider.forecast_advised_economy_current_ws_range_ms = max(
        float(settings.forecast_advised_economy_current_ws_range_ms),
        0.0,
    )
    provider.forecast_advised_economy_current_dir_shift_deg = max(
        float(settings.forecast_advised_economy_current_dir_shift_deg),
        0.0,
    )
    provider.forecast_advised_economy_current_history_s = max(
        float(settings.forecast_advised_economy_current_history_s),
        float(provider.update_interval_s),
    )
    provider.forecast_advised_economy_mild_enabled = bool(
        settings.forecast_advised_economy_mild_enabled
    )
    provider.forecast_advised_economy_mild_min_headroom_deg = max(
        float(settings.forecast_advised_economy_mild_min_headroom_deg),
        0.0,
    )
    provider.forecast_advised_economy_mild_rise_norm = max(
        float(settings.forecast_advised_economy_mild_rise_norm),
        0.0,
    )
    provider.forecast_advised_economy_mild_max_speed_range_ms = max(
        float(settings.forecast_advised_economy_mild_max_speed_range_ms),
        0.0,
    )
    provider.forecast_advised_economy_mild_restart_err_kg = max(
        float(settings.forecast_advised_economy_mild_restart_err_kg),
        0.0,
    )
    provider.forecast_advised_economy_high_stable_clean_required = bool(
        settings.forecast_advised_economy_high_stable_clean_required
    )
    provider.forecast_advised_economy_high_stable_early_min_ms = max(
        float(settings.forecast_advised_economy_high_stable_early_min_ms),
        0.0,
    )
    provider.forecast_advised_economy_high_stable_near_min_ms = max(
        float(settings.forecast_advised_economy_high_stable_near_min_ms),
        0.0,
    )
    provider.forecast_advised_economy_high_stable_far_min_ms = max(
        float(settings.forecast_advised_economy_high_stable_far_min_ms),
        0.0,
    )
    provider.forecast_advised_economy_high_stable_near_range_max_ms = max(
        float(settings.forecast_advised_economy_high_stable_near_range_max_ms),
        0.0,
    )
    provider.forecast_advised_economy_high_stable_far_range_max_ms = max(
        float(settings.forecast_advised_economy_high_stable_far_range_max_ms),
        0.0,
    )
    provider.forecast_advised_economy_high_stable_dir_shift_max_deg = max(
        float(settings.forecast_advised_economy_high_stable_dir_shift_max_deg),
        0.0,
    )
    provider.forecast_advised_economy_min_peak_drop_ms = max(
        float(settings.forecast_advised_economy_min_peak_drop_ms),
        0.0,
    )
    provider.forecast_advised_economy_hold_current_on_candidate = bool(
        settings.forecast_advised_economy_hold_current_on_candidate
    )
    provider.forecast_advised_economy_suppression_only = bool(
        settings.forecast_advised_economy_suppression_only
    )
    provider.forecast_advised_economy_suppression_restart_err_kg = max(
        float(settings.forecast_advised_economy_suppression_restart_err_kg),
        0.0,
    )
    provider.forecast_advised_economy_runaway_release_enabled = bool(
        settings.forecast_advised_economy_runaway_release_enabled
    )
    provider.forecast_advised_economy_runaway_pump_rate_m3_min = max(
        float(settings.forecast_advised_economy_runaway_pump_rate_m3_min),
        0.0,
    )
    provider.forecast_advised_economy_runaway_backlog_kg = max(
        float(settings.forecast_advised_economy_runaway_backlog_kg),
        0.0,
    )
    provider.forecast_advised_economy_runaway_cooldown_s = max(
        float(settings.forecast_advised_economy_runaway_cooldown_s),
        0.0,
    )
    provider.forecast_advised_economy_shadow_opportunity_enabled = bool(
        settings.forecast_advised_economy_shadow_opportunity_enabled
    )
    provider.forecast_advised_economy_shadow_min_latch_delta = max(
        int(settings.forecast_advised_economy_shadow_min_latch_delta),
        0,
    )
    provider.forecast_advised_economy_shadow_min_target_motion_kg_s = max(
        float(settings.forecast_advised_economy_shadow_min_target_motion_kg_s),
        0.0,
    )
    provider.forecast_advised_economy_shadow_min_pump_rate_m3_min = max(
        float(settings.forecast_advised_economy_shadow_min_pump_rate_m3_min),
        0.0,
    )
    provider.forecast_advised_economy_shadow_min_backlog_kg = max(
        float(settings.forecast_advised_economy_shadow_min_backlog_kg),
        0.0,
    )
    provider.forecast_advised_economy_shadow_latch_s = max(
        float(settings.forecast_advised_economy_shadow_latch_s),
        0.0,
    )
    provider.forecast_pause_resume_max_target_err_kg = max(
        float(settings.forecast_pause_resume_max_target_err_kg),
        0.0,
    )
    provider.sustained_active_recompute_enabled = bool(
        settings.sustained_active_recompute_enabled
    )
    provider.sustained_active_recompute_min_active_buckets = max(
        int(settings.sustained_active_recompute_min_active_buckets),
        1,
    )
    provider.sustained_active_recompute_pitch_deg = max(
        float(settings.sustained_active_recompute_pitch_deg),
        0.0,
    )
    provider.sustained_active_recompute_roll_deg = max(
        float(settings.sustained_active_recompute_roll_deg),
        0.0,
    )
    provider.sustained_active_recompute_target_err_kg = max(
        float(settings.sustained_active_recompute_target_err_kg),
        0.0,
    )
    provider.sustained_active_recompute_pump_rate_m3_min = max(
        float(settings.sustained_active_recompute_pump_rate_m3_min),
        0.0,
    )
    provider.sustained_active_recompute_min_target_age_s = max(
        float(settings.sustained_active_recompute_min_target_age_s),
        0.0,
    )
    if settings.stale_active_target_refresh_mode not in (
        "off",
        "refresh_only_v1",
        "refresh_only_guarded_v1",
    ):
        raise ValueError(
            "unsupported stale_active_target_refresh_mode="
            f"{settings.stale_active_target_refresh_mode!r}"
        )
    provider.stale_active_target_refresh_mode = str(settings.stale_active_target_refresh_mode)
    provider.stale_active_target_refresh_min_active_buckets = max(
        int(settings.stale_active_target_refresh_min_active_buckets),
        1,
    )
    provider.stale_active_target_refresh_pitch_deg = max(
        float(settings.stale_active_target_refresh_pitch_deg),
        0.0,
    )
    provider.stale_active_target_refresh_roll_deg = max(
        float(settings.stale_active_target_refresh_roll_deg),
        0.0,
    )
    provider.stale_active_target_refresh_target_err_kg = max(
        float(settings.stale_active_target_refresh_target_err_kg),
        0.0,
    )
    provider.stale_active_target_refresh_pump_rate_m3_min = max(
        float(settings.stale_active_target_refresh_pump_rate_m3_min),
        0.0,
    )
    provider.stale_active_target_refresh_min_target_age_s = max(
        float(settings.stale_active_target_refresh_min_target_age_s),
        0.0,
    )
    provider.stale_active_target_refresh_min_delta_kg = max(
        float(settings.stale_active_target_refresh_min_delta_kg),
        0.0,
    )
    provider.stale_active_target_refresh_cooldown_s = max(
        float(settings.stale_active_target_refresh_cooldown_s),
        0.0,
    )
    provider.stale_active_target_refresh_no_recovery_eps_deg = max(
        float(settings.stale_active_target_refresh_no_recovery_eps_deg),
        0.0,
    )
    if settings.reactive_floor_predictive_veto not in (
        "off",
        "on",
        "theta_only_gated",
        "relaxed_veto_on_theta_only",
    ):
        raise ValueError(
            "unsupported reactive_floor_predictive_veto="
            f"{settings.reactive_floor_predictive_veto!r}"
        )
    if settings.high_posture_metric not in ("max_axis", "hybrid"):
        raise ValueError(f"unsupported high_posture_metric={settings.high_posture_metric!r}")
    if settings.reactive_floor_action not in ("active_small", "active_medium"):
        raise ValueError(f"unsupported reactive_floor_action={settings.reactive_floor_action!r}")
    provider.reactive_floor_predictive_veto_mode = str(settings.reactive_floor_predictive_veto)
    provider.reactive_floor_predictive_veto_enabled = (
        provider.reactive_floor_predictive_veto_mode != "off"
    )
    provider.high_posture_metric = str(settings.high_posture_metric)
    provider.high_posture_enter = max(float(settings.high_posture_enter), 0.0)
    provider.high_posture_exit = max(float(settings.high_posture_exit), 0.0)
    provider.high_posture_min_duration_s = max(float(settings.high_posture_min_duration_s), 0.0)
    provider.reactive_floor_theta_only_enter = max(
        float(settings.reactive_floor_theta_only_enter),
        0.0,
    )
    provider.reactive_floor_theta_only_min_duration_s = max(
        float(settings.reactive_floor_theta_only_min_duration_s),
        0.0,
    )
    provider.reactive_floor_action = str(settings.reactive_floor_action)
    provider.reactive_floor_medium_delay_s = max(
        float(settings.reactive_floor_medium_delay_s),
        0.0,
    )
    if settings.reactive_floor_post_exit_mode not in ("off", "early_stop", "target_cap"):
        raise ValueError(
            "unsupported reactive_floor_post_exit_mode="
            f"{settings.reactive_floor_post_exit_mode!r}"
        )
    provider.reactive_floor_post_exit_mode = str(settings.reactive_floor_post_exit_mode)
    provider.reactive_floor_post_exit_enabled = (
        provider.reactive_floor_post_exit_mode != "off"
    )
    provider.reactive_floor_post_exit_target_cap_kg = max(
        float(settings.reactive_floor_post_exit_target_cap_kg),
        0.0,
    )
    provider.reactive_floor_post_exit_margin_deg = max(
        float(settings.reactive_floor_post_exit_margin_deg),
        0.0,
    )
    provider.reactive_floor_post_exit_worsening_eps_deg = max(
        float(settings.reactive_floor_post_exit_worsening_eps_deg),
        0.0,
    )
    provider.oracle_preemptive_prevent_enabled = bool(
        settings.oracle_preemptive_prevent_enabled
    )
    provider.oracle_preemptive_enter_deg = max(
        float(settings.oracle_preemptive_enter_deg),
        0.0,
    )
    provider.oracle_preemptive_floor_deg = max(
        float(settings.oracle_preemptive_floor_deg),
        provider.oracle_preemptive_enter_deg,
    )
    provider.oracle_preemptive_min_future_norm = max(
        float(settings.oracle_preemptive_min_future_norm),
        0.0,
    )
    provider.oracle_preemptive_rise_margin_norm = max(
        float(settings.oracle_preemptive_rise_margin_norm),
        0.0,
    )
    provider.oracle_preemptive_min_delta_kg = max(
        float(settings.oracle_preemptive_min_delta_kg),
        0.0,
    )
    provider.oracle_preemptive_cooldown_s = max(
        float(settings.oracle_preemptive_cooldown_s),
        0.0,
    )
    provider.h120_oracle_probe_enabled = bool(settings.h120_oracle_probe_enabled)
    provider.h120_oracle_probe_short_delay_s = max(
        float(settings.h120_oracle_probe_short_delay_s),
        0.0,
    )
    provider.h120_oracle_probe_far_high_norm = max(
        float(settings.h120_oracle_probe_far_high_norm),
        0.0,
    )
    provider.h120_oracle_probe_intensify_margin_norm = max(
        float(settings.h120_oracle_probe_intensify_margin_norm),
        0.0,
    )
    provider.h120_oracle_probe_preemptive_enter_deg = max(
        float(settings.h120_oracle_probe_preemptive_enter_deg),
        0.0,
    )
    provider.h120_oracle_probe_preemptive_floor_deg = max(
        float(settings.h120_oracle_probe_preemptive_floor_deg),
        provider.h120_oracle_probe_preemptive_enter_deg,
    )
    provider.h120_oracle_probe_min_delta_kg = max(
        float(settings.h120_oracle_probe_min_delta_kg),
        0.0,
    )
    provider.h120_oracle_probe_cooldown_s = max(
        float(settings.h120_oracle_probe_cooldown_s),
        0.0,
    )
    # h120 learned risk scheduler (default-off). Uses the 60-120min
    # far-horizon pressure blocks only to (a) shorten the delayed-medium
    # delay and (b) suppress early_stop at most once per floor episode.
    # It NEVER decides whether high posture recovers (that stays with the
    # reactive floor), NEVER refreshes the primary target preemptively,
    # and NEVER suppresses recovery via relief prediction.
    provider.h120_risk_scheduler_enabled = bool(settings.h120_risk_scheduler_enabled)
    provider.h120_scheduler_delay_normal_s = max(float(settings.h120_scheduler_delay_normal_s), 0.0)
    provider.h120_scheduler_delay_risk_aware_s = max(float(settings.h120_scheduler_delay_risk_aware_s), 0.0)
    provider.h120_scheduler_delay_high_risk_s = max(float(settings.h120_scheduler_delay_high_risk_s), 0.0)
    provider.h120_scheduler_far_high_norm = max(float(settings.h120_scheduler_far_high_norm), 0.0)
    provider.h120_scheduler_intensify_margin_norm = max(float(settings.h120_scheduler_intensify_margin_norm), 0.0)
    provider.h120_scheduler_relief_margin_norm = max(float(settings.h120_scheduler_relief_margin_norm), 0.0)
    provider.h120_scheduler_early_stop_suppress_max_per_episode = max(
        int(settings.h120_scheduler_early_stop_suppress_max_per_episode), 0
    )
    provider.h120_scheduler_prefloor_probe_enabled = bool(
        settings.h120_scheduler_prefloor_probe_enabled
    )
    if settings.h120_scheduler_prefloor_action not in ("pump_saving", "active_small"):
        raise ValueError(
            "unsupported h120_scheduler_prefloor_action="
            f"{settings.h120_scheduler_prefloor_action!r}"
        )
    provider.h120_scheduler_prefloor_action = str(settings.h120_scheduler_prefloor_action)
    provider.h120_scheduler_prefloor_enter_deg = max(
        float(settings.h120_scheduler_prefloor_enter_deg), 0.0
    )
    provider.h120_scheduler_prefloor_floor_deg = max(
        float(settings.h120_scheduler_prefloor_floor_deg),
        provider.h120_scheduler_prefloor_enter_deg,
    )
    provider.h120_scheduler_prefloor_min_delta_kg = max(
        float(settings.h120_scheduler_prefloor_min_delta_kg), 0.0
    )
    provider.h120_scheduler_prefloor_max_delta_kg = max(
        float(settings.h120_scheduler_prefloor_max_delta_kg), 0.0
    )
    provider.h120_scheduler_prefloor_cooldown_s = max(
        float(settings.h120_scheduler_prefloor_cooldown_s), 0.0
    )
    provider.h120_axis_micro_enabled = bool(settings.h120_axis_micro_enabled)
    provider.h120_axis_micro_enter_deg = max(float(settings.h120_axis_micro_enter_deg), 0.0)
    provider.h120_axis_micro_floor_deg = max(
        float(settings.h120_axis_micro_floor_deg), provider.h120_axis_micro_enter_deg
    )
    provider.h120_axis_micro_rise_floor_norm = max(
        float(settings.h120_axis_micro_rise_floor_norm), 0.0
    )
    provider.h120_axis_micro_worsening_eps_deg = max(
        float(settings.h120_axis_micro_worsening_eps_deg), 0.0
    )
    provider.h120_axis_micro_delta_kg = max(float(settings.h120_axis_micro_delta_kg), 0.0)
    provider.h120_axis_micro_cooldown_s = max(float(settings.h120_axis_micro_cooldown_s), 0.0)
    if settings.h120_floor_shaping_mode not in (
        "off",
        "axis_shaped",
        "bounded_medium",
        "axis_shaped_bounded_medium",
        "stale_repair",
    ):
        raise ValueError(
            f"unsupported h120_floor_shaping_mode={settings.h120_floor_shaping_mode!r}"
        )
    provider.h120_floor_shaping_mode = str(settings.h120_floor_shaping_mode)
    provider.h120_floor_shaping_enabled = provider.h120_floor_shaping_mode != "off"
    provider.h120_floor_shaping_far_high_norm = max(
        float(settings.h120_floor_shaping_far_high_norm), 0.0
    )
    provider.h120_floor_shaping_slow_response_eps_deg = max(
        float(settings.h120_floor_shaping_slow_response_eps_deg), 0.0
    )
    provider.h120_floor_shaping_medium_delay_s = max(
        float(settings.h120_floor_shaping_medium_delay_s), 0.0
    )
    provider.h120_pareto_mode_selector_enabled = bool(
        settings.h120_pareto_mode_selector_enabled
    )
    provider.h120_pareto_mode_selector_far_high_norm = max(
        float(settings.h120_pareto_mode_selector_far_high_norm), 0.0
    )
    provider.h120_pareto_mode_selector_intensify_margin_norm = max(
        float(settings.h120_pareto_mode_selector_intensify_margin_norm), 0.0
    )
    provider.h120_pareto_mode_selector_relief_margin_norm = max(
        float(settings.h120_pareto_mode_selector_relief_margin_norm), 0.0
    )
    provider.h120_pareto_mode_selector_slow_response_eps_deg = max(
        float(settings.h120_pareto_mode_selector_slow_response_eps_deg), 0.0
    )
    provider.relief_economy_enabled = bool(settings.relief_economy_enabled)
    if settings.relief_economy_horizon not in ("near", "far"):
        raise ValueError(f"unsupported relief_economy_horizon={settings.relief_economy_horizon!r}")
    provider.relief_economy_horizon = str(settings.relief_economy_horizon)
    provider.relief_economy_safe_deg = max(float(settings.relief_economy_safe_deg), 0.0)
    provider.relief_economy_worsening_eps_deg = max(
        float(settings.relief_economy_worsening_eps_deg), 0.0
    )
    provider.relief_economy_low_norm = max(float(settings.relief_economy_low_norm), 0.0)
    provider.relief_economy_high_norm = max(float(settings.relief_economy_high_norm), 0.0)
    provider.relief_envelope_enabled = bool(settings.relief_envelope_enabled)
    if settings.relief_envelope_horizon not in ("near", "far"):
        raise ValueError(
            f"unsupported relief_envelope_horizon={settings.relief_envelope_horizon!r}"
        )
    provider.relief_envelope_horizon = str(settings.relief_envelope_horizon)
    provider.relief_envelope_allowed_deg = (
        max(float(settings.relief_envelope_allowed_0_20_deg), 0.0),
        max(float(settings.relief_envelope_allowed_20_40_deg), 0.0),
        max(float(settings.relief_envelope_allowed_40_60_deg), 0.0),
        max(float(settings.relief_envelope_allowed_60_120_deg), 0.0),
    )
    provider.relief_envelope_max_duration_s = max(
        float(settings.relief_envelope_max_duration_s), 0.0
    )
    provider.relief_envelope_debt_budget_deg_s = max(
        float(settings.relief_envelope_debt_budget_deg_s), 0.0
    )
    provider.relief_envelope_near_limit_deg = max(
        float(settings.relief_envelope_near_limit_deg), 0.0
    )
    provider.relief_envelope_near_limit_budget_s = max(
        float(settings.relief_envelope_near_limit_budget_s), 0.0
    )
    provider.relief_envelope_worsening_eps_deg = max(
        float(settings.relief_envelope_worsening_eps_deg), 0.0
    )
    provider.relief_envelope_low_norm = max(float(settings.relief_envelope_low_norm), 0.0)
    provider.relief_envelope_high_norm = max(float(settings.relief_envelope_high_norm), 0.0)
    provider.relief_envelope_reintensify_norm = max(
        float(settings.relief_envelope_reintensify_norm), 0.0
    )
    provider.relief_envelope_single_axis_only = bool(settings.relief_envelope_single_axis_only)
    provider.relief_envelope_axis_minor_ratio_max = max(
        float(settings.relief_envelope_axis_minor_ratio_max), 0.0
    )
    provider.relief_envelope_axis_minor_abs_max_deg = max(
        float(settings.relief_envelope_axis_minor_abs_max_deg), 0.0
    )
    provider.relief_envelope_adaptive_axis_cap = bool(
        settings.relief_envelope_adaptive_axis_cap
    )
    provider.relief_envelope_cap_relief_norm = max(
        float(settings.relief_envelope_cap_relief_norm), 0.0
    )
    provider.relief_envelope_cap_hold_fraction = min(
        max(float(settings.relief_envelope_cap_hold_fraction), 0.0),
        1.0,
    )
    provider.economy_pump_budget_enabled = bool(settings.economy_pump_budget_enabled)
    provider.economy_pump_budget_m3 = max(float(settings.economy_pump_budget_m3), 0.0)
    provider.economy_pump_budget_hold_refresh_fraction = min(
        max(float(settings.economy_pump_budget_hold_refresh_fraction), 0.0),
        1.0,
    )
    provider.economy_pump_budget_dynamic_refresh_enabled = bool(
        settings.economy_pump_budget_dynamic_refresh_enabled
    )
    provider.economy_pump_budget_dynamic_refresh_enter_deg = max(
        float(settings.economy_pump_budget_dynamic_refresh_enter_deg), 0.0
    )
    provider.economy_pump_budget_dynamic_refresh_full_deg = max(
        float(settings.economy_pump_budget_dynamic_refresh_full_deg),
        provider.economy_pump_budget_dynamic_refresh_enter_deg + 1e-6,
    )
    provider.economy_pump_budget_dynamic_refresh_max_fraction = min(
        max(float(settings.economy_pump_budget_dynamic_refresh_max_fraction), 0.0),
        1.0,
    )
    if settings.economy_pump_budget_allocator_mode not in (
        "budget",
        "strict_relief",
        "broad_safe",
        "posture_only",
        "opportunity",
        "relief_decay_auto",
        "relief_decay_auto_v2",
        "relief_decay_episode_auto_v1",
        "residual_high_auto",
        "residual_high_plateau_auto",
        "residual_high_plateau_strict_auto",
        "neutral_mhs_broader_auto",
        "neutral_mhs_clean_auto",
        "psc_4hao_regime_addback_v2_auto",
        "dual_specialist_auto",
        "gusty_oscillation_budget_auto",
        "gusty_oscillation_soft_cap_auto",
        "direction_reversal_avoidance_auto",
        "direction_reversal_layered_auto",
        "direction_reversal_layered_v5_auto",
        "direction_reversal_layered_v6_auto",
        "direction_reversal_layered_v7_auto",
        "direction_reversal_layered_v8_auto",
        "regime_auto",
        "regime_auto_failclosed",
    ):
        raise ValueError(
            "economy_pump_budget_allocator_mode must be one of "
            "budget, strict_relief, broad_safe, posture_only, opportunity, "
            "relief_decay_auto, relief_decay_auto_v2, relief_decay_episode_auto_v1, "
            "residual_high_auto, residual_high_plateau_auto, "
            "residual_high_plateau_strict_auto, neutral_mhs_broader_auto, "
            "neutral_mhs_clean_auto, psc_4hao_regime_addback_v2_auto, "
            "dual_specialist_auto, "
            "gusty_oscillation_budget_auto, "
            "gusty_oscillation_soft_cap_auto, direction_reversal_avoidance_auto, "
            "direction_reversal_layered_auto, direction_reversal_layered_v5_auto, "
            "direction_reversal_layered_v6_auto, direction_reversal_layered_v7_auto, "
            "direction_reversal_layered_v8_auto, "
            "regime_auto, regime_auto_failclosed"
        )
    provider.economy_pump_budget_allocator_mode = str(
        settings.economy_pump_budget_allocator_mode
    )
    provider.economy_pump_budget_require_relief = bool(settings.economy_pump_budget_require_relief)
    provider.economy_pump_budget_relief_norm = max(
        float(settings.economy_pump_budget_relief_norm),
        0.0,
    )
    provider.economy_pump_budget_relief_drop_norm = max(
        float(settings.economy_pump_budget_relief_drop_norm),
        0.0,
    )
    provider.economy_pump_budget_forecast_smart = bool(settings.economy_pump_budget_forecast_smart)
    provider.economy_pump_budget_smart_posture_deg = max(
        float(settings.economy_pump_budget_smart_posture_deg),
        0.0,
    )
    provider.economy_pump_budget_smart_pressure_norm = max(
        float(settings.economy_pump_budget_smart_pressure_norm),
        0.0,
    )
    provider.economy_pump_budget_opportunity_window_s = max(
        float(settings.economy_pump_budget_opportunity_window_s),
        0.0,
    )
    provider.economy_pump_budget_opportunity_min_pump_m3 = max(
        float(settings.economy_pump_budget_opportunity_min_pump_m3),
        0.0,
    )
    provider.economy_pump_budget_opportunity_min_refresh = max(
        int(settings.economy_pump_budget_opportunity_min_refresh),
        0,
    )
    provider.economy_pump_budget_opportunity_min_delta_kg = max(
        float(settings.economy_pump_budget_opportunity_min_delta_kg),
        0.0,
    )
    provider.hold_relief_debt_enabled = bool(settings.hold_relief_debt_enabled)
    provider.hold_relief_debt_threshold_deg = max(
        float(settings.hold_relief_debt_threshold_deg),
        0.0,
    )
    provider.hold_relief_debt_response_eps_deg = max(
        float(settings.hold_relief_debt_response_eps_deg),
        0.0,
    )
    provider.hold_relief_debt_increment = max(float(settings.hold_relief_debt_increment), 0.0)
    provider.hold_relief_debt_max = max(float(settings.hold_relief_debt_max), 0.0)
    provider.hold_relief_debt_lowrisk_pressure_norm = max(
        float(settings.hold_relief_debt_lowrisk_pressure_norm),
        0.0,
    )
    provider.hold_relief_debt_lowrisk_safe_deg = max(
        float(settings.hold_relief_debt_lowrisk_safe_deg),
        0.0,
    )
    if settings.forced_prefix_mode not in ("off", "raw_action", "final_action", "target_update"):
        raise ValueError(f"unsupported forced_prefix_mode={settings.forced_prefix_mode!r}")
    provider.forced_prefix_mode = str(settings.forced_prefix_mode)
    provider.forced_prefix_actions: dict[int, dict[str, Any]] = {
        int(k): dict(v)
        for k, v in (settings.forced_prefix_actions or {}).items()
    }
