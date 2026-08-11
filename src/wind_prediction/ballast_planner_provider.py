"""Adapter that exposes the A1.4 ballast planner as a preview_trim_provider.

It plugs into archive/legacy_fowt_control/controllers_extras.py::ClosedLoopPolicy
through its `preview_trim_provider` slot. On every 10-min replay bucket it
runs the full 125-sequence rolling planner and emits the *first* block's
action as a (pitch_bias_deg, roll_bias_deg) setpoint bias.

Two bias shapes are supported:
  - "hold": bias = first_avec, held flat until the next bucket. Original A2
    behaviour. Produces a DC offset under sustained-wind windows.
  - "event_decay": when the planner's first_action changes between buckets a
    fresh impulse is started at the new target; otherwise the bias decays
    exponentially with time-constant `decay_tau_s` so a constant-action
    sustained window contributes only a single transient. Defaults
    `decay_tau_s = 300 s` (≈ 5 min half-life of e^-1) — see Lemmer & Cheng
    2017 on transient preview FF for FOWT.

Only this file changes; planner core and the closed-loop are untouched.
"""
from __future__ import annotations

import math
import os
from datetime import datetime, timedelta
from functools import cmp_to_key
from typing import Any

import numpy as np

from .ballast_planner import (
    ACTIONS,
    PlannerConfig,
    _ALL_SEQUENCES,
    action_vec,
    apply_posture_state_to_blocks,
    compare_sequences,
    compute_pressure_blocks,
    economic_scalar_cost,
    evaluate_sequence,
    norm_term,
    planner_action_vec,
    pressure_proxy_vec,
    select_best_economic,
    tank_signal,
)
from .forecast_adapter import ForecastModelAdapter
from .forecast_contract import (
    ForecastContract,
    forecast_from_oracle_sample,
    validate_forecast_result,
)
from .control_telemetry_defaults import (
    frozen_active_intent_record_defaults,
    frozen_hold_comfort_preview_defaults,
    frozen_hold_comfort_record_defaults,
    frozen_overlay_preview_defaults,
    frozen_overlay_record_defaults,
    frozen_pump_suppression_preview_defaults,
    frozen_pump_suppression_record_defaults,
    frozen_recovery_mode_record_defaults,
)
from .replay_dataset import Fino1ReplayDataset
from .safety_supervisor import (
    apply_reactive_floor_state_defaults,
    posture_vector_deg,
    reactive_floor_log_fields,
    reactive_floor_posture_metrics,
    reactive_floor_theta_gate_decision,
    reactive_floor_veto_decision,
)
from .target_lifecycle import (
    apply_primary_target_transition,
    apply_primary_target_state_defaults,
    consume_primary_refresh_owner,
    mark_primary_target_reused,
    plant_primary_masses,
    primary_target_at_current,
    primary_target_capped_to_current,
    primary_target_from_delta,
    primary_target_from_paused,
    primary_target_proposal_metrics,
    remember_paused_primary_target,
)


def complete_event_probabilities_available(
    event_probs: dict[str, float] | None,
    expected_columns,
) -> bool:
    expected = tuple(str(name) for name in expected_columns)
    if not expected or not isinstance(event_probs, dict):
        return False
    try:
        values = [float(event_probs[name]) for name in expected]
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values))) and all(
        0.0 <= value <= 1.0 for value in values
    )


_EVENT_RISK_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)
_HOLD_RELEASE_ENTER_PITCH_DEG = 2.5
_HOLD_RELEASE_ENTER_ROLL_DEG = 2.5
_HOLD_FORECAST_POSTURE_RELIEF_NORM = 0.20
_TRUSTED_EVENT_ATTENTION_KEYS = (
    "ballast_attention_event",
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)
_TRUSTED_EVENT_DYNAMIC_KEYS = (
    "speed_ramp_ge_3ms",
    "direction_shift_ge_45deg",
    "vector_change_ge_train_p90",
)
_TRUSTED_EVENT_HIGHWIND_KEY = "future_speed_ge_train_p95"
_FORECAST_RELIEF_MARGIN_NORM = 0.25
_FORECAST_REINTENSIFY_HIGH_NORM = 1.05
_FORECAST_REINTENSIFY_RISE_NORM = 0.10


class BallastPlannerPreviewProvider:
    """A1.4 planner exposed as a preview_trim_provider."""

    def __init__(
        self,
        replay_dataset: Fino1ReplayDataset,
        start_timestamp: datetime,
        cfg: PlannerConfig,
        block_discounts: list[float],
        plant_info: dict[str, Any] | None = None,
        scale: float = 1.0,
        bias_shape: str = "hold",
        decay_tau_s: float = 300.0,
        objective_mode: str = "lex",
        event_reset_mode: str = "action",
        target_reset_tol_deg: float = 0.02,
        forecast_adapter: ForecastModelAdapter | None = None,
        setpoint_bias_sign: float = 1.0,
        ff_channel_enabled: bool = False,
        ff_scale: float = 1.0,
        setpoint_channel_enabled: bool = True,
        prediction_primary_enabled: bool = False,
        prediction_primary_scale: float = 1.0,
        prediction_primary_disable_setpoint: bool = True,
        primary_hold_target_mode: str = "current",
        trusted_event_gate_enabled: bool = False,
        trusted_event_highwind_threshold: float = 0.90,
        trusted_event_attention_threshold: float = 0.75,
        trusted_event_attention_min_heads: int = 2,
        trusted_event_dynamic_enabled: bool = False,
        trusted_event_dynamic_threshold: float = 0.85,
        trusted_event_dynamic_min_heads: int = 2,
        relief_medium_cap_enabled: bool = False,
        relief_medium_cap_event_threshold: float = 0.70,
        relief_medium_cap_ratio: float = 0.25,
        relief_medium_cap_adaptive: bool = False,
        relief_medium_cap_pitch_abs_max_deg: float = 2.0,
        relief_medium_cap_roll_abs_max_deg: float = 1.5,
        active_posture_refresh_enabled: bool = False,
        active_posture_refresh_enter_deg: float = 4.0,
        active_posture_refresh_exit_deg: float = 3.2,
        active_posture_refresh_update_interval_s: float = 120.0,
        active_posture_refresh_action_name: str = "active_small",
        active_posture_refresh_pressure_max_norm: float = 0.0,
        active_posture_refresh_min_target_age_s: float = 0.0,
        active_posture_refresh_axis_mode: str = "max",
        active_posture_refresh_worsening_eps_deg_s: float = 0.0,
        active_posture_refresh_debt_deg: float = 0.0,
        active_posture_refresh_include_hold: bool = False,
        fallback_risk_active_release_enabled: bool = False,
        fallback_risk_active_release_enter_deg: float = 5.5,
        fallback_risk_active_release_exit_deg: float = 4.8,
        fallback_risk_active_release_worsening_eps_deg_s: float = 0.05,
        fallback_risk_active_release_pump_rate_m3_min: float = 10.0,
        fallback_risk_active_release_backlog_kg: float = 500.0,
        no_preview_myopic_horizon_enabled: bool = False,
        primary_stall_refresh_enabled: bool = False,
        primary_stall_pitch_deg: float = 5.0,
        primary_stall_roll_deg: float = 4.0,
        primary_stall_target_err_kg: float = 1500.0,
        primary_stall_pump_rate_m3_min: float = 0.5,
        primary_stall_min_age_s: float = 300.0,
        primary_stall_pitch_axis_bias: bool = False,
        primary_stall_pitch_axis_roll_max_deg: float = 2.0,
        active_effectiveness_refresh_enabled: bool = False,
        active_effectiveness_refresh_min_active_buckets: int = 2,
        active_effectiveness_refresh_pitch_deg: float = 3.0,
        active_effectiveness_refresh_roll_deg: float = 3.0,
        active_effectiveness_refresh_target_err_kg: float = 1500.0,
        active_effectiveness_refresh_pump_rate_m3_min: float = 0.5,
        active_effectiveness_refresh_min_target_age_s: float = 0.0,
        active_effectiveness_refresh_min_pressure_norm: float = 0.0,
        active_effectiveness_refresh_response_eps_deg: float = 0.3,
        active_effectiveness_refresh_required_worsening_deg: float = 0.0,
        active_effectiveness_refresh_min_delta_kg: float = 1500.0,
        active_effectiveness_refresh_episode_budget_m3: float = 180.0,
        active_effectiveness_refresh_lowrisk_pressure_norm: float = 0.5,
        gusty_oscillation_enabled: bool = False,
        gusty_oscillation_min_target_age_s: float = 600.0,
        gusty_oscillation_target_scale: float = 1.0,
        gusty_oscillation_headroom_scaled: bool = False,
        gusty_oscillation_low_posture_deg: float = 1.5,
        gusty_oscillation_high_posture_deg: float = 3.2,
        gusty_oscillation_low_posture_scale: float = 0.55,
        gusty_oscillation_high_posture_scale: float = 0.90,
        gusty_oscillation_short_gate_enabled: bool = False,
        gusty_oscillation_short_min_range_ms: float = 0.75,
        gusty_oscillation_short_min_pressure_norm: float = 0.75,
        gusty_oscillation_short_max_dir_shift_deg: float = 45.0,
        gusty_oscillation_short_min_turns: int = 1,
        gusty_hold_current_gate_mode: str = "off",
        gusty_hold_current_isolated_primary_enabled: bool = False,
        gusty_hold_current_runtime_release_enabled: bool = False,
        gusty_hold_current_runtime_release_deg: float = 4.2,
        gusty_hold_current_runtime_release_cooldown_s: float = 300.0,
        gusty_hold_current_debt_guard_enabled: bool = False,
        gusty_hold_current_debt_guard_target_motion_kg_s: float = 220.0,
        gusty_hold_current_debt_guard_target_err_kg: float = 1200.0,
        gusty_hold_current_debt_guard_min_age_s: float = 600.0,
        gusty_hold_current_day_budget_enabled: bool = False,
        gusty_hold_current_day_budget_max_buckets: int = 0,
        gusty_hold_current_day_budget_window_s: float = 86400.0,
        direction_reversal_hold_current_gate_mode: str = "off",
        forecast_control_trust_gate_enabled: bool = False,
        forecast_control_trust_event_threshold: float = 0.60,
        forecast_control_trust_current_stability_required: bool = False,
        forecast_control_trust_max_speed_range_ms: float = 1.0,
        forecast_control_trust_max_dir_shift_deg: float = 10.0,
        forecast_control_trust_suppression_rise_norm: float = 0.0,
        forecast_control_trust_min_pressure_norm: float = 1.0,
        forecast_pressure_trust_gate_enabled: bool = False,
        forecast_pressure_trust_current_norm_max: float = 0.65,
        forecast_pressure_trust_future_norm_min: float = 1.00,
        forecast_pressure_trust_event_threshold: float = 0.75,
        forecast_pressure_trust_untrusted_scale: float = 0.55,
        forecast_pressure_trust_min_direction_dot: float = 0.0,
        forecast_pressure_trust_spike_margin_norm: float = 0.35,
        forecast_advised_economy_enabled: bool = False,
        forecast_advised_economy_min_headroom_deg: float = 0.35,
        forecast_advised_economy_rise_norm: float = 0.20,
        forecast_advised_economy_max_speed_range_ms: float = 0.0,
        forecast_advised_economy_max_dir_shift_deg: float = 0.0,
        forecast_advised_economy_confirm_buckets: int = 1,
        forecast_advised_economy_hold_confirm_buckets: int = 0,
        forecast_advised_economy_min_axis_deg: float = 0.0,
        forecast_advised_economy_min_pressure_norm: float = 0.0,
        forecast_advised_economy_posture_release_deg: float = 0.0,
        forecast_advised_economy_release_cooldown_s: float = 0.0,
        forecast_advised_economy_latch_s: float = 0.0,
        forecast_advised_economy_current_ws_range_ms: float = 0.0,
        forecast_advised_economy_current_dir_shift_deg: float = 0.0,
        forecast_advised_economy_current_history_s: float = 1800.0,
        forecast_advised_economy_mild_enabled: bool = False,
        forecast_advised_economy_mild_min_headroom_deg: float = 0.10,
        forecast_advised_economy_mild_rise_norm: float = 0.35,
        forecast_advised_economy_mild_max_speed_range_ms: float = 0.0,
        forecast_advised_economy_mild_restart_err_kg: float = 900.0,
        forecast_advised_economy_high_stable_clean_required: bool = False,
        forecast_advised_economy_high_stable_early_min_ms: float = 14.0,
        forecast_advised_economy_high_stable_near_min_ms: float = 14.0,
        forecast_advised_economy_high_stable_far_min_ms: float = 11.5,
        forecast_advised_economy_high_stable_near_range_max_ms: float = 3.0,
        forecast_advised_economy_high_stable_far_range_max_ms: float = 3.0,
        forecast_advised_economy_high_stable_dir_shift_max_deg: float = 20.0,
        forecast_advised_economy_min_peak_drop_ms: float = 0.0,
        forecast_advised_economy_hold_current_on_candidate: bool = False,
        forecast_advised_economy_suppression_only: bool = False,
        forecast_advised_economy_suppression_restart_err_kg: float = 2000.0,
        forecast_advised_economy_runaway_release_enabled: bool = False,
        forecast_advised_economy_runaway_pump_rate_m3_min: float = 8.0,
        forecast_advised_economy_runaway_backlog_kg: float = 1200.0,
        forecast_advised_economy_runaway_cooldown_s: float = 1200.0,
        forecast_advised_economy_shadow_opportunity_enabled: bool = False,
        forecast_advised_economy_shadow_min_latch_delta: int = 30,
        forecast_advised_economy_shadow_min_target_motion_kg_s: float = 30.0,
        forecast_advised_economy_shadow_min_pump_rate_m3_min: float = 2.0,
        forecast_advised_economy_shadow_min_backlog_kg: float = 600.0,
        forecast_advised_economy_shadow_latch_s: float = 1800.0,
        forecast_pause_resume_max_target_err_kg: float = 0.0,
        sustained_active_recompute_enabled: bool = False,
        sustained_active_recompute_min_active_buckets: int = 4,
        sustained_active_recompute_pitch_deg: float = 4.5,
        sustained_active_recompute_roll_deg: float = 4.5,
        sustained_active_recompute_target_err_kg: float = 1500.0,
        sustained_active_recompute_pump_rate_m3_min: float = 0.5,
        sustained_active_recompute_min_target_age_s: float = 1800.0,
        stale_active_target_refresh_mode: str = "off",
        stale_active_target_refresh_min_active_buckets: int = 2,
        stale_active_target_refresh_pitch_deg: float = 5.0,
        stale_active_target_refresh_roll_deg: float = 5.0,
        stale_active_target_refresh_target_err_kg: float = 500.0,
        stale_active_target_refresh_pump_rate_m3_min: float = 0.5,
        stale_active_target_refresh_min_target_age_s: float = 600.0,
        stale_active_target_refresh_min_delta_kg: float = 1500.0,
        stale_active_target_refresh_cooldown_s: float = 1200.0,
        stale_active_target_refresh_no_recovery_eps_deg: float = 0.10,
        reactive_floor_predictive_veto: str = "off",
        high_posture_metric: str = "max_axis",
        high_posture_enter: float = 5.0,
        high_posture_exit: float = 4.5,
        high_posture_min_duration_s: float = 60.0,
        reactive_floor_theta_only_enter: float = 6.0,
        reactive_floor_theta_only_min_duration_s: float = 120.0,
        reactive_floor_action: str = "active_small",
        reactive_floor_medium_delay_s: float = 0.0,
        reactive_floor_post_exit_mode: str = "off",
        reactive_floor_post_exit_target_cap_kg: float = 500.0,
        reactive_floor_post_exit_margin_deg: float = 0.0,
        reactive_floor_post_exit_worsening_eps_deg: float = 0.05,
        oracle_preemptive_prevent_enabled: bool = False,
        oracle_preemptive_enter_deg: float = 3.0,
        oracle_preemptive_floor_deg: float = 5.0,
        oracle_preemptive_min_future_norm: float = 0.75,
        oracle_preemptive_rise_margin_norm: float = 0.25,
        oracle_preemptive_min_delta_kg: float = 500.0,
        oracle_preemptive_cooldown_s: float = 1200.0,
        h120_oracle_probe_enabled: bool = False,
        h120_oracle_probe_short_delay_s: float = 600.0,
        h120_oracle_probe_far_high_norm: float = 1.10,
        h120_oracle_probe_intensify_margin_norm: float = 0.30,
        h120_oracle_probe_preemptive_enter_deg: float = 3.0,
        h120_oracle_probe_preemptive_floor_deg: float = 5.0,
        h120_oracle_probe_min_delta_kg: float = 500.0,
        h120_oracle_probe_cooldown_s: float = 1200.0,
        h120_risk_scheduler_enabled: bool = False,
        h120_scheduler_delay_normal_s: float = 1200.0,
        h120_scheduler_delay_risk_aware_s: float = 900.0,
        h120_scheduler_delay_high_risk_s: float = 600.0,
        h120_scheduler_far_high_norm: float = 0.90,
        h120_scheduler_intensify_margin_norm: float = 0.30,
        h120_scheduler_relief_margin_norm: float = 0.25,
        h120_scheduler_early_stop_suppress_max_per_episode: int = 1,
        h120_scheduler_prefloor_probe_enabled: bool = False,
        h120_scheduler_prefloor_action: str = "active_small",
        h120_scheduler_prefloor_enter_deg: float = 3.0,
        h120_scheduler_prefloor_floor_deg: float = 5.0,
        h120_scheduler_prefloor_min_delta_kg: float = 500.0,
        h120_scheduler_prefloor_max_delta_kg: float = 0.0,
        h120_scheduler_prefloor_cooldown_s: float = 1200.0,
        h120_axis_micro_enabled: bool = False,
        h120_axis_micro_enter_deg: float = 4.0,
        h120_axis_micro_floor_deg: float = 5.0,
        h120_axis_micro_rise_floor_norm: float = 0.8,
        h120_axis_micro_worsening_eps_deg: float = 0.05,
        h120_axis_micro_delta_kg: float = 2000.0,
        h120_axis_micro_cooldown_s: float = 600.0,
        h120_floor_shaping_mode: str = "off",
        h120_floor_shaping_far_high_norm: float = 0.90,
        h120_floor_shaping_slow_response_eps_deg: float = 0.05,
        h120_floor_shaping_medium_delay_s: float = 600.0,
        h120_pareto_mode_selector_enabled: bool = False,
        h120_pareto_mode_selector_far_high_norm: float = 0.90,
        h120_pareto_mode_selector_intensify_margin_norm: float = 0.30,
        h120_pareto_mode_selector_relief_margin_norm: float = 0.25,
        h120_pareto_mode_selector_slow_response_eps_deg: float = 0.05,
        relief_economy_enabled: bool = False,
        relief_economy_horizon: str = "far",
        relief_economy_safe_deg: float = 3.0,
        relief_economy_worsening_eps_deg: float = 0.05,
        relief_economy_low_norm: float = 0.5,
        relief_economy_high_norm: float = 0.9,
        relief_envelope_enabled: bool = False,
        relief_envelope_horizon: str = "far",
        relief_envelope_allowed_0_20_deg: float = 4.5,
        relief_envelope_allowed_20_40_deg: float = 4.2,
        relief_envelope_allowed_40_60_deg: float = 3.9,
        relief_envelope_allowed_60_120_deg: float = 3.5,
        relief_envelope_max_duration_s: float = 600.0,
        relief_envelope_debt_budget_deg_s: float = 300.0,
        relief_envelope_near_limit_deg: float = 4.7,
        relief_envelope_near_limit_budget_s: float = 0.0,
        relief_envelope_worsening_eps_deg: float = 0.0,
        relief_envelope_low_norm: float = 0.5,
        relief_envelope_high_norm: float = 0.9,
        relief_envelope_reintensify_norm: float = 0.9,
        relief_envelope_single_axis_only: bool = False,
        relief_envelope_axis_minor_ratio_max: float = 0.25,
        relief_envelope_axis_minor_abs_max_deg: float = 0.05,
        relief_envelope_adaptive_axis_cap: bool = False,
        relief_envelope_cap_relief_norm: float = 0.55,
        relief_envelope_cap_hold_fraction: float = 0.0,
        economy_pump_budget_enabled: bool = False,
        economy_pump_budget_m3: float = 0.0,
        economy_pump_budget_hold_refresh_fraction: float = 0.0,
        economy_pump_budget_dynamic_refresh_enabled: bool = False,
        economy_pump_budget_dynamic_refresh_enter_deg: float = 3.5,
        economy_pump_budget_dynamic_refresh_full_deg: float = 4.7,
        economy_pump_budget_dynamic_refresh_max_fraction: float = 0.85,
        economy_pump_budget_allocator_mode: str = "budget",
        economy_pump_budget_require_relief: bool = False,
        economy_pump_budget_relief_norm: float = 0.7,
        economy_pump_budget_relief_drop_norm: float = 0.2,
        economy_pump_budget_forecast_smart: bool = False,
        economy_pump_budget_smart_posture_deg: float = 4.25,
        economy_pump_budget_smart_pressure_norm: float = 0.9,
        economy_pump_budget_opportunity_window_s: float = 1200.0,
        economy_pump_budget_opportunity_min_pump_m3: float = 120.0,
        economy_pump_budget_opportunity_min_refresh: int = 3,
        economy_pump_budget_opportunity_min_delta_kg: float = 100000.0,
        hold_relief_debt_enabled: bool = False,
        hold_relief_debt_threshold_deg: float = 3.0,
        hold_relief_debt_response_eps_deg: float = 0.3,
        hold_relief_debt_increment: float = 1.0,
        hold_relief_debt_max: float = 4.0,
        hold_relief_debt_lowrisk_pressure_norm: float = 0.5,
        hold_relief_debt_lowrisk_safe_deg: float = 2.5,
        far_horizon_enabled: bool = False,
        far_horizon_relief_gate_enabled: bool = False,
        far_horizon_weight_60_80: float = 0.50,
        far_horizon_weight_80_100: float = 0.35,
        far_horizon_weight_100_120: float = 0.25,
        far_horizon_relief_pitch_abs_max_deg: float = 4.0,
        far_horizon_relief_roll_abs_max_deg: float = 4.0,
        forced_prefix_actions: dict[int, dict[str, Any]] | None = None,
        forced_prefix_mode: str = "off",
    ) -> None:
        self.replay_dataset = replay_dataset
        self.start_timestamp = start_timestamp
        self.cfg = cfg
        self.block_discounts = list(block_discounts)
        self.update_interval_s = int(replay_dataset.update_interval_s)
        self.scale = float(scale)
        if bias_shape not in ("hold", "event_decay"):
            raise ValueError(f"unsupported bias_shape={bias_shape!r}")
        if objective_mode not in ("lex", "economic"):
            raise ValueError(f"unsupported objective_mode={objective_mode!r}")
        if event_reset_mode not in ("action", "target_change", "active_bucket"):
            raise ValueError(f"unsupported event_reset_mode={event_reset_mode!r}")
        self.bias_shape = str(bias_shape)
        self.decay_tau_s = float(decay_tau_s)
        self.objective_mode = str(objective_mode)
        self.event_reset_mode = str(event_reset_mode)
        self.target_reset_tol_deg = float(target_reset_tol_deg)
        self.forecast_adapter = forecast_adapter
        # Latest blend-forecast diagnostic, populated by _forecast_uv each
        # planner step when the forecast_adapter is a BlendedForecastAdapter.
        # Empty for plain/oracle/persistence/current_only sources.
        self._last_forecast_blend_diagnostic: dict[str, Any] = {}
        if abs(float(setpoint_bias_sign)) < 1e-9:
            raise ValueError("setpoint_bias_sign must be nonzero")
        self.setpoint_bias_sign = float(setpoint_bias_sign)
        # Feedforward channel: emit per-tank mass delta (kg) directly into the
        # closed-loop command stream BEFORE the rate limiter, AFTER the heave
        # balancer. This bypasses the PI deadband, which is the structural
        # ceiling of the setpoint-shift channel (verified 2026-05-04).
        # The FF channel uses the PLANNER FRAME (compensation_direction) sign
        # of first_avec, NOT the setpoint frame -- because we are directly
        # commanding ballast motion, not shifting a PI target.
        self.ff_channel_enabled = bool(ff_channel_enabled)
        self.ff_scale = float(ff_scale)
        self.setpoint_channel_enabled = bool(setpoint_channel_enabled)
        # Prediction-primary prototype: instead of nudging the PI setpoint, emit
        # an absolute three-tank mass target that ClosedLoopPolicy can apply
        # directly before the plant pump latch/rate layer. Default-off preserves
        # all existing production behavior.
        self.prediction_primary_enabled = bool(prediction_primary_enabled)
        self.prediction_primary_scale = float(prediction_primary_scale)
        self.prediction_primary_disable_setpoint = bool(prediction_primary_disable_setpoint)
        if primary_hold_target_mode not in ("current", "pause", "pi_release", "forecast_pause"):
            raise ValueError(
                f"unsupported primary_hold_target_mode={primary_hold_target_mode!r}"
        )
        self.primary_hold_target_mode = str(primary_hold_target_mode)
        self.trusted_event_gate_enabled = bool(trusted_event_gate_enabled)
        self.trusted_event_highwind_threshold = float(trusted_event_highwind_threshold)
        self.trusted_event_attention_threshold = float(trusted_event_attention_threshold)
        self.trusted_event_attention_min_heads = max(1, int(trusted_event_attention_min_heads))
        self.trusted_event_dynamic_enabled = bool(trusted_event_dynamic_enabled)
        self.trusted_event_dynamic_threshold = float(trusted_event_dynamic_threshold)
        self.trusted_event_dynamic_min_heads = max(1, int(trusted_event_dynamic_min_heads))
        self.far_horizon_enabled = bool(far_horizon_enabled)
        self.far_horizon_relief_gate_enabled = bool(far_horizon_relief_gate_enabled)
        self.far_horizon_weights = (
            max(float(far_horizon_weight_60_80), 0.0),
            max(float(far_horizon_weight_80_100), 0.0),
            max(float(far_horizon_weight_100_120), 0.0),
        )
        self.far_horizon_relief_pitch_abs_max_deg = max(
            float(far_horizon_relief_pitch_abs_max_deg),
            0.0,
        )
        self.far_horizon_relief_roll_abs_max_deg = max(
            float(far_horizon_relief_roll_abs_max_deg),
            0.0,
        )
        self.relief_medium_cap_enabled = bool(relief_medium_cap_enabled)
        self.relief_medium_cap_event_threshold = float(relief_medium_cap_event_threshold)
        self.relief_medium_cap_ratio = float(relief_medium_cap_ratio)
        self.relief_medium_cap_adaptive = bool(relief_medium_cap_adaptive)
        self.relief_medium_cap_pitch_abs_max_deg = max(
            float(relief_medium_cap_pitch_abs_max_deg),
            0.0,
        )
        self.relief_medium_cap_roll_abs_max_deg = max(
            float(relief_medium_cap_roll_abs_max_deg),
            0.0,
        )
        self.active_posture_refresh_enabled = bool(active_posture_refresh_enabled)
        self.active_posture_refresh_enter_deg = max(
            float(active_posture_refresh_enter_deg),
            0.0,
        )
        self.active_posture_refresh_exit_deg = max(
            float(active_posture_refresh_exit_deg),
            0.0,
        )
        self.active_posture_refresh_update_interval_s = max(
            float(active_posture_refresh_update_interval_s),
            1.0,
        )
        self.active_posture_refresh_pressure_max_norm = max(
            float(active_posture_refresh_pressure_max_norm),
            0.0,
        )
        self.active_posture_refresh_min_target_age_s = max(
            float(active_posture_refresh_min_target_age_s),
            0.0,
        )
        if active_posture_refresh_axis_mode not in ("max", "axis_debt"):
            raise ValueError(
                "unsupported active_posture_refresh_axis_mode="
                f"{active_posture_refresh_axis_mode!r}"
            )
        self.active_posture_refresh_axis_mode = str(active_posture_refresh_axis_mode)
        self.active_posture_refresh_worsening_eps_deg_s = max(
            float(active_posture_refresh_worsening_eps_deg_s),
            0.0,
        )
        self.active_posture_refresh_debt_deg = max(
            float(active_posture_refresh_debt_deg),
            0.0,
        )
        self.active_posture_refresh_include_hold = bool(
            active_posture_refresh_include_hold
        )
        if active_posture_refresh_action_name not in (
            "pump_saving",
            "active_small",
            "active_medium",
        ):
            raise ValueError(
                "unsupported active_posture_refresh_action_name="
                f"{active_posture_refresh_action_name!r}"
            )
        self.active_posture_refresh_action_name = str(
            active_posture_refresh_action_name
        )
        self.fallback_risk_active_release_enabled = bool(
            fallback_risk_active_release_enabled
        )
        self.fallback_risk_active_release_enter_deg = max(
            float(fallback_risk_active_release_enter_deg),
            0.0,
        )
        self.fallback_risk_active_release_exit_deg = max(
            float(fallback_risk_active_release_exit_deg),
            0.0,
        )
        self.fallback_risk_active_release_worsening_eps_deg_s = max(
            float(fallback_risk_active_release_worsening_eps_deg_s),
            0.0,
        )
        self.fallback_risk_active_release_pump_rate_m3_min = max(
            float(fallback_risk_active_release_pump_rate_m3_min),
            0.0,
        )
        self.fallback_risk_active_release_backlog_kg = max(
            float(fallback_risk_active_release_backlog_kg),
            0.0,
        )
        self.no_preview_myopic_horizon_enabled = bool(no_preview_myopic_horizon_enabled)
        self.primary_stall_refresh_enabled = bool(primary_stall_refresh_enabled)
        self.primary_stall_pitch_deg = float(primary_stall_pitch_deg)
        self.primary_stall_roll_deg = float(primary_stall_roll_deg)
        self.primary_stall_target_err_kg = float(primary_stall_target_err_kg)
        self.primary_stall_pump_rate_m3_min = float(primary_stall_pump_rate_m3_min)
        self.primary_stall_min_age_s = float(primary_stall_min_age_s)
        self.primary_stall_pitch_axis_bias = bool(primary_stall_pitch_axis_bias)
        self.primary_stall_pitch_axis_roll_max_deg = float(
            primary_stall_pitch_axis_roll_max_deg
        )
        self.active_effectiveness_refresh_enabled = bool(
            active_effectiveness_refresh_enabled
        )
        self.active_effectiveness_refresh_min_active_buckets = max(
            int(active_effectiveness_refresh_min_active_buckets),
            1,
        )
        self.active_effectiveness_refresh_pitch_deg = max(
            float(active_effectiveness_refresh_pitch_deg),
            0.0,
        )
        self.active_effectiveness_refresh_roll_deg = max(
            float(active_effectiveness_refresh_roll_deg),
            0.0,
        )
        self.active_effectiveness_refresh_target_err_kg = max(
            float(active_effectiveness_refresh_target_err_kg),
            0.0,
        )
        self.active_effectiveness_refresh_pump_rate_m3_min = max(
            float(active_effectiveness_refresh_pump_rate_m3_min),
            0.0,
        )
        self.active_effectiveness_refresh_min_target_age_s = max(
            float(active_effectiveness_refresh_min_target_age_s),
            0.0,
        )
        self.active_effectiveness_refresh_min_pressure_norm = max(
            float(active_effectiveness_refresh_min_pressure_norm),
            0.0,
        )
        self.active_effectiveness_refresh_response_eps_deg = max(
            float(active_effectiveness_refresh_response_eps_deg),
            0.0,
        )
        self.active_effectiveness_refresh_required_worsening_deg = max(
            float(active_effectiveness_refresh_required_worsening_deg),
            0.0,
        )
        self.active_effectiveness_refresh_min_delta_kg = max(
            float(active_effectiveness_refresh_min_delta_kg),
            0.0,
        )
        self.active_effectiveness_refresh_episode_budget_m3 = max(
            float(active_effectiveness_refresh_episode_budget_m3),
            0.0,
        )
        self.active_effectiveness_refresh_lowrisk_pressure_norm = max(
            float(active_effectiveness_refresh_lowrisk_pressure_norm),
            0.0,
        )
        self.gusty_oscillation_enabled = bool(gusty_oscillation_enabled)
        self.gusty_oscillation_min_target_age_s = max(
            float(gusty_oscillation_min_target_age_s),
            0.0,
        )
        self.gusty_oscillation_target_scale = min(
            max(float(gusty_oscillation_target_scale), 0.0),
            1.0,
        )
        self.gusty_oscillation_headroom_scaled = bool(
            gusty_oscillation_headroom_scaled
        )
        self.gusty_oscillation_low_posture_deg = max(
            float(gusty_oscillation_low_posture_deg),
            0.0,
        )
        self.gusty_oscillation_high_posture_deg = max(
            float(gusty_oscillation_high_posture_deg),
            self.gusty_oscillation_low_posture_deg,
        )
        self.gusty_oscillation_low_posture_scale = min(
            max(float(gusty_oscillation_low_posture_scale), 0.0),
            1.0,
        )
        self.gusty_oscillation_high_posture_scale = min(
            max(float(gusty_oscillation_high_posture_scale), 0.0),
            1.0,
        )
        self.gusty_oscillation_short_gate_enabled = bool(
            gusty_oscillation_short_gate_enabled
        )
        self.gusty_oscillation_short_min_range_ms = max(
            float(gusty_oscillation_short_min_range_ms),
            0.0,
        )
        self.gusty_oscillation_short_min_pressure_norm = max(
            float(gusty_oscillation_short_min_pressure_norm),
            0.0,
        )
        self.gusty_oscillation_short_max_dir_shift_deg = max(
            float(gusty_oscillation_short_max_dir_shift_deg),
            0.0,
        )
        self.gusty_oscillation_short_min_turns = max(
            int(gusty_oscillation_short_min_turns),
            0,
        )
        if gusty_hold_current_gate_mode not in ("off", "gate_on", "gate_off"):
            raise ValueError(
                "gusty_hold_current_gate_mode must be off, gate_on, or gate_off"
            )
        self.gusty_hold_current_gate_mode = str(gusty_hold_current_gate_mode)
        self.gusty_hold_current_isolated_primary_enabled = bool(
            gusty_hold_current_isolated_primary_enabled
        )
        if direction_reversal_hold_current_gate_mode not in (
            "off",
            "gate_on",
            "gate_off",
        ):
            raise ValueError(
                "direction_reversal_hold_current_gate_mode must be off, gate_on, or gate_off"
            )
        self.direction_reversal_hold_current_gate_mode = str(
            direction_reversal_hold_current_gate_mode
        )
        self.forecast_control_trust_gate_enabled = bool(
            forecast_control_trust_gate_enabled
        )
        self.forecast_control_trust_event_threshold = min(
            max(float(forecast_control_trust_event_threshold), 0.0),
            1.0,
        )
        self.forecast_control_trust_current_stability_required = bool(
            forecast_control_trust_current_stability_required
        )
        self.forecast_control_trust_max_speed_range_ms = max(
            float(forecast_control_trust_max_speed_range_ms),
            0.0,
        )
        self.forecast_control_trust_max_dir_shift_deg = max(
            float(forecast_control_trust_max_dir_shift_deg),
            0.0,
        )
        self.forecast_control_trust_suppression_rise_norm = max(
            float(forecast_control_trust_suppression_rise_norm),
            0.0,
        )
        self.forecast_control_trust_min_pressure_norm = max(
            float(forecast_control_trust_min_pressure_norm),
            0.0,
        )
        self.forecast_pressure_trust_gate_enabled = bool(
            forecast_pressure_trust_gate_enabled
        )
        self.forecast_pressure_trust_current_norm_max = max(
            float(forecast_pressure_trust_current_norm_max),
            0.0,
        )
        self.forecast_pressure_trust_future_norm_min = max(
            float(forecast_pressure_trust_future_norm_min),
            0.0,
        )
        self.forecast_pressure_trust_event_threshold = min(
            max(float(forecast_pressure_trust_event_threshold), 0.0),
            1.0,
        )
        self.forecast_pressure_trust_untrusted_scale = min(
            max(float(forecast_pressure_trust_untrusted_scale), 0.0),
            1.0,
        )
        self.forecast_pressure_trust_min_direction_dot = float(
            forecast_pressure_trust_min_direction_dot
        )
        self.forecast_pressure_trust_spike_margin_norm = max(
            float(forecast_pressure_trust_spike_margin_norm),
            0.0,
        )
        self.gusty_hold_current_runtime_release_enabled = bool(
            gusty_hold_current_runtime_release_enabled
        )
        self.gusty_hold_current_runtime_release_deg = max(
            float(gusty_hold_current_runtime_release_deg),
            0.0,
        )
        self.gusty_hold_current_runtime_release_cooldown_s = max(
            float(gusty_hold_current_runtime_release_cooldown_s),
            0.0,
        )
        self.gusty_hold_current_debt_guard_enabled = bool(
            gusty_hold_current_debt_guard_enabled
        )
        self.gusty_hold_current_debt_guard_target_motion_kg_s = max(
            float(gusty_hold_current_debt_guard_target_motion_kg_s),
            0.0,
        )
        self.gusty_hold_current_debt_guard_target_err_kg = max(
            float(gusty_hold_current_debt_guard_target_err_kg),
            0.0,
        )
        self.gusty_hold_current_debt_guard_min_age_s = max(
            float(gusty_hold_current_debt_guard_min_age_s),
            0.0,
        )
        self.gusty_hold_current_day_budget_enabled = bool(
            gusty_hold_current_day_budget_enabled
        )
        self.gusty_hold_current_day_budget_max_buckets = max(
            int(gusty_hold_current_day_budget_max_buckets),
            0,
        )
        self.gusty_hold_current_day_budget_window_s = max(
            float(gusty_hold_current_day_budget_window_s),
            1.0,
        )
        self.forecast_advised_economy_enabled = bool(
            forecast_advised_economy_enabled
        )
        self.forecast_advised_economy_min_headroom_deg = max(
            float(forecast_advised_economy_min_headroom_deg),
            0.0,
        )
        self.forecast_advised_economy_rise_norm = max(
            float(forecast_advised_economy_rise_norm),
            0.0,
        )
        self.forecast_advised_economy_max_speed_range_ms = max(
            float(forecast_advised_economy_max_speed_range_ms),
            0.0,
        )
        self.forecast_advised_economy_max_dir_shift_deg = max(
            float(forecast_advised_economy_max_dir_shift_deg),
            0.0,
        )
        self.forecast_advised_economy_confirm_buckets = max(
            int(forecast_advised_economy_confirm_buckets),
            1,
        )
        self.forecast_advised_economy_hold_confirm_buckets = max(
            int(forecast_advised_economy_hold_confirm_buckets),
            0,
        )
        self.forecast_advised_economy_min_axis_deg = max(
            float(forecast_advised_economy_min_axis_deg),
            0.0,
        )
        self.forecast_advised_economy_min_pressure_norm = max(
            float(forecast_advised_economy_min_pressure_norm),
            0.0,
        )
        self.forecast_advised_economy_posture_release_deg = max(
            float(forecast_advised_economy_posture_release_deg),
            0.0,
        )
        self.forecast_advised_economy_release_cooldown_s = max(
            float(forecast_advised_economy_release_cooldown_s),
            0.0,
        )
        self.forecast_advised_economy_latch_s = max(
            float(forecast_advised_economy_latch_s),
            0.0,
        )
        self.forecast_advised_economy_current_ws_range_ms = max(
            float(forecast_advised_economy_current_ws_range_ms),
            0.0,
        )
        self.forecast_advised_economy_current_dir_shift_deg = max(
            float(forecast_advised_economy_current_dir_shift_deg),
            0.0,
        )
        self.forecast_advised_economy_current_history_s = max(
            float(forecast_advised_economy_current_history_s),
            float(self.update_interval_s),
        )
        self.forecast_advised_economy_mild_enabled = bool(
            forecast_advised_economy_mild_enabled
        )
        self.forecast_advised_economy_mild_min_headroom_deg = max(
            float(forecast_advised_economy_mild_min_headroom_deg),
            0.0,
        )
        self.forecast_advised_economy_mild_rise_norm = max(
            float(forecast_advised_economy_mild_rise_norm),
            0.0,
        )
        self.forecast_advised_economy_mild_max_speed_range_ms = max(
            float(forecast_advised_economy_mild_max_speed_range_ms),
            0.0,
        )
        self.forecast_advised_economy_mild_restart_err_kg = max(
            float(forecast_advised_economy_mild_restart_err_kg),
            0.0,
        )
        self.forecast_advised_economy_high_stable_clean_required = bool(
            forecast_advised_economy_high_stable_clean_required
        )
        self.forecast_advised_economy_high_stable_early_min_ms = max(
            float(forecast_advised_economy_high_stable_early_min_ms),
            0.0,
        )
        self.forecast_advised_economy_high_stable_near_min_ms = max(
            float(forecast_advised_economy_high_stable_near_min_ms),
            0.0,
        )
        self.forecast_advised_economy_high_stable_far_min_ms = max(
            float(forecast_advised_economy_high_stable_far_min_ms),
            0.0,
        )
        self.forecast_advised_economy_high_stable_near_range_max_ms = max(
            float(forecast_advised_economy_high_stable_near_range_max_ms),
            0.0,
        )
        self.forecast_advised_economy_high_stable_far_range_max_ms = max(
            float(forecast_advised_economy_high_stable_far_range_max_ms),
            0.0,
        )
        self.forecast_advised_economy_high_stable_dir_shift_max_deg = max(
            float(forecast_advised_economy_high_stable_dir_shift_max_deg),
            0.0,
        )
        self.forecast_advised_economy_min_peak_drop_ms = max(
            float(forecast_advised_economy_min_peak_drop_ms),
            0.0,
        )
        self.forecast_advised_economy_hold_current_on_candidate = bool(
            forecast_advised_economy_hold_current_on_candidate
        )
        self.forecast_advised_economy_suppression_only = bool(
            forecast_advised_economy_suppression_only
        )
        self.forecast_advised_economy_suppression_restart_err_kg = max(
            float(forecast_advised_economy_suppression_restart_err_kg),
            0.0,
        )
        self.forecast_advised_economy_runaway_release_enabled = bool(
            forecast_advised_economy_runaway_release_enabled
        )
        self.forecast_advised_economy_runaway_pump_rate_m3_min = max(
            float(forecast_advised_economy_runaway_pump_rate_m3_min),
            0.0,
        )
        self.forecast_advised_economy_runaway_backlog_kg = max(
            float(forecast_advised_economy_runaway_backlog_kg),
            0.0,
        )
        self.forecast_advised_economy_runaway_cooldown_s = max(
            float(forecast_advised_economy_runaway_cooldown_s),
            0.0,
        )
        self.forecast_advised_economy_shadow_opportunity_enabled = bool(
            forecast_advised_economy_shadow_opportunity_enabled
        )
        self.forecast_advised_economy_shadow_min_latch_delta = max(
            int(forecast_advised_economy_shadow_min_latch_delta),
            0,
        )
        self.forecast_advised_economy_shadow_min_target_motion_kg_s = max(
            float(forecast_advised_economy_shadow_min_target_motion_kg_s),
            0.0,
        )
        self.forecast_advised_economy_shadow_min_pump_rate_m3_min = max(
            float(forecast_advised_economy_shadow_min_pump_rate_m3_min),
            0.0,
        )
        self.forecast_advised_economy_shadow_min_backlog_kg = max(
            float(forecast_advised_economy_shadow_min_backlog_kg),
            0.0,
        )
        self.forecast_advised_economy_shadow_latch_s = max(
            float(forecast_advised_economy_shadow_latch_s),
            0.0,
        )
        self.forecast_pause_resume_max_target_err_kg = max(
            float(forecast_pause_resume_max_target_err_kg),
            0.0,
        )
        self.sustained_active_recompute_enabled = bool(
            sustained_active_recompute_enabled
        )
        self.sustained_active_recompute_min_active_buckets = max(
            int(sustained_active_recompute_min_active_buckets),
            1,
        )
        self.sustained_active_recompute_pitch_deg = max(
            float(sustained_active_recompute_pitch_deg),
            0.0,
        )
        self.sustained_active_recompute_roll_deg = max(
            float(sustained_active_recompute_roll_deg),
            0.0,
        )
        self.sustained_active_recompute_target_err_kg = max(
            float(sustained_active_recompute_target_err_kg),
            0.0,
        )
        self.sustained_active_recompute_pump_rate_m3_min = max(
            float(sustained_active_recompute_pump_rate_m3_min),
            0.0,
        )
        self.sustained_active_recompute_min_target_age_s = max(
            float(sustained_active_recompute_min_target_age_s),
            0.0,
        )
        if stale_active_target_refresh_mode not in (
            "off",
            "refresh_only_v1",
            "refresh_only_guarded_v1",
        ):
            raise ValueError(
                "unsupported stale_active_target_refresh_mode="
                f"{stale_active_target_refresh_mode!r}"
            )
        self.stale_active_target_refresh_mode = str(stale_active_target_refresh_mode)
        self.stale_active_target_refresh_min_active_buckets = max(
            int(stale_active_target_refresh_min_active_buckets),
            1,
        )
        self.stale_active_target_refresh_pitch_deg = max(
            float(stale_active_target_refresh_pitch_deg),
            0.0,
        )
        self.stale_active_target_refresh_roll_deg = max(
            float(stale_active_target_refresh_roll_deg),
            0.0,
        )
        self.stale_active_target_refresh_target_err_kg = max(
            float(stale_active_target_refresh_target_err_kg),
            0.0,
        )
        self.stale_active_target_refresh_pump_rate_m3_min = max(
            float(stale_active_target_refresh_pump_rate_m3_min),
            0.0,
        )
        self.stale_active_target_refresh_min_target_age_s = max(
            float(stale_active_target_refresh_min_target_age_s),
            0.0,
        )
        self.stale_active_target_refresh_min_delta_kg = max(
            float(stale_active_target_refresh_min_delta_kg),
            0.0,
        )
        self.stale_active_target_refresh_cooldown_s = max(
            float(stale_active_target_refresh_cooldown_s),
            0.0,
        )
        self.stale_active_target_refresh_no_recovery_eps_deg = max(
            float(stale_active_target_refresh_no_recovery_eps_deg),
            0.0,
        )
        if reactive_floor_predictive_veto not in (
            "off",
            "on",
            "theta_only_gated",
            "relaxed_veto_on_theta_only",
        ):
            raise ValueError(
                "unsupported reactive_floor_predictive_veto="
                f"{reactive_floor_predictive_veto!r}"
            )
        if high_posture_metric not in ("max_axis", "hybrid"):
            raise ValueError(f"unsupported high_posture_metric={high_posture_metric!r}")
        if reactive_floor_action not in ("active_small", "active_medium"):
            raise ValueError(f"unsupported reactive_floor_action={reactive_floor_action!r}")
        self.reactive_floor_predictive_veto_mode = str(reactive_floor_predictive_veto)
        self.reactive_floor_predictive_veto_enabled = (
            self.reactive_floor_predictive_veto_mode != "off"
        )
        self.high_posture_metric = str(high_posture_metric)
        self.high_posture_enter = max(float(high_posture_enter), 0.0)
        self.high_posture_exit = max(float(high_posture_exit), 0.0)
        self.high_posture_min_duration_s = max(float(high_posture_min_duration_s), 0.0)
        self.reactive_floor_theta_only_enter = max(
            float(reactive_floor_theta_only_enter),
            0.0,
        )
        self.reactive_floor_theta_only_min_duration_s = max(
            float(reactive_floor_theta_only_min_duration_s),
            0.0,
        )
        self.reactive_floor_action = str(reactive_floor_action)
        self.reactive_floor_medium_delay_s = max(
            float(reactive_floor_medium_delay_s),
            0.0,
        )
        if reactive_floor_post_exit_mode not in ("off", "early_stop", "target_cap"):
            raise ValueError(
                "unsupported reactive_floor_post_exit_mode="
                f"{reactive_floor_post_exit_mode!r}"
            )
        self.reactive_floor_post_exit_mode = str(reactive_floor_post_exit_mode)
        self.reactive_floor_post_exit_enabled = (
            self.reactive_floor_post_exit_mode != "off"
        )
        self.reactive_floor_post_exit_target_cap_kg = max(
            float(reactive_floor_post_exit_target_cap_kg),
            0.0,
        )
        self.reactive_floor_post_exit_margin_deg = max(
            float(reactive_floor_post_exit_margin_deg),
            0.0,
        )
        self.reactive_floor_post_exit_worsening_eps_deg = max(
            float(reactive_floor_post_exit_worsening_eps_deg),
            0.0,
        )
        self.oracle_preemptive_prevent_enabled = bool(
            oracle_preemptive_prevent_enabled
        )
        self.oracle_preemptive_enter_deg = max(
            float(oracle_preemptive_enter_deg),
            0.0,
        )
        self.oracle_preemptive_floor_deg = max(
            float(oracle_preemptive_floor_deg),
            self.oracle_preemptive_enter_deg,
        )
        self.oracle_preemptive_min_future_norm = max(
            float(oracle_preemptive_min_future_norm),
            0.0,
        )
        self.oracle_preemptive_rise_margin_norm = max(
            float(oracle_preemptive_rise_margin_norm),
            0.0,
        )
        self.oracle_preemptive_min_delta_kg = max(
            float(oracle_preemptive_min_delta_kg),
            0.0,
        )
        self.oracle_preemptive_cooldown_s = max(
            float(oracle_preemptive_cooldown_s),
            0.0,
        )
        self.h120_oracle_probe_enabled = bool(h120_oracle_probe_enabled)
        self.h120_oracle_probe_short_delay_s = max(
            float(h120_oracle_probe_short_delay_s),
            0.0,
        )
        self.h120_oracle_probe_far_high_norm = max(
            float(h120_oracle_probe_far_high_norm),
            0.0,
        )
        self.h120_oracle_probe_intensify_margin_norm = max(
            float(h120_oracle_probe_intensify_margin_norm),
            0.0,
        )
        self.h120_oracle_probe_preemptive_enter_deg = max(
            float(h120_oracle_probe_preemptive_enter_deg),
            0.0,
        )
        self.h120_oracle_probe_preemptive_floor_deg = max(
            float(h120_oracle_probe_preemptive_floor_deg),
            self.h120_oracle_probe_preemptive_enter_deg,
        )
        self.h120_oracle_probe_min_delta_kg = max(
            float(h120_oracle_probe_min_delta_kg),
            0.0,
        )
        self.h120_oracle_probe_cooldown_s = max(
            float(h120_oracle_probe_cooldown_s),
            0.0,
        )
        # h120 learned risk scheduler (default-off). Uses the 60-120min
        # far-horizon pressure blocks only to (a) shorten the delayed-medium
        # delay and (b) suppress early_stop at most once per floor episode.
        # It NEVER decides whether high posture recovers (that stays with the
        # reactive floor), NEVER refreshes the primary target preemptively,
        # and NEVER suppresses recovery via relief prediction.
        self.h120_risk_scheduler_enabled = bool(h120_risk_scheduler_enabled)
        self.h120_scheduler_delay_normal_s = max(float(h120_scheduler_delay_normal_s), 0.0)
        self.h120_scheduler_delay_risk_aware_s = max(float(h120_scheduler_delay_risk_aware_s), 0.0)
        self.h120_scheduler_delay_high_risk_s = max(float(h120_scheduler_delay_high_risk_s), 0.0)
        self.h120_scheduler_far_high_norm = max(float(h120_scheduler_far_high_norm), 0.0)
        self.h120_scheduler_intensify_margin_norm = max(float(h120_scheduler_intensify_margin_norm), 0.0)
        self.h120_scheduler_relief_margin_norm = max(float(h120_scheduler_relief_margin_norm), 0.0)
        self.h120_scheduler_early_stop_suppress_max_per_episode = max(
            int(h120_scheduler_early_stop_suppress_max_per_episode), 0
        )
        self.h120_scheduler_prefloor_probe_enabled = bool(
            h120_scheduler_prefloor_probe_enabled
        )
        if h120_scheduler_prefloor_action not in ("pump_saving", "active_small"):
            raise ValueError(
                "unsupported h120_scheduler_prefloor_action="
                f"{h120_scheduler_prefloor_action!r}"
            )
        self.h120_scheduler_prefloor_action = str(h120_scheduler_prefloor_action)
        self.h120_scheduler_prefloor_enter_deg = max(
            float(h120_scheduler_prefloor_enter_deg), 0.0
        )
        self.h120_scheduler_prefloor_floor_deg = max(
            float(h120_scheduler_prefloor_floor_deg),
            self.h120_scheduler_prefloor_enter_deg,
        )
        self.h120_scheduler_prefloor_min_delta_kg = max(
            float(h120_scheduler_prefloor_min_delta_kg), 0.0
        )
        self.h120_scheduler_prefloor_max_delta_kg = max(
            float(h120_scheduler_prefloor_max_delta_kg), 0.0
        )
        self.h120_scheduler_prefloor_cooldown_s = max(
            float(h120_scheduler_prefloor_cooldown_s), 0.0
        )
        self.h120_axis_micro_enabled = bool(h120_axis_micro_enabled)
        self.h120_axis_micro_enter_deg = max(float(h120_axis_micro_enter_deg), 0.0)
        self.h120_axis_micro_floor_deg = max(
            float(h120_axis_micro_floor_deg), self.h120_axis_micro_enter_deg
        )
        self.h120_axis_micro_rise_floor_norm = max(
            float(h120_axis_micro_rise_floor_norm), 0.0
        )
        self.h120_axis_micro_worsening_eps_deg = max(
            float(h120_axis_micro_worsening_eps_deg), 0.0
        )
        self.h120_axis_micro_delta_kg = max(float(h120_axis_micro_delta_kg), 0.0)
        self.h120_axis_micro_cooldown_s = max(float(h120_axis_micro_cooldown_s), 0.0)
        if h120_floor_shaping_mode not in (
            "off",
            "axis_shaped",
            "bounded_medium",
            "axis_shaped_bounded_medium",
            "stale_repair",
        ):
            raise ValueError(
                f"unsupported h120_floor_shaping_mode={h120_floor_shaping_mode!r}"
            )
        self.h120_floor_shaping_mode = str(h120_floor_shaping_mode)
        self.h120_floor_shaping_enabled = self.h120_floor_shaping_mode != "off"
        self.h120_floor_shaping_far_high_norm = max(
            float(h120_floor_shaping_far_high_norm), 0.0
        )
        self.h120_floor_shaping_slow_response_eps_deg = max(
            float(h120_floor_shaping_slow_response_eps_deg), 0.0
        )
        self.h120_floor_shaping_medium_delay_s = max(
            float(h120_floor_shaping_medium_delay_s), 0.0
        )
        self.h120_pareto_mode_selector_enabled = bool(
            h120_pareto_mode_selector_enabled
        )
        self.h120_pareto_mode_selector_far_high_norm = max(
            float(h120_pareto_mode_selector_far_high_norm), 0.0
        )
        self.h120_pareto_mode_selector_intensify_margin_norm = max(
            float(h120_pareto_mode_selector_intensify_margin_norm), 0.0
        )
        self.h120_pareto_mode_selector_relief_margin_norm = max(
            float(h120_pareto_mode_selector_relief_margin_norm), 0.0
        )
        self.h120_pareto_mode_selector_slow_response_eps_deg = max(
            float(h120_pareto_mode_selector_slow_response_eps_deg), 0.0
        )
        self.relief_economy_enabled = bool(relief_economy_enabled)
        if relief_economy_horizon not in ("near", "far"):
            raise ValueError(f"unsupported relief_economy_horizon={relief_economy_horizon!r}")
        self.relief_economy_horizon = str(relief_economy_horizon)
        self.relief_economy_safe_deg = max(float(relief_economy_safe_deg), 0.0)
        self.relief_economy_worsening_eps_deg = max(
            float(relief_economy_worsening_eps_deg), 0.0
        )
        self.relief_economy_low_norm = max(float(relief_economy_low_norm), 0.0)
        self.relief_economy_high_norm = max(float(relief_economy_high_norm), 0.0)
        self.relief_envelope_enabled = bool(relief_envelope_enabled)
        if relief_envelope_horizon not in ("near", "far"):
            raise ValueError(
                f"unsupported relief_envelope_horizon={relief_envelope_horizon!r}"
            )
        self.relief_envelope_horizon = str(relief_envelope_horizon)
        self.relief_envelope_allowed_deg = (
            max(float(relief_envelope_allowed_0_20_deg), 0.0),
            max(float(relief_envelope_allowed_20_40_deg), 0.0),
            max(float(relief_envelope_allowed_40_60_deg), 0.0),
            max(float(relief_envelope_allowed_60_120_deg), 0.0),
        )
        self.relief_envelope_max_duration_s = max(
            float(relief_envelope_max_duration_s), 0.0
        )
        self.relief_envelope_debt_budget_deg_s = max(
            float(relief_envelope_debt_budget_deg_s), 0.0
        )
        self.relief_envelope_near_limit_deg = max(
            float(relief_envelope_near_limit_deg), 0.0
        )
        self.relief_envelope_near_limit_budget_s = max(
            float(relief_envelope_near_limit_budget_s), 0.0
        )
        self.relief_envelope_worsening_eps_deg = max(
            float(relief_envelope_worsening_eps_deg), 0.0
        )
        self.relief_envelope_low_norm = max(float(relief_envelope_low_norm), 0.0)
        self.relief_envelope_high_norm = max(float(relief_envelope_high_norm), 0.0)
        self.relief_envelope_reintensify_norm = max(
            float(relief_envelope_reintensify_norm), 0.0
        )
        self.relief_envelope_single_axis_only = bool(relief_envelope_single_axis_only)
        self.relief_envelope_axis_minor_ratio_max = max(
            float(relief_envelope_axis_minor_ratio_max), 0.0
        )
        self.relief_envelope_axis_minor_abs_max_deg = max(
            float(relief_envelope_axis_minor_abs_max_deg), 0.0
        )
        self.relief_envelope_adaptive_axis_cap = bool(
            relief_envelope_adaptive_axis_cap
        )
        self.relief_envelope_cap_relief_norm = max(
            float(relief_envelope_cap_relief_norm), 0.0
        )
        self.relief_envelope_cap_hold_fraction = min(
            max(float(relief_envelope_cap_hold_fraction), 0.0),
            1.0,
        )
        self.economy_pump_budget_enabled = bool(economy_pump_budget_enabled)
        self.economy_pump_budget_m3 = max(float(economy_pump_budget_m3), 0.0)
        self.economy_pump_budget_hold_refresh_fraction = min(
            max(float(economy_pump_budget_hold_refresh_fraction), 0.0),
            1.0,
        )
        self.economy_pump_budget_dynamic_refresh_enabled = bool(
            economy_pump_budget_dynamic_refresh_enabled
        )
        self.economy_pump_budget_dynamic_refresh_enter_deg = max(
            float(economy_pump_budget_dynamic_refresh_enter_deg), 0.0
        )
        self.economy_pump_budget_dynamic_refresh_full_deg = max(
            float(economy_pump_budget_dynamic_refresh_full_deg),
            self.economy_pump_budget_dynamic_refresh_enter_deg + 1e-6,
        )
        self.economy_pump_budget_dynamic_refresh_max_fraction = min(
            max(float(economy_pump_budget_dynamic_refresh_max_fraction), 0.0),
            1.0,
        )
        if economy_pump_budget_allocator_mode not in (
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
        self.economy_pump_budget_allocator_mode = str(
            economy_pump_budget_allocator_mode
        )
        self.economy_pump_budget_require_relief = bool(economy_pump_budget_require_relief)
        self.economy_pump_budget_relief_norm = max(
            float(economy_pump_budget_relief_norm),
            0.0,
        )
        self.economy_pump_budget_relief_drop_norm = max(
            float(economy_pump_budget_relief_drop_norm),
            0.0,
        )
        self.economy_pump_budget_forecast_smart = bool(economy_pump_budget_forecast_smart)
        self.economy_pump_budget_smart_posture_deg = max(
            float(economy_pump_budget_smart_posture_deg),
            0.0,
        )
        self.economy_pump_budget_smart_pressure_norm = max(
            float(economy_pump_budget_smart_pressure_norm),
            0.0,
        )
        self.economy_pump_budget_opportunity_window_s = max(
            float(economy_pump_budget_opportunity_window_s),
            0.0,
        )
        self.economy_pump_budget_opportunity_min_pump_m3 = max(
            float(economy_pump_budget_opportunity_min_pump_m3),
            0.0,
        )
        self.economy_pump_budget_opportunity_min_refresh = max(
            int(economy_pump_budget_opportunity_min_refresh),
            0,
        )
        self.economy_pump_budget_opportunity_min_delta_kg = max(
            float(economy_pump_budget_opportunity_min_delta_kg),
            0.0,
        )
        self.hold_relief_debt_enabled = bool(hold_relief_debt_enabled)
        self.hold_relief_debt_threshold_deg = max(
            float(hold_relief_debt_threshold_deg),
            0.0,
        )
        self.hold_relief_debt_response_eps_deg = max(
            float(hold_relief_debt_response_eps_deg),
            0.0,
        )
        self.hold_relief_debt_increment = max(float(hold_relief_debt_increment), 0.0)
        self.hold_relief_debt_max = max(float(hold_relief_debt_max), 0.0)
        self.hold_relief_debt_lowrisk_pressure_norm = max(
            float(hold_relief_debt_lowrisk_pressure_norm),
            0.0,
        )
        self.hold_relief_debt_lowrisk_safe_deg = max(
            float(hold_relief_debt_lowrisk_safe_deg),
            0.0,
        )
        if forced_prefix_mode not in ("off", "raw_action", "final_action", "target_update"):
            raise ValueError(f"unsupported forced_prefix_mode={forced_prefix_mode!r}")
        self.forced_prefix_mode = str(forced_prefix_mode)
        self.forced_prefix_actions: dict[int, dict[str, Any]] = {
            int(k): dict(v)
            for k, v in (forced_prefix_actions or {}).items()
        }
        self._default_plant_info = plant_info or {
            "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
            "pump_fullspeed_any": 0,
            "pump_total_backlog_kg": 0.0,
            "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        }
        # Cached planner-frame target (for FF) and zeroed mass FF state.
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        apply_primary_target_state_defaults(
            self,
            self._default_plant_info["tank_masses"],
            refresh_owner="init",
        )
        self._forecast_advised_economy_candidate = False
        self._forecast_advised_economy_boundary_veto = False
        self._forecast_advised_economy_headroom_deg = 0.0
        self._forecast_advised_economy_near_rise_norm = 0.0
        self._forecast_advised_economy_posture_release = False
        self._forecast_advised_economy_max_axis_deg = 0.0
        self._forecast_advised_economy_clear_streak = 0
        self._forecast_advised_economy_confirmed = False
        self._forecast_advised_economy_release_cooldown_until_s = 0.0
        self._forecast_advised_economy_latched_until_s = 0.0
        self._forecast_advised_economy_latched_active = False
        self._forecast_advised_current_ws_range_ms = 0.0
        self._forecast_advised_current_dir_shift_deg = 0.0
        self._forecast_advised_current_history_n = 0
        self._forecast_advised_current_stability_veto = False
        self._forecast_advised_high_stable_clean_gate = False
        self._forecast_advised_high_stable_clean_reason = "disabled"
        self._forecast_advised_hold_current_override = False
        self._forecast_advised_runaway_release = False
        self._forecast_advised_runaway_pump_rate_m3_min = 0.0
        self._forecast_advised_runaway_backlog_kg = 0.0
        self._forecast_advised_shadow_opportunity = False
        self._forecast_advised_shadow_latch_delta = 0
        self._forecast_advised_shadow_target_motion_kg_s = 0.0
        self._forecast_advised_shadow_pump_rate_m3_min = 0.0
        self._forecast_advised_shadow_backlog_kg = 0.0
        self._forecast_advised_shadow_prev_latch_count = 0
        self._forecast_advised_shadow_latched_until_s = 0.0
        self._forecast_advised_shadow_latched_active = False
        self._forecast_advised_economy_reason = "disabled"
        self._forecast_pause_resume_guard_active = False
        self._forecast_pause_resume_guard_reason = "not_evaluated"
        self._forecast_pause_resume_target_err_kg = 0.0
        self._gusty_oscillation_gate_active = False
        self._gusty_oscillation_gate_reason = "disabled"
        self._gusty_oscillation_last_scale = 1.0
        self._gusty_short_blend_applied = False
        self._gusty_short_target_delta_before_kg = 0.0
        self._gusty_short_target_delta_after_kg = 0.0
        self._gusty_hold_current_gate_evaluated = False
        self._gusty_hold_current_allowed = False
        self._gusty_hold_current_blocked = False
        self._gusty_hold_current_reason = "not_evaluated"
        self._direction_reversal_hold_current_gate_evaluated = False
        self._direction_reversal_hold_current_allowed = False
        self._direction_reversal_hold_current_blocked = False
        self._direction_reversal_hold_current_reason = "not_evaluated"
        self._gusty_hold_current_latched_active = False
        self._gusty_hold_current_runtime_released = False
        self._gusty_hold_current_runtime_release_reason = "not_evaluated"
        self._gusty_hold_current_runtime_release_until_s = 0.0
        self._gusty_hold_current_runtime_release_count = 0
        self._gusty_hold_current_debt_guard_active = False
        self._gusty_hold_current_debt_guard_reason = "not_evaluated"
        self._gusty_hold_current_debt_guard_target_motion_kg_s = 0.0
        self._gusty_hold_current_debt_guard_target_err_kg = 0.0
        self._gusty_hold_current_day_budget_used_buckets = 0
        self._gusty_hold_current_day_budget_remaining_buckets = (
            int(self.gusty_hold_current_day_budget_max_buckets)
            if bool(self.gusty_hold_current_day_budget_enabled)
            else 0
        )
        self._gusty_hold_current_day_budget_window_start_s = 0.0
        self._gusty_hold_current_day_budget_active = False
        self._gusty_hold_current_day_budget_blocked = False
        self._gusty_hold_current_day_budget_reason = "init"
        self._fallback_risk_active_release_active = False
        self._fallback_risk_active_release_reason = "init"
        self._fallback_risk_active_release_pitch_abs_deg = 0.0
        self._fallback_risk_active_release_roll_abs_deg = 0.0
        self._fallback_risk_active_release_pitch_worsening_deg_s = 0.0
        self._fallback_risk_active_release_roll_worsening_deg_s = 0.0
        self._fallback_risk_active_release_pump_rate_m3_min = 0.0
        self._fallback_risk_active_release_backlog_kg = 0.0
        self._fallback_risk_active_release_count = 0
        # Objective-mode state (default-off prototype): counter increments while
        # posture exceeds threshold, decrements while below. Recovery is
        # latched between enter_buckets and exit_buckets (hysteresis).
        self._persistent_high_buckets = 0
        self._persistent_below_buckets = 0
        self._objective_mode_latched_recovery = False
        # Diagnostic: count mode transitions to detect chattering.
        self._objective_mode_switch_count = 0
        self._objective_mode_last_label = "economic"
        self._event_risk_raw_probs = [0.0, 0.0, 0.0]
        self._event_risk_probs = [0.0, 0.0, 0.0]
        self._event_risk_block_scales = [1.0, 1.0, 1.0]
        self._trusted_event_gate_diag: dict[str, Any] = {
            "enabled": 0,
            "trusted": 1,
            "reason": "init",
            "highwind_hit": 0,
            "attention_hits": 0,
            "dynamic_hits": 0,
            "highwind_prob": 0.0,
            "attention_threshold": float(self.trusted_event_attention_threshold),
            "highwind_threshold": float(self.trusted_event_highwind_threshold),
            "dynamic_threshold": float(self.trusted_event_dynamic_threshold),
        }
        self._raw_pressure_block_norms = [0.0, 0.0, 0.0]
        self._raw_pressure_block_vecs = [np.zeros(2, dtype=float) for _ in range(3)]
        self._effective_pressure_block_norms = [0.0, 0.0, 0.0]
        self._effective_pressure_block_vecs = [
            np.zeros(2, dtype=float) for _ in range(3)
        ]
        self._reset_forecast_pressure_trust_state("init", reset_observation=True)
        self._reset_far_horizon_state("init")
        self._reset_forecast_speed_shape_state("init")
        self._reset_psc_v2_short_state("init")
        self._h120_oracle_probe_count = 0
        self._h120_oracle_probe_short_delay_count = 0
        self._h120_oracle_probe_early_stop_veto_count = 0
        self._h120_oracle_probe_preemptive_count = 0
        self._h120_oracle_probe_last_preemptive_s = -1.0e12
        self._reset_h120_oracle_probe_bucket_state("init")
        # h120 risk scheduler per-case counters + per-episode latch
        self._h120_scheduler_early_stop_suppressed_count = 0
        self._h120_scheduler_remote_risk_trigger_count = 0
        self._h120_scheduler_early_stop_suppressed_this_episode = 0
        self._h120_scheduler_prefloor_count = 0
        self._h120_scheduler_prefloor_last_s = -1.0e12
        self._h120_axis_micro_count = 0
        self._h120_axis_micro_last_s = -1.0e12
        self._h120_axis_micro_active = False
        self._h120_axis_micro_reason = "init"
        self._h120_axis_micro_axis = "none"
        self._h120_axis_micro_delta_mean_kg = 0.0
        self._relief_economy_active = False
        self._relief_economy_reason = "init"
        self._relief_economy_near_low = False
        self._relief_economy_far_low = False
        self._relief_economy_delay_count = 0
        self._relief_envelope_active = False
        self._relief_envelope_reason = "init"
        self._relief_envelope_allowed_max_axis_deg = 0.0
        self._relief_envelope_relief_block_index = -1
        self._relief_envelope_relief_time_min = 0.0
        self._relief_envelope_axis_cap_requested_delta_deg = 0.0
        self._relief_envelope_axis_cap_pursued_delta_deg = 0.0
        self._relief_envelope_axis_cap_held_delta_deg = 0.0
        self._relief_envelope_axis_cap_covered_axes = 0
        self._relief_envelope_axis_cap_mode = "none"
        self._economy_pump_budget_used_m3 = 0.0
        self._economy_pump_budget_last_s: float | None = None
        self._economy_pump_budget_active = False
        self._economy_pump_budget_reason = "init"
        self._economy_pump_budget_hold_count = 0
        self._economy_pump_budget_refresh_count = 0
        self._economy_pump_budget_partial_refresh_count = 0
        self._economy_pump_budget_partial_refresh_fraction = 0.0
        self._economy_pump_budget_partial_delta_mean_kg = 0.0
        self._economy_pump_budget_blocked_delta_mean_kg = 0.0
        self._economy_pump_budget_opportunity_pump_m3 = 0.0
        self._economy_pump_budget_opportunity_refresh_count = 0
        self._economy_pump_budget_opportunity_delta_sum_kg = 0.0
        self._economy_pump_budget_opportunity_score = 0.0
        self._economy_pump_budget_opportunity_active = False
        self._economy_pump_budget_plateau_detected = False
        self._economy_pump_budget_plateau_confidence = 0.0
        self._economy_pump_budget_boundary_veto_reason = "init"
        self._economy_pump_budget_failclosed_release_active = False
        self._economy_pump_budget_failclosed_release_reason = "init"
        self._economy_pump_budget_clean_start_checked = False
        self._economy_pump_budget_clean_start_eligible = False
        self._economy_pump_budget_initial_max_axis_deg = float("nan")
        self._relief_decay_auto_v2_latched = False
        self._relief_decay_auto_v2_latch_reason = "init"
        self._relief_decay_auto_v2_latch_count = 0
        self._economy_pump_budget_hold_latched = False
        self._economy_pump_budget_forced_release_count = 0
        self._economy_pump_budget_forced_release_reason = "init"
        self._relief_envelope_eligible_count = 0
        self._relief_envelope_pump_opportunity_count = 0
        self._relief_envelope_target_refresh_delayed_count = 0
        self._relief_envelope_relaxation_active_time_s = 0.0
        self._relief_envelope_current_duration_s = 0.0
        self._relief_envelope_safety_debt_deg_s = 0.0
        self._relief_envelope_near_limit_dwell_s = 0.0
        self._relief_envelope_debt_exit_count = 0
        self._relief_envelope_false_relief_count = 0
        self._relief_envelope_reintensification_veto_count = 0
        self._relief_envelope_direction_mismatch_veto_count = 0
        self._relief_envelope_axis_shape_veto_count = 0
        self._relief_envelope_axis_cap_count = 0
        self._relief_envelope_axis_cap_requested_delta_deg = 0.0
        self._relief_envelope_axis_cap_pursued_delta_deg = 0.0
        self._relief_envelope_axis_cap_held_delta_deg = 0.0
        self._relief_envelope_axis_cap_covered_axes = 0
        self._relief_envelope_axis_cap_mode = "none"
        self._economy_pump_budget_used_m3 = 0.0
        self._economy_pump_budget_last_s = None
        self._economy_pump_budget_active = False
        self._economy_pump_budget_reason = "reset"
        self._economy_pump_budget_hold_count = 0
        self._economy_pump_budget_refresh_count = 0
        self._economy_pump_budget_partial_refresh_count = 0
        self._economy_pump_budget_partial_refresh_fraction = 0.0
        self._economy_pump_budget_partial_delta_mean_kg = 0.0
        self._economy_pump_budget_blocked_delta_mean_kg = 0.0
        self._economy_pump_budget_opportunity_pump_m3 = 0.0
        self._economy_pump_budget_opportunity_refresh_count = 0
        self._economy_pump_budget_opportunity_delta_sum_kg = 0.0
        self._economy_pump_budget_opportunity_score = 0.0
        self._economy_pump_budget_opportunity_active = False
        self._economy_pump_budget_plateau_detected = False
        self._economy_pump_budget_plateau_confidence = 0.0
        self._economy_pump_budget_boundary_veto_reason = "reset"
        self._economy_pump_budget_failclosed_release_active = False
        self._economy_pump_budget_failclosed_release_reason = "reset"
        self._economy_pump_budget_clean_start_checked = False
        self._economy_pump_budget_clean_start_eligible = False
        self._economy_pump_budget_initial_max_axis_deg = float("nan")
        self._relief_decay_auto_v2_latched = False
        self._relief_decay_auto_v2_latch_reason = "reset"
        self._relief_decay_auto_v2_latch_count = 0
        self._economy_pump_budget_hold_latched = False
        self._economy_pump_budget_forced_release_count = 0
        self._economy_pump_budget_forced_release_reason = "reset"
        self._relief_envelope_posture_worsening_exit_count = 0
        self._relief_envelope_a2_unique_relax_count = 0
        self._relief_envelope_a2_unique_pump_saved_m3 = 0.0
        self._relief_envelope_last_update_s = -1.0e12
        self._relief_envelope_prev_active = False
        self._relief_envelope_evaluated_this_bucket = False
        self._h120_floor_shaping_active = False
        self._h120_floor_shaping_reason = "init"
        self._h120_floor_shaping_effect = "none"
        self._h120_floor_shaping_axis = "none"
        self._h120_floor_shaping_count = 0
        self._h120_floor_shaping_delta_mean_kg = 0.0
        self._h120_pareto_mode_selector_active = False
        self._h120_pareto_mode_selector_reason = "init"
        self._h120_pareto_mode_selector_selected_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_selected_effect = "balanced"
        self._h120_pareto_mode_selector_count = 0
        self._h120_pareto_mode_selector_switch_count = 0
        self._h120_pareto_mode_selector_last_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_episode_active = False
        self._h120_pareto_mode_selector_episode_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_far_persistent_high = False
        self._h120_pareto_mode_selector_far_intensification = False
        self._h120_pareto_mode_selector_far_reintensification = False
        self._h120_pareto_mode_selector_far_relief = False
        self._h120_pareto_mode_selector_far_direction_consistent = False
        self._h120_pareto_mode_selector_recovery_slow = False
        self._h120_pareto_mode_selector_far_max_norm = 0.0
        self._h120_pareto_mode_selector_far_min_norm = 0.0
        self._h120_pareto_mode_selector_near_last_norm = 0.0
        self._h120_pareto_mode_selector_current_response_deg = float("nan")
        self._reset_h120_scheduler_bucket_state("init")
        self._forecast_has_future = False
        self._forecast_event_probs_available = False
        self._reset_forecast_control_trust_state("init")
        self._relief_medium_cap_active = False
        self._relief_medium_cap_reason = "disabled"
        self._relief_medium_cap_relief_margin = 0.0
        self._active_posture_refresh_active = False
        self._active_posture_refresh_reason = "disabled"
        self._active_posture_refresh_last_update_s: float | None = None
        self._active_posture_refresh_pitch_abs_deg = 0.0
        self._active_posture_refresh_roll_abs_deg = 0.0
        self._active_posture_refresh_action_name = ""
        self._active_posture_refresh_count = 0
        self._active_posture_refresh_axis_mode_active = "max"
        self._active_posture_refresh_axis_mask = "none"
        self._active_posture_refresh_pitch_worsening_deg_s = 0.0
        self._active_posture_refresh_roll_worsening_deg_s = 0.0
        self._active_posture_refresh_last_posture_time_s: float | None = None
        self._active_posture_refresh_last_pitch_abs_deg = 0.0
        self._active_posture_refresh_last_roll_abs_deg = 0.0
        self._hold_future_posture_evidence_active = False
        self._hold_future_posture_evidence_reason = "init"
        self._hold_relief_debt_level = 0.0
        self._hold_relief_debt_planner_level = 0.0
        self._hold_relief_debt_applied = False
        self._hold_relief_debt_reason = "disabled"
        self._hold_relief_debt_wait_reason = ""
        self._hold_relief_debt_pending = False
        self._hold_relief_debt_pending_attitude_deg = float("nan")
        self._hold_relief_debt_pending_reason = ""
        self._hold_relief_debt_current_response_deg = float("nan")
        self._hold_relief_debt_lowrisk_safe = False
        self._hold_relief_debt_fallback_dominated = False
        self._posture_action_refresh_active = False
        self._primary_stall_refresh_active = False
        self._primary_stall_refresh_reason = "disabled"
        self._primary_stall_axis_bias_active = False
        self._primary_stall_target_err_mean_kg = 0.0
        self._primary_stall_pump_rate_m3_min = 0.0
        self._active_effectiveness_refresh_active = False
        self._active_effectiveness_refresh_reason = "disabled"
        self._active_effectiveness_repeated_active_buckets = 0
        self._active_effectiveness_pitch_abs_deg = 0.0
        self._active_effectiveness_roll_abs_deg = 0.0
        self._active_effectiveness_attitude_abs_deg = 0.0
        self._active_effectiveness_prev_attitude_abs_deg = float("nan")
        self._active_effectiveness_current_response_deg = float("nan")
        self._active_effectiveness_future_relief_status = "unknown"
        self._active_effectiveness_target_err_mean_kg = 0.0
        self._active_effectiveness_pump_rate_m3_min = 0.0
        self._active_effectiveness_target_age_s = 0.0
        self._active_effectiveness_pressure_max_norm = 0.0
        self._active_effectiveness_proposal_delta_mean_kg = 0.0
        self._active_effectiveness_proposal_delta_m3 = 0.0
        self._active_effectiveness_budget_used_m3 = 0.0
        self._active_effectiveness_lowrisk_safe = False
        self._active_effectiveness_fallback_dominated = False
        self._active_effectiveness_target_ready = False
        self._active_effectiveness_pump_idle = False
        self._forecast_speed_peak_count = 0
        self._forecast_speed_oscillation_range_ms = 0.0
        self._forecast_speed_range_available = False
        self._forecast_speed_range_reason = "reset"
        self._forecast_speed_range_window_n = 0
        self._forecast_speed_range_ms = 0.0
        self._forecast_speed_gusty_oscillation_candidate = False
        self._sustained_active_recompute_active = False
        self._sustained_active_recompute_reason = "disabled"
        self._stale_active_target_refresh_active = False
        self._stale_active_target_refresh_reason = "disabled"
        self._stale_active_target_refresh_candidate_delta_mean_kg = 0.0
        self._stale_active_target_refresh_delta_after_kg = 0.0
        self._stale_active_target_refresh_target_err_before_kg = 0.0
        self._stale_active_target_refresh_target_age_before_s = 0.0
        self._stale_active_target_refresh_pump_rate_m3_min = 0.0
        self._stale_active_target_refresh_posture_abs_deg = 0.0
        self._stale_active_target_refresh_response_deg = float("nan")
        self._stale_active_target_refresh_last_s = -1.0e12
        self._stale_active_target_refresh_count = 0
        self._stale_active_target_refresh_rejected_count = 0
        self._stale_active_target_refresh_no_effect_count = 0
        apply_reactive_floor_state_defaults(
            self,
            reactive_floor_action=self.reactive_floor_action,
        )
        self._oracle_preemptive_active = False
        self._oracle_preemptive_reason = "disabled"
        self._oracle_preemptive_count = 0
        self._oracle_preemptive_last_s = -1.0e12
        self._oracle_preemptive_posture_metric_deg = 0.0
        self._oracle_preemptive_current_norm = 0.0
        self._oracle_preemptive_future_max_norm = 0.0
        self._oracle_preemptive_rise_norm = 0.0
        self._oracle_preemptive_delta_mean_kg = 0.0
        self._h120_oracle_probe_count = 0
        self._h120_oracle_probe_short_delay_count = 0
        self._h120_oracle_probe_early_stop_veto_count = 0
        self._h120_oracle_probe_preemptive_count = 0
        self._h120_oracle_probe_last_preemptive_s = -1.0e12
        self._reset_h120_oracle_probe_bucket_state("reset")
        self._h120_scheduler_early_stop_suppressed_count = 0
        self._h120_scheduler_remote_risk_trigger_count = 0
        self._h120_scheduler_early_stop_suppressed_this_episode = 0
        self._h120_scheduler_prefloor_count = 0
        self._h120_scheduler_prefloor_last_s = -1.0e12
        self._h120_axis_micro_count = 0
        self._h120_axis_micro_last_s = -1.0e12
        self._h120_axis_micro_active = False
        self._h120_axis_micro_reason = "reset"
        self._h120_axis_micro_axis = "none"
        self._h120_axis_micro_delta_mean_kg = 0.0
        self._relief_economy_active = False
        self._relief_economy_reason = "reset"
        self._relief_economy_near_low = False
        self._relief_economy_far_low = False
        self._relief_economy_delay_count = 0
        self._relief_envelope_active = False
        self._relief_envelope_reason = "reset"
        self._relief_envelope_allowed_max_axis_deg = 0.0
        self._relief_envelope_relief_block_index = -1
        self._relief_envelope_relief_time_min = 0.0
        self._relief_envelope_eligible_count = 0
        self._relief_envelope_pump_opportunity_count = 0
        self._relief_envelope_target_refresh_delayed_count = 0
        self._relief_envelope_relaxation_active_time_s = 0.0
        self._relief_envelope_current_duration_s = 0.0
        self._relief_envelope_safety_debt_deg_s = 0.0
        self._relief_envelope_near_limit_dwell_s = 0.0
        self._relief_envelope_debt_exit_count = 0
        self._relief_envelope_false_relief_count = 0
        self._relief_envelope_reintensification_veto_count = 0
        self._relief_envelope_direction_mismatch_veto_count = 0
        self._relief_envelope_axis_shape_veto_count = 0
        self._relief_envelope_axis_cap_count = 0
        self._relief_envelope_axis_cap_requested_delta_deg = 0.0
        self._relief_envelope_axis_cap_pursued_delta_deg = 0.0
        self._relief_envelope_axis_cap_held_delta_deg = 0.0
        self._relief_envelope_axis_cap_covered_axes = 0
        self._relief_envelope_axis_cap_mode = "none"
        self._relief_envelope_posture_worsening_exit_count = 0
        self._relief_envelope_a2_unique_relax_count = 0
        self._relief_envelope_a2_unique_pump_saved_m3 = 0.0
        self._relief_envelope_last_update_s = -1.0e12
        self._relief_envelope_prev_active = False
        self._relief_envelope_evaluated_this_bucket = False
        self._h120_floor_shaping_active = False
        self._h120_floor_shaping_reason = "reset"
        self._h120_floor_shaping_effect = "none"
        self._h120_floor_shaping_axis = "none"
        self._h120_floor_shaping_count = 0
        self._h120_floor_shaping_delta_mean_kg = 0.0
        self._h120_pareto_mode_selector_active = False
        self._h120_pareto_mode_selector_reason = "reset"
        self._h120_pareto_mode_selector_selected_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_selected_effect = "balanced"
        self._h120_pareto_mode_selector_count = 0
        self._h120_pareto_mode_selector_switch_count = 0
        self._h120_pareto_mode_selector_last_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_episode_active = False
        self._h120_pareto_mode_selector_episode_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_far_persistent_high = False
        self._h120_pareto_mode_selector_far_intensification = False
        self._h120_pareto_mode_selector_far_reintensification = False
        self._h120_pareto_mode_selector_far_relief = False
        self._h120_pareto_mode_selector_far_direction_consistent = False
        self._h120_pareto_mode_selector_recovery_slow = False
        self._h120_pareto_mode_selector_far_max_norm = 0.0
        self._h120_pareto_mode_selector_far_min_norm = 0.0
        self._h120_pareto_mode_selector_near_last_norm = 0.0
        self._h120_pareto_mode_selector_current_response_deg = float("nan")
        self._reset_h120_scheduler_bucket_state("reset")
        self._last_bucket = -1
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_action = "hold"
        self._transition_time_s = 0.0
        self._reset_forced_prefix_state()
        self.records: list[dict[str, Any]] = []  # per-bucket planner log

    def _reset_forced_prefix_state(self) -> None:
        self._forced_prefix_active = False
        self._forced_prefix_target_update_active = False
        self._forced_prefix_mode_active = ""
        self._forced_prefix_bucket = -1
        self._forced_action_label = ""
        self._forced_pitch_deg = 0.0
        self._forced_roll_deg = 0.0
        self._forced_vector_mode = "unavailable"
        self._forced_source_label = ""
        self._forced_original_raw_action = ""
        self._forced_original_final_action = ""
        self._forced_target_update_action = ""

    def _forced_prefix_entry(self, bucket: int) -> dict[str, Any] | None:
        if self.forced_prefix_mode == "off":
            return None
        entry = self.forced_prefix_actions.get(int(bucket))
        return dict(entry) if entry else None

    @staticmethod
    def _entry_float(entry: dict[str, Any], *names: str) -> float | None:
        for name in names:
            if name not in entry:
                continue
            value = entry.get(name)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def _forced_prefix_action_vec(
        self,
        entry: dict[str, Any],
        action: str,
        blocks: list[dict[str, Any]],
        plant_info: dict[str, Any],
    ) -> tuple[np.ndarray, str]:
        pitch = self._entry_float(
            entry,
            "forced_pitch_deg",
            "planner_action_pitch_deg",
            "pitch_deg",
            "planner_pitch_deg",
        )
        roll = self._entry_float(
            entry,
            "forced_roll_deg",
            "planner_action_roll_deg",
            "roll_deg",
            "planner_roll_deg",
        )
        if pitch is not None and roll is not None:
            return np.array([pitch, roll], dtype=float), "logged"
        try:
            avec, _ = planner_action_vec(action, 0, blocks, plant_info, self.cfg)
            return np.asarray(avec, dtype=float).reshape(2), "reconstructed"
        except Exception:
            return np.zeros(2, dtype=float), "unavailable"

    def _apply_forced_prefix_action(
        self,
        stage: str,
        bucket: int,
        current_action: str,
        current_avec: np.ndarray,
        blocks: list[dict[str, Any]],
        plant_info: dict[str, Any],
    ) -> tuple[str, np.ndarray]:
        if self.forced_prefix_mode != stage:
            return current_action, current_avec
        entry = self._forced_prefix_entry(bucket)
        if entry is None:
            return current_action, current_avec
        action = str(entry.get("action", entry.get("forced_action_label", ""))).strip()
        if action not in ACTIONS:
            self._forced_prefix_active = True
            self._forced_prefix_mode_active = stage
            self._forced_prefix_bucket = int(bucket)
            self._forced_action_label = action
            self._forced_vector_mode = "unavailable"
            self._forced_source_label = str(entry.get("source_label", "manual"))
            return current_action, current_avec
        forced_avec, vector_mode = self._forced_prefix_action_vec(
            entry,
            action,
            blocks,
            plant_info,
        )
        if vector_mode == "unavailable":
            return current_action, current_avec
        self._forced_prefix_active = True
        self._forced_prefix_mode_active = stage
        self._forced_prefix_bucket = int(bucket)
        self._forced_action_label = action
        self._forced_pitch_deg = float(forced_avec[0])
        self._forced_roll_deg = float(forced_avec[1])
        self._forced_vector_mode = vector_mode
        self._forced_source_label = str(entry.get("source_label", "manual"))
        return action, forced_avec

    def reset(self) -> None:
        self._last_bucket = -1
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        apply_primary_target_state_defaults(
            self,
            self._default_plant_info["tank_masses"],
            refresh_owner="reset",
        )
        if hasattr(self, "_forecast_deadband_held_target_kg"):
            del self._forecast_deadband_held_target_kg
        self._forecast_advised_economy_candidate = False
        self._forecast_advised_economy_boundary_veto = False
        self._forecast_advised_economy_headroom_deg = 0.0
        self._forecast_advised_economy_near_rise_norm = 0.0
        self._forecast_advised_economy_posture_release = False
        self._forecast_advised_economy_max_axis_deg = 0.0
        self._forecast_advised_economy_clear_streak = 0
        self._forecast_advised_economy_confirmed = False
        self._forecast_advised_economy_release_cooldown_until_s = 0.0
        self._forecast_advised_economy_latched_until_s = 0.0
        self._forecast_advised_economy_latched_active = False
        self._forecast_advised_current_ws_range_ms = 0.0
        self._forecast_advised_current_dir_shift_deg = 0.0
        self._forecast_advised_current_history_n = 0
        self._forecast_advised_current_stability_veto = False
        self._forecast_advised_high_stable_clean_gate = False
        self._forecast_advised_high_stable_clean_reason = "disabled"
        self._forecast_advised_hold_current_override = False
        self._forecast_advised_runaway_release = False
        self._forecast_advised_runaway_pump_rate_m3_min = 0.0
        self._forecast_advised_runaway_backlog_kg = 0.0
        self._forecast_advised_shadow_opportunity = False
        self._forecast_advised_shadow_latch_delta = 0
        self._forecast_advised_shadow_target_motion_kg_s = 0.0
        self._forecast_advised_shadow_pump_rate_m3_min = 0.0
        self._forecast_advised_shadow_backlog_kg = 0.0
        self._forecast_advised_shadow_prev_latch_count = 0
        self._forecast_advised_shadow_latched_until_s = 0.0
        self._forecast_advised_shadow_latched_active = False
        self._forecast_advised_economy_reason = "disabled"
        self._forecast_pause_resume_guard_active = False
        self._forecast_pause_resume_guard_reason = "not_evaluated"
        self._forecast_pause_resume_target_err_kg = 0.0
        self._gusty_oscillation_gate_active = False
        self._gusty_oscillation_gate_reason = "disabled"
        self._gusty_oscillation_last_scale = 1.0
        self._gusty_short_blend_applied = False
        self._gusty_short_target_delta_before_kg = 0.0
        self._gusty_short_target_delta_after_kg = 0.0
        self._gusty_hold_current_gate_evaluated = False
        self._gusty_hold_current_allowed = False
        self._gusty_hold_current_blocked = False
        self._gusty_hold_current_reason = "not_evaluated"
        self._gusty_hold_current_latched_active = False
        self._gusty_hold_current_runtime_released = False
        self._gusty_hold_current_runtime_release_reason = "not_evaluated"
        self._gusty_hold_current_runtime_release_until_s = 0.0
        self._gusty_hold_current_runtime_release_count = 0
        self._gusty_hold_current_debt_guard_active = False
        self._gusty_hold_current_debt_guard_reason = "not_evaluated"
        self._gusty_hold_current_debt_guard_target_motion_kg_s = 0.0
        self._gusty_hold_current_debt_guard_target_err_kg = 0.0
        self._gusty_hold_current_day_budget_used_buckets = 0
        self._gusty_hold_current_day_budget_remaining_buckets = (
            int(self.gusty_hold_current_day_budget_max_buckets)
            if bool(self.gusty_hold_current_day_budget_enabled)
            else 0
        )
        self._gusty_hold_current_day_budget_window_start_s = 0.0
        self._gusty_hold_current_day_budget_active = False
        self._gusty_hold_current_day_budget_blocked = False
        self._gusty_hold_current_day_budget_reason = "reset"
        # Objective-mode state (default-off prototype): counter increments while
        # posture exceeds threshold, decrements while below. Recovery is
        # latched between enter_buckets and exit_buckets (hysteresis).
        self._persistent_high_buckets = 0
        self._persistent_below_buckets = 0
        self._objective_mode_latched_recovery = False
        # Diagnostic: count mode transitions to detect chattering.
        self._objective_mode_switch_count = 0
        self._objective_mode_last_label = "economic"
        self._event_risk_raw_probs = [0.0, 0.0, 0.0]
        self._event_risk_probs = [0.0, 0.0, 0.0]
        self._event_risk_block_scales = [1.0, 1.0, 1.0]
        self._trusted_event_gate_diag = {
            "enabled": 0,
            "trusted": 1,
            "reason": "reset",
            "highwind_hit": 0,
            "attention_hits": 0,
            "dynamic_hits": 0,
            "highwind_prob": 0.0,
            "attention_threshold": float(self.trusted_event_attention_threshold),
            "highwind_threshold": float(self.trusted_event_highwind_threshold),
            "dynamic_threshold": float(self.trusted_event_dynamic_threshold),
        }
        self._raw_pressure_block_norms = [0.0, 0.0, 0.0]
        self._raw_pressure_block_vecs = [np.zeros(2, dtype=float) for _ in range(3)]
        self._effective_pressure_block_norms = [0.0, 0.0, 0.0]
        self._effective_pressure_block_vecs = [
            np.zeros(2, dtype=float) for _ in range(3)
        ]
        self._reset_forecast_pressure_trust_state("reset", reset_observation=True)
        self._reset_far_horizon_state("reset")
        self._reset_forecast_speed_shape_state("reset")
        self._reset_psc_v2_short_state("reset")
        self._forecast_has_future = False
        self._forecast_event_probs_available = False
        self._reset_forecast_control_trust_state("reset")
        self._active_posture_refresh_active = False
        self._active_posture_refresh_reason = "disabled"
        self._active_posture_refresh_last_update_s = None
        self._active_posture_refresh_pitch_abs_deg = 0.0
        self._active_posture_refresh_roll_abs_deg = 0.0
        self._active_posture_refresh_action_name = ""
        self._active_posture_refresh_count = 0
        self._active_posture_refresh_axis_mode_active = "max"
        self._active_posture_refresh_axis_mask = "none"
        self._active_posture_refresh_pitch_worsening_deg_s = 0.0
        self._active_posture_refresh_roll_worsening_deg_s = 0.0
        self._active_posture_refresh_last_posture_time_s = None
        self._active_posture_refresh_last_pitch_abs_deg = 0.0
        self._active_posture_refresh_last_roll_abs_deg = 0.0
        self._hold_future_posture_evidence_active = False
        self._hold_future_posture_evidence_reason = "reset"
        self._hold_relief_debt_level = 0.0
        self._hold_relief_debt_planner_level = 0.0
        self._hold_relief_debt_applied = False
        self._hold_relief_debt_reason = "disabled"
        self._hold_relief_debt_wait_reason = ""
        self._hold_relief_debt_pending = False
        self._hold_relief_debt_pending_attitude_deg = float("nan")
        self._hold_relief_debt_pending_reason = ""
        self._hold_relief_debt_current_response_deg = float("nan")
        self._hold_relief_debt_lowrisk_safe = False
        self._hold_relief_debt_fallback_dominated = False
        self._active_effectiveness_refresh_active = False
        self._active_effectiveness_refresh_reason = "disabled"
        self._active_effectiveness_repeated_active_buckets = 0
        self._active_effectiveness_pitch_abs_deg = 0.0
        self._active_effectiveness_roll_abs_deg = 0.0
        self._active_effectiveness_attitude_abs_deg = 0.0
        self._active_effectiveness_prev_attitude_abs_deg = float("nan")
        self._active_effectiveness_current_response_deg = float("nan")
        self._active_effectiveness_future_relief_status = "unknown"
        self._active_effectiveness_target_err_mean_kg = 0.0
        self._active_effectiveness_pump_rate_m3_min = 0.0
        self._active_effectiveness_target_age_s = 0.0
        self._active_effectiveness_pressure_max_norm = 0.0
        self._active_effectiveness_proposal_delta_mean_kg = 0.0
        self._active_effectiveness_proposal_delta_m3 = 0.0
        self._active_effectiveness_budget_used_m3 = 0.0
        self._active_effectiveness_lowrisk_safe = False
        self._active_effectiveness_fallback_dominated = False
        self._active_effectiveness_target_ready = False
        self._active_effectiveness_pump_idle = False
        self._forecast_speed_peak_count = 0
        self._forecast_speed_oscillation_range_ms = 0.0
        self._forecast_speed_range_available = False
        self._forecast_speed_range_reason = "reset"
        self._forecast_speed_range_window_n = 0
        self._forecast_speed_range_ms = 0.0
        self._forecast_speed_gusty_oscillation_candidate = False
        self._sustained_active_recompute_active = False
        self._sustained_active_recompute_reason = "disabled"
        self._stale_active_target_refresh_active = False
        self._stale_active_target_refresh_reason = "disabled"
        self._stale_active_target_refresh_candidate_delta_mean_kg = 0.0
        self._stale_active_target_refresh_delta_after_kg = 0.0
        self._stale_active_target_refresh_target_err_before_kg = 0.0
        self._stale_active_target_refresh_target_age_before_s = 0.0
        self._stale_active_target_refresh_pump_rate_m3_min = 0.0
        self._stale_active_target_refresh_posture_abs_deg = 0.0
        self._stale_active_target_refresh_response_deg = float("nan")
        self._stale_active_target_refresh_last_s = -1.0e12
        self._stale_active_target_refresh_count = 0
        self._stale_active_target_refresh_rejected_count = 0
        self._stale_active_target_refresh_no_effect_count = 0
        apply_reactive_floor_state_defaults(
            self,
            reactive_floor_action=self.reactive_floor_action,
        )
        self._oracle_preemptive_active = False
        self._oracle_preemptive_reason = "disabled"
        self._oracle_preemptive_count = 0
        self._oracle_preemptive_last_s = -1.0e12
        self._oracle_preemptive_posture_metric_deg = 0.0
        self._oracle_preemptive_current_norm = 0.0
        self._oracle_preemptive_future_max_norm = 0.0
        self._oracle_preemptive_rise_norm = 0.0
        self._oracle_preemptive_delta_mean_kg = 0.0
        self._target_action = "hold"
        self._transition_time_s = 0.0
        self.records = []

    def _bucket_timestamp(self, current_time: float) -> datetime:
        return self.replay_dataset.simulation_timestamp(self.start_timestamp, current_time)

    def _plant_info_from(self, plant_info_prev: dict | None) -> dict:
        info = dict(self._default_plant_info)
        if plant_info_prev:
            if "tank_masses" in plant_info_prev:
                tm = np.asarray(plant_info_prev["tank_masses"], dtype=float).reshape(-1)[:3]
                if tm.size == 3:
                    info["tank_masses"] = tm
            if "pump_fullspeed_any" in plant_info_prev:
                info["pump_fullspeed_any"] = int(plant_info_prev["pump_fullspeed_any"])
            if "pump_rate_cmd_m3_min" in plant_info_prev:
                pr = np.asarray(
                    plant_info_prev["pump_rate_cmd_m3_min"],
                    dtype=float,
                ).reshape(-1)[:3]
                if pr.size == 3:
                    info["pump_rate_cmd_m3_min"] = pr
            if "pump_total_backlog_kg" in plant_info_prev:
                info["pump_total_backlog_kg"] = float(
                    plant_info_prev["pump_total_backlog_kg"]
                )
            if "pump_latch_switch_count" in plant_info_prev:
                info["pump_latch_switch_count"] = int(
                    plant_info_prev["pump_latch_switch_count"]
                )
            if "pump_target_motion_kg_s" in plant_info_prev:
                info["pump_target_motion_kg_s"] = float(
                    plant_info_prev["pump_target_motion_kg_s"]
                )
        return info

    @staticmethod
    def _attach_posture_info(plant_info: dict[str, Any], state) -> dict[str, Any]:
        info = dict(plant_info)
        if state is None:
            info["posture_vec_deg"] = np.zeros(2, dtype=float)
            return info
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size <= 4:
            info["posture_vec_deg"] = np.zeros(2, dtype=float)
            return info
        # Planner vector order is [pitch, roll].
        info["posture_vec_deg"] = np.array(
            [
                float(np.degrees(state_arr[4])),
                float(np.degrees(state_arr[3])),
            ],
            dtype=float,
        )
        return info

    def _hold_relief_debt_observe_before_planner(
        self,
        plant_info: dict[str, Any],
        raw_pressure_norms: list[float] | None = None,
    ) -> dict[str, Any]:
        """Update wait-debt from the previous high-posture hold, then expose it.

        The debt is intentionally tiny and planner-internal: it only penalizes
        future hold/pump_saving candidates. It never directly changes the final
        action, target lifecycle, forecast, or pump execution.
        """
        info = dict(plant_info)
        self._hold_relief_debt_applied = False
        self._hold_relief_debt_reason = "disabled"
        self._hold_relief_debt_wait_reason = ""
        self._hold_relief_debt_current_response_deg = float("nan")
        self._hold_relief_debt_lowrisk_safe = False
        self._hold_relief_debt_fallback_dominated = bool(
            int(info.get("pump_fullspeed_any", 0))
        )
        if not self.hold_relief_debt_enabled:
            info["hold_relief_debt_level"] = 0.0
            self._hold_relief_debt_planner_level = 0.0
            return info

        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        attitude_abs = float(max(abs(posture[0]), abs(posture[1])))
        norms_src = (
            [float(x) for x in raw_pressure_norms[:3]]
            if raw_pressure_norms is not None
            else [float(x) for x in self._raw_pressure_block_norms[:3]]
        )
        max_pressure = max(norms_src) if norms_src else 0.0
        self._hold_relief_debt_lowrisk_safe = (
            attitude_abs <= self.hold_relief_debt_lowrisk_safe_deg
            and max_pressure <= self.hold_relief_debt_lowrisk_pressure_norm
        )

        if self._hold_relief_debt_pending:
            prev = float(self._hold_relief_debt_pending_attitude_deg)
            response = prev - attitude_abs if math.isfinite(prev) else float("nan")
            self._hold_relief_debt_current_response_deg = float(response)
            blocked = (
                self._hold_relief_debt_fallback_dominated
                or self._hold_relief_debt_lowrisk_safe
                or attitude_abs <= self.hold_relief_debt_threshold_deg
            )
            if blocked:
                self._hold_relief_debt_level = 0.0
                self._hold_relief_debt_reason = "cleared_or_blocked"
            elif response < self.hold_relief_debt_response_eps_deg:
                self._hold_relief_debt_level = min(
                    self.hold_relief_debt_max,
                    self._hold_relief_debt_level + self.hold_relief_debt_increment,
                )
                self._hold_relief_debt_reason = "previous_wait_not_realized"
            else:
                self._hold_relief_debt_level = max(
                    0.0,
                    self._hold_relief_debt_level - self.hold_relief_debt_increment,
                )
                self._hold_relief_debt_reason = "previous_wait_improved"
            self._hold_relief_debt_wait_reason = self._hold_relief_debt_pending_reason
        else:
            if (
                self._hold_relief_debt_fallback_dominated
                or self._hold_relief_debt_lowrisk_safe
                or attitude_abs <= self.hold_relief_debt_threshold_deg
            ):
                self._hold_relief_debt_level = 0.0
                self._hold_relief_debt_reason = "cleared_without_pending_wait"
            else:
                self._hold_relief_debt_reason = "no_pending_wait"

        self._hold_relief_debt_pending = False
        self._hold_relief_debt_pending_attitude_deg = float("nan")
        self._hold_relief_debt_pending_reason = ""
        if self._hold_relief_debt_fallback_dominated or self._hold_relief_debt_lowrisk_safe:
            debt_for_planner = 0.0
        else:
            debt_for_planner = self._hold_relief_debt_level
        self._hold_relief_debt_applied = debt_for_planner > 0.0
        self._hold_relief_debt_planner_level = float(debt_for_planner)
        info["hold_relief_debt_level"] = float(debt_for_planner)
        return info

    def _hold_relief_debt_observe_after_action(
        self,
        new_action: str,
        plant_info: dict[str, Any],
    ) -> None:
        if not self.hold_relief_debt_enabled:
            return
        action = str(new_action)
        if action not in ("hold", "pump_saving"):
            self._hold_relief_debt_level = max(
                0.0,
                self._hold_relief_debt_level - self.hold_relief_debt_increment,
            )
            return
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        attitude_abs = float(max(abs(posture[0]), abs(posture[1])))
        if attitude_abs <= self.hold_relief_debt_threshold_deg:
            return
        max_pressure = max(float(x) for x in self._raw_pressure_block_norms[:3])
        if (
            attitude_abs <= self.hold_relief_debt_lowrisk_safe_deg
            and max_pressure <= self.hold_relief_debt_lowrisk_pressure_norm
        ):
            return
        wait_reason = ""
        if self._hold_future_posture_evidence_active:
            wait_reason = str(self._hold_future_posture_evidence_reason)
        elif max_pressure <= self.hold_relief_debt_lowrisk_pressure_norm:
            wait_reason = "low_pressure_wait"
        if not wait_reason:
            return
        self._hold_relief_debt_pending = True
        self._hold_relief_debt_pending_attitude_deg = attitude_abs
        self._hold_relief_debt_pending_reason = wait_reason

    def _forecast_uv(self, sample) -> tuple[np.ndarray, str, dict[str, float]]:
        contract = ForecastContract(
            future_steps=int(self.replay_dataset.future_steps),
            event_columns=tuple(self.replay_dataset.event_columns),
            require_event_probs=self.forecast_adapter is None,
        )
        if self.forecast_adapter is None:
            forecast = forecast_from_oracle_sample(
                y_uv_raw=sample.y_uv_raw,
                y_event=sample.y_event,
                event_columns=self.replay_dataset.event_columns,
                model_version="oracle_future",
                timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self._last_forecast_blend_diagnostic = {}
            return (
                np.asarray(forecast.wind_uv_raw, dtype=float),
                str(forecast.model_version),
                dict(forecast.event_probs),
            )
        forecast = self.forecast_adapter.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        forecast = validate_forecast_result(forecast, contract)
        # Capture optional blend diagnostic. Only BlendedForecastAdapter
        # populates this; plain ForecastModelAdapter leaves it unset.
        diag = getattr(self.forecast_adapter, "last_diagnostic", None)
        self._last_forecast_blend_diagnostic = dict(diag) if isinstance(diag, dict) else {}
        return (
            np.asarray(forecast.wind_uv_raw, dtype=float),
            str(forecast.model_version),
            dict(forecast.event_probs),
        )

    def _oracle_preemptive_blocks(self, sample) -> list[dict[str, Any]]:
        """Oracle pressure blocks for the preemptive ceiling test.

        The main planner can still use a learned forecast. This helper reads
        the replay sample's true future only for the default-off preemptive
        diagnostic, so the result measures an oracle ceiling rather than a
        learned-model claim.
        """
        block_discounts = self.block_discounts
        uv = np.asarray(sample.y_uv_raw, dtype=float)
        return compute_pressure_blocks(uv, block_discounts, self.cfg)

    def _oracle_preemptive_log_fields(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}oracle_preemptive_enabled": int(
                self.oracle_preemptive_prevent_enabled
            ),
            f"{prefix}oracle_preemptive_active": int(self._oracle_preemptive_active),
            f"{prefix}oracle_preemptive_reason": self._oracle_preemptive_reason,
            f"{prefix}oracle_preemptive_count": int(self._oracle_preemptive_count),
            f"{prefix}oracle_preemptive_posture_metric_deg": float(
                self._oracle_preemptive_posture_metric_deg
            ),
            f"{prefix}oracle_preemptive_current_norm": float(
                self._oracle_preemptive_current_norm
            ),
            f"{prefix}oracle_preemptive_future_max_norm": float(
                self._oracle_preemptive_future_max_norm
            ),
            f"{prefix}oracle_preemptive_rise_norm": float(
                self._oracle_preemptive_rise_norm
            ),
            f"{prefix}oracle_preemptive_delta_mean_kg": float(
                self._oracle_preemptive_delta_mean_kg
            ),
        }

    def _reset_h120_oracle_probe_bucket_state(self, reason: str = "not_evaluated") -> None:
        self._h120_oracle_probe_active = False
        self._h120_oracle_probe_reason = str(reason)
        self._h120_oracle_probe_action_effect = "none"
        self._h120_oracle_probe_effective_medium_delay_s = float(
            self.reactive_floor_medium_delay_s
        )
        self._h120_oracle_probe_far_persistent_high = False
        self._h120_oracle_probe_far_intensification = False
        self._h120_oracle_probe_far_reintensification = False
        self._h120_oracle_probe_posture_metric_deg = 0.0
        self._h120_oracle_probe_far_max_norm = 0.0
        self._h120_oracle_probe_far_min_norm = 0.0
        self._h120_oracle_probe_near_last_norm = 0.0
        self._h120_oracle_probe_far_rise_norm = 0.0
        self._h120_oracle_probe_delta_mean_kg = 0.0
        self._h120_oracle_probe_preemptive_active = False
        self._h120_oracle_probe_short_delay_active = False
        self._h120_oracle_probe_early_stop_veto_active = False

    def _h120_oracle_probe_refresh_signals(
        self,
        plant_info: dict[str, Any] | None = None,
    ) -> None:
        if not self.h120_oracle_probe_enabled:
            self._h120_oracle_probe_reason = "disabled"
            return
        if not self.far_horizon_enabled:
            self._h120_oracle_probe_reason = "far_horizon_disabled"
            return
        if not self._far_horizon_available:
            self._h120_oracle_probe_reason = self._far_horizon_reason
            return

        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        near_last = float(norms[2])
        far = [float(x) for x in norms[3:6]]
        far_min = float(min(far))
        far_max = float(max(far))
        far_rise = far_max - near_last
        persistent_high = bool(far_min >= float(self.h120_oracle_probe_far_high_norm))
        intensification = bool(
            far_max >= float(self.h120_oracle_probe_far_high_norm)
            and far_rise >= float(self.h120_oracle_probe_intensify_margin_norm)
        )
        reintensification = bool(
            far[2] >= float(self.h120_oracle_probe_far_high_norm)
            and far[2] >= far[0] + float(self.h120_oracle_probe_intensify_margin_norm)
        )
        posture = np.zeros(2, dtype=float)
        if plant_info is not None:
            posture = np.asarray(
                plant_info.get("posture_vec_deg", posture),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        posture_abs = np.abs(posture[:2])
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        posture_balance = float(
            np.min(posture_abs) / max(float(np.max(posture_abs)), 1e-9)
        )

        self._h120_oracle_probe_posture_metric_deg = float(max_axis)
        self._h120_oracle_probe_far_max_norm = far_max
        self._h120_oracle_probe_far_min_norm = far_min
        self._h120_oracle_probe_near_last_norm = near_last
        self._h120_oracle_probe_far_rise_norm = far_rise
        self._h120_oracle_probe_far_persistent_high = persistent_high
        self._h120_oracle_probe_far_intensification = intensification
        self._h120_oracle_probe_far_reintensification = reintensification
        if persistent_high:
            self._h120_oracle_probe_reason = "far_persistent_high"
        elif intensification:
            self._h120_oracle_probe_reason = "far_intensification"
        elif reintensification:
            self._h120_oracle_probe_reason = "far_reintensification"
        else:
            self._h120_oracle_probe_reason = "no_far_high_pressure"

    def _h120_oracle_probe_far_pressure_risk(self) -> bool:
        return bool(
            self._h120_oracle_probe_far_persistent_high
            or self._h120_oracle_probe_far_intensification
            or self._h120_oracle_probe_far_reintensification
            or self._far_horizon_hidden_intensification
        )

    def _h120_oracle_probe_prepare_floor_delay(
        self,
        plant_info: dict[str, Any],
    ) -> None:
        self._h120_oracle_probe_refresh_signals(plant_info)
        self._h120_oracle_probe_effective_medium_delay_s = float(
            self.reactive_floor_medium_delay_s
        )
        if not self.h120_oracle_probe_enabled:
            return
        if self.reactive_floor_medium_delay_s <= 0.0:
            self._h120_oracle_probe_reason = "medium_delay_disabled"
            return
        if not self._h120_oracle_probe_far_pressure_risk():
            return
        short_delay = min(
            float(self.reactive_floor_medium_delay_s),
            float(self.h120_oracle_probe_short_delay_s),
        )
        if short_delay < float(self.reactive_floor_medium_delay_s):
            self._h120_oracle_probe_active = True
            self._h120_oracle_probe_short_delay_active = True
            self._h120_oracle_probe_action_effect = "shorten_medium_delay"
            self._h120_oracle_probe_effective_medium_delay_s = short_delay
            self._h120_oracle_probe_short_delay_count += 1
            self._h120_oracle_probe_count += 1

    def _h120_oracle_probe_preemptive_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        floor_refresh_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        self._h120_oracle_probe_preemptive_active = False
        self._h120_oracle_probe_delta_mean_kg = 0.0
        if not self.h120_oracle_probe_enabled:
            return False, np.zeros(2, dtype=float), "disabled"
        self._h120_oracle_probe_refresh_signals(plant_info)
        if floor_refresh_pending or self._reactive_floor_latched:
            self._h120_oracle_probe_reason = "floor_already_active"
            return False, np.zeros(2, dtype=float), "floor_already_active"
        if not self._h120_oracle_probe_far_pressure_risk():
            return False, np.zeros(2, dtype=float), self._h120_oracle_probe_reason

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        self._h120_oracle_probe_posture_metric_deg = float(max_axis)
        if max_axis < float(self.h120_oracle_probe_preemptive_enter_deg):
            self._h120_oracle_probe_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.h120_oracle_probe_preemptive_floor_deg):
            self._h120_oracle_probe_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"
        since_last = float(current_time) - float(self._h120_oracle_probe_last_preemptive_s)
        if since_last < float(self.h120_oracle_probe_cooldown_s):
            self._h120_oracle_probe_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"

        avec = action_vec("active_small", posture, self.cfg)
        delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        self._h120_oracle_probe_delta_mean_kg = delta_mean
        if delta_mean < float(self.h120_oracle_probe_min_delta_kg):
            self._h120_oracle_probe_reason = "delta_too_small"
            return False, avec, "delta_too_small"

        self._h120_oracle_probe_active = True
        self._h120_oracle_probe_preemptive_active = True
        self._h120_oracle_probe_action_effect = "preemptive_active_small"
        self._h120_oracle_probe_reason = "far_pressure_preemptive_active_small"
        self._h120_oracle_probe_preemptive_count += 1
        self._h120_oracle_probe_count += 1
        self._h120_oracle_probe_last_preemptive_s = float(current_time)
        return True, avec, "far_pressure_preemptive_active_small"

    def _h120_oracle_probe_log_fields(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}h120_oracle_probe_enabled": int(self.h120_oracle_probe_enabled),
            f"{prefix}h120_oracle_probe_active": int(self._h120_oracle_probe_active),
            f"{prefix}h120_oracle_probe_reason": self._h120_oracle_probe_reason,
            f"{prefix}h120_oracle_probe_action_effect": (
                self._h120_oracle_probe_action_effect
            ),
            f"{prefix}h120_oracle_probe_count": int(self._h120_oracle_probe_count),
            f"{prefix}h120_oracle_probe_short_delay_active": int(
                self._h120_oracle_probe_short_delay_active
            ),
            f"{prefix}h120_oracle_probe_short_delay_count": int(
                self._h120_oracle_probe_short_delay_count
            ),
            f"{prefix}h120_oracle_probe_early_stop_veto_active": int(
                self._h120_oracle_probe_early_stop_veto_active
            ),
            f"{prefix}h120_oracle_probe_early_stop_veto_count": int(
                self._h120_oracle_probe_early_stop_veto_count
            ),
            f"{prefix}h120_oracle_probe_preemptive_active": int(
                self._h120_oracle_probe_preemptive_active
            ),
            f"{prefix}h120_oracle_probe_preemptive_count": int(
                self._h120_oracle_probe_preemptive_count
            ),
            f"{prefix}h120_oracle_probe_effective_medium_delay_s": float(
                self._h120_oracle_probe_effective_medium_delay_s
            ),
            f"{prefix}h120_oracle_probe_far_persistent_high": int(
                self._h120_oracle_probe_far_persistent_high
            ),
            f"{prefix}h120_oracle_probe_far_intensification": int(
                self._h120_oracle_probe_far_intensification
            ),
            f"{prefix}h120_oracle_probe_far_reintensification": int(
                self._h120_oracle_probe_far_reintensification
            ),
            f"{prefix}h120_oracle_probe_posture_metric_deg": float(
                self._h120_oracle_probe_posture_metric_deg
            ),
            f"{prefix}h120_oracle_probe_near_last_norm": float(
                self._h120_oracle_probe_near_last_norm
            ),
            f"{prefix}h120_oracle_probe_far_min_norm": float(
                self._h120_oracle_probe_far_min_norm
            ),
            f"{prefix}h120_oracle_probe_far_max_norm": float(
                self._h120_oracle_probe_far_max_norm
            ),
            f"{prefix}h120_oracle_probe_far_rise_norm": float(
                self._h120_oracle_probe_far_rise_norm
            ),
            f"{prefix}h120_oracle_probe_delta_mean_kg": float(
                self._h120_oracle_probe_delta_mean_kg
            ),
        }

    # ------------------------------------------------------------------
    # h120 learned risk scheduler (default-off remote-risk delay/early-stop layer)
    # ------------------------------------------------------------------
    def _reset_h120_scheduler_bucket_state(self, reason: str = "not_evaluated") -> None:
        self._h120_scheduler_risk_tier = "normal"
        self._h120_scheduler_reason = str(reason)
        self._h120_scheduler_far_persistent_high_pressure = False
        self._h120_scheduler_far_intensification = False
        self._h120_scheduler_far_reintensification_after_relief = False
        self._h120_scheduler_far_direction_consistent_with_current_posture = False
        self._h120_scheduler_far_signflip_risk = False
        self._h120_scheduler_effective_medium_delay_s = float(
            self.reactive_floor_medium_delay_s
        )
        self._h120_scheduler_remote_risk_active = False
        self._h120_scheduler_early_stop_suppressed_active = False
        self._h120_scheduler_floor_episode_active_for_budget = False
        self._h120_scheduler_far_max_norm = 0.0
        self._h120_scheduler_far_min_norm = 0.0
        self._h120_scheduler_near_max_norm = 0.0
        self._h120_scheduler_near_last_norm = 0.0
        self._h120_scheduler_prefloor_active = False
        self._h120_scheduler_prefloor_reason = str(reason)
        self._h120_scheduler_prefloor_delta_mean_kg = 0.0

    def _h120_scheduler_refresh(self, plant_info: dict[str, Any] | None = None) -> None:
        """Recompute the remote-risk tier from the 60-120min far-horizon blocks.

        Idempotent per bucket: call once at the start of the floor phase. Sets
        the effective delayed-medium delay and the remote_risk flag used by the
        early_stop guard. Does NOT touch target generation, does NOT decide
        whether high posture recovers, does NOT use relief to weaken the floor.
        """
        self._reset_h120_scheduler_bucket_state("not_evaluated")
        if not self.h120_risk_scheduler_enabled:
            self._h120_scheduler_reason = "disabled"
            return
        # When enabled, the scheduler OWNS the delay; base = normal-tier delay.
        self._h120_scheduler_effective_medium_delay_s = float(
            self.h120_scheduler_delay_normal_s
        )
        floor_episode_active = bool(
            self._reactive_floor_latched or self._reactive_floor_post_exit_episode_active
        )
        if not floor_episode_active:
            self._h120_scheduler_early_stop_suppressed_this_episode = 0
        self._h120_scheduler_floor_episode_active_for_budget = floor_episode_active
        if not self.far_horizon_enabled:
            self._h120_scheduler_reason = "far_horizon_disabled"
            return
        if not self._far_horizon_available:
            self._h120_scheduler_reason = self._far_horizon_reason
            return

        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        near = [float(x) for x in norms[:3]]
        far = [float(x) for x in norms[3:6]]
        near_max = float(max(near))
        near_last = float(near[2])
        far_min = float(min(far))
        far_max = float(max(far))
        high_norm = float(self.h120_scheduler_far_high_norm)
        intensify_margin = float(self.h120_scheduler_intensify_margin_norm)
        relief_margin = float(self.h120_scheduler_relief_margin_norm)

        persistent_high = bool(far_min >= high_norm)
        intensification = bool(far_max >= near_last + intensify_margin)
        near_relief = bool(near_last <= near_max - relief_margin)
        reintensification = bool(
            near_relief and far_max >= near_last + intensify_margin
        )

        # Direction consistency: far pressure vector aligned with current
        # posture (would push attitude further in the offending direction).
        direction_consistent = False
        posture = np.zeros(2, dtype=float)
        if plant_info is not None:
            posture = np.asarray(
                plant_info.get("posture_vec_deg", posture), dtype=float
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            posture = posture[:2]
        posture_norm = float(np.linalg.norm(posture))
        far_vecs = (
            self._far_horizon_vecs[3:6]
            if len(self._far_horizon_vecs) >= 6
            else []
        )
        if far_vecs and posture_norm > 1e-6:
            far_vec_sum = np.sum(np.asarray(far_vecs, dtype=float), axis=0)
            fv_norm = float(np.linalg.norm(far_vec_sum))
            if fv_norm > 1e-6:
                cos = float(np.dot(far_vec_sum, posture) / (fv_norm * posture_norm))
                # Pure direction alignment, decoupled from magnitude so the
                # tier rule can combine it explicitly with the danger flags.
                direction_consistent = bool(cos > 0.3)

        signflip_risk = bool(
            self._far_horizon_reversal or self._far_horizon_direction_shift
        )

        # Direction-gated tier rule (controller-relevant far-risk shapes):
        #   far_danger = far pressure stays high / rises / re-intensifies.
        #   high_risk ONLY when far_danger AND the far pressure direction is
        #   aligned with the current posture (would worsen the SAME axis).
        #   far_danger with mismatched direction -> risk_aware (cautious, not
        #   full escalation): a strong remote wind that does not load the
        #   current attitude axis should not trigger aggressive medium.
        #   signflip/reversal alone -> risk_aware. Otherwise normal.
        far_danger = bool(persistent_high or intensification or reintensification)
        if far_danger and direction_consistent:
            tier = "high_risk"
        elif far_danger or signflip_risk:
            tier = "risk_aware"
        else:
            tier = "normal"

        delay_map = {
            "normal": self.h120_scheduler_delay_normal_s,
            "risk_aware": self.h120_scheduler_delay_risk_aware_s,
            "high_risk": self.h120_scheduler_delay_high_risk_s,
        }
        self._h120_scheduler_risk_tier = tier
        self._h120_scheduler_reason = "ok"
        self._h120_scheduler_far_persistent_high_pressure = persistent_high
        self._h120_scheduler_far_intensification = intensification
        self._h120_scheduler_far_reintensification_after_relief = reintensification
        self._h120_scheduler_far_direction_consistent_with_current_posture = direction_consistent
        self._h120_scheduler_far_signflip_risk = signflip_risk
        self._h120_scheduler_effective_medium_delay_s = float(delay_map[tier])
        self._h120_scheduler_remote_risk_active = bool(tier == "high_risk")
        self._h120_scheduler_far_max_norm = far_max
        self._h120_scheduler_far_min_norm = far_min
        self._h120_scheduler_near_max_norm = near_max
        self._h120_scheduler_near_last_norm = near_last
        # Counter increment is done by the single per-bucket call site
        # (before _reactive_floor_needed) to avoid double-counting when this
        # method is also called from the early_stop guard in the same bucket.

    def _h120_scheduler_try_suppress_early_stop(self) -> bool:
        """Return True if early_stop should be suppressed this bucket.

        Only when remote risk (high_risk tier) is active AND the per-episode
        suppress budget is not exhausted. Increments the suppress counters.
        Never permanently disables early_stop — bounded by max_per_episode.
        """
        if not self.h120_risk_scheduler_enabled:
            return False
        if not self._h120_scheduler_remote_risk_active:
            return False
        if (
            self._h120_scheduler_early_stop_suppressed_this_episode
            >= self.h120_scheduler_early_stop_suppress_max_per_episode
        ):
            return False
        self._h120_scheduler_early_stop_suppressed_this_episode += 1
        self._h120_scheduler_early_stop_suppressed_count += 1
        self._h120_scheduler_early_stop_suppressed_active = True
        return True

    def _h120_scheduler_prefloor_probe_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        floor_refresh_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        """Conservative pre-floor coupling probe for oracle far-risk tests.

        Default-off.  Unlike scheduler v1's floor-internal levers, this can
        refresh a small posture-oriented target before the floor enters,
        but only in the 3-5 degree band and only under high-risk, direction-
        consistent far pressure.  It never weakens the floor and never uses
        relief suppression.
        """
        self._h120_scheduler_prefloor_active = False
        self._h120_scheduler_prefloor_delta_mean_kg = 0.0
        if not (
            self.h120_risk_scheduler_enabled
            and self.h120_scheduler_prefloor_probe_enabled
        ):
            self._h120_scheduler_prefloor_reason = "disabled"
            return False, np.zeros(2, dtype=float), "disabled"
        if floor_refresh_pending or self._reactive_floor_latched:
            self._h120_scheduler_prefloor_reason = "floor_already_active"
            return False, np.zeros(2, dtype=float), "floor_already_active"
        self._h120_scheduler_refresh(plant_info)
        if not self._h120_scheduler_remote_risk_active:
            self._h120_scheduler_prefloor_reason = "not_high_risk"
            return False, np.zeros(2, dtype=float), "not_high_risk"
        if not self._h120_scheduler_far_direction_consistent_with_current_posture:
            self._h120_scheduler_prefloor_reason = "direction_not_consistent"
            return False, np.zeros(2, dtype=float), "direction_not_consistent"

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if max_axis < float(self.h120_scheduler_prefloor_enter_deg):
            self._h120_scheduler_prefloor_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.h120_scheduler_prefloor_floor_deg):
            self._h120_scheduler_prefloor_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"
        since_last = float(current_time) - float(self._h120_scheduler_prefloor_last_s)
        if since_last < float(self.h120_scheduler_prefloor_cooldown_s):
            self._h120_scheduler_prefloor_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"

        avec = action_vec(self.h120_scheduler_prefloor_action, posture, self.cfg)
        delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        if (
            float(self.h120_scheduler_prefloor_max_delta_kg) > 0.0
            and delta_mean > float(self.h120_scheduler_prefloor_max_delta_kg)
        ):
            scale = float(self.h120_scheduler_prefloor_max_delta_kg) / max(
                delta_mean, 1e-9
            )
            avec = avec * scale
            delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        self._h120_scheduler_prefloor_delta_mean_kg = delta_mean
        if delta_mean < float(self.h120_scheduler_prefloor_min_delta_kg):
            self._h120_scheduler_prefloor_reason = "delta_too_small"
            return False, avec, "delta_too_small"

        self._h120_scheduler_prefloor_active = True
        self._h120_scheduler_prefloor_reason = (
            f"prefloor_{self.h120_scheduler_prefloor_action}"
        )
        self._h120_scheduler_prefloor_count += 1
        self._h120_scheduler_prefloor_last_s = float(current_time)
        return True, avec, str(self._h120_scheduler_prefloor_reason)

    def _h120_axis_micro_prepare_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        higher_priority_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        """Corrected pre-floor axis-aware micro prepare for oracle ceiling.

        Distinct from the prefloor target-refresh probe (NO-GO): this fires
        ONLY on the direction-clean + real-rise eligible window validated by
        the Step 1 audit, and applies a small single-axis nudge (capped well
        below active_small) on the CURRENT dominant posture axis.  It never
        touches the floor, never refreshes a posture target wholesale, never
        uses the far signal to pick the axis (axis = current posture).
        """
        self._h120_axis_micro_active = False
        self._h120_axis_micro_axis = "none"
        self._h120_axis_micro_delta_mean_kg = 0.0
        if not (self.h120_axis_micro_enabled and self.far_horizon_enabled):
            self._h120_axis_micro_reason = "disabled"
            return False, np.zeros(2, dtype=float), "disabled"
        if higher_priority_pending or self._reactive_floor_latched:
            self._h120_axis_micro_reason = "floor_or_higher_active"
            return False, np.zeros(2, dtype=float), "floor_or_higher_active"

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        pitch_abs, roll_abs = abs(float(posture[0])), abs(float(posture[1]))
        max_axis = max(pitch_abs, roll_abs)
        if max_axis < float(self.h120_axis_micro_enter_deg):
            self._h120_axis_micro_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.h120_axis_micro_floor_deg):
            self._h120_axis_micro_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"

        # eligibility: pump idle + worsening (reuse floor-computed quantities)
        if not bool(self._reactive_floor_pump_idle):
            self._h120_axis_micro_reason = "pump_not_idle"
            return False, np.zeros(2, dtype=float), "pump_not_idle"
        resp = float(self._reactive_floor_current_response_deg)
        worsening = math.isfinite(resp) and resp <= -float(
            self.h120_axis_micro_worsening_eps_deg
        )
        if not worsening:
            self._h120_axis_micro_reason = "not_worsening"
            return False, np.zeros(2, dtype=float), "not_worsening"

        # direction-clean + real-rise gate (Step 1 validated)
        if self._far_horizon_reversal or self._far_horizon_direction_shift:
            self._h120_axis_micro_reason = "direction_not_clean"
            return False, np.zeros(2, dtype=float), "direction_not_clean"
        if float(self._far_horizon_far_max) < float(
            self.h120_axis_micro_rise_floor_norm
        ):
            self._h120_axis_micro_reason = "no_far_rise"
            return False, np.zeros(2, dtype=float), "no_far_rise"

        since_last = float(current_time) - float(self._h120_axis_micro_last_s)
        if since_last < float(self.h120_axis_micro_cooldown_s):
            self._h120_axis_micro_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"

        # axis-aware micro: active_small direction, single dominant axis, capped
        base = action_vec("active_small", posture, self.cfg)
        base = np.asarray(base, dtype=float).reshape(2)
        axis_micro = np.zeros(2, dtype=float)
        if pitch_abs >= roll_abs:
            axis_micro[0] = base[0]
            self._h120_axis_micro_axis = "pitch"
        else:
            axis_micro[1] = base[1]
            self._h120_axis_micro_axis = "roll"
        delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(axis_micro))))
        if delta_mean <= 0.0:
            self._h120_axis_micro_reason = "zero_action"
            return False, np.zeros(2, dtype=float), "zero_action"
        if delta_mean > float(self.h120_axis_micro_delta_kg):
            axis_micro = axis_micro * (
                float(self.h120_axis_micro_delta_kg) / max(delta_mean, 1e-9)
            )
            delta_mean = float(
                np.mean(np.abs(self._primary_mass_delta_kg(axis_micro)))
            )
        self._h120_axis_micro_delta_mean_kg = delta_mean
        self._h120_axis_micro_active = True
        self._h120_axis_micro_reason = f"axis_micro_{self._h120_axis_micro_axis}"
        self._h120_axis_micro_count += 1
        self._h120_axis_micro_last_s = float(current_time)
        return True, axis_micro, str(self._h120_axis_micro_reason)

    def _h120_axis_micro_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_axis_micro_enabled": int(self.h120_axis_micro_enabled),
            f"{p}h120_axis_micro_active": int(self._h120_axis_micro_active),
            f"{p}h120_axis_micro_reason": str(self._h120_axis_micro_reason),
            f"{p}h120_axis_micro_axis": str(self._h120_axis_micro_axis),
            f"{p}h120_axis_micro_count": int(self._h120_axis_micro_count),
            f"{p}h120_axis_micro_delta_mean_kg": float(
                self._h120_axis_micro_delta_mean_kg
            ),
        }

    def _relief_economy_delay_needed(self, plant_info: dict[str, Any]) -> bool:
        """Default-off economy lever: delay a NON-URGENT target refresh when the
        forecast shows clear relief and posture is safe.

        This is the A1/A2 oracle audit lever. It only ever DELAYS (reuses the
        current target instead of refreshing) on the non-urgent economy path; it
        never touches the reactive floor, recovery, safety fallback, or any
        high-posture action. horizon='near' uses only the 0-60min blocks (A1);
        horizon='far' additionally uses the 60-120min far blocks (A2).
        """
        self._relief_economy_active = False
        self._relief_economy_near_low = False
        self._relief_economy_far_low = False
        if not self.relief_economy_enabled:
            self._relief_economy_reason = "disabled"
            return False
        if self._reactive_floor_latched:
            self._relief_economy_reason = "floor_latched"
            return False
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)), dtype=float
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        max_axis = max(pitch_abs, roll_abs)
        posture_abs = np.abs(posture[:2])
        posture_balance = float(
            np.min(posture_abs) / max(float(np.max(posture_abs)), 1e-9)
        )
        if max_axis >= float(self.relief_economy_safe_deg):
            self._relief_economy_reason = "posture_not_safe"
            return False
        resp = float(self._active_effectiveness_current_response_deg)
        if math.isfinite(resp) and resp <= -float(self.relief_economy_worsening_eps_deg):
            self._relief_economy_reason = "posture_worsening"
            return False

        low = float(self.relief_economy_low_norm)
        high = float(self.relief_economy_high_norm)
        near = [float(x) for x in self._raw_pressure_block_norms[:3]]
        if len(near) < 3:
            self._relief_economy_reason = "no_near_blocks"
            return False
        immediate_calm = near[0] < high
        near_low = bool(immediate_calm and near[2] < low and max(near) < high)
        self._relief_economy_near_low = near_low
        relief = near_low
        if self.relief_economy_horizon == "far":
            far = [float(x) for x in list(self._far_horizon_norms)[3:6]]
            far_low = bool(
                immediate_calm and len(far) == 3 and min(far) < low and max(far) < high
            )
            self._relief_economy_far_low = far_low
            relief = near_low or far_low
        if not relief:
            self._relief_economy_reason = "no_relief"
            return False
        self._relief_economy_active = True
        self._relief_economy_reason = (
            "relief_far"
            if (self.relief_economy_horizon == "far" and self._relief_economy_far_low and not near_low)
            else "relief_near"
        )
        return True

    def _relief_economy_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}relief_economy_enabled": int(self.relief_economy_enabled),
            f"{p}relief_economy_horizon": str(self.relief_economy_horizon),
            f"{p}relief_economy_active": int(self._relief_economy_active),
            f"{p}relief_economy_reason": str(self._relief_economy_reason),
            f"{p}relief_economy_near_low": int(self._relief_economy_near_low),
            f"{p}relief_economy_far_low": int(self._relief_economy_far_low),
            f"{p}relief_economy_delay_count": int(self._relief_economy_delay_count),
        }

    def _economy_budget_observe(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        *,
        hard_domain: bool,
    ) -> None:
        """Integrate pump use into the economy budget outside hard domains."""
        if not self.economy_pump_budget_enabled:
            self._economy_pump_budget_reason = "disabled"
            self._economy_pump_budget_hold_latched = False
            self._economy_pump_budget_last_s = float(current_time)
            return
        now = float(current_time)
        if (
            str(self.economy_pump_budget_allocator_mode)
            in ("neutral_mhs_clean_auto", "dual_specialist_auto")
            and not self._economy_pump_budget_clean_start_checked
        ):
            posture = np.asarray(
                plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            initial_max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
            self._economy_pump_budget_initial_max_axis_deg = float(initial_max_axis)
            self._economy_pump_budget_clean_start_eligible = bool(
                (not hard_domain) and initial_max_axis <= 0.06
            )
            self._economy_pump_budget_clean_start_checked = True
        if self._economy_pump_budget_last_s is None:
            self._economy_pump_budget_last_s = now
            return
        dt = max(0.0, now - float(self._economy_pump_budget_last_s))
        self._economy_pump_budget_last_s = now
        if hard_domain:
            self._economy_pump_budget_hold_latched = False
            return
        pump_rate = np.asarray(
            plant_info.get("pump_rate_cmd_m3_min", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        pump_m3 = float(np.sum(np.abs(pump_rate))) * dt / 60.0
        self._economy_pump_budget_used_m3 += pump_m3
        if now <= float(self.economy_pump_budget_opportunity_window_s):
            self._economy_pump_budget_opportunity_pump_m3 += pump_m3

    def _economy_budget_allows_refresh(
        self,
        *,
        would_refresh: bool,
        refresh_avec: np.ndarray | None = None,
        plant_info: dict[str, Any] | None = None,
    ) -> bool:
        self._economy_pump_budget_active = False
        self._economy_pump_budget_blocked_delta_mean_kg = 0.0
        if not self.economy_pump_budget_enabled:
            self._economy_pump_budget_reason = "disabled"
            self._economy_pump_budget_hold_latched = False
            return True
        if self.economy_pump_budget_m3 <= 0.0:
            self._economy_pump_budget_reason = "no_budget_configured"
            self._economy_pump_budget_hold_latched = False
            return True
        if not would_refresh:
            self._economy_pump_budget_reason = "no_refresh"
            return True
        if self._economy_budget_in_opportunity_window():
            self._economy_pump_budget_opportunity_refresh_count += 1
            if refresh_avec is not None:
                self._economy_pump_budget_opportunity_delta_sum_kg += float(
                    np.mean(np.abs(self._primary_mass_delta_kg(refresh_avec)))
                )
        info = plant_info or {}
        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)), dtype=float
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        hard_domain = bool(
            self._reactive_floor_latched
            or max_axis >= float(self.high_posture_enter)
            or int(info.get("pump_fullspeed_any", 0)) > 0
        )
        if hard_domain:
            self._economy_pump_budget_refresh_count += 1
            self._economy_pump_budget_reason = "hard_domain_release"
            self._economy_pump_budget_hold_latched = False
            return True
        if self._economy_pump_budget_used_m3 < self.economy_pump_budget_m3:
            self._economy_pump_budget_refresh_count += 1
            self._economy_pump_budget_reason = "budget_available"
            self._economy_pump_budget_hold_latched = False
            return True
        allow_hold, reason = self._economy_budget_allocator_decision(plant_info)
        if not allow_hold:
            self._economy_pump_budget_refresh_count += 1
            self._economy_pump_budget_reason = reason
            self._economy_pump_budget_hold_latched = False
            return True
        self._economy_pump_budget_active = True
        self._economy_pump_budget_hold_latched = True
        self._economy_pump_budget_hold_count += 1
        if refresh_avec is not None:
            self._economy_pump_budget_blocked_delta_mean_kg = float(
                np.mean(np.abs(self._primary_mass_delta_kg(refresh_avec)))
            )
        self._economy_pump_budget_reason = reason
        return False

    def _economy_budget_in_opportunity_window(self) -> bool:
        if self.economy_pump_budget_opportunity_window_s <= 0.0:
            return False
        if self._economy_pump_budget_last_s is None:
            return True
        return float(self._economy_pump_budget_last_s) <= float(
            self.economy_pump_budget_opportunity_window_s
        )

    def _update_relief_decay_auto_v2_latch(
        self,
        plant_info: dict[str, Any] | None,
    ) -> None:
        if str(self.economy_pump_budget_allocator_mode) not in (
            "relief_decay_auto_v2",
            "relief_decay_episode_auto_v1",
            "dual_specialist_auto",
        ):
            return
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        near0 = float(self._raw_pressure_block_norms[0])
        near1 = float(self._raw_pressure_block_norms[1])
        near2 = float(self._raw_pressure_block_norms[2])
        high_then_decay_shape = bool(
            max(near0, near1) >= 0.65
            and (max(near0, near1) - near2)
            >= float(self.economy_pump_budget_relief_drop_norm)
        )
        rise_then_fall = bool(
            near1 >= near0 + 0.10
            and near2 <= near1 - float(self.economy_pump_budget_relief_drop_norm)
        )
        episode_style = str(self.economy_pump_budget_allocator_mode) in (
            "relief_decay_episode_auto_v1",
            "dual_specialist_auto",
        )
        if episode_style:
            # The episode-level selector is intentionally less dependent on
            # the soft learned drop detector, which can fire briefly in
            # sign-flip boundary cases.  It uses either the strict casebook-like
            # decay shape or the early high/stable episode entry below.
            wind_speed_relief_decay = bool(
                self._forecast_speed_shape_available
                and self._forecast_speed_relief_decay_strict_candidate
            )
        else:
            wind_speed_relief_decay = bool(
                self._forecast_speed_shape_available
                and self._forecast_speed_relief_decay_candidate
            )
        wind_speed_episode_entry = bool(
            episode_style
            and self._forecast_speed_shape_available
            and self._forecast_speed_early_max_ms >= 18.0
            and self._forecast_speed_dir_shift_deg < 45.0
        )
        direction_ok = bool(
            (not self._far_horizon_reversal)
            and (not self._far_horizon_direction_shift)
            and (not self._h120_scheduler_far_signflip_risk)
            and self._relief_envelope_direction_ok(posture)
        )
        reintensify = bool(
            near2 >= float(self.economy_pump_budget_smart_pressure_norm)
            and near2 > near0 + 0.05
        )
        far_boundary = bool(
            self._far_horizon_reversal
            or self._far_horizon_direction_shift
            or self._far_horizon_hidden_intensification
            or self._h120_scheduler_far_signflip_risk
            or self._h120_scheduler_far_reintensification_after_relief
            or self._h120_oracle_probe_far_reintensification
        )
        fallback_guard_ok = bool(
            not (
                self._active_effectiveness_fallback_dominated
                or self._hold_relief_debt_fallback_dominated
            )
        )
        if far_boundary or reintensify or not direction_ok or not fallback_guard_ok:
            if far_boundary:
                self._relief_decay_auto_v2_latch_reason = "far_boundary_release"
            elif reintensify:
                self._relief_decay_auto_v2_latch_reason = "reintensification_release"
            elif not direction_ok:
                self._relief_decay_auto_v2_latch_reason = "direction_veto_release"
            else:
                self._relief_decay_auto_v2_latch_reason = "fallback_veto_release"
            self._relief_decay_auto_v2_latched = False
            return
        if max_axis >= float(self.economy_pump_budget_smart_posture_deg):
            return
        if (
            wind_speed_relief_decay
            or wind_speed_episode_entry
            or high_then_decay_shape
            or rise_then_fall
        ):
            if not self._relief_decay_auto_v2_latched:
                self._relief_decay_auto_v2_latch_count += 1
            if wind_speed_relief_decay:
                self._relief_decay_auto_v2_latch_reason = "wind_speed_decay"
            elif wind_speed_episode_entry:
                self._relief_decay_auto_v2_latch_reason = "wind_speed_high_stable_episode"
            elif high_then_decay_shape:
                self._relief_decay_auto_v2_latch_reason = "high_then_decay"
            else:
                self._relief_decay_auto_v2_latch_reason = "rise_then_fall"
            self._relief_decay_auto_v2_latched = True

    def _economy_budget_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        remaining = (
            float(self.economy_pump_budget_m3) - float(self._economy_pump_budget_used_m3)
            if self.economy_pump_budget_enabled and self.economy_pump_budget_m3 > 0.0
            else 0.0
        )
        return {
            f"{p}economy_pump_budget_enabled": int(self.economy_pump_budget_enabled),
            f"{p}economy_pump_budget_m3": float(self.economy_pump_budget_m3),
            f"{p}economy_pump_budget_hold_refresh_fraction": float(
                self.economy_pump_budget_hold_refresh_fraction
            ),
            f"{p}economy_pump_budget_dynamic_refresh_enabled": int(
                self.economy_pump_budget_dynamic_refresh_enabled
            ),
            f"{p}economy_pump_budget_dynamic_refresh_enter_deg": float(
                self.economy_pump_budget_dynamic_refresh_enter_deg
            ),
            f"{p}economy_pump_budget_dynamic_refresh_full_deg": float(
                self.economy_pump_budget_dynamic_refresh_full_deg
            ),
            f"{p}economy_pump_budget_dynamic_refresh_max_fraction": float(
                self.economy_pump_budget_dynamic_refresh_max_fraction
            ),
            f"{p}economy_pump_budget_allocator_mode": str(
                self.economy_pump_budget_allocator_mode
            ),
            f"{p}economy_pump_budget_require_relief": int(
                self.economy_pump_budget_require_relief
            ),
            f"{p}economy_pump_budget_relief_norm": float(
                self.economy_pump_budget_relief_norm
            ),
            f"{p}economy_pump_budget_relief_drop_norm": float(
                self.economy_pump_budget_relief_drop_norm
            ),
            f"{p}economy_pump_budget_forecast_smart": int(
                self.economy_pump_budget_forecast_smart
            ),
            f"{p}economy_pump_budget_smart_posture_deg": float(
                self.economy_pump_budget_smart_posture_deg
            ),
            f"{p}economy_pump_budget_smart_pressure_norm": float(
                self.economy_pump_budget_smart_pressure_norm
            ),
            f"{p}economy_pump_budget_opportunity_window_s": float(
                self.economy_pump_budget_opportunity_window_s
            ),
            f"{p}economy_pump_budget_opportunity_min_pump_m3": float(
                self.economy_pump_budget_opportunity_min_pump_m3
            ),
            f"{p}economy_pump_budget_opportunity_min_refresh": int(
                self.economy_pump_budget_opportunity_min_refresh
            ),
            f"{p}economy_pump_budget_opportunity_min_delta_kg": float(
                self.economy_pump_budget_opportunity_min_delta_kg
            ),
            f"{p}economy_pump_budget_used_m3": float(
                self._economy_pump_budget_used_m3
            ),
            f"{p}economy_pump_budget_remaining_m3": float(remaining),
            f"{p}economy_pump_budget_active": int(self._economy_pump_budget_active),
            f"{p}economy_pump_budget_reason": str(self._economy_pump_budget_reason),
            f"{p}economy_pump_budget_hold_count": int(
                self._economy_pump_budget_hold_count
            ),
            f"{p}economy_pump_budget_refresh_count": int(
                self._economy_pump_budget_refresh_count
            ),
            f"{p}economy_pump_budget_partial_refresh_count": int(
                self._economy_pump_budget_partial_refresh_count
            ),
            f"{p}economy_pump_budget_partial_refresh_fraction": float(
                self._economy_pump_budget_partial_refresh_fraction
            ),
            f"{p}economy_pump_budget_partial_delta_mean_kg": float(
                self._economy_pump_budget_partial_delta_mean_kg
            ),
            f"{p}economy_pump_budget_blocked_delta_mean_kg": float(
                self._economy_pump_budget_blocked_delta_mean_kg
            ),
            f"{p}economy_pump_budget_opportunity_pump_m3": float(
                self._economy_pump_budget_opportunity_pump_m3
            ),
            f"{p}economy_pump_budget_opportunity_refresh_count": int(
                self._economy_pump_budget_opportunity_refresh_count
            ),
            f"{p}economy_pump_budget_opportunity_delta_sum_kg": float(
                self._economy_pump_budget_opportunity_delta_sum_kg
            ),
            f"{p}economy_pump_budget_opportunity_score": float(
                self._economy_pump_budget_opportunity_score
            ),
            f"{p}economy_pump_budget_opportunity_active": int(
                self._economy_pump_budget_opportunity_active
            ),
            f"{p}economy_pump_budget_plateau_detected": int(
                self._economy_pump_budget_plateau_detected
            ),
            f"{p}economy_pump_budget_plateau_confidence_proxy": float(
                self._economy_pump_budget_plateau_confidence
            ),
            f"{p}economy_pump_budget_boundary_veto_reason": str(
                self._economy_pump_budget_boundary_veto_reason
            ),
            f"{p}economy_pump_budget_clean_start_checked": int(
                self._economy_pump_budget_clean_start_checked
            ),
            f"{p}economy_pump_budget_clean_start_eligible": int(
                self._economy_pump_budget_clean_start_eligible
            ),
            f"{p}economy_pump_budget_initial_max_axis_deg": float(
                self._economy_pump_budget_initial_max_axis_deg
            ),
            f"{p}relief_decay_auto_v2_latched": int(
                self._relief_decay_auto_v2_latched
            ),
            f"{p}relief_decay_auto_v2_latch_reason": str(
                self._relief_decay_auto_v2_latch_reason
            ),
            f"{p}relief_decay_auto_v2_latch_count": int(
                self._relief_decay_auto_v2_latch_count
            ),
            f"{p}economy_pump_budget_hold_latched": int(
                self._economy_pump_budget_hold_latched
            ),
            f"{p}economy_pump_budget_forced_release_count": int(
                self._economy_pump_budget_forced_release_count
            ),
            f"{p}economy_pump_budget_forced_release_reason": str(
                self._economy_pump_budget_forced_release_reason
            ),
        }

    def _economy_budget_allocator_decision(
        self,
        plant_info: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        """Return whether an exhausted-budget hold has a valid reason."""
        mode = str(self.economy_pump_budget_allocator_mode)
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        max_axis = max(pitch_abs, roll_abs)
        posture_abs = np.abs(posture[:2])
        posture_balance = float(
            np.min(posture_abs) / max(float(np.max(posture_abs)), 1e-9)
        )
        near0 = float(self._raw_pressure_block_norms[0])
        near1 = float(self._raw_pressure_block_norms[1])
        near2 = float(self._raw_pressure_block_norms[2])
        raw_vecs = [
            np.asarray(v, dtype=float).reshape(-1)
            for v in self._raw_pressure_block_vecs[:3]
        ]
        while len(raw_vecs) < 3:
            raw_vecs.append(np.zeros(2, dtype=float))
        vec0 = raw_vecs[0][:2]
        vec2 = raw_vecs[2][:2]
        if vec0.size < 2:
            vec0 = np.pad(vec0, (0, 2 - vec0.size))
        if vec2.size < 2:
            vec2 = np.pad(vec2, (0, 2 - vec2.size))
        dot02 = float(np.dot(vec0, vec2))
        pressure_max = max(near0, near1, near2)
        pressure_min = min(near0, near1, near2)
        pressure_range = pressure_max - pressure_min
        relief = bool(
            near2 <= float(self.economy_pump_budget_relief_norm)
            and (near0 - near2) >= float(self.economy_pump_budget_relief_drop_norm)
        )
        high_then_relief = bool(
            max(near0, near1) >= float(self.economy_pump_budget_smart_pressure_norm)
            and near2 <= float(self.economy_pump_budget_relief_norm)
            and (max(near0, near1) - near2)
            >= float(self.economy_pump_budget_relief_drop_norm)
        )
        high_then_decay_shape = bool(
            max(near0, near1) >= 0.65
            and (max(near0, near1) - near2)
            >= float(self.economy_pump_budget_relief_drop_norm)
        )
        rise_then_fall = bool(
            near1 >= near0 + 0.10
            and near2 <= near1 - float(self.economy_pump_budget_relief_drop_norm)
        )
        episode_style = mode in ("relief_decay_episode_auto_v1", "dual_specialist_auto")
        if episode_style:
            # Keep the episode selector from being opened by the soft learned
            # drop detector alone; boundary smoke tests showed that signal can
            # appear for one bucket in sign-flip cases.  The episode entry
            # itself is handled separately by wind_speed_episode_entry.
            wind_speed_relief_decay = bool(
                self._forecast_speed_shape_available
                and self._forecast_speed_relief_decay_strict_candidate
            )
        else:
            wind_speed_relief_decay = bool(
                self._forecast_speed_shape_available
                and self._forecast_speed_relief_decay_candidate
            )
        wind_speed_episode_entry = bool(
            episode_style
            and self._forecast_speed_shape_available
            and self._forecast_speed_early_max_ms >= 18.0
            and self._forecast_speed_dir_shift_deg < 45.0
        )
        direction_ok = bool(
            (not self._far_horizon_reversal)
            and (not self._far_horizon_direction_shift)
            and (not self._h120_scheduler_far_signflip_risk)
            and self._relief_envelope_direction_ok(posture)
        )
        reintensify = bool(
            near2 >= float(self.economy_pump_budget_smart_pressure_norm)
            and near2 > near0 + 0.05
        )
        far_boundary = bool(
            self._far_horizon_reversal
            or self._far_horizon_direction_shift
            or self._far_horizon_hidden_intensification
            or self._h120_scheduler_far_signflip_risk
            or self._h120_scheduler_far_reintensification_after_relief
            or self._h120_oracle_probe_far_reintensification
        )
        floor_risk = bool(
            max_axis >= float(self.economy_pump_budget_smart_posture_deg)
            or pressure_max >= float(self.economy_pump_budget_smart_pressure_norm)
        )
        posture_floor_risk = bool(
            max_axis >= float(self.economy_pump_budget_smart_posture_deg)
        )
        fallback_guard_ok = bool(
            not (
                self._active_effectiveness_fallback_dominated
                or self._hold_relief_debt_fallback_dominated
            )
        )
        lowrisk_redundant = bool(
            max_axis < 3.0
            and pressure_max <= float(self.economy_pump_budget_relief_norm)
            and direction_ok
        )
        residual_high_economy = bool(
            max_axis < float(self.economy_pump_budget_smart_posture_deg)
            and pressure_max < float(self.economy_pump_budget_smart_pressure_norm)
            and near0 >= 0.65
            and not reintensify
            and direction_ok
        )
        residual_high_stable = bool(
            max_axis < float(self.economy_pump_budget_smart_posture_deg)
            and pressure_max < float(self.economy_pump_budget_smart_pressure_norm)
            and max(near0, near1) >= 0.65
            and near2 <= max(near0, near1) + 0.05
            and not reintensify
            and direction_ok
        )
        residual_high_plateau = bool(
            max_axis < float(self.economy_pump_budget_smart_posture_deg)
            and min(near0, near1, near2)
            >= float(self.economy_pump_budget_smart_pressure_norm)
            and pressure_range <= max(
                0.15,
                0.75 * float(self.economy_pump_budget_relief_drop_norm),
            )
            and near2 <= near0 + 0.10
            and not reintensify
            and direction_ok
        )
        residual_high_plateau_strict = bool(
            residual_high_plateau
            and posture_balance >= 0.85
            and max_axis >= 1.0
        )
        psc_v2_short_ok = bool(
            self._psc_v2_short_available
            and self._psc_v2_short_dir_shift_deg <= 8.0
        )
        psc_v2_neutral_frame = bool(
            residual_high_plateau
            and psc_v2_short_ok
            and 14.5 <= self._psc_v2_short_max_ms <= 19.5
            and self._psc_v2_short_range_ms <= 0.80
        )
        psc_v2_neutral_v2c_balanced_roll = bool(
            1.00 <= pitch_abs <= 1.45
            and roll_abs >= 2.20
            and roll_abs >= pitch_abs + 0.75
            and 16.0 <= self._psc_v2_short_max_ms <= 16.7
            and 0.22 <= self._psc_v2_short_range_ms <= 0.32
            and self._psc_v2_short_peak_to_late_mean_drop_ms >= 0.18
        )
        psc_v2_neutral_v2c_roll_load = bool(
            1.80 <= pitch_abs <= 2.10
            and roll_abs >= 2.30
            and roll_abs >= pitch_abs + 0.20
            and 16.8 <= self._psc_v2_short_max_ms <= 17.5
            and self._psc_v2_short_range_ms <= 0.22
            and self._psc_v2_short_peak_to_late_mean_drop_ms >= 0.10
        )
        psc_v2_neutral_v2c = bool(
            psc_v2_neutral_frame
            and (
                psc_v2_neutral_v2c_balanced_roll
                or psc_v2_neutral_v2c_roll_load
            )
        )
        psc_v2_base_neutral = bool(
            psc_v2_neutral_v2c
        )
        psc_v2_lowrisk = bool(
            lowrisk_redundant
            and (
                not self._psc_v2_short_available
                or self._psc_v2_short_max_ms <= 2.25
            )
        )
        psc_v2_p1_short_low_peak = bool(
            psc_v2_short_ok
            and pressure_max >= 0.90
            and 14.5 <= self._psc_v2_short_max_ms <= 16.8
            and 0.20 <= self._psc_v2_short_range_ms <= 0.90
        )
        psc_v2_p1_short_high_peak = bool(
            psc_v2_short_ok
            and pressure_max >= 0.90
            and 20.0 <= self._psc_v2_short_max_ms <= 23.5
            and 0.35 <= self._psc_v2_short_range_ms <= 1.45
        )
        psc_v2_p1 = bool(
            psc_v2_p1_short_low_peak or psc_v2_p1_short_high_peak
        )
        psc_v2_w1 = bool(
            self._psc_v2_short_available
            and self._psc_v2_short_max_ms <= 14.5
            and self._psc_v2_short_dir_shift_deg <= 20.0
            and (
                pressure_max <= 0.35
                or self._psc_v2_short_max_ms >= 13.0
            )
        )
        psc_v2_w2_flat = bool(
            self._psc_v2_short_available
            and pressure_max >= float(self.economy_pump_budget_smart_pressure_norm)
            and 18.0 <= self._psc_v2_short_max_ms <= 20.0
            and self._psc_v2_short_range_ms <= 0.45
            and self._psc_v2_short_dir_shift_deg <= 5.0
            and self._psc_v2_short_peak_to_late_mean_drop_ms >= 0.0
        )
        psc_v2_w2_decay = bool(
            self._psc_v2_short_available
            and pressure_max >= float(self.economy_pump_budget_smart_pressure_norm)
            and 16.0 <= self._psc_v2_short_max_ms <= 18.3
            and 0.45 <= self._psc_v2_short_range_ms <= 0.95
            and self._psc_v2_short_peak_to_late_mean_drop_ms >= 0.50
            and self._psc_v2_short_dir_shift_deg <= 6.0
        )
        psc_v2_w2 = bool(psc_v2_w2_flat or psc_v2_w2_decay)
        plateau_confidence = 0.0
        plateau_confidence += 1.0 if min(near0, near1, near2) >= float(
            self.economy_pump_budget_smart_pressure_norm
        ) else 0.0
        plateau_confidence += 1.0 if pressure_range <= 0.15 else 0.0
        plateau_confidence += 1.0 if near2 <= near0 + 0.10 else 0.0
        plateau_confidence += 1.0 if max_axis < float(
            self.economy_pump_budget_smart_posture_deg
        ) else 0.0
        plateau_confidence += 1.0 if direction_ok else 0.0
        self._economy_pump_budget_plateau_detected = bool(residual_high_plateau)
        self._economy_pump_budget_plateau_confidence = float(plateau_confidence / 5.0)
        self._economy_pump_budget_boundary_veto_reason = "none"
        if far_boundary:
            self._economy_pump_budget_boundary_veto_reason = "far_boundary"
        elif not direction_ok:
            self._economy_pump_budget_boundary_veto_reason = "direction"
        elif reintensify:
            self._economy_pump_budget_boundary_veto_reason = "reintensification"
        elif not fallback_guard_ok:
            self._economy_pump_budget_boundary_veto_reason = "fallback"
        elif posture_floor_risk:
            self._economy_pump_budget_boundary_veto_reason = "posture_floor"
        if mode == "budget":
            if self.economy_pump_budget_forecast_smart:
                if floor_risk:
                    return False, "budget_smart_floor_risk_release"
                if reintensify:
                    return False, "budget_smart_reintensification_release"
            return True, "budget_exhausted_hold"
        if mode == "relief_decay_auto":
            if floor_risk:
                return False, "relief_decay_auto_floor_risk_release"
            if reintensify:
                return False, "relief_decay_auto_reintensification_release"
            if not direction_ok:
                return False, "relief_decay_auto_direction_veto_release"
            if high_then_relief:
                return True, "relief_decay_auto_hold:high_then_relief"
            if rise_then_fall:
                return True, "relief_decay_auto_hold:rise_then_fall"
            return False, "relief_decay_auto_no_opportunity_release"
        if mode in ("relief_decay_auto_v2", "relief_decay_episode_auto_v1"):
            # v1 accidentally treated high near-pressure as a release signal,
            # which conflicts with the target regime: high/rising load followed
            # by forecast relief.  v2 releases only on posture proximity to the
            # hard floor or true boundary signals; hard-domain logic above
            # still overrides before this branch.
            opportunity_now = bool(
                wind_speed_relief_decay
                or wind_speed_episode_entry
                or high_then_decay_shape
                or rise_then_fall
            )
            if opportunity_now and direction_ok and not reintensify and fallback_guard_ok:
                if not self._relief_decay_auto_v2_latched:
                    self._relief_decay_auto_v2_latch_count += 1
                if wind_speed_relief_decay:
                    self._relief_decay_auto_v2_latch_reason = "wind_speed_decay"
                elif wind_speed_episode_entry:
                    self._relief_decay_auto_v2_latch_reason = (
                        "wind_speed_high_stable_episode"
                    )
                elif high_then_decay_shape:
                    self._relief_decay_auto_v2_latch_reason = "high_then_decay"
                else:
                    self._relief_decay_auto_v2_latch_reason = "rise_then_fall"
                self._relief_decay_auto_v2_latched = True
            if posture_floor_risk:
                return False, "relief_decay_auto_v2_posture_floor_release"
            if reintensify:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "reintensification_release"
                return False, "relief_decay_auto_v2_reintensification_release"
            if not direction_ok:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "direction_veto_release"
                return False, "relief_decay_auto_v2_direction_veto_release"
            if not fallback_guard_ok:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "fallback_veto_release"
                return False, "relief_decay_auto_v2_fallback_veto_release"
            if self._relief_decay_auto_v2_latched:
                return True, (
                    "relief_decay_auto_v2_hold:"
                    f"{self._relief_decay_auto_v2_latch_reason}"
                )
            return False, "relief_decay_auto_v2_no_opportunity_release"
        if mode == "residual_high_auto":
            if floor_risk:
                return False, "residual_high_auto_floor_risk_release"
            if reintensify:
                return False, "residual_high_auto_reintensification_release"
            if not direction_ok:
                return False, "residual_high_auto_direction_veto_release"
            if not fallback_guard_ok:
                return False, "residual_high_auto_fallback_veto_release"
            if residual_high_stable:
                return True, "residual_high_auto_hold:stable_high_load"
            return False, "residual_high_auto_no_opportunity_release"
        if mode in (
            "residual_high_plateau_auto",
            "residual_high_plateau_strict_auto",
            "neutral_mhs_broader_auto",
        ):
            if posture_floor_risk:
                return False, f"{mode}_posture_floor_release"
            if reintensify:
                return False, f"{mode}_reintensification_release"
            if far_boundary:
                return False, f"{mode}_boundary_veto_release"
            if not direction_ok:
                return False, f"{mode}_direction_veto_release"
            if not fallback_guard_ok:
                return False, f"{mode}_fallback_veto_release"
            if mode == "residual_high_plateau_strict_auto":
                if residual_high_plateau_strict:
                    return True, f"{mode}_hold:balanced_stable_plateau"
                return False, f"{mode}_no_balanced_plateau_release"
            if mode == "neutral_mhs_broader_auto":
                if residual_high_plateau:
                    return True, f"{mode}_hold:neutral_headroom_stable_plateau"
                return False, f"{mode}_no_neutral_headroom_plateau_release"
            if residual_high_plateau:
                return True, f"{mode}_hold:stable_plateau"
            return False, f"{mode}_no_plateau_release"
        if mode == "neutral_mhs_clean_auto":
            if not bool(self._forecast_has_future):
                return False, "neutral_mhs_clean_no_future_preview_release"
            if not self._economy_pump_budget_clean_start_checked:
                return False, "neutral_mhs_clean_start_unchecked_release"
            if not self._economy_pump_budget_clean_start_eligible:
                return False, "neutral_mhs_clean_start_veto_release"
            if posture_floor_risk:
                return False, "neutral_mhs_clean_posture_floor_release"
            if reintensify:
                return False, "neutral_mhs_clean_reintensification_release"
            if far_boundary:
                return False, "neutral_mhs_clean_boundary_veto_release"
            if not direction_ok:
                return False, "neutral_mhs_clean_direction_veto_release"
            if not fallback_guard_ok:
                return False, "neutral_mhs_clean_fallback_veto_release"
            if residual_high_plateau:
                return True, "neutral_mhs_clean_hold:clean_start_stable_plateau"
            return False, "neutral_mhs_clean_no_plateau_release"
        if mode == "psc_4hao_regime_addback_v2_auto":
            if not bool(self._forecast_has_future):
                return False, "psc_v2_no_future_preview_release"
            if posture_floor_risk:
                return False, "psc_v2_posture_floor_release"
            if not fallback_guard_ok:
                return False, "psc_v2_fallback_veto_release"
            if psc_v2_base_neutral:
                return True, "psc_v2_hold:neutral_mhs_broader_short6"
            if psc_v2_neutral_frame:
                return False, "psc_v2_neutral_v2c_veto_release"
            if psc_v2_lowrisk:
                return True, "psc_v2_hold:lowrisk_redundant"
            if psc_v2_p1 and not reintensify and not far_boundary and direction_ok:
                if psc_v2_p1_short_high_peak:
                    return True, "psc_v2_hold:p1_short6_high_peak"
                return True, "psc_v2_hold:p1_short6_low_peak"
            if psc_v2_w1:
                return True, "psc_v2_hold:w1_short6_low_pressure_boundary"
            if psc_v2_w2 and not far_boundary:
                if psc_v2_w2_flat:
                    return True, "psc_v2_hold:w2_short6_flat_high"
                return True, "psc_v2_hold:w2_short6_decay_high"
            if far_boundary:
                return False, "psc_v2_boundary_veto_release"
            if reintensify:
                return False, "psc_v2_reintensification_release"
            if not direction_ok:
                return False, "psc_v2_direction_veto_release"
            return False, "psc_v2_fail_closed_release"
        if mode == "dual_specialist_auto":
            relief_opportunity_now = bool(
                wind_speed_relief_decay
                or wind_speed_episode_entry
                or high_then_decay_shape
                or rise_then_fall
            )
            if relief_opportunity_now and direction_ok and not reintensify and fallback_guard_ok:
                if not self._relief_decay_auto_v2_latched:
                    self._relief_decay_auto_v2_latch_count += 1
                if wind_speed_relief_decay:
                    self._relief_decay_auto_v2_latch_reason = "wind_speed_decay"
                elif wind_speed_episode_entry:
                    self._relief_decay_auto_v2_latch_reason = (
                        "wind_speed_high_stable_episode"
                    )
                elif high_then_decay_shape:
                    self._relief_decay_auto_v2_latch_reason = "high_then_decay"
                else:
                    self._relief_decay_auto_v2_latch_reason = "rise_then_fall"
                self._relief_decay_auto_v2_latched = True
            if posture_floor_risk:
                self._relief_decay_auto_v2_latched = False
                return False, "dual_specialist_posture_floor_release"
            if reintensify:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "reintensification_release"
                return False, "dual_specialist_reintensification_release"
            if far_boundary:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "far_boundary_release"
                return False, "dual_specialist_boundary_veto_release"
            if not direction_ok:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "direction_veto_release"
                return False, "dual_specialist_direction_veto_release"
            if not fallback_guard_ok:
                self._relief_decay_auto_v2_latched = False
                self._relief_decay_auto_v2_latch_reason = "fallback_veto_release"
                return False, "dual_specialist_fallback_veto_release"
            if self._relief_decay_auto_v2_latched:
                return True, (
                    "dual_specialist_hold:relief_decay:"
                    f"{self._relief_decay_auto_v2_latch_reason}"
                )
            if not self._economy_pump_budget_clean_start_checked:
                return False, "dual_specialist_clean_start_unchecked_release"
            if (
                self._economy_pump_budget_clean_start_eligible
                and residual_high_plateau_strict
            ):
                return True, "dual_specialist_hold:neutral_mhs_clean_plateau_strict"
            if self._economy_pump_budget_clean_start_eligible:
                return False, "dual_specialist_clean_no_plateau_release"
            return False, "dual_specialist_no_opportunity_release"
        if mode == "gusty_oscillation_budget_auto":
            # Gusty windows are not steady relief.  The saving mechanism is to
            # avoid chasing repeated short peaks after a modest economy budget
            # has already been spent, while the unchanged hard floor still owns
            # true safety recovery.  High pressure alone is not a release here;
            # posture proximity, reversal, re-intensification, and fallback are.
            if not bool(self._forecast_has_future):
                return False, "gusty_budget_no_future_preview_release"
            if posture_floor_risk:
                return False, "gusty_budget_posture_floor_release"
            if far_boundary:
                return False, "gusty_budget_boundary_veto_release"
            if not direction_ok:
                return False, "gusty_budget_direction_veto_release"
            if reintensify:
                return False, "gusty_budget_reintensification_release"
            if not fallback_guard_ok:
                return False, "gusty_budget_fallback_veto_release"
            if pressure_max < 0.65:
                return False, "gusty_budget_low_pump_opportunity_release"
            if self._forecast_speed_gusty_oscillation_candidate:
                return True, "gusty_budget_hold:repeated_forecast_peaks"
            return False, "gusty_budget_no_oscillation_release"
        if mode in (
            "direction_reversal_avoidance_auto",
            "direction_reversal_layered_auto",
            "direction_reversal_layered_v5_auto",
            "direction_reversal_layered_v6_auto",
            "direction_reversal_layered_v7_auto",
            "direction_reversal_layered_v8_auto",
        ):
            # Direction-reversal windows are usually a boundary for broad holds,
            # but they can also contain avoidable pump: do not chase a refresh
            # in the current direction when the forecast says the pressure
            # vector is about to reverse.  The layered variant adds one more
            # distinction: low-posture reversal can save pump, but high-posture
            # reversal is a warning regime and must release to normal control.
            layered_mode = mode in (
                "direction_reversal_layered_auto",
                "direction_reversal_layered_v5_auto",
                "direction_reversal_layered_v6_auto",
                "direction_reversal_layered_v7_auto",
                "direction_reversal_layered_v8_auto",
            )
            future_reversal = bool(
                bool(self._forecast_has_future)
                and dot02 < 0.0
                and pressure_max >= 0.65
            )
            posture_limit = float(self.economy_pump_budget_smart_posture_deg)
            if layered_mode:
                posture_limit = min(posture_limit, 2.50)
            posture_safe = bool(
                max_axis < posture_limit
            )
            posture_axis_relieved = True
            if mode in (
                "direction_reversal_layered_v5_auto",
                "direction_reversal_layered_v6_auto",
                "direction_reversal_layered_v7_auto",
                "direction_reversal_layered_v8_auto",
            ) and max_axis >= 1.0:
                posture_mag = float(np.linalg.norm(posture[:2]))
                if posture_mag <= 1e-9:
                    posture_axis_relieved = False
                else:
                    posture_dir = posture[:2] / posture_mag
                    proj0 = float(np.dot(vec0, posture_dir))
                    proj2 = float(np.dot(vec2, posture_dir))
                    posture_axis_relieved = bool(
                        (proj0 - proj2)
                        >= max(0.15, _HOLD_FORECAST_POSTURE_RELIEF_NORM)
                        or (proj0 > 0.0 and proj2 < 0.0)
                    )
            no_far_intensification = bool(
                not self._far_horizon_hidden_intensification
                and not self._h120_scheduler_far_reintensification_after_relief
                and not self._h120_oracle_probe_far_reintensification
            )
            if not bool(self._forecast_has_future):
                return False, "direction_reversal_no_future_preview_release"
            trust_ok, trust_reason = self._forecast_control_trust_decision(
                "direction_reversal_hold"
            )
            if self.forecast_control_trust_gate_enabled and not trust_ok:
                return False, f"direction_reversal_trust_veto:{trust_reason}"
            if not future_reversal:
                return False, "direction_reversal_no_reversal_release"
            if (
                mode == "direction_reversal_layered_v8_auto"
                and pressure_max >= 1.20
                and dot02 < -0.05
            ):
                return False, "direction_reversal_layered_v8_high_pressure_reversal_warning"
            if not posture_safe:
                if layered_mode:
                    return False, "direction_reversal_layered_posture_warning_release"
                return False, "direction_reversal_posture_floor_release"
            if mode in (
                    "direction_reversal_layered_v5_auto",
                    "direction_reversal_layered_v6_auto",
                    "direction_reversal_layered_v7_auto",
                    "direction_reversal_layered_v8_auto",
                ) and not posture_axis_relieved:
                return False, "direction_reversal_layered_v5_no_posture_axis_relief_release"
            if not no_far_intensification:
                return False, "direction_reversal_intensification_release"
            if not fallback_guard_ok:
                return False, "direction_reversal_fallback_veto_release"
            if layered_mode:
                if mode in (
                    "direction_reversal_layered_v5_auto",
                    "direction_reversal_layered_v6_auto",
                    "direction_reversal_layered_v7_auto",
                    "direction_reversal_layered_v8_auto",
                ):
                    return True, "direction_reversal_layered_v5_hold:low_posture_axis_relief"
                return True, "direction_reversal_layered_hold:low_posture_wrong_direction"
            return True, "direction_reversal_hold:avoid_wrong_direction_refresh"
        if mode == "regime_auto":
            if floor_risk:
                return False, "regime_auto_floor_risk_release"
            if reintensify:
                return False, "regime_auto_reintensification_release"
            if not direction_ok:
                return False, "regime_auto_direction_veto_release"
            if not fallback_guard_ok:
                return False, "regime_auto_fallback_veto_release"
            if high_then_relief:
                return True, "regime_auto_hold:transient_peak_future_decay"
            if rise_then_fall:
                return True, "regime_auto_hold:rise_then_fall"
            if lowrisk_redundant:
                return True, "regime_auto_hold:lowrisk_redundant_pump"
            if residual_high_economy:
                return True, "regime_auto_hold:residual_high_economy"
            return False, "regime_auto_no_opportunity_release"
        if mode == "posture_only":
            if floor_risk:
                return False, "posture_only_floor_risk_release"
            return True, "posture_only_debt_hold"
        if mode == "opportunity":
            pump_ok = (
                self._economy_pump_budget_opportunity_pump_m3
                >= float(self.economy_pump_budget_opportunity_min_pump_m3)
            )
            refresh_ok = (
                self._economy_pump_budget_opportunity_refresh_count
                >= int(self.economy_pump_budget_opportunity_min_refresh)
            )
            avg_delta = (
                self._economy_pump_budget_opportunity_delta_sum_kg
                / max(float(self._economy_pump_budget_opportunity_refresh_count), 1.0)
            )
            delta_ok = avg_delta >= float(
                self.economy_pump_budget_opportunity_min_delta_kg
            )
            pressure_guard_ok = (
                (not reintensify)
                and pressure_max < float(self.economy_pump_budget_smart_pressure_norm)
            )
            posture_guard_ok = max_axis < float(self.economy_pump_budget_smart_posture_deg)
            direction_guard_ok = bool(direction_ok)
            self._economy_pump_budget_opportunity_score = float(
                (1.0 if pump_ok else 0.0)
                + (1.0 if refresh_ok else 0.0)
                + (1.0 if delta_ok else 0.0)
                + (0.5 if pressure_guard_ok else 0.0)
                + (0.5 if posture_guard_ok else 0.0)
                + (0.5 if direction_guard_ok else 0.0)
                + (0.5 if fallback_guard_ok else 0.0)
            )
            self._economy_pump_budget_opportunity_active = bool(
                pump_ok and refresh_ok and delta_ok
            )
            if not self._economy_pump_budget_opportunity_active:
                return False, "opportunity_not_confirmed_release"
            if not pressure_guard_ok:
                return False, "opportunity_pressure_veto_release"
            if not posture_guard_ok:
                return False, "opportunity_posture_veto_release"
            if not direction_guard_ok:
                return False, "opportunity_direction_veto_release"
            if not fallback_guard_ok:
                return False, "opportunity_fallback_veto_release"
            return True, "opportunity_budget_hold"
        if floor_risk:
            return False, "floor_risk_release"
        if reintensify:
            return False, "reintensification_veto_release"
        if mode == "strict_relief":
            if relief:
                return True, "justified_hold:near_relief_decay"
            return False, "no_relief_budget_release"
        if mode == "broad_safe":
            if relief:
                return True, "justified_hold:near_relief_decay"
            if pressure_max < float(self.economy_pump_budget_smart_pressure_norm):
                return True, "justified_hold:no_imminent_worsening"
            return False, "no_safe_forecast_budget_release"
        return False, "unknown_allocator_mode_release"

    def _economy_budget_hold_release_needed(
        self,
        plant_info: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        """Release an active budget hold before the hard floor has to rescue it."""
        self._economy_pump_budget_forced_release_reason = "not_active"
        if not (
            self.economy_pump_budget_enabled
            and self._economy_pump_budget_hold_latched
        ):
            return False, "not_active"
        mode = str(self.economy_pump_budget_allocator_mode)
        if mode not in (
            "residual_high_plateau_auto",
            "residual_high_plateau_strict_auto",
            "neutral_mhs_broader_auto",
            "neutral_mhs_clean_auto",
            "psc_4hao_regime_addback_v2_auto",
            "dual_specialist_auto",
            "gusty_oscillation_budget_auto",
            "direction_reversal_avoidance_auto",
            "direction_reversal_layered_v5_auto",
            "direction_reversal_layered_v6_auto",
            "direction_reversal_layered_v7_auto",
            "direction_reversal_layered_v8_auto",
        ):
            return False, "mode_no_forced_release"
        info = plant_info or {}
        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if self._reactive_floor_latched or max_axis >= float(self.high_posture_enter):
            return True, "hard_floor_release"
        if max_axis >= max(0.0, float(self.high_posture_enter) - 0.35):
            return True, "near_hard_floor_release"
        allow_hold, reason = self._economy_budget_allocator_decision(info)
        if not allow_hold:
            return True, f"allocator_release:{reason}"
        return False, "hold_still_valid"

    def _economy_budget_observe_from_state(
        self,
        state,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> dict[str, Any]:
        """Charge non-hard pump use to the economy budget on every cadence."""
        info = self._attach_posture_info(plant_info, state)
        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        hard_domain = bool(
            self._reactive_floor_latched
            or max_axis >= float(self.high_posture_enter)
            or int(info.get("pump_fullspeed_any", 0)) > 0
        )
        self._economy_budget_observe(
            info,
            float(current_time),
            hard_domain=hard_domain,
        )
        return info

    def _relief_envelope_block_choice(
        self,
        *,
        horizon: str | None = None,
    ) -> tuple[bool, int, float, float, bool]:
        """Return relief eligibility for the first low-pressure block.

        The returned tuple is:
        (has_relief, block_index, block_mid_min, allowed_max_axis_deg, a2_unique).
        Blocks 0..2 are 0-60min; blocks 3..5 are 60-120min.
        """
        selected_horizon = str(horizon if horizon is not None else self.relief_envelope_horizon)
        norms = [float(x) for x in self._raw_pressure_block_norms[:3]]
        while len(norms) < 3:
            norms.append(0.0)
        if selected_horizon == "far":
            far = [float(x) for x in list(self._far_horizon_norms)[3:6]]
            while len(far) < 3:
                far.append(0.0)
            norms = norms + far
        low = float(self.relief_envelope_low_norm)
        high = float(self.relief_envelope_high_norm)
        reintensify = float(self.relief_envelope_reintensify_norm)
        block_mid = [10.0, 30.0, 50.0, 70.0, 90.0, 110.0]
        allowed = [
            float(self.relief_envelope_allowed_deg[0]),
            float(self.relief_envelope_allowed_deg[1]),
            float(self.relief_envelope_allowed_deg[2]),
            float(self.relief_envelope_allowed_deg[3]),
            float(self.relief_envelope_allowed_deg[3]),
            float(self.relief_envelope_allowed_deg[3]),
        ]
        # If the immediate hour is high, do not wait for a distant relief: the
        # soft band may only relax when the near horizon is not demanding a hard
        # recovery push. This keeps the v1.6 floor and near-risk behavior intact.
        near_max = max(norms[:3]) if len(norms) >= 3 else 0.0
        if near_max >= high:
            return False, -1, 0.0, 0.0, False
        for idx, norm in enumerate(norms):
            if norm >= low:
                continue
            future = norms[idx + 1 :]
            if future and max(future) >= reintensify:
                self._relief_envelope_reintensification_veto_count += 1
                self._relief_envelope_reason = "reintensification_veto"
                return False, -1, 0.0, 0.0, False
            return True, idx, block_mid[idx], allowed[idx], idx >= 3
        return False, -1, 0.0, 0.0, False

    def _relief_envelope_direction_ok(self, posture: np.ndarray) -> bool:
        # If the near blocks are nearly reversing direction, avoid waiting: the
        # platform can trade pitch against roll under sign-flip / mismatch.
        vecs = [np.asarray(v, dtype=float).reshape(-1) for v in self._raw_pressure_block_vecs[:3]]
        if len(vecs) >= 2 and vecs[0].size >= 2 and vecs[1].size >= 2:
            n0 = float(np.linalg.norm(vecs[0][:2]))
            n1 = float(np.linalg.norm(vecs[1][:2]))
            if n0 > 1e-9 and n1 > 1e-9 and float(np.dot(vecs[0][:2], vecs[1][:2])) < 0.0:
                return False
        if posture.size >= 2 and len(vecs) >= 1 and vecs[0].size >= 2:
            # Only veto strong opposition between current posture direction and
            # immediate pressure direction. Weak dot products are treated as
            # uncertain, not as a trigger.
            pnorm = float(np.linalg.norm(posture[:2]))
            vnorm = float(np.linalg.norm(vecs[0][:2]))
            if pnorm > 1e-9 and vnorm > 1e-9:
                cos = float(np.dot(posture[:2], vecs[0][:2]) / (pnorm * vnorm))
                if cos < -0.25:
                    return False
        return True

    def _relief_envelope_action_shape_ok(self, avec: np.ndarray | None) -> bool:
        if not self.relief_envelope_single_axis_only:
            return True
        if avec is None:
            return True
        arr = np.asarray(avec, dtype=float).reshape(-1)
        if arr.size < 2:
            return True
        a0 = abs(float(arr[0]))
        a1 = abs(float(arr[1]))
        major = max(a0, a1)
        minor = min(a0, a1)
        if major <= 1e-9:
            return True
        if minor <= float(self.relief_envelope_axis_minor_abs_max_deg):
            return True
        return (minor / major) <= float(self.relief_envelope_axis_minor_ratio_max)

    def _relief_envelope_axis_cap_decision(
        self,
        avec: np.ndarray | None,
    ) -> tuple[str, np.ndarray | None]:
        """Return the adaptive relief-envelope action for an eligible refresh.

        This is an additive B1 lever. Single-axis and all-axes-relieved
        opportunities keep the existing A1 full-delay path. Mixed-axis actions are
        capped only on axes whose 0-60min pressure is already relieved; unrelieved
        axes remain fully pursued.
        """
        self._relief_envelope_axis_cap_requested_delta_deg = 0.0
        self._relief_envelope_axis_cap_pursued_delta_deg = 0.0
        self._relief_envelope_axis_cap_held_delta_deg = 0.0
        self._relief_envelope_axis_cap_covered_axes = 0
        self._relief_envelope_axis_cap_mode = "none"
        if not self.relief_envelope_adaptive_axis_cap:
            return "disabled", None
        if avec is None:
            return "veto_no_action", None
        arr = np.asarray(avec, dtype=float).reshape(-1)
        if arr.size < 2:
            return "veto_no_action", None
        arr = arr[:2].copy()
        if float(np.linalg.norm(arr)) <= 1e-12:
            return "veto_no_action", None
        # Do not erode A1's current wins: if the action already passes the shape
        # guard, keep the full-delay behavior.
        if self._relief_envelope_action_shape_ok(arr):
            self._relief_envelope_axis_cap_mode = "full_delay_shape_ok"
            return "full_delay", None
        vecs = [
            np.asarray(v, dtype=float).reshape(-1)
            for v in self._raw_pressure_block_vecs[:3]
        ]
        if len(vecs) < 3 or any(v.size < 2 for v in vecs):
            self._relief_envelope_axis_shape_veto_count += 1
            self._relief_envelope_axis_cap_mode = "veto_no_pressure_vec"
            self._relief_envelope_reason = "axis_shape_veto"
            return "veto", None
        axis_pressure = np.max(
            np.abs(np.vstack([v[:2] for v in vecs])),
            axis=0,
        )
        relief_axis = axis_pressure <= float(self.relief_envelope_cap_relief_norm)
        if not np.any(relief_axis):
            self._relief_envelope_axis_shape_veto_count += 1
            self._relief_envelope_axis_cap_mode = "veto_no_relieved_axis"
            self._relief_envelope_reason = "axis_shape_veto"
            return "veto", None
        if bool(np.all(relief_axis)):
            # Full relief is the existing full-delay case, not a cap case.
            self._relief_envelope_axis_cap_covered_axes = int(np.sum(relief_axis))
            self._relief_envelope_axis_cap_mode = "full_delay_all_axes_relieved"
            return "full_delay", None
        capped = arr.copy()
        hold_fraction = float(self.relief_envelope_cap_hold_fraction)
        capped[relief_axis] = capped[relief_axis] * hold_fraction
        requested = float(np.sum(np.abs(arr)))
        pursued = float(np.sum(np.abs(capped)))
        held = max(requested - pursued, 0.0)
        if held <= 1e-9 or pursued <= 1e-12:
            self._relief_envelope_axis_shape_veto_count += 1
            self._relief_envelope_axis_cap_mode = "veto_degenerate_cap"
            self._relief_envelope_reason = "axis_shape_veto"
            return "veto", None
        self._relief_envelope_axis_cap_requested_delta_deg = requested
        self._relief_envelope_axis_cap_pursued_delta_deg = pursued
        self._relief_envelope_axis_cap_held_delta_deg = held
        self._relief_envelope_axis_cap_covered_axes = int(np.sum(relief_axis))
        self._relief_envelope_axis_cap_mode = "partial_axis_cap"
        return "cap", capped

    def _debug_safe_hold_active(self, plant_info: dict[str, Any]) -> bool:
        """DEBUG-ONLY (default off via env FOWT_RELIEF_DEBUG_SAFE_HOLD=1).

        Forced-positive probe: returns True when posture is safe (below the hard
        high-posture enter threshold) and the reactive floor is not latched, so a
        non-urgent target refresh can be held instead of executed. This exists only
        to prove the hold mechanism can change closed-loop pump; it never overrides
        the floor and is never on in normal runs.
        """
        if os.environ.get("FOWT_RELIEF_DEBUG_SAFE_HOLD") != "1":
            return False
        if self._reactive_floor_latched:
            return False
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)), dtype=float
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        return max_axis < float(self.high_posture_enter)

    def _relief_envelope_delay_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        *,
        would_refresh: bool,
        refresh_avec: np.ndarray | None = None,
        ignore_action_shape: bool = False,
        count_delay: bool = True,
    ) -> bool:
        self._relief_envelope_evaluated_this_bucket = True
        self._relief_envelope_active = False
        self._relief_envelope_allowed_max_axis_deg = 0.0
        self._relief_envelope_relief_block_index = -1
        self._relief_envelope_relief_time_min = 0.0
        now = float(current_time)
        step_s = (
            self.update_interval_s
            if self._relief_envelope_last_update_s < -1.0e11
            else max(0.0, now - float(self._relief_envelope_last_update_s))
        )
        self._relief_envelope_last_update_s = now
        if not self.relief_envelope_enabled:
            self._relief_envelope_reason = "disabled"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)), dtype=float
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if self._reactive_floor_latched or max_axis >= float(self.high_posture_enter):
            self._relief_envelope_reason = "hard_floor_domain"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        if max_axis < 3.0:
            self._relief_envelope_reason = "below_soft_band"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        if not self._relief_envelope_direction_ok(posture):
            self._relief_envelope_direction_mismatch_veto_count += 1
            self._relief_envelope_reason = "direction_mismatch_veto"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        if (not ignore_action_shape) and not self._relief_envelope_action_shape_ok(refresh_avec):
            self._relief_envelope_axis_shape_veto_count += 1
            self._relief_envelope_reason = "axis_shape_veto"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        resp = float(self._active_effectiveness_current_response_deg)
        if (
            math.isfinite(resp)
            and resp < -float(self.relief_envelope_worsening_eps_deg)
        ):
            self._relief_envelope_posture_worsening_exit_count += 1
            self._relief_envelope_reason = "posture_worsening_exit"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        has_relief, block_idx, block_mid, allowed_deg, a2_unique = (
            self._relief_envelope_block_choice()
        )
        if not has_relief:
            if self._relief_envelope_reason not in ("reintensification_veto",):
                self._relief_envelope_reason = "no_relief"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        self._relief_envelope_relief_block_index = int(block_idx)
        self._relief_envelope_relief_time_min = float(block_mid)
        self._relief_envelope_allowed_max_axis_deg = float(allowed_deg)
        if max_axis > float(allowed_deg):
            self._relief_envelope_reason = "outside_allowed_envelope"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        projected_duration = (
            float(self._relief_envelope_current_duration_s) + step_s
            if self._relief_envelope_prev_active
            else step_s
        )
        if projected_duration > float(self.relief_envelope_max_duration_s):
            self._relief_envelope_debt_exit_count += 1
            self._relief_envelope_reason = "duration_exit"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        excess = max(0.0, max_axis - 3.0)
        projected_debt = float(self._relief_envelope_safety_debt_deg_s) + excess * step_s
        if projected_debt > float(self.relief_envelope_debt_budget_deg_s):
            self._relief_envelope_debt_exit_count += 1
            self._relief_envelope_reason = "debt_budget_exit"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        projected_near_dwell = float(self._relief_envelope_near_limit_dwell_s)
        if max_axis > float(self.relief_envelope_near_limit_deg):
            projected_near_dwell += step_s
        if projected_near_dwell > float(self.relief_envelope_near_limit_budget_s):
            self._relief_envelope_debt_exit_count += 1
            self._relief_envelope_reason = "near_limit_dwell_exit"
            self._relief_envelope_prev_active = False
            self._relief_envelope_current_duration_s = 0.0
            return False
        self._relief_envelope_safety_debt_deg_s = projected_debt
        self._relief_envelope_near_limit_dwell_s = projected_near_dwell
        self._relief_envelope_current_duration_s = projected_duration
        self._relief_envelope_active = True
        self._relief_envelope_relaxation_active_time_s += step_s
        if not self._relief_envelope_prev_active:
            self._relief_envelope_eligible_count += 1
            if a2_unique:
                self._relief_envelope_a2_unique_relax_count += 1
        if would_refresh:
            self._relief_envelope_pump_opportunity_count += 1
            if count_delay:
                self._relief_envelope_target_refresh_delayed_count += 1
        self._relief_envelope_reason = "a2_unique_relief" if a2_unique else "near_relief"
        self._relief_envelope_prev_active = True
        return bool(would_refresh)

    def _relief_envelope_reset_bucket_log(self) -> None:
        self._relief_envelope_evaluated_this_bucket = False
        self._relief_envelope_active = False
        self._relief_envelope_reason = "not_evaluated"
        self._relief_envelope_allowed_max_axis_deg = 0.0
        self._relief_envelope_relief_block_index = -1
        self._relief_envelope_relief_time_min = 0.0

    def _relief_envelope_mark_not_applicable(self) -> None:
        if self._relief_envelope_evaluated_this_bucket:
            return
        self._relief_envelope_active = False
        self._relief_envelope_reason = (
            "disabled" if not self.relief_envelope_enabled else "not_applicable"
        )
        self._relief_envelope_allowed_max_axis_deg = 0.0
        self._relief_envelope_relief_block_index = -1
        self._relief_envelope_relief_time_min = 0.0
        self._relief_envelope_axis_cap_requested_delta_deg = 0.0
        self._relief_envelope_axis_cap_pursued_delta_deg = 0.0
        self._relief_envelope_axis_cap_held_delta_deg = 0.0
        self._relief_envelope_axis_cap_covered_axes = 0
        self._relief_envelope_axis_cap_mode = "none"
        self._relief_envelope_prev_active = False
        self._relief_envelope_current_duration_s = 0.0

    def _relief_envelope_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}relief_envelope_enabled": int(self.relief_envelope_enabled),
            f"{p}relief_envelope_horizon": str(self.relief_envelope_horizon),
            f"{p}relief_envelope_active": int(self._relief_envelope_active),
            f"{p}relief_envelope_reason": str(self._relief_envelope_reason),
            f"{p}relief_envelope_allowed_max_axis_deg": float(
                self._relief_envelope_allowed_max_axis_deg
            ),
            f"{p}relief_envelope_relief_block_index": int(
                self._relief_envelope_relief_block_index
            ),
            f"{p}relief_envelope_relief_time_min": float(
                self._relief_envelope_relief_time_min
            ),
            f"{p}relief_envelope_eligible_count": int(
                self._relief_envelope_eligible_count
            ),
            f"{p}relief_envelope_pump_opportunity_count": int(
                self._relief_envelope_pump_opportunity_count
            ),
            f"{p}relief_envelope_target_refresh_delayed_count": int(
                self._relief_envelope_target_refresh_delayed_count
            ),
            f"{p}relief_envelope_relaxation_active_time_s": float(
                self._relief_envelope_relaxation_active_time_s
            ),
            f"{p}relief_envelope_current_duration_s": float(
                self._relief_envelope_current_duration_s
            ),
            f"{p}relief_envelope_safety_debt_used": float(
                self._relief_envelope_safety_debt_deg_s
            ),
            f"{p}relief_envelope_near_limit_dwell_s": float(
                self._relief_envelope_near_limit_dwell_s
            ),
            f"{p}relief_envelope_debt_exit_count": int(
                self._relief_envelope_debt_exit_count
            ),
            f"{p}relief_envelope_false_relief_count": int(
                self._relief_envelope_false_relief_count
            ),
            f"{p}relief_envelope_reintensification_veto_count": int(
                self._relief_envelope_reintensification_veto_count
            ),
            f"{p}relief_envelope_direction_mismatch_veto_count": int(
                self._relief_envelope_direction_mismatch_veto_count
            ),
            f"{p}relief_envelope_axis_shape_veto_count": int(
                self._relief_envelope_axis_shape_veto_count
            ),
            f"{p}relief_envelope_axis_cap_enabled": int(
                self.relief_envelope_adaptive_axis_cap
            ),
            f"{p}relief_envelope_axis_cap_count": int(
                self._relief_envelope_axis_cap_count
            ),
            f"{p}relief_envelope_axis_cap_mode": str(
                self._relief_envelope_axis_cap_mode
            ),
            f"{p}relief_envelope_axis_cap_requested_delta_deg": float(
                self._relief_envelope_axis_cap_requested_delta_deg
            ),
            f"{p}relief_envelope_axis_cap_pursued_delta_deg": float(
                self._relief_envelope_axis_cap_pursued_delta_deg
            ),
            f"{p}relief_envelope_axis_cap_held_delta_deg": float(
                self._relief_envelope_axis_cap_held_delta_deg
            ),
            f"{p}relief_envelope_axis_cap_covered_axes": int(
                self._relief_envelope_axis_cap_covered_axes
            ),
            f"{p}relief_envelope_posture_worsening_exit_count": int(
                self._relief_envelope_posture_worsening_exit_count
            ),
            f"{p}relief_envelope_a2_unique_relax_count": int(
                self._relief_envelope_a2_unique_relax_count
            ),
            f"{p}relief_envelope_a2_unique_pump_saved_m3": float(
                self._relief_envelope_a2_unique_pump_saved_m3
            ),
        }

    def _h120_floor_shaping_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_floor_shaping_mode": str(self.h120_floor_shaping_mode),
            f"{p}h120_floor_shaping_active": int(self._h120_floor_shaping_active),
            f"{p}h120_floor_shaping_reason": str(self._h120_floor_shaping_reason),
            f"{p}h120_floor_shaping_effect": str(self._h120_floor_shaping_effect),
            f"{p}h120_floor_shaping_axis": str(self._h120_floor_shaping_axis),
            f"{p}h120_floor_shaping_count": int(self._h120_floor_shaping_count),
            f"{p}h120_floor_shaping_delta_mean_kg": float(
                self._h120_floor_shaping_delta_mean_kg
            ),
            f"{p}h120_floor_shaping_far_max_norm": float(self._far_horizon_far_max),
            f"{p}h120_floor_shaping_current_response_deg": float(
                self._reactive_floor_current_response_deg
            ),
        }

    def _h120_pareto_mode_selector_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_pareto_mode_selector_enabled": int(
                self.h120_pareto_mode_selector_enabled
            ),
            f"{p}h120_pareto_mode_selector_active": int(
                self._h120_pareto_mode_selector_active
            ),
            f"{p}h120_pareto_mode_selector_reason": str(
                self._h120_pareto_mode_selector_reason
            ),
            f"{p}h120_pareto_mode_selector_selected_mode": str(
                self._h120_pareto_mode_selector_selected_mode
            ),
            f"{p}h120_pareto_mode_selector_selected_effect": str(
                self._h120_pareto_mode_selector_selected_effect
            ),
            f"{p}h120_pareto_mode_selector_episode_active": int(
                self._h120_pareto_mode_selector_episode_active
            ),
            f"{p}h120_pareto_mode_selector_episode_mode": str(
                self._h120_pareto_mode_selector_episode_mode
            ),
            f"{p}h120_pareto_mode_selector_count": int(
                self._h120_pareto_mode_selector_count
            ),
            f"{p}h120_pareto_mode_selector_switch_count": int(
                self._h120_pareto_mode_selector_switch_count
            ),
            f"{p}h120_pareto_mode_selector_far_persistent_high": int(
                self._h120_pareto_mode_selector_far_persistent_high
            ),
            f"{p}h120_pareto_mode_selector_far_intensification": int(
                self._h120_pareto_mode_selector_far_intensification
            ),
            f"{p}h120_pareto_mode_selector_far_reintensification": int(
                self._h120_pareto_mode_selector_far_reintensification
            ),
            f"{p}h120_pareto_mode_selector_far_relief": int(
                self._h120_pareto_mode_selector_far_relief
            ),
            f"{p}h120_pareto_mode_selector_far_direction_consistent": int(
                self._h120_pareto_mode_selector_far_direction_consistent
            ),
            f"{p}h120_pareto_mode_selector_recovery_slow": int(
                self._h120_pareto_mode_selector_recovery_slow
            ),
            f"{p}h120_pareto_mode_selector_far_max_norm": float(
                self._h120_pareto_mode_selector_far_max_norm
            ),
            f"{p}h120_pareto_mode_selector_far_min_norm": float(
                self._h120_pareto_mode_selector_far_min_norm
            ),
            f"{p}h120_pareto_mode_selector_near_last_norm": float(
                self._h120_pareto_mode_selector_near_last_norm
            ),
            f"{p}h120_pareto_mode_selector_current_response_deg": float(
                self._h120_pareto_mode_selector_current_response_deg
            ),
        }

    def _h120_scheduler_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_scheduler_enabled": int(self.h120_risk_scheduler_enabled),
            f"{p}h120_scheduler_risk_tier": str(self._h120_scheduler_risk_tier),
            f"{p}h120_scheduler_reason": str(self._h120_scheduler_reason),
            f"{p}h120_scheduler_far_persistent_high_pressure": int(
                self._h120_scheduler_far_persistent_high_pressure
            ),
            f"{p}h120_scheduler_far_intensification": int(
                self._h120_scheduler_far_intensification
            ),
            f"{p}h120_scheduler_far_reintensification_after_relief": int(
                self._h120_scheduler_far_reintensification_after_relief
            ),
            f"{p}h120_scheduler_far_direction_consistent_with_current_posture": int(
                self._h120_scheduler_far_direction_consistent_with_current_posture
            ),
            f"{p}h120_scheduler_far_signflip_risk": int(
                self._h120_scheduler_far_signflip_risk
            ),
            f"{p}h120_scheduler_effective_medium_delay_s": float(
                self._h120_scheduler_effective_medium_delay_s
            ),
            f"{p}h120_scheduler_remote_risk_active": int(
                self._h120_scheduler_remote_risk_active
            ),
            f"{p}h120_scheduler_floor_episode_active_for_budget": int(
                self._h120_scheduler_floor_episode_active_for_budget
            ),
            f"{p}h120_scheduler_early_stop_suppressed_active": int(
                self._h120_scheduler_early_stop_suppressed_active
            ),
            f"{p}h120_scheduler_early_stop_suppressed_count": int(
                self._h120_scheduler_early_stop_suppressed_count
            ),
            f"{p}h120_scheduler_remote_risk_trigger_count": int(
                self._h120_scheduler_remote_risk_trigger_count
            ),
            f"{p}h120_scheduler_far_max_norm": float(self._h120_scheduler_far_max_norm),
            f"{p}h120_scheduler_far_min_norm": float(self._h120_scheduler_far_min_norm),
            f"{p}h120_scheduler_near_max_norm": float(self._h120_scheduler_near_max_norm),
            f"{p}h120_scheduler_near_last_norm": float(self._h120_scheduler_near_last_norm),
            f"{p}h120_scheduler_prefloor_probe_enabled": int(
                self.h120_scheduler_prefloor_probe_enabled
            ),
            f"{p}h120_scheduler_prefloor_action": str(
                self.h120_scheduler_prefloor_action
            ),
            f"{p}h120_scheduler_prefloor_active": int(
                self._h120_scheduler_prefloor_active
            ),
            f"{p}h120_scheduler_prefloor_reason": str(
                self._h120_scheduler_prefloor_reason
            ),
            f"{p}h120_scheduler_prefloor_count": int(
                self._h120_scheduler_prefloor_count
            ),
            f"{p}h120_scheduler_prefloor_delta_mean_kg": float(
                self._h120_scheduler_prefloor_delta_mean_kg
            ),
        }

    def _h120_pareto_mode_selector_refresh(
        self,
        plant_info: dict[str, Any] | None = None,
    ) -> None:
        """Default-off oracle mode selector for the static floor Pareto modes.

        This is a diagnostic selector only. It does not alter the floor logic;
        it only records which static mode the oracle would choose among:
        - v1.4 economy: active_small + early_stop
        - v1.6 balanced: active_small + delayed medium
        - v1.5 safety: active_medium + early_stop
        """
        self._h120_pareto_mode_selector_active = False
        self._h120_pareto_mode_selector_reason = "disabled"
        self._h120_pareto_mode_selector_selected_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_selected_effect = "balanced"
        self._h120_pareto_mode_selector_far_persistent_high = False
        self._h120_pareto_mode_selector_far_intensification = False
        self._h120_pareto_mode_selector_far_reintensification = False
        self._h120_pareto_mode_selector_far_relief = False
        self._h120_pareto_mode_selector_far_direction_consistent = False
        self._h120_pareto_mode_selector_recovery_slow = False
        self._h120_pareto_mode_selector_far_max_norm = 0.0
        self._h120_pareto_mode_selector_far_min_norm = 0.0
        self._h120_pareto_mode_selector_near_last_norm = 0.0
        self._h120_pareto_mode_selector_current_response_deg = float("nan")
        if not self._reactive_floor_latched:
            self._h120_pareto_mode_selector_episode_active = False
            self._h120_pareto_mode_selector_episode_mode = "v16_delay1200"
        if not self.h120_pareto_mode_selector_enabled:
            return
        if not self.far_horizon_enabled or not self._far_horizon_available:
            self._h120_pareto_mode_selector_reason = "far_horizon_unavailable"
            return

        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        near_last = float(norms[2])
        far = [float(x) for x in norms[3:6]]
        far_min = float(min(far))
        far_max = float(max(far))
        high_norm = float(self.h120_pareto_mode_selector_far_high_norm)
        intensify_margin = float(self.h120_pareto_mode_selector_intensify_margin_norm)
        relief_margin = float(self.h120_pareto_mode_selector_relief_margin_norm)
        persistent_high = bool(far_min >= high_norm)
        intensification = bool(far_max >= near_last + intensify_margin)
        near_relief = bool(near_last <= float(norms[0]) - relief_margin)
        reintensification = bool(near_relief and far_max >= near_last + intensify_margin)

        posture = np.zeros(2, dtype=float)
        if plant_info is not None:
            posture = np.asarray(
                plant_info.get("posture_vec_deg", posture),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            posture = posture[:2]
        posture_norm = float(np.linalg.norm(posture))
        far_vecs = self._far_horizon_vecs[3:6] if len(self._far_horizon_vecs) >= 6 else []
        direction_consistent = False
        if far_vecs and posture_norm > 1e-6:
            far_vec_sum = np.sum(np.asarray(far_vecs, dtype=float), axis=0)
            fv_norm = float(np.linalg.norm(far_vec_sum))
            if fv_norm > 1e-6:
                cos = float(np.dot(far_vec_sum, posture) / (fv_norm * posture_norm))
                direction_consistent = bool(cos > 0.3)
        current_response = float(self._reactive_floor_current_response_deg)
        recovery_slow = bool(math.isfinite(current_response) and current_response <= float(self.h120_pareto_mode_selector_slow_response_eps_deg))

        self._h120_pareto_mode_selector_far_persistent_high = persistent_high
        self._h120_pareto_mode_selector_far_intensification = intensification
        self._h120_pareto_mode_selector_far_reintensification = reintensification
        self._h120_pareto_mode_selector_far_relief = near_relief
        self._h120_pareto_mode_selector_far_direction_consistent = direction_consistent
        self._h120_pareto_mode_selector_recovery_slow = recovery_slow
        self._h120_pareto_mode_selector_far_max_norm = far_max
        self._h120_pareto_mode_selector_far_min_norm = far_min
        self._h120_pareto_mode_selector_near_last_norm = near_last
        self._h120_pareto_mode_selector_current_response_deg = current_response

        # Conservative three-way rule:
        # - relief / weak far danger -> v1.4 economy
        # - persistent high / intensification / reintensification with slow
        #   recovery or direction support -> v1.5 safety
        # - otherwise -> v1.6 balanced
        if (not persistent_high and not intensification and not reintensification) or (
            near_relief and not direction_consistent
        ):
            self._h120_pareto_mode_selector_selected_mode = "v14_early_stop"
            self._h120_pareto_mode_selector_selected_effect = "economy"
            self._h120_pareto_mode_selector_reason = "relief_or_weak_far_risk"
        elif (persistent_high or intensification or reintensification) and (
            direction_consistent or recovery_slow
        ):
            self._h120_pareto_mode_selector_selected_mode = "v15_medium"
            self._h120_pareto_mode_selector_selected_effect = "safety"
            self._h120_pareto_mode_selector_reason = "far_danger_or_slow_recovery"
        else:
            self._h120_pareto_mode_selector_selected_mode = "v16_delay1200"
            self._h120_pareto_mode_selector_selected_effect = "balanced"
            self._h120_pareto_mode_selector_reason = "default_balanced"

        if self._h120_pareto_mode_selector_episode_active:
            self._h120_pareto_mode_selector_selected_mode = str(
                self._h120_pareto_mode_selector_episode_mode
            )
            if self._h120_pareto_mode_selector_selected_mode == "v14_early_stop":
                self._h120_pareto_mode_selector_selected_effect = "economy"
            elif self._h120_pareto_mode_selector_selected_mode == "v15_medium":
                self._h120_pareto_mode_selector_selected_effect = "safety"
            else:
                self._h120_pareto_mode_selector_selected_effect = "balanced"
            self._h120_pareto_mode_selector_reason = (
                f"episode_latched:{self._h120_pareto_mode_selector_reason}"
            )

        self._h120_pareto_mode_selector_active = True
        if (
            self._h120_pareto_mode_selector_selected_mode
            != self._h120_pareto_mode_selector_last_mode
        ):
            self._h120_pareto_mode_selector_switch_count += 1
            self._h120_pareto_mode_selector_last_mode = self._h120_pareto_mode_selector_selected_mode
        self._h120_pareto_mode_selector_count += 1

    def _oracle_preemptive_needed(
        self,
        sample,
        plant_info: dict[str, Any],
        current_time: float,
        floor_refresh_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        self._oracle_preemptive_active = False
        self._oracle_preemptive_reason = "disabled"
        self._oracle_preemptive_posture_metric_deg = 0.0
        self._oracle_preemptive_current_norm = 0.0
        self._oracle_preemptive_future_max_norm = 0.0
        self._oracle_preemptive_rise_norm = 0.0
        self._oracle_preemptive_delta_mean_kg = 0.0
        if not self.oracle_preemptive_prevent_enabled:
            return False, np.zeros(2, dtype=float), "disabled"
        if floor_refresh_pending or self._reactive_floor_latched:
            self._oracle_preemptive_reason = "floor_already_active"
            return False, np.zeros(2, dtype=float), "floor_already_active"
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        self._oracle_preemptive_posture_metric_deg = float(max_axis)
        if max_axis < float(self.oracle_preemptive_enter_deg):
            self._oracle_preemptive_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.oracle_preemptive_floor_deg):
            self._oracle_preemptive_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"
        since_last = float(current_time) - float(self._oracle_preemptive_last_s)
        if since_last < float(self.oracle_preemptive_cooldown_s):
            self._oracle_preemptive_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"
        oracle_blocks = self._oracle_preemptive_blocks(sample)
        if len(oracle_blocks) < 3:
            self._oracle_preemptive_reason = "insufficient_oracle_blocks"
            return False, np.zeros(2, dtype=float), "insufficient_oracle_blocks"
        raw_norms = [
            norm_term(
                np.asarray(
                    b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                    dtype=float,
                ),
                self.cfg,
            )
            for b in oracle_blocks[:3]
        ]
        current_norm = float(raw_norms[0])
        future_max = float(max(raw_norms[1:3]))
        rise = future_max - current_norm
        self._oracle_preemptive_current_norm = current_norm
        self._oracle_preemptive_future_max_norm = future_max
        self._oracle_preemptive_rise_norm = rise
        if future_max < float(self.oracle_preemptive_min_future_norm):
            self._oracle_preemptive_reason = "future_pressure_low"
            return False, np.zeros(2, dtype=float), "future_pressure_low"
        if rise < float(self.oracle_preemptive_rise_margin_norm):
            self._oracle_preemptive_reason = "future_not_rising_enough"
            return False, np.zeros(2, dtype=float), "future_not_rising_enough"
        avec = action_vec("active_small", posture, self.cfg)
        delta = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        self._oracle_preemptive_delta_mean_kg = delta
        if delta < float(self.oracle_preemptive_min_delta_kg):
            self._oracle_preemptive_reason = "proposal_delta_below_threshold"
            return False, avec, "proposal_delta_below_threshold"
        self._oracle_preemptive_active = True
        self._oracle_preemptive_reason = "oracle_preemptive_refresh"
        self._oracle_preemptive_count += 1
        self._oracle_preemptive_last_s = float(current_time)
        return True, avec, "oracle_preemptive_refresh"

    def _event_risk_scales(self, event_probs: dict[str, float]) -> tuple[list[float], list[float]]:
        probs = [float(event_probs.get(key, 0.0)) for key in _EVENT_RISK_KEYS]
        return probs, [1.0, 1.0, 1.0]

    def _trusted_event_gate(self, event_probs: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        raw = {str(key): float(value) for key, value in event_probs.items()}
        attention_hits = sum(
            int(raw.get(key, 0.0) >= self.trusted_event_attention_threshold)
            for key in _TRUSTED_EVENT_ATTENTION_KEYS
        )
        dynamic_hits = sum(
            int(raw.get(key, 0.0) >= self.trusted_event_dynamic_threshold)
            for key in _TRUSTED_EVENT_DYNAMIC_KEYS
        )
        highwind_hit = (
            raw.get(_TRUSTED_EVENT_HIGHWIND_KEY, 0.0)
            >= self.trusted_event_highwind_threshold
        )
        attention_pass = attention_hits >= self.trusted_event_attention_min_heads
        dynamic_pass = (
            self.trusted_event_dynamic_enabled
            and dynamic_hits >= self.trusted_event_dynamic_min_heads
        )
        trusted = bool(highwind_hit or attention_pass or dynamic_pass)
        if not self.trusted_event_gate_enabled:
            reason = "disabled"
            effective = dict(raw)
        elif trusted:
            reasons = []
            if highwind_hit:
                reasons.append("highwind")
            if attention_pass:
                reasons.append("attention_multihead")
            if dynamic_pass:
                reasons.append("dynamic_multihead")
            reason = "+".join(reasons) if reasons else "trusted"
            effective = dict(raw)
        else:
            reason = "untrusted_event_signal"
            effective = dict(raw)
            for key in _EVENT_RISK_KEYS:
                effective[key] = 0.0
            effective["ballast_attention_event"] = 0.0

        diag = {
            "enabled": int(self.trusted_event_gate_enabled),
            "trusted": int(trusted) if self.trusted_event_gate_enabled else 1,
            "reason": reason,
            "highwind_hit": int(highwind_hit),
            "attention_hits": int(attention_hits),
            "dynamic_hits": int(dynamic_hits),
            "highwind_prob": float(raw.get(_TRUSTED_EVENT_HIGHWIND_KEY, 0.0)),
            "attention_threshold": float(self.trusted_event_attention_threshold),
            "highwind_threshold": float(self.trusted_event_highwind_threshold),
            "dynamic_threshold": float(self.trusted_event_dynamic_threshold),
        }
        return effective, diag

    def _trusted_event_gate_log_fields(self, prefix: str = "") -> dict[str, Any]:
        diag = dict(self._trusted_event_gate_diag)
        raw = list(self._event_risk_raw_probs)
        effective = list(self._event_risk_probs)
        while len(raw) < 3:
            raw.append(0.0)
        while len(effective) < 3:
            effective.append(0.0)
        return {
            f"{prefix}event_risk_raw_prob_0_20m": float(raw[0]),
            f"{prefix}event_risk_raw_prob_20_40m": float(raw[1]),
            f"{prefix}event_risk_raw_prob_40_60m": float(raw[2]),
            f"{prefix}trusted_event_gate_enabled": int(diag.get("enabled", 0)),
            f"{prefix}trusted_event_gate_trusted": int(diag.get("trusted", 1)),
            f"{prefix}trusted_event_gate_reason": str(diag.get("reason", "")),
            f"{prefix}trusted_event_highwind_hit": int(diag.get("highwind_hit", 0)),
            f"{prefix}trusted_event_attention_hits": int(diag.get("attention_hits", 0)),
            f"{prefix}trusted_event_dynamic_hits": int(diag.get("dynamic_hits", 0)),
            f"{prefix}trusted_event_highwind_prob": float(diag.get("highwind_prob", 0.0)),
            f"{prefix}trusted_event_highwind_threshold": float(
                diag.get("highwind_threshold", self.trusted_event_highwind_threshold)
            ),
            f"{prefix}trusted_event_attention_threshold": float(
                diag.get("attention_threshold", self.trusted_event_attention_threshold)
            ),
            f"{prefix}trusted_event_dynamic_threshold": float(
                diag.get("dynamic_threshold", self.trusted_event_dynamic_threshold)
            ),
            f"{prefix}event_risk_effective_prob_0_20m": float(effective[0]),
            f"{prefix}event_risk_effective_prob_20_40m": float(effective[1]),
            f"{prefix}event_risk_effective_prob_40_60m": float(effective[2]),
        }

    def _reset_forecast_pressure_trust_state(
        self,
        reason: str,
        reset_observation: bool = False,
    ) -> None:
        reason = str(reason)
        self._forecast_pressure_trust_enabled = int(
            self.forecast_pressure_trust_gate_enabled
        )
        self._forecast_pressure_trust_trusted = True
        self._forecast_pressure_trust_reason = reason
        self._forecast_pressure_trust_block_scales = [1.0, 1.0, 1.0]
        self._forecast_pressure_trust_current_norm = 0.0
        self._forecast_pressure_trust_future_max_norm = 0.0
        self._forecast_pressure_trust_rise_norm = 0.0
        self._forecast_pressure_trust_relief_margin_norm = 0.0
        self._forecast_pressure_trust_dot01 = 0.0
        self._forecast_pressure_trust_dot02 = 0.0
        self._forecast_pressure_trust_current_support = 0
        self._forecast_pressure_trust_event_support = 0
        self._forecast_pressure_trust_pressure_shape_support = 0
        self._forecast_pressure_trust_spike_veto = 0
        if reset_observation:
            self._forecast_pressure_trust_current_observed_norm = 0.0
            self._forecast_pressure_trust_current_speed_ms = 0.0
            self._forecast_pressure_trust_current_dir_deg = 0.0
            self._forecast_pressure_trust_current_wind_stability_veto = 0
            self._forecast_pressure_trust_future_speed_max_ms = 0.0
            self._forecast_pressure_trust_current_obs_available = False
            self._forecast_pressure_trust_current_obs_reason = reason

    def _observe_forecast_pressure_current_support(self, wind_obs) -> None:
        self._forecast_pressure_trust_current_obs_available = False
        self._forecast_pressure_trust_current_obs_reason = "missing_current_wind"
        self._forecast_pressure_trust_current_observed_norm = 0.0
        self._forecast_pressure_trust_current_speed_ms = 0.0
        self._forecast_pressure_trust_current_dir_deg = 0.0
        if wind_obs is None:
            return
        try:
            ws = float(wind_obs.get("ws", 0.0))
            wd = float(wind_obs.get("wd_deg", 0.0))
        except Exception:
            return
        if not (math.isfinite(ws) and math.isfinite(wd)):
            self._forecast_pressure_trust_current_obs_reason = "invalid_current_wind"
            return
        wd = float(wd % 360.0)
        uv = np.asarray(
            [[-ws * math.sin(math.radians(wd)), -ws * math.cos(math.radians(wd))]],
            dtype=float,
        )
        self._forecast_pressure_trust_current_obs_available = True
        self._forecast_pressure_trust_current_obs_reason = "ok"
        self._forecast_pressure_trust_current_speed_ms = float(max(ws, 0.0))
        self._forecast_pressure_trust_current_dir_deg = wd
        self._forecast_pressure_trust_current_observed_norm = norm_term(
            pressure_proxy_vec(uv, self.cfg),
            self.cfg,
        )

    @staticmethod
    def _pressure_block_dot(
        vecs: list[np.ndarray],
        left: int,
        right: int,
    ) -> float:
        if left >= len(vecs) or right >= len(vecs):
            return 0.0
        a = np.asarray(vecs[left], dtype=float).reshape(-1)
        b = np.asarray(vecs[right], dtype=float).reshape(-1)
        if a.size < 2:
            a = np.pad(a, (0, 2 - a.size))
        if b.size < 2:
            b = np.pad(b, (0, 2 - b.size))
        return float(np.dot(a[:2], b[:2]))

    def _forecast_pressure_trust_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        scales = list(self._forecast_pressure_trust_block_scales)
        while len(scales) < 3:
            scales.append(1.0)
        effective_norms = list(self._effective_pressure_block_norms)
        raw_norms = list(self._raw_pressure_block_norms)
        while len(effective_norms) < 3:
            effective_norms.append(0.0)
        while len(raw_norms) < 3:
            raw_norms.append(0.0)
        return {
            f"{p}forecast_pressure_trust_gate_enabled": int(
                self.forecast_pressure_trust_gate_enabled
            ),
            f"{p}forecast_pressure_trust_trusted": int(
                self._forecast_pressure_trust_trusted
            ),
            f"{p}forecast_pressure_trust_reason": str(
                self._forecast_pressure_trust_reason
            ),
            f"{p}forecast_pressure_trust_block0_scale": float(scales[0]),
            f"{p}forecast_pressure_trust_block1_scale": float(scales[1]),
            f"{p}forecast_pressure_trust_block2_scale": float(scales[2]),
            f"{p}forecast_pressure_trust_current_norm": float(
                self._forecast_pressure_trust_current_norm
            ),
            f"{p}forecast_pressure_trust_future_max_norm": float(
                self._forecast_pressure_trust_future_max_norm
            ),
            f"{p}forecast_pressure_trust_rise_norm": float(
                self._forecast_pressure_trust_rise_norm
            ),
            f"{p}forecast_pressure_trust_relief_margin_norm": float(
                self._forecast_pressure_trust_relief_margin_norm
            ),
            f"{p}forecast_pressure_trust_dot01": float(
                self._forecast_pressure_trust_dot01
            ),
            f"{p}forecast_pressure_trust_dot02": float(
                self._forecast_pressure_trust_dot02
            ),
            f"{p}forecast_pressure_trust_current_support": int(
                self._forecast_pressure_trust_current_support
            ),
            f"{p}forecast_pressure_trust_current_observed_norm": float(
                self._forecast_pressure_trust_current_observed_norm
            ),
            f"{p}forecast_pressure_trust_current_speed_ms": float(
                self._forecast_pressure_trust_current_speed_ms
            ),
            f"{p}forecast_pressure_trust_current_dir_deg": float(
                self._forecast_pressure_trust_current_dir_deg
            ),
            f"{p}forecast_pressure_trust_current_wind_stability_veto": int(
                self._forecast_pressure_trust_current_wind_stability_veto
            ),
            f"{p}forecast_pressure_trust_future_speed_max_ms": float(
                self._forecast_pressure_trust_future_speed_max_ms
            ),
            f"{p}forecast_pressure_trust_event_support": int(
                self._forecast_pressure_trust_event_support
            ),
            f"{p}forecast_pressure_trust_pressure_shape_support": int(
                self._forecast_pressure_trust_pressure_shape_support
            ),
            f"{p}forecast_pressure_trust_spike_veto": int(
                self._forecast_pressure_trust_spike_veto
            ),
            f"{p}effective_pressure_block0_norm": float(effective_norms[0]),
            f"{p}effective_pressure_block1_norm": float(effective_norms[1]),
            f"{p}effective_pressure_block2_norm": float(effective_norms[2]),
            f"{p}effective_pressure_future_max_norm": float(max(effective_norms[1:3])),
            f"{p}raw_pressure_future_max_norm": float(max(raw_norms[1:3])),
        }

    def _apply_forecast_pressure_trust_gate(
        self,
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Shape controller-facing future pressure while preserving raw diagnostics."""
        self._reset_forecast_pressure_trust_state("disabled")
        effective = [dict(b) for b in blocks]
        vecs = [
            np.asarray(
                b.get("pressure_vec", np.zeros(2, dtype=float)),
                dtype=float,
            )
            for b in effective[:3]
        ]
        while len(vecs) < 3:
            vecs.append(np.zeros(2, dtype=float))
        norms = [norm_term(v, self.cfg) for v in vecs[:3]]
        while len(norms) < 3:
            norms.append(0.0)

        current_norm = float(norms[0])
        future_max = float(max(norms[1:3]))
        rise = float(future_max - current_norm)
        relief_margin = float(current_norm - min(norms[1:3]))
        dot01 = self._pressure_block_dot(vecs, 0, 1)
        dot02 = self._pressure_block_dot(vecs, 0, 2)
        max_event = max(float(x) for x in self._event_risk_probs) if self._event_risk_probs else 0.0
        trusted_event = bool(int(self._trusted_event_gate_diag.get("trusted", 1)))
        event_support = bool(
            trusted_event
            and (
                max_event >= self.forecast_pressure_trust_event_threshold
                or int(self._trusted_event_gate_diag.get("highwind_hit", 0)) > 0
                or int(self._trusted_event_gate_diag.get("attention_hits", 0))
                >= self.trusted_event_attention_min_heads
            )
        )
        current_obs_norm = float(self._forecast_pressure_trust_current_observed_norm)
        current_wind_stable = not bool(self._forecast_advised_current_stability_veto)
        forecast_start_speed = float(
            self._forecast_speed_start_ms
            if self._forecast_speed_shape_available
            else self._psc_v2_short_start_ms
        )
        forecast_speed_support = bool(
            self._forecast_speed_range_available
            and self._forecast_speed_range_ms <= 4.0
            and self._forecast_speed_dir_shift_deg <= 45.0
            and forecast_start_speed >= 12.0
        )
        current_support = bool(
            (
                current_obs_norm >= self.forecast_pressure_trust_current_norm_max
                or forecast_speed_support
            )
            and current_wind_stable
        )
        direction_support = bool(
            dot01 >= self.forecast_pressure_trust_min_direction_dot
            and dot02 >= self.forecast_pressure_trust_min_direction_dot
        )
        single_spike = bool(
            future_max >= self.forecast_pressure_trust_future_norm_min
            and abs(float(norms[1] - norms[2]))
            > self.forecast_pressure_trust_spike_margin_norm
            and min(float(norms[1]), float(norms[2]))
            < self.forecast_pressure_trust_future_norm_min
        )
        pressure_shape_support = bool(direction_support and not single_spike)

        self._forecast_pressure_trust_enabled = int(
            self.forecast_pressure_trust_gate_enabled
        )
        self._forecast_pressure_trust_current_norm = current_norm
        self._forecast_pressure_trust_future_max_norm = future_max
        self._forecast_pressure_trust_rise_norm = rise
        self._forecast_pressure_trust_relief_margin_norm = relief_margin
        self._forecast_pressure_trust_dot01 = dot01
        self._forecast_pressure_trust_dot02 = dot02
        self._forecast_pressure_trust_current_support = int(current_support)
        self._forecast_pressure_trust_current_wind_stability_veto = int(
            self._forecast_advised_current_stability_veto
        )
        if self._forecast_speed_range_available:
            self._forecast_pressure_trust_future_speed_max_ms = float(
                max(
                    self._forecast_speed_early_max_ms,
                    self._forecast_speed_near_max_ms,
                    self._forecast_speed_far_max_ms,
                    self._psc_v2_short_max_ms,
                    forecast_start_speed,
                )
            )
        else:
            self._forecast_pressure_trust_future_speed_max_ms = 0.0
        self._forecast_pressure_trust_event_support = int(event_support)
        self._forecast_pressure_trust_pressure_shape_support = int(
            pressure_shape_support
        )
        self._forecast_pressure_trust_spike_veto = int(single_spike)

        if not self.forecast_pressure_trust_gate_enabled:
            self._forecast_pressure_trust_trusted = True
            self._forecast_pressure_trust_reason = "disabled"
        elif not bool(self._forecast_has_future):
            self._forecast_pressure_trust_trusted = True
            self._forecast_pressure_trust_reason = "no_future_preview"
        elif future_max < self.forecast_pressure_trust_future_norm_min:
            self._forecast_pressure_trust_trusted = True
            self._forecast_pressure_trust_reason = "future_pressure_below_gate"
        elif event_support:
            self._forecast_pressure_trust_trusted = True
            self._forecast_pressure_trust_reason = "trusted_event_support"
        elif current_support and pressure_shape_support:
            self._forecast_pressure_trust_trusted = True
            self._forecast_pressure_trust_reason = "current_and_shape_support"
        elif pressure_shape_support and rise <= 0.0:
            self._forecast_pressure_trust_trusted = True
            self._forecast_pressure_trust_reason = "shape_supported_no_rise"
        else:
            self._forecast_pressure_trust_trusted = False
            reasons: list[str] = []
            if not current_support:
                reasons.append("weak_current_support")
            if not event_support:
                reasons.append("no_trusted_event_support")
            if not pressure_shape_support:
                reasons.append(
                    "single_horizon_spike"
                    if single_spike
                    else "direction_inconsistent"
                )
            self._forecast_pressure_trust_reason = (
                "+".join(reasons) if reasons else "untrusted_future_pressure"
            )

        scales = [1.0, 1.0, 1.0]
        if self.forecast_pressure_trust_gate_enabled and not self._forecast_pressure_trust_trusted:
            scales[0] = float(self.forecast_pressure_trust_untrusted_scale)
            scales[1] = float(self.forecast_pressure_trust_untrusted_scale)
            scales[2] = float(self.forecast_pressure_trust_untrusted_scale)
        self._forecast_pressure_trust_block_scales = scales

        shaped: list[dict[str, Any]] = []
        for idx, block in enumerate(effective):
            item = dict(block)
            scale = float(scales[idx]) if idx < len(scales) else 1.0
            pressure_vec = np.asarray(
                item.get("pressure_vec", np.zeros(2, dtype=float)),
                dtype=float,
            )
            item["pressure_vec_untrusted_input"] = pressure_vec.copy()
            item["pressure_vec"] = pressure_vec * scale
            item["pressure_norm"] = norm_term(
                np.asarray(item["pressure_vec"], dtype=float),
                self.cfg,
            )
            item["forecast_pressure_trust_scale"] = scale
            item["forecast_pressure_trust_trusted"] = int(
                self._forecast_pressure_trust_trusted
            )
            item["forecast_pressure_trust_reason"] = str(
                self._forecast_pressure_trust_reason
            )
            shaped.append(item)

        self._effective_pressure_block_vecs = [
            np.asarray(
                b.get("pressure_vec", np.zeros(2, dtype=float)),
                dtype=float,
            )
            for b in shaped[:3]
        ]
        while len(self._effective_pressure_block_vecs) < 3:
            self._effective_pressure_block_vecs.append(np.zeros(2, dtype=float))
        self._effective_pressure_block_norms = [
            norm_term(v, self.cfg) for v in self._effective_pressure_block_vecs[:3]
        ]
        while len(self._effective_pressure_block_norms) < 3:
            self._effective_pressure_block_norms.append(0.0)
        return shaped

    def _reset_forecast_control_trust_state(self, reason: str) -> None:
        self._forecast_control_trust_ok = True
        self._forecast_control_trust_reason = str(reason)
        self._forecast_control_trust_purpose = "none"
        self._forecast_control_trust_pressure_max = 0.0
        self._forecast_control_trust_near_rise = 0.0
        self._forecast_control_trust_dot02 = 0.0
        self._forecast_control_trust_event_prob_max = 0.0
        self._forecast_control_trust_speed_range_ms = 0.0
        self._forecast_control_trust_dir_shift_deg = 0.0
        self._forecast_control_trust_current_stability_veto = False
        self._forecast_control_trust_current_ws_range_limit_ms = 0.0
        self._forecast_control_trust_current_dir_shift_limit_deg = 0.0
        self._forecast_control_trust_by_purpose = {
            "suppression": (True, str(reason)),
            "direction_reversal_hold": (True, str(reason)),
            "gusty_hold_current": (True, str(reason)),
        }

    def _set_forecast_control_trust_result(
        self,
        purpose: str,
        ok: bool,
        reason: str,
    ) -> tuple[bool, str]:
        purpose_key = str(purpose)
        reason_text = str(reason)
        self._forecast_control_trust_ok = bool(ok)
        self._forecast_control_trust_reason = reason_text
        self._forecast_control_trust_by_purpose[purpose_key] = (
            bool(ok),
            reason_text,
        )
        return bool(ok), reason_text

    def _forecast_control_trust_decision(self, purpose: str) -> tuple[bool, str]:
        self._forecast_control_trust_purpose = str(purpose)
        norms = [float(x) for x in self._raw_pressure_block_norms[:3]]
        while len(norms) < 3:
            norms.append(0.0)
        vecs = [np.asarray(v, dtype=float).reshape(-1) for v in self._raw_pressure_block_vecs[:3]]
        while len(vecs) < 3:
            vecs.append(np.zeros(2, dtype=float))
        dot02 = float(np.dot(vecs[0][:2], vecs[2][:2]))
        pressure_max = max(norms)
        near_rise = float(norms[2] - norms[0])
        event_prob_max = max(float(x) for x in self._event_risk_probs) if self._event_risk_probs else 0.0
        self._forecast_control_trust_pressure_max = float(pressure_max)
        self._forecast_control_trust_near_rise = float(near_rise)
        self._forecast_control_trust_dot02 = float(dot02)
        self._forecast_control_trust_event_prob_max = float(event_prob_max)
        self._forecast_control_trust_speed_range_ms = float(self._forecast_speed_range_ms)
        self._forecast_control_trust_dir_shift_deg = float(self._forecast_speed_dir_shift_deg)
        self._forecast_control_trust_current_stability_veto = bool(
            self._forecast_advised_current_stability_veto
        )
        if not self.forecast_control_trust_gate_enabled:
            return self._set_forecast_control_trust_result(
                purpose,
                True,
                "disabled",
            )
        if not bool(self._forecast_has_future):
            return self._set_forecast_control_trust_result(
                purpose,
                False,
                "no_future_preview",
            )
        if (
            self.forecast_control_trust_current_stability_required
            and self._forecast_advised_current_stability_veto
        ):
            return self._set_forecast_control_trust_result(
                purpose,
                False,
                "current_wind_unstable",
            )
        if self.forecast_control_trust_max_speed_range_ms > 0.0:
            if not self._forecast_speed_range_available:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "forecast_speed_range_unavailable",
                )
            if (
                self._forecast_speed_range_ms
                > self.forecast_control_trust_max_speed_range_ms
            ):
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "forecast_speed_range_untrusted",
                )
        if self.forecast_control_trust_max_dir_shift_deg > 0.0:
            if not self._forecast_speed_range_available:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "forecast_direction_shift_unavailable",
                )
            if (
                self._forecast_speed_dir_shift_deg
                > self.forecast_control_trust_max_dir_shift_deg
            ):
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "forecast_direction_shift_untrusted",
                )
        if (
            event_prob_max >= self.forecast_control_trust_event_threshold
            and str(purpose) in {"suppression", "gusty_hold_current", "direction_reversal_hold"}
        ):
            return self._set_forecast_control_trust_result(
                purpose,
                False,
                "event_risk_untrusted_for_economy",
            )
        if str(purpose) == "suppression":
            if near_rise > self.forecast_control_trust_suppression_rise_norm:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "near_pressure_rising_untrusted",
                )
            if pressure_max < self.forecast_control_trust_min_pressure_norm:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "pressure_opportunity_too_low",
                )
        elif str(purpose) == "direction_reversal_hold":
            if dot02 >= 0.0:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "no_reversal_trust",
                )
            if pressure_max < self.forecast_control_trust_min_pressure_norm:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "reversal_pressure_too_low",
                )
        elif str(purpose) == "gusty_hold_current":
            if not (
                self._forecast_speed_gusty_short_candidate
                or self._forecast_speed_gusty_oscillation_candidate
            ):
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "not_gusty_forecast",
                )
            if dot02 <= 0.0:
                return self._set_forecast_control_trust_result(
                    purpose,
                    False,
                    "gusty_direction_boundary",
                )
        return self._set_forecast_control_trust_result(
            purpose,
            True,
            "trusted",
        )

    def _select_objective_mode_cfg(
        self,
        planner_info: dict[str, Any],
        planner_blocks: list[dict[str, Any]],
    ) -> tuple["PlannerConfig", dict[str, Any]]:
        """Decide economic vs recovery mode based on posture persistence and
        future pressure trend; return effective cfg + metadata.

        Default path (objective_mode_active=False) returns self.cfg unchanged
        and "economic" mode label, preserving all prior behavior.
        """
        import dataclasses as _dc

        if not bool(getattr(self.cfg, "objective_mode_active", False)):
            return self.cfg, {
                "objective_mode_active": "economic",
                "objective_mode_persistent_high_buckets": 0,
                "objective_mode_future_rising": 0,
            }

        # 1. Update persistent-high counter based on current posture norm.
        # Counter increments while above threshold (consecutive count for ENTER);
        # tracked separately, _persistent_below_buckets counts consecutive
        # sub-threshold buckets for EXIT hysteresis. This asymmetric design
        # lets recovery latch immediately at high posture but exit only after
        # N sustained sub-threshold buckets, preventing chattering at the
        # boundary.
        posture_vec = np.asarray(
            planner_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)[:2]
        if posture_vec.size < 2:
            posture_vec = np.pad(posture_vec, (0, 2 - posture_vec.size))
        posture_norm_val = norm_term(posture_vec, self.cfg)
        threshold = float(self.cfg.recovery_mode_posture_norm_threshold)
        if posture_norm_val > threshold:
            self._persistent_high_buckets += 1
            self._persistent_below_buckets = 0
        else:
            self._persistent_below_buckets = (
                getattr(self, "_persistent_below_buckets", 0) + 1
            )
            # Decay the high counter for diagnostics only; it is not the latch
            # trigger anymore.
            self._persistent_high_buckets = max(self._persistent_high_buckets - 1, 0)

        # 2. Asymmetric hysteresis: ENTER on first high bucket >= enter_buckets,
        # EXIT only after exit_buckets consecutive below-threshold buckets.
        if self._persistent_high_buckets >= int(self.cfg.recovery_mode_enter_buckets):
            self._objective_mode_latched_recovery = True
        if self._persistent_below_buckets >= int(self.cfg.recovery_mode_exit_buckets):
            self._objective_mode_latched_recovery = False

        # 3. Future-rising signal from raw planner blocks.
        block_norms_raw = [
            norm_term(
                np.asarray(b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))), dtype=float),
                self.cfg,
            )
            for b in planner_blocks[:3]
        ]
        while len(block_norms_raw) < 3:
            block_norms_raw.append(0.0)
        future_rising = (
            max(block_norms_raw[1], block_norms_raw[2])
            > block_norms_raw[0] + float(self.cfg.recovery_mode_future_rising_norm_diff)
        )

        # 4. Recovery mode only when persistent-high AND NOT future-rising.
        use_recovery = self._objective_mode_latched_recovery and not future_rising
        new_label = "recovery" if use_recovery else "economic"
        if new_label != self._objective_mode_last_label:
            self._objective_mode_switch_count += 1
            self._objective_mode_last_label = new_label
        meta = {
            "objective_mode_active": new_label,
            "objective_mode_persistent_high_buckets": int(self._persistent_high_buckets),
            "objective_mode_future_rising": int(future_rising),
            "objective_mode_posture_norm": float(posture_norm_val),
            "objective_mode_switch_count": int(self._objective_mode_switch_count),
            "objective_mode_block_norms_raw": [float(x) for x in block_norms_raw[:3]],
        }
        if not use_recovery:
            return self.cfg, meta
        # Build effective cfg for recovery: smooth_huber + elevated weights.
        recovery_cfg = _dc.replace(
            self.cfg,
            attitude_zone_form="smooth_huber",
            attitude_zone_delta_norm=float(self.cfg.recovery_mode_zone_delta_norm),
            attitude_zone_terminal_delta_norm=float(self.cfg.recovery_mode_zone_delta_norm),
            w_attitude_residual=float(self.cfg.recovery_mode_w_attitude),
            w_terminal_residual=float(self.cfg.recovery_mode_w_terminal),
        )
        return recovery_cfg, meta

    @staticmethod
    def _direction_delta_deg(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na <= 1e-9 or nb <= 1e-9:
            return 0.0
        cosv = float(np.dot(a, b) / (na * nb))
        return float(math.degrees(math.acos(max(-1.0, min(1.0, cosv)))))

    def _observe_current_wind_stability(self, sample) -> None:
        self._forecast_advised_current_ws_range_ms = 0.0
        self._forecast_advised_current_dir_shift_deg = 0.0
        self._forecast_advised_current_history_n = 0
        self._forecast_advised_current_stability_veto = False
        if (
            not self.forecast_advised_economy_enabled
            and not self.forecast_control_trust_current_stability_required
        ):
            return
        if (
            self.forecast_advised_economy_current_ws_range_ms <= 0.0
            and self.forecast_advised_economy_current_dir_shift_deg <= 0.0
        ):
            if not self.forecast_control_trust_current_stability_required:
                return
            ws_limit = 1.5
            dir_limit = 15.0
        else:
            ws_limit = float(self.forecast_advised_economy_current_ws_range_ms)
            dir_limit = float(self.forecast_advised_economy_current_dir_shift_deg)
        self._forecast_control_trust_current_ws_range_limit_ms = float(ws_limit)
        self._forecast_control_trust_current_dir_shift_limit_deg = float(dir_limit)
        span_rows = max(
            2,
            int(round(self.forecast_advised_economy_current_history_s / 600.0)),
        )
        history_end = getattr(sample, "history_end", None)
        if history_end is None:
            return
        trace_source = getattr(
            self.replay_dataset,
            "_trace_rows_all",
            getattr(self.replay_dataset, "trace_rows", []),
        )
        history_start = history_end - timedelta(
            seconds=float(self.forecast_advised_economy_current_history_s)
        )
        trace_rows = [
            row
            for row in trace_source
            if history_start <= row.timestamp <= history_end
        ]
        if len(trace_rows) < 2:
            return
        trace_rows = trace_rows[-span_rows:]
        ws = np.asarray([row.wind_speed_ms for row in trace_rows], dtype=float)
        wd = np.asarray([row.wind_dir_deg for row in trace_rows], dtype=float)
        finite = np.isfinite(ws) & np.isfinite(wd)
        if int(np.sum(finite)) < 2:
            return
        ws = ws[finite]
        wd = wd[finite]
        self._forecast_advised_current_history_n = int(ws.size)
        self._forecast_advised_current_ws_range_ms = float(np.max(ws) - np.min(ws))
        wd_rad = np.radians(wd)
        uv = np.column_stack((-ws * np.sin(wd_rad), -ws * np.cos(wd_rad)))
        split = max(1, int(uv.shape[0]) // 2)
        early = np.mean(uv[:split, :], axis=0)
        late = np.mean(uv[split:, :], axis=0)
        self._forecast_advised_current_dir_shift_deg = self._direction_delta_deg(
            early,
            late,
        )
        ws_veto = ws_limit > 0.0 and self._forecast_advised_current_ws_range_ms > ws_limit
        dir_veto = (
            dir_limit > 0.0
            and self._forecast_advised_current_dir_shift_deg > dir_limit
        )
        self._forecast_advised_current_stability_veto = bool(ws_veto or dir_veto)

    def _reset_forecast_speed_shape_state(self, reason: str) -> None:
        self._forecast_speed_shape_available = False
        self._forecast_speed_shape_reason = str(reason)
        self._forecast_speed_start_ms = 0.0
        self._forecast_speed_early_max_ms = 0.0
        self._forecast_speed_near_max_ms = 0.0
        self._forecast_speed_far_max_ms = 0.0
        self._forecast_speed_late_mean_ms = 0.0
        self._forecast_speed_early_rise_ms = 0.0
        self._forecast_speed_peak_to_late_mean_drop_ms = 0.0
        self._forecast_speed_dir_shift_deg = 0.0
        self._forecast_speed_peak_count = 0
        self._forecast_speed_oscillation_range_ms = 0.0
        self._forecast_speed_near_range_ms = 0.0
        self._forecast_speed_far_range_ms = 0.0
        self._forecast_speed_range_available = False
        self._forecast_speed_range_reason = str(reason)
        self._forecast_speed_range_window_n = 0
        self._forecast_speed_range_ms = 0.0
        self._forecast_speed_relief_decay_strict_candidate = False
        self._forecast_speed_relief_decay_candidate = False
        self._forecast_speed_gusty_oscillation_candidate = False
        self._forecast_speed_gusty_short_candidate = False
        self._forecast_speed_gusty_short_reason = str(reason)
        self._forecast_speed_short_turn_count = 0

    def _reset_psc_v2_short_state(self, reason: str) -> None:
        self._psc_v2_short_available = False
        self._psc_v2_short_reason = str(reason)
        self._psc_v2_short_window_n = 0
        self._psc_v2_short_start_ms = 0.0
        self._psc_v2_short_max_ms = 0.0
        self._psc_v2_short_min_ms = 0.0
        self._psc_v2_short_mean_ms = 0.0
        self._psc_v2_short_early_max_ms = 0.0
        self._psc_v2_short_late_mean_ms = 0.0
        self._psc_v2_short_early_rise_ms = 0.0
        self._psc_v2_short_peak_to_late_mean_drop_ms = 0.0
        self._psc_v2_short_range_ms = 0.0
        self._psc_v2_short_dir_shift_deg = 0.0
        self._psc_v2_short_first_last_drop_ms = 0.0
        self._psc_v2_short_last_minus_first_ms = 0.0

    def _observe_psc_v2_short_shape(self, uv: np.ndarray) -> None:
        """Runtime v2 telemetry from the available 6x10min preview.

        The offline regime-addback-v2 labels used 12 future buckets.  The
        current provider model exposes only six, so this keeps a separate
        short-horizon feature set instead of weakening the original 12-bucket
        ``forecast_speed_shape_*`` semantics.
        """
        arr = np.asarray(uv, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 6 or arr.shape[1] < 2:
            self._reset_psc_v2_short_state("insufficient_horizon")
            return
        window = arr[:6, :2]
        speed = np.linalg.norm(window, axis=1)
        early = speed[:3]
        late = speed[3:6]
        early_vec = np.mean(window[:3, :], axis=0)
        late_vec = np.mean(window[3:6, :], axis=0)
        start = float(speed[0])
        early_max = float(np.max(early))
        late_mean = float(np.mean(late))
        self._psc_v2_short_available = True
        self._psc_v2_short_reason = "ok"
        self._psc_v2_short_window_n = int(speed.shape[0])
        self._psc_v2_short_start_ms = start
        self._psc_v2_short_max_ms = float(np.max(speed))
        self._psc_v2_short_min_ms = float(np.min(speed))
        self._psc_v2_short_mean_ms = float(np.mean(speed))
        self._psc_v2_short_early_max_ms = early_max
        self._psc_v2_short_late_mean_ms = late_mean
        self._psc_v2_short_early_rise_ms = float(early_max - start)
        self._psc_v2_short_peak_to_late_mean_drop_ms = float(early_max - late_mean)
        self._psc_v2_short_range_ms = float(np.max(speed) - np.min(speed))
        self._psc_v2_short_dir_shift_deg = self._direction_delta_deg(
            early_vec,
            late_vec,
        )
        self._psc_v2_short_first_last_drop_ms = float(speed[0] - speed[-1])
        self._psc_v2_short_last_minus_first_ms = float(speed[-1] - speed[0])

    def _observe_forecast_speed_short_gusty(self, uv_arr: np.ndarray) -> None:
        """6-bucket learned-runtime gusty/chatter detector.

        This is separate from the original 12-bucket C3 telemetry. It is meant
        to answer a narrower question: does the available learned preview look
        chattery enough that a young target refresh should be low-pass filtered?
        """
        arr = np.asarray(uv_arr, dtype=float)
        self._forecast_speed_gusty_short_candidate = False
        self._forecast_speed_gusty_short_reason = "insufficient_short_horizon"
        self._forecast_speed_short_turn_count = 0
        if arr.ndim != 2 or arr.shape[0] < 6 or arr.shape[1] < 2:
            return
        window = arr[:6, :2]
        speed = np.linalg.norm(window, axis=1)
        speed_range = float(np.max(speed) - np.min(speed))
        early_vec = np.mean(window[:3, :], axis=0)
        late_vec = np.mean(window[3:6, :], axis=0)
        dir_shift = self._direction_delta_deg(early_vec, late_vec)
        deltas = np.diff(speed)
        eps = 0.05
        signs = [int(np.sign(x)) for x in deltas if abs(float(x)) >= eps]
        turn_count = int(
            sum(1 for prev, cur in zip(signs, signs[1:]) if prev != cur)
        )
        self._forecast_speed_short_turn_count = turn_count
        if speed_range < float(self.gusty_oscillation_short_min_range_ms):
            self._forecast_speed_gusty_short_reason = "short_range_low"
            return
        if turn_count < int(self.gusty_oscillation_short_min_turns):
            self._forecast_speed_gusty_short_reason = "short_monotone"
            return
        if dir_shift > float(self.gusty_oscillation_short_max_dir_shift_deg):
            self._forecast_speed_gusty_short_reason = "short_direction_shift"
            return
        self._forecast_speed_gusty_short_candidate = True
        self._forecast_speed_gusty_short_reason = "short_gusty_candidate"

    def _observe_forecast_speed_shape(self, uv: np.ndarray) -> None:
        uv_arr = np.asarray(uv, dtype=float)
        if uv_arr.ndim == 2 and uv_arr.shape[0] >= 2 and uv_arr.shape[1] >= 2:
            speed_for_range = np.linalg.norm(uv_arr[:, :2], axis=1)
            split = max(1, int(uv_arr.shape[0]) // 2)
            early_vec_for_range = np.mean(uv_arr[:split, :2], axis=0)
            late_vec_for_range = np.mean(uv_arr[split:, :2], axis=0)
            self._forecast_speed_range_available = True
            self._forecast_speed_range_reason = "ok"
            self._forecast_speed_range_window_n = int(speed_for_range.shape[0])
            self._forecast_speed_range_ms = float(
                np.max(speed_for_range) - np.min(speed_for_range)
            )
            self._forecast_speed_dir_shift_deg = self._direction_delta_deg(
                early_vec_for_range,
                late_vec_for_range,
            )
        else:
            self._forecast_speed_range_available = False
            self._forecast_speed_range_reason = "insufficient_horizon"
            self._forecast_speed_range_window_n = 0
            self._forecast_speed_range_ms = 0.0
        if uv_arr.ndim != 2 or uv_arr.shape[0] < 12 or uv_arr.shape[1] < 2:
            self._reset_forecast_speed_shape_state("insufficient_horizon")
            if uv_arr.ndim == 2 and uv_arr.shape[0] >= 2 and uv_arr.shape[1] >= 2:
                speed_for_range = np.linalg.norm(uv_arr[:, :2], axis=1)
                split = max(1, int(uv_arr.shape[0]) // 2)
                early_vec_for_range = np.mean(uv_arr[:split, :2], axis=0)
                late_vec_for_range = np.mean(uv_arr[split:, :2], axis=0)
                self._forecast_speed_range_available = True
                self._forecast_speed_range_reason = "range_only"
                self._forecast_speed_range_window_n = int(speed_for_range.shape[0])
                self._forecast_speed_range_ms = float(
                    np.max(speed_for_range) - np.min(speed_for_range)
                )
                self._forecast_speed_dir_shift_deg = self._direction_delta_deg(
                    early_vec_for_range,
                    late_vec_for_range,
                )
                self._observe_forecast_speed_short_gusty(uv_arr)
            return
        speed = np.linalg.norm(uv_arr[:, :2], axis=1)
        early = np.asarray(speed[:6], dtype=float)
        late = np.asarray(speed[6:12], dtype=float)
        peak_count = int(
            np.sum((speed[1:-1] > speed[:-2]) & (speed[1:-1] > speed[2:]))
        )
        oscillation_range = float(np.max(speed) - np.min(speed))
        early_max = float(np.max(early))
        near_max = float(np.max(speed[:6]))
        far_max = float(np.max(speed[6:12]))
        late_mean = float(np.mean(late))
        start = float(speed[0])
        early_rise = float(early_max - start)
        drop_mean = float(early_max - late_mean)
        near_range = float(np.max(speed[:6]) - np.min(speed[:6]))
        far_range = float(np.max(speed[6:12]) - np.min(speed[6:12]))
        early_vec = np.mean(uv_arr[:6, :2], axis=0)
        late_vec = np.mean(uv_arr[6:12, :2], axis=0)
        dir_shift = self._direction_delta_deg(early_vec, late_vec)
        strict_candidate = (
            early_max >= 16.0
            and early_rise >= 2.0
            and drop_mean >= 3.0
            and dir_shift < 45.0
        )
        gusty_candidate = bool(
            early_max >= 15.0
            and peak_count >= 2
            and drop_mean <= 4.0
            and dir_shift < 45.0
        )
        # The learned h120 forecast often enters the window already near the
        # peak or slightly underestimates the absolute peak.  Keep the original
        # casebook rule as telemetry, but use a softer learned-runtime detector
        # for the default-off dispatcher.
        learned_candidate = early_max >= 14.5 and drop_mean >= 2.0 and dir_shift < 45.0
        self._forecast_speed_shape_available = True
        self._forecast_speed_shape_reason = "ok"
        self._forecast_speed_start_ms = start
        self._forecast_speed_early_max_ms = early_max
        self._forecast_speed_near_max_ms = near_max
        self._forecast_speed_far_max_ms = far_max
        self._forecast_speed_late_mean_ms = late_mean
        self._forecast_speed_early_rise_ms = early_rise
        self._forecast_speed_peak_to_late_mean_drop_ms = drop_mean
        self._forecast_speed_dir_shift_deg = float(dir_shift)
        self._forecast_speed_peak_count = int(peak_count)
        self._forecast_speed_oscillation_range_ms = float(oscillation_range)
        self._forecast_speed_near_range_ms = near_range
        self._forecast_speed_far_range_ms = far_range
        self._forecast_speed_range_available = True
        self._forecast_speed_range_reason = "ok"
        self._forecast_speed_range_window_n = int(speed.shape[0])
        self._forecast_speed_range_ms = float(oscillation_range)
        self._forecast_speed_relief_decay_strict_candidate = bool(strict_candidate)
        self._forecast_speed_relief_decay_candidate = bool(
            strict_candidate or learned_candidate
        )
        self._forecast_speed_gusty_oscillation_candidate = bool(gusty_candidate)
        self._observe_forecast_speed_short_gusty(uv_arr)

    def _forecast_speed_shape_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}forecast_speed_shape_available": int(
                self._forecast_speed_shape_available
            ),
            f"{p}forecast_speed_shape_reason": str(
                self._forecast_speed_shape_reason
            ),
            f"{p}forecast_speed_start_ms": float(self._forecast_speed_start_ms),
            f"{p}forecast_speed_early_max_ms": float(
                self._forecast_speed_early_max_ms
            ),
            f"{p}forecast_speed_near_max_ms": float(
                self._forecast_speed_near_max_ms
            ),
            f"{p}forecast_speed_far_max_ms": float(
                self._forecast_speed_far_max_ms
            ),
            f"{p}forecast_speed_late_mean_ms": float(
                self._forecast_speed_late_mean_ms
            ),
            f"{p}forecast_speed_early_rise_ms": float(
                self._forecast_speed_early_rise_ms
            ),
            f"{p}forecast_speed_peak_to_late_mean_drop_ms": float(
                self._forecast_speed_peak_to_late_mean_drop_ms
            ),
            f"{p}forecast_speed_dir_shift_deg": float(
                self._forecast_speed_dir_shift_deg
            ),
            f"{p}forecast_speed_peak_count": int(self._forecast_speed_peak_count),
            f"{p}forecast_speed_oscillation_range_ms": float(
                self._forecast_speed_oscillation_range_ms
            ),
            f"{p}forecast_speed_near_range_ms": float(
                self._forecast_speed_near_range_ms
            ),
            f"{p}forecast_speed_far_range_ms": float(
                self._forecast_speed_far_range_ms
            ),
            f"{p}forecast_speed_range_available": int(
                self._forecast_speed_range_available
            ),
            f"{p}forecast_speed_range_reason": str(
                self._forecast_speed_range_reason
            ),
            f"{p}forecast_speed_range_window_n": int(
                self._forecast_speed_range_window_n
            ),
            f"{p}forecast_speed_range_ms": float(self._forecast_speed_range_ms),
            f"{p}forecast_speed_relief_decay_strict_candidate": int(
                self._forecast_speed_relief_decay_strict_candidate
            ),
            f"{p}forecast_speed_relief_decay_candidate": int(
                self._forecast_speed_relief_decay_candidate
            ),
            f"{p}forecast_speed_gusty_oscillation_candidate": int(
                self._forecast_speed_gusty_oscillation_candidate
            ),
            f"{p}forecast_speed_gusty_short_candidate": int(
                self._forecast_speed_gusty_short_candidate
            ),
            f"{p}forecast_speed_gusty_short_reason": str(
                self._forecast_speed_gusty_short_reason
            ),
            f"{p}forecast_speed_short_turn_count": int(
                self._forecast_speed_short_turn_count
            ),
        }

    def _psc_v2_short_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}psc_v2_short_available": int(self._psc_v2_short_available),
            f"{p}psc_v2_short_reason": str(self._psc_v2_short_reason),
            f"{p}psc_v2_short_window_n": int(self._psc_v2_short_window_n),
            f"{p}psc_v2_short_start_ms": float(self._psc_v2_short_start_ms),
            f"{p}psc_v2_short_max_ms": float(self._psc_v2_short_max_ms),
            f"{p}psc_v2_short_min_ms": float(self._psc_v2_short_min_ms),
            f"{p}psc_v2_short_mean_ms": float(self._psc_v2_short_mean_ms),
            f"{p}psc_v2_short_early_max_ms": float(
                self._psc_v2_short_early_max_ms
            ),
            f"{p}psc_v2_short_late_mean_ms": float(
                self._psc_v2_short_late_mean_ms
            ),
            f"{p}psc_v2_short_early_rise_ms": float(
                self._psc_v2_short_early_rise_ms
            ),
            f"{p}psc_v2_short_peak_to_late_mean_drop_ms": float(
                self._psc_v2_short_peak_to_late_mean_drop_ms
            ),
            f"{p}psc_v2_short_range_ms": float(self._psc_v2_short_range_ms),
            f"{p}psc_v2_short_dir_shift_deg": float(
                self._psc_v2_short_dir_shift_deg
            ),
            f"{p}psc_v2_short_first_last_drop_ms": float(
                self._psc_v2_short_first_last_drop_ms
            ),
            f"{p}psc_v2_short_last_minus_first_ms": float(
                self._psc_v2_short_last_minus_first_ms
            ),
        }

    def _gusty_oscillation_refresh_gate(
        self,
        plant_info: dict[str, Any],
        current_time: float | None = None,
    ) -> tuple[bool, str]:
        """Default-off thin forecast-shape gate for gusty oscillation smoothing.

        The gate is intentionally narrow: if the forecast does not look like a
        gusty oscillation candidate, or if boundary/fallback/high-posture
        conditions are present, it does nothing. When active, it only blocks
        refresh-style target churn until the active target has aged enough to
        avoid chasing each short peak.
        """
        self._gusty_oscillation_gate_active = False
        self._gusty_oscillation_gate_reason = "disabled"
        if not self.gusty_oscillation_enabled:
            return False, "disabled"
        trust_ok, trust_reason = self._forecast_control_trust_decision(
            "gusty_hold_current"
        )
        if self.forecast_control_trust_gate_enabled and not trust_ok:
            self._gusty_oscillation_gate_reason = (
                f"forecast_control_trust_veto:{trust_reason}"
            )
            return False, self._gusty_oscillation_gate_reason
        if (
            current_time is not None
            and bool(self.gusty_hold_current_runtime_release_enabled)
            and str(self.gusty_hold_current_gate_mode) == "gate_on"
            and float(current_time)
            < float(self._gusty_hold_current_runtime_release_until_s)
        ):
            self._gusty_oscillation_gate_reason = "runtime_release_cooldown"
            return False, "runtime_release_cooldown"
        if bool(self.gusty_oscillation_short_gate_enabled):
            if not self._forecast_speed_gusty_short_candidate:
                reason = f"not_short_gusty:{self._forecast_speed_gusty_short_reason}"
                self._gusty_oscillation_gate_reason = reason
                return False, reason
            pressure_norm = max(float(x) for x in self._raw_pressure_block_norms[:3])
            if pressure_norm < float(self.gusty_oscillation_short_min_pressure_norm):
                self._gusty_oscillation_gate_reason = "short_pressure_low"
                return False, "short_pressure_low"
            dot02 = float(
                np.dot(self._raw_pressure_block_vecs[0], self._raw_pressure_block_vecs[2])
            )
            if dot02 <= 0.0:
                self._gusty_oscillation_gate_reason = "short_direction_veto"
                return False, "short_direction_veto"
        elif (
            not self._forecast_speed_shape_available
            or not self._forecast_speed_gusty_oscillation_candidate
        ):
            self._gusty_oscillation_gate_reason = "not_gusty"
            return False, "not_gusty"
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if max_axis >= float(self.economy_pump_budget_smart_posture_deg):
            self._gusty_oscillation_gate_reason = "posture_not_safe"
            return False, "posture_not_safe"
        if bool(
            self._reactive_floor_latched
            or int((plant_info or {}).get("pump_fullspeed_any", 0)) > 0
        ):
            self._gusty_oscillation_gate_reason = "hard_or_pump_release"
            return False, "hard_or_pump_release"
        if bool(
            self._far_horizon_reversal
            or self._far_horizon_direction_shift
            or self._far_horizon_hidden_intensification
            or self._h120_scheduler_far_signflip_risk
            or self._h120_scheduler_far_reintensification_after_relief
            or self._h120_oracle_probe_far_reintensification
        ):
            self._gusty_oscillation_gate_reason = "boundary_veto"
            return False, "boundary_veto"
        if bool(
            self._active_effectiveness_fallback_dominated
            or self._hold_relief_debt_fallback_dominated
        ):
            self._gusty_oscillation_gate_reason = "fallback_veto"
            return False, "fallback_veto"
        target_age_s = float(self._active_effectiveness_target_age_s)
        self._gusty_hold_current_debt_guard_active = False
        self._gusty_hold_current_debt_guard_reason = "not_evaluated"
        self._gusty_hold_current_debt_guard_target_motion_kg_s = float(
            abs((plant_info or {}).get("pump_target_motion_kg_s", 0.0) or 0.0)
        )
        target_err_kg = (
            self._current_primary_target_error_mean_kg(plant_info)
            if bool(self._primary_target_initialized)
            else 0.0
        )
        self._gusty_hold_current_debt_guard_target_err_kg = float(target_err_kg)
        if (
            bool(self.gusty_hold_current_debt_guard_enabled)
            and target_age_s >= float(self.gusty_hold_current_debt_guard_min_age_s)
            and (
                self._gusty_hold_current_debt_guard_target_motion_kg_s
                >= float(self.gusty_hold_current_debt_guard_target_motion_kg_s)
                or target_err_kg
                >= float(self.gusty_hold_current_debt_guard_target_err_kg)
            )
        ):
            self._gusty_hold_current_debt_guard_active = True
            if (
                self._gusty_hold_current_debt_guard_target_motion_kg_s
                >= float(self.gusty_hold_current_debt_guard_target_motion_kg_s)
            ):
                self._gusty_hold_current_debt_guard_reason = "target_motion_debt"
            else:
                self._gusty_hold_current_debt_guard_reason = "target_error_debt"
            self._gusty_oscillation_gate_reason = (
                f"debt_guard:{self._gusty_hold_current_debt_guard_reason}"
            )
            return False, self._gusty_oscillation_gate_reason
        if target_age_s < float(self.gusty_oscillation_min_target_age_s):
            if (
                current_time is not None
                and self.gusty_hold_current_gate_mode == "gate_on"
                and self._gusty_hold_current_day_budget_exhausted(
                    float(current_time)
                )
            ):
                self._gusty_oscillation_gate_reason = (
                    f"day_budget:{self._gusty_hold_current_day_budget_reason}"
                )
                return False, self._gusty_oscillation_gate_reason
            self._gusty_oscillation_gate_active = True
            self._gusty_oscillation_gate_reason = "gusty_oscillation_target_age_hold"
            return True, "gusty_oscillation_target_age_hold"
        self._gusty_oscillation_gate_reason = "gusty_candidate_no_hold"
        return False, "gusty_candidate_no_hold"

    def _refresh_gusty_hold_current_day_budget(self, current_time: float) -> None:
        if not bool(self.gusty_hold_current_day_budget_enabled):
            self._gusty_hold_current_day_budget_active = False
            self._gusty_hold_current_day_budget_blocked = False
            self._gusty_hold_current_day_budget_reason = "disabled"
            self._gusty_hold_current_day_budget_remaining_buckets = 0
            return
        window_s = float(self.gusty_hold_current_day_budget_window_s)
        t = max(float(current_time), 0.0)
        if (
            self._gusty_hold_current_day_budget_window_start_s <= 0.0
            or t - float(self._gusty_hold_current_day_budget_window_start_s)
            >= window_s
        ):
            self._gusty_hold_current_day_budget_window_start_s = t
            self._gusty_hold_current_day_budget_used_buckets = 0
        max_buckets = int(self.gusty_hold_current_day_budget_max_buckets)
        remaining = (
            max(max_buckets - int(self._gusty_hold_current_day_budget_used_buckets), 0)
            if max_buckets > 0
            else 0
        )
        self._gusty_hold_current_day_budget_active = True
        self._gusty_hold_current_day_budget_remaining_buckets = int(remaining)
        if max_buckets > 0 and remaining <= 0:
            self._gusty_hold_current_day_budget_blocked = True
            self._gusty_hold_current_day_budget_reason = "exhausted"
        else:
            self._gusty_hold_current_day_budget_blocked = False
            self._gusty_hold_current_day_budget_reason = "available"

    def _gusty_hold_current_day_budget_exhausted(
        self,
        current_time: float,
    ) -> bool:
        self._refresh_gusty_hold_current_day_budget(current_time)
        return bool(self._gusty_hold_current_day_budget_blocked)

    def _record_gusty_hold_current_day_budget_use(self, current_time: float) -> None:
        if not bool(self.gusty_hold_current_day_budget_enabled):
            return
        self._refresh_gusty_hold_current_day_budget(current_time)
        if self._gusty_hold_current_day_budget_blocked:
            return
        self._gusty_hold_current_day_budget_used_buckets += 1
        self._refresh_gusty_hold_current_day_budget(current_time)

    def _gusty_hold_current_runtime_release_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> tuple[bool, str]:
        """Release a previously-entered C3 hold when posture drifts upward.

        The entry gate runs only at forecast/target decision buckets.  The
        posture cost in the balanced C3 runs appears between those buckets, so
        this guard watches the live state and releases before hard-floor logic
        has to own the correction.
        """
        self._gusty_hold_current_runtime_released = False
        self._gusty_hold_current_runtime_release_reason = "not_evaluated"
        if not bool(self.gusty_hold_current_runtime_release_enabled):
            return False, "runtime_release_disabled"
        if str(self.gusty_hold_current_gate_mode) != "gate_on":
            return False, "gate_not_on"
        if not bool(self._gusty_hold_current_latched_active):
            return False, "not_latched"
        if float(current_time) < float(self._gusty_hold_current_runtime_release_until_s):
            return True, "runtime_release_cooldown"
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if max_axis >= float(self.gusty_hold_current_runtime_release_deg):
            return True, "runtime_posture_release"
        if bool(
            self._reactive_floor_latched
            or int((plant_info or {}).get("pump_fullspeed_any", 0)) > 0
        ):
            return True, "runtime_hard_or_pump_release"
        return False, "runtime_hold_ok"

    def _gusty_oscillation_scaled_target_blend(self, plant_info: dict[str, Any]) -> float:
        """Return a posture-headroom-aware blend scale for gusty refresh churn.

        Lower values hold more of the existing target, removing repeated-peak
        chatter.  Higher values pursue the newly proposed target more normally.
        """
        base = float(self.gusty_oscillation_target_scale)
        if not bool(self.gusty_oscillation_headroom_scaled):
            self._gusty_oscillation_last_scale = base
            return base
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        low = float(self.gusty_oscillation_low_posture_deg)
        high = float(self.gusty_oscillation_high_posture_deg)
        low_scale = float(self.gusty_oscillation_low_posture_scale)
        high_scale = float(self.gusty_oscillation_high_posture_scale)
        if high <= low + 1e-9:
            frac = 1.0
        else:
            frac = min(max((max_axis - low) / (high - low), 0.0), 1.0)
        scaled = low_scale + frac * (high_scale - low_scale)
        # Keep the configured base as the lower bound so environment overrides
        # cannot accidentally make the refined profile more aggressive than
        # requested.
        scale = min(max(scaled, base, 0.0), 1.0)
        self._gusty_oscillation_last_scale = scale
        return scale

    def _gusty_oscillation_soft_cap_action(
        self,
        first_avec: np.ndarray,
        plant_info: dict[str, Any],
    ) -> tuple[np.ndarray, bool, str]:
        """Reduce, rather than block, target pursuit in gusty forecast windows."""
        if str(self.economy_pump_budget_allocator_mode) != "gusty_oscillation_soft_cap_auto":
            return first_avec, False, "mode_inactive"
        # The gusty specialist is implemented at the target-refresh boundary
        # below as a partial refresh, not as an absolute scaling of the newly
        # proposed target.  Absolute scaling was too close to broad relaxation:
        # it can save pump by simply pulling the target toward zero rather
        # than by filtering oscillatory refresh churn.
        return first_avec, False, "refresh_blend_only"
        if not (
            self.gusty_oscillation_enabled
            and self._forecast_has_future
            and self._forecast_speed_shape_available
            and self._forecast_speed_gusty_oscillation_candidate
        ):
            return first_avec, False, "not_gusty"
        info = plant_info or {}
        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if (
            self._reactive_floor_latched
            or max_axis >= float(self.economy_pump_budget_smart_posture_deg)
            or int(info.get("pump_fullspeed_any", 0)) > 0
        ):
            return first_avec, False, "hard_or_posture_release"
        if bool(
            self._far_horizon_reversal
            or self._far_horizon_direction_shift
            or self._far_horizon_hidden_intensification
            or self._h120_scheduler_far_signflip_risk
            or self._h120_scheduler_far_reintensification_after_relief
            or self._h120_oracle_probe_far_reintensification
        ):
            return first_avec, False, "boundary_veto"
        if bool(
            self._active_effectiveness_fallback_dominated
            or self._hold_relief_debt_fallback_dominated
        ):
            return first_avec, False, "fallback_veto"
        scale = float(self.gusty_oscillation_target_scale)
        if scale >= 0.999:
            return first_avec, False, "scale_noop"
        avec = np.asarray(first_avec, dtype=float).reshape(2)
        if float(np.linalg.norm(avec)) <= 1e-12:
            return first_avec, False, "zero_action"
        return avec * scale, True, "gusty_soft_cap:target_scale"

    def _reset_far_horizon_state(self, reason: str) -> None:
        self._far_horizon_available = False
        self._far_horizon_reason = str(reason)
        self._far_horizon_vecs = [np.zeros(2, dtype=float) for _ in range(6)]
        self._far_horizon_norms = [0.0 for _ in range(6)]
        self._far_horizon_near_max = 0.0
        self._far_horizon_near_last = 0.0
        self._far_horizon_far_min = 0.0
        self._far_horizon_far_max = 0.0
        self._far_horizon_near_weighted_sum = 0.0
        self._far_horizon_far_weighted_sum = 0.0
        self._far_horizon_far_over_near_weighted = 0.0
        self._far_horizon_dir_shift_deg = 0.0
        self._far_horizon_dot_60_120 = 0.0
        self._far_horizon_hidden_relief = False
        self._far_horizon_hidden_intensification = False
        self._far_horizon_reversal = False
        self._far_horizon_direction_shift = False
        self._far_horizon_hint_any = False
        self._far_horizon_relief_gate_active = False
        self._far_horizon_relief_gate_reason = str(reason)

    def _observe_far_horizon(self, uv: np.ndarray) -> None:
        if not self.far_horizon_enabled:
            self._reset_far_horizon_state("disabled")
            return
        uv_arr = np.asarray(uv, dtype=float)
        if uv_arr.ndim != 2 or uv_arr.shape[0] < 12:
            self._reset_far_horizon_state("insufficient_horizon")
            return

        slices = ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12))
        vecs = [np.asarray(pressure_proxy_vec(uv_arr[start:end], self.cfg), dtype=float) for start, end in slices]
        norms = [float(norm_term(v, self.cfg)) for v in vecs]
        near = np.asarray(norms[:3], dtype=float)
        far = np.asarray(norms[3:6], dtype=float)
        near_w = np.asarray(self.block_discounts[:3], dtype=float)
        if near_w.size < 3:
            near_w = np.pad(near_w, (0, 3 - near_w.size), constant_values=1.0)
        far_w = np.asarray(self.far_horizon_weights, dtype=float)
        if far_w.size < 3:
            far_w = np.pad(far_w, (0, 3 - far_w.size), constant_values=0.0)

        near_max = float(np.max(near))
        near_last = float(near[-1])
        far_min = float(np.min(far))
        far_max = float(np.max(far))
        dot_60_120 = float(np.dot(vecs[2], vecs[5]))
        dir_shift = self._direction_delta_deg(vecs[2], vecs[5])

        hidden_relief = (
            near_max >= 1.0
            and far_min <= max(0.75, near_max - 0.35)
            and far_min <= near_last - 0.25
        )
        hidden_intensification = (
            near_max <= 0.85
            and far_max >= 1.10
            and far_max >= near_max + 0.35
        )
        far_reversal = near_max >= 0.55 and far_max >= 0.55 and dot_60_120 < 0.0
        far_direction_shift = near_max >= 0.55 and far_max >= 0.55 and dir_shift >= 60.0

        near_weighted = float(np.sum(near * near_w[:3]))
        far_weighted = float(np.sum(far * far_w[:3]))
        self._far_horizon_available = True
        self._far_horizon_reason = "ok"
        self._far_horizon_vecs = vecs
        self._far_horizon_norms = norms
        self._far_horizon_near_max = near_max
        self._far_horizon_near_last = near_last
        self._far_horizon_far_min = far_min
        self._far_horizon_far_max = far_max
        self._far_horizon_near_weighted_sum = near_weighted
        self._far_horizon_far_weighted_sum = far_weighted
        self._far_horizon_far_over_near_weighted = far_weighted / max(near_weighted, 1e-9)
        self._far_horizon_dir_shift_deg = float(dir_shift)
        self._far_horizon_dot_60_120 = dot_60_120
        self._far_horizon_hidden_relief = bool(hidden_relief)
        self._far_horizon_hidden_intensification = bool(hidden_intensification)
        self._far_horizon_reversal = bool(far_reversal)
        self._far_horizon_direction_shift = bool(far_direction_shift)
        self._far_horizon_hint_any = bool(
            hidden_relief or hidden_intensification or far_reversal or far_direction_shift
        )
        self._far_horizon_relief_gate_active = False
        self._far_horizon_relief_gate_reason = "not_evaluated"

    def _far_horizon_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        return {
            f"{p}far_horizon_enabled": int(self.far_horizon_enabled),
            f"{p}far_horizon_relief_gate_enabled": int(
                self.far_horizon_relief_gate_enabled
            ),
            f"{p}far_horizon_available": int(self._far_horizon_available),
            f"{p}far_horizon_reason": self._far_horizon_reason,
            f"{p}far_horizon_norm_0_20": float(norms[0]),
            f"{p}far_horizon_norm_20_40": float(norms[1]),
            f"{p}far_horizon_norm_40_60": float(norms[2]),
            f"{p}far_horizon_norm_60_80": float(norms[3]),
            f"{p}far_horizon_norm_80_100": float(norms[4]),
            f"{p}far_horizon_norm_100_120": float(norms[5]),
            f"{p}far_horizon_near_max": float(self._far_horizon_near_max),
            f"{p}far_horizon_near_last": float(self._far_horizon_near_last),
            f"{p}far_horizon_far_min": float(self._far_horizon_far_min),
            f"{p}far_horizon_far_max": float(self._far_horizon_far_max),
            f"{p}far_horizon_near_weighted_sum": float(
                self._far_horizon_near_weighted_sum
            ),
            f"{p}far_horizon_far_weighted_sum": float(
                self._far_horizon_far_weighted_sum
            ),
            f"{p}far_horizon_far_over_near_weighted": float(
                self._far_horizon_far_over_near_weighted
            ),
            f"{p}far_horizon_dir_shift_deg": float(self._far_horizon_dir_shift_deg),
            f"{p}far_horizon_dot_60_120": float(self._far_horizon_dot_60_120),
            f"{p}far_horizon_hidden_relief": int(self._far_horizon_hidden_relief),
            f"{p}far_horizon_hidden_intensification": int(
                self._far_horizon_hidden_intensification
            ),
            f"{p}far_horizon_reversal": int(self._far_horizon_reversal),
            f"{p}far_horizon_direction_shift": int(self._far_horizon_direction_shift),
            f"{p}far_horizon_hint_any": int(self._far_horizon_hint_any),
            f"{p}far_horizon_relief_gate_active": int(
                self._far_horizon_relief_gate_active
            ),
            f"{p}far_horizon_relief_gate_reason": (
                self._far_horizon_relief_gate_reason
            ),
        }

    def _run_planner(
        self,
        sample,
        plant_info: dict,
        forecast_has_future: bool = True,
    ) -> tuple[dict[str, Any], np.ndarray, str, list[dict[str, Any]]]:
        uv, forecast_source, event_probs = self._forecast_uv(sample)
        self._forecast_has_future = bool(forecast_has_future)
        self._forecast_event_probs_available = complete_event_probabilities_available(
            event_probs,
            self.replay_dataset.event_columns,
        )
        self._observe_current_wind_stability(sample)
        self._observe_forecast_pressure_current_support(
            getattr(sample, "wind_obs", None)
        )
        if self._forecast_has_future:
            self._observe_forecast_speed_shape(uv)
            self._observe_psc_v2_short_shape(uv)
            self._observe_far_horizon(uv)
        else:
            self._reset_forecast_speed_shape_state("no_future_preview")
            self._reset_psc_v2_short_state("no_future_preview")
            self._reset_far_horizon_state("no_future_preview")
        self._reset_h120_oracle_probe_bucket_state("not_evaluated")
        plant_info = dict(plant_info)
        plant_info["forecast_has_future"] = int(self._forecast_has_future)
        block_discounts = self.block_discounts
        if not self._forecast_has_future:
            # A no-preview source repeats the current wind only as an unknown
            # future placeholder. Discounting those repeated blocks would make
            # "unknown" look like forecast relief, so keep the repeated current
            # pressure flat across the planner horizon.
            block_discounts = [1.0 for _ in self.block_discounts]
        raw_blocks = compute_pressure_blocks(uv, block_discounts, self.cfg)
        self._raw_pressure_block_vecs = [
            np.asarray(
                b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                dtype=float,
            )
            for b in raw_blocks[:3]
        ]
        while len(self._raw_pressure_block_vecs) < 3:
            self._raw_pressure_block_vecs.append(np.zeros(2, dtype=float))
        self._raw_pressure_block_norms = [
            norm_term(v, self.cfg) for v in self._raw_pressure_block_vecs[:3]
        ]
        while len(self._raw_pressure_block_norms) < 3:
            self._raw_pressure_block_norms.append(0.0)
        plant_info = self._hold_relief_debt_observe_before_planner(
            plant_info,
            self._raw_pressure_block_norms,
        )
        self._update_relief_decay_auto_v2_latch(plant_info)
        self._event_risk_raw_probs = [
            float(event_probs.get(key, 0.0)) for key in _EVENT_RISK_KEYS
        ]
        effective_event_probs, gate_diag = self._trusted_event_gate(event_probs)
        self._trusted_event_gate_diag = gate_diag
        event_probs_vec, event_scales = self._event_risk_scales(effective_event_probs)
        self._event_risk_probs = event_probs_vec
        self._event_risk_block_scales = event_scales
        blocks = raw_blocks
        blocks = self._apply_forecast_pressure_trust_gate(blocks)
        planner_blocks, posture_meta = apply_posture_state_to_blocks(
            blocks,
            plant_info,
            self.cfg,
        )
        planner_info = dict(plant_info)
        planner_info.update(posture_meta)
        # Objective-mode switch (default-off prototype). When active, decide
        # between economic (quadratic, preemptive-sensitive) and recovery
        # (smooth_huber, deadzone) cost shapes based on persistent_high_posture
        # AND future_rising_pressure booleans. The chosen cfg is used for
        # this bucket only; other state (block discounts, posture residual,
        # etc.) is untouched.
        active_cfg, mode_meta = self._select_objective_mode_cfg(
            planner_info, planner_blocks
        )
        planner_info.update(mode_meta)
        sequences = _ALL_SEQUENCES
        if self.no_preview_myopic_horizon_enabled and not self._forecast_has_future:
            # A strict no-preview controller should choose the immediate action
            # from current state/pressure only, not optimize a 60-minute
            # persistence pseudo-forecast. Keep the same planner/cost/action
            # library, but restrict future blocks to hold for this diagnostic.
            sequences = [(a, "hold", "hold") for a in ACTIONS]
        evaluated = [
            evaluate_sequence(seq, planner_blocks, planner_info, active_cfg)
            for seq in sequences
        ]
        # Diagnostic: count how many candidate sequences were rejected by the
        # safety_floor specifically. Helps distinguish "mechanism never fires"
        # from "mechanism fires but other feasible sequences also exist".
        safety_floor_reject_count = sum(
            1 for r in evaluated
            if isinstance(r.get("hard_reject_reason"), str)
            and r["hard_reject_reason"].startswith("safety_floor_violated")
        )
        total_sequences = len(evaluated)
        if self.objective_mode == "economic":
            feasible, scalars = select_best_economic(evaluated, active_cfg)
            best_scalar = float(scalars[0]) if scalars else float("nan")
        else:
            feasible = sorted(
                [r for r in evaluated if not r["hard_reject_reason"]],
                key=cmp_to_key(lambda a, b: compare_sequences(a, b, active_cfg)[0]),
            )
            best_scalar = float("nan")
        if not feasible:
            # Safety-floor planned reactive handoff: if every candidate is
            # hard-rejected specifically by safety_floor_violated, return the
            # "least-bad" sequence (smallest predicted safety excursion) with
            # planned_reactive_handoff=1, instead of falling back to hold.
            # Hold-then-do-nothing would push pitch over IEC 5° in this
            # operating window; the safety-floor reject already proved every
            # sequence violates that. The least-bad sequence is by definition
            # the most aggressive action the planner has — which matches what
            # a reactive PI controller would emit in the same conditions.
            safety_rejects = [
                r for r in evaluated
                if isinstance(r.get("hard_reject_reason"), str)
                and r["hard_reject_reason"].startswith("safety_floor_violated")
            ]
            if safety_rejects and bool(getattr(active_cfg, "safety_floor_active", False)):
                def _safety_max_excursion(r: dict[str, Any]) -> float:
                    pitch = abs(float(r.get("safety_floor_predicted_pitch", 0.0)))
                    roll = abs(float(r.get("safety_floor_predicted_roll", 0.0)))
                    return max(pitch, roll)
                best_effort = min(safety_rejects, key=_safety_max_excursion)
                best = dict(best_effort)
                best["planned_reactive_handoff"] = 1
                best["hard_reject_reason"] = ""
                best["selection_reason"] = "planned_reactive_handoff"
                best["best_scalar_cost"] = float("nan")
                best["objective_mode_active"] = str(
                    mode_meta.get("objective_mode_active", "economic")
                )
                best["objective_mode_persistent_high_buckets"] = int(
                    mode_meta.get("objective_mode_persistent_high_buckets", 0)
                )
                best["objective_mode_future_rising"] = int(
                    mode_meta.get("objective_mode_future_rising", 0)
                )
                first_action = best["sequence"][0]
                first_avec, _ = planner_action_vec(
                    first_action,
                    0,
                    planner_blocks,
                    planner_info,
                    active_cfg,
                )
                return best, first_avec, forecast_source, blocks
            return {"sequence": ("hold", "hold", "hold"), "first_action": "hold",
                    "best_scalar_cost": best_scalar,
                    "planned_reactive_handoff": 0}, np.zeros(2), forecast_source, blocks
        best = dict(feasible[0])
        best["best_scalar_cost"] = best_scalar
        best["planned_reactive_handoff"] = 0
        best["safety_floor_reject_count"] = int(safety_floor_reject_count)
        best["safety_floor_total_sequences"] = int(total_sequences)
        # Stamp the active objective mode into best for downstream logging.
        best["objective_mode_active"] = str(mode_meta.get("objective_mode_active", "economic"))
        best["objective_mode_persistent_high_buckets"] = int(
            mode_meta.get("objective_mode_persistent_high_buckets", 0)
        )
        best["objective_mode_future_rising"] = int(
            mode_meta.get("objective_mode_future_rising", 0)
        )
        first_action = best["sequence"][0]
        first_avec, _ = planner_action_vec(
            first_action,
            0,
            planner_blocks,
            planner_info,
            active_cfg,
        )
        return best, first_avec, forecast_source, blocks

    def _relief_medium_cap_action(
        self,
        current_action: str,
        first_avec: np.ndarray,
        blocks: list[dict[str, Any]],
        state,
    ) -> tuple[str, np.ndarray]:
        """Cap active_medium to a smaller, forecast-relief action in comfortable windows."""
        self._relief_medium_cap_active = False
        self._relief_medium_cap_reason = "disabled"
        self._relief_medium_cap_relief_margin = 0.0
        if not self.relief_medium_cap_enabled:
            return current_action, first_avec
        if str(current_action) != "active_medium":
            self._relief_medium_cap_reason = "current_action_not_medium"
            return current_action, first_avec
        if len(blocks) < 3:
            self._relief_medium_cap_reason = "missing_blocks"
            return current_action, first_avec

        if state is not None:
            state_arr = np.asarray(state, dtype=float).reshape(-1)
            if state_arr.size > 4:
                pitch_abs = abs(float(np.degrees(state_arr[4])))
                roll_abs = abs(float(np.degrees(state_arr[3])))
                if pitch_abs > self.relief_medium_cap_pitch_abs_max_deg:
                    self._relief_medium_cap_reason = "current_pitch_not_comfortable"
                    return current_action, first_avec
                if roll_abs > self.relief_medium_cap_roll_abs_max_deg:
                    self._relief_medium_cap_reason = "current_roll_not_comfortable"
                    return current_action, first_avec

        max_event = max(float(x) for x in self._event_risk_probs) if self._event_risk_probs else 0.0
        if max_event < self.relief_medium_cap_event_threshold:
            self._relief_medium_cap_reason = "event_risk_not_confident"
            return current_action, first_avec

        raw_vecs = [
            np.asarray(
                b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                dtype=float,
            )
            for b in blocks[:3]
        ]
        raw_norms = [norm_term(v, self.cfg) for v in raw_vecs]
        while len(raw_norms) < 3:
            raw_norms.append(0.0)
        while len(raw_vecs) < 3:
            raw_vecs.append(np.zeros(2, dtype=float))
        relief_margin = raw_norms[0] - raw_norms[2]
        self._relief_medium_cap_relief_margin = float(relief_margin)
        if relief_margin < _FORECAST_RELIEF_MARGIN_NORM:
            self._relief_medium_cap_reason = "raw_future_relief_too_small"
            return current_action, first_avec
        if raw_norms[0] < 1.0:
            self._relief_medium_cap_reason = "current_pressure_not_high"
            return current_action, first_avec

        vec0 = raw_vecs[0]
        vec2 = raw_vecs[2]
        if float(np.dot(vec0, vec2)) < 0.0:
            self._relief_medium_cap_reason = "future_direction_flip"
            return current_action, first_avec

        first_vec = np.asarray(first_avec, dtype=float).reshape(-1)
        roll_pitch_ratio = 0.0
        if first_vec.size >= 2:
            roll_pitch_ratio = abs(float(first_vec[1])) / max(abs(float(first_vec[0])), 1e-9)

        pressure_vec = np.asarray(blocks[0]["pressure_vec"], dtype=float)
        pressure_norm = float(np.linalg.norm(pressure_vec))
        if pressure_norm <= 1e-9:
            self._relief_medium_cap_reason = "cap_vector_zero"
            return current_action, first_avec
        direction = pressure_vec / pressure_norm
        db = np.array([self.cfg.deadband_pitch_deg, self.cfg.deadband_roll_deg], dtype=float)
        cap_ratio = min(
            max(float(self.relief_medium_cap_ratio), float(self.cfg.active_small_ratio)),
            float(self.cfg.active_medium_ratio),
        )
        if self.relief_medium_cap_adaptive:
            if roll_pitch_ratio > 1.0 and max_event >= 0.90:
                cap_ratio = min(
                    max(float(self.relief_medium_cap_ratio), float(self.cfg.active_small_ratio)),
                    float(self.cfg.active_medium_ratio),
                )
                self._relief_medium_cap_reason = "forecast_relief_caps_medium_roll_heavy"
            else:
                cap_ratio = float(self.cfg.active_small_ratio)
                self._relief_medium_cap_reason = "forecast_relief_caps_medium_to_small"
        small_vec = direction * db * cap_ratio
        self._relief_medium_cap_active = True
        if not self.relief_medium_cap_adaptive:
            self._relief_medium_cap_reason = "forecast_relief_caps_medium"
        return "active_small", small_vec

    def _far_horizon_relief_gate_action(
        self,
        current_action: str,
        first_avec: np.ndarray,
        blocks: list[dict[str, Any]],
        state,
    ) -> tuple[str, np.ndarray]:
        """Use 60-120min relief only to cap medium actions in comfortable posture.

        This is intentionally weaker than changing planner costs: the far horizon
        may be noisy, so it can only reduce an active_medium command to the
        local active_small direction when the current attitude is not already in
        a high-risk posture.
        """
        self._far_horizon_relief_gate_active = False
        self._far_horizon_relief_gate_reason = "disabled"
        if not self.far_horizon_relief_gate_enabled:
            return current_action, first_avec
        if not self.far_horizon_enabled:
            self._far_horizon_relief_gate_reason = "far_horizon_disabled"
            return current_action, first_avec
        if not self._far_horizon_available:
            self._far_horizon_relief_gate_reason = self._far_horizon_reason
            return current_action, first_avec
        if not self._far_horizon_hidden_relief:
            self._far_horizon_relief_gate_reason = "no_far_hidden_relief"
            return current_action, first_avec
        if str(current_action) != "active_medium":
            self._far_horizon_relief_gate_reason = "current_action_not_medium"
            return current_action, first_avec
        if state is not None:
            state_arr = np.asarray(state, dtype=float).reshape(-1)
            if state_arr.size > 4:
                pitch_abs = abs(float(np.degrees(state_arr[4])))
                roll_abs = abs(float(np.degrees(state_arr[3])))
                if pitch_abs > self.far_horizon_relief_pitch_abs_max_deg:
                    self._far_horizon_relief_gate_reason = "current_pitch_too_high"
                    return current_action, first_avec
                if roll_abs > self.far_horizon_relief_roll_abs_max_deg:
                    self._far_horizon_relief_gate_reason = "current_roll_too_high"
                    return current_action, first_avec
        if len(blocks) < 1:
            self._far_horizon_relief_gate_reason = "missing_blocks"
            return current_action, first_avec

        pressure_vec = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float)
        pressure_norm = float(np.linalg.norm(pressure_vec))
        if pressure_norm <= 1e-9:
            self._far_horizon_relief_gate_reason = "pressure_vector_zero"
            return current_action, first_avec
        direction = pressure_vec / pressure_norm
        db = np.array([self.cfg.deadband_pitch_deg, self.cfg.deadband_roll_deg], dtype=float)
        small_vec = direction * db * float(self.cfg.active_small_ratio)
        self._far_horizon_relief_gate_active = True
        self._far_horizon_relief_gate_reason = "far_60_120_relief_caps_medium_to_small"
        return "active_small", small_vec

    def _observe_forecast_advised_economy_decision(
        self,
        blocks: list[dict[str, Any]],
        plant_info: dict[str, Any] | None = None,
        current_time: float | None = None,
        proposed_action: str | None = None,
    ) -> tuple[list[float], float]:
        norms = [float(b.get("pressure_norm", 0.0)) for b in blocks[:3]]
        while len(norms) < 3:
            norms.append(0.0)
        vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float) if blocks else np.zeros(2)
        vec2 = np.asarray(blocks[2].get("pressure_vec", np.zeros(2)), dtype=float) if len(blocks) >= 3 else np.zeros(2)
        dot02 = float(np.dot(vec0, vec2))
        plateau_norms = [float(x) for x in self._raw_pressure_block_norms[:3]]
        while len(plateau_norms) < 3:
            plateau_norms.append(0.0)
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        near_min = float(min(plateau_norms))
        near_rise = float(plateau_norms[2] - plateau_norms[0])
        direction_ok = bool(self._relief_envelope_direction_ok(posture))
        reintensify = bool(
            plateau_norms[2] >= _FORECAST_REINTENSIFY_HIGH_NORM
            and plateau_norms[2] > plateau_norms[0] + 0.05
        )
        far_boundary = bool(
            self._far_horizon_reversal
            or self._far_horizon_direction_shift
            or self._far_horizon_hidden_intensification
            or self._h120_scheduler_far_signflip_risk
            or self._h120_scheduler_far_reintensification_after_relief
            or self._h120_oracle_probe_far_reintensification
        )
        fallback_guard_ok = bool(
            not (
                self._active_effectiveness_fallback_dominated
                or self._hold_relief_debt_fallback_dominated
            )
        )
        headroom_deg = float(self.high_posture_enter) - float(max_axis)
        posture_release_veto = bool(
            self.forecast_advised_economy_posture_release_deg > 0.0
            and max_axis >= self.forecast_advised_economy_posture_release_deg
        )
        now_s = (
            float(current_time)
            if current_time is not None
            else float(self._economy_pump_budget_last_s or 0.0)
        )
        if (
            posture_release_veto
            and self.forecast_advised_economy_release_cooldown_s > 0.0
        ):
            self._forecast_advised_economy_release_cooldown_until_s = max(
                float(self._forecast_advised_economy_release_cooldown_until_s),
                now_s + float(self.forecast_advised_economy_release_cooldown_s),
            )
        release_cooldown_veto = bool(
            self.forecast_advised_economy_release_cooldown_s > 0.0
            and now_s < float(self._forecast_advised_economy_release_cooldown_until_s)
        )
        forecast_advised_was_active = bool(self._forecast_advised_economy_candidate)
        pump_rate_vec = np.asarray(
            (plant_info or {}).get("pump_rate_cmd_m3_min", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        pump_rate_total = float(np.sum(np.abs(pump_rate_vec)))
        backlog_total = float(
            (plant_info or {}).get("pump_total_backlog_kg", 0.0) or 0.0
        )
        latch_count = int((plant_info or {}).get("pump_latch_switch_count", 0) or 0)
        latch_delta = max(0, latch_count - int(self._forecast_advised_shadow_prev_latch_count))
        self._forecast_advised_shadow_prev_latch_count = latch_count
        target_motion = float(
            abs((plant_info or {}).get("pump_target_motion_kg_s", 0.0) or 0.0)
        )
        shadow_opportunity_now = bool(
            latch_delta >= self.forecast_advised_economy_shadow_min_latch_delta
            or target_motion
            >= self.forecast_advised_economy_shadow_min_target_motion_kg_s
            or pump_rate_total
            >= self.forecast_advised_economy_shadow_min_pump_rate_m3_min
            or backlog_total >= self.forecast_advised_economy_shadow_min_backlog_kg
        )
        if shadow_opportunity_now and self.forecast_advised_economy_shadow_latch_s > 0.0:
            self._forecast_advised_shadow_latched_until_s = max(
                float(self._forecast_advised_shadow_latched_until_s),
                now_s + float(self.forecast_advised_economy_shadow_latch_s),
            )
        self._forecast_advised_shadow_latched_active = bool(
            now_s < float(self._forecast_advised_shadow_latched_until_s)
        )
        shadow_opportunity = bool(
            shadow_opportunity_now or self._forecast_advised_shadow_latched_active
        )
        self._forecast_advised_shadow_opportunity = shadow_opportunity
        self._forecast_advised_shadow_latch_delta = int(latch_delta)
        self._forecast_advised_shadow_target_motion_kg_s = target_motion
        self._forecast_advised_shadow_pump_rate_m3_min = pump_rate_total
        self._forecast_advised_shadow_backlog_kg = backlog_total
        shadow_opportunity_entry_veto = bool(
            self.forecast_advised_economy_shadow_opportunity_enabled
            and not shadow_opportunity
            and not forecast_advised_was_active
        )
        runaway_release_veto = bool(
            self.forecast_advised_economy_runaway_release_enabled
            and forecast_advised_was_active
            and (
                pump_rate_total
                >= self.forecast_advised_economy_runaway_pump_rate_m3_min
                or backlog_total >= self.forecast_advised_economy_runaway_backlog_kg
            )
        )
        self._forecast_advised_runaway_release = runaway_release_veto
        self._forecast_advised_runaway_pump_rate_m3_min = pump_rate_total
        self._forecast_advised_runaway_backlog_kg = backlog_total
        if (
            runaway_release_veto
            and self.forecast_advised_economy_runaway_cooldown_s > 0.0
        ):
            self._forecast_advised_economy_release_cooldown_until_s = max(
                float(self._forecast_advised_economy_release_cooldown_until_s),
                now_s + float(self.forecast_advised_economy_runaway_cooldown_s),
            )
            release_cooldown_veto = True
        min_axis_entry_veto = bool(
            self.forecast_advised_economy_min_axis_deg > 0.0
            and max_axis < self.forecast_advised_economy_min_axis_deg
            and not forecast_advised_was_active
        )
        low_pressure_entry_veto = bool(
            self.forecast_advised_economy_min_pressure_norm > 0.0
            and near_min < self.forecast_advised_economy_min_pressure_norm
            and not forecast_advised_was_active
        )
        current_stability_entry_veto = bool(
            self._forecast_advised_current_stability_veto
            and not forecast_advised_was_active
        )
        high_stable_clean_gate = bool(
            self._forecast_speed_shape_available
            and self._forecast_speed_early_max_ms
            >= self.forecast_advised_economy_high_stable_early_min_ms
            and self._forecast_speed_near_max_ms
            >= self.forecast_advised_economy_high_stable_near_min_ms
            and self._forecast_speed_far_max_ms
            >= self.forecast_advised_economy_high_stable_far_min_ms
            and self._forecast_speed_near_range_ms
            <= self.forecast_advised_economy_high_stable_near_range_max_ms
            and self._forecast_speed_far_range_ms
            <= self.forecast_advised_economy_high_stable_far_range_max_ms
            and self._forecast_speed_dir_shift_deg
            <= self.forecast_advised_economy_high_stable_dir_shift_max_deg
        )
        high_stable_clean_veto = bool(
            self.forecast_advised_economy_high_stable_clean_required
            and not high_stable_clean_gate
            and not forecast_advised_was_active
        )
        peak_drop_entry_veto = bool(
            self.forecast_advised_economy_min_peak_drop_ms > 0.0
            and self._forecast_speed_shape_available
            and self._forecast_speed_peak_to_late_mean_drop_ms
            < self.forecast_advised_economy_min_peak_drop_ms
            and not forecast_advised_was_active
        )
        trust_ok, trust_reason = self._forecast_control_trust_decision("suppression")
        forecast_control_trust_veto = bool(
            self.forecast_control_trust_gate_enabled and not trust_ok
        )
        forecast_advised_boundary_veto = bool(
            not bool(self._forecast_has_future)
            or (
                self.forecast_advised_economy_max_speed_range_ms > 0.0
                and not self._forecast_speed_range_available
            )
            or forecast_control_trust_veto
            or dot02 < 0.0
            or near_rise > self.forecast_advised_economy_rise_norm
            or (
                self.forecast_advised_economy_max_speed_range_ms > 0.0
                and self._forecast_speed_range_ms
                > self.forecast_advised_economy_max_speed_range_ms
            )
            or (
                self.forecast_advised_economy_max_dir_shift_deg > 0.0
                and self._forecast_speed_dir_shift_deg
                > self.forecast_advised_economy_max_dir_shift_deg
            )
            or current_stability_entry_veto
            or reintensify
            or not direction_ok
            or far_boundary
            or not fallback_guard_ok
            or posture_release_veto
            or release_cooldown_veto
            or runaway_release_veto
            or min_axis_entry_veto
            or low_pressure_entry_veto
            or high_stable_clean_veto
            or peak_drop_entry_veto
            or shadow_opportunity_entry_veto
            or headroom_deg < self.forecast_advised_economy_min_headroom_deg
        )
        forecast_advised_candidate = bool(
            self.forecast_advised_economy_enabled
            and not forecast_advised_boundary_veto
        )
        if forecast_advised_candidate:
            self._forecast_advised_economy_clear_streak += 1
        else:
            self._forecast_advised_economy_clear_streak = 0
        required_confirm_buckets = int(self.forecast_advised_economy_confirm_buckets)
        if (
            self.forecast_advised_economy_hold_confirm_buckets > 0
            and str(proposed_action or "") == "hold"
        ):
            required_confirm_buckets = max(
                required_confirm_buckets,
                int(self.forecast_advised_economy_hold_confirm_buckets),
            )
        forecast_advised_confirmed = bool(
            self._forecast_advised_economy_clear_streak
            >= required_confirm_buckets
        )
        forecast_advised_candidate = bool(
            forecast_advised_candidate and forecast_advised_confirmed
        )
        forecast_advised_mild_boundary_veto = bool(
            not bool(self._forecast_has_future)
            or (
                self.forecast_advised_economy_mild_max_speed_range_ms > 0.0
                and not self._forecast_speed_range_available
            )
            or forecast_control_trust_veto
            or dot02 < 0.0
            or near_rise > self.forecast_advised_economy_mild_rise_norm
            or (
                self.forecast_advised_economy_mild_max_speed_range_ms > 0.0
                and self._forecast_speed_range_ms
                > self.forecast_advised_economy_mild_max_speed_range_ms
            )
            or current_stability_entry_veto
            or reintensify
            or not direction_ok
            or far_boundary
            or not fallback_guard_ok
            or posture_release_veto
            or release_cooldown_veto
            or runaway_release_veto
            or min_axis_entry_veto
            or low_pressure_entry_veto
            or high_stable_clean_veto
            or peak_drop_entry_veto
            or shadow_opportunity_entry_veto
            or headroom_deg < self.forecast_advised_economy_mild_min_headroom_deg
        )
        forecast_advised_mild_candidate = bool(
            self.forecast_advised_economy_enabled
            and self.forecast_advised_economy_mild_enabled
            and not forecast_advised_candidate
            and not forecast_advised_mild_boundary_veto
        )
        if not self.forecast_advised_economy_enabled:
            forecast_advised_reason = "disabled"
        elif forecast_advised_candidate:
            forecast_advised_reason = "forecast_clear_with_posture_headroom"
        elif forecast_advised_mild_candidate:
            forecast_advised_reason = "forecast_clear_with_mild_posture_headroom"
        elif (
            self._forecast_advised_economy_clear_streak > 0
            and not forecast_advised_confirmed
        ):
            forecast_advised_reason = (
                "forecast_clear_hold_confirming"
                if str(proposed_action or "") == "hold"
                else "forecast_clear_confirming"
            )
        elif not bool(self._forecast_has_future):
            forecast_advised_reason = "no_future_forecast"
        elif (
            self.forecast_advised_economy_max_speed_range_ms > 0.0
            and not self._forecast_speed_range_available
        ):
            forecast_advised_reason = "future_speed_range_unavailable_veto"
        elif forecast_control_trust_veto:
            forecast_advised_reason = f"forecast_control_trust_veto:{trust_reason}"
        elif posture_release_veto:
            forecast_advised_reason = "forecast_advised_posture_release"
        elif runaway_release_veto:
            forecast_advised_reason = "forecast_advised_runaway_release"
        elif release_cooldown_veto:
            forecast_advised_reason = "forecast_advised_release_cooldown"
        elif min_axis_entry_veto:
            forecast_advised_reason = "insufficient_posture_deviation"
        elif low_pressure_entry_veto:
            forecast_advised_reason = "insufficient_pressure_opportunity"
        elif high_stable_clean_veto:
            if not self._forecast_speed_shape_available:
                forecast_advised_reason = "high_stable_clean_forecast_unavailable"
            elif (
                self._forecast_speed_early_max_ms
                < self.forecast_advised_economy_high_stable_early_min_ms
            ):
                forecast_advised_reason = "high_stable_clean_early_max_veto"
            elif (
                self._forecast_speed_near_max_ms
                < self.forecast_advised_economy_high_stable_near_min_ms
            ):
                forecast_advised_reason = "high_stable_clean_near_max_veto"
            elif (
                self._forecast_speed_far_max_ms
                < self.forecast_advised_economy_high_stable_far_min_ms
            ):
                forecast_advised_reason = "high_stable_clean_far_max_veto"
            elif (
                self._forecast_speed_near_range_ms
                > self.forecast_advised_economy_high_stable_near_range_max_ms
            ):
                forecast_advised_reason = "high_stable_clean_near_range_veto"
            elif (
                self._forecast_speed_far_range_ms
                > self.forecast_advised_economy_high_stable_far_range_max_ms
            ):
                forecast_advised_reason = "high_stable_clean_far_range_veto"
            elif (
                self._forecast_speed_dir_shift_deg
                > self.forecast_advised_economy_high_stable_dir_shift_max_deg
            ):
                forecast_advised_reason = "high_stable_clean_direction_veto"
            else:
                forecast_advised_reason = "high_stable_clean_veto"
        elif peak_drop_entry_veto:
            forecast_advised_reason = "insufficient_future_peak_drop_veto"
        elif shadow_opportunity_entry_veto:
            forecast_advised_reason = "insufficient_runtime_pump_opportunity"
        elif headroom_deg < self.forecast_advised_economy_min_headroom_deg:
            if (
                self.forecast_advised_economy_mild_enabled
                and headroom_deg
                >= self.forecast_advised_economy_mild_min_headroom_deg
            ):
                forecast_advised_reason = "insufficient_strong_headroom_mild_blocked"
            else:
                forecast_advised_reason = "insufficient_posture_headroom"
        elif dot02 < 0.0:
            forecast_advised_reason = "direction_reversal_veto"
        elif near_rise > self.forecast_advised_economy_rise_norm:
            if (
                self.forecast_advised_economy_mild_enabled
                and near_rise <= self.forecast_advised_economy_mild_rise_norm
            ):
                forecast_advised_reason = "near_pressure_rising_mild_blocked"
            else:
                forecast_advised_reason = "near_pressure_rising_veto"
        elif (
            self.forecast_advised_economy_max_speed_range_ms > 0.0
            and self._forecast_speed_range_ms
            > self.forecast_advised_economy_max_speed_range_ms
        ):
            if (
                self.forecast_advised_economy_mild_enabled
                and self.forecast_advised_economy_mild_max_speed_range_ms > 0.0
                and self._forecast_speed_range_ms
                <= self.forecast_advised_economy_mild_max_speed_range_ms
            ):
                forecast_advised_reason = "future_speed_range_mild_blocked"
            else:
                forecast_advised_reason = "future_speed_range_veto"
        elif (
            self.forecast_advised_economy_max_dir_shift_deg > 0.0
            and self._forecast_speed_dir_shift_deg
            > self.forecast_advised_economy_max_dir_shift_deg
        ):
            forecast_advised_reason = "future_direction_shift_veto"
        elif current_stability_entry_veto:
            if (
                self.forecast_advised_economy_current_ws_range_ms > 0.0
                and self._forecast_advised_current_ws_range_ms
                > self.forecast_advised_economy_current_ws_range_ms
            ):
                forecast_advised_reason = "current_wind_speed_unstable_veto"
            elif (
                self.forecast_advised_economy_current_dir_shift_deg > 0.0
                and self._forecast_advised_current_dir_shift_deg
                > self.forecast_advised_economy_current_dir_shift_deg
            ):
                forecast_advised_reason = "current_wind_direction_unstable_veto"
            else:
                forecast_advised_reason = "current_wind_unstable_veto"
        elif reintensify:
            forecast_advised_reason = "reintensification_veto"
        elif not direction_ok:
            forecast_advised_reason = "axis_direction_veto"
        elif far_boundary:
            forecast_advised_reason = "far_boundary_veto"
        elif not fallback_guard_ok:
            forecast_advised_reason = "fallback_dominated_veto"
        else:
            forecast_advised_reason = "boundary_veto"
        self._forecast_advised_economy_candidate = bool(
            forecast_advised_candidate or forecast_advised_mild_candidate
        )
        self._forecast_advised_economy_boundary_veto = bool(
            forecast_advised_boundary_veto
            and not forecast_advised_mild_candidate
        )
        self._forecast_advised_economy_headroom_deg = headroom_deg
        self._forecast_advised_economy_near_rise_norm = near_rise
        self._forecast_advised_economy_posture_release = posture_release_veto
        self._forecast_advised_economy_max_axis_deg = max_axis
        self._forecast_advised_economy_confirmed = forecast_advised_confirmed
        self._forecast_advised_high_stable_clean_gate = high_stable_clean_gate
        self._forecast_advised_high_stable_clean_reason = (
            "passed" if high_stable_clean_gate else forecast_advised_reason
        )
        self._forecast_advised_economy_reason = forecast_advised_reason
        forecast_advised_latch_veto = bool(
            not self.forecast_advised_economy_enabled
            or forecast_advised_boundary_veto
            or posture_release_veto
            or release_cooldown_veto
        )
        if (
            self.forecast_advised_economy_latch_s <= 0.0
            or forecast_advised_latch_veto
        ):
            self._forecast_advised_economy_latched_until_s = 0.0
            self._forecast_advised_economy_latched_active = False
        elif forecast_advised_candidate:
            # Avoid fragmenting a valid forecast-clear economy episode into
            # one-bucket on/off chatter; hard boundary and posture vetoes above
            # still release immediately.
            self._forecast_advised_economy_latched_until_s = max(
                float(self._forecast_advised_economy_latched_until_s),
                now_s + float(self.forecast_advised_economy_latch_s),
            )
            self._forecast_advised_economy_latched_active = True
        else:
            self._forecast_advised_economy_latched_active = bool(
                now_s < float(self._forecast_advised_economy_latched_until_s)
            )
        return norms, dot02

    def _forecast_advised_pump_suppression_record_fields(self) -> dict[str, Any]:
        fields = frozen_pump_suppression_record_defaults()
        if not (
            self.forecast_advised_economy_enabled
            and self.forecast_advised_economy_suppression_only
        ):
            return fields
        if not self._forecast_advised_economy_candidate:
            fields["pump_suppression_reason"] = str(
                self._forecast_advised_economy_reason
            )
            return fields
        fields.update(
            {
                "pump_suppression_active": 1,
                "pump_suppression_restart_err_kg": float(
                    self.forecast_advised_economy_suppression_restart_err_kg
                ),
                "pump_suppression_reason": (
                    "forecast_advised_economy_suppression_only"
                ),
            }
        )
        return fields

    def _forecast_advised_pump_suppression_preview_fields(self) -> dict[str, Any]:
        fields = frozen_pump_suppression_preview_defaults()
        if not (
            self.forecast_advised_economy_enabled
            and self.forecast_advised_economy_suppression_only
        ):
            return fields
        if not self._forecast_advised_economy_candidate:
            fields["preview_pump_suppression_reason"] = str(
                self._forecast_advised_economy_reason
            )
            return fields
        fields.update(
            {
                "preview_pump_suppression_active": 1,
                "preview_pump_restart_err_kg": float(
                    self.forecast_advised_economy_suppression_restart_err_kg
                ),
                "preview_pump_suppression_reason": (
                    "forecast_advised_economy_suppression_only"
                ),
            }
        )
        return fields

    def _should_reset_event(self, new_action: str, new_pitch: float, new_roll: float,
                            current_time: float) -> bool:
        if new_action != self._target_action:
            return True
        if self.bias_shape != "event_decay" or new_action == "hold":
            return False
        if self.event_reset_mode == "active_bucket":
            return True
        if self.event_reset_mode == "target_change":
            delta = float(np.hypot(new_pitch - self._target_pitch, new_roll - self._target_roll))
            return delta >= self.target_reset_tol_deg
        return False

    def _shaped_bias(self, current_time: float) -> tuple[float, float]:
        if self.bias_shape == "hold":
            return self._target_pitch * self.scale, self._target_roll * self.scale
        # event_decay
        dt_since = max(0.0, float(current_time) - float(self._transition_time_s))
        decay = math.exp(-dt_since / max(self.decay_tau_s, 1e-3))
        return (self._target_pitch * decay * self.scale,
                self._target_roll * decay * self.scale)

    def _shaped_mass_ff_kg(self, current_time: float) -> np.ndarray:
        """Per-tank mass feedforward (kg), planner frame, decayed like bias_shape.

        Uses the SAME bias_shape and decay_tau_s as the setpoint bias. The
        per-tank mass is computed from the planner-frame avec (no sign flip)
        via tank_signal × action_mass_quantum_kg, which matches the planner's
        own internal tank update formula in update_tanks(...).
        """
        if not self.ff_channel_enabled:
            return np.zeros(3, dtype=float)
        if self.bias_shape == "hold":
            decay = 1.0
        else:
            dt_since = max(0.0, float(current_time) - float(self._transition_time_s))
            decay = math.exp(-dt_since / max(self.decay_tau_s, 1e-3))
        avec = np.array([self._target_pitch_planner_frame, self._target_roll_planner_frame],
                        dtype=float) * decay * self.ff_scale
        if float(np.linalg.norm(avec)) < 1e-12:
            return np.zeros(3, dtype=float)
        sig = tank_signal(avec, self.cfg)
        return sig * self.cfg.action_mass_quantum_kg

    def _primary_mass_delta_kg(self, avec: np.ndarray) -> np.ndarray:
        """Mass-domain prediction-primary target delta for one planner action.

        Unlike the FF channel, this is held as an absolute bucket target instead
        of being repeatedly added on top of the PI command. This is the minimal
        "prediction decides pump target" path requested in the pump-first v2
        redesign.
        """
        avec = np.asarray(avec, dtype=float).reshape(2)
        if float(np.linalg.norm(avec)) < 1e-12:
            return np.zeros(3, dtype=float)
        return tank_signal(avec, self.cfg) * self.cfg.action_mass_quantum_kg * self.prediction_primary_scale

    def _update_primary_target(self, plant_info: dict[str, Any], first_avec: np.ndarray,
                               current_time: float) -> None:
        consume_primary_refresh_owner(self)
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        apply_primary_target_transition(
            self,
            primary_target_from_delta(
                masses,
                self._primary_mass_delta_kg(first_avec),
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
            ),
        )
        if float(np.linalg.norm(first_avec)) > 1e-12:
            remember_paused_primary_target(
                self,
                action=str(self._target_action),
                avec=first_avec,
            )

    def _apply_economy_budget_blocked_refresh(
        self,
        plant_info: dict[str, Any],
        refresh_avec: np.ndarray,
        current_time: float,
    ) -> None:
        """Apply the budget-blocked economy branch as hold or partial refresh.

        `economy_pump_budget_hold_refresh_fraction=0` preserves the historical
        behavior: hold the target at current masses.  A value in (0, 1] pursues
        that fraction of the blocked refresh delta, giving No.4 a real mild
        action knob instead of a binary hold/refresh switch.
        """
        fraction = min(
            max(float(self.economy_pump_budget_hold_refresh_fraction), 0.0),
            1.0,
        )
        if self.economy_pump_budget_dynamic_refresh_enabled:
            posture = np.asarray(
                (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
            enter = float(self.economy_pump_budget_dynamic_refresh_enter_deg)
            full = float(self.economy_pump_budget_dynamic_refresh_full_deg)
            if max_axis >= enter:
                span = max(full - enter, 1e-6)
                t = min(max((max_axis - enter) / span, 0.0), 1.0)
                target_fraction = float(
                    self.economy_pump_budget_dynamic_refresh_max_fraction
                )
                fraction = max(fraction, fraction + (target_fraction - fraction) * t)
                fraction = min(max(fraction, 0.0), 1.0)
        self._economy_pump_budget_partial_refresh_fraction = float(fraction)
        if fraction <= 1e-12:
            self._economy_pump_budget_partial_delta_mean_kg = 0.0
            self._hold_primary_target_at_current(plant_info, current_time)
            return
        scaled_avec = np.asarray(refresh_avec, dtype=float).reshape(2) * fraction
        self._economy_pump_budget_partial_refresh_count += 1
        self._economy_pump_budget_partial_delta_mean_kg = float(
            np.mean(np.abs(self._primary_mass_delta_kg(scaled_avec)))
        )
        self._update_primary_target(plant_info, scaled_avec, current_time)

    def _reuse_primary_target(self) -> None:
        mark_primary_target_reused(
            self,
            refresh_owner=str(self._primary_refresh_owner_pending or "reuse"),
        )

    def _direction_reversal_layered_reuse_safe(
        self,
        plant_info: dict[str, Any] | None,
    ) -> bool:
        if str(self.economy_pump_budget_allocator_mode) not in (
            "direction_reversal_layered_auto",
            "direction_reversal_layered_v5_auto",
            "direction_reversal_layered_v6_auto",
            "direction_reversal_layered_v7_auto",
            "direction_reversal_layered_v8_auto",
        ):
            return True
        info = plant_info or {}
        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if max_axis >= float(self.economy_pump_budget_smart_posture_deg):
            self._economy_pump_budget_forced_release_reason = (
                "direction_reversal_layered_reuse_posture_warning"
            )
            self._economy_pump_budget_hold_latched = False
            return False
        allow_hold, reason = self._economy_budget_allocator_decision(info)
        if not allow_hold:
            self._economy_pump_budget_forced_release_reason = (
                f"direction_reversal_layered_reuse_allocator:{reason}"
            )
            self._economy_pump_budget_hold_latched = False
            return False
        return True

    def _reuse_or_refresh_direction_reversal_layered(
        self,
        plant_info: dict[str, Any],
        refresh_avec: np.ndarray,
        current_time: float,
    ) -> None:
        if self._direction_reversal_layered_reuse_safe(plant_info):
            posture = np.asarray(
                (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
            dynamic_reuse_active = bool(
                self.economy_pump_budget_dynamic_refresh_enabled
                and max_axis
                >= float(self.economy_pump_budget_dynamic_refresh_enter_deg)
            )
            posture_release_active = (
                max_axis >= float(self.economy_pump_budget_smart_posture_deg)
                or dynamic_reuse_active
            )
            if (
                self.economy_pump_budget_hold_refresh_fraction > 0.0
                and posture_release_active
            ):
                if str(self._primary_refresh_owner_pending or "unset") in (
                    "",
                    "unset",
                    "reuse",
                ):
                    self._primary_refresh_owner_pending = (
                        "reuse_dynamic_mild_refresh"
                        if dynamic_reuse_active
                        else "reuse_mild_refresh"
                    )
                self._apply_economy_budget_blocked_refresh(
                    plant_info,
                    refresh_avec,
                    current_time,
                )
            else:
                self._reuse_primary_target()
        else:
            if str(self._primary_refresh_owner_pending or "unset") == "unset":
                self._primary_refresh_owner_pending = "direction_reversal_layered_forced_refresh"
            self._update_primary_target(plant_info, refresh_avec, current_time)

    def _direction_reversal_layered_warning_active(
        self,
        plant_info: dict[str, Any] | None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """High-posture direction reversal is a warning, not a saving window."""
        if str(self.economy_pump_budget_allocator_mode) not in (
            "direction_reversal_layered_v7_auto",
            "direction_reversal_layered_v8_auto",
        ):
            return False, "mode_disabled"
        info = plant_info or {}
        posture = np.asarray(
            info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        warning_posture = max(
            float(self.economy_pump_budget_smart_posture_deg),
            min(float(self.high_posture_enter), 3.0),
        )
        if max_axis < warning_posture:
            return False, "posture_below_warning_band"
        if not bool(self._forecast_has_future):
            return False, "no_future_preview"
        if blocks:
            vecs = [
                np.asarray(
                    b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                    dtype=float,
                )
                for b in blocks[:3]
            ]
            norms = [norm_term(v, self.cfg) for v in vecs]
        else:
            vecs = [np.asarray(v, dtype=float) for v in self._raw_pressure_block_vecs[:3]]
            norms = [float(x) for x in self._raw_pressure_block_norms[:3]]
        while len(vecs) < 3:
            vecs.append(np.zeros(2, dtype=float))
        while len(norms) < 3:
            norms.append(0.0)
        if max(float(x) for x in norms[:3]) < 0.65:
            return False, "pressure_below_reversal_window"
        if float(np.dot(vecs[0], vecs[2])) < 0.0:
            return True, "high_posture_near_reversal"
        if not self._direction_reversal_layered_reuse_safe(info):
            return True, str(self._economy_pump_budget_forced_release_reason)
        return False, "no_reversal_warning"

    def _direction_reversal_layered_entry_veto_active(
        self,
        plant_info: dict[str, Any] | None,
        blocks: list[dict[str, Any]] | None = None,
    ) -> tuple[bool, str]:
        """Fail closed before installing targets in the high-load reversal subtype."""
        if str(self.economy_pump_budget_allocator_mode) != "direction_reversal_layered_v8_auto":
            return False, "mode_disabled"
        if not bool(self._forecast_has_future):
            return False, "no_future_preview"
        if blocks:
            vecs = [
                np.asarray(
                    b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                    dtype=float,
                )
                for b in blocks[:3]
            ]
            norms = [norm_term(v, self.cfg) for v in vecs]
        else:
            vecs = [np.asarray(v, dtype=float) for v in self._raw_pressure_block_vecs[:3]]
            norms = [float(x) for x in self._raw_pressure_block_norms[:3]]
        while len(vecs) < 3:
            vecs.append(np.zeros(2, dtype=float))
        while len(norms) < 3:
            norms.append(0.0)
        dot02 = float(np.dot(vecs[0], vecs[2]))
        if max(float(x) for x in norms[:3]) >= 1.20 and dot02 < -0.05:
            return True, "high_pressure_strong_reversal_entry_veto"
        return self._direction_reversal_layered_warning_active(plant_info, blocks)

    def _current_primary_target_error_mean_kg(self, plant_info: dict[str, Any]) -> float:
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        return float(np.mean(np.abs(masses - self._primary_target_kg)))

    def _release_primary_target_after_floor_exit(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        reason: str,
    ) -> None:
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        before = float(np.mean(np.abs(self._primary_target_kg - masses)))
        apply_primary_target_transition(
            self,
            primary_target_at_current(
                masses,
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
                initialized=True,
                clip_target=False,
            ),
        )
        self._reactive_floor_post_exit_active = True
        self._reactive_floor_post_exit_released = True
        self._reactive_floor_post_exit_release_count += 1
        self._reactive_floor_post_exit_reason = reason
        self._reactive_floor_post_exit_delta_before_kg = before
        self._reactive_floor_post_exit_delta_after_kg = 0.0

    def _cap_primary_target_after_floor_exit(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        cap_kg: float,
        reason: str,
    ) -> None:
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        before = float(np.mean(np.abs(self._primary_target_kg - masses)))
        apply_primary_target_transition(
            self,
            primary_target_capped_to_current(
                masses,
                self._primary_target_kg,
                cap_kg=float(cap_kg),
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
            ),
        )
        after = float(np.mean(np.abs(self._primary_target_kg - masses)))
        self._reactive_floor_post_exit_active = True
        self._reactive_floor_post_exit_released = False
        self._reactive_floor_post_exit_cap_count += 1
        self._reactive_floor_post_exit_reason = reason
        self._reactive_floor_post_exit_delta_before_kg = before
        self._reactive_floor_post_exit_delta_after_kg = after

    def _reactive_floor_post_exit_update(
        self,
        state,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> None:
        self._reactive_floor_post_exit_active = False
        self._reactive_floor_post_exit_reason = "disabled"
        self._reactive_floor_post_exit_released = False
        self._reactive_floor_post_exit_delta_before_kg = 0.0
        self._reactive_floor_post_exit_delta_after_kg = 0.0
        if not (
            self.reactive_floor_post_exit_enabled
            and self.reactive_floor_predictive_veto_enabled
            and self.prediction_primary_enabled
            and self._primary_target_initialized
        ):
            return
        if not self._reactive_floor_post_exit_episode_active:
            self._reactive_floor_post_exit_reason = "episode_already_handled"
            return
        if not math.isfinite(float(self._reactive_floor_last_refresh_s)):
            self._reactive_floor_post_exit_reason = "no_floor_refresh"
            return
        if float(self._reactive_floor_last_refresh_s) < -1.0e11:
            self._reactive_floor_post_exit_reason = "no_floor_refresh"
            return

        plant_info = self._attach_posture_info(plant_info, state)
        posture_metrics = reactive_floor_posture_metrics(
            plant_info,
            high_posture_metric=self.high_posture_metric,
            high_posture_enter=self.high_posture_enter,
            high_posture_exit=self.high_posture_exit,
        )
        metric = float(posture_metrics.metric_value_deg)
        prev_metric = float(self._reactive_floor_post_exit_prev_metric_deg)
        if not math.isfinite(prev_metric):
            current_response = float("nan")
            not_worsening = True
        else:
            current_response = prev_metric - float(metric)
            not_worsening = current_response >= -float(
                self.reactive_floor_post_exit_worsening_eps_deg
            )
        self._reactive_floor_post_exit_prev_metric_deg = float(metric)
        self._reactive_floor_post_exit_posture_metric_deg = float(metric)
        self._reactive_floor_post_exit_current_response_deg = float(current_response)

        if metric > float(self.high_posture_exit) + float(
            self.reactive_floor_post_exit_margin_deg
        ):
            self._reactive_floor_post_exit_reason = "above_exit"
            return
        if not not_worsening:
            self._reactive_floor_post_exit_reason = "posture_worsening"
            return
        if self._target_action == "hold":
            self._reactive_floor_post_exit_reason = "hold_action"
            return
        if self.h120_oracle_probe_enabled:
            self._h120_oracle_probe_refresh_signals(plant_info)
            if self._h120_oracle_probe_far_pressure_risk():
                self._h120_oracle_probe_active = True
                self._h120_oracle_probe_early_stop_veto_active = True
                self._h120_oracle_probe_action_effect = "early_stop_veto"
                self._h120_oracle_probe_reason = "far_pressure_vetoes_early_stop"
                self._h120_oracle_probe_early_stop_veto_count += 1
                self._h120_oracle_probe_count += 1
                self._reactive_floor_post_exit_reason = "h120_far_pressure_veto"
                return
        if self.h120_risk_scheduler_enabled:
            # Remote-risk early_stop guard: at most one suppression per floor
            # episode. Never permanently disables early_stop — the next
            # exit-eligible bucket (after the budget is used) will proceed.
            self._h120_scheduler_refresh(plant_info)
            if self._h120_scheduler_try_suppress_early_stop():
                self._reactive_floor_post_exit_reason = "h120_scheduler_remote_risk_suppress"
                return

        if self.reactive_floor_post_exit_mode == "early_stop":
            self._release_primary_target_after_floor_exit(
                plant_info,
                float(current_time),
                "early_stop_after_exit",
            )
            self._reactive_floor_post_exit_episode_active = False
        elif self.reactive_floor_post_exit_mode == "target_cap":
            target_err = self._current_primary_target_error_mean_kg(plant_info)
            if target_err <= float(self.reactive_floor_post_exit_target_cap_kg):
                self._reactive_floor_post_exit_reason = "already_within_cap"
                self._reactive_floor_post_exit_delta_before_kg = float(target_err)
                self._reactive_floor_post_exit_delta_after_kg = float(target_err)
                return
            self._cap_primary_target_after_floor_exit(
                plant_info,
                float(current_time),
                float(self.reactive_floor_post_exit_target_cap_kg),
                "target_cap_after_exit",
            )
            self._reactive_floor_post_exit_episode_active = False

    @staticmethod
    def _active_action_like(action: str) -> bool:
        return str(action).startswith("active")

    def _reactive_floor_log_fields(self, prefix: str = "") -> dict[str, Any]:
        return reactive_floor_log_fields(self, prefix)

    def _reactive_floor_action_vec(self, plant_info: dict[str, Any]) -> np.ndarray:
        self._h120_floor_shaping_active = False
        self._h120_floor_shaping_reason = "off"
        self._h120_floor_shaping_effect = "none"
        self._h120_floor_shaping_axis = "none"
        self._h120_floor_shaping_delta_mean_kg = 0.0
        posture = posture_vector_deg(plant_info)
        action_name = str(self.reactive_floor_action)
        delay_s = float(self.reactive_floor_medium_delay_s)
        pareto_episode_mode = str(self._h120_pareto_mode_selector_selected_mode)
        if self.h120_pareto_mode_selector_enabled:
            if not self._h120_pareto_mode_selector_episode_active:
                self._h120_pareto_mode_selector_episode_active = True
                self._h120_pareto_mode_selector_episode_mode = pareto_episode_mode
            mode = str(self._h120_pareto_mode_selector_episode_mode)
            pareto_episode_mode = mode
            self._h120_pareto_mode_selector_selected_mode = mode
            if mode == "v14_early_stop":
                self._h120_pareto_mode_selector_selected_effect = "economy"
            elif mode == "v15_medium":
                self._h120_pareto_mode_selector_selected_effect = "safety"
            else:
                self._h120_pareto_mode_selector_selected_effect = "balanced"
            if mode == "v14_early_stop":
                action_name = "active_small"
                delay_s = 0.0
            elif mode == "v15_medium":
                action_name = "active_medium"
                delay_s = 0.0
            else:
                action_name = "active_small"
                delay_s = 1200.0
        if self.h120_oracle_probe_enabled:
            delay_s = float(self._h120_oracle_probe_effective_medium_delay_s)
        if self.h120_risk_scheduler_enabled:
            delay_s = float(self._h120_scheduler_effective_medium_delay_s)
        floor_shaping_mode = str(self.h120_floor_shaping_mode)
        far_supported = (
            self.far_horizon_enabled
            and float(self._far_horizon_far_max) >= float(self.h120_floor_shaping_far_high_norm)
        )
        response = float(self._reactive_floor_current_response_deg)
        recovery_slow = (
            math.isfinite(response)
            and response <= float(self.h120_floor_shaping_slow_response_eps_deg)
        )
        if (
            floor_shaping_mode in ("bounded_medium", "axis_shaped_bounded_medium")
            and self._reactive_floor_latched
            and action_name == "active_small"
            and far_supported
            and recovery_slow
        ):
            delay_s = min(
                delay_s if delay_s > 0.0 else float(self.reactive_floor_medium_delay_s),
                float(self.h120_floor_shaping_medium_delay_s),
            )
        if (
            delay_s > 0.0
            and action_name == "active_small"
            and self._reactive_floor_latched
            and self._reactive_floor_elapsed_s >= delay_s
        ):
            action_name = "active_medium"
            if floor_shaping_mode in ("bounded_medium", "axis_shaped_bounded_medium"):
                self._h120_floor_shaping_active = True
                self._h120_floor_shaping_effect = "bounded_medium"
                self._h120_floor_shaping_reason = "recovery_slow_far_supported"
        self._reactive_floor_resolved_action = action_name
        self._reactive_floor_medium_delay_active = (
            action_name != self.reactive_floor_action
            and not (
                self.h120_pareto_mode_selector_enabled
                and pareto_episode_mode == "v15_medium"
            )
        )
        avec = action_vec(action_name, posture, self.cfg)
        if (
            floor_shaping_mode in ("axis_shaped", "axis_shaped_bounded_medium")
            and self._reactive_floor_latched
            and far_supported
            and float(np.linalg.norm(avec)) > 1e-12
        ):
            axis_vec = np.zeros(2, dtype=float)
            if abs(float(posture[0])) >= abs(float(posture[1])):
                axis_vec[0] = avec[0]
                self._h120_floor_shaping_axis = "pitch"
            else:
                axis_vec[1] = avec[1]
                self._h120_floor_shaping_axis = "roll"
            if float(np.linalg.norm(axis_vec)) > 1e-12:
                avec = axis_vec
                self._h120_floor_shaping_active = True
                self._h120_floor_shaping_effect = (
                    "axis_shaped_bounded_medium"
                    if self._h120_floor_shaping_effect == "bounded_medium"
                    else "axis_shaped"
                )
                self._h120_floor_shaping_reason = (
                    self._h120_floor_shaping_reason
                    if self._h120_floor_shaping_reason != "off"
                    else "axis_shaped_far_supported"
                )
        if self._h120_floor_shaping_active:
            self._h120_floor_shaping_count += 1
            self._h120_floor_shaping_delta_mean_kg = float(
                np.mean(np.abs(self._primary_mass_delta_kg(avec)))
            )
        return avec

    def _reactive_floor_needed(
        self,
        plant_info: dict[str, Any],
        blocks: list[dict[str, Any]],
        current_time: float,
    ) -> tuple[bool, np.ndarray, str]:
        self._reactive_floor_active = False
        self._reactive_floor_veto_active = False
        self._reactive_floor_reason = "disabled"
        self._reactive_floor_veto_reason = "disabled"
        self._reactive_floor_delta_mean_kg = 0.0
        self._reactive_floor_trigger_class = "none"
        self._reactive_floor_elapsed_s = 0.0
        self._reactive_floor_resolved_action = self.reactive_floor_action
        self._reactive_floor_medium_delay_active = False
        self._reactive_floor_theta_only_gate_pass = False
        self._reactive_floor_theta_only_gate_reason = "disabled"
        self._reactive_floor_theta_only_elapsed_s = 0.0
        if not self.reactive_floor_predictive_veto_enabled:
            return False, np.zeros(2, dtype=float), "disabled"
        posture_metrics = reactive_floor_posture_metrics(
            plant_info,
            high_posture_metric=self.high_posture_metric,
            high_posture_enter=self.high_posture_enter,
            high_posture_exit=self.high_posture_exit,
        )
        max_axis = float(posture_metrics.max_axis_deg)
        theta_total = float(posture_metrics.theta_total_deg)
        theta_only_now = bool(posture_metrics.theta_only_now)
        trigger_class = str(posture_metrics.trigger_class)
        metric_value = float(posture_metrics.metric_value_deg)
        self._reactive_floor_max_axis_deg = float(max_axis)
        self._reactive_floor_theta_total_deg = float(theta_total)
        self._reactive_floor_posture_metric_deg = float(metric_value)
        enter = bool(posture_metrics.enter)
        exit_ready = bool(posture_metrics.exit_ready)
        if enter:
            if self._reactive_floor_enter_start_s is None:
                self._reactive_floor_enter_start_s = float(current_time)
            elapsed = float(current_time) - float(self._reactive_floor_enter_start_s)
            if elapsed >= float(self.high_posture_min_duration_s):
                self._reactive_floor_latched = True
        elif self._reactive_floor_latched and exit_ready:
            self._reactive_floor_latched = False
            self._reactive_floor_enter_start_s = None
        elif not self._reactive_floor_latched:
            self._reactive_floor_enter_start_s = None
        if not self._reactive_floor_latched:
            self._reactive_floor_reason = "posture_below_floor"
            return False, np.zeros(2, dtype=float), "posture_below_floor"
        if self._reactive_floor_enter_start_s is not None:
            self._reactive_floor_elapsed_s = max(
                0.0,
                float(current_time) - float(self._reactive_floor_enter_start_s),
            )

        target_err = self._current_primary_target_error_mean_kg(plant_info)
        target_age = max(0.0, float(current_time) - float(self._primary_target_last_reset_s))
        pump_rate = np.asarray(
            plant_info.get("pump_rate_cmd_m3_min", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        pump_total = float(np.sum(np.abs(pump_rate)))
        current_response = float(self._active_effectiveness_current_response_deg)
        if not math.isfinite(current_response):
            current_response = float("nan")
        future_status = self._classify_active_effectiveness_future_relief(blocks)
        raw_norms = [
            norm_term(
                np.asarray(
                    b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                    dtype=float,
                ),
                self.cfg,
            )
            for b in blocks[:3]
        ]
        max_risk = max(raw_norms) if raw_norms else 0.0
        target_stale = bool(
            self._primary_target_initialized
            and self._primary_target_reused
            and target_err <= 500.0
            and pump_total <= 0.5
        )
        pump_idle = pump_total <= 0.5
        short_risk_low = max_risk <= 0.75
        forecast_relief_clear = future_status == "clear"
        posture_not_worsening = bool(
            math.isfinite(current_response) and current_response >= -0.05
        )
        theta_only_latched = (
            self.high_posture_metric == "hybrid"
            and max_axis <= float(self.high_posture_enter)
            and theta_total > float(self.high_posture_enter) + 0.5
        )
        if trigger_class == "none":
            trigger_class = "theta_total_only" if theta_only_latched else "max_axis"
        self._reactive_floor_trigger_class = trigger_class
        if theta_only_latched:
            theta_elapsed = (
                0.0
                if self._reactive_floor_enter_start_s is None
                else max(0.0, float(current_time) - float(self._reactive_floor_enter_start_s))
            )
        else:
            theta_elapsed = 0.0
        self._reactive_floor_theta_only_elapsed_s = float(theta_elapsed)
        self._reactive_floor_target_stale = target_stale
        self._reactive_floor_target_err_mean_kg = float(target_err)
        self._reactive_floor_target_age_s = float(target_age)
        self._reactive_floor_pump_idle = pump_idle
        self._reactive_floor_current_response_deg = float(current_response)
        self._reactive_floor_forecast_relief_clear = bool(forecast_relief_clear)
        self._reactive_floor_short_risk_low = bool(short_risk_low)

        theta_gate = reactive_floor_theta_gate_decision(
            mode=self.reactive_floor_predictive_veto_mode,
            theta_only_latched=theta_only_latched,
            theta_total_deg=theta_total,
            theta_elapsed_s=theta_elapsed,
            theta_only_enter_deg=self.reactive_floor_theta_only_enter,
            theta_only_min_duration_s=self.reactive_floor_theta_only_min_duration_s,
            posture_not_worsening=posture_not_worsening,
            target_stale=target_stale,
            pump_idle=pump_idle,
            forecast_relief_clear=forecast_relief_clear,
        )
        self._reactive_floor_theta_only_gate_pass = bool(theta_gate.pass_gate)
        self._reactive_floor_theta_only_gate_reason = str(theta_gate.reason)

        if not theta_gate.pass_gate:
            self._reactive_floor_reason = "theta_only_watch"
            return False, np.zeros(2, dtype=float), "theta_only_watch"

        veto_decision = reactive_floor_veto_decision(
            mode=self.reactive_floor_predictive_veto_mode,
            theta_only_latched=theta_only_latched,
            forecast_relief_clear=forecast_relief_clear,
            posture_not_worsening=posture_not_worsening,
            target_stale=target_stale,
            short_risk_low=short_risk_low,
        )
        if veto_decision.veto:
            self._reactive_floor_veto_active = True
            self._reactive_floor_veto_reason = str(veto_decision.reason)
            self._reactive_floor_veto_count += 1
            self._reactive_floor_reason = "vetoed"
            return False, np.zeros(2, dtype=float), "vetoed"

        self._h120_oracle_probe_prepare_floor_delay(plant_info)
        avec = self._reactive_floor_action_vec(plant_info)
        if float(np.linalg.norm(avec)) <= 1e-12:
            self._reactive_floor_reason = "zero_floor_action"
            return False, avec, "zero_floor_action"
        self._reactive_floor_delta_mean_kg = float(
            np.mean(np.abs(self._primary_mass_delta_kg(avec)))
        )
        self._reactive_floor_active = True
        self._reactive_floor_reason = "reactive_floor_refresh"
        self._reactive_floor_count += 1
        self._reactive_floor_last_refresh_s = float(current_time)
        self._reactive_floor_post_exit_episode_active = True
        self._reactive_floor_post_exit_prev_metric_deg = float(metric_value)
        return True, avec, "reactive_floor_refresh"

    def _classify_active_effectiveness_future_relief(
        self,
        blocks: list[dict[str, Any]],
    ) -> str:
        if len(blocks) < 2:
            return "unknown"
        vec0 = np.asarray(
            blocks[0].get("pressure_vec_raw", blocks[0].get("pressure_vec", np.zeros(2))),
            dtype=float,
        )
        vec1 = np.asarray(
            blocks[1].get("pressure_vec_raw", blocks[1].get("pressure_vec", np.zeros(2))),
            dtype=float,
        )
        norm0 = norm_term(vec0, self.cfg)
        norm1 = norm_term(vec1, self.cfg)
        if norm0 - norm1 >= _FORECAST_RELIEF_MARGIN_NORM:
            return "clear"
        if float(np.dot(vec0, vec1)) < 0.0:
            return "clear"
        return "absent"

    def _observe_active_effectiveness_bucket(
        self,
        previous_action: str,
        new_action: str,
        plant_info: dict[str, Any],
        blocks: list[dict[str, Any]],
        current_time: float,
    ) -> None:
        self._active_effectiveness_refresh_active = False
        self._active_effectiveness_refresh_reason = "not_evaluated"
        active_now = self._active_action_like(new_action)
        active_prev = self._active_action_like(previous_action)
        if active_now:
            self._active_effectiveness_repeated_active_buckets = (
                self._active_effectiveness_repeated_active_buckets + 1
                if active_prev
                else 1
            )
        else:
            self._active_effectiveness_repeated_active_buckets = 0
            self._active_effectiveness_budget_used_m3 = 0.0

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        attitude_abs = max(pitch_abs, roll_abs)
        prev = float(self._active_effectiveness_prev_attitude_abs_deg)
        self._active_effectiveness_pitch_abs_deg = pitch_abs
        self._active_effectiveness_roll_abs_deg = roll_abs
        self._active_effectiveness_attitude_abs_deg = attitude_abs
        self._active_effectiveness_current_response_deg = (
            prev - attitude_abs if math.isfinite(prev) else float("nan")
        )
        self._active_effectiveness_prev_attitude_abs_deg = attitude_abs

        raw_norms = [
            norm_term(
                np.asarray(
                    b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                    dtype=float,
                ),
                self.cfg,
            )
            for b in blocks[:3]
        ]
        pressure_max = max(raw_norms) if raw_norms else 0.0
        self._active_effectiveness_pressure_max_norm = float(pressure_max)
        self._active_effectiveness_future_relief_status = (
            self._classify_active_effectiveness_future_relief(blocks)
        )
        self._active_effectiveness_lowrisk_safe = bool(
            pitch_abs < 2.5
            and roll_abs < 2.5
            and pressure_max < float(self.active_effectiveness_refresh_lowrisk_pressure_norm)
        )
        high = (
            pitch_abs >= float(self.active_effectiveness_refresh_pitch_deg)
            or roll_abs >= float(self.active_effectiveness_refresh_roll_deg)
        )
        if not high:
            self._active_effectiveness_budget_used_m3 = 0.0

        self._active_effectiveness_target_err_mean_kg = (
            self._current_primary_target_error_mean_kg(plant_info)
        )
        pump_rate = np.asarray(
            plant_info.get("pump_rate_cmd_m3_min", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        self._active_effectiveness_pump_rate_m3_min = float(np.sum(np.abs(pump_rate)))
        self._active_effectiveness_target_age_s = max(
            0.0,
            float(current_time) - float(self._primary_target_last_reset_s),
        )
        self._active_effectiveness_target_ready = bool(
            self._active_effectiveness_target_err_mean_kg
            <= float(self.active_effectiveness_refresh_target_err_kg)
        )
        self._active_effectiveness_pump_idle = bool(
            self._active_effectiveness_pump_rate_m3_min
            <= float(self.active_effectiveness_refresh_pump_rate_m3_min)
        )
        self._active_effectiveness_fallback_dominated = bool(
            int(plant_info.get("pump_fullspeed_any", 0)) > 0
        )
        self._active_effectiveness_proposal_delta_mean_kg = 0.0
        self._active_effectiveness_proposal_delta_m3 = 0.0

    def _active_effectiveness_refresh_needed(
        self,
        plant_info: dict[str, Any],
        first_avec: np.ndarray,
        new_action: str,
    ) -> tuple[bool, str]:
        self._active_effectiveness_refresh_active = False
        self._active_effectiveness_refresh_reason = "disabled"
        if not self.active_effectiveness_refresh_enabled:
            return False, "disabled"
        if not self._active_action_like(new_action):
            self._active_effectiveness_refresh_reason = "current_action_not_active"
            return False, "current_action_not_active"
        if (
            self._active_effectiveness_repeated_active_buckets
            < self.active_effectiveness_refresh_min_active_buckets
        ):
            self._active_effectiveness_refresh_reason = "repeated_active_below_threshold"
            return False, "repeated_active_below_threshold"
        if self._active_effectiveness_lowrisk_safe:
            self._active_effectiveness_refresh_reason = "lowrisk_safe_posture"
            return False, "lowrisk_safe_posture"
        if (
            self._active_effectiveness_pitch_abs_deg
            < float(self.active_effectiveness_refresh_pitch_deg)
            and self._active_effectiveness_roll_abs_deg
            < float(self.active_effectiveness_refresh_roll_deg)
        ):
            self._active_effectiveness_refresh_reason = "posture_inside_attention_band"
            return False, "posture_inside_attention_band"
        if self._active_effectiveness_fallback_dominated:
            self._active_effectiveness_refresh_reason = "fallback_dominated"
            return False, "fallback_dominated"
        if (
            self._active_effectiveness_target_age_s
            < float(self.active_effectiveness_refresh_min_target_age_s)
        ):
            self._active_effectiveness_refresh_reason = "target_age_below_threshold"
            return False, "target_age_below_threshold"
        if (
            self._active_effectiveness_pressure_max_norm
            < float(self.active_effectiveness_refresh_min_pressure_norm)
        ):
            self._active_effectiveness_refresh_reason = "pressure_below_threshold"
            return False, "pressure_below_threshold"
        current_response = float(self._active_effectiveness_current_response_deg)
        if not math.isfinite(current_response):
            self._active_effectiveness_refresh_reason = "missing_response_history"
            return False, "missing_response_history"
        required_worsening = float(
            self.active_effectiveness_refresh_required_worsening_deg
        )
        if required_worsening > 0.0 and current_response > -required_worsening:
            self._active_effectiveness_refresh_reason = "posture_not_worsening_enough"
            return False, "posture_not_worsening_enough"
        if current_response > float(self.active_effectiveness_refresh_response_eps_deg):
            self._active_effectiveness_refresh_reason = "current_posture_improving"
            return False, "current_posture_improving"
        if self._active_effectiveness_future_relief_status == "clear":
            self._active_effectiveness_refresh_reason = "future_relief_clear"
            return False, "future_relief_clear"
        if not (
            self._active_effectiveness_target_ready
            or self._active_effectiveness_pump_idle
        ):
            self._active_effectiveness_refresh_reason = "target_and_pump_still_working"
            return False, "target_and_pump_still_working"
        gusty_hold, gusty_reason = self._gusty_oscillation_refresh_gate(plant_info)
        if gusty_hold:
            self._active_effectiveness_refresh_reason = gusty_reason
            return False, gusty_reason

        avec = np.asarray(first_avec, dtype=float).reshape(2)
        if float(np.linalg.norm(avec)) <= 1e-12:
            self._active_effectiveness_refresh_reason = "zero_action"
            return False, "zero_action"
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        proposal_metrics = primary_target_proposal_metrics(
            masses,
            self._primary_mass_delta_kg(avec),
            self._primary_target_kg,
            tank_capacity_kg=self.cfg.tank_capacity_kg,
        )
        delta_mean = float(proposal_metrics.target_delta_mean_kg)
        delta_m3 = float(proposal_metrics.target_delta_m3)
        self._active_effectiveness_proposal_delta_mean_kg = delta_mean
        self._active_effectiveness_proposal_delta_m3 = delta_m3
        if delta_mean < float(self.active_effectiveness_refresh_min_delta_kg):
            self._active_effectiveness_refresh_reason = "proposal_delta_below_threshold"
            return False, "proposal_delta_below_threshold"
        if (
            self.active_effectiveness_refresh_episode_budget_m3 > 0.0
            and self._active_effectiveness_budget_used_m3 + delta_m3
            > float(self.active_effectiveness_refresh_episode_budget_m3)
        ):
            self._active_effectiveness_refresh_reason = "episode_budget_exhausted"
            return False, "episode_budget_exhausted"

        self._active_effectiveness_budget_used_m3 += delta_m3
        self._active_effectiveness_refresh_active = True
        self._active_effectiveness_refresh_reason = "active_ineffective_refresh"
        return True, "active_ineffective_refresh"

    def _sustained_active_recompute_needed(self, new_action: str) -> tuple[bool, str]:
        self._sustained_active_recompute_active = False
        self._sustained_active_recompute_reason = "disabled"
        if not self.sustained_active_recompute_enabled:
            return False, "disabled"
        if not self._active_action_like(new_action):
            self._sustained_active_recompute_reason = "current_action_not_active"
            return False, "current_action_not_active"
        if (
            self._active_effectiveness_repeated_active_buckets
            < self.sustained_active_recompute_min_active_buckets
        ):
            self._sustained_active_recompute_reason = "repeated_active_below_threshold"
            return False, "repeated_active_below_threshold"
        if (
            self._active_effectiveness_pitch_abs_deg
            < float(self.sustained_active_recompute_pitch_deg)
            and self._active_effectiveness_roll_abs_deg
            < float(self.sustained_active_recompute_roll_deg)
        ):
            self._sustained_active_recompute_reason = "posture_inside_attention_band"
            return False, "posture_inside_attention_band"
        if self._active_effectiveness_fallback_dominated:
            self._sustained_active_recompute_reason = "fallback_dominated"
            return False, "fallback_dominated"
        if (
            self._active_effectiveness_target_age_s
            < float(self.sustained_active_recompute_min_target_age_s)
        ):
            self._sustained_active_recompute_reason = "target_age_below_threshold"
            return False, "target_age_below_threshold"
        if (
            self._active_effectiveness_target_err_mean_kg
            > float(self.sustained_active_recompute_target_err_kg)
        ):
            self._sustained_active_recompute_reason = "target_still_tracking"
            return False, "target_still_tracking"
        if (
            self._active_effectiveness_pump_rate_m3_min
            > float(self.sustained_active_recompute_pump_rate_m3_min)
        ):
            self._sustained_active_recompute_reason = "pump_still_active"
            return False, "pump_still_active"
        self._sustained_active_recompute_active = True
        self._sustained_active_recompute_reason = "sustained_active_recompute"
        return True, "sustained_active_recompute"

    def _stale_active_target_log_fields(self, prefix: str = "") -> dict[str, Any]:
        if self.stale_active_target_refresh_mode == "off":
            return {}
        return {
            f"{prefix}stale_active_target_refresh_mode": (
                self.stale_active_target_refresh_mode
            ),
            f"{prefix}stale_active_target_refresh_active": int(
                self._stale_active_target_refresh_active
            ),
            f"{prefix}stale_active_target_refresh_reason": (
                self._stale_active_target_refresh_reason
            ),
            f"{prefix}stale_active_target_refresh_candidate_delta_mean_kg": float(
                self._stale_active_target_refresh_candidate_delta_mean_kg
            ),
            f"{prefix}stale_active_target_refresh_delta_after_kg": float(
                self._stale_active_target_refresh_delta_after_kg
            ),
            f"{prefix}stale_active_target_refresh_target_err_before_kg": float(
                self._stale_active_target_refresh_target_err_before_kg
            ),
            f"{prefix}stale_active_target_refresh_target_age_before_s": float(
                self._stale_active_target_refresh_target_age_before_s
            ),
            f"{prefix}stale_active_target_refresh_pump_rate_m3_min": float(
                self._stale_active_target_refresh_pump_rate_m3_min
            ),
            f"{prefix}stale_active_target_refresh_posture_abs_deg": float(
                self._stale_active_target_refresh_posture_abs_deg
            ),
            f"{prefix}stale_active_target_refresh_response_deg": float(
                self._stale_active_target_refresh_response_deg
            ),
            f"{prefix}stale_active_target_refresh_count": int(
                self._stale_active_target_refresh_count
            ),
            f"{prefix}stale_active_target_refresh_rejected_count": int(
                self._stale_active_target_refresh_rejected_count
            ),
            f"{prefix}stale_active_target_refresh_no_effect_count": int(
                self._stale_active_target_refresh_no_effect_count
            ),
        }

    def _stale_active_target_refresh_needed(
        self,
        plant_info: dict[str, Any],
        first_avec: np.ndarray,
        new_action: str,
        current_time: float,
    ) -> tuple[bool, str]:
        self._stale_active_target_refresh_active = False
        self._stale_active_target_refresh_reason = "disabled"
        self._stale_active_target_refresh_candidate_delta_mean_kg = 0.0
        self._stale_active_target_refresh_delta_after_kg = 0.0
        self._stale_active_target_refresh_target_err_before_kg = float(
            self._active_effectiveness_target_err_mean_kg
        )
        self._stale_active_target_refresh_target_age_before_s = float(
            self._active_effectiveness_target_age_s
        )
        self._stale_active_target_refresh_pump_rate_m3_min = float(
            self._active_effectiveness_pump_rate_m3_min
        )
        self._stale_active_target_refresh_posture_abs_deg = float(
            self._active_effectiveness_attitude_abs_deg
        )
        self._stale_active_target_refresh_response_deg = float(
            self._active_effectiveness_current_response_deg
        )
        if self.stale_active_target_refresh_mode == "off":
            return False, "disabled"
        if not self._active_action_like(new_action):
            self._stale_active_target_refresh_reason = "current_action_not_active"
            return False, "current_action_not_active"
        if not self._primary_target_initialized:
            self._stale_active_target_refresh_reason = "target_not_initialized"
            return False, "target_not_initialized"
        if not self._primary_target_reused:
            self._stale_active_target_refresh_reason = "target_not_reused"
            return False, "target_not_reused"
        if (
            self._active_effectiveness_repeated_active_buckets
            < self.stale_active_target_refresh_min_active_buckets
        ):
            self._stale_active_target_refresh_reason = (
                "repeated_active_below_threshold"
            )
            return False, "repeated_active_below_threshold"
        if (
            self._active_effectiveness_pitch_abs_deg
            < float(self.stale_active_target_refresh_pitch_deg)
            and self._active_effectiveness_roll_abs_deg
            < float(self.stale_active_target_refresh_roll_deg)
        ):
            self._stale_active_target_refresh_reason = "posture_inside_attention_band"
            return False, "posture_inside_attention_band"
        if (
            self._active_effectiveness_target_age_s
            < float(self.stale_active_target_refresh_min_target_age_s)
        ):
            self._stale_active_target_refresh_reason = "target_age_below_threshold"
            return False, "target_age_below_threshold"
        if (
            self._active_effectiveness_target_err_mean_kg
            > float(self.stale_active_target_refresh_target_err_kg)
        ):
            self._stale_active_target_refresh_reason = "target_not_reached"
            return False, "target_not_reached"
        if (
            self._active_effectiveness_pump_rate_m3_min
            > float(self.stale_active_target_refresh_pump_rate_m3_min)
        ):
            self._stale_active_target_refresh_reason = "pump_still_active"
            return False, "pump_still_active"
        if self.stale_active_target_refresh_mode == "refresh_only_guarded_v1":
            elapsed = max(
                0.0,
                float(current_time) - float(self._stale_active_target_refresh_last_s),
            )
            if elapsed < float(self.stale_active_target_refresh_cooldown_s):
                self._stale_active_target_refresh_reason = "cooldown_active"
                return False, "cooldown_active"
            response = float(self._active_effectiveness_current_response_deg)
            if not math.isfinite(response):
                self._stale_active_target_refresh_reason = "missing_response_history"
                return False, "missing_response_history"
            if response > float(self.stale_active_target_refresh_no_recovery_eps_deg):
                self._stale_active_target_refresh_reason = "posture_recovering"
                return False, "posture_recovering"
        gusty_hold, gusty_reason = self._gusty_oscillation_refresh_gate(plant_info)
        if gusty_hold:
            self._stale_active_target_refresh_reason = gusty_reason
            return False, gusty_reason
        avec = np.asarray(first_avec, dtype=float).reshape(2)
        if float(np.linalg.norm(avec)) <= 1e-12:
            self._stale_active_target_refresh_reason = "zero_action"
            return False, "zero_action"
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        proposal_metrics = primary_target_proposal_metrics(
            masses,
            self._primary_mass_delta_kg(avec),
            self._primary_target_kg,
            tank_capacity_kg=self.cfg.tank_capacity_kg,
        )
        candidate_delta = float(proposal_metrics.target_delta_mean_kg)
        after_delta = float(proposal_metrics.after_delta_mean_kg)
        self._stale_active_target_refresh_candidate_delta_mean_kg = candidate_delta
        self._stale_active_target_refresh_delta_after_kg = after_delta
        if candidate_delta < float(self.stale_active_target_refresh_min_delta_kg):
            self._stale_active_target_refresh_no_effect_count += 1
            self._stale_active_target_refresh_reason = "refresh_no_effect"
            return False, "refresh_no_effect"
        self._stale_active_target_refresh_active = True
        self._stale_active_target_refresh_reason = "stale_active_target_refresh"
        self._stale_active_target_refresh_last_s = float(current_time)
        self._stale_active_target_refresh_count += 1
        return True, "stale_active_target_refresh"

    def _stall_refresh_action_vec(
        self,
        first_avec: np.ndarray,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> tuple[np.ndarray, bool, str]:
        self._primary_stall_target_err_mean_kg = self._current_primary_target_error_mean_kg(
            plant_info
        )
        pump_rate = np.asarray(
            plant_info.get("pump_rate_cmd_m3_min", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        self._primary_stall_pump_rate_m3_min = float(np.sum(np.abs(pump_rate)))
        self._primary_stall_axis_bias_active = False
        if not self.primary_stall_refresh_enabled:
            return first_avec, False, "disabled"
        if float(np.linalg.norm(first_avec)) <= 1e-12:
            return first_avec, False, "zero_action"
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        pitch = float(posture[0])
        roll = float(posture[1])
        if (
            abs(pitch) < float(self.primary_stall_pitch_deg)
            and abs(roll) < float(self.primary_stall_roll_deg)
        ):
            return first_avec, False, "posture_inside_stall_band"
        if self._primary_stall_target_err_mean_kg > float(self.primary_stall_target_err_kg):
            return first_avec, False, "target_not_reached"
        if self._primary_stall_pump_rate_m3_min > float(self.primary_stall_pump_rate_m3_min):
            return first_avec, False, "pump_still_active"
        age_s = max(0.0, float(current_time) - float(self._primary_target_last_reset_s))
        if age_s < float(self.primary_stall_min_age_s):
            return first_avec, False, "target_too_young"

        avec = np.asarray(first_avec, dtype=float).reshape(2).copy()
        if (
            self.primary_stall_pitch_axis_bias
            and abs(pitch) >= float(self.primary_stall_pitch_deg)
            and abs(roll) <= float(self.primary_stall_pitch_axis_roll_max_deg)
        ):
            pitch_step = max(
                abs(float(avec[0])),
                float(self.cfg.active_small_ratio) * float(self.cfg.deadband_pitch_deg),
            )
            # Planner-frame positive pitch action produces negative physical pitch
            # response, so use the measured pitch sign for compensation.
            avec[0] = math.copysign(pitch_step, pitch)
            avec[1] = 0.0
            self._primary_stall_axis_bias_active = True
            return avec, True, "pitch_axis_stall_refresh"
        return avec, True, "stalled_target_refresh"

    def _remember_primary_target_for_pause(
        self,
        previous_action: str,
        previous_avec: np.ndarray,
    ) -> None:
        if self.primary_hold_target_mode not in ("pause", "forecast_pause"):
            return
        if str(previous_action) == "hold" or not self._primary_target_initialized:
            return
        if float(np.mean(np.abs(self._primary_target_kg - self._primary_anchor_masses_kg))) <= 1.0:
            return
        remember_paused_primary_target(
            self,
            action=str(previous_action),
            avec=previous_avec,
        )

    def _resume_paused_primary_target(
        self,
        plant_info: dict[str, Any],
        first_avec: np.ndarray,
        current_time: float,
    ) -> bool:
        self._forecast_pause_resume_guard_active = False
        self._forecast_pause_resume_guard_reason = "not_evaluated"
        self._forecast_pause_resume_target_err_kg = 0.0
        if (
            self.primary_hold_target_mode not in ("pause", "forecast_pause")
            or not self._paused_primary_valid
        ):
            return False
        first_avec = np.asarray(first_avec, dtype=float).reshape(2)
        if float(np.linalg.norm(first_avec)) <= 1e-12:
            return False
        if float(np.linalg.norm(self._paused_primary_avec)) <= 1e-12:
            return False
        if float(np.dot(first_avec, self._paused_primary_avec)) < 0.0:
            self._paused_primary_valid = False
            self._forecast_pause_resume_guard_reason = "direction_mismatch"
            return False

        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        resume_target_err_kg = float(
            np.mean(np.abs(self._paused_primary_target_kg - masses))
        )
        self._forecast_pause_resume_target_err_kg = resume_target_err_kg
        if (
            self.primary_hold_target_mode == "forecast_pause"
            and self.forecast_pause_resume_max_target_err_kg > 0.0
            and resume_target_err_kg
            > float(self.forecast_pause_resume_max_target_err_kg)
        ):
            self._paused_primary_valid = False
            self._forecast_pause_resume_guard_active = True
            self._forecast_pause_resume_guard_reason = "target_error_debt"
            self._primary_refresh_owner_pending = "forecast_pause_resume_debt_pi_release"
            self._release_primary_target_to_pi(plant_info, float(current_time))
            return True

        self._forecast_pause_resume_guard_reason = "resume_allowed"
        apply_primary_target_transition(
            self,
            primary_target_from_paused(
                masses,
                self._paused_primary_target_kg,
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
            ),
        )
        return True

    def _hold_primary_target_at_current(
        self,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> None:
        consume_primary_refresh_owner(self)
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        apply_primary_target_transition(
            self,
            primary_target_at_current(
                masses,
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
                initialized=True,
                clip_target=True,
            ),
        )

    def synchronize_forecast_deadband_hold(
        self,
        plant_info: dict[str, Any] | None,
        current_time: float,
        *,
        blend: float = 1.0,
        start_new: bool = True,
    ) -> np.ndarray:
        """Commit a forecast-approved reduction of the pending primary target."""
        info = self._plant_info_from(plant_info)
        masses = plant_primary_masses(info, self._primary_anchor_masses_kg)
        if start_new or not hasattr(self, "_forecast_deadband_held_target_kg"):
            blend = float(np.clip(blend, 0.0, 1.0))
            pending = np.asarray(self._primary_target_kg, dtype=float).reshape(-1)[:3]
            held_target = np.clip(
                pending + blend * (masses - pending),
                0.0,
                float(self.cfg.tank_capacity_kg),
            )
            self._forecast_deadband_held_target_kg = held_target.copy()
        held_target = np.asarray(
            self._forecast_deadband_held_target_kg,
            dtype=float,
        ).reshape(-1)[:3]
        self._paused_primary_valid = False
        self._primary_refresh_owner_pending = "forecast_safe_deadband_hold_current"
        self._target_action = "hold"
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        apply_primary_target_transition(
            self,
            primary_target_from_paused(
                masses,
                held_target,
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
            ),
        )
        return self._primary_target_kg.copy()

    def exit_forecast_deadband_hold(
        self,
        plant_info: dict[str, Any] | None,
        current_time: float,
    ) -> None:
        """Discard the held target and force a fresh candidate evaluation."""
        info = self._plant_info_from(plant_info)
        self._paused_primary_valid = False
        if hasattr(self, "_forecast_deadband_held_target_kg"):
            del self._forecast_deadband_held_target_kg
        self._primary_refresh_owner_pending = "forecast_safe_deadband_exit_replan"
        self._release_primary_target_to_pi(info, float(current_time))
        self._target_action = "hold"
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        self._last_bucket = -1

    def _release_primary_target_to_pi(self, plant_info: dict[str, Any], current_time: float) -> None:
        consume_primary_refresh_owner(self)
        masses = plant_primary_masses(plant_info, self._primary_anchor_masses_kg)
        apply_primary_target_transition(
            self,
            primary_target_at_current(
                masses,
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
                initialized=False,
                clip_target=False,
            ),
        )

    def _hold_has_future_relief_or_reversal(self, state) -> tuple[bool, str]:
        if not bool(self._forecast_has_future):
            return False, "no_future_preview"
        if state is None:
            return False, "missing_state"
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size <= 4:
            return False, "invalid_state"
        posture_vec = np.array(
            [float(np.degrees(state_arr[4])), float(np.degrees(state_arr[3]))],
            dtype=float,
        )
        posture_mag = float(np.linalg.norm(posture_vec))
        if posture_mag <= 1e-9:
            return False, "no_posture_direction"
        posture_dir = posture_vec / posture_mag
        norms = [float(x) for x in self._raw_pressure_block_norms[:3]]
        while len(norms) < 3:
            norms.append(0.0)
        vec0 = np.asarray(self._raw_pressure_block_vecs[0], dtype=float)
        vec2 = np.asarray(self._raw_pressure_block_vecs[2], dtype=float)
        proj0 = float(np.dot(vec0, posture_dir))
        proj2 = float(np.dot(vec2, posture_dir))
        if proj0 - proj2 >= _FORECAST_RELIEF_MARGIN_NORM:
            return True, "future_posture_relief"
        if proj0 > 0.0 and proj2 < 0.0:
            return True, "future_posture_reversal"
        return False, "no_posture_relief_or_reversal"

    def _should_release_hold_to_pi(self, state) -> tuple[bool, str]:
        self._hold_future_posture_evidence_active = False
        self._hold_future_posture_evidence_reason = "not_evaluated"
        if state is None:
            return False, "missing_state"
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size <= 4:
            return False, "invalid_state"
        pitch_abs = abs(float(np.degrees(state_arr[4])))
        roll_abs = abs(float(np.degrees(state_arr[3])))
        if (
            pitch_abs < _HOLD_RELEASE_ENTER_PITCH_DEG
            and roll_abs < _HOLD_RELEASE_ENTER_ROLL_DEG
        ):
            return False, "inside_comfort_band"
        has_relief, relief_reason = self._hold_has_future_relief_or_reversal(state)
        self._hold_future_posture_evidence_active = bool(has_relief)
        self._hold_future_posture_evidence_reason = str(relief_reason)
        if self.primary_hold_target_mode != "forecast_pause":
            return False, f"mode_disabled:{relief_reason}"
        if (
            str(self.economy_pump_budget_allocator_mode)
            in (
                "direction_reversal_avoidance_auto",
                "direction_reversal_layered_auto",
                "direction_reversal_layered_v5_auto",
                "direction_reversal_layered_v6_auto",
                "direction_reversal_layered_v7_auto",
                "direction_reversal_layered_v8_auto",
            )
            and max(pitch_abs, roll_abs)
            >= float(self.economy_pump_budget_smart_posture_deg)
        ):
            return True, (
                "release_to_pi:direction_reversal_posture_above_safe_band:"
                f"{relief_reason}"
            )
        if has_relief:
            return False, f"preview_pause:{relief_reason}"
        return True, f"release_to_pi:{relief_reason}"

    def _active_posture_refresh_correction(
        self,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> tuple[bool, np.ndarray]:
        """Refresh an active target when current posture drifts between buckets."""
        prev_active = bool(self._active_posture_refresh_active)
        self._active_posture_refresh_active = False
        self._active_posture_refresh_reason = "disabled"
        self._active_posture_refresh_pitch_abs_deg = 0.0
        self._active_posture_refresh_roll_abs_deg = 0.0
        self._active_posture_refresh_action_name = ""
        self._active_posture_refresh_axis_mode_active = str(
            self.active_posture_refresh_axis_mode
        )
        self._active_posture_refresh_axis_mask = "none"
        self._active_posture_refresh_pitch_worsening_deg_s = 0.0
        self._active_posture_refresh_roll_worsening_deg_s = 0.0
        if not self.active_posture_refresh_enabled:
            return False, np.zeros(2, dtype=float)
        allowed_actions = {"active_small", "active_medium"}
        if self.active_posture_refresh_include_hold:
            allowed_actions.add("hold")
        if str(self._target_action) not in allowed_actions:
            self._active_posture_refresh_reason = "current_action_not_active"
            return False, np.zeros(2, dtype=float)
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        self._active_posture_refresh_pitch_abs_deg = pitch_abs
        self._active_posture_refresh_roll_abs_deg = roll_abs
        prev_t = self._active_posture_refresh_last_posture_time_s
        if prev_t is not None:
            dt_s = max(float(current_time) - float(prev_t), 1e-6)
            pitch_worsening = (
                pitch_abs - float(self._active_posture_refresh_last_pitch_abs_deg)
            ) / dt_s
            roll_worsening = (
                roll_abs - float(self._active_posture_refresh_last_roll_abs_deg)
            ) / dt_s
        else:
            pitch_worsening = 0.0
            roll_worsening = 0.0
        self._active_posture_refresh_pitch_worsening_deg_s = float(pitch_worsening)
        self._active_posture_refresh_roll_worsening_deg_s = float(roll_worsening)
        self._active_posture_refresh_last_posture_time_s = float(current_time)
        self._active_posture_refresh_last_pitch_abs_deg = pitch_abs
        self._active_posture_refresh_last_roll_abs_deg = roll_abs
        threshold = (
            float(self.active_posture_refresh_exit_deg)
            if prev_active
            else float(self.active_posture_refresh_enter_deg)
        )
        if max(pitch_abs, roll_abs) < threshold:
            self._active_posture_refresh_reason = "posture_inside_active_band"
            return False, np.zeros(2, dtype=float)
        if self.active_posture_refresh_pressure_max_norm > 0.0:
            max_pressure = max(float(x) for x in self._raw_pressure_block_norms[:3])
            if max_pressure > float(self.active_posture_refresh_pressure_max_norm):
                self._active_posture_refresh_reason = "pressure_not_relieved"
                return False, np.zeros(2, dtype=float)
        target_age_s = max(
            0.0,
            float(current_time) - float(self._primary_target_last_reset_s),
        )
        if target_age_s < float(self.active_posture_refresh_min_target_age_s):
            self._active_posture_refresh_reason = "target_age_below_threshold"
            return False, np.zeros(2, dtype=float)
        posture_for_action = posture
        if self.active_posture_refresh_axis_mode == "axis_debt":
            eps = float(self.active_posture_refresh_worsening_eps_deg_s)
            debt_deg = float(self.active_posture_refresh_debt_deg)
            pitch_debt = pitch_abs >= threshold and (
                (eps <= 0.0 or pitch_worsening >= eps)
                or (debt_deg > 0.0 and pitch_abs >= debt_deg)
            )
            roll_debt = roll_abs >= threshold and (
                (eps <= 0.0 or roll_worsening >= eps)
                or (debt_deg > 0.0 and roll_abs >= debt_deg)
            )
            if not pitch_debt and not roll_debt:
                self._active_posture_refresh_reason = "axis_debt_gate_not_met"
                return False, np.zeros(2, dtype=float)
            mask = np.array([pitch_debt, roll_debt], dtype=bool)
            posture_for_action = np.where(mask, posture, 0.0)
            if pitch_debt and roll_debt:
                self._active_posture_refresh_axis_mask = "pitch_roll"
            elif pitch_debt:
                self._active_posture_refresh_axis_mask = "pitch"
            else:
                self._active_posture_refresh_axis_mask = "roll"
        avec = action_vec(
            self.active_posture_refresh_action_name,
            posture_for_action,
            self.cfg,
        )
        if norm_term(avec, self.cfg) <= 1e-9:
            self._active_posture_refresh_reason = "correction_vector_zero"
            return False, np.zeros(2, dtype=float)
        self._active_posture_refresh_active = True
        self._active_posture_refresh_reason = "active_posture_between_bucket"
        self._active_posture_refresh_action_name = (
            self.active_posture_refresh_action_name
        )
        return True, avec

    def _fallback_risk_active_release_needed(
        self,
        plant_info: dict[str, Any],
    ) -> tuple[bool, str]:
        self._fallback_risk_active_release_active = False
        self._fallback_risk_active_release_reason = "disabled"
        self._fallback_risk_active_release_pitch_abs_deg = 0.0
        self._fallback_risk_active_release_roll_abs_deg = 0.0
        self._fallback_risk_active_release_pitch_worsening_deg_s = 0.0
        self._fallback_risk_active_release_roll_worsening_deg_s = 0.0
        self._fallback_risk_active_release_pump_rate_m3_min = 0.0
        self._fallback_risk_active_release_backlog_kg = 0.0
        if not self.fallback_risk_active_release_enabled:
            return False, "disabled"
        if str(self._target_action) not in ("active_small", "active_medium"):
            self._fallback_risk_active_release_reason = "current_action_not_active"
            return False, "current_action_not_active"
        if not self._primary_target_initialized:
            self._fallback_risk_active_release_reason = "target_not_initialized"
            return False, "target_not_initialized"
        posture = np.asarray(
            (plant_info or {}).get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        pitch_deg = float(posture[0])
        roll_deg = float(posture[1])
        pitch_abs = abs(pitch_deg)
        roll_abs = abs(roll_deg)
        self._fallback_risk_active_release_pitch_abs_deg = pitch_abs
        self._fallback_risk_active_release_roll_abs_deg = roll_abs
        axis_deg = max(pitch_abs, roll_abs)
        if axis_deg < float(self.fallback_risk_active_release_enter_deg):
            self._fallback_risk_active_release_reason = "posture_inside_risk_band"
            return False, "posture_inside_risk_band"

        prev_pitch = (
            float(self.records[-1].get("pitch_deg", pitch_deg))
            if self.records
            else pitch_deg
        )
        prev_roll = (
            float(self.records[-1].get("roll_deg", roll_deg))
            if self.records
            else roll_deg
        )
        pitch_worsening = pitch_deg * (pitch_deg - prev_pitch)
        roll_worsening = roll_deg * (roll_deg - prev_roll)
        self._fallback_risk_active_release_pitch_worsening_deg_s = float(
            pitch_worsening
        )
        self._fallback_risk_active_release_roll_worsening_deg_s = float(
            roll_worsening
        )
        pump_rate = np.asarray(
            (plant_info or {}).get("pump_rate_cmd_m3_min", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        pump_rate_total = float(np.sum(np.abs(pump_rate)))
        backlog = float((plant_info or {}).get("pump_backlog_max_kg", 0.0))
        self._fallback_risk_active_release_pump_rate_m3_min = pump_rate_total
        self._fallback_risk_active_release_backlog_kg = backlog
        live_worsening = bool(
            pitch_worsening
            >= float(self.fallback_risk_active_release_worsening_eps_deg_s)
            or roll_worsening
            >= float(self.fallback_risk_active_release_worsening_eps_deg_s)
        )
        catchup_active = bool(
            pump_rate_total >= float(self.fallback_risk_active_release_pump_rate_m3_min)
            and backlog >= float(self.fallback_risk_active_release_backlog_kg)
        )
        if not live_worsening and not catchup_active:
            self._fallback_risk_active_release_reason = (
                "posture_high_but_not_worsening"
            )
            return False, "posture_high_but_not_worsening"
        self._fallback_risk_active_release_active = True
        self._fallback_risk_active_release_count += 1
        reason = "fallback_risk_active_release"
        if live_worsening and catchup_active:
            reason = "fallback_risk_worsening_and_catchup"
        elif catchup_active:
            reason = "fallback_risk_catchup"
        self._fallback_risk_active_release_reason = reason
        return True, reason

    def _maybe_update_primary_target_between_buckets(
        self,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> None:
        """Run default-off active-target refresh checks between 10-min replans.

        The planner still updates forecast actions every bucket, but current
        posture diagnostics may refresh active targets at a modest cadence.
        """
        if not self.prediction_primary_enabled:
            return
        direction_reversal_reuse_guard = bool(
            str(self.economy_pump_budget_allocator_mode)
            in (
                "direction_reversal_avoidance_auto",
                "direction_reversal_layered_auto",
                "direction_reversal_layered_v5_auto",
                "direction_reversal_layered_v6_auto",
                "direction_reversal_layered_v7_auto",
            )
            and self._primary_target_reused
        )
        if (
            str(self._target_action) == "hold"
            and not self.active_posture_refresh_include_hold
        ) or direction_reversal_reuse_guard:
            return
        active_refresh, active_avec = self._active_posture_refresh_correction(
            plant_info,
            float(current_time),
        )
        fallback_release, fallback_reason = (
            self._fallback_risk_active_release_needed(plant_info)
        )
        if fallback_release:
            self._primary_refresh_owner_pending = (
                f"fallback_risk_active_release:{fallback_reason}"
            )
            self._release_primary_target_to_pi(plant_info, float(current_time))
            self._active_posture_refresh_last_update_s = float(current_time)
            return
        if not active_refresh:
            return
        due = (
            self._active_posture_refresh_last_update_s is None
            or float(current_time)
            - float(self._active_posture_refresh_last_update_s)
            >= self.active_posture_refresh_update_interval_s
        )
        if not due:
            self._active_posture_refresh_reason = "update_interval_not_due"
            return
        self._economy_budget_observe(
            plant_info,
            float(current_time),
            hard_domain=False,
        )
        if self._economy_budget_allows_refresh(
            would_refresh=True,
            refresh_avec=active_avec,
            plant_info=plant_info,
        ):
            self._primary_refresh_owner_pending = "between_bucket_active_posture"
            self._update_primary_target(
                plant_info,
                active_avec,
                float(current_time),
            )
        else:
            self._primary_refresh_owner_pending = (
                "between_bucket_active_posture_mild_refresh"
                if self.economy_pump_budget_hold_refresh_fraction > 0.0
                else "between_bucket_active_posture_hold_current"
            )
            self._apply_economy_budget_blocked_refresh(
                plant_info,
                active_avec,
                float(current_time),
            )
        self._active_posture_refresh_last_update_s = float(current_time)
        self._active_posture_refresh_count += 1

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        bucket = int(np.floor(max(float(current_time), 0.0) / self.update_interval_s + 1e-9))
        replanned = bucket != self._last_bucket
        plant_info_now = self._economy_budget_observe_from_state(
            state,
            self._plant_info_from(plant_info_prev),
            float(current_time),
        )

        # Re-plan only on bucket boundaries.
        if replanned:
            self._last_bucket = bucket
            self._reset_forced_prefix_state()
            self._gusty_short_blend_applied = False
            self._gusty_short_target_delta_before_kg = 0.0
            self._gusty_short_target_delta_after_kg = 0.0
            self._gusty_hold_current_gate_evaluated = False
            self._gusty_hold_current_allowed = False
            self._gusty_hold_current_blocked = False
            self._gusty_hold_current_reason = "not_evaluated"
            self._direction_reversal_hold_current_gate_evaluated = False
            self._direction_reversal_hold_current_allowed = False
            self._direction_reversal_hold_current_blocked = False
            self._direction_reversal_hold_current_reason = "not_evaluated"
            self._forecast_advised_hold_current_override = False
            self._forecast_pause_resume_guard_active = False
            self._forecast_pause_resume_guard_reason = "not_evaluated"
            self._forecast_pause_resume_target_err_kg = 0.0
            ts = self._bucket_timestamp(current_time)
            sample = self.replay_dataset.sample_for_history_end(ts)
            if sample is None:
                # Fall back to zero target; transition timer NOT reset, decay continues.
                if self._target_action != "hold":
                    self._transition_time_s = float(current_time)
                self._target_action = "hold"
                self._target_pitch = 0.0
                self._target_roll = 0.0
                self._target_pitch_planner_frame = 0.0
                self._target_roll_planner_frame = 0.0
                if self.forecast_advised_economy_enabled:
                    # The forecast-advised economy profile may only own the
                    # primary target when a forecast sample exists.  Missing
                    # forecast data must fail closed to the baseline PI path
                    # instead of installing a zero/hold primary target.
                    self._primary_refresh_owner_pending = "missing_sample_fail_closed"
                    self._release_primary_target_to_pi(
                        self._plant_info_from(plant_info_prev),
                        float(current_time),
                    )
                    self._forecast_advised_economy_candidate = False
                    self._forecast_advised_economy_boundary_veto = True
                    self._forecast_advised_economy_headroom_deg = 0.0
                    self._forecast_advised_economy_near_rise_norm = 0.0
                    self._forecast_advised_economy_posture_release = False
                    self._forecast_advised_economy_max_axis_deg = 0.0
                    self._forecast_advised_economy_reason = (
                        "missing_sample_fail_closed"
                    )
                else:
                    self._primary_refresh_owner_pending = "missing_sample_zero_target"
                    self._update_primary_target(
                        self._plant_info_from(plant_info_prev),
                        np.zeros(2, dtype=float),
                        float(current_time),
                    )
                self._raw_pressure_block_norms = [0.0, 0.0, 0.0]
                self._raw_pressure_block_vecs = [np.zeros(2, dtype=float) for _ in range(3)]
                self._effective_pressure_block_norms = [0.0, 0.0, 0.0]
                self._effective_pressure_block_vecs = [
                    np.zeros(2, dtype=float) for _ in range(3)
                ]
                self._forecast_has_future = False
                self._forecast_event_probs_available = False
                self._event_risk_raw_probs = [0.0, 0.0, 0.0]
                self._event_risk_probs = [0.0, 0.0, 0.0]
                self._event_risk_block_scales = [1.0, 1.0, 1.0]
                self._reset_forecast_control_trust_state("missing_sample")
                self._reset_forecast_pressure_trust_state(
                    "missing_sample",
                    reset_observation=True,
                )
            else:
                plant_info = plant_info_now
                previous_action = str(self._target_action)
                previous_avec = np.array(
                    [self._target_pitch_planner_frame, self._target_roll_planner_frame],
                    dtype=float,
                )
                forecast_has_future = True
                if self.forecast_adapter is not None:
                    forecast_has_future = bool(
                        getattr(self.forecast_adapter, "provides_future_preview", True)
                    )
                best, first_avec, forecast_source, blocks = self._run_planner(
                    sample,
                    plant_info,
                    forecast_has_future=forecast_has_future,
                )
                original_first_action = str(best["sequence"][0])
                self._forced_original_raw_action = original_first_action
                raw_forced_action, raw_forced_avec = self._apply_forced_prefix_action(
                    "raw_action",
                    bucket,
                    original_first_action,
                    first_avec,
                    blocks,
                    plant_info,
                )
                if self._forced_prefix_active and self._forced_prefix_mode_active == "raw_action":
                    seq = tuple(best.get("sequence", ("hold", "hold", "hold")))
                    best = dict(best)
                    best["sequence"] = (raw_forced_action,) + tuple(seq[1:])
                    first_avec = raw_forced_avec
                pipeline_raw_action = str(best["sequence"][0])
                new_action = pipeline_raw_action
                new_action, first_avec = self._relief_medium_cap_action(
                    new_action,
                    first_avec,
                    blocks,
                    state,
                )
                new_action, first_avec = self._far_horizon_relief_gate_action(
                    new_action,
                    first_avec,
                    blocks,
                    state,
                )
                final_before_force = str(new_action)
                self._forced_original_final_action = final_before_force
                new_action, first_avec = self._apply_forced_prefix_action(
                    "final_action",
                    bucket,
                    new_action,
                    first_avec,
                    blocks,
                    plant_info,
                )
                self._observe_forecast_advised_economy_decision(
                    blocks,
                    plant_info,
                    float(current_time),
                    proposed_action=new_action,
                )
                if (
                    self.forecast_advised_economy_hold_current_on_candidate
                    and self._forecast_advised_economy_candidate
                    and str(new_action) != "hold"
                ):
                    # In the broad P2 economy family, a clear forecast means
                    # "do not chase this target update", not "issue a larger
                    # active target".  The continuous validation path therefore
                    # converts candidate active refreshes into a current-mass
                    # hold, while boundary/floor vetoes above still fail closed.
                    self._forecast_advised_hold_current_override = True
                    new_action = "hold"
                    first_avec = np.zeros(2, dtype=float)
                # first_avec is in PLANNER FRAME (compensation direction: which way to push
                # ballast to oppose wind pressure). For pump-first operation the default
                # setpoint sign is +1.0 because the signpos experiment was the only
                # pump-saving path; -1.0 remains an attitude-first diagnostic mode.
                planner_pitch = float(first_avec[0])
                planner_roll = float(first_avec[1])
                new_pitch = planner_pitch * self.setpoint_bias_sign
                new_roll = planner_roll * self.setpoint_bias_sign
                event_reset = self._should_reset_event(new_action, new_pitch, new_roll, float(current_time))
                posture_action_refresh = False
                self._posture_action_refresh_active = False
                self._sustained_active_recompute_active = False
                self._sustained_active_recompute_reason = "not_evaluated"
                stall_avec, stall_refresh, stall_reason = self._stall_refresh_action_vec(
                    first_avec,
                    plant_info,
                    float(current_time),
                )
                self._primary_stall_refresh_active = bool(
                    stall_refresh and new_action != "hold"
                )
                self._primary_stall_refresh_reason = (
                    stall_reason if new_action != "hold" else "hold_action"
                )
                if self._primary_stall_refresh_active:
                    first_avec = stall_avec
                    planner_pitch = float(first_avec[0])
                    planner_roll = float(first_avec[1])
                    new_pitch = planner_pitch * self.setpoint_bias_sign
                    new_roll = planner_roll * self.setpoint_bias_sign
                    event_reset = True
                first_avec, gusty_soft_cap_active, _gusty_soft_cap_reason = (
                    self._gusty_oscillation_soft_cap_action(first_avec, plant_info)
                )
                if gusty_soft_cap_active:
                    planner_pitch = float(first_avec[0])
                    planner_roll = float(first_avec[1])
                    new_pitch = planner_pitch * self.setpoint_bias_sign
                    new_roll = planner_roll * self.setpoint_bias_sign
                direction_reversal_entry_veto, direction_reversal_entry_veto_reason = (
                    self._direction_reversal_layered_entry_veto_active(
                        plant_info, blocks
                    )
                )
                if direction_reversal_entry_veto:
                    event_reset = False
                    posture_action_refresh = False
                    new_action = "hold"
                    first_avec = np.zeros(2, dtype=float)
                    planner_pitch = 0.0
                    planner_roll = 0.0
                    new_pitch = 0.0
                    new_roll = 0.0
                self._observe_active_effectiveness_bucket(
                    previous_action,
                    new_action,
                    plant_info,
                    blocks,
                    float(current_time),
                )
                if event_reset:
                    self._transition_time_s = float(current_time)
                    self._target_action = new_action
                self._target_pitch = new_pitch
                self._target_roll = new_roll
                self._target_pitch_planner_frame = planner_pitch
                self._target_roll_planner_frame = planner_roll
                direction_reversal_warning_active, direction_reversal_warning_reason = (
                    self._direction_reversal_layered_warning_active(plant_info, blocks)
                )
                if direction_reversal_warning_active:
                    event_reset = False
                target_update_action = str(new_action)
                target_update_avec = first_avec
                self._relief_envelope_reset_bucket_log()
                target_forced_action, target_forced_avec = self._apply_forced_prefix_action(
                    "target_update",
                    bucket,
                    target_update_action,
                    target_update_avec,
                    blocks,
                    plant_info,
                )
                if (
                    self._forced_prefix_active
                    and self._forced_prefix_mode_active == "target_update"
                ):
                    self._forced_prefix_target_update_active = True
                    self._forced_target_update_action = target_forced_action
                    target_update_action = target_forced_action
                    target_update_avec = target_forced_avec
                # h120 risk scheduler: single per-bucket refresh point. Sets
                # the effective delayed-medium delay read by the floor action
                # and the remote_risk flag read by the early_stop guard.
                self._h120_scheduler_refresh(plant_info)
                if self._h120_scheduler_remote_risk_active:
                    self._h120_scheduler_remote_risk_trigger_count += 1
                self._h120_pareto_mode_selector_refresh(plant_info)
                reactive_floor_refresh, reactive_floor_avec, _ = (
                    self._reactive_floor_needed(
                        plant_info,
                        blocks,
                        float(current_time),
                    )
                )
                oracle_preemptive_refresh, oracle_preemptive_avec, _ = (
                    self._oracle_preemptive_needed(
                        sample,
                        plant_info,
                        float(current_time),
                        reactive_floor_refresh,
                    )
                )
                h120_preemptive_refresh, h120_preemptive_avec, _ = (
                    self._h120_oracle_probe_preemptive_needed(
                        plant_info,
                        float(current_time),
                        reactive_floor_refresh or oracle_preemptive_refresh,
                    )
                )
                h120_prefloor_refresh, h120_prefloor_avec, _ = (
                    self._h120_scheduler_prefloor_probe_needed(
                        plant_info,
                        float(current_time),
                        (
                            reactive_floor_refresh
                            or oracle_preemptive_refresh
                            or h120_preemptive_refresh
                        ),
                    )
                )
                h120_axis_micro_refresh, h120_axis_micro_avec, _ = (
                    self._h120_axis_micro_prepare_needed(
                        plant_info,
                        float(current_time),
                        (
                            reactive_floor_refresh
                            or oracle_preemptive_refresh
                            or h120_preemptive_refresh
                            or h120_prefloor_refresh
                        ),
                    )
                )
                budget_hold_release, budget_hold_release_reason = (
                    self._economy_budget_hold_release_needed(plant_info)
                )
                sustained_recompute, _ = (
                    self._sustained_active_recompute_needed(new_action)
                    if (
                        not event_reset
                        and self._primary_target_initialized
                        and not reactive_floor_refresh
                        and not oracle_preemptive_refresh
                        and not h120_preemptive_refresh
                        and not h120_prefloor_refresh
                        and not h120_axis_micro_refresh
                    )
                    else (False, "event_reset_or_uninitialized")
                )
                if self._forced_prefix_target_update_active:
                    if target_update_action == "hold":
                        self._primary_refresh_owner_pending = "forced_target_hold_current"
                        self._hold_primary_target_at_current(plant_info, float(current_time))
                    else:
                        self._primary_refresh_owner_pending = "forced_target_update"
                        self._update_primary_target(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                elif reactive_floor_refresh:
                    self._primary_refresh_owner_pending = "reactive_floor"
                    self._update_primary_target(
                        plant_info,
                        reactive_floor_avec,
                        float(current_time),
                    )
                elif oracle_preemptive_refresh:
                    self._primary_refresh_owner_pending = "oracle_preemptive"
                    self._update_primary_target(
                        plant_info,
                        oracle_preemptive_avec,
                        float(current_time),
                    )
                elif h120_preemptive_refresh:
                    self._primary_refresh_owner_pending = "h120_preemptive"
                    self._update_primary_target(
                        plant_info,
                        h120_preemptive_avec,
                        float(current_time),
                    )
                elif h120_prefloor_refresh:
                    self._primary_refresh_owner_pending = "h120_prefloor"
                    self._update_primary_target(
                        plant_info,
                        h120_prefloor_avec,
                        float(current_time),
                    )
                elif h120_axis_micro_refresh:
                    self._primary_refresh_owner_pending = "h120_axis_micro"
                    self._update_primary_target(
                        plant_info,
                        h120_axis_micro_avec,
                        float(current_time),
                    )
                elif direction_reversal_entry_veto:
                    self._primary_refresh_owner_pending = (
                        "direction_reversal_v8_entry_veto_pi_release"
                    )
                    self._release_primary_target_to_pi(plant_info, float(current_time))
                elif budget_hold_release:
                    self._economy_pump_budget_hold_latched = False
                    self._economy_pump_budget_forced_release_count += 1
                    self._economy_pump_budget_forced_release_reason = (
                        budget_hold_release_reason
                    )
                    self._primary_refresh_owner_pending = "budget_hold_release"
                    self._update_primary_target(
                        plant_info,
                        target_update_avec,
                        float(current_time),
                    )
                elif (
                    self.gusty_hold_current_isolated_primary_enabled
                    and new_action != "hold"
                ):
                    self._primary_refresh_owner_pending = (
                        "gusty_hold_current_isolated_nonhold_pi_release"
                    )
                    self._release_primary_target_to_pi(plant_info, float(current_time))
                elif (
                    self.prediction_primary_enabled
                    and new_action == "hold"
                ):
                    self._remember_primary_target_for_pause(previous_action, previous_avec)
                    forecast_hold_fail_closed = bool(
                        self.forecast_advised_economy_enabled
                        and not self._forecast_advised_economy_candidate
                    )
                    release_hold, release_reason = self._should_release_hold_to_pi(state)
                    if forecast_hold_fail_closed:
                        self._primary_refresh_owner_pending = (
                            "forecast_advised_hold_noncandidate_pi_release"
                        )
                        self._release_primary_target_to_pi(
                            plant_info,
                            float(current_time),
                        )
                    elif self.primary_hold_target_mode == "pi_release" or release_hold:
                        self._primary_refresh_owner_pending = "hold_action_pi_release"
                        self._release_primary_target_to_pi(plant_info, float(current_time))
                    else:
                        hold_current_allowed = True
                        hold_current_reason = "ungated"
                        if self.gusty_hold_current_gate_mode == "gate_off":
                            self._gusty_hold_current_gate_evaluated = True
                            hold_current_allowed = False
                            hold_current_reason = "gate_off_ablation"
                        elif self.gusty_hold_current_gate_mode == "gate_on":
                            self._gusty_hold_current_gate_evaluated = True
                            hold_current_allowed, hold_current_reason = (
                                self._gusty_oscillation_refresh_gate(
                                    plant_info,
                                    float(current_time),
                                )
                            )
                        if self.direction_reversal_hold_current_gate_mode == "gate_off":
                            self._direction_reversal_hold_current_gate_evaluated = True
                            hold_current_allowed = False
                            hold_current_reason = "direction_reversal_gate_off_ablation"
                        elif (
                            self.direction_reversal_hold_current_gate_mode == "gate_on"
                        ):
                            self._direction_reversal_hold_current_gate_evaluated = True
                            hold_current_allowed, hold_current_reason = (
                                self._economy_budget_allocator_decision(plant_info)
                            )
                            hold_current_reason = (
                                f"direction_reversal_gate:{hold_current_reason}"
                            )
                        self._gusty_hold_current_allowed = bool(hold_current_allowed)
                        self._gusty_hold_current_blocked = not bool(
                            hold_current_allowed
                        )
                        self._gusty_hold_current_reason = str(hold_current_reason)
                        self._direction_reversal_hold_current_allowed = bool(
                            hold_current_allowed
                        )
                        self._direction_reversal_hold_current_blocked = not bool(
                            hold_current_allowed
                        )
                        self._direction_reversal_hold_current_reason = str(
                            hold_current_reason
                        )
                        if hold_current_allowed:
                            self._primary_refresh_owner_pending = (
                                "hold_action_hold_current_gusty_gated"
                                if self.gusty_hold_current_gate_mode != "off"
                                else (
                                    "hold_action_hold_current_direction_reversal_gated"
                                    if self.direction_reversal_hold_current_gate_mode
                                    != "off"
                                    else "hold_action_hold_current"
                                )
                            )
                            self._hold_primary_target_at_current(
                                plant_info,
                                float(current_time),
                            )
                            if self.gusty_hold_current_gate_mode == "gate_on":
                                self._record_gusty_hold_current_day_budget_use(
                                    float(current_time)
                                )
                            self._gusty_hold_current_latched_active = bool(
                                self.gusty_hold_current_gate_mode == "gate_on"
                                or self.direction_reversal_hold_current_gate_mode
                                == "gate_on"
                            )
                        else:
                            self._gusty_hold_current_latched_active = False
                            blocked_owner = (
                                "hold_action_hold_current_gusty_blocked_pi_release"
                                if self.gusty_hold_current_gate_mode != "off"
                                else "hold_action_hold_current_direction_reversal_blocked_pi_release"
                            )
                            self._primary_refresh_owner_pending = blocked_owner
                            self._release_primary_target_to_pi(
                                plant_info,
                                float(current_time),
                            )
                elif previous_action == "hold" and self._resume_paused_primary_target(
                    plant_info,
                    target_update_avec,
                    float(current_time),
                ):
                    pass
                elif (
                    event_reset
                    or not self._primary_target_initialized
                    or sustained_recompute
                ):
                    # event_reset-level trigger: sustained-active + stalled-pump
                    # + posture-above-band is treated structurally as "the
                    # active-execution chain is idling on a stale target —
                    # ask the planner to recompute". Same priority as a
                    # discrete action change. The narrow conjunctive
                    # active_effectiveness_refresh below remains for the
                    # additional worsening-driven case it was tuned for.
                    relief_envelope_delay = False
                    relief_envelope_cap_mode = "none"
                    relief_envelope_cap_avec = None
                    if self._primary_target_initialized:
                        if self.relief_envelope_adaptive_axis_cap:
                            relief_envelope_eligible = self._relief_envelope_delay_needed(
                                plant_info,
                                float(current_time),
                                would_refresh=True,
                                refresh_avec=target_update_avec,
                                ignore_action_shape=True,
                                count_delay=False,
                            )
                            if relief_envelope_eligible:
                                relief_envelope_cap_mode, relief_envelope_cap_avec = (
                                    self._relief_envelope_axis_cap_decision(
                                        target_update_avec
                                    )
                                )
                                relief_envelope_delay = (
                                    relief_envelope_cap_mode == "full_delay"
                                )
                                if relief_envelope_delay:
                                    self._relief_envelope_target_refresh_delayed_count += 1
                        else:
                            relief_envelope_delay = self._relief_envelope_delay_needed(
                                plant_info,
                                float(current_time),
                                would_refresh=True,
                                refresh_avec=target_update_avec,
                            )
                    if (
                        self._primary_target_initialized
                        and self._debug_safe_hold_active(plant_info)
                    ):
                        # DEBUG-ONLY forced-positive (default off): hold instead of
                        # refresh on non-floor refreshes while posture is safe, to
                        # measure savable pump. Never overrides the floor.
                        self._relief_envelope_target_refresh_delayed_count += 1
                        self._primary_refresh_owner_pending = "event_debug_safe_hold_reuse"
                        self._reuse_or_refresh_direction_reversal_layered(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif relief_envelope_delay:
                        self._primary_refresh_owner_pending = "event_relief_envelope_reuse"
                        self._reuse_or_refresh_direction_reversal_layered(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif (
                        relief_envelope_cap_mode == "cap"
                        and relief_envelope_cap_avec is not None
                    ):
                        self._relief_envelope_axis_cap_count += 1
                        self._primary_refresh_owner_pending = "event_relief_envelope_cap"
                        self._update_primary_target(
                            plant_info,
                            relief_envelope_cap_avec,
                            float(current_time),
                        )
                    elif (
                        self._primary_target_initialized
                        and str(self.economy_pump_budget_allocator_mode)
                        == "gusty_oscillation_soft_cap_auto"
                        and self._gusty_oscillation_refresh_gate(
                            plant_info,
                            float(current_time),
                        )[0]
                    ):
                        old_avec = np.array(
                            [
                                self._target_pitch_planner_frame,
                                self._target_roll_planner_frame,
                            ],
                            dtype=float,
                        )
                        scale = self._gusty_oscillation_scaled_target_blend(plant_info)
                        blended_avec = old_avec + scale * (
                            np.asarray(target_update_avec, dtype=float) - old_avec
                        )
                        masses = plant_primary_masses(
                            plant_info,
                            self._primary_anchor_masses_kg,
                        )
                        proposed_target = np.clip(
                            masses + self._primary_mass_delta_kg(target_update_avec),
                            0.0,
                            self.cfg.tank_capacity_kg,
                        )
                        blended_target = np.clip(
                            masses + self._primary_mass_delta_kg(blended_avec),
                            0.0,
                            self.cfg.tank_capacity_kg,
                        )
                        self._gusty_short_blend_applied = True
                        self._gusty_short_target_delta_before_kg = float(
                            np.mean(np.abs(proposed_target - self._primary_target_kg))
                        )
                        self._gusty_short_target_delta_after_kg = float(
                            np.mean(np.abs(blended_target - self._primary_target_kg))
                        )
                        self._primary_refresh_owner_pending = "event_gusty_blend"
                        self._update_primary_target(
                            plant_info,
                            blended_avec,
                            float(current_time),
                        )
                    elif (
                        self._primary_target_initialized
                        and not self._economy_budget_allows_refresh(
                            would_refresh=True,
                            refresh_avec=target_update_avec,
                            plant_info=plant_info,
                        )
                    ):
                        self._primary_refresh_owner_pending = (
                            "event_budget_mild_refresh"
                            if self.economy_pump_budget_hold_refresh_fraction > 0.0
                            else "event_budget_hold_current"
                        )
                        self._apply_economy_budget_blocked_refresh(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    else:
                        if not self._primary_target_initialized:
                            self._primary_refresh_owner_pending = "initial_target"
                        elif sustained_recompute:
                            self._primary_refresh_owner_pending = "sustained_recompute"
                        elif event_reset:
                            self._primary_refresh_owner_pending = "event_reset"
                        else:
                            self._primary_refresh_owner_pending = "event_refresh"
                        self._update_primary_target(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                elif direction_reversal_warning_active:
                    self._primary_refresh_owner_pending = "direction_reversal_v7_warning_pi_release"
                    self._release_primary_target_to_pi(plant_info, float(current_time))
                else:
                    active_effective_refresh, _ = self._active_effectiveness_refresh_needed(
                        plant_info,
                        target_update_avec,
                        new_action,
                    )
                    stale_target_refresh, _ = self._stale_active_target_refresh_needed(
                        plant_info,
                        target_update_avec,
                        new_action,
                        float(current_time),
                    )
                    relief_economy_delay = self._relief_economy_delay_needed(plant_info)
                    would_refresh = bool(
                        active_effective_refresh
                        or stale_target_refresh
                    )
                    relief_envelope_delay = False
                    relief_envelope_cap_mode = "none"
                    relief_envelope_cap_avec = None
                    if self.relief_envelope_adaptive_axis_cap:
                        relief_envelope_eligible = self._relief_envelope_delay_needed(
                            plant_info,
                            float(current_time),
                            would_refresh=would_refresh,
                            refresh_avec=target_update_avec,
                            ignore_action_shape=True,
                            count_delay=False,
                        )
                        if relief_envelope_eligible and would_refresh:
                            relief_envelope_cap_mode, relief_envelope_cap_avec = (
                                self._relief_envelope_axis_cap_decision(
                                    target_update_avec
                                )
                            )
                            relief_envelope_delay = (
                                relief_envelope_cap_mode == "full_delay"
                            )
                            if relief_envelope_delay:
                                self._relief_envelope_target_refresh_delayed_count += 1
                    else:
                        relief_envelope_delay = self._relief_envelope_delay_needed(
                            plant_info,
                            float(current_time),
                            would_refresh=would_refresh,
                            refresh_avec=target_update_avec,
                        )
                    if would_refresh and self._debug_safe_hold_active(plant_info):
                        # DEBUG-ONLY forced-positive (default off): hold instead of
                        # refresh on the economy branch while posture is safe.
                        self._relief_envelope_target_refresh_delayed_count += 1
                        self._primary_refresh_owner_pending = "economy_debug_safe_hold_reuse"
                        self._reuse_or_refresh_direction_reversal_layered(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif relief_envelope_delay and would_refresh:
                        self._primary_refresh_owner_pending = "economy_relief_envelope_reuse"
                        self._reuse_or_refresh_direction_reversal_layered(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif relief_economy_delay and would_refresh:
                        # Non-urgent economy refresh delayed under clear forecast
                        # relief + safe posture. Floor/recovery/safety are never
                        # reached here (earlier elif branches). Pure delay: hold
                        # the current target instead of refreshing toward a new one.
                        self._relief_economy_delay_count += 1
                        self._primary_refresh_owner_pending = "economy_relief_delay_reuse"
                        self._reuse_or_refresh_direction_reversal_layered(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif would_refresh and not self._economy_budget_allows_refresh(
                        would_refresh=True,
                        refresh_avec=target_update_avec,
                        plant_info=plant_info,
                    ):
                        self._primary_refresh_owner_pending = (
                            "economy_budget_mild_refresh"
                            if self.economy_pump_budget_hold_refresh_fraction > 0.0
                            else "economy_budget_hold_current"
                        )
                        self._apply_economy_budget_blocked_refresh(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif (
                        relief_envelope_cap_mode == "cap"
                        and relief_envelope_cap_avec is not None
                    ):
                        self._relief_envelope_axis_cap_count += 1
                        self._primary_refresh_owner_pending = "economy_relief_envelope_cap"
                        self._update_primary_target(
                            plant_info,
                            relief_envelope_cap_avec,
                            float(current_time),
                        )
                    elif active_effective_refresh:
                        self._primary_refresh_owner_pending = "active_effectiveness"
                        self._update_primary_target(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    elif stale_target_refresh:
                        self._primary_refresh_owner_pending = "stale_active"
                        self._update_primary_target(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                    else:
                        if (
                            self.stale_active_target_refresh_mode != "off"
                            and not self._stale_active_target_refresh_active
                            and self._stale_active_target_refresh_reason
                            not in ("disabled", "not_evaluated", "refresh_no_effect")
                        ):
                            self._stale_active_target_refresh_rejected_count += 1
                        self._primary_refresh_owner_pending = "reuse"
                        self._reuse_or_refresh_direction_reversal_layered(
                            plant_info,
                            target_update_avec,
                            float(current_time),
                        )
                self._relief_envelope_mark_not_applicable()
                self._hold_relief_debt_observe_after_action(new_action, plant_info)
                self.records.append({
                    "bucket": bucket,
                    "current_time_s": float(current_time),
                    "history_end": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "first_action": new_action,
                    "planner_first_action_raw": pipeline_raw_action,
                    "forced_prefix_active": int(self._forced_prefix_active),
                    "forced_prefix_mode": self._forced_prefix_mode_active,
                    "forced_prefix_bucket": int(self._forced_prefix_bucket),
                    "forced_action_label": self._forced_action_label,
                    "forced_pitch_deg": float(self._forced_pitch_deg),
                    "forced_roll_deg": float(self._forced_roll_deg),
                    "forced_vector_mode": self._forced_vector_mode,
                    "forced_source_label": self._forced_source_label,
                    "forced_original_raw_action": self._forced_original_raw_action,
                    "forced_original_final_action": self._forced_original_final_action,
                    "forced_target_update_active": int(
                        self._forced_prefix_target_update_active
                    ),
                    "forced_target_update_action": self._forced_target_update_action,
                    "best_sequence": ">".join(best["sequence"]),
                    # planner frame: compensation direction (sign as planner sees it)
                    "planner_action_pitch_deg": planner_pitch,
                    "planner_action_roll_deg": planner_roll,
                    "setpoint_bias_sign": float(self.setpoint_bias_sign),
                    # Objective-mode diagnostics (default-off prototype).
                    "objective_mode_active": str(best.get("objective_mode_active", "economic")),
                    "objective_mode_persistent_high_buckets": int(
                        best.get("objective_mode_persistent_high_buckets", 0)
                    ),
                    "objective_mode_future_rising": int(
                        best.get("objective_mode_future_rising", 0)
                    ),
                    "objective_mode_switch_count": int(self._objective_mode_switch_count),
                    # Safety-floor planned reactive handoff (safety_floor_v1 profile).
                    # Per-bucket signal: 1 when ALL planner sequences violated the
                    # safety_floor and the bucket was routed to the least-bad
                    # (most aggressive) sequence with the handoff label.
                    "planned_reactive_handoff": int(best.get("planned_reactive_handoff", 0)),
                    "safety_floor_predicted_pitch": float(
                        best.get("safety_floor_predicted_pitch", 0.0)
                    ),
                    "safety_floor_predicted_roll": float(
                        best.get("safety_floor_predicted_roll", 0.0)
                    ),
                    "safety_floor_reject_count": int(
                        best.get("safety_floor_reject_count", 0)
                    ),
                    "safety_floor_total_sequences": int(
                        best.get("safety_floor_total_sequences", 0)
                    ),
                    "prediction_primary_enabled": int(self.prediction_primary_enabled),
                    "prediction_primary_scale": float(self.prediction_primary_scale),
                    "prediction_primary_target_t1_kg": float(self._primary_target_kg[0]),
                    "prediction_primary_target_t2_kg": float(self._primary_target_kg[1]),
                    "prediction_primary_target_t3_kg": float(self._primary_target_kg[2]),
                    "prediction_primary_delta_abs_mean_kg": float(np.mean(np.abs(self._primary_delta_kg))),
                    "prediction_primary_event_reset": int(event_reset),
                    "prediction_primary_posture_action_refresh": int(
                        posture_action_refresh
                    ),
                    "prediction_primary_stall_refresh": int(
                        self._primary_stall_refresh_active
                    ),
                    "prediction_primary_stall_refresh_reason": (
                        self._primary_stall_refresh_reason
                    ),
                    "prediction_primary_stall_axis_bias": int(
                        self._primary_stall_axis_bias_active
                    ),
                    "prediction_primary_stall_target_err_mean_kg": float(
                        self._primary_stall_target_err_mean_kg
                    ),
                    "prediction_primary_stall_pump_rate_m3_min": float(
                        self._primary_stall_pump_rate_m3_min
                    ),
                    **frozen_active_intent_record_defaults("prediction_"),
                    "active_effectiveness_refresh_enabled": int(
                        self.active_effectiveness_refresh_enabled
                    ),
                    "active_effectiveness_refresh_active": int(
                        self._active_effectiveness_refresh_active
                    ),
                    "active_effectiveness_refresh_reason": (
                        self._active_effectiveness_refresh_reason
                    ),
                    "active_effectiveness_repeated_active_buckets": int(
                        self._active_effectiveness_repeated_active_buckets
                    ),
                    "active_effectiveness_pitch_abs_deg": float(
                        self._active_effectiveness_pitch_abs_deg
                    ),
                    "active_effectiveness_roll_abs_deg": float(
                        self._active_effectiveness_roll_abs_deg
                    ),
                    "active_effectiveness_attitude_abs_deg": float(
                        self._active_effectiveness_attitude_abs_deg
                    ),
                    "active_effectiveness_current_response_deg": float(
                        self._active_effectiveness_current_response_deg
                    ),
                    "active_effectiveness_future_relief_status": (
                        self._active_effectiveness_future_relief_status
                    ),
                    "active_effectiveness_target_err_mean_kg": float(
                        self._active_effectiveness_target_err_mean_kg
                    ),
                    "active_effectiveness_pump_rate_m3_min": float(
                        self._active_effectiveness_pump_rate_m3_min
                    ),
                    "active_effectiveness_target_age_s": float(
                        self._active_effectiveness_target_age_s
                    ),
                    "active_effectiveness_pressure_max_norm": float(
                        self._active_effectiveness_pressure_max_norm
                    ),
                    "active_effectiveness_target_ready": int(
                        self._active_effectiveness_target_ready
                    ),
                    "active_effectiveness_pump_idle": int(
                        self._active_effectiveness_pump_idle
                    ),
                    "active_effectiveness_proposal_delta_mean_kg": float(
                        self._active_effectiveness_proposal_delta_mean_kg
                    ),
                    "active_effectiveness_proposal_delta_m3": float(
                        self._active_effectiveness_proposal_delta_m3
                    ),
                    "active_effectiveness_budget_used_m3": float(
                        self._active_effectiveness_budget_used_m3
                    ),
                    "active_effectiveness_lowrisk_safe": int(
                        self._active_effectiveness_lowrisk_safe
                    ),
                    "active_effectiveness_fallback_dominated": int(
                        self._active_effectiveness_fallback_dominated
                    ),
                    "sustained_active_recompute_enabled": int(
                        self.sustained_active_recompute_enabled
                    ),
                    "sustained_active_recompute_active": int(
                        self._sustained_active_recompute_active
                    ),
                    "sustained_active_recompute_reason": (
                        self._sustained_active_recompute_reason
                    ),
                    "active_posture_refresh_enabled": int(
                        self.active_posture_refresh_enabled
                    ),
                    "active_posture_refresh_active": int(
                        self._active_posture_refresh_active
                    ),
                    "active_posture_refresh_reason": (
                        self._active_posture_refresh_reason
                    ),
                    "active_posture_refresh_pitch_abs_deg": float(
                        self._active_posture_refresh_pitch_abs_deg
                    ),
                    "active_posture_refresh_roll_abs_deg": float(
                        self._active_posture_refresh_roll_abs_deg
                    ),
                    "active_posture_refresh_action_name": (
                        self._active_posture_refresh_action_name
                    ),
                    "active_posture_refresh_pressure_max_norm": float(
                        self.active_posture_refresh_pressure_max_norm
                    ),
                    "active_posture_refresh_min_target_age_s": float(
                        self.active_posture_refresh_min_target_age_s
                    ),
                    "active_posture_refresh_axis_mode": str(
                        self.active_posture_refresh_axis_mode
                    ),
                    "active_posture_refresh_axis_mode_active": (
                        self._active_posture_refresh_axis_mode_active
                    ),
                    "active_posture_refresh_axis_mask": (
                        self._active_posture_refresh_axis_mask
                    ),
                    "active_posture_refresh_worsening_eps_deg_s": float(
                        self.active_posture_refresh_worsening_eps_deg_s
                    ),
                    "active_posture_refresh_debt_deg": float(
                        self.active_posture_refresh_debt_deg
                    ),
                    "active_posture_refresh_pitch_worsening_deg_s": float(
                        self._active_posture_refresh_pitch_worsening_deg_s
                    ),
                    "active_posture_refresh_roll_worsening_deg_s": float(
                        self._active_posture_refresh_roll_worsening_deg_s
                    ),
                    "active_posture_refresh_count": int(
                        self._active_posture_refresh_count
                    ),
                    **self._stale_active_target_log_fields(),
                    **self._reactive_floor_log_fields(),
                    **self._oracle_preemptive_log_fields(),
                    **self._h120_oracle_probe_log_fields(),
                    **self._h120_scheduler_log_fields(),
                    **self._h120_axis_micro_log_fields(),
                    **self._relief_economy_log_fields(),
                    **self._relief_envelope_log_fields(),
                    **self._economy_budget_log_fields(),
                    **self._h120_floor_shaping_log_fields(),
                    **self._h120_pareto_mode_selector_log_fields(),
                    "prediction_primary_target_refreshed": int(self._primary_target_refreshed),
                    "prediction_primary_target_reused": int(self._primary_target_reused),
                    "prediction_primary_target_resumed": int(self._primary_target_resumed),
                    "prediction_primary_refresh_owner": self._primary_refresh_owner,
                    "prediction_primary_hold_target_mode": self.primary_hold_target_mode,
                    "prediction_primary_target_age_s": float(current_time) - float(self._primary_target_last_reset_s),
                    # setpoint frame (after sign contract applied): what controller sees
                    "target_pitch_deg": new_pitch,
                    "target_roll_deg": new_roll,
                    "transition_time_s": float(self._transition_time_s),
                    "objective_mode": self.objective_mode,
                    "event_reset_mode": self.event_reset_mode,
                    "forecast_source": forecast_source,
                    # Blend-forecast attribution telemetry. Populated only
                    # when forecast_adapter is a BlendedForecastAdapter; for
                    # baseline/e15/oracle/persistence runs all blend_* fields
                    # are blank/0 so non-blend runs gain only inert columns.
                    "blend_mode": str(
                        self._last_forecast_blend_diagnostic.get("blend_mode", "")
                    ),
                    "blend_alpha": float(
                        self._last_forecast_blend_diagnostic.get("blend_alpha", 0.0)
                    ),
                    "blend_alpha_candidate": float(
                        self._last_forecast_blend_diagnostic.get("blend_alpha_candidate", 0.0)
                    ),
                    "blend_floor_ratio": float(
                        self._last_forecast_blend_diagnostic.get("blend_floor_ratio", 1.0)
                    ),
                    "blend_floor_triggered": int(
                        self._last_forecast_blend_diagnostic.get("blend_floor_triggered", 0)
                    ),
                    "blend_floor_shrink_amount": float(
                        self._last_forecast_blend_diagnostic.get("blend_floor_shrink_amount", 0.0)
                    ),
                    "blend_baseline_relief": int(
                        self._last_forecast_blend_diagnostic.get("blend_baseline_relief", 0)
                    ),
                    "blend_baseline_weak_relief": int(
                        self._last_forecast_blend_diagnostic.get("blend_baseline_weak_relief", 0)
                    ),
                    "blend_baseline_confirmed_relief": int(
                        self._last_forecast_blend_diagnostic.get("blend_baseline_confirmed_relief", 0)
                    ),
                    "blend_baseline_intensify": int(
                        self._last_forecast_blend_diagnostic.get("blend_baseline_intensify", 0)
                    ),
                    "blend_baseline_strong_intensify": int(
                        self._last_forecast_blend_diagnostic.get("blend_baseline_strong_intensify", 0)
                    ),
                    "blend_e15_relief": int(
                        self._last_forecast_blend_diagnostic.get("blend_e15_relief", 0)
                    ),
                    "blend_e15_strong_relief": int(
                        self._last_forecast_blend_diagnostic.get("blend_e15_strong_relief", 0)
                    ),
                    "blend_e15_relief_blocks_count": int(
                        self._last_forecast_blend_diagnostic.get("blend_e15_relief_blocks_count", 0)
                    ),
                    "blend_e15_intensify": int(
                        self._last_forecast_blend_diagnostic.get("blend_e15_intensify", 0)
                    ),
                    "blend_swing_proxy": int(
                        self._last_forecast_blend_diagnostic.get("blend_swing_proxy", 0)
                    ),
                    "blend_confirmation_pass": int(
                        self._last_forecast_blend_diagnostic.get("blend_confirmation_pass", 0)
                    ),
                    "blend_rule_reason": str(
                        self._last_forecast_blend_diagnostic.get("blend_rule_reason", "")
                    ),
                    "best_scalar_cost": float(best.get("best_scalar_cost", float("nan"))),
                    "max_envelope_norm": float(best.get("costs", {}).get("max_envelope_norm", float("nan"))),
                    "envelope_violation_cost": float(best.get("costs", {}).get("envelope_violation_cost", float("nan"))),
                    "envelope_barrier_triggered": int(
                        float(best.get("costs", {}).get("envelope_barrier_triggered", 0.0)) > 0.5
                    ),
                    "posture_hold_barrier_triggered": int(
                        float(best.get("costs", {}).get("posture_hold_barrier_triggered", 0.0)) > 0.5
                    ),
                    "posture_hold_barrier_cost": float(
                        best.get("costs", {}).get("posture_hold_barrier_cost", 0.0)
                    ),
                    "posture_hold_norm": float(
                        best.get("costs", {}).get("posture_hold_norm", 0.0)
                    ),
                    "posture_hold_forecast_credit": float(
                        best.get("costs", {}).get("posture_hold_forecast_credit", 0.0)
                    ),
                    "hold_relief_debt_enabled": int(self.hold_relief_debt_enabled),
                    "hold_relief_debt_applied": int(self._hold_relief_debt_applied),
                    "hold_relief_debt_planner_level": float(
                        self._hold_relief_debt_planner_level
                    ),
                    "hold_relief_debt_level": float(self._hold_relief_debt_level),
                    "hold_relief_debt_cost": float(
                        best.get("costs", {}).get("hold_relief_debt_cost", 0.0)
                    ),
                    "hold_relief_debt_reason": self._hold_relief_debt_reason,
                    "hold_relief_debt_wait_reason": self._hold_relief_debt_wait_reason,
                    "hold_relief_debt_pending": int(self._hold_relief_debt_pending),
                    "hold_relief_debt_pending_reason": (
                        self._hold_relief_debt_pending_reason
                    ),
                    "hold_relief_debt_current_response_deg": float(
                        self._hold_relief_debt_current_response_deg
                    ),
                    "hold_relief_debt_lowrisk_safe": int(
                        self._hold_relief_debt_lowrisk_safe
                    ),
                    "hold_relief_debt_fallback_dominated": int(
                        self._hold_relief_debt_fallback_dominated
                    ),
                    "posture_hold_action_posture_directed": int(
                        float(
                            best.get("costs", {}).get(
                                "posture_hold_action_posture_directed", 0.0
                            )
                        )
                        > 0.5
                    ),
                    "posture_hold_barrier_reason": str(
                        best.get("posture_hold_barrier_reason", "")
                    ),
                    "current_pitch_deg": float(
                        np.asarray(plant_info.get("posture_vec_deg", [0.0, 0.0]), dtype=float)[0]
                    ),
                    "current_roll_deg": float(
                        np.asarray(plant_info.get("posture_vec_deg", [0.0, 0.0]), dtype=float)[1]
                    ),
                    "forecast_has_future": int(bool(forecast_has_future)),
                    "forecast_control_trust_gate_enabled": int(
                        self.forecast_control_trust_gate_enabled
                    ),
                    "forecast_control_trust_ok": int(
                        self._forecast_control_trust_ok
                    ),
                    "forecast_control_trust_reason": str(
                        self._forecast_control_trust_reason
                    ),
                    "forecast_control_trust_purpose": str(
                        self._forecast_control_trust_purpose
                    ),
                    "forecast_control_trust_pressure_max": float(
                        self._forecast_control_trust_pressure_max
                    ),
                    "forecast_control_trust_near_rise": float(
                        self._forecast_control_trust_near_rise
                    ),
                    "forecast_control_trust_dot02": float(
                        self._forecast_control_trust_dot02
                    ),
                    "forecast_control_trust_event_prob_max": float(
                        self._forecast_control_trust_event_prob_max
                    ),
                    "forecast_control_trust_speed_range_ms": float(
                        self._forecast_control_trust_speed_range_ms
                    ),
                    "forecast_control_trust_dir_shift_deg": float(
                        self._forecast_control_trust_dir_shift_deg
                    ),
                    "forecast_control_trust_current_ws_range_limit_ms": float(
                        self._forecast_control_trust_current_ws_range_limit_ms
                    ),
                    "forecast_control_trust_current_dir_shift_limit_deg": float(
                        self._forecast_control_trust_current_dir_shift_limit_deg
                    ),
                    "forecast_control_trust_suppression_ok": int(
                        self._forecast_control_trust_by_purpose["suppression"][0]
                    ),
                    "forecast_control_trust_suppression_reason": str(
                        self._forecast_control_trust_by_purpose["suppression"][1]
                    ),
                    "forecast_control_trust_direction_reversal_hold_ok": int(
                        self._forecast_control_trust_by_purpose[
                            "direction_reversal_hold"
                        ][0]
                    ),
                    "forecast_control_trust_direction_reversal_hold_reason": str(
                        self._forecast_control_trust_by_purpose[
                            "direction_reversal_hold"
                        ][1]
                    ),
                    "forecast_control_trust_gusty_hold_current_ok": int(
                        self._forecast_control_trust_by_purpose[
                            "gusty_hold_current"
                        ][0]
                    ),
                    "forecast_control_trust_gusty_hold_current_reason": str(
                        self._forecast_control_trust_by_purpose[
                            "gusty_hold_current"
                        ][1]
                    ),
                    **self._forecast_pressure_trust_log_fields(),
                    "pump_work_cost": float(best.get("costs", {}).get("pump_work_cost", float("nan"))),
                    "active_medium_gate_reason": str(best.get("active_medium_gate_reason", "")),
                    "reverse_release_reason": str(best.get("reverse_release_reason", "")),
                    **self._forecast_advised_pump_suppression_record_fields(),
                    "gusty_oscillation_enabled": int(self.gusty_oscillation_enabled),
                    "gusty_oscillation_gate_active": int(
                        self._gusty_oscillation_gate_active
                    ),
                    "gusty_oscillation_gate_reason": str(
                        self._gusty_oscillation_gate_reason
                    ),
                    "gusty_oscillation_min_target_age_s": float(
                        self.gusty_oscillation_min_target_age_s
                    ),
                    "gusty_oscillation_headroom_scaled": int(
                        self.gusty_oscillation_headroom_scaled
                    ),
                    "gusty_oscillation_target_scale": float(
                        self.gusty_oscillation_target_scale
                    ),
                    "gusty_oscillation_last_scale": float(
                        self._gusty_oscillation_last_scale
                    ),
                    "gusty_oscillation_short_gate_enabled": int(
                        self.gusty_oscillation_short_gate_enabled
                    ),
                    "gusty_short_blend_applied": int(
                        self._gusty_short_blend_applied
                    ),
                    "gusty_short_target_delta_before_kg": float(
                        self._gusty_short_target_delta_before_kg
                    ),
                    "gusty_short_target_delta_after_kg": float(
                        self._gusty_short_target_delta_after_kg
                    ),
                    "gusty_hold_current_gate_mode": str(
                        self.gusty_hold_current_gate_mode
                    ),
                    "gusty_hold_current_isolated_primary_enabled": int(
                        self.gusty_hold_current_isolated_primary_enabled
                    ),
                    "gusty_hold_current_gate_evaluated": int(
                        self._gusty_hold_current_gate_evaluated
                    ),
                    "gusty_hold_current_allowed": int(
                        self._gusty_hold_current_allowed
                    ),
                    "gusty_hold_current_blocked": int(
                        self._gusty_hold_current_blocked
                    ),
                    "gusty_hold_current_reason": str(
                        self._gusty_hold_current_reason
                    ),
                    "direction_reversal_hold_current_gate_mode": str(
                        self.direction_reversal_hold_current_gate_mode
                    ),
                    "direction_reversal_hold_current_gate_evaluated": int(
                        self._direction_reversal_hold_current_gate_evaluated
                    ),
                    "direction_reversal_hold_current_allowed": int(
                        self._direction_reversal_hold_current_allowed
                    ),
                    "direction_reversal_hold_current_blocked": int(
                        self._direction_reversal_hold_current_blocked
                    ),
                    "direction_reversal_hold_current_reason": str(
                        self._direction_reversal_hold_current_reason
                    ),
                    "gusty_hold_current_latched_active": int(
                        self._gusty_hold_current_latched_active
                    ),
                    "gusty_hold_current_runtime_release_enabled": int(
                        self.gusty_hold_current_runtime_release_enabled
                    ),
                    "gusty_hold_current_runtime_release_deg": float(
                        self.gusty_hold_current_runtime_release_deg
                    ),
                    "gusty_hold_current_runtime_released": int(
                        self._gusty_hold_current_runtime_released
                    ),
                    "gusty_hold_current_runtime_release_reason": str(
                        self._gusty_hold_current_runtime_release_reason
                    ),
                    "gusty_hold_current_runtime_release_count": int(
                        self._gusty_hold_current_runtime_release_count
                    ),
                    "gusty_hold_current_debt_guard_enabled": int(
                        self.gusty_hold_current_debt_guard_enabled
                    ),
                    "gusty_hold_current_debt_guard_active": int(
                        self._gusty_hold_current_debt_guard_active
                    ),
                    "gusty_hold_current_debt_guard_reason": str(
                        self._gusty_hold_current_debt_guard_reason
                    ),
                    "gusty_hold_current_debt_guard_target_motion_kg_s": float(
                        self._gusty_hold_current_debt_guard_target_motion_kg_s
                    ),
                    "gusty_hold_current_debt_guard_target_err_kg": float(
                        self._gusty_hold_current_debt_guard_target_err_kg
                    ),
                    "gusty_hold_current_day_budget_enabled": int(
                        self.gusty_hold_current_day_budget_enabled
                    ),
                    "gusty_hold_current_day_budget_max_buckets": int(
                        self.gusty_hold_current_day_budget_max_buckets
                    ),
                    "gusty_hold_current_day_budget_used_buckets": int(
                        self._gusty_hold_current_day_budget_used_buckets
                    ),
                    "gusty_hold_current_day_budget_remaining_buckets": int(
                        self._gusty_hold_current_day_budget_remaining_buckets
                    ),
                    "gusty_hold_current_day_budget_blocked": int(
                        self._gusty_hold_current_day_budget_blocked
                    ),
                    "gusty_hold_current_day_budget_reason": str(
                        self._gusty_hold_current_day_budget_reason
                    ),
                    "forecast_advised_economy_enabled": int(
                        self.forecast_advised_economy_enabled
                    ),
                    "forecast_advised_economy_candidate": int(
                        self._forecast_advised_economy_candidate
                    ),
                    "forecast_advised_economy_boundary_veto": int(
                        self._forecast_advised_economy_boundary_veto
                    ),
                    "forecast_advised_economy_headroom_deg": float(
                        self._forecast_advised_economy_headroom_deg
                    ),
                    "forecast_advised_economy_near_rise_norm": float(
                        self._forecast_advised_economy_near_rise_norm
                    ),
                    "forecast_advised_economy_posture_release": int(
                        self._forecast_advised_economy_posture_release
                    ),
                    "forecast_advised_economy_max_axis_deg": float(
                        self._forecast_advised_economy_max_axis_deg
                    ),
                    "forecast_advised_economy_min_axis_deg": float(
                        self.forecast_advised_economy_min_axis_deg
                    ),
                    "forecast_advised_economy_min_pressure_norm": float(
                        self.forecast_advised_economy_min_pressure_norm
                    ),
                    "forecast_advised_economy_posture_release_deg": float(
                        self.forecast_advised_economy_posture_release_deg
                    ),
                    "forecast_advised_economy_release_cooldown_until_s": float(
                        self._forecast_advised_economy_release_cooldown_until_s
                    ),
                    "forecast_advised_economy_latched_active": int(
                        self._forecast_advised_economy_latched_active
                    ),
                    "forecast_advised_economy_latched_until_s": float(
                        self._forecast_advised_economy_latched_until_s
                    ),
                    "forecast_advised_economy_speed_range_ms": float(
                        self._forecast_speed_range_ms
                    ),
                    "forecast_advised_economy_speed_range_available": int(
                        self._forecast_speed_range_available
                    ),
                    "forecast_advised_economy_speed_range_reason": str(
                        self._forecast_speed_range_reason
                    ),
                    "forecast_advised_economy_speed_range_window_n": int(
                        self._forecast_speed_range_window_n
                    ),
                    "forecast_advised_economy_max_speed_range_ms": float(
                        self.forecast_advised_economy_max_speed_range_ms
                    ),
                    "forecast_advised_economy_max_dir_shift_deg": float(
                        self.forecast_advised_economy_max_dir_shift_deg
                    ),
                    "forecast_advised_economy_confirm_buckets": int(
                        self.forecast_advised_economy_confirm_buckets
                    ),
                    "forecast_advised_economy_hold_confirm_buckets": int(
                        self.forecast_advised_economy_hold_confirm_buckets
                    ),
                    "forecast_advised_economy_clear_streak": int(
                        self._forecast_advised_economy_clear_streak
                    ),
                    "forecast_advised_economy_confirmed": int(
                        self._forecast_advised_economy_confirmed
                    ),
                    "forecast_advised_current_ws_range_ms": float(
                        self._forecast_advised_current_ws_range_ms
                    ),
                    "forecast_advised_current_dir_shift_deg": float(
                        self._forecast_advised_current_dir_shift_deg
                    ),
                    "forecast_advised_current_history_n": int(
                        self._forecast_advised_current_history_n
                    ),
                    "forecast_advised_current_stability_veto": int(
                        self._forecast_advised_current_stability_veto
                    ),
                    "forecast_advised_current_ws_range_limit_ms": float(
                        self.forecast_advised_economy_current_ws_range_ms
                    ),
                    "forecast_advised_current_dir_shift_limit_deg": float(
                        self.forecast_advised_economy_current_dir_shift_deg
                    ),
                    "forecast_advised_current_history_s": float(
                        self.forecast_advised_economy_current_history_s
                    ),
                    "forecast_advised_economy_mild_max_speed_range_ms": float(
                        self.forecast_advised_economy_mild_max_speed_range_ms
                    ),
                    "forecast_advised_high_stable_clean_required": int(
                        self.forecast_advised_economy_high_stable_clean_required
                    ),
                    "forecast_advised_high_stable_clean_gate": int(
                        self._forecast_advised_high_stable_clean_gate
                    ),
                    "forecast_advised_high_stable_clean_reason": str(
                        self._forecast_advised_high_stable_clean_reason
                    ),
                    "forecast_advised_hold_current_override": int(
                        self._forecast_advised_hold_current_override
                    ),
                    "forecast_advised_runaway_release": int(
                        self._forecast_advised_runaway_release
                    ),
                    "forecast_advised_runaway_pump_rate_m3_min": float(
                        self._forecast_advised_runaway_pump_rate_m3_min
                    ),
                    "forecast_advised_runaway_backlog_kg": float(
                        self._forecast_advised_runaway_backlog_kg
                    ),
                    "forecast_advised_shadow_opportunity": int(
                        self._forecast_advised_shadow_opportunity
                    ),
                    "forecast_advised_shadow_latch_delta": int(
                        self._forecast_advised_shadow_latch_delta
                    ),
                    "forecast_advised_shadow_target_motion_kg_s": float(
                        self._forecast_advised_shadow_target_motion_kg_s
                    ),
                    "forecast_advised_shadow_pump_rate_m3_min": float(
                        self._forecast_advised_shadow_pump_rate_m3_min
                    ),
                    "forecast_advised_shadow_backlog_kg": float(
                        self._forecast_advised_shadow_backlog_kg
                    ),
                    "forecast_advised_shadow_latched_active": int(
                        self._forecast_advised_shadow_latched_active
                    ),
                    "forecast_advised_economy_min_peak_drop_ms": float(
                        self.forecast_advised_economy_min_peak_drop_ms
                    ),
                    "forecast_pause_resume_max_target_err_kg": float(
                        self.forecast_pause_resume_max_target_err_kg
                    ),
                    "forecast_pause_resume_guard_active": int(
                        self._forecast_pause_resume_guard_active
                    ),
                    "forecast_pause_resume_guard_reason": str(
                        self._forecast_pause_resume_guard_reason
                    ),
                    "forecast_pause_resume_target_err_kg": float(
                        self._forecast_pause_resume_target_err_kg
                    ),
                    "forecast_advised_economy_reason": str(
                        self._forecast_advised_economy_reason
                    ),
                    "pressure_block0_norm": float(self._effective_pressure_block_norms[0]),
                    "pressure_block1_norm": float(self._effective_pressure_block_norms[1]),
                    "pressure_block2_norm": float(self._effective_pressure_block_norms[2]),
                    "pressure_block02_dot": float(
                        np.dot(
                            self._effective_pressure_block_vecs[0],
                            self._effective_pressure_block_vecs[2],
                        )
                    ),
                    "raw_pressure_block0_norm": float(self._raw_pressure_block_norms[0]),
                    "raw_pressure_block1_norm": float(self._raw_pressure_block_norms[1]),
                    "raw_pressure_block2_norm": float(self._raw_pressure_block_norms[2]),
                    "raw_pressure_block02_dot": float(
                        np.dot(self._raw_pressure_block_vecs[0], self._raw_pressure_block_vecs[2])
                    ),
                    "raw_pressure_block0_pitch": float(
                        self._raw_pressure_block_vecs[0][0]
                        if len(self._raw_pressure_block_vecs) > 0
                        and np.asarray(self._raw_pressure_block_vecs[0]).size > 0
                        else 0.0
                    ),
                    "raw_pressure_block0_roll": float(
                        self._raw_pressure_block_vecs[0][1]
                        if len(self._raw_pressure_block_vecs) > 0
                        and np.asarray(self._raw_pressure_block_vecs[0]).size > 1
                        else 0.0
                    ),
                    "raw_pressure_block1_pitch": float(
                        self._raw_pressure_block_vecs[1][0]
                        if len(self._raw_pressure_block_vecs) > 1
                        and np.asarray(self._raw_pressure_block_vecs[1]).size > 0
                        else 0.0
                    ),
                    "raw_pressure_block1_roll": float(
                        self._raw_pressure_block_vecs[1][1]
                        if len(self._raw_pressure_block_vecs) > 1
                        and np.asarray(self._raw_pressure_block_vecs[1]).size > 1
                        else 0.0
                    ),
                    "raw_pressure_block2_pitch": float(
                        self._raw_pressure_block_vecs[2][0]
                        if len(self._raw_pressure_block_vecs) > 2
                        and np.asarray(self._raw_pressure_block_vecs[2]).size > 0
                        else 0.0
                    ),
                    "raw_pressure_block2_roll": float(
                        self._raw_pressure_block_vecs[2][1]
                        if len(self._raw_pressure_block_vecs) > 2
                        and np.asarray(self._raw_pressure_block_vecs[2]).size > 1
                        else 0.0
                    ),
                    **self._far_horizon_log_fields(),
                    **self._forecast_speed_shape_log_fields(),
                    **self._psc_v2_short_log_fields(),
                    **frozen_overlay_record_defaults(),
                    **self._trusted_event_gate_log_fields(),
                    "event_risk_prob_0_20m": float(self._event_risk_probs[0]),
                    "event_risk_prob_20_40m": float(self._event_risk_probs[1]),
                    "event_risk_prob_40_60m": float(self._event_risk_probs[2]),
                    "event_risk_scale_0_20m": float(self._event_risk_block_scales[0]),
                    "event_risk_scale_20_40m": float(self._event_risk_block_scales[1]),
                    "event_risk_scale_40_60m": float(self._event_risk_block_scales[2]),
                    "relief_medium_cap_enabled": int(self.relief_medium_cap_enabled),
                    "relief_medium_cap_active": int(self._relief_medium_cap_active),
                    "relief_medium_cap_reason": self._relief_medium_cap_reason,
                    "relief_medium_cap_margin": float(self._relief_medium_cap_relief_margin),
                    **frozen_hold_comfort_record_defaults(),
                    "hold_future_posture_evidence_active": int(
                        self._hold_future_posture_evidence_active
                    ),
                    "hold_future_posture_evidence_reason": (
                        self._hold_future_posture_evidence_reason
                    ),
                    **frozen_recovery_mode_record_defaults(),
                })
        elif self.prediction_primary_enabled:
            self._maybe_update_primary_target_between_buckets(
                plant_info_now,
                float(current_time),
            )
            release_gusty, gusty_runtime_reason = (
                self._gusty_hold_current_runtime_release_needed(
                    plant_info_now,
                    float(current_time),
                )
            )
            if release_gusty:
                self._gusty_hold_current_latched_active = False
                self._gusty_hold_current_runtime_released = True
                self._gusty_hold_current_runtime_release_reason = str(
                    gusty_runtime_reason
                )
                self._gusty_hold_current_runtime_release_until_s = (
                    float(current_time)
                    + float(self.gusty_hold_current_runtime_release_cooldown_s)
                )
                self._gusty_hold_current_runtime_release_count += 1
                self._primary_refresh_owner_pending = (
                    f"gusty_runtime_release:{gusty_runtime_reason}"
                )
                self._release_primary_target_to_pi(plant_info_now, float(current_time))
            release_fallback_risk, fallback_risk_reason = (
                self._fallback_risk_active_release_needed(plant_info_now)
            )
            if release_fallback_risk:
                self._primary_refresh_owner_pending = (
                    f"fallback_risk_active_release:{fallback_risk_reason}"
                )
                self._release_primary_target_to_pi(plant_info_now, float(current_time))

        if (
            self.forecast_advised_economy_enabled
            and self.forecast_advised_economy_suppression_only
            and self._primary_target_initialized
        ):
            self._primary_refresh_owner_pending = "forecast_advised_suppression_only_pi_release"
            self._release_primary_target_to_pi(
                self._plant_info_from(plant_info_prev),
                float(current_time),
            )

        forecast_advised_fail_closed = bool(
            self.forecast_advised_economy_enabled
            and not self._forecast_advised_economy_candidate
        )
        if forecast_advised_fail_closed and self._primary_target_initialized:
            self._release_primary_target_to_pi(
                self._plant_info_from(plant_info_prev),
                float(current_time),
            )
        pitch_bias, roll_bias = self._shaped_bias(current_time)
        if (
            not self.setpoint_channel_enabled
            or (self.prediction_primary_enabled and self.prediction_primary_disable_setpoint)
            or forecast_advised_fail_closed
        ):
            pitch_bias = 0.0
            roll_bias = 0.0
        if self.prediction_primary_enabled:
            self._reactive_floor_post_exit_update(
                state,
                self._plant_info_from(plant_info_prev),
                float(current_time),
            )
        mass_ff_kg = (
            np.zeros(3, dtype=float)
            if forecast_advised_fail_closed
            else self._shaped_mass_ff_kg(current_time)
        )
        primary_channel_enabled = bool(
            self.prediction_primary_enabled
            and self._primary_target_initialized
            and not forecast_advised_fail_closed
        )
        primary_active = int(
            primary_channel_enabled
            and float(np.mean(np.abs(self._primary_target_kg - self._primary_anchor_masses_kg))) > 1.0
        )
        return {
            "pitch_bias_deg": pitch_bias,
            "roll_bias_deg": roll_bias,
            "preview_pitch_bias_deg": pitch_bias,
            "preview_roll_bias_deg": roll_bias,
            "preview_mass_ff_kg": [float(x) for x in mass_ff_kg],
            "preview_ff_channel_enabled": int(self.ff_channel_enabled),
            "preview_setpoint_channel_enabled": int(self.setpoint_channel_enabled),
            "preview_primary_enabled": int(primary_channel_enabled),
            "preview_primary_active": int(primary_active),
            "preview_primary_target_kg": [float(x) for x in self._primary_target_kg],
            "preview_primary_delta_kg": [float(x) for x in self._primary_delta_kg],
            "preview_primary_action": self._target_action,
            "preview_primary_hold_target_mode": self.primary_hold_target_mode,
            "preview_fallback_risk_active_release_enabled": int(
                self.fallback_risk_active_release_enabled
            ),
            "preview_fallback_risk_active_release_active": int(
                self._fallback_risk_active_release_active
            ),
            "preview_fallback_risk_active_release_reason": (
                self._fallback_risk_active_release_reason
            ),
            "preview_fallback_risk_active_release_pitch_abs_deg": float(
                self._fallback_risk_active_release_pitch_abs_deg
            ),
            "preview_fallback_risk_active_release_roll_abs_deg": float(
                self._fallback_risk_active_release_roll_abs_deg
            ),
            "preview_fallback_risk_active_release_pitch_worsening_deg_s": float(
                self._fallback_risk_active_release_pitch_worsening_deg_s
            ),
            "preview_fallback_risk_active_release_roll_worsening_deg_s": float(
                self._fallback_risk_active_release_roll_worsening_deg_s
            ),
            "preview_fallback_risk_active_release_pump_rate_m3_min": float(
                self._fallback_risk_active_release_pump_rate_m3_min
            ),
            "preview_fallback_risk_active_release_backlog_kg": float(
                self._fallback_risk_active_release_backlog_kg
            ),
            "preview_fallback_risk_active_release_count": int(
                self._fallback_risk_active_release_count
            ),
            "preview_primary_event_reset": int(self._primary_target_refreshed),
            "preview_primary_posture_action_refresh": int(
                self._posture_action_refresh_active
            ),
            "preview_primary_stall_refresh": int(self._primary_stall_refresh_active),
            "preview_primary_stall_refresh_reason": self._primary_stall_refresh_reason,
            "preview_primary_stall_axis_bias": int(self._primary_stall_axis_bias_active),
            "preview_primary_stall_target_err_mean_kg": float(
                self._primary_stall_target_err_mean_kg
            ),
            "preview_primary_stall_pump_rate_m3_min": float(
                self._primary_stall_pump_rate_m3_min
            ),
            **frozen_active_intent_record_defaults("preview_"),
            "preview_active_effectiveness_refresh_enabled": int(
                self.active_effectiveness_refresh_enabled
            ),
            "preview_active_effectiveness_refresh_active": int(
                self._active_effectiveness_refresh_active
            ),
            "preview_active_effectiveness_refresh_reason": (
                self._active_effectiveness_refresh_reason
            ),
            "preview_sustained_active_recompute_enabled": int(
                self.sustained_active_recompute_enabled
            ),
            "preview_sustained_active_recompute_active": int(
                self._sustained_active_recompute_active
            ),
            "preview_sustained_active_recompute_reason": (
                self._sustained_active_recompute_reason
            ),
            **self._stale_active_target_log_fields(prefix="preview_"),
            "preview_active_effectiveness_repeated_active_buckets": int(
                self._active_effectiveness_repeated_active_buckets
            ),
            "preview_active_effectiveness_pitch_abs_deg": float(
                self._active_effectiveness_pitch_abs_deg
            ),
            "preview_active_effectiveness_roll_abs_deg": float(
                self._active_effectiveness_roll_abs_deg
            ),
            "preview_active_effectiveness_attitude_abs_deg": float(
                self._active_effectiveness_attitude_abs_deg
            ),
            "preview_active_effectiveness_current_response_deg": float(
                self._active_effectiveness_current_response_deg
            ),
            "preview_active_effectiveness_future_relief_status": (
                self._active_effectiveness_future_relief_status
            ),
            "preview_active_effectiveness_target_err_mean_kg": float(
                self._active_effectiveness_target_err_mean_kg
            ),
            "preview_active_effectiveness_pump_rate_m3_min": float(
                self._active_effectiveness_pump_rate_m3_min
            ),
            "preview_active_effectiveness_target_age_s": float(
                self._active_effectiveness_target_age_s
            ),
            "preview_active_effectiveness_pressure_max_norm": float(
                self._active_effectiveness_pressure_max_norm
            ),
            "preview_active_effectiveness_target_ready": int(
                self._active_effectiveness_target_ready
            ),
            "preview_active_effectiveness_pump_idle": int(
                self._active_effectiveness_pump_idle
            ),
            "preview_active_effectiveness_proposal_delta_mean_kg": float(
                self._active_effectiveness_proposal_delta_mean_kg
            ),
            "preview_active_effectiveness_proposal_delta_m3": float(
                self._active_effectiveness_proposal_delta_m3
            ),
            "preview_active_effectiveness_budget_used_m3": float(
                self._active_effectiveness_budget_used_m3
            ),
            "preview_active_effectiveness_lowrisk_safe": int(
                self._active_effectiveness_lowrisk_safe
            ),
            "preview_active_effectiveness_fallback_dominated": int(
                self._active_effectiveness_fallback_dominated
            ),
            "preview_primary_target_refreshed": int(self._primary_target_refreshed),
            "preview_primary_target_reused": int(self._primary_target_reused),
            "preview_primary_target_resumed": int(self._primary_target_resumed),
            "preview_primary_refresh_owner": self._primary_refresh_owner,
            "preview_primary_target_age_s": max(0.0, float(current_time) - float(self._primary_target_last_reset_s)),
            **self._reactive_floor_log_fields("preview_"),
            **self._oracle_preemptive_log_fields("preview_"),
            **self._h120_oracle_probe_log_fields("preview_"),
            **self._h120_scheduler_log_fields("preview_"),
            **self._h120_axis_micro_log_fields("preview_"),
            **self._relief_economy_log_fields("preview_"),
            **self._relief_envelope_log_fields("preview_"),
            **self._economy_budget_log_fields("preview_"),
            **self._h120_floor_shaping_log_fields("preview_"),
            **self._h120_pareto_mode_selector_log_fields("preview_"),
            "preview_forced_prefix_active": int(self._forced_prefix_active),
            "preview_forced_prefix_mode": self._forced_prefix_mode_active,
            "preview_forced_action_label": self._forced_action_label,
            "preview_forced_pitch_deg": float(self._forced_pitch_deg),
            "preview_forced_roll_deg": float(self._forced_roll_deg),
            "preview_forced_vector_mode": self._forced_vector_mode,
            "preview_forced_source_label": self._forced_source_label,
            "preview_forced_target_update_active": int(
                self._forced_prefix_target_update_active
            ),
            "planner_action_pitch_deg": (
                float(self.records[-1].get("planner_action_pitch_deg", 0.0))
                if self.records
                else 0.0
            ),
            "planner_action_roll_deg": (
                float(self.records[-1].get("planner_action_roll_deg", 0.0))
                if self.records
                else 0.0
            ),
            "preview_forecast_has_future": int(bool(self._forecast_has_future)),
            "preview_forecast_event_probs_available": int(
                bool(self._forecast_event_probs_available)
            ),
            "preview_forecast_control_trust_gate_enabled": int(
                self.forecast_control_trust_gate_enabled
            ),
            "preview_forecast_control_trust_ok": int(
                self._forecast_control_trust_ok
            ),
            "preview_forecast_control_trust_reason": str(
                self._forecast_control_trust_reason
            ),
            "preview_forecast_control_trust_purpose": str(
                self._forecast_control_trust_purpose
            ),
            "preview_forecast_control_trust_pressure_max": float(
                self._forecast_control_trust_pressure_max
            ),
            "preview_forecast_control_trust_near_rise": float(
                self._forecast_control_trust_near_rise
            ),
            "preview_forecast_control_trust_dot02": float(
                self._forecast_control_trust_dot02
            ),
            "preview_forecast_control_trust_event_prob_max": float(
                self._forecast_control_trust_event_prob_max
            ),
            "preview_forecast_control_trust_speed_range_ms": float(
                self._forecast_control_trust_speed_range_ms
            ),
            "preview_forecast_control_trust_dir_shift_deg": float(
                self._forecast_control_trust_dir_shift_deg
            ),
            "preview_forecast_control_trust_current_ws_range_limit_ms": float(
                self._forecast_control_trust_current_ws_range_limit_ms
            ),
            "preview_forecast_control_trust_current_dir_shift_limit_deg": float(
                self._forecast_control_trust_current_dir_shift_limit_deg
            ),
            "preview_forecast_control_trust_suppression_ok": int(
                self._forecast_control_trust_by_purpose["suppression"][0]
            ),
            "preview_forecast_control_trust_suppression_reason": str(
                self._forecast_control_trust_by_purpose["suppression"][1]
            ),
            "preview_forecast_control_trust_direction_reversal_hold_ok": int(
                self._forecast_control_trust_by_purpose[
                    "direction_reversal_hold"
                ][0]
            ),
            "preview_forecast_control_trust_direction_reversal_hold_reason": str(
                self._forecast_control_trust_by_purpose[
                    "direction_reversal_hold"
                ][1]
            ),
            "preview_forecast_control_trust_gusty_hold_current_ok": int(
                self._forecast_control_trust_by_purpose[
                    "gusty_hold_current"
                ][0]
            ),
            "preview_forecast_control_trust_gusty_hold_current_reason": str(
                self._forecast_control_trust_by_purpose[
                    "gusty_hold_current"
                ][1]
            ),
            **self._forecast_pressure_trust_log_fields("preview_"),
            "preview_posture_hold_barrier_triggered": (
                int(self.records[-1].get("posture_hold_barrier_triggered", 0))
                if self.records
                else 0
            ),
            "preview_posture_hold_barrier_cost": (
                float(self.records[-1].get("posture_hold_barrier_cost", 0.0))
                if self.records
                else 0.0
            ),
            "preview_posture_hold_norm": (
                float(self.records[-1].get("posture_hold_norm", 0.0))
                if self.records
                else 0.0
            ),
            "preview_posture_hold_forecast_credit": (
                float(self.records[-1].get("posture_hold_forecast_credit", 0.0))
                if self.records
                else 0.0
            ),
            "preview_hold_relief_debt_enabled": int(self.hold_relief_debt_enabled),
            "preview_hold_relief_debt_applied": int(self._hold_relief_debt_applied),
            "preview_hold_relief_debt_planner_level": float(
                self._hold_relief_debt_planner_level
            ),
            "preview_hold_relief_debt_level": float(self._hold_relief_debt_level),
            "preview_hold_relief_debt_cost": (
                float(self.records[-1].get("hold_relief_debt_cost", 0.0))
                if self.records
                else 0.0
            ),
            "preview_hold_relief_debt_reason": self._hold_relief_debt_reason,
            "preview_hold_relief_debt_wait_reason": self._hold_relief_debt_wait_reason,
            "preview_hold_relief_debt_pending": int(self._hold_relief_debt_pending),
            "preview_hold_relief_debt_pending_reason": (
                self._hold_relief_debt_pending_reason
            ),
            "preview_hold_relief_debt_current_response_deg": float(
                self._hold_relief_debt_current_response_deg
            ),
            "preview_hold_relief_debt_lowrisk_safe": int(
                self._hold_relief_debt_lowrisk_safe
            ),
            "preview_hold_relief_debt_fallback_dominated": int(
                self._hold_relief_debt_fallback_dominated
            ),
            "preview_posture_hold_action_posture_directed": (
                int(self.records[-1].get("posture_hold_action_posture_directed", 0))
                if self.records
                else 0
            ),
            "preview_posture_hold_barrier_reason": (
                str(self.records[-1].get("posture_hold_barrier_reason", ""))
                if self.records
                else ""
            ),
            **self._forecast_advised_pump_suppression_preview_fields(),
            "preview_gusty_oscillation_enabled": int(self.gusty_oscillation_enabled),
            "preview_gusty_oscillation_gate_active": int(
                self._gusty_oscillation_gate_active
            ),
            "preview_gusty_oscillation_gate_reason": str(
                self._gusty_oscillation_gate_reason
            ),
            "preview_gusty_oscillation_min_target_age_s": float(
                self.gusty_oscillation_min_target_age_s
            ),
            "preview_gusty_oscillation_headroom_scaled": int(
                self.gusty_oscillation_headroom_scaled
            ),
            "preview_gusty_oscillation_target_scale": float(
                self.gusty_oscillation_target_scale
            ),
            "preview_gusty_oscillation_last_scale": float(
                self._gusty_oscillation_last_scale
            ),
            "preview_gusty_oscillation_short_gate_enabled": int(
                self.gusty_oscillation_short_gate_enabled
            ),
            "preview_gusty_short_blend_applied": int(
                self._gusty_short_blend_applied
            ),
            "preview_gusty_short_target_delta_before_kg": float(
                self._gusty_short_target_delta_before_kg
            ),
            "preview_gusty_short_target_delta_after_kg": float(
                self._gusty_short_target_delta_after_kg
            ),
            "preview_gusty_hold_current_gate_mode": str(
                self.gusty_hold_current_gate_mode
            ),
            "preview_gusty_hold_current_isolated_primary_enabled": int(
                self.gusty_hold_current_isolated_primary_enabled
            ),
            "preview_gusty_hold_current_gate_evaluated": int(
                self._gusty_hold_current_gate_evaluated
            ),
            "preview_gusty_hold_current_allowed": int(
                self._gusty_hold_current_allowed
            ),
            "preview_gusty_hold_current_blocked": int(
                self._gusty_hold_current_blocked
            ),
            "preview_gusty_hold_current_reason": str(
                self._gusty_hold_current_reason
            ),
            "preview_direction_reversal_hold_current_gate_mode": str(
                self.direction_reversal_hold_current_gate_mode
            ),
            "preview_direction_reversal_hold_current_gate_evaluated": int(
                self._direction_reversal_hold_current_gate_evaluated
            ),
            "preview_direction_reversal_hold_current_allowed": int(
                self._direction_reversal_hold_current_allowed
            ),
            "preview_direction_reversal_hold_current_blocked": int(
                self._direction_reversal_hold_current_blocked
            ),
            "preview_direction_reversal_hold_current_reason": str(
                self._direction_reversal_hold_current_reason
            ),
            "preview_gusty_hold_current_latched_active": int(
                self._gusty_hold_current_latched_active
            ),
            "preview_gusty_hold_current_runtime_release_enabled": int(
                self.gusty_hold_current_runtime_release_enabled
            ),
            "preview_gusty_hold_current_runtime_release_deg": float(
                self.gusty_hold_current_runtime_release_deg
            ),
            "preview_gusty_hold_current_runtime_released": int(
                self._gusty_hold_current_runtime_released
            ),
            "preview_gusty_hold_current_runtime_release_reason": str(
                self._gusty_hold_current_runtime_release_reason
            ),
            "preview_gusty_hold_current_runtime_release_count": int(
                self._gusty_hold_current_runtime_release_count
            ),
            "preview_gusty_hold_current_debt_guard_enabled": int(
                self.gusty_hold_current_debt_guard_enabled
            ),
            "preview_gusty_hold_current_debt_guard_active": int(
                self._gusty_hold_current_debt_guard_active
            ),
            "preview_gusty_hold_current_debt_guard_reason": str(
                self._gusty_hold_current_debt_guard_reason
            ),
            "preview_gusty_hold_current_debt_guard_target_motion_kg_s": float(
                self._gusty_hold_current_debt_guard_target_motion_kg_s
            ),
            "preview_gusty_hold_current_debt_guard_target_err_kg": float(
                self._gusty_hold_current_debt_guard_target_err_kg
            ),
            "preview_gusty_hold_current_day_budget_enabled": int(
                self.gusty_hold_current_day_budget_enabled
            ),
            "preview_gusty_hold_current_day_budget_max_buckets": int(
                self.gusty_hold_current_day_budget_max_buckets
            ),
            "preview_gusty_hold_current_day_budget_used_buckets": int(
                self._gusty_hold_current_day_budget_used_buckets
            ),
            "preview_gusty_hold_current_day_budget_remaining_buckets": int(
                self._gusty_hold_current_day_budget_remaining_buckets
            ),
            "preview_gusty_hold_current_day_budget_blocked": int(
                self._gusty_hold_current_day_budget_blocked
            ),
            "preview_gusty_hold_current_day_budget_reason": str(
                self._gusty_hold_current_day_budget_reason
            ),
            "preview_forecast_advised_economy_enabled": int(
                self.forecast_advised_economy_enabled
            ),
            "preview_forecast_advised_economy_candidate": int(
                self._forecast_advised_economy_candidate
            ),
            "preview_forecast_advised_economy_boundary_veto": int(
                self._forecast_advised_economy_boundary_veto
            ),
            "preview_forecast_advised_economy_headroom_deg": float(
                self._forecast_advised_economy_headroom_deg
            ),
            "preview_forecast_advised_economy_near_rise_norm": float(
                self._forecast_advised_economy_near_rise_norm
            ),
            "preview_forecast_advised_economy_posture_release": int(
                self._forecast_advised_economy_posture_release
            ),
            "preview_forecast_advised_economy_max_axis_deg": float(
                self._forecast_advised_economy_max_axis_deg
            ),
            "preview_forecast_advised_economy_min_axis_deg": float(
                self.forecast_advised_economy_min_axis_deg
            ),
            "preview_forecast_advised_economy_min_pressure_norm": float(
                self.forecast_advised_economy_min_pressure_norm
            ),
            "preview_forecast_advised_economy_posture_release_deg": float(
                self.forecast_advised_economy_posture_release_deg
            ),
            "preview_forecast_advised_economy_release_cooldown_until_s": float(
                self._forecast_advised_economy_release_cooldown_until_s
            ),
            "preview_forecast_advised_economy_latched_active": int(
                self._forecast_advised_economy_latched_active
            ),
            "preview_forecast_advised_economy_latched_until_s": float(
                self._forecast_advised_economy_latched_until_s
            ),
            "preview_forecast_advised_economy_speed_range_ms": float(
                self._forecast_speed_range_ms
            ),
            "preview_forecast_advised_economy_speed_range_available": int(
                self._forecast_speed_range_available
            ),
            "preview_forecast_advised_economy_speed_range_reason": str(
                self._forecast_speed_range_reason
            ),
            "preview_forecast_advised_economy_speed_range_window_n": int(
                self._forecast_speed_range_window_n
            ),
            "preview_forecast_advised_economy_max_speed_range_ms": float(
                self.forecast_advised_economy_max_speed_range_ms
            ),
            "preview_forecast_advised_economy_max_dir_shift_deg": float(
                self.forecast_advised_economy_max_dir_shift_deg
            ),
            "preview_forecast_advised_economy_confirm_buckets": int(
                self.forecast_advised_economy_confirm_buckets
            ),
            "preview_forecast_advised_economy_hold_confirm_buckets": int(
                self.forecast_advised_economy_hold_confirm_buckets
            ),
            "preview_forecast_advised_economy_clear_streak": int(
                self._forecast_advised_economy_clear_streak
            ),
            "preview_forecast_advised_economy_confirmed": int(
                self._forecast_advised_economy_confirmed
            ),
            "preview_forecast_advised_current_ws_range_ms": float(
                self._forecast_advised_current_ws_range_ms
            ),
            "preview_forecast_advised_current_dir_shift_deg": float(
                self._forecast_advised_current_dir_shift_deg
            ),
            "preview_forecast_advised_current_history_n": int(
                self._forecast_advised_current_history_n
            ),
            "preview_forecast_advised_current_stability_veto": int(
                self._forecast_advised_current_stability_veto
            ),
            "preview_forecast_advised_current_ws_range_limit_ms": float(
                self.forecast_advised_economy_current_ws_range_ms
            ),
            "preview_forecast_advised_current_dir_shift_limit_deg": float(
                self.forecast_advised_economy_current_dir_shift_deg
            ),
            "preview_forecast_advised_current_history_s": float(
                self.forecast_advised_economy_current_history_s
            ),
            "preview_forecast_advised_economy_mild_max_speed_range_ms": float(
                self.forecast_advised_economy_mild_max_speed_range_ms
            ),
            "preview_forecast_advised_high_stable_clean_required": int(
                self.forecast_advised_economy_high_stable_clean_required
            ),
            "preview_forecast_advised_high_stable_clean_gate": int(
                self._forecast_advised_high_stable_clean_gate
            ),
            "preview_forecast_advised_high_stable_clean_reason": str(
                self._forecast_advised_high_stable_clean_reason
            ),
            "preview_forecast_advised_hold_current_override": int(
                self._forecast_advised_hold_current_override
            ),
            "preview_forecast_advised_runaway_release": int(
                self._forecast_advised_runaway_release
            ),
            "preview_forecast_advised_runaway_pump_rate_m3_min": float(
                self._forecast_advised_runaway_pump_rate_m3_min
            ),
            "preview_forecast_advised_runaway_backlog_kg": float(
                self._forecast_advised_runaway_backlog_kg
            ),
            "preview_forecast_advised_shadow_opportunity": int(
                self._forecast_advised_shadow_opportunity
            ),
            "preview_forecast_advised_shadow_latch_delta": int(
                self._forecast_advised_shadow_latch_delta
            ),
            "preview_forecast_advised_shadow_target_motion_kg_s": float(
                self._forecast_advised_shadow_target_motion_kg_s
            ),
            "preview_forecast_advised_shadow_pump_rate_m3_min": float(
                self._forecast_advised_shadow_pump_rate_m3_min
            ),
            "preview_forecast_advised_shadow_backlog_kg": float(
                self._forecast_advised_shadow_backlog_kg
            ),
            "preview_forecast_advised_shadow_latched_active": int(
                self._forecast_advised_shadow_latched_active
            ),
            "preview_forecast_advised_economy_min_peak_drop_ms": float(
                self.forecast_advised_economy_min_peak_drop_ms
            ),
            "preview_forecast_advised_economy_reason": str(
                self._forecast_advised_economy_reason
            ),
            "preview_pressure_block0_norm": float(self._effective_pressure_block_norms[0]),
            "preview_pressure_block1_norm": float(self._effective_pressure_block_norms[1]),
            "preview_pressure_block2_norm": float(self._effective_pressure_block_norms[2]),
            "preview_pressure_block02_dot": float(
                np.dot(
                    self._effective_pressure_block_vecs[0],
                    self._effective_pressure_block_vecs[2],
                )
            ),
            "preview_raw_pressure_block0_norm": float(self._raw_pressure_block_norms[0]),
            "preview_raw_pressure_block1_norm": float(self._raw_pressure_block_norms[1]),
            "preview_raw_pressure_block2_norm": float(self._raw_pressure_block_norms[2]),
            "preview_raw_pressure_block02_dot": float(
                np.dot(self._raw_pressure_block_vecs[0], self._raw_pressure_block_vecs[2])
            ),
            **self._far_horizon_log_fields("preview_"),
            **self._forecast_speed_shape_log_fields("preview_"),
            **self._psc_v2_short_log_fields("preview_"),
            **frozen_overlay_preview_defaults(),
            **self._trusted_event_gate_log_fields("preview_"),
            "preview_event_risk_prob_0_20m": float(self._event_risk_probs[0]),
            "preview_event_risk_prob_20_40m": float(self._event_risk_probs[1]),
            "preview_event_risk_prob_40_60m": float(self._event_risk_probs[2]),
            "preview_event_risk_scale_0_20m": float(self._event_risk_block_scales[0]),
            "preview_event_risk_scale_20_40m": float(self._event_risk_block_scales[1]),
            "preview_event_risk_scale_40_60m": float(self._event_risk_block_scales[2]),
            "preview_relief_medium_cap_enabled": int(self.relief_medium_cap_enabled),
            "preview_relief_medium_cap_active": int(self._relief_medium_cap_active),
            "preview_relief_medium_cap_reason": self._relief_medium_cap_reason,
            "preview_relief_medium_cap_margin": float(self._relief_medium_cap_relief_margin),
            **frozen_hold_comfort_preview_defaults(),
            "preview_hold_future_posture_evidence_active": int(
                self._hold_future_posture_evidence_active
            ),
            "preview_hold_future_posture_evidence_reason": (
                self._hold_future_posture_evidence_reason
            ),
            "source": f"ballast_planner_{self.objective_mode}_{self.bias_shape}",
            "preview_trim_source": f"ballast_planner_{self.objective_mode}_{self.bias_shape}",
            "preview_gain": float(self.scale),
            "risk_level": "active" if self._target_action != "hold" else "hold",
            "preview_mode": f"ballast_planner_{self.objective_mode}_{self.bias_shape}",
            "first_action": self._target_action,
            **frozen_recovery_mode_record_defaults("preview_"),
        }
