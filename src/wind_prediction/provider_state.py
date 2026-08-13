"""Provider construction, mutable state, and replay input helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from .ballast_planner import (
    ACTIONS,
    PlannerConfig,
    planner_action_vec,
)
from .forecast_adapter import ForecastModelAdapter
from .replay_dataset import Fino1ReplayDataset
from .safety_supervisor import (
    apply_reactive_floor_state_defaults,
)
from .target_lifecycle import (
    apply_primary_target_state_defaults,
)

from .provider_defaults import normalize_provider_options
from .provider_settings import apply_provider_settings


_PLANT_INFO_THREE_VECTOR_DTYPES = {
    "tank_masses": float,
    "target_ballast_mass": float,
    "pump_rate_cmd_m3_min": float,
    "pump_net_rate_m3_min": float,
    "pump_latched": bool,
    "pump_on_elapsed_s": float,
    "pump_off_elapsed_s": float,
    "pump_near_target_s": float,
    "pump_stage_idx": int,
}

_PLANT_INFO_SCALAR_CASTS = {
    "pump_fullspeed_any": int,
    "pump_total_backlog_kg": float,
    "pump_latch_switch_count": int,
    "pump_target_motion_kg_s": float,
    "pump_stage_switch_count": int,
}


def _copy_plant_info_value(value: Any) -> Any:
    return value.copy() if isinstance(value, np.ndarray) else value


def _three_vector_or_none(value: Any, dtype: type) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=dtype).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 3:
        return None
    return array[:3].copy()


class ProviderStateMixin:
    """Internal behavior group for ``BallastPlannerPreviewProvider``."""

    def __init__(
        self,
        replay_dataset: Fino1ReplayDataset,
        start_timestamp: datetime,
        cfg: PlannerConfig,
        block_discounts: list[float],
        plant_info: dict[str, Any] | None = None,
        forecast_adapter: ForecastModelAdapter | None = None,
        **options: Any,
    ) -> None:
        settings = normalize_provider_options(options)
        self.replay_dataset = replay_dataset
        self.start_timestamp = start_timestamp
        self.cfg = cfg
        self.block_discounts = list(block_discounts)
        self.update_interval_s = int(replay_dataset.update_interval_s)
        apply_provider_settings(self, settings, forecast_adapter)
        default_tank_masses = np.array(
            [1108000.0, 1362000.0, 1362000.0],
            dtype=float,
        )
        self._default_plant_info = {
            "tank_masses": default_tank_masses,
            "target_ballast_mass": default_tank_masses.copy(),
            "pump_fullspeed_any": 0,
            "pump_total_backlog_kg": 0.0,
            "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
            "pump_net_rate_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
            "pump_latched": np.array([False, False, False], dtype=bool),
            "pump_on_elapsed_s": np.array([0.0, 0.0, 0.0], dtype=float),
            "pump_off_elapsed_s": np.array([0.0, 0.0, 0.0], dtype=float),
            "pump_near_target_s": np.array([0.0, 0.0, 0.0], dtype=float),
            "pump_stage_idx": np.array([0, 0, 0], dtype=int),
            "pump_latch_switch_count": 0,
            "pump_target_motion_kg_s": 0.0,
            "pump_stage_switch_count": 0,
        }
        if plant_info:
            known_fields = set(_PLANT_INFO_THREE_VECTOR_DTYPES) | set(
                _PLANT_INFO_SCALAR_CASTS
            )
            for name, value in plant_info.items():
                if name not in known_fields:
                    self._default_plant_info[name] = _copy_plant_info_value(value)
            self._default_plant_info = self._plant_info_from(plant_info)
        self.reset()

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
        # Cached planner-frame target (for FF) and zeroed mass FF state.
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        if hasattr(self, "_forecast_deadband_held_target_kg"):
            del self._forecast_deadband_held_target_kg
        apply_primary_target_state_defaults(
            self,
            self._default_plant_info["tank_masses"],
            refresh_owner="reset",
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
        self._gusty_hold_current_day_budget_reason = "reset"
        self._fallback_risk_active_release_active = False
        self._fallback_risk_active_release_reason = "reset"
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
        self._h120_oracle_probe_count = 0
        self._h120_oracle_probe_short_delay_count = 0
        self._h120_oracle_probe_early_stop_veto_count = 0
        self._h120_oracle_probe_preemptive_count = 0
        self._h120_oracle_probe_last_preemptive_s = -1.0e12
        self._reset_h120_oracle_probe_bucket_state("reset")
        # h120 risk scheduler per-case counters + per-episode latch
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
        self._relief_envelope_axis_cap_requested_delta_deg = 0.0
        self._relief_envelope_axis_cap_pursued_delta_deg = 0.0
        self._relief_envelope_axis_cap_held_delta_deg = 0.0
        self._relief_envelope_axis_cap_covered_axes = 0
        self._relief_envelope_axis_cap_mode = "none"
        self._economy_pump_budget_used_m3 = 0.0
        self._economy_pump_budget_last_s: float | None = None
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
        self._forecast_has_future = False
        self._forecast_event_probs_available = False
        self._reset_forecast_control_trust_state("reset")
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

    def _bucket_timestamp(self, current_time: float) -> datetime:
        return self.replay_dataset.simulation_timestamp(self.start_timestamp, current_time)

    def _plant_info_from(self, plant_info_prev: dict | None) -> dict:
        info = {
            name: _copy_plant_info_value(value)
            for name, value in self._default_plant_info.items()
        }
        if not plant_info_prev:
            return info

        copied_current_masses = False
        for name, dtype in _PLANT_INFO_THREE_VECTOR_DTYPES.items():
            if name not in plant_info_prev:
                continue
            array = _three_vector_or_none(plant_info_prev[name], dtype)
            if array is None:
                continue
            info[name] = array
            copied_current_masses = copied_current_masses or name == "tank_masses"

        # Older plant snapshots did not publish an execution target.  Preserve
        # their former hold-current behavior instead of reviving a stale default.
        if copied_current_masses and "target_ballast_mass" not in plant_info_prev:
            info["target_ballast_mass"] = info["tank_masses"].copy()

        for name, cast in _PLANT_INFO_SCALAR_CASTS.items():
            if name not in plant_info_prev:
                continue
            try:
                info[name] = cast(plant_info_prev[name])
            except (TypeError, ValueError):
                continue
        return info

    @staticmethod
    def _attach_posture_info(plant_info: dict[str, Any], state) -> dict[str, Any]:
        info = dict(plant_info)
        if state is None:
            info["posture_vec_deg"] = np.zeros(2, dtype=float)
            info["posture_rate_vec_deg_s"] = np.zeros(2, dtype=float)
            return info
        state_arr = np.asarray(state, dtype=float).reshape(-1)
        if state_arr.size <= 4:
            info["posture_vec_deg"] = np.zeros(2, dtype=float)
            info["posture_rate_vec_deg_s"] = np.zeros(2, dtype=float)
            return info
        # Planner vector order is [pitch, roll].
        info["posture_vec_deg"] = np.array(
            [
                float(np.degrees(state_arr[4])),
                float(np.degrees(state_arr[3])),
            ],
            dtype=float,
        )
        if state_arr.size > 10:
            info["posture_rate_vec_deg_s"] = np.array(
                [
                    float(np.degrees(state_arr[10])),
                    float(np.degrees(state_arr[9])),
                ],
                dtype=float,
            )
        else:
            info["posture_rate_vec_deg_s"] = np.zeros(2, dtype=float)
        return info
