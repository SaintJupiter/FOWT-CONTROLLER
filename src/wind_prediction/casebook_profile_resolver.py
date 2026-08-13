"""Resolve named casebook profiles into one explicit controller configuration.

This module quarantines historical experiment branches from the simulation
runner. New formal studies should use an explicit protocol rather than adding
another named branch here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .casebook_profile_gate import enforce_casebook_profile_gate


@dataclass(frozen=True)
class ProfileResolution:
    control_profile: str
    registry_group: str
    target_shape: dict[str, Any]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def resolve_primary_control_profile(
    *,
    args: Any,
    primary_safety_override: Callable[[str], dict[str, Any]],
    profile_registry_path: Path,
    repo_root: Path,
) -> ProfileResolution:
    explicit_cli_overrides = dict(
        getattr(args, "_explicit_cli_overrides", {}) or {}
    )
    primary_control_profile = str(args.primary_control_profile)
    primary_profile_registry_group = enforce_casebook_profile_gate(
        primary_control_profile,
        allow_isolated_profile=bool(args.allow_isolated_profile),
        registry_path=profile_registry_path,
        repo_root=repo_root,
    )
    primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile == "safe_pump_freeze_v1":
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        primary_target_shape_cfg["safe_pump_freeze_enabled"] = True
        primary_target_shape_cfg["safe_pump_freeze_enter_axis_deg"] = float(
            os.environ.get("FOWT_SAFE_PUMP_FREEZE_ENTER_AXIS_DEG", 2.6)
        )
        primary_target_shape_cfg["safe_pump_freeze_exit_axis_deg"] = float(
            os.environ.get("FOWT_SAFE_PUMP_FREEZE_EXIT_AXIS_DEG", 3.8)
        )
        primary_target_shape_cfg["safe_pump_freeze_min_delta_kg"] = float(
            os.environ.get("FOWT_SAFE_PUMP_FREEZE_MIN_DELTA_KG", 0.0)
        )
    if primary_control_profile in {
        "dc_preserving_deadband_v1",
        "dc_preserving_deadband_engineered_v1",
        "dc_preserving_deadband_engineered_recovery_observer_v1",
        "dc_preserving_deadband_w1_veto_v1",
        "dc_preserving_deadband_conservative_v1",
        "dc_preserving_deadband_w1_pid_veto_v1",
        "dc_preserving_deadband_forecast_adaptive_v1",
        "dc_preserving_deadband_forecast_widen_v1",
    }:
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        conservative_deadband = (
            primary_control_profile == "dc_preserving_deadband_conservative_v1"
        )
        engineered_deadband = (
            primary_control_profile
            in {
                "dc_preserving_deadband_engineered_v1",
                "dc_preserving_deadband_engineered_recovery_observer_v1",
            }
        )
        if (conservative_deadband or engineered_deadband) and str(args.primary_pump_profile) == "raw":
            args.primary_pump_profile = "engineered_minimal"
        if primary_control_profile == "dc_preserving_deadband_engineered_recovery_observer_v1":
            args.recovery_mode = True
        primary_target_shape_cfg["deadband_target_release_enabled"] = (
            os.environ.get("FOWT_DC_ENABLE_TARGET_RELEASE", "1").lower()
            not in {"0", "false", "no"}
        )
        primary_target_shape_cfg["deadband_target_release_pitch_deg"] = float(
            os.environ.get("FOWT_DC_DEADBAND_PITCH_DEG", 1.2 if conservative_deadband else 1.5)
        )
        primary_target_shape_cfg["deadband_target_release_roll_deg"] = float(
            os.environ.get("FOWT_DC_DEADBAND_ROLL_DEG", 1.2 if conservative_deadband else 1.5)
        )
        primary_target_shape_cfg["deadband_target_release_exit_pitch_deg"] = float(
            os.environ.get("FOWT_DC_DEADBAND_EXIT_PITCH_DEG", 1.2 if conservative_deadband else 1.5)
        )
        primary_target_shape_cfg["deadband_target_release_exit_roll_deg"] = float(
            os.environ.get("FOWT_DC_DEADBAND_EXIT_ROLL_DEG", 1.2 if conservative_deadband else 1.5)
        )
        primary_target_shape_cfg["deadband_target_release_near_zero_deg"] = float(
            os.environ.get("FOWT_DC_DEADBAND_NEAR_ZERO_DEG", 0.15)
        )
        primary_target_shape_cfg["deadband_target_release_rate_eps_deg_s"] = float(
            os.environ.get("FOWT_DC_DEADBAND_RATE_EPS_DEG_S", 0.002)
        )
        primary_target_shape_cfg["deadband_target_release_blend"] = float(
            os.environ.get("FOWT_DC_DEADBAND_BLEND", 1.0)
        )
        primary_target_shape_cfg["deadband_target_release_max_delta_kg"] = float(
            os.environ.get("FOWT_DC_DEADBAND_MAX_DELTA_KG", 6000.0)
        )
        primary_target_shape_cfg["deadband_target_release_require_both_axes"] = (
            os.environ.get("FOWT_DC_DEADBAND_REQUIRE_BOTH_AXES", "1").lower()
            not in {"0", "false", "no"}
        )
        primary_target_shape_cfg["deadband_target_release_use_pid_deadband"] = (
            os.environ.get("FOWT_DC_DEADBAND_USE_PID_DEADBAND", "0").lower()
            not in {"0", "false", "no"}
        )
        primary_target_shape_cfg["deadband_target_release_require_zero_motion"] = (
            os.environ.get("FOWT_DC_DEADBAND_REQUIRE_ZERO_MOTION", "1").lower()
            not in {"0", "false", "no"}
        )
        primary_target_shape_cfg["deadband_target_release_allow_moving_to_zero"] = (
            os.environ.get("FOWT_DC_DEADBAND_ALLOW_MOVING_TO_ZERO", "1").lower()
            not in {"0", "false", "no"}
        )
        primary_target_shape_cfg["deadband_target_release_latch_enabled"] = (
            os.environ.get("FOWT_DC_DEADBAND_LATCH_ENABLED", "0").lower()
            not in {"0", "false", "no"}
        )
        primary_target_shape_cfg["deadband_target_release_reset_limiter"] = (
            os.environ.get("FOWT_DC_DEADBAND_RESET_LIMITER", "1").lower()
            not in {"0", "false", "no"}
        )
        if primary_control_profile in {
            "dc_preserving_deadband_w1_veto_v1",
            "dc_preserving_deadband_conservative_v1",
            "dc_preserving_deadband_w1_pid_veto_v1",
            "dc_preserving_deadband_forecast_adaptive_v1",
            "dc_preserving_deadband_forecast_widen_v1",
        }:
            primary_target_shape_cfg["deadband_target_release_forecast_veto_enabled"] = True
            primary_target_shape_cfg["deadband_target_release_veto_dot02_max"] = float(
                os.environ.get("FOWT_DC_W1_VETO_DOT02_MAX", 0.0)
            )
            primary_target_shape_cfg["deadband_target_release_veto_near_rise_norm"] = float(
                os.environ.get("FOWT_DC_W1_VETO_NEAR_RISE_NORM", 0.05)
            )
            primary_target_shape_cfg["deadband_target_release_veto_min_pressure_norm"] = float(
                os.environ.get("FOWT_DC_W1_VETO_MIN_PRESSURE_NORM", 0.45)
            )
        if primary_control_profile == "dc_preserving_deadband_forecast_adaptive_v1":
            # Prediction-timed PID attitude deadband: wide when forecast-calm,
            # contracted to tight tracking when forecast flags near-rise/reversal.
            primary_target_shape_cfg["deadband_forecast_adaptive_enabled"] = True
            primary_target_shape_cfg["deadband_forecast_adaptive_tight_pitch_deg"] = float(
                os.environ.get("FOWT_DC_FC_ADAPTIVE_TIGHT_PITCH_DEG", 0.5)
            )
            primary_target_shape_cfg["deadband_forecast_adaptive_tight_roll_deg"] = float(
                os.environ.get("FOWT_DC_FC_ADAPTIVE_TIGHT_ROLL_DEG", 0.4)
            )
        if primary_control_profile == "dc_preserving_deadband_forecast_widen_v1":
            # Forecast-causal saving: base PID band is tight; widen to a capped value
            # only when the forecast confirms a safe decay and posture has headroom.
            primary_target_shape_cfg["deadband_forecast_widen_enabled"] = True
            primary_target_shape_cfg["deadband_forecast_widen_cap_pitch_deg"] = float(
                os.environ.get("FOWT_DC_FC_WIDEN_CAP_PITCH_DEG", 2.0)
            )
            primary_target_shape_cfg["deadband_forecast_widen_cap_roll_deg"] = float(
                os.environ.get("FOWT_DC_FC_WIDEN_CAP_ROLL_DEG", 2.0)
            )
            primary_target_shape_cfg["deadband_forecast_widen_decay_norm"] = float(
                os.environ.get("FOWT_DC_FC_WIDEN_DECAY_NORM", 0.05)
            )
            primary_target_shape_cfg["deadband_forecast_widen_posture_gate_deg"] = float(
                os.environ.get("FOWT_DC_FC_WIDEN_POSTURE_GATE_DEG", 1.5)
            )
    if primary_control_profile == "dc_target_lowpass_v1":
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        primary_target_shape_cfg["enable_rate_limit"] = True
        primary_target_shape_cfg["rate_limit_m3_min"] = float(
            os.environ.get("FOWT_DC_TARGET_LOWPASS_RATE_M3_MIN", 3.0)
        )
        primary_target_shape_cfg["deadband_target_release_enabled"] = False
    if primary_control_profile == "dc_target_ac_filter_v1":
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        primary_target_shape_cfg["target_ac_filter_enabled"] = True
        primary_target_shape_cfg["target_ac_filter_tau_s"] = float(
            os.environ.get("FOWT_AC_FILTER_TAU_S", 120.0)
        )
        primary_target_shape_cfg["target_ac_filter_reversal_decay_tau_s"] = float(
            os.environ.get("FOWT_AC_FILTER_REVERSAL_DECAY_TAU_S", 300.0)
        )
        primary_target_shape_cfg["target_ac_filter_attenuation"] = float(
            os.environ.get("FOWT_AC_FILTER_ATTENUATION", 0.65)
        )
        primary_target_shape_cfg["target_ac_filter_min_reversal_score"] = float(
            os.environ.get("FOWT_AC_FILTER_MIN_REVERSAL_SCORE", 2.0)
        )
        primary_target_shape_cfg["target_ac_filter_min_delta_kg"] = float(
            os.environ.get("FOWT_AC_FILTER_MIN_DELTA_KG", 500.0)
        )
        primary_target_shape_cfg["target_ac_filter_posture_release_deg"] = float(
            os.environ.get("FOWT_AC_FILTER_POSTURE_RELEASE_DEG", 4.5)
        )
        primary_target_shape_cfg["deadband_target_release_enabled"] = False
    if primary_control_profile in {
        "rawenv_holdpause_barrier_v1",
        "rawenv_holdpause_barrier_reliefcap030_v1",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "rawenv_holdcurrent_barrier_reliefcap_adaptive_v1",
        "rawenv_forecastsafe_deadband_candidate_v1",
        "rawenv_forecastveto_deadband_candidate_v1",
        "rawenv_forecastveto_deadband_fullsync_legacy_v1",
        "rawenv_forecastgated_command_deadband_v1",
        "rawenv_pirelease_barrier_reliefcap_adaptive_v1",
        "rawenv_holdpause_barrier_reliefcap_state_v1",
        "rawenv_forecastpause_barrier_reliefcap_adaptive_v1",
        "regime_auto_pump_saving_v1",
        "psc_4hao_mild_fraction_v1",
        "psc_4hao_debt_aware_v1",
        "psc_4hao_dynamic_refresh_v1",
        "psc_4hao_generalized_gate_v1",
        "psc_4hao_regime_addback_v2_provider_v1",
        "psc_4hao_guarded_v2",
        "psc_4hao_guarded_v3",
        "psc_4hao_regime_gated_v1",
        "psc_4hao_chain_guard_v1",
        "psc_4hao_exec_guard_v1",
        "relief_decay_smart_release_v1",
        "relief_decay_dispatcher_v2",
        "residual_high_economy_v1",
        "residual_high_plateau_v2",
        "residual_high_plateau_strict_v2",
        "neutral_mhs_clean_budget_v1",
        "dual_specialist_pump_saving_v1",
        "residual_high_plateau_suppression_v1",
        "gusty_oscillation_smoothing_v1",
        "gusty_oscillation_budget_auto_v1",
        "direction_reversal_avoidance_v1",
        "safety_floor_v1",
    }:
        args.planner_envelope_raw = True
        args.planner_envelope_barrier = True
        args.planner_envelope_barrier_const = 50.0
        args.primary_hold_target_mode = "pause"
    if primary_control_profile in {
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "rawenv_forecastgated_command_deadband_v1",
    }:
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
    if primary_control_profile in {
        "rawenv_holdcurrent_barrier_reliefcap_adaptive_v1",
        "rawenv_forecastsafe_deadband_candidate_v1",
        "rawenv_forecastveto_deadband_candidate_v1",
        "rawenv_forecastveto_deadband_fullsync_legacy_v1",
    }:
        # Candidate-level deadband experiment: retain the production planner and
        # make its existing hold action release the primary target to current
        # tank masses instead of continuing to chase a paused target.
        args.primary_hold_target_mode = (
            "current"
            if primary_control_profile == "rawenv_holdcurrent_barrier_reliefcap_adaptive_v1"
            else "pause"
        )
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(
            os.environ.get(
                "FOWT_TUNE_GAIN",
                0.40
                if primary_control_profile in {
                    "rawenv_forecastsafe_deadband_candidate_v1",
                    "rawenv_forecastveto_deadband_candidate_v1",
                    "rawenv_forecastveto_deadband_fullsync_legacy_v1",
                }
                else 0.45,
            )
        )
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        if primary_control_profile in {
            "rawenv_forecastsafe_deadband_candidate_v1",
            "rawenv_forecastveto_deadband_candidate_v1",
            "rawenv_forecastveto_deadband_fullsync_legacy_v1",
        }:
            legacy_fullsync_mode = (
                primary_control_profile
                == "rawenv_forecastveto_deadband_fullsync_legacy_v1"
            )
            forecast_veto_mode = (
                primary_control_profile
                in {
                    "rawenv_forecastveto_deadband_candidate_v1",
                    "rawenv_forecastveto_deadband_fullsync_legacy_v1",
                }
            )
            if legacy_fullsync_mode:
                args.primary_hold_target_mode = "current"
            primary_target_shape_cfg["forecast_safe_deadband_enabled"] = True
            primary_target_shape_cfg["forecast_safe_deadband_gate_mode"] = (
                "forecast_veto" if forecast_veto_mode else "conservative"
            )
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_provider_target"
            ] = bool(forecast_veto_mode)
            primary_target_shape_cfg[
                "forecast_safe_deadband_min_target_error_kg"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_MIN_TARGET_ERROR_KG", 1.0))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_require_recovering"
            ] = bool(forecast_veto_mode and not legacy_fullsync_mode)
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_require_pump_demand"
            ] = bool(forecast_veto_mode and not legacy_fullsync_mode)
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_refresh_current"
            ] = bool(legacy_fullsync_mode)
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_startup_delay_s"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_STARTUP_DELAY_S", 0.0))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_reentry_cooldown_s"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_REENTRY_COOLDOWN_S", 0.0))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_min_blend"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_MIN_BLEND", 1.0))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_full_pressure_norm"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_FULL_PRESSURE_NORM", 1.0))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_min_pressure_norm"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_MIN_PRESSURE_NORM", 1.5))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_future_rise_max"
            ] = float(
                os.environ.get(
                    "FOWT_FC_SAFE_DEADBAND_SYNC_FUTURE_RISE_MAX",
                    1.0e9 if legacy_fullsync_mode else 0.10,
                )
            )
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_near_zero_deg"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_NEAR_ZERO_DEG", 0.15))
            primary_target_shape_cfg[
                "forecast_safe_deadband_sync_rate_eps_deg_s"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_SYNC_RATE_EPS_DEG_S", 0.002))
            primary_target_shape_cfg["forecast_safe_deadband_pitch_enter_deg"] = float(
                os.environ.get("FOWT_FC_SAFE_DEADBAND_PITCH_ENTER_DEG", 1.5)
            )
            primary_target_shape_cfg["forecast_safe_deadband_roll_enter_deg"] = float(
                os.environ.get("FOWT_FC_SAFE_DEADBAND_ROLL_ENTER_DEG", 1.5)
            )
            primary_target_shape_cfg["forecast_safe_deadband_pitch_exit_deg"] = float(
                os.environ.get("FOWT_FC_SAFE_DEADBAND_PITCH_EXIT_DEG", 1.2)
            )
            primary_target_shape_cfg["forecast_safe_deadband_roll_exit_deg"] = float(
                os.environ.get("FOWT_FC_SAFE_DEADBAND_ROLL_EXIT_DEG", 1.2)
            )
            primary_target_shape_cfg[
                "forecast_safe_deadband_event_probability_max"
            ] = float(
                os.environ.get(
                    "FOWT_FC_SAFE_DEADBAND_EVENT_MAX",
                    0.99 if forecast_veto_mode else 0.75,
                )
            )
            primary_target_shape_cfg["forecast_safe_deadband_future_rise_max"] = float(
                os.environ.get("FOWT_FC_SAFE_DEADBAND_FUTURE_RISE_MAX", 0.05)
            )
            primary_target_shape_cfg[
                "forecast_safe_deadband_minimum_direction_dot"
            ] = float(os.environ.get("FOWT_FC_SAFE_DEADBAND_DIRECTION_DOT_MIN", 0.0))
            primary_target_shape_cfg["forecast_safe_deadband_posture_gate_deg"] = float(
                os.environ.get(
                    "FOWT_FC_SAFE_DEADBAND_POSTURE_GATE_DEG",
                    1.5 if forecast_veto_mode else 1.2,
                )
            )
            primary_target_shape_cfg["forecast_safe_deadband_allowed_actions"] = (
                (
                    "hold",
                    "pump_saving",
                    "active_small",
                    "active_medium",
                    "active_reverse_small",
                )
                if forecast_veto_mode
                else ("hold", "active_small")
            )
            primary_target_shape_cfg["forecast_safe_deadband_reject_unsafe_hold"] = False
    if primary_control_profile == "rawenv_forecastgated_command_deadband_v1":
        # Keep the planner target lifecycle unchanged. The learned forecast only
        # admits a command-layer release after the current posture has entered
        # the widened deadband and is no longer moving away from zero.
        args.primary_hold_target_mode = "pause"
        primary_target_shape_cfg["forecast_safe_deadband_enabled"] = True
        primary_target_shape_cfg["forecast_safe_deadband_gate_mode"] = "forecast_veto"
        primary_target_shape_cfg["forecast_safe_deadband_sync_provider_target"] = False
        primary_target_shape_cfg["forecast_safe_deadband_apply_pid_deadband"] = False
        primary_target_shape_cfg["forecast_safe_deadband_release_primary_target"] = False
        primary_target_shape_cfg["forecast_safe_deadband_pitch_enter_deg"] = float(
            os.environ.get("FOWT_FC_SAFE_DEADBAND_PITCH_ENTER_DEG", 1.5)
        )
        primary_target_shape_cfg["forecast_safe_deadband_roll_enter_deg"] = float(
            os.environ.get("FOWT_FC_SAFE_DEADBAND_ROLL_ENTER_DEG", 1.5)
        )
        primary_target_shape_cfg["forecast_safe_deadband_pitch_exit_deg"] = float(
            os.environ.get("FOWT_FC_SAFE_DEADBAND_PITCH_EXIT_DEG", 1.2)
        )
        primary_target_shape_cfg["forecast_safe_deadband_roll_exit_deg"] = float(
            os.environ.get("FOWT_FC_SAFE_DEADBAND_ROLL_EXIT_DEG", 1.2)
        )
        primary_target_shape_cfg["forecast_safe_deadband_posture_gate_deg"] = float(
            os.environ.get("FOWT_FC_SAFE_DEADBAND_POSTURE_GATE_DEG", 1.5)
        )
        primary_target_shape_cfg["forecast_safe_deadband_event_probability_max"] = float(
            os.environ.get("FOWT_FC_SAFE_DEADBAND_EVENT_MAX", 0.99)
        )
        primary_target_shape_cfg[
            "forecast_safe_deadband_require_event_probability"
        ] = True
        primary_target_shape_cfg["forecast_safe_deadband_allowed_actions"] = (
            "hold",
            "pump_saving",
            "active_small",
            "active_medium",
            "active_reverse_small",
        )
        primary_target_shape_cfg["forecast_safe_deadband_reject_unsafe_hold"] = False
        primary_target_shape_cfg["deadband_target_release_enabled"] = True
        primary_target_shape_cfg["deadband_target_release_require_forecast_safe"] = True
        primary_target_shape_cfg["deadband_target_release_pitch_deg"] = float(
            os.environ.get("FOWT_FC_COMMAND_RELEASE_PITCH_DEG", 1.0)
        )
        primary_target_shape_cfg["deadband_target_release_roll_deg"] = float(
            os.environ.get("FOWT_FC_COMMAND_RELEASE_ROLL_DEG", 1.0)
        )
        primary_target_shape_cfg["deadband_target_release_exit_pitch_deg"] = float(
            os.environ.get("FOWT_FC_COMMAND_RELEASE_EXIT_PITCH_DEG", 1.3)
        )
        primary_target_shape_cfg["deadband_target_release_exit_roll_deg"] = float(
            os.environ.get("FOWT_FC_COMMAND_RELEASE_EXIT_ROLL_DEG", 1.3)
        )
        primary_target_shape_cfg["deadband_target_release_near_zero_deg"] = float(
            os.environ.get("FOWT_FC_COMMAND_RELEASE_NEAR_ZERO_DEG", 0.15)
        )
        primary_target_shape_cfg["deadband_target_release_rate_eps_deg_s"] = float(
            os.environ.get("FOWT_FC_COMMAND_RELEASE_RATE_EPS_DEG_S", 0.002)
        )
        primary_target_shape_cfg["deadband_target_release_blend"] = 1.0
        primary_target_shape_cfg["deadband_target_release_require_both_axes"] = True
        primary_target_shape_cfg["deadband_target_release_use_pid_deadband"] = False
        primary_target_shape_cfg["deadband_target_release_reset_limiter"] = True
        # Preserve the planner's candidate set, but refresh an aged active target
        # in the current posture-recovery direction when moderate posture debt
        # develops. This prevents a same-name active_small action from retaining
        # one absolute tank target for the remainder of a six-hour case.
        args.active_posture_refresh = True
        args.active_posture_refresh_enter_deg = float(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_ENTER_DEG", 1.5)
        )
        args.active_posture_refresh_exit_deg = float(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_EXIT_DEG", 1.2)
        )
        args.active_posture_refresh_update_interval_s = float(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_INTERVAL_S", 300.0)
        )
        args.active_posture_refresh_action_name = str(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_ACTION", "active_small")
        )
        args.active_posture_refresh_min_target_age_s = float(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_MIN_AGE_S", 300.0)
        )
        args.active_posture_refresh_axis_mode = str(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_AXIS_MODE", "axis_debt")
        )
        args.active_posture_refresh_worsening_eps_deg_s = float(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_WORSENING_EPS", 0.0001)
        )
        args.active_posture_refresh_debt_deg = float(
            os.environ.get("FOWT_FC_ACTIVE_REFRESH_DEBT_DEG", 1.5)
        )
        args.active_posture_refresh_include_hold = (
            str(os.environ.get("FOWT_FC_ACTIVE_REFRESH_INCLUDE_HOLD", "1"))
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
    if primary_control_profile == "regime_auto_pump_saving_v1":
        # Runtime selector for multiple pump-saving regimes.  It is still
        # default-off and keeps v1.6 hard floor / recovery / fallback behavior.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "regime_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "psc_4hao_mild_fraction_v1":
        # No.4 mild-arm experiment.  This keeps the same safety floor and
        # regime selector as regime_auto, but changes the budget-exhausted
        # action from binary hold-vs-refresh to partial refresh.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(os.environ.get("FOWT_4HAO_BUDGET_M3", 100.0))
        args.economy_pump_budget_hold_refresh_fraction = float(
            os.environ.get("FOWT_4HAO_MILD_FRACTION", 0.50)
        )
        args.economy_pump_budget_allocator_mode = str(
            os.environ.get("FOWT_4HAO_ALLOCATOR_MODE", "regime_auto")
        )
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_4HAO_RELEASE_DEG", 3.5)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_4HAO_PRESSURE_NORM", 1.05)
        )
        args.economy_pump_budget_relief_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_NORM", 0.7)
        )
        args.economy_pump_budget_relief_drop_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_DROP_NORM", 0.2)
        )
        args.hold_comfort_release = True
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_action_name = "active_small"
        args.active_posture_refresh = (
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_REFRESH", "0").lower()
            not in {"0", "false", "no"}
        )
        args.active_posture_refresh_enter_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ENTER_DEG", 4.0)
        )
        args.active_posture_refresh_exit_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_EXIT_DEG", 3.2)
        )
        args.active_posture_refresh_update_interval_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_INTERVAL_S", 120.0)
        )
        args.active_posture_refresh_action_name = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ACTION", "active_small")
        )
        args.active_posture_refresh_pressure_max_norm = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_PRESSURE_MAX_NORM", 0.0)
        )
        args.active_posture_refresh_min_target_age_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_MIN_TARGET_AGE_S", 0.0)
        )
        args.active_posture_refresh_axis_mode = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_AXIS_MODE", "max")
        )
        args.active_posture_refresh_worsening_eps_deg_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_WORSENING_EPS_DEG_S", 0.0)
        )
        args.active_posture_refresh_debt_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_DEBT_DEG", 0.0)
        )
        args.hold_comfort_forecast_veto = True
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_4HAO_POSTURE_RELIEF_NORM", 0.20)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_4HAO_VETO_MAX_S", 300.0)
        )
        args.fallback_risk_active_release = _env_bool(
            "FOWT_4HAO_FALLBACK_RISK_RELEASE",
            bool(args.fallback_risk_active_release),
        )
        args.fallback_risk_active_release_enter_deg = float(
            os.environ.get("FOWT_4HAO_FALLBACK_RISK_ENTER_DEG", 5.5)
        )
        args.fallback_risk_active_release_exit_deg = float(
            os.environ.get("FOWT_4HAO_FALLBACK_RISK_EXIT_DEG", 4.8)
        )
        args.fallback_risk_active_release_worsening_eps_deg_s = float(
            os.environ.get("FOWT_4HAO_FALLBACK_RISK_WORSENING_EPS", 0.05)
        )
        args.fallback_risk_active_release_pump_rate_m3_min = float(
            os.environ.get("FOWT_4HAO_FALLBACK_RISK_PUMP_RATE", 10.0)
        )
        args.fallback_risk_active_release_backlog_kg = float(
            os.environ.get("FOWT_4HAO_FALLBACK_RISK_BACKLOG_KG", 500.0)
        )
    if primary_control_profile in {
        "psc_4hao_debt_aware_v1",
        "psc_4hao_guarded_v2",
        "psc_4hao_guarded_v3",
        "psc_4hao_regime_gated_v1",
        "psc_4hao_chain_guard_v1",
        "psc_4hao_exec_guard_v1",
    }:
        # No.4 18% candidate: keep the real partial-refresh mild action, then
        # charge repeated high-posture waits that fail to realize forecast relief.
        is_guarded = primary_control_profile in {
            "psc_4hao_guarded_v2",
            "psc_4hao_guarded_v3",
            "psc_4hao_regime_gated_v1",
            "psc_4hao_chain_guard_v1",
            "psc_4hao_exec_guard_v1",
        }
        is_hard_guarded = primary_control_profile == "psc_4hao_guarded_v3"
        is_regime_gated = primary_control_profile == "psc_4hao_regime_gated_v1"
        is_chain_guard = primary_control_profile == "psc_4hao_chain_guard_v1"
        is_exec_guard = primary_control_profile == "psc_4hao_exec_guard_v1"
        if is_guarded and str(args.primary_safety_profile) == "default":
            args.primary_safety_profile = str(
                os.environ.get("FOWT_4HAO_PRIMARY_SAFETY_PROFILE", "strict")
            )
            primary_target_shape_cfg = primary_safety_override(
                str(args.primary_safety_profile)
            )
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(os.environ.get("FOWT_4HAO_BUDGET_M3", 100.0))
        args.economy_pump_budget_hold_refresh_fraction = float(
            os.environ.get("FOWT_4HAO_MILD_FRACTION", 0.35)
        )
        if is_hard_guarded:
            args.economy_pump_budget_dynamic_refresh = True
            args.economy_pump_budget_dynamic_refresh_enter_deg = float(
                os.environ.get("FOWT_4HAO_DYNAMIC_ENTER_DEG", 3.2)
            )
            args.economy_pump_budget_dynamic_refresh_full_deg = float(
                os.environ.get("FOWT_4HAO_DYNAMIC_FULL_DEG", 4.2)
            )
            args.economy_pump_budget_dynamic_refresh_max_fraction = float(
                os.environ.get("FOWT_4HAO_DYNAMIC_MAX_FRACTION", 1.0)
            )
        args.economy_pump_budget_allocator_mode = str(
            os.environ.get(
                "FOWT_4HAO_ALLOCATOR_MODE",
                "dual_specialist_auto" if is_regime_gated else "regime_auto",
            )
        )
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_4HAO_RELEASE_DEG", 3.2)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_4HAO_PRESSURE_NORM", 1.05)
        )
        args.economy_pump_budget_relief_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_NORM", 0.7)
        )
        args.economy_pump_budget_relief_drop_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_DROP_NORM", 0.2)
        )
        args.hold_comfort_release = True
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_action_name = "active_small"
        args.active_posture_refresh = (
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_REFRESH",
                "1" if is_guarded else "0",
            ).lower()
            not in {"0", "false", "no"}
        )
        args.active_posture_refresh_enter_deg = float(
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_ENTER_DEG",
                3.6 if is_hard_guarded else 4.0,
            )
        )
        args.active_posture_refresh_exit_deg = float(
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_EXIT_DEG",
                2.8 if is_hard_guarded else 3.2,
            )
        )
        args.active_posture_refresh_update_interval_s = float(
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_INTERVAL_S",
                60.0 if is_hard_guarded else 120.0,
            )
        )
        args.active_posture_refresh_action_name = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ACTION", "active_small")
        )
        args.active_posture_refresh_pressure_max_norm = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_PRESSURE_MAX_NORM", 0.0)
        )
        args.active_posture_refresh_min_target_age_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_MIN_TARGET_AGE_S", 0.0)
        )
        args.active_posture_refresh_axis_mode = str(
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_AXIS_MODE",
                "axis_debt" if is_guarded else "max",
            )
        )
        args.active_posture_refresh_worsening_eps_deg_s = float(
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_WORSENING_EPS_DEG_S",
                0.0005 if is_guarded else 0.0,
            )
        )
        args.active_posture_refresh_debt_deg = float(
            os.environ.get(
                "FOWT_4HAO_ACTIVE_POSTURE_DEBT_DEG",
                4.5 if is_guarded else 0.0,
            )
        )
        args.hold_comfort_forecast_veto = True
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_4HAO_POSTURE_RELIEF_NORM", 0.20)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_4HAO_VETO_MAX_S", 300.0)
        )
        args.hold_relief_debt = True
        args.hold_relief_debt_weight = float(
            os.environ.get("FOWT_4HAO_DEBT_WEIGHT", 6.0)
        )
        args.hold_relief_debt_threshold_deg = float(
            os.environ.get("FOWT_4HAO_DEBT_THRESHOLD_DEG", 3.0)
        )
        args.hold_relief_debt_response_eps_deg = float(
            os.environ.get("FOWT_4HAO_DEBT_RESPONSE_EPS_DEG", 0.10)
        )
        args.hold_relief_debt_increment = float(
            os.environ.get("FOWT_4HAO_DEBT_INCREMENT", 1.0)
        )
        args.hold_relief_debt_max = float(os.environ.get("FOWT_4HAO_DEBT_MAX", 6.0))
        args.hold_relief_debt_lowrisk_pressure_norm = float(
            os.environ.get("FOWT_4HAO_DEBT_LOWRISK_PRESSURE_NORM", 0.35)
        )
        args.hold_relief_debt_lowrisk_safe_deg = float(
            os.environ.get("FOWT_4HAO_DEBT_LOWRISK_SAFE_DEG", 2.3)
        )
        args.fallback_risk_active_release = _env_bool(
            "FOWT_4HAO_FALLBACK_RISK_RELEASE",
            bool(is_guarded or args.fallback_risk_active_release),
        )
        args.fallback_risk_active_release_enter_deg = float(
            os.environ.get(
                "FOWT_4HAO_FALLBACK_RISK_ENTER_DEG",
                4.8 if is_guarded else 5.5,
            )
        )
        args.fallback_risk_active_release_exit_deg = float(
            os.environ.get(
                "FOWT_4HAO_FALLBACK_RISK_EXIT_DEG",
                4.2 if is_guarded else 4.8,
            )
        )
        args.fallback_risk_active_release_worsening_eps_deg_s = float(
            os.environ.get(
                "FOWT_4HAO_FALLBACK_RISK_WORSENING_EPS",
                0.03 if is_guarded else 0.05,
            )
        )
        args.fallback_risk_active_release_pump_rate_m3_min = float(
            os.environ.get(
                "FOWT_4HAO_FALLBACK_RISK_PUMP_RATE",
                6.0 if is_guarded else 10.0,
            )
        )
        args.fallback_risk_active_release_backlog_kg = float(
            os.environ.get(
                "FOWT_4HAO_FALLBACK_RISK_BACKLOG_KG",
                400.0 if is_guarded else 500.0,
            )
        )
        if is_chain_guard:
            # The current trained preview is 6x10min, so use near-horizon
            # evidence and measured posture response instead of unavailable
            # 60-120min far-horizon guards.
            args.reactive_floor_predictive_veto = str(
                os.environ.get("FOWT_4HAO_CHAIN_REACTIVE_FLOOR", "theta_only_gated")
            )
            args.high_posture_enter = float(
                os.environ.get("FOWT_4HAO_CHAIN_FLOOR_ENTER_DEG", 4.8)
            )
            args.high_posture_exit = float(
                os.environ.get("FOWT_4HAO_CHAIN_FLOOR_EXIT_DEG", 4.2)
            )
            args.high_posture_min_duration_s = float(
                os.environ.get("FOWT_4HAO_CHAIN_FLOOR_HOLD_S", 30.0)
            )
            args.reactive_floor_theta_only_enter = float(
                os.environ.get("FOWT_4HAO_CHAIN_THETA_ENTER_DEG", 5.8)
            )
            args.reactive_floor_theta_only_min_duration_s = float(
                os.environ.get("FOWT_4HAO_CHAIN_THETA_HOLD_S", 60.0)
            )
            args.reactive_floor_action = str(
                os.environ.get("FOWT_4HAO_CHAIN_FLOOR_ACTION", "active_small")
            )
            args.reactive_floor_post_exit_mode = str(
                os.environ.get("FOWT_4HAO_CHAIN_POST_EXIT", "target_cap")
            )
            args.reactive_floor_post_exit_target_cap_kg = float(
                os.environ.get("FOWT_4HAO_CHAIN_POST_EXIT_CAP_KG", 500.0)
            )
            args.sustained_active_recompute = _env_bool(
                "FOWT_4HAO_CHAIN_SUSTAINED_RECOMPUTE", True
            )
            args.sustained_active_recompute_min_active_buckets = int(
                os.environ.get("FOWT_4HAO_CHAIN_RECOMPUTE_BUCKETS", 2)
            )
            args.sustained_active_recompute_pitch_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_RECOMPUTE_PITCH_DEG", 4.2)
            )
            args.sustained_active_recompute_roll_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_RECOMPUTE_ROLL_DEG", 4.0)
            )
            args.sustained_active_recompute_min_target_age_s = float(
                os.environ.get("FOWT_4HAO_CHAIN_RECOMPUTE_AGE_S", 600.0)
            )
            args.stale_active_target_refresh = str(
                os.environ.get(
                    "FOWT_4HAO_CHAIN_STALE_TARGET",
                    "refresh_only_guarded_v1",
                )
            )
            args.stale_active_target_refresh_pitch_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_STALE_PITCH_DEG", 4.6)
            )
            args.stale_active_target_refresh_roll_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_STALE_ROLL_DEG", 4.2)
            )
            args.stale_active_target_refresh_min_target_age_s = float(
                os.environ.get("FOWT_4HAO_CHAIN_STALE_AGE_S", 600.0)
            )
            args.stale_active_target_refresh_cooldown_s = float(
                os.environ.get("FOWT_4HAO_CHAIN_STALE_COOLDOWN_S", 600.0)
            )
            args.active_effectiveness_refresh = _env_bool(
                "FOWT_4HAO_CHAIN_ACTIVE_EFFECTIVENESS", True
            )
            args.active_effectiveness_refresh_min_active_buckets = int(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_BUCKETS", 2)
            )
            args.active_effectiveness_refresh_pitch_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_PITCH_DEG", 4.2)
            )
            args.active_effectiveness_refresh_roll_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_ROLL_DEG", 4.0)
            )
            args.active_effectiveness_refresh_min_target_age_s = float(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_AGE_S", 300.0)
            )
            args.active_effectiveness_refresh_response_eps_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_RESPONSE_DEG", 0.05)
            )
            args.active_effectiveness_refresh_min_delta_kg = float(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_DELTA_KG", 500.0)
            )
            args.active_effectiveness_refresh_episode_budget_m3 = float(
                os.environ.get("FOWT_4HAO_CHAIN_ACTIVE_EFF_BUDGET_M3", 240.0)
            )
            args.no_unexplained_hold_action = _env_bool(
                "FOWT_4HAO_CHAIN_NO_HOLD_ACTION", True
            )
            args.no_unexplained_hold_threshold_deg = float(
                os.environ.get("FOWT_4HAO_CHAIN_NO_HOLD_DEG", 3.8)
            )
            args.no_unexplained_hold_trigger_buckets = int(
                os.environ.get("FOWT_4HAO_CHAIN_NO_HOLD_BUCKETS", 2)
            )
        if is_exec_guard:
            primary_target_shape_cfg["primary_safety_prefallback_enabled"] = _env_bool(
                "FOWT_4HAO_EXEC_PREFALLBACK", True
            )
            primary_target_shape_cfg["primary_safety_prefallback_pitch_enter_deg"] = float(
                os.environ.get("FOWT_4HAO_EXEC_PREFALLBACK_PITCH_ENTER_DEG", 4.8)
            )
            primary_target_shape_cfg["primary_safety_prefallback_roll_enter_deg"] = float(
                os.environ.get("FOWT_4HAO_EXEC_PREFALLBACK_ROLL_ENTER_DEG", 4.2)
            )
            primary_target_shape_cfg["primary_safety_prefallback_pitch_exit_deg"] = float(
                os.environ.get("FOWT_4HAO_EXEC_PREFALLBACK_PITCH_EXIT_DEG", 4.1)
            )
            primary_target_shape_cfg["primary_safety_prefallback_roll_exit_deg"] = float(
                os.environ.get("FOWT_4HAO_EXEC_PREFALLBACK_ROLL_EXIT_DEG", 3.6)
            )
            primary_target_shape_cfg["primary_safety_prefallback_enter_hold_s"] = float(
                os.environ.get("FOWT_4HAO_EXEC_PREFALLBACK_HOLD_S", 0.0)
            )
    if primary_control_profile == "psc_4hao_dynamic_refresh_v1":
        # No.4 18% candidate: keep low-posture economy mild, but dynamically
        # raise the blocked-refresh fraction as live posture enters the debt band.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(os.environ.get("FOWT_4HAO_BUDGET_M3", 100.0))
        args.economy_pump_budget_hold_refresh_fraction = float(
            os.environ.get("FOWT_4HAO_MILD_FRACTION", 0.35)
        )
        args.economy_pump_budget_dynamic_refresh = True
        args.economy_pump_budget_dynamic_refresh_enter_deg = float(
            os.environ.get("FOWT_4HAO_DYNAMIC_ENTER_DEG", 3.4)
        )
        args.economy_pump_budget_dynamic_refresh_full_deg = float(
            os.environ.get("FOWT_4HAO_DYNAMIC_FULL_DEG", 4.7)
        )
        args.economy_pump_budget_dynamic_refresh_max_fraction = float(
            os.environ.get("FOWT_4HAO_DYNAMIC_MAX_FRACTION", 0.85)
        )
        args.economy_pump_budget_allocator_mode = str(
            os.environ.get("FOWT_4HAO_ALLOCATOR_MODE", "regime_auto")
        )
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_4HAO_RELEASE_DEG", 3.2)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_4HAO_PRESSURE_NORM", 1.05)
        )
        args.economy_pump_budget_relief_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_NORM", 0.7)
        )
        args.economy_pump_budget_relief_drop_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_DROP_NORM", 0.2)
        )
        args.hold_comfort_release = True
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_action_name = "active_small"
        args.active_posture_refresh = (
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_REFRESH", "0").lower()
            not in {"0", "false", "no"}
        )
        args.active_posture_refresh_enter_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ENTER_DEG", 4.0)
        )
        args.active_posture_refresh_exit_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_EXIT_DEG", 3.2)
        )
        args.active_posture_refresh_update_interval_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_INTERVAL_S", 120.0)
        )
        args.active_posture_refresh_action_name = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ACTION", "active_small")
        )
        args.active_posture_refresh_pressure_max_norm = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_PRESSURE_MAX_NORM", 0.0)
        )
        args.active_posture_refresh_min_target_age_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_MIN_TARGET_AGE_S", 0.0)
        )
        args.active_posture_refresh_axis_mode = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_AXIS_MODE", "max")
        )
        args.active_posture_refresh_worsening_eps_deg_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_WORSENING_EPS_DEG_S", 0.0)
        )
        args.active_posture_refresh_debt_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_DEBT_DEG", 0.0)
        )
        args.hold_comfort_forecast_veto = True
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_4HAO_POSTURE_RELIEF_NORM", 0.20)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_4HAO_VETO_MAX_S", 300.0)
        )
    if primary_control_profile in {
        "psc_4hao_generalized_gate_v1",
        "psc_4hao_regime_addback_v2_provider_v1",
    }:
        # No.4 generalized-gate candidate from the 2026-06-03 broad validation:
        # keep the same dynamic mild-refresh action as refresh_on, but admit
        # only the neutral/headroom stable-plateau family.  Relief/boundary
        # regimes are intentionally release-only unless a separate narrow
        # specialist is selected.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(os.environ.get("FOWT_4HAO_BUDGET_M3", 100.0))
        args.economy_pump_budget_hold_refresh_fraction = float(
            os.environ.get("FOWT_4HAO_MILD_FRACTION", 0.35)
        )
        args.economy_pump_budget_dynamic_refresh = True
        args.economy_pump_budget_dynamic_refresh_enter_deg = float(
            os.environ.get("FOWT_4HAO_DYNAMIC_ENTER_DEG", 3.4)
        )
        args.economy_pump_budget_dynamic_refresh_full_deg = float(
            os.environ.get("FOWT_4HAO_DYNAMIC_FULL_DEG", 4.7)
        )
        args.economy_pump_budget_dynamic_refresh_max_fraction = float(
            os.environ.get("FOWT_4HAO_DYNAMIC_MAX_FRACTION", 0.85)
        )
        default_allocator = (
            "psc_4hao_regime_addback_v2_auto"
            if primary_control_profile == "psc_4hao_regime_addback_v2_provider_v1"
            else "neutral_mhs_broader_auto"
        )
        args.economy_pump_budget_allocator_mode = str(
            os.environ.get("FOWT_4HAO_ALLOCATOR_MODE", default_allocator)
        )
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_4HAO_RELEASE_DEG", 3.2)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_4HAO_PRESSURE_NORM", 1.05)
        )
        args.economy_pump_budget_relief_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_NORM", 0.7)
        )
        args.economy_pump_budget_relief_drop_norm = float(
            os.environ.get("FOWT_4HAO_RELIEF_DROP_NORM", 0.2)
        )
        args.hold_comfort_release = True
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_ENTER_DEG", 3.0)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_4HAO_COMFORT_EXIT_DEG", 2.4)
        )
        args.hold_comfort_action_name = "active_small"
        args.active_posture_refresh = (
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_REFRESH", "1").lower()
            not in {"0", "false", "no"}
        )
        args.active_posture_refresh_enter_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ENTER_DEG", 4.0)
        )
        args.active_posture_refresh_exit_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_EXIT_DEG", 3.2)
        )
        args.active_posture_refresh_update_interval_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_INTERVAL_S", 120.0)
        )
        args.active_posture_refresh_action_name = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_ACTION", "active_small")
        )
        args.active_posture_refresh_pressure_max_norm = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_PRESSURE_MAX_NORM", 0.0)
        )
        args.active_posture_refresh_min_target_age_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_MIN_TARGET_AGE_S", 0.0)
        )
        args.active_posture_refresh_axis_mode = str(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_AXIS_MODE", "max")
        )
        args.active_posture_refresh_worsening_eps_deg_s = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_WORSENING_EPS_DEG_S", 0.0)
        )
        args.active_posture_refresh_debt_deg = float(
            os.environ.get("FOWT_4HAO_ACTIVE_POSTURE_DEBT_DEG", 0.0)
        )
        args.hold_comfort_forecast_veto = True
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_4HAO_POSTURE_RELIEF_NORM", 0.20)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_4HAO_VETO_MAX_S", 300.0)
        )
    if primary_control_profile == "relief_decay_smart_release_v1":
        # Fixed specialist profile from h120 pump-saving work:
        # v1.6 hard floor plus a 100 m3 economy budget and forecast-smart
        # release. It is intended for ex-ante relief/decay opportunity regimes;
        # do not use it as the global mixed-regime default.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "relief_decay_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "relief_decay_dispatcher_v2":
        # Corrected relief/decay dispatcher.  It keeps the v1.6 hard floor,
        # recovery and fallback untouched, but allows exhausted-budget holds
        # when the live near-horizon forecast shows a transient peak followed
        # by decay.  Unlike v1, high near-pressure is not by itself a release:
        # high-then-relief is exactly the intended opportunity.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "relief_decay_auto_v2"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "relief_decay_episode_auto_v1":
        # Episode-level relief/decay selector.  This is a bounded test profile:
        # it keeps the same v1.6 hard safety floor and budget mechanics as v2,
        # but latches the economy mode earlier when the live forecast already
        # shows a high, direction-stable episode even before the later drop is
        # visible to the learned model.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "relief_decay_episode_auto_v1"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "residual_high_economy_v1":
        # Fixed specialist profile for stable residual-high regimes.  It is
        # deliberately separate from the relief/decay specialist because the
        # opportunity is broader: avoid non-hard-safety economy chasing under
        # high but non-intensifying near pressure.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "residual_high_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "residual_high_plateau_v2":
        # v2 specialist for high-pressure plateaus.  Unlike residual_high_auto,
        # high pressure itself is not a release condition; the release/veto
        # boundary is posture-floor proximity, fallback dominance,
        # re-intensification, or direction/reversal risk.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "residual_high_plateau_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "residual_high_plateau_strict_v2":
        # Conservative automatic plateau profile.  It intentionally keeps the
        # broad residual_high_plateau_v2 action and budget unchanged, but
        # requires a balanced two-axis posture signature before entering so
        # signflip/catch-up boundaries are less likely to be admitted.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "residual_high_plateau_strict_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "neutral_mhs_clean_budget_v1":
        # Specialist profile for neutral moderate-high steady cases that start
        # from a very clean posture.  Forecast shape identifies the steady high
        # load; the startup posture gate keeps the aggressive economy hold out
        # of safety-margin-heavy lookalikes.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "neutral_mhs_clean_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "dual_specialist_pump_saving_v1":
        # Integration prototype, not the current paper-ready mainline.  It runs
        # the relief/decay episode specialist first, otherwise allows the
        # neutral-MHS clean-start specialist.  Mixed validation showed positive
        # aggregate pump saving but boundary/catch-up harm, so keep it for
        # diagnosis until boundary abstention is stronger.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = 100.0
        args.economy_pump_budget_allocator_mode = "dual_specialist_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = 4.7
        args.economy_pump_budget_smart_pressure_norm = 1.05
        args.economy_pump_budget_relief_norm = 0.7
        args.economy_pump_budget_relief_drop_norm = 0.2
    if primary_control_profile == "residual_high_plateau_suppression_v1":
        # Continuous-pursuit suppressor for steady residual-high plateaus.
        # This deliberately does not enable the refresh-level economy budget:
        # the hypothesis is that plateau pump is target-pursuit chatter between
        # refreshes, so the test should isolate the continuous suppression hook.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.pump_suppression = True
        args.pump_suppression_plateau = True
        args.pump_suppression_plateau_only = True
        args.pump_suppression_plateau_high_norm = 1.05
        args.pump_suppression_plateau_range_norm = 0.15
        args.pump_suppression_plateau_rise_norm = 0.10
        args.pump_suppression_plateau_restart_err_kg = 2000.0
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        primary_target_shape_cfg["preview_pump_suppression_allow_latched"] = True
    if primary_control_profile == "gusty_oscillation_smoothing_v1":
        # One-shot candidate for the mined gusty/oscillatory regime.  It does
        # not open the budget hold; it only delays young active/stale target
        # refreshes when learned forecast speed has repeated peaks, acting as a
        # low-pass on target pursuit.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = False
        args.pump_suppression = True
        args.pump_suppression_plateau_only = True
        args.pump_suppression_restart_err_kg = 2000.0
        args.gusty_oscillation = True
        args.gusty_oscillation_min_target_age_s = 600.0
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        primary_target_shape_cfg["preview_pump_suppression_allow_latched"] = True
    if primary_control_profile == "gusty_oscillation_budget_auto_v1":
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(os.environ.get("FOWT_GUSTY_BUDGET_M3", 100.0))
        args.economy_pump_budget_allocator_mode = "gusty_oscillation_budget_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_GUSTY_POSTURE_RELEASE_DEG", 4.7)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_GUSTY_REINTENSIFY_NORM", 1.05)
        )
        args.gusty_oscillation = True
        args.gusty_oscillation_min_target_age_s = float(
            os.environ.get("FOWT_GUSTY_MIN_TARGET_AGE_S", 900.0)
        )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile == "gusty_oscillation_soft_cap_v1":
        # C3/gusty conservative candidate: do not open the broad budget hold
        # and do not enter the pause-style target mode.  This profile only
        # rate-limits young target refreshes when the learned forecast shows
        # repeated oscillatory peaks, so it should remove churn without
        # relaxing posture tracking for a full episode.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = False
        args.economy_pump_budget_m3 = 0.0
        args.economy_pump_budget_allocator_mode = "gusty_oscillation_soft_cap_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_GUSTY_SOFT_POSTURE_RELEASE_DEG", 3.8)
        )
        args.gusty_oscillation = True
        args.gusty_oscillation_min_target_age_s = float(
            os.environ.get("FOWT_GUSTY_REFRESH_COOLDOWN_S", 1200.0)
        )
        args.gusty_oscillation_target_scale = float(
            os.environ.get("FOWT_GUSTY_SOFT_TARGET_SCALE", 0.70)
        )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile == "gusty_oscillation_refined_v1":
        # C3 refined: preserve the repeated-peak anti-chatter mechanism, but
        # reduce suppression as current posture consumes headroom.  The goal is
        # a cleaner 30%+ regime result rather than the aggressive 65% endpoint.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = False
        args.economy_pump_budget_m3 = 0.0
        args.economy_pump_budget_allocator_mode = "gusty_oscillation_soft_cap_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_GUSTY_REFINED_POSTURE_RELEASE_DEG", 3.2)
        )
        args.gusty_oscillation = True
        args.gusty_oscillation_min_target_age_s = float(
            os.environ.get("FOWT_GUSTY_REFINED_REFRESH_COOLDOWN_S", 3600.0)
        )
        args.gusty_oscillation_target_scale = float(
            os.environ.get("FOWT_GUSTY_REFINED_BASE_SCALE", 0.55)
        )
        args.gusty_oscillation_headroom_scaled = True
        args.gusty_oscillation_low_posture_deg = float(
            os.environ.get("FOWT_GUSTY_REFINED_LOW_POSTURE_DEG", 1.5)
        )
        args.gusty_oscillation_high_posture_deg = float(
            os.environ.get("FOWT_GUSTY_REFINED_HIGH_POSTURE_DEG", 3.2)
        )
        args.gusty_oscillation_low_posture_scale = float(
            os.environ.get("FOWT_GUSTY_REFINED_LOW_SCALE", 0.55)
        )
        args.gusty_oscillation_high_posture_scale = float(
            os.environ.get("FOWT_GUSTY_REFINED_HIGH_SCALE", 0.90)
        )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile in (
        "gusty_oscillation_true_gate_v1",
        "gusty_oscillation_true_gate_off_v1",
    ):
        # True C3 attribution profile: no posture residual and no broad budget
        # hold.  The only intended difference between on/off is whether the
        # 6-bucket learned-runtime gusty gate may blend target refreshes.
        args.planner_posture_state_residual = False
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = False
        args.economy_pump_budget_m3 = 0.0
        args.economy_pump_budget_allocator_mode = "gusty_oscillation_soft_cap_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_POSTURE_RELEASE_DEG", 3.2)
        )
        args.gusty_oscillation = primary_control_profile == "gusty_oscillation_true_gate_v1"
        args.gusty_oscillation_short_gate = True
        args.gusty_oscillation_min_target_age_s = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_MIN_TARGET_AGE_S", 1200.0)
        )
        args.gusty_oscillation_target_scale = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_BASE_SCALE", 0.0)
        )
        args.gusty_oscillation_headroom_scaled = True
        args.gusty_oscillation_low_posture_deg = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_LOW_POSTURE_DEG", 1.5)
        )
        args.gusty_oscillation_high_posture_deg = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_HIGH_POSTURE_DEG", 3.2)
        )
        args.gusty_oscillation_low_posture_scale = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_LOW_SCALE", 0.65)
        )
        args.gusty_oscillation_high_posture_scale = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_HIGH_SCALE", 0.95)
        )
        args.gusty_oscillation_short_min_range_ms = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_MIN_RANGE_MS", 0.75)
        )
        args.gusty_oscillation_short_min_pressure_norm = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_MIN_PRESSURE_NORM", 0.75)
        )
        args.gusty_oscillation_short_max_dir_shift_deg = float(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_MAX_DIR_SHIFT_DEG", 45.0)
        )
        args.gusty_oscillation_short_min_turns = int(
            os.environ.get("FOWT_GUSTY_TRUE_GATE_MIN_TURNS", 1)
        )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile in (
        "gusty_hold_current_gated_v1",
        "gusty_hold_current_gate_off_v1",
        "gusty_hold_current_balanced_v1",
        "gusty_hold_current_balanced_release_v1",
        "gusty_hold_current_debt_guard_v1",
        "gusty_hold_current_isolated_v1",
        "gusty_hold_current_isolated_gate_off_v1",
    ):
        # C3 ownership experiment: isolate the dominant hold-action current-
        # target lifecycle branch.  No posture residual, no broad budget hold,
        # and no event-reset gusty blend branch.
        args.planner_posture_state_residual = False
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = False
        args.economy_pump_budget_m3 = 0.0
        args.economy_pump_budget_allocator_mode = "budget"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_GUSTY_HOLD_CURRENT_POSTURE_RELEASE_DEG", 3.2)
        )
        args.gusty_oscillation = primary_control_profile in (
            "gusty_hold_current_gated_v1",
            "gusty_hold_current_balanced_v1",
            "gusty_hold_current_balanced_release_v1",
            "gusty_hold_current_debt_guard_v1",
            "gusty_hold_current_isolated_v1",
        )
        args.gusty_oscillation_short_gate = True
        args.gusty_hold_current_gate_mode = (
            "gate_on"
            if primary_control_profile
            in (
                "gusty_hold_current_gated_v1",
                "gusty_hold_current_balanced_v1",
                "gusty_hold_current_balanced_release_v1",
                "gusty_hold_current_debt_guard_v1",
                "gusty_hold_current_isolated_v1",
            )
            else "gate_off"
        )
        args.gusty_hold_current_isolated_primary = primary_control_profile in (
            "gusty_hold_current_isolated_v1",
            "gusty_hold_current_isolated_gate_off_v1",
        )
        args.gusty_oscillation_min_target_age_s = float(
            os.environ.get("FOWT_GUSTY_HOLD_CURRENT_MIN_TARGET_AGE_S", 1200.0)
        )
        default_min_range = (
            0.50
            if primary_control_profile
            in (
                "gusty_hold_current_balanced_v1",
                "gusty_hold_current_balanced_release_v1",
                "gusty_hold_current_debt_guard_v1",
            )
            or primary_control_profile
            in ("gusty_hold_current_isolated_v1", "gusty_hold_current_isolated_gate_off_v1")
            else 0.75
        )
        args.gusty_oscillation_short_min_range_ms = float(
            os.environ.get("FOWT_GUSTY_HOLD_CURRENT_MIN_RANGE_MS", default_min_range)
        )
        args.gusty_oscillation_short_min_pressure_norm = float(
            os.environ.get("FOWT_GUSTY_HOLD_CURRENT_MIN_PRESSURE_NORM", 0.75)
        )
        args.gusty_oscillation_short_max_dir_shift_deg = float(
            os.environ.get("FOWT_GUSTY_HOLD_CURRENT_MAX_DIR_SHIFT_DEG", 45.0)
        )
        default_min_turns = (
            0
            if primary_control_profile
            in (
                "gusty_hold_current_balanced_v1",
                "gusty_hold_current_balanced_release_v1",
                "gusty_hold_current_debt_guard_v1",
            )
            else 1
        )
        args.gusty_oscillation_short_min_turns = int(
            os.environ.get("FOWT_GUSTY_HOLD_CURRENT_MIN_TURNS", default_min_turns)
        )
        if primary_control_profile == "gusty_hold_current_isolated_v1":
            args.gusty_hold_current_day_budget = bool(
                int(os.environ.get("FOWT_C3_HOLD_CURRENT_DAY_BUDGET", "1"))
            )
            args.gusty_hold_current_day_budget_max_buckets = int(
                os.environ.get("FOWT_C3_HOLD_CURRENT_DAY_BUDGET_MAX_BUCKETS", 3)
            )
            args.gusty_hold_current_day_budget_window_s = float(
                os.environ.get("FOWT_C3_HOLD_CURRENT_DAY_BUDGET_WINDOW_S", 86400.0)
            )
        if primary_control_profile in (
            "gusty_hold_current_balanced_release_v1",
            "gusty_hold_current_debt_guard_v1",
            "gusty_hold_current_isolated_v1",
        ):
            args.gusty_hold_current_runtime_release = True
            args.gusty_hold_current_runtime_release_deg = float(
                os.environ.get("FOWT_GUSTY_HOLD_CURRENT_RUNTIME_RELEASE_DEG", 4.2)
            )
            args.gusty_hold_current_runtime_release_cooldown_s = float(
                os.environ.get(
                    "FOWT_GUSTY_HOLD_CURRENT_RUNTIME_RELEASE_COOLDOWN_S",
                    300.0,
                )
            )
        if primary_control_profile == "gusty_hold_current_debt_guard_v1":
            args.gusty_hold_current_debt_guard = True
            args.gusty_hold_current_debt_guard_target_motion_kg_s = float(
                os.environ.get("FOWT_C3_DEBT_GUARD_TARGET_MOTION_KG_S", 220.0)
            )
            args.gusty_hold_current_debt_guard_target_err_kg = float(
                os.environ.get("FOWT_C3_DEBT_GUARD_TARGET_ERR_KG", 1200.0)
            )
            args.gusty_hold_current_debt_guard_min_age_s = float(
                os.environ.get("FOWT_C3_DEBT_GUARD_MIN_AGE_S", 600.0)
            )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile == "direction_reversal_avoidance_v1":
        # Direction-reversal economy specialist.  Unlike the broad reversal
        # veto, this profile treats a strong predicted sign flip as evidence
        # that the current economy refresh may be the wrong direction to chase.
        # It only delays non-hard-domain economy refreshes after a small budget
        # has already been spent; hard floor, fallback, recovery, and pump
        # penalty remain unchanged.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_BUDGET_M3", 60.0)
        )
        args.economy_pump_budget_allocator_mode = "direction_reversal_avoidance_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_POSTURE_RELEASE_DEG", 3.0)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_PRESSURE_NORM", 1.05)
        )
        args.hold_comfort_release = True
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ENTER_DEG", 2.5)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ENTER_DEG", 2.5)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_EXIT_DEG", 2.0)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_EXIT_DEG", 2.0)
        )
        args.hold_comfort_action_name = "active_small"
        args.hold_comfort_forecast_veto = True
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_POSTURE_RELIEF_NORM", 0.25)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_VETO_MAX_S", 300.0)
        )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile == "direction_reversal_avoidance_v2":
        # Direction-reversal posture-specific specialist.  v1 showed that
        # broad pause/hold can save pump in reversal windows but over-relaxes
        # high-posture cases.  v2 keeps the same predictor evidence, but lets
        # a planner hold freeze the target only when the future pressure is
        # projected to relieve the current posture; otherwise hold releases
        # to the normal PI target path.
        args.primary_hold_target_mode = "forecast_pause"
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_BUDGET_M3", 60.0)
        )
        args.economy_pump_budget_allocator_mode = "direction_reversal_avoidance_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_POSTURE_RELEASE_DEG", 3.0)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_PRESSURE_NORM", 1.05)
        )
        args.hold_comfort_release = True
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ENTER_DEG", 2.5)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ENTER_DEG", 2.5)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_EXIT_DEG", 2.0)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_EXIT_DEG", 2.0)
        )
        args.hold_comfort_action_name = "active_small"
        args.hold_comfort_forecast_veto = True
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_POSTURE_RELIEF_NORM", 0.25)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_VETO_MAX_S", 300.0)
        )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile in {
        "direction_reversal_layered_v3",
        "direction_reversal_layered_v4",
        "direction_reversal_layered_v5",
        "direction_reversal_layered_v6",
        "direction_reversal_layered_v7",
        "direction_reversal_layered_v8",
        "direction_reversal_layered_v8_holdgate_v1",
        "direction_reversal_layered_v8_holdgate_off_v1",
    }:
        # Direction-reversal layered specialist.  Low posture + forecast sign
        # flip means the current economy target may be the wrong direction to
        # chase, so hold can save pump.  Once posture leaves the safe band, the
        # same forecast becomes a warning context and the hold is released.
        args.primary_hold_target_mode = "forecast_pause"
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = True
        args.economy_pump_budget_m3 = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_BUDGET_M3", 60.0)
        )
        if primary_control_profile in {
            "direction_reversal_layered_v5",
            "direction_reversal_layered_v6",
            "direction_reversal_layered_v7",
            "direction_reversal_layered_v8",
            "direction_reversal_layered_v8_holdgate_v1",
            "direction_reversal_layered_v8_holdgate_off_v1",
        }:
            args.economy_pump_budget_allocator_mode = (
                {
                    "direction_reversal_layered_v5": "direction_reversal_layered_v5_auto",
                    "direction_reversal_layered_v6": "direction_reversal_layered_v6_auto",
                    "direction_reversal_layered_v7": "direction_reversal_layered_v7_auto",
                    "direction_reversal_layered_v8": "direction_reversal_layered_v8_auto",
                    "direction_reversal_layered_v8_holdgate_v1": "direction_reversal_layered_v8_auto",
                    "direction_reversal_layered_v8_holdgate_off_v1": "direction_reversal_layered_v8_auto",
                }[primary_control_profile]
            )
        else:
            args.economy_pump_budget_allocator_mode = "direction_reversal_layered_auto"
        args.economy_pump_budget_forecast_smart = True
        args.economy_pump_budget_smart_posture_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_LAYERED_RELEASE_DEG", 2.5)
        )
        args.economy_pump_budget_smart_pressure_norm = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_PRESSURE_NORM", 1.05)
        )
        args.hold_comfort_release = True
        if primary_control_profile == "direction_reversal_layered_v4":
            default_enter = 2.5
            default_exit = 2.0
            default_action = "active_medium"
        else:
            default_enter = 1.8
            default_exit = 1.4
            default_action = "active_small"
        args.hold_comfort_enter_pitch_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ENTER_DEG", default_enter)
        )
        args.hold_comfort_enter_roll_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ENTER_DEG", default_enter)
        )
        args.hold_comfort_exit_pitch_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_EXIT_DEG", default_exit)
        )
        args.hold_comfort_exit_roll_deg = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_EXIT_DEG", default_exit)
        )
        args.hold_comfort_action_name = str(
            os.environ.get("FOWT_DIRECTION_REVERSAL_COMFORT_ACTION", default_action)
        )
        args.hold_comfort_forecast_veto = False
        args.hold_comfort_veto_posture_specific_only = True
        args.hold_comfort_veto_posture_relief_norm = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_POSTURE_RELIEF_NORM", 0.15)
        )
        args.hold_comfort_veto_max_s = float(
            os.environ.get("FOWT_DIRECTION_REVERSAL_VETO_MAX_S", 180.0)
        )
        if primary_control_profile in {
            "direction_reversal_layered_v8_holdgate_v1",
            "direction_reversal_layered_v8_holdgate_off_v1",
        }:
            args.direction_reversal_hold_current_gate_mode = (
                "gate_on"
                if primary_control_profile == "direction_reversal_layered_v8_holdgate_v1"
                else "gate_off"
            )
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
    if primary_control_profile in {
        "forecast_advised_economy_mode_v1",
        "forecast_advised_economy_latched_suppression_v1",
        "forecast_advised_economy_mode_v2",
        "forecast_advised_economy_guarded_v1",
        "forecast_advised_economy_guarded_stale_v1",
        "forecast_advised_economy_continuous_guard_v1",
        "forecast_advised_economy_relaxed_v1",
        "forecast_advised_economy_runaway_release_v1",
        "forecast_advised_economy_shadow_opportunity_v1",
        "forecast_advised_economy_guarded_holdconfirm_v1",
        "forecast_advised_economy_latched_v1",
        "forecast_advised_economy_clean_v1",
        "forecast_advised_economy_lockout_v1",
        "forecast_advised_economy_balanced_v1",
        "neutral_mixed_steady_suppression_v1",
    }:
        # Broad, operator-enabled economy mode for pump-active neutral/moderate
        # regimes.  Forecast is used as a permissive launch key and boundary
        # veto, while the v1.6 hard floor/fallback/recovery remain untouched.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_TUNE_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_TUNE_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_TUNE_WATT", 1.0))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_TUNE_WTERM", 1.0))
        args.planner_posture_hold_forecast_credit = 0.0
        args.economy_pump_budget = False
        args.pump_suppression = True
        args.pump_suppression_plateau_only = True
        args.pump_suppression_restart_err_kg = float(
            os.environ.get("FOWT_FORECAST_ADVISED_RESTART_ERR_KG", 2500.0)
        )
        args.forecast_advised_economy = True
        mixed_steady_profile = (
            primary_control_profile == "neutral_mixed_steady_suppression_v1"
        )
        args.forecast_advised_economy_min_headroom_deg = float(
            os.environ.get(
                "FOWT_FORECAST_ADVISED_MIN_HEADROOM_DEG",
                0.35 if not mixed_steady_profile else 0.30,
            )
        )
        args.forecast_advised_economy_rise_norm = float(
            os.environ.get("FOWT_FORECAST_ADVISED_RISE_NORM", 0.20)
        )
        if primary_control_profile in {
            "forecast_advised_economy_clean_v1",
            "forecast_advised_economy_lockout_v1",
            "forecast_advised_economy_balanced_v1",
            "forecast_advised_economy_latched_v1",
        }:
            lockout_profile = primary_control_profile == "forecast_advised_economy_lockout_v1"
            balanced_profile = (
                primary_control_profile == "forecast_advised_economy_balanced_v1"
            )
            latched_profile = primary_control_profile == "forecast_advised_economy_latched_v1"
            args.forecast_advised_economy_max_speed_range_ms = float(
                os.environ.get("FOWT_FORECAST_ADVISED_MAX_SPEED_RANGE_MS", 2.5)
            )
            args.forecast_advised_economy_max_dir_shift_deg = float(
                os.environ.get("FOWT_FORECAST_ADVISED_MAX_DIR_SHIFT_DEG", 20.0)
            )
            args.forecast_advised_economy_confirm_buckets = int(
                os.environ.get("FOWT_FORECAST_ADVISED_CONFIRM_BUCKETS", 2)
            )
            args.forecast_advised_economy_min_axis_deg = float(
                os.environ.get("FOWT_FORECAST_ADVISED_MIN_AXIS_DEG", 0.0)
            )
            args.forecast_advised_economy_min_pressure_norm = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_MIN_PRESSURE_NORM",
                    0.65
                    if lockout_profile or latched_profile
                    else (0.60 if balanced_profile else 0.0),
                )
            )
            args.forecast_advised_economy_posture_release_deg = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_POSTURE_RELEASE_DEG",
                    3.8 if balanced_profile else 3.0,
                )
            )
            args.forecast_advised_economy_release_cooldown_s = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_RELEASE_COOLDOWN_S",
                    7200.0 if lockout_profile else (600.0 if balanced_profile else 0.0),
                )
            )
            args.forecast_advised_economy_latch_s = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_LATCH_S",
                    1800.0 if latched_profile else 0.0,
                )
            )
            args.forecast_advised_economy_current_ws_range_ms = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_CURRENT_WS_RANGE_MS",
                    1.5 if lockout_profile else (2.0 if balanced_profile else 0.0),
                )
            )
            args.forecast_advised_economy_current_dir_shift_deg = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_CURRENT_DIR_SHIFT_DEG",
                    15.0 if lockout_profile else (20.0 if balanced_profile else 0.0),
                )
            )
            args.forecast_advised_economy_current_history_s = float(
                os.environ.get("FOWT_FORECAST_ADVISED_CURRENT_HISTORY_S", 1800.0)
            )
            if primary_control_profile == "forecast_advised_economy_clean_v1":
                args.forecast_advised_economy_high_stable_clean = True
        if mixed_steady_profile:
            args.forecast_advised_economy_max_speed_range_ms = float(
                os.environ.get("FOWT_MIXED_STEADY_MAX_SPEED_RANGE_MS", 3.0)
            )
            args.forecast_advised_economy_max_dir_shift_deg = float(
                os.environ.get("FOWT_MIXED_STEADY_MAX_DIR_SHIFT_DEG", 20.0)
            )
            args.forecast_advised_economy_min_peak_drop_ms = float(
                os.environ.get("FOWT_MIXED_STEADY_MIN_PEAK_DROP_MS", 0.35)
            )
        if primary_control_profile == "forecast_advised_economy_mode_v2":
            args.forecast_advised_economy_mild = True
            args.forecast_advised_economy_mild_min_headroom_deg = float(
                os.environ.get("FOWT_FORECAST_ADVISED_MILD_MIN_HEADROOM_DEG", 0.10)
            )
            args.forecast_advised_economy_mild_rise_norm = float(
                os.environ.get("FOWT_FORECAST_ADVISED_MILD_RISE_NORM", 0.35)
            )
            args.forecast_advised_economy_mild_restart_err_kg = float(
                os.environ.get("FOWT_FORECAST_ADVISED_MILD_RESTART_ERR_KG", 900.0)
            )
        if primary_control_profile in {
            "forecast_advised_economy_guarded_v1",
            "forecast_advised_economy_guarded_stale_v1",
            "forecast_advised_economy_continuous_guard_v1",
            "forecast_advised_economy_relaxed_v1",
            "forecast_advised_economy_runaway_release_v1",
            "forecast_advised_economy_shadow_opportunity_v1",
            "forecast_advised_economy_guarded_holdconfirm_v1",
        }:
            default_confirm = (
                1
                if primary_control_profile
                == "forecast_advised_economy_guarded_holdconfirm_v1"
                else 2
            )
            args.forecast_advised_economy_confirm_buckets = int(
                os.environ.get("FOWT_FORECAST_ADVISED_CONFIRM_BUCKETS", default_confirm)
            )
            args.forecast_advised_economy_hold_confirm_buckets = int(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_HOLD_CONFIRM_BUCKETS",
                    2
                    if primary_control_profile
                    == "forecast_advised_economy_guarded_holdconfirm_v1"
                    else 0,
                )
            )
            args.forecast_advised_economy_posture_release_deg = float(
                os.environ.get("FOWT_FORECAST_ADVISED_POSTURE_RELEASE_DEG", 4.8)
            )
            args.forecast_advised_economy_release_cooldown_s = float(
                os.environ.get("FOWT_FORECAST_ADVISED_RELEASE_COOLDOWN_S", 1200.0)
            )
            if primary_control_profile == "forecast_advised_economy_continuous_guard_v1":
                # Continuous-run variant: the 1h positive-window P2 profile was
                # too permissive in ordinary/background and ramp/event boundary
                # traces.  Keep the same suppression interface, but require a
                # material mid/high pressure opportunity and let event-risk
                # forecasts veto suppression before it creates catch-up pump.
                args.primary_hold_target_mode = "forecast_pause"
                args.forecast_advised_economy_hold_current_on_candidate = False
                args.forecast_advised_economy_suppression_only = True
                args.forecast_advised_economy_min_pressure_norm = float(
                    os.environ.get("FOWT_CONT_P2_MIN_PRESSURE_NORM", 0.55)
                )
                args.pump_suppression_event_risk_guard = True
                args.pump_suppression_event_risk_guard_threshold = float(
                    os.environ.get("FOWT_CONT_P2_EVENT_RISK_THRESHOLD", 0.70)
                )
            if primary_control_profile == "forecast_advised_economy_relaxed_v1":
                # P2 relaxed variant for the updated attitude semantics.  Long
                # 24h probes showed that owning a forecast primary target
                # accumulates pump debt, so this profile now uses forecast
                # only as a low-gain suppression signal and releases the
                # target-owning channel back to PI.
                args.pump_suppression = False
                args.pump_suppression_plateau = False
                args.pump_suppression_plateau_only = False
                args.pump_suppression_event_risk_guard = False
                args.pump_suppression_restart_err_kg = 2000.0
                args.pump_suppression_relief_margin_norm = 0.25
                args.pump_suppression_low_risk_norm = 0.75
                args.pump_suppression_plateau_high_norm = 1.05
                args.pump_suppression_plateau_range_norm = 0.15
                args.pump_suppression_plateau_rise_norm = 0.10
                args.pump_suppression_plateau_restart_err_kg = 2000.0
                args.pump_suppression_event_risk_guard_threshold = 0.70
                args.primary_hold_target_mode = "forecast_pause"
                args.forecast_advised_economy_hold_current_on_candidate = False
                args.forecast_advised_economy_suppression_only = True
                args.forecast_advised_economy_min_pressure_norm = float(
                    os.environ.get("FOWT_RELAXED_P2_MIN_PRESSURE_NORM", 0.55)
                )
                args.forecast_advised_economy_posture_release_deg = float(
                    os.environ.get("FOWT_RELAXED_P2_POSTURE_RELEASE_DEG", 5.2)
                )
                args.forecast_advised_economy_release_cooldown_s = float(
                    os.environ.get("FOWT_RELAXED_P2_RELEASE_COOLDOWN_S", 900.0)
                )
                args.forecast_pause_resume_max_target_err_kg = float(
                    os.environ.get("FOWT_RELAXED_P2_RESUME_MAX_TARGET_ERR_KG", 5000.0)
                )
            if primary_control_profile == "forecast_advised_economy_runaway_release_v1":
                args.forecast_advised_economy_runaway_release_enabled = True
                args.forecast_advised_economy_runaway_pump_rate_m3_min = float(
                    os.environ.get("FOWT_P2_RUNAWAY_PUMP_RATE_M3_MIN", 8.0)
                )
                args.forecast_advised_economy_runaway_backlog_kg = float(
                    os.environ.get("FOWT_P2_RUNAWAY_BACKLOG_KG", 1200.0)
                )
                args.forecast_advised_economy_runaway_cooldown_s = float(
                    os.environ.get("FOWT_P2_RUNAWAY_COOLDOWN_S", 1200.0)
                )
            if primary_control_profile == "forecast_advised_economy_shadow_opportunity_v1":
                args.forecast_advised_economy_shadow_opportunity_enabled = True
                args.forecast_advised_economy_shadow_min_latch_delta = int(
                    os.environ.get("FOWT_P2_SHADOW_MIN_LATCH_DELTA", 20)
                )
                args.forecast_advised_economy_shadow_min_target_motion_kg_s = float(
                    os.environ.get("FOWT_P2_SHADOW_MIN_TARGET_MOTION_KG_S", 30.0)
                )
                args.forecast_advised_economy_shadow_min_pump_rate_m3_min = float(
                    os.environ.get("FOWT_P2_SHADOW_MIN_PUMP_RATE_M3_MIN", 2.0)
                )
                args.forecast_advised_economy_shadow_min_backlog_kg = float(
                    os.environ.get("FOWT_P2_SHADOW_MIN_BACKLOG_KG", 500.0)
                )
                args.forecast_advised_economy_shadow_latch_s = float(
                    os.environ.get("FOWT_P2_SHADOW_LATCH_S", 1800.0)
                )
            if primary_control_profile in {
                "forecast_advised_economy_guarded_stale_v1",
                "forecast_advised_economy_continuous_guard_v1",
                "forecast_advised_economy_relaxed_v1",
                "forecast_advised_economy_runaway_release_v1",
                "forecast_advised_economy_shadow_opportunity_v1",
            }:
                args.stale_active_target_refresh = "refresh_only_guarded_v1"
                args.stale_active_target_refresh_min_active_buckets = int(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_MIN_ACTIVE_BUCKETS", 2)
                )
                args.stale_active_target_refresh_pitch_deg = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_PITCH_DEG", 3.5)
                )
                args.stale_active_target_refresh_roll_deg = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_ROLL_DEG", 3.5)
                )
                args.stale_active_target_refresh_target_err_kg = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_TARGET_ERR_KG", 2500.0)
                )
                args.stale_active_target_refresh_pump_rate_m3_min = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_PUMP_RATE_M3_MIN", 1.0)
                )
                args.stale_active_target_refresh_min_target_age_s = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_MIN_TARGET_AGE_S", 600.0)
                )
                args.stale_active_target_refresh_min_delta_kg = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_MIN_DELTA_KG", 1000.0)
                )
                args.stale_active_target_refresh_cooldown_s = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_COOLDOWN_S", 1200.0)
                )
                args.stale_active_target_refresh_no_recovery_eps_deg = float(
                    os.environ.get("FOWT_STALE_ACTIVE_TARGET_REFRESH_NO_RECOVERY_EPS_DEG", 0.10)
                )
        args.economy_pump_budget_smart_posture_deg = 4.7
        primary_target_shape_cfg = primary_safety_override(str(args.primary_safety_profile))
        primary_target_shape_cfg["preview_pump_suppression_allow_latched"] = True
        if primary_control_profile in {
            "forecast_advised_economy_latched_suppression_v1",
            "forecast_advised_economy_relaxed_v1",
        }:
            primary_target_shape_cfg[
                "preview_pump_suppression_allow_forecast_advised_latched"
            ] = True
        if primary_control_profile in {
            "forecast_advised_economy_clean_v1",
            "forecast_advised_economy_lockout_v1",
            "forecast_advised_economy_relaxed_v1",
        }:
            primary_target_shape_cfg[
                "preview_pump_suppression_forecast_advised_max_axis_deg"
            ] = float(
                os.environ.get(
                    "FOWT_FORECAST_ADVISED_SUPPRESSION_MAX_AXIS_DEG",
                    5.2
                    if primary_control_profile == "forecast_advised_economy_relaxed_v1"
                    else 3.0,
                )
            )
        if (
            primary_control_profile == "forecast_advised_economy_relaxed_v1"
            and _env_bool("FOWT_RELAXED_P2_SAFE_FREEZE", False)
        ):
            primary_target_shape_cfg["safe_pump_freeze_enabled"] = True
            primary_target_shape_cfg["safe_pump_freeze_enter_axis_deg"] = float(
                os.environ.get("FOWT_RELAXED_P2_SAFE_FREEZE_ENTER_DEG", 1.0)
            )
            primary_target_shape_cfg["safe_pump_freeze_exit_axis_deg"] = float(
                os.environ.get("FOWT_RELAXED_P2_SAFE_FREEZE_EXIT_DEG", 1.4)
            )
            primary_target_shape_cfg["safe_pump_freeze_min_delta_kg"] = float(
                os.environ.get("FOWT_RELAXED_P2_SAFE_FREEZE_MIN_DELTA_KG", 0.0)
            )
    if primary_control_profile == "h120_safety_watch_v1":
        # Default-off warning / safety-margin profile, not a pump-saving mode.
        # It preserves the v1.6 hard floor, fallback, recovery, and pump
        # penalty. Future high-risk forecast is only used as an early-warning
        # key for a small pre-floor posture refresh and a bounded axis micro
        # nudge before the hard floor would otherwise take ownership.
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_FLOOR_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_FLOOR_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_FLOOR_WATT", 0.30))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_FLOOR_WTERM", 0.30))
        args.planner_posture_hold_forecast_credit = 0.0
        args.far_horizon = True
        args.h120_risk_scheduler = True
        args.h120_scheduler_delay_normal_s = 1200.0
        args.h120_scheduler_delay_risk_aware_s = 900.0
        args.h120_scheduler_delay_high_risk_s = 600.0
        args.h120_scheduler_far_high_norm = 0.90
        args.h120_scheduler_intensify_margin_norm = 0.30
        args.h120_scheduler_relief_margin_norm = 0.25
        args.h120_scheduler_early_stop_suppress_max_per_episode = 1
        args.h120_scheduler_prefloor_probe = True
        args.h120_scheduler_prefloor_action = "active_small"
        args.h120_scheduler_prefloor_enter_deg = 3.0
        args.h120_scheduler_prefloor_floor_deg = 5.0
        args.h120_scheduler_prefloor_min_delta_kg = 500.0
        args.h120_scheduler_prefloor_max_delta_kg = 1500.0
        args.h120_scheduler_prefloor_cooldown_s = 1200.0
        args.h120_axis_micro = True
        args.h120_axis_micro_enter_deg = 4.0
        args.h120_axis_micro_floor_deg = 5.0
        args.h120_axis_micro_rise_floor_norm = 0.80
        args.h120_axis_micro_worsening_eps_deg = 0.05
        args.h120_axis_micro_delta_kg = 2000.0
        args.h120_axis_micro_cooldown_s = 600.0
    if primary_control_profile == "safety_floor_v1":
        # Same overlays as rawenv_holdpause_barrier_reliefcap_adaptive_v1 plus
        # hard safety_floor at 5deg IEC. Lexicographic cost: pump dominant,
        # attitude_residual kept as small tie-break (0.3) so the planner
        # prefers feasible solutions with margin from the safety floor under
        # forecast noise (Codex review 2026-05-17).
        args.planner_posture_state_residual = True
        args.planner_posture_state_gain = float(os.environ.get("FOWT_FLOOR_GAIN", 0.40))
        args.planner_posture_state_clip_norm = float(os.environ.get("FOWT_FLOOR_CLIP", 6.0))
        args.planner_attitude_residual_weight = float(os.environ.get("FOWT_FLOOR_WATT", 0.30))
        args.planner_terminal_residual_weight = float(os.environ.get("FOWT_FLOOR_WTERM", 0.30))
        args.planner_posture_hold_forecast_credit = 0.0
    if primary_control_profile == "rawenv_pirelease_barrier_reliefcap_adaptive_v1":
        args.primary_hold_target_mode = "pi_release"
    if primary_control_profile == "rawenv_holdpause_barrier_reliefcap030_v1":
        args.relief_medium_cap = True
        args.relief_medium_cap_ratio = 0.30
        args.relief_medium_cap_event_threshold = 0.70
    if primary_control_profile in {
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "rawenv_holdcurrent_barrier_reliefcap_adaptive_v1",
        "rawenv_forecastsafe_deadband_candidate_v1",
        "rawenv_forecastveto_deadband_candidate_v1",
        "rawenv_forecastveto_deadband_fullsync_legacy_v1",
        "rawenv_forecastgated_command_deadband_v1",
        "rawenv_pirelease_barrier_reliefcap_adaptive_v1",
        "rawenv_holdpause_barrier_reliefcap_state_v1",
        "rawenv_forecastpause_barrier_reliefcap_adaptive_v1",
        "regime_auto_pump_saving_v1",
        "relief_decay_smart_release_v1",
        "relief_decay_dispatcher_v2",
        "relief_decay_episode_auto_v1",
        "gusty_oscillation_smoothing_v1",
        "gusty_oscillation_budget_auto_v1",
        "direction_reversal_avoidance_v2",
        "direction_reversal_layered_v3",
        "direction_reversal_layered_v4",
        "direction_reversal_layered_v5",
        "direction_reversal_layered_v6",
        "direction_reversal_layered_v7",
        "direction_reversal_layered_v8",
        "direction_reversal_layered_v8_holdgate_v1",
        "direction_reversal_layered_v8_holdgate_off_v1",
        "safety_floor_v1",
    }:
        args.relief_medium_cap = True
        args.relief_medium_cap_ratio = 0.30
        args.relief_medium_cap_event_threshold = 0.70
        args.relief_medium_cap_adaptive = True
    if primary_control_profile == "rawenv_forecastpause_barrier_reliefcap_adaptive_v1":
        args.primary_hold_target_mode = "forecast_pause"
    if primary_control_profile == "rawenv_holdpause_barrier_reliefcap_state_v1":
        # Retained as a diagnostic; the new defaults already enable
        # posture_state_residual_active=True so this profile only differs from
        # the main line by explicit historical labeling.
        args.planner_posture_state_residual = True
    # NOTE: 14 abandoned profile branches pruned 2026-05-11 (see
    # configs/planner_overlay_audit.md). Their REDUNDANT overlays (preview_lead,
    # hold_comfort_*, posture_veto, hold_risk_micro, hold_feedback_lite,
    # evidence_*, quiet_posture_*, sequence_lead) are subsumed by the new
    # state-feedback cost defaults (w_attitude_residual=1.0,
    # posture_state_residual_active=True). The underlying mechanisms remain
    # in provider code as default-off diagnostics; only the profile bundles
    # are removed.
    # Named profiles supply defaults. Values explicitly supplied on the
    # command line remain authoritative so an experiment cannot silently run
    # with a profile value different from the requested one.
    for dest, value in explicit_cli_overrides.items():
        setattr(args, str(dest), value)

    return ProfileResolution(
        control_profile=str(primary_control_profile),
        registry_group=str(primary_profile_registry_group),
        target_shape=dict(primary_target_shape_cfg),
    )
