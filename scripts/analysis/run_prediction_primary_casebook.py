#!/usr/bin/env python3
"""Prediction-primary casebook: 10 diverse 1h visual comparisons.

Each case runs closed_only and prediction_primary_econ for one hour and writes
a pitch/roll/pump-rate comparison figure. This is a short visual audit, not a
long validation campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from wind_prediction.ballast_planner import PlannerConfig, action_vec, tank_signal
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.provider_factory import build_forecast_assisted_provider
from wind_prediction.casebook_profile_resolver import resolve_primary_control_profile
from wind_prediction.casebook_profile_gate import (
    CASEBOOK_PROFILE_REGISTRY_RELATIVE,
    ISOLATED_PROFILE_UNLOCK_ENV,
    enforce_casebook_profile_gate,
)
from wind_prediction.forecast_adapter import (
    BlendedForecastAdapter,
    CurrentOnlyForecastAdapter,
    ForecastModelAdapter,
    PersistenceMeanForecastAdapter,
)
from wind_prediction.execution_rollout import ExecutionRolloutConfig
from wind_prediction.attitude_metrics import (
    ATTITUDE_EXPOSURE_METRIC_KEYS,
    attitude_metric_semantics_markdown_rows,
    compute_attitude_exposure_metrics,
)
from wind_prediction.input_files import resolve_mooring_stiffness_file
from wind_prediction.planner_runtime_config import (
    load_planner_runtime_config,
    validate_execution_target_refresh_contract,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset
from wind_prediction.run_protocol import (
    build_casebook_run_identity,
    build_casebook_run_protocol,
    write_run_identity,
    write_run_protocol,
)
from wind_prediction.validation_protocol import enforce_development_experiment_bounds


DT = 1.0
N_STEPS = 3600
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
plt = None
CASEBOOK_PROFILE_REGISTRY_PATH = repo_root / CASEBOOK_PROFILE_REGISTRY_RELATIVE
PRUNED_OVERLAY_FLAG_GUARDS = {
    "event_risk_pressure_boost": "--event-risk-pressure-boost",
    "event_risk_pressure_floor": "--event-risk-pressure-floor",
    "preview_lead_action": "--preview-lead-action",
    "sequence_lead_action": "--sequence-lead-action",
    "hold_risk_micro_action": "--hold-risk-micro-action",
    "quiet_posture_action": "--quiet-posture-action",
    "posture_action_refresh": "--posture-action-refresh",
    "no_unexplained_hold_action": "--no-unexplained-hold-action",
    "medium_escalation": "--medium-escalation",
    "recovery_mode": "--recovery-mode",
    "planner_posture_hold_barrier": "--planner-posture-hold-barrier",
    "planner_posture_hold_action_uses_posture_vec": "--planner-posture-hold-action-uses-posture-vec",
    "active_intent_reproposal": "--active-intent-reproposal",
    "active_intent_reproposal_eligibility": "--active-intent-reproposal-eligibility",
    "pump_suppression": "--pump-suppression",
    "pump_suppression_plateau": "--pump-suppression-plateau",
    "pump_suppression_plateau_only": "--pump-suppression-plateau-only",
    "pump_suppression_event_risk_guard": "--pump-suppression-event-risk-guard",
    "hold_comfort_release": "--hold-comfort-release",
    "hold_comfort_no_preview_only": "--hold-comfort-no-preview-only",
    "hold_comfort_forecast_veto": "--hold-comfort-forecast-veto",
    "hold_comfort_veto_posture_specific_only": "--hold-comfort-veto-posture-specific-only",
}
PRUNED_OVERLAY_VALUE_GUARDS = {
    "event_risk_pressure_boost_gain": ("--event-risk-pressure-boost-gain", 0.35),
    "event_risk_pressure_boost_threshold": (
        "--event-risk-pressure-boost-threshold",
        0.50,
    ),
    "event_risk_pressure_boost_max": ("--event-risk-pressure-boost-max", 0.35),
    "event_risk_pressure_floor_threshold": (
        "--event-risk-pressure-floor-threshold",
        0.70,
    ),
    "event_risk_pressure_floor_norm": ("--event-risk-pressure-floor-norm", 0.85),
    "event_risk_pressure_floor_max_lift": (
        "--event-risk-pressure-floor-max-lift",
        0.50,
    ),
    "preview_lead_event_threshold": ("--preview-lead-event-threshold", 0.90),
    "preview_lead_current_norm_max": ("--preview-lead-current-norm-max", 0.65),
    "preview_lead_future_norm_min": ("--preview-lead-future-norm-min", 1.00),
    "preview_lead_current_wind_max_ms": (
        "--preview-lead-current-wind-max-ms",
        8.0,
    ),
    "preview_lead_pitch_abs_max_deg": ("--preview-lead-pitch-abs-max-deg", 2.0),
    "preview_lead_roll_abs_max_deg": ("--preview-lead-roll-abs-max-deg", 1.5),
    "sequence_lead_pitch_abs_min_deg": (
        "--sequence-lead-pitch-abs-min-deg",
        2.0,
    ),
    "sequence_lead_roll_abs_min_deg": (
        "--sequence-lead-roll-abs-min-deg",
        2.0,
    ),
    "sequence_lead_min_future_norm": ("--sequence-lead-min-future-norm", 0.95),
    "hold_risk_micro_event_threshold": ("--hold-risk-micro-event-threshold", 0.80),
    "hold_risk_micro_pressure_norm_min": (
        "--hold-risk-micro-pressure-norm-min",
        1.00,
    ),
    "hold_risk_micro_pitch_abs_min_deg": (
        "--hold-risk-micro-pitch-abs-min-deg",
        0.00,
    ),
    "hold_risk_micro_roll_abs_min_deg": (
        "--hold-risk-micro-roll-abs-min-deg",
        4.50,
    ),
    "hold_risk_micro_pitch_abs_max_deg": (
        "--hold-risk-micro-pitch-abs-max-deg",
        2.00,
    ),
    "hold_risk_micro_roll_abs_max_deg": (
        "--hold-risk-micro-roll-abs-max-deg",
        0.00,
    ),
    "no_unexplained_hold_threshold_deg": (
        "--no-unexplained-hold-threshold-deg",
        3.0,
    ),
    "no_unexplained_hold_trigger_buckets": (
        "--no-unexplained-hold-trigger-buckets",
        2.0,
    ),
    "no_unexplained_hold_reset_buckets": (
        "--no-unexplained-hold-reset-buckets",
        1.0,
    ),
    "no_unexplained_hold_target_err_kg": (
        "--no-unexplained-hold-target-err-kg",
        1500.0,
    ),
    "no_unexplained_hold_pump_rate_m3_min": (
        "--no-unexplained-hold-pump-rate-m3-min",
        0.5,
    ),
    "active_intent_reproposal_min_delta_kg": (
        "--active-intent-reproposal-min-delta-kg",
        1500.0,
    ),
    "active_intent_reproposal_eligibility_pitch_deg": (
        "--active-intent-reproposal-eligibility-pitch-deg",
        4.0,
    ),
    "active_intent_reproposal_eligibility_roll_deg": (
        "--active-intent-reproposal-eligibility-roll-deg",
        3.0,
    ),
    "active_intent_reproposal_eligibility_target_err_kg": (
        "--active-intent-reproposal-eligibility-target-err-kg",
        2500.0,
    ),
    "active_intent_reproposal_eligibility_pump_rate_m3_min": (
        "--active-intent-reproposal-eligibility-pump-rate-m3-min",
        1.0,
    ),
    "planner_posture_hold_pitch_deg": ("--planner-posture-hold-pitch-deg", 3.0),
    "planner_posture_hold_roll_deg": ("--planner-posture-hold-roll-deg", 3.0),
    "planner_posture_hold_barrier_const": (
        "--planner-posture-hold-barrier-const",
        10.0,
    ),
    "planner_posture_hold_low_risk_norm_max": (
        "--planner-posture-hold-low-risk-norm-max",
        0.0,
    ),
    "pump_suppression_restart_err_kg": ("--pump-suppression-restart-err-kg", 2000.0),
    "pump_suppression_relief_margin_norm": (
        "--pump-suppression-relief-margin-norm",
        0.25,
    ),
    "pump_suppression_low_risk_norm": ("--pump-suppression-low-risk-norm", 0.75),
    "pump_suppression_plateau_high_norm": (
        "--pump-suppression-plateau-high-norm",
        1.05,
    ),
    "pump_suppression_plateau_range_norm": (
        "--pump-suppression-plateau-range-norm",
        0.15,
    ),
    "pump_suppression_plateau_rise_norm": (
        "--pump-suppression-plateau-rise-norm",
        0.10,
    ),
    "pump_suppression_plateau_restart_err_kg": (
        "--pump-suppression-plateau-restart-err-kg",
        2000.0,
    ),
    "pump_suppression_event_risk_guard_threshold": (
        "--pump-suppression-event-risk-guard-threshold",
        0.70,
    ),
    "hold_comfort_enter_pitch_deg": ("--hold-comfort-enter-pitch-deg", 2.5),
    "hold_comfort_enter_roll_deg": ("--hold-comfort-enter-roll-deg", 2.5),
    "hold_comfort_exit_pitch_deg": ("--hold-comfort-exit-pitch-deg", 2.0),
    "hold_comfort_exit_roll_deg": ("--hold-comfort-exit-roll-deg", 2.0),
    "hold_comfort_rate_eps_deg_s": ("--hold-comfort-rate-eps-deg-s", 0.002),
    "hold_comfort_update_interval_s": ("--hold-comfort-update-interval-s", 60.0),
    "hold_comfort_min_hold_s": ("--hold-comfort-min-hold-s", 0.0),
    "hold_comfort_min_release_s": ("--hold-comfort-min-release-s", 0.0),
    "hold_comfort_pressure_max_norm": ("--hold-comfort-pressure-max-norm", 0.0),
    "hold_comfort_current_pressure_max_norm": (
        "--hold-comfort-current-pressure-max-norm",
        0.0,
    ),
    "hold_comfort_veto_event_threshold": ("--hold-comfort-veto-event-threshold", 0.70),
    "hold_comfort_veto_relief_margin_norm": (
        "--hold-comfort-veto-relief-margin-norm",
        0.25,
    ),
    "hold_comfort_veto_posture_relief_norm": (
        "--hold-comfort-veto-posture-relief-norm",
        0.20,
    ),
    "hold_comfort_veto_low_risk_norm": ("--hold-comfort-veto-low-risk-norm", 0.75),
    "hold_comfort_veto_low_risk_posture_max_deg": (
        "--hold-comfort-veto-low-risk-posture-max-deg",
        0.0,
    ),
    "hold_comfort_veto_max_s": ("--hold-comfort-veto-max-s", 900.0),
}
PRUNED_OVERLAY_TEXT_GUARDS = {
    "preview_lead_action_name": ("--preview-lead-action-name", "active_small"),
    "sequence_lead_action_name": ("--sequence-lead-action-name", "pump_saving"),
    "hold_risk_micro_action_name": ("--hold-risk-micro-action-name", "pump_saving"),
    "hold_comfort_action_name": ("--hold-comfort-action-name", "active_small"),
}
PRUNED_OVERLAY_BOOLEAN_OPTIONAL_GUARDS = {
    "hold_comfort_between_bucket_enabled": (
        "--hold-comfort-between-bucket-enabled",
        True,
    ),
}


class ReactivePrimaryProvider:
    """Same execution shell as prediction-primary, but no forecast horizon.

    This diagnostic provider emits prediction-primary mass targets from the
    measured current pitch/roll only. It is not a wind forecast baseline: it
    does not call a forecast adapter, build future blocks, or repeat current
    wind into the future.
    """

    def __init__(
        self,
        cfg: PlannerConfig,
        action_name: str = "pump_saving",
        enter_pitch_deg: float = 3.0,
        enter_roll_deg: float = 3.0,
        exit_pitch_deg: float = 2.4,
        exit_roll_deg: float = 2.4,
        update_interval_s: float = 600.0,
        prediction_primary_scale: float = 1.0,
    ) -> None:
        if action_name not in {"pump_saving", "active_small", "active_medium"}:
            raise ValueError(f"unsupported reactive primary action {action_name!r}")
        self.cfg = cfg
        self.action_name = str(action_name)
        self.enter_pitch_deg = float(enter_pitch_deg)
        self.enter_roll_deg = float(enter_roll_deg)
        self.exit_pitch_deg = float(exit_pitch_deg)
        self.exit_roll_deg = float(exit_roll_deg)
        self.update_interval_s = max(float(update_interval_s), 1.0)
        self.prediction_primary_scale = float(prediction_primary_scale)
        self.records: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self._last_update_s: float | None = None
        self._latched = False
        self._target_kg = np.zeros(3, dtype=float)
        self._delta_kg = np.zeros(3, dtype=float)
        self._action = "hold"
        self._target_initialized = False
        self.records = []

    def _current_masses(self, plant_info_prev) -> np.ndarray:
        if isinstance(plant_info_prev, dict) and "tank_masses" in plant_info_prev:
            masses = np.asarray(plant_info_prev["tank_masses"], dtype=float).reshape(-1)
            if masses.size >= 3:
                return masses[:3].astype(float)
        return np.zeros(3, dtype=float)

    def _posture_vec_deg(self, state) -> np.ndarray:
        arr = np.asarray(state, dtype=float).reshape(-1)
        if arr.size <= 4:
            return np.zeros(2, dtype=float)
        return np.array([float(np.degrees(arr[4])), float(np.degrees(arr[3]))], dtype=float)

    def _target_from_action(self, masses: np.ndarray, avec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if float(np.linalg.norm(avec)) <= 1e-12:
            delta = np.zeros(3, dtype=float)
        else:
            delta = (
                tank_signal(avec, self.cfg)
                * self.cfg.action_mass_quantum_kg
                * self.prediction_primary_scale
            )
        target = np.clip(masses + delta, 0.0, self.cfg.tank_capacity_kg)
        return target, target - masses

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        del wind_obs
        t = float(current_time)
        masses = self._current_masses(plant_info_prev)
        posture = self._posture_vec_deg(state)
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        enter = pitch_abs >= self.enter_pitch_deg or roll_abs >= self.enter_roll_deg
        exit_ready = pitch_abs <= self.exit_pitch_deg and roll_abs <= self.exit_roll_deg
        if enter:
            self._latched = True
        elif self._latched and exit_ready:
            self._latched = False

        due = (
            self._last_update_s is None
            or t - float(self._last_update_s) >= self.update_interval_s
        )
        refreshed = False
        if due or not self._target_initialized:
            self._last_update_s = t
            refreshed = True
            if self._latched and float(np.linalg.norm(posture)) > 1e-9:
                avec = action_vec(self.action_name, posture, self.cfg)
                target, delta = self._target_from_action(masses, avec)
                self._action = self.action_name
            else:
                target = np.clip(masses.copy(), 0.0, self.cfg.tank_capacity_kg)
                delta = np.zeros(3, dtype=float)
                self._action = "hold"
            self._target_kg = target
            self._delta_kg = delta
            self._target_initialized = True

        active = int(float(np.mean(np.abs(self._delta_kg))) > 1.0)
        self.records.append(
            {
                "current_time_s": t,
                "first_action": self._action,
                "forecast_source": "reactive_primary_no_forecast",
                "forecast_has_future": 0,
                "current_pitch_deg": float(posture[0]),
                "current_roll_deg": float(posture[1]),
                "prediction_primary_target_t1_kg": float(self._target_kg[0]),
                "prediction_primary_target_t2_kg": float(self._target_kg[1]),
                "prediction_primary_target_t3_kg": float(self._target_kg[2]),
                "prediction_primary_delta_abs_mean_kg": float(np.mean(np.abs(self._delta_kg))),
                "prediction_primary_target_refreshed": int(refreshed),
                "prediction_primary_target_reused": int(not refreshed),
                "prediction_primary_target_resumed": 0,
                "prediction_primary_hold_target_mode": "reactive_current",
                "prediction_primary_target_age_s": 0.0
                if self._last_update_s is None
                else t - float(self._last_update_s),
                "target_pitch_deg": 0.0,
                "target_roll_deg": 0.0,
                "preview_reactive_primary_latched": int(self._latched),
                "preview_reactive_primary_enter_pitch_deg": float(self.enter_pitch_deg),
                "preview_reactive_primary_enter_roll_deg": float(self.enter_roll_deg),
                "preview_reactive_primary_exit_pitch_deg": float(self.exit_pitch_deg),
                "preview_reactive_primary_exit_roll_deg": float(self.exit_roll_deg),
            }
        )
        return {
            "pitch_bias_deg": 0.0,
            "roll_bias_deg": 0.0,
            "preview_pitch_bias_deg": 0.0,
            "preview_roll_bias_deg": 0.0,
            "preview_mass_ff_kg": [0.0, 0.0, 0.0],
            "preview_ff_channel_enabled": 0,
            "preview_setpoint_channel_enabled": 0,
            "preview_primary_enabled": 1,
            "preview_primary_active": active,
            "preview_primary_target_kg": [float(x) for x in self._target_kg],
            "preview_primary_delta_kg": [float(x) for x in self._delta_kg],
            "preview_primary_action": self._action,
            "preview_primary_hold_target_mode": "reactive_current",
            "preview_primary_event_reset": int(refreshed),
            "preview_primary_posture_action_refresh": 0,
            "preview_primary_target_refreshed": int(refreshed),
            "preview_primary_target_reused": int(not refreshed),
            "preview_primary_target_resumed": 0,
            "preview_primary_target_age_s": 0.0
            if self._last_update_s is None
            else t - float(self._last_update_s),
            "preview_forecast_has_future": 0,
            "preview_reactive_primary_latched": int(self._latched),
        }


def _require_pyplot():
    global plt
    if plt is not None:
        return plt
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ModuleNotFoundError as exc:  # pragma: no cover - optional for --skip-figures runs
        raise RuntimeError(
            "matplotlib is required for plotting; rerun with --skip-figures "
            "or install matplotlib in this environment"
        ) from exc
    plt = pyplot
    return plt


# Frozen non-preview reactive baselines for closed_only. The default
# engineered_minimal profile is the closed_baseline_v1 reference: ordinary PI
# feedback plus pump actuator constraints, with no forecast or planner access.
CLOSED_BASELINE_V1_PUMP_CFG = {
    "pump_stop_err_kg": 300.0,
    "pump_restart_err_kg": 500.0,
    "pump_min_on_s": 20.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 10.0,
    "pump_global_quiet_hold_s": 0.0,
    "pump_global_quiet_backlog_kg": float("inf"),
    "pump_target_quiet_rate_kg_s": float("inf"),
    "pump_ramp_up_m3_min_per_s": 2.0,
    "pump_ramp_down_m3_min_per_s": 3.0,
}

CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG = {
    **CLOSED_BASELINE_V1_PUMP_CFG,
    "pump_min_off_s": 20.0,
    "pump_hold_before_stop_s": 15.0,
    "pump_ramp_up_m3_min_per_s": 0.5,
    "pump_ramp_down_m3_min_per_s": 0.75,
}

CLOSED_RELAXED_ECONOMY_RAMP1P5_PUMP_CFG = {
    **CLOSED_BASELINE_V1_PUMP_CFG,
    "pump_stop_err_kg": 700.0,
    "pump_restart_err_kg": 1100.0,
    "pump_min_off_s": 20.0,
    "pump_hold_before_stop_s": 15.0,
    "pump_ramp_up_m3_min_per_s": 1.5,
    "pump_ramp_down_m3_min_per_s": 2.25,
}

CLOSED_SMOOTH_ECON700_PUMP_CFG = {
    **CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    "pump_stop_err_kg": 700.0,
    "pump_restart_err_kg": 1100.0,
}

CLOSED_PUMP_PROFILES = {
    "raw": None,
    "formal": CLOSED_BASELINE_V1_PUMP_CFG,
    "engineered_minimal": CLOSED_BASELINE_V1_PUMP_CFG,
    # Display-friendly closed profile. It reduces pump-rate total variation by
    # about 41% in the 5-case check, with <=3.28% pump-work increase and nearly
    # unchanged attitude metrics. The formal paper baseline remains
    # engineered_minimal unless explicitly changed.
    "actuator_smoothed_v2": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    "smooth": CLOSED_ACTUATOR_SMOOTHING_CANDIDATE_PUMP_CFG,
    # Supplemental non-preview references for baseline-fairness checks.
    "relaxed_economy_ramp1p5": CLOSED_RELAXED_ECONOMY_RAMP1P5_PUMP_CFG,
    "smooth_econ700": CLOSED_SMOOTH_ECON700_PUMP_CFG,
}

CLOSED_TARGET_SHAPE_PROFILES = {
    "raw": {},
    "formal": {},
    "engineered_minimal": {},
    "actuator_smoothed_v2": {},
    "smooth": {},
    "relaxed_economy_ramp1p5": {},
    "smooth_econ700": {},
}

PRIMARY_SAFETY_PROFILES = {
    "default": {},
    "bucket_guard": {
        "primary_safety_bucket_guard_enabled": True,
        "primary_safety_bucket_guard_bucket_s": 600.0,
        "primary_safety_bucket_guard_pitch_enter_deg": 4.0,
        "primary_safety_bucket_guard_roll_enter_deg": 4.0,
        "primary_safety_bucket_guard_pitch_exit_deg": 3.5,
        "primary_safety_bucket_guard_roll_exit_deg": 3.5,
        "primary_safety_bucket_guard_improve_tol_deg": 0.0,
        "primary_safety_bucket_guard_exit_required_windows": 1,
        "primary_safety_bucket_guard_max_active_windows": 1,
        "primary_safety_bucket_guard_hold_only": False,
        "primary_safety_bucket_guard_require_current_high": False,
    },
    "risk_gate": {
        "primary_safety_pitch_enter_deg": 4.0,
        "primary_safety_roll_enter_deg": 4.0,
        "primary_safety_pitch_exit_deg": 3.5,
        "primary_safety_roll_exit_deg": 3.5,
        "primary_safety_enter_hold_s": 0.0,
        "primary_safety_exit_hold_s": 0.0,
        "primary_safety_exit_required_windows": 1,
        "primary_safety_emergency_pitch_enter_deg": 8.0,
        "primary_safety_emergency_roll_enter_deg": 6.5,
        "primary_safety_bucket_guard_enabled": True,
        "primary_safety_bucket_guard_bucket_s": 600.0,
        "primary_safety_bucket_guard_pitch_enter_deg": 4.0,
        "primary_safety_bucket_guard_roll_enter_deg": 4.0,
        "primary_safety_bucket_guard_pitch_exit_deg": 3.5,
        "primary_safety_bucket_guard_roll_exit_deg": 3.5,
        "primary_safety_bucket_guard_improve_tol_deg": 0.0,
        "primary_safety_bucket_guard_exit_required_windows": 1,
        "primary_safety_bucket_guard_max_active_windows": 1,
        "primary_safety_bucket_guard_hold_only": False,
        "primary_safety_bucket_guard_require_current_high": False,
    },
    "hold_risk_gate": {
        "primary_safety_pitch_enter_deg": 4.0,
        "primary_safety_roll_enter_deg": 4.0,
        "primary_safety_pitch_exit_deg": 3.5,
        "primary_safety_roll_exit_deg": 3.5,
        "primary_safety_enter_hold_s": 0.0,
        "primary_safety_exit_hold_s": 0.0,
        "primary_safety_exit_required_windows": 1,
        "primary_safety_emergency_pitch_enter_deg": 8.0,
        "primary_safety_emergency_roll_enter_deg": 6.5,
        "primary_safety_hold_risk_gate_enabled": True,
        "primary_safety_hold_risk_gate_action": "hold",
        "primary_safety_bucket_guard_enabled": True,
        "primary_safety_bucket_guard_bucket_s": 600.0,
        "primary_safety_bucket_guard_pitch_enter_deg": 4.0,
        "primary_safety_bucket_guard_roll_enter_deg": 4.0,
        "primary_safety_bucket_guard_pitch_exit_deg": 3.5,
        "primary_safety_bucket_guard_roll_exit_deg": 3.5,
        "primary_safety_bucket_guard_improve_tol_deg": 0.0,
        "primary_safety_bucket_guard_exit_required_windows": 1,
        "primary_safety_bucket_guard_max_active_windows": 1,
        "primary_safety_bucket_guard_hold_only": True,
        "primary_safety_bucket_guard_require_current_high": False,
    },
    "strict": {
        "primary_safety_pitch_enter_deg": 5.5,
        "primary_safety_roll_enter_deg": 4.5,
        "primary_safety_pitch_exit_deg": 4.8,
        "primary_safety_roll_exit_deg": 3.8,
        "primary_safety_enter_hold_s": 10.0,
        "primary_safety_emergency_pitch_enter_deg": 7.5,
        "primary_safety_emergency_roll_enter_deg": 6.5,
    },
    "hold_feedback_lite": {
        "primary_hold_feedback_enabled": True,
        "primary_hold_feedback_enter_pitch_deg": 3.0,
        "primary_hold_feedback_enter_roll_deg": 3.0,
        "primary_hold_feedback_exit_pitch_deg": 2.4,
        "primary_hold_feedback_exit_roll_deg": 2.4,
        "primary_hold_feedback_max_delta_kg": 15000.0,
        "primary_hold_feedback_zero_delta_eps_kg": 1.0,
    },
    "evidence_hold_feedback_lite": {
        "primary_hold_feedback_enabled": True,
        "primary_hold_feedback_enter_pitch_deg": 3.0,
        "primary_hold_feedback_enter_roll_deg": 3.0,
        "primary_hold_feedback_exit_pitch_deg": 2.4,
        "primary_hold_feedback_exit_roll_deg": 2.4,
        "primary_hold_feedback_max_delta_kg": 8000.0,
        "primary_hold_feedback_zero_delta_eps_kg": 1.0,
        "primary_hold_feedback_require_no_future_evidence": True,
        "primary_hold_feedback_future_evidence_max_pause_s": 600.0,
    },
    "evidence_hold_feedback_soft": {
        "primary_hold_feedback_enabled": True,
        "primary_hold_feedback_enter_pitch_deg": 3.0,
        "primary_hold_feedback_enter_roll_deg": 3.0,
        "primary_hold_feedback_exit_pitch_deg": 2.4,
        "primary_hold_feedback_exit_roll_deg": 2.4,
        "primary_hold_feedback_max_delta_kg": 3000.0,
        "primary_hold_feedback_zero_delta_eps_kg": 1.0,
        "primary_hold_feedback_require_no_future_evidence": True,
        "primary_hold_feedback_future_evidence_max_pause_s": 600.0,
    },
}


CASES = [
    ("01_onset_strong", "2024-11-27 19:40:00", "Strong onset: calm now, very strong future wind"),
    ("02_onset_signflip", "2023-10-31 06:20:00", "Onset with direction flip"),
    ("03_onset_moderate", "2024-10-10 03:50:00", "Moderate onset"),
    ("04_decay_strong", "2024-09-27 13:00:00", "Strong decay: high now, weak future wind"),
    ("05_decay_signflip", "2022-03-20 19:00:00", "Decay with direction flip"),
    ("06_signflip_high", "2023-10-03 06:30:00", "High-pressure sign-flip"),
    ("07_signflip_sustained", "2023-03-14 04:40:00", "Sustained high sign-flip"),
    ("08_lowrisk_quiet", "2021-12-20 14:30:00", "Low-risk quiet window"),
    ("09_high_pressure_event", "2022-02-04 11:00:00", "High-pressure high-event normal window"),
    ("10_residual_high", "2024-09-05 18:10:00", "Residual-high normal window"),
    # Fallbacks in case a timestamp is unavailable in the replay dataset.
    ("11_clean_low_pressure", "2023-09-11 03:40:00", "Clean low-pressure normal window"),
    ("12_onset_late", "2024-12-06 08:40:00", "Late onset fallback"),
]


def discover_excel(explicit_path: str | Path | None = None) -> str:
    return str(
        resolve_mooring_stiffness_file(
            explicit_path=explicit_path,
            search_dirs=(
                repo_root / "data",
                repo_root / "archive" / "legacy_fowt_control" / "data",
            ),
        )
    )


def pct_abs(x: np.ndarray, q: float) -> float:
    return float(np.percentile(np.abs(np.asarray(x, dtype=float)), q))


def work_m3(pump: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(pump, dtype=float)), dx=DT) / 60.0)


def closed_pump_cfg(profile: str) -> dict | None:
    cfg = CLOSED_PUMP_PROFILES[str(profile)]
    return None if cfg is None else dict(cfg)


def closed_target_shape_override(profile: str) -> dict:
    return dict(CLOSED_TARGET_SHAPE_PROFILES[str(profile)])


def primary_safety_override(profile: str) -> dict:
    return dict(PRIMARY_SAFETY_PROFILES[str(profile)])


def _register_pruned_overlay_legacy_args(parser: argparse.ArgumentParser) -> None:
    for attr, flag in PRUNED_OVERLAY_FLAG_GUARDS.items():
        parser.add_argument(flag, dest=attr, action="store_true", default=False, help=argparse.SUPPRESS)
    for attr, (flag, default) in PRUNED_OVERLAY_VALUE_GUARDS.items():
        parser.add_argument(flag, dest=attr, type=float, default=float(default), help=argparse.SUPPRESS)
    for attr, (flag, default) in PRUNED_OVERLAY_TEXT_GUARDS.items():
        parser.add_argument(flag, dest=attr, default=str(default), help=argparse.SUPPRESS)
    for attr, (flag, default) in PRUNED_OVERLAY_BOOLEAN_OPTIONAL_GUARDS.items():
        parser.add_argument(
            flag,
            dest=attr,
            action=argparse.BooleanOptionalAction,
            default=bool(default),
            help=argparse.SUPPRESS,
        )


def _enforce_pruned_overlay_flags(args: argparse.Namespace) -> None:
    enabled_flags = [
        flag
        for attr, flag in PRUNED_OVERLAY_FLAG_GUARDS.items()
        if bool(getattr(args, attr, False))
    ]
    changed_values = [
        f"{flag}={float(getattr(args, attr)):.6g}"
        for attr, (flag, default) in PRUNED_OVERLAY_VALUE_GUARDS.items()
        if abs(float(getattr(args, attr, default)) - float(default)) > 1e-12
    ]
    changed_values.extend(
        f"{flag}={getattr(args, attr)!r}"
        for attr, (flag, default) in PRUNED_OVERLAY_TEXT_GUARDS.items()
        if str(getattr(args, attr, default)) != str(default)
    )
    changed_values.extend(
        f"{flag}={getattr(args, attr)!r}"
        for attr, (flag, default) in PRUNED_OVERLAY_BOOLEAN_OPTIONAL_GUARDS.items()
        if bool(getattr(args, attr, default)) != bool(default)
    )
    if not enabled_flags and not changed_values:
        return
    raise ValueError(
        "Frozen redundant overlay option(s) cannot be enabled for new runs: "
        f"{', '.join(enabled_flags + changed_values)}. "
        "See configs/overlay_freeze_registry_v1.json."
    )


def smooth_wind_trace(trace: dict, transition_s: float) -> dict:
    transition_s = float(transition_s)
    if transition_s <= 0.0:
        return trace
    dt = float(trace.get("dt", DT))
    update_s = float(trace.get("update_interval_s", 0.0))
    if update_s <= 0.0:
        return trace
    hold_steps = int(round(update_s / dt))
    if hold_steps <= 1:
        return trace
    trans_steps = min(max(1, int(round(transition_s / dt))), hold_steps)
    ws = np.asarray(trace["ws"], dtype=float).copy()
    wd = np.asarray(trace["wd"], dtype=float).copy()
    n_bucket = int(np.ceil(len(ws) / hold_steps))
    for k in range(1, n_bucket):
        start = k * hold_steps
        if start >= len(ws):
            break
        end = min(start + trans_steps, len(ws))
        prev_ws = float(ws[start - 1])
        prev_wd = float(wd[start - 1])
        curr_ws = float(ws[start])
        curr_wd = float(wd[start])
        wd_delta = ((curr_wd - prev_wd + 180.0) % 360.0) - 180.0
        denom = max(end - start - 1, 1)
        for i, idx in enumerate(range(start, end)):
            frac = 0.5 * (1.0 - np.cos(np.pi * i / denom))
            ws[idx] = prev_ws + frac * (curr_ws - prev_ws)
            wd[idx] = (prev_wd + frac * wd_delta) % 360.0
    out = dict(trace)
    out["ws"] = ws
    out["wd"] = wd
    out["mean_lpf_tau_s"] = float(transition_s)
    out["transition_shape"] = "raised_cosine"
    return out


def summarize(df: pd.DataFrame) -> dict:
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    zeros = pd.Series(np.zeros(len(df)))
    time_s = df["t_s"].to_numpy(dtype=float) if "t_s" in df.columns else None
    out = {
        "pitch_mean": float(np.mean(pitch)),
        "pitch_abs_p50": pct_abs(pitch, 50),
        "pitch_abs_p95": pct_abs(pitch, 95),
        "pitch_abs_max": pct_abs(pitch, 100),
        "roll_mean": float(np.mean(roll)),
        "roll_abs_p50": pct_abs(roll, 50),
        "roll_abs_p95": pct_abs(roll, 95),
        "roll_abs_max": pct_abs(roll, 100),
        "pump_work_m3": work_m3(pump),
        "pump_duty": float(np.mean(np.abs(pump) > 1e-6)),
        "latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
        "primary_delta_mean_kg": float(
            np.mean(np.abs(df.get("preview_primary_delta_mean_kg", zeros).to_numpy(dtype=float)))
        ),
        "primary_candidate_ratio": float(
            np.mean(df.get("preview_primary_candidate_applied", zeros).to_numpy(dtype=float))
        ),
        "primary_applied_ratio": float(
            np.mean(df.get("preview_primary_applied", zeros).to_numpy(dtype=float))
        ),
        "primary_safety_fallback_ratio": float(
            np.mean(df.get("preview_primary_safety_fallback", zeros).to_numpy(dtype=float))
        ),
        "primary_safety_active_ratio": float(
            np.mean(df.get("preview_primary_safety_active", zeros).to_numpy(dtype=float))
        ),
        "primary_safety_transition_count": int(
            np.sum(
                np.abs(
                    np.diff(
                        df.get("preview_primary_safety_active", zeros).to_numpy(dtype=float),
                        prepend=0.0,
                    )
                )
                > 0.5
            )
        ),
        "primary_bucket_guard_active_ratio": float(
            np.mean(df.get("preview_primary_bucket_guard_active", zeros).to_numpy(dtype=float))
        ),
        "primary_bucket_guard_transition_count": int(
            np.sum(
                np.abs(
                    np.diff(
                        df.get("preview_primary_bucket_guard_active", zeros).to_numpy(dtype=float),
                        prepend=0.0,
                    )
                )
                > 0.5
            )
        ),
        "preview_pump_suppression_ratio": float(
            np.mean(df.get("preview_pump_suppression_active", zeros).to_numpy(dtype=float))
        ),
        "preview_suppression_delta_mean_kg": float(
            np.mean(df.get("preview_suppression_delta_mean_kg", zeros).to_numpy(dtype=float))
        ),
        "safe_pump_freeze_ratio": float(
            np.mean(df.get("safe_pump_freeze_active", zeros).to_numpy(dtype=float))
        ),
        "safe_pump_freeze_delta_mean_kg": float(
            np.mean(df.get("safe_pump_freeze_delta_mean_kg", zeros).to_numpy(dtype=float))
        ),
        "preview_event_risk_boost_ratio": float(
            np.mean(df.get("preview_event_risk_pressure_boost_enabled", zeros).to_numpy(dtype=float))
        ),
        "preview_event_risk_scale_mean": float(
            np.mean(
                df.get(
                    "preview_event_risk_scale_0_20m",
                    pd.Series(np.ones(len(df)), index=df.index),
                ).to_numpy(dtype=float)
            )
        ),
        "preview_event_risk_floor_ratio": float(
            np.mean(
                np.max(
                    np.column_stack(
                        [
                            df.get("preview_event_risk_floor_active_0_20m", zeros).to_numpy(dtype=float),
                            df.get("preview_event_risk_floor_active_20_40m", zeros).to_numpy(dtype=float),
                            df.get("preview_event_risk_floor_active_40_60m", zeros).to_numpy(dtype=float),
                        ]
                    ),
                    axis=1,
                )
            )
        ),
        "preview_event_risk_floor_norm_mean": float(
            np.mean(
                np.column_stack(
                    [
                        df.get("preview_event_risk_floor_norm_0_20m", zeros).to_numpy(dtype=float),
                        df.get("preview_event_risk_floor_norm_20_40m", zeros).to_numpy(dtype=float),
                        df.get("preview_event_risk_floor_norm_40_60m", zeros).to_numpy(dtype=float),
                    ]
                )
            )
        ),
        "preview_trusted_event_gate_enabled_ratio": float(
            np.mean(df.get("preview_trusted_event_gate_enabled", zeros).to_numpy(dtype=float))
        ),
        "preview_trusted_event_gate_untrusted_ratio": float(
            np.mean(
                (
                    df.get("preview_trusted_event_gate_enabled", zeros).to_numpy(dtype=float)
                    > 0.5
                )
                & (
                    df.get(
                        "preview_trusted_event_gate_trusted",
                        pd.Series(np.ones(len(df)), index=df.index),
                    ).to_numpy(dtype=float)
                    < 0.5
                )
            )
        ),
        "preview_event_risk_raw_prob_mean": float(
            np.mean(
                np.column_stack(
                    [
                        df.get(
                            "preview_event_risk_raw_prob_0_20m",
                            df.get("preview_event_risk_prob_0_20m", zeros),
                        ).to_numpy(dtype=float),
                        df.get(
                            "preview_event_risk_raw_prob_20_40m",
                            df.get("preview_event_risk_prob_20_40m", zeros),
                        ).to_numpy(dtype=float),
                        df.get(
                            "preview_event_risk_raw_prob_40_60m",
                            df.get("preview_event_risk_prob_40_60m", zeros),
                        ).to_numpy(dtype=float),
                    ]
                )
            )
        ),
        "preview_event_risk_effective_prob_mean": float(
            np.mean(
                np.column_stack(
                    [
                        df.get(
                            "preview_event_risk_effective_prob_0_20m",
                            df.get("preview_event_risk_prob_0_20m", zeros),
                        ).to_numpy(dtype=float),
                        df.get(
                            "preview_event_risk_effective_prob_20_40m",
                            df.get("preview_event_risk_prob_20_40m", zeros),
                        ).to_numpy(dtype=float),
                        df.get(
                            "preview_event_risk_effective_prob_40_60m",
                            df.get("preview_event_risk_prob_40_60m", zeros),
                        ).to_numpy(dtype=float),
                    ]
                )
            )
        ),
        "preview_forecast_pressure_trust_enabled_ratio": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_gate_enabled",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_block_ratio": float(
            np.mean(
                (
                    df.get(
                        "preview_forecast_pressure_trust_gate_enabled",
                        zeros,
                    ).to_numpy(dtype=float)
                    > 0.5
                )
                & (
                    df.get(
                        "preview_forecast_pressure_trust_trusted",
                        pd.Series(np.ones(len(df)), index=df.index),
                    ).to_numpy(dtype=float)
                    < 0.5
                )
            )
        ),
        "preview_forecast_pressure_trust_scale_mean": float(
            np.mean(
                np.column_stack(
                    [
                        df.get(
                            "preview_forecast_pressure_trust_block0_scale",
                            pd.Series(np.ones(len(df)), index=df.index),
                        ).to_numpy(dtype=float),
                        df.get(
                            "preview_forecast_pressure_trust_block1_scale",
                            pd.Series(np.ones(len(df)), index=df.index),
                        ).to_numpy(dtype=float),
                        df.get(
                            "preview_forecast_pressure_trust_block2_scale",
                            pd.Series(np.ones(len(df)), index=df.index),
                        ).to_numpy(dtype=float),
                    ]
                )
            )
        ),
        "preview_raw_pressure_future_max_mean": float(
            np.mean(
                df.get(
                    "preview_raw_pressure_future_max_norm",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_effective_pressure_future_max_mean": float(
            np.mean(
                df.get(
                    "preview_effective_pressure_future_max_norm",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_current_support_ratio": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_current_support",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_current_observed_norm_mean": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_current_observed_norm",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_current_speed_mean_ms": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_current_speed_ms",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_future_speed_max_mean_ms": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_future_speed_max_ms",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_event_support_ratio": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_event_support",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_forecast_pressure_trust_shape_support_ratio": float(
            np.mean(
                df.get(
                    "preview_forecast_pressure_trust_pressure_shape_support",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_lead_action_ratio": float(
            np.mean(df.get("preview_lead_action_active", zeros).to_numpy(dtype=float))
        ),
        "preview_relief_medium_cap_ratio": float(
            np.mean(df.get("preview_relief_medium_cap_active", zeros).to_numpy(dtype=float))
        ),
        "preview_medium_escalation_ratio": float(
            np.mean(df.get("preview_medium_escalation_active", zeros).to_numpy(dtype=float))
        ),
        "preview_far_horizon_available_ratio": float(
            np.mean(df.get("preview_far_horizon_available", zeros).to_numpy(dtype=float))
        ),
        "preview_far_horizon_hint_ratio": float(
            np.mean(df.get("preview_far_horizon_hint_any", zeros).to_numpy(dtype=float))
        ),
        "preview_far_horizon_relief_ratio": float(
            np.mean(df.get("preview_far_horizon_hidden_relief", zeros).to_numpy(dtype=float))
        ),
        "preview_far_horizon_relief_gate_ratio": float(
            np.mean(
                df.get(
                    "preview_far_horizon_relief_gate_active",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_active_effectiveness_refresh_ratio": float(
            np.mean(
                df.get(
                    "preview_active_effectiveness_refresh_active",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_sustained_active_recompute_ratio": float(
            np.mean(
                df.get(
                    "preview_sustained_active_recompute_active",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_hold_relief_debt_ratio": float(
            np.mean(
                df.get("preview_hold_relief_debt_applied", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_hold_relief_debt_level_mean": float(
            np.mean(
                df.get("preview_hold_relief_debt_level", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_hold_comfort_release_ratio": float(
            np.mean(
                df.get("preview_hold_comfort_release_active", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_hold_comfort_veto_ratio": float(
            np.mean(
                df.get("preview_hold_comfort_forecast_veto_active", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_hold_risk_micro_action_ratio": float(
            np.mean(
                df.get("preview_hold_risk_micro_action_active", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_no_unexplained_hold_action_ratio": float(
            np.mean(
                df.get(
                    "preview_no_unexplained_hold_action_active",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_primary_hold_feedback_ratio": float(
            np.mean(
                df.get("preview_primary_hold_feedback_active", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_primary_hold_feedback_delta_mean_kg": float(
            np.mean(
                df.get(
                    "preview_primary_hold_feedback_delta_mean_kg",
                    zeros,
                ).to_numpy(dtype=float)
            )
        ),
        "preview_posture_hold_barrier_ratio": float(
            np.mean(
                df.get("preview_posture_hold_barrier_triggered", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_posture_hold_barrier_cost_mean": float(
            np.mean(
                df.get("preview_posture_hold_barrier_cost", zeros).to_numpy(dtype=float)
            )
        ),
        "preview_posture_hold_norm_mean": float(
            np.mean(df.get("preview_posture_hold_norm", zeros).to_numpy(dtype=float))
        ),
        "primary_target_resumed_ratio": float(
            np.mean(df.get("preview_primary_target_resumed", zeros).to_numpy(dtype=float))
        ),
        "preview_recovery_mode_ratio": float(
            np.mean(
                df.get("preview_recovery_mode_active", zeros).to_numpy(dtype=float)
            )
        ),
    }
    out.update(
        compute_attitude_exposure_metrics(
            pitch,
            roll,
            time_s=time_s,
            fallback_dt_s=DT,
        )
    )
    return out


def add_prefixed_attitude_exposure(
    row: dict,
    closed_summary: dict,
    primary_summary: dict,
    *,
    has_closed: bool,
) -> None:
    for key in ATTITUDE_EXPOSURE_METRIC_KEYS:
        primary_value = float(primary_summary.get(key, 0.0))
        closed_value = (
            float(closed_summary.get(key, np.nan)) if has_closed else np.nan
        )
        row[f"primary_{key}"] = primary_value
        row[f"closed_{key}"] = closed_value
        row[f"d_{key}"] = (
            primary_value - closed_value if np.isfinite(closed_value) else np.nan
        )


def _planner_short_action(action: str) -> str:
    mapping = {
        "hold": "hold",
        "pump_saving": "save",
        "active_small": "small",
        "active_medium": "medium",
        "active_reverse_small": "reverse",
    }
    return mapping.get(str(action), str(action))


def _format_plan_note(row: pd.Series) -> str:
    action = _planner_short_action(str(row.get("first_action", "hold")))
    tp = float(row.get("target_pitch_deg", 0.0))
    tr = float(row.get("target_roll_deg", 0.0))
    b0 = float(row.get("pressure_block0_norm", 0.0))
    b1 = float(row.get("pressure_block1_norm", 0.0))
    b2 = float(row.get("pressure_block2_norm", 0.0))
    reused = int(float(row.get("prediction_primary_target_reused", 0.0))) if "prediction_primary_target_reused" in row else 0
    refresh = "reuse" if reused else "new"
    if action == "hold":
        return f"hold\nfuture {b0:.2f}/{b1:.2f}/{b2:.2f}"
    return (
        f"{action} {refresh}\n"
        f"target {tp:+.2f}/{tr:+.2f} deg\n"
        f"future {b0:.2f}/{b1:.2f}/{b2:.2f}"
    )


def _circular_mean_deg(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    rad = np.deg2rad(vals)
    mean_sin = float(np.mean(np.sin(rad)))
    mean_cos = float(np.mean(np.cos(rad)))
    return float((np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360.0) % 360.0)


def _format_prediction_role(row: pd.Series | None) -> str:
    if row is None:
        return "Pred: n/a"
    action = str(row.get("first_action", "hold"))
    short = _planner_short_action(action)
    role_map = {
        "hold": "hold",
        "save": "pump-save",
        "small": "pre-small",
        "medium": "pre-med",
        "reverse": "release",
    }
    role = role_map.get(short, short)
    b0 = float(row.get("pressure_block0_norm", 0.0))
    b1 = float(row.get("pressure_block1_norm", 0.0))
    b2 = float(row.get("pressure_block2_norm", 0.0))
    dkg = float(row.get("prediction_primary_delta_abs_mean_kg", 0.0))
    refreshed = int(float(row.get("prediction_primary_target_refreshed", 0.0))) > 0
    reused = int(float(row.get("prediction_primary_target_reused", 0.0))) > 0
    mode = "new" if refreshed or not reused else "reuse"
    if role == "hold" or dkg <= 1.0:
        return f"Pred: hold\nrisk {b0:.2f}/{b1:.2f}/{b2:.2f}"
    return f"Pred: {role}\n{mode}, dM {dkg / 1000.0:.0f}t"


def _planner_row_for_interval(
    planner_log: pd.DataFrame | None,
    start_s: float,
    end_s: float,
) -> pd.Series | None:
    if planner_log is None or planner_log.empty or "current_time_s" not in planner_log.columns:
        return None
    times = planner_log["current_time_s"].to_numpy(dtype=float)
    mask = (times >= float(start_s) - 1e-6) & (times < float(end_s) - 1e-6)
    if np.any(mask):
        return planner_log.iloc[int(np.flatnonzero(mask)[0])]
    before = np.flatnonzero(times <= float(start_s) + 1e-6)
    if before.size > 0:
        return planner_log.iloc[int(before[-1])]
    return planner_log.iloc[0]


def _same_wind_block(a: dict, b: dict, ws_tol: float = 0.05, wd_tol: float = 0.5) -> bool:
    if not (np.isfinite(a["ws"]) and np.isfinite(b["ws"]) and np.isfinite(a["wd"]) and np.isfinite(b["wd"])):
        return False
    wd_delta = abs(((float(a["wd"]) - float(b["wd"]) + 180.0) % 360.0) - 180.0)
    return abs(float(a["ws"]) - float(b["ws"])) <= float(ws_tol) and wd_delta <= float(wd_tol)


def _annotate_wind_prediction_strip(
    ax,
    wind_df: pd.DataFrame,
    planner_log: pd.DataFrame | None,
    duration_min: float,
    block_min: float = 10.0,
) -> None:
    pyplot = _require_pyplot()
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([])
    ax.set_ylabel("Wind\nPred", rotation=0, ha="right", va="center", labelpad=28)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#bdbdbd")
    ax.grid(False)

    t = wind_df["t_s"].to_numpy(dtype=float) / 60.0
    ws = wind_df.get("wind_speed", pd.Series(np.zeros(len(wind_df)))).to_numpy(dtype=float)
    wd = wind_df.get("wind_dir_deg", pd.Series(np.zeros(len(wind_df)))).to_numpy(dtype=float)
    cmap = pyplot.get_cmap("Blues")
    n_blocks = int(np.ceil(float(duration_min) / float(block_min)))
    blocks: list[dict] = []
    for i in range(n_blocks):
        start = i * float(block_min)
        end = min((i + 1) * float(block_min), float(duration_min))
        if end <= start:
            continue
        mask = (t >= start - 1e-9) & (t < end - 1e-9)
        if not np.any(mask):
            mask = np.abs(t - start) <= 0.5
        ws_val = float(np.nanmean(ws[mask])) if np.any(mask) else float("nan")
        wd_val = _circular_mean_deg(wd[mask]) if np.any(mask) else float("nan")
        blocks.append(
            {
                "start": start,
                "end": end,
                "mid": 0.5 * (start + end),
                "ws": ws_val,
                "wd": wd_val,
                "planner_row": _planner_row_for_interval(planner_log, start * 60.0, end * 60.0),
            }
        )

    groups: list[dict] = []
    for block in blocks:
        if groups and _same_wind_block(groups[-1], block):
            groups[-1]["end"] = block["end"]
        else:
            groups.append({"start": block["start"], "end": block["end"], "ws": block["ws"], "wd": block["wd"]})

    for group in groups:
        speed_norm = 0.0 if not np.isfinite(group["ws"]) else float(np.clip(group["ws"] / 25.0, 0.0, 1.0))
        face = cmap(0.20 + 0.70 * speed_norm)
        ax.axvspan(group["start"], group["end"], facecolor=face, alpha=0.72, lw=0.0)
        ax.axvline(group["start"], color="#ffffff", lw=0.9, alpha=0.85)
        group_mid = 0.5 * (group["start"] + group["end"])
        wind_label = (
            f"WS {group['ws']:.1f} m/s  WD {group['wd']:.0f} deg"
            if np.isfinite(group["ws"]) and np.isfinite(group["wd"])
            else "WS n/a  WD n/a"
        )
        ax.text(group_mid, 0.76, wind_label, ha="center", va="center", fontsize=7.0, color="#0f2438")

    for block in blocks:
        ax.text(
            block["mid"],
            0.25,
            _format_prediction_role(block["planner_row"]),
            ha="center",
            va="center",
            fontsize=6.4,
            color="#2d2d2d",
            linespacing=1.12,
        )
    ax.axvline(float(duration_min), color="#ffffff", lw=0.8, alpha=0.75)
    ax.text(
        0.0,
        1.06,
        "10-min replay blocks: current wind and prediction-primary role",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )


def _annotate_planner(ax, planner_log: pd.DataFrame | None, duration_min: float) -> None:
    if planner_log is None or planner_log.empty:
        return
    y0, y1 = ax.get_ylim()
    span = max(y1 - y0, 1e-6)
    y = y1 + 0.045 * span
    for _, row in planner_log.iterrows():
        start_min = float(row.get("current_time_s", 0.0)) / 60.0
        if start_min >= duration_min:
            continue
        interval_min = 10.0
        if "current_time_s" in planner_log.columns and len(planner_log) > 1:
            times = np.sort(planner_log["current_time_s"].to_numpy(dtype=float))
            diffs = np.diff(times)
            diffs = diffs[diffs > 1e-6]
            if diffs.size > 0:
                interval_min = float(np.median(diffs) / 60.0)
        ax.text(
            min(start_min + 0.5 * interval_min, duration_min - 1.0),
            y,
            _format_plan_note(row),
            ha="center",
            va="bottom",
            fontsize=7.0,
            color="#3f3f3f",
            linespacing=1.1,
            clip_on=False,
        )
    ax.set_ylim(y0, y1 + 0.24 * span)


def plot_case(
    case_id: str,
    label: str,
    ts_str: str,
    closed: pd.DataFrame,
    primary: pd.DataFrame,
    out_dir: Path,
    primary_label: str,
    duration_min: float,
    planner_log: pd.DataFrame | None = None,
) -> Path:
    if plt is None:
        _require_pyplot()
    t_closed = closed["t_s"].to_numpy(dtype=float) / 60.0
    t_primary = primary["t_s"].to_numpy(dtype=float) / 60.0
    c = "#2468B2"
    p = "#D85C27"
    fig_w = max(12.5, min(24.0, 12.5 * duration_min / 60.0))
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(fig_w, 9.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.78, 1.0, 1.0, 1.0]},
    )
    strip_ax = axes[0]
    series_axes = axes[1:]
    wind_source = primary if {"wind_speed", "wind_dir_deg"}.issubset(primary.columns) else closed
    _annotate_wind_prediction_strip(strip_ax, wind_source, planner_log, duration_min)
    rows = [
        ("Pitch angle (deg)", "pitch_deg"),
        ("Roll angle (deg)", "roll_deg"),
        ("Total pump rate (m3/min)", "pump_total_rate_m3_min"),
    ]
    for ax, (ylabel, col) in zip(series_axes, rows):
        ax.plot(t_closed, closed[col].to_numpy(dtype=float), color=c, lw=1.45, label="closed_only")
        ax.plot(t_primary, primary[col].to_numpy(dtype=float), color=p, lw=1.45, label=primary_label)
        ax.axhline(0.0, color="#777777", lw=0.8, alpha=0.55)
        for x in np.arange(20.0, duration_min, 20.0):
            ax.axvline(x, color="#999999", lw=0.9, ls="--", alpha=0.55)
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#d9d9d9", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    series_axes[0].legend(loc="upper left", frameon=False, ncol=2)
    series_axes[2].set_xlabel("Time (min)")
    series_axes[2].set_xlim(0, duration_min)
    sc = summarize(closed)
    sp = summarize(primary)
    metric_text = (
        f"pump: {sc['pump_work_m3']:.1f} -> {sp['pump_work_m3']:.1f} m3 "
        f"({(sp['pump_work_m3'] - sc['pump_work_m3']) / max(sc['pump_work_m3'], 1e-9) * 100:+.1f}%)\n"
        f"pitch p95: {sc['pitch_abs_p95']:.2f} -> {sp['pitch_abs_p95']:.2f} deg; "
        f"roll p95: {sc['roll_abs_p95']:.2f} -> {sp['roll_abs_p95']:.2f} deg"
    )
    series_axes[0].text(
        0.995,
        0.03,
        metric_text,
        transform=series_axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.92),
    )
    fig.suptitle(f"{case_id}: {label} ({ts_str})", fontsize=13, y=1.01)
    out_path = out_dir / f"{case_id}_closed_vs_primary.png"
    fig.savefig(out_path, dpi=165, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _register_pruned_overlay_legacy_args(parser)
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/prediction_primary_casebook",
        help="Output directory for figures, timeseries, planner logs, and report.",
    )
    parser.add_argument(
        "--primary-label",
        default="prediction_primary_econ",
        help="Label/name for the prediction-primary variant in outputs.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=float(N_STEPS * DT),
        help="Simulation duration per case in seconds.",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated canonical case ids to run, e.g. 01,04,05,09,10. Empty means first 10 cases.",
    )
    parser.add_argument(
        "--cases-csv",
        default="",
        help="Optional CSV with case_id,timestamp,label columns. Overrides the built-in case list.",
    )
    parser.add_argument(
        "--stiffness-file",
        default="",
        help=(
            "Mooring stiffness workbook. When omitted, one unambiguous "
            "stiffness-named workbook is discovered from the project data directories."
        ),
    )
    parser.add_argument(
        "--planner-runtime-config",
        default=str(repo_root / "configs" / "planner_runtime_v1.json"),
        help="Versioned planner sign and forecast-discount configuration.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "Skip PNG plotting during batch runs. Raw 1Hz timeseries, planner logs, "
            "summary CSV, and report are still written."
        ),
    )
    parser.add_argument(
        "--dry-config",
        action="store_true",
        help="Print the effective controller configuration and exit before simulation.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Run only the prediction-primary variant. Useful for learned/persistence/oracle "
            "screening where the closed baseline would be identical and can be added later."
        ),
    )
    parser.add_argument(
        "--reactive-primary-only",
        action="store_true",
        help=(
            "Run the primary variant as a same-shell reactive baseline: no forecast "
            "adapter, no future blocks, and no current-wind-as-future persistence. "
            "Mass targets come only from current pitch/roll posture."
        ),
    )
    parser.add_argument(
        "--reactive-primary-action",
        choices=("pump_saving", "active_small", "active_medium"),
        default="pump_saving",
        help="Action strength for --reactive-primary-only.",
    )
    parser.add_argument(
        "--reactive-primary-enter-deg",
        type=float,
        default=3.0,
        help="Pitch/roll absolute-degree threshold that starts reactive_primary.",
    )
    parser.add_argument(
        "--reactive-primary-exit-deg",
        type=float,
        default=2.4,
        help="Pitch/roll absolute-degree threshold that exits reactive_primary.",
    )
    parser.add_argument(
        "--reactive-primary-update-interval-s",
        type=float,
        default=600.0,
        help="Target refresh interval for --reactive-primary-only.",
    )
    parser.add_argument(
        "--wind-transition-s",
        type=float,
        default=0.0,
        help="Raised-cosine transition duration for replay wind bucket changes. 0 keeps hard 10-min steps.",
    )
    parser.add_argument(
        "--bucket-snapshot-dir",
        default="",
        help="Optional analysis-only directory for bucket snapshots. Default disabled.",
    )
    parser.add_argument(
        "--bucket-snapshot-case-ids",
        default="",
        help="Comma-separated case ids for bucket snapshot export, e.g. 02 or 02_fr_relief_09.",
    )
    parser.add_argument(
        "--bucket-snapshot-ids",
        default="",
        help="Comma-separated 10-minute bucket indices to snapshot, e.g. 7,8,11. Default none.",
    )
    parser.add_argument(
        "--bucket-snapshot-horizon-buckets",
        type=int,
        default=3,
        help="Number of future buckets of wind to include in each snapshot.",
    )
    parser.add_argument(
        "--forced-prefix-actions",
        default="",
        help=(
            "Default-off diagnostic: JSON/CSV file with forced prefix rows "
            "(case_id,bucket,action,optional vector/source fields)."
        ),
    )
    parser.add_argument(
        "--forced-prefix-mode",
        choices=("off", "raw_action", "final_action", "target_update"),
        default="off",
        help="Default-off forced-prefix injection layer.",
    )
    parser.add_argument(
        "--forced-prefix-buckets",
        default="",
        help="Optional comma-separated bucket allowlist for forced-prefix rows.",
    )
    parser.add_argument(
        "--closed-pump-profile",
        choices=sorted(CLOSED_PUMP_PROFILES),
        default="engineered_minimal",
        help="Pump-side profile for closed_only only. Default is the frozen non-preview baseline.",
    )
    parser.add_argument(
        "--primary-pump-profile",
        choices=sorted(CLOSED_PUMP_PROFILES),
        default="raw",
        help=(
            "Default-off pump-side profile for prediction_primary. Default raw preserves "
            "legacy candidate runs; use engineered_minimal for actuator-matched validation."
        ),
    )
    parser.add_argument(
        "--primary-scale",
        type=float,
        default=1.0,
        help="Mass-target scale for prediction_primary only.",
    )
    parser.add_argument(
        "--primary-control-profile",
        default="manual",
        metavar="PROFILE",
        help=(
            "Controller profile name. Profiles are validated against "
            f"{CASEBOOK_PROFILE_REGISTRY_RELATIVE}; production and documented "
            "diagnostics run directly, while isolated experiments require "
            "--allow-isolated-profile."
        ),
    )
    parser.add_argument(
        "--allow-isolated-profile",
        action="store_true",
        help=(
            "Explicitly unlock profiles classified as isolated_experiment or "
            f"prune_candidate in {CASEBOOK_PROFILE_REGISTRY_RELATIVE}. "
            "Production and documented diagnostic profiles do not need this."
        ),
    )
    parser.add_argument(
        "--primary-safety-profile",
        choices=sorted(PRIMARY_SAFETY_PROFILES),
        default="default",
        help="Runtime fallback profile for prediction_primary only.",
    )
    parser.add_argument(
        "--forecast-source",
        choices=(
            "oracle",
            "persistence_mean",
            "current_only",
            "learned",
            "blend_soft03",
            "blend_dynamic_alpha",
            "blend_risk_floor_v1",
            "blend_confirmed_relief_v1",
        ),
        default="oracle",
        help="Forecast source for prediction_primary. Oracle uses replay future y_uv_raw.",
    )
    parser.add_argument(
        "--model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
        help="Trained forecast model directory used when --forecast-source=learned.",
    )
    parser.add_argument(
        "--blend-baseline-model-dir",
        default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
        help="Baseline model directory for blended forecast probes.",
    )
    parser.add_argument(
        "--blend-relief-model-dir",
        default="outputs/wind_prediction/lstm_synth_relief_t030_e15_v1",
        help="Relief-proposal model directory for blended forecast probes.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
        help="Replay dataset directory.",
    )
    parser.add_argument(
        "--replay-split",
        choices=("test", "validation", "train"),
        default="test",
        help=(
            "Replay split used to resolve case timestamps. Default is test for "
            "backward compatibility; mixed casebooks should be run split-by-split."
        ),
    )
    parser.add_argument(
        "--persistence-history-min",
        type=float,
        default=10.0,
        help="History mean window for --forecast-source=persistence_mean.",
    )
    parser.add_argument(
        "--event-reset-mode",
        choices=("action", "target_change", "active_bucket"),
        default="action",
        help="Prediction-primary target refresh policy. Default preserves current behavior.",
    )
    parser.add_argument(
        "--planner-envelope-raw",
        action="store_true",
        help=(
            "Evaluate planner envelope costs on raw undiscounted pressure blocks. "
            "Default preserves discounted-envelope behavior."
        ),
    )
    parser.add_argument(
        "--planner-envelope-barrier",
        action="store_true",
        help=(
            "Enable the existing zone-MPC envelope barrier: if a sequence violates "
            "the envelope and uses only weak actions, add a fixed scalar cost."
        ),
    )
    parser.add_argument(
        "--planner-envelope-barrier-const",
        type=float,
        default=50.0,
        help="Fixed scalar cost used by --planner-envelope-barrier.",
    )
    parser.add_argument(
        "--planner-posture-hold-relief-margin-norm",
        type=float,
        default=0.25,
        help="Raw block0-to-block2 relief margin that credits forecast hold.",
    )
    parser.add_argument(
        "--planner-posture-hold-forecast-credit",
        type=float,
        default=0.0,
        help=(
            "Maximum fraction of posture-state residual relieved by future "
            "evidence. Default 0 keeps learned/reactive comparisons "
            "source-symmetric."
        ),
    )
    parser.add_argument(
        "--far-horizon",
        action="store_true",
        help="Compute and log 60-120 minute forecast pressure features when available.",
    )
    parser.add_argument(
        "--far-horizon-relief-gate",
        action="store_true",
        help=(
            "Default-off weak action influence: when 60-120min relief is clear "
            "and current posture is comfortable, cap active_medium to active_small."
        ),
    )
    parser.add_argument(
        "--far-horizon-weight-60-80",
        type=float,
        default=0.50,
        help="Weight for the 60-80min far-horizon pressure block.",
    )
    parser.add_argument(
        "--far-horizon-weight-80-100",
        type=float,
        default=0.35,
        help="Weight for the 80-100min far-horizon pressure block.",
    )
    parser.add_argument(
        "--far-horizon-weight-100-120",
        type=float,
        default=0.25,
        help="Weight for the 100-120min far-horizon pressure block.",
    )
    parser.add_argument(
        "--far-horizon-relief-pitch-abs-max-deg",
        type=float,
        default=4.0,
        help="Current pitch cap for allowing the far-horizon relief gate.",
    )
    parser.add_argument(
        "--far-horizon-relief-roll-abs-max-deg",
        type=float,
        default=4.0,
        help="Current roll cap for allowing the far-horizon relief gate.",
    )
    parser.add_argument(
        "--no-preview-myopic-horizon",
        action="store_true",
        help=(
            "Diagnostic fairness mode for current_only/no-preview sources: keep "
            "the same planner and action library, but score only immediate "
            "actions followed by hold instead of optimizing a 60-minute "
            "persistence pseudo-preview."
        ),
    )
    parser.add_argument(
        "--primary-stall-refresh",
        action="store_true",
        help=(
            "Default-off execution diagnostic: when an active planner action has "
            "apparently reached its old target but posture remains high and pumps "
            "are quiet, refresh the primary target instead of reusing it."
        ),
    )
    parser.add_argument(
        "--primary-stall-pitch-deg",
        type=float,
        default=5.0,
        help="Absolute pitch threshold for --primary-stall-refresh.",
    )
    parser.add_argument(
        "--primary-stall-roll-deg",
        type=float,
        default=4.0,
        help="Absolute roll threshold for --primary-stall-refresh.",
    )
    parser.add_argument(
        "--primary-stall-target-err-kg",
        type=float,
        default=1500.0,
        help=(
            "Mean absolute tank-target error below which the previous target is "
            "treated as reached for --primary-stall-refresh."
        ),
    )
    parser.add_argument(
        "--primary-stall-pump-rate-m3-min",
        type=float,
        default=0.5,
        help=(
            "Total absolute pump command rate below which the pumps are treated "
            "as quiet for --primary-stall-refresh."
        ),
    )
    parser.add_argument(
        "--primary-stall-min-age-s",
        type=float,
        default=300.0,
        help="Minimum age of the current primary target before stall refresh may fire.",
    )
    parser.add_argument(
        "--primary-stall-pitch-axis-bias",
        action="store_true",
        help=(
            "When stall refresh fires with high pitch but low roll, refresh along "
            "the pitch axis instead of preserving a mixed pitch/roll action."
        ),
    )
    parser.add_argument(
        "--primary-stall-pitch-axis-roll-max-deg",
        type=float,
        default=2.0,
        help="Roll must be below this absolute value for pitch-axis stall bias.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh",
        action="store_true",
        help=(
            "Default-off target-refresh prototype: refresh repeated active "
            "targets only when the current active path is stale, target-ready, "
            "pump-idle, and materially different from the current target."
        ),
    )
    parser.add_argument(
        "--active-effectiveness-refresh-min-active-buckets",
        type=int,
        default=2,
        help="Minimum repeated active buckets before an effectiveness refresh may fire.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-pitch-deg",
        type=float,
        default=3.0,
        help="Pitch attention band for the effectiveness-refresh gate.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-roll-deg",
        type=float,
        default=3.0,
        help="Roll attention band for the effectiveness-refresh gate.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-target-err-kg",
        type=float,
        default=1500.0,
        help="Target-err threshold that counts as effectively ready for refresh.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-pump-rate-m3-min",
        type=float,
        default=0.5,
        help="Pump-idle threshold used by the effectiveness-refresh gate.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-min-target-age-s",
        type=float,
        default=0.0,
        help=(
            "Require the current primary target to be at least this old before "
            "an active-effectiveness refresh may fire. Default 0 preserves the "
            "original default-off prototype behavior."
        ),
    )
    parser.add_argument(
        "--active-effectiveness-refresh-min-pressure-norm",
        type=float,
        default=0.0,
        help=(
            "Require max raw pressure norm across planner blocks to exceed this "
            "value before an effectiveness refresh may fire."
        ),
    )
    parser.add_argument(
        "--active-effectiveness-refresh-response-eps-deg",
        type=float,
        default=0.3,
        help="Minimum improvement before the effectiveness-refresh gate backs off.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-required-worsening-deg",
        type=float,
        default=0.0,
        help=(
            "If positive, require the bucket-to-bucket max attitude to worsen by "
            "at least this many degrees before refreshing a stale active target."
        ),
    )
    parser.add_argument(
        "--active-effectiveness-refresh-min-delta-kg",
        type=float,
        default=1500.0,
        help="Minimum mean abs proposal delta required before refreshing target.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-episode-budget-m3",
        type=float,
        default=180.0,
        help="Episode-level pump budget for effectiveness refresh.",
    )
    parser.add_argument(
        "--active-effectiveness-refresh-lowrisk-pressure-norm",
        type=float,
        default=0.5,
        help="Low-risk forecast pressure cap for effectiveness refresh rejection.",
    )
    parser.add_argument(
        "--sustained-active-recompute",
        action="store_true",
        help=(
            "Default-off STRUCTURAL fix for sustained-active stalled-pump idle. "
            "When N consecutive active buckets accumulate while posture remains "
            "above an attention band AND the previous target has been reached "
            "AND pump rate is essentially zero (i.e., the active-execution chain "
            "is idling), force a primary-target recompute at the event_reset "
            "level (same priority as action-change). Distinct from "
            "active_effectiveness_refresh: this mechanism does NOT require "
            "pressure/worsening/proposal-delta/budget conjunctions — it just "
            "says 'sustained active + pump idle + posture still high ⇒ ask the "
            "planner to recompute target'. Designed to address the case where "
            "PP is nominally active but execution stalls and the lifecycle "
            "re-uses a stale target indefinitely."
        ),
    )
    parser.add_argument(
        "--sustained-active-recompute-min-active-buckets",
        type=int,
        default=4,
        help="Consecutive active buckets required before recompute can trigger.",
    )
    parser.add_argument(
        "--sustained-active-recompute-pitch-deg",
        type=float,
        default=4.5,
        help="Pitch threshold (deg, absolute) above which recompute can trigger.",
    )
    parser.add_argument(
        "--sustained-active-recompute-roll-deg",
        type=float,
        default=4.5,
        help="Roll threshold (deg, absolute) above which recompute can trigger.",
    )
    parser.add_argument(
        "--sustained-active-recompute-target-err-kg",
        type=float,
        default=1500.0,
        help="Mean abs target error below which target is considered reached.",
    )
    parser.add_argument(
        "--sustained-active-recompute-pump-rate-m3-min",
        type=float,
        default=0.5,
        help="Total pump rate below which the execution chain is considered idle.",
    )
    parser.add_argument(
        "--sustained-active-recompute-min-target-age-s",
        type=float,
        default=1800.0,
        help="Minimum target age (seconds) before recompute can trigger.",
    )
    parser.add_argument(
        "--stale-active-target-refresh",
        choices=("off", "refresh_only_v1", "refresh_only_guarded_v1"),
        default="off",
        help=(
            "Default-off target-lifecycle candidate: when a high-posture active "
            "bucket keeps reusing a nearly reached target with idle pumps, "
            "refresh the target from the current active action. Does not "
            "upgrade active_small to active_medium."
        ),
    )
    parser.add_argument(
        "--stale-active-target-refresh-min-active-buckets",
        type=int,
        default=2,
        help="Repeated active buckets required before stale-active refresh may fire.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-pitch-deg",
        type=float,
        default=5.0,
        help="Absolute pitch threshold for stale-active refresh.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-roll-deg",
        type=float,
        default=5.0,
        help="Absolute roll threshold for stale-active refresh.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-target-err-kg",
        type=float,
        default=500.0,
        help="Target-error threshold below which the stale target is considered reached.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-pump-rate-m3-min",
        type=float,
        default=0.5,
        help="Total pump rate below which execution is considered idle.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-min-target-age-s",
        type=float,
        default=600.0,
        help="Minimum target age before stale-active refresh may fire.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-min-delta-kg",
        type=float,
        default=1500.0,
        help="Minimum proposal-vs-current-target delta required to accept refresh.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-cooldown-s",
        type=float,
        default=1200.0,
        help="Cooldown between accepted guarded stale-active refreshes.",
    )
    parser.add_argument(
        "--stale-active-target-refresh-no-recovery-eps-deg",
        type=float,
        default=0.10,
        help="Guarded mode rejects refresh if posture is already recovering by this much.",
    )
    parser.add_argument(
        "--reactive-floor-predictive-veto",
        choices=("off", "on", "theta_only_gated", "relaxed_veto_on_theta_only"),
        default="off",
        help=(
            "Default-off provider-layer probe: after sustained high posture, "
            "refresh a current-posture active_small recovery target unless "
            "forecast relief/current stability/target health veto the floor."
        ),
    )
    parser.add_argument(
        "--high-posture-metric",
        choices=("max_axis", "hybrid"),
        default="max_axis",
        help=(
            "High-posture entry metric for reactive floor. hybrid means "
            "max_axis>enter OR theta_total>enter+0.5."
        ),
    )
    parser.add_argument(
        "--high-posture-enter",
        type=float,
        default=5.0,
        help="High-posture enter threshold in deg for reactive floor.",
    )
    parser.add_argument(
        "--high-posture-exit",
        type=float,
        default=4.5,
        help="High-posture exit threshold in deg for reactive floor hysteresis.",
    )
    parser.add_argument(
        "--high-posture-min-duration-s",
        type=float,
        default=60.0,
        help="Minimum sustained high-posture duration before reactive floor latches.",
    )
    parser.add_argument(
        "--reactive-floor-theta-only-enter",
        type=float,
        default=6.0,
        help=(
            "v1.2 weak-trigger threshold: theta-total-only recovery is stronger "
            "evidence when theta_total exceeds this value."
        ),
    )
    parser.add_argument(
        "--reactive-floor-theta-only-min-duration-s",
        type=float,
        default=120.0,
        help=(
            "v1.2 weak-trigger threshold: theta-total-only recovery is stronger "
            "evidence after this sustained duration."
        ),
    )
    parser.add_argument(
        "--reactive-floor-action",
        choices=("active_small", "active_medium"),
        default="active_small",
        help=(
            "Default-off v1.5 probe: action strength used when reactive floor "
            "refreshes the recovery target. Defaults to active_small, preserving v1.4."
        ),
    )
    parser.add_argument(
        "--reactive-floor-medium-delay-s",
        type=float,
        default=0.0,
        help=(
            "Default-off v1.6 probe: when --reactive-floor-action=active_small, "
            "switch floor recovery to active_medium only after this many seconds "
            "of a still-latched floor episode. 0 disables delayed medium escalation."
        ),
    )
    parser.add_argument(
        "--reactive-floor-post-exit-mode",
        choices=("off", "early_stop", "target_cap"),
        default="off",
        help=(
            "Default-off v1.4 probe: after reactive floor has recovered posture "
            "to the exit band, either release the recovery target or cap the "
            "remaining target error."
        ),
    )
    parser.add_argument(
        "--reactive-floor-post-exit-target-cap-kg",
        type=float,
        default=500.0,
        help="Remaining mean target error cap used by --reactive-floor-post-exit-mode=target_cap.",
    )
    parser.add_argument(
        "--reactive-floor-post-exit-margin-deg",
        type=float,
        default=0.0,
        help="Extra margin above high-posture exit threshold before post-exit v1.4 can act.",
    )
    parser.add_argument(
        "--reactive-floor-post-exit-worsening-eps-deg",
        type=float,
        default=0.05,
        help="Allowed per-step worsening tolerance for post-exit v1.4 intervention.",
    )
    parser.add_argument(
        "--oracle-preemptive-prevent-trigger",
        action="store_true",
        help=(
            "Default-off oracle ceiling probe: before reactive floor, use true "
            "future pressure to refresh a small recovery target in the 3-5 deg "
            "posture band when near-future pressure clearly rises."
        ),
    )
    parser.add_argument(
        "--oracle-preemptive-enter-deg",
        type=float,
        default=3.0,
        help="Lower max-axis posture bound for oracle preemptive probe.",
    )
    parser.add_argument(
        "--oracle-preemptive-floor-deg",
        type=float,
        default=5.0,
        help="Upper max-axis bound; at/above this, reactive floor owns recovery.",
    )
    parser.add_argument(
        "--oracle-preemptive-min-future-norm",
        type=float,
        default=0.75,
        help="Minimum oracle future pressure norm required for preemptive refresh.",
    )
    parser.add_argument(
        "--oracle-preemptive-rise-margin-norm",
        type=float,
        default=0.25,
        help="Required oracle future pressure rise over current block norm.",
    )
    parser.add_argument(
        "--oracle-preemptive-min-delta-kg",
        type=float,
        default=500.0,
        help="Minimum mean tank target delta for accepting preemptive refresh.",
    )
    parser.add_argument(
        "--oracle-preemptive-cooldown-s",
        type=float,
        default=1200.0,
        help="Cooldown between oracle preemptive refreshes.",
    )
    parser.add_argument(
        "--h120-oracle-probe",
        action="store_true",
        help=(
            "Default-off 120min oracle value probe: use logged 60-120min "
            "pressure only for small diagnostic actions on top of v1.6."
        ),
    )
    parser.add_argument(
        "--h120-oracle-probe-short-delay-s",
        type=float,
        default=600.0,
        help="Temporary delayed-medium threshold when 60-120min high pressure is clear.",
    )
    parser.add_argument(
        "--h120-oracle-probe-far-high-norm",
        type=float,
        default=1.10,
        help="Pressure norm threshold for persistent far-horizon high pressure.",
    )
    parser.add_argument(
        "--h120-oracle-probe-intensify-margin-norm",
        type=float,
        default=0.30,
        help="Required 60-120min rise over 40-60min pressure for intensification.",
    )
    parser.add_argument(
        "--h120-oracle-probe-preemptive-enter-deg",
        type=float,
        default=3.0,
        help="Lower max-axis posture bound for h120 preemptive active_small.",
    )
    parser.add_argument(
        "--h120-oracle-probe-preemptive-floor-deg",
        type=float,
        default=5.0,
        help="Upper max-axis posture bound; at/above this, v1.6 floor owns recovery.",
    )
    parser.add_argument(
        "--h120-oracle-probe-min-delta-kg",
        type=float,
        default=500.0,
        help="Minimum mean tank target delta for accepting h120 preemptive refresh.",
    )
    parser.add_argument(
        "--h120-oracle-probe-cooldown-s",
        type=float,
        default=1200.0,
        help="Cooldown between h120 preemptive refreshes.",
    )
    parser.add_argument(
        "--h120-risk-scheduler",
        action="store_true",
        help=(
            "Default-off h120 learned/oracle remote-risk scheduler on top of "
            "v1.6 floor. Uses ONLY 60-120min far-horizon pressure to (a) shorten "
            "the delayed-medium delay and (b) suppress early_stop at most once "
            "per floor episode. Never decides recovery, never refreshes target "
            "preemptively, never uses relief to weaken the floor."
        ),
    )
    parser.add_argument(
        "--h120-scheduler-delay-normal-s", type=float, default=1200.0,
        help="delayed-medium delay under normal remote-risk tier.",
    )
    parser.add_argument(
        "--h120-scheduler-delay-risk-aware-s", type=float, default=900.0,
        help="delayed-medium delay under risk_aware tier.",
    )
    parser.add_argument(
        "--h120-scheduler-delay-high-risk-s", type=float, default=600.0,
        help="delayed-medium delay under high_risk tier.",
    )
    parser.add_argument(
        "--h120-scheduler-far-high-norm", type=float, default=0.90,
        help="60-120min pressure norm threshold for persistent high pressure.",
    )
    parser.add_argument(
        "--h120-scheduler-intensify-margin-norm", type=float, default=0.30,
        help="Required far_max rise over near_last for far intensification.",
    )
    parser.add_argument(
        "--h120-scheduler-relief-margin-norm", type=float, default=0.25,
        help="Near-horizon dip threshold used for re-intensification-after-relief.",
    )
    parser.add_argument(
        "--h120-scheduler-early-stop-suppress-max-per-episode", type=int, default=1,
        help="Max early_stop suppressions per floor episode under high_risk.",
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-probe",
        action="store_true",
        help=(
            "Default-off coupling probe: under h120 high_risk before floor entry, "
            "refresh a conservative active_small posture target in the 3-5deg band. "
            "Does not weaken the reactive floor or use relief suppression."
        ),
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-action",
        choices=("pump_saving", "active_small"),
        default="active_small",
        help="Posture action used by the default-off h120 prefloor probe.",
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-enter-deg", type=float, default=3.0,
        help="Lower max-axis posture bound for h120 prefloor active_small probe.",
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-floor-deg", type=float, default=5.0,
        help="Upper max-axis posture bound for h120 prefloor probe.",
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-min-delta-kg", type=float, default=500.0,
        help="Minimum mean target delta for accepting h120 prefloor probe refresh.",
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-max-delta-kg", type=float, default=0.0,
        help="Optional mean target delta cap for h120 prefloor probe; 0 disables the cap.",
    )
    parser.add_argument(
        "--h120-scheduler-prefloor-cooldown-s", type=float, default=1200.0,
        help="Cooldown between h120 prefloor probe refreshes.",
    )
    parser.add_argument(
        "--h120-axis-micro",
        action="store_true",
        help=(
            "Default-off corrected pre-floor axis-aware micro prepare. Fires "
            "only on the direction-clean + real-rise eligible window with a "
            "small single-axis nudge on the current dominant posture axis."
        ),
    )
    parser.add_argument(
        "--h120-axis-micro-enter-deg", type=float, default=4.0,
        help="Lower max-axis posture bound for axis-micro prepare.",
    )
    parser.add_argument(
        "--h120-axis-micro-floor-deg", type=float, default=5.0,
        help="Upper max-axis posture bound (floor) for axis-micro prepare.",
    )
    parser.add_argument(
        "--h120-axis-micro-rise-floor-norm", type=float, default=0.8,
        help="Minimum far_horizon_far_max norm required for axis-micro prepare.",
    )
    parser.add_argument(
        "--h120-axis-micro-worsening-eps-deg", type=float, default=0.05,
        help="Minimum posture worsening (deg/bucket) for axis-micro prepare.",
    )
    parser.add_argument(
        "--h120-axis-micro-delta-kg", type=float, default=2000.0,
        help="Mean mass-delta cap for the axis-micro nudge (kept << active_small).",
    )
    parser.add_argument(
        "--h120-axis-micro-cooldown-s", type=float, default=600.0,
        help="Cooldown between axis-micro prepare nudges.",
    )
    parser.add_argument(
        "--relief-economy",
        action="store_true",
        help=(
            "Default-off relief-aware economy lever (A1/A2 audit). Delays a "
            "NON-URGENT economy target refresh when forecast relief is clear and "
            "posture is safe. Never touches floor/recovery/safety/high-posture."
        ),
    )
    parser.add_argument(
        "--relief-economy-horizon", choices=["near", "far"], default="far",
        help="near = 0-60min relief (A1); far = 0-120min incl. far blocks (A2).",
    )
    parser.add_argument(
        "--relief-economy-safe-deg", type=float, default=3.0,
        help="Max-axis posture must be below this (safe) for the lever to act.",
    )
    parser.add_argument(
        "--relief-economy-worsening-eps-deg", type=float, default=0.05,
        help="Lever skips if posture is worsening by more than this per bucket.",
    )
    parser.add_argument(
        "--relief-economy-low-norm", type=float, default=0.5,
        help="Relief: horizon pressure norm must drop below this.",
    )
    parser.add_argument(
        "--relief-economy-high-norm", type=float, default=0.9,
        help="Relief guard: no horizon pressure norm above this (no spike/re-intensify).",
    )
    parser.add_argument(
        "--relief-envelope",
        action="store_true",
        help=(
            "Default-off relief-conditioned soft posture envelope. It only "
            "delays non-hard target refreshes in the 3-5deg soft band; the "
            "reactive floor remains hard at 5deg."
        ),
    )
    parser.add_argument(
        "--relief-envelope-horizon", choices=["near", "far"], default="far",
        help="near = 0-60min relief envelope (A1); far = 0-120min envelope (A2).",
    )
    parser.add_argument("--relief-envelope-allowed-0-20-deg", type=float, default=4.5)
    parser.add_argument("--relief-envelope-allowed-20-40-deg", type=float, default=4.2)
    parser.add_argument("--relief-envelope-allowed-40-60-deg", type=float, default=3.9)
    parser.add_argument("--relief-envelope-allowed-60-120-deg", type=float, default=3.5)
    parser.add_argument("--relief-envelope-max-duration-s", type=float, default=600.0)
    parser.add_argument("--relief-envelope-debt-budget-deg-s", type=float, default=300.0)
    parser.add_argument("--relief-envelope-near-limit-deg", type=float, default=4.7)
    parser.add_argument("--relief-envelope-near-limit-budget-s", type=float, default=0.0)
    parser.add_argument("--relief-envelope-worsening-eps-deg", type=float, default=0.0)
    parser.add_argument("--relief-envelope-low-norm", type=float, default=0.5)
    parser.add_argument("--relief-envelope-high-norm", type=float, default=0.9)
    parser.add_argument("--relief-envelope-reintensify-norm", type=float, default=0.9)
    parser.add_argument(
        "--relief-envelope-single-axis-only",
        action="store_true",
        help=(
            "Only allow relief-envelope delay when the delayed refresh action is "
            "single-axis / dominant-axis clear. Default off."
        ),
    )
    parser.add_argument(
        "--relief-envelope-axis-minor-ratio-max",
        type=float,
        default=0.25,
        help="Minor/major action-axis ratio allowed by --relief-envelope-single-axis-only.",
    )
    parser.add_argument(
        "--relief-envelope-axis-minor-abs-max-deg",
        type=float,
        default=0.05,
        help="Absolute minor-axis action magnitude allowed by --relief-envelope-single-axis-only.",
    )
    parser.add_argument(
        "--relief-envelope-adaptive-axis-cap",
        action="store_true",
        help=(
            "B1 default-off relief-envelope variant: mixed-axis refreshes that "
            "would be blocked by --relief-envelope-single-axis-only may pursue "
            "only the non-relieved axes instead of fully refreshing or fully delaying."
        ),
    )
    parser.add_argument(
        "--relief-envelope-cap-relief-norm",
        type=float,
        default=0.55,
        help="Per-axis 0-60min pressure magnitude below this is treated as relief-covered.",
    )
    parser.add_argument(
        "--relief-envelope-cap-hold-fraction",
        type=float,
        default=0.0,
        help="Fraction of the relieved-axis action to still pursue under adaptive cap.",
    )
    parser.add_argument(
        "--forecast-control-trust-gate",
        action="store_true",
        help=(
            "Default-off fail-closed trust gate for forecast-enabled economy "
            "branches. When enabled, P2 suppression and W1/C3 hold-current "
            "gates must pass forecast-shape/current-stability/event-risk checks."
        ),
    )
    parser.add_argument(
        "--forecast-control-trust-event-threshold",
        type=float,
        default=0.60,
        help="Event probability above which forecast economy/hold permissions are vetoed.",
    )
    parser.add_argument(
        "--forecast-control-trust-current-stability-required",
        action="store_true",
        help="Require the existing current-wind stability check to pass before forecast economy/hold permissions.",
    )
    parser.add_argument(
        "--forecast-control-trust-max-speed-range-ms",
        type=float,
        default=1.0,
        help="Veto forecast economy/hold permissions when forecast speed range exceeds this; <=0 disables.",
    )
    parser.add_argument(
        "--forecast-control-trust-max-dir-shift-deg",
        type=float,
        default=10.0,
        help="Veto forecast economy/hold permissions when forecast direction shift exceeds this; <=0 disables.",
    )
    parser.add_argument(
        "--forecast-control-trust-suppression-rise-norm",
        type=float,
        default=0.0,
        help="Maximum allowed block2-block0 pressure rise for forecast-advised suppression.",
    )
    parser.add_argument(
        "--forecast-control-trust-min-pressure-norm",
        type=float,
        default=1.0,
        help="Minimum max near-pressure norm required before forecast economy/hold permissions can act.",
    )
    parser.add_argument(
        "--forecast-pressure-trust-gate",
        action="store_true",
        help=(
            "Default-off pressure-side trust gate. When enabled, untrusted "
            "learned future pressure is scaled in controller-facing pressure "
            "blocks while raw pressure diagnostics remain intact."
        ),
    )
    parser.add_argument(
        "--forecast-pressure-trust-current-norm-max",
        type=float,
        default=0.65,
        help="Current pressure norm above which future pressure has current-observation support.",
    )
    parser.add_argument(
        "--forecast-pressure-trust-future-norm-min",
        type=float,
        default=1.00,
        help="Minimum future pressure norm before pressure trust shaping can activate.",
    )
    parser.add_argument(
        "--forecast-pressure-trust-event-threshold",
        type=float,
        default=0.75,
        help="Effective trusted event probability that supports future pressure.",
    )
    parser.add_argument(
        "--forecast-pressure-trust-untrusted-scale",
        type=float,
        default=0.55,
        help="Scale applied to future effective pressure blocks when pressure trust fails.",
    )
    parser.add_argument(
        "--forecast-pressure-trust-min-direction-dot",
        type=float,
        default=0.0,
        help="Minimum pressure-vector dot product required for direction consistency.",
    )
    parser.add_argument(
        "--forecast-pressure-trust-spike-margin-norm",
        type=float,
        default=0.35,
        help="Norm gap that marks an isolated single-horizon future-pressure spike.",
    )
    parser.add_argument(
        "--economy-pump-budget",
        action="store_true",
        help=(
            "Default-off aggressive economy mode: once the non-hard economy pump "
            "budget is exhausted, non-floor target refreshes reuse the current target."
        ),
    )
    parser.add_argument(
        "--economy-pump-budget-m3",
        type=float,
        default=0.0,
        help="Per-case economy pump budget in m3. Hard floor/recovery branches are exempt.",
    )
    parser.add_argument(
        "--economy-pump-budget-hold-refresh-fraction",
        type=float,
        default=0.0,
        help=(
            "No.4 mild-arm knob: when an exhausted economy budget would hold "
            "current target, pursue this fraction of the blocked refresh delta. "
            "0 preserves historical hold behavior; 1 is full refresh."
        ),
    )
    parser.add_argument(
        "--economy-pump-budget-dynamic-refresh",
        action="store_true",
        help=(
            "No.4 dynamic mild-arm knob: when a budget-blocked refresh is "
            "partially pursued, raise the fraction as live pitch/roll enters "
            "the high-posture debt band."
        ),
    )
    parser.add_argument(
        "--economy-pump-budget-dynamic-refresh-enter-deg",
        type=float,
        default=3.5,
        help="Max-axis posture where dynamic partial-refresh scaling starts.",
    )
    parser.add_argument(
        "--economy-pump-budget-dynamic-refresh-full-deg",
        type=float,
        default=4.7,
        help="Max-axis posture where dynamic scaling reaches max fraction.",
    )
    parser.add_argument(
        "--economy-pump-budget-dynamic-refresh-max-fraction",
        type=float,
        default=0.85,
        help="Upper partial-refresh fraction reached in the high-posture band.",
    )
    parser.add_argument(
        "--economy-pump-budget-allocator-mode",
        choices=(
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
            "neutral_mhs_clean_auto",
            "dual_specialist_auto",
            "regime_auto",
        ),
        default="budget",
        help=(
            "How budget-exhausted holds are justified. budget is the blind "
            "negative control; strict_relief/broad_safe require forecast-state "
            "reasons; posture_only ignores forecast; opportunity uses early "
            "controller-waste evidence before allowing aggressive holds; "
            "relief_decay_auto enables holds only when the live 0-60min forecast "
            "shows a transient peak followed by decay; relief_decay_auto_v2 uses "
            "the same opportunity signal but does not release merely because near "
            "pressure is high; relief_decay_episode_auto_v1 additionally latches "
            "early high/stable wind episodes before the drop is visible; residual_high_auto is a "
            "specialist for stable high-load economy holds; residual_high_plateau_auto "
            "targets high-pressure plateaus with explicit boundary releases; "
            "dual_specialist_auto combines the validated relief-decay episode "
            "selector with the neutral-MHS clean-start selector; "
            "regime_auto additionally recognizes residual-high and low-risk "
            "redundant-pump opportunities."
        ),
    )
    parser.add_argument(
        "--economy-pump-budget-require-relief",
        action="store_true",
        help=(
            "Only freeze non-hard economy refreshes after budget exhaustion when "
            "0-60min pressure is clearly decaying. Otherwise release the refresh."
        ),
    )
    parser.add_argument(
        "--economy-pump-budget-relief-norm",
        type=float,
        default=0.7,
        help="Required block2 pressure norm for relief-supported budget hold.",
    )
    parser.add_argument(
        "--economy-pump-budget-relief-drop-norm",
        type=float,
        default=0.2,
        help="Required block0-block2 pressure drop for relief-supported budget hold.",
    )
    parser.add_argument(
        "--economy-pump-budget-forecast-smart",
        action="store_true",
        help=(
            "Allow budget-exhausted refreshes when posture is close to the hard "
            "floor or near pressure is high, reducing high-posture idle rebound."
        ),
    )
    parser.add_argument(
        "--economy-pump-budget-smart-posture-deg",
        type=float,
        default=4.25,
        help="Forecast-smart budget releases refreshes above this max-axis posture.",
    )
    parser.add_argument(
        "--economy-pump-budget-smart-pressure-norm",
        type=float,
        default=0.9,
        help="Forecast-smart budget releases refreshes when near pressure exceeds this norm.",
    )
    parser.add_argument(
        "--economy-pump-budget-opportunity-window-s",
        type=float,
        default=1200.0,
        help="Initial window used to confirm an opportunity-mode economy budget case.",
    )
    parser.add_argument(
        "--economy-pump-budget-opportunity-min-pump-m3",
        type=float,
        default=120.0,
        help="Minimum early non-hard pump use required for opportunity-mode holds.",
    )
    parser.add_argument(
        "--economy-pump-budget-opportunity-min-refresh",
        type=int,
        default=3,
        help="Minimum early target-refresh count required for opportunity-mode holds.",
    )
    parser.add_argument(
        "--economy-pump-budget-opportunity-min-delta-kg",
        type=float,
        default=100000.0,
        help="Minimum mean early target delta required for opportunity-mode holds.",
    )
    parser.add_argument(
        "--h120-floor-shaping-mode",
        choices=[
            "off",
            "axis_shaped",
            "bounded_medium",
            "axis_shaped_bounded_medium",
            "stale_repair",
        ],
        default="off",
        help=(
            "Default-off h120 floor-active action-shaping oracle probe. "
            "It only acts after the reactive floor is already latched."
        ),
    )
    parser.add_argument(
        "--h120-floor-shaping-far-high-norm",
        type=float,
        default=0.90,
        help="Far-horizon norm threshold used by floor-active shaping probes.",
    )
    parser.add_argument(
        "--h120-floor-shaping-slow-response-eps-deg",
        type=float,
        default=0.05,
        help="Recovery response threshold for floor-active shaping probes.",
    )
    parser.add_argument(
        "--h120-floor-shaping-medium-delay-s",
        type=float,
        default=600.0,
        help="Bounded medium delay used by h120 floor-active shaping probes.",
    )
    parser.add_argument(
        "--h120-pareto-mode-selector",
        action="store_true",
        help=(
            "Default-off oracle selector for the static floor Pareto modes. "
            "It chooses among v1.4 early_stop, v1.6 delay1200, and v1.5 "
            "always-medium at floor-episode level using 60-120min far-horizon "
            "shapes only."
        ),
    )
    parser.add_argument(
        "--h120-pareto-mode-selector-far-high-norm",
        type=float,
        default=0.90,
        help="Far-horizon norm threshold used by the oracle Pareto mode selector.",
    )
    parser.add_argument(
        "--h120-pareto-mode-selector-intensify-margin-norm",
        type=float,
        default=0.30,
        help="Far-horizon rise margin used by the oracle Pareto mode selector.",
    )
    parser.add_argument(
        "--h120-pareto-mode-selector-relief-margin-norm",
        type=float,
        default=0.25,
        help="Near-horizon relief margin used by the oracle Pareto mode selector.",
    )
    parser.add_argument(
        "--h120-pareto-mode-selector-slow-response-eps-deg",
        type=float,
        default=0.05,
        help="Recovery response threshold used by the oracle Pareto mode selector.",
    )
    parser.add_argument(
        "--hold-relief-debt",
        action="store_true",
        help=(
            "Default-off planner-cost prototype: if a high-posture hold was "
            "justified by future relief or low pressure but the next bucket "
            "does not improve, add a small debt to subsequent hold/pump_saving "
            "candidate sequences. Does not directly override actions."
        ),
    )
    parser.add_argument(
        "--hold-relief-debt-weight",
        type=float,
        default=1.0,
        help="Scalar weight for the planner hold-relief debt cost.",
    )
    parser.add_argument(
        "--hold-relief-debt-threshold-deg",
        type=float,
        default=3.0,
        help="Pitch/roll attention band above which hold relief debt can accrue.",
    )
    parser.add_argument(
        "--hold-relief-debt-response-eps-deg",
        type=float,
        default=0.3,
        help="Required bucket-to-bucket attitude reduction to count relief as realized.",
    )
    parser.add_argument(
        "--hold-relief-debt-increment",
        type=float,
        default=1.0,
        help="Debt increment when a previous high-posture wait did not realize relief.",
    )
    parser.add_argument(
        "--hold-relief-debt-max",
        type=float,
        default=4.0,
        help="Maximum accumulated hold-relief debt level.",
    )
    parser.add_argument(
        "--hold-relief-debt-lowrisk-pressure-norm",
        type=float,
        default=0.5,
        help="Pressure norm below which a high hold is considered quiet/low-pressure waiting.",
    )
    parser.add_argument(
        "--hold-relief-debt-lowrisk-safe-deg",
        type=float,
        default=2.5,
        help="Posture cap for rejecting hold-relief debt in low-risk safe posture.",
    )
    parser.add_argument(
        "--planner-posture-state-residual",
        action="store_true",
        help=(
            "Include current pitch/roll as planner initial residual state, so "
            "hold actions must account for existing attitude error."
        ),
    )
    parser.add_argument(
        "--planner-objective-mode-active",
        action="store_true",
        help=(
            "Enable the 2-state objective-mode prototype (default-off). When "
            "persistent_high_posture AND NOT future_rising, switch attitude "
            "cost to smooth_huber (deadzone) with elevated w_attitude; "
            "otherwise stay in quadratic (preemptive). Diagnostic fields "
            "objective_mode_active / persistent_high_buckets / future_rising / "
            "switch_count are emitted per planner bucket. See "
            "scripts/analysis/cost_replay_objective_mode_v1.py for math."
        ),
    )
    parser.add_argument(
        "--planner-posture-state-gain",
        type=float,
        default=0.35,
        help="Gain from current posture error to planner residual state.",
    )
    parser.add_argument(
        "--planner-posture-state-decay",
        type=float,
        default=0.85,
        help="Per-block decay applied to current-posture residual state.",
    )
    parser.add_argument(
        "--planner-posture-state-clip-norm",
        type=float,
        default=1.5,
        help="Deadband-normalized clipping for current-posture residual state.",
    )
    parser.add_argument(
        "--planner-attitude-residual-weight",
        type=float,
        default=0.0,
        help=(
            "Economic cost weight for accumulated residual attitude norm. "
            "Default 0 keeps legacy envelope-zone pump-first behavior unless "
            "a posture-aware profile overrides it."
        ),
    )
    parser.add_argument(
        "--planner-terminal-residual-weight",
        type=float,
        default=0.0,
        help=(
            "Economic cost weight for terminal residual attitude norm. "
            "Default 0 keeps legacy envelope-zone pump-first behavior unless "
            "a posture-aware profile overrides it."
        ),
    )
    parser.add_argument(
        "--planner-pressure-norm-cap",
        type=float,
        default=1.5,
        help=(
            "Deadband-normalized cap for pressure_proxy_vec. Default 1.5 preserves "
            "the current planner; larger values test whether high-wind saturation "
            "is compressing learned/persistence forecast differences."
        ),
    )
    parser.add_argument(
        "--target-reset-tol-deg",
        type=float,
        default=0.02,
        help="Planner-frame target-change tolerance for --event-reset-mode=target_change.",
    )
    parser.add_argument(
        "--primary-hold-target-mode",
        choices=("current", "pause", "pi_release", "forecast_pause"),
        default="current",
        help=(
            "Prediction-primary hold target handling. current preserves existing behavior; "
            "pause temporarily anchors hold at current masses but can resume the prior "
            "same-direction active target instead of re-issuing a fresh target; "
            "pi_release releases hold buckets back to the ordinary PI/heave command."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy",
        action="store_true",
        help=(
            "Enable the broad forecast-advised economy suppression branch. "
            "This is default-off and intended for operator/Pareto mode tests."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-min-headroom-deg",
        type=float,
        default=0.35,
        help="Minimum hard-floor headroom required by forecast-advised economy mode.",
    )
    parser.add_argument(
        "--forecast-advised-economy-rise-norm",
        type=float,
        default=0.20,
        help="Maximum near-horizon raw pressure rise allowed by forecast-advised economy mode.",
    )
    parser.add_argument(
        "--forecast-advised-economy-max-speed-range-ms",
        type=float,
        default=0.0,
        help=(
            "Optional maximum 0-120min forecast speed range for the strong "
            "forecast-advised economy branch; 0 disables this veto."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-max-dir-shift-deg",
        type=float,
        default=0.0,
        help=(
            "Optional forecast direction-shift veto for the strong "
            "forecast-advised economy branch; 0 disables this veto."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-confirm-buckets",
        type=int,
        default=1,
        help=(
            "Number of consecutive clear forecast buckets required before the "
            "strong forecast-advised economy branch can suppress pursuit."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-hold-confirm-buckets",
        type=int,
        default=0,
        help=(
            "Optional stronger confirmation count when the current planner "
            "action is hold; 0 uses --forecast-advised-economy-confirm-buckets."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-min-axis-deg",
        type=float,
        default=0.0,
        help=(
            "Optional minimum current max-axis posture deviation required before "
            "forecast-advised economy suppression may enter; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-min-pressure-norm",
        type=float,
        default=0.0,
        help=(
            "Optional minimum near-horizon pressure norm required before "
            "forecast-advised economy suppression may enter; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-posture-release-deg",
        type=float,
        default=0.0,
        help=(
            "Optional true max-axis posture threshold that releases the "
            "forecast-advised economy branch before the hard floor; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-release-cooldown-s",
        type=float,
        default=0.0,
        help=(
            "Optional cooldown after a forecast-advised posture release before "
            "new suppression may enter again; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-latch-s",
        type=float,
        default=0.0,
        help=(
            "Optional latch duration after a confirmed forecast-advised economy "
            "entry, used to avoid one-bucket suppression chatter; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-current-ws-range-ms",
        type=float,
        default=0.0,
        help=(
            "Optional observed-history wind-speed range veto for the "
            "forecast-advised economy branch; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-current-dir-shift-deg",
        type=float,
        default=0.0,
        help=(
            "Optional observed-history wind-direction shift veto for the "
            "forecast-advised economy branch; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-current-history-s",
        type=float,
        default=1800.0,
        help="Observed wind history span used by the current-stability veto.",
    )
    parser.add_argument(
        "--forecast-advised-economy-mild",
        action="store_true",
        help=(
            "Enable a lower-intensity forecast-advised economy branch for "
            "forecast-clear windows with less posture headroom."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-mild-min-headroom-deg",
        type=float,
        default=0.10,
        help="Minimum hard-floor headroom for the mild forecast-advised economy branch.",
    )
    parser.add_argument(
        "--forecast-advised-economy-mild-rise-norm",
        type=float,
        default=0.35,
        help="Maximum near-horizon raw pressure rise allowed by the mild branch.",
    )
    parser.add_argument(
        "--forecast-advised-economy-mild-max-speed-range-ms",
        type=float,
        default=0.0,
        help=(
            "Optional maximum 0-120min forecast speed range for the mild "
            "forecast-advised economy branch; 0 disables this veto."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-mild-restart-err-kg",
        type=float,
        default=900.0,
        help="Restart-error threshold for the mild forecast-advised economy branch.",
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-clean",
        action="store_true",
        help=(
            "Require the strict 52-case high-stable clean forecast gate before "
            "forecast-advised economy suppression may enter."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-early-min-ms",
        type=float,
        default=14.0,
        help="Minimum first-40min forecast speed max for the high-stable clean gate.",
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-near-min-ms",
        type=float,
        default=14.0,
        help="Minimum first-60min forecast speed max for the high-stable clean gate.",
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-far-min-ms",
        type=float,
        default=11.5,
        help="Minimum 60-120min forecast speed max for the high-stable clean gate.",
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-near-range-max-ms",
        type=float,
        default=3.0,
        help="Maximum first-60min forecast speed range for the high-stable clean gate.",
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-far-range-max-ms",
        type=float,
        default=3.0,
        help="Maximum 60-120min forecast speed range for the high-stable clean gate.",
    )
    parser.add_argument(
        "--forecast-advised-economy-high-stable-dir-shift-max-deg",
        type=float,
        default=20.0,
        help="Maximum 0-120min forecast direction shift for the high-stable clean gate.",
    )
    parser.add_argument(
        "--forecast-advised-economy-min-peak-drop-ms",
        type=float,
        default=0.0,
        help=(
            "Optional minimum forecast early-peak to 60-120min mean speed drop "
            "before forecast-advised economy suppression may enter; 0 disables."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-hold-current-on-candidate",
        action="store_true",
        help=(
            "For continuous broad-economy validation, convert active target "
            "refreshes into current-mass holds while the forecast-advised "
            "candidate is clear.  This isolates the intended 'do not chase' "
            "mechanism from planner active-target pursuit."
        ),
    )
    parser.add_argument(
        "--forecast-advised-economy-suppression-only",
        action="store_true",
        help=(
            "Do not let forecast-advised economy own the prediction-primary "
            "mass target.  The branch may still suppress small pump pursuit, "
            "but the main target path is released back to PI."
        ),
    )
    parser.add_argument(
        "--forecast-pause-resume-max-target-err-kg",
        type=float,
        default=0.0,
        help=(
            "For primary_hold_target_mode=forecast_pause, release to PI instead "
            "of resuming a paused target when its mean mass error exceeds this "
            "threshold. <=0 disables the guard."
        ),
    )
    parser.add_argument(
        "--gusty-oscillation",
        action="store_true",
        help=(
            "Enable the default-off gusty oscillation refresh gate. It blocks "
            "short-age active/stale target refreshes when learned forecast speed "
            "shows repeated peaks, so the target does not chase every gust."
        ),
    )
    parser.add_argument(
        "--gusty-oscillation-min-target-age-s",
        type=float,
        default=600.0,
        help=(
            "Minimum active target age before a gusty oscillation candidate can "
            "refresh again; below this age, refresh-style target churn is held."
        ),
    )
    parser.add_argument(
        "--gusty-oscillation-target-scale",
        type=float,
        default=1.0,
        help=(
            "Optional soft cap for gusty oscillation target pursuit; 1 leaves "
            "targets unchanged, 0.5 pursues half of each gusty target update."
        ),
    )
    parser.add_argument(
        "--gusty-oscillation-headroom-scaled",
        action="store_true",
        help=(
            "Scale gusty target pursuit by current posture headroom: strong "
            "suppression at low posture, weaker suppression near the release band."
        ),
    )
    parser.add_argument(
        "--gusty-oscillation-low-posture-deg",
        type=float,
        default=1.5,
        help="Posture below which the gusty refined branch uses the low-posture scale.",
    )
    parser.add_argument(
        "--gusty-oscillation-high-posture-deg",
        type=float,
        default=3.2,
        help="Posture above which the gusty refined branch uses the high-posture scale.",
    )
    parser.add_argument(
        "--gusty-oscillation-low-posture-scale",
        type=float,
        default=0.55,
        help="Target blend scale at low posture for the gusty refined branch.",
    )
    parser.add_argument(
        "--gusty-oscillation-high-posture-scale",
        type=float,
        default=0.90,
        help="Target blend scale near the release band for the gusty refined branch.",
    )
    parser.add_argument(
        "--gusty-oscillation-short-gate",
        action="store_true",
        help="Use the 6-bucket learned-runtime gusty gate instead of the legacy 12-bucket gate.",
    )
    parser.add_argument(
        "--gusty-oscillation-short-min-range-ms",
        type=float,
        default=0.75,
        help="Minimum 6-bucket forecast speed range for the short C3 gate.",
    )
    parser.add_argument(
        "--gusty-oscillation-short-min-pressure-norm",
        type=float,
        default=0.75,
        help="Minimum raw pressure-block norm before the short C3 gate may blend targets.",
    )
    parser.add_argument(
        "--gusty-oscillation-short-max-dir-shift-deg",
        type=float,
        default=45.0,
        help="Maximum 6-bucket direction shift before the short C3 gate releases.",
    )
    parser.add_argument(
        "--gusty-oscillation-short-min-turns",
        type=int,
        default=1,
        help="Minimum number of speed-derivative sign changes for the short C3 gate.",
    )
    parser.add_argument(
        "--gusty-hold-current-gate-mode",
        choices=("off", "gate_on", "gate_off"),
        default="off",
        help=(
            "Default-off C3 attribution guard for the hold-action current-target "
            "branch. gate_on allows hold-current only through the short gusty "
            "gate; gate_off blocks it for ablation."
        ),
    )
    parser.add_argument(
        "--gusty-hold-current-isolated-primary",
        action="store_true",
        help=(
            "Fail closed to the PI path outside the C3 hold-current branch so "
            "gate_on/gate_off isolate the forecast-gated hold-current effect."
        ),
    )
    parser.add_argument(
        "--gusty-hold-current-runtime-release",
        action="store_true",
        help="Release a C3 hold-current target between forecast buckets if live posture approaches the floor.",
    )
    parser.add_argument(
        "--gusty-hold-current-runtime-release-deg",
        type=float,
        default=4.2,
        help="Max-axis posture threshold for runtime release of a gated C3 hold-current target.",
    )
    parser.add_argument(
        "--gusty-hold-current-runtime-release-cooldown-s",
        type=float,
        default=300.0,
        help="Cooldown after runtime release before the C3 hold-current branch may re-enter.",
    )
    parser.add_argument(
        "--gusty-hold-current-debt-guard",
        action="store_true",
        help=(
            "Release C3 hold-current when target-motion or primary-target "
            "error debt becomes too large; this prevents forecast-gated holds "
            "from turning into delayed catch-up pump."
        ),
    )
    parser.add_argument(
        "--gusty-hold-current-debt-guard-target-motion-kg-s",
        type=float,
        default=220.0,
        help="Target-motion threshold for the C3 debt guard.",
    )
    parser.add_argument(
        "--gusty-hold-current-debt-guard-target-err-kg",
        type=float,
        default=1200.0,
        help="Primary target-error threshold for the C3 debt guard.",
    )
    parser.add_argument(
        "--gusty-hold-current-debt-guard-min-age-s",
        type=float,
        default=600.0,
        help="Minimum target age before the C3 debt guard may release a hold.",
    )
    parser.add_argument(
        "--gusty-hold-current-day-budget",
        action="store_true",
        help=(
            "Limit C3 hold-current entries over a day/window so repeated "
            "forecast-gated holds cannot accumulate into late catch-up pump."
        ),
    )
    parser.add_argument(
        "--gusty-hold-current-day-budget-max-buckets",
        type=int,
        default=0,
        help="Maximum C3 hold-current buckets per budget window; <=0 disables the cap.",
    )
    parser.add_argument(
        "--gusty-hold-current-day-budget-window-s",
        type=float,
        default=86400.0,
        help="Budget window for the C3 hold-current bucket cap.",
    )
    parser.add_argument(
        "--direction-reversal-hold-current-gate-mode",
        choices=("off", "gate_on", "gate_off"),
        default="off",
        help=(
            "Default-off W1 attribution guard for the hold-action current-target "
            "branch. gate_on allows hold-current only through the direction-"
            "reversal allocator; gate_off blocks it for ablation."
        ),
    )
    parser.add_argument(
        "--trusted-event-gate",
        action="store_true",
        help=(
            "Default-off control-facing event gate. When enabled, event-risk "
            "boost/floor uses only probabilities that pass highwind or multihead "
            "attention evidence."
        ),
    )
    parser.add_argument(
        "--trusted-event-highwind-threshold",
        type=float,
        default=0.90,
        help="High-wind event probability threshold for trusting event-risk signals.",
    )
    parser.add_argument(
        "--trusted-event-attention-threshold",
        type=float,
        default=0.75,
        help="Per-head attention event probability threshold for trusted-event gate.",
    )
    parser.add_argument(
        "--trusted-event-attention-min-heads",
        type=int,
        default=2,
        help="Minimum attention heads above threshold before event-risk signals are trusted.",
    )
    parser.add_argument(
        "--trusted-event-dynamic",
        action="store_true",
        help="Also allow dynamic heads to pass the trusted-event gate.",
    )
    parser.add_argument(
        "--trusted-event-dynamic-threshold",
        type=float,
        default=0.85,
        help="Per-head dynamic event probability threshold for trusted-event gate.",
    )
    parser.add_argument(
        "--trusted-event-dynamic-min-heads",
        type=int,
        default=2,
        help="Minimum dynamic heads above threshold before event-risk signals are trusted.",
    )
    parser.add_argument(
        "--relief-medium-cap",
        action="store_true",
        help="Cap active_medium to active_small in high-confidence future-relief windows.",
    )
    parser.add_argument(
        "--relief-medium-cap-event-threshold",
        type=float,
        default=0.70,
        help="Minimum segment event probability required for relief medium cap.",
    )
    parser.add_argument(
        "--relief-medium-cap-ratio",
        type=float,
        default=0.25,
        help="Action ratio used when capping active_medium in future-relief windows.",
    )
    parser.add_argument(
        "--relief-medium-cap-adaptive",
        action="store_true",
        help=(
            "Use a direction-aware relief cap: pitch-dominant medium actions are "
            "capped to active_small, while roll-dominant medium actions keep the "
            "configured relief-medium-cap-ratio."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh",
        action="store_true",
        help=(
            "Default-off No.4 guard: when prediction-primary is already active "
            "but live posture drifts high between planner buckets, refresh the "
            "active target with a bounded posture-directed action."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh-enter-deg",
        type=float,
        default=4.0,
        help="Max-axis posture where active between-bucket refresh may start.",
    )
    parser.add_argument(
        "--active-posture-refresh-exit-deg",
        type=float,
        default=3.2,
        help="Hysteresis exit band for active between-bucket refresh.",
    )
    parser.add_argument(
        "--active-posture-refresh-update-interval-s",
        type=float,
        default=120.0,
        help="Minimum interval between active posture refreshes inside a bucket.",
    )
    parser.add_argument(
        "--active-posture-refresh-action-name",
        choices=("pump_saving", "active_small", "active_medium"),
        default="active_small",
        help="Bounded action used by active between-bucket posture refresh.",
    )
    parser.add_argument(
        "--active-posture-refresh-pressure-max-norm",
        type=float,
        default=0.0,
        help=(
            "If positive, active posture refresh can trigger only when all raw "
            "forecast pressure blocks stay below this norm.  This targets "
            "relief-stage posture debt instead of correcting high-pressure "
            "economy holds globally."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh-min-target-age-s",
        type=float,
        default=0.0,
        help=(
            "If positive, active posture refresh can trigger only after the "
            "current active target has aged at least this long.  This targets "
            "6h delayed debt and avoids early mainline refresh churn."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh-axis-mode",
        choices=("max", "axis_debt"),
        default="max",
        help=(
            "Default max preserves legacy max-axis refresh. axis_debt masks the "
            "action vector to only axes that are above the active band and "
            "worsening or already in the configured debt band."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh-worsening-eps-deg-s",
        type=float,
        default=0.0,
        help=(
            "When axis_mode=axis_debt, require abs posture on an axis to worsen "
            "by at least this many deg/s unless the axis is already above "
            "--active-posture-refresh-debt-deg."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh-debt-deg",
        type=float,
        default=0.0,
        help=(
            "When axis_mode=axis_debt, an axis at or above this absolute posture "
            "is treated as debt even without a positive worsening slope. 0 disables."
        ),
    )
    parser.add_argument(
        "--active-posture-refresh-include-hold",
        action="store_true",
        help=(
            "Allow the bounded posture refresh to update a retained hold "
            "candidate instead of releasing the complete planner to PI."
        ),
    )
    parser.add_argument(
        "--fallback-risk-active-release",
        action="store_true",
        help=(
            "Default-off No.4 boundary guard: release an active primary target "
            "to PI when live posture is near the hard fallback band and still "
            "worsening or catching up with high pump/backlog."
        ),
    )
    parser.add_argument(
        "--fallback-risk-active-release-enter-deg",
        type=float,
        default=5.5,
        help="Max-axis live posture where fallback-risk active release may start.",
    )
    parser.add_argument(
        "--fallback-risk-active-release-exit-deg",
        type=float,
        default=4.8,
        help="Reserved hysteresis exit band reported for fallback-risk active release.",
    )
    parser.add_argument(
        "--fallback-risk-active-release-worsening-eps-deg-s",
        type=float,
        default=0.05,
        help="Minimum same-sign one-step posture worsening for fallback-risk release.",
    )
    parser.add_argument(
        "--fallback-risk-active-release-pump-rate-m3-min",
        type=float,
        default=10.0,
        help="Pump-rate catch-up threshold for fallback-risk active release.",
    )
    parser.add_argument(
        "--fallback-risk-active-release-backlog-kg",
        type=float,
        default=500.0,
        help="Pump backlog catch-up threshold for fallback-risk active release.",
    )
    args = parser.parse_args()
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    explicit_dests: set[str] = set()
    for token in sys.argv[1:]:
        if not token.startswith("--"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit_dests.add(dest)
    args._explicit_cli_overrides = {
        dest: getattr(args, dest)
        for dest in explicit_dests
        if hasattr(args, dest)
    }
    return args


def _case_key_aliases(case_num: int, raw_case_id: str, numbered_case_id: str) -> set[str]:
    return {
        str(case_num),
        f"{case_num:02d}",
        str(raw_case_id),
        str(numbered_case_id),
        str(numbered_case_id).split("_", 1)[-1],
    }


def _load_forced_prefix_actions(path_str: str) -> list[dict[str, Any]]:
    if not str(path_str).strip():
        return []
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"--forced-prefix-actions not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            rows = data.get("rows", data.get("actions", []))
        else:
            rows = data
        if not isinstance(rows, list):
            raise ValueError("forced-prefix JSON must be a list or contain rows/actions list")
        return [dict(r) for r in rows]
    df = pd.read_csv(path)
    return df.to_dict("records")


def _forced_prefix_for_case(
    rows: list[dict[str, Any]],
    case_num: int,
    raw_case_id: str,
    numbered_case_id: str,
    bucket_allowlist: set[int] | None,
) -> dict[int, dict[str, Any]]:
    aliases = _case_key_aliases(case_num, raw_case_id, numbered_case_id)
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        row_case = str(
            row.get("case_id", row.get("case", row.get("raw_case_id", "")))
        ).strip()
        if row_case and row_case not in aliases:
            continue
        if "bucket" not in row:
            continue
        try:
            bucket = int(row["bucket"])
        except (TypeError, ValueError):
            continue
        if bucket_allowlist is not None and bucket not in bucket_allowlist:
            continue
        action = str(
            row.get("action", row.get("forced_action_label", row.get("first_action", "")))
        ).strip()
        if not action:
            continue
        entry = dict(row)
        entry["action"] = action
        entry.setdefault(
            "source_label",
            row.get("forced_source_label", row.get("source", "manual")),
        )
        out[bucket] = entry
    return out


def _load_indexed_cases(cases_csv: str) -> list[tuple[int, str, str, str]]:
    if not cases_csv:
        return [
            (idx, raw_case_id, ts_str, label)
            for idx, (raw_case_id, ts_str, label) in enumerate(CASES[:10], start=1)
        ]
    path = Path(cases_csv)
    if not path.is_absolute():
        path = repo_root / path
    df = pd.read_csv(path)
    required = {"timestamp"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"--cases-csv missing required columns: {sorted(missing)}")
    rows: list[tuple[int, str, str, str]] = []
    for idx, row in enumerate(df.to_dict("records"), start=1):
        raw_case_id = str(row.get("case_id", "") or row.get("group", "") or f"case_{idx:02d}")
        label = str(row.get("label", "") or row.get("selection_group", "") or raw_case_id)
        rows.append((idx, raw_case_id, str(row["timestamp"]), label))
    return rows


def _numbered_case_id(case_num: int, raw_case_id: str) -> str:
    clean = str(raw_case_id).strip() or f"case_{case_num:02d}"
    if len(clean) > 3 and clean[:2].isdigit() and clean[2] == "_":
        clean = clean[3:]
    elif "_" in clean and clean.split("_", 1)[0].isdigit():
        clean = clean.split("_", 1)[1]
    clean = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in clean)
    return f"{case_num:02d}_{clean}"


def enforce_casebook_experiment_limits(
    *,
    duration_s: float,
    case_count: int,
) -> None:
    """Apply the six-hour, 30-case boundary to every newly started run."""

    enforce_development_experiment_bounds(
        duration_s=duration_s,
        case_count=case_count,
    )


def _make_forecast_adapter(args: argparse.Namespace):
    source = str(args.forecast_source)
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    if source == "oracle":
        return None
    if source == "persistence_mean":
        return PersistenceMeanForecastAdapter(
            dataset_dir=dataset_dir,
            history_minutes=float(args.persistence_history_min),
        )
    if source == "current_only":
        return CurrentOnlyForecastAdapter(
            dataset_dir=dataset_dir,
            history_minutes=float(args.persistence_history_min),
        )
    if source == "learned":
        model_dir = Path(args.model_dir)
        if not model_dir.is_absolute():
            model_dir = repo_root / model_dir
        return ForecastModelAdapter(
            model_dir=model_dir,
            dataset_dir=dataset_dir,
            device="cpu",
        )
    if source in {
        "blend_soft03",
        "blend_dynamic_alpha",
        "blend_risk_floor_v1",
        "blend_confirmed_relief_v1",
    }:
        baseline_model_dir = Path(args.blend_baseline_model_dir)
        if not baseline_model_dir.is_absolute():
            baseline_model_dir = repo_root / baseline_model_dir
        relief_model_dir = Path(args.blend_relief_model_dir)
        if not relief_model_dir.is_absolute():
            relief_model_dir = repo_root / relief_model_dir
        mode_map = {
            "blend_soft03": "soft_blend_03",
            "blend_dynamic_alpha": "dynamic_alpha",
            "blend_risk_floor_v1": "risk_floor_blend_v1",
            "blend_confirmed_relief_v1": "confirmed_relief_alpha_v1",
        }
        return BlendedForecastAdapter(
            baseline_model_dir=baseline_model_dir,
            relief_model_dir=relief_model_dir,
            dataset_dir=dataset_dir,
            mode=mode_map[source],
            device="cpu",
        )
    raise ValueError(f"unsupported forecast source: {source}")


def main() -> None:
    args = parse_args()
    profile_resolution = resolve_primary_control_profile(
        args=args,
        primary_safety_override=primary_safety_override,
        profile_registry_path=CASEBOOK_PROFILE_REGISTRY_PATH,
        repo_root=repo_root,
    )
    primary_control_profile = profile_resolution.control_profile
    primary_profile_registry_group = profile_resolution.registry_group
    primary_target_shape_cfg = profile_resolution.target_shape
    _enforce_pruned_overlay_flags(args)
    t0 = time.perf_counter()
    n_steps = int(round(float(args.duration_s) / DT))
    if n_steps <= 0:
        raise ValueError("--duration-s must be positive")
    duration_min = float(n_steps * DT / 60.0)
    selected_ids = {s.strip() for s in str(args.case_ids).split(",") if s.strip()}
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    primary_variant = str(args.primary_label)
    closed_profile = str(args.closed_pump_profile)
    closed_profile_cfg = closed_pump_cfg(closed_profile)
    primary_pump_profile = str(args.primary_pump_profile)
    primary_profile_cfg = closed_pump_cfg(primary_pump_profile)
    closed_target_shape_cfg = closed_target_shape_override(closed_profile)
    primary_target_shape_cfg = dict(primary_target_shape_cfg)
    primary_scale = float(args.primary_scale)
    forced_rows = _load_forced_prefix_actions(str(args.forced_prefix_actions))
    forced_bucket_allowlist = (
        {
            int(s.strip())
            for s in str(args.forced_prefix_buckets).split(",")
            if s.strip()
        }
        if str(args.forced_prefix_buckets).strip()
        else None
    )
    fig_dir = out_dir / "figures"
    ts_dir = out_dir / "timeseries"
    log_dir = out_dir / "planner_logs"

    planner_runtime_path = Path(args.planner_runtime_config)
    if not planner_runtime_path.is_absolute():
        planner_runtime_path = repo_root / planner_runtime_path
    planner_runtime = load_planner_runtime_config(planner_runtime_path)
    validate_execution_target_refresh_contract(
        planner_runtime,
        event_reset_mode=str(args.event_reset_mode),
        bias_shape="event_decay",
    )
    args.planner_runtime_config = str(planner_runtime_path.resolve())
    args.target_lifecycle_mode = str(planner_runtime.target_lifecycle_mode)
    args.candidate_action_mode = str(planner_runtime.candidate_action_mode)
    args.forecast_action_policy_mode = str(
        planner_runtime.forecast_action_policy_mode
    )
    args.forecast_action_policy_config = planner_runtime.forecast_action_policy
    if planner_runtime.target_lifecycle_mode == "planner_authoritative":
        # The clean v2 planner owns all economic target updates.  The submitted
        # v1 command-layer release remains available only under legacy mode.
        primary_target_shape_cfg["deadband_target_release_enabled"] = False
        primary_target_shape_cfg["forecast_safe_deadband_sync_provider_target"] = False
        primary_target_shape_cfg["forecast_safe_deadband_release_primary_target"] = False
    rollout_runtime = planner_runtime.execution_rollout
    rollout_config = ExecutionRolloutConfig(
        block_duration_s=rollout_runtime.block_duration_s,
        internal_step_s=rollout_runtime.internal_step_s,
        water_density_kg_m3=rollout_runtime.water_density_kg_m3,
        max_pump_rate_m3_min=rollout_runtime.max_pump_rate_m3_min,
        target_slew_enabled=rollout_runtime.target_slew_enabled,
        target_slew_rate_m3_min=rollout_runtime.target_slew_rate_m3_min,
        stop_error_kg=rollout_runtime.stop_error_kg,
        restart_error_kg=rollout_runtime.restart_error_kg,
        min_on_s=rollout_runtime.min_on_s,
        min_off_s=rollout_runtime.min_off_s,
        near_target_hold_s=rollout_runtime.near_target_hold_s,
        ramp_up_m3_min_per_s=rollout_runtime.ramp_up_m3_min_per_s,
        ramp_down_m3_min_per_s=rollout_runtime.ramp_down_m3_min_per_s,
        tank_capacity_kg=rollout_runtime.tank_capacity_kg,
        pump_rate_schedule_m3_min=rollout_runtime.pump_rate_schedule_m3_min,
    )
    cfg = PlannerConfig(
        pressure_sign_multiplier=planner_runtime.pressure_sign_multiplier,
        candidate_action_mode=planner_runtime.candidate_action_mode,
        pressure_aggregation_mode=planner_runtime.pressure_aggregation_mode,
        lead_reliability_enabled=planner_runtime.lead_reliability_enabled,
        execution_rollout_active=rollout_runtime.enabled,
        execution_rollout=rollout_config,
        target_replan_policy_mode=planner_runtime.target_replan_policy.mode,
        target_replan_rate_noise_tolerance_deg_s=(
            planner_runtime.target_replan_policy.rate_noise_tolerance_deg_s
        ),
        pressure_norm_cap=float(args.planner_pressure_norm_cap),
        envelope_use_discount=not bool(args.planner_envelope_raw),
        envelope_barrier_active=bool(args.planner_envelope_barrier),
        envelope_barrier_const=float(args.planner_envelope_barrier_const),
        posture_hold_relief_margin_norm=float(
            args.planner_posture_hold_relief_margin_norm
        ),
        posture_hold_forecast_credit=float(args.planner_posture_hold_forecast_credit),
        posture_state_residual_active=bool(args.planner_posture_state_residual),
        posture_state_gain=float(args.planner_posture_state_gain),
        posture_state_decay=float(args.planner_posture_state_decay),
        posture_state_clip_norm=float(args.planner_posture_state_clip_norm),
        w_attitude_residual=float(args.planner_attitude_residual_weight),
        w_terminal_residual=float(args.planner_terminal_residual_weight),
        objective_mode_active=bool(getattr(args, "planner_objective_mode_active", False)),
        hold_relief_debt_active=bool(args.hold_relief_debt),
        hold_relief_debt_weight=float(args.hold_relief_debt_weight),
        safety_floor_active=(
            primary_control_profile in {"safety_floor_v1", "h120_safety_watch_v1"}
        ),
        safety_floor_pitch_deg=float(os.environ.get("FOWT_FLOOR_PITCH_DEG", 5.0)),
        safety_floor_roll_deg=float(os.environ.get("FOWT_FLOOR_ROLL_DEG", 5.0)),
    )
    discounts = list(planner_runtime.default_discount_blocks)
    if bool(args.dry_config):
        dry = {
            "primary_control_profile": str(args.primary_control_profile),
            "primary_profile_registry_group": str(primary_profile_registry_group),
            "allow_isolated_profile": int(bool(args.allow_isolated_profile)),
            "planner_runtime_schema": str(planner_runtime.schema_version),
            "planner_target_lifecycle_mode": str(
                planner_runtime.target_lifecycle_mode
            ),
            "planner_candidate_action_mode": str(
                planner_runtime.candidate_action_mode
            ),
            "planner_pressure_aggregation_mode": str(
                planner_runtime.pressure_aggregation_mode
            ),
            "planner_lead_reliability_enabled": int(
                bool(planner_runtime.lead_reliability_enabled)
            ),
            "planner_forecast_action_policy_mode": str(
                planner_runtime.forecast_action_policy_mode
            ),
            "planner_target_replan_policy_mode": str(
                planner_runtime.target_replan_policy.mode
            ),
            "planner_execution_rollout_enabled": int(
                bool(rollout_runtime.enabled)
            ),
            "planner_execution_target_slew_enabled": int(
                bool(rollout_runtime.target_slew_enabled)
            ),
            "primary_safety_profile": str(args.primary_safety_profile),
            "primary_safety_pitch_enter_deg": float(
                primary_target_shape_cfg.get("primary_safety_pitch_enter_deg", np.nan)
            ),
            "primary_safety_roll_enter_deg": float(
                primary_target_shape_cfg.get("primary_safety_roll_enter_deg", np.nan)
            ),
            "primary_safety_pitch_exit_deg": float(
                primary_target_shape_cfg.get("primary_safety_pitch_exit_deg", np.nan)
            ),
            "primary_safety_roll_exit_deg": float(
                primary_target_shape_cfg.get("primary_safety_roll_exit_deg", np.nan)
            ),
            "primary_safety_emergency_pitch_enter_deg": float(
                primary_target_shape_cfg.get(
                    "primary_safety_emergency_pitch_enter_deg",
                    np.nan,
                )
            ),
            "primary_safety_emergency_roll_enter_deg": float(
                primary_target_shape_cfg.get(
                    "primary_safety_emergency_roll_enter_deg",
                    np.nan,
                )
            ),
            "primary_safety_enter_hold_s": float(
                primary_target_shape_cfg.get("primary_safety_enter_hold_s", np.nan)
            ),
            "forecast_source_requested": str(args.forecast_source),
            "forecast_control_trust_gate_enabled": int(
                bool(args.forecast_control_trust_gate)
            ),
            "forecast_control_trust_event_threshold": float(
                args.forecast_control_trust_event_threshold
            ),
            "forecast_control_trust_current_stability_required": int(
                bool(args.forecast_control_trust_current_stability_required)
            ),
            "forecast_control_trust_max_speed_range_ms": float(
                args.forecast_control_trust_max_speed_range_ms
            ),
            "forecast_control_trust_max_dir_shift_deg": float(
                args.forecast_control_trust_max_dir_shift_deg
            ),
            "forecast_control_trust_suppression_rise_norm": float(
                args.forecast_control_trust_suppression_rise_norm
            ),
            "forecast_control_trust_min_pressure_norm": float(
                args.forecast_control_trust_min_pressure_norm
            ),
            "forecast_pressure_trust_gate_enabled": int(
                bool(args.forecast_pressure_trust_gate)
            ),
            "forecast_pressure_trust_current_norm_max": float(
                args.forecast_pressure_trust_current_norm_max
            ),
            "forecast_pressure_trust_future_norm_min": float(
                args.forecast_pressure_trust_future_norm_min
            ),
            "forecast_pressure_trust_event_threshold": float(
                args.forecast_pressure_trust_event_threshold
            ),
            "forecast_pressure_trust_untrusted_scale": float(
                args.forecast_pressure_trust_untrusted_scale
            ),
            "forecast_pressure_trust_min_direction_dot": float(
                args.forecast_pressure_trust_min_direction_dot
            ),
            "forecast_pressure_trust_spike_margin_norm": float(
                args.forecast_pressure_trust_spike_margin_norm
            ),
            "trusted_event_gate_enabled": int(bool(args.trusted_event_gate)),
            "trusted_event_highwind_threshold": float(args.trusted_event_highwind_threshold),
            "trusted_event_attention_threshold": float(args.trusted_event_attention_threshold),
            "trusted_event_attention_min_heads": int(args.trusted_event_attention_min_heads),
            "trusted_event_dynamic_enabled": int(bool(args.trusted_event_dynamic)),
            "trusted_event_dynamic_threshold": float(args.trusted_event_dynamic_threshold),
            "trusted_event_dynamic_min_heads": int(args.trusted_event_dynamic_min_heads),
            "reactive_primary_only": int(bool(args.reactive_primary_only)),
            "primary_hold_target_mode": str(args.primary_hold_target_mode),
            "forecast_safe_deadband_enabled": int(
                bool(primary_target_shape_cfg.get("forecast_safe_deadband_enabled", False))
            ),
            "forecast_safe_deadband_gate_mode": str(
                primary_target_shape_cfg.get("forecast_safe_deadband_gate_mode", "off")
            ),
            "forecast_safe_deadband_pitch_enter_deg": float(
                primary_target_shape_cfg.get("forecast_safe_deadband_pitch_enter_deg", 0.0)
            ),
            "forecast_safe_deadband_roll_enter_deg": float(
                primary_target_shape_cfg.get("forecast_safe_deadband_roll_enter_deg", 0.0)
            ),
            "forecast_safe_deadband_pitch_exit_deg": float(
                primary_target_shape_cfg.get("forecast_safe_deadband_pitch_exit_deg", 0.0)
            ),
            "forecast_safe_deadband_roll_exit_deg": float(
                primary_target_shape_cfg.get("forecast_safe_deadband_roll_exit_deg", 0.0)
            ),
            "forecast_safe_deadband_event_probability_max": float(
                primary_target_shape_cfg.get(
                    "forecast_safe_deadband_event_probability_max", 0.0
                )
            ),
            "forecast_safe_deadband_require_event_probability": int(
                bool(
                    primary_target_shape_cfg.get(
                        "forecast_safe_deadband_require_event_probability", False
                    )
                )
            ),
            "deadband_target_release_enabled": int(
                bool(primary_target_shape_cfg.get("deadband_target_release_enabled", False))
            ),
            "deadband_target_release_pitch_deg": float(
                primary_target_shape_cfg.get("deadband_target_release_pitch_deg", 0.0)
            ),
            "deadband_target_release_roll_deg": float(
                primary_target_shape_cfg.get("deadband_target_release_roll_deg", 0.0)
            ),
            "deadband_target_release_exit_pitch_deg": float(
                primary_target_shape_cfg.get(
                    "deadband_target_release_exit_pitch_deg", 0.0
                )
            ),
            "deadband_target_release_exit_roll_deg": float(
                primary_target_shape_cfg.get(
                    "deadband_target_release_exit_roll_deg", 0.0
                )
            ),
            "planner_envelope_use_discount": int(bool(cfg.envelope_use_discount)),
            "planner_envelope_barrier_active": int(bool(cfg.envelope_barrier_active)),
            "planner_posture_state_residual_active": int(
                bool(cfg.posture_state_residual_active)
            ),
            "planner_posture_state_gain": float(cfg.posture_state_gain),
            "planner_posture_state_decay": float(cfg.posture_state_decay),
            "planner_posture_state_clip_norm": float(cfg.posture_state_clip_norm),
            "planner_attitude_residual_weight": float(cfg.w_attitude_residual),
            "planner_terminal_residual_weight": float(cfg.w_terminal_residual),
            "planner_posture_hold_forecast_credit": float(
                cfg.posture_hold_forecast_credit
            ),
            "hold_relief_debt_enabled": int(bool(args.hold_relief_debt)),
            "hold_relief_debt_weight": float(args.hold_relief_debt_weight),
            "relief_medium_cap_enabled": int(bool(args.relief_medium_cap)),
            "relief_medium_cap_adaptive": int(bool(args.relief_medium_cap_adaptive)),
            "hold_comfort_release_enabled": 0,
            "active_posture_refresh_enabled": int(bool(args.active_posture_refresh)),
            "active_posture_refresh_enter_deg": float(
                args.active_posture_refresh_enter_deg
            ),
            "active_posture_refresh_exit_deg": float(
                args.active_posture_refresh_exit_deg
            ),
            "active_posture_refresh_update_interval_s": float(
                args.active_posture_refresh_update_interval_s
            ),
            "active_posture_refresh_action_name": str(
                args.active_posture_refresh_action_name
            ),
            "active_posture_refresh_pressure_max_norm": float(
                args.active_posture_refresh_pressure_max_norm
            ),
            "active_posture_refresh_min_target_age_s": float(
                args.active_posture_refresh_min_target_age_s
            ),
            "active_posture_refresh_axis_mode": str(
                args.active_posture_refresh_axis_mode
            ),
            "active_posture_refresh_worsening_eps_deg_s": float(
                args.active_posture_refresh_worsening_eps_deg_s
            ),
            "active_posture_refresh_debt_deg": float(
                args.active_posture_refresh_debt_deg
            ),
            "active_posture_refresh_include_hold": int(
                bool(args.active_posture_refresh_include_hold)
            ),
            "fallback_risk_active_release_enabled": int(
                bool(args.fallback_risk_active_release)
            ),
            "fallback_risk_active_release_enter_deg": float(
                args.fallback_risk_active_release_enter_deg
            ),
            "fallback_risk_active_release_worsening_eps_deg_s": float(
                args.fallback_risk_active_release_worsening_eps_deg_s
            ),
            "preview_lead_action_enabled": int(bool(args.preview_lead_action)),
            "primary_stall_refresh_enabled": int(bool(args.primary_stall_refresh)),
            "primary_stall_pitch_axis_bias": int(bool(args.primary_stall_pitch_axis_bias)),
            "primary_stall_pitch_deg": float(args.primary_stall_pitch_deg),
            "primary_stall_roll_deg": float(args.primary_stall_roll_deg),
            "primary_stall_target_err_kg": float(args.primary_stall_target_err_kg),
            "primary_stall_pump_rate_m3_min": float(args.primary_stall_pump_rate_m3_min),
            "primary_stall_min_age_s": float(args.primary_stall_min_age_s),
            "no_unexplained_hold_action_enabled": 0,
            "no_unexplained_hold_threshold_deg": 0.0,
            "no_unexplained_hold_trigger_buckets": 0,
            "medium_escalation_enabled": 0,
            "recovery_mode_enabled": 0,
            "far_horizon_enabled": int(bool(args.far_horizon)),
            "far_horizon_relief_gate_enabled": int(bool(args.far_horizon_relief_gate)),
            "far_horizon_weights": [
                float(args.far_horizon_weight_60_80),
                float(args.far_horizon_weight_80_100),
                float(args.far_horizon_weight_100_120),
            ],
            "h120_oracle_probe_enabled": int(bool(args.h120_oracle_probe)),
            "h120_oracle_probe_short_delay_s": float(
                args.h120_oracle_probe_short_delay_s
            ),
            "h120_oracle_probe_far_high_norm": float(
                args.h120_oracle_probe_far_high_norm
            ),
            "h120_oracle_probe_intensify_margin_norm": float(
                args.h120_oracle_probe_intensify_margin_norm
            ),
            "h120_risk_scheduler_enabled": int(bool(args.h120_risk_scheduler)),
            "h120_scheduler_delay_normal_s": float(args.h120_scheduler_delay_normal_s),
            "h120_scheduler_delay_risk_aware_s": float(
                args.h120_scheduler_delay_risk_aware_s
            ),
            "h120_scheduler_delay_high_risk_s": float(
                args.h120_scheduler_delay_high_risk_s
            ),
            "h120_scheduler_far_high_norm": float(args.h120_scheduler_far_high_norm),
            "h120_scheduler_intensify_margin_norm": float(
                args.h120_scheduler_intensify_margin_norm
            ),
            "h120_scheduler_relief_margin_norm": float(
                args.h120_scheduler_relief_margin_norm
            ),
            "h120_scheduler_early_stop_suppress_max_per_episode": int(
                args.h120_scheduler_early_stop_suppress_max_per_episode
            ),
            "h120_scheduler_prefloor_probe_enabled": int(
                bool(args.h120_scheduler_prefloor_probe)
            ),
            "h120_scheduler_prefloor_action": str(
                args.h120_scheduler_prefloor_action
            ),
            "h120_scheduler_prefloor_enter_deg": float(
                args.h120_scheduler_prefloor_enter_deg
            ),
            "h120_scheduler_prefloor_floor_deg": float(
                args.h120_scheduler_prefloor_floor_deg
            ),
            "h120_scheduler_prefloor_min_delta_kg": float(
                args.h120_scheduler_prefloor_min_delta_kg
            ),
            "h120_scheduler_prefloor_max_delta_kg": float(
                args.h120_scheduler_prefloor_max_delta_kg
            ),
            "h120_scheduler_prefloor_cooldown_s": float(
                args.h120_scheduler_prefloor_cooldown_s
            ),
            "h120_axis_micro_enabled": int(bool(args.h120_axis_micro)),
            "h120_axis_micro_enter_deg": float(args.h120_axis_micro_enter_deg),
            "h120_axis_micro_floor_deg": float(args.h120_axis_micro_floor_deg),
            "h120_axis_micro_rise_floor_norm": float(
                args.h120_axis_micro_rise_floor_norm
            ),
            "h120_axis_micro_worsening_eps_deg": float(
                args.h120_axis_micro_worsening_eps_deg
            ),
            "h120_axis_micro_delta_kg": float(args.h120_axis_micro_delta_kg),
            "h120_axis_micro_cooldown_s": float(args.h120_axis_micro_cooldown_s),
            "relief_economy_enabled": int(bool(args.relief_economy)),
            "relief_economy_horizon": str(args.relief_economy_horizon),
            "relief_economy_safe_deg": float(args.relief_economy_safe_deg),
            "relief_economy_worsening_eps_deg": float(
                args.relief_economy_worsening_eps_deg
            ),
            "relief_economy_low_norm": float(args.relief_economy_low_norm),
            "relief_economy_high_norm": float(args.relief_economy_high_norm),
            "relief_envelope_enabled": int(bool(args.relief_envelope)),
            "relief_envelope_horizon": str(args.relief_envelope_horizon),
            "relief_envelope_allowed_0_20_deg": float(
                args.relief_envelope_allowed_0_20_deg
            ),
            "relief_envelope_allowed_20_40_deg": float(
                args.relief_envelope_allowed_20_40_deg
            ),
            "relief_envelope_allowed_40_60_deg": float(
                args.relief_envelope_allowed_40_60_deg
            ),
            "relief_envelope_allowed_60_120_deg": float(
                args.relief_envelope_allowed_60_120_deg
            ),
            "relief_envelope_max_duration_s": float(
                args.relief_envelope_max_duration_s
            ),
            "relief_envelope_debt_budget_deg_s": float(
                args.relief_envelope_debt_budget_deg_s
            ),
            "relief_envelope_near_limit_deg": float(
                args.relief_envelope_near_limit_deg
            ),
            "relief_envelope_near_limit_budget_s": float(
                args.relief_envelope_near_limit_budget_s
            ),
            "relief_envelope_worsening_eps_deg": float(
                args.relief_envelope_worsening_eps_deg
            ),
            "relief_envelope_low_norm": float(args.relief_envelope_low_norm),
            "relief_envelope_high_norm": float(args.relief_envelope_high_norm),
            "relief_envelope_reintensify_norm": float(
                args.relief_envelope_reintensify_norm
            ),
            "relief_envelope_single_axis_only": int(
                bool(args.relief_envelope_single_axis_only)
            ),
            "relief_envelope_axis_minor_ratio_max": float(
                args.relief_envelope_axis_minor_ratio_max
            ),
            "relief_envelope_axis_minor_abs_max_deg": float(
                args.relief_envelope_axis_minor_abs_max_deg
            ),
            "relief_envelope_adaptive_axis_cap": int(
                bool(args.relief_envelope_adaptive_axis_cap)
            ),
            "relief_envelope_cap_relief_norm": float(
                args.relief_envelope_cap_relief_norm
            ),
            "relief_envelope_cap_hold_fraction": float(
                args.relief_envelope_cap_hold_fraction
            ),
            "economy_pump_budget_enabled": int(bool(args.economy_pump_budget)),
            "economy_pump_budget_m3": float(args.economy_pump_budget_m3),
            "economy_pump_budget_hold_refresh_fraction": float(
                args.economy_pump_budget_hold_refresh_fraction
            ),
            "economy_pump_budget_dynamic_refresh_enabled": int(
                bool(args.economy_pump_budget_dynamic_refresh)
            ),
            "economy_pump_budget_dynamic_refresh_enter_deg": float(
                args.economy_pump_budget_dynamic_refresh_enter_deg
            ),
            "economy_pump_budget_dynamic_refresh_full_deg": float(
                args.economy_pump_budget_dynamic_refresh_full_deg
            ),
            "economy_pump_budget_dynamic_refresh_max_fraction": float(
                args.economy_pump_budget_dynamic_refresh_max_fraction
            ),
            "economy_pump_budget_allocator_mode": str(
                args.economy_pump_budget_allocator_mode
            ),
            "economy_pump_budget_require_relief": int(
                bool(args.economy_pump_budget_require_relief)
            ),
            "economy_pump_budget_relief_norm": float(args.economy_pump_budget_relief_norm),
            "economy_pump_budget_relief_drop_norm": float(
                args.economy_pump_budget_relief_drop_norm
            ),
            "economy_pump_budget_forecast_smart": int(
                bool(args.economy_pump_budget_forecast_smart)
            ),
            "economy_pump_budget_smart_posture_deg": float(
                args.economy_pump_budget_smart_posture_deg
            ),
            "economy_pump_budget_smart_pressure_norm": float(
                args.economy_pump_budget_smart_pressure_norm
            ),
            "economy_pump_budget_opportunity_window_s": float(
                args.economy_pump_budget_opportunity_window_s
            ),
            "economy_pump_budget_opportunity_min_pump_m3": float(
                args.economy_pump_budget_opportunity_min_pump_m3
            ),
            "economy_pump_budget_opportunity_min_refresh": int(
                args.economy_pump_budget_opportunity_min_refresh
            ),
            "economy_pump_budget_opportunity_min_delta_kg": float(
                args.economy_pump_budget_opportunity_min_delta_kg
            ),
            "forecast_advised_economy_enabled": int(bool(args.forecast_advised_economy)),
            "forecast_advised_economy_confirm_buckets": int(
                args.forecast_advised_economy_confirm_buckets
            ),
            "forecast_advised_economy_hold_confirm_buckets": int(
                args.forecast_advised_economy_hold_confirm_buckets
            ),
            "forecast_advised_economy_posture_release_deg": float(
                args.forecast_advised_economy_posture_release_deg
            ),
            "forecast_advised_economy_release_cooldown_s": float(
                args.forecast_advised_economy_release_cooldown_s
            ),
            "forecast_pause_resume_max_target_err_kg": float(
                args.forecast_pause_resume_max_target_err_kg
            ),
            "stale_active_target_refresh_mode": str(args.stale_active_target_refresh),
            "stale_active_target_refresh_min_active_buckets": int(
                args.stale_active_target_refresh_min_active_buckets
            ),
            "stale_active_target_refresh_pitch_deg": float(
                args.stale_active_target_refresh_pitch_deg
            ),
            "stale_active_target_refresh_roll_deg": float(
                args.stale_active_target_refresh_roll_deg
            ),
            "stale_active_target_refresh_target_err_kg": float(
                args.stale_active_target_refresh_target_err_kg
            ),
            "stale_active_target_refresh_pump_rate_m3_min": float(
                args.stale_active_target_refresh_pump_rate_m3_min
            ),
            "stale_active_target_refresh_min_target_age_s": float(
                args.stale_active_target_refresh_min_target_age_s
            ),
            "stale_active_target_refresh_min_delta_kg": float(
                args.stale_active_target_refresh_min_delta_kg
            ),
            "stale_active_target_refresh_cooldown_s": float(
                args.stale_active_target_refresh_cooldown_s
            ),
            "stale_active_target_refresh_no_recovery_eps_deg": float(
                args.stale_active_target_refresh_no_recovery_eps_deg
            ),
            "h120_floor_shaping_mode": str(args.h120_floor_shaping_mode),
            "h120_floor_shaping_far_high_norm": float(
                args.h120_floor_shaping_far_high_norm
            ),
            "h120_floor_shaping_slow_response_eps_deg": float(
                args.h120_floor_shaping_slow_response_eps_deg
            ),
            "h120_floor_shaping_medium_delay_s": float(
                args.h120_floor_shaping_medium_delay_s
            ),
            "forced_prefix_enabled": int(
                bool(forced_rows) and str(args.forced_prefix_mode) != "off"
            ),
            "forced_prefix_mode": str(args.forced_prefix_mode),
            "forced_prefix_rows": int(len(forced_rows)),
        }
        print(json.dumps(dry, indent=2, sort_keys=True))
        return

    indexed_cases = _load_indexed_cases(str(args.cases_csv))
    if selected_ids:
        indexed_cases = [
            item
            for item in indexed_cases
            if f"{item[0]:02d}" in selected_ids
            or str(item[0]) in selected_ids
            or item[1] in selected_ids
        ]
        if not indexed_cases:
            raise ValueError(
                "--case-ids did not match any loaded case. "
                f"Requested: {sorted(selected_ids)}. "
                "Use numeric ids such as 01,04,10 for the built-in case list, "
                "or pass --cases-csv with matching case_id values."
            )
    enforce_casebook_experiment_limits(
        duration_s=float(args.duration_s),
        case_count=len(indexed_cases),
    )

    for d in (fig_dir, ts_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)
    excel_path = discover_excel(args.stiffness_file)
    args.stiffness_file = excel_path
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = repo_root / dataset_dir
    replay = Fino1ReplayDataset(
        dataset_dir=dataset_dir,
        split=str(args.replay_split),
    )
    forecast_adapter = _make_forecast_adapter(args)
    forecast_source_label = (
        "oracle_future" if forecast_adapter is None else str(forecast_adapter.model_version)
    )

    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    rows: list[dict] = []
    issues: list[str] = []
    completed = 0
    snapshot_case_ids = {
        s.strip()
        for s in str(args.bucket_snapshot_case_ids).split(",")
        if s.strip()
    }
    snapshot_bucket_ids = [
        int(s.strip())
        for s in str(args.bucket_snapshot_ids).split(",")
        if s.strip()
    ]
    snapshot_dir = (
        Path(args.bucket_snapshot_dir)
        if str(args.bucket_snapshot_dir).strip()
        else None
    )
    if snapshot_dir is not None and not snapshot_dir.is_absolute():
        snapshot_dir = repo_root / snapshot_dir
    for case_num, raw_case_id, ts_str, label in indexed_cases:
        case_id = _numbered_case_id(case_num, raw_case_id)
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        try:
            row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
            wind_trace = smooth_wind_trace(wind_trace, float(args.wind_transition_s))
        except Exception as e:
            issues.append(f"{raw_case_id} {ts_str}: wind_trace failed: {e}")
            continue

        dfs: dict[str, pd.DataFrame] = {}
        providers: dict[str, BallastPlannerPreviewProvider | None] = {}
        variants = (primary_variant,) if bool(args.primary_only) else ("closed_only", primary_variant)
        for variant in variants:
            provider = None
            if variant == primary_variant:
                if primary_control_profile in {
                    "noop_control_v1",
                    "dc_preserving_deadband_v1",
                    "dc_preserving_deadband_engineered_v1",
                    "dc_preserving_deadband_w1_pid_veto_v1",
                    "dc_target_lowpass_v1",
                    "dc_target_ac_filter_v1",
                }:
                    provider = None
                elif primary_control_profile in {
                    "dc_preserving_deadband_w1_veto_v1",
                    "dc_preserving_deadband_conservative_v1",
                    "dc_preserving_deadband_forecast_adaptive_v1",
                    "dc_preserving_deadband_forecast_widen_v1",
                }:
                    provider = BallastPlannerPreviewProvider(
                        replay_dataset=replay,
                        start_timestamp=ts,
                        cfg=cfg,
                        block_discounts=discounts,
                        bias_shape="event_decay",
                        objective_mode="economic",
                        event_reset_mode=str(args.event_reset_mode),
                        target_reset_tol_deg=float(args.target_reset_tol_deg),
                        setpoint_channel_enabled=False,
                        prediction_primary_enabled=False,
                        ff_channel_enabled=False,
                        far_horizon_enabled=True,
                        forecast_adapter=forecast_adapter,
                    )
                elif bool(args.reactive_primary_only):
                    provider = ReactivePrimaryProvider(
                        cfg=cfg,
                        action_name=str(args.reactive_primary_action),
                        enter_pitch_deg=float(args.reactive_primary_enter_deg),
                        enter_roll_deg=float(args.reactive_primary_enter_deg),
                        exit_pitch_deg=float(args.reactive_primary_exit_deg),
                        exit_roll_deg=float(args.reactive_primary_exit_deg),
                        update_interval_s=float(args.reactive_primary_update_interval_s),
                        prediction_primary_scale=primary_scale,
                    )
                else:
                    forced_for_case = _forced_prefix_for_case(
                        forced_rows,
                        case_num,
                        raw_case_id,
                        case_id,
                        forced_bucket_allowlist,
                    )
                    provider = build_forecast_assisted_provider(
                        replay_dataset=replay,
                        start_timestamp=ts,
                        planner=cfg,
                        block_discounts=discounts,
                        args=args,
                        primary_scale=primary_scale,
                        forecast_adapter=forecast_adapter,
                        forced_prefix_actions=forced_for_case,
                    )
            providers[variant] = provider
            run_name = f"{case_id}_{ts_str.replace(':', '').replace(' ', '_')}_{variant}"
            print(f"--- {run_name} ---", flush=True)
            try:
                use_snapshots = (
                    snapshot_dir is not None
                    and bool(snapshot_bucket_ids)
                    and (
                        not snapshot_case_ids
                        or f"{case_num:02d}" in snapshot_case_ids
                        or str(case_num) in snapshot_case_ids
                        or raw_case_id in snapshot_case_ids
                        or case_id in snapshot_case_ids
                    )
                    and variant == primary_variant
                )
                controller_override = None
                if variant == primary_variant and primary_control_profile in {
                    "dc_preserving_deadband_v1",
                    "dc_preserving_deadband_engineered_v1",
                    "dc_preserving_deadband_engineered_recovery_observer_v1",
                    "dc_preserving_deadband_w1_veto_v1",
                    "dc_preserving_deadband_conservative_v1",
                    "dc_preserving_deadband_w1_pid_veto_v1",
                    "dc_preserving_deadband_forecast_adaptive_v1",
                    "dc_preserving_deadband_forecast_widen_v1",
                }:
                    _tight_base = primary_control_profile in {
                        "dc_preserving_deadband_w1_pid_veto_v1",
                        "dc_preserving_deadband_forecast_widen_v1",
                    }
                    conservative_deadband = (
                        primary_control_profile
                        == "dc_preserving_deadband_conservative_v1"
                    )
                    pid_deadband_pitch_default = (
                        1.2 if conservative_deadband else (1.0 if _tight_base else 1.5)
                    )
                    pid_deadband_roll_default = (
                        1.2 if conservative_deadband else (0.8 if _tight_base else 1.5)
                    )
                    pid_deadband_exit_default = (
                        0.65 if conservative_deadband else (0.5 if _tight_base else 0.8)
                    )
                    controller_override = {
                        "deadband_pitch": float(
                            os.environ.get(
                                "FOWT_DC_PID_DEADBAND_PITCH_DEG",
                                pid_deadband_pitch_default,
                            )
                        ),
                        "deadband_roll": float(
                            os.environ.get(
                                "FOWT_DC_PID_DEADBAND_ROLL_DEG",
                                pid_deadband_roll_default,
                            )
                        ),
                        "deadband_exit_ratio": float(
                            os.environ.get(
                                "FOWT_DC_PID_DEADBAND_EXIT_RATIO",
                                pid_deadband_exit_default,
                            )
                        ),
                    }
                _, timeseries = run_closed_loop_case(
                    excel_path=excel_path,
                    case_name=run_name,
                    dt=DT,
                    n_steps=n_steps,
                    wind_trace=wind_trace,
                    control_enabled=True,
                    record_timeseries=True,
                    experiment_protocol="main",
                    platform_profile="default",
                    start_from_heave_equilibrium=True,
                    preview_trim_provider=provider,
                    pump_cfg=closed_profile_cfg if variant == "closed_only" else primary_profile_cfg,
                    controller_cfg=controller_override,
                    target_shape_override_cfg=(
                        closed_target_shape_cfg if variant == "closed_only" else primary_target_shape_cfg
                    ),
                    bucket_snapshot_dir=snapshot_dir if use_snapshots else None,
                    bucket_snapshot_ids=snapshot_bucket_ids if use_snapshots else None,
                    bucket_snapshot_horizon_buckets=int(
                        args.bucket_snapshot_horizon_buckets
                    ),
                )
            except Exception as e:
                issues.append(f"{run_name}: {type(e).__name__}: {e}")
                issues.append(traceback.format_exc())
                dfs = {}
                break
            df = pd.DataFrame(timeseries)
            df.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
            dfs[variant] = df
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(log_dir / f"{run_name}_planner_log.csv", index=False)

        if primary_variant not in dfs:
            continue
        has_closed = "closed_only" in dfs
        if not has_closed and not bool(args.primary_only):
            continue

        planner_log = None
        provider = providers.get(primary_variant)
        if provider is not None and provider.records:
            planner_log = pd.DataFrame(provider.records)
        fig_path: Path | None = None
        if has_closed and not bool(args.skip_figures):
            fig_path = plot_case(
                case_id,
                label,
                ts_str,
                dfs["closed_only"],
                dfs[primary_variant],
                fig_dir,
                primary_variant,
                duration_min=duration_min,
                planner_log=planner_log,
            )
        sc = summarize(dfs["closed_only"]) if has_closed else {}
        sp = summarize(dfs[primary_variant])
        if planner_log is not None and not planner_log.empty:
            zeros_log = pd.Series(np.zeros(len(planner_log)), index=planner_log.index)
            sp["preview_far_horizon_available_ratio"] = float(
                np.mean(
                    planner_log.get(
                        "far_horizon_available",
                        zeros_log,
                    ).to_numpy(dtype=float)
                )
            )
            sp["preview_far_horizon_hint_ratio"] = float(
                np.mean(
                    planner_log.get(
                        "far_horizon_hint_any",
                        zeros_log,
                    ).to_numpy(dtype=float)
                )
            )
            sp["preview_far_horizon_relief_ratio"] = float(
                np.mean(
                    planner_log.get(
                        "far_horizon_hidden_relief",
                        zeros_log,
                    ).to_numpy(dtype=float)
                )
            )
            sp["preview_far_horizon_relief_gate_ratio"] = float(
                np.mean(
                    planner_log.get(
                        "far_horizon_relief_gate_active",
                        zeros_log,
                    ).to_numpy(dtype=float)
                )
            )
            sp["h120_oracle_probe_active_ratio"] = float(
                np.mean(
                    planner_log.get(
                        "h120_oracle_probe_active",
                        zeros_log,
                    ).to_numpy(dtype=float)
                )
            )
            sp["h120_oracle_probe_preemptive_count"] = float(
                planner_log.get(
                    "h120_oracle_probe_preemptive_active",
                    zeros_log,
                ).to_numpy(dtype=float).sum()
            )
            sp["h120_oracle_probe_short_delay_count"] = float(
                planner_log.get(
                    "h120_oracle_probe_short_delay_active",
                    zeros_log,
                ).to_numpy(dtype=float).sum()
            )
            sp["h120_oracle_probe_early_stop_veto_count"] = float(
                planner_log.get(
                    "h120_oracle_probe_early_stop_veto_active",
                    zeros_log,
                ).to_numpy(dtype=float).sum()
            )
            sp["h120_scheduler_remote_risk_ratio"] = float(
                np.mean(
                    planner_log.get(
                        "h120_scheduler_remote_risk_active",
                        zeros_log,
                    ).to_numpy(dtype=float)
                )
            )
            sp["h120_scheduler_risk_aware_rows"] = float(
                (
                    planner_log.get(
                        "h120_scheduler_risk_tier",
                        pd.Series("", index=planner_log.index),
                    )
                    .astype(str)
                    .eq("risk_aware")
                    .sum()
                )
            )
            sp["h120_scheduler_high_risk_rows"] = float(
                (
                    planner_log.get(
                        "h120_scheduler_risk_tier",
                        pd.Series("", index=planner_log.index),
                    )
                    .astype(str)
                    .eq("high_risk")
                    .sum()
                )
            )
            sp["h120_scheduler_early_stop_suppressed_count"] = float(
                planner_log.get(
                    "h120_scheduler_early_stop_suppressed_active",
                    zeros_log,
                ).to_numpy(dtype=float).sum()
            )
            sp["h120_scheduler_prefloor_active_count"] = float(
                planner_log.get(
                    "h120_scheduler_prefloor_active",
                    zeros_log,
                ).to_numpy(dtype=float).sum()
            )
        else:
            sp["h120_oracle_probe_active_ratio"] = 0.0
            sp["h120_oracle_probe_preemptive_count"] = 0.0
            sp["h120_oracle_probe_short_delay_count"] = 0.0
            sp["h120_oracle_probe_early_stop_veto_count"] = 0.0
            sp["h120_scheduler_remote_risk_ratio"] = 0.0
            sp["h120_scheduler_risk_aware_rows"] = 0.0
            sp["h120_scheduler_high_risk_rows"] = 0.0
            sp["h120_scheduler_early_stop_suppressed_count"] = 0.0
            sp["h120_scheduler_prefloor_active_count"] = 0.0
        row = {
            "case_id": case_id,
            "timestamp": ts_str,
            "label": label,
            "figure": str(fig_path.relative_to(repo_root)) if fig_path is not None else "",
            "closed_pump_work_m3": sc.get("pump_work_m3", np.nan),
            "primary_pump_work_m3": sp["pump_work_m3"],
            "d_pump_work_pct": (
                (sp["pump_work_m3"] - sc["pump_work_m3"]) / max(sc["pump_work_m3"], 1e-9) * 100.0
                if has_closed
                else np.nan
            ),
            "closed_pitch_p95": sc.get("pitch_abs_p95", np.nan),
            "primary_pitch_p95": sp["pitch_abs_p95"],
            "d_pitch_p95": sp["pitch_abs_p95"] - sc["pitch_abs_p95"] if has_closed else np.nan,
            "closed_roll_p95": sc.get("roll_abs_p95", np.nan),
            "primary_roll_p95": sp["roll_abs_p95"],
            "d_roll_p95": sp["roll_abs_p95"] - sc["roll_abs_p95"] if has_closed else np.nan,
            "closed_latch_switches": sc.get("latch_switches", np.nan),
            "primary_latch_switches": sp["latch_switches"],
            "primary_delta_mean_kg": sp["primary_delta_mean_kg"],
            "primary_candidate_ratio": sp["primary_candidate_ratio"],
            "primary_applied_ratio": sp["primary_applied_ratio"],
            "primary_safety_fallback_ratio": sp["primary_safety_fallback_ratio"],
            "primary_safety_active_ratio": sp["primary_safety_active_ratio"],
            "primary_safety_transition_count": sp["primary_safety_transition_count"],
            "preview_pump_suppression_ratio": sp["preview_pump_suppression_ratio"],
            "preview_suppression_delta_mean_kg": sp["preview_suppression_delta_mean_kg"],
            "safe_pump_freeze_ratio": sp["safe_pump_freeze_ratio"],
            "safe_pump_freeze_delta_mean_kg": sp["safe_pump_freeze_delta_mean_kg"],
            "preview_event_risk_boost_ratio": sp["preview_event_risk_boost_ratio"],
            "preview_event_risk_scale_mean": sp["preview_event_risk_scale_mean"],
            "preview_event_risk_floor_ratio": sp["preview_event_risk_floor_ratio"],
            "preview_event_risk_floor_norm_mean": sp["preview_event_risk_floor_norm_mean"],
            "preview_trusted_event_gate_enabled_ratio": sp[
                "preview_trusted_event_gate_enabled_ratio"
            ],
            "preview_trusted_event_gate_untrusted_ratio": sp[
                "preview_trusted_event_gate_untrusted_ratio"
            ],
            "preview_event_risk_raw_prob_mean": sp[
                "preview_event_risk_raw_prob_mean"
            ],
            "preview_event_risk_effective_prob_mean": sp[
                "preview_event_risk_effective_prob_mean"
            ],
            "preview_forecast_pressure_trust_enabled_ratio": sp[
                "preview_forecast_pressure_trust_enabled_ratio"
            ],
            "preview_forecast_pressure_trust_block_ratio": sp[
                "preview_forecast_pressure_trust_block_ratio"
            ],
            "preview_forecast_pressure_trust_scale_mean": sp[
                "preview_forecast_pressure_trust_scale_mean"
            ],
            "preview_raw_pressure_future_max_mean": sp[
                "preview_raw_pressure_future_max_mean"
            ],
            "preview_effective_pressure_future_max_mean": sp[
                "preview_effective_pressure_future_max_mean"
            ],
            "preview_forecast_pressure_trust_current_support_ratio": sp[
                "preview_forecast_pressure_trust_current_support_ratio"
            ],
            "preview_forecast_pressure_trust_current_observed_norm_mean": sp[
                "preview_forecast_pressure_trust_current_observed_norm_mean"
            ],
            "preview_forecast_pressure_trust_current_speed_mean_ms": sp[
                "preview_forecast_pressure_trust_current_speed_mean_ms"
            ],
            "preview_forecast_pressure_trust_future_speed_max_mean_ms": sp[
                "preview_forecast_pressure_trust_future_speed_max_mean_ms"
            ],
            "preview_forecast_pressure_trust_event_support_ratio": sp[
                "preview_forecast_pressure_trust_event_support_ratio"
            ],
            "preview_forecast_pressure_trust_shape_support_ratio": sp[
                "preview_forecast_pressure_trust_shape_support_ratio"
            ],
            "preview_lead_action_ratio": sp["preview_lead_action_ratio"],
            "preview_relief_medium_cap_ratio": sp["preview_relief_medium_cap_ratio"],
            "preview_medium_escalation_ratio": sp["preview_medium_escalation_ratio"],
            "preview_far_horizon_available_ratio": sp[
                "preview_far_horizon_available_ratio"
            ],
            "preview_far_horizon_hint_ratio": sp["preview_far_horizon_hint_ratio"],
            "preview_far_horizon_relief_ratio": sp["preview_far_horizon_relief_ratio"],
            "preview_far_horizon_relief_gate_ratio": sp[
                "preview_far_horizon_relief_gate_ratio"
            ],
            "preview_active_effectiveness_refresh_ratio": sp[
                "preview_active_effectiveness_refresh_ratio"
            ],
            "preview_sustained_active_recompute_ratio": sp[
                "preview_sustained_active_recompute_ratio"
            ],
            "preview_hold_comfort_release_ratio": sp[
                "preview_hold_comfort_release_ratio"
            ],
            "preview_hold_comfort_veto_ratio": sp["preview_hold_comfort_veto_ratio"],
            "preview_hold_risk_micro_action_ratio": sp[
                "preview_hold_risk_micro_action_ratio"
            ],
            "preview_no_unexplained_hold_action_ratio": sp[
                "preview_no_unexplained_hold_action_ratio"
            ],
            "preview_primary_hold_feedback_ratio": sp[
                "preview_primary_hold_feedback_ratio"
            ],
            "preview_primary_hold_feedback_delta_mean_kg": sp[
                "preview_primary_hold_feedback_delta_mean_kg"
            ],
            "preview_posture_hold_barrier_ratio": sp[
                "preview_posture_hold_barrier_ratio"
            ],
            "preview_posture_hold_barrier_cost_mean": sp[
                "preview_posture_hold_barrier_cost_mean"
            ],
            "preview_posture_hold_norm_mean": sp["preview_posture_hold_norm_mean"],
            "closed_pump_profile": closed_profile,
            "primary_pump_profile": primary_pump_profile,
            "primary_scale": primary_scale,
            "primary_control_profile": str(args.primary_control_profile),
            "economy_pump_budget_hold_refresh_fraction": float(
                args.economy_pump_budget_hold_refresh_fraction
            ),
            "economy_pump_budget_dynamic_refresh_enabled": int(
                bool(args.economy_pump_budget_dynamic_refresh)
            ),
            "economy_pump_budget_dynamic_refresh_enter_deg": float(
                args.economy_pump_budget_dynamic_refresh_enter_deg
            ),
            "economy_pump_budget_dynamic_refresh_full_deg": float(
                args.economy_pump_budget_dynamic_refresh_full_deg
            ),
            "economy_pump_budget_dynamic_refresh_max_fraction": float(
                args.economy_pump_budget_dynamic_refresh_max_fraction
            ),
            "primary_safety_profile": str(args.primary_safety_profile),
            "forecast_source_requested": str(args.forecast_source),
            "forecast_source_effective": forecast_source_label,
            "forecast_control_trust_gate_enabled": int(
                bool(args.forecast_control_trust_gate)
            ),
            "forecast_control_trust_event_threshold": float(
                args.forecast_control_trust_event_threshold
            ),
            "forecast_control_trust_current_stability_required": int(
                bool(args.forecast_control_trust_current_stability_required)
            ),
            "forecast_control_trust_max_speed_range_ms": float(
                args.forecast_control_trust_max_speed_range_ms
            ),
            "forecast_control_trust_max_dir_shift_deg": float(
                args.forecast_control_trust_max_dir_shift_deg
            ),
            "forecast_control_trust_suppression_rise_norm": float(
                args.forecast_control_trust_suppression_rise_norm
            ),
            "forecast_control_trust_min_pressure_norm": float(
                args.forecast_control_trust_min_pressure_norm
            ),
            "event_reset_mode": str(args.event_reset_mode),
            "primary_hold_target_mode": str(args.primary_hold_target_mode),
            "primary_target_resumed_ratio": sp["primary_target_resumed_ratio"],
            "planner_envelope_use_discount": int(bool(cfg.envelope_use_discount)),
            "planner_envelope_mode": "discounted" if cfg.envelope_use_discount else "raw",
            "planner_envelope_barrier_active": int(bool(cfg.envelope_barrier_active)),
            "planner_envelope_barrier_const": float(cfg.envelope_barrier_const),
            "planner_posture_hold_barrier_active": 0,
            "planner_posture_hold_pitch_deg": 0.0,
            "planner_posture_hold_roll_deg": 0.0,
            "planner_posture_hold_barrier_const": 0.0,
            "planner_posture_hold_relief_margin_norm": float(
                cfg.posture_hold_relief_margin_norm
            ),
            "planner_posture_hold_forecast_credit": float(
                cfg.posture_hold_forecast_credit
            ),
            "planner_posture_hold_action_uses_posture_vec": 0,
            "planner_posture_hold_low_risk_norm_max": 0.0,
            "planner_posture_state_residual_active": int(
                bool(cfg.posture_state_residual_active)
            ),
            "planner_posture_state_gain": float(cfg.posture_state_gain),
            "planner_posture_state_decay": float(cfg.posture_state_decay),
            "planner_posture_state_clip_norm": float(cfg.posture_state_clip_norm),
            "planner_attitude_residual_weight": float(cfg.w_attitude_residual),
            "planner_terminal_residual_weight": float(cfg.w_terminal_residual),
            "planner_pressure_norm_cap": float(cfg.pressure_norm_cap),
            "target_reset_tol_deg": float(args.target_reset_tol_deg),
            "pump_suppression_enabled": 0,
            "pump_suppression_restart_err_kg": 0.0,
            "pump_suppression_relief_margin_norm": 0.0,
            "pump_suppression_low_risk_norm": 0.0,
            "pump_suppression_plateau_enabled": 0,
            "pump_suppression_plateau_only": 0,
            "pump_suppression_plateau_high_norm": 0.0,
            "pump_suppression_plateau_range_norm": 0.0,
            "pump_suppression_plateau_rise_norm": 0.0,
            "pump_suppression_plateau_restart_err_kg": 0.0,
            "forecast_advised_economy_enabled": int(
                bool(args.forecast_advised_economy)
            ),
            "forecast_advised_economy_suppression_only": int(
                bool(args.forecast_advised_economy_suppression_only)
            ),
            "forecast_advised_economy_min_headroom_deg": float(
                args.forecast_advised_economy_min_headroom_deg
            ),
            "forecast_advised_economy_rise_norm": float(
                args.forecast_advised_economy_rise_norm
            ),
            "forecast_advised_economy_max_speed_range_ms": float(
                args.forecast_advised_economy_max_speed_range_ms
            ),
            "forecast_advised_economy_max_dir_shift_deg": float(
                args.forecast_advised_economy_max_dir_shift_deg
            ),
            "forecast_advised_economy_confirm_buckets": int(
                args.forecast_advised_economy_confirm_buckets
            ),
            "forecast_advised_economy_hold_confirm_buckets": int(
                args.forecast_advised_economy_hold_confirm_buckets
            ),
            "forecast_advised_economy_min_axis_deg": float(
                args.forecast_advised_economy_min_axis_deg
            ),
            "forecast_advised_economy_min_pressure_norm": float(
                args.forecast_advised_economy_min_pressure_norm
            ),
            "forecast_advised_economy_posture_release_deg": float(
                args.forecast_advised_economy_posture_release_deg
            ),
            "forecast_advised_economy_release_cooldown_s": float(
                args.forecast_advised_economy_release_cooldown_s
            ),
            "forecast_advised_economy_latch_s": float(
                args.forecast_advised_economy_latch_s
            ),
            "forecast_advised_economy_current_ws_range_ms": float(
                args.forecast_advised_economy_current_ws_range_ms
            ),
            "forecast_advised_economy_current_dir_shift_deg": float(
                args.forecast_advised_economy_current_dir_shift_deg
            ),
            "forecast_advised_economy_current_history_s": float(
                args.forecast_advised_economy_current_history_s
            ),
            "forecast_advised_economy_mild_enabled": int(
                bool(args.forecast_advised_economy_mild)
            ),
            "forecast_advised_economy_mild_min_headroom_deg": float(
                args.forecast_advised_economy_mild_min_headroom_deg
            ),
            "forecast_advised_economy_mild_rise_norm": float(
                args.forecast_advised_economy_mild_rise_norm
            ),
            "forecast_advised_economy_mild_max_speed_range_ms": float(
                args.forecast_advised_economy_mild_max_speed_range_ms
            ),
            "forecast_advised_economy_mild_restart_err_kg": float(
                args.forecast_advised_economy_mild_restart_err_kg
            ),
            "forecast_advised_economy_high_stable_clean_required": int(
                bool(args.forecast_advised_economy_high_stable_clean)
            ),
            "forecast_advised_economy_high_stable_early_min_ms": float(
                args.forecast_advised_economy_high_stable_early_min_ms
            ),
            "forecast_advised_economy_high_stable_near_min_ms": float(
                args.forecast_advised_economy_high_stable_near_min_ms
            ),
            "forecast_advised_economy_high_stable_far_min_ms": float(
                args.forecast_advised_economy_high_stable_far_min_ms
            ),
            "forecast_advised_economy_high_stable_near_range_max_ms": float(
                args.forecast_advised_economy_high_stable_near_range_max_ms
            ),
            "forecast_advised_economy_high_stable_far_range_max_ms": float(
                args.forecast_advised_economy_high_stable_far_range_max_ms
            ),
            "forecast_advised_economy_high_stable_dir_shift_max_deg": float(
                args.forecast_advised_economy_high_stable_dir_shift_max_deg
            ),
            "forecast_advised_economy_min_peak_drop_ms": float(
                args.forecast_advised_economy_min_peak_drop_ms
            ),
            "gusty_oscillation_enabled": int(bool(args.gusty_oscillation)),
            "gusty_oscillation_min_target_age_s": float(
                args.gusty_oscillation_min_target_age_s
            ),
            "gusty_oscillation_target_scale": float(
                args.gusty_oscillation_target_scale
            ),
            "gusty_oscillation_headroom_scaled": int(
                bool(args.gusty_oscillation_headroom_scaled)
            ),
            "gusty_oscillation_low_posture_deg": float(
                args.gusty_oscillation_low_posture_deg
            ),
            "gusty_oscillation_high_posture_deg": float(
                args.gusty_oscillation_high_posture_deg
            ),
            "gusty_oscillation_low_posture_scale": float(
                args.gusty_oscillation_low_posture_scale
            ),
            "gusty_oscillation_high_posture_scale": float(
                args.gusty_oscillation_high_posture_scale
            ),
            "gusty_oscillation_short_gate_enabled": int(
                bool(args.gusty_oscillation_short_gate)
            ),
            "gusty_oscillation_short_min_range_ms": float(
                args.gusty_oscillation_short_min_range_ms
            ),
            "gusty_oscillation_short_min_pressure_norm": float(
                args.gusty_oscillation_short_min_pressure_norm
            ),
            "gusty_oscillation_short_max_dir_shift_deg": float(
                args.gusty_oscillation_short_max_dir_shift_deg
            ),
            "gusty_oscillation_short_min_turns": int(
                args.gusty_oscillation_short_min_turns
            ),
            "gusty_hold_current_gate_mode": str(
                args.gusty_hold_current_gate_mode
            ),
            "gusty_hold_current_isolated_primary_enabled": int(
                bool(args.gusty_hold_current_isolated_primary)
            ),
            "gusty_hold_current_runtime_release_enabled": int(
                bool(args.gusty_hold_current_runtime_release)
            ),
            "gusty_hold_current_runtime_release_deg": float(
                args.gusty_hold_current_runtime_release_deg
            ),
            "gusty_hold_current_runtime_release_cooldown_s": float(
                args.gusty_hold_current_runtime_release_cooldown_s
            ),
            "gusty_hold_current_debt_guard_enabled": int(
                bool(args.gusty_hold_current_debt_guard)
            ),
            "gusty_hold_current_debt_guard_target_motion_kg_s": float(
                args.gusty_hold_current_debt_guard_target_motion_kg_s
            ),
            "gusty_hold_current_debt_guard_target_err_kg": float(
                args.gusty_hold_current_debt_guard_target_err_kg
            ),
            "gusty_hold_current_debt_guard_min_age_s": float(
                args.gusty_hold_current_debt_guard_min_age_s
            ),
            "gusty_hold_current_day_budget_enabled": int(
                bool(args.gusty_hold_current_day_budget)
            ),
            "gusty_hold_current_day_budget_max_buckets": int(
                args.gusty_hold_current_day_budget_max_buckets
            ),
            "gusty_hold_current_day_budget_window_s": float(
                args.gusty_hold_current_day_budget_window_s
            ),
            "direction_reversal_hold_current_gate_mode": str(
                args.direction_reversal_hold_current_gate_mode
            ),
            "event_risk_pressure_boost_enabled": int(bool(args.event_risk_pressure_boost)),
            "event_risk_pressure_boost_gain": float(args.event_risk_pressure_boost_gain),
            "event_risk_pressure_boost_threshold": float(args.event_risk_pressure_boost_threshold),
            "event_risk_pressure_boost_max": float(args.event_risk_pressure_boost_max),
            "event_risk_pressure_floor_enabled": int(bool(args.event_risk_pressure_floor)),
            "event_risk_pressure_floor_threshold": float(args.event_risk_pressure_floor_threshold),
            "event_risk_pressure_floor_norm": float(args.event_risk_pressure_floor_norm),
            "event_risk_pressure_floor_max_lift": float(args.event_risk_pressure_floor_max_lift),
            "trusted_event_gate_enabled": int(bool(args.trusted_event_gate)),
            "trusted_event_highwind_threshold": float(args.trusted_event_highwind_threshold),
            "trusted_event_attention_threshold": float(args.trusted_event_attention_threshold),
            "trusted_event_attention_min_heads": int(args.trusted_event_attention_min_heads),
            "trusted_event_dynamic_enabled": int(bool(args.trusted_event_dynamic)),
            "trusted_event_dynamic_threshold": float(args.trusted_event_dynamic_threshold),
            "trusted_event_dynamic_min_heads": int(args.trusted_event_dynamic_min_heads),
            "forecast_pressure_trust_gate_enabled": int(
                bool(args.forecast_pressure_trust_gate)
            ),
            "forecast_pressure_trust_current_norm_max": float(
                args.forecast_pressure_trust_current_norm_max
            ),
            "forecast_pressure_trust_future_norm_min": float(
                args.forecast_pressure_trust_future_norm_min
            ),
            "forecast_pressure_trust_event_threshold": float(
                args.forecast_pressure_trust_event_threshold
            ),
            "forecast_pressure_trust_untrusted_scale": float(
                args.forecast_pressure_trust_untrusted_scale
            ),
            "forecast_pressure_trust_min_direction_dot": float(
                args.forecast_pressure_trust_min_direction_dot
            ),
            "forecast_pressure_trust_spike_margin_norm": float(
                args.forecast_pressure_trust_spike_margin_norm
            ),
            "preview_lead_action_enabled": int(bool(args.preview_lead_action)),
            "preview_lead_event_threshold": float(args.preview_lead_event_threshold),
            "preview_lead_current_norm_max": float(args.preview_lead_current_norm_max),
            "preview_lead_future_norm_min": float(args.preview_lead_future_norm_min),
            "preview_lead_current_wind_max_ms": float(args.preview_lead_current_wind_max_ms),
            "preview_lead_pitch_abs_max_deg": float(args.preview_lead_pitch_abs_max_deg),
            "preview_lead_roll_abs_max_deg": float(args.preview_lead_roll_abs_max_deg),
            "preview_lead_action_name": str(args.preview_lead_action_name),
            "pump_suppression_event_risk_guard_enabled": 0,
            "pump_suppression_event_risk_guard_threshold": 0.0,
            "relief_medium_cap_enabled": int(bool(args.relief_medium_cap)),
            "relief_medium_cap_event_threshold": float(args.relief_medium_cap_event_threshold),
            "relief_medium_cap_ratio": float(args.relief_medium_cap_ratio),
            "relief_medium_cap_adaptive": int(bool(args.relief_medium_cap_adaptive)),
            "far_horizon_enabled": int(bool(args.far_horizon)),
            "far_horizon_relief_gate_enabled": int(bool(args.far_horizon_relief_gate)),
            "far_horizon_weight_60_80": float(args.far_horizon_weight_60_80),
            "far_horizon_weight_80_100": float(args.far_horizon_weight_80_100),
            "far_horizon_weight_100_120": float(args.far_horizon_weight_100_120),
            "far_horizon_relief_pitch_abs_max_deg": float(
                args.far_horizon_relief_pitch_abs_max_deg
            ),
            "far_horizon_relief_roll_abs_max_deg": float(
                args.far_horizon_relief_roll_abs_max_deg
            ),
            "h120_oracle_probe_enabled": int(bool(args.h120_oracle_probe)),
            "h120_oracle_probe_active_ratio": float(
                sp["h120_oracle_probe_active_ratio"]
            ),
            "h120_oracle_probe_preemptive_count": float(
                sp["h120_oracle_probe_preemptive_count"]
            ),
            "h120_oracle_probe_short_delay_count": float(
                sp["h120_oracle_probe_short_delay_count"]
            ),
            "h120_oracle_probe_early_stop_veto_count": float(
                sp["h120_oracle_probe_early_stop_veto_count"]
            ),
            "h120_risk_scheduler_enabled": int(bool(args.h120_risk_scheduler)),
            "h120_scheduler_remote_risk_ratio": float(
                sp["h120_scheduler_remote_risk_ratio"]
            ),
            "h120_scheduler_risk_aware_rows": float(
                sp["h120_scheduler_risk_aware_rows"]
            ),
            "h120_scheduler_high_risk_rows": float(
                sp["h120_scheduler_high_risk_rows"]
            ),
            "h120_scheduler_early_stop_suppressed_count": float(
                sp["h120_scheduler_early_stop_suppressed_count"]
            ),
            "h120_scheduler_prefloor_active_count": float(
                sp["h120_scheduler_prefloor_active_count"]
            ),
            "h120_scheduler_delay_normal_s": float(args.h120_scheduler_delay_normal_s),
            "h120_scheduler_delay_risk_aware_s": float(
                args.h120_scheduler_delay_risk_aware_s
            ),
            "h120_scheduler_delay_high_risk_s": float(
                args.h120_scheduler_delay_high_risk_s
            ),
            "h120_scheduler_far_high_norm": float(args.h120_scheduler_far_high_norm),
            "h120_scheduler_intensify_margin_norm": float(
                args.h120_scheduler_intensify_margin_norm
            ),
            "h120_scheduler_relief_margin_norm": float(
                args.h120_scheduler_relief_margin_norm
            ),
            "h120_scheduler_early_stop_suppress_max_per_episode": int(
                args.h120_scheduler_early_stop_suppress_max_per_episode
            ),
            "h120_scheduler_prefloor_probe_enabled": int(
                bool(args.h120_scheduler_prefloor_probe)
            ),
            "h120_scheduler_prefloor_action": str(
                args.h120_scheduler_prefloor_action
            ),
            "h120_scheduler_prefloor_active_count": float(
                sp["h120_scheduler_prefloor_active_count"]
            ),
            "medium_escalation_enabled": 0,
            "no_unexplained_hold_action_enabled": 0,
            "no_unexplained_hold_threshold_deg": 0.0,
            "no_unexplained_hold_trigger_buckets": 0,
            "recovery_mode_enabled": 0,
            "preview_recovery_mode_ratio": sp["preview_recovery_mode_ratio"],
            "hold_comfort_release_enabled": 0,
            "active_posture_refresh_enabled": int(bool(args.active_posture_refresh)),
            "active_posture_refresh_enter_deg": float(
                args.active_posture_refresh_enter_deg
            ),
            "active_posture_refresh_exit_deg": float(
                args.active_posture_refresh_exit_deg
            ),
            "active_posture_refresh_update_interval_s": float(
                args.active_posture_refresh_update_interval_s
            ),
            "active_posture_refresh_action_name": str(
                args.active_posture_refresh_action_name
            ),
            "active_posture_refresh_pressure_max_norm": float(
                args.active_posture_refresh_pressure_max_norm
            ),
            "active_posture_refresh_min_target_age_s": float(
                args.active_posture_refresh_min_target_age_s
            ),
            "active_posture_refresh_axis_mode": str(
                args.active_posture_refresh_axis_mode
            ),
            "active_posture_refresh_worsening_eps_deg_s": float(
                args.active_posture_refresh_worsening_eps_deg_s
            ),
            "active_posture_refresh_debt_deg": float(
                args.active_posture_refresh_debt_deg
            ),
            "active_posture_refresh_include_hold": int(
                bool(args.active_posture_refresh_include_hold)
            ),
            "economy_pump_budget_hold_refresh_fraction": float(
                args.economy_pump_budget_hold_refresh_fraction
            ),
            "economy_pump_budget_dynamic_refresh_enabled": int(
                bool(args.economy_pump_budget_dynamic_refresh)
            ),
            "economy_pump_budget_dynamic_refresh_enter_deg": float(
                args.economy_pump_budget_dynamic_refresh_enter_deg
            ),
            "economy_pump_budget_dynamic_refresh_full_deg": float(
                args.economy_pump_budget_dynamic_refresh_full_deg
            ),
            "economy_pump_budget_dynamic_refresh_max_fraction": float(
                args.economy_pump_budget_dynamic_refresh_max_fraction
            ),
            "hold_comfort_no_preview_only": 0,
            "hold_comfort_enter_pitch_deg": 0.0,
            "hold_comfort_enter_roll_deg": 0.0,
            "hold_comfort_exit_pitch_deg": 0.0,
            "hold_comfort_exit_roll_deg": 0.0,
            "hold_comfort_rate_eps_deg_s": 0.0,
            "hold_comfort_action_name": "",
            "hold_comfort_pressure_max_norm": 0.0,
            "hold_comfort_current_pressure_max_norm": 0.0,
            "hold_comfort_forecast_veto_enabled": 0,
            "hold_comfort_veto_event_threshold": 0.0,
            "hold_comfort_veto_relief_margin_norm": 0.0,
            "hold_comfort_veto_posture_relief_norm": 0.0,
            "hold_comfort_veto_posture_specific_only": 0,
            "hold_comfort_veto_low_risk_norm": 0.0,
            "hold_comfort_veto_low_risk_posture_max_deg": 0.0,
            "hold_risk_micro_action_enabled": int(bool(args.hold_risk_micro_action)),
            "hold_risk_micro_event_threshold": float(
                args.hold_risk_micro_event_threshold
            ),
            "hold_risk_micro_pressure_norm_min": float(
                args.hold_risk_micro_pressure_norm_min
            ),
            "hold_risk_micro_pitch_abs_min_deg": float(
                args.hold_risk_micro_pitch_abs_min_deg
            ),
            "hold_risk_micro_roll_abs_min_deg": float(
                args.hold_risk_micro_roll_abs_min_deg
            ),
            "hold_risk_micro_pitch_abs_max_deg": float(
                args.hold_risk_micro_pitch_abs_max_deg
            ),
            "hold_risk_micro_roll_abs_max_deg": float(
                args.hold_risk_micro_roll_abs_max_deg
            ),
            "hold_risk_micro_action_name": str(args.hold_risk_micro_action_name),
            "primary_stall_refresh_enabled": int(bool(args.primary_stall_refresh)),
            "primary_stall_pitch_deg": float(args.primary_stall_pitch_deg),
            "primary_stall_roll_deg": float(args.primary_stall_roll_deg),
            "primary_stall_target_err_kg": float(args.primary_stall_target_err_kg),
            "primary_stall_pump_rate_m3_min": float(args.primary_stall_pump_rate_m3_min),
            "primary_stall_min_age_s": float(args.primary_stall_min_age_s),
            "primary_stall_pitch_axis_bias": int(bool(args.primary_stall_pitch_axis_bias)),
            "primary_stall_pitch_axis_roll_max_deg": float(
                args.primary_stall_pitch_axis_roll_max_deg
            ),
            "active_intent_reproposal_enabled": 0,
            "active_intent_reproposal_min_delta_kg": 0.0,
            "active_intent_reproposal_eligibility_enabled": 0,
            "active_intent_reproposal_eligibility_pitch_deg": 0.0,
            "active_intent_reproposal_eligibility_roll_deg": 0.0,
            "active_intent_reproposal_eligibility_target_err_kg": 0.0,
            "active_intent_reproposal_eligibility_pump_rate_m3_min": 0.0,
            "active_effectiveness_refresh_enabled": int(
                bool(args.active_effectiveness_refresh)
            ),
            "active_effectiveness_refresh_min_active_buckets": int(
                args.active_effectiveness_refresh_min_active_buckets
            ),
            "active_effectiveness_refresh_pitch_deg": float(
                args.active_effectiveness_refresh_pitch_deg
            ),
            "active_effectiveness_refresh_roll_deg": float(
                args.active_effectiveness_refresh_roll_deg
            ),
            "active_effectiveness_refresh_target_err_kg": float(
                args.active_effectiveness_refresh_target_err_kg
            ),
            "active_effectiveness_refresh_pump_rate_m3_min": float(
                args.active_effectiveness_refresh_pump_rate_m3_min
            ),
            "active_effectiveness_refresh_min_target_age_s": float(
                args.active_effectiveness_refresh_min_target_age_s
            ),
            "active_effectiveness_refresh_min_pressure_norm": float(
                args.active_effectiveness_refresh_min_pressure_norm
            ),
            "active_effectiveness_refresh_response_eps_deg": float(
                args.active_effectiveness_refresh_response_eps_deg
            ),
            "active_effectiveness_refresh_required_worsening_deg": float(
                args.active_effectiveness_refresh_required_worsening_deg
            ),
            "active_effectiveness_refresh_min_delta_kg": float(
                args.active_effectiveness_refresh_min_delta_kg
            ),
            "active_effectiveness_refresh_episode_budget_m3": float(
                args.active_effectiveness_refresh_episode_budget_m3
            ),
            "active_effectiveness_refresh_lowrisk_pressure_norm": float(
                args.active_effectiveness_refresh_lowrisk_pressure_norm
            ),
            "sustained_active_recompute_enabled": int(
                bool(args.sustained_active_recompute)
            ),
            "sustained_active_recompute_min_active_buckets": int(
                args.sustained_active_recompute_min_active_buckets
            ),
            "sustained_active_recompute_pitch_deg": float(
                args.sustained_active_recompute_pitch_deg
            ),
            "sustained_active_recompute_roll_deg": float(
                args.sustained_active_recompute_roll_deg
            ),
            "sustained_active_recompute_target_err_kg": float(
                args.sustained_active_recompute_target_err_kg
            ),
            "sustained_active_recompute_pump_rate_m3_min": float(
                args.sustained_active_recompute_pump_rate_m3_min
            ),
            "sustained_active_recompute_min_target_age_s": float(
                args.sustained_active_recompute_min_target_age_s
            ),
            "stale_active_target_refresh_mode": str(
                args.stale_active_target_refresh
            ),
            "stale_active_target_refresh_min_active_buckets": int(
                args.stale_active_target_refresh_min_active_buckets
            ),
            "stale_active_target_refresh_pitch_deg": float(
                args.stale_active_target_refresh_pitch_deg
            ),
            "stale_active_target_refresh_roll_deg": float(
                args.stale_active_target_refresh_roll_deg
            ),
            "stale_active_target_refresh_target_err_kg": float(
                args.stale_active_target_refresh_target_err_kg
            ),
            "stale_active_target_refresh_pump_rate_m3_min": float(
                args.stale_active_target_refresh_pump_rate_m3_min
            ),
            "stale_active_target_refresh_min_target_age_s": float(
                args.stale_active_target_refresh_min_target_age_s
            ),
            "stale_active_target_refresh_min_delta_kg": float(
                args.stale_active_target_refresh_min_delta_kg
            ),
            "stale_active_target_refresh_cooldown_s": float(
                args.stale_active_target_refresh_cooldown_s
            ),
            "stale_active_target_refresh_no_recovery_eps_deg": float(
                args.stale_active_target_refresh_no_recovery_eps_deg
            ),
            "reactive_floor_predictive_veto": str(
                args.reactive_floor_predictive_veto
            ),
            "high_posture_metric": str(args.high_posture_metric),
            "high_posture_enter": float(args.high_posture_enter),
            "high_posture_exit": float(args.high_posture_exit),
            "high_posture_min_duration_s": float(args.high_posture_min_duration_s),
            "reactive_floor_theta_only_enter": float(
                args.reactive_floor_theta_only_enter
            ),
            "reactive_floor_theta_only_min_duration_s": float(
                args.reactive_floor_theta_only_min_duration_s
            ),
            "reactive_floor_action": str(args.reactive_floor_action),
            "reactive_floor_post_exit_mode": str(
                args.reactive_floor_post_exit_mode
            ),
            "reactive_floor_post_exit_target_cap_kg": float(
                args.reactive_floor_post_exit_target_cap_kg
            ),
            "reactive_floor_post_exit_margin_deg": float(
                args.reactive_floor_post_exit_margin_deg
            ),
            "reactive_floor_post_exit_worsening_eps_deg": float(
                args.reactive_floor_post_exit_worsening_eps_deg
            ),
            "oracle_preemptive_prevent_trigger": int(
                bool(args.oracle_preemptive_prevent_trigger)
            ),
            "oracle_preemptive_enter_deg": float(args.oracle_preemptive_enter_deg),
            "oracle_preemptive_floor_deg": float(args.oracle_preemptive_floor_deg),
            "oracle_preemptive_min_future_norm": float(
                args.oracle_preemptive_min_future_norm
            ),
            "oracle_preemptive_rise_margin_norm": float(
                args.oracle_preemptive_rise_margin_norm
            ),
            "oracle_preemptive_min_delta_kg": float(
                args.oracle_preemptive_min_delta_kg
            ),
            "oracle_preemptive_cooldown_s": float(
                args.oracle_preemptive_cooldown_s
            ),
            "h120_oracle_probe_enabled": int(bool(args.h120_oracle_probe)),
            "h120_oracle_probe_short_delay_s": float(
                args.h120_oracle_probe_short_delay_s
            ),
            "h120_oracle_probe_far_high_norm": float(
                args.h120_oracle_probe_far_high_norm
            ),
            "h120_oracle_probe_intensify_margin_norm": float(
                args.h120_oracle_probe_intensify_margin_norm
            ),
            "h120_oracle_probe_preemptive_enter_deg": float(
                args.h120_oracle_probe_preemptive_enter_deg
            ),
            "h120_oracle_probe_preemptive_floor_deg": float(
                args.h120_oracle_probe_preemptive_floor_deg
            ),
            "h120_oracle_probe_min_delta_kg": float(
                args.h120_oracle_probe_min_delta_kg
            ),
            "h120_oracle_probe_cooldown_s": float(
                args.h120_oracle_probe_cooldown_s
            ),
            "h120_risk_scheduler_enabled": int(bool(args.h120_risk_scheduler)),
            "h120_scheduler_delay_normal_s": float(args.h120_scheduler_delay_normal_s),
            "h120_scheduler_delay_risk_aware_s": float(
                args.h120_scheduler_delay_risk_aware_s
            ),
            "h120_scheduler_delay_high_risk_s": float(
                args.h120_scheduler_delay_high_risk_s
            ),
            "h120_scheduler_far_high_norm": float(args.h120_scheduler_far_high_norm),
            "h120_scheduler_intensify_margin_norm": float(
                args.h120_scheduler_intensify_margin_norm
            ),
            "h120_scheduler_relief_margin_norm": float(
                args.h120_scheduler_relief_margin_norm
            ),
            "h120_scheduler_early_stop_suppress_max_per_episode": int(
                args.h120_scheduler_early_stop_suppress_max_per_episode
            ),
            "h120_scheduler_prefloor_probe_enabled": int(
                bool(args.h120_scheduler_prefloor_probe)
            ),
            "h120_scheduler_prefloor_action": str(
                args.h120_scheduler_prefloor_action
            ),
            "h120_scheduler_prefloor_enter_deg": float(
                args.h120_scheduler_prefloor_enter_deg
            ),
            "h120_scheduler_prefloor_floor_deg": float(
                args.h120_scheduler_prefloor_floor_deg
            ),
            "h120_scheduler_prefloor_min_delta_kg": float(
                args.h120_scheduler_prefloor_min_delta_kg
            ),
            "h120_scheduler_prefloor_max_delta_kg": float(
                args.h120_scheduler_prefloor_max_delta_kg
            ),
            "h120_scheduler_prefloor_cooldown_s": float(
                args.h120_scheduler_prefloor_cooldown_s
            ),
        }
        add_prefixed_attitude_exposure(row, sc, sp, has_closed=has_closed)
        rows.append(row)
        completed += 1
        saved_label = f"saved {fig_path.name}" if fig_path is not None else "figures skipped"
        if has_closed:
            print(
                f"   {saved_label}: pump {row['d_pump_work_pct']:+.1f}%, "
                f"d_pitch_p95={row['d_pitch_p95']:+.3f}, d_roll_p95={row['d_roll_p95']:+.3f}",
                flush=True,
            )
        else:
            print(
                f"   {saved_label}: primary pump {row['primary_pump_work_m3']:.2f} m3, "
                f"pitch_p95={row['primary_pitch_p95']:.3f}, roll_p95={row['primary_roll_p95']:.3f}",
                flush=True,
            )

    elapsed_s = time.perf_counter() - t0
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "casebook_summary.csv", index=False)
    protocol = build_casebook_run_protocol(
        args=args,
        cfg=cfg,
        repo_root=repo_root,
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        closed_pump_profile=closed_profile,
        primary_pump_profile=primary_pump_profile,
        primary_scale=primary_scale,
        forecast_source_effective=forecast_source_label,
        completed_cases=len(summary),
        issue_count=len(issues),
        elapsed_s=elapsed_s,
        summary_rows=summary,
    )
    protocol_path = write_run_protocol(out_dir / "run_protocol.json", protocol)
    run_identity = build_casebook_run_identity(
        protocol=protocol,
        args=args,
        planner_config=cfg,
        repo_root=repo_root,
        out_dir=out_dir,
        dataset_dir=dataset_dir,
        forecast_source_effective=forecast_source_label,
        created_at_utc=protocol.created_at_utc,
    )
    identity_path = write_run_identity(out_dir / "run_identity.json", run_identity)
    lines = [
        "# Prediction-primary casebook",
        "",
        f"- elapsed: `{elapsed_s:.1f}s`",
        f"- completed cases: `{len(summary)}`",
        f"- run protocol: `{protocol_path.relative_to(repo_root) if protocol_path.is_relative_to(repo_root) else protocol_path}`",
        f"- run identity: `{identity_path.relative_to(repo_root) if identity_path.is_relative_to(repo_root) else identity_path}`",
        f"- stable run id: `{run_identity.run_id}`",
        f"- variants: `{'primary_only' if bool(args.primary_only) else 'closed_only vs ' + primary_variant}`",
        f"- closed-only pump profile: `{closed_profile}`",
        f"- prediction-primary scale: `{primary_scale:.3f}`",
        f"- prediction-primary control profile: `{args.primary_control_profile}`",
        f"- prediction-primary safety profile: `{args.primary_safety_profile}`",
        f"- forecast source requested: `{args.forecast_source}`",
        f"- forecast source effective: `{forecast_source_label}`",
        f"- event reset mode: `{args.event_reset_mode}`",
        f"- planner envelope mode: `{'discounted' if cfg.envelope_use_discount else 'raw'}`",
        f"- planner envelope barrier: `{int(bool(cfg.envelope_barrier_active))}`",
        f"- planner envelope barrier const: `{float(cfg.envelope_barrier_const):.3f}`",
        "- planner posture hold barrier: `0`",
        "- planner posture hold band: `0.00/0.00 deg`",
        "- planner posture hold barrier const: `0.000`",
        f"- planner pressure norm cap: `{float(cfg.pressure_norm_cap):.3f}`",
        f"- target reset tol: `{float(args.target_reset_tol_deg):.3f} deg`",
        f"- forecast-advised suppression-only: `{int(bool(args.forecast_advised_economy_suppression_only))}`",
        f"- event-risk pressure boost: `{int(bool(args.event_risk_pressure_boost))}`",
        f"- event-risk pressure floor: `{int(bool(args.event_risk_pressure_floor))}`",
        f"- trusted event gate: `{int(bool(args.trusted_event_gate))}` "
        f"(hi>={float(args.trusted_event_highwind_threshold):.2f}, "
        f"att>={float(args.trusted_event_attention_threshold):.2f}, "
        f"k={int(args.trusted_event_attention_min_heads)})",
        f"- forecast pressure trust gate: `{int(bool(args.forecast_pressure_trust_gate))}` "
        f"(future>={float(args.forecast_pressure_trust_future_norm_min):.2f}, "
        f"scale={float(args.forecast_pressure_trust_untrusted_scale):.2f})",
        f"- preview lead action: `{int(bool(args.preview_lead_action))}`",
        "- suppression event-risk guard: `0`",
        f"- relief medium cap: `{int(bool(args.relief_medium_cap))}`",
        f"- relief medium cap ratio: `{float(args.relief_medium_cap_ratio):.3f}`",
        "- medium escalation: `0`",
        f"- active effectiveness refresh: `{int(bool(args.active_effectiveness_refresh))}`",
        f"- hold relief debt: `{int(bool(args.hold_relief_debt))}`",
        "- no-unexplained-hold action: `0`",
        f"- recovery mode: `{int(bool(args.recovery_mode))}`",
        "- hold comfort release: `0`",
        "- hold comfort no-preview only: `0`",
        "- hold comfort enter/exit: `0.0/0.0 -> 0.0/0.0 deg`",
        "- hold comfort forecast veto: `0`",
        f"- hold-risk micro action: `{int(bool(args.hold_risk_micro_action))}`",
        f"- replay dataset: `{dataset_dir.relative_to(repo_root) if dataset_dir.is_relative_to(repo_root) else dataset_dir}`",
        f"- duration per case: `{duration_min:.0f} min`",
        f"- replay wind transition: `{float(args.wind_transition_s):.1f}s raised-cosine`",
        f"- figures skipped: `{int(bool(args.skip_figures))}`",
        f"- primary-only screening: `{int(bool(args.primary_only))}`",
        "",
        "## Attitude metric semantics",
        "",
        "- exposure axis: `max(abs(pitch_deg), abs(roll_deg))`",
        "- startup window: `0 <= t < 300s`; steady window: `t >= 300s`",
        "- full CSV columns include `time`, `max_continuous`, and `area` for thresholds `3/4/5/7.5/10 deg`, with `primary_`, `closed_`, and `d_` prefixes.",
        "- thresholds are layered reporting bands, not single-point pass/fail limits; `t>5` is service-pressure accounting, not a failure by itself.",
        "",
        "| threshold | label | meaning | control use |",
        "|---:|---|---|---|",
        *attitude_metric_semantics_markdown_rows(),
        "",
        "| case | label | primary pump | pump delta | d_pitch_p95 | d_roll_p95 | primary t>5 | d t>5 | primary t>7.5 | d t>7.5 | primary run>7.5 | primary t>10 | d t>10 | primary run>10 | primary steady area>5 | d steady area>5 | suppression | pressure trust block | pressure scale | risk boost | risk floor | lead | relief cap | med esc | active eff | no-hold | recovery | posture barrier | comfort release | comfort veto | hold micro | safety fallback | latch switches | figure |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        pump_txt = "n/a" if pd.isna(r["d_pump_work_pct"]) else f"{r['d_pump_work_pct']:+.1f}%"
        d_pitch_txt = "n/a" if pd.isna(r["d_pitch_p95"]) else f"{r['d_pitch_p95']:+.3f}"
        d_roll_txt = "n/a" if pd.isna(r["d_roll_p95"]) else f"{r['d_roll_p95']:+.3f}"
        d_time5_txt = (
            "n/a"
            if pd.isna(r["d_time_over_5deg_s"])
            else f"{r['d_time_over_5deg_s']:+.0f}s"
        )
        d_time75_txt = (
            "n/a"
            if pd.isna(r["d_time_over_7p5deg_s"])
            else f"{r['d_time_over_7p5deg_s']:+.0f}s"
        )
        d_time10_txt = (
            "n/a"
            if pd.isna(r["d_time_over_10deg_s"])
            else f"{r['d_time_over_10deg_s']:+.0f}s"
        )
        d_steady_area5_txt = (
            "n/a"
            if pd.isna(r["d_steady_after_300s_area_over_5deg_deg_s"])
            else f"{r['d_steady_after_300s_area_over_5deg_deg_s']:+.1f}"
        )
        latch_txt = (
            "n/a"
            if pd.isna(r["closed_latch_switches"])
            else f"{int(r['closed_latch_switches'])}->{int(r['primary_latch_switches'])}"
        )
        lines.append(
            f"| {r['case_id']} | {r['label']} | {r['primary_pump_work_m3']:.2f} | {pump_txt} | "
            f"{d_pitch_txt} | {d_roll_txt} | "
            f"{r['primary_time_over_5deg_s']:.0f}s | "
            f"{d_time5_txt} | "
            f"{r['primary_time_over_7p5deg_s']:.0f}s | "
            f"{d_time75_txt} | "
            f"{r['primary_max_continuous_over_7p5deg_s']:.0f}s | "
            f"{r['primary_time_over_10deg_s']:.0f}s | "
            f"{d_time10_txt} | "
            f"{r['primary_max_continuous_over_10deg_s']:.0f}s | "
            f"{r['primary_steady_after_300s_area_over_5deg_deg_s']:.1f} | "
            f"{d_steady_area5_txt} | "
            f"{r['preview_pump_suppression_ratio'] * 100:.1f}% | "
            f"{r['preview_forecast_pressure_trust_block_ratio'] * 100:.1f}% | "
            f"{r['preview_forecast_pressure_trust_scale_mean']:.2f}x | "
            f"{r['preview_event_risk_scale_mean']:.2f}x | "
            f"{r['preview_event_risk_floor_ratio'] * 100:.1f}% | "
            f"{r['preview_lead_action_ratio'] * 100:.1f}% | "
            f"{r['preview_relief_medium_cap_ratio'] * 100:.1f}% | "
            f"{r['preview_medium_escalation_ratio'] * 100:.1f}% | "
            f"{r['preview_active_effectiveness_refresh_ratio'] * 100:.1f}% | "
            f"{r['preview_no_unexplained_hold_action_ratio'] * 100:.1f}% | "
            f"{r['preview_recovery_mode_ratio'] * 100:.1f}% | "
            f"{r['preview_posture_hold_barrier_ratio'] * 100:.1f}% | "
            f"{r['preview_hold_comfort_release_ratio'] * 100:.1f}% | "
            f"{r['preview_hold_comfort_veto_ratio'] * 100:.1f}% | "
            f"{r['preview_hold_risk_micro_action_ratio'] * 100:.1f}% | "
            f"{r['primary_safety_fallback_ratio'] * 100:.1f}% | "
            f"{latch_txt} | "
            f"`{r['figure']}` |"
        )
    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]
    (out_dir / "casebook_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {out_dir / 'casebook_report.md'}")


if __name__ == "__main__":
    main()
