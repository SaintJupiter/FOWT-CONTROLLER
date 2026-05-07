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
from datetime import datetime
from functools import cmp_to_key
from typing import Any

import numpy as np

from .ballast_planner import (
    PlannerConfig,
    _ALL_SEQUENCES,
    action_vec,
    compare_sequences,
    compute_pressure_blocks,
    economic_scalar_cost,
    evaluate_sequence,
    norm_term,
    select_best_economic,
    tank_signal,
)
from .forecast_adapter import ForecastModelAdapter
from .replay_dataset import Fino1ReplayDataset

_EVENT_RISK_KEYS = (
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)


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
        pump_suppression_enabled: bool = False,
        pump_suppression_restart_err_kg: float = 2000.0,
        pump_suppression_relief_margin_norm: float = 0.25,
        pump_suppression_low_risk_norm: float = 0.75,
        event_risk_pressure_boost_enabled: bool = False,
        event_risk_pressure_boost_gain: float = 0.35,
        event_risk_pressure_boost_threshold: float = 0.50,
        event_risk_pressure_boost_max: float = 0.35,
        event_risk_pressure_floor_enabled: bool = False,
        event_risk_pressure_floor_threshold: float = 0.70,
        event_risk_pressure_floor_norm: float = 0.85,
        event_risk_pressure_floor_max_lift: float = 0.50,
        preview_lead_action_enabled: bool = False,
        preview_lead_event_threshold: float = 0.90,
        preview_lead_current_norm_max: float = 0.65,
        preview_lead_future_norm_min: float = 1.00,
        preview_lead_current_wind_max_ms: float = 8.0,
        preview_lead_pitch_abs_max_deg: float = 2.0,
        preview_lead_roll_abs_max_deg: float = 1.5,
        preview_lead_min_direction_norm: float = 0.05,
        preview_lead_action_name: str = "active_small",
        pump_suppression_event_risk_guard_enabled: bool = False,
        pump_suppression_event_risk_guard_threshold: float = 0.70,
        relief_medium_cap_enabled: bool = False,
        relief_medium_cap_event_threshold: float = 0.70,
        relief_medium_cap_ratio: float = 0.25,
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
        if primary_hold_target_mode not in ("current", "pause"):
            raise ValueError(
                f"unsupported primary_hold_target_mode={primary_hold_target_mode!r}"
            )
        self.primary_hold_target_mode = str(primary_hold_target_mode)
        self.pump_suppression_enabled = bool(pump_suppression_enabled)
        self.pump_suppression_restart_err_kg = float(pump_suppression_restart_err_kg)
        self.pump_suppression_relief_margin_norm = float(pump_suppression_relief_margin_norm)
        self.pump_suppression_low_risk_norm = float(pump_suppression_low_risk_norm)
        self.event_risk_pressure_boost_enabled = bool(event_risk_pressure_boost_enabled)
        self.event_risk_pressure_boost_gain = float(event_risk_pressure_boost_gain)
        self.event_risk_pressure_boost_threshold = float(event_risk_pressure_boost_threshold)
        self.event_risk_pressure_boost_max = float(event_risk_pressure_boost_max)
        self.event_risk_pressure_floor_enabled = bool(event_risk_pressure_floor_enabled)
        self.event_risk_pressure_floor_threshold = float(event_risk_pressure_floor_threshold)
        self.event_risk_pressure_floor_norm = float(event_risk_pressure_floor_norm)
        self.event_risk_pressure_floor_max_lift = float(event_risk_pressure_floor_max_lift)
        if preview_lead_action_name not in ("pump_saving", "active_small"):
            raise ValueError(f"unsupported preview_lead_action_name={preview_lead_action_name!r}")
        self.preview_lead_action_enabled = bool(preview_lead_action_enabled)
        self.preview_lead_event_threshold = float(preview_lead_event_threshold)
        self.preview_lead_current_norm_max = float(preview_lead_current_norm_max)
        self.preview_lead_future_norm_min = float(preview_lead_future_norm_min)
        self.preview_lead_current_wind_max_ms = float(preview_lead_current_wind_max_ms)
        self.preview_lead_pitch_abs_max_deg = float(preview_lead_pitch_abs_max_deg)
        self.preview_lead_roll_abs_max_deg = float(preview_lead_roll_abs_max_deg)
        self.preview_lead_min_direction_norm = float(preview_lead_min_direction_norm)
        self.preview_lead_action_name = str(preview_lead_action_name)
        self.pump_suppression_event_risk_guard_enabled = bool(
            pump_suppression_event_risk_guard_enabled
        )
        self.pump_suppression_event_risk_guard_threshold = float(
            pump_suppression_event_risk_guard_threshold
        )
        self.relief_medium_cap_enabled = bool(relief_medium_cap_enabled)
        self.relief_medium_cap_event_threshold = float(relief_medium_cap_event_threshold)
        self.relief_medium_cap_ratio = float(relief_medium_cap_ratio)
        self._default_plant_info = plant_info or {
            "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
            "pump_fullspeed_any": 0,
            "pump_total_backlog_kg": 0.0,
            "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        }
        # Cached planner-frame target (for FF) and zeroed mass FF state.
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        self._primary_anchor_masses_kg = np.asarray(
            self._default_plant_info["tank_masses"], dtype=float
        ).copy()
        self._primary_delta_kg = np.zeros(3, dtype=float)
        self._primary_target_kg = self._primary_anchor_masses_kg.copy()
        self._primary_target_initialized = False
        self._primary_target_refreshed = False
        self._primary_target_reused = False
        self._primary_target_resumed = False
        self._primary_target_last_reset_s = 0.0
        self._paused_primary_target_kg = self._primary_anchor_masses_kg.copy()
        self._paused_primary_delta_kg = np.zeros(3, dtype=float)
        self._paused_primary_avec = np.zeros(2, dtype=float)
        self._paused_primary_action = "hold"
        self._paused_primary_valid = False
        self._pump_suppression_active = False
        self._pump_suppression_reason = "disabled"
        self._pump_suppression_restart_err_kg_eff = 0.0
        self._pump_suppression_block_norms = [0.0, 0.0, 0.0]
        self._pump_suppression_block02_dot = 0.0
        self._event_risk_probs = [0.0, 0.0, 0.0]
        self._event_risk_block_scales = [1.0, 1.0, 1.0]
        self._event_risk_floor_active = [0, 0, 0]
        self._event_risk_floor_norms = [0.0, 0.0, 0.0]
        self._raw_pressure_block_norms = [0.0, 0.0, 0.0]
        self._preview_lead_active = False
        self._preview_lead_reason = "disabled"
        self._preview_lead_block_index = -1
        self._preview_lead_action_name = ""
        self._relief_medium_cap_active = False
        self._relief_medium_cap_reason = "disabled"
        self._relief_medium_cap_relief_margin = 0.0
        self._last_bucket = -1
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_action = "hold"
        self._transition_time_s = 0.0
        self.records: list[dict[str, Any]] = []  # per-bucket planner log

    @staticmethod
    def _zero_bias(source: str) -> dict[str, Any]:
        return {
            "pitch_bias_deg": 0.0,
            "roll_bias_deg": 0.0,
            "preview_pitch_bias_deg": 0.0,
            "preview_roll_bias_deg": 0.0,
            "source": str(source),
            "preview_trim_source": str(source),
            "preview_gain": 0.0,
            "risk_level": "passive",
            "preview_mode": "ballast_planner_a14",
            "first_action": "hold",
            "preview_pump_suppression_active": 0,
            "preview_pump_restart_err_kg": 0.0,
            "preview_pump_suppression_reason": str(source),
            "preview_primary_enabled": 0,
            "preview_primary_active": 0,
            "preview_primary_target_kg": [0.0, 0.0, 0.0],
            "preview_primary_delta_kg": [0.0, 0.0, 0.0],
            "preview_primary_action": "hold",
            "preview_primary_hold_target_mode": "current",
            "preview_primary_target_resumed": 0,
            "preview_pressure_block0_norm": 0.0,
            "preview_pressure_block1_norm": 0.0,
            "preview_pressure_block2_norm": 0.0,
            "preview_pressure_block02_dot": 0.0,
            "preview_event_risk_pressure_boost_enabled": 0,
            "preview_event_risk_prob_0_20m": 0.0,
            "preview_event_risk_prob_20_40m": 0.0,
            "preview_event_risk_prob_40_60m": 0.0,
            "preview_event_risk_scale_0_20m": 1.0,
            "preview_event_risk_scale_20_40m": 1.0,
            "preview_event_risk_scale_40_60m": 1.0,
            "preview_event_risk_pressure_floor_enabled": 0,
            "preview_event_risk_floor_active_0_20m": 0,
            "preview_event_risk_floor_active_20_40m": 0,
            "preview_event_risk_floor_active_40_60m": 0,
            "preview_event_risk_floor_norm_0_20m": 0.0,
            "preview_event_risk_floor_norm_20_40m": 0.0,
            "preview_event_risk_floor_norm_40_60m": 0.0,
            "preview_lead_action_enabled": 0,
            "preview_lead_action_active": 0,
            "preview_lead_action_reason": str(source),
            "preview_lead_action_block_index": -1,
            "preview_lead_action_name": "",
            "preview_pump_suppression_event_risk_guard_enabled": 0,
            "preview_pump_suppression_event_risk_guard_threshold": 0.0,
            "preview_relief_medium_cap_enabled": 0,
            "preview_relief_medium_cap_active": 0,
            "preview_relief_medium_cap_reason": str(source),
            "preview_relief_medium_cap_margin": 0.0,
        }

    def reset(self) -> None:
        self._last_bucket = -1
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        self._primary_anchor_masses_kg = np.asarray(
            self._default_plant_info["tank_masses"], dtype=float
        ).copy()
        self._primary_delta_kg = np.zeros(3, dtype=float)
        self._primary_target_kg = self._primary_anchor_masses_kg.copy()
        self._primary_target_initialized = False
        self._primary_target_refreshed = False
        self._primary_target_reused = False
        self._primary_target_resumed = False
        self._primary_target_last_reset_s = 0.0
        self._paused_primary_target_kg = self._primary_anchor_masses_kg.copy()
        self._paused_primary_delta_kg = np.zeros(3, dtype=float)
        self._paused_primary_avec = np.zeros(2, dtype=float)
        self._paused_primary_action = "hold"
        self._paused_primary_valid = False
        self._pump_suppression_active = False
        self._pump_suppression_reason = "disabled"
        self._pump_suppression_restart_err_kg_eff = 0.0
        self._pump_suppression_block_norms = [0.0, 0.0, 0.0]
        self._pump_suppression_block02_dot = 0.0
        self._event_risk_probs = [0.0, 0.0, 0.0]
        self._event_risk_block_scales = [1.0, 1.0, 1.0]
        self._event_risk_floor_active = [0, 0, 0]
        self._event_risk_floor_norms = [0.0, 0.0, 0.0]
        self._raw_pressure_block_norms = [0.0, 0.0, 0.0]
        self._preview_lead_active = False
        self._preview_lead_reason = "disabled"
        self._preview_lead_block_index = -1
        self._preview_lead_action_name = ""
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
        return info

    def _forecast_uv(self, sample) -> tuple[np.ndarray, str, dict[str, float]]:
        if self.forecast_adapter is None:
            event_probs = {
                name: float(sample.y_event[idx])
                for idx, name in enumerate(self.replay_dataset.event_columns)
            }
            return np.asarray(sample.y_uv_raw, dtype=float), "oracle_future", event_probs
        forecast = self.forecast_adapter.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        return (
            np.asarray(forecast.wind_uv_raw, dtype=float),
            str(forecast.model_version),
            dict(forecast.event_probs),
        )

    def _event_risk_scales(self, event_probs: dict[str, float]) -> tuple[list[float], list[float]]:
        probs = [float(event_probs.get(key, 0.0)) for key in _EVENT_RISK_KEYS]
        if not self.event_risk_pressure_boost_enabled:
            return probs, [1.0, 1.0, 1.0]
        scales: list[float] = []
        threshold = float(self.event_risk_pressure_boost_threshold)
        gain = float(self.event_risk_pressure_boost_gain)
        cap = max(0.0, float(self.event_risk_pressure_boost_max))
        for prob in probs:
            excess = max(0.0, min(1.0, prob) - threshold)
            boost = min(cap, gain * excess / max(1.0 - threshold, 1e-6))
            scales.append(1.0 + boost)
        return probs, scales

    def _apply_event_risk_scales(self, blocks: list[dict[str, Any]], scales: list[float]) -> list[dict[str, Any]]:
        if not self.event_risk_pressure_boost_enabled:
            return blocks
        scaled: list[dict[str, Any]] = []
        for idx, block in enumerate(blocks):
            factor = float(scales[idx]) if idx < len(scales) else 1.0
            item = dict(block)
            item["pressure_vec"] = np.asarray(block["pressure_vec"], dtype=float) * factor
            item["pressure_vec_raw"] = np.asarray(block.get("pressure_vec_raw", block["pressure_vec"]), dtype=float) * factor
            item["pressure_norm"] = norm_term(np.asarray(item["pressure_vec"], dtype=float), self.cfg)
            item["event_risk_pressure_scale"] = factor
            scaled.append(item)
        return scaled

    def _apply_event_risk_pressure_floor(
        self,
        blocks: list[dict[str, Any]],
        event_probs: list[float],
    ) -> list[dict[str, Any]]:
        self._event_risk_floor_active = [0, 0, 0]
        self._event_risk_floor_norms = [0.0, 0.0, 0.0]
        if not self.event_risk_pressure_floor_enabled:
            return blocks

        floored: list[dict[str, Any]] = []
        threshold = float(self.event_risk_pressure_floor_threshold)
        floor_norm = max(0.0, float(self.event_risk_pressure_floor_norm))
        max_lift = max(0.0, float(self.event_risk_pressure_floor_max_lift))
        for idx, block in enumerate(blocks):
            item = dict(block)
            prob = float(event_probs[idx]) if idx < len(event_probs) else 0.0
            current_norm = float(item.get("pressure_norm", 0.0))
            desired_norm = current_norm
            if prob >= threshold and floor_norm > current_norm:
                desired_norm = min(floor_norm, current_norm + max_lift)

            if desired_norm > current_norm + 1e-9:
                pressure_vec = np.asarray(item["pressure_vec"], dtype=float)
                actual_norm = norm_term(pressure_vec, self.cfg)
                # The event head only says "future wind change is important";
                # it does not encode direction. If UV gives no direction, do not
                # invent one here.
                if actual_norm > 1e-9:
                    factor = desired_norm / actual_norm
                    item["pressure_vec"] = pressure_vec * factor
                    item["pressure_vec_raw"] = (
                        np.asarray(item.get("pressure_vec_raw", pressure_vec), dtype=float) * factor
                    )
                    item["pressure_norm"] = norm_term(
                        np.asarray(item["pressure_vec"], dtype=float), self.cfg
                    )
                    item["event_risk_pressure_floor_active"] = 1
                    item["event_risk_pressure_floor_factor"] = factor
                    self._event_risk_floor_active[idx] = 1
                    self._event_risk_floor_norms[idx] = float(item["pressure_norm"])
                else:
                    item["event_risk_pressure_floor_active"] = 0
                    item["event_risk_pressure_floor_factor"] = 1.0
                    self._event_risk_floor_norms[idx] = current_norm
            else:
                item["event_risk_pressure_floor_active"] = 0
                item["event_risk_pressure_floor_factor"] = 1.0
                self._event_risk_floor_norms[idx] = current_norm
            floored.append(item)
        return floored

    def _run_planner(self, sample, plant_info: dict) -> tuple[dict[str, Any], np.ndarray, str, list[dict[str, Any]]]:
        uv, forecast_source, event_probs = self._forecast_uv(sample)
        raw_blocks = compute_pressure_blocks(uv, self.block_discounts, self.cfg)
        self._raw_pressure_block_norms = [
            float(b.get("pressure_norm", 0.0)) for b in raw_blocks[:3]
        ]
        while len(self._raw_pressure_block_norms) < 3:
            self._raw_pressure_block_norms.append(0.0)
        event_probs_vec, event_scales = self._event_risk_scales(event_probs)
        self._event_risk_probs = event_probs_vec
        self._event_risk_block_scales = event_scales
        blocks = self._apply_event_risk_scales(raw_blocks, event_scales)
        blocks = self._apply_event_risk_pressure_floor(blocks, event_probs_vec)
        evaluated = [evaluate_sequence(seq, blocks, plant_info, self.cfg) for seq in _ALL_SEQUENCES]
        if self.objective_mode == "economic":
            feasible, scalars = select_best_economic(evaluated, self.cfg)
            best_scalar = float(scalars[0]) if scalars else float("nan")
        else:
            feasible = sorted(
                [r for r in evaluated if not r["hard_reject_reason"]],
                key=cmp_to_key(lambda a, b: compare_sequences(a, b, self.cfg)[0]),
            )
            best_scalar = float("nan")
        if not feasible:
            return {"sequence": ("hold", "hold", "hold"), "first_action": "hold",
                    "best_scalar_cost": best_scalar}, np.zeros(2), forecast_source, blocks
        best = dict(feasible[0])
        best["best_scalar_cost"] = best_scalar
        first_action = best["sequence"][0]
        first_avec = action_vec(first_action, blocks[0]["pressure_vec"], self.cfg)
        return best, first_avec, forecast_source, blocks

    def _preview_lead_action(
        self,
        best: dict[str, Any],
        first_avec: np.ndarray,
        blocks: list[dict[str, Any]],
        state,
    ) -> tuple[str, np.ndarray]:
        self._preview_lead_active = False
        self._preview_lead_reason = "disabled"
        self._preview_lead_block_index = -1
        self._preview_lead_action_name = ""
        if not self.preview_lead_action_enabled:
            return str(best.get("sequence", ("hold",))[0]), first_avec

        sequence = tuple(best.get("sequence", ("hold", "hold", "hold")))
        current_action = str(sequence[0]) if sequence else "hold"
        if current_action != "hold":
            self._preview_lead_reason = "current_action_not_hold"
            return current_action, first_avec
        if len(blocks) < 2:
            self._preview_lead_reason = "missing_future_blocks"
            return current_action, first_avec
        if state is not None:
            state_arr = np.asarray(state, dtype=float).reshape(-1)
            if state_arr.size > 4:
                pitch_abs = abs(float(np.degrees(state_arr[4])))
                roll_abs = abs(float(np.degrees(state_arr[3])))
                if pitch_abs > self.preview_lead_pitch_abs_max_deg:
                    self._preview_lead_reason = "current_pitch_not_comfortable"
                    return current_action, first_avec
                if roll_abs > self.preview_lead_roll_abs_max_deg:
                    self._preview_lead_reason = "current_roll_not_comfortable"
                    return current_action, first_avec
        current_norm = float(self._raw_pressure_block_norms[0])
        if current_norm > self.preview_lead_current_norm_max:
            self._preview_lead_reason = "current_pressure_not_low"
            return current_action, first_avec
        current_speed_proxy = math.sqrt(max(current_norm, 0.0)) * 12.0
        if current_speed_proxy > self.preview_lead_current_wind_max_ms:
            self._preview_lead_reason = "current_wind_not_low"
            return current_action, first_avec

        chosen_idx = -1
        for idx in range(1, min(3, len(blocks))):
            if float(self._event_risk_probs[idx]) < self.preview_lead_event_threshold:
                continue
            future_norm = float(blocks[idx].get("pressure_norm", 0.0))
            if future_norm < self.preview_lead_future_norm_min:
                continue
            vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float)
            vecf = np.asarray(blocks[idx].get("pressure_vec", np.zeros(2)), dtype=float)
            if norm_term(vecf, self.cfg) < self.preview_lead_min_direction_norm:
                continue
            # Skip ambiguous sign-flips: lead only when the forecast direction is
            # consistent with the near block or the near block is too small to
            # carry a reliable direction.
            if (
                norm_term(vec0, self.cfg) >= self.preview_lead_min_direction_norm
                and float(np.dot(vec0, vecf)) < 0.0
            ):
                continue
            chosen_idx = idx
            break

        if chosen_idx < 0:
            self._preview_lead_reason = "no_eligible_future_action"
            return current_action, first_avec

        lead_name = self.preview_lead_action_name
        lead_vec = action_vec(lead_name, blocks[chosen_idx]["pressure_vec"], self.cfg)
        if norm_term(lead_vec, self.cfg) <= 1e-9:
            self._preview_lead_reason = "lead_vector_zero"
            return current_action, first_avec

        self._preview_lead_active = True
        self._preview_lead_reason = "future_action_prepump"
        self._preview_lead_block_index = int(chosen_idx)
        self._preview_lead_action_name = lead_name
        return lead_name, lead_vec

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
                if pitch_abs > self.preview_lead_pitch_abs_max_deg:
                    self._relief_medium_cap_reason = "current_pitch_not_comfortable"
                    return current_action, first_avec
                if roll_abs > self.preview_lead_roll_abs_max_deg:
                    self._relief_medium_cap_reason = "current_roll_not_comfortable"
                    return current_action, first_avec

        max_event = max(float(x) for x in self._event_risk_probs) if self._event_risk_probs else 0.0
        if max_event < self.relief_medium_cap_event_threshold:
            self._relief_medium_cap_reason = "event_risk_not_confident"
            return current_action, first_avec

        norms = [float(b.get("pressure_norm", 0.0)) for b in blocks[:3]]
        relief_margin = norms[0] - norms[2]
        self._relief_medium_cap_relief_margin = float(relief_margin)
        if relief_margin < self.pump_suppression_relief_margin_norm:
            self._relief_medium_cap_reason = "future_relief_too_small"
            return current_action, first_avec
        if norms[0] < 1.0:
            self._relief_medium_cap_reason = "current_pressure_not_high"
            return current_action, first_avec

        vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float)
        vec2 = np.asarray(blocks[2].get("pressure_vec", np.zeros(2)), dtype=float)
        if float(np.dot(vec0, vec2)) < 0.0:
            self._relief_medium_cap_reason = "future_direction_flip"
            return current_action, first_avec

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
        small_vec = direction * db * cap_ratio
        self._relief_medium_cap_active = True
        self._relief_medium_cap_reason = "forecast_relief_caps_medium"
        return "active_small", small_vec

    def _suppression_decision(self, blocks: list[dict[str, Any]]) -> tuple[bool, str, float, list[float], float]:
        norms = [float(b.get("pressure_norm", 0.0)) for b in blocks[:3]]
        while len(norms) < 3:
            norms.append(0.0)
        vec0 = np.asarray(blocks[0].get("pressure_vec", np.zeros(2)), dtype=float) if blocks else np.zeros(2)
        vec2 = np.asarray(blocks[2].get("pressure_vec", np.zeros(2)), dtype=float) if len(blocks) >= 3 else np.zeros(2)
        dot02 = float(np.dot(vec0, vec2))
        if not self.pump_suppression_enabled:
            return False, "disabled", 0.0, norms, dot02
        if (
            self.pump_suppression_event_risk_guard_enabled
            and max(self._event_risk_probs) >= self.pump_suppression_event_risk_guard_threshold
        ):
            return False, "event_risk_guard_blocks_suppression", 0.0, norms, dot02
        if norms[2] + self.pump_suppression_relief_margin_norm <= norms[0]:
            return True, "forecast_relief_block2_below_block0", self.pump_suppression_restart_err_kg, norms, dot02
        if dot02 < 0.0:
            return True, "forecast_sign_flip_block0_block2", self.pump_suppression_restart_err_kg, norms, dot02
        if max(norms) <= self.pump_suppression_low_risk_norm:
            return True, "forecast_low_risk_all_blocks", self.pump_suppression_restart_err_kg, norms, dot02
        return False, "forecast_not_safe_for_suppression", 0.0, norms, dot02

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
        masses = np.asarray(plant_info.get("tank_masses", self._primary_anchor_masses_kg), dtype=float).reshape(-1)
        if masses.size < 3:
            masses = self._primary_anchor_masses_kg
        masses = masses[:3].astype(float)
        self._primary_anchor_masses_kg = masses.copy()
        self._primary_delta_kg = self._primary_mass_delta_kg(first_avec)
        self._primary_target_kg = np.clip(
            self._primary_anchor_masses_kg + self._primary_delta_kg,
            0.0,
            self.cfg.tank_capacity_kg,
        )
        self._primary_target_initialized = True
        self._primary_target_refreshed = True
        self._primary_target_reused = False
        self._primary_target_resumed = False
        self._primary_target_last_reset_s = float(current_time)
        if float(np.linalg.norm(first_avec)) > 1e-12:
            self._paused_primary_target_kg = self._primary_target_kg.copy()
            self._paused_primary_delta_kg = self._primary_delta_kg.copy()
            self._paused_primary_avec = np.asarray(first_avec, dtype=float).reshape(2).copy()
            self._paused_primary_action = str(self._target_action)
            self._paused_primary_valid = True

    def _reuse_primary_target(self) -> None:
        self._primary_target_refreshed = False
        self._primary_target_reused = True
        self._primary_target_resumed = False

    def _remember_primary_target_for_pause(
        self,
        previous_action: str,
        previous_avec: np.ndarray,
    ) -> None:
        if self.primary_hold_target_mode != "pause":
            return
        if str(previous_action) == "hold" or not self._primary_target_initialized:
            return
        if float(np.mean(np.abs(self._primary_target_kg - self._primary_anchor_masses_kg))) <= 1.0:
            return
        self._paused_primary_target_kg = self._primary_target_kg.copy()
        self._paused_primary_delta_kg = self._primary_delta_kg.copy()
        self._paused_primary_avec = np.asarray(previous_avec, dtype=float).reshape(2).copy()
        self._paused_primary_action = str(previous_action)
        self._paused_primary_valid = True

    def _resume_paused_primary_target(
        self,
        plant_info: dict[str, Any],
        first_avec: np.ndarray,
        current_time: float,
    ) -> bool:
        if self.primary_hold_target_mode != "pause" or not self._paused_primary_valid:
            return False
        first_avec = np.asarray(first_avec, dtype=float).reshape(2)
        if float(np.linalg.norm(first_avec)) <= 1e-12:
            return False
        if float(np.linalg.norm(self._paused_primary_avec)) <= 1e-12:
            return False
        if float(np.dot(first_avec, self._paused_primary_avec)) < 0.0:
            self._paused_primary_valid = False
            return False

        masses = np.asarray(
            plant_info.get("tank_masses", self._primary_anchor_masses_kg),
            dtype=float,
        ).reshape(-1)
        if masses.size < 3:
            masses = self._primary_anchor_masses_kg
        masses = masses[:3].astype(float)
        self._primary_anchor_masses_kg = masses.copy()
        self._primary_target_kg = np.clip(
            self._paused_primary_target_kg.copy(),
            0.0,
            self.cfg.tank_capacity_kg,
        )
        self._primary_delta_kg = self._primary_target_kg - self._primary_anchor_masses_kg
        self._primary_target_initialized = True
        self._primary_target_refreshed = False
        self._primary_target_reused = True
        self._primary_target_resumed = True
        self._primary_target_last_reset_s = float(current_time)
        return True

    def _hold_primary_target_at_current(
        self,
        plant_info: dict[str, Any],
        current_time: float,
    ) -> None:
        masses = np.asarray(
            plant_info.get("tank_masses", self._primary_anchor_masses_kg),
            dtype=float,
        ).reshape(-1)
        if masses.size < 3:
            masses = self._primary_anchor_masses_kg
        masses = masses[:3].astype(float)
        self._primary_anchor_masses_kg = masses.copy()
        self._primary_delta_kg = np.zeros(3, dtype=float)
        self._primary_target_kg = np.clip(
            masses.copy(),
            0.0,
            self.cfg.tank_capacity_kg,
        )
        self._primary_target_initialized = True
        self._primary_target_refreshed = True
        self._primary_target_reused = False
        self._primary_target_resumed = False
        self._primary_target_last_reset_s = float(current_time)

    def compute(self, state, wind_obs, plant_info_prev, current_time):
        bucket = int(np.floor(max(float(current_time), 0.0) / self.update_interval_s + 1e-9))

        # Re-plan only on bucket boundaries.
        if bucket != self._last_bucket:
            self._last_bucket = bucket
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
                self._update_primary_target(
                    self._plant_info_from(plant_info_prev),
                    np.zeros(2, dtype=float),
                    float(current_time),
                )
                self._pump_suppression_active = False
                self._pump_suppression_reason = "missing_sample"
                self._pump_suppression_restart_err_kg_eff = 0.0
                self._pump_suppression_block_norms = [0.0, 0.0, 0.0]
                self._pump_suppression_block02_dot = 0.0
                self._event_risk_floor_active = [0, 0, 0]
                self._event_risk_floor_norms = [0.0, 0.0, 0.0]
                self._raw_pressure_block_norms = [0.0, 0.0, 0.0]
                self._preview_lead_active = False
                self._preview_lead_reason = "missing_sample"
                self._preview_lead_block_index = -1
                self._preview_lead_action_name = ""
            else:
                plant_info = self._plant_info_from(plant_info_prev)
                previous_action = str(self._target_action)
                previous_avec = np.array(
                    [self._target_pitch_planner_frame, self._target_roll_planner_frame],
                    dtype=float,
                )
                best, first_avec, forecast_source, blocks = self._run_planner(sample, plant_info)
                original_first_action = str(best["sequence"][0])
                new_action, first_avec = self._preview_lead_action(
                    best,
                    first_avec,
                    blocks,
                    state,
                )
                new_action, first_avec = self._relief_medium_cap_action(
                    new_action,
                    first_avec,
                    blocks,
                    state,
                )
                (
                    suppression_active,
                    suppression_reason,
                    suppression_restart_err_kg,
                    suppression_norms,
                    suppression_dot02,
                ) = self._suppression_decision(blocks)
                # first_avec is in PLANNER FRAME (compensation direction: which way to push
                # ballast to oppose wind pressure). For pump-first operation the default
                # setpoint sign is +1.0 because the signpos experiment was the only
                # pump-saving path; -1.0 remains an attitude-first diagnostic mode.
                planner_pitch = float(first_avec[0])
                planner_roll = float(first_avec[1])
                new_pitch = planner_pitch * self.setpoint_bias_sign
                new_roll = planner_roll * self.setpoint_bias_sign
                event_reset = self._should_reset_event(new_action, new_pitch, new_roll, float(current_time))
                if event_reset:
                    self._transition_time_s = float(current_time)
                    self._target_action = new_action
                self._target_pitch = new_pitch
                self._target_roll = new_roll
                self._target_pitch_planner_frame = planner_pitch
                self._target_roll_planner_frame = planner_roll
                if self.prediction_primary_enabled and new_action == "hold":
                    self._remember_primary_target_for_pause(previous_action, previous_avec)
                    self._hold_primary_target_at_current(plant_info, float(current_time))
                elif self._resume_paused_primary_target(
                    plant_info,
                    first_avec,
                    float(current_time),
                ):
                    pass
                elif event_reset or not self._primary_target_initialized:
                    self._update_primary_target(plant_info, first_avec, float(current_time))
                else:
                    self._reuse_primary_target()
                self._pump_suppression_active = bool(suppression_active)
                self._pump_suppression_reason = str(suppression_reason)
                self._pump_suppression_restart_err_kg_eff = float(suppression_restart_err_kg)
                self._pump_suppression_block_norms = [float(x) for x in suppression_norms[:3]]
                self._pump_suppression_block02_dot = float(suppression_dot02)
                self.records.append({
                    "bucket": bucket,
                    "current_time_s": float(current_time),
                    "history_end": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "first_action": new_action,
                    "planner_first_action_raw": original_first_action,
                    "best_sequence": ">".join(best["sequence"]),
                    # planner frame: compensation direction (sign as planner sees it)
                    "planner_action_pitch_deg": planner_pitch,
                    "planner_action_roll_deg": planner_roll,
                    "setpoint_bias_sign": float(self.setpoint_bias_sign),
                    "prediction_primary_enabled": int(self.prediction_primary_enabled),
                    "prediction_primary_scale": float(self.prediction_primary_scale),
                    "prediction_primary_target_t1_kg": float(self._primary_target_kg[0]),
                    "prediction_primary_target_t2_kg": float(self._primary_target_kg[1]),
                    "prediction_primary_target_t3_kg": float(self._primary_target_kg[2]),
                    "prediction_primary_delta_abs_mean_kg": float(np.mean(np.abs(self._primary_delta_kg))),
                    "prediction_primary_event_reset": int(event_reset),
                    "prediction_primary_target_refreshed": int(self._primary_target_refreshed),
                    "prediction_primary_target_reused": int(self._primary_target_reused),
                    "prediction_primary_target_resumed": int(self._primary_target_resumed),
                    "prediction_primary_hold_target_mode": self.primary_hold_target_mode,
                    "prediction_primary_target_age_s": float(current_time) - float(self._primary_target_last_reset_s),
                    # setpoint frame (after sign contract applied): what controller sees
                    "target_pitch_deg": new_pitch,
                    "target_roll_deg": new_roll,
                    "transition_time_s": float(self._transition_time_s),
                    "objective_mode": self.objective_mode,
                    "event_reset_mode": self.event_reset_mode,
                    "forecast_source": forecast_source,
                    "best_scalar_cost": float(best.get("best_scalar_cost", float("nan"))),
                    "max_envelope_norm": float(best.get("costs", {}).get("max_envelope_norm", float("nan"))),
                    "envelope_violation_cost": float(best.get("costs", {}).get("envelope_violation_cost", float("nan"))),
                    "envelope_barrier_triggered": int(
                        float(best.get("costs", {}).get("envelope_barrier_triggered", 0.0)) > 0.5
                    ),
                    "pump_work_cost": float(best.get("costs", {}).get("pump_work_cost", float("nan"))),
                    "active_medium_gate_reason": str(best.get("active_medium_gate_reason", "")),
                    "reverse_release_reason": str(best.get("reverse_release_reason", "")),
                    "pump_suppression_active": int(self._pump_suppression_active),
                    "pump_suppression_restart_err_kg": float(self._pump_suppression_restart_err_kg_eff),
                    "pump_suppression_reason": self._pump_suppression_reason,
                    "pressure_block0_norm": float(self._pump_suppression_block_norms[0]),
                    "pressure_block1_norm": float(self._pump_suppression_block_norms[1]),
                    "pressure_block2_norm": float(self._pump_suppression_block_norms[2]),
                    "pressure_block02_dot": float(self._pump_suppression_block02_dot),
                    "event_risk_pressure_boost_enabled": int(self.event_risk_pressure_boost_enabled),
                    "event_risk_prob_0_20m": float(self._event_risk_probs[0]),
                    "event_risk_prob_20_40m": float(self._event_risk_probs[1]),
                    "event_risk_prob_40_60m": float(self._event_risk_probs[2]),
                    "event_risk_scale_0_20m": float(self._event_risk_block_scales[0]),
                    "event_risk_scale_20_40m": float(self._event_risk_block_scales[1]),
                    "event_risk_scale_40_60m": float(self._event_risk_block_scales[2]),
                    "event_risk_pressure_floor_enabled": int(
                        self.event_risk_pressure_floor_enabled
                    ),
                    "event_risk_floor_active_0_20m": int(self._event_risk_floor_active[0]),
                    "event_risk_floor_active_20_40m": int(self._event_risk_floor_active[1]),
                    "event_risk_floor_active_40_60m": int(self._event_risk_floor_active[2]),
                    "event_risk_floor_norm_0_20m": float(self._event_risk_floor_norms[0]),
                    "event_risk_floor_norm_20_40m": float(self._event_risk_floor_norms[1]),
                    "event_risk_floor_norm_40_60m": float(self._event_risk_floor_norms[2]),
                    "preview_lead_action_enabled": int(self.preview_lead_action_enabled),
                    "preview_lead_action_active": int(self._preview_lead_active),
                    "preview_lead_action_reason": self._preview_lead_reason,
                    "preview_lead_action_block_index": int(self._preview_lead_block_index),
                    "preview_lead_action_name": self._preview_lead_action_name,
                    "pump_suppression_event_risk_guard_enabled": int(
                        self.pump_suppression_event_risk_guard_enabled
                    ),
                    "pump_suppression_event_risk_guard_threshold": float(
                        self.pump_suppression_event_risk_guard_threshold
                    ),
                    "relief_medium_cap_enabled": int(self.relief_medium_cap_enabled),
                    "relief_medium_cap_active": int(self._relief_medium_cap_active),
                    "relief_medium_cap_reason": self._relief_medium_cap_reason,
                    "relief_medium_cap_margin": float(self._relief_medium_cap_relief_margin),
                })

        pitch_bias, roll_bias = self._shaped_bias(current_time)
        if (
            not self.setpoint_channel_enabled
            or (self.prediction_primary_enabled and self.prediction_primary_disable_setpoint)
        ):
            pitch_bias = 0.0
            roll_bias = 0.0
        mass_ff_kg = self._shaped_mass_ff_kg(current_time)
        primary_active = int(
            self.prediction_primary_enabled
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
            "preview_primary_enabled": int(self.prediction_primary_enabled),
            "preview_primary_active": int(primary_active),
            "preview_primary_target_kg": [float(x) for x in self._primary_target_kg],
            "preview_primary_delta_kg": [float(x) for x in self._primary_delta_kg],
            "preview_primary_action": self._target_action,
            "preview_primary_hold_target_mode": self.primary_hold_target_mode,
            "preview_primary_event_reset": int(self._primary_target_refreshed),
            "preview_primary_target_refreshed": int(self._primary_target_refreshed),
            "preview_primary_target_reused": int(self._primary_target_reused),
            "preview_primary_target_resumed": int(self._primary_target_resumed),
            "preview_primary_target_age_s": max(0.0, float(current_time) - float(self._primary_target_last_reset_s)),
            "preview_pump_suppression_active": int(self._pump_suppression_active),
            "preview_pump_restart_err_kg": float(self._pump_suppression_restart_err_kg_eff),
            "preview_pump_suppression_reason": str(self._pump_suppression_reason),
            "preview_pressure_block0_norm": float(self._pump_suppression_block_norms[0]),
            "preview_pressure_block1_norm": float(self._pump_suppression_block_norms[1]),
            "preview_pressure_block2_norm": float(self._pump_suppression_block_norms[2]),
            "preview_pressure_block02_dot": float(self._pump_suppression_block02_dot),
            "preview_event_risk_pressure_boost_enabled": int(self.event_risk_pressure_boost_enabled),
            "preview_event_risk_prob_0_20m": float(self._event_risk_probs[0]),
            "preview_event_risk_prob_20_40m": float(self._event_risk_probs[1]),
            "preview_event_risk_prob_40_60m": float(self._event_risk_probs[2]),
            "preview_event_risk_scale_0_20m": float(self._event_risk_block_scales[0]),
            "preview_event_risk_scale_20_40m": float(self._event_risk_block_scales[1]),
            "preview_event_risk_scale_40_60m": float(self._event_risk_block_scales[2]),
            "preview_event_risk_pressure_floor_enabled": int(
                self.event_risk_pressure_floor_enabled
            ),
            "preview_event_risk_floor_active_0_20m": int(self._event_risk_floor_active[0]),
            "preview_event_risk_floor_active_20_40m": int(self._event_risk_floor_active[1]),
            "preview_event_risk_floor_active_40_60m": int(self._event_risk_floor_active[2]),
            "preview_event_risk_floor_norm_0_20m": float(self._event_risk_floor_norms[0]),
            "preview_event_risk_floor_norm_20_40m": float(self._event_risk_floor_norms[1]),
            "preview_event_risk_floor_norm_40_60m": float(self._event_risk_floor_norms[2]),
            "preview_lead_action_enabled": int(self.preview_lead_action_enabled),
            "preview_lead_action_active": int(self._preview_lead_active),
            "preview_lead_action_reason": self._preview_lead_reason,
            "preview_lead_action_block_index": int(self._preview_lead_block_index),
            "preview_lead_action_name": self._preview_lead_action_name,
            "preview_pump_suppression_event_risk_guard_enabled": int(
                self.pump_suppression_event_risk_guard_enabled
            ),
            "preview_pump_suppression_event_risk_guard_threshold": float(
                self.pump_suppression_event_risk_guard_threshold
            ),
            "preview_relief_medium_cap_enabled": int(self.relief_medium_cap_enabled),
            "preview_relief_medium_cap_active": int(self._relief_medium_cap_active),
            "preview_relief_medium_cap_reason": self._relief_medium_cap_reason,
            "preview_relief_medium_cap_margin": float(self._relief_medium_cap_relief_margin),
            "source": f"ballast_planner_{self.objective_mode}_{self.bias_shape}",
            "preview_trim_source": f"ballast_planner_{self.objective_mode}_{self.bias_shape}",
            "preview_gain": float(self.scale),
            "risk_level": "active" if self._target_action != "hold" else "hold",
            "preview_mode": f"ballast_planner_{self.objective_mode}_{self.bias_shape}",
            "first_action": self._target_action,
        }
