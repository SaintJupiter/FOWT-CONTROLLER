"""Forecast trust, shape, stability, and horizon analysis."""

from __future__ import annotations

import math
from datetime import (
    timedelta,
)
from typing import Any

import numpy as np

from .ballast_planner import (
    norm_term,
    pressure_proxy_vec,
)
from .forecast_admission import (
    TrustedEventGateConfig,
    trusted_event_gate,
)



class ProviderForecastMixin:
    """Internal behavior group for ``BallastPlannerPreviewProvider``."""

    def _trusted_event_gate(self, event_probs: dict[str, float]) -> tuple[dict[str, float], dict[str, Any]]:
        return trusted_event_gate(
            event_probs,
            TrustedEventGateConfig(
                enabled=bool(self.trusted_event_gate_enabled),
                attention_threshold=float(self.trusted_event_attention_threshold),
                highwind_threshold=float(self.trusted_event_highwind_threshold),
                dynamic_threshold=float(self.trusted_event_dynamic_threshold),
                attention_min_heads=int(self.trusted_event_attention_min_heads),
                dynamic_min_heads=int(self.trusted_event_dynamic_min_heads),
                dynamic_enabled=bool(self.trusted_event_dynamic_enabled),
            ),
        )

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
