"""Economy, relief, and long-horizon experimental policy helpers."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np


from .provider_common import (
    HOLD_FORECAST_POSTURE_RELIEF_NORM as _HOLD_FORECAST_POSTURE_RELIEF_NORM,
)


@dataclass(frozen=True)
class _EconomySignals:
    direction_ok: bool
    dot02: float
    fallback_guard_ok: bool
    far_boundary: bool
    floor_risk: bool
    high_then_decay_shape: bool
    high_then_relief: bool
    lowrisk_redundant: bool
    max_axis: float
    mode: str
    posture: np.ndarray
    posture_floor_risk: bool
    pressure_max: float
    psc_v2_base_neutral: bool
    psc_v2_lowrisk: bool
    psc_v2_neutral_frame: bool
    psc_v2_p1: bool
    psc_v2_p1_short_high_peak: bool
    psc_v2_w1: bool
    psc_v2_w2: bool
    psc_v2_w2_flat: bool
    reintensify: bool
    relief: bool
    residual_high_economy: bool
    residual_high_plateau: bool
    residual_high_plateau_strict: bool
    residual_high_stable: bool
    rise_then_fall: bool
    vec0: np.ndarray
    vec2: np.ndarray
    wind_speed_episode_entry: bool
    wind_speed_relief_decay: bool



class ProviderEconomyMixin:
    """Internal behavior group for ``BallastPlannerPreviewProvider``."""

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










    # ------------------------------------------------------------------
    # h120 learned risk scheduler (default-off remote-risk delay/early-stop layer)
    # ------------------------------------------------------------------

        # Counter increment is done by the single per-bucket call site
        # (before _reactive_floor_needed) to avoid double-counting when this
        # method is also called from the early_stop guard in the same bucket.





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
        signals = self._economy_budget_signals(plant_info)
        return self._economy_budget_mode_decision(signals)

    def _economy_budget_signals(
        self,
        plant_info: dict[str, Any] | None,
    ) -> _EconomySignals:
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
        return _EconomySignals(
            direction_ok=direction_ok,
            dot02=dot02,
            fallback_guard_ok=fallback_guard_ok,
            far_boundary=far_boundary,
            floor_risk=floor_risk,
            high_then_decay_shape=high_then_decay_shape,
            high_then_relief=high_then_relief,
            lowrisk_redundant=lowrisk_redundant,
            max_axis=max_axis,
            mode=mode,
            posture=posture,
            posture_floor_risk=posture_floor_risk,
            pressure_max=pressure_max,
            psc_v2_base_neutral=psc_v2_base_neutral,
            psc_v2_lowrisk=psc_v2_lowrisk,
            psc_v2_neutral_frame=psc_v2_neutral_frame,
            psc_v2_p1=psc_v2_p1,
            psc_v2_p1_short_high_peak=psc_v2_p1_short_high_peak,
            psc_v2_w1=psc_v2_w1,
            psc_v2_w2=psc_v2_w2,
            psc_v2_w2_flat=psc_v2_w2_flat,
            reintensify=reintensify,
            relief=relief,
            residual_high_economy=residual_high_economy,
            residual_high_plateau=residual_high_plateau,
            residual_high_plateau_strict=residual_high_plateau_strict,
            residual_high_stable=residual_high_stable,
            rise_then_fall=rise_then_fall,
            vec0=vec0,
            vec2=vec2,
            wind_speed_episode_entry=wind_speed_episode_entry,
            wind_speed_relief_decay=wind_speed_relief_decay,
        )

    def _economy_budget_mode_decision(
        self,
        signals: _EconomySignals,
    ) -> tuple[bool, str]:
        mode = signals.mode
        floor_risk = signals.floor_risk
        reintensify = signals.reintensify
        direction_ok = signals.direction_ok
        high_then_relief = signals.high_then_relief
        rise_then_fall = signals.rise_then_fall
        wind_speed_relief_decay = signals.wind_speed_relief_decay
        wind_speed_episode_entry = signals.wind_speed_episode_entry
        high_then_decay_shape = signals.high_then_decay_shape
        fallback_guard_ok = signals.fallback_guard_ok
        posture_floor_risk = signals.posture_floor_risk
        residual_high_stable = signals.residual_high_stable
        far_boundary = signals.far_boundary
        residual_high_plateau_strict = signals.residual_high_plateau_strict
        residual_high_plateau = signals.residual_high_plateau
        psc_v2_base_neutral = signals.psc_v2_base_neutral
        psc_v2_neutral_frame = signals.psc_v2_neutral_frame
        psc_v2_lowrisk = signals.psc_v2_lowrisk
        psc_v2_p1 = signals.psc_v2_p1
        psc_v2_p1_short_high_peak = signals.psc_v2_p1_short_high_peak
        psc_v2_w1 = signals.psc_v2_w1
        psc_v2_w2 = signals.psc_v2_w2
        psc_v2_w2_flat = signals.psc_v2_w2_flat
        pressure_max = signals.pressure_max
        dot02 = signals.dot02
        max_axis = signals.max_axis
        posture = signals.posture
        vec0 = signals.vec0
        vec2 = signals.vec2
        lowrisk_redundant = signals.lowrisk_redundant
        residual_high_economy = signals.residual_high_economy
        relief = signals.relief
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
