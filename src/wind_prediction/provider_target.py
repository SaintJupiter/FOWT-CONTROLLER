"""Primary target lifecycle, safety refresh, pause, and release logic."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .ballast_planner import (
    action_vec,
    norm_term,
    tank_signal,
)
from .safety_supervisor import (
    posture_vector_deg,
    reactive_floor_log_fields,
    reactive_floor_posture_metrics,
    reactive_floor_theta_gate_decision,
    reactive_floor_veto_decision,
)
from .target_lifecycle import (
    apply_primary_target_transition,
    consume_primary_refresh_owner,
    mark_primary_target_reused,
    plant_primary_masses,
    primary_target_at_current,
    primary_target_blended_to_current,
    primary_target_capped_to_current,
    primary_target_from_delta,
    primary_target_from_paused,
    primary_target_proposal_metrics,
    remember_paused_primary_target,
)

from .provider_common import (
    FORECAST_RELIEF_MARGIN_NORM as _FORECAST_RELIEF_MARGIN_NORM,
    HOLD_RELEASE_ENTER_PITCH_DEG as _HOLD_RELEASE_ENTER_PITCH_DEG,
    HOLD_RELEASE_ENTER_ROLL_DEG as _HOLD_RELEASE_ENTER_ROLL_DEG,
)


class ProviderTargetMixin:
    """Internal behavior group for ``BallastPlannerPreviewProvider``."""

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
            pending = np.asarray(self._primary_target_kg, dtype=float).reshape(-1)[:3]
            transition = primary_target_blended_to_current(
                masses,
                pending,
                blend=blend,
                tank_capacity_kg=self.cfg.tank_capacity_kg,
                current_time_s=float(current_time),
            )
            self._forecast_deadband_held_target_kg = transition.target_kg.copy()
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
