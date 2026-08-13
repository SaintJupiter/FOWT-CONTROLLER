"""Planner evaluation and forecast-advised action shaping."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any

import numpy as np

from .ballast_planner import (
    MAINTAIN_TARGET_ACTION,
    STOP_EXECUTION_ACTION,
    apply_posture_state_to_blocks,
    candidate_actions,
    candidate_sequences,
    compare_sequences,
    compute_pressure_blocks,
    evaluate_sequence,
    norm_term,
    planner_action_vec,
    select_best_economic,
)
from .forecast_admission import (
    EVENT_RISK_KEYS as _EVENT_RISK_KEYS,
)
from .forecast_action_policy import (
    ForecastActionPolicyResult,
    build_forecast_action_policy_snapshot,
    evaluate_forecast_action_policy,
)
from .control_telemetry_defaults import (
    frozen_pump_suppression_preview_defaults,
    frozen_pump_suppression_record_defaults,
)
from .provider_common import (
    FORECAST_REINTENSIFY_HIGH_NORM as _FORECAST_REINTENSIFY_HIGH_NORM,
    FORECAST_RELIEF_MARGIN_NORM as _FORECAST_RELIEF_MARGIN_NORM,
    complete_event_probabilities_available,
)


_FORECAST_POLICY_FAMILY_BY_ACTION = {
    "active_medium": "strengthen",
    "hold": "hold",
    MAINTAIN_TARGET_ACTION: "hold",
    STOP_EXECUTION_ACTION: "release",
    "active_reverse_small": "reverse",
}


def _current_wind_uv_ms(wind_obs: Any) -> np.ndarray | None:
    """Convert one replay observation to the forecast model's UV convention."""

    if not isinstance(wind_obs, dict):
        return None
    try:
        speed = float(wind_obs["ws"])
        direction_deg = float(wind_obs["wd_deg"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(speed) or not np.isfinite(direction_deg) or speed < 0.0:
        return None
    direction_rad = np.deg2rad(direction_deg)
    return np.asarray(
        [-speed * np.sin(direction_rad), -speed * np.cos(direction_rad)],
        dtype=float,
    )


@dataclass(frozen=True)
class ForecastActionEnforcement:
    """Audit record for the first-action candidate boundary."""

    applied: bool
    reason: str
    candidate_count_before: int
    candidate_count_after: int
    blocked_first_actions: tuple[str, ...] = ()
    posture_strengthen_restored: bool = False


def enforce_forecast_policy_on_first_actions(
    sequences: list[tuple[str, ...]],
    *,
    mode: str,
    result: ForecastActionPolicyResult | None,
    posture_strengthening_required: bool = False,
) -> tuple[list[tuple[str, ...]], ForecastActionEnforcement]:
    """Restrict only candidate first actions when policy authority is enabled.

    Ordinary and reduced forward actions remain available to the planner. The
    forecast policy controls the four action families it explicitly describes:
    strengthening, target hold, target release, and reverse correction.
    """

    original = list(sequences)
    normalized_mode = str(mode)
    if normalized_mode != "enforce":
        return original, ForecastActionEnforcement(
            applied=False,
            reason=f"mode_{normalized_mode}",
            candidate_count_before=len(original),
            candidate_count_after=len(original),
        )
    if result is None or not result.enabled:
        return original, ForecastActionEnforcement(
            applied=False,
            reason="policy_unavailable",
            candidate_count_before=len(original),
            candidate_count_after=len(original),
        )
    if not result.stages:
        return original, ForecastActionEnforcement(
            applied=False,
            reason="policy_has_no_stages",
            candidate_count_before=len(original),
            candidate_count_after=len(original),
        )

    first_stage = result.stages[0]
    authorization = {
        "strengthen": bool(first_stage.strengthen.allowed),
        "hold": bool(first_stage.hold.allowed),
        "release": bool(first_stage.release.allowed),
        "reverse": bool(first_stage.reverse.allowed),
    }
    posture_strengthen_restored = bool(
        posture_strengthening_required and not authorization["strengthen"]
    )
    if posture_strengthening_required:
        # Future wind evidence may authorize a pre-emptive strong action, but
        # it must not veto a corrective action already supported by measured
        # posture. Restoring the candidate does not force it; normal sequence
        # evaluation still decides whether the stronger action is worthwhile.
        authorization["strengthen"] = True
    filtered: list[tuple[str, ...]] = []
    blocked_first_actions: set[str] = set()
    for sequence in original:
        if not sequence:
            continue
        first_action = str(sequence[0])
        family = _FORECAST_POLICY_FAMILY_BY_ACTION.get(first_action)
        if family is not None and not authorization[family]:
            blocked_first_actions.add(first_action)
            continue
        filtered.append(sequence)

    blocked = tuple(sorted(blocked_first_actions))
    if not filtered and original:
        # Authorization is advisory at this boundary unless at least one
        # candidate remains available for the normal planner evaluation. An
        # empty policy result must not synthesize an unevaluated lifecycle
        # action such as stop_execution.
        return original, ForecastActionEnforcement(
            applied=False,
            reason="empty_filter_rollback",
            candidate_count_before=len(original),
            candidate_count_after=len(original),
            blocked_first_actions=blocked,
            posture_strengthen_restored=posture_strengthen_restored,
        )

    return filtered, ForecastActionEnforcement(
        applied=bool(filtered != original),
        reason=("filtered" if filtered != original else "no_candidates_blocked"),
        candidate_count_before=len(original),
        candidate_count_after=len(filtered),
        blocked_first_actions=blocked,
        posture_strengthen_restored=posture_strengthen_restored,
    )


class ProviderPlanningMixin:
    """Internal behavior group for ``BallastPlannerPreviewProvider``."""

    def _run_planner(
        self,
        sample,
        plant_info: dict,
        forecast_has_future: bool = True,
    ) -> tuple[dict[str, Any], np.ndarray, str, list[dict[str, Any]]]:
        evidence = self._forecast_uv(sample)
        uv = np.asarray(evidence.uv_ms, dtype=float)
        forecast_source = str(evidence.source)
        event_probs = dict(evidence.event_probs)
        self._forecast_evidence = evidence
        self._forecast_action_policy_result = evaluate_forecast_action_policy(
            evidence,
            current_uv_ms=_current_wind_uv_ms(getattr(sample, "wind_obs", None)),
            config=self.forecast_action_policy_config,
        )
        self._forecast_action_policy_snapshot = (
            build_forecast_action_policy_snapshot(
                self._forecast_action_policy_result,
                mode=self.forecast_action_policy_mode,
            )
        )
        self._forecast_has_future = bool(
            forecast_has_future and evidence.provides_future_preview
        )
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
        raw_blocks = compute_pressure_blocks(
            uv,
            block_discounts,
            self.cfg,
            lead_reliability=(
                evidence.lead_reliability
                if self._forecast_has_future
                else np.ones(uv.shape[0], dtype=float)
            ),
            event_probs=event_probs,
        )
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
        planner_info["primary_target_kg"] = np.asarray(
            self._primary_target_kg,
            dtype=float,
        ).reshape(3).copy()
        planner_info["prediction_primary_scale"] = float(
            self.prediction_primary_scale
        )
        planner_info["primary_hold_target_mode"] = str(
            self.primary_hold_target_mode
        )
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
        # Retain the exact evaluation context for supervision. Any later
        # action/vector correction must be checked against these same forecast
        # blocks, plant state, and objective settings before it is committed.
        self._last_planner_blocks = planner_blocks
        self._last_planner_info = planner_info
        self._last_planner_cfg = active_cfg
        sequences = candidate_sequences(active_cfg)
        if self.no_preview_myopic_horizon_enabled and not self._forecast_has_future:
            # A strict no-preview controller should choose the immediate action
            # from current state/pressure only, not optimize a 60-minute
            # persistence pseudo-forecast. Keep the same planner/cost/action
            # library, but restrict future blocks to hold for this diagnostic.
            actions = candidate_actions(active_cfg)
            future_action = (
                "maintain_target"
                if active_cfg.candidate_action_mode
                == "explicit_target_lifecycle"
                else "hold"
            )
            sequences = [(a, future_action, future_action) for a in actions]
        sequences, policy_enforcement = enforce_forecast_policy_on_first_actions(
            sequences,
            mode=self.forecast_action_policy_mode,
            result=self._forecast_action_policy_result,
            posture_strengthening_required=bool(
                planner_blocks
                and np.linalg.norm(
                    np.asarray(
                        planner_blocks[0].get(
                            "posture_action_demand_vec",
                            np.zeros(2, dtype=float),
                        ),
                        dtype=float,
                    )
                ) > 1e-12
            ),
        )
        self._forecast_action_policy_enforcement = policy_enforcement
        # A short diagnostic showed that deleting no-op candidates from this
        # legacy provider path can force repeated target rewrites and increase
        # both pumping and posture error. Posture recovery authority therefore
        # belongs in the new controller core, where lifecycle, candidate cost,
        # and actuator execution are evaluated as one transaction.
        self._posture_action_enforcement = None
        if not sequences:
            fallback_action = (
                STOP_EXECUTION_ACTION
                if active_cfg.candidate_action_mode == "explicit_target_lifecycle"
                else "hold"
            )
            return {
                "sequence": (fallback_action,) * 3,
                "first_action": fallback_action,
                "best_scalar_cost": float("nan"),
                "planned_reactive_handoff": 0,
                "selection_reason": "forecast_action_policy_empty_safe_fallback",
            }, np.zeros(2), forecast_source, blocks
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
            fallback_action = (
                STOP_EXECUTION_ACTION
                if active_cfg.candidate_action_mode == "explicit_target_lifecycle"
                else "hold"
            )
            return {"sequence": (fallback_action,) * 3, "first_action": fallback_action,
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
