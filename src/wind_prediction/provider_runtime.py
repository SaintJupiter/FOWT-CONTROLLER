"""Per-cycle provider orchestration and telemetry assembly."""

from __future__ import annotations


import numpy as np

from .action_plan import (
    TargetOperation,
    apply_action_correction,
    build_candidate_plan,
    commit_action_plan,
    record_evaluation,
)
from .ballast_planner import (
    MAINTAIN_TARGET_ACTION,
    STOP_EXECUTION_ACTION,
    candidate_actions,
    economic_scalar_cost,
    evaluate_sequence,
)
from .control_contracts import DecisionTraceStep
from .target_lifecycle import (
    apply_planner_authoritative_target,
    mark_primary_target_reused,
    plant_primary_masses,
)



class ProviderRuntimeMixin:
    def _release_action_name(self) -> str:
        cfg = getattr(self, "_last_planner_cfg", None)
        if getattr(cfg, "candidate_action_mode", "legacy_five") == (
            "explicit_target_lifecycle"
        ):
            return STOP_EXECUTION_ACTION
        return "hold"

    @staticmethod
    def _is_release_action(action: str) -> bool:
        return str(action) in {"hold", STOP_EXECUTION_ACTION}

    @staticmethod
    def _decision_vector(action_vec) -> tuple[float, ...]:
        """Return the planner-frame vector used by one trace step."""

        return tuple(
            float(value)
            for value in np.asarray(action_vec, dtype=float).reshape(-1)[:2]
        )

    def _record_decision_trace_step(
        self,
        *,
        name: str,
        input_action: str,
        input_vector,
        output_action: str,
        output_vector,
        reason: str = "",
    ) -> DecisionTraceStep:
        """Append a behavior-neutral trace step for post-ranking supervision."""

        input_vec = self._decision_vector(input_vector)
        output_vec = self._decision_vector(output_vector)
        step = DecisionTraceStep(
            name=str(name),
            input_action=str(input_action),
            output_action=str(output_action),
            changed=(
                str(input_action) != str(output_action)
                or input_vec != output_vec
            ),
            reason=str(reason),
            metadata={
                "input_vector": input_vec,
                "output_vector": output_vec,
            },
        )
        self._decision_trace_steps = (
            *tuple(getattr(self, "_decision_trace_steps", ())),
            step,
        )
        return step

    @staticmethod
    def _decision_trace_input(action: str, action_vec):
        return (
            str(action),
            np.asarray(action_vec, dtype=float).reshape(-1)[:2].copy(),
        )

    def _record_decision_transition(
        self,
        name: str,
        trace_input,
        output_action: str,
        output_vector,
        reason: str,
    ) -> None:
        input_action, input_vector = trace_input
        self._record_decision_trace_step(
            name=name,
            input_action=input_action,
            input_vector=input_vector,
            output_action=output_action,
            output_vector=output_vector,
            reason=reason,
        )

    def _apply_traced_action_supervision(
        self,
        *,
        best,
        first_avec,
        bucket: int,
        blocks,
        plant_info,
        state,
        current_time: float,
    ):
        """Apply action-label supervisors while recording every transition."""

        original_action = str(best["sequence"][0])
        self._forced_original_raw_action = original_action
        before = self._decision_trace_input(original_action, first_avec)
        forced_action, forced_avec = self._apply_forced_prefix_action(
            "raw_action",
            bucket,
            original_action,
            first_avec,
            blocks,
            plant_info,
        )
        if self._forced_prefix_active and self._forced_prefix_mode_active == "raw_action":
            sequence = tuple(best.get("sequence", ("hold", "hold", "hold")))
            best = dict(best)
            best["sequence"] = (forced_action,) + tuple(sequence[1:])
            first_avec = forced_avec
        pipeline_raw_action = str(best["sequence"][0])
        raw_reason = (
            str(self._forced_source_label)
            if self._forced_prefix_active
            and self._forced_prefix_mode_active == "raw_action"
            else "not_applied"
        )
        self._record_decision_transition(
            "raw_forced", before, pipeline_raw_action, first_avec, raw_reason
        )

        new_action = pipeline_raw_action
        before = self._decision_trace_input(new_action, first_avec)
        new_action, first_avec = self._relief_medium_cap_action(
            new_action, first_avec, blocks, state
        )
        self._record_decision_transition(
            "medium_cap",
            before,
            new_action,
            first_avec,
            str(self._relief_medium_cap_reason),
        )

        before = self._decision_trace_input(new_action, first_avec)
        new_action, first_avec = self._far_horizon_relief_gate_action(
            new_action, first_avec, blocks, state
        )
        self._record_decision_transition(
            "far_cap",
            before,
            new_action,
            first_avec,
            str(self._far_horizon_relief_gate_reason),
        )

        self._forced_original_final_action = str(new_action)
        before = self._decision_trace_input(new_action, first_avec)
        new_action, first_avec = self._apply_forced_prefix_action(
            "final_action",
            bucket,
            new_action,
            first_avec,
            blocks,
            plant_info,
        )
        final_reason = (
            str(self._forced_source_label)
            if self._forced_prefix_active
            and self._forced_prefix_mode_active == "final_action"
            else "not_applied"
        )
        self._record_decision_transition(
            "final_forced", before, new_action, first_avec, final_reason
        )

        self._observe_forecast_advised_economy_decision(
            blocks,
            plant_info,
            float(current_time),
            proposed_action=new_action,
        )
        before = self._decision_trace_input(new_action, first_avec)
        if (
            self.forecast_advised_economy_hold_current_on_candidate
            and self._forecast_advised_economy_candidate
            and not self._is_release_action(new_action)
        ):
            self._forecast_advised_hold_current_override = True
            new_action = self._release_action_name()
            first_avec = np.zeros(2, dtype=float)
        self._record_decision_transition(
            "economy_hold",
            before,
            new_action,
            first_avec,
            str(self._forecast_advised_economy_reason),
        )
        return best, pipeline_raw_action, new_action, first_avec

    def _apply_traced_vector_supervision(
        self,
        *,
        action: str,
        action_vec,
        plant_info,
        blocks,
        current_time: float,
        event_reset: bool,
        posture_action_refresh: bool,
    ):
        """Apply vector supervisors and the final veto with trace capture."""

        before = self._decision_trace_input(action, action_vec)
        stall_avec, stall_refresh, stall_reason = self._stall_refresh_action_vec(
            action_vec, plant_info, float(current_time)
        )
        self._primary_stall_refresh_active = bool(
            stall_refresh and not self._is_release_action(action)
        )
        self._primary_stall_refresh_reason = (
            stall_reason if not self._is_release_action(action) else "release_action"
        )
        if self._primary_stall_refresh_active:
            action_vec = stall_avec
            event_reset = True
        self._record_decision_transition(
            "stall_vector_rewrite",
            before,
            action,
            action_vec,
            str(self._primary_stall_refresh_reason),
        )

        before = self._decision_trace_input(action, action_vec)
        action_vec, _gust_active, gust_reason = (
            self._gusty_oscillation_soft_cap_action(action_vec, plant_info)
        )
        self._record_decision_transition(
            "gust_cap", before, action, action_vec, str(gust_reason)
        )

        direction_veto, direction_reason = (
            self._direction_reversal_layered_entry_veto_active(plant_info, blocks)
        )
        before = self._decision_trace_input(action, action_vec)
        if direction_veto:
            event_reset = False
            posture_action_refresh = False
            action = self._release_action_name()
            action_vec = np.zeros(2, dtype=float)
        self._record_decision_transition(
            "direction_reversal_veto",
            before,
            action,
            action_vec,
            str(direction_reason),
        )
        return (
            action,
            action_vec,
            bool(event_reset),
            bool(posture_action_refresh),
            bool(direction_veto),
            str(direction_reason),
        )

    @staticmethod
    def _target_operation_for_action(action: str) -> TargetOperation:
        normalized = str(action)
        if normalized == "hold":
            return TargetOperation.HOLD_CURRENT
        if normalized == STOP_EXECUTION_ACTION:
            return TargetOperation.RELEASE
        if normalized == MAINTAIN_TARGET_ACTION:
            return TargetOperation.CONTINUE
        return TargetOperation.SET_DELTA

    def _build_ranked_action_plan(self, best, action: str, avec):
        evaluated_target = tuple(
            float(value)
            for value in best.get("first_execution_requested_target_kg", ())
        )
        plan = build_candidate_plan(
            str(action),
            self._target_operation_for_action(action),
            action_vector=np.asarray(avec, dtype=float).reshape(-1)[:2],
            target_masses_kg=evaluated_target,
        )
        raw_score = best.get("best_scalar_cost")
        score = None
        if raw_score is not None:
            value = float(raw_score)
            if np.isfinite(value):
                score = value
        safety_passed = bool(
            not best.get("hard_reject_reason")
            and not best.get("planned_reactive_handoff", 0)
        )
        return record_evaluation(
            plan,
            str(action),
            action_vector=np.asarray(avec, dtype=float).reshape(-1)[:2],
            target_masses_kg=evaluated_target,
            safety_passed=safety_passed,
            score=score,
            summary=str(best.get("selection_reason", "ranked_candidate")),
        )

    def _finalize_supervised_action(
        self,
        *,
        plan,
        best,
        action: str,
        avec,
    ):
        """Re-evaluate the exact action/vector left by supervision."""

        vector = np.asarray(avec, dtype=float).reshape(-1)[:2]
        if bool(best.get("planned_reactive_handoff", 0)):
            release_action = self._release_action_name()
            self._action_plan_snapshot = {
                "status": "reactive_handoff",
                "candidate_action": plan.candidate_action,
                "final_action": release_action,
                "correction_reason": "all_candidates_failed_safety",
                "safety_passed": 0,
                "hard_reject_reason": str(best.get("hard_reject_reason", "")),
            }
            return release_action, np.zeros(2, dtype=float), None, True

        operation = self._target_operation_for_action(action)
        corrected = apply_action_correction(
            plan,
            str(action),
            target_operation=operation,
            action_vector=vector,
            reason="post_ranking_supervision",
        )
        final_row = None
        if corrected.needs_reevaluation:
            if str(action) not in candidate_actions(self._last_planner_cfg):
                final_row = {"hard_reject_reason": "unknown_supervised_action"}
            else:
                original_sequence = tuple(best.get("sequence", (plan.candidate_action,)))
                final_sequence = (str(action),) + tuple(original_sequence[1:])
                final_row = evaluate_sequence(
                    final_sequence,
                    self._last_planner_blocks,
                    self._last_planner_info,
                    self._last_planner_cfg,
                    action_vec_overrides={0: vector},
                )

            if final_row.get("hard_reject_reason"):
                snapshot = commit_action_plan(plan)
                snapshot["status"] = "correction_rejected"
                snapshot["rejected_action"] = str(action)
                snapshot["rejected_action_vector"] = vector.tolist()
                snapshot["rejected_reason"] = str(
                    final_row.get("hard_reject_reason", "")
                )
                self._action_plan_snapshot = snapshot
                return (
                    plan.candidate_action,
                    np.asarray(plan.candidate_action_vector, dtype=float),
                    None,
                    False,
                )

            score = None
            if self.objective_mode == "economic":
                score = float(economic_scalar_cost(final_row, self._last_planner_cfg))
            evaluated_target = tuple(
                float(value)
                for value in final_row.get(
                    "first_execution_requested_target_kg",
                    (),
                )
            )
            corrected = apply_action_correction(
                corrected,
                str(action),
                target_operation=operation,
                action_vector=vector,
                target_masses_kg=evaluated_target,
                reason="post_ranking_supervision_reevaluated",
            )
            corrected = record_evaluation(
                corrected,
                str(action),
                action_vector=vector,
                target_masses_kg=evaluated_target,
                safety_passed=True,
                score=score,
                summary="post_ranking_supervision_reevaluated",
            )

        snapshot = commit_action_plan(corrected)
        snapshot["status"] = (
            "reevaluated" if final_row is not None else "candidate_unchanged"
        )
        snapshot["hard_reject_reason"] = ""
        self._action_plan_snapshot = snapshot
        return str(action), vector, final_row, False

    def _reset_bucket_transients(self, bucket: int) -> None:
        self._last_bucket = bucket
        self._decision_trace_steps = ()
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
        self._action_plan_snapshot = {"status": "not_evaluated"}

    def _handle_missing_forecast_sample(
        self,
        current_time: float,
        plant_info_prev,
    ) -> None:
        if self._target_action != "hold":
            self._transition_time_s = float(current_time)
        self._target_action = "hold"
        self._target_pitch = 0.0
        self._target_roll = 0.0
        self._target_pitch_planner_frame = 0.0
        self._target_roll_planner_frame = 0.0
        plant_info = self._plant_info_from(plant_info_prev)
        if self.forecast_advised_economy_enabled:
            self._primary_refresh_owner_pending = "missing_sample_fail_closed"
            self._release_primary_target_to_pi(plant_info, float(current_time))
            self._forecast_advised_economy_candidate = False
            self._forecast_advised_economy_boundary_veto = True
            self._forecast_advised_economy_headroom_deg = 0.0
            self._forecast_advised_economy_near_rise_norm = 0.0
            self._forecast_advised_economy_posture_release = False
            self._forecast_advised_economy_max_axis_deg = 0.0
            self._forecast_advised_economy_reason = "missing_sample_fail_closed"
        else:
            self._primary_refresh_owner_pending = "missing_sample_zero_target"
            self._update_primary_target(
                plant_info,
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

    def _apply_hold_target_policy(
        self,
        *,
        state,
        plant_info,
        current_time: float,
        previous_action: str,
        previous_avec,
    ) -> None:
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

    def _apply_event_target_refresh(
        self,
        *,
        plant_info,
        target_update_avec,
        current_time: float,
        event_reset: bool,
        sustained_recompute: bool,
    ) -> None:
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

    def _apply_reuse_target_refresh(
        self,
        *,
        plant_info,
        target_update_avec,
        new_action: str,
        current_time: float,
    ) -> None:
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

    def _commit_planned_target(
        self,
        *,
        bucket: int,
        current_time: float,
        sample,
        state,
        plant_info,
        blocks,
        previous_action: str,
        previous_avec,
        new_action: str,
        first_avec,
        event_reset: bool,
        direction_reversal_entry_veto: bool,
        direction_reversal_warning_active: bool,
        planned_reactive_handoff: bool,
    ) -> None:
        if str(self.target_lifecycle_mode) == "planner_authoritative":
            self._commit_planner_authoritative_target(
                current_time=float(current_time),
                plant_info=plant_info,
                new_action=str(new_action),
                planned_reactive_handoff=bool(planned_reactive_handoff),
            )
            return

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
        elif planned_reactive_handoff:
            self._primary_refresh_owner_pending = "planner_safety_handoff_to_feedback"
            self._release_primary_target_to_pi(plant_info, float(current_time))
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
            self._apply_hold_target_policy(
                state=state,
                plant_info=plant_info,
                current_time=float(current_time),
                previous_action=previous_action,
                previous_avec=previous_avec,
            )
        elif (
            previous_action == "hold"
            and not event_reset
            and self._resume_paused_primary_target(
            plant_info,
            target_update_avec,
            float(current_time),
            )
        ):
            pass
        elif (
            event_reset
            or not self._primary_target_initialized
            or sustained_recompute
        ):
            self._apply_event_target_refresh(
                plant_info=plant_info,
                target_update_avec=target_update_avec,
                current_time=float(current_time),
                event_reset=bool(event_reset),
                sustained_recompute=bool(sustained_recompute),
            )
        elif direction_reversal_warning_active:
            self._primary_refresh_owner_pending = "direction_reversal_v7_warning_pi_release"
            self._release_primary_target_to_pi(plant_info, float(current_time))
        else:
            self._apply_reuse_target_refresh(
                plant_info=plant_info,
                target_update_avec=target_update_avec,
                new_action=new_action,
                current_time=float(current_time),
            )
        self._relief_envelope_mark_not_applicable()
        self._hold_relief_debt_observe_after_action(new_action, plant_info)

    def _commit_planner_authoritative_target(
        self,
        *,
        current_time: float,
        plant_info,
        new_action: str,
        planned_reactive_handoff: bool,
    ) -> None:
        """Commit exactly the target that survived planner safety review."""

        current_masses = plant_primary_masses(
            plant_info,
            self._primary_anchor_masses_kg,
        )
        if planned_reactive_handoff:
            operation = TargetOperation.RELEASE.value
            requested_target = None
            owner = "planner_safety_handoff_to_feedback"
        else:
            operation = str(
                self._action_plan_snapshot.get(
                    "target_operation",
                    self._target_operation_for_action(new_action).value,
                )
            )
            requested_values = self._action_plan_snapshot.get(
                "final_target_masses_kg",
                (),
            )
            requested_target = (
                np.asarray(requested_values, dtype=float).reshape(-1)[:3]
                if requested_values
                else None
            )
            owner = f"planner_authoritative:{new_action}"
        if operation == TargetOperation.CONTINUE.value:
            if requested_target is None:
                raise ValueError(
                    "planner-authoritative continue requires the evaluated "
                    "existing target"
                )
            current_target = np.asarray(
                self._primary_target_kg,
                dtype=float,
            ).reshape(3)
            if not np.allclose(
                requested_target,
                current_target,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(
                    "planner-authoritative continue target does not match "
                    "the provider target state"
                )
            mark_primary_target_reused(self, refresh_owner=owner)
        else:
            apply_planner_authoritative_target(
                self,
                operation=operation,
                current_masses_kg=current_masses,
                requested_target_kg=requested_target,
                tank_capacity_kg=float(self.cfg.tank_capacity_kg),
                current_time_s=float(current_time),
                refresh_owner=owner,
            )
        self._relief_envelope_reset_bucket_log()
        self._relief_envelope_mark_not_applicable()
        self._hold_relief_debt_observe_after_action(new_action, plant_info)

    def _update_planner_bucket(
        self,
        *,
        bucket: int,
        current_time: float,
        state,
        plant_info_now,
        plant_info_prev,
        replanned: bool,
    ) -> None:
        self._reset_bucket_transients(bucket)
        ts = self._bucket_timestamp(current_time)
        sample = self.replay_dataset.sample_for_history_end(ts)
        if sample is None:
            self._handle_missing_forecast_sample(current_time, plant_info_prev)
            return
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
        ranked_first_avec = np.asarray(first_avec, dtype=float).reshape(-1)[:2].copy()
        ranked_action_plan = self._build_ranked_action_plan(
            best,
            original_first_action,
            ranked_first_avec,
        )
        best, pipeline_raw_action, new_action, first_avec = (
            self._apply_traced_action_supervision(
                best=best,
                first_avec=first_avec,
                bucket=bucket,
                blocks=blocks,
                plant_info=plant_info,
                state=state,
                current_time=float(current_time),
            )
        )
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
        (
            new_action,
            first_avec,
            event_reset,
            posture_action_refresh,
            direction_reversal_entry_veto,
            direction_reversal_entry_veto_reason,
        ) = self._apply_traced_vector_supervision(
            action=new_action,
            action_vec=first_avec,
            plant_info=plant_info,
            blocks=blocks,
            current_time=float(current_time),
            event_reset=bool(event_reset),
            posture_action_refresh=bool(posture_action_refresh),
        )
        new_action, first_avec, supervised_row, reactive_handoff = (
            self._finalize_supervised_action(
                plan=ranked_action_plan,
                best=best,
                action=new_action,
                avec=first_avec,
            )
        )
        planner_pitch = float(first_avec[0])
        planner_roll = float(first_avec[1])
        new_pitch = planner_pitch * self.setpoint_bias_sign
        new_roll = planner_roll * self.setpoint_bias_sign
        event_reset = bool(
            not reactive_handoff
            and self._should_reset_event(
                new_action,
                new_pitch,
                new_roll,
                float(current_time),
            )
        )
        best["supervised_action"] = str(new_action)
        best["supervised_action_reevaluated"] = int(supervised_row is not None)
        best["supervised_reactive_handoff"] = int(reactive_handoff)
        if supervised_row is not None:
            best["supervised_hard_reject_reason"] = str(
                supervised_row.get("hard_reject_reason", "")
            )
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
        self._commit_planned_target(
            bucket=bucket,
            current_time=float(current_time),
            sample=sample,
            state=state,
            plant_info=plant_info,
            blocks=blocks,
            previous_action=previous_action,
            previous_avec=previous_avec,
            new_action=new_action,
            first_avec=first_avec,
            event_reset=bool(event_reset),
            direction_reversal_entry_veto=bool(direction_reversal_entry_veto),
            direction_reversal_warning_active=bool(direction_reversal_warning_active),
            planned_reactive_handoff=bool(reactive_handoff),
        )
        self._append_replan_record(
            best=best,
            bucket=bucket,
            current_time=float(current_time),
            event_reset=bool(event_reset),
            forecast_has_future=bool(forecast_has_future),
            forecast_source=str(forecast_source),
            new_action=str(new_action),
            new_pitch=float(new_pitch),
            new_roll=float(new_roll),
            pipeline_raw_action=str(pipeline_raw_action),
            planner_pitch=float(planner_pitch),
            planner_roll=float(planner_roll),
            plant_info=plant_info,
            posture_action_refresh=bool(posture_action_refresh),
            ts=ts,
        )

    """Internal behavior group for ``BallastPlannerPreviewProvider``."""

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
            self._update_planner_bucket(
                bucket=bucket,
                current_time=float(current_time),
                state=state,
                plant_info_now=plant_info_now,
                plant_info_prev=plant_info_prev,
                replanned=replanned,
            )

        if (
            self.target_lifecycle_mode == "legacy"
            and self.forecast_advised_economy_enabled
            and self.forecast_advised_economy_suppression_only
            and self._primary_target_initialized
        ):
            self._primary_refresh_owner_pending = "forecast_advised_suppression_only_pi_release"
            self._release_primary_target_to_pi(
                self._plant_info_from(plant_info_prev),
                float(current_time),
            )

        forecast_advised_fail_closed = bool(
            self.target_lifecycle_mode == "legacy"
            and self.forecast_advised_economy_enabled
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
        return self._build_runtime_output(
            current_time=float(current_time),
            pitch_bias=float(pitch_bias),
            roll_bias=float(roll_bias),
            mass_ff_kg=mass_ff_kg,
            primary_channel_enabled=bool(primary_channel_enabled),
            primary_active=int(primary_active),
        )
