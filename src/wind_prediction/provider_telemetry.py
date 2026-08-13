"""Per-cycle provider output and diagnostic telemetry assembly."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping

import numpy as np

from .control_telemetry_defaults import (
    frozen_active_intent_record_defaults,
    frozen_hold_comfort_preview_defaults,
    frozen_hold_comfort_record_defaults,
    frozen_overlay_preview_defaults,
    frozen_overlay_record_defaults,
    frozen_recovery_mode_record_defaults,
)


class ProviderTelemetryMixin:
    """Build the external cycle result without changing control state."""

    _DECISION_RECORD_SCHEMA_VERSION = "v2_decision_record_v2"

    @classmethod
    def _json_safe_value(cls, value):
        """Return strict-JSON data while preserving missing diagnostics as null."""

        if isinstance(value, np.ndarray):
            return [cls._json_safe_value(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return cls._json_safe_value(value.item())
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._json_safe_value(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    @classmethod
    def _finite_float_or_none(cls, value):
        try:
            normalized = float(value)
        except (TypeError, ValueError):
            return None
        return normalized if math.isfinite(normalized) else None

    @classmethod
    def _three_vector_or_none(cls, value):
        try:
            array = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return None
        if array.size != 3 or not np.all(np.isfinite(array)):
            return None
        return [float(item) for item in array]

    def _forecast_decision_record(self) -> dict[str, object]:
        evidence = getattr(self, "_forecast_evidence", None)
        if evidence is None:
            return {
                "status": "missing",
                "source": "",
                "model_version": "",
                "origin_time": None,
                "provides_future_preview": None,
                "sample_period_s": None,
                "points": [],
                "event_probabilities": {},
                "metadata": {},
            }

        try:
            uv = np.asarray(getattr(evidence, "uv_ms", ()), dtype=float)
        except (TypeError, ValueError):
            uv = np.empty((0, 2), dtype=float)
        try:
            reliability = np.asarray(
                getattr(evidence, "lead_reliability", ()),
                dtype=float,
            ).reshape(-1)
        except (TypeError, ValueError):
            reliability = np.empty(0, dtype=float)
        sample_period_s = self._finite_float_or_none(
            getattr(evidence, "sample_period_s", None)
        )
        points = []
        if uv.ndim == 2 and uv.shape[1] == 2:
            for index, vector in enumerate(uv):
                lead_reliability = (
                    self._finite_float_or_none(reliability[index])
                    if index < reliability.size
                    else None
                )
                points.append({
                    "lead_index": index,
                    "lead_time_s": (
                        float((index + 1) * sample_period_s)
                        if sample_period_s is not None
                        else None
                    ),
                    "u_ms": self._finite_float_or_none(vector[0]),
                    "v_ms": self._finite_float_or_none(vector[1]),
                    "reliability": lead_reliability,
                })
        return {
            "status": "available" if points else "invalid_or_empty",
            "source": str(getattr(evidence, "source", "")),
            "model_version": str(getattr(evidence, "model_version", "")),
            "origin_time": getattr(evidence, "origin_time", None),
            "provides_future_preview": bool(
                getattr(evidence, "provides_future_preview", False)
            ),
            "sample_period_s": sample_period_s,
            "points": points,
            "event_probabilities": self._json_safe_value(
                dict(getattr(evidence, "event_probs", {}) or {})
            ),
            "metadata": self._json_safe_value(
                dict(getattr(evidence, "metadata", {}) or {})
            ),
        }

    def _forecast_action_policy_decision_record(self, best) -> dict[str, object]:
        """Serialize optional action-authorization policy output.

        The policy producer is deliberately outside this telemetry mixin. Both
        shadow and enforce modes use the same snapshot shape so changing policy
        authority does not change the audit schema.
        """

        best_row = dict(best or {})
        candidate_filter = self._forecast_action_policy_filter_decision_record(
            best_row
        )
        snapshot = getattr(self, "_forecast_action_policy_snapshot", None)
        if snapshot is None:
            snapshot = best_row.get("forecast_action_policy")
        if snapshot is None:
            return {
                "status": "missing",
                "mode": None,
                "stage_labels": None,
                "authorized_actions": None,
                "blocked_actions": None,
                "threshold_source": None,
                "reasons": None,
                "candidate_filter": candidate_filter,
            }
        if not isinstance(snapshot, Mapping):
            return {
                "status": "invalid",
                "mode": None,
                "stage_labels": None,
                "authorized_actions": None,
                "blocked_actions": None,
                "threshold_source": None,
                "reasons": None,
                "candidate_filter": candidate_filter,
            }

        def optional_value(name: str):
            if name not in snapshot or snapshot[name] is None:
                return None
            return self._json_safe_value(snapshot[name])

        mode = optional_value("mode")
        if mode is not None:
            mode = str(mode)
        return {
            "status": "available",
            "mode": mode,
            "stage_labels": optional_value("stage_labels"),
            "authorized_actions": optional_value("authorized_actions"),
            "blocked_actions": optional_value("blocked_actions"),
            "threshold_source": optional_value("threshold_source"),
            "reasons": optional_value("reasons"),
            "candidate_filter": candidate_filter,
        }

    def _forecast_action_policy_filter_decision_record(
        self,
        best_row: Mapping[str, object],
    ) -> dict[str, object]:
        """Serialize the effective first-action filter boundary."""

        enforcement = getattr(self, "_forecast_action_policy_enforcement", None)
        if enforcement is None:
            enforcement = best_row.get("forecast_action_policy_enforcement")
        missing = {
            "status": "missing",
            "applied": None,
            "reason": None,
            "candidate_count_before": None,
            "candidate_count_after": None,
            "blocked_first_actions": None,
            "posture_strengthen_restored": None,
        }
        if enforcement is None:
            return missing

        def read(name: str):
            if isinstance(enforcement, Mapping):
                return enforcement.get(name)
            return getattr(enforcement, name, None)

        applied = read("applied")
        reason = read("reason")
        count_before = read("candidate_count_before")
        count_after = read("candidate_count_after")
        blocked = read("blocked_first_actions")
        posture_strengthen_restored = read("posture_strengthen_restored")
        if posture_strengthen_restored is None:
            posture_strengthen_restored = False
        try:
            count_before = int(count_before)
            count_after = int(count_after)
        except (TypeError, ValueError):
            return {**missing, "status": "invalid"}
        if (
            not isinstance(applied, (bool, np.bool_))
            or reason is None
            or count_before < 0
            or count_after < 0
            or isinstance(blocked, (str, bytes))
        ):
            return {**missing, "status": "invalid"}
        try:
            blocked_actions = [str(action) for action in blocked]
        except TypeError:
            return {**missing, "status": "invalid"}
        return {
            "status": "available",
            "applied": bool(applied),
            "reason": str(reason),
            "candidate_count_before": count_before,
            "candidate_count_after": count_after,
            "blocked_first_actions": blocked_actions,
            "posture_strengthen_restored": bool(posture_strengthen_restored),
        }

    def _candidate_decision_record(self, best) -> dict[str, object]:
        best_row = dict(best or {})
        snapshot = getattr(self, "_candidate_evaluations_snapshot", None)
        if snapshot is None:
            snapshot = best_row.get("candidate_evaluations")
        if snapshot is None:
            source_rows = [best_row] if best_row else []
            coverage = "selected_only" if source_rows else "missing"
        else:
            try:
                source_rows = [dict(row or {}) for row in snapshot]
            except (TypeError, ValueError):
                source_rows = []
            coverage = "full" if source_rows else "missing"

        best_sequence = tuple(best_row.get("sequence", ()))
        rows = []
        rejection_reasons = Counter()
        for row in source_rows:
            reason = str(row.get("hard_reject_reason", "") or "")
            if reason:
                rejection_reasons[reason] += 1
            sequence = tuple(row.get("sequence", ()))
            first_requested_target = self._three_vector_or_none(
                row.get("first_execution_requested_target_kg")
            )
            first_shaped_target = self._three_vector_or_none(
                row.get("first_execution_shaped_target_kg")
            )
            rows.append({
                "sequence": list(sequence),
                "selected": bool(sequence and sequence == best_sequence),
                "feasible": not bool(reason),
                "rejection_status": "rejected" if reason else "accepted",
                "hard_reject_reason": reason or None,
                "selection_reason": (
                    str(row["selection_reason"])
                    if row.get("selection_reason") not in (None, "")
                    else None
                ),
                "first_stage": {
                    "action": str(sequence[0]) if sequence else None,
                    "target_status": (
                        "available"
                        if first_requested_target is not None
                        or first_shaped_target is not None
                        else "missing"
                    ),
                    "requested_target_masses_kg": first_requested_target,
                    "shaped_target_masses_kg": first_shaped_target,
                },
                "scalar_cost": self._finite_float_or_none(
                    row.get("best_scalar_cost", row.get("scalar_cost"))
                ),
                "costs": self._json_safe_value(dict(row.get("costs", {}) or {})),
            })

        # Older planner outputs expose only the winning row and an aggregate
        # safety-floor count. Retain that information without claiming that the
        # complete rejection distribution was logged.
        safety_reject_count = int(best_row.get("safety_floor_reject_count", 0) or 0)
        if coverage != "full" and safety_reject_count > 0:
            rejection_reasons["safety_floor_violated"] += safety_reject_count

        second_sequence = best_row.get("second_best_sequence")
        best_cost = self._finite_float_or_none(best_row.get("best_scalar_cost"))
        second_cost = self._finite_float_or_none(best_row.get("second_best_scalar_cost"))
        absolute_gap = (
            float(second_cost - best_cost)
            if best_cost is not None and second_cost is not None
            else None
        )
        relative_gap_pct = (
            float(absolute_gap / max(abs(best_cost), 1e-12) * 100.0)
            if absolute_gap is not None and best_cost is not None
            else None
        )
        total_evaluated = best_row.get("safety_floor_total_sequences")
        if coverage == "full":
            total_evaluated = len(rows)
        elif total_evaluated is None:
            total_evaluated = len(rows) if rows else None
        rejected_count = (
            sum(row["rejection_status"] == "rejected" for row in rows)
            if coverage == "full"
            else None
        )
        selected_row = next(
            (row for row in rows if row["selected"]),
            rows[0] if coverage == "selected_only" and rows else None,
        )
        return {
            "coverage": coverage,
            "rows": rows,
            "rows_logged": len(rows),
            "total_evaluated": total_evaluated,
            "rejected_count": rejected_count,
            "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
            "selected_first_stage": (
                dict(selected_row["first_stage"])
                if selected_row is not None
                else {
                    "action": None,
                    "target_status": "missing",
                    "requested_target_masses_kg": None,
                    "shaped_target_masses_kg": None,
                }
            ),
            "best_sequence": list(best_sequence),
            "second_best_sequence": (
                list(second_sequence)
                if isinstance(second_sequence, (list, tuple))
                else None
            ),
            "best_scalar_cost": best_cost,
            "second_best_scalar_cost": second_cost,
            "best_second_gap": {
                "status": "available" if absolute_gap is not None else "missing",
                "absolute": absolute_gap,
                "relative_pct": relative_gap_pct,
            },
        }

    def _posture_action_authority_decision_record(self) -> dict[str, object]:
        """Serialize the completed-target posture-action boundary."""

        enforcement = getattr(self, "_posture_action_enforcement", None)
        missing = {
            "status": "missing",
            "applied": None,
            "reason": None,
            "candidate_count_before": None,
            "candidate_count_after": None,
            "blocked_first_actions": None,
            "required_axes": None,
        }
        if enforcement is None:
            return missing

        def read(name: str):
            if isinstance(enforcement, Mapping):
                return enforcement.get(name)
            return getattr(enforcement, name, None)

        try:
            count_before = int(read("candidate_count_before"))
            count_after = int(read("candidate_count_after"))
            blocked = [str(value) for value in read("blocked_first_actions")]
            required_axes = [bool(value) for value in read("required_axes")]
        except (TypeError, ValueError):
            return {**missing, "status": "invalid"}
        applied = read("applied")
        reason = read("reason")
        if (
            not isinstance(applied, (bool, np.bool_))
            or reason is None
            or count_before < 0
            or count_after < 0
            or len(required_axes) != 2
        ):
            return {**missing, "status": "invalid"}
        return {
            "status": "available",
            "applied": bool(applied),
            "reason": str(reason),
            "candidate_count_before": count_before,
            "candidate_count_after": count_after,
            "blocked_first_actions": blocked,
            "required_axes": required_axes,
        }

    def _target_execution_decision_record(
        self,
        *,
        action_plan: Mapping[str, object],
        best,
        plant_info,
        planner_pitch: float,
        planner_roll: float,
    ) -> dict[str, object]:
        plan = dict(action_plan or {})
        evaluation = dict(plan.get("evaluation", {}) or {})
        info = dict(plant_info or {})
        current_masses = self._three_vector_or_none(info.get("tank_masses"))
        observed_execution_target = self._three_vector_or_none(
            info.get("target_ballast_mass")
        )
        committed_target = self._three_vector_or_none(
            getattr(self, "_primary_target_kg", None)
        )
        candidate_target = self._three_vector_or_none(
            plan.get("candidate_target_masses_kg")
        )
        final_target = self._three_vector_or_none(plan.get("final_target_masses_kg"))

        def debt(target):
            if target is None or current_masses is None:
                return {
                    "status": "missing",
                    "per_tank_kg": None,
                    "absolute_total_kg": None,
                    "absolute_mean_kg": None,
                    "maximum_absolute_kg": None,
                }
            values = np.asarray(target, dtype=float) - np.asarray(
                current_masses,
                dtype=float,
            )
            absolute = np.abs(values)
            return {
                "status": "available",
                "per_tank_kg": values.tolist(),
                "absolute_total_kg": float(np.sum(absolute)),
                "absolute_mean_kg": float(np.mean(absolute)),
                "maximum_absolute_kg": float(np.max(absolute)),
            }

        safety_passed = evaluation.get("safety_passed")
        best_sequence = tuple((best or {}).get("sequence", ()))
        target_initialized = bool(
            getattr(
                self,
                "_primary_target_initialized",
                committed_target is not None,
            )
        )
        best_row = dict(best or {})
        replan_mode = best_row.get("target_replan_policy_mode")
        target_replan = {
            "status": "available" if replan_mode is not None else "missing",
            "mode": str(replan_mode) if replan_mode is not None else None,
            "required": (
                bool(best_row.get("target_replan_required"))
                if replan_mode is not None
                else None
            ),
            "target_completed": (
                bool(best_row.get("target_replan_target_completed"))
                if replan_mode is not None
                else None
            ),
            "outside_envelope_axes": (
                [
                    bool(best_row.get("target_replan_outside_pitch")),
                    bool(best_row.get("target_replan_outside_roll")),
                ]
                if replan_mode is not None
                else None
            ),
            "recovering_axes": (
                [
                    bool(best_row.get("target_replan_recovering_pitch")),
                    bool(best_row.get("target_replan_recovering_roll")),
                ]
                if replan_mode is not None
                else None
            ),
            "remaining_target_direction_helpful": (
                bool(best_row.get("target_replan_target_direction_helpful"))
                if replan_mode is not None
                else None
            ),
            "action_redirected": (
                bool(best_row.get("target_replan_action_redirected"))
                if replan_mode is not None
                else None
            ),
            "action_corrective": (
                bool(best_row.get("target_replan_action_corrective"))
                if replan_mode is not None
                else None
            ),
            "reason": (
                str(best_row.get("target_replan_reason", ""))
                if replan_mode is not None
                else None
            ),
        }
        return {
            "planner": {
                "action": str(best_sequence[0]) if best_sequence else "",
                "action_vector_deg": [float(planner_pitch), float(planner_roll)],
                "target_operation": str(plan.get("target_operation", "")),
                "target_masses_kg": candidate_target,
            },
            "safety_review": {
                "status": str(plan.get("status", "not_evaluated")),
                "candidate_action": str(plan.get("candidate_action", "")),
                "approved_action": str(plan.get("final_action", "")),
                "safety_passed": (
                    bool(safety_passed) if safety_passed is not None else None
                ),
                "correction_reason": str(plan.get("correction_reason", "")),
                "rejected_reason": str(plan.get("rejected_reason", "")),
            },
            "target_commit": {
                "status": (
                    "missing"
                    if committed_target is None
                    else "available"
                    if target_initialized
                    else "not_initialized"
                ),
                "operation": str(plan.get("target_operation", "")),
                "approved_target_masses_kg": final_target,
                "committed_target_masses_kg": committed_target,
                "refresh_owner": str(
                    getattr(self, "_primary_refresh_owner", "")
                ),
            },
            "target_replan": target_replan,
            "execution_observation": {
                "timing": "plant_step_before_current_decision",
                "current_masses_kg": current_masses,
                "execution_target_masses_kg": observed_execution_target,
                "actual_mass_delta_kg": self._three_vector_or_none(
                    info.get("tank_mass_delta_kg", info.get("pump_delta"))
                ),
                "actual_rate_m3_min": self._three_vector_or_none(
                    info.get("pump_net_rate_m3_min", info.get("pump_rate_cmd_m3_min"))
                ),
                "backlog_total_kg": self._finite_float_or_none(
                    info.get("pump_total_backlog_kg")
                ),
            },
            "target_debt": {
                "basis": "committed_target_minus_current_masses",
                **debt(committed_target),
                "observed_execution_target": debt(observed_execution_target),
            },
        }

    def _v2_decision_record_log_fields(
        self,
        *,
        best=None,
        plant_info=None,
        planner_pitch: float = 0.0,
        planner_roll: float = 0.0,
        prefix: str = "",
    ) -> dict[str, object]:
        """Serialize one attribution record without changing provider state."""

        action_plan = dict(getattr(self, "_action_plan_snapshot", {}) or {})
        forecast = self._forecast_decision_record()
        forecast_action_policy = self._forecast_action_policy_decision_record(best)
        forecast_action_filter = forecast_action_policy["candidate_filter"]
        posture_action_authority = (
            self._posture_action_authority_decision_record()
        )
        candidates = self._candidate_decision_record(best)
        target_execution = self._target_execution_decision_record(
            action_plan=action_plan,
            best=best,
            plant_info=plant_info,
            planner_pitch=float(planner_pitch),
            planner_roll=float(planner_roll),
        )
        record = {
            "schema_version": self._DECISION_RECORD_SCHEMA_VERSION,
            "forecast": forecast,
            "forecast_action_policy": forecast_action_policy,
            "posture_action_authority": posture_action_authority,
            "candidates": candidates,
            **target_execution,
            "post_ranking_trace": json.loads(
                self._decision_trace_log_fields()["decision_trace_json"]
            ),
        }
        debt_total = target_execution["target_debt"]["absolute_total_kg"]
        target_replan = target_execution["target_replan"]
        gap = candidates["best_second_gap"]

        def compact_optional(value):
            if value is None:
                return None
            return json.dumps(
                self._json_safe_value(value),
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )

        return {
            f"{prefix}decision_record_schema_version": (
                self._DECISION_RECORD_SCHEMA_VERSION
            ),
            f"{prefix}decision_record_json": json.dumps(
                self._json_safe_value(record),
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            f"{prefix}decision_record_forecast_point_count": len(
                forecast["points"]
            ),
            f"{prefix}decision_record_forecast_action_policy_status": (
                forecast_action_policy["status"]
            ),
            f"{prefix}decision_record_forecast_action_policy_mode": (
                forecast_action_policy["mode"]
            ),
            f"{prefix}decision_record_forecast_action_policy_stage_labels_json": (
                compact_optional(forecast_action_policy["stage_labels"])
            ),
            f"{prefix}decision_record_forecast_action_policy_authorized_actions_json": (
                compact_optional(forecast_action_policy["authorized_actions"])
            ),
            f"{prefix}decision_record_forecast_action_policy_blocked_actions_json": (
                compact_optional(forecast_action_policy["blocked_actions"])
            ),
            f"{prefix}decision_record_forecast_action_policy_threshold_source_json": (
                compact_optional(forecast_action_policy["threshold_source"])
            ),
            f"{prefix}decision_record_forecast_action_policy_reasons_json": (
                compact_optional(forecast_action_policy["reasons"])
            ),
            f"{prefix}decision_record_forecast_action_policy_filter_status": (
                forecast_action_filter["status"]
            ),
            f"{prefix}decision_record_forecast_action_policy_filter_applied": (
                forecast_action_filter["applied"]
            ),
            f"{prefix}decision_record_forecast_action_policy_filter_reason": (
                forecast_action_filter["reason"]
            ),
            f"{prefix}decision_record_forecast_action_policy_candidate_count_before": (
                forecast_action_filter["candidate_count_before"]
            ),
            f"{prefix}decision_record_forecast_action_policy_candidate_count_after": (
                forecast_action_filter["candidate_count_after"]
            ),
            f"{prefix}decision_record_forecast_action_policy_blocked_first_actions_json": (
                compact_optional(forecast_action_filter["blocked_first_actions"])
            ),
            f"{prefix}decision_record_forecast_action_policy_posture_strengthen_restored": (
                forecast_action_filter["posture_strengthen_restored"]
            ),
            f"{prefix}decision_record_posture_action_authority_status": (
                posture_action_authority["status"]
            ),
            f"{prefix}decision_record_posture_action_authority_applied": (
                posture_action_authority["applied"]
            ),
            f"{prefix}decision_record_posture_action_authority_reason": (
                posture_action_authority["reason"]
            ),
            f"{prefix}decision_record_posture_action_authority_blocked_first_actions_json": (
                compact_optional(posture_action_authority["blocked_first_actions"])
            ),
            f"{prefix}decision_record_posture_action_authority_required_axes_json": (
                compact_optional(posture_action_authority["required_axes"])
            ),
            f"{prefix}decision_record_candidate_coverage": candidates["coverage"],
            f"{prefix}decision_record_candidate_rows_logged": candidates[
                "rows_logged"
            ],
            f"{prefix}decision_record_best_second_gap": gap["absolute"],
            f"{prefix}decision_record_target_debt_abs_total_kg": debt_total,
            f"{prefix}decision_record_target_replan_status": target_replan[
                "status"
            ],
            f"{prefix}decision_record_target_replan_mode": target_replan["mode"],
            f"{prefix}decision_record_target_replan_required": target_replan[
                "required"
            ],
            f"{prefix}decision_record_target_replan_action_redirected": (
                target_replan["action_redirected"]
            ),
            f"{prefix}decision_record_target_replan_action_corrective": (
                target_replan["action_corrective"]
            ),
            f"{prefix}decision_record_target_replan_reason": target_replan[
                "reason"
            ],
        }

    def _decision_trace_log_fields(self, prefix: str = "") -> dict[str, object]:
        """Serialize the current post-ranking trace without mutating it."""

        trace = tuple(getattr(self, "_decision_trace_steps", ()))
        payload = []
        for step in trace:
            entry = {
                "name": str(step.name),
                "input_action": str(step.input_action),
                "output_action": str(step.output_action),
                "changed": int(bool(step.changed)),
                "reason": str(step.reason),
            }
            metadata = dict(step.metadata or {})
            if metadata:
                entry["metadata"] = metadata
            payload.append(entry)
        return {
            f"{prefix}decision_trace_json": json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
            f"{prefix}decision_trace_modification_count": int(
                sum(bool(step.changed) for step in trace)
            ),
        }

    def _append_replan_record(
        self,
        *,
        best,
        bucket: int,
        current_time: float,
        event_reset: bool,
        forecast_has_future: bool,
        forecast_source: str,
        new_action: str,
        new_pitch: float,
        new_roll: float,
        pipeline_raw_action: str,
        planner_pitch: float,
        planner_roll: float,
        plant_info,
        posture_action_refresh: bool,
        ts,
    ) -> None:
        action_plan = dict(getattr(self, "_action_plan_snapshot", {}) or {})
        action_evaluation = dict(action_plan.get("evaluation", {}) or {})
        self.records.append({
            "bucket": bucket,
            "current_time_s": float(current_time),
            "history_end": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "first_action": new_action,
            "planner_first_action_raw": pipeline_raw_action,
            "candidate_action": str(
                action_plan.get("candidate_action", pipeline_raw_action)
            ),
            "supervised_action": str(
                action_plan.get("final_action", new_action)
            ),
            "action_plan_status": str(
                action_plan.get("status", "not_evaluated")
            ),
            "action_correction_reason": str(
                action_plan.get("correction_reason", "")
            ),
            "action_correction_rejected_reason": str(
                action_plan.get("rejected_reason", "")
            ),
            "supervised_action_safety_passed": int(
                bool(action_evaluation.get("safety_passed", False))
            ),
            **self._decision_trace_log_fields(),
            **self._v2_decision_record_log_fields(
                best=best,
                plant_info=plant_info,
                planner_pitch=planner_pitch,
                planner_roll=planner_roll,
            ),
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
            "target_replan_policy_mode": str(
                best.get("target_replan_policy_mode", "off")
            ),
            "target_replan_required": int(
                best.get("target_replan_required", 0)
            ),
            "target_replan_target_completed": int(
                best.get("target_replan_target_completed", 0)
            ),
            "target_replan_outside_pitch": int(
                best.get("target_replan_outside_pitch", 0)
            ),
            "target_replan_outside_roll": int(
                best.get("target_replan_outside_roll", 0)
            ),
            "target_replan_recovering_pitch": int(
                best.get("target_replan_recovering_pitch", 0)
            ),
            "target_replan_recovering_roll": int(
                best.get("target_replan_recovering_roll", 0)
            ),
            "target_replan_target_direction_helpful": int(
                best.get("target_replan_target_direction_helpful", 0)
            ),
            "target_replan_action_redirected": int(
                best.get("target_replan_action_redirected", 0)
            ),
            "target_replan_action_corrective": int(
                best.get("target_replan_action_corrective", 0)
            ),
            "target_replan_reason": str(best.get("target_replan_reason", "")),
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


    def _build_runtime_output(
        self,
        *,
        current_time: float,
        pitch_bias: float,
        roll_bias: float,
        mass_ff_kg,
        primary_channel_enabled: bool,
        primary_active: int,
    ):
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
            **self._decision_trace_log_fields(prefix="preview_"),
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
