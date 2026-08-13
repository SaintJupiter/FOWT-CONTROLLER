from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from wind_prediction.control_contracts import DecisionTraceStep
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.provider_telemetry import ProviderTelemetryMixin


class _TelemetryProbe(ProviderTelemetryMixin):
    pass


class ProviderTelemetrySchemaTest(unittest.TestCase):
    def test_complete_record_preserves_stage_semantics(self) -> None:
        provider = _TelemetryProbe()
        provider._forecast_evidence = ForecastEvidence(
            source="learned_lstm",
            model_version="model-v2",
            origin_time="2026-01-01 00:00:00",
            sample_period_s=600.0,
            uv_ms=np.array([[4.0, 3.0], [5.0, 2.0]], dtype=float),
            event_probs={"wind_change": 0.75},
            lead_reliability=np.array([1.0, 0.8], dtype=float),
            provides_future_preview=True,
        )
        provider._candidate_evaluations_snapshot = (
            {
                "sequence": ("hold", "hold", "hold"),
                "hard_reject_reason": "safety_floor_violated_at_block_0",
                "selection_reason": "hard_reject",
                "first_execution_requested_target_kg": [1000.0, 1000.0, 1000.0],
                "first_execution_shaped_target_kg": [1000.0, 1000.0, 1000.0],
                "costs": {"pump_work_cost": 0.0},
            },
            {
                "sequence": ("active_small", "hold", "hold"),
                "hard_reject_reason": "",
                "selection_reason": "feasible",
                "scalar_cost": 10.0,
                "first_execution_requested_target_kg": [1010.0, 990.0, 1000.0],
                "first_execution_shaped_target_kg": [1008.0, 992.0, 1000.0],
                "costs": {"pump_work_cost": 2.5},
            },
            {
                "sequence": ("active_medium", "hold", "hold"),
                "hard_reject_reason": "",
                "selection_reason": "feasible",
                "scalar_cost": 11.5,
                "costs": {"pump_work_cost": 3.0},
            },
        )
        provider._action_plan_snapshot = {
            "status": "reevaluated",
            "candidate_action": "active_small",
            "final_action": "active_small",
            "target_operation": "set_delta",
            "candidate_target_masses_kg": [1010.0, 990.0, 1000.0],
            "final_target_masses_kg": [1010.0, 990.0, 1000.0],
            "correction_reason": "",
            "rejected_reason": "",
            "evaluation": {"safety_passed": True},
        }
        provider._primary_target_kg = np.array([1010.0, 990.0, 1000.0])
        provider._primary_refresh_owner = "planner"
        provider._forecast_action_policy_snapshot = {
            "mode": "shadow",
            "stage_labels": ["0-20 min", "20-40 min", "40-60 min"],
            "authorized_actions": {
                "0-20 min": ["hold", "active_small"],
                "20-40 min": ["hold"],
                "40-60 min": ["hold", "active_reverse_small"],
            },
            "blocked_actions": {
                "0-20 min": ["active_medium"],
                "20-40 min": ["active_medium", "active_reverse_small"],
                "40-60 min": [],
            },
            "threshold_source": {
                "name": "validation_metrics",
                "version": "v1",
            },
            "reasons": {
                "0-20 min": ["reliability_below_medium_threshold"],
                "20-40 min": ["forecast_relief"],
                "40-60 min": ["persistent_direction_reversal"],
            },
        }
        provider._forecast_action_policy_enforcement = {
            "applied": False,
            "reason": "mode_shadow",
            "candidate_count_before": 125,
            "candidate_count_after": 125,
            "blocked_first_actions": (),
        }
        provider._posture_action_enforcement = {
            "applied": True,
            "reason": "completed_target_posture_not_recovering",
            "candidate_count_before": 75,
            "candidate_count_after": 45,
            "blocked_first_actions": ["maintain_target", "stop_execution"],
            "required_axes": [False, True],
        }
        provider._decision_trace_steps = (
            DecisionTraceStep(
                name="final_safety_review",
                input_action="active_small",
                output_action="active_small",
                changed=False,
                reason="accepted",
            ),
        )
        best = {
            "sequence": ("active_small", "hold", "hold"),
            "best_scalar_cost": 10.0,
            "second_best_sequence": ("active_medium", "hold", "hold"),
            "second_best_scalar_cost": 11.5,
            "costs": {"pump_work_cost": 2.5},
            "target_replan_policy_mode": "enforce",
            "target_replan_required": 1,
            "target_replan_target_completed": 1,
            "target_replan_outside_pitch": 1,
            "target_replan_outside_roll": 0,
            "target_replan_recovering_pitch": 0,
            "target_replan_recovering_roll": 0,
            "target_replan_target_direction_helpful": 0,
            "target_replan_action_redirected": 1,
            "target_replan_action_corrective": 1,
            "target_replan_reason": "completed_target_requires_replan",
        }
        plant_info = {
            "tank_masses": np.array([1000.0, 1000.0, 1000.0]),
            "target_ballast_mass": np.array([1005.0, 995.0, 1000.0]),
            "tank_mass_delta_kg": np.array([2.0, -2.0, 0.0]),
            "pump_net_rate_m3_min": np.array([0.2, -0.2, 0.0]),
            "pump_total_backlog_kg": 10.0,
        }

        fields = provider._v2_decision_record_log_fields(
            best=best,
            plant_info=plant_info,
            planner_pitch=0.3,
            planner_roll=-0.2,
        )
        record = json.loads(fields["decision_record_json"])

        self.assertEqual(record["schema_version"], "v2_decision_record_v2")
        self.assertEqual(record["forecast"]["points"][1]["lead_time_s"], 1200.0)
        self.assertEqual(record["forecast"]["points"][1]["reliability"], 0.8)
        self.assertEqual(record["candidates"]["coverage"], "full")
        self.assertEqual(record["candidates"]["rejected_count"], 1)
        self.assertEqual(
            record["candidates"]["rejection_reason_counts"],
            {"safety_floor_violated_at_block_0": 1},
        )
        self.assertEqual(
            record["candidates"]["rows"][0]["hard_reject_reason"],
            "safety_floor_violated_at_block_0",
        )
        self.assertIsNone(
            record["candidates"]["rows"][1]["hard_reject_reason"]
        )
        self.assertEqual(
            record["candidates"]["selected_first_stage"],
            {
                "action": "active_small",
                "target_status": "available",
                "requested_target_masses_kg": [1010.0, 990.0, 1000.0],
                "shaped_target_masses_kg": [1008.0, 992.0, 1000.0],
            },
        )
        self.assertEqual(
            record["candidates"]["best_second_gap"],
            {"status": "available", "absolute": 1.5, "relative_pct": 15.0},
        )
        self.assertEqual(record["planner"]["target_masses_kg"][0], 1010.0)
        self.assertEqual(record["forecast_action_policy"]["mode"], "shadow")
        self.assertEqual(
            record["forecast_action_policy"]["authorized_actions"]["0-20 min"],
            ["hold", "active_small"],
        )
        self.assertEqual(
            record["forecast_action_policy"]["blocked_actions"]["0-20 min"],
            ["active_medium"],
        )
        self.assertEqual(
            record["forecast_action_policy"]["threshold_source"],
            {"name": "validation_metrics", "version": "v1"},
        )
        self.assertEqual(
            fields["decision_record_forecast_action_policy_mode"],
            "shadow",
        )
        self.assertEqual(
            record["forecast_action_policy"]["candidate_filter"],
            {
                "status": "available",
                "applied": False,
                "reason": "mode_shadow",
                "candidate_count_before": 125,
                "candidate_count_after": 125,
                "blocked_first_actions": [],
                "posture_strengthen_restored": False,
            },
        )
        self.assertFalse(
            fields["decision_record_forecast_action_policy_filter_applied"]
        )
        self.assertEqual(
            record["posture_action_authority"],
            {
                "status": "available",
                "applied": True,
                "reason": "completed_target_posture_not_recovering",
                "candidate_count_before": 75,
                "candidate_count_after": 45,
                "blocked_first_actions": [
                    "maintain_target",
                    "stop_execution",
                ],
                "required_axes": [False, True],
            },
        )
        self.assertTrue(
            fields["decision_record_posture_action_authority_applied"]
        )
        self.assertEqual(
            fields["decision_record_forecast_action_policy_candidate_count_before"],
            125,
        )
        self.assertEqual(
            fields["decision_record_forecast_action_policy_candidate_count_after"],
            125,
        )
        self.assertTrue(record["safety_review"]["safety_passed"])
        self.assertEqual(record["target_commit"]["refresh_owner"], "planner")
        self.assertTrue(record["target_replan"]["required"])
        self.assertTrue(record["target_replan"]["action_redirected"])
        self.assertTrue(record["target_replan"]["action_corrective"])
        self.assertEqual(
            record["target_replan"]["reason"],
            "completed_target_requires_replan",
        )
        self.assertEqual(
            fields["decision_record_target_replan_mode"],
            "enforce",
        )
        self.assertTrue(fields["decision_record_target_replan_action_redirected"])
        self.assertTrue(fields["decision_record_target_replan_action_corrective"])
        self.assertEqual(
            record["execution_observation"]["timing"],
            "plant_step_before_current_decision",
        )
        self.assertEqual(
            record["execution_observation"]["actual_mass_delta_kg"],
            [2.0, -2.0, 0.0],
        )
        self.assertEqual(record["target_debt"]["per_tank_kg"], [10.0, -10.0, 0.0])
        self.assertEqual(fields["decision_record_target_debt_abs_total_kg"], 20.0)
        self.assertNotIn("NaN", fields["decision_record_json"])

    def test_missing_data_is_explicit_and_does_not_mutate_state(self) -> None:
        provider = _TelemetryProbe()
        provider._decision_trace_steps = ()
        provider._action_plan_snapshot = {"status": "not_evaluated"}
        before = copy.deepcopy(provider.__dict__)

        fields = provider._v2_decision_record_log_fields(
            best={"sequence": ("hold",), "best_scalar_cost": float("nan")},
            plant_info={"tank_masses": [1.0, 2.0]},
        )
        record = json.loads(fields["decision_record_json"])

        self.assertEqual(provider.__dict__, before)
        self.assertEqual(record["forecast"]["status"], "missing")
        self.assertEqual(record["forecast_action_policy"]["status"], "missing")
        self.assertEqual(record["target_replan"]["status"], "missing")
        self.assertIsNone(record["forecast_action_policy"]["mode"])
        self.assertIsNone(record["forecast_action_policy"]["stage_labels"])
        self.assertIsNone(record["forecast_action_policy"]["authorized_actions"])
        self.assertIsNone(record["forecast_action_policy"]["blocked_actions"])
        self.assertIsNone(record["forecast_action_policy"]["threshold_source"])
        self.assertIsNone(record["forecast_action_policy"]["reasons"])
        self.assertEqual(
            record["forecast_action_policy"]["candidate_filter"],
            {
                "status": "missing",
                "applied": None,
                "reason": None,
                "candidate_count_before": None,
                "candidate_count_after": None,
                "blocked_first_actions": None,
                "posture_strengthen_restored": None,
            },
        )
        self.assertIsNone(
            fields["decision_record_forecast_action_policy_authorized_actions_json"]
        )
        self.assertIsNone(
            fields["decision_record_forecast_action_policy_blocked_actions_json"]
        )
        self.assertIsNone(
            fields["decision_record_forecast_action_policy_reasons_json"]
        )
        self.assertIsNone(
            fields["decision_record_forecast_action_policy_filter_applied"]
        )
        self.assertIsNone(
            fields[
                "decision_record_forecast_action_policy_blocked_first_actions_json"
            ]
        )
        self.assertEqual(record["candidates"]["coverage"], "selected_only")
        self.assertEqual(
            record["candidates"]["selected_first_stage"],
            {
                "action": "hold",
                "target_status": "missing",
                "requested_target_masses_kg": None,
                "shaped_target_masses_kg": None,
            },
        )
        self.assertIsNone(record["candidates"]["best_scalar_cost"])
        self.assertEqual(record["candidates"]["best_second_gap"]["status"], "missing")
        self.assertEqual(record["target_commit"]["status"], "missing")
        self.assertEqual(record["target_debt"]["status"], "missing")
        self.assertIsNone(fields["decision_record_best_second_gap"])
        self.assertIsNone(fields["decision_record_target_debt_abs_total_kg"])
        self.assertNotIn("NaN", fields["decision_record_json"])

    def test_enforce_policy_uses_the_same_schema_as_shadow_policy(self) -> None:
        provider = _TelemetryProbe()
        provider._decision_trace_steps = ()
        provider._action_plan_snapshot = {"status": "not_evaluated"}
        provider._forecast_action_policy_snapshot = {
            "mode": "enforce",
            "stage_labels": ["near"],
            "authorized_actions": {"near": ["active_small"]},
            "blocked_actions": {"near": ["active_medium"]},
            "threshold_source": "held_out_validation",
            "reasons": {"near": ["medium_not_authorized"]},
        }
        provider._forecast_action_policy_enforcement = {
            "applied": True,
            "reason": "filtered",
            "candidate_count_before": 125,
            "candidate_count_after": 75,
            "blocked_first_actions": ["active_medium"],
            "posture_strengthen_restored": True,
        }

        fields = provider._v2_decision_record_log_fields(
            best={"sequence": ("active_small",)},
        )
        policy = json.loads(fields["decision_record_json"])[
            "forecast_action_policy"
        ]

        self.assertEqual(policy["status"], "available")
        self.assertEqual(policy["mode"], "enforce")
        self.assertEqual(policy["stage_labels"], ["near"])
        self.assertEqual(policy["authorized_actions"], {"near": ["active_small"]})
        self.assertEqual(policy["blocked_actions"], {"near": ["active_medium"]})
        self.assertEqual(policy["threshold_source"], "held_out_validation")
        self.assertEqual(policy["reasons"], {"near": ["medium_not_authorized"]})
        self.assertEqual(
            policy["candidate_filter"],
            {
                "status": "available",
                "applied": True,
                "reason": "filtered",
                "candidate_count_before": 125,
                "candidate_count_after": 75,
                "blocked_first_actions": ["active_medium"],
                "posture_strengthen_restored": True,
            },
        )

    def test_empty_filter_rollback_is_explicit_in_stable_telemetry(self) -> None:
        provider = _TelemetryProbe()
        provider._decision_trace_steps = ()
        provider._action_plan_snapshot = {"status": "not_evaluated"}
        provider._forecast_action_policy_snapshot = {
            "mode": "enforce",
            "stage_labels": ["near"],
            "authorized_actions": {"near": []},
            "blocked_actions": {"near": ["maintain_target"]},
            "threshold_source": "held_out_validation",
            "reasons": {"near": ["hold_not_authorized"]},
        }
        provider._forecast_action_policy_enforcement = {
            "applied": False,
            "reason": "empty_filter_rollback",
            "candidate_count_before": 1,
            "candidate_count_after": 1,
            "blocked_first_actions": ["maintain_target"],
        }

        fields = provider._v2_decision_record_log_fields(
            best={"sequence": ("maintain_target",)},
        )
        candidate_filter = json.loads(fields["decision_record_json"])[
            "forecast_action_policy"
        ]["candidate_filter"]

        self.assertFalse(candidate_filter["applied"])
        self.assertEqual(candidate_filter["reason"], "empty_filter_rollback")
        self.assertEqual(candidate_filter["candidate_count_before"], 1)
        self.assertEqual(candidate_filter["candidate_count_after"], 1)
        self.assertEqual(
            candidate_filter["blocked_first_actions"],
            ["maintain_target"],
        )
        self.assertEqual(
            fields["decision_record_forecast_action_policy_filter_reason"],
            "empty_filter_rollback",
        )
        self.assertEqual(
            json.loads(
                fields[
                    "decision_record_forecast_action_policy_blocked_first_actions_json"
                ]
            ),
            ["maintain_target"],
        )

    def test_legacy_trace_fields_are_unchanged(self) -> None:
        provider = _TelemetryProbe()
        provider._decision_trace_steps = (
            DecisionTraceStep(
                name="cap",
                input_action="active_medium",
                output_action="active_small",
                changed=True,
                reason="limit",
                metadata={"value": np.float64(1.25)},
            ),
        )

        fields = provider._decision_trace_log_fields()

        self.assertEqual(fields["decision_trace_modification_count"], 1)
        self.assertEqual(
            json.loads(fields["decision_trace_json"])[0]["output_action"],
            "active_small",
        )


if __name__ == "__main__":
    unittest.main()
