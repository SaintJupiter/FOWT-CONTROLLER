import json
import unittest
from dataclasses import FrozenInstanceError

from wind_prediction.action_plan import (
    TargetOperation,
    apply_action_correction,
    build_candidate_plan,
    commit_action_plan,
    record_evaluation,
)


class ActionPlanTests(unittest.TestCase):
    def test_candidate_plan_has_explicit_immutable_identity(self):
        plan = build_candidate_plan("active_medium", TargetOperation.SET_DELTA)

        self.assertEqual(plan.candidate_action, "active_medium")
        self.assertEqual(plan.final_action, "active_medium")
        self.assertEqual(plan.target_operation, TargetOperation.SET_DELTA)
        self.assertTrue(plan.needs_reevaluation)
        with self.assertRaises(FrozenInstanceError):
            plan.final_action = "active_small"

    def test_target_operation_vocabulary_is_closed_and_stable(self):
        self.assertEqual(
            {operation.value for operation in TargetOperation},
            {
                "set_delta",
                "continue",
                "hold_current",
                "pause",
                "resume",
                "release",
            },
        )

    def test_matching_safe_evaluation_can_commit(self):
        plan = build_candidate_plan("active_medium", "set_delta")
        evaluated = record_evaluation(
            plan,
            "active_medium",
            safety_passed=True,
            score=0.21,
            summary="rank=1; envelope clear",
        )

        snapshot = commit_action_plan(evaluated)

        self.assertEqual(snapshot["candidate_action"], "active_medium")
        self.assertEqual(snapshot["final_action"], "active_medium")
        self.assertEqual(snapshot["target_operation"], "set_delta")
        self.assertFalse(snapshot["needs_reevaluation"])
        self.assertTrue(snapshot["evaluation"]["safety_passed"])
        json.dumps(snapshot)

    def test_ranked_action_downgrade_cannot_commit_without_reevaluation(self):
        ranked = record_evaluation(
            build_candidate_plan("active_medium", "set_delta"),
            "active_medium",
            safety_passed=True,
            score=0.21,
            summary="rank=1",
        )

        corrected = apply_action_correction(
            ranked,
            "active_small",
            reason="supervisor capacity cap",
        )

        self.assertEqual(corrected.candidate_action, "active_medium")
        self.assertEqual(corrected.final_action, "active_small")
        self.assertTrue(corrected.needs_reevaluation)
        self.assertEqual(corrected.evaluation.evaluated_action, "active_medium")
        with self.assertRaisesRegex(ValueError, "does not match the evaluated action"):
            commit_action_plan(corrected)

    def test_downgraded_action_can_commit_after_matching_reevaluation(self):
        selected = record_evaluation(
            build_candidate_plan("active_medium", "set_delta"),
            "active_medium",
            safety_passed=True,
        )
        corrected = apply_action_correction(
            selected,
            "active_small",
            reason="supervisor capacity cap",
        )

        reevaluated = record_evaluation(
            corrected,
            "active_small",
            safety_passed=True,
            score=0.32,
            summary="corrected action re-ranked",
        )
        snapshot = commit_action_plan(reevaluated)

        self.assertFalse(reevaluated.needs_reevaluation)
        self.assertEqual(snapshot["candidate_action"], "active_medium")
        self.assertEqual(snapshot["final_action"], "active_small")
        self.assertEqual(snapshot["correction_reason"], "supervisor capacity cap")

    def test_target_operation_correction_also_requires_reevaluation(self):
        evaluated = record_evaluation(
            build_candidate_plan("hold", "continue"),
            "hold",
            safety_passed=True,
        )

        corrected = apply_action_correction(
            evaluated,
            "hold",
            target_operation="hold_current",
            reason="safety hold",
        )

        self.assertTrue(corrected.needs_reevaluation)
        with self.assertRaisesRegex(ValueError, "target operation does not match"):
            commit_action_plan(corrected)

    def test_action_vector_change_requires_reevaluation_even_when_label_is_same(self):
        evaluated = record_evaluation(
            build_candidate_plan(
                "active_medium",
                "set_delta",
                action_vector=(0.2, -0.1),
            ),
            "active_medium",
            action_vector=(0.2, -0.1),
            safety_passed=True,
        )

        corrected = apply_action_correction(
            evaluated,
            "active_medium",
            action_vector=(0.1, -0.05),
            reason="axis amplitude cap",
        )

        self.assertTrue(corrected.needs_reevaluation)
        with self.assertRaisesRegex(ValueError, "action vector does not match"):
            commit_action_plan(corrected)

    def test_target_change_requires_reevaluation_even_when_action_is_same(self):
        evaluated = record_evaluation(
            build_candidate_plan(
                "active_small",
                "set_delta",
                target_masses_kg=(100.0, 200.0, 300.0),
            ),
            "active_small",
            target_masses_kg=(100.0, 200.0, 300.0),
            safety_passed=True,
        )

        corrected = apply_action_correction(
            evaluated,
            "active_small",
            target_masses_kg=(110.0, 190.0, 300.0),
            reason="target capacity projection",
        )

        self.assertTrue(corrected.needs_reevaluation)
        with self.assertRaisesRegex(ValueError, "target masses do not match"):
            commit_action_plan(corrected)

    def test_unsafe_final_action_cannot_commit(self):
        evaluated = record_evaluation(
            build_candidate_plan("active_small", "set_delta"),
            "active_small",
            safety_passed=False,
            summary="tank envelope violation",
        )

        self.assertFalse(evaluated.needs_reevaluation)
        with self.assertRaisesRegex(ValueError, "failed safety evaluation"):
            commit_action_plan(evaluated)


if __name__ == "__main__":
    unittest.main()
