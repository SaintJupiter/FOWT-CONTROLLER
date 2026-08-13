import unittest

import numpy as np

from wind_prediction.attitude_rollout_shadow import (
    AttitudeShadowCriteria,
    evaluate_attitude_rollout_shadow,
)


class AttitudeRolloutShadowTests(unittest.TestCase):
    @staticmethod
    def _criteria(**overrides):
        values = {
            "min_samples": 4,
            "min_distinct_cases": 2,
            "max_axis_mae_deg": 0.25,
            "max_axis_p95_abs_error_deg": 0.5,
            "max_axis_abs_bias_deg": 0.2,
            "min_change_sign_agreement": 0.75,
            "sign_tolerance_deg": 0.01,
        }
        values.update(overrides)
        return AttitudeShadowCriteria(**values)

    def test_perfect_shadow_data_is_report_only_but_trial_eligible(self):
        current = np.array(
            [[0.0, 0.0], [0.5, -0.5], [-0.5, 0.5], [0.2, -0.2]]
        )
        actual = np.array(
            [[0.4, -0.3], [0.8, -0.2], [-0.8, 0.2], [0.0, 0.0]]
        )
        report = evaluate_attitude_rollout_shadow(
            actual,
            actual,
            current_pitch_roll_deg=current,
            case_ids=["case-a", "case-a", "case-b", "case-b"],
            criteria=self._criteria(),
        )

        self.assertTrue(report.evidence.meets_ranking_trial_evidence)
        self.assertEqual(
            report.evidence.status,
            "eligible_for_controlled_ranking_trial",
        )
        self.assertTrue(report.evidence.report_only)
        self.assertFalse(report.evidence.sorting_authority_enabled)
        self.assertEqual(report.pitch.mae_deg, 0.0)
        self.assertEqual(report.roll.change_sign_agreement, 1.0)

    def test_bias_quantiles_and_vector_error_are_computed_from_prediction_minus_actual(self):
        predicted = [[1.0, -1.0], [2.0, 1.0], [3.0, -2.0], [4.0, 2.0]]
        actual = [[0.5, -1.5], [2.5, 0.0], [2.0, -1.0], [5.0, 2.5]]
        report = evaluate_attitude_rollout_shadow(
            predicted,
            actual,
            case_ids=["a", "a", "b", "b"],
            criteria=self._criteria(
                max_axis_mae_deg=2.0,
                max_axis_p95_abs_error_deg=2.0,
                max_axis_abs_bias_deg=2.0,
                min_change_sign_agreement=0.0,
            ),
        )

        pitch_error = np.array([0.5, -0.5, 1.0, -1.0])
        roll_error = np.array([0.5, 1.0, -1.0, -0.5])
        self.assertAlmostEqual(report.pitch.bias_deg, np.mean(pitch_error))
        self.assertAlmostEqual(report.pitch.mae_deg, np.mean(np.abs(pitch_error)))
        self.assertAlmostEqual(
            report.roll.abs_error_quantiles_deg[0.95],
            np.quantile(np.abs(roll_error), 0.95),
        )
        np.testing.assert_allclose(
            report.residuals.vector_error_deg,
            np.hypot(pitch_error, roll_error),
        )

    def test_value_and_change_sign_agreement_are_reported_separately(self):
        current = [[1.0, -1.0], [1.0, -1.0], [1.0, -1.0], [1.0, -1.0]]
        actual = [[1.5, -1.5], [0.5, -0.5], [1.5, -0.5], [0.5, -1.5]]
        predicted = [[1.4, -1.4], [1.4, -1.4], [0.6, -1.4], [0.6, -0.6]]
        report = evaluate_attitude_rollout_shadow(
            predicted,
            actual,
            current_pitch_roll_deg=current,
            case_ids=["a", "a", "b", "b"],
            criteria=self._criteria(
                max_axis_mae_deg=2.0,
                max_axis_p95_abs_error_deg=2.0,
                max_axis_abs_bias_deg=2.0,
                min_change_sign_agreement=0.0,
            ),
        )

        self.assertEqual(report.pitch.value_sign_agreement, 1.0)
        self.assertEqual(report.roll.value_sign_agreement, 1.0)
        self.assertEqual(report.pitch.change_sign_agreement, 0.5)
        self.assertEqual(report.roll.change_sign_agreement, 0.25)

    def test_missing_current_attitude_cannot_pass_change_direction_evidence(self):
        actual = [[0.2, -0.1], [0.3, -0.2], [0.4, -0.3], [0.5, -0.4]]
        report = evaluate_attitude_rollout_shadow(
            actual,
            actual,
            case_ids=["a", "a", "b", "b"],
            criteria=self._criteria(),
        )

        self.assertTrue(np.isnan(report.pitch.change_sign_agreement))
        self.assertFalse(report.evidence.checks["pitch_change_sign"])
        self.assertFalse(report.evidence.meets_ranking_trial_evidence)
        self.assertEqual(report.evidence.status, "shadow_accuracy_not_ready")

    def test_insufficient_independent_cases_is_reported(self):
        actual = np.zeros((4, 2))
        report = evaluate_attitude_rollout_shadow(
            actual,
            actual,
            current_pitch_roll_deg=actual,
            case_ids=["same-case"] * 4,
            criteria=self._criteria(),
        )

        self.assertFalse(report.evidence.meets_ranking_trial_evidence)
        self.assertEqual(report.evidence.status, "insufficient_shadow_data")
        self.assertIn("distinct_case_count", report.evidence.reasons)

    def test_large_errors_fail_accuracy_evidence(self):
        current = np.zeros((4, 2))
        actual = np.ones((4, 2))
        predicted = -np.ones((4, 2))
        report = evaluate_attitude_rollout_shadow(
            predicted,
            actual,
            current_pitch_roll_deg=current,
            case_ids=["a", "a", "b", "b"],
            criteria=self._criteria(),
        )

        self.assertFalse(report.evidence.meets_ranking_trial_evidence)
        self.assertEqual(report.evidence.status, "shadow_accuracy_not_ready")
        self.assertIn("pitch_mae", report.evidence.reasons)
        self.assertIn("roll_change_sign", report.evidence.reasons)

    def test_summary_is_json_friendly_and_excludes_residual_arrays(self):
        actual = [[0.0, 0.0]] * 4
        report = evaluate_attitude_rollout_shadow(
            actual,
            actual,
            current_pitch_roll_deg=actual,
            case_ids=["a", "a", "b", "b"],
            criteria=self._criteria(min_change_sign_agreement=0.0),
        )
        summary = report.to_dict()

        self.assertNotIn("residuals", summary)
        self.assertEqual(summary["sample_count"], 4)
        self.assertFalse(summary["evidence"]["sorting_authority_enabled"])

    def test_invalid_shapes_and_case_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_attitude_rollout_shadow([[0.0, 0.0]], [[0.0]])
        with self.assertRaises(ValueError):
            evaluate_attitude_rollout_shadow(
                [[0.0, 0.0]],
                [[0.0, 0.0]],
                case_ids=[""],
            )
        with self.assertRaises(ValueError):
            AttitudeShadowCriteria(quantiles=(0.5, 0.9))


if __name__ == "__main__":
    unittest.main()
