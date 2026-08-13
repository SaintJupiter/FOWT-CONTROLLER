import unittest
from dataclasses import FrozenInstanceError

from wind_prediction.target_replan_policy import (
    TargetReplanConfig,
    TargetReplanResult,
    evaluate_target_replan,
)


class TargetReplanPolicyTests(unittest.TestCase):
    @staticmethod
    def config() -> TargetReplanConfig:
        return TargetReplanConfig(
            pitch_envelope_deg=1.2,
            roll_envelope_deg=0.96,
            target_stop_error_kg=300.0,
            rate_noise_tolerance_deg_s=0.002,
        )

    def evaluate(
        self,
        *,
        posture=(0.0, 0.0),
        rate=(0.0, 0.0),
        current=(1000.0, 1000.0, 1000.0),
        target=(1000.0, 1000.0, 1000.0),
        direction=(0.0, 0.0),
    ) -> TargetReplanResult:
        return evaluate_target_replan(
            posture,
            rate,
            current,
            target,
            self.config(),
            target_recovery_direction=direction,
        )

    def test_config_and_result_are_immutable(self):
        config = self.config()
        result = self.evaluate()

        with self.assertRaises(FrozenInstanceError):
            config.target_stop_error_kg = 1.0
        with self.assertRaises(FrozenInstanceError):
            result.requires_replan = True

    def test_target_completion_uses_per_tank_inclusive_stop_error(self):
        completed = self.evaluate(target=(1300.0, 700.0, 1000.0))
        unfinished = self.evaluate(target=(1300.01, 1000.0, 1000.0))

        self.assertTrue(completed.target_completed)
        self.assertFalse(unfinished.target_completed)

    def test_completed_target_outside_envelope_does_not_create_new_work(self):
        result = self.evaluate(posture=(2.0, 0.0), rate=(0.01, 0.0))

        self.assertEqual(result.outside_envelope_axes, (True, False))
        self.assertEqual(result.recovering_axes, (False, False))
        self.assertFalse(result.requires_replan)
        self.assertEqual(result.reason, "completed_target_has_no_execution_debt")

    def test_completed_target_outside_envelope_and_stationary_does_not_replan(self):
        result = self.evaluate(posture=(0.0, -2.0), rate=(0.0, 0.0))

        self.assertEqual(result.outside_envelope_axes, (False, True))
        self.assertFalse(result.requires_replan)

    def test_completed_target_outside_envelope_but_recovering_can_wait(self):
        result = self.evaluate(posture=(2.0, -2.0), rate=(-0.01, 0.01))

        self.assertEqual(result.recovering_axes, (True, True))
        self.assertFalse(result.requires_replan)
        self.assertEqual(result.reason, "outside_envelope_recovering")

    def test_inside_envelope_does_not_require_replan(self):
        result = self.evaluate(posture=(1.2, -0.96), rate=(0.0, 0.0))

        self.assertEqual(result.outside_envelope_axes, (False, False))
        self.assertFalse(result.requires_replan)
        self.assertEqual(result.reason, "inside_envelope")

    def test_unfinished_restorative_target_can_continue(self):
        result = self.evaluate(
            posture=(2.0, -2.0),
            rate=(0.0, 0.0),
            target=(1400.0, 600.0, 1000.0),
            direction=(0.3, -0.2),
        )

        self.assertFalse(result.target_completed)
        self.assertTrue(result.target_direction_helpful)
        self.assertFalse(result.requires_replan)
        self.assertEqual(result.reason, "unfinished_target_direction_helpful")

    def test_unfinished_nonrestorative_target_requires_replan(self):
        result = self.evaluate(
            posture=(2.0, -2.0),
            rate=(0.0, 0.0),
            target=(1400.0, 600.0, 1000.0),
            direction=(-0.3, -0.2),
        )

        self.assertFalse(result.target_direction_helpful)
        self.assertTrue(result.requires_replan)
        self.assertEqual(
            result.reason,
            "unfinished_target_direction_not_helpful",
        )

    def test_one_unrecovered_outside_axis_prevents_whole_state_recovery(self):
        result = self.evaluate(
            posture=(2.0, -2.0),
            rate=(-0.01, -0.01),
        )

        self.assertEqual(result.recovering_axes, (True, False))
        self.assertFalse(result.requires_replan)

    def test_direction_only_needs_to_help_unrecovered_outside_axes(self):
        result = self.evaluate(
            posture=(2.0, -2.0),
            rate=(-0.01, 0.0),
            target=(1400.0, 600.0, 1000.0),
            direction=(0.0, -0.2),
        )

        self.assertEqual(result.recovering_axes, (True, False))
        self.assertTrue(result.target_direction_helpful)
        self.assertFalse(result.requires_replan)

    def test_common_mode_debt_mapping_to_zero_is_not_helpful(self):
        seen_debt = []

        def common_mode_mapper(debt):
            seen_debt.append(debt)
            return (0.0, 0.0)

        result = evaluate_target_replan(
            posture_deg=(2.0, 0.0),
            posture_rate_deg_s=(0.0, 0.0),
            current_masses_kg=(1000.0, 1000.0, 1000.0),
            target_masses_kg=(1400.0, 1400.0, 1400.0),
            config=self.config(),
            debt_to_recovery_direction=common_mode_mapper,
        )

        self.assertEqual(seen_debt, [(400.0, 400.0, 400.0)])
        self.assertFalse(result.target_completed)
        self.assertFalse(result.target_direction_helpful)
        self.assertTrue(result.requires_replan)

    def test_completed_target_has_no_remaining_helpful_direction(self):
        result = self.evaluate(
            posture=(2.0, 0.0),
            rate=(0.0, 0.0),
            target=(1100.0, 1000.0, 1000.0),
            direction=(1.0, 0.0),
        )

        self.assertTrue(result.target_completed)
        self.assertFalse(result.target_direction_helpful)
        self.assertFalse(result.requires_replan)

    def test_rate_inside_noise_tolerance_is_not_recovery(self):
        result = self.evaluate(posture=(2.0, 0.0), rate=(-0.002, 0.0))

        self.assertEqual(result.recovering_axes, (False, False))
        self.assertFalse(result.requires_replan)

    def test_exactly_one_direction_source_is_required(self):
        arguments = dict(
            posture_deg=(0.0, 0.0),
            posture_rate_deg_s=(0.0, 0.0),
            current_masses_kg=(1000.0, 1000.0, 1000.0),
            target_masses_kg=(1000.0, 1000.0, 1000.0),
            config=self.config(),
        )

        with self.assertRaises(ValueError):
            evaluate_target_replan(**arguments)
        with self.assertRaises(ValueError):
            evaluate_target_replan(
                **arguments,
                target_recovery_direction=(0.0, 0.0),
                debt_to_recovery_direction=lambda debt: (0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
