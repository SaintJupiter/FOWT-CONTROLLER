import unittest

import numpy as np

from wind_prediction.ballast_planner import (
    _ALL_SEQUENCES,
    PlannerConfig,
    apply_posture_state_to_blocks,
    compute_pressure_blocks,
    evaluate_sequence,
    planner_action_vec,
)


class BallastPlannerIntrastageForecastTests(unittest.TestCase):
    @staticmethod
    def plant():
        return {
            "tank_masses": np.asarray(
                [1108000.0, 1362000.0, 1362000.0],
                dtype=float,
            ),
            "pump_fullspeed_any": 0,
        }

    @staticmethod
    def block(mean, steps):
        mean_vec = np.asarray(mean, dtype=float)
        step_vecs = np.asarray(steps, dtype=float)
        return {
            "pressure_vec": mean_vec,
            "pressure_vec_raw": mean_vec.copy(),
            "pressure_norm": float(np.linalg.norm(mean_vec)),
            "pressure_step_vecs": step_vecs.copy(),
            "pressure_step_vecs_raw": step_vecs.copy(),
            "pressure_step_vecs_evaluation": step_vecs.copy(),
            "within_block_reversal": int(
                float(np.dot(step_vecs[0], step_vecs[1])) < 0.0
            ),
        }

    def test_default_path_keeps_three_stages_and_125_candidates(self):
        self.assertEqual(PlannerConfig().pressure_aggregation_mode, "legacy_mean")
        self.assertEqual(len(_ALL_SEQUENCES), 5 ** 3)
        self.assertTrue(all(len(sequence) == 3 for sequence in _ALL_SEQUENCES))

    def test_all_125_candidates_use_the_same_three_stage_structure(self):
        uv = np.asarray(
            [
                [7.0, 1.0],
                [9.0, 2.0],
                [8.0, -1.0],
                [6.0, -2.0],
                [5.0, 1.0],
                [4.0, 0.0],
            ],
            dtype=float,
        )
        cfg = PlannerConfig(
            pressure_aggregation_mode="intrastage_stepwise",
            posture_state_residual_active=False,
        )
        blocks = compute_pressure_blocks(uv, [1.0, 0.85, 0.7], cfg)

        evaluated = [
            evaluate_sequence(sequence, blocks, self.plant(), cfg)
            for sequence in _ALL_SEQUENCES
        ]

        self.assertEqual(len(evaluated), 125)
        self.assertTrue(all(len(row["sequence"]) == 3 for row in evaluated))
        self.assertTrue(
            all(row.get("intrastage_forecast_evaluation") == 1 for row in evaluated)
        )

    def test_v1_numeric_snapshot_is_unchanged(self):
        blocks = [
            self.block((1.2, -0.4), ((1.2, -0.4), (1.2, -0.4))),
            self.block((0.7, 0.3), ((0.7, 0.3), (0.7, 0.3))),
            self.block((-0.2, 0.6), ((-0.2, 0.6), (-0.2, 0.6))),
        ]

        result = evaluate_sequence(
            ("hold", "hold", "hold"),
            blocks,
            self.plant(),
            PlannerConfig(),
        )

        self.assertEqual(result["hard_reject_reason"], "")
        self.assertNotIn("intrastage_forecast_evaluation", result)
        self.assertEqual(
            result["costs"]["attitude_residual_cost"],
            2.9231249999999998,
        )
        self.assertEqual(
            result["costs"]["terminal_residual_cost"],
            0.6024999999999997,
        )
        self.assertEqual(
            result["costs"]["envelope_violation_cost"],
            0.006944444444444432,
        )

    def test_both_leads_enter_residual_cost_without_doubling_stage_weight(self):
        blocks = [
            self.block((1.0, 0.0), ((0.0, 0.0), (2.0, 0.0))),
            self.block((0.0, 0.0), ((0.0, 0.0), (0.0, 0.0))),
            self.block((0.0, 0.0), ((0.0, 0.0), (0.0, 0.0))),
        ]
        common = dict(
            deadband_pitch_deg=1.0,
            deadband_roll_deg=1.0,
            pitch_envelope_deg=10.0,
            roll_envelope_deg=10.0,
        )

        block_mean = evaluate_sequence(
            ("hold", "hold", "hold"),
            blocks,
            self.plant(),
            PlannerConfig(pressure_aggregation_mode="mean_step_force", **common),
        )
        intrastage = evaluate_sequence(
            ("hold", "hold", "hold"),
            blocks,
            self.plant(),
            PlannerConfig(
                pressure_aggregation_mode="intrastage_stepwise",
                **common,
            ),
        )

        self.assertEqual(block_mean["costs"]["attitude_residual_cost"], 1.0)
        self.assertEqual(intrastage["costs"]["attitude_residual_cost"], 2.0)
        self.assertEqual(intrastage["intrastage_points_evaluated"], 6)
        self.assertEqual(intrastage["costs"]["pump_work_cost"], 0.0)
        self.assertEqual(intrastage["costs"]["pump_duration_cost"], 0.0)

    def test_peak_and_reversal_cannot_cancel_in_safety_check(self):
        reversing = self.block(
            (0.0, 0.0),
            ((6.0, 0.0), (-6.0, 0.0)),
        )
        quiet = self.block(
            (0.0, 0.0),
            ((0.0, 0.0), (0.0, 0.0)),
        )
        blocks = [reversing, quiet, quiet]
        common = dict(
            safety_floor_active=True,
            safety_floor_pitch_deg=5.0,
            safety_floor_roll_deg=5.0,
            posture_state_residual_active=False,
        )

        block_mean = evaluate_sequence(
            ("hold", "hold", "hold"),
            blocks,
            self.plant(),
            PlannerConfig(pressure_aggregation_mode="mean_step_force", **common),
        )
        intrastage = evaluate_sequence(
            ("hold", "hold", "hold"),
            blocks,
            self.plant(),
            PlannerConfig(
                pressure_aggregation_mode="intrastage_stepwise",
                **common,
            ),
        )

        self.assertEqual(block_mean["hard_reject_reason"], "")
        self.assertEqual(
            intrastage["hard_reject_reason"],
            "safety_floor_violated_at_block_0_lead_0",
        )
        self.assertEqual(intrastage["intrastage_reversal_blocks_evaluated"], 1)
        self.assertEqual(intrastage["intrastage_safety_lead_index"], 0)

    def test_order_with_same_block_mean_changes_terminal_residual(self):
        quiet = self.block(
            (0.0, 0.0),
            ((0.0, 0.0), (0.0, 0.0)),
        )
        rising = self.block(
            (1.0, 0.0),
            ((0.0, 0.0), (2.0, 0.0)),
        )
        falling = self.block(
            (1.0, 0.0),
            ((2.0, 0.0), (0.0, 0.0)),
        )
        cfg = PlannerConfig(
            pressure_aggregation_mode="intrastage_stepwise",
            deadband_pitch_deg=1.0,
            deadband_roll_deg=1.0,
        )

        rising_result = evaluate_sequence(
            ("hold", "hold", "hold"),
            [quiet, quiet, rising],
            self.plant(),
            cfg,
        )
        falling_result = evaluate_sequence(
            ("hold", "hold", "hold"),
            [quiet, quiet, falling],
            self.plant(),
            cfg,
        )

        self.assertEqual(
            rising_result["costs"]["terminal_residual_cost"],
            4.0,
        )
        self.assertEqual(
            falling_result["costs"]["terminal_residual_cost"],
            0.0,
        )

    def test_posture_state_is_added_to_every_intrastage_view(self):
        uv = np.asarray(
            [[4.0, 0.0], [5.0, 0.0]] * 3,
            dtype=float,
        )
        cfg = PlannerConfig(
            pressure_aggregation_mode="intrastage_stepwise",
            posture_state_gain=0.5,
            posture_state_decay=1.0,
        )
        blocks = compute_pressure_blocks(uv, [1.0, 0.85, 0.7], cfg)
        adjusted, _ = apply_posture_state_to_blocks(
            blocks,
            {"posture_vec_deg": np.asarray([2.0, -1.0])},
            cfg,
        )

        self.assertEqual(adjusted[0]["pressure_step_vecs_evaluation"].shape, (2, 2))
        self.assertEqual(adjusted[0]["pressure_step_vecs_safety"].shape, (2, 2))
        np.testing.assert_allclose(
            adjusted[0]["pressure_step_vecs_evaluation"]
            - blocks[0]["pressure_step_vecs_evaluation"],
            np.asarray([[1.0, -0.5], [1.0, -0.5]]),
        )

    def test_posture_above_deadband_can_direct_action_with_subunit_gain(self):
        cfg = PlannerConfig(
            posture_state_gain=0.4,
            posture_state_decay=1.0,
            deadband_pitch_deg=1.0,
            deadband_roll_deg=0.8,
        )
        blocks = [self.block((0.0, 0.0), ((0.0, 0.0), (0.0, 0.0)))] * 3
        adjusted, _ = apply_posture_state_to_blocks(
            blocks,
            {"posture_vec_deg": np.asarray([0.0, 1.0])},
            cfg,
        )

        action, _ = planner_action_vec(
            "active_small",
            0,
            adjusted,
            {},
            cfg,
        )

        self.assertEqual(float(action[0]), 0.0)
        self.assertGreater(float(action[1]), 0.0)

    def test_posture_below_deadband_does_not_direct_action(self):
        cfg = PlannerConfig(
            posture_state_gain=0.4,
            posture_state_decay=1.0,
            deadband_pitch_deg=1.0,
            deadband_roll_deg=0.8,
        )
        blocks = [self.block((0.0, 0.0), ((0.0, 0.0), (0.0, 0.0)))] * 3
        adjusted, _ = apply_posture_state_to_blocks(
            blocks,
            {"posture_vec_deg": np.asarray([0.0, 0.7])},
            cfg,
        )

        action, _ = planner_action_vec(
            "active_small",
            0,
            adjusted,
            {},
            cfg,
        )

        np.testing.assert_allclose(action, np.zeros(2))


if __name__ == "__main__":
    unittest.main()
