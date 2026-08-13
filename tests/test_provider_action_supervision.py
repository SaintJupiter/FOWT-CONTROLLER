import unittest

import numpy as np

from wind_prediction.ballast_planner import (
    PlannerConfig,
    economic_scalar_cost,
    evaluate_sequence,
)
from wind_prediction.provider_runtime import ProviderRuntimeMixin


class _Provider(ProviderRuntimeMixin):
    objective_mode = "economic"


class ProviderActionSupervisionTests(unittest.TestCase):
    @staticmethod
    def _blocks():
        vector = np.asarray([2.0, 0.0], dtype=float)
        return [
            {
                "pressure_vec": vector.copy(),
                "pressure_vec_raw": vector.copy(),
                "pressure_vec_safety": vector.copy(),
                "pressure_norm": 2.0,
            }
            for _ in range(3)
        ]

    @staticmethod
    def _plant(posture=(0.0, 0.0)):
        return {
            "tank_masses": np.asarray([500000.0, 500000.0, 500000.0]),
            "posture_vec_deg": np.asarray(posture, dtype=float),
        }

    def _ranked(self, cfg, plant):
        provider = _Provider()
        provider._last_planner_cfg = cfg
        provider._last_planner_blocks = self._blocks()
        provider._last_planner_info = plant
        row = evaluate_sequence(
            ("active_small", "hold", "hold"),
            provider._last_planner_blocks,
            plant,
            cfg,
        )
        row["best_scalar_cost"] = economic_scalar_cost(row, cfg)
        plan = provider._build_ranked_action_plan(
            row,
            "active_small",
            np.asarray([0.15, 0.0]),
        )
        return provider, row, plan

    def test_changed_vector_is_reevaluated_before_commit(self):
        provider, best, plan = self._ranked(PlannerConfig(), self._plant())

        action, vector, final_row, handoff = provider._finalize_supervised_action(
            plan=plan,
            best=best,
            action="active_small",
            avec=np.asarray([0.10, 0.0]),
        )

        self.assertEqual(action, "active_small")
        np.testing.assert_allclose(vector, [0.10, 0.0])
        self.assertIsNotNone(final_row)
        self.assertFalse(handoff)
        self.assertEqual(provider._action_plan_snapshot["status"], "reevaluated")

    def test_unsafe_hold_correction_reverts_to_ranked_safe_action(self):
        cfg = PlannerConfig(
            safety_floor_active=True,
            safety_floor_pitch_deg=5.0,
            safety_floor_roll_deg=5.0,
            safety_floor_current_engage_margin_deg=0.0,
        )
        provider, best, plan = self._ranked(cfg, self._plant(posture=(5.1, 0.0)))

        action, vector, final_row, handoff = provider._finalize_supervised_action(
            plan=plan,
            best=best,
            action="hold",
            avec=np.zeros(2),
        )

        self.assertEqual(action, "active_small")
        np.testing.assert_allclose(vector, [0.15, 0.0])
        self.assertIsNone(final_row)
        self.assertFalse(handoff)
        self.assertEqual(
            provider._action_plan_snapshot["status"], "correction_rejected"
        )

    def test_all_candidates_unsafe_routes_explicit_handoff(self):
        provider, best, plan = self._ranked(PlannerConfig(), self._plant())
        best["planned_reactive_handoff"] = 1
        best["hard_reject_reason"] = "safety_floor_violated_at_block_0"

        action, vector, final_row, handoff = provider._finalize_supervised_action(
            plan=plan,
            best=best,
            action="active_small",
            avec=np.asarray([0.15, 0.0]),
        )

        self.assertEqual(action, "hold")
        np.testing.assert_allclose(vector, np.zeros(2))
        self.assertIsNone(final_row)
        self.assertTrue(handoff)
        self.assertEqual(
            provider._action_plan_snapshot["status"], "reactive_handoff"
        )


if __name__ == "__main__":
    unittest.main()
