import unittest
from datetime import datetime

import numpy as np

from src.wind_prediction.ballast_planner import PlannerConfig
from src.wind_prediction.ballast_planner_provider import (
    BallastPlannerPreviewProvider,
)


class _ReplayDataset:
    update_interval_s = 600


def _provider(plant_info=None):
    return BallastPlannerPreviewProvider(
        _ReplayDataset(),
        datetime(2020, 1, 1),
        PlannerConfig(),
        [1.0, 1.0, 1.0],
        plant_info=plant_info,
    )


class ProviderPlantInfoTests(unittest.TestCase):
    vector_fields = (
        "tank_masses",
        "target_ballast_mass",
        "pump_rate_cmd_m3_min",
        "pump_net_rate_m3_min",
        "pump_latched",
        "pump_on_elapsed_s",
        "pump_off_elapsed_s",
        "pump_near_target_s",
        "pump_stage_idx",
    )

    def test_real_execution_state_fields_are_preserved(self):
        plant_info = {
            "tank_masses": np.array([101.0, 202.0, 303.0]),
            "target_ballast_mass": np.array([111.0, 190.0, 333.0]),
            "pump_rate_cmd_m3_min": np.array([4.0, 5.0, 6.0]),
            "pump_net_rate_m3_min": np.array([4.0, -5.0, 0.0]),
            "pump_latched": np.array([True, True, False]),
            "pump_on_elapsed_s": np.array([12.0, 24.0, 0.0]),
            "pump_off_elapsed_s": np.array([0.0, 0.0, 18.0]),
            "pump_near_target_s": np.array([1.0, 2.0, 3.0]),
            "pump_stage_idx": np.array([3, 2, 0]),
            "pump_fullspeed_any": 1,
            "pump_total_backlog_kg": 52.5,
            "pump_latch_switch_count": 7,
            "pump_target_motion_kg_s": 2.5,
            "pump_stage_switch_count": 4,
        }
        provider = _provider(plant_info)

        snapshot = provider._plant_info_from(plant_info)

        for name in self.vector_fields:
            np.testing.assert_array_equal(snapshot[name], plant_info[name])
        self.assertEqual(snapshot["pump_net_rate_m3_min"][1], -5.0)
        self.assertEqual(snapshot["pump_total_backlog_kg"], 52.5)
        self.assertEqual(snapshot["pump_latch_switch_count"], 7)
        self.assertEqual(snapshot["pump_target_motion_kg_s"], 2.5)
        self.assertEqual(snapshot["pump_stage_switch_count"], 4)

    def test_snapshot_arrays_do_not_share_storage(self):
        plant_info = {
            name: np.arange(3, dtype=float) + index
            for index, name in enumerate(self.vector_fields)
        }
        plant_info["pump_latched"] = np.array([True, False, True])
        plant_info["pump_stage_idx"] = np.array([1, 2, 3])
        provider = _provider(plant_info)

        snapshot = provider._plant_info_from(plant_info)

        for name in self.vector_fields:
            self.assertFalse(
                np.shares_memory(provider._default_plant_info[name], plant_info[name])
            )
            self.assertFalse(
                np.shares_memory(snapshot[name], plant_info[name])
            )
            self.assertFalse(
                np.shares_memory(snapshot[name], provider._default_plant_info[name])
            )

        original_mass = plant_info["tank_masses"][0]
        snapshot["tank_masses"][0] = -1.0
        self.assertEqual(plant_info["tank_masses"][0], original_mass)
        self.assertNotEqual(provider._default_plant_info["tank_masses"][0], -1.0)

    def test_missing_fields_fall_back_without_rejecting_legacy_input(self):
        provider = _provider(
            {
                "tank_masses": np.array([10.0, 20.0, 30.0]),
                "pump_rate_cmd_m3_min": np.array([1.0, 2.0, 3.0]),
                "pump_fullspeed_any": 1,
                "pump_total_backlog_kg": 90.0,
            }
        )

        snapshot = provider._plant_info_from(
            {
                "tank_masses": np.array([40.0, 50.0, 60.0]),
                "pump_net_rate_m3_min": [9.0, -9.0],
            }
        )

        np.testing.assert_array_equal(snapshot["tank_masses"], [40.0, 50.0, 60.0])
        np.testing.assert_array_equal(
            snapshot["target_ballast_mass"],
            snapshot["tank_masses"],
        )
        np.testing.assert_array_equal(snapshot["pump_net_rate_m3_min"], np.zeros(3))
        np.testing.assert_array_equal(snapshot["pump_latched"], [False, False, False])
        np.testing.assert_array_equal(snapshot["pump_on_elapsed_s"], np.zeros(3))
        np.testing.assert_array_equal(snapshot["pump_off_elapsed_s"], np.zeros(3))
        np.testing.assert_array_equal(snapshot["pump_near_target_s"], np.zeros(3))
        np.testing.assert_array_equal(
            snapshot["pump_stage_idx"],
            np.zeros(3, dtype=int),
        )
        np.testing.assert_array_equal(
            snapshot["pump_rate_cmd_m3_min"],
            [1.0, 2.0, 3.0],
        )
        self.assertEqual(snapshot["pump_fullspeed_any"], 1)
        self.assertEqual(snapshot["pump_total_backlog_kg"], 90.0)

    def test_posture_snapshot_uses_planner_pitch_roll_order(self):
        state = np.zeros(12, dtype=float)
        state[3] = np.radians(-1.25)
        state[4] = np.radians(2.5)
        state[9] = np.radians(-0.04)
        state[10] = np.radians(0.08)

        snapshot = _provider()._attach_posture_info({}, state)

        np.testing.assert_allclose(snapshot["posture_vec_deg"], [2.5, -1.25])
        np.testing.assert_allclose(
            snapshot["posture_rate_vec_deg_s"],
            [0.08, -0.04],
        )

    def test_posture_rate_defaults_to_zero_for_legacy_short_state(self):
        state = np.zeros(6, dtype=float)
        state[3] = np.radians(0.5)
        state[4] = np.radians(-0.75)

        snapshot = _provider()._attach_posture_info({}, state)

        np.testing.assert_allclose(snapshot["posture_vec_deg"], [-0.75, 0.5])
        np.testing.assert_array_equal(
            snapshot["posture_rate_vec_deg_s"],
            np.zeros(2),
        )

    def test_missing_state_publishes_zero_posture_and_rate(self):
        snapshot = _provider()._attach_posture_info({}, None)

        np.testing.assert_array_equal(snapshot["posture_vec_deg"], np.zeros(2))
        np.testing.assert_array_equal(
            snapshot["posture_rate_vec_deg_s"],
            np.zeros(2),
        )


if __name__ == "__main__":
    unittest.main()
