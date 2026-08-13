import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(SRC_DIR))

import test_plant_execution_rollout_parity as _parity  # noqa: E402
from target_execution import TargetSlewLimiter  # noqa: E402
from wind_prediction.execution_rollout import simulate_execution_step  # noqa: E402


SIX_HOURS_S = 21_600
STAGE_DURATION_S = 2_400


class SixHourExecutionConsistencyTests(unittest.TestCase):
    @staticmethod
    def _requested_targets(initial_kg, capacity_kg):
        initial = np.asarray(initial_kg, dtype=float)
        return np.asarray(
            [
                initial + [120_000.0, -90_000.0, 60_000.0],
                initial + [-100_000.0, 130_000.0, -70_000.0],
                initial,
                initial + [180_000.0, 20_000.0, -160_000.0],
                initial + [-160_000.0, -100_000.0, 180_000.0],
                initial + [80_000.0, -170_000.0, 110_000.0],
                [
                    capacity_kg + 50_000.0,
                    initial[1] - 220_000.0,
                    initial[2] + 140_000.0,
                ],
                [
                    -50_000.0,
                    initial[1] + 200_000.0,
                    initial[2] - 180_000.0,
                ],
                initial,
            ],
            dtype=float,
        )

    def test_rollout_matches_frozen_plant_for_six_hours(self):
        initial = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
        fixture = _parity.PlantExecutionRolloutParityTests
        plant = fixture._new_plant(self, initial)
        config = fixture._rollout_config(plant, target_slew_enabled=True)
        rollout_state = fixture._rollout_state(initial)
        limiter = TargetSlewLimiter(
            dt=1.0,
            rho=plant.rho,
            max_capacity=plant.tank_capacity,
            target_slew_rate_m3_min=10.0,
            enabled=True,
        )
        limiter.reset(initial)
        requested_targets = self._requested_targets(initial, plant.tank_capacity)
        self.assertEqual(
            len(requested_targets) * STAGE_DURATION_S,
            SIX_HOURS_S,
        )

        plant_masses = np.empty((SIX_HOURS_S, 3), dtype=float)
        rollout_masses = np.empty_like(plant_masses)
        plant_targets = np.empty_like(plant_masses)
        rollout_targets = np.empty_like(plant_masses)
        plant_net_rates = np.empty_like(plant_masses)
        rollout_net_rates = np.empty_like(plant_masses)
        plant_command_rates = np.empty_like(plant_masses)
        rollout_command_rates = np.empty_like(plant_masses)
        plant_latches = np.empty((SIX_HOURS_S, 3), dtype=bool)
        rollout_latches = np.empty_like(plant_latches)
        plant_active = np.empty_like(plant_latches)
        rollout_active = np.empty_like(plant_latches)
        plant_cumulative_volume = np.empty(SIX_HOURS_S, dtype=float)
        rollout_cumulative_volume = np.empty(SIX_HOURS_S, dtype=float)

        plant_volume_m3 = 0.0
        rollout_volume_m3 = 0.0
        for second in range(SIX_HOURS_S):
            stage = second // STAGE_DURATION_S
            requested = requested_targets[stage]
            shaped_target, _ = limiter.update(requested)

            plant.set_ballast_target(*shaped_target)
            _, info = plant.step(0.0, 0.0, 1.0, float(second))
            rollout = simulate_execution_step(rollout_state, requested, config)
            rollout_state = rollout.state

            plant_masses[second] = info["tank_masses"]
            rollout_masses[second] = rollout_state.masses_kg
            plant_targets[second] = info["target_ballast_mass"]
            rollout_targets[second] = rollout.shaped_target_kg
            plant_net_rates[second] = info["pump_net_rate_m3_min"]
            rollout_net_rates[second] = rollout_state.pump_rates_m3_min
            plant_command_rates[second] = info["pump_rate_smoothed_m3_min"]
            rollout_command_rates[second] = rollout_state.pump_command_rates_m3_min
            plant_latches[second] = info["pump_latched"]
            rollout_latches[second] = rollout_state.pump_latched
            plant_active[second] = np.asarray(info["pump_active"], dtype=bool)
            rollout_active[second] = (
                rollout_state.pump_command_rates_m3_min > 1e-9
            )

            plant_volume_m3 += float(
                np.sum(np.abs(info["tank_mass_delta_kg"])) / plant.rho
            )
            rollout_volume_m3 += rollout.transferred_volume_m3
            plant_cumulative_volume[second] = plant_volume_m3
            rollout_cumulative_volume[second] = rollout_volume_m3

        np.testing.assert_allclose(
            rollout_masses,
            plant_masses,
            rtol=0.0,
            atol=1e-7,
            err_msg="three-tank masses diverged during the six-hour execution",
        )
        np.testing.assert_allclose(
            rollout_targets,
            plant_targets,
            rtol=0.0,
            atol=1e-9,
            err_msg="rate-limited targets diverged during the six-hour execution",
        )
        np.testing.assert_allclose(
            rollout_net_rates,
            plant_net_rates,
            rtol=0.0,
            atol=1e-9,
            err_msg="signed pump flows diverged during the six-hour execution",
        )
        np.testing.assert_allclose(
            rollout_command_rates,
            plant_command_rates,
            rtol=0.0,
            atol=1e-9,
            err_msg="pump command rates diverged during the six-hour execution",
        )
        np.testing.assert_array_equal(
            rollout_latches,
            plant_latches,
            err_msg="pump latch states diverged during the six-hour execution",
        )
        np.testing.assert_array_equal(
            rollout_active,
            plant_active,
            err_msg="pump active states diverged during the six-hour execution",
        )
        np.testing.assert_allclose(
            rollout_cumulative_volume,
            plant_cumulative_volume,
            rtol=0.0,
            atol=1e-7,
            err_msg="cumulative pumped volume diverged during the six-hour execution",
        )

        self.assertGreater(plant_cumulative_volume[-1], 0.0)
        self.assertTrue(np.all(np.any(plant_net_rates > 0.0, axis=0)))
        self.assertTrue(np.all(np.any(plant_net_rates < 0.0, axis=0)))
        self.assertTrue(np.all(np.any(plant_latches, axis=0)))
        self.assertTrue(np.all(np.any(~plant_latches, axis=0)))
        self.assertTrue(np.all(plant_masses >= 0.0))
        self.assertTrue(np.all(rollout_masses >= 0.0))
        self.assertTrue(np.all(plant_masses <= plant.tank_capacity))
        self.assertTrue(np.all(rollout_masses <= plant.tank_capacity))
        self.assertTrue(np.all(plant_targets >= 0.0))
        self.assertTrue(np.all(rollout_targets >= 0.0))
        self.assertTrue(np.all(plant_targets <= plant.tank_capacity))
        self.assertTrue(np.all(rollout_targets <= plant.tank_capacity))


if __name__ == "__main__":
    unittest.main()
