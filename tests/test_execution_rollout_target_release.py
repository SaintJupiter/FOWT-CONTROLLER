import unittest

import numpy as np

from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionTargetOperation,
    simulate_execution_step,
)


class ExecutionRolloutTargetReleaseTests(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "block_duration_s": 1.0,
            "water_density_kg_m3": 60.0,
            "max_pump_rate_m3_min": 10.0,
            "target_slew_enabled": True,
            "target_slew_rate_m3_min": 1.0,
            "stop_error_kg": 10.0,
            "tank_capacity_kg": 1000.0,
            "internal_step_s": 1.0,
            "restart_error_kg": 20.0,
            "min_on_s": 3.0,
            "min_off_s": 2.0,
            "near_target_hold_s": 2.0,
            "ramp_up_m3_min_per_s": 10.0,
            "ramp_down_m3_min_per_s": 3.0,
            "pump_rate_schedule_m3_min": ((0.0, 0.0), (1000.0, 10.0)),
        }
        values.update(overrides)
        return ExecutionRolloutConfig(**values)

    @staticmethod
    def state(*, latched=True, on_elapsed_s=0.0, command_rate=9.0):
        return ExecutionRolloutState(
            masses_kg=np.array([100.0, 200.0, 300.0]),
            target_masses_kg=np.array([500.0, 200.0, 300.0]),
            primary_target_kg=np.array([800.0, 200.0, 300.0]),
            pump_rates_m3_min=np.array([command_rate, 0.0, 0.0]),
            pump_latched=np.array([latched, False, False]),
            pump_on_elapsed_s=np.array([on_elapsed_s, 0.0, 0.0]),
            pump_off_elapsed_s=np.array([0.0, 2.0, 2.0]),
            pump_near_target_s=np.zeros(3),
            pump_command_rates_m3_min=np.array([command_rate, 0.0, 0.0]),
            last_flow_directions=np.array([1.0, 0.0, 0.0]),
        )

    def test_track_request_is_identical_to_legacy_target_argument(self):
        cfg = self.config(min_on_s=0.0, near_target_hold_s=0.0)
        state = self.state(latched=False, command_rate=0.0)
        target = np.array([400.0, 200.0, 300.0])

        legacy = simulate_execution_step(state, target, cfg)
        explicit = simulate_execution_step(
            state,
            ExecutionRolloutRequest.track(target),
            cfg,
        )

        np.testing.assert_allclose(explicit.state.masses_kg, legacy.state.masses_kg)
        np.testing.assert_allclose(
            explicit.state.rate_limited_target_kg,
            legacy.state.rate_limited_target_kg,
        )
        np.testing.assert_allclose(
            explicit.state.primary_target_masses_kg,
            legacy.state.primary_target_masses_kg,
        )
        np.testing.assert_allclose(explicit.pump_volume_m3, legacy.pump_volume_m3)
        self.assertEqual(explicit.starts, legacy.starts)
        self.assertEqual(explicit.stops, legacy.stops)
        self.assertEqual(explicit.target_operation, ExecutionTargetOperation.TRACK)
        self.assertFalse(explicit.target_slew_reset)

    def test_release_resets_primary_and_rate_limited_targets_to_current_masses(self):
        cfg = self.config()
        state = self.state()

        result = simulate_execution_step(
            state,
            ExecutionRolloutRequest.release_to_current(),
            cfg,
        )

        np.testing.assert_allclose(result.requested_target_kg, state.masses_kg)
        np.testing.assert_allclose(result.shaped_target_kg, state.masses_kg)
        np.testing.assert_allclose(
            result.state.primary_target_masses_kg,
            state.masses_kg,
        )
        np.testing.assert_allclose(
            result.state.rate_limited_target_kg,
            state.masses_kg,
        )
        np.testing.assert_allclose(result.mass_delta_kg, np.zeros(3))
        self.assertEqual(
            result.target_operation,
            ExecutionTargetOperation.RELEASE_TO_CURRENT,
        )
        self.assertTrue(result.target_slew_reset)

    def test_release_keeps_latch_until_normal_stop_dwell_is_satisfied(self):
        cfg = self.config()
        request = ExecutionRolloutRequest.release_to_current()

        first = simulate_execution_step(self.state(), request, cfg)
        self.assertTrue(first.state.pump_latched[0])
        self.assertEqual(first.stops, 0)
        self.assertAlmostEqual(first.state.pump_on_elapsed_s[0], 1.0)
        self.assertAlmostEqual(first.state.pump_near_target_s[0], 1.0)
        self.assertAlmostEqual(first.state.pump_command_rates_m3_min[0], 6.0)

        second = simulate_execution_step(first.state, request, cfg)
        self.assertTrue(second.state.pump_latched[0])
        self.assertEqual(second.stops, 0)
        self.assertAlmostEqual(second.state.pump_on_elapsed_s[0], 2.0)
        self.assertAlmostEqual(second.state.pump_near_target_s[0], 2.0)

        third = simulate_execution_step(second.state, request, cfg)
        self.assertFalse(third.state.pump_latched[0])
        self.assertEqual(third.stops, 1)
        np.testing.assert_array_equal(third.stop_counts, [1, 0, 0])
        np.testing.assert_allclose(third.mass_delta_kg, np.zeros(3))

    def test_release_does_not_restart_an_idle_pump_from_stale_targets(self):
        cfg = self.config(block_duration_s=4.0)
        state = self.state(latched=False, command_rate=0.0)

        result = simulate_execution_step(
            state,
            ExecutionRolloutRequest.release_to_current(),
            cfg,
        )

        self.assertEqual(result.starts, 0)
        self.assertEqual(result.stops, 0)
        self.assertFalse(result.state.pump_latched[0])
        self.assertAlmostEqual(result.transferred_volume_m3, 0.0)
        np.testing.assert_allclose(result.state.masses_kg, state.masses_kg)

    def test_release_request_rejects_an_external_target(self):
        with self.assertRaisesRegex(ValueError, "derives its target"):
            ExecutionRolloutRequest(
                operation=ExecutionTargetOperation.RELEASE_TO_CURRENT,
                target_masses_kg=(1.0, 2.0, 3.0),
            )


if __name__ == "__main__":
    unittest.main()
