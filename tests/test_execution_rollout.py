import unittest

import numpy as np

from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    simulate_execution_step,
)


CONSTANT_ONE_M3_MIN = ((0.0, 1.0), (10000.0, 1.0))


class ExecutionRolloutTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ExecutionRolloutConfig(
            block_duration_s=60.0,
            water_density_kg_m3=1000.0,
            max_pump_rate_m3_min=1.0,
            target_slew_enabled=True,
            target_slew_rate_m3_min=2.0,
            stop_error_kg=0.0,
            tank_capacity_kg=10000.0,
            internal_step_s=1.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=CONSTANT_ONE_M3_MIN,
        )

    @staticmethod
    def state(
        masses,
        target=None,
        rates=None,
        latched=None,
        **kwargs,
    ):
        masses = np.asarray(masses, dtype=float)
        return ExecutionRolloutState(
            masses_kg=masses,
            target_masses_kg=np.asarray(target if target is not None else masses),
            pump_rates_m3_min=np.asarray(
                rates if rates is not None else np.zeros(3), dtype=float
            ),
            pump_latched=np.asarray(
                latched if latched is not None else np.zeros(3), dtype=bool
            ),
            **kwargs,
        )

    def test_target_and_mass_motion_keep_existing_api_but_use_internal_steps(self):
        state = self.state([1000.0, 1000.0, 1000.0])
        result = simulate_execution_step(
            state,
            np.array([5000.0, 1000.0, 1000.0]),
            self.cfg,
        )

        self.assertAlmostEqual(result.shaped_target_kg[0], 3000.0)
        self.assertAlmostEqual(result.mass_delta_kg[0], 1000.0)
        self.assertAlmostEqual(result.transferred_volume_m3, 1.0)
        self.assertEqual(result.starts, 1)
        self.assertFalse(result.target_reached)
        np.testing.assert_allclose(
            result.state.primary_target_masses_kg,
            [5000.0, 1000.0, 1000.0],
        )
        np.testing.assert_allclose(
            result.state.rate_limited_target_kg,
            result.shaped_target_kg,
        )

    def test_capacity_clip_and_hold_report_actual_pumped_volume(self):
        state = self.state([9900.0, 1000.0, 1000.0])
        result = simulate_execution_step(
            state,
            np.array([12000.0, 1000.0, 1000.0]),
            self.cfg,
        )
        self.assertEqual(result.state.masses_kg[0], self.cfg.tank_capacity_kg)
        self.assertAlmostEqual(result.pump_volume_m3[0], 0.1)

        held = simulate_execution_step(
            result.state,
            result.state.masses_kg,
            self.cfg,
        )
        self.assertAlmostEqual(held.transferred_volume_m3, 0.0)
        np.testing.assert_allclose(held.mass_delta_kg, np.zeros(3))

    def test_disabled_target_slew_applies_requested_target_immediately(self):
        cfg = ExecutionRolloutConfig(
            block_duration_s=1.0,
            water_density_kg_m3=60.0,
            max_pump_rate_m3_min=1.0,
            target_slew_enabled=False,
            target_slew_rate_m3_min=0.001,
            stop_error_kg=0.0,
            tank_capacity_kg=1000.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=CONSTANT_ONE_M3_MIN,
        )
        result = simulate_execution_step(
            self.state([0.0, 0.0, 0.0]),
            [100.0, 0.0, 0.0],
            cfg,
        )

        self.assertAlmostEqual(result.shaped_target_kg[0], 100.0)
        self.assertAlmostEqual(result.mass_delta_kg[0], 1.0)

    def test_restart_hysteresis_and_minimum_off_time_gate_start(self):
        cfg = ExecutionRolloutConfig(
            block_duration_s=2.0,
            water_density_kg_m3=60.0,
            max_pump_rate_m3_min=1.0,
            target_slew_rate_m3_min=1000.0,
            stop_error_kg=10.0,
            tank_capacity_kg=1000.0,
            restart_error_kg=50.0,
            min_on_s=0.0,
            min_off_s=3.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=CONSTANT_ONE_M3_MIN,
        )
        waiting = self.state(
            [0.0, 0.0, 0.0],
            target=[100.0, 0.0, 0.0],
            pump_off_elapsed_s=np.zeros(3),
        )

        first = simulate_execution_step(waiting, [100.0, 0.0, 0.0], cfg)
        self.assertEqual(first.starts, 0)
        self.assertFalse(first.state.pump_latched[0])
        self.assertEqual(first.transferred_volume_m3, 0.0)

        second = simulate_execution_step(first.state, [100.0, 0.0, 0.0], cfg)
        self.assertEqual(second.starts, 1)
        self.assertTrue(second.state.pump_latched[0])
        self.assertAlmostEqual(second.pump_runtime_s[0], 2.0)

        in_deadband = self.state(
            [0.0, 0.0, 0.0],
            target=[40.0, 0.0, 0.0],
        )
        held_off = simulate_execution_step(in_deadband, [40.0, 0.0, 0.0], cfg)
        self.assertEqual(held_off.starts, 0)
        self.assertFalse(held_off.state.pump_latched[0])

    def test_minimum_on_and_near_target_dwell_gate_stop(self):
        cfg = ExecutionRolloutConfig(
            block_duration_s=2.0,
            water_density_kg_m3=60.0,
            stop_error_kg=10.0,
            tank_capacity_kg=1000.0,
            restart_error_kg=20.0,
            min_on_s=3.0,
            min_off_s=0.0,
            near_target_hold_s=2.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
        )
        running = self.state(
            [100.0, 0.0, 0.0],
            latched=[True, False, False],
            pump_on_elapsed_s=np.zeros(3),
            pump_off_elapsed_s=np.zeros(3),
            pump_near_target_s=np.zeros(3),
        )

        first = simulate_execution_step(running, [100.0, 0.0, 0.0], cfg)
        self.assertTrue(first.state.pump_latched[0])
        self.assertEqual(first.stops, 0)
        self.assertAlmostEqual(first.state.pump_on_elapsed_s[0], 2.0)
        self.assertAlmostEqual(first.state.pump_near_target_s[0], 2.0)

        second = simulate_execution_step(first.state, [100.0, 0.0, 0.0], cfg)
        self.assertFalse(second.state.pump_latched[0])
        self.assertEqual(second.stops, 1)
        np.testing.assert_array_equal(second.stop_counts, [1, 0, 0])

    def test_flow_ramp_sets_real_volume_runtime_and_signed_flow(self):
        cfg = ExecutionRolloutConfig(
            block_duration_s=3.0,
            water_density_kg_m3=60.0,
            max_pump_rate_m3_min=6.0,
            target_slew_rate_m3_min=1000.0,
            stop_error_kg=0.0,
            tank_capacity_kg=1000.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=2.0,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=((0.0, 6.0), (1000.0, 6.0)),
        )
        result = simulate_execution_step(
            self.state([0.0, 0.0, 0.0], target=[100.0, 0.0, 0.0]),
            [100.0, 0.0, 0.0],
            cfg,
        )

        self.assertAlmostEqual(result.mass_delta_kg[0], 12.0)
        self.assertAlmostEqual(result.pump_volume_m3[0], 0.2)
        self.assertAlmostEqual(result.transferred_volume_m3, 0.2)
        self.assertAlmostEqual(result.pump_runtime_s[0], 3.0)
        self.assertAlmostEqual(result.active_time_s, 3.0)
        self.assertAlmostEqual(result.state.signed_flow_m3_min[0], 6.0)

    def test_direction_reversal_and_per_tank_metrics_are_counted(self):
        cfg = ExecutionRolloutConfig(
            block_duration_s=1.0,
            water_density_kg_m3=60.0,
            max_pump_rate_m3_min=1.0,
            target_slew_rate_m3_min=1000.0,
            stop_error_kg=0.0,
            tank_capacity_kg=1000.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=CONSTANT_ONE_M3_MIN,
        )
        state = self.state(
            [50.0, 50.0, 50.0],
            target=[0.0, 100.0, 50.0],
            rates=[1.0, -1.0, 0.0],
            latched=[True, True, False],
        )
        result = simulate_execution_step(state, [0.0, 100.0, 50.0], cfg)

        np.testing.assert_allclose(result.per_tank_mass_delta_kg, [-1.0, 1.0, 0.0])
        np.testing.assert_allclose(result.pump_volume_m3, [1.0 / 60.0, 1.0 / 60.0, 0.0])
        np.testing.assert_allclose(result.pump_runtime_s, [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result.direction_switch_counts, [1, 1, 0])
        self.assertEqual(result.direction_switches, 2)
        np.testing.assert_allclose(result.state.signed_flow_m3_min, [-1.0, 1.0, 0.0])

    def test_actual_volume_accumulates_both_sides_of_an_in_block_reversal(self):
        cfg = ExecutionRolloutConfig(
            block_duration_s=5.0,
            water_density_kg_m3=60.0,
            max_pump_rate_m3_min=1.0,
            target_slew_enabled=True,
            target_slew_rate_m3_min=2.0,
            stop_error_kg=0.0,
            tank_capacity_kg=1000.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=CONSTANT_ONE_M3_MIN,
        )
        state = self.state(
            [50.0, 0.0, 0.0],
            target=[55.0, 0.0, 0.0],
            rates=[1.0, 0.0, 0.0],
            latched=[True, False, False],
        )
        result = simulate_execution_step(state, [45.0, 0.0, 0.0], cfg)

        self.assertAlmostEqual(result.mass_delta_kg[0], -2.0)
        self.assertAlmostEqual(result.pump_volume_m3[0], 4.0 / 60.0)
        self.assertGreater(
            result.pump_volume_m3[0],
            abs(result.mass_delta_kg[0]) / cfg.water_density_kg_m3,
        )
        self.assertEqual(result.direction_switches, 1)

    def test_from_plant_info_restores_signed_flow_commands_and_timers(self):
        info = {
            "tank_masses": [100.0, 200.0, 300.0],
            "target_ballast_mass": [150.0, 150.0, 300.0],
            "target_final_kg": [500.0, 100.0, 300.0],
            "pump_net_rate_m3_min": [-1.0, 2.0, 0.0],
            "pump_rate_smoothed_m3_min": [1.5, 2.5, 0.0],
            "pump_latched": [True, True, False],
            "pump_on_elapsed_s": [4.0, 5.0, 0.0],
            "pump_off_elapsed_s": [0.0, 0.0, 8.0],
            "pump_near_target_s": [1.0, 2.0, 3.0],
        }
        state = ExecutionRolloutState.from_plant_info(info, self.cfg)

        np.testing.assert_allclose(state.signed_flow_m3_min, [-1.0, 2.0, 0.0])
        np.testing.assert_allclose(state.pump_command_rates_m3_min, [1.5, 2.5, 0.0])
        np.testing.assert_allclose(state.primary_target_kg, [500.0, 100.0, 300.0])
        np.testing.assert_allclose(state.rate_limited_target_kg, [150.0, 150.0, 300.0])
        np.testing.assert_allclose(state.pump_on_elapsed_s, [4.0, 5.0, 0.0])
        np.testing.assert_allclose(state.pump_off_elapsed_s, [0.0, 0.0, 8.0])
        np.testing.assert_allclose(state.pump_near_target_s, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
