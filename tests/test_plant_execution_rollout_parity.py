import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = ROOT / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from core_model import FloatingPlatform  # noqa: E402
from target_execution import TargetSlewLimiter  # noqa: E402
from wind_prediction.execution_rollout import (  # noqa: E402
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    simulate_execution_step,
)


FROZEN_PUMP_CFG = {
    "pump_stop_err_kg": 300.0,
    "pump_restart_err_kg": 500.0,
    "pump_min_on_s": 20.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 10.0,
    "pump_global_quiet_backlog_kg": float("inf"),
    "pump_target_quiet_rate_kg_s": float("inf"),
    "pump_global_quiet_hold_s": 0.0,
    "pump_ramp_up_m3_min_per_s": 2.0,
    "pump_ramp_down_m3_min_per_s": 3.0,
}

FROZEN_RATE_SCHEDULE_M3_MIN = (
    (0.0, 0.0),
    (200.0, 4.0),
    (300.0, 6.0),
    (500.0, 8.0),
    (700.0, 10.0),
    (1000.0, 12.0),
    (2000.0, 14.0),
    (3000.0, 15.0),
)


class PlantExecutionRolloutParityTests(unittest.TestCase):
    def _new_plant(self, masses_kg):
        with redirect_stdout(io.StringIO()):
            plant = FloatingPlatform(
                ROOT / "tests" / "missing_mooring_for_pump_parity.xlsx",
                pump_cfg=FROZEN_PUMP_CFG,
                allow_linear_mooring_fallback=True,
            )
        plant.force_ballast_mass(*np.asarray(masses_kg, dtype=float))
        return plant

    @staticmethod
    def _rollout_config(plant, *, target_slew_enabled=True):
        return ExecutionRolloutConfig(
            block_duration_s=1.0,
            water_density_kg_m3=plant.rho,
            max_pump_rate_m3_min=15.0,
            target_slew_enabled=target_slew_enabled,
            target_slew_rate_m3_min=10.0,
            stop_error_kg=300.0,
            tank_capacity_kg=plant.tank_capacity,
            internal_step_s=1.0,
            restart_error_kg=500.0,
            min_on_s=20.0,
            min_off_s=12.0,
            near_target_hold_s=10.0,
            ramp_up_m3_min_per_s=2.0,
            ramp_down_m3_min_per_s=3.0,
            pump_rate_schedule_m3_min=FROZEN_RATE_SCHEDULE_M3_MIN,
        )

    @staticmethod
    def _rollout_state(masses_kg, *, target_kg=None):
        masses = np.asarray(masses_kg, dtype=float)
        target = masses if target_kg is None else np.asarray(target_kg, dtype=float)
        return ExecutionRolloutState(
            masses_kg=masses,
            target_masses_kg=target,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            primary_target_kg=target,
            pump_on_elapsed_s=np.zeros(3),
            pump_off_elapsed_s=np.full(3, 12.0),
            pump_near_target_s=np.zeros(3),
            pump_command_rates_m3_min=np.zeros(3),
            last_flow_directions=np.zeros(3),
        )

    def _assert_state_matches(self, plant, info, rollout_state, *, second):
        prefix = f"physical mismatch at second {second}"
        np.testing.assert_allclose(
            rollout_state.masses_kg,
            plant.current_ballast_mass,
            rtol=0.0,
            atol=1e-7,
            err_msg=f"{prefix}: tank mass",
        )
        np.testing.assert_allclose(
            rollout_state.pump_rates_m3_min,
            info["pump_net_rate_m3_min"],
            rtol=0.0,
            atol=1e-9,
            err_msg=f"{prefix}: signed pump flow",
        )
        np.testing.assert_allclose(
            rollout_state.pump_command_rates_m3_min,
            info["pump_rate_smoothed_m3_min"],
            rtol=0.0,
            atol=1e-9,
            err_msg=f"{prefix}: commanded pump flow",
        )
        np.testing.assert_array_equal(
            rollout_state.pump_latched,
            info["pump_latched"],
            err_msg=f"{prefix}: pump latch",
        )
        for name, plant_key in (
            ("pump_on_elapsed_s", "pump_on_elapsed_s"),
            ("pump_off_elapsed_s", "pump_off_elapsed_s"),
            ("pump_near_target_s", "pump_near_target_s"),
        ):
            np.testing.assert_allclose(
                getattr(rollout_state, name),
                info[plant_key],
                rtol=0.0,
                atol=1e-9,
                err_msg=f"{prefix}: {name}",
            )

    def test_target_slew_pump_state_and_direction_reversal_match_each_second(self):
        initial = np.array([1108000.0, 1362000.0, 1362000.0])
        plant = self._new_plant(initial)
        config = self._rollout_config(plant, target_slew_enabled=True)
        rollout_state = self._rollout_state(initial)
        limiter = TargetSlewLimiter(
            dt=1.0,
            rho=plant.rho,
            max_capacity=plant.tank_capacity,
            target_slew_rate_m3_min=10.0,
            enabled=True,
        )
        limiter.reset(initial)

        positive = initial + np.array([8000.0, -6000.0, 4000.0])
        reversed_target = initial + np.array([-7000.0, 5000.0, -3000.0])
        final_target = initial + np.array([1500.0, -1200.0, 900.0])
        seen_directions = [set(), set(), set()]

        for second in range(180):
            if second < 55:
                requested = positive
            elif second < 135:
                requested = reversed_target
            else:
                requested = final_target

            shaped_target, _ = limiter.update(requested)
            plant.set_ballast_target(*shaped_target)
            _, info = plant.step(0.0, 0.0, 1.0, float(second))
            rollout = simulate_execution_step(rollout_state, requested, config)
            rollout_state = rollout.state

            np.testing.assert_allclose(
                rollout.shaped_target_kg,
                shaped_target,
                rtol=0.0,
                atol=1e-9,
                err_msg=f"target slew mismatch at second {second}",
            )
            self._assert_state_matches(
                plant,
                info,
                rollout_state,
                second=second,
            )
            for pump_idx, direction in enumerate(np.sign(info["pump_net_rate_m3_min"])):
                if direction:
                    seen_directions[pump_idx].add(int(direction))

        for pump_idx, directions in enumerate(seen_directions):
            self.assertEqual(
                directions,
                {-1, 1},
                f"pump {pump_idx + 1} did not exercise both flow directions",
            )

    def test_restart_threshold_is_strictly_greater_than_500_kg(self):
        initial = np.array([1108000.0, 1362000.0, 1362000.0])
        plant = self._new_plant(initial)
        config = self._rollout_config(plant, target_slew_enabled=False)
        rollout_state = self._rollout_state(initial)

        at_boundary = initial + np.array([500.0, 0.0, 0.0])
        plant.set_ballast_target(*at_boundary)
        _, info = plant.step(0.0, 0.0, 1.0, 0.0)
        rollout = simulate_execution_step(rollout_state, at_boundary, config)
        rollout_state = rollout.state
        self._assert_state_matches(plant, info, rollout_state, second=0)
        self.assertFalse(bool(info["pump_latched"][0]))

        above_boundary = initial + np.array([500.001, 0.0, 0.0])
        plant.set_ballast_target(*above_boundary)
        _, info = plant.step(0.0, 0.0, 1.0, 1.0)
        rollout = simulate_execution_step(rollout_state, above_boundary, config)
        self._assert_state_matches(plant, info, rollout.state, second=1)
        self.assertTrue(bool(info["pump_latched"][0]))

    def test_stop_threshold_is_strictly_less_than_300_kg_after_dwell(self):
        initial = np.array([1108000.0, 1362000.0, 1362000.0])
        plant = self._new_plant(initial)
        config = self._rollout_config(plant, target_slew_enabled=False)
        rollout_state = self._rollout_state(initial)

        far_target = initial + np.array([20000.0, 0.0, 0.0])
        for second in range(22):
            plant.set_ballast_target(*far_target)
            _, info = plant.step(0.0, 0.0, 1.0, float(second))
            rollout = simulate_execution_step(rollout_state, far_target, config)
            rollout_state = rollout.state
            self._assert_state_matches(plant, info, rollout_state, second=second)
        self.assertTrue(bool(info["pump_latched"][0]))

        exact_stop_boundary = plant.current_ballast_mass.copy()
        exact_stop_boundary[0] += 300.0
        plant.set_ballast_target(*exact_stop_boundary)
        _, info = plant.step(0.0, 0.0, 1.0, 22.0)
        rollout = simulate_execution_step(
            rollout_state,
            exact_stop_boundary,
            config,
        )
        rollout_state = rollout.state
        self._assert_state_matches(plant, info, rollout_state, second=22)
        self.assertTrue(bool(info["pump_latched"][0]))

        below_stop_boundary = plant.current_ballast_mass.copy()
        below_stop_boundary[0] += 299.0
        for second in range(23, 34):
            plant.set_ballast_target(*below_stop_boundary)
            _, info = plant.step(0.0, 0.0, 1.0, float(second))
            rollout = simulate_execution_step(
                rollout_state,
                below_stop_boundary,
                config,
            )
            rollout_state = rollout.state
            self._assert_state_matches(plant, info, rollout_state, second=second)
        self.assertFalse(bool(info["pump_latched"][0]))


if __name__ == "__main__":
    unittest.main()
