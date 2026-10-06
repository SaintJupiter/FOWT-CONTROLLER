import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import BallastEndpointPathSample
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
)
from wind_prediction.physical_endpoint_reachability import (
    evaluate_physical_endpoint_reachability,
)


def _config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=30.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=1.0,
        tank_capacity_kg=2_000.0,
        internal_step_s=1.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (2_000.0, 1.0)),
    )


def _state() -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=[1_000.0, 1_000.0, 1_000.0],
        target_masses_kg=[1_000.0, 1_000.0, 1_000.0],
        primary_target_kg=[1_000.0, 1_000.0, 1_000.0],
        pump_rates_m3_min=[0.0, 0.0, 0.0],
        pump_latched=[False, False, False],
        pump_off_elapsed_s=[100.0, 100.0, 100.0],
    )


def _endpoint() -> BallastEndpointPathSample:
    return BallastEndpointPathSample(
        fraction=1.0,
        hypothetical_tank_masses_kg=[1_900.0, 500.0, 600.0],
        mass_delta_from_actual_kg=[900.0, -500.0, -400.0],
        net_mass_delta_from_actual_kg=0.0,
        within_capacity=True,
    )


class PhysicalEndpointReachabilityTests(unittest.TestCase):
    def test_returns_actual_reached_mass_remaining_error_and_transient_total_change(self):
        result = evaluate_physical_endpoint_reachability(
            lead_time_s=600.0,
            endpoint_sample=_endpoint(),
            execution_state=_state(),
            execution_config=_config(),
            execution_duration_s=30.0,
        )

        np.testing.assert_allclose(
            result.execution_step.state.actual_masses_kg,
            [1_500.0, 500.0, 600.0],
        )
        np.testing.assert_allclose(
            result.execution_start_state.actual_masses_kg,
            [1_000.0, 1_000.0, 1_000.0],
        )
        np.testing.assert_allclose(
            result.execution_start_state.primary_target_masses_kg,
            [1_000.0, 1_000.0, 1_000.0],
        )
        np.testing.assert_array_equal(
            result.execution_start_state.pump_latched,
            [False, False, False],
        )
        np.testing.assert_allclose(
            result.remaining_mass_to_endpoint_kg,
            [400.0, 0.0, 0.0],
            atol=1e-9,
        )
        self.assertFalse(result.endpoint_reached)
        self.assertAlmostEqual(result.executed_net_mass_delta_kg, -400.0)
        self.assertAlmostEqual(
            result.executed_net_mass_delta_kg,
            np.sum(result.execution_step.mass_delta_kg),
        )

    def test_rejects_duration_beyond_lead_or_endpoint_from_another_actual_state(self):
        with self.assertRaisesRegex(ValueError, "must not exceed lead_time_s"):
            evaluate_physical_endpoint_reachability(
                lead_time_s=20.0,
                endpoint_sample=_endpoint(),
                execution_state=_state(),
                execution_config=_config(),
                execution_duration_s=30.0,
            )
        with self.assertRaisesRegex(ValueError, "execution_config.block_duration_s"):
            evaluate_physical_endpoint_reachability(
                lead_time_s=600.0,
                endpoint_sample=_endpoint(),
                execution_state=_state(),
                execution_config=_config(),
                execution_duration_s=31.0,
            )
        mismatched = ExecutionRolloutState(
            masses_kg=[999.0, 1_000.0, 1_000.0],
            target_masses_kg=[999.0, 1_000.0, 1_000.0],
            primary_target_kg=[999.0, 1_000.0, 1_000.0],
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
            pump_off_elapsed_s=[100.0, 100.0, 100.0],
        )
        with self.assertRaisesRegex(ValueError, "formed from execution_state"):
            evaluate_physical_endpoint_reachability(
                lead_time_s=600.0,
                endpoint_sample=_endpoint(),
                execution_state=mismatched,
                execution_config=_config(),
                execution_duration_s=30.0,
            )

    def test_rejects_a_step_with_tampered_pump_timing(self):
        result = evaluate_physical_endpoint_reachability(
            lead_time_s=600.0,
            endpoint_sample=_endpoint(),
            execution_state=_state(),
            execution_config=_config(),
            execution_duration_s=30.0,
        )

        with self.assertRaisesRegex(ValueError, "must be reproducible"):
            replace(
                result,
                execution_step=replace(
                    result.execution_step,
                    active_time_s=result.execution_step.active_time_s + 1.0,
                ),
            )

    def test_module_does_not_import_controller_or_candidate_ranking(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_endpoint_reachability.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("controller", "candidate", "planner")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
