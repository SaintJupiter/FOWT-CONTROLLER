import ast
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import IncrementalState, RotorGeneralizedLoad, RotorNormalLoad
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
)
from wind_prediction.physical_execution_platform_path import (
    advance_physical_execution_platform_path,
)

from tests.test_preview_mpc_runtime import (
    CAPACITIES,
    COORDINATES,
    REFERENCE_MASSES,
    runtime,
)


def _rotor_load(x_force_n: float) -> RotorGeneralizedLoad:
    generalized = np.zeros(6)
    generalized[0] = x_force_n
    force = np.zeros(3)
    force[0] = x_force_n
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.zeros(3),
        normal_load=RotorNormalLoad(
            thrust_n=abs(x_force_n),
            force_platform_n=force,
        ),
        generalized_load_platform=generalized,
    )


class PhysicalExecutionPlatformPathBoundaryTests(unittest.TestCase):
    def test_realised_rotor_load_is_sampled_along_the_interval(self):
        config = ExecutionRolloutConfig(
            block_duration_s=20.0,
            internal_step_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=0.2,
            tank_capacity_kg=float(CAPACITIES[0]),
            pump_rate_schedule_m3_min=((0.0, 0.2), (1.0e9, 0.2)),
        )
        execution_state = ExecutionRolloutState(
            masses_kg=REFERENCE_MASSES,
            target_masses_kg=REFERENCE_MASSES,
            primary_target_kg=REFERENCE_MASSES,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )

        path = advance_physical_execution_platform_path(
            platform_state=IncrementalState.zeros(),
            execution_state=execution_state,
            execution_request=ExecutionRolloutRequest.release_to_current(),
            execution_config=config,
            runtime_assembly=runtime(),
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
            rotor_load=_rotor_load(0.0),
            rotor_load_substeps=(_rotor_load(25.0), _rotor_load(75.0)),
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=20.0,
        )

        self.assertEqual(len(path.substeps), 2)
        self.assertAlmostEqual(path.substeps[0].platform_step.loads.wind[0], 25.0)
        self.assertAlmostEqual(path.substeps[1].platform_step.loads.wind[0], 75.0)

    def test_module_does_not_import_legacy_decision_layers(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_execution_platform_path.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "controller_core",
            "forecast_action_policy",
            "decision_demand",
            "action_target",
            "action_plan",
            "candidate",
            "planner",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )

    def test_first_interval_rollout_does_not_import_legacy_decision_layers(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_first_interval_lifecycle_rollout.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "controller_core",
            "forecast_action_policy",
            "decision_demand",
            "action_target",
            "action_plan",
            "candidate",
            "planner",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
