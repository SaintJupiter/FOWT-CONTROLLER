import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from fowt_platform import (
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    assemble_ballast_model_snapshot,
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    simulate_execution_step,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)


class ExecutionToPlatformDataflowTests(unittest.TestCase):
    def runtime_assembly(self, damping):
        runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            REFERENCE_MANIFEST,
            damping,
        )
        self.assertEqual(
            runtime.provenance,
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        return runtime

    @staticmethod
    def execution_config(*, density_kg_m3: float) -> ExecutionRolloutConfig:
        return ExecutionRolloutConfig(
            block_duration_s=60.0,
            water_density_kg_m3=density_kg_m3,
            max_pump_rate_m3_min=1.0,
            target_slew_enabled=False,
            stop_error_kg=0.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
            pump_rate_schedule_m3_min=((0.0, 1.0), (20_000.0, 1.0)),
            tank_capacity_kg=TANK_CAPACITY_KG,
            internal_step_s=1.0,
        )

    @staticmethod
    def initial_execution_state() -> ExecutionRolloutState:
        return ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )

    def test_actual_pumped_mass_drives_one_consistent_platform_snapshot(self):
        density_kg_m3 = 1025.0
        execution = simulate_execution_step(
            self.initial_execution_state(),
            REFERENCE_TANK_MASSES_KG + np.array([10_000.0, 0.0, 0.0]),
            self.execution_config(density_kg_m3=density_kg_m3),
        )

        expected_delta = np.array([density_kg_m3, 0.0, 0.0])
        np.testing.assert_allclose(execution.mass_delta_kg, expected_delta)
        np.testing.assert_allclose(
            execution.state.actual_masses_kg,
            REFERENCE_TANK_MASSES_KG + expected_delta,
        )

        runtime_assembly = self.runtime_assembly(np.zeros((6, 6)))
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=execution.state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
        )

        np.testing.assert_allclose(snapshot.tank_mass_deltas_kg, expected_delta)
        baseline = runtime_assembly.base_matrices
        self.assertAlmostEqual(
            snapshot.matrices.mass[0, 0] - baseline.mass[0, 0],
            density_kg_m3,
        )
        self.assertAlmostEqual(
            snapshot.incremental_ballast_load[2],
            -9.81 * density_kg_m3,
        )
        np.testing.assert_allclose(
            snapshot.incremental_ballast_load[4],
            9.81 * density_kg_m3 * TANK_COORDINATES_M[0, 0],
            rtol=0.0,
            atol=1.0e-5,
        )

        loads = IncrementalLoads(
            wind=np.zeros(6),
            wave=np.zeros(6),
            ballast=snapshot.incremental_ballast_load,
            other=np.zeros(6),
        )
        derivative = IncrementalPlatformModel(snapshot.matrices).derivative(
            IncrementalState.zeros(),
            loads,
        )
        np.testing.assert_allclose(
            snapshot.matrices.mass @ derivative.velocity_rate,
            snapshot.incremental_ballast_load,
            rtol=1e-12,
            atol=1e-8,
        )

    def test_two_execution_periods_preserve_actual_mass_and_balanced_transfer(self):
        density_kg_m3 = 1025.0
        config = self.execution_config(density_kg_m3=density_kg_m3)
        requested_target = REFERENCE_TANK_MASSES_KG + np.array(
            [-3_000.0, 3_000.0, 0.0]
        )

        first = simulate_execution_step(
            self.initial_execution_state(),
            requested_target,
            config,
        )
        second = simulate_execution_step(first.state, requested_target, config)

        per_period_delta = np.array([-density_kg_m3, density_kg_m3, 0.0])
        np.testing.assert_allclose(first.mass_delta_kg, per_period_delta)
        np.testing.assert_allclose(second.mass_delta_kg, per_period_delta)
        np.testing.assert_allclose(
            second.state.actual_masses_kg,
            REFERENCE_TANK_MASSES_KG + 2.0 * per_period_delta,
        )

        runtime_assembly = self.runtime_assembly(np.zeros((6, 6)))
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=second.state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        baseline = runtime_assembly.base_matrices

        self.assertAlmostEqual(float(np.sum(snapshot.tank_mass_deltas_kg)), 0.0)
        self.assertAlmostEqual(snapshot.incremental_ballast_load[2], 0.0)
        np.testing.assert_allclose(
            snapshot.matrices.mass[:3, :3],
            baseline.mass[:3, :3],
        )
        self.assertLess(snapshot.incremental_ballast_load[3], 0.0)
        self.assertLess(snapshot.incremental_ballast_load[4], 0.0)
        self.assertFalse(
            np.array_equal(snapshot.matrices.mass[3:, 3:], baseline.mass[3:, 3:])
        )

    def test_two_short_steps_keep_pump_and_platform_updates_causal(self):
        density_kg_m3 = 1025.0
        short_step_s = 1.0
        config = replace(
            self.execution_config(density_kg_m3=density_kg_m3),
            block_duration_s=short_step_s,
        )
        requested_target = REFERENCE_TANK_MASSES_KG + np.array(
            [-3_000.0, 3_000.0, 0.0]
        )
        damping = np.zeros((6, 6))
        runtime_assembly = self.runtime_assembly(damping)
        tank_capacities = np.full(3, TANK_CAPACITY_KG)
        platform_state = IncrementalState.zeros()
        execution_state = self.initial_execution_state()

        snapshot_n = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=execution_state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=tank_capacities,
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        platform_state = IncrementalPlatformModel(
            snapshot_n.matrices
        ).advance_frozen_step(
            platform_state,
            IncrementalLoads(
                wind=np.zeros(6),
                wave=np.zeros(6),
                ballast=snapshot_n.incremental_ballast_load,
                other=np.zeros(6),
            ),
            duration_s=short_step_s,
        )
        execution_n1 = simulate_execution_step(
            execution_state,
            requested_target,
            config,
        )
        np.testing.assert_array_equal(platform_state.position, np.zeros(6))
        np.testing.assert_array_equal(platform_state.velocity, np.zeros(6))

        snapshot_n1 = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=execution_n1.state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=tank_capacities,
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        platform_state_n2 = IncrementalPlatformModel(
            snapshot_n1.matrices
        ).advance_frozen_step(
            platform_state,
            IncrementalLoads(
                wind=np.zeros(6),
                wave=np.zeros(6),
                ballast=snapshot_n1.incremental_ballast_load,
                other=np.zeros(6),
            ),
            duration_s=short_step_s,
        )
        execution_n2 = simulate_execution_step(
            execution_n1.state,
            requested_target,
            config,
        )

        expected_step_delta = np.array(
            [-density_kg_m3 / 60.0, density_kg_m3 / 60.0, 0.0]
        )
        np.testing.assert_allclose(execution_n1.mass_delta_kg, expected_step_delta)
        np.testing.assert_allclose(execution_n2.mass_delta_kg, expected_step_delta)
        np.testing.assert_allclose(
            execution_n2.state.actual_masses_kg,
            REFERENCE_TANK_MASSES_KG + 2.0 * expected_step_delta,
        )
        self.assertGreater(np.linalg.norm(platform_state_n2.velocity[3:5]), 0.0)


if __name__ == "__main__":
    unittest.main()
