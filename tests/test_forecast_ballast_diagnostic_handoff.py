import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    IncrementalState,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    diagnose_forecast_ballast_redistribution,
)
from wind_prediction.controller_platform_handoff import advance_controller_platform_substep
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
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


def _zero_rotor_load() -> RotorGeneralizedLoad:
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.zeros(3),
        normal_load=RotorNormalLoad(
            thrust_n=0.0,
            force_platform_n=np.zeros(3),
        ),
        generalized_load_platform=np.zeros(6),
    )


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=1.0,
        water_density_kg_m3=1025.0,
        max_pump_rate_m3_min=15.0,
        target_slew_enabled=False,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 15.0),),
        tank_capacity_kg=TANK_CAPACITY_KG,
        internal_step_s=1.0,
    )


def _execution_state() -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=REFERENCE_TANK_MASSES_KG,
        primary_target_kg=REFERENCE_TANK_MASSES_KG,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
    )


class ForecastBallastDiagnosticHandoffTests(unittest.TestCase):
    def test_zero_net_endpoint_target_can_have_transient_mass_change_with_independent_pumps(self):
        runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            REFERENCE_MANIFEST,
            np.zeros((6, 6)),
        )
        self.assertEqual(
            runtime_assembly.provenance,
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        current_load = np.zeros(6)
        future_load = np.array([0.0, 0.0, 0.0, 0.0, 1.0e8, 0.0])
        diagnostic = diagnose_forecast_ballast_redistribution(
            matrices=runtime_assembly.base_matrices,
            current_generalized_load=current_load,
            future_generalized_load=future_load,
            actual_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        request = ExecutionRolloutRequest.track(
            diagnostic.allocation.target_tank_masses_kg
        )
        initial_execution = _execution_state()
        config = _execution_config()

        self.assertFalse(diagnostic.allocation.capacity_limited)
        self.assertAlmostEqual(
            float(np.sum(diagnostic.allocation.tank_mass_deltas_kg)),
            0.0,
            places=9,
        )

        first = advance_controller_platform_substep(
            platform_state=IncrementalState.zeros(),
            execution_state=initial_execution,
            execution_request=request,
            execution_config=config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=_zero_rotor_load(),
            wave_load=np.zeros(6),
            other_load=current_load,
            duration_s=1.0,
        )

        np.testing.assert_allclose(
            first.ballast_snapshot.actual_tank_masses_kg,
            initial_execution.actual_masses_kg,
        )
        np.testing.assert_allclose(first.platform_step.loads.other, current_load)
        self.assertGreater(np.linalg.norm(first.execution_step.mass_delta_kg), 0.0)
        self.assertNotAlmostEqual(
            float(np.sum(first.execution_step.mass_delta_kg)),
            0.0,
            places=9,
        )

        second = advance_controller_platform_substep(
            platform_state=first.next_platform_state,
            execution_state=first.next_execution_state,
            execution_request=request,
            execution_config=config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=_zero_rotor_load(),
            wave_load=np.zeros(6),
            other_load=future_load,
            duration_s=1.0,
        )

        np.testing.assert_allclose(
            second.ballast_snapshot.actual_tank_masses_kg,
            first.next_execution_state.actual_masses_kg,
        )
        np.testing.assert_allclose(second.platform_step.loads.other, future_load)
        self.assertEqual(
            second.ballast_snapshot.runtime_provenance,
            runtime_assembly.provenance,
        )


if __name__ == "__main__":
    unittest.main()
