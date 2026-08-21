import unittest

import numpy as np

from fowt_platform import (
    PlatformMatrices,
    allocate_pitch_roll_moment_to_tanks,
    diagnose_forecast_ballast_redistribution,
    diagnose_pitch_roll_restoring_demand,
    sample_ballast_endpoint_path,
)


TANK_COORDINATES_M = np.array(
    [
        [2.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -2.0, -1.0],
    ]
)
CAPACITIES_KG = np.full(3, 1_000.0)
ACTUAL_MASSES_KG = np.full(3, 500.0)


def _matrices_with_pitch_roll_stiffness(stiffness_pitch_roll):
    restoring = np.eye(6)
    restoring[np.ix_([4, 3], [4, 3])] = stiffness_pitch_roll
    return PlatformMatrices(
        mass=np.eye(6),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=restoring,
        mooring_stiffness=np.zeros((6, 6)),
    )


class ForecastBallastDiagnosticTests(unittest.TestCase):
    def diagnose(self, *, current, future, actual=ACTUAL_MASSES_KG):
        return diagnose_forecast_ballast_redistribution(
            matrices=_matrices_with_pitch_roll_stiffness(np.diag([200.0, 100.0])),
            current_generalized_load=current,
            future_generalized_load=future,
            actual_tank_masses_kg=actual,
            tank_capacities_kg=CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

    def test_zero_load_change_keeps_tanks_unchanged_and_has_no_remaining_load(self):
        result = self.diagnose(
            current=[1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
            future=[1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
        )

        np.testing.assert_array_equal(
            result.allocation.target_tank_masses_kg,
            ACTUAL_MASSES_KG,
        )
        np.testing.assert_array_equal(
            result.remaining_relative_pitch_roll_load_nm,
            [0.0, 0.0],
        )

    def test_composed_diagnostic_matches_the_two_explicit_steps_and_cancels_pure_axes(self):
        current = np.zeros(6)
        future = [0.0, 0.0, 0.0, 30.0, -40.0, 0.0]
        matrices = _matrices_with_pitch_roll_stiffness(np.diag([200.0, 100.0]))

        result = diagnose_forecast_ballast_redistribution(
            matrices=matrices,
            current_generalized_load=current,
            future_generalized_load=future,
            actual_tank_masses_kg=ACTUAL_MASSES_KG,
            tank_capacities_kg=CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )
        direct_restoring = diagnose_pitch_roll_restoring_demand(
            matrices=matrices,
            current_generalized_load=current,
            future_generalized_load=future,
        )
        direct_allocation = allocate_pitch_roll_moment_to_tanks(
            requested_pitch_roll_moment_nm=direct_restoring.counteracting_pitch_roll_moment_nm,
            actual_tank_masses_kg=ACTUAL_MASSES_KG,
            tank_capacities_kg=CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

        np.testing.assert_allclose(
            result.restoring_diagnostic.relative_pitch_roll_load_nm,
            [-40.0, 30.0],
        )
        np.testing.assert_allclose(
            result.allocation.tank_mass_deltas_kg,
            direct_allocation.tank_mass_deltas_kg,
        )
        np.testing.assert_allclose(
            result.allocation.achieved_pitch_roll_moment_nm,
            direct_restoring.counteracting_pitch_roll_moment_nm,
        )
        np.testing.assert_allclose(
            result.remaining_relative_pitch_roll_load_nm,
            [0.0, 0.0],
            atol=1e-12,
        )

    def test_diagnostic_endpoint_can_be_sampled_without_entering_controller_or_pump_logic(self):
        result = self.diagnose(
            current=np.zeros(6),
            future=[0.0, 0.0, 0.0, 30.0, -40.0, 0.0],
        )

        samples = sample_ballast_endpoint_path(
            actual_tank_masses_kg=ACTUAL_MASSES_KG,
            diagnostic_target_tank_masses_kg=result.allocation.target_tank_masses_kg,
            tank_capacities_kg=CAPACITIES_KG,
            fractions=(0.0, 0.5, 1.0),
        )

        np.testing.assert_allclose(
            samples[0].hypothetical_tank_masses_kg,
            ACTUAL_MASSES_KG,
        )
        np.testing.assert_allclose(
            samples[-1].hypothetical_tank_masses_kg,
            result.allocation.target_tank_masses_kg,
        )
        np.testing.assert_allclose(
            samples[1].mass_delta_from_actual_kg,
            0.5 * result.allocation.tank_mass_deltas_kg,
        )
        for sample in samples:
            with self.subTest(fraction=sample.fraction):
                self.assertAlmostEqual(sample.net_mass_delta_from_actual_kg, 0.0)
                self.assertTrue(sample.within_capacity)

    def test_capacity_limit_leaves_a_remaining_external_load_with_the_documented_sign(self):
        baseline = self.diagnose(
            current=np.zeros(6),
            future=[0.0, 0.0, 0.0, 0.0, 2_000.0, 0.0],
        )
        result = self.diagnose(
            current=np.zeros(6),
            future=[0.0, 0.0, 0.0, 0.0, 2_000.0, 0.0],
            actual=np.array([20.0, 500.0, 500.0]),
        )

        self.assertFalse(baseline.allocation.capacity_limited)
        self.assertTrue(result.allocation.capacity_limited)
        self.assertGreater(
            np.linalg.norm(result.remaining_relative_pitch_roll_load_nm),
            0.0,
        )
        np.testing.assert_allclose(
            result.remaining_relative_pitch_roll_load_nm,
            -result.allocation.residual_pitch_roll_moment_nm,
        )
        self.assertAlmostEqual(
            float(np.sum(result.allocation.tank_mass_deltas_kg)),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
