import unittest

import numpy as np

from fowt_platform import (
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    PlatformMatrices,
    solve_incremental_static_offset,
)


class IncrementalPlatformModelTests(unittest.TestCase):
    def setUp(self):
        self.matrices = PlatformMatrices(
            mass=np.diag([10.0, 12.0, 14.0, 20.0, 22.0, 24.0]),
            damping=np.diag([1.0, 1.0, 2.0, 3.0, 3.0, 1.0]),
            hydrostatic_stiffness=np.diag([0.0, 0.0, 30.0, 40.0, 50.0, 0.0]),
            mooring_stiffness=np.diag([4.0, 5.0, 0.0, 0.0, 0.0, 6.0]),
            weight_stiffness=np.diag([0.0, 0.0, 0.0, 2.0, 3.0, 0.0]),
        )
        self.model = IncrementalPlatformModel(self.matrices)

    def test_fixed_equilibrium_has_zero_derivative_without_incremental_loads(self):
        derivative = self.model.derivative(
            IncrementalState.zeros(),
            IncrementalLoads.zeros(),
        )

        np.testing.assert_array_equal(derivative.position_rate, np.zeros(6))
        np.testing.assert_array_equal(derivative.velocity_rate, np.zeros(6))

    def test_hydrostatic_and_mooring_restoring_are_applied_once(self):
        position = np.array([2.0, -3.0, 0.5, 0.1, -0.2, 0.3])
        state = IncrementalState(position=position, velocity=np.zeros(6))

        derivative = self.model.derivative(state, IncrementalLoads.zeros())

        expected_restoring = -(
            self.matrices.hydrostatic_stiffness
            + self.matrices.mooring_stiffness
            + self.matrices.weight_stiffness
        ) @ position
        expected_acceleration = np.linalg.solve(
            self.matrices.mass,
            expected_restoring,
        )
        np.testing.assert_allclose(derivative.velocity_rate, expected_acceleration)

    def test_named_load_components_sum_before_solving_acceleration(self):
        loads = IncrementalLoads(
            wind=[1.0, 2.0, 0.0, 0.0, 3.0, 0.0],
            wave=[0.0, -1.0, 2.0, 0.0, 0.0, 1.0],
            ballast=[0.0, 0.0, -4.0, 5.0, 0.0, 0.0],
            other=[1.0, 0.0, 0.0, 0.0, 0.0, -2.0],
        )

        derivative = self.model.derivative(IncrementalState.zeros(), loads)

        np.testing.assert_allclose(
            derivative.velocity_rate,
            np.linalg.solve(self.matrices.mass, loads.total),
        )

    def test_state_rejects_wrong_dof_shape(self):
        with self.assertRaisesRegex(ValueError, "position must have shape \\(6,\\)"):
            IncrementalState(position=np.zeros(7), velocity=np.zeros(6))

    def test_mass_matrix_must_be_symmetric_positive_definite(self):
        invalid_mass = np.eye(6)
        invalid_mass[0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "mass must be positive definite"):
            PlatformMatrices(
                mass=invalid_mass,
                damping=np.eye(6),
                hydrostatic_stiffness=np.eye(6),
                mooring_stiffness=np.eye(6),
            )

    def test_static_offset_balances_a_known_constant_incremental_load(self):
        expected_position = np.array([2.0, -2.0, 0.5, 0.1, -0.1, 1.0])
        static_load = np.array([8.0, -10.0, 15.0, 4.2, -5.3, 6.0])

        offset = solve_incremental_static_offset(
            matrices=self.matrices,
            incremental_static_load=static_load,
        )

        np.testing.assert_allclose(offset.position, expected_position)
        np.testing.assert_allclose(offset.residual_load, np.zeros(6), atol=1e-12)
        derivative = self.model.derivative(
            IncrementalState(position=offset.position, velocity=np.zeros(6)),
            IncrementalLoads(wind=static_load, wave=np.zeros(6), ballast=np.zeros(6), other=np.zeros(6)),
        )
        np.testing.assert_allclose(derivative.velocity_rate, np.zeros(6), atol=1e-12)

    def test_static_offset_rejects_an_unrestrained_load_direction(self):
        matrices = PlatformMatrices(
            mass=np.eye(6),
            damping=np.eye(6),
            hydrostatic_stiffness=np.diag([0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            mooring_stiffness=np.zeros((6, 6)),
        )

        with self.assertRaisesRegex(ValueError, "restoring stiffness is singular"):
            solve_incremental_static_offset(
                matrices=matrices,
                incremental_static_load=np.ones(6),
            )

    def test_coupled_mass_matrix_maps_load_to_expected_acceleration(self):
        mass = np.diag([10.0, 12.0, 14.0, 20.0, 22.0, 24.0])
        mass[0, 4] = mass[4, 0] = 2.0
        mass[1, 3] = mass[3, 1] = -1.5
        matrices = PlatformMatrices(
            mass=mass,
            damping=np.zeros((6, 6)),
            hydrostatic_stiffness=np.eye(6),
            mooring_stiffness=np.zeros((6, 6)),
        )
        model = IncrementalPlatformModel(matrices)
        expected_acceleration = np.array([0.2, -0.1, 0.3, 0.04, -0.05, 0.02])
        applied_load = mass @ expected_acceleration

        derivative = model.derivative(
            IncrementalState.zeros(),
            IncrementalLoads(
                wind=applied_load,
                wave=np.zeros(6),
                ballast=np.zeros(6),
                other=np.zeros(6),
            ),
        )

        np.testing.assert_allclose(derivative.velocity_rate, expected_acceleration)

    def test_negative_damping_is_rejected(self):
        damping = np.eye(6)
        damping[2, 2] = -0.1
        with self.assertRaisesRegex(ValueError, "damping must be positive semidefinite"):
            PlatformMatrices(
                mass=np.eye(6),
                damping=damping,
                hydrostatic_stiffness=np.eye(6),
                mooring_stiffness=np.zeros((6, 6)),
            )

    def test_unstable_total_restoring_stiffness_is_rejected(self):
        hydrostatic = np.eye(6)
        mooring = np.zeros((6, 6))
        mooring[4, 4] = -2.0
        with self.assertRaisesRegex(
            ValueError,
            "total restoring stiffness must be positive semidefinite",
        ):
            PlatformMatrices(
                mass=np.eye(6),
                damping=np.eye(6),
                hydrostatic_stiffness=hydrostatic,
                mooring_stiffness=mooring,
            )


if __name__ == "__main__":
    unittest.main()
