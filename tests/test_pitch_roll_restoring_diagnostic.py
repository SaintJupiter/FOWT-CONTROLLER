import unittest

import numpy as np

from fowt_platform import (
    PlatformMatrices,
    diagnose_pitch_roll_restoring_demand,
)


def _matrices_with_pitch_roll_stiffness(stiffness_pitch_roll):
    restoring = np.eye(6)
    restoring[np.ix_([4, 3], [4, 3])] = stiffness_pitch_roll
    return PlatformMatrices(
        mass=np.eye(6),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=restoring,
        mooring_stiffness=np.zeros((6, 6)),
    )


class PitchRollRestoringDiagnosticTests(unittest.TestCase):
    def test_equal_loads_have_zero_relative_diagnostic(self):
        diagnostic = diagnose_pitch_roll_restoring_demand(
            matrices=_matrices_with_pitch_roll_stiffness(
                np.diag([200.0, 100.0])
            ),
            current_generalized_load=[1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
            future_generalized_load=[1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
        )

        np.testing.assert_array_equal(diagnostic.relative_pitch_roll_load_nm, [0.0, 0.0])
        np.testing.assert_array_equal(
            diagnostic.counteracting_pitch_roll_moment_nm,
            [0.0, 0.0],
        )
        np.testing.assert_array_equal(
            diagnostic.constrained_pitch_roll_response_rad,
            [0.0, 0.0],
        )

    def test_pitch_and_roll_outputs_use_the_documented_pitch_roll_order(self):
        diagnostic = diagnose_pitch_roll_restoring_demand(
            matrices=_matrices_with_pitch_roll_stiffness(
                np.diag([200.0, 100.0])
            ),
            current_generalized_load=np.zeros(6),
            future_generalized_load=[0.0, 0.0, 0.0, 30.0, -40.0, 0.0],
        )

        np.testing.assert_allclose(
            diagnostic.relative_pitch_roll_load_nm,
            [-40.0, 30.0],
        )
        np.testing.assert_allclose(
            diagnostic.constrained_pitch_roll_response_rad,
            [-0.2, 0.3],
        )
        np.testing.assert_allclose(
            diagnostic.counteracting_pitch_roll_moment_nm,
            [40.0, -30.0],
        )

    def test_coupled_pitch_roll_stiffness_uses_the_local_two_by_two_solution(self):
        stiffness = np.array([[200.0, 40.0], [40.0, 100.0]])
        diagnostic = diagnose_pitch_roll_restoring_demand(
            matrices=_matrices_with_pitch_roll_stiffness(stiffness),
            current_generalized_load=np.zeros(6),
            future_generalized_load=[0.0, 0.0, 0.0, -16.0, 28.0, 0.0],
        )

        expected_response = np.linalg.solve(stiffness, [28.0, -16.0])
        np.testing.assert_allclose(
            diagnostic.constrained_pitch_roll_response_rad,
            expected_response,
        )
        np.testing.assert_allclose(
            diagnostic.pitch_roll_stiffness_nm_per_rad,
            stiffness,
        )
        np.testing.assert_allclose(
            diagnostic.counteracting_pitch_roll_moment_nm,
            [-28.0, 16.0],
        )

    def test_singular_pitch_roll_block_is_rejected_without_full_static_solve(self):
        matrices = _matrices_with_pitch_roll_stiffness(
            np.array([[200.0, 0.0], [0.0, 0.0]])
        )

        with self.assertRaisesRegex(ValueError, "pitch-roll restoring stiffness"):
            diagnose_pitch_roll_restoring_demand(
                matrices=matrices,
                current_generalized_load=np.zeros(6),
                future_generalized_load=np.zeros(6),
            )

    def test_translational_load_change_is_outside_this_diagnostic_boundary(self):
        diagnostic = diagnose_pitch_roll_restoring_demand(
            matrices=_matrices_with_pitch_roll_stiffness(
                np.diag([200.0, 100.0])
            ),
            current_generalized_load=np.zeros(6),
            future_generalized_load=[1000.0, -500.0, 200.0, 0.0, 0.0, 80.0],
        )

        np.testing.assert_array_equal(diagnostic.relative_pitch_roll_load_nm, [0.0, 0.0])
        np.testing.assert_array_equal(
            diagnostic.constrained_pitch_roll_response_rad,
            [0.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
