import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    IncrementalState,
    PlatformMatrices,
    load_volturnus_reference_components,
    simulate_linear_free_response,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)


def _diagonal_matrices(*, damping_ratio: float = 0.0) -> PlatformMatrices:
    mass = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    angular_frequency = np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    stiffness = np.diag(np.diag(mass) * angular_frequency**2)
    damping = np.diag(
        2.0 * damping_ratio * np.diag(mass) * angular_frequency
    )
    return PlatformMatrices(
        mass=mass,
        damping=damping,
        hydrostatic_stiffness=stiffness,
        mooring_stiffness=np.zeros((6, 6)),
    )


class PlatformFreeResponseTests(unittest.TestCase):
    def test_undamped_single_mode_matches_analytic_solution(self):
        matrices = _diagonal_matrices()
        initial_position = np.zeros(6)
        initial_position[3] = 0.1
        samples = np.linspace(0.0, 20.0, 101)

        response = simulate_linear_free_response(
            matrices=matrices,
            initial_state=IncrementalState(
                position=initial_position,
                velocity=np.zeros(6),
            ),
            time_s=samples,
        )

        expected_position = 0.1 * np.cos(1.1 * samples)
        expected_velocity = -0.11 * np.sin(1.1 * samples)
        np.testing.assert_allclose(response.position[:, 3], expected_position)
        np.testing.assert_allclose(response.velocity[:, 3], expected_velocity)
        np.testing.assert_allclose(
            response.mechanical_energy,
            response.mechanical_energy[0],
            rtol=1e-12,
            atol=1e-14,
        )

    def test_positive_damping_does_not_increase_mechanical_energy(self):
        response = simulate_linear_free_response(
            matrices=_diagonal_matrices(damping_ratio=0.05),
            initial_state=IncrementalState(
                position=np.full(6, 0.1),
                velocity=np.zeros(6),
            ),
            time_s=np.linspace(0.0, 40.0, 201),
        )

        energy_change = np.diff(response.mechanical_energy)
        tolerance = 1e-12 * response.mechanical_energy[0]
        self.assertTrue(np.all(energy_change <= tolerance))
        self.assertLess(
            response.mechanical_energy[-1],
            response.mechanical_energy[0],
        )

    def test_coupled_rank_deficient_damping_does_not_increase_energy(self):
        matrices = _diagonal_matrices()
        damping_factor = np.array([1.0, -0.5, 0.0, 0.8, 0.0, -0.2])
        coupled_damping = np.outer(damping_factor, damping_factor)
        coupled_matrices = PlatformMatrices(
            mass=matrices.mass,
            damping=coupled_damping,
            hydrostatic_stiffness=matrices.hydrostatic_stiffness,
            mooring_stiffness=matrices.mooring_stiffness,
        )

        response = simulate_linear_free_response(
            matrices=coupled_matrices,
            initial_state=IncrementalState(
                position=np.full(6, 0.1),
                velocity=np.linspace(-0.1, 0.1, 6),
            ),
            time_s=np.linspace(0.0, 20.0, 201),
        )

        tolerance = 1e-12 * response.mechanical_energy[0]
        self.assertTrue(np.all(np.diff(response.mechanical_energy) <= tolerance))

    def test_zero_state_stays_at_fixed_equilibrium(self):
        response = simulate_linear_free_response(
            matrices=_diagonal_matrices(damping_ratio=0.05),
            initial_state=IncrementalState.zeros(),
            time_s=[0.0, 1.0, 2.0],
        )

        np.testing.assert_array_equal(response.position, np.zeros((3, 6)))
        np.testing.assert_array_equal(response.velocity, np.zeros((3, 6)))
        np.testing.assert_array_equal(response.mechanical_energy, np.zeros(3))

    def test_time_samples_must_be_strictly_increasing(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            simulate_linear_free_response(
                matrices=_diagonal_matrices(),
                initial_state=IncrementalState.zeros(),
                time_s=[0.0, 1.0, 1.0],
            )

    def test_reference_pitch_response_reproduces_matrix_modal_period(self):
        reference = load_volturnus_reference_components(REFERENCE_MANIFEST)
        matrices = reference.assemble_whole_system_candidate(
            damping=np.zeros((6, 6))
        )
        initial_position = np.zeros(6)
        initial_position[4] = np.deg2rad(3.0)
        samples = np.arange(0.0, 140.0 + 0.1, 0.1)

        response = simulate_linear_free_response(
            matrices=matrices,
            initial_state=IncrementalState(
                position=initial_position,
                velocity=np.zeros(6),
            ),
            time_s=samples,
        )
        pitch = response.position[:, 4]
        peak_indices = np.flatnonzero(
            (pitch[1:-1] > pitch[:-2]) & (pitch[1:-1] >= pitch[2:])
        ) + 1
        period_s = float(np.median(np.diff(samples[peak_indices])))

        self.assertGreaterEqual(peak_indices.size, 4)
        self.assertAlmostEqual(period_s, 26.85, delta=0.1)
        relative_energy_drift = np.max(
            np.abs(
                response.mechanical_energy / response.mechanical_energy[0] - 1.0
            )
        )
        self.assertLess(relative_energy_drift, 1e-11)


if __name__ == "__main__":
    unittest.main()
