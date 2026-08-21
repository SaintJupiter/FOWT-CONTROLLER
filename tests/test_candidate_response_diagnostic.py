import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    IncrementalState,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    compare_hypothetical_candidate_response,
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


def _rotor_load() -> RotorGeneralizedLoad:
    force = np.array([20_000.0, -5_000.0, 0.0])
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.array([8.0, -2.0, 0.0]),
        normal_load=RotorNormalLoad(thrust_n=float(np.linalg.norm(force)), force_platform_n=force),
        generalized_load_platform=np.array(
            [20_000.0, -5_000.0, 0.0, 0.0, 2_400_000.0, 600_000.0]
        ),
    )


class CandidateResponseDiagnosticTests(unittest.TestCase):
    def _compare(self, candidate_tank_masses_kg):
        runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            REFERENCE_MANIFEST,
            np.diag([8.0e4, 8.0e4, 1.0e5, 2.0e7, 2.0e7, 1.0e7]),
        )
        self.assertEqual(
            runtime_assembly.provenance,
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        return compare_hypothetical_candidate_response(
            runtime_assembly=runtime_assembly,
            platform_state=IncrementalState.zeros(),
            actual_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            candidate_tank_masses_kg=candidate_tank_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=_rotor_load(),
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=30.0,
        )

    def test_matching_candidate_reproduces_actual_frozen_response(self):
        diagnostic = self._compare(REFERENCE_TANK_MASSES_KG)

        np.testing.assert_allclose(
            diagnostic.candidate_snapshot.actual_tank_masses_kg,
            diagnostic.actual_snapshot.actual_tank_masses_kg,
        )
        np.testing.assert_allclose(
            diagnostic.candidate_response.loads.total,
            diagnostic.actual_response.loads.total,
        )
        np.testing.assert_allclose(diagnostic.candidate_minus_actual_position, np.zeros(6))
        np.testing.assert_allclose(diagnostic.candidate_minus_actual_velocity, np.zeros(6))

    def test_candidate_changes_only_ballast_branch_for_shared_environmental_loads(self):
        candidate = REFERENCE_TANK_MASSES_KG + np.array([8_000.0, -4_000.0, -4_000.0])
        diagnostic = self._compare(candidate)

        np.testing.assert_allclose(
            diagnostic.actual_response.loads.wind,
            diagnostic.candidate_response.loads.wind,
        )
        np.testing.assert_allclose(
            diagnostic.actual_response.loads.wave,
            diagnostic.candidate_response.loads.wave,
        )
        np.testing.assert_allclose(
            diagnostic.actual_response.loads.other,
            diagnostic.candidate_response.loads.other,
        )
        self.assertGreater(
            np.linalg.norm(
                diagnostic.candidate_snapshot.incremental_ballast_load
                - diagnostic.actual_snapshot.incremental_ballast_load
            ),
            0.0,
        )
        self.assertGreater(
            np.linalg.norm(diagnostic.candidate_minus_actual_position[3:5]),
            0.0,
        )

    def test_rejects_hypothetical_candidate_outside_tank_capacity(self):
        invalid_candidate = REFERENCE_TANK_MASSES_KG.copy()
        invalid_candidate[0] = TANK_CAPACITY_KG + 1.0

        with self.assertRaisesRegex(ValueError, "within each tank capacity"):
            self._compare(invalid_candidate)


if __name__ == "__main__":
    unittest.main()
