import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    assemble_ballast_model_snapshot,
    load_volturnus_reference_components,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITIES_KG = np.full(3, 1_896_250.0)


class BallastModelSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_volturnus_reference_components(REFERENCE_MANIFEST)

    def assemble(self, actual_masses):
        return assemble_ballast_model_snapshot(
            reference=self.reference,
            damping=np.zeros((6, 6)),
            actual_tank_masses_kg=actual_masses,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=TANK_CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
        )

    def test_reference_tank_state_preserves_source_mass_and_has_zero_load(self):
        snapshot = self.assemble(REFERENCE_TANK_MASSES_KG)
        source = self.reference.assemble_whole_system_candidate(
            damping=np.zeros((6, 6)),
        )

        for name in (
            "mass",
            "damping",
            "hydrostatic_stiffness",
            "mooring_stiffness",
            "weight_stiffness",
            "restoring_stiffness",
        ):
            np.testing.assert_array_equal(
                getattr(snapshot.matrices, name),
                getattr(source, name),
            )
        np.testing.assert_array_equal(
            snapshot.incremental_ballast_load,
            np.zeros(6),
        )

    def test_one_tank_state_updates_mass_and_load_from_same_delta(self):
        actual = REFERENCE_TANK_MASSES_KG + np.array(
            [-20_000.0, 10_000.0, 10_000.0]
        )

        snapshot = self.assemble(actual)

        np.testing.assert_array_equal(
            snapshot.tank_mass_deltas_kg,
            [-20_000.0, 10_000.0, 10_000.0],
        )
        self.assertAlmostEqual(snapshot.incremental_ballast_load[2], 0.0)
        self.assertNotEqual(snapshot.incremental_ballast_load[4], 0.0)
        self.assertFalse(
            np.array_equal(
                snapshot.matrices.mass,
                self.reference.whole_system_rigid_body_mass
                + self.reference.added_mass,
            )
        )

    def test_invalid_actual_mass_is_rejected_before_matrix_assembly(self):
        invalid = REFERENCE_TANK_MASSES_KG.copy()
        invalid[0] = TANK_CAPACITIES_KG[0] + 1.0

        with self.assertRaisesRegex(ValueError, "tank capacity"):
            self.assemble(invalid)


if __name__ == "__main__":
    unittest.main()
