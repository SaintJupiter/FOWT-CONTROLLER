import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    assemble_ballast_model_snapshot,
    load_volturnus_source_consistent_static_reference,
    solve_incremental_static_offset,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)


class SourceConsistentReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = load_volturnus_source_consistent_static_reference(MANIFEST)

    def test_inventory_uses_only_elastodyn_and_first_order_wamit_sources(self):
        names = [item.name for item in self.reference.mass_items]
        self.assertEqual(
            names,
            ["platform", "tower", "nacelle", "yaw_bearing", "hub", "blades"],
        )
        self.assertEqual(
            self.reference.model_version,
            "IEAWindSystems IEA-15-240-RWT v1.1.16",
        )
        self.assertEqual(self.reference.gravity_m_s2, 9.81)
        self.assertEqual(self.reference.seawater_density_kg_m3, 1025.0)
        expected = {
            "platform": (17838000.0, -14.4),
            "tower": (1466657.178309603, 57.685169457893046),
            "nacelle": (644857.0, 148.701),
            "yaw_bearing": (28249.0, 144.386),
            "hub": (69131.0, 150.00008534857722),
            "blades": (205547.98090660915, 150.00008534857722),
        }
        actual = {
            item.name: (item.mass_kg, item.vertical_position_m)
            for item in self.reference.mass_items
        }
        self.assertEqual(set(actual), set(expected))
        for name, expected_values in expected.items():
            np.testing.assert_allclose(actual[name], expected_values, rtol=1.0e-12)
        self.assertAlmostEqual(
            self.reference.total_mass_kg, 20252442.15921621, places=4
        )
        self.assertAlmostEqual(
            self.reference.vertical_center_of_mass_m,
            -1.5351887163872497,
            delta=1.0e-9,
        )

    def test_weight_stiffness_follows_the_source_derived_vertical_center(self):
        expected = (
            -self.reference.total_mass_kg
            * self.reference.gravity_m_s2
            * self.reference.vertical_center_of_mass_m
        )
        np.testing.assert_allclose(
            self.reference.weight_stiffness[3, 3], expected, rtol=1.0e-12
        )
        np.testing.assert_allclose(
            self.reference.weight_stiffness[4, 4], expected, rtol=1.0e-12
        )
        np.testing.assert_array_equal(
            self.reference.weight_stiffness[[0, 1, 2, 5], :],
            np.zeros((4, 6)),
        )

    def test_static_pitch_stiffness_has_the_expected_no_mooring_openfast_scale(self):
        # The 21 Aug conditional OpenFAST audit with MoorDyn disabled gave
        # 2.492832 GNm/rad.  The two conditions retain different free-coordinate
        # responses, so this is a static-scale check rather than a full-model fit.
        self.assertAlmostEqual(
            self.reference.pitch_restoring_stiffness_without_mooring_nm_per_rad
            / 1.0e9,
            2.492832,
            delta=0.025,
        )

    def test_first_order_wamit_terms_remain_available_but_mooring_is_not_injected(self):
        self.assertEqual(self.reference.hydrostatic_stiffness.shape, (6, 6))
        self.assertEqual(self.reference.infinite_frequency_added_mass.shape, (6, 6))
        self.assertGreater(self.reference.hydrostatic_stiffness[4, 4], 2.0e9)
        self.assertGreater(self.reference.infinite_frequency_added_mass[4, 4], 1.0e10)

    def test_static_restoring_aligned_runtime_matches_moordyn_enabled_scale(self):
        runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            MANIFEST,
            np.zeros((6, 6)),
        )
        unit_pitch_moment = np.zeros(6)
        unit_pitch_moment[4] = 1.0
        compliance = solve_incremental_static_offset(
            runtime.base_matrices,
            unit_pitch_moment,
        ).position[4]
        stiffness_gnm_per_rad = 1.0 / compliance / 1.0e9

        self.assertEqual(
            runtime.provenance,
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        self.assertAlmostEqual(stiffness_gnm_per_rad, 2.73377, delta=0.025)

    def test_static_restoring_aligned_runtime_keeps_its_base_at_zero_tank_delta(self):
        runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            MANIFEST,
            np.zeros((6, 6)),
        )
        masses = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime,
            actual_tank_masses_kg=masses,
            reference_tank_masses_kg=masses,
            tank_capacities_kg=np.full(3, 1_896_250.0),
            tank_coordinates_m=np.array(
                [
                    [46.2, 0.0, -10.0],
                    [-23.1, 46.2 * 0.866, -10.0],
                    [-23.1, -46.2 * 0.866, -10.0],
                ]
            ),
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
                getattr(runtime.base_matrices, name),
            )


if __name__ == "__main__":
    unittest.main()
