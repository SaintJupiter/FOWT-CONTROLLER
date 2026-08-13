import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    ballast_gravity_load_about_reference,
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
    analyze_undamped_modes,
    load_volturnus_reference_components,
    rigid_body_mass_matrix_about_reference,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)


class ReferencePlatformComponentTests(unittest.TestCase):
    def test_loader_keeps_source_components_separate(self):
        components = load_volturnus_reference_components(MANIFEST)

        self.assertEqual(components.model_version, "IEAWindSystems IEA-15-240-RWT v1.1.16")
        self.assertEqual(components.reference_point, "SWL and tower-axis intersection")
        self.assertEqual(components.platform_only_mass_kg, 17838000.0)
        self.assertEqual(components.whole_system_rigid_body_mass.shape, (6, 6))
        self.assertEqual(components.added_mass.shape, (6, 6))
        self.assertEqual(components.hydrostatic_stiffness.shape, (6, 6))
        self.assertEqual(components.mooring_stiffness.shape, (6, 6))

    def test_whole_system_assembly_adds_added_mass_once(self):
        components = load_volturnus_reference_components(MANIFEST)
        damping = np.diag([1.0e6, 1.0e6, 2.0e6, 1.0e9, 1.0e9, 1.0e9])

        matrices = components.assemble_whole_system_candidate(damping=damping)

        np.testing.assert_allclose(
            matrices.mass,
            components.whole_system_rigid_body_mass + components.added_mass,
        )
        np.testing.assert_allclose(
            matrices.hydrostatic_stiffness,
            0.5
            * (
                components.hydrostatic_stiffness
                + components.hydrostatic_stiffness.T
            ),
        )
        np.testing.assert_allclose(
            matrices.mooring_stiffness,
            0.5 * (components.mooring_stiffness + components.mooring_stiffness.T),
        )
        expected_weight_stiffness = np.zeros((6, 6))
        expected_weight_stiffness[3, 3] = (
            -components.whole_system_mass_kg
            * 9.81
            * components.whole_system_center_of_mass[2]
        )
        expected_weight_stiffness[4, 4] = expected_weight_stiffness[3, 3]
        np.testing.assert_allclose(
            matrices.weight_stiffness,
            expected_weight_stiffness,
        )

    def test_whole_system_mass_properties_reconstruct_source_rigid_matrix(self):
        components = load_volturnus_reference_components(MANIFEST)

        reconstructed = rigid_body_mass_matrix_about_reference(
            total_mass_kg=components.whole_system_mass_kg,
            center_of_mass_m=components.whole_system_center_of_mass,
            inertia_about_reference_kg_m2=components.whole_system_rigid_body_mass[
                3:, 3:
            ],
        )

        np.testing.assert_allclose(
            reconstructed,
            components.whole_system_rigid_body_mass,
            rtol=0.0,
            atol=4.0e4,
        )

    def test_zero_ballast_delta_preserves_the_source_candidate_exactly(self):
        components = load_volturnus_reference_components(MANIFEST)
        damping = np.diag([1.0e6, 1.0e6, 2.0e6, 1.0e9, 1.0e9, 1.0e9])
        tank_coordinates = np.array(
            [[46.2, 0.0, -10.0], [-23.1, 40.0, -10.0], [-23.1, -40.0, -10.0]]
        )

        baseline = components.assemble_whole_system_candidate(damping=damping)
        adjusted = components.assemble_ballast_adjusted_candidate(
            damping=damping,
            tank_mass_deltas_kg=np.zeros(3),
            tank_coordinates_m=tank_coordinates,
        )

        np.testing.assert_array_equal(adjusted.mass, baseline.mass)

    def test_balanced_tank_changes_preserve_total_mass_and_change_distribution(self):
        components = load_volturnus_reference_components(MANIFEST)
        tank_coordinates = np.array(
            [[46.2, 0.0, -10.0], [-23.1, 40.0, -10.0], [-23.1, -40.0, -10.0]]
        )
        deltas = np.array([100000.0, -50000.0, -50000.0])

        properties = components.mass_properties_with_ballast_deltas(
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )
        updated = components.rigid_body_mass_with_ballast_deltas(
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )

        self.assertEqual(properties.total_mass_kg, components.whole_system_mass_kg)
        np.testing.assert_array_equal(
            updated[:3, :3],
            components.whole_system_rigid_body_mass[:3, :3],
        )
        self.assertFalse(
            np.array_equal(updated, components.whole_system_rigid_body_mass)
        )
        self.assertGreater(np.min(np.linalg.eigvalsh(updated)), 0.0)

    def test_ballast_adjusted_candidate_uses_updated_mass_and_incremental_weight_once(self):
        components = load_volturnus_reference_components(MANIFEST)
        damping = np.zeros((6, 6))
        tank_coordinates = np.array(
            [[46.2, 0.0, -10.0], [-23.1, 40.0, -10.0], [-23.1, -40.0, -10.0]]
        )
        deltas = np.array([100000.0, -50000.0, -50000.0])
        matrices = components.assemble_ballast_adjusted_candidate(
            damping=damping,
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )
        ballast_load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )

        derivative = IncrementalPlatformModel(matrices).derivative(
            IncrementalState.zeros(),
            IncrementalLoads(
                wind=np.zeros(6),
                wave=np.zeros(6),
                ballast=ballast_load,
                other=np.zeros(6),
            ),
        )

        np.testing.assert_allclose(
            matrices.mass @ derivative.velocity_rate,
            ballast_load,
            rtol=1e-12,
            atol=1e-8,
        )

    def test_updated_weight_stiffness_and_incremental_load_match_linear_gravity_moment(self):
        components = load_volturnus_reference_components(MANIFEST)
        tank_coordinates = np.array(
            [[46.2, 0.0, -10.0], [-23.1, 40.0, -10.0], [-23.1, -40.0, -10.0]]
        )
        deltas = np.array([100000.0, 20000.0, -50000.0])
        properties = components.mass_properties_with_ballast_deltas(
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )
        candidate = components.assemble_ballast_adjusted_candidate(
            damping=np.zeros((6, 6)),
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )
        incremental_load = ballast_gravity_load_about_reference(
            tank_mass_deltas_kg=deltas,
            tank_coordinates_m=tank_coordinates,
        )
        position = np.array([0.0, 0.0, 0.0, 0.02, -0.03, 0.0])

        model_gravity_moment = (
            incremental_load - candidate.weight_stiffness @ position
        )[3:5]
        mass = properties.total_mass_kg
        x, y, z = properties.center_of_mass_m
        expected = np.array(
            [
                -mass * 9.81 * y + mass * 9.81 * z * position[3],
                mass * 9.81 * x + mass * 9.81 * z * position[4],
            ]
        )

        np.testing.assert_allclose(model_gravity_moment, expected, atol=1e-8)

    def test_assembly_requires_an_explicit_damping_matrix(self):
        components = load_volturnus_reference_components(MANIFEST)

        with self.assertRaisesRegex(ValueError, "damping must be provided explicitly"):
            components.assemble_whole_system_candidate(damping=None)

    def test_assembly_rejects_material_source_matrix_asymmetry(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        hydrostatic = payload["direct_model_values"][
            "hydrostatic_hull_stiffness_about_prp"
        ]
        hydrostatic[3][5] = 1.0e8
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asymmetric.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            components = load_volturnus_reference_components(path)

            with self.assertRaisesRegex(ValueError, "relative asymmetry"):
                components.assemble_whole_system_candidate(damping=np.eye(6))

    def test_loader_rejects_manifest_promoted_to_runtime_by_label_only(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["identity"]["status"] = "runtime_ready"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "evidence-only source manifest"):
                load_volturnus_reference_components(path)

    def test_candidate_modes_match_reference_report_frequency_scale(self):
        components = load_volturnus_reference_components(MANIFEST)
        candidate = components.assemble_whole_system_candidate(
            damping=np.zeros((6, 6))
        )

        modes = analyze_undamped_modes(
            mass=candidate.mass,
            stiffness=candidate.restoring_stiffness,
        )

        report_frequencies_hz = np.array([0.007, 0.007, 0.011, 0.036, 0.036, 0.049])
        np.testing.assert_allclose(
            modes.frequency_hz,
            report_frequencies_hz,
            rtol=0.15,
            atol=0.0,
        )

    def test_report_values_close_the_nominal_vertical_static_balance(self):
        components = load_volturnus_reference_components(MANIFEST)
        evidence = components.vertical_balance_evidence

        self.assertLess(abs(evidence.residual_n), 3000.0)
        self.assertLess(
            abs(evidence.residual_n) / evidence.mooring_vertical_pretension_n,
            5.0e-4,
        )


if __name__ == "__main__":
    unittest.main()
