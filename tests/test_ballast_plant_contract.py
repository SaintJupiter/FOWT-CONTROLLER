import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from core_model import FloatingPlatform


class BallastPlantContractTests(unittest.TestCase):
    def _plant(self, directory, platform_profile="default"):
        path = Path(directory) / "stiffness.xlsx"
        pd.DataFrame(
            {
                "offsetx-x (m)": [-10.0, 0.0, 10.0],
                "force (N)": [-2.0e6, 0.0, 2.0e6],
            }
        ).to_excel(path, index=False)
        return FloatingPlatform(path, platform_profile=platform_profile)

    def test_independent_inflow_and_outflow_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory)
            current = plant.current_ballast_mass.copy()
            plant.set_ballast_target(
                current[0] + 10000.0,
                current[1] - 10000.0,
                current[2],
            )

            _, info = plant.step(0.0, 0.0, 1.0, 0.0)

            self.assertGreater(info["pump_net_rate_m3_min"][0], 0.0)
            self.assertLess(info["pump_net_rate_m3_min"][1], 0.0)
            self.assertEqual(info["pump_net_rate_m3_min"][2], 0.0)
            self.assertGreater(info["pump_inflow_m3_min"][0], 0.0)
            self.assertGreater(info["pump_outflow_m3_min"][1], 0.0)
            self.assertAlmostEqual(
                info["ballast_total_delta_kg"],
                float(np.sum(info["tank_mass_delta_kg"])),
            )
            self.assertAlmostEqual(info["ballast_total_delta_kg"], 0.0)

    def test_unbalanced_inflow_updates_total_mass_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory)
            current = plant.current_ballast_mass.copy()
            plant.set_ballast_target(
                current[0] + 10000.0,
                current[1],
                current[2],
            )

            _, info = plant.step(0.0, 0.0, 1.0, 0.0)

            self.assertGreater(info["ballast_total_delta_kg"], 0.0)
            self.assertAlmostEqual(
                info["ballast_total_kg"],
                float(np.sum(info["tank_masses"])),
            )
            self.assertAlmostEqual(
                plant.M_total[0, 0] / 1.6,
                plant.mass_dry + info["ballast_total_kg"],
            )

    def test_reported_transfer_respects_capacity_and_command_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory)
            plant.force_ballast_mass(
                plant.tank_capacity - 10.0,
                10.0,
                plant.current_ballast_mass[2],
            )
            plant.set_ballast_target(
                plant.tank_capacity + 10000.0,
                -10000.0,
                plant.current_ballast_mass[2],
            )

            _, info = plant.step(0.0, 0.0, 1.0, 0.0)

            self.assertTrue(np.all(info["tank_masses"] >= 0.0))
            self.assertTrue(np.all(info["tank_masses"] <= plant.tank_capacity))
            self.assertTrue(
                np.all(
                    np.abs(info["pump_net_rate_m3_min"])
                    <= info["pump_rate_cmd_m3_min"] + 1e-12
                )
            )
            self.assertTrue(np.all(np.isfinite(plant.state)))

    def test_multistep_pump_execution_starts_runs_and_stops_near_target(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory)
            initial = plant.current_ballast_mass.copy()
            target = initial.copy()
            target[0] += 2000.0
            plant.set_ballast_target(*target)

            signed_rates = []
            for _ in range(30):
                _, info = plant.step(0.0, 0.0, 1.0, 0.0)
                signed_rates.append(float(info["pump_net_rate_m3_min"][0]))

            self.assertGreater(max(signed_rates), 0.0)
            self.assertGreater(plant.current_ballast_mass[0], initial[0])
            self.assertLessEqual(plant.current_ballast_mass[0], target[0] + 1e-9)

            for _ in range(300):
                _, info = plant.step(0.0, 0.0, 1.0, 0.0)
            self.assertAlmostEqual(
                float(info["pump_net_rate_m3_min"][0]),
                0.0,
                places=12,
            )
            self.assertLess(
                abs(float(plant.current_ballast_mass[0] - target[0])),
                100.0,
            )

    def test_research_profile_updates_mass_center_and_inertia(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory, "research_incremental_v1")
            initial_mass = float(plant.mass_properties.total_mass_kg)
            initial_center = np.array(plant.mass_properties.center_of_mass_m)
            initial_inertia = np.diag(
                plant.mass_properties.inertia_about_reference_kg_m2
            ).copy()
            initial_effective_mass = plant.M_total.copy()
            changed = plant.current_ballast_mass.copy()
            changed[0] += 10000.0

            plant.force_ballast_mass(*changed)

            self.assertAlmostEqual(
                plant.mass_properties.total_mass_kg,
                initial_mass + 10000.0,
            )
            self.assertFalse(
                np.allclose(plant.mass_properties.center_of_mass_m, initial_center)
            )
            self.assertFalse(
                np.allclose(
                    np.diag(plant.mass_properties.inertia_about_reference_kg_m2),
                    initial_inertia,
                )
            )
            self.assertAlmostEqual(
                plant.M_total[0, 0] - initial_effective_mass[0, 0],
                10000.0,
            )

    def test_real_pump_step_updates_incremental_mass_properties(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory, "research_incremental_v1")
            current = plant.current_ballast_mass.copy()
            plant.set_ballast_target(
                current[0] + 10000.0,
                current[1] - 10000.0,
                current[2],
            )

            _, info = plant.step(0.0, 0.0, 1.0, 0.0)
            deltas = plant.current_ballast_mass - plant.reference_ballast_mass
            reference = plant.reference_mass_properties
            expected_mass = reference.total_mass_kg + float(np.sum(deltas))
            expected_first_moment = (
                reference.total_mass_kg * reference.center_of_mass_m
                + np.sum(deltas[:, None] * plant.tank_pos, axis=0)
            )
            radius_squared = np.einsum("ij,ij->i", plant.tank_pos, plant.tank_pos)
            expected_inertia = (
                reference.inertia_about_reference_kg_m2
                + np.sum(
                    deltas[:, None, None]
                    * (
                        radius_squared[:, None, None] * np.eye(3)
                        - plant.tank_pos[:, :, None]
                        * plant.tank_pos[:, None, :]
                    ),
                    axis=0,
                )
            )

            self.assertGreater(deltas[0], 0.0)
            self.assertLess(deltas[1], 0.0)
            self.assertAlmostEqual(info["platform_total_mass_kg"], expected_mass)
            np.testing.assert_allclose(
                info["platform_center_of_mass_m"],
                expected_first_moment / expected_mass,
            )
            np.testing.assert_allclose(
                info["platform_inertia_about_reference_kg_m2"],
                expected_inertia,
            )
            np.testing.assert_allclose(
                plant.M_total,
                plant._rigid_body_mass_matrix(plant.mass_properties)
                + plant.reference_added_mass_matrix,
            )

    def test_research_profile_reference_state_is_not_recounted(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory, "research_incremental_v1")

            self.assertEqual(
                plant.mass_property_mode,
                "reference_delta_point_mass",
            )
            self.assertAlmostEqual(
                plant.mass_properties.total_mass_kg,
                plant.reference_mass_properties.total_mass_kg,
            )
            np.testing.assert_array_equal(
                plant.mass_properties.center_of_mass_m,
                plant.reference_mass_properties.center_of_mass_m,
            )
            np.testing.assert_array_equal(
                plant.mass_properties.inertia_about_reference_kg_m2,
                plant.reference_mass_properties.inertia_about_reference_kg_m2,
            )

    def test_profile_ballast_override_initializes_pump_cache_consistently(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory)
            configured = np.array([1000.0, 2000.0, 3000.0])
            path = Path(directory) / "second_stiffness.xlsx"
            pd.DataFrame(
                {
                    "offsetx-x (m)": [-10.0, 0.0, 10.0],
                    "force (N)": [-2.0e6, 0.0, 2.0e6],
                }
            ).to_excel(path, index=False)
            plant = FloatingPlatform(
                path,
                platform_cfg={"default_ballast": configured.tolist()},
            )

            np.testing.assert_array_equal(
                plant._pump_prev_target_ballast_mass,
                configured,
            )

    def test_profile_mooring_override_is_not_replaced_by_constructor_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.xlsx"
            configured = [11.0, 22.0, 0.0, 0.0, 0.0, 0.0]
            plant = FloatingPlatform(
                path,
                platform_cfg={"K_mooring_lin": configured},
                allow_linear_mooring_fallback=True,
            )

            np.testing.assert_array_equal(plant.K_mooring_lin, configured)

    def test_unknown_platform_configuration_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stiffness.xlsx"
            pd.DataFrame(
                {
                    "offsetx-x (m)": [-10.0, 0.0, 10.0],
                    "force (N)": [-2.0e6, 0.0, 2.0e6],
                }
            ).to_excel(path, index=False)
            with self.assertRaisesRegex(
                ValueError,
                "unsupported platform configuration fields",
            ):
                FloatingPlatform(
                    path,
                    platform_cfg={"misspelled_mass": 1.0},
                )

    def test_audit_only_profile_is_rejected_by_control_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stiffness.xlsx"
            pd.DataFrame(
                {
                    "offsetx-x (m)": [-10.0, 0.0, 10.0],
                    "force (N)": [-2.0e6, 0.0, 2.0e6],
                }
            ).to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, "audit-only"):
                FloatingPlatform(
                    path,
                    platform_profile="reference_mapped_volturnus_s",
                )

    def test_audit_only_profile_requires_explicit_audit_purpose(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stiffness.xlsx"
            pd.DataFrame(
                {
                    "offsetx-x (m)": [-10.0, 0.0, 10.0],
                    "force (N)": [-2.0e6, 0.0, 2.0e6],
                }
            ).to_excel(path, index=False)
            plant = FloatingPlatform(
                path,
                platform_profile="reference_mapped_volturnus_s",
                platform_profile_purpose="audit",
            )

            self.assertEqual(
                plant.platform_profile_status,
                "audit_only_not_for_control_validation",
            )

    def test_complete_reference_configuration_does_not_recount_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.xlsx"
            reference_inertia = np.diag([2.0e10, 2.1e10, 3.0e10])
            plant = FloatingPlatform(
                path,
                platform_cfg={
                    "load_reference_mode": "reference_incremental",
                    "mass_property_mode": "reference_delta_point_mass",
                    "mooring_reference_mode": "zero_at_reference",
                    "reference_property_mode": "complete_reference",
                    "reference_total_mass_kg": 2.0e7,
                    "reference_center_of_mass_m": [0.0, 0.0, -8.0],
                    "reference_inertia_about_reference_kg_m2": (
                        reference_inertia.tolist()
                    ),
                },
                allow_linear_mooring_fallback=True,
            )

            self.assertEqual(plant.platform_profile_name, "custom")
            self.assertEqual(plant.reference_total_mass, 2.0e7)
            self.assertEqual(plant.mass_properties.total_mass_kg, 2.0e7)
            np.testing.assert_array_equal(
                plant.reference_mass_properties.inertia_about_reference_kg_m2,
                reference_inertia,
            )

    def test_named_profile_cannot_be_partially_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.xlsx"
            with self.assertRaisesRegex(ValueError, "named platform profiles"):
                FloatingPlatform(
                    path,
                    platform_profile="research_incremental_v1",
                    platform_cfg={"mass_property_mode": "legacy_diagonal"},
                    allow_linear_mooring_fallback=True,
                )

    def test_resolved_platform_identity_contains_actual_default_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            plant = self._plant(directory)
            identity = plant.resolved_platform_identity()

            self.assertEqual(
                identity["schema_version"],
                "floating_platform_identity.v1",
            )
            self.assertEqual(identity["profile"]["requested_name"], "default")
            self.assertEqual(
                identity["structure"]["dry_mass_kg"],
                plant.mass_dry,
            )
            self.assertEqual(
                identity["ballast_system"]["tank_positions_m"],
                plant.tank_pos.tolist(),
            )
            self.assertEqual(
                identity["reference_mass_properties"]["effective_mass_matrix"],
                plant.M_total.tolist(),
            )

    def test_resolved_platform_identity_distinguishes_custom_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.xlsx"
            first = FloatingPlatform(
                path,
                platform_cfg={"mass_dry": 1.0e7},
                allow_linear_mooring_fallback=True,
            )
            second = FloatingPlatform(
                path,
                platform_cfg={"mass_dry": 1.1e7},
                allow_linear_mooring_fallback=True,
            )

            self.assertEqual(first.platform_profile_name, "default+custom")
            self.assertEqual(second.platform_profile_name, "default+custom")
            self.assertNotEqual(
                first.resolved_platform_identity(),
                second.resolved_platform_identity(),
            )


if __name__ == "__main__":
    unittest.main()
