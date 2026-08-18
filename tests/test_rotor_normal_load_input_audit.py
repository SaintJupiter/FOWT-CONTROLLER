import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_rotor_normal_load_input_audit.py"
SPEC = importlib.util.spec_from_file_location("rotor_normal_load_input_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RotorNormalLoadInputAuditTests(unittest.TestCase):
    def test_frozen_table_lookup_node_has_traceable_source_and_positive_pitch_load(self):
        result = MODULE.run_audit()

        self.assertEqual(
            result["evidence_level"],
            "source_bound_quasi_steady_rotor_normal_load_input",
        )
        self.assertTrue(result["checks"]["all_checks_pass"])
        self.assertTrue(
            result["boundaries"]["uses_fixed_reference_relative_wind_kinematics"]
        )
        self.assertAlmostEqual(
            result["table_lookup_node"][
                "prescribed_aligned_ambient_wind_speed_mps"
            ],
            10.74,
        )
        self.assertEqual(
            result["table_lookup_node"]["pitch_source"], "ElastoDyn initial value"
        )
        self.assertEqual(
            result["table_lookup_node"]["rotor_speed_source"],
            "ElastoDyn initial value",
        )
        self.assertAlmostEqual(result["source_parameters"]["initial_blade_pitch_deg"], 1.0)
        self.assertAlmostEqual(result["source_parameters"]["initial_rotor_speed_rpm"], 7.55)
        self.assertGreater(result["table_lookup_node"]["thrust_coefficient"], 0.0)
        self.assertGreater(result["rotor_normal_load"]["thrust_n"], 0.0)
        self.assertAlmostEqual(
            result["fixed_reference_relative_air"][
                "relative_normal_wind_speed_mps"
            ],
            result["table_lookup_node"]["table_wind_speed_mps"],
        )
        np.testing.assert_allclose(
            result["fixed_reference_relative_air"][
                "ambient_air_velocity_platform_mps"
            ],
            result["fixed_reference_relative_air"][
                "relative_air_velocity_at_hub_platform_mps"
            ],
        )
        np.testing.assert_allclose(
            result["fixed_reference_relative_air"][
                "ambient_air_velocity_platform_mps"
            ],
            result["table_lookup_node"]["table_wind_speed_mps"]
            * np.asarray(
                result["source_parameters"][
                    "downwind_rotor_normal_frozen_equilibrium"
                ]
            ),
        )
        self.assertTrue(
            result["checks"]["relative_normal_wind_speed_matches_table_layer"]
        )
        self.assertGreater(
            result["generalized_load_about_platform_reference"]["pitch_nm"],
            0.0,
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                result["source_parameters"][
                    "downwind_rotor_normal_frozen_equilibrium"
                ]
            ),
            1.0,
        )
        self.assertNotAlmostEqual(
            result["source_parameters"]["downwind_rotor_normal_openfast_reference"][2],
            result["source_parameters"]["downwind_rotor_normal_frozen_equilibrium"][2],
        )
        self.assertNotEqual(
            result["source_parameters"]["hub_from_platform_reference_openfast_reference_m"],
            result["source_parameters"]["hub_from_platform_reference_frozen_equilibrium_m"],
        )
        raw_normal = np.asarray(
            result["source_parameters"]["downwind_rotor_normal_openfast_reference"]
        )
        pitch = math.radians(
            result["source"]["candidate_reference_state"]["state"]["pitch_deg"]
        )
        expected_normal = np.array(
            [
                math.cos(pitch) * raw_normal[0] - math.sin(pitch) * raw_normal[2],
                raw_normal[1],
                math.sin(pitch) * raw_normal[0] + math.cos(pitch) * raw_normal[2],
            ]
        )
        np.testing.assert_allclose(
            result["source_parameters"]["downwind_rotor_normal_frozen_equilibrium"],
            expected_normal,
        )

    def test_rejects_wind_speed_not_represented_by_single_layer_table(self):
        with self.assertRaisesRegex(ValueError, "sole wind-speed layer"):
            MODULE.run_audit(wind_speed_mps=5.0)

    def test_rejects_rotor_state_outside_the_published_table_domain(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            MODULE.run_audit(pitch_deg=35.0)

    def test_explicit_operating_state_override_is_labeled_as_an_override(self):
        result = MODULE.run_audit(pitch_deg=1.5, rotor_speed_rpm=7.5)

        self.assertEqual(
            result["table_lookup_node"]["pitch_source"], "explicit audit override"
        )
        self.assertEqual(
            result["table_lookup_node"]["rotor_speed_source"],
            "explicit audit override",
        )

    def test_rejects_a_negative_ct_entry_from_the_published_table(self):
        table = MODULE.load_rosco_rotor_performance_table_from_zip(
            MODULE.MODEL_ZIP,
            MODULE.ROTOR_TABLE_MEMBER,
            expected_archive_sha256=MODULE.MODEL_ARCHIVE_SHA256,
        )
        index = np.argwhere(table.ct < 0.0)[0]
        target_tsr = table.tip_speed_ratio[index[0]]
        target_pitch = table.pitch_deg[index[1]]
        rpm = (
            target_tsr
            * table.wind_speed_mps[0]
            * 60.0
            / (2.0 * np.pi * 120.97)
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MODULE.run_audit(pitch_deg=target_pitch, rotor_speed_rpm=rpm)


if __name__ == "__main__":
    unittest.main()
