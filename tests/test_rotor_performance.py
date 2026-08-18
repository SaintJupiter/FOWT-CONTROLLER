import unittest
from pathlib import Path

import numpy as np

from fowt_platform.rotor_performance import (
    load_rosco_rotor_performance_table_from_zip,
    parse_rosco_rotor_performance_table,
)


FIXTURE = """# Pitch angle vector, 2 entries - x axis (matrix columns) (deg)
0.0  10.0
# TSR vector, 2 entries - y axis (matrix rows) (-)
4.0  8.0
# Wind speed vector - z axis (m/s)
10.0

# Power coefficient
0.1 0.2
0.3 0.4

# Thrust coefficient
0.5 0.6
0.7 0.8

# Torque coefficient
0.01 0.02
0.03 0.04
"""

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ARCHIVE = REPOSITORY_ROOT / (
    "references/fowt_reference_models/3_核心必用_官方模型文件包_IEAWindSystems_v1.1.16/"
    "IEAWindSystems_IEA-15-240-RWT_v1.1.16.zip"
)
REFERENCE_MEMBER = "IEA-15-240-RWT-1.1.16/OpenFAST/IEA-15-240-RWT/Cp_Ct_Cq.IEA15MW.txt"
REFERENCE_ARCHIVE_SHA256 = (
    "c96c6be345abf9440170764e2d9a2c201b5da6191192cef66537c5bc16805b80"
)


class RotorPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.table = parse_rosco_rotor_performance_table(FIXTURE, source_name="fixture")

    def test_table_keeps_documented_rows_and_columns(self):
        self.assertEqual(self.table.ct.shape, (2, 2))
        np.testing.assert_allclose(self.table.pitch_deg, [0.0, 10.0])
        np.testing.assert_allclose(self.table.tip_speed_ratio, [4.0, 8.0])
        self.assertAlmostEqual(self.table.ct_at(pitch_deg=10.0, tip_speed_ratio=8.0), 0.8)

    def test_ct_uses_bilinear_interpolation_inside_the_grid(self):
        self.assertAlmostEqual(self.table.ct_at(pitch_deg=5.0, tip_speed_ratio=6.0), 0.65)

    def test_ct_rejects_extrapolation(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.table.ct_at(pitch_deg=-0.1, tip_speed_ratio=6.0)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.table.ct_at(pitch_deg=5.0, tip_speed_ratio=8.1)

    def test_parser_rejects_matrix_with_axis_mismatch(self):
        malformed = FIXTURE.replace("0.7 0.8", "0.7")
        with self.assertRaisesRegex(ValueError, "Invalid matrix row"):
            parse_rosco_rotor_performance_table(malformed)

    def test_parser_rejects_numeric_rows_beyond_the_declared_matrix_shape(self):
        malformed = FIXTURE.replace(
            "0.7 0.8\n\n# Torque coefficient",
            "0.7 0.8\n0.9 1.0\n\n# Torque coefficient",
        )
        with self.assertRaisesRegex(ValueError, "more rows"):
            parse_rosco_rotor_performance_table(malformed)

    def test_parser_rejects_multiple_wind_speed_metadata_entries(self):
        multilayer = FIXTURE.replace("10.0\n\n# Power coefficient", "10.0 12.0\n\n# Power coefficient")
        with self.assertRaisesRegex(ValueError, "two-dimensional ROSCO table"):
            parse_rosco_rotor_performance_table(multilayer)

    def test_parser_rejects_single_point_interpolation_axis(self):
        single_pitch = (
            FIXTURE.replace("0.0  10.0", "0.0", 1)
            .replace("0.1 0.2", "0.1")
            .replace("0.3 0.4", "0.3")
            .replace("0.5 0.6", "0.5")
            .replace("0.7 0.8", "0.7")
            .replace("0.01 0.02", "0.01")
            .replace("0.03 0.04", "0.03")
        )
        with self.assertRaisesRegex(ValueError, "at least 2 entries"):
            parse_rosco_rotor_performance_table(single_pitch)

    def test_frozen_public_table_preserves_documented_axis_order(self):
        table = load_rosco_rotor_performance_table_from_zip(
            REFERENCE_ARCHIVE,
            REFERENCE_MEMBER,
            expected_archive_sha256=REFERENCE_ARCHIVE_SHA256,
        )
        self.assertEqual(table.ct.shape, (26, 36))
        self.assertEqual(table.pitch_deg[0], -5.0)
        self.assertEqual(table.pitch_deg[-1], 30.0)
        self.assertEqual(table.tip_speed_ratio[0], 2.0)
        self.assertEqual(table.tip_speed_ratio[-1], 14.5)
        self.assertEqual(table.wind_speed_metadata_mps[0], 10.74)
        self.assertEqual(table.source_archive_sha256, REFERENCE_ARCHIVE_SHA256)
        self.assertAlmostEqual(table.ct_at(pitch_deg=0.0, tip_speed_ratio=7.0), 0.614890)

    def test_frozen_public_table_uses_the_four_neighboring_entries(self):
        table = load_rosco_rotor_performance_table_from_zip(
            REFERENCE_ARCHIVE,
            REFERENCE_MEMBER,
            expected_archive_sha256=REFERENCE_ARCHIVE_SHA256,
        )
        self.assertAlmostEqual(
            table.ct_at(pitch_deg=0.5, tip_speed_ratio=7.25), 0.624166, places=6
        )

    def test_frozen_public_table_rejects_unexpected_archive_identity(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_rosco_rotor_performance_table_from_zip(
                REFERENCE_ARCHIVE,
                REFERENCE_MEMBER,
                expected_archive_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
