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

    def test_frozen_public_table_preserves_documented_axis_order(self):
        table = load_rosco_rotor_performance_table_from_zip(
            REFERENCE_ARCHIVE, REFERENCE_MEMBER
        )
        self.assertEqual(table.ct.shape, (26, 36))
        self.assertEqual(table.pitch_deg[0], -5.0)
        self.assertEqual(table.pitch_deg[-1], 30.0)
        self.assertEqual(table.tip_speed_ratio[0], 2.0)
        self.assertEqual(table.tip_speed_ratio[-1], 14.5)
        self.assertAlmostEqual(table.ct_at(pitch_deg=0.0, tip_speed_ratio=7.0), 0.614890)

    def test_frozen_public_table_uses_the_four_neighboring_entries(self):
        table = load_rosco_rotor_performance_table_from_zip(
            REFERENCE_ARCHIVE, REFERENCE_MEMBER
        )
        self.assertAlmostEqual(
            table.ct_at(pitch_deg=0.5, tip_speed_ratio=7.25), 0.624166, places=6
        )


if __name__ == "__main__":
    unittest.main()
