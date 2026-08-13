import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


LEGACY_DIR = Path(__file__).resolve().parents[1] / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from core_model import FloatingPlatform  # noqa: E402


class MooringInputValidationTests(unittest.TestCase):
    def test_valid_stiffness_workbook_enables_table_mooring(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stiffness.xlsx"
            pd.DataFrame(
                {
                    "offsetx-x (m)": [-10.0, 0.0, 10.0],
                    "force (N)": [-2.0e6, 0.0, 2.0e6],
                }
            ).to_excel(path, index=False)

            plant = FloatingPlatform(path)

            self.assertTrue(plant.use_nonlinear_mooring)
            self.assertEqual(plant.mooring_source_path, str(path.resolve()))

    def test_invalid_workbook_fails_instead_of_silent_linear_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequency-matrix.xlsx"
            pd.DataFrame({"frequency": [0.1, 0.2], "value": [1.0, 2.0]}).to_excel(
                path,
                index=False,
            )

            with self.assertRaisesRegex(ValueError, "displacement and force"):
                FloatingPlatform(path)

    def test_linear_fallback_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frequency-matrix.xlsx"
            pd.DataFrame({"frequency": [0.1], "value": [1.0]}).to_excel(
                path,
                index=False,
            )

            with redirect_stdout(io.StringIO()):
                plant = FloatingPlatform(path, allow_linear_mooring_fallback=True)

            self.assertFalse(plant.use_nonlinear_mooring)
            self.assertIsNone(plant.mooring_source_path)

    def test_research_profile_removes_zero_displacement_force_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stiffness.xlsx"
            pd.DataFrame(
                {
                    "offsetx-x (m)": [-10.0, 0.0, 10.0],
                    "force (N)": [-1.9e6, 1.0e5, 2.1e6],
                }
            ).to_excel(path, index=False)

            plant = FloatingPlatform(
                path,
                platform_profile="research_incremental_v1",
            )

            self.assertEqual(plant.mooring_reference_force_raw, 1.0e5)
            self.assertAlmostEqual(plant._get_mooring_force(0.0, 0), 0.0)
            self.assertLess(plant._get_mooring_force(1.0, 0), 0.0)
            self.assertGreater(plant._get_mooring_force(-1.0, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
