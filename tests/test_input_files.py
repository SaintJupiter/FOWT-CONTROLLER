import tempfile
import unittest
from pathlib import Path

from src.wind_prediction.input_files import resolve_mooring_stiffness_file


class MooringInputResolutionTests(unittest.TestCase):
    def test_prefers_only_stiffness_workbook(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = root / "平台水平刚度曲线.xlsx"
            expected.touch()
            (root / "frequency-matrix.xlsx").touch()
            (root / "Thrust force.xlsx").touch()

            resolved = resolve_mooring_stiffness_file(search_dirs=(root,))

            self.assertEqual(resolved, expected.resolve())

    def test_explicit_path_has_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            explicit = root / "selected.xlsx"
            explicit.touch()
            (root / "other_stiffness.xlsx").touch()

            resolved = resolve_mooring_stiffness_file(
                explicit_path=explicit,
                search_dirs=(root,),
            )

            self.assertEqual(resolved, explicit.resolve())

    def test_ambiguous_automatic_discovery_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "surge_stiffness.xlsx").touch()
            (root / "平台刚度.xlsx").touch()

            with self.assertRaisesRegex(RuntimeError, "Multiple mooring stiffness"):
                resolve_mooring_stiffness_file(search_dirs=(root,))

    def test_non_mooring_workbooks_do_not_satisfy_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frequency-matrix.xlsx").touch()

            with self.assertRaisesRegex(FileNotFoundError, "No mooring stiffness"):
                resolve_mooring_stiffness_file(search_dirs=(root,))


if __name__ == "__main__":
    unittest.main()
