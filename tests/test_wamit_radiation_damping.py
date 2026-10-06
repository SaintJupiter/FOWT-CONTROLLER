import unittest
from pathlib import Path

import numpy as np

from fowt_platform import load_local_radiation_damping


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "configs/reference_platforms/volturnus_s_openfast_v1_1_16.json"
)


class WamitRadiationDampingTests(unittest.TestCase):
    def test_loads_source_bound_positive_semidefinite_local_matrix(self):
        result = load_local_radiation_damping(
            MANIFEST,
            requested_period_s=28.5,
        )

        self.assertEqual(result.requested_period_s, 28.5)
        self.assertAlmostEqual(result.lower_source_period_s, 25.13274)
        self.assertAlmostEqual(result.upper_source_period_s, 31.41593)
        self.assertEqual(result.wamit_characteristic_length_m, 1.0)
        self.assertEqual(result.seawater_density_kg_m3, 1025.0)
        self.assertEqual(len(result.archive_sha256), 64)
        np.testing.assert_allclose(
            result.damping_matrix,
            result.damping_matrix.T,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(result.damping_matrix))),
            -1e-8,
        )
        self.assertAlmostEqual(result.damping_matrix[4, 4], 27785.25617783)

    def test_rejects_period_outside_finite_wamit_table(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            load_local_radiation_damping(
                MANIFEST,
                requested_period_s=1_000.0,
                verify_archive_hash=False,
            )


if __name__ == "__main__":
    unittest.main()
