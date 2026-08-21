import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_openfast_moment_angle_consistency.py"
SPEC = importlib.util.spec_from_file_location("openfast_moment_angle_consistency", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpenFastMomentAngleConsistencyTests(unittest.TestCase):
    def test_replaces_and_reads_back_all_six_add_f0_components(self):
        source = (
            "0 AddF0 - preload\n"
            "0\n"
            "0\n"
            "0\n"
            "0\n"
            "0\n"
            "0 0 0 0 0 0 AddCLin - next matrix\n"
        )
        preload = np.array([1.0, -2.0, 3.0, 4.0e6, -5.0e6, 6.0e6])

        updated = MODULE._set_hydrodyn_add_f0(source, preload)

        np.testing.assert_allclose(MODULE._read_hydrodyn_add_f0(updated), preload)
        self.assertIn("AddF0", updated.splitlines()[0])
        self.assertEqual(updated.splitlines()[6], "0 0 0 0 0 0 AddCLin - next matrix")

    def test_disables_wind_aero_and_servo_without_touching_platform_modules(self):
        source = (
            "1 CompInflow - inflow\n"
            "2 CompAero - aero\n"
            "1 CompServo - servo\n"
            "1 CompHydro - hydro\n"
            "3 CompMooring - mooring\n"
        )

        updated = MODULE._disable_aerodynamic_modules(source)

        lines = updated.splitlines()
        self.assertTrue(lines[0].startswith("0 "))
        self.assertTrue(lines[1].startswith("0 "))
        self.assertTrue(lines[2].startswith("0 "))
        self.assertTrue(lines[3].startswith("1 "))
        self.assertTrue(lines[4].startswith("3 "))

    def test_rotates_openfast_preload_into_frozen_equilibrium_axes(self):
        nominal = {"surge_m": 0.0, "heave_m": 0.0, "pitch_deg": -10.0}
        preload = np.array([0.0, 0.0, 0.0, 1.0e7, 2.0e7, 3.0e7])

        result = MODULE._frozen_equilibrium_load(preload, nominal_reference=nominal)

        self.assertEqual(result.shape, (6,))
        self.assertFalse(np.allclose(result[3:], preload[3:]))
        self.assertAlmostEqual(np.linalg.norm(result[3:]), np.linalg.norm(preload[3:]))

    def test_release_stability_requires_small_range_and_small_tail_drift(self):
        phase = {
            "late_window_statistics": {
                "second_window_statistics": {
                    "roll_deg": {"range": 0.003},
                    "pitch_deg": {"range": 0.004},
                },
                "second_minus_first_mean": {
                    "roll_deg": 0.001,
                    "pitch_deg": -0.002,
                },
            }
        }

        result = MODULE._release_tail_stability(phase)

        self.assertTrue(result["passes"])
        phase["late_window_statistics"]["second_minus_first_mean"]["pitch_deg"] = 0.006
        self.assertFalse(MODULE._release_tail_stability(phase)["passes"])

    def test_static_solution_keeps_the_corrected_runtime_provenance(self):
        solution = MODULE._static_low_order_solution(np.zeros(6))

        self.assertEqual(
            solution["runtime_provenance"],
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        self.assertLess(solution["static_residual_max_abs"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
