import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_openfast_large_angle_pitch_sweep.py"
SPEC = importlib.util.spec_from_file_location("openfast_large_angle_pitch_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpenFastLargeAnglePitchSweepTests(unittest.TestCase):
    def test_target_pitch_sequence_requires_positive_increasing_values(self):
        self.assertEqual(MODULE._target_pitch_sequence((2, 3, 5)), (2.0, 3.0, 5.0))
        with self.assertRaises(ValueError):
            MODULE._target_pitch_sequence((2, 2))
        with self.assertRaises(ValueError):
            MODULE._target_pitch_sequence((2, -3))

    def test_frozen_and_openfast_reference_loads_round_trip(self):
        nominal = {"surge_m": 0.0, "heave_m": 0.0, "pitch_deg": -10.0}
        frozen = np.array([1.0, -2.0, 3.0, 4.0e6, -5.0e6, 6.0e6])

        openfast = MODULE._frozen_equilibrium_load_to_openfast_reference(
            frozen,
            nominal_reference=nominal,
        )
        round_trip = MODULE._frozen_equilibrium_load(
            openfast,
            nominal_reference=nominal,
        )

        np.testing.assert_allclose(round_trip, frozen, rtol=1e-12, atol=1e-7)

    def test_target_case_recovers_requested_low_order_pitch(self):
        nominal = {"surge_m": 0.0, "heave_m": 0.0, "pitch_deg": -10.0}
        target = MODULE._target_case(7.0, nominal_reference=nominal)
        low_order = MODULE._static_low_order_solution(
            MODULE._frozen_equilibrium_load(
                target["openfast_preload"],
                nominal_reference=nominal,
            )
        )

        self.assertAlmostEqual(low_order["static_offset"]["pitch_deg"], 7.0, places=10)
        self.assertEqual(
            target["low_order_runtime_provenance"],
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        frozen_load = target["frozen_equilibrium_load_si"]
        self.assertEqual(frozen_load["surge_n"], 0.0)
        self.assertEqual(frozen_load["sway_n"], 0.0)
        self.assertEqual(frozen_load["heave_n"], 0.0)
        self.assertEqual(frozen_load["roll_nm"], 0.0)
        self.assertEqual(frozen_load["yaw_nm"], 0.0)


if __name__ == "__main__":
    unittest.main()
