import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validation" / "run_openfast_aligned_pitch_free_response.py"
SPEC = importlib.util.spec_from_file_location("openfast_aligned_pitch_free_response", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class OpenFastAlignedPitchFreeResponseTests(unittest.TestCase):
    def test_peak_spacing_recovers_the_known_period(self):
        time = np.arange(0.0, 100.1, 0.1)
        values = np.cos(2.0 * np.pi * time / 20.0)

        period_s, peaks = MODULE._dominant_positive_period_s(
            time,
            values,
            minimum_amplitude=0.5,
        )

        self.assertAlmostEqual(period_s, 20.0, places=10)
        self.assertGreaterEqual(peaks.size, 4)

    def test_initial_state_offsets_only_pitch(self):
        reference = {"surge_m": 0.4, "heave_m": -0.3, "pitch_deg": -1.5}

        state = MODULE._release_initial_state(reference, pitch_increment_deg=1.0)

        self.assertEqual(state["surge_m"], 0.4)
        self.assertEqual(state["heave_m"], -0.3)
        self.assertEqual(state["pitch_deg"], -0.5)
        self.assertEqual(state["roll_deg"], 0.0)

    def test_period_requires_multiple_positive_peaks(self):
        with self.assertRaisesRegex(ValueError, "at least three positive peaks"):
            MODULE._dominant_positive_period_s(
                [0.0, 1.0, 2.0],
                [0.0, 1.0, 0.0],
                minimum_amplitude=0.1,
            )


if __name__ == "__main__":
    unittest.main()
