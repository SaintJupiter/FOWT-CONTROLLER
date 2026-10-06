from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIRECTORY = ROOT / "scripts" / "validation"
if str(VALIDATION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIRECTORY))

from preview_mpc_parameter_study_cases import (  # noqa: E402
    load_frozen_staged_cases,
)


class PreviewMPCParameterStudyCaseRegistryTests(unittest.TestCase):
    def test_frozen_wind_only_registry_has_expected_stages_and_counts(self):
        cases, identity = load_frozen_staged_cases(repository_root=ROOT)

        self.assertEqual(
            {name: len(items) for name, items in cases.items()},
            {"stage10": 10, "stage20": 20, "stage30": 30},
        )
        self.assertFalse(identity["controller_outputs_used"])
        self.assertEqual(identity["within_and_cross_stage_overlap_count"], 0)
        self.assertEqual(identity["historical_overlap_count"], 0)
        self.assertEqual(identity["minimum_actual_start_separation_s"], 259200.0)
        self.assertEqual(
            set(case.stratum for case in cases["stage10"]),
            {
                "future_relief_or_reversal",
                "low_disturbance",
                "oscillation",
                "strengthening",
            },
        )


if __name__ == "__main__":
    unittest.main()
