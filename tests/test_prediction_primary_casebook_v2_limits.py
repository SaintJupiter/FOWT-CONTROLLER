from __future__ import annotations

import unittest

from scripts.analysis.run_prediction_primary_casebook import (
    enforce_casebook_experiment_limits,
)


class PredictionPrimaryCasebookV2LimitTests(unittest.TestCase):
    def test_thirty_six_hour_cases_are_allowed(self) -> None:
        enforce_casebook_experiment_limits(
            duration_s=21600,
            case_count=30,
        )

    def test_thirty_one_cases_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed 30 cases"):
            enforce_casebook_experiment_limits(
                duration_s=21600,
                case_count=31,
            )

    def test_non_six_hour_case_duration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "--duration-s 21600"):
            enforce_casebook_experiment_limits(
                duration_s=7200,
                case_count=1,
            )

    def test_frozen_v1_reproduction_must_also_be_split_into_bounded_batches(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed 30 cases"):
            enforce_casebook_experiment_limits(
                duration_s=21600,
                case_count=31,
            )


if __name__ == "__main__":
    unittest.main()
