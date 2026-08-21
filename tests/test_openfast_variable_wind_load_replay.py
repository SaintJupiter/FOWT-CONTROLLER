"""Unit checks for the controlled load-replay protocol helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/run_openfast_variable_wind_load_replay.py"
SPEC = importlib.util.spec_from_file_location("openfast_variable_wind_replay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class OpenFastVariableWindReplayTests(unittest.TestCase):
    def test_controlled_wind_stays_in_declared_range_and_starts_at_base(self):
        time_s = np.arange(0.0, 901.0, 0.5)
        speed = MODULE.controlled_wind_speed_mps(time_s)

        self.assertTrue(np.all(speed >= 4.5 - 1.0e-12))
        self.assertTrue(np.all(speed <= 5.5 + 1.0e-12))
        self.assertTrue(np.allclose(speed[time_s <= 60.0], 5.0))
        full_period_index = int(np.where(np.isclose(time_s, 660.0))[0][0])
        self.assertAlmostEqual(float(speed[full_period_index]), 5.0, places=12)

    def test_common_time_axis_rejects_misaligned_runs(self):
        with self.assertRaisesRegex(ValueError, "share one time axis"):
            MODULE._assert_common_time_axis(
                np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.6, 1.2])
            )

    def test_analysis_window_contains_only_complete_periods_after_ramp(self):
        start_s, end_s, period_count = MODULE._full_period_analysis_interval_s(
            duration_s=900.0
        )

        self.assertEqual((start_s, end_s, period_count), (300.0, 900.0, 2))
        self.assertGreaterEqual(start_s, MODULE.INITIAL_HOLD_S + MODULE.RAMP_S)

    def test_response_metrics_report_exact_common_response_without_time_shift(self):
        value = np.array([0.0, 0.1, -0.1, 0.2, -0.2])
        metrics = MODULE._response_metrics(
            value,
            value.copy(),
            np.zeros_like(value),
            np.ones(value.shape, dtype=bool),
        )

        self.assertAlmostEqual(metrics["rmse_deg"], 0.0)
        self.assertAlmostEqual(metrics["zero_lag_correlation"], 1.0)
        self.assertAlmostEqual(metrics["amplitude_ratio_low_order_over_openfast"], 1.0)

    def test_response_metrics_reports_fundamental_phase_without_time_shift(self):
        time_s = np.linspace(300.0, 900.0, 1201)
        openfast = np.sin(2.0 * np.pi * time_s / 300.0)
        low_order = np.sin(2.0 * np.pi * time_s / 300.0 + np.deg2rad(30.0))
        metrics = MODULE._response_metrics(
            openfast,
            low_order,
            np.zeros_like(openfast),
            np.ones(openfast.shape, dtype=bool),
            time_s=time_s,
            fundamental_period_s=300.0,
        )

        self.assertAlmostEqual(
            metrics["fundamental_phase_difference_low_order_minus_openfast_deg"],
            30.0,
            places=3,
        )

    def test_replay_requires_an_explicit_platform_matrix_set(self):
        time = np.array([0.0, 1.0, 2.0])
        loads = np.zeros((3, 6))

        with self.assertRaisesRegex(TypeError, "matrices must be PlatformMatrices"):
            MODULE._advance_replay(time, loads, matrices=None)


if __name__ == "__main__":
    unittest.main()
