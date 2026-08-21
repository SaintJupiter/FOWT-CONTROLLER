"""Focused checks for the FINO1 wind-load-scale diagnostic helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/validation/run_fino1_wind_load_scale_audit.py"
SPEC = importlib.util.spec_from_file_location("fino1_wind_load_scale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_legacy_moment_components_have_the_expected_resultant() -> None:
    resultant, roll, pitch = MODULE.legacy_wind_moment_components_mnm(
        np.array([10.0]), np.array([0.0])
    )
    assert np.isclose(resultant[0], pitch[0])
    assert np.isclose(roll[0], 0.0, atol=1.0e-12)


def test_contiguous_changes_exclude_time_gaps_and_invalid_points() -> None:
    timestamp = pd.Series(pd.to_datetime(["2021-01-01 00:00", "2021-01-01 00:10", "2021-01-01 00:30"]))
    changes = MODULE._contiguous_ten_minute_changes(
        timestamp,
        np.array([10.0, 16.0, 30.0]),
        np.array([True, True, True]),
    )
    assert np.allclose(changes, [6.0])


def test_six_hour_coverage_includes_start_and_excludes_endpoint() -> None:
    timestamp = pd.Series(pd.to_datetime(["2021-01-01 00:00", "2021-01-01 05:50", "2021-01-01 06:00"]))
    mask = MODULE._coverage_mask(timestamp, pd.Series(pd.to_datetime(["2021-01-01 00:00"])))
    assert mask.tolist() == [True, True, False]
