#!/usr/bin/env python3
"""Run the conservative-deadband positive selector validation.

This reproduces the 170-case positive-only readout with a tighter
deadband, trading part of the pump saving for lower attitude increase.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = Path("outputs/wind_prediction/positive_tilt_conservative_20260611")

OLD_CASES = Path(
    "outputs/wind_prediction/selector_mixed_6h_limit20_v1/"
    "casebooks/selector_mixed_limit20_6h_cases.csv"
)
NEW_CASES = Path(
    "outputs/wind_prediction/selector_positive_add40_6h_v1/"
    "casebooks/positive_add40_6h_cases.csv"
)

OLD_OUT = OUT_ROOT / "deadband12_existing101_6h"
NEW_OUT = OUT_ROOT / "deadband12_add40_120case_6h"
COMBINED_OUT = OUT_ROOT / "combined_positive_summary_170case_6h"

DEADBAND_ENV = {
    "FOWT_DC_DEADBAND_PITCH_DEG": "1.2",
    "FOWT_DC_DEADBAND_ROLL_DEG": "1.2",
    "FOWT_DC_DEADBAND_EXIT_PITCH_DEG": "1.2",
    "FOWT_DC_DEADBAND_EXIT_ROLL_DEG": "1.2",
    "FOWT_DC_PID_DEADBAND_PITCH_DEG": "1.2",
    "FOWT_DC_PID_DEADBAND_ROLL_DEG": "1.2",
    "FOWT_DC_PID_DEADBAND_EXIT_DEG": "0.65",
}


def _run_casebook(cases_csv: Path, out_dir: Path) -> None:
    env = os.environ.copy()
    env.update(DEADBAND_ENV)
    cmd = [
        sys.executable,
        "scripts/analysis/run_prediction_primary_casebook.py",
        "--cases-csv",
        str(cases_csv),
        "--out-dir",
        str(out_dir),
        "--duration-s",
        "21600",
        "--skip-figures",
        "--primary-control-profile",
        "dc_preserving_deadband_engineered_v1",
        "--allow-isolated-profile",
        "--primary-pump-profile",
        "engineered_minimal",
        "--closed-pump-profile",
        "engineered_minimal",
        "--forecast-source",
        "learned",
        "--event-reset-mode",
        "action",
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _summarize() -> None:
    cmd = [
        sys.executable,
        "scripts/analysis/summarize_selector_positive_add40_combined_v1.py",
        "--old-summary",
        str(OLD_OUT / "casebook_summary.csv"),
        "--new-summary",
        str(NEW_OUT / "casebook_summary.csv"),
        "--out-dir",
        str(COMBINED_OUT),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _write_protocol() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Reduce positive-only attitude increase to about 0.4 deg by experimentally rerunning the controller with lower pump-saving aggressiveness.",
        "changed_control_parameters": DEADBAND_ENV,
        "unchanged_casebooks": {
            "existing_101_cases": str(OLD_CASES),
            "add40_120_cases": str(NEW_CASES),
        },
        "unchanged_runtime_options": {
            "duration_s": 21600,
            "primary_control_profile": "dc_preserving_deadband_engineered_v1",
            "primary_pump_profile": "engineered_minimal",
            "closed_pump_profile": "engineered_minimal",
            "forecast_source": "learned",
            "event_reset_mode": "action",
            "skip_figures": True,
        },
        "outputs": {
            "existing_101": str(OLD_OUT),
            "add40_120": str(NEW_OUT),
            "combined_positive_170": str(COMBINED_OUT),
        },
    }
    (OUT_ROOT / "conservative_deadband_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )


def main() -> None:
    _write_protocol()
    _run_casebook(OLD_CASES, OLD_OUT)
    _run_casebook(NEW_CASES, NEW_OUT)
    _summarize()


if __name__ == "__main__":
    main()
