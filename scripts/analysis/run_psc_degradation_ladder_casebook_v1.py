#!/usr/bin/env python3
"""Run the deployable PSC degradation-ladder arms on a supplied casebook.

This is a thin reproducibility wrapper around run_prediction_primary_casebook.py.
It keeps the arm definitions used by the 2026-05-31 PSC selector audit in one
place so larger mixed-pool checks do not depend on hand-copied commands.
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "analysis" / "run_prediction_primary_casebook.py"


ARMS: dict[str, list[str]] = {
    "blind_deadband": [
        "--primary-control-profile",
        "dc_preserving_deadband_v1",
        "--forecast-source",
        "current_only",
    ],
    "current_forecast_adaptive": [
        "--primary-control-profile",
        "dc_preserving_deadband_forecast_adaptive_v1",
        "--forecast-source",
        "current_only",
    ],
    "learned_forecast_adaptive": [
        "--primary-control-profile",
        "dc_preserving_deadband_forecast_adaptive_v1",
        "--forecast-source",
        "learned",
    ],
    "current_rawenv_mainline": [
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--forecast-source",
        "current_only",
        "--planner-envelope-raw",
        "--planner-envelope-barrier",
        "--relief-medium-cap",
        "--relief-medium-cap-ratio",
        "0.30",
    ],
    "learned_rawenv_mainline": [
        "--primary-control-profile",
        "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
        "--forecast-source",
        "learned",
        "--planner-envelope-raw",
        "--planner-envelope-barrier",
        "--relief-medium-cap",
        "--relief-medium-cap-ratio",
        "0.30",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=7200.0)
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument("--replay-split", choices=("test", "validation", "train"), default="test")
    parser.add_argument("--model-dir", default="outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1")
    parser.add_argument(
        "--arms",
        default=",".join(ARMS),
        help="Comma-separated subset of arms to run.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _command(args: argparse.Namespace, arm: str) -> list[str]:
    out_dir = args.output_dir / arm
    return [
        sys.executable,
        str(RUNNER),
        "--out-dir",
        str(out_dir),
        "--cases-csv",
        str(args.cases_csv),
        "--duration-s",
        str(float(args.duration_s)),
        "--dataset-dir",
        str(args.dataset_dir),
        "--replay-split",
        str(args.replay_split),
        "--model-dir",
        str(args.model_dir),
        "--primary-label",
        arm,
        "--primary-only",
        "--skip-figures",
        *ARMS[arm],
    ]


def _read_case_metrics(run_dir: Path, arm: str) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / "casebook_summary.csv")
    rows: list[dict[str, Any]] = []
    for row in summary.to_dict("records"):
        case_id = str(row["case_id"])
        matches = glob.glob(str(run_dir / "timeseries" / f"{case_id}_*_{arm}_timeseries.csv"))
        if not matches:
            raise FileNotFoundError(f"missing timeseries for {arm}/{case_id}")
        ts = pd.read_csv(matches[0], low_memory=False)
        max_axis = ts[["pitch_deg", "roll_deg"]].abs().max(axis=1)
        fallback = (
            ts.get("preview_primary_safety_fallback", pd.Series(0, index=ts.index))
            .fillna(0)
            .astype(float)
            .gt(0)
        )
        rows.append(
            {
                "arm": arm,
                "case_id": case_id,
                "label": str(row.get("label", "")),
                "pump_m3": float(row["primary_pump_work_m3"]),
                "time_gt5_s": int((max_axis > 5.0).sum()),
                "time_gt6_s": int((max_axis > 6.0).sum()),
                "max_axis_deg": float(max_axis.max()),
                "p95_axis_deg": float(max_axis.quantile(0.95)),
                "fallback_s": int(fallback.sum()),
                "latch_switches": float(row.get("primary_latch_switches", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def summarize(output_dir: Path, arms: list[str]) -> None:
    metrics = pd.concat([_read_case_metrics(output_dir / arm, arm) for arm in arms], ignore_index=True)
    metrics.to_csv(output_dir / "degradation_ladder_case_metrics.csv", index=False)
    summary = (
        metrics.groupby("arm", sort=False)
        .agg(
            cases=("case_id", "size"),
            pump_m3=("pump_m3", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            time_gt6_s=("time_gt6_s", "sum"),
            fallback_s=("fallback_s", "sum"),
            latch_switches=("latch_switches", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "degradation_ladder_summary.csv", index=False)
    print(summary.to_string(index=False))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arms = [arm.strip() for arm in str(args.arms).split(",") if arm.strip()]
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")

    for arm in arms:
        out_dir = args.output_dir / arm
        summary = out_dir / "casebook_summary.csv"
        if args.skip_existing and summary.exists():
            print(f"skip existing {arm}: {summary}")
            continue
        cmd = _command(args, arm)
        print("+ " + " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if not args.dry_run:
        summarize(args.output_dir, arms)


if __name__ == "__main__":
    main()
