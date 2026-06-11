#!/usr/bin/env python3
"""Planner dry-run runner (A1.4 gate).

Replaces the old a1_pilot / a1_1 / a1_2 / a1_3 / a1_4 scripts.
All planner logic lives in src/wind_prediction/ballast_planner.py.
This script only handles IO, window selection, and the summary report.

Usage:
    python scripts/analysis/run_planner_dryrun.py
    python scripts/analysis/run_planner_dryrun.py --windows expanded
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction.ballast_planner import PlannerConfig, run_planner_on_windows
from wind_prediction.replay_dataset import Fino1ReplayDataset


BASE_OUT = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", choices=["original", "expanded"], default="original",
                        help="original=14-window set; expanded=re-labelled 4-group set")
    args = parser.parse_args()

    t0 = time.perf_counter()
    BASE_OUT.mkdir(parents=True, exist_ok=True)

    sign_cfg = json.loads((BASE_OUT / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((BASE_OUT / "diagnostics" / "a1_block_discount_config.json").read_text())

    if args.windows == "expanded":
        windows_path = BASE_OUT / "window_selection_expanded_preview.csv"
    else:
        windows_path = BASE_OUT / "window_selection_preview.csv"
    windows = pd.read_csv(windows_path)

    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    discount_profiles = {
        "default_discount": discount_cfg["default_discount_blocks"],
        "no_discount": discount_cfg["no_discount_blocks"],
    }

    results = run_planner_on_windows(windows, replay, cfg, discount_profiles)

    tag = "expanded" if args.windows == "expanded" else "a1_4"
    results.to_csv(BASE_OUT / f"{tag}_window_results.csv", index=False)

    action_rows = []
    for (profile, group), sub in results.groupby(["discount_profile", "a1_group"]):
        total = len(sub)
        for action, cnt in sub["first_action"].value_counts().items():
            action_rows.append({"discount_profile": profile, "a1_group": group,
                                 "first_action": action, "count": int(cnt),
                                 "ratio": round(cnt / max(total, 1), 3)})
    summary = pd.DataFrame(action_rows)
    summary.to_csv(BASE_OUT / f"{tag}_action_summary.csv", index=False)

    lex = (results.groupby(["discount_profile", "winning_dimension"])
           .size().reset_index(name="count")
           .sort_values(["discount_profile", "count"], ascending=[True, False]))
    lex.to_csv(BASE_OUT / f"diagnostics/{tag}_lex_summary.csv", index=False)

    # ---- quick console report ----
    default = results[results["discount_profile"] == "default_discount"]
    print(f"\n=== Planner dry-run ({args.windows}, {time.perf_counter()-t0:.1f}s) ===")
    print("\nAction distribution (default_discount):")
    print(summary[summary["discount_profile"] == "default_discount"].to_string(index=False))
    print("\nWinning dimension (default_discount):")
    print(lex[lex["discount_profile"] == "default_discount"].to_string(index=False))


if __name__ == "__main__":
    main()
