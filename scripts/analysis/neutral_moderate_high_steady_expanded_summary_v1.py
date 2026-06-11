#!/usr/bin/env python3
"""Summarize expanded neutral moderate-high steady A0 and export clean subset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_moderate_high_steady_expanded_v1"
OUT_RAW = BASE / "raw_tables"
CASEBOOKS = BASE / "casebooks"
A0 = BASE / "a0_1h" / "casebook_summary.csv"
FEATURES = OUT_RAW / "neutral_moderate_high_steady_expanded_selected_features.csv"


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    a0 = pd.read_csv(A0)
    feat = pd.read_csv(FEATURES)
    feat = feat.copy()
    feat["timestamp"] = feat["future_start"].astype(str)
    compact = feat[
        [
            "timestamp",
            "split",
            "early_max_ms",
            "peak_to_late_drop_ms",
            "near_range_ms",
            "far_range_ms",
            "dir_shift_abs_max_deg",
            "selection_score",
        ]
    ]
    df = a0.merge(compact, on="timestamp", how="left")
    df["p95_max_axis_deg"] = df[["primary_pitch_p95", "primary_roll_p95"]].abs().max(axis=1)
    df["material_a0_pump"] = df["primary_pump_work_m3"] >= 100.0
    df["clean_economy_opportunity"] = (
        (df["primary_pump_work_m3"] >= 100.0)
        & (df["primary_safety_fallback_ratio"] <= 0.01)
        & (df["p95_max_axis_deg"] <= 4.0)
    )
    df["safety_margin_heavy"] = (
        (df["primary_safety_fallback_ratio"] >= 0.05)
        | (df["p95_max_axis_deg"] >= 5.0)
    )
    df.to_csv(OUT_RAW / "neutral_mhs_expanded_a0_opportunity_per_case.csv", index=False)

    summary = (
        df.groupby("split")
        .agg(
            cases=("case_id", "size"),
            total_a0_pump_m3=("primary_pump_work_m3", "sum"),
            clean_cases=("clean_economy_opportunity", "sum"),
            safety_margin_heavy_cases=("safety_margin_heavy", "sum"),
            median_p95_max_axis_deg=("p95_max_axis_deg", "median"),
            median_fallback_ratio=("primary_safety_fallback_ratio", "median"),
            median_a0_pump_m3=("primary_pump_work_m3", "median"),
        )
        .reset_index()
    )
    summary["clean_case_share"] = summary["clean_cases"] / summary["cases"]
    summary.to_csv(OUT_RAW / "neutral_mhs_expanded_a0_opportunity_summary.csv", index=False)

    clean = df[df["clean_economy_opportunity"]].copy()
    clean_casebook = pd.DataFrame(
        {
            "case_id": [f"neutral_mhs_clean_exp_{i+1:02d}" for i in range(len(clean))],
            "timestamp": clean["timestamp"].astype(str).values,
            "label": clean["label"].astype(str).values + " | clean_a0_opportunity",
        }
    )
    clean_casebook.to_csv(CASEBOOKS / "neutral_mhs_expanded_clean_cases.csv", index=False)

    print(summary.to_string(index=False))
    print(f"clean cases: {len(clean_casebook)}")
    print(CASEBOOKS / "neutral_mhs_expanded_clean_cases.csv")


if __name__ == "__main__":
    main()
