#!/usr/bin/env python3
"""Expand the neutral moderate-high steady opportunity casebook.

The rule is fixed from the prior neutral profile:

* neutral_subtype == neutral_moderate_high_steady
* select high, steady, low-direction-change windows by pre-outcome wind-shape score
* keep test/validation balanced and spaced in time

This script does not inspect controller outcomes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
PROFILE = ROOT / "neutral_untyped_profile_v1" / "raw_tables" / "neutral_untyped_profile_rows.csv"
OUT = ROOT / "neutral_moderate_high_steady_expanded_v1"
OUT_RAW = OUT / "raw_tables"
CASEBOOKS = OUT / "casebooks"


MAX_PER_SPLIT = 16
MIN_SPACING_S = 12 * 3600


def score(df: pd.DataFrame) -> pd.Series:
    # Fixed pre-outcome score: high pressure, low variation, low direction change.
    return (
        1.5 * df["early_max_ms"].fillna(0.0)
        + 0.5 * df["near_max_ms"].fillna(0.0)
        - 0.7 * df["near_range_ms"].fillna(0.0)
        - 0.7 * df["far_range_ms"].fillna(0.0)
        - 0.04 * df["dir_shift_abs_max_deg"].fillna(0.0)
    )


def spaced_top(df: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    work = df.copy()
    work["selection_score"] = score(work)
    work = work.sort_values("selection_score", ascending=False)
    keep = []
    taken: list[pd.Timestamp] = []
    for _, row in work.iterrows():
        t = pd.to_datetime(row["future_start"])
        if all(abs((t - prev).total_seconds()) >= MIN_SPACING_S for prev in taken):
            keep.append(row)
            taken.append(t)
        if len(keep) >= max_cases:
            break
    return pd.DataFrame(keep)


def casebook(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        label = (
            "neutral_moderate_high_steady_expanded"
            f" | split={row['split']}"
            f" | early={float(row['early_max_ms']):.1f}"
            f" | drop={float(row['peak_to_late_drop_ms']):.1f}"
            f" | dir={float(row['dir_shift_abs_max_deg']):.0f}"
        )
        rows.append(
            {
                "case_id": f"neutral_mhs_exp_{i:02d}",
                "timestamp": str(row["future_start"]),
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROFILE)
    pool = df[df["neutral_subtype"] == "neutral_moderate_high_steady"].copy()
    selected = []
    for split in ("test", "validation"):
        selected.append(spaced_top(pool[pool["split"] == split], MAX_PER_SPLIT))
    sel = pd.concat(selected, ignore_index=True)
    sel.to_csv(OUT_RAW / "neutral_moderate_high_steady_expanded_selected_features.csv", index=False)
    cb = casebook(sel)
    cb.to_csv(CASEBOOKS / "neutral_moderate_high_steady_expanded_cases.csv", index=False)
    summary = (
        sel.groupby("split")
        .agg(
            cases=("neutral_subtype", "size"),
            early_max_median_ms=("early_max_ms", "median"),
            peak_to_late_drop_median_ms=("peak_to_late_drop_ms", "median"),
            dir_shift_median_deg=("dir_shift_abs_max_deg", "median"),
            score_median=("selection_score", "median"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_RAW / "neutral_moderate_high_steady_expanded_selection_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(CASEBOOKS / "neutral_moderate_high_steady_expanded_cases.csv")


if __name__ == "__main__":
    main()
