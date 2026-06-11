#!/usr/bin/env python3
"""Build a broader neutral moderate-high steady casebook with fixed rules."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
PROFILE = ROOT / "neutral_untyped_profile_v1" / "raw_tables" / "neutral_untyped_profile_rows.csv"
OUT = ROOT / "neutral_moderate_high_steady_broader_v1"
RAW_DIR = OUT / "raw_tables"
CASEBOOKS = OUT / "casebooks"

MAX_PER_SPLIT = 32
MIN_SPACING_S = 6 * 3600


def score(df: pd.DataFrame) -> pd.Series:
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


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROFILE)
    pool = df[df["neutral_subtype"] == "neutral_moderate_high_steady"].copy()
    selected = pd.concat(
        [spaced_top(pool[pool["split"] == split], MAX_PER_SPLIT) for split in ("test", "validation")],
        ignore_index=True,
    )
    selected.to_csv(RAW_DIR / "neutral_mhs_broader_selected_features.csv", index=False)
    rows = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        rows.append(
            {
                "case_id": f"neutral_mhs_broader_{i:02d}",
                "timestamp": str(row["future_start"]),
                "label": (
                    "neutral_moderate_high_steady_broader"
                    f" | split={row['split']}"
                    f" | early={float(row['early_max_ms']):.1f}"
                    f" | drop={float(row['peak_to_late_drop_ms']):.1f}"
                    f" | dir={float(row['dir_shift_abs_max_deg']):.0f}"
                ),
            }
        )
    casebook = pd.DataFrame(rows)
    casebook.to_csv(CASEBOOKS / "neutral_mhs_broader_cases.csv", index=False)
    summary = (
        selected.groupby("split")
        .agg(
            cases=("neutral_subtype", "size"),
            early_max_median_ms=("early_max_ms", "median"),
            peak_to_late_drop_median_ms=("peak_to_late_drop_ms", "median"),
            near_range_median_ms=("near_range_ms", "median"),
            far_range_median_ms=("far_range_ms", "median"),
            dir_shift_median_deg=("dir_shift_abs_max_deg", "median"),
            score_median=("selection_score", "median"),
        )
        .reset_index()
    )
    summary.to_csv(RAW_DIR / "neutral_mhs_broader_selection_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(CASEBOOKS / "neutral_mhs_broader_cases.csv")


if __name__ == "__main__":
    main()
