#!/usr/bin/env python3
"""Build small smoke-test casebooks from neutral_untyped subtypes.

This is a pre-controller screen.  It exports representative cases for the two
neutral subtypes that are plausible enough to justify an A0 pump-opportunity
check:

* neutral_mild_decay
* neutral_moderate_high_steady

The selection uses only pre-outcome wind-shape features already computed by
neutral_untyped_profile_v1.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
PROFILE = ROOT / "neutral_untyped_profile_v1" / "raw_tables" / "neutral_untyped_profile_rows.csv"
OUT = ROOT / "neutral_untyped_opportunity_screen_v1"
OUT_RAW = OUT / "raw_tables"
CASEBOOKS = OUT / "casebooks"


TARGETS = {
    "neutral_mild_decay": {
        "max_cases": 8,
        "score": lambda df: (
            1.2 * df["early_max_ms"].fillna(0)
            + 1.5 * df["peak_to_late_drop_ms"].fillna(0)
            + 0.4 * df["near_range_ms"].fillna(0)
            - 0.03 * df["dir_shift_abs_max_deg"].fillna(0)
        ),
    },
    "neutral_moderate_high_steady": {
        "max_cases": 8,
        "score": lambda df: (
            1.5 * df["early_max_ms"].fillna(0)
            - 0.8 * df["near_range_ms"].fillna(0)
            - 0.8 * df["far_range_ms"].fillna(0)
            - 0.05 * df["dir_shift_abs_max_deg"].fillna(0)
        ),
    },
}


def spaced_top(df: pd.DataFrame, score: pd.Series, max_cases: int, min_spacing_s: int = 3600) -> pd.DataFrame:
    work = df.copy()
    work["score"] = score
    work = work.sort_values("score", ascending=False)
    keep = []
    taken: list[pd.Timestamp] = []
    for _, row in work.iterrows():
        t = pd.to_datetime(row["future_start"])
        if all(abs((t - prev).total_seconds()) >= min_spacing_s for prev in taken):
            keep.append(row)
            taken.append(t)
        if len(keep) >= max_cases:
            break
    if not keep:
        return work.head(max_cases)
    return pd.DataFrame(keep)


def casebook_rows(df: pd.DataFrame, subtype: str) -> pd.DataFrame:
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        label = (
            f"{subtype} | split={row['split']} | "
            f"early={float(row['early_max_ms']):.1f} | "
            f"drop={float(row['peak_to_late_drop_ms']):.1f} | "
            f"dir={float(row['dir_shift_abs_max_deg']):.0f}"
        )
        rows.append(
            {
                "case_id": f"{subtype}_{i:02d}",
                "timestamp": str(row["future_start"]),
                "label": label,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROFILE)
    summary_rows = []
    for subtype, cfg in TARGETS.items():
        sub = df[df["neutral_subtype"] == subtype].copy()
        selected = spaced_top(sub, cfg["score"](sub), int(cfg["max_cases"]))
        selected.to_csv(OUT_RAW / f"{subtype}_selected_features.csv", index=False)
        cb = casebook_rows(selected, subtype)
        cb.to_csv(CASEBOOKS / f"{subtype}_cases.csv", index=False)
        summary_rows.append(
            {
                "subtype": subtype,
                "available_rows": int(len(sub)),
                "selected_cases": int(len(cb)),
                "early_max_median_ms": float(selected["early_max_ms"].median()),
                "peak_to_late_drop_median_ms": float(selected["peak_to_late_drop_ms"].median()),
                "dir_shift_median_deg": float(selected["dir_shift_abs_max_deg"].median()),
                "casebook": str(CASEBOOKS / f"{subtype}_cases.csv"),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_RAW / "neutral_opportunity_casebook_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
