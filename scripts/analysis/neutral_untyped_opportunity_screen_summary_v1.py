#!/usr/bin/env python3
"""Summarize A0 pump-opportunity smoke runs for neutral_untyped candidates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_untyped_opportunity_screen_v1"
OUT_RAW = BASE / "raw_tables"
OUT_PAPER = BASE / "paper_ready"

RUNS = {
    "neutral_mild_decay": BASE / "a0_neutral_mild_decay_1h" / "casebook_summary.csv",
    "neutral_moderate_high_steady": BASE / "a0_neutral_moderate_high_steady_1h" / "casebook_summary.csv",
}


def load() -> pd.DataFrame:
    parts = []
    for subtype, path in RUNS.items():
        df = pd.read_csv(path)
        df["neutral_subtype"] = subtype
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    df = load()
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

    per_case_cols = [
        "neutral_subtype",
        "case_id",
        "timestamp",
        "primary_pump_work_m3",
        "p95_max_axis_deg",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
        "material_a0_pump",
        "clean_economy_opportunity",
        "safety_margin_heavy",
        "label",
    ]
    per_case = df[per_case_cols].copy()
    per_case.to_csv(OUT_RAW / "neutral_a0_opportunity_per_case.csv", index=False)

    summary = (
        df.groupby("neutral_subtype")
        .agg(
            cases=("case_id", "size"),
            total_a0_pump_m3=("primary_pump_work_m3", "sum"),
            median_a0_pump_m3=("primary_pump_work_m3", "median"),
            material_pump_cases=("material_a0_pump", "sum"),
            clean_economy_cases=("clean_economy_opportunity", "sum"),
            safety_margin_heavy_cases=("safety_margin_heavy", "sum"),
            median_p95_max_axis_deg=("p95_max_axis_deg", "median"),
            median_fallback_ratio=("primary_safety_fallback_ratio", "median"),
            median_latch_switches=("primary_latch_switches", "median"),
        )
        .reset_index()
    )
    summary["clean_case_share"] = summary["clean_economy_cases"] / summary["cases"]
    summary.to_csv(OUT_RAW / "neutral_a0_opportunity_summary.csv", index=False)

    cols = [
        "neutral_subtype",
        "cases",
        "total_a0_pump_m3",
        "median_a0_pump_m3",
        "material_pump_cases",
        "clean_economy_cases",
        "safety_margin_heavy_cases",
        "median_p95_max_axis_deg",
        "median_fallback_ratio",
        "clean_case_share",
    ]
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in summary[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{float(val):.3f}" if isinstance(val, float) else str(val))
        rows.append("| " + " | ".join(vals) + " |")

    decision = [
        "# Neutral Untyped A0 Opportunity Screen v1",
        "",
        "This is a 1h A0-only smoke screen. It checks whether two neutral subtypes have material pump opportunity before any controller action is designed.",
        "",
        "\n".join(rows),
        "",
        "## Interpretation",
        "",
        "- Both candidate subtypes have material A0 pump, so neither is immediately discarded.",
        "- Both also contain safety-margin-heavy cases, especially repeated high-posture patterns. A future pump-saving action must be gated away from those cases.",
        "- `neutral_moderate_high_steady` has the cleaner economy-opportunity share in this small screen; it is the better next candidate if one more fixed-action test is allowed.",
        "- `neutral_mild_decay` has pump opportunity, but several samples are already high-posture/fallback-heavy, so it is riskier and should not be merged blindly into the mature relief/decay action.",
        "",
        "## Next Step",
        "",
        "If continuing, run one short fixed-action smoke on the clean subset only: baseline A0 vs a conservative existing action. Do not create a new controller branch yet.",
        "",
    ]
    text = "\n".join(decision)
    (BASE / "decision.md").write_text(text, encoding="utf-8")
    (OUT_PAPER / "neutral_a0_opportunity_screen_summary.md").write_text(text, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nWrote {BASE / 'decision.md'}")


if __name__ == "__main__":
    main()
