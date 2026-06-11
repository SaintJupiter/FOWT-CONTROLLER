#!/usr/bin/env python3
"""Compare A0 vs a fixed budget100 smoke on the neutral moderate-high clean subset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_untyped_opportunity_screen_v1"
OUT_RAW = BASE / "raw_tables"
OUT_PAPER = BASE / "paper_ready"

A0 = BASE / "a0_neutral_moderate_high_steady_1h" / "casebook_summary.csv"
BUDGET = BASE / "budget100_neutral_moderate_high_steady_clean_1h" / "casebook_summary.csv"
OPP = OUT_RAW / "neutral_a0_opportunity_per_case.csv"


def compact(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        "timestamp",
        "case_id",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
    ]
    out = df[cols].copy()
    return out.rename(columns={c: f"{prefix}_{c}" for c in cols if c != "timestamp"})


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    opp = pd.read_csv(OPP)
    clean_ts = set(
        opp[
            (opp["neutral_subtype"] == "neutral_moderate_high_steady")
            & (opp["clean_economy_opportunity"])
        ]["timestamp"].astype(str)
    )
    a0 = pd.read_csv(A0)
    budget = pd.read_csv(BUDGET)
    a0 = a0[a0["timestamp"].astype(str).isin(clean_ts)].copy()
    merged = compact(a0, "a0").merge(compact(budget, "budget100"), on="timestamp", how="inner")
    merged["pump_saved_m3"] = merged["a0_primary_pump_work_m3"] - merged["budget100_primary_pump_work_m3"]
    merged["pump_saving_pct"] = 100.0 * merged["pump_saved_m3"] / merged["a0_primary_pump_work_m3"].clip(lower=1e-9)
    merged["a0_p95_max_axis_deg"] = merged[["a0_primary_pitch_p95", "a0_primary_roll_p95"]].abs().max(axis=1)
    merged["budget100_p95_max_axis_deg"] = merged[
        ["budget100_primary_pitch_p95", "budget100_primary_roll_p95"]
    ].abs().max(axis=1)
    merged["delta_p95_max_axis_deg"] = merged["budget100_p95_max_axis_deg"] - merged["a0_p95_max_axis_deg"]
    merged["delta_fallback_ratio"] = (
        merged["budget100_primary_safety_fallback_ratio"] - merged["a0_primary_safety_fallback_ratio"]
    )
    merged.to_csv(OUT_RAW / "neutral_moderate_high_steady_clean_budget100_delta.csv", index=False)

    total_a0 = float(merged["a0_primary_pump_work_m3"].sum())
    total_budget = float(merged["budget100_primary_pump_work_m3"].sum())
    saved = total_a0 - total_budget
    saving_pct = 100.0 * saved / max(total_a0, 1e-9)
    summary = pd.DataFrame(
        [
            {
                "subset": "neutral_moderate_high_steady_clean_1h",
                "cases": int(len(merged)),
                "a0_pump_m3": total_a0,
                "budget100_pump_m3": total_budget,
                "pump_saved_m3": saved,
                "pump_saving_pct": saving_pct,
                "delta_fallback_ratio_max": float(merged["delta_fallback_ratio"].max()),
                "delta_p95_max_axis_mean_deg": float(merged["delta_p95_max_axis_deg"].mean()),
                "top_case_share_of_saved": float(
                    merged["pump_saved_m3"].max() / saved if saved > 1e-9 else 0.0
                ),
            }
        ]
    )
    summary.to_csv(OUT_RAW / "neutral_moderate_high_steady_clean_budget100_summary.csv", index=False)

    md = [
        "# Neutral Moderate-High Steady Fixed-Action Smoke",
        "",
        "This is a 1h smoke test on the clean A0-opportunity subset only. It is not a mature automatic strategy.",
        "",
        f"- cases: {len(merged)}",
        f"- A0 pump: {total_a0:.2f} m3",
        f"- budget100 pump: {total_budget:.2f} m3",
        f"- pump saved: {saved:.2f} m3 ({saving_pct:.2f}%)",
        f"- max fallback-ratio delta: {float(merged['delta_fallback_ratio'].max()):.4f}",
        f"- mean p95 max-axis delta: {float(merged['delta_p95_max_axis_deg'].mean()):.3f} deg",
        "",
        "## Interpretation",
        "",
        "The clean neutral moderate-high steady subset has a real budget-style pump-saving pool in this short smoke.",
        "However, this is still a blind fixed-action result, not an automatic forecast-gated controller. The next methodological step would be outcome-labeling plus pre-action detector separability, not threshold tuning.",
        "",
    ]
    text = "\n".join(md)
    (BASE / "neutral_moderate_high_steady_clean_budget100_decision.md").write_text(text, encoding="utf-8")
    (OUT_PAPER / "neutral_moderate_high_steady_clean_budget100_summary.md").write_text(
        text, encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
