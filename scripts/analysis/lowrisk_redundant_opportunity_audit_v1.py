#!/usr/bin/env python3
"""Read-only screen for low-risk redundant refill opportunities.

This script does not run the controller and does not tune thresholds.  It uses
the existing locked53 budget100 mechanism table to ask whether low-risk /
quiet-pump-saving is a broad third opportunity domain or just a sparse artifact.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("outputs/wind_prediction")
IN_TABLE = (
    BASE
    / "economy_pump_budget_closed_loop_probe_v1"
    / "budget100_final_candidate_v1"
    / "raw_tables"
    / "budget100_regime_mechanism_case_table.csv"
)
OUT = (
    BASE
    / "regime_conditioned_policy_development_v1"
    / "lowrisk_redundant_opportunity_audit_v1"
)
OUT_RAW = OUT / "raw_tables"
OUT_PAPER = OUT / "paper_ready"


def boolish(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.fillna("").astype(str).str.lower().isin(["1", "true", "yes", "y"])
    return pd.to_numeric(series, errors="coerce").fillna(0) > 0


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._\n"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].head(max_rows).iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_TABLE)

    numeric_cols = [
        "pump_m3_a0",
        "pump_saved_m3",
        "pump_saving_pct",
        "delta_time_gt5_s",
        "delta_fallback_time_s",
        "delta_p95_max_axis_deg",
        "max_near_peak",
        "max_near_drop",
        "max_far_relief_drop",
        "near_high_rows",
        "high_then_relief_rows",
        "reversal_or_signflip_rows",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    lowrisk_label = boolish(df.get("is_lowrisk_label", pd.Series(False, index=df.index)))
    low_pressure = df["max_near_peak"] <= 0.75
    no_relief_decay = ~boolish(df.get("forecast_relief_decay_candidate", pd.Series(False, index=df.index)))
    no_reversal = df["reversal_or_signflip_rows"] <= 0
    material_pump = df["pump_m3_a0"] >= 100.0
    lowrisk_candidate = lowrisk_label | (low_pressure & no_relief_decay & no_reversal)
    redundant_candidate = lowrisk_candidate & material_pump

    df["lowrisk_candidate"] = lowrisk_candidate.astype(int)
    df["redundant_refill_candidate"] = redundant_candidate.astype(int)

    groups = {
        "all_locked53": pd.Series(True, index=df.index),
        "lowrisk_or_quiet_shape": lowrisk_candidate,
        "lowrisk_with_material_a0_pump_ge100": redundant_candidate,
        "known_lowrisk_labels": lowrisk_label,
        "low_pressure_material_pump_no_reversal": low_pressure & material_pump & no_reversal,
    }
    rows = []
    for name, mask in groups.items():
        g = df[mask].copy()
        a0 = float(g["pump_m3_a0"].sum())
        saved = float(g["pump_saved_m3"].sum())
        positive = g[g["pump_saved_m3"] > 0]
        positive_saved = float(positive["pump_saved_m3"].sum())
        rows.append(
            {
                "group": name,
                "cases": len(g),
                "a0_pump_m3": a0,
                "budget100_saved_m3": saved,
                "budget100_saving_pct": 100.0 * saved / a0 if a0 else 0.0,
                "positive_cases": len(positive),
                "delta_time_gt5_s": float(g["delta_time_gt5_s"].sum()),
                "delta_fallback_time_s": float(g["delta_fallback_time_s"].sum()),
                "mean_delta_p95_deg": float(g["delta_p95_max_axis_deg"].mean()) if len(g) else 0.0,
                "top_case_share": float(positive["pump_saved_m3"].max() / positive_saved)
                if positive_saved > 0 and len(positive)
                else 0.0,
            }
        )
    summary = pd.DataFrame(rows)

    candidates = df[redundant_candidate].copy()
    candidates = candidates.sort_values("pump_saved_m3", ascending=False)
    case_cols = [
        "case",
        "case_id",
        "label",
        "pump_m3_a0",
        "pump_saved_m3",
        "pump_saving_pct",
        "delta_time_gt5_s",
        "delta_fallback_time_s",
        "delta_p95_max_axis_deg",
        "max_near_peak",
        "max_near_drop",
        "reversal_or_signflip_rows",
    ]
    case_cols = [c for c in case_cols if c in candidates.columns]
    summary.to_csv(OUT_RAW / "lowrisk_redundant_summary.csv", index=False)
    candidates[case_cols].to_csv(OUT_RAW / "lowrisk_redundant_candidate_cases.csv", index=False)
    df.to_csv(OUT_RAW / "lowrisk_redundant_screen_all_cases.csv", index=False)

    best = summary[summary["group"] == "lowrisk_with_material_a0_pump_ge100"].iloc[0]
    if int(best["cases"]) >= 8 and float(best["budget100_saving_pct"]) >= 10.0 and float(best["top_case_share"]) < 0.5:
        decision = "CONDITIONAL-GO for more lowrisk mining before a specialist."
    else:
        decision = "NO-GO for a lowrisk automatic specialist now; evidence remains sparse or concentrated."

    lines = ["# Lowrisk redundant refill opportunity audit v1\n"]
    lines.append(f"## Decision\n\n{decision}\n")
    lines.append(
        "This is a read-only screen using the existing locked53 mechanism table. "
        "It asks whether low-risk redundant refill is broad enough to justify a new controller branch.\n"
    )
    lines.append("## Summary\n\n")
    lines.append(
        markdown_table(
            summary,
            [
                "group",
                "cases",
                "a0_pump_m3",
                "budget100_saved_m3",
                "budget100_saving_pct",
                "positive_cases",
                "delta_time_gt5_s",
                "delta_fallback_time_s",
                "mean_delta_p95_deg",
                "top_case_share",
            ],
        )
    )
    lines.append("\n## Candidate Cases\n\n")
    lines.append(markdown_table(candidates, case_cols, max_rows=20))
    lines.append("\n## Interpretation\n\n")
    lines.append(
        "- If the material lowrisk candidate set is small or top-case dominated, "
        "do not add a specialist; keep this as telemetry/advisory.\n"
    )
    lines.append(
        "- A controller branch is only justified after more cases are mined under the same pre-outcome rule.\n"
    )
    text = "\n".join(lines)
    (OUT / "decision.md").write_text(text, encoding="utf-8")
    (OUT_PAPER / "lowrisk_redundant_opportunity_summary.md").write_text(text, encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nWrote", OUT)


if __name__ == "__main__":
    main()
