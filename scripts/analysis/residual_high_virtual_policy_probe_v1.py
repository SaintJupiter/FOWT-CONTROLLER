#!/usr/bin/env python3
"""Virtual policy probes for the residual-high candidate regime.

This is a read-only screen: it combines observed A0/budget100 deltas with
runtime-observable labels and forecast-shape features to estimate whether a
residual-high specialist is worth implementing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
MATRIX = BASE / "residual_high_7200/raw_tables/candidate_matrix_case_deltas.csv"
TAX = BASE / "raw_tables/regime_taxonomy_case_table.csv"
OUT = BASE / "residual_high_7200/raw_tables"
PAPER = BASE / "residual_high_7200/paper_ready"


def policy_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    label = df["label"].fillna("").str.lower()
    case = df["case_id"].fillna("").str.lower()
    saved = df["pump_saved_m3_vs_baseline"].fillna(0.0)
    p95_cost = df["delta_p95_vs_baseline"].fillna(0.0)
    fallback_cost = df["delta_fallback_time_s_vs_baseline"].fillna(0.0)
    all_cases = pd.Series(True, index=df.index)

    return {
        "budget100_all_residual_high": all_cases,
        "exclude_future_relief_label": ~label.str.contains("future_relief|fr_relief"),
        "high_pressure_or_onset_only": label.str.contains("b/high|high_pressure|onset|decay"),
        "observed_low_cost_budget100": (p95_cost <= 0.25) & (fallback_cost <= 0.0),
        "observed_positive_and_low_cost": (saved > 40.0) & (p95_cost <= 0.25) & (fallback_cost <= 0.0),
        "regime_auto_current_hits": saved > 1e9,  # placeholder replaced below
    }


def summarize_virtual(df: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, float | str | int]:
    selected = df[mask]
    a0_total = df["pump_m3_baseline"].sum()
    saved = selected["pump_saved_m3_vs_baseline"].sum()
    return {
        "virtual_policy": name,
        "selected_cases": len(selected),
        "selected_a0_pump_m3": selected["pump_m3_baseline"].sum(),
        "saved_m3_vs_baseline": saved,
        "saving_pct_of_full_residual_high_baseline": 100.0 * saved / a0_total if a0_total else 0.0,
        "saving_pct_within_selected": (
            100.0 * saved / selected["pump_m3_baseline"].sum()
            if selected["pump_m3_baseline"].sum()
            else 0.0
        ),
        "delta_fallback_time_s": selected["delta_fallback_time_s_vs_baseline"].sum(),
        "mean_delta_p95_deg": selected["delta_p95_vs_baseline"].mean() if len(selected) else 0.0,
        "selected_cases_list": ";".join(selected["case_id"].astype(str).tolist()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(MATRIX)
    tax = pd.read_csv(TAX)
    budget = cases[cases["arm"] == "budget100"].copy()
    auto = cases[cases["arm"] == "regime_auto"][["case_id", "pump_saved_m3_vs_baseline"]].rename(
        columns={"pump_saved_m3_vs_baseline": "regime_auto_saved_m3"}
    )
    tax_cols = [c for c in ["case", "case_id", "label", "max_near_peak", "max_near_drop", "max_far_relief_drop"] if c in tax.columns]
    enriched = budget.merge(auto, on="case_id", how="left").merge(
        tax[tax_cols].rename(columns={"case": "taxonomy_case", "label": "taxonomy_label"}),
        on="case_id",
        how="left",
    )
    masks = policy_masks(enriched)
    masks["regime_auto_current_hits"] = enriched["regime_auto_saved_m3"].fillna(0.0) > 0.0

    rows = [summarize_virtual(enriched, mask, name) for name, mask in masks.items()]
    out = pd.DataFrame(rows).sort_values("saved_m3_vs_baseline", ascending=False)
    out.to_csv(OUT / "residual_high_virtual_policy_probe.csv", index=False)
    enriched.to_csv(OUT / "residual_high_virtual_policy_case_features.csv", index=False)

    lines = ["# Residual-High Virtual Policy Probe\n"]
    lines.append(
        "This is a read-only probe using observed budget100 deltas. It estimates whether a residual-high specialist is worth implementing before changing controller code.\n"
    )
    lines.append(out.to_string(index=False))
    lines.append("\n\n## Provisional Decision\n")
    best = out.iloc[0]
    lines.append(
        f"The largest virtual saving is `{best['virtual_policy']}` with {best['saved_m3_vs_baseline']:.1f} m3 saved "
        f"({best['saving_pct_of_full_residual_high_baseline']:.1f}% of the full residual-high baseline). "
        "Use this only as a design screen, not as a deployable result.\n"
    )
    (PAPER / "residual_high_virtual_policy_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
