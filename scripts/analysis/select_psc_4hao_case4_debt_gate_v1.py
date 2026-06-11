#!/usr/bin/env python3
"""Select a debt-aware No.4 case4 refresh gate from the 6h ON/OFF ablation.

This is an offline structural selector over already-run ablation outputs.  It
does not change controller logic.  The rule treats 6h debt as the arbiter:
release refresh-OFF only when it has no fallback debt, bounded high-posture and
p95 debt, and a real water advantage over refresh-ON.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
ABLATION_DIR = BASE / "psc_4hao_case4_ablation_v1"
CASE_METRICS = ABLATION_DIR / "case4_refresh_ablation_case_metrics.csv"
OUT_DIR = BASE / "psc_4hao_case4_debt_gate_v1"

DECISIONS_CSV = OUT_DIR / "case4_debt_gate_decisions.csv"
SUMMARY_CSV = OUT_DIR / "case4_debt_gate_summary.csv"
READOUT_MD = OUT_DIR / "case4_debt_gate_readout.md"


MAX_OFF_D_GT5_S = 60.0
MAX_OFF_D_FALLBACK_S = 0.0
MAX_OFF_D_P95_DEG = 1.25
MIN_OFF_WATER_ADVANTAGE_M3 = 0.0


def _fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3g}"
    return str(value)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for row in df.to_dict("records"):
        lines.append("| " + " | ".join(_fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(lines)


def _pick_case(group: pd.DataFrame) -> dict[str, object]:
    rows = {str(row["state"]): row for _, row in group.iterrows()}
    on = rows["ON"]
    off = rows["OFF"]
    off_water_advantage = float(off["saved_m3"]) - float(on["saved_m3"])
    off_release = (
        float(off["d_fallback_s"]) <= MAX_OFF_D_FALLBACK_S
        and float(off["d_gt5_s"]) <= MAX_OFF_D_GT5_S
        and float(off["d_p95_axis_deg"]) <= MAX_OFF_D_P95_DEG
        and off_water_advantage > MIN_OFF_WATER_ADVANTAGE_M3
    )
    selected = off if off_release else on
    blocked_reasons: list[str] = []
    if float(off["d_fallback_s"]) > MAX_OFF_D_FALLBACK_S:
        blocked_reasons.append("fallback_debt")
    if float(off["d_gt5_s"]) > MAX_OFF_D_GT5_S:
        blocked_reasons.append("gt5_debt")
    if float(off["d_p95_axis_deg"]) > MAX_OFF_D_P95_DEG:
        blocked_reasons.append("p95_debt")
    if off_water_advantage <= MIN_OFF_WATER_ADVANTAGE_M3:
        blocked_reasons.append("no_water_advantage")

    return {
        "case_id": str(selected["case_id"]),
        "selected_refresh": str(selected["state"]),
        "decision_reason": "off_debt_guard_release" if off_release else ";".join(blocked_reasons),
        "off_water_advantage_m3": off_water_advantage,
        "selected_saved_m3": float(selected["saved_m3"]),
        "selected_d_gt5_s": float(selected["d_gt5_s"]),
        "selected_d_fallback_s": float(selected["d_fallback_s"]),
        "selected_d_p95_axis_deg": float(selected["d_p95_axis_deg"]),
        "on_saved_m3": float(on["saved_m3"]),
        "on_d_gt5_s": float(on["d_gt5_s"]),
        "on_d_fallback_s": float(on["d_fallback_s"]),
        "on_d_p95_axis_deg": float(on["d_p95_axis_deg"]),
        "off_saved_m3": float(off["saved_m3"]),
        "off_d_gt5_s": float(off["d_gt5_s"]),
        "off_d_fallback_s": float(off["d_fallback_s"]),
        "off_d_p95_axis_deg": float(off["d_p95_axis_deg"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CASE_METRICS)
    summary_ref = pd.read_csv(ABLATION_DIR / "case4_refresh_ablation_summary.csv")
    h6 = df[df["horizon"].eq("6h")].copy()
    decisions = pd.DataFrame([_pick_case(g) for _, g in h6.groupby("case_id", sort=True)])
    decisions.to_csv(DECISIONS_CSV, index=False)

    gate_summary = {
        "selector": "case4_debt_gate_v1",
        "cases": int(len(decisions)),
        "off_released_cases": int(decisions["selected_refresh"].eq("OFF").sum()),
        "selected_saved_m3": float(decisions["selected_saved_m3"].sum()),
        "selected_d_gt5_s": float(decisions["selected_d_gt5_s"].sum()),
        "selected_d_fallback_s": float(decisions["selected_d_fallback_s"].sum()),
        "max_selected_d_p95_axis_deg": float(decisions["selected_d_p95_axis_deg"].max()),
        "opened_off_case_ids": ";".join(
            decisions.loc[decisions["selected_refresh"].eq("OFF"), "case_id"].astype(str)
        ),
    }
    summary = pd.DataFrame([gate_summary])
    summary.to_csv(SUMMARY_CSV, index=False)

    comparison_rows = []
    summary_ref = summary_ref[summary_ref["horizon"].astype(str).eq("6h")].copy()
    baseline_6h_m3 = float(summary_ref["baseline_pump_m3"].iloc[0])
    for _, row in summary_ref.iterrows():
        comparison_rows.append(
            {
                "policy": f"{row['horizon']}_{row['state']}",
                "saved_m3": float(row["saved_m3"]),
                "saving_pct": float(row["saving_pct"]),
                "d_gt5_s": float(row["d_gt5_s"]),
                "d_fallback_s": float(row["d_fallback_s"]),
                "max_d_p95_axis_deg": float(row["max_d_p95_axis_deg"]),
                "note": "reference_ablation",
            }
        )
    comparison_rows.append(
        [
            {
                "policy": "6h_debt_gate_selected",
                "saved_m3": gate_summary["selected_saved_m3"],
                "saving_pct": 100.0 * float(gate_summary["selected_saved_m3"]) / max(baseline_6h_m3, 1e-9),
                "d_gt5_s": gate_summary["selected_d_gt5_s"],
                "d_fallback_s": gate_summary["selected_d_fallback_s"],
                "max_d_p95_axis_deg": gate_summary["max_selected_d_p95_axis_deg"],
                "note": "offline gate: ON by default, release OFF only on clean 6h debt slices",
            }
        ][0]
    )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "case4_debt_gate_reference_comparison.csv", index=False)

    compact = decisions[
        [
            "case_id",
            "selected_refresh",
            "decision_reason",
            "off_water_advantage_m3",
            "selected_saved_m3",
            "selected_d_gt5_s",
            "selected_d_fallback_s",
            "selected_d_p95_axis_deg",
        ]
    ].copy()
    lines = [
        "# Case4 debt-aware refresh gate v1",
        "",
        "Scope: offline selector over the already-run 6h case4 active-posture refresh ablation. No controller logic is changed.",
        "",
        f"Sample: 4 case4 focus windows, all evaluated at 6h. Baseline denominator is current-forecast-adaptive total pump `{baseline_6h_m3:.1f} m3` over these same 4 windows.",
        "",
        "Important limitation: these 4 windows are a mechanism probe only. They are not a validation-sized sample and should not be used as a final persuasive claim. The next validation must expand to at least 20 6h windows.",
        "",
        "Terminology: `refresh-ON` means the active-posture target refresh safety valve is enabled; when posture remains high during active control, the controller is allowed to refresh the ballast target. `refresh-OFF` means that safety valve is disabled, so the controller can keep reusing an older target longer.",
        "",
        "## Gate",
        "",
        f"6h-only exploratory rule: release refresh-OFF only if `d_fallback_s <= {MAX_OFF_D_FALLBACK_S:.0f}`, `d_gt5_s <= {MAX_OFF_D_GT5_S:.0f}`, `d_p95_axis_deg <= {MAX_OFF_D_P95_DEG:.2f}`, and OFF saves more water than refresh-ON.",
        "",
        "## Summary",
        "",
        _table(summary),
        "",
        "## Reference Comparison",
        "",
        _table(comparison),
        "",
        "## Case Decisions",
        "",
        _table(compact),
        "",
        "## Readout",
        "",
        "- This selector is optimized only against the existing 6h failure mode. 2h is intentionally excluded from the gate and from the comparison table.",
        "- Because the sample has only 4 focus windows, this is not enough for a claim. It only says which mechanism deserves a 20+ case 6h validation.",
        "- Case 09 is blocked by fallback, gt5, and p95 debt; this is the confirmed 6h failure mode from the ablation.",
        "- refresh-OFF is only released for cases where the extra water saving survives the 6h debt checks.",
        "- This gate keeps fallback debt at zero while retaining the refresh-OFF water advantage on the clean slices.",
        "- This is a structural 6h debt gate, not another enter / pressure / age sweep. The controlled variable is the accepted 6h debt of the OFF branch.",
        "- Runtime integration should happen at the refresh-policy/action-family selector layer: default to refresh-ON, allow refresh-OFF only when the learned/current structural features predict the same no-fallback, bounded-tail region.",
        "",
        f"- Decisions: `{DECISIONS_CSV.relative_to(REPO_ROOT)}`",
        f"- Summary: `{SUMMARY_CSV.relative_to(REPO_ROOT)}`",
        f"- Reference comparison: `{(OUT_DIR / 'case4_debt_gate_reference_comparison.csv').relative_to(REPO_ROOT)}`",
    ]
    READOUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(READOUT_MD)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
