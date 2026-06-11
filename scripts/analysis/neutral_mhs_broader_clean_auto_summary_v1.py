#!/usr/bin/env python3
"""Summarize broader neutral-MHS clean-start auto validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_moderate_high_steady_broader_v1"
RAW_DIR = BASE / "raw_tables"
PAPER_DIR = BASE / "paper_ready"
A0 = RAW_DIR / "neutral_mhs_broader_a0_opportunity_per_case.csv"
RUN = BASE / "neutral_mhs_clean_auto_learned_1h"
LOG_DIR = RUN / "planner_logs"


def max_axis_p95(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            pd.to_numeric(df["primary_pitch_p95"], errors="coerce").abs(),
            pd.to_numeric(df["primary_roll_p95"], errors="coerce").abs(),
        ],
        axis=1,
    ).max(axis=1)


def read_log(case_id: str) -> dict[str, object]:
    matches = sorted(LOG_DIR.glob(f"{case_id}_*_planner_log.csv"))
    if not matches:
        return {"case_id": case_id, "planner_log_found": 0}
    log = pd.read_csv(matches[0])
    active = pd.to_numeric(log.get("economy_pump_budget_active", 0), errors="coerce").fillna(0)
    eligible = pd.to_numeric(
        log.get("economy_pump_budget_clean_start_eligible", 0),
        errors="coerce",
    ).fillna(0)
    hold_count = pd.to_numeric(log.get("economy_pump_budget_hold_count", 0), errors="coerce").fillna(0)
    initial_axis = pd.to_numeric(
        log.get("economy_pump_budget_initial_max_axis_deg", np.nan),
        errors="coerce",
    ).dropna()
    reason = ""
    if "economy_pump_budget_reason" in log.columns:
        counts = log["economy_pump_budget_reason"].astype(str).value_counts()
        if len(counts):
            reason = str(counts.index[0])
    return {
        "case_id": case_id,
        "planner_log_found": 1,
        "auto_budget_active_rows": int((active > 0).sum()),
        "auto_clean_start_eligible": int(eligible.max() > 0),
        "auto_final_hold_count": int(hold_count.max()),
        "auto_initial_max_axis_deg": float(initial_axis.iloc[-1]) if len(initial_axis) else np.nan,
        "auto_top_budget_reason": reason,
    }


def metrics(df: pd.DataFrame, group: str) -> dict[str, object]:
    a0_pump = float(df["a0_pump_m3"].sum())
    auto_pump = float(df["auto_pump_m3"].sum())
    saved = a0_pump - auto_pump
    positive = df.loc[df["pump_saved_m3"] > 0, "pump_saved_m3"]
    return {
        "group": group,
        "cases": int(len(df)),
        "a0_pump_m3": a0_pump,
        "auto_pump_m3": auto_pump,
        "pump_saved_m3": saved,
        "pump_saving_pct": 100.0 * saved / a0_pump if a0_pump else np.nan,
        "active_cases": int((df["auto_budget_active_rows"] > 0).sum()),
        "eligible_cases": int((df["auto_clean_start_eligible"] > 0).sum()),
        "max_delta_fallback_ratio": float(df["delta_fallback_ratio"].max()),
        "mean_delta_fallback_ratio": float(df["delta_fallback_ratio"].mean()),
        "max_delta_p95_max_axis_deg": float(df["delta_p95_max_axis_deg"].max()),
        "mean_delta_p95_max_axis_deg": float(df["delta_p95_max_axis_deg"].mean()),
        "top_case_share_of_saved": float(positive.max() / saved) if saved > 0 and len(positive) else 0.0,
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    a0 = pd.read_csv(A0)
    auto = pd.read_csv(RUN / "casebook_summary.csv")
    forecast_source = (
        str(auto["forecast_source_effective"].dropna().iloc[0])
        if "forecast_source_effective" in auto.columns and len(auto["forecast_source_effective"].dropna())
        else "unknown"
    )
    logs = pd.DataFrame([read_log(str(cid)) for cid in auto["case_id"]])
    auto = auto.merge(logs, on="case_id", how="left")
    auto["auto_p95_max_axis_deg"] = max_axis_p95(auto)

    keep = [
        "case_id",
        "split",
        "early_max_ms",
        "peak_to_late_drop_ms",
        "near_range_ms",
        "far_range_ms",
        "dir_shift_abs_max_deg",
        "selection_score",
        "p95_max_axis_deg",
        "primary_pump_work_m3",
        "primary_safety_fallback_ratio",
        "clean_economy_opportunity",
        "safety_margin_heavy",
        "initial_max_axis_deg",
    ]
    joined = a0[keep].rename(
        columns={
            "primary_pump_work_m3": "a0_pump_m3",
            "primary_safety_fallback_ratio": "a0_fallback_ratio",
            "p95_max_axis_deg": "a0_p95_max_axis_deg",
        }
    ).merge(
        auto[
            [
                "case_id",
                "primary_pump_work_m3",
                "primary_safety_fallback_ratio",
                "auto_p95_max_axis_deg",
                "auto_budget_active_rows",
                "auto_clean_start_eligible",
                "auto_final_hold_count",
                "auto_initial_max_axis_deg",
                "auto_top_budget_reason",
            ]
        ].rename(
            columns={
                "primary_pump_work_m3": "auto_pump_m3",
                "primary_safety_fallback_ratio": "auto_fallback_ratio",
            }
        ),
        on="case_id",
        how="inner",
    )
    joined["pump_saved_m3"] = joined["a0_pump_m3"] - joined["auto_pump_m3"]
    joined["pump_saving_pct"] = 100.0 * joined["pump_saved_m3"] / joined["a0_pump_m3"]
    joined["delta_fallback_ratio"] = joined["auto_fallback_ratio"] - joined["a0_fallback_ratio"]
    joined["delta_p95_max_axis_deg"] = joined["auto_p95_max_axis_deg"] - joined["a0_p95_max_axis_deg"]
    joined.to_csv(RAW_DIR / "neutral_mhs_broader_clean_auto_learned_delta.csv", index=False)

    summary = pd.DataFrame(
        [
            metrics(joined, "all_64_broader"),
            metrics(joined[joined["clean_economy_opportunity"]], "clean_32"),
            metrics(joined[joined["safety_margin_heavy"]], "safety_heavy_32"),
            metrics(joined[joined["split"] == "test"], "test_split"),
            metrics(joined[joined["split"] == "validation"], "validation_split"),
        ]
    )
    summary.to_csv(RAW_DIR / "neutral_mhs_broader_clean_auto_learned_summary.csv", index=False)

    all_row = summary.loc[summary["group"] == "all_64_broader"].iloc[0]
    clean_row = summary.loc[summary["group"] == "clean_32"].iloc[0]
    heavy_row = summary.loc[summary["group"] == "safety_heavy_32"].iloc[0]
    md = [
        "# Neutral MHS Clean-Start Auto Broader Validation",
        "",
        f"Run: `{RUN.name}`. Forecast source: `{forecast_source}`.",
        "",
        "## Result",
        "",
        f"- all 64 broader candidates: {all_row['pump_saving_pct']:.2f}% pump saving ({all_row['pump_saved_m3']:.2f} m3)",
        f"- clean-start opportunity subset (32 cases): {clean_row['pump_saving_pct']:.2f}% pump saving ({clean_row['pump_saved_m3']:.2f} m3)",
        f"- safety-heavy lookalikes (32 cases): {heavy_row['pump_saving_pct']:.2f}% pump saving ({heavy_row['pump_saved_m3']:.2f} m3)",
        f"- active cases: clean {int(clean_row['active_cases'])}/32, safety-heavy {int(heavy_row['active_cases'])}/32",
        f"- max fallback-ratio delta: {all_row['max_delta_fallback_ratio']:.4f}",
        f"- max p95-axis delta: {all_row['max_delta_p95_max_axis_deg']:.3f} deg",
        f"- top-case share of saved pump: {all_row['top_case_share_of_saved']:.3f}",
        "",
        "## Interpretation",
        "",
        "This broader fixed-rule validation supports a second automatic specialist: neutral moderate-high steady wind can save pump when the current platform state is nearly level, while the same wind shape must abstain when current posture is already safety-margin-heavy.",
        "The action is therefore not opened by wind-shape alone. It requires both learned steady-high forecast shape and a clean current posture gate.",
        "",
        "## Files",
        "",
        "- `raw_tables/neutral_mhs_broader_clean_auto_learned_delta.csv`",
        "- `raw_tables/neutral_mhs_broader_clean_auto_learned_summary.csv`",
    ]
    (BASE / "neutral_mhs_broader_clean_auto_learned_decision.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    (PAPER_DIR / "neutral_mhs_broader_clean_auto_learned_summary.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
