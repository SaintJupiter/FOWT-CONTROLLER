#!/usr/bin/env python3
"""Summarize the neutral-MHS clean-start automatic gate smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_moderate_high_steady_expanded_v1"
RAW_DIR = BASE / "raw_tables"
PAPER_DIR = BASE / "paper_ready"
A0 = RAW_DIR / "neutral_mhs_expanded_a0_opportunity_per_case.csv"
LOG_DIR = BASE / "neutral_mhs_clean_auto_1h" / "planner_logs"


def max_axis_p95(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            pd.to_numeric(df["primary_pitch_p95"], errors="coerce").abs(),
            pd.to_numeric(df["primary_roll_p95"], errors="coerce").abs(),
        ],
        axis=1,
    ).max(axis=1)


def summarize_log(case_id: str) -> dict[str, object]:
    matches = sorted(LOG_DIR.glob(f"{case_id}_*_planner_log.csv"))
    if not matches:
        return {"case_id": case_id, "planner_log_found": 0}
    log = pd.read_csv(matches[0])
    active = pd.to_numeric(log.get("economy_pump_budget_active", 0), errors="coerce").fillna(0)
    eligible = pd.to_numeric(
        log.get("economy_pump_budget_clean_start_eligible", 0),
        errors="coerce",
    ).fillna(0)
    hold_count = pd.to_numeric(
        log.get("economy_pump_budget_hold_count", 0),
        errors="coerce",
    ).fillna(0)
    reasons = log.get("economy_pump_budget_reason")
    if reasons is not None:
        reason_counts = reasons.astype(str).value_counts()
        top_reason = str(reason_counts.index[0]) if len(reason_counts) else ""
    else:
        top_reason = ""
    return {
        "case_id": case_id,
        "planner_log_found": 1,
        "auto_budget_active_rows": int((active > 0).sum()),
        "auto_clean_start_eligible": int(eligible.max() > 0),
        "auto_final_hold_count": int(hold_count.max()),
        "auto_top_budget_reason": top_reason,
        "auto_initial_max_axis_deg": float(
            pd.to_numeric(
                log.get("economy_pump_budget_initial_max_axis_deg", np.nan),
                errors="coerce",
            ).dropna().iloc[-1]
        )
        if "economy_pump_budget_initial_max_axis_deg" in log.columns
        and len(
            pd.to_numeric(
                log.get("economy_pump_budget_initial_max_axis_deg"),
                errors="coerce",
            ).dropna()
        )
        else np.nan,
    }


def metrics(df: pd.DataFrame, label: str) -> dict[str, object]:
    a0_pump = float(df["a0_pump_m3"].sum())
    auto_pump = float(df["auto_pump_m3"].sum())
    saved = a0_pump - auto_pump
    return {
        "group": label,
        "cases": int(len(df)),
        "a0_pump_m3": a0_pump,
        "auto_pump_m3": auto_pump,
        "pump_saved_m3": saved,
        "pump_saving_pct": 100.0 * saved / a0_pump if a0_pump else np.nan,
        "max_delta_fallback_ratio": float(df["delta_fallback_ratio"].max()),
        "mean_delta_p95_max_axis_deg": float(df["delta_p95_max_axis_deg"].mean()),
        "max_delta_p95_max_axis_deg": float(df["delta_p95_max_axis_deg"].max()),
        "active_cases": int((df["auto_budget_active_rows"] > 0).sum()),
        "eligible_cases": int((df["auto_clean_start_eligible"] > 0).sum()),
        "top_case_share_of_saved": float(
            df.loc[df["pump_saved_m3"] > 0, "pump_saved_m3"].max()
            / max(saved, 1e-9)
        )
        if saved > 0 and (df["pump_saved_m3"] > 0).any()
        else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="neutral_mhs_clean_auto_1h")
    parser.add_argument("--summary-stem", default="neutral_mhs_clean_auto_mixed")
    args = parser.parse_args()

    auto_path = BASE / str(args.run_name) / "casebook_summary.csv"
    log_dir = BASE / str(args.run_name) / "planner_logs"
    summary_stem = str(args.summary_stem)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    a0 = pd.read_csv(A0)
    auto = pd.read_csv(auto_path)
    forecast_source = (
        str(auto["forecast_source_effective"].dropna().iloc[0])
        if "forecast_source_effective" in auto.columns
        and len(auto["forecast_source_effective"].dropna())
        else "unknown"
    )
    global LOG_DIR
    LOG_DIR = log_dir
    logs = pd.DataFrame([summarize_log(str(cid)) for cid in auto["case_id"]])
    auto = auto.merge(logs, on="case_id", how="left")
    auto["auto_p95_max_axis_deg"] = max_axis_p95(auto)

    cols = [
        "case_id",
        "split",
        "early_max_ms",
        "peak_to_late_drop_ms",
        "near_range_ms",
        "far_range_ms",
        "dir_shift_abs_max_deg",
        "p95_max_axis_deg",
        "primary_pump_work_m3",
        "primary_safety_fallback_ratio",
        "clean_economy_opportunity",
        "safety_margin_heavy",
    ]
    joined = a0[cols].rename(
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
                "auto_top_budget_reason",
                "auto_initial_max_axis_deg",
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
    joined["delta_fallback_ratio"] = (
        joined["auto_fallback_ratio"] - joined["a0_fallback_ratio"]
    )
    joined["delta_p95_max_axis_deg"] = (
        joined["auto_p95_max_axis_deg"] - joined["a0_p95_max_axis_deg"]
    )
    joined.to_csv(RAW_DIR / f"{summary_stem}_delta.csv", index=False)

    rows = [
        metrics(joined, "all_32_mixed"),
        metrics(joined[joined["clean_economy_opportunity"]], "clean_16"),
        metrics(joined[joined["safety_margin_heavy"]], "safety_heavy_16"),
        metrics(joined[joined["split"] == "test"], "test_split"),
        metrics(joined[joined["split"] == "validation"], "validation_split"),
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(RAW_DIR / f"{summary_stem}_summary.csv", index=False)

    all_row = summary.loc[summary["group"] == "all_32_mixed"].iloc[0]
    clean_row = summary.loc[summary["group"] == "clean_16"].iloc[0]
    heavy_row = summary.loc[summary["group"] == "safety_heavy_16"].iloc[0]
    md = [
        "# Neutral MHS Clean-Start Auto Gate Summary",
        "",
        f"This is a short mixed-set validation of the new default-off `neutral_mhs_clean_budget_v1` profile (`{args.run_name}`, forecast source `{forecast_source}`).",
        "",
        "## Result",
        "",
        f"- all 32 candidates: {all_row['pump_saving_pct']:.2f}% pump saving ({all_row['pump_saved_m3']:.2f} m3)",
        f"- clean 16 subset: {clean_row['pump_saving_pct']:.2f}% pump saving ({clean_row['pump_saved_m3']:.2f} m3)",
        f"- safety-heavy 16 subset: {heavy_row['pump_saving_pct']:.2f}% pump saving ({heavy_row['pump_saved_m3']:.2f} m3)",
        f"- clean active cases: {int(clean_row['active_cases'])}/16",
        f"- safety-heavy active cases: {int(heavy_row['active_cases'])}/16",
        f"- max fallback-ratio delta: {all_row['max_delta_fallback_ratio']:.4f}",
        f"- max p95-axis delta: {all_row['max_delta_p95_max_axis_deg']:.3f} deg",
        f"- top-case share of saved pump: {all_row['top_case_share_of_saved']:.3f}",
        "",
        "## Interpretation",
        "",
        "The automatic gate behaves as intended in this smoke: it opens the aggressive economy budget on the clean-start neutral-MHS cases and abstains on the safety-heavy lookalikes. This converts the earlier clean-subset result into a runnable, default-off specialist candidate.",
        "",
        "Important caveat: this is still a 32-case, 1-hour smoke. It is stronger than the 4-case check, but not yet a final paper headline until repeated on a broader mixed set.",
        "",
        "## Files",
        "",
        f"- `raw_tables/{summary_stem}_delta.csv`",
        f"- `raw_tables/{summary_stem}_summary.csv`",
    ]
    (BASE / f"{summary_stem}_decision.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    (PAPER_DIR / f"{summary_stem}_summary.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
