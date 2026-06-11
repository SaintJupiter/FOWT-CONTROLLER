#!/usr/bin/env python3
"""Read-only shadow audit for an economy pump-budget layer.

The audit estimates where a pump budget would bite before implementing any
closed-loop budget controller. It classifies 1 Hz pump work into hard-safety
proxy rows and economy/comfort proxy rows using existing preview telemetry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
RUNS = {
    "A0": BASE / "runs/A0_learned_v16",
    "A1": BASE / "runs/A1_learned_near_envelope_medium_axis_guard",
}
OUT = REPO / "outputs/wind_prediction/economy_pump_budget_shadow_audit_v1"
RAW = OUT / "raw_tables"
PAPER = OUT / "paper_ready"
DEBUG = OUT / "debug"


def ensure_dirs() -> None:
    for d in (OUT, RAW, PAPER, DEBUG):
        d.mkdir(parents=True, exist_ok=True)


def case_id_from_path(path: Path) -> str:
    suffix = "_prediction_primary_econ_timeseries.csv"
    return path.name[: -len(suffix)]


def read_ts(path: Path) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=1).columns
    wanted = [
        "t_s",
        "pitch_deg",
        "roll_deg",
        "pump_total_rate_m3_min",
        "preview_primary_safety_active",
        "preview_primary_safety_hard_active",
        "preview_primary_safety_fallback",
        "preview_primary_bucket_guard_active",
        "preview_recovery_mode_active",
        "preview_primary_event_reset",
        "preview_primary_target_refreshed",
        "preview_primary_target_reused",
        "preview_hold_comfort_release_active",
        "preview_hold_comfort_forecast_veto_active",
        "preview_primary_action",
        "target_source",
    ]
    use = [c for c in wanted if c in cols]
    df = pd.read_csv(path, usecols=use)
    for c in wanted:
        if c not in df.columns:
            if c in ("preview_primary_action", "target_source"):
                df[c] = ""
            else:
                df[c] = 0
    return df


def classify_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pump = pd.to_numeric(out["pump_total_rate_m3_min"], errors="coerce").fillna(0.0).abs()
    out["pump_m3_row"] = pump / 60.0
    out["max_axis"] = np.maximum(
        pd.to_numeric(out["pitch_deg"], errors="coerce").fillna(0.0).abs(),
        pd.to_numeric(out["roll_deg"], errors="coerce").fillna(0.0).abs(),
    )
    hard = (
        pd.to_numeric(out["preview_primary_safety_active"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out["preview_primary_safety_hard_active"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out["preview_primary_safety_fallback"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out["preview_primary_bucket_guard_active"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(out["preview_recovery_mode_active"], errors="coerce").fillna(0).gt(0)
        | out["max_axis"].ge(5.0)
    )
    out["hard_safety_proxy"] = hard.astype(int)
    out["economy_proxy"] = ((~hard) & pump.gt(1e-9)).astype(int)
    out["comfort_proxy"] = (
        pd.to_numeric(out["preview_hold_comfort_release_active"], errors="coerce").fillna(0).gt(0)
        & (~hard)
        & pump.gt(1e-9)
    ).astype(int)
    out["refresh_proxy"] = (
        pd.to_numeric(out["preview_primary_target_refreshed"], errors="coerce").fillna(0).gt(0)
        & (~hard)
    ).astype(int)
    out["event_reset_proxy"] = (
        pd.to_numeric(out["preview_primary_event_reset"], errors="coerce").fillna(0).gt(0)
        & (~hard)
    ).astype(int)
    return out


def summarize_run(run_name: str, run_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    budget_rows = []
    budgets = [1.00, 0.95, 0.90, 0.85, 0.80, 0.75]
    for path in sorted((run_path / "timeseries").glob("*_prediction_primary_econ_timeseries.csv")):
        case = case_id_from_path(path)
        df = classify_rows(read_ts(path))
        total = float(df["pump_m3_row"].sum())
        hard = float(df.loc[df["hard_safety_proxy"].astype(bool), "pump_m3_row"].sum())
        econ = float(df.loc[df["economy_proxy"].astype(bool), "pump_m3_row"].sum())
        comfort = float(df.loc[df["comfort_proxy"].astype(bool), "pump_m3_row"].sum())
        refresh = float((df["refresh_proxy"] * df["pump_m3_row"]).sum())
        event_reset = float((df["event_reset_proxy"] * df["pump_m3_row"]).sum())
        time3 = int((df["max_axis"] > 3.0).sum())
        time4 = int((df["max_axis"] > 4.0).sum())
        time5 = int((df["max_axis"] > 5.0).sum())
        p95 = float(np.percentile(df["max_axis"], 95))
        rows.append(
            {
                "run": run_name,
                "case": case,
                "total_pump_m3": total,
                "hard_safety_proxy_pump_m3": hard,
                "economy_proxy_pump_m3": econ,
                "comfort_proxy_pump_m3": comfort,
                "refresh_proxy_pump_m3": refresh,
                "event_reset_proxy_pump_m3": event_reset,
                "economy_share": econ / max(total, 1e-9),
                "time_gt3_s": time3,
                "time_gt4_s": time4,
                "time_gt5_s": time5,
                "p95_max_axis_deg": p95,
            }
        )
        for frac in budgets:
            allowed_total = frac * total
            # Shadow estimate: hard-safety pump is exempt; only economy proxy can
            # be cut. If hard exceeds the requested total, no economy budget remains.
            allowed_econ = max(0.0, allowed_total - hard)
            econ_cut = max(0.0, econ - allowed_econ)
            achieved_total = total - econ_cut
            budget_rows.append(
                {
                    "run": run_name,
                    "case": case,
                    "budget_fraction_of_case_total": frac,
                    "requested_total_budget_m3": allowed_total,
                    "hard_safety_proxy_pump_m3": hard,
                    "economy_proxy_pump_m3": econ,
                    "economy_cut_shadow_m3": econ_cut,
                    "achieved_shadow_total_m3": achieved_total,
                    "shadow_saving_pct": econ_cut / max(total, 1e-9) * 100.0,
                    "budget_infeasible_due_to_hard_pump": int(hard > allowed_total),
                    "time_gt3_s": time3,
                    "time_gt4_s": time4,
                    "time_gt5_s": time5,
                    "p95_max_axis_deg": p95,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(budget_rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    d = df if max_rows is None else df.head(max_rows)
    lines = ["| " + " | ".join(d.columns) + " |", "| " + " | ".join(["---"] * len(d.columns)) + " |"]
    for _, row in d.iterrows():
        vals = []
        for c in d.columns:
            v = row[c]
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    case_tables = []
    budget_tables = []
    for name, path in RUNS.items():
        cases, budgets = summarize_run(name, path)
        case_tables.append(cases)
        budget_tables.append(budgets)
    case_table = pd.concat(case_tables, ignore_index=True)
    budget_table = pd.concat(budget_tables, ignore_index=True)

    run_summary = (
        case_table.groupby("run", as_index=False)
        .agg(
            cases=("case", "nunique"),
            total_pump_m3=("total_pump_m3", "sum"),
            hard_safety_proxy_pump_m3=("hard_safety_proxy_pump_m3", "sum"),
            economy_proxy_pump_m3=("economy_proxy_pump_m3", "sum"),
            comfort_proxy_pump_m3=("comfort_proxy_pump_m3", "sum"),
            refresh_proxy_pump_m3=("refresh_proxy_pump_m3", "sum"),
            event_reset_proxy_pump_m3=("event_reset_proxy_pump_m3", "sum"),
            time_gt3_s=("time_gt3_s", "sum"),
            time_gt4_s=("time_gt4_s", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            p95_max_axis_deg_mean=("p95_max_axis_deg", "mean"),
        )
    )
    run_summary["economy_share"] = run_summary["economy_proxy_pump_m3"] / run_summary["total_pump_m3"].clip(lower=1e-9)
    run_summary["hard_share"] = run_summary["hard_safety_proxy_pump_m3"] / run_summary["total_pump_m3"].clip(lower=1e-9)

    budget_summary = (
        budget_table.groupby(["run", "budget_fraction_of_case_total"], as_index=False)
        .agg(
            cases=("case", "nunique"),
            requested_total_budget_m3=("requested_total_budget_m3", "sum"),
            hard_safety_proxy_pump_m3=("hard_safety_proxy_pump_m3", "sum"),
            economy_proxy_pump_m3=("economy_proxy_pump_m3", "sum"),
            economy_cut_shadow_m3=("economy_cut_shadow_m3", "sum"),
            achieved_shadow_total_m3=("achieved_shadow_total_m3", "sum"),
            infeasible_cases=("budget_infeasible_due_to_hard_pump", "sum"),
            time_gt3_s=("time_gt3_s", "sum"),
            time_gt4_s=("time_gt4_s", "sum"),
            time_gt5_s=("time_gt5_s", "sum"),
            p95_max_axis_deg_mean=("p95_max_axis_deg", "mean"),
        )
    )
    totals = run_summary.set_index("run")["total_pump_m3"].to_dict()
    budget_summary["shadow_saving_pct_vs_run"] = budget_summary.apply(
        lambda r: r["economy_cut_shadow_m3"] / max(float(totals.get(r["run"], 0.0)), 1e-9) * 100.0,
        axis=1,
    )

    case_table.to_csv(RAW / "economy_budget_case_pump_classification.csv", index=False)
    budget_table.to_csv(RAW / "economy_budget_shadow_case_table.csv", index=False)
    run_summary.to_csv(RAW / "economy_budget_run_summary.csv", index=False)
    budget_summary.to_csv(RAW / "economy_budget_shadow_frontier.csv", index=False)

    text = f"""# Economy Pump Budget Shadow Audit v1

## Purpose

This is a read-only pre-check before implementing a closed-loop economy pump
budget. It estimates how much pump belongs to hard-safety proxy rows versus
economy/comfort proxy rows, then asks whether 90/85/80/75% total-pump budgets
would mostly cut economy pump or collide with hard-safety pump.

## Run Summary

{md_table(run_summary)}

## Shadow Budget Frontier

{md_table(budget_summary)}

## Interpretation

- This is not a closed-loop result. Cutting economy pump can change posture and
  may later rebound into floor/recovery pump.
- If the shadow frontier cannot reach a target without colliding with hard-safety
  proxy pump, a closed-loop budget layer is unlikely to reach that target safely.
- If the shadow frontier has enough economy proxy pump, the next step is a
  default-off closed-loop budget layer with blind, forecast-smart, and scrambled
  controls.
"""
    (PAPER / "economy_pump_budget_shadow_audit.md").write_text(text)
    (OUT / "decision.md").write_text(text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
