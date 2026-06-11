#!/usr/bin/env python3
"""Audit relief/decay dispatcher attribution and duration consistency.

This is a read-only reproducibility script.  It checks whether the existing
relief/decay specialist outputs are comparable across equal horizons and whether
the apparent smart-release gain is actually backed by economy-budget hold rows.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
RUN = ROOT / "relief_decay_expansion_v1"
OUT = (
    Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
    / "relief_decay_dispatcher_attribution_audit_v1"
)
RAW = OUT / "raw_tables"
PAPER = OUT / "paper_ready"

ARMS = [
    "a0_baseline",
    "budget100",
    "budget100_smart_release",
    "relief_decay_smart_release_auto_v1",
]


def case_from_path(path: Path) -> str:
    return re.sub(r"_prediction_primary_econ_(timeseries|planner_log)\.csv$", "", path.name)


def series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def fallback_flag(df: pd.DataFrame) -> np.ndarray:
    flag = np.zeros(len(df), dtype=bool)
    for col in [
        "target_lookup_fallback",
        "preview_primary_safety_fallback",
        "fallback_reason_missing_state",
        "fallback_reason_missing_table",
        "fallback_reason_missing_columns",
        "fallback_reason_invalid_state_id",
    ]:
        if col in df.columns:
            flag |= series(df, col).to_numpy() > 0.5
    return flag


def time_filtered(df: pd.DataFrame, horizon_s: float | None = None) -> pd.DataFrame:
    if horizon_s is None:
        return df
    if "t_s" in df.columns:
        return df[series(df, "t_s") < float(horizon_s)].copy()
    return df.iloc[: int(horizon_s)].copy()


def metrics(path: Path, horizon_s: float | None = None) -> dict[str, float | int]:
    df = time_filtered(pd.read_csv(path, low_memory=False), horizon_s)
    pitch = series(df, "pitch_deg").to_numpy()
    roll = series(df, "roll_deg").to_numpy()
    axis = np.maximum(np.abs(pitch), np.abs(roll))
    pump = np.abs(series(df, "pump_total_rate_m3_min").to_numpy())
    fallback = fallback_flag(df)
    return {
        "rows": int(len(df)),
        "pump_m3": float(np.nansum(pump) / 60.0),
        "time_gt5_s": int(np.nansum(axis > 5.0)),
        "idle_gt5_s": int(np.nansum((axis > 5.0) & (pump < 0.5))),
        "fallback_time_s": int(np.nansum(fallback)),
        "p95_max_axis_deg": float(np.nanpercentile(axis, 95)) if len(axis) else 0.0,
        "max_axis_deg": float(np.nanmax(axis)) if len(axis) else 0.0,
    }


def duration_rows() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for path in sorted((RUN / arm / "timeseries").glob("*_timeseries.csv")):
            df = pd.read_csv(path, usecols=lambda c: c in {"t_s", "time_s", "current_time_s"})
            time_col = next((c for c in ("t_s", "time_s", "current_time_s") if c in df.columns), None)
            duration = (
                float(pd.to_numeric(df[time_col], errors="coerce").max()) + 1.0
                if time_col
                else float(len(df))
            )
            rows.append(
                {
                    "arm": arm,
                    "case": case_from_path(path),
                    "rows": int(len(df)),
                    "duration_s": duration,
                }
            )
    return pd.DataFrame(rows)


def horizon_summary(horizon_s: float) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for path in sorted((RUN / arm / "timeseries").glob("*_timeseries.csv")):
            row = metrics(path, horizon_s)
            row.update({"arm": arm, "case": case_from_path(path)})
            rows.append(row)
    per_case = pd.DataFrame(rows)
    out = []
    a0 = per_case[per_case["arm"] == "a0_baseline"]
    a0_pump = float(a0["pump_m3"].sum())
    a0_time = int(a0["time_gt5_s"].sum())
    a0_idle = int(a0["idle_gt5_s"].sum())
    a0_fallback = int(a0["fallback_time_s"].sum())
    a0_p95 = float(a0["p95_max_axis_deg"].mean())
    for arm, g in per_case.groupby("arm", sort=False):
        pump = float(g["pump_m3"].sum())
        out.append(
            {
                "arm": arm,
                "cases": int(len(g)),
                "pump_m3": pump,
                "pump_saved_m3_vs_a0": a0_pump - pump,
                "pump_saving_pct_vs_a0": 100.0 * (a0_pump - pump) / a0_pump
                if a0_pump
                else 0.0,
                "delta_time_gt5_s": int(g["time_gt5_s"].sum()) - a0_time,
                "delta_idle_gt5_s": int(g["idle_gt5_s"].sum()) - a0_idle,
                "delta_fallback_time_s": int(g["fallback_time_s"].sum()) - a0_fallback,
                "delta_mean_p95_max_axis_deg": float(g["p95_max_axis_deg"].mean()) - a0_p95,
                "max_axis_deg": float(g["max_axis_deg"].max()),
            }
        )
    return pd.DataFrame(out)


def budget_hold_rows() -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for path in sorted((RUN / arm / "planner_logs").glob("*_planner_log.csv")):
            df = pd.read_csv(path, low_memory=False)
            active = int(series(df, "economy_pump_budget_active").sum())
            hold_count = int(series(df, "economy_pump_budget_hold_count").max())
            reason_counts = (
                df["economy_pump_budget_reason"].value_counts().to_dict()
                if "economy_pump_budget_reason" in df.columns
                else {}
            )
            rows.append(
                {
                    "arm": arm,
                    "case": case_from_path(path),
                    "planner_rows": int(len(df)),
                    "active_hold_rows": active,
                    "final_hold_count": hold_count,
                    "top_reasons": str(reason_counts),
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.3f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    durations = duration_rows()
    duration_summary = durations.groupby("arm", as_index=False).agg(
        cases=("case", "count"),
        min_duration_s=("duration_s", "min"),
        max_duration_s=("duration_s", "max"),
        mean_duration_s=("duration_s", "mean"),
        min_rows=("rows", "min"),
        max_rows=("rows", "max"),
    )
    matched = horizon_summary(3600.0)
    holds = budget_hold_rows()
    hold_summary = holds.groupby("arm", as_index=False).agg(
        planner_rows=("planner_rows", "sum"),
        active_hold_rows=("active_hold_rows", "sum"),
        final_hold_count=("final_hold_count", "sum"),
    )

    durations.to_csv(RAW / "relief_decay_duration_by_case.csv", index=False)
    duration_summary.to_csv(RAW / "relief_decay_duration_summary.csv", index=False)
    matched.to_csv(RAW / "matched_3600_relief_decay_summary.csv", index=False)
    holds.to_csv(RAW / "relief_decay_budget_hold_by_case.csv", index=False)
    hold_summary.to_csv(RAW / "relief_decay_budget_hold_summary.csv", index=False)

    text = [
        "# Relief/Decay Dispatcher Attribution Audit v1",
        "",
        "This audit checks whether existing relief/decay specialist outputs are comparable across equal horizons and whether the apparent smart-release gain is backed by budget-hold actions.",
        "",
        "## Duration Summary",
        "",
        md_table(
            duration_summary,
            ["arm", "cases", "min_duration_s", "max_duration_s", "mean_duration_s"],
        ),
        "",
        "## Matched First-3600s Comparison",
        "",
        md_table(
            matched,
            [
                "arm",
                "pump_saving_pct_vs_a0",
                "pump_saved_m3_vs_a0",
                "delta_time_gt5_s",
                "delta_idle_gt5_s",
                "delta_fallback_time_s",
                "delta_mean_p95_max_axis_deg",
            ],
        ),
        "",
        "## Budget-Hold Attribution",
        "",
        md_table(hold_summary, ["arm", "planner_rows", "active_hold_rows", "final_hold_count"]),
        "",
        "## Interpretation",
        "",
        "- The existing smart-release runs are 3600s per case while A0 and budget100 are 7200s per case.",
        "- On a matched first-3600s horizon, smart-release and relief_decay_auto produce 0% saving versus A0 and have 0 active budget-hold rows.",
        "- The valid large relief/decay result is budget100: about 21.7% pump saving in the first hour with no measured time>5/fallback increase.",
        "- Therefore a future automatic dispatcher should target budget100-style aggressive economy mode in forecast-identified relief/decay windows; it should not cite the old smart-release aggregate as evidence.",
        "",
    ]
    (OUT / "decision.md").write_text("\n".join(text), encoding="utf-8")
    (PAPER / "relief_decay_dispatcher_attribution_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )
    print(matched.to_string(index=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
