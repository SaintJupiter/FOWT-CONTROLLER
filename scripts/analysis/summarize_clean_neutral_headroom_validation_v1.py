#!/usr/bin/env python3
"""Summarize clean-neutral/headroom validation runs.

The casebook summary has pump and fallback ratios, but it does not directly
report time above the posture floor.  This script joins the summary with the
per-case time-series files so broad-economy runs can be judged by net pump and
posture cost before any larger validation is launched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.2f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _read_timeseries_metrics(path: Path, prefix: str) -> dict[str, float]:
    df = pd.read_csv(path, usecols=["t_s", "pitch_deg", "roll_deg", "pump_total_rate_m3_min"])
    t = df["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    axis = np.maximum(df["pitch_deg"].abs(), df["roll_deg"].abs())
    return {
        f"{prefix}_initial_axis_deg": float(axis.iloc[0]),
        f"{prefix}_time_gt5_s": float(axis.ge(5.0).sum() * dt),
        f"{prefix}_time_gt4_s": float(axis.ge(4.0).sum() * dt),
        f"{prefix}_p95_axis_deg": float(np.percentile(axis, 95)),
        f"{prefix}_max_axis_deg": float(axis.max()),
        f"{prefix}_pump_integral_m3": float(df["pump_total_rate_m3_min"].sum() * dt / 60.0),
    }


def summarize_run(run_dir: Path, label: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "casebook_summary.csv"
    ts_dir = run_dir / "timeseries"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not ts_dir.exists():
        raise FileNotFoundError(ts_dir)

    summary = pd.read_csv(summary_path)
    summary["case_prefix"] = summary["case_id"].astype(str).str[:2]
    rows: list[dict[str, float | str]] = []
    for closed_path in sorted(ts_dir.glob("*_closed_only_timeseries.csv")):
        stem = closed_path.name.replace("_closed_only_timeseries.csv", "")
        primary_path = ts_dir / f"{stem}_prediction_primary_econ_timeseries.csv"
        if not primary_path.exists():
            continue
        row: dict[str, float | str] = {
            "case_prefix": stem[:2],
            "case_file": stem,
        }
        row.update(_read_timeseries_metrics(closed_path, "closed"))
        row.update(_read_timeseries_metrics(primary_path, "primary"))
        rows.append(row)

    metrics = pd.DataFrame(rows)
    keep = [
        "case_prefix",
        "case_id",
        "timestamp",
        "label",
        "closed_pump_work_m3",
        "primary_pump_work_m3",
        "d_pump_work_pct",
        "primary_safety_fallback_ratio",
    ]
    joined = metrics.merge(summary[[c for c in keep if c in summary.columns]], on="case_prefix", how="left")
    joined["saved_m3"] = joined["closed_pump_integral_m3"] - joined["primary_pump_integral_m3"]
    joined["saving_pct"] = 100.0 * joined["saved_m3"] / joined["closed_pump_integral_m3"].replace(0.0, np.nan)
    joined["d_time_gt5_s"] = joined["primary_time_gt5_s"] - joined["closed_time_gt5_s"]
    joined["d_time_gt4_s"] = joined["primary_time_gt4_s"] - joined["closed_time_gt4_s"]
    joined["d_p95_axis_deg"] = joined["primary_p95_axis_deg"] - joined["closed_p95_axis_deg"]
    joined["d_max_axis_deg"] = joined["primary_max_axis_deg"] - joined["closed_max_axis_deg"]
    joined["primary_fallback_s"] = joined.get("primary_safety_fallback_ratio", 0.0).fillna(0.0) * 3600.0
    joined["fallback_active"] = joined["primary_fallback_s"] > 0.0
    joined["timegt5_worse_gt300s"] = joined["d_time_gt5_s"] > 300.0
    joined["run_label"] = label

    closed_pump = float(joined["closed_pump_integral_m3"].sum())
    primary_pump = float(joined["primary_pump_integral_m3"].sum())
    saved = closed_pump - primary_pump
    positive_saved = joined["saved_m3"].clip(lower=0.0)
    summary_row = pd.DataFrame(
        [
            {
                "run_label": label,
                "cases": int(len(joined)),
                "closed_pump_m3": closed_pump,
                "primary_pump_m3": primary_pump,
                "saved_m3": saved,
                "saving_pct": 100.0 * saved / closed_pump if closed_pump else 0.0,
                "delta_time_gt5_s": float(joined["d_time_gt5_s"].sum()),
                "delta_time_gt4_s": float(joined["d_time_gt4_s"].sum()),
                "primary_fallback_s": float(joined["primary_fallback_s"].sum()),
                "fallback_cases": int(joined["fallback_active"].sum()),
                "mean_delta_p95_axis_deg": float(joined["d_p95_axis_deg"].mean()),
                "max_delta_p95_axis_deg": float(joined["d_p95_axis_deg"].max()),
                "top_case_share": float(positive_saved.max() / max(positive_saved.sum(), 1e-9)),
                "bad_cases_timegt5_gt300": int(joined["timegt5_worse_gt300s"].sum()),
            }
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_dir / f"{label}_case_runtime_delta.csv", index=False)
    summary_row.to_csv(out_dir / f"{label}_summary.csv", index=False)
    return joined, summary_row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    _, summary = summarize_run(args.run_dir, args.label, args.out_dir)
    print(_md_table(summary))


if __name__ == "__main__":
    main()
