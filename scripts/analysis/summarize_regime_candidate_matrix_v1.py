#!/usr/bin/env python3
"""Summarize an isolated regime candidate matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_arm(base: Path, arm: str) -> pd.DataFrame:
    path = base / arm / "casebook_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "case_id": df["case_id"].astype(str),
            "timestamp": df["timestamp"].astype(str),
            "label": df["label"].astype(str),
            "arm": arm,
            "pump_m3": pd.to_numeric(df["primary_pump_work_m3"], errors="coerce").fillna(0.0),
            "pitch_p95": pd.to_numeric(df["primary_pitch_p95"], errors="coerce"),
            "roll_p95": pd.to_numeric(df["primary_roll_p95"], errors="coerce"),
            "fallback_ratio": pd.to_numeric(df["primary_safety_fallback_ratio"], errors="coerce").fillna(0.0),
        }
    )
    out["max_p95"] = out[["pitch_p95", "roll_p95"]].max(axis=1)
    out["fallback_time_s"] = out["fallback_ratio"] * duration_from_casebook(path)
    return out


def duration_from_casebook(path: Path) -> float:
    report = path.with_name("casebook_report.md")
    if not report.exists():
        return 0.0
    for line in report.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "duration per case:" not in line:
            continue
        # Example: - duration per case: `120 min`
        if "`" in line:
            token = line.split("`", 2)[1]
            parts = token.split()
            if len(parts) >= 2 and parts[1].startswith("min"):
                return float(parts[0]) * 60.0
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--baseline", default="a0_baseline")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    raw = out_dir / "raw_tables"
    paper = out_dir / "paper_ready"
    raw.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    frames = [load_arm(base, arm) for arm in args.arms]
    all_df = pd.concat(frames, ignore_index=True)
    baseline = all_df[all_df["arm"] == args.baseline].set_index(["case_id", "timestamp"])
    rows = []
    case_rows = []
    for arm, group in all_df.groupby("arm"):
        g = group.set_index(["case_id", "timestamp"])
        joined = g.join(
            baseline[["pump_m3", "max_p95", "fallback_time_s"]],
            rsuffix="_baseline",
            how="left",
        )
        joined["pump_saved_m3_vs_baseline"] = joined["pump_m3_baseline"] - joined["pump_m3"]
        joined["delta_p95_vs_baseline"] = joined["max_p95"] - joined["max_p95_baseline"]
        joined["delta_fallback_time_s_vs_baseline"] = (
            joined["fallback_time_s"] - joined["fallback_time_s_baseline"]
        )
        a0_pump = joined["pump_m3_baseline"].sum()
        saved = joined["pump_saved_m3_vs_baseline"].sum()
        rows.append(
            {
                "arm": arm,
                "cases": len(joined),
                "pump_m3": joined["pump_m3"].sum(),
                "pump_saved_m3_vs_baseline": saved,
                "pump_saving_pct_vs_baseline": 100.0 * saved / a0_pump if a0_pump else 0.0,
                "delta_fallback_time_s_vs_baseline": joined[
                    "delta_fallback_time_s_vs_baseline"
                ].sum(),
                "mean_delta_p95_vs_baseline": joined["delta_p95_vs_baseline"].mean(),
                "top_case_saved_m3": joined["pump_saved_m3_vs_baseline"].max(),
            }
        )
        tmp = joined.reset_index()
        tmp["arm"] = arm
        case_rows.append(tmp)

    summary = pd.DataFrame(rows).sort_values("pump_saved_m3_vs_baseline", ascending=False)
    cases = pd.concat(case_rows, ignore_index=True)
    summary.to_csv(raw / "candidate_matrix_summary.csv", index=False)
    cases.to_csv(raw / "candidate_matrix_case_deltas.csv", index=False)

    lines = ["# Regime Candidate Matrix Summary\n"]
    lines.append(summary.to_string(index=False))
    lines.append("\n\n## Interpretation Placeholder\n")
    lines.append("Use this table to decide whether this regime should become a mature specialist, remain a candidate, or be rejected.\n")
    (paper / "candidate_matrix_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
