#!/usr/bin/env python3
"""Audit sustained high-posture buckets where prediction-primary keeps holding."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--casebook-dir",
        default="outputs/wind_prediction/current_program_5case_figures",
        help="Casebook output directory containing timeseries/ and planner_logs/.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/no_unexplained_hold_audit",
        help="Directory for audit CSV/markdown outputs.",
    )
    parser.add_argument("--threshold-deg", type=float, default=3.0)
    parser.add_argument("--bucket-s", type=float, default=600.0)
    parser.add_argument("--trigger-buckets", type=int, default=2)
    parser.add_argument("--target-err-kg", type=float, default=1500.0)
    parser.add_argument("--pump-idle-m3-min", type=float, default=0.5)
    return parser.parse_args()


def _num(series: pd.Series | int | float, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series([float(series)])


def _run_lengths(flags: list[bool]) -> list[int]:
    run = 0
    out: list[int] = []
    for flag in flags:
        if flag:
            run += 1
        else:
            run = 0
        out.append(run)
    return out


def _bucket_timeseries(path: Path, bucket_s: float, threshold_deg: float) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["bucket"] = np.floor(_num(df["t_s"]) / float(bucket_s)).astype(int)
    rows: list[dict] = []
    for bucket, group in df.groupby("bucket", sort=True):
        pitch = _num(group["pitch_deg"])
        roll = _num(group["roll_deg"])
        action = (
            str(group["preview_primary_action"].mode().iloc[0])
            if "preview_primary_action" in group
            else ""
        )
        target_err = _num(group.get("preview_primary_stall_target_err_mean_kg", 0.0))
        pump_rate = _num(group.get("pump_total_rate_m3_min", 0.0))
        fallback = _num(group.get("preview_primary_safety_fallback", 0.0))
        stall_refresh = _num(group.get("preview_primary_stall_refresh", 0.0))
        stall_reason = (
            str(group["preview_primary_stall_refresh_reason"].mode().iloc[0])
            if "preview_primary_stall_refresh_reason" in group
            else ""
        )
        rows.append(
            {
                "bucket": int(bucket),
                "start_min": float(bucket) * float(bucket_s) / 60.0,
                "end_min": float(bucket + 1) * float(bucket_s) / 60.0,
                "pitch_abs_p95": float(pitch.abs().quantile(0.95)),
                "roll_abs_p95": float(roll.abs().quantile(0.95)),
                "attitude_abs_p95": float(np.maximum(pitch.abs(), roll.abs()).quantile(0.95)),
                "high_ratio": float(
                    ((pitch.abs() > threshold_deg) | (roll.abs() > threshold_deg)).mean()
                ),
                "primary_action": action,
                "target_err_mean_kg": float(target_err.mean()),
                "pump_rate_mean_m3_min": float(pump_rate.mean()),
                "pump_idle_ratio": float((pump_rate <= 0.5).mean()),
                "fallback_ratio": float(fallback.mean()),
                "stall_refresh_ratio": float(stall_refresh.mean()),
                "stall_refresh_reason": stall_reason,
            }
        )
    out = pd.DataFrame(rows)
    out["high_bucket"] = (
        (out["pitch_abs_p95"] > threshold_deg) | (out["roll_abs_p95"] > threshold_deg)
    )
    out["continuous_high_buckets"] = _run_lengths([bool(x) for x in out["high_bucket"]])
    return out


def main() -> None:
    args = parse_args()
    casebook_dir = Path(args.casebook_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_dir = casebook_dir / "timeseries"
    log_dir = casebook_dir / "planner_logs"

    rows: list[pd.DataFrame] = []
    for ts_path in sorted(ts_dir.glob("*prediction_primary*_timeseries.csv")):
        case_id = ts_path.name.split("_202")[0]
        bucket_df = _bucket_timeseries(ts_path, args.bucket_s, args.threshold_deg)
        log_paths = sorted(log_dir.glob(f"{case_id}_*_planner_log.csv"))
        if log_paths:
            log = pd.read_csv(log_paths[0])
            keep = [
                c
                for c in [
                    "bucket",
                    "first_action",
                    "planner_first_action_raw",
                    "best_sequence",
                    "prediction_primary_stall_refresh",
                    "prediction_primary_stall_refresh_reason",
                    "prediction_primary_stall_target_err_mean_kg",
                    "prediction_primary_stall_pump_rate_m3_min",
                    "pressure_block0_norm",
                    "pressure_block1_norm",
                    "pressure_block2_norm",
                    "raw_pressure_block0_norm",
                    "raw_pressure_block1_norm",
                    "raw_pressure_block2_norm",
                    "recovery_mode_active",
                    "recovery_mode_entry_reason",
                    "recovery_mode_suggested_level",
                ]
                if c in log.columns
            ]
            bucket_df = bucket_df.merge(log[keep], on="bucket", how="left")
        bucket_df.insert(0, "case_id", case_id)
        rows.append(bucket_df)

    if not rows:
        raise SystemExit(f"no prediction-primary timeseries found under {ts_dir}")

    audit = pd.concat(rows, ignore_index=True)
    action = audit["primary_action"].astype(str)
    planner_action = audit.get("first_action", action).astype(str)
    target_err = _num(
        audit.get("prediction_primary_stall_target_err_mean_kg", audit["target_err_mean_kg"])
    )
    pump_rate = _num(
        audit.get("prediction_primary_stall_pump_rate_m3_min", audit["pump_rate_mean_m3_min"])
    )
    fallback = _num(audit["fallback_ratio"])
    stall_refresh = _num(audit.get("prediction_primary_stall_refresh", audit["stall_refresh_ratio"]))
    audit["hold_like"] = action.isin(["hold", "pump_saving"]) | planner_action.isin(
        ["hold", "pump_saving"]
    )
    audit["target_reached"] = target_err <= float(args.target_err_kg)
    audit["pump_idle"] = pump_rate <= float(args.pump_idle_m3_min)
    audit["fallback_dominated"] = fallback > 0.05
    audit["no_unexplained_hold_candidate"] = (
        audit["high_bucket"]
        & (audit["continuous_high_buckets"] >= int(args.trigger_buckets))
        & audit["hold_like"]
        & audit["target_reached"]
        & audit["pump_idle"]
        & ~audit["fallback_dominated"]
    )
    audit["primary_stall_would_cover"] = stall_refresh > 0.0
    audit["primary_stall_reason_observed"] = audit.get(
        "prediction_primary_stall_refresh_reason",
        audit.get("stall_refresh_reason", ""),
    )

    audit_path = out_dir / "no_unexplained_hold_bucket_audit.csv"
    audit.to_csv(audit_path, index=False)

    case_rows = []
    for case_id, group in audit.groupby("case_id", sort=True):
        high = group[group["high_bucket"]]
        candidates = group[group["no_unexplained_hold_candidate"]]
        case_rows.append(
            {
                "case_id": case_id,
                "buckets": int(len(group)),
                "high_buckets": int(len(high)),
                "max_continuous_high_buckets": int(group["continuous_high_buckets"].max()),
                "no_unexplained_hold_candidates": int(len(candidates)),
                "candidate_ratio_of_high": (
                    float(len(candidates) / len(high)) if len(high) else 0.0
                ),
                "primary_stall_covered_candidates": int(
                    candidates["primary_stall_would_cover"].sum()
                ),
                "primary_stall_candidate_coverage": (
                    float(candidates["primary_stall_would_cover"].mean())
                    if len(candidates)
                    else 0.0
                ),
                "dominant_candidate_action": (
                    ";".join(candidates["primary_action"].astype(str).value_counts().head(3).index)
                    if len(candidates)
                    else ""
                ),
                "candidate_pressure0_mean": (
                    float(_num(candidates.get("pressure_block0_norm", 0.0)).mean())
                    if len(candidates)
                    else 0.0
                ),
                "candidate_pressure2_mean": (
                    float(_num(candidates.get("pressure_block2_norm", 0.0)).mean())
                    if len(candidates)
                    else 0.0
                ),
            }
        )
    summary = pd.DataFrame(case_rows)
    summary_path = out_dir / "no_unexplained_hold_case_summary.csv"
    summary.to_csv(summary_path, index=False)

    candidates = audit[audit["no_unexplained_hold_candidate"]]
    md = out_dir / "no_unexplained_hold_audit.md"
    lines = [
        "# No-Unexplained-Hold Audit",
        "",
        f"Source: `{casebook_dir}`",
        f"Threshold: `{float(args.threshold_deg):.2f} deg`; trigger: `{int(args.trigger_buckets)} consecutive buckets`.",
        "",
        "## Case Summary",
        "",
        "| case | high buckets | max continuous high | no-unexplained-hold candidates | primary_stall coverage | candidate pressure0->2 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['case_id']} | {int(row['high_buckets'])} | "
            f"{int(row['max_continuous_high_buckets'])} | "
            f"{int(row['no_unexplained_hold_candidates'])} | "
            f"{row['primary_stall_candidate_coverage'] * 100:.1f}% | "
            f"{row['candidate_pressure0_mean']:.2f}->{row['candidate_pressure2_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Candidate buckets found: `{len(candidates)}`.",
            "- A candidate means high posture persisted past the configured bucket budget while the planner/action path remained hold-like, target was already reached, pump was idle, and fallback was not dominant.",
            "- `primary_stall_refresh` coverage is reported separately because it only operates on non-hold action vectors; pure hold buckets are expected to show low or zero coverage.",
            "",
            f"Bucket audit: `{audit_path}`",
            f"Case summary: `{summary_path}`",
        ]
    )
    md.write_text("\n".join(lines) + "\n")

    print(summary.to_string(index=False))
    print(md)


if __name__ == "__main__":
    main()
