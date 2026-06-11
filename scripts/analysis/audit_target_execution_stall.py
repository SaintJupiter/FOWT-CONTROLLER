#!/usr/bin/env python3
"""Bucket-level audit for prediction-primary target/execution stall.

This is a read-only diagnostic. It joins planner bucket logs with 1 Hz
timeseries and classifies each bucket by whether an active planner command is
tracking, dominated by fallback, or has reached the old tank target while
attitude remains high.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _p95_abs(series: pd.Series) -> float:
    vals = np.abs(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, 0.95))


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _target_columns(ts: pd.DataFrame) -> tuple[list[str], str]:
    preview = [
        "preview_primary_target_t1_kg",
        "preview_primary_target_t2_kg",
        "preview_primary_target_t3_kg",
    ]
    ordinary = ["target_tank1_kg", "target_tank2_kg", "target_tank3_kg"]
    if all(c in ts.columns for c in preview):
        return preview, "preview_primary_target"
    return ordinary, "closed_loop_target"


def _find_case_files(run_dir: Path, case_filter: list[str] | None) -> list[tuple[str, Path, Path]]:
    log_dir = run_dir / "planner_logs"
    ts_dir = run_dir / "timeseries"
    if not log_dir.exists() or not ts_dir.exists():
        raise FileNotFoundError(f"missing planner_logs/timeseries under {run_dir}")
    out: list[tuple[str, Path, Path]] = []
    for log_path in sorted(log_dir.glob("*_planner_log.csv")):
        stem = log_path.name.replace("_planner_log.csv", "")
        ts_path = ts_dir / f"{stem}_timeseries.csv"
        if not ts_path.exists():
            continue
        if case_filter and not any(token in stem for token in case_filter):
            continue
        out.append((stem, log_path, ts_path))
    return out


def audit_case(
    case_name: str,
    planner_log_path: Path,
    timeseries_path: Path,
    *,
    bucket_s: float,
    posture_pitch_deg: float,
    posture_roll_deg: float,
    target_err_kg: float,
    target_change_kg: float,
    pump_idle_rate_m3_min: float,
    fallback_threshold: float,
    improve_eps_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    log = pd.read_csv(planner_log_path)
    ts = pd.read_csv(timeseries_path)
    if "current_time_s" not in log.columns or "t_s" not in ts.columns:
        raise ValueError(f"missing time columns for {case_name}")

    mass_cols = ["tank1_kg", "tank2_kg", "tank3_kg"]
    target_cols, target_source = _target_columns(ts)
    required = ["pitch_deg", "roll_deg", "pump_total_rate_m3_min", *mass_cols, *target_cols]
    missing = [c for c in required if c not in ts.columns]
    if missing:
        raise ValueError(f"{case_name}: missing timeseries columns {missing}")

    fallback_col = _first_existing(
        ts,
        [
            "preview_primary_safety_fallback",
            "primary_safety_fallback",
            "safety_fallback",
        ],
    )

    rows: list[dict[str, Any]] = []
    prev_target_logged: np.ndarray | None = None
    for _, plan in log.iterrows():
        start = _as_float(plan.get("current_time_s"))
        end = start + float(bucket_s)
        bucket_ts = ts[(ts["t_s"] >= start - 1e-9) & (ts["t_s"] < end - 1e-9)]
        if bucket_ts.empty:
            continue
        first = bucket_ts.iloc[0]
        last = bucket_ts.iloc[-1]
        action = str(plan.get("first_action", "hold"))
        active = action != "hold"

        mass = bucket_ts[mass_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        target = bucket_ts[target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mass_start = mass[0]
        mass_end = mass[-1]
        target_start = target[0]
        target_end = target[-1]
        target_err = np.nanmean(np.abs(mass - target), axis=1)
        target_err_mean = float(np.nanmean(target_err))
        target_err_end = float(target_err[-1])
        actual_delta_mean_kg = float(np.nanmean(np.abs(mass_end - mass_start)))
        target_motion_mean_kg = float(np.nanmean(np.abs(target_end - target_start)))
        logged_target = np.array(
            [
                _as_float(plan.get("prediction_primary_target_t1_kg"), float("nan")),
                _as_float(plan.get("prediction_primary_target_t2_kg"), float("nan")),
                _as_float(plan.get("prediction_primary_target_t3_kg"), float("nan")),
            ],
            dtype=float,
        )
        if prev_target_logged is None or not np.all(np.isfinite(prev_target_logged)):
            target_delta_from_prev_mean_kg = float("nan")
            target_changed_from_prev = False
        elif np.all(np.isfinite(logged_target)):
            target_delta_from_prev_mean_kg = float(np.nanmean(np.abs(logged_target - prev_target_logged)))
            target_changed_from_prev = target_delta_from_prev_mean_kg >= float(target_change_kg)
        else:
            target_delta_from_prev_mean_kg = float("nan")
            target_changed_from_prev = False
        if np.all(np.isfinite(logged_target)):
            prev_target_logged = logged_target

        pump_rate = pd.to_numeric(
            bucket_ts["pump_total_rate_m3_min"],
            errors="coerce",
        ).fillna(0.0)
        pump_idle_ratio = float(np.mean(np.abs(pump_rate.to_numpy(dtype=float)) <= pump_idle_rate_m3_min))
        pump_work_m3 = float(np.nansum(pump_rate.to_numpy(dtype=float)) / 60.0)
        fallback_ratio = 0.0
        if fallback_col is not None:
            fallback_vals = pd.to_numeric(bucket_ts[fallback_col], errors="coerce").fillna(0.0)
            fallback_ratio = float(np.mean(fallback_vals.to_numpy(dtype=float) > 0.5))

        pitch_start = _as_float(first.get("pitch_deg"))
        roll_start = _as_float(first.get("roll_deg"))
        pitch_end = _as_float(last.get("pitch_deg"))
        roll_end = _as_float(last.get("roll_deg"))
        pitch_abs_start = abs(pitch_start)
        roll_abs_start = abs(roll_start)
        pitch_abs_end = abs(pitch_end)
        roll_abs_end = abs(roll_end)
        posture_high = (
            pitch_abs_start >= float(posture_pitch_deg)
            or roll_abs_start >= float(posture_roll_deg)
        )
        pitch_high_start = pitch_abs_start >= float(posture_pitch_deg)
        roll_high_start = roll_abs_start >= float(posture_roll_deg)
        pitch_high_end = pitch_abs_end >= float(posture_pitch_deg)
        roll_high_end = roll_abs_end >= float(posture_roll_deg)
        pitch_high = pitch_high_start or pitch_high_end
        roll_high = roll_high_start or roll_high_end
        pitch_improving = pitch_abs_end <= pitch_abs_start - float(improve_eps_deg)
        roll_improving = roll_abs_end <= roll_abs_start - float(improve_eps_deg)
        pitch_worsened_into_problem = (
            pitch_abs_end >= float(posture_pitch_deg)
            and pitch_abs_end >= pitch_abs_start + float(improve_eps_deg)
        )
        roll_worsened_into_problem = (
            roll_abs_end >= float(posture_roll_deg)
            and roll_abs_end >= roll_abs_start + float(improve_eps_deg)
        )
        if pitch_high_start and roll_high_start:
            attitude_improving = pitch_improving and roll_improving
        elif pitch_high_start:
            attitude_improving = pitch_improving
        elif roll_high_start:
            attitude_improving = roll_improving
        else:
            attitude_improving = pitch_improving or roll_improving
        if pitch_worsened_into_problem or roll_worsened_into_problem:
            attitude_improving = False

        target_reached = target_err_end <= float(target_err_kg)
        pump_idle = pump_idle_ratio >= 0.80
        pump_active = not pump_idle
        fallback_dominated = fallback_ratio >= float(fallback_threshold)
        target_updated = (
            target_changed_from_prev
            or _as_float(plan.get("prediction_primary_target_refreshed")) > 0.5
            or target_motion_mean_kg >= float(target_change_kg)
        )
        target_stall_candidate = (
            active
            and posture_high
            and target_reached
            and pump_idle
            and not target_updated
            and not attitude_improving
        )
        pitch_authority_limited = (
            active
            and pitch_high
            and not pitch_improving
            and not target_stall_candidate
            and (target_updated or pump_active or actual_delta_mean_kg >= float(target_change_kg) or not target_reached)
        )
        normal_improving = active and posture_high and attitude_improving
        if not active:
            classification = "hold"
        elif fallback_dominated:
            classification = "fallback_dominated"
        elif not posture_high:
            classification = "off_band"
        elif target_stall_candidate:
            classification = "target_stall"
        elif pitch_authority_limited:
            classification = "pitch_authority_limited"
        elif normal_improving:
            classification = "normal_improving"
        elif not target_reached:
            classification = "tracking"
        else:
            classification = "active_pumping"

        row = {
            "case": case_name,
            "bucket": int(_as_float(plan.get("bucket"), len(rows))),
            "bucket_start_s": start,
            "first_action": action,
            "planner_action_pitch_deg": _as_float(plan.get("planner_action_pitch_deg")),
            "planner_action_roll_deg": _as_float(plan.get("planner_action_roll_deg")),
            "target_t1_kg": _as_float(plan.get("prediction_primary_target_t1_kg")),
            "target_t2_kg": _as_float(plan.get("prediction_primary_target_t2_kg")),
            "target_t3_kg": _as_float(plan.get("prediction_primary_target_t3_kg")),
            "actual_t1_start_kg": _as_float(first.get("tank1_kg")),
            "actual_t2_start_kg": _as_float(first.get("tank2_kg")),
            "actual_t3_start_kg": _as_float(first.get("tank3_kg")),
            "target_err_mean_kg": target_err_mean,
            "target_err_end_kg": target_err_end,
            "target_delta_from_prev_mean_kg": target_delta_from_prev_mean_kg,
            "target_changed_from_prev": int(target_changed_from_prev),
            "target_motion_mean_kg": target_motion_mean_kg,
            "actual_delta_mean_kg": actual_delta_mean_kg,
            "target_refreshed": int(_as_float(plan.get("prediction_primary_target_refreshed")) > 0.5),
            "target_reused": int(_as_float(plan.get("prediction_primary_target_reused")) > 0.5),
            "target_resumed": int(_as_float(plan.get("prediction_primary_target_resumed")) > 0.5),
            "target_age_s": _as_float(plan.get("prediction_primary_target_age_s")),
            "pump_work_m3": pump_work_m3,
            "pump_idle_ratio": pump_idle_ratio,
            "fallback_ratio": fallback_ratio,
            "pitch_start_deg": pitch_start,
            "roll_start_deg": roll_start,
            "pitch_end_deg": pitch_end,
            "roll_end_deg": roll_end,
            "d_abs_pitch_deg": pitch_abs_end - pitch_abs_start,
            "d_abs_roll_deg": roll_abs_end - roll_abs_start,
            "pitch_p95_abs_deg": _p95_abs(bucket_ts["pitch_deg"]),
            "roll_p95_abs_deg": _p95_abs(bucket_ts["roll_deg"]),
            "posture_high": int(posture_high),
            "pitch_high": int(pitch_high),
            "roll_high": int(roll_high),
            "pitch_high_start": int(pitch_high_start),
            "roll_high_start": int(roll_high_start),
            "pitch_high_end": int(pitch_high_end),
            "roll_high_end": int(roll_high_end),
            "pitch_worsened_into_problem": int(pitch_worsened_into_problem),
            "roll_worsened_into_problem": int(roll_worsened_into_problem),
            "target_reached": int(target_reached),
            "attitude_improving": int(attitude_improving),
            "pitch_improving": int(pitch_improving),
            "roll_improving": int(roll_improving),
            "target_updated": int(target_updated),
            "target_stall_candidate": int(target_stall_candidate),
            "pitch_authority_limited": int(pitch_authority_limited),
            "normal_improving": int(normal_improving),
            "classification": classification,
            "target_source": target_source,
        }
        rows.append(row)

    active_rows = [r for r in rows if r["first_action"] != "hold"]
    high_active = [r for r in active_rows if r["posture_high"]]
    class_counts = {
        name: sum(r["classification"] == name for r in rows)
        for name in [
            "target_stall",
            "pitch_authority_limited",
            "fallback_dominated",
            "normal_improving",
            "tracking",
            "active_pumping",
            "off_band",
            "hold",
        ]
    }
    diagnostic_rows = [
        r for r in rows if r["classification"] in {
            "target_stall",
            "pitch_authority_limited",
            "fallback_dominated",
            "normal_improving",
            "tracking",
            "active_pumping",
        }
    ]
    summary = {
        "case": case_name,
        "buckets": len(rows),
        "active_buckets": len(active_rows),
        "high_active_buckets": len(high_active),
        "target_stall_buckets": class_counts["target_stall"],
        "target_stall_ratio_active": class_counts["target_stall"] / len(active_rows)
        if active_rows
        else 0.0,
        "target_stall_ratio_high_active": class_counts["target_stall"] / len(high_active)
        if high_active
        else 0.0,
        "pitch_authority_limited_buckets": class_counts["pitch_authority_limited"],
        "fallback_dominated_buckets": class_counts["fallback_dominated"],
        "normal_improving_buckets": class_counts["normal_improving"],
        "tracking_buckets": class_counts["tracking"],
        "active_pumping_buckets": class_counts["active_pumping"],
        "off_band_buckets": class_counts["off_band"],
        "hold_buckets": class_counts["hold"],
        "target_stall_share_diagnostic": class_counts["target_stall"] / len(diagnostic_rows)
        if diagnostic_rows
        else 0.0,
        "pitch_authority_share_diagnostic": class_counts["pitch_authority_limited"] / len(diagnostic_rows)
        if diagnostic_rows
        else 0.0,
        "fallback_share_diagnostic": class_counts["fallback_dominated"] / len(diagnostic_rows)
        if diagnostic_rows
        else 0.0,
        "normal_improving_share_diagnostic": class_counts["normal_improving"] / len(diagnostic_rows)
        if diagnostic_rows
        else 0.0,
        "mean_pump_work_m3_per_bucket": float(np.mean([r["pump_work_m3"] for r in rows]))
        if rows
        else 0.0,
        "mean_pitch_p95_abs_deg": float(np.nanmean([r["pitch_p95_abs_deg"] for r in rows]))
        if rows
        else float("nan"),
        "mean_roll_p95_abs_deg": float(np.nanmean([r["roll_p95_abs_deg"] for r in rows]))
        if rows
        else float("nan"),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-filter", action="append", default=None)
    parser.add_argument("--bucket-s", type=float, default=600.0)
    parser.add_argument("--posture-pitch-deg", type=float, default=5.0)
    parser.add_argument("--posture-roll-deg", type=float, default=4.0)
    parser.add_argument("--target-err-kg", type=float, default=1500.0)
    parser.add_argument("--target-change-kg", type=float, default=1500.0)
    parser.add_argument("--pump-idle-rate-m3-min", type=float, default=0.5)
    parser.add_argument("--fallback-threshold", type=float, default=0.20)
    parser.add_argument("--improve-eps-deg", type=float, default=0.10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_files = _find_case_files(args.run_dir, args.case_filter)
    if not case_files:
        raise SystemExit(f"no case files found in {args.run_dir}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case_name, log_path, ts_path in case_files:
        rows, summary = audit_case(
            case_name,
            log_path,
            ts_path,
            bucket_s=float(args.bucket_s),
            posture_pitch_deg=float(args.posture_pitch_deg),
            posture_roll_deg=float(args.posture_roll_deg),
            target_err_kg=float(args.target_err_kg),
            target_change_kg=float(args.target_change_kg),
            pump_idle_rate_m3_min=float(args.pump_idle_rate_m3_min),
            fallback_threshold=float(args.fallback_threshold),
            improve_eps_deg=float(args.improve_eps_deg),
        )
        all_rows.extend(rows)
        summaries.append(summary)

    bucket_csv = args.out_dir / "bucket_target_execution_audit.csv"
    summary_csv = args.out_dir / "case_target_execution_summary.csv"
    pd.DataFrame(all_rows).to_csv(bucket_csv, index=False)
    pd.DataFrame(summaries).to_csv(summary_csv, index=False)

    print(f"Wrote {bucket_csv}")
    print(f"Wrote {summary_csv}")
    for summary in summaries:
        print(
            f"{summary['case']}: active={summary['active_buckets']} "
            f"high_active={summary['high_active_buckets']} "
            f"target_stall={summary['target_stall_buckets']} "
            f"stall/high_active={summary['target_stall_ratio_high_active']:.2f} "
            f"pitch_limited={summary['pitch_authority_limited_buckets']} "
            f"fallback_dom={summary['fallback_dominated_buckets']} "
            f"normal_improving={summary['normal_improving_buckets']}"
        )


if __name__ == "__main__":
    main()
