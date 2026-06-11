#!/usr/bin/env python3
"""Build a bucket-level command-chain trace from existing casebook outputs.

This is read-only. It merges planner logs with 1 Hz timeseries so each bucket
shows the full path:

planner action -> implied target proposal -> accepted primary target ->
final execution target/fallback -> pump/mass motion -> attitude response.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from wind_prediction.ballast_planner import PlannerConfig, tank_signal  # noqa: E402


PITCH_BASIS = np.array([-1.0, 0.5, 0.5], dtype=float)
ROLL_BASIS = np.array([0.0, 1.0, -1.0], dtype=float)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _mean_abs(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def _axis_coeff(delta: np.ndarray, basis: np.ndarray) -> float:
    denom = float(np.dot(basis, basis))
    if denom <= 0.0 or not np.all(np.isfinite(delta)):
        return float("nan")
    return float(np.dot(delta, basis) / denom)


def _axis_label(pitch: float, roll: float, ratio: float = 1.5) -> str:
    ap = abs(float(pitch))
    ar = abs(float(roll))
    if ap < 1e-9 and ar < 1e-9:
        return "none"
    if ap >= ratio * max(ar, 1e-9):
        return "pitch"
    if ar >= ratio * max(ap, 1e-9):
        return "roll"
    return "mixed"


def _find_pairs(run_dir: Path, case_filters: list[str] | None) -> list[tuple[str, Path, Path]]:
    pairs = []
    for log_path in sorted((run_dir / "planner_logs").glob("*_planner_log.csv")):
        stem = log_path.name.replace("_planner_log.csv", "")
        if case_filters and not any(token in stem for token in case_filters):
            continue
        ts_path = run_dir / "timeseries" / f"{stem}_timeseries.csv"
        if ts_path.exists():
            pairs.append((stem, log_path, ts_path))
    if not pairs:
        raise FileNotFoundError(f"no planner/timeseries pairs found under {run_dir}")
    return pairs


def _primary_scale(run_dir: Path, case_name: str) -> float:
    summary_path = run_dir / "casebook_summary.csv"
    if not summary_path.exists():
        return 1.0
    try:
        summary = pd.read_csv(summary_path)
    except Exception:
        return 1.0
    if "primary_scale" not in summary.columns or "case_id" not in summary.columns:
        return 1.0
    case_id = case_name.split("_202")[0]
    match = summary[summary["case_id"].astype(str).apply(lambda x: x in case_name or case_id.endswith(x))]
    if match.empty:
        match = summary[summary["case_id"].astype(str).apply(lambda x: x in case_name)]
    if match.empty:
        return 1.0
    return _as_float(match.iloc[0].get("primary_scale"), 1.0)


def trace_case(
    run_dir: Path,
    case_name: str,
    log_path: Path,
    ts_path: Path,
    *,
    bucket_s: float,
    pitch_high_deg: float,
    roll_high_deg: float,
) -> pd.DataFrame:
    cfg = PlannerConfig()
    scale = _primary_scale(run_dir, case_name)
    log = pd.read_csv(log_path)
    ts = pd.read_csv(ts_path)
    rows: list[dict[str, Any]] = []
    prev_primary_target: np.ndarray | None = None

    for _, plan in log.iterrows():
        start = _as_float(plan.get("current_time_s"))
        end = start + float(bucket_s)
        seg = ts[(ts["t_s"] >= start - 1e-9) & (ts["t_s"] < end - 1e-9)]
        if seg.empty:
            continue
        first = seg.iloc[0]
        last = seg.iloc[-1]

        mass_start = np.array([_as_float(first.get(f"tank{i}_kg"), float("nan")) for i in (1, 2, 3)])
        mass_end = np.array([_as_float(last.get(f"tank{i}_kg"), float("nan")) for i in (1, 2, 3)])
        primary_target = np.array(
            [
                _as_float(plan.get("prediction_primary_target_t1_kg"), float("nan")),
                _as_float(plan.get("prediction_primary_target_t2_kg"), float("nan")),
                _as_float(plan.get("prediction_primary_target_t3_kg"), float("nan")),
            ],
            dtype=float,
        )
        final_target_start = np.array(
            [
                _as_float(first.get(f"target_tank{i}_kg"), float("nan"))
                for i in (1, 2, 3)
            ],
            dtype=float,
        )
        final_target_end = np.array(
            [
                _as_float(last.get(f"target_tank{i}_kg"), float("nan"))
                for i in (1, 2, 3)
            ],
            dtype=float,
        )

        avec = np.array(
            [
                _as_float(plan.get("planner_action_pitch_deg")),
                _as_float(plan.get("planner_action_roll_deg")),
            ],
            dtype=float,
        )
        proposal_delta = tank_signal(avec, cfg) * cfg.action_mass_quantum_kg * scale
        proposal_target = np.clip(mass_start + proposal_delta, 0.0, cfg.tank_capacity_kg)
        if prev_primary_target is None:
            primary_target_delta_mean = float("nan")
            proposal_vs_prev_primary_mean = float("nan")
            primary_target_before = np.full(3, np.nan, dtype=float)
        else:
            primary_target_before = prev_primary_target.copy()
            primary_target_delta_mean = _mean_abs(primary_target - prev_primary_target)
            proposal_vs_prev_primary_mean = _mean_abs(proposal_target - prev_primary_target)

        fallback_vals = seg.get("preview_primary_safety_fallback")
        fallback_ratio = (
            float(np.mean(pd.to_numeric(fallback_vals, errors="coerce").fillna(0.0).to_numpy() > 0.5))
            if fallback_vals is not None
            else 0.0
        )
        pump_rate = pd.to_numeric(seg["pump_total_rate_m3_min"], errors="coerce").fillna(0.0)
        pump_work_m3 = float(np.sum(pump_rate.to_numpy(dtype=float)) / 60.0)
        pump_active_ratio = float(np.mean(np.abs(pump_rate.to_numpy(dtype=float)) > 0.5))

        pitch_start = _as_float(first.get("pitch_deg"))
        pitch_end = _as_float(last.get("pitch_deg"))
        roll_start = _as_float(first.get("roll_deg"))
        roll_end = _as_float(last.get("roll_deg"))
        d_abs_pitch = abs(pitch_end) - abs(pitch_start)
        d_abs_roll = abs(roll_end) - abs(roll_start)
        high_posture = max(abs(pitch_start), abs(pitch_end)) >= pitch_high_deg or max(abs(roll_start), abs(roll_end)) >= roll_high_deg
        fallback_overlay_mean = _mean_abs(final_target_end - primary_target)

        if fallback_ratio >= 0.20:
            chain_label = "fallback_overlay"
        elif (
            str(plan.get("first_action", "hold")) != "hold"
            and high_posture
            and primary_target_delta_mean == 0.0
            and proposal_vs_prev_primary_mean >= 1500.0
        ):
            chain_label = "active_intent_blocked_by_reuse"
        elif pump_active_ratio < 0.20 and _mean_abs(primary_target - mass_end) <= 1500.0 and high_posture:
            chain_label = "target_reached_high_posture"
        elif pump_active_ratio >= 0.20 and d_abs_pitch > 0.10 and max(abs(pitch_start), abs(pitch_end)) >= pitch_high_deg:
            chain_label = "mass_moved_pitch_worse"
        elif d_abs_pitch <= -0.10 or d_abs_roll <= -0.10:
            chain_label = "improving"
        else:
            chain_label = "neutral_or_off_band"

        rows.append(
            {
                "case": case_name,
                "bucket": int(_as_float(plan.get("bucket"), len(rows))),
                "bucket_start_s": start,
                "first_action": str(plan.get("first_action", "hold")),
                "best_sequence": str(plan.get("best_sequence", "")),
                "action_pitch_deg": avec[0],
                "action_roll_deg": avec[1],
                "action_axis": _axis_label(avec[0], avec[1]),
                "pressure_block0_norm": _as_float(plan.get("pressure_block0_norm"), float("nan")),
                "pressure_block1_norm": _as_float(plan.get("pressure_block1_norm"), float("nan")),
                "pressure_block2_norm": _as_float(plan.get("pressure_block2_norm"), float("nan")),
                "proposal_delta_t1_kg": proposal_delta[0],
                "proposal_delta_t2_kg": proposal_delta[1],
                "proposal_delta_t3_kg": proposal_delta[2],
                "proposal_delta_mean_abs_kg": _mean_abs(proposal_delta),
                "proposal_pitch_coeff_kg": _axis_coeff(proposal_delta, PITCH_BASIS),
                "proposal_roll_coeff_kg": _axis_coeff(proposal_delta, ROLL_BASIS),
                "proposal_vs_prev_primary_mean_kg": proposal_vs_prev_primary_mean,
                "primary_target_before_t1_kg": primary_target_before[0],
                "primary_target_before_t2_kg": primary_target_before[1],
                "primary_target_before_t3_kg": primary_target_before[2],
                "primary_target_t1_kg": primary_target[0],
                "primary_target_t2_kg": primary_target[1],
                "primary_target_t3_kg": primary_target[2],
                "primary_target_delta_from_prev_mean_kg": primary_target_delta_mean,
                "primary_target_refreshed": int(_as_float(plan.get("prediction_primary_target_refreshed")) > 0.5),
                "primary_target_reused": int(_as_float(plan.get("prediction_primary_target_reused")) > 0.5),
                "primary_target_resumed": int(_as_float(plan.get("prediction_primary_target_resumed")) > 0.5),
                "primary_target_age_s": _as_float(plan.get("prediction_primary_target_age_s"), float("nan")),
                "active_intent_reproposal": int(_as_float(plan.get("prediction_primary_active_intent_reproposal")) > 0.5),
                "active_intent_reproposal_reason": str(plan.get("prediction_primary_active_intent_reproposal_reason", "")),
                "active_intent_reproposal_delta_mean_kg": _as_float(plan.get("prediction_primary_active_intent_reproposal_delta_mean_kg"), float("nan")),
                "final_target_delta_from_primary_start_mean_kg": _mean_abs(final_target_start - primary_target),
                "final_target_delta_from_primary_end_mean_kg": fallback_overlay_mean,
                "fallback_ratio": fallback_ratio,
                "final_target_start_t1_kg": final_target_start[0],
                "final_target_start_t2_kg": final_target_start[1],
                "final_target_start_t3_kg": final_target_start[2],
                "final_target_end_t1_kg": final_target_end[0],
                "final_target_end_t2_kg": final_target_end[1],
                "final_target_end_t3_kg": final_target_end[2],
                "actual_tank1_start_kg": mass_start[0],
                "actual_tank2_start_kg": mass_start[1],
                "actual_tank3_start_kg": mass_start[2],
                "actual_tank1_end_kg": mass_end[0],
                "actual_tank2_end_kg": mass_end[1],
                "actual_tank3_end_kg": mass_end[2],
                "mass_delta_t1_kg": mass_end[0] - mass_start[0],
                "mass_delta_t2_kg": mass_end[1] - mass_start[1],
                "mass_delta_t3_kg": mass_end[2] - mass_start[2],
                "mass_delta_mean_abs_kg": _mean_abs(mass_end - mass_start),
                "mass_pitch_coeff_kg": _axis_coeff(mass_end - mass_start, PITCH_BASIS),
                "mass_roll_coeff_kg": _axis_coeff(mass_end - mass_start, ROLL_BASIS),
                "target_err_start_mean_kg": _mean_abs(primary_target - mass_start),
                "target_err_end_mean_kg": _mean_abs(primary_target - mass_end),
                "pump_work_m3": pump_work_m3,
                "pump_active_ratio": pump_active_ratio,
                "pitch_start_deg": pitch_start,
                "pitch_end_deg": pitch_end,
                "roll_start_deg": roll_start,
                "roll_end_deg": roll_end,
                "d_abs_pitch_deg": d_abs_pitch,
                "d_abs_roll_deg": d_abs_roll,
                "chain_label": chain_label,
            }
        )
        if np.all(np.isfinite(primary_target)):
            prev_primary_target = primary_target

    out = pd.DataFrame(rows)
    for lead in (1, 2, 3):
        out[f"future{lead}_d_abs_pitch_from_now_deg"] = out["d_abs_pitch_deg"].shift(-lead)
        out[f"future{lead}_d_abs_roll_from_now_deg"] = out["d_abs_roll_deg"].shift(-lead)
    return out


def write_summary(trace: pd.DataFrame, out_path: Path) -> None:
    lines = ["# Command Chain Trace Summary", ""]
    for case, df in trace.groupby("case", sort=False):
        counts = df["chain_label"].value_counts()
        lines.append(f"## {case}")
        lines.append("")
        for label, count in counts.items():
            lines.append(f"- {label}: {int(count)}/{len(df)}")
        lines.append("")
        high = df[(df["pitch_start_deg"].abs() >= 5.0) | (df["pitch_end_deg"].abs() >= 5.0)]
        if not high.empty:
            lines.append("High-pitch buckets:")
            keep = [
                "bucket",
                "first_action",
                "action_axis",
                "primary_target_delta_from_prev_mean_kg",
                "proposal_vs_prev_primary_mean_kg",
                "fallback_ratio",
                "pump_work_m3",
                "pitch_start_deg",
                "pitch_end_deg",
                "roll_start_deg",
                "roll_end_deg",
                "chain_label",
            ]
            lines.append("")
            lines.append("```csv")
            lines.append(high[keep].to_csv(index=False).strip())
            lines.append("```")
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--case-filter", action="append", default=None)
    parser.add_argument("--bucket-s", type=float, default=600.0)
    parser.add_argument("--pitch-high-deg", type=float, default=5.0)
    parser.add_argument("--roll-high-deg", type=float, default=4.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    for case, log_path, ts_path in _find_pairs(args.run_dir, args.case_filter):
        traces.append(
            trace_case(
                args.run_dir,
                case,
                log_path,
                ts_path,
                bucket_s=args.bucket_s,
                pitch_high_deg=args.pitch_high_deg,
                roll_high_deg=args.roll_high_deg,
            )
        )
    trace = pd.concat(traces, ignore_index=True)
    csv_path = args.out_dir / "command_chain_trace.csv"
    md_path = args.out_dir / "command_chain_summary.md"
    trace.to_csv(csv_path, index=False)
    write_summary(trace, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
