#!/usr/bin/env python3
"""Read-only pitch-authority audit for prediction-primary case logs.

This diagnostic focuses on high-pitch buckets and traces whether a planner
active action turns into tank targets, actual mass motion, and subsequent pitch
response. It does not run a controller or modify controller parameters.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PITCH_BASIS = np.array([-1.0, 0.5, 0.5], dtype=float)
ROLL_BASIS = np.array([0.0, 1.0, -1.0], dtype=float)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _axis_from_action(pitch: float, roll: float, ratio: float) -> str:
    ap = abs(pitch)
    ar = abs(roll)
    if ap < 1e-9 and ar < 1e-9:
        return "none"
    if ap >= ratio * max(ar, 1e-9):
        return "pitch"
    if ar >= ratio * max(ap, 1e-9):
        return "roll"
    return "mixed"


def _basis_coeff(delta: np.ndarray, basis: np.ndarray) -> float:
    denom = float(np.dot(basis, basis))
    if denom <= 0.0 or not np.all(np.isfinite(delta)):
        return float("nan")
    return float(np.dot(delta, basis) / denom)


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _case_files(run_dir: Path, token: str) -> tuple[str, Path, Path]:
    log_dir = run_dir / "planner_logs"
    ts_dir = run_dir / "timeseries"
    matches: list[tuple[str, Path, Path]] = []
    for log_path in sorted(log_dir.glob("*_planner_log.csv")):
        stem = log_path.name.replace("_planner_log.csv", "")
        if token not in stem:
            continue
        ts_path = ts_dir / f"{stem}_timeseries.csv"
        if ts_path.exists():
            matches.append((stem, log_path, ts_path))
    if not matches:
        raise FileNotFoundError(f"no planner/timeseries pair matching {token!r} under {run_dir}")
    if len(matches) > 1:
        names = ", ".join(m[0] for m in matches)
        raise ValueError(f"ambiguous case token {token!r}: {names}")
    return matches[0]


def _mean_abs(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    return float(np.mean(np.abs(vals)))


def _fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def audit_case(
    case_name: str,
    planner_log_path: Path,
    timeseries_path: Path,
    *,
    bucket_s: float,
    pitch_high_deg: float,
    axis_ratio: float,
    target_weak_kg: float,
    mass_weak_kg: float,
    fallback_threshold: float,
    improve_eps_deg: float,
    lookahead_buckets: int,
) -> tuple[pd.DataFrame, str]:
    log = pd.read_csv(planner_log_path)
    ts = pd.read_csv(timeseries_path)

    fallback_col = _first_existing(
        ts,
        [
            "preview_primary_safety_fallback",
            "primary_safety_fallback",
            "safety_fallback",
        ],
    )

    rows: list[dict[str, Any]] = []
    prev_target: np.ndarray | None = None
    for _, plan in log.iterrows():
        start = _as_float(plan.get("current_time_s"))
        end = start + float(bucket_s)
        bucket = ts[(ts["t_s"] >= start - 1e-9) & (ts["t_s"] < end - 1e-9)]
        if bucket.empty:
            continue

        first = bucket.iloc[0]
        last = bucket.iloc[-1]
        action = str(plan.get("first_action", "hold"))
        action_pitch = _as_float(plan.get("planner_action_pitch_deg"))
        action_roll = _as_float(plan.get("planner_action_roll_deg"))
        action_axis = _axis_from_action(action_pitch, action_roll, axis_ratio)

        target = np.array(
            [
                _as_float(plan.get("prediction_primary_target_t1_kg"), float("nan")),
                _as_float(plan.get("prediction_primary_target_t2_kg"), float("nan")),
                _as_float(plan.get("prediction_primary_target_t3_kg"), float("nan")),
            ],
            dtype=float,
        )
        mass_start = np.array(
            [
                _as_float(first.get("tank1_kg"), float("nan")),
                _as_float(first.get("tank2_kg"), float("nan")),
                _as_float(first.get("tank3_kg"), float("nan")),
            ],
            dtype=float,
        )
        mass_end = np.array(
            [
                _as_float(last.get("tank1_kg"), float("nan")),
                _as_float(last.get("tank2_kg"), float("nan")),
                _as_float(last.get("tank3_kg"), float("nan")),
            ],
            dtype=float,
        )

        if prev_target is None or not np.all(np.isfinite(prev_target)):
            target_delta = target - mass_start
            target_delta_source = "from_bucket_start_mass"
        else:
            target_delta = target - prev_target
            target_delta_source = "from_previous_bucket_target"
        if np.all(np.isfinite(target)):
            prev_target = target

        actual_delta = mass_end - mass_start
        target_pitch_coeff = _basis_coeff(target_delta, PITCH_BASIS)
        target_roll_coeff = _basis_coeff(target_delta, ROLL_BASIS)
        actual_pitch_coeff = _basis_coeff(actual_delta, PITCH_BASIS)
        actual_roll_coeff = _basis_coeff(actual_delta, ROLL_BASIS)

        target_err_end = _mean_abs(target - mass_end)
        target_err_start = _mean_abs(target - mass_start)
        target_reused = int(_as_float(plan.get("prediction_primary_target_reused")) > 0.5)
        target_refreshed = int(_as_float(plan.get("prediction_primary_target_refreshed")) > 0.5)

        pump_rate = pd.to_numeric(bucket["pump_total_rate_m3_min"], errors="coerce").fillna(0.0)
        pump_volume_m3 = float(np.sum(pump_rate.to_numpy(dtype=float)) / 60.0)
        pump_active_ratio = float(np.mean(np.abs(pump_rate.to_numpy(dtype=float)) > 0.5))
        pump_active = pump_active_ratio >= 0.20

        fallback_ratio = 0.0
        if fallback_col:
            fallback_vals = pd.to_numeric(bucket[fallback_col], errors="coerce").fillna(0.0)
            fallback_ratio = float(np.mean(fallback_vals.to_numpy(dtype=float) > 0.5))

        pitch_start = _as_float(first.get("pitch_deg"))
        pitch_end = _as_float(last.get("pitch_deg"))
        roll_start = _as_float(first.get("roll_deg"))
        roll_end = _as_float(last.get("roll_deg"))
        d_abs_pitch = abs(pitch_end) - abs(pitch_start)
        d_abs_roll = abs(roll_end) - abs(roll_start)
        pitch_high = max(abs(pitch_start), abs(pitch_end)) >= float(pitch_high_deg)
        pitch_immediate_improving = d_abs_pitch <= -float(improve_eps_deg)
        pitch_immediate_worse = d_abs_pitch >= float(improve_eps_deg)

        active = action != "hold"
        fallback_overlay = fallback_ratio >= float(fallback_threshold)
        pitch_target_weak = (
            active
            and pitch_high
            and not fallback_overlay
            and abs(target_pitch_coeff) < float(target_weak_kg)
        )
        pitch_mass_weak = (
            active
            and pitch_high
            and not fallback_overlay
            and abs(target_pitch_coeff) >= float(target_weak_kg)
            and abs(actual_pitch_coeff) < float(mass_weak_kg)
            and target_err_end > float(mass_weak_kg)
        )
        pitch_authority_limited = (
            active
            and pitch_high
            and not fallback_overlay
            and abs(target_pitch_coeff) >= float(target_weak_kg)
            and abs(actual_pitch_coeff) >= float(mass_weak_kg)
            and not pitch_immediate_improving
        )
        normal_improving = active and pitch_high and not fallback_overlay and pitch_immediate_improving

        rows.append(
            {
                "case": case_name,
                "bucket": int(_as_float(plan.get("bucket"), len(rows))),
                "bucket_start_s": start,
                "first_action": action,
                "action_pitch_deg": action_pitch,
                "action_roll_deg": action_roll,
                "action_axis": action_axis,
                "target_delta_source": target_delta_source,
                "target_delta_t1_kg": float(target_delta[0]),
                "target_delta_t2_kg": float(target_delta[1]),
                "target_delta_t3_kg": float(target_delta[2]),
                "target_delta_mean_abs_kg": _mean_abs(target_delta),
                "target_pitch_coeff_kg": target_pitch_coeff,
                "target_roll_coeff_kg": target_roll_coeff,
                "actual_delta_t1_kg": float(actual_delta[0]),
                "actual_delta_t2_kg": float(actual_delta[1]),
                "actual_delta_t3_kg": float(actual_delta[2]),
                "actual_delta_mean_abs_kg": _mean_abs(actual_delta),
                "actual_pitch_coeff_kg": actual_pitch_coeff,
                "actual_roll_coeff_kg": actual_roll_coeff,
                "target_reused": target_reused,
                "target_refreshed": target_refreshed,
                "target_age_s": _as_float(plan.get("prediction_primary_target_age_s")),
                "target_err_start_kg": target_err_start,
                "target_err_end_kg": target_err_end,
                "pump_volume_m3": pump_volume_m3,
                "pump_active_ratio": pump_active_ratio,
                "pump_active": int(pump_active),
                "fallback_ratio": fallback_ratio,
                "pitch_start_deg": pitch_start,
                "pitch_end_deg": pitch_end,
                "roll_start_deg": roll_start,
                "roll_end_deg": roll_end,
                "d_abs_pitch_deg": d_abs_pitch,
                "d_abs_roll_deg": d_abs_roll,
                "pitch_high": int(pitch_high),
                "pitch_immediate_improving": int(pitch_immediate_improving),
                "pitch_immediate_worse": int(pitch_immediate_worse),
                "pitch_target_weak": int(pitch_target_weak),
                "pitch_mass_weak": int(pitch_mass_weak),
                "pitch_authority_limited": int(pitch_authority_limited),
                "fallback_overlay": int(fallback_overlay),
                "normal_improving": int(normal_improving),
            }
        )

    df = pd.DataFrame(rows)
    for lead in range(1, int(lookahead_buckets) + 1):
        df[f"future{lead}_d_abs_pitch_from_now_deg"] = np.nan
        df[f"future{lead}_d_abs_roll_from_now_deg"] = np.nan
        df[f"future{lead}_pitch_end_deg"] = np.nan
        df[f"future{lead}_roll_end_deg"] = np.nan
    for i, row in df.iterrows():
        for lead in range(1, int(lookahead_buckets) + 1):
            j = i + lead
            if j >= len(df):
                continue
            df.loc[i, f"future{lead}_d_abs_pitch_from_now_deg"] = (
                abs(float(df.loc[j, "pitch_end_deg"])) - abs(float(row["pitch_end_deg"]))
            )
            df.loc[i, f"future{lead}_d_abs_roll_from_now_deg"] = (
                abs(float(df.loc[j, "roll_end_deg"])) - abs(float(row["roll_end_deg"]))
            )
            df.loc[i, f"future{lead}_pitch_end_deg"] = float(df.loc[j, "pitch_end_deg"])
            df.loc[i, f"future{lead}_roll_end_deg"] = float(df.loc[j, "roll_end_deg"])

    delayed_flags: list[int] = []
    for _, row in df.iterrows():
        if not int(row["pitch_high"]) or int(row["normal_improving"]) or int(row["fallback_overlay"]):
            delayed_flags.append(0)
            continue
        delayed = False
        for lead in range(1, int(lookahead_buckets) + 1):
            val = row.get(f"future{lead}_d_abs_pitch_from_now_deg")
            if pd.notna(val) and float(val) <= -float(improve_eps_deg):
                delayed = True
                break
        delayed_flags.append(int(delayed))
    df["delayed_improving"] = delayed_flags

    def labels_for(row: pd.Series) -> str:
        labels = []
        for name in [
            "pitch_target_weak",
            "pitch_mass_weak",
            "pitch_authority_limited",
            "fallback_overlay",
            "delayed_improving",
            "normal_improving",
        ]:
            if int(row.get(name, 0)) > 0:
                labels.append(name)
        return ";".join(labels) if labels else "off_band_or_hold"

    df["labels"] = df.apply(labels_for, axis=1)
    high = df[df["pitch_high"] > 0].copy()
    if high.empty:
        summary = f"# Pitch Authority Audit\n\nNo pitch >= {pitch_high_deg:g} deg buckets found for `{case_name}`.\n"
        return df, summary

    counts = {
        "pitch_target_weak": int(high["pitch_target_weak"].sum()),
        "pitch_mass_weak": int(high["pitch_mass_weak"].sum()),
        "pitch_authority_limited": int(high["pitch_authority_limited"].sum()),
        "fallback_overlay": int(high["fallback_overlay"].sum()),
        "delayed_improving": int(high["delayed_improving"].sum()),
        "normal_improving": int(high["normal_improving"].sum()),
    }
    n_high = len(high)
    dominant = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    lines = [
        "# Pitch Authority Audit",
        "",
        f"Run case: `{case_name}`",
        f"High-pitch buckets audited: {n_high} (threshold |pitch| >= {pitch_high_deg:g} deg)",
        "",
        "## Label Counts",
        "",
    ]
    for name, count in dominant:
        lines.append(f"- {name}: {count}/{n_high} ({_fmt_pct(count / n_high)})")

    lines.extend(
        [
            "",
            "## Plain-Language Diagnosis",
            "",
        ]
    )
    if counts["fallback_overlay"] >= max(counts["pitch_target_weak"], counts["pitch_authority_limited"]):
        lines.append("- Fallback is a major overlay in the high-pitch segment, so some buckets cannot be attributed to target reuse alone.")
    if counts["pitch_target_weak"] > 0:
        lines.append("- There is direct evidence that planner active intent sometimes fails to create a new pitch-direction target; this supports keeping stall-refresh as a default-off candidate, not as a mainline fix.")
    if counts["pitch_authority_limited"] > 0:
        lines.append("- There is also evidence that tank mass moves while pitch still worsens; this points to pitch-direction authority/physical-response audit rather than more refresh tuning.")
    if counts["pitch_mass_weak"] > 0:
        lines.append("- Some buckets ask for pitch target motion but actual tank mass motion is weak; this would point to pump/rate execution.")
    if counts["normal_improving"] > 0 or counts["delayed_improving"] > 0:
        lines.append("- Some high-pitch buckets are improving immediately or with delay, so the whole 09 segment is not one uniform failure mode.")

    if counts["pitch_target_weak"] >= max(counts["fallback_overlay"], counts["pitch_authority_limited"]):
        main_bottleneck = "target-level pitch intent is weak or stale in most high-pitch buckets"
    elif counts["fallback_overlay"] >= counts["pitch_authority_limited"]:
        main_bottleneck = "fallback overlay is the largest high-pitch factor"
    else:
        main_bottleneck = "pitch physical authority is the largest high-pitch factor"

    if counts["pitch_authority_limited"] > 0:
        next_direction = (
            "action vector -> tank target projection -> actual mass projection -> pitch response, "
            "especially buckets with mass motion but worsening pitch"
        )
    elif counts["pitch_target_weak"] > 0:
        next_direction = (
            "target lifecycle / target projection first, because active pitch intent often does not "
            "become a new pitch-direction target"
        )
    else:
        next_direction = "fallback overlay and safety interaction first"

    lines.extend(
        [
            "",
            "## Answer",
            "",
            f"- Main bottleneck: {main_bottleneck}.",
            "- Stall-refresh evidence: partial. It can address stale target buckets, but it cannot explain fallback-overlay buckets and should remain default-off until checked on more cases.",
            f"- Next audit direction: {next_direction}.",
            "",
        ]
    )
    return df, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--case-filter", default="fr_relief_09")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bucket-s", type=float, default=600.0)
    parser.add_argument("--pitch-high-deg", type=float, default=5.0)
    parser.add_argument("--axis-ratio", type=float, default=1.5)
    parser.add_argument("--target-weak-kg", type=float, default=5000.0)
    parser.add_argument("--mass-weak-kg", type=float, default=5000.0)
    parser.add_argument("--fallback-threshold", type=float, default=0.20)
    parser.add_argument("--improve-eps-deg", type=float, default=0.10)
    parser.add_argument("--lookahead-buckets", type=int, default=3)
    args = parser.parse_args()

    case_name, log_path, ts_path = _case_files(args.run_dir, args.case_filter)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, summary = audit_case(
        case_name,
        log_path,
        ts_path,
        bucket_s=float(args.bucket_s),
        pitch_high_deg=float(args.pitch_high_deg),
        axis_ratio=float(args.axis_ratio),
        target_weak_kg=float(args.target_weak_kg),
        mass_weak_kg=float(args.mass_weak_kg),
        fallback_threshold=float(args.fallback_threshold),
        improve_eps_deg=float(args.improve_eps_deg),
        lookahead_buckets=int(args.lookahead_buckets),
    )
    csv_path = args.out_dir / "pitch_authority_bucket_audit.csv"
    md_path = args.out_dir / "pitch_authority_summary.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(summary, encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(summary)


if __name__ == "__main__":
    main()
