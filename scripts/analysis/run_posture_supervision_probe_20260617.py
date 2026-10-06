#!/usr/bin/env python3
"""Offline posture-supervision probe for prediction-primary outputs.

This script intentionally does not modify or invoke the shared controller.  It
reads paired closed_only / prediction_primary timeseries and applies a minimal
supervision overlay: prediction is allowed to reduce pump action only while the
measured posture has enough margin and the recent posture/wind trend is stable;
otherwise the overlay falls back to the closed feedback trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = (
    REPO_ROOT / "outputs" / "wind_prediction" / "positive_tilt_conservative_20260611"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "outputs"
    / "wind_prediction"
    / "posture_supervision_probe_20260617"
)


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    case_key: str
    timestamp: str
    label: str
    selector_stratum: str
    ranking_mean_d_axis_tilt_deg: float
    closed_path: Path
    primary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--ranking-csv",
        type=Path,
        default=(
            DEFAULT_SOURCE_ROOT
            / "actual_tilt_change_analysis_20260613"
            / "actual_tilt_change_per_case_original170.csv"
        ),
    )
    parser.add_argument(
        "--timeseries-dir",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "deadband12_existing101_6h" / "timeseries",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--near-axis-deg", type=float, default=4.5)
    parser.add_argument("--trend-axis-min-deg", type=float, default=0.6)
    parser.add_argument("--trend-window-s", type=float, default=180.0)
    parser.add_argument("--trend-increase-deg", type=float, default=0.02)
    parser.add_argument("--wind-window-s", type=float, default=600.0)
    parser.add_argument("--wind-speed-range-max-ms", type=float, default=2.0)
    parser.add_argument("--wind-dir-range-max-deg", type=float, default=20.0)
    parser.add_argument("--pump-reduction-eps-m3-min", type=float, default=0.05)
    parser.add_argument("--fallback-latch-s", type=float, default=600.0)
    parser.add_argument("--release-axis-deg", type=float, default=1.2)
    return parser.parse_args()


def timestamp_token(timestamp: str) -> str:
    return str(timestamp).replace(":", "").replace(" ", "_")


def find_timeseries_pair(
    timeseries_dir: Path,
    case_key: str,
    timestamp: str,
) -> tuple[Path, Path] | None:
    token = timestamp_token(timestamp)
    closed = sorted(timeseries_dir.glob(f"*{case_key}*{token}*closed_only_timeseries.csv"))
    primary = sorted(
        timeseries_dir.glob(f"*{case_key}*{token}*prediction_primary_econ_timeseries.csv")
    )
    if not closed or not primary:
        return None
    return closed[0], primary[0]


def select_cases(
    ranking_csv: Path,
    timeseries_dir: Path,
    sample_size: int,
) -> list[CaseFiles]:
    ranking = pd.read_csv(ranking_csv)
    ranking = ranking.sort_values("mean_d_axis_tilt_deg", ascending=False)
    selected: list[CaseFiles] = []
    for _, row in ranking.iterrows():
        case_key = str(row["case_key"])
        timestamp = str(row["timestamp"])
        pair = find_timeseries_pair(timeseries_dir, case_key, timestamp)
        if pair is None:
            continue
        selected.append(
            CaseFiles(
                case_id=str(row["case_id"]),
                case_key=case_key,
                timestamp=timestamp,
                label=str(row.get("label", "")),
                selector_stratum=str(row.get("selector_stratum", "")),
                ranking_mean_d_axis_tilt_deg=float(row["mean_d_axis_tilt_deg"]),
                closed_path=pair[0],
                primary_path=pair[1],
            )
        )
        if len(selected) >= int(sample_size):
            break
    if not selected:
        raise RuntimeError(
            "No cases with paired closed_only and prediction_primary timeseries were found."
        )
    return selected


def as_float(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df:
        return pd.Series(np.full(len(df), float(default)), index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(float(default))


def max_axis(df: pd.DataFrame) -> pd.Series:
    return np.maximum(as_float(df, "pitch_deg").abs(), as_float(df, "roll_deg").abs())


def total_tilt(df: pd.DataFrame) -> pd.Series:
    pitch = as_float(df, "pitch_deg")
    roll = as_float(df, "roll_deg")
    return np.sqrt(pitch * pitch + roll * roll)


def work_m3(rate: Iterable[float], dt_s: float) -> float:
    values = np.asarray(list(rate), dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.trapezoid(np.abs(values), dx=float(dt_s)) / 60.0)


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def pump_onoff_count(df: pd.DataFrame) -> int:
    total = 0
    pump_cols = [c for c in ("pump_rate1_m3min", "pump_rate2_m3min", "pump_rate3_m3min") if c in df]
    if not pump_cols:
        pump_cols = ["pump_total_rate_m3_min"]
    for col in pump_cols:
        on = as_float(df, col).abs().to_numpy(dtype=float) > 1e-6
        if on.size < 2:
            continue
        total += int(np.sum(on[1:] != on[:-1]))
    return total


def target_change_count(df: pd.DataFrame) -> int:
    target_cols = [
        c
        for c in ("target_tank1_kg", "target_tank2_kg", "target_tank3_kg")
        if c in df
    ]
    if not target_cols:
        return 0
    targets = df[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    changed = targets.diff().abs().max(axis=1).fillna(0.0) > 1.0
    return int(changed.sum())


def exposure_time(axis: pd.Series, threshold: float, dt_s: float) -> float:
    return float((axis.to_numpy(dtype=float) > float(threshold)).sum()) * float(dt_s)


def max_continuous_over(axis: pd.Series, threshold: float, dt_s: float) -> float:
    mask = axis.to_numpy(dtype=float) > float(threshold)
    best = 0
    cur = 0
    for value in mask:
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return float(best) * float(dt_s)


def summarize_variant(df: pd.DataFrame, dt_s: float) -> dict[str, float]:
    axis = max_axis(df)
    total = total_tilt(df)
    pump = as_float(df, "pump_total_rate_m3_min")
    out = {
        "pump_work_m3": work_m3(pump, dt_s),
        "mean_axis_tilt_deg": float(axis.mean()),
        "p95_axis_tilt_deg": percentile(axis, 95),
        "max_axis_tilt_deg": percentile(axis, 100),
        "mean_total_tilt_deg": float(total.mean()),
        "p95_total_tilt_deg": percentile(total, 95),
        "time_over_5_s": exposure_time(axis, 5.0, dt_s),
        "time_over_7p5_s": exposure_time(axis, 7.5, dt_s),
        "time_over_10_s": exposure_time(axis, 10.0, dt_s),
        "max_continuous_over_5_s": max_continuous_over(axis, 5.0, dt_s),
        "pump_onoff_count": float(pump_onoff_count(df)),
        "target_change_count": float(target_change_count(df)),
    }
    return out


def circular_rolling_range_deg(deg: pd.Series, window: int) -> pd.Series:
    radians = np.deg2rad(pd.to_numeric(deg, errors="coerce").fillna(0.0).to_numpy())
    unwrapped = np.rad2deg(np.unwrap(radians))
    s = pd.Series(unwrapped, index=deg.index)
    return s.rolling(window=window, min_periods=2).max() - s.rolling(
        window=window,
        min_periods=2,
    ).min()


def build_supervised_overlay(
    closed: pd.DataFrame,
    primary: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, float]]:
    n = min(len(closed), len(primary))
    closed = closed.iloc[:n].reset_index(drop=True)
    primary = primary.iloc[:n].reset_index(drop=True)
    dt_s = float(args.dt_s)
    trend_window = max(1, int(round(float(args.trend_window_s) / dt_s)))
    wind_window = max(2, int(round(float(args.wind_window_s) / dt_s)))
    latch_steps = max(0, int(round(float(args.fallback_latch_s) / dt_s)))

    primary_axis = max_axis(primary)
    primary_rate = as_float(primary, "pump_total_rate_m3_min").abs()
    closed_rate = as_float(closed, "pump_total_rate_m3_min").abs()
    pump_reducing = primary_rate < (closed_rate - float(args.pump_reduction_eps_m3_min))

    trend_delta = primary_axis - primary_axis.shift(trend_window).fillna(primary_axis.iloc[0])
    posture_near = primary_axis >= float(args.near_axis_deg)
    posture_worsening = (
        (primary_axis >= float(args.trend_axis_min_deg))
        & (trend_delta >= float(args.trend_increase_deg))
    )

    wind_speed_range = as_float(primary, "wind_speed").rolling(
        window=wind_window,
        min_periods=2,
    ).max() - as_float(primary, "wind_speed").rolling(window=wind_window, min_periods=2).min()
    wind_dir_range = circular_rolling_range_deg(as_float(primary, "wind_dir_deg"), wind_window)
    wind_uncertain = (
        (wind_speed_range.fillna(0.0) >= float(args.wind_speed_range_max_ms))
        | (wind_dir_range.fillna(0.0) >= float(args.wind_dir_range_max_deg))
    ) & (primary_axis >= float(args.trend_axis_min_deg))

    raw_fallback = pump_reducing & (posture_near | posture_worsening | wind_uncertain)
    fallback = np.zeros(n, dtype=bool)
    latch_remaining = 0
    reasons: list[str] = []
    for i in range(n):
        reason_bits = []
        if bool(pump_reducing.iloc[i]):
            if bool(posture_near.iloc[i]):
                reason_bits.append("near_threshold")
            if bool(posture_worsening.iloc[i]):
                reason_bits.append("posture_worsening")
            if bool(wind_uncertain.iloc[i]):
                reason_bits.append("wind_uncertain")
        if bool(raw_fallback.iloc[i]):
            latch_remaining = latch_steps
        release_ready = (
            primary_axis.iloc[i] <= float(args.release_axis_deg)
            and trend_delta.iloc[i] <= 0.0
            and not bool(wind_uncertain.iloc[i])
        )
        if latch_remaining > 0 and not release_ready:
            fallback[i] = True
            latch_remaining -= 1
        elif bool(raw_fallback.iloc[i]):
            fallback[i] = True
        else:
            fallback[i] = False
            if release_ready:
                latch_remaining = 0
        if fallback[i] and not reason_bits:
            reason_bits.append("latched_fallback")
        reasons.append("|".join(reason_bits) if fallback[i] else "prediction_allowed")

    probe = primary.copy()
    cols_to_overlay = [
        "pitch_deg",
        "roll_deg",
        "heave_m",
        "pump_rate1_m3min",
        "pump_rate2_m3min",
        "pump_rate3_m3min",
        "pump_total_rate_m3_min",
        "target_tank1_kg",
        "target_tank2_kg",
        "target_tank3_kg",
    ]
    for col in cols_to_overlay:
        if col in probe and col in closed:
            probe.loc[fallback, col] = closed.loc[fallback, col].to_numpy()
    probe["posture_supervisor_fallback"] = fallback.astype(int)
    probe["posture_supervisor_reason"] = reasons
    probe["posture_supervisor_primary_axis_deg"] = primary_axis
    probe["posture_supervisor_primary_axis_trend_deg"] = trend_delta
    probe["posture_supervisor_primary_pump_reducing"] = pump_reducing.astype(int)
    probe["posture_supervisor_wind_speed_range_ms"] = wind_speed_range.fillna(0.0)
    probe["posture_supervisor_wind_dir_range_deg"] = wind_dir_range.fillna(0.0)

    aux = {
        "fallback_active_ratio": float(np.mean(fallback)) if n else 0.0,
        "fallback_active_s": float(np.sum(fallback)) * dt_s,
        "raw_fallback_s": float(raw_fallback.sum()) * dt_s,
        "near_threshold_s": float((pump_reducing & posture_near).sum()) * dt_s,
        "posture_worsening_s": float((pump_reducing & posture_worsening).sum()) * dt_s,
        "wind_uncertain_s": float((pump_reducing & wind_uncertain).sum()) * dt_s,
        "pump_reducing_s": float(pump_reducing.sum()) * dt_s,
    }
    return probe, aux


def add_prefixed(row: dict[str, object], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def pct_saving(closed_pump: float, variant_pump: float) -> float:
    return (float(closed_pump) - float(variant_pump)) / max(float(closed_pump), 1e-9) * 100.0


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def aggregate_rows(rows: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for variant in ("closed", "primary", "probe"):
        out[f"{variant}_pump_work_m3"] = float(rows[f"{variant}_pump_work_m3"].sum())
        out[f"{variant}_mean_axis_tilt_deg"] = float(rows[f"{variant}_mean_axis_tilt_deg"].mean())
        out[f"{variant}_p95_axis_tilt_deg"] = float(rows[f"{variant}_p95_axis_tilt_deg"].mean())
        out[f"{variant}_time_over_5_s"] = float(rows[f"{variant}_time_over_5_s"].sum())
        out[f"{variant}_time_over_7p5_s"] = float(rows[f"{variant}_time_over_7p5_s"].sum())
        out[f"{variant}_time_over_10_s"] = float(rows[f"{variant}_time_over_10_s"].sum())
        out[f"{variant}_pump_onoff_count"] = float(rows[f"{variant}_pump_onoff_count"].sum())
    closed_pump = out["closed_pump_work_m3"]
    out["primary_pump_saving_pct"] = pct_saving(closed_pump, out["primary_pump_work_m3"])
    out["probe_pump_saving_pct"] = pct_saving(closed_pump, out["probe_pump_work_m3"])
    out["primary_d_mean_axis_deg"] = (
        out["primary_mean_axis_tilt_deg"] - out["closed_mean_axis_tilt_deg"]
    )
    out["probe_d_mean_axis_deg"] = (
        out["probe_mean_axis_tilt_deg"] - out["closed_mean_axis_tilt_deg"]
    )
    out["primary_d_p95_axis_deg"] = (
        out["primary_p95_axis_tilt_deg"] - out["closed_p95_axis_tilt_deg"]
    )
    out["probe_d_p95_axis_deg"] = (
        out["probe_p95_axis_tilt_deg"] - out["closed_p95_axis_tilt_deg"]
    )
    out["fallback_active_s"] = float(rows["probe_fallback_active_s"].sum())
    out["fallback_active_ratio_mean"] = float(rows["probe_fallback_active_ratio"].mean())
    return out


def build_report(rows: pd.DataFrame, aggregate: dict[str, float], args: argparse.Namespace) -> str:
    pump_loss_pp = aggregate["primary_pump_saving_pct"] - aggregate["probe_pump_saving_pct"]
    mean_axis_reduction = aggregate["primary_d_mean_axis_deg"] - aggregate["probe_d_mean_axis_deg"]
    p95_axis_reduction = aggregate["primary_d_p95_axis_deg"] - aggregate["probe_d_p95_axis_deg"]
    if mean_axis_reduction >= 0.10:
        conclusion = (
            "This minimal supervision overlay materially reduces the sampled mean attitude increase, "
            "but it pays for that reduction by giving back part of the pump saving."
        )
    elif p95_axis_reduction >= 0.10:
        conclusion = (
            "This minimal supervision overlay compresses the p95 attitude tail, but its reduction of "
            "the sampled mean attitude increase is still small."
        )
    elif mean_axis_reduction > 0.0:
        conclusion = (
            "This minimal supervision overlay only slightly reduces the sampled attitude increase under "
            "the current thresholds."
        )
    else:
        conclusion = (
            "This minimal supervision overlay did not reduce the sampled mean attitude increase under "
            "the current thresholds."
        )
    lines = [
        "# Posture Supervision Probe (2026-06-17)",
        "",
        "## Scope",
        "",
        "- Shared controller files were not modified.",
        "- Source windows: existing paired 6h `closed_only` and `prediction_primary_econ` timeseries from `positive_tilt_conservative_20260611/deadband12_existing101_6h`.",
        "- Selection: highest `mean_d_axis_tilt_deg` windows with complete 1Hz timeseries in `actual_tilt_change_per_case_original170.csv`.",
        "- Mechanism statement for paper language: prediction-assisted adjustment is constrained by closed-loop posture feedback; when posture evolution is adverse, the controller falls back to current feedback adjustment.",
        "",
        "## Existing Hooks Found",
        "",
        "- `src/wind_prediction/ballast_planner.py`: deadband, capacity/fullspeed guards, current-state hard gate, safety floor, and posture-state residual cost already exist.",
        "- `src/wind_prediction/safety_supervisor.py`: reactive-floor posture metrics and forecast-veto decisions already exist.",
        "- `scripts/analysis/run_prediction_primary_casebook.py`: default-off `--reactive-floor-predictive-veto`, `safe_pump_freeze`, and deadband/forecast-veto profiles exist. The deadband12 source run used here is a target-shape profile, so this probe keeps the test as an independent offline overlay rather than changing the shared profile.",
        "",
        "## Probe Rule",
        "",
        f"- Prediction pump reduction is supervised only when primary pump rate is at least `{args.pump_reduction_eps_m3_min:g} m3/min` below closed feedback.",
        f"- Fallback is triggered by max-axis posture near `{args.near_axis_deg:g} deg`, by a `{args.trend_increase_deg:g} deg` rise over `{args.trend_window_s:g}s` above `{args.trend_axis_min_deg:g} deg`, or by observed wind variability above `{args.wind_speed_range_max_ms:g} m/s` / `{args.wind_dir_range_max_deg:g} deg` over `{args.wind_window_s:g}s`.",
        f"- Fallback is latched up to `{args.fallback_latch_s:g}s` and released when posture is below `{args.release_axis_deg:g} deg`, not rising, and wind is stable.",
        "",
        "## Aggregate Result",
        "",
        "| metric | closed_only | current prediction_primary | posture supervision probe |",
        "| --- | ---: | ---: | ---: |",
        f"| pump work (m3) | {fmt(aggregate['closed_pump_work_m3'], 2)} | {fmt(aggregate['primary_pump_work_m3'], 2)} | {fmt(aggregate['probe_pump_work_m3'], 2)} |",
        f"| pump saving vs closed | 0.00% | {fmt(aggregate['primary_pump_saving_pct'], 2)}% | {fmt(aggregate['probe_pump_saving_pct'], 2)}% |",
        f"| mean max-axis attitude (deg) | {fmt(aggregate['closed_mean_axis_tilt_deg'])} | {fmt(aggregate['primary_mean_axis_tilt_deg'])} | {fmt(aggregate['probe_mean_axis_tilt_deg'])} |",
        f"| delta mean max-axis vs closed (deg) | 0.000 | {fmt(aggregate['primary_d_mean_axis_deg'])} | {fmt(aggregate['probe_d_mean_axis_deg'])} |",
        f"| mean per-case p95 max-axis (deg) | {fmt(aggregate['closed_p95_axis_tilt_deg'])} | {fmt(aggregate['primary_p95_axis_tilt_deg'])} | {fmt(aggregate['probe_p95_axis_tilt_deg'])} |",
        f"| delta p95 max-axis vs closed (deg) | 0.000 | {fmt(aggregate['primary_d_p95_axis_deg'])} | {fmt(aggregate['probe_d_p95_axis_deg'])} |",
        f"| T>5s | {fmt(aggregate['closed_time_over_5_s'], 0)} | {fmt(aggregate['primary_time_over_5_s'], 0)} | {fmt(aggregate['probe_time_over_5_s'], 0)} |",
        f"| T>7.5s | {fmt(aggregate['closed_time_over_7p5_s'], 0)} | {fmt(aggregate['primary_time_over_7p5_s'], 0)} | {fmt(aggregate['probe_time_over_7p5_s'], 0)} |",
        f"| T>10s | {fmt(aggregate['closed_time_over_10_s'], 0)} | {fmt(aggregate['primary_time_over_10_s'], 0)} | {fmt(aggregate['probe_time_over_10_s'], 0)} |",
        f"| pump on/off transitions | {fmt(aggregate['closed_pump_onoff_count'], 0)} | {fmt(aggregate['primary_pump_onoff_count'], 0)} | {fmt(aggregate['probe_pump_onoff_count'], 0)} |",
        "",
        "## Judgment",
        "",
        f"- {conclusion}",
        f"- Mean max-axis attitude delta is reduced by `{fmt(mean_axis_reduction)}` deg; p95 delta is reduced by `{fmt(p95_axis_reduction)}` deg.",
        f"- Pump-saving cost is `{fmt(pump_loss_pp, 2)}` percentage points versus current prediction_primary on this sample.",
        f"- Fallback occupies `{fmt(aggregate['fallback_active_s'], 0)}s` total, average `{fmt(aggregate['fallback_active_ratio_mean'] * 100.0, 1)}%` of each 6h window.",
        "",
        "## Case Table",
        "",
        "`probe_case_table.csv` contains per-case closed/current/probe metrics, fallback reasons, and pump-saving tradeoffs.",
        "",
        "| case | stratum | primary saving | probe saving | primary d_mean_axis | probe d_mean_axis | primary d_p95 | probe d_p95 | fallback % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in rows.iterrows():
        lines.append(
            f"| {r['case_id']} | {r['selector_stratum']} | "
            f"{float(r['primary_pump_saving_pct']):.1f}% | "
            f"{float(r['probe_pump_saving_pct']):.1f}% | "
            f"{float(r['primary_d_mean_axis_tilt_deg']):.3f} | "
            f"{float(r['probe_d_mean_axis_tilt_deg']):.3f} | "
            f"{float(r['primary_d_p95_axis_tilt_deg']):.3f} | "
            f"{float(r['probe_d_p95_axis_tilt_deg']):.3f} | "
            f"{float(r['probe_fallback_active_ratio']) * 100.0:.1f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    ts_out = out_dir / "timeseries"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_out.mkdir(parents=True, exist_ok=True)

    selected = select_cases(args.ranking_csv, args.timeseries_dir, args.sample_size)
    case_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for case in selected:
        closed = pd.read_csv(case.closed_path)
        primary = pd.read_csv(case.primary_path)
        n = min(len(closed), len(primary))
        closed = closed.iloc[:n].reset_index(drop=True)
        primary = primary.iloc[:n].reset_index(drop=True)
        probe, aux = build_supervised_overlay(closed, primary, args)

        closed_metrics = summarize_variant(closed, args.dt_s)
        primary_metrics = summarize_variant(primary, args.dt_s)
        probe_metrics = summarize_variant(probe, args.dt_s)
        row: dict[str, object] = {
            "case_id": case.case_id,
            "case_key": case.case_key,
            "timestamp": case.timestamp,
            "selector_stratum": case.selector_stratum,
            "ranking_mean_d_axis_tilt_deg": case.ranking_mean_d_axis_tilt_deg,
            "label": case.label,
            "closed_timeseries": str(case.closed_path),
            "primary_timeseries": str(case.primary_path),
        }
        add_prefixed(row, "closed", closed_metrics)
        add_prefixed(row, "primary", primary_metrics)
        add_prefixed(row, "probe", probe_metrics)
        for key, value in aux.items():
            row[f"probe_{key}"] = value
        row["primary_pump_saving_pct"] = pct_saving(
            closed_metrics["pump_work_m3"],
            primary_metrics["pump_work_m3"],
        )
        row["probe_pump_saving_pct"] = pct_saving(
            closed_metrics["pump_work_m3"],
            probe_metrics["pump_work_m3"],
        )
        row["probe_pump_saving_loss_pp"] = (
            float(row["primary_pump_saving_pct"]) - float(row["probe_pump_saving_pct"])
        )
        row["primary_d_mean_axis_tilt_deg"] = (
            primary_metrics["mean_axis_tilt_deg"] - closed_metrics["mean_axis_tilt_deg"]
        )
        row["probe_d_mean_axis_tilt_deg"] = (
            probe_metrics["mean_axis_tilt_deg"] - closed_metrics["mean_axis_tilt_deg"]
        )
        row["primary_d_p95_axis_tilt_deg"] = (
            primary_metrics["p95_axis_tilt_deg"] - closed_metrics["p95_axis_tilt_deg"]
        )
        row["probe_d_p95_axis_tilt_deg"] = (
            probe_metrics["p95_axis_tilt_deg"] - closed_metrics["p95_axis_tilt_deg"]
        )
        row["probe_mean_axis_delta_reduction_deg"] = (
            float(row["primary_d_mean_axis_tilt_deg"])
            - float(row["probe_d_mean_axis_tilt_deg"])
        )
        row["probe_p95_axis_delta_reduction_deg"] = (
            float(row["primary_d_p95_axis_tilt_deg"])
            - float(row["probe_d_p95_axis_tilt_deg"])
        )
        case_rows.append(row)
        selection_rows.append(
            {
                "case_id": case.case_id,
                "case_key": case.case_key,
                "timestamp": case.timestamp,
                "selector_stratum": case.selector_stratum,
                "ranking_mean_d_axis_tilt_deg": case.ranking_mean_d_axis_tilt_deg,
                "closed_timeseries": str(case.closed_path),
                "primary_timeseries": str(case.primary_path),
            }
        )

        keep_cols = [
            c
            for c in [
                "t_s",
                "wind_speed",
                "wind_dir_deg",
                "pitch_deg",
                "roll_deg",
                "pump_rate1_m3min",
                "pump_rate2_m3min",
                "pump_rate3_m3min",
                "pump_total_rate_m3_min",
                "posture_supervisor_fallback",
                "posture_supervisor_reason",
                "posture_supervisor_primary_axis_deg",
                "posture_supervisor_primary_axis_trend_deg",
                "posture_supervisor_primary_pump_reducing",
                "posture_supervisor_wind_speed_range_ms",
                "posture_supervisor_wind_dir_range_deg",
            ]
            if c in probe.columns
        ]
        write_csv(
            ts_out / f"{case.case_id}_{timestamp_token(case.timestamp)}_posture_supervision_probe_timeseries.csv",
            probe[keep_cols],
        )

    rows_df = pd.DataFrame(case_rows)
    write_csv(out_dir / "probe_case_table.csv", rows_df)
    write_csv(out_dir / "probe_case_selection.csv", pd.DataFrame(selection_rows))
    aggregate = aggregate_rows(rows_df)
    report = build_report(rows_df, aggregate, args)
    (out_dir / "probe_summary.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_dir / 'probe_summary.md'}")
    print(f"Wrote {out_dir / 'probe_case_table.csv'}")


if __name__ == "__main__":
    main()
