#!/usr/bin/env python3
"""Read-only diagnostics for relief-envelope pump-saving headroom.

Inputs are existing locked-holdout logs/timeseries. The script does not run the
controller and does not modify controller code. It answers:

* which relief-envelope veto/exit reasons dominate;
* whether current delay episodes are cap-limited;
* how much net pump is addressable in safe-relief non-floor refresh windows;
* whether axis-shape vetoes include a safe unlock subset.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKED_DIR = (
    REPO_ROOT
    / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
)
OUT_DIR = REPO_ROOT / "outputs/wind_prediction/relief_envelope_saving_ceiling_diagnostics_v1"
A0_DIR = LOCKED_DIR / "runs/A0_learned_v16"
A1_DIR = LOCKED_DIR / "runs/A1_learned_near_envelope_medium_axis_guard"


def _dirs() -> dict[str, Path]:
    dirs = {"out": OUT_DIR, "raw": OUT_DIR / "raw_tables", "paper": OUT_DIR / "paper_ready"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _case_key(path: Path) -> str:
    suffixes = (
        "_prediction_primary_econ_planner_log.csv",
        "_prediction_primary_econ_timeseries.csv",
    )
    for suffix in suffixes:
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return path.stem


def _timestamp_from_case(case: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{6})", case)
    if not match:
        return ""
    hhmmss = match.group(2)
    return f"{match.group(1)} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"


def _metadata() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    path = LOCKED_DIR / "locked_casebook_manifest.csv"
    if not path.exists():
        return {}, {}
    df = pd.read_csv(path)
    return (
        df.set_index("case_id").to_dict(orient="index"),
        df.set_index(df["timestamp"].astype(str)).to_dict(orient="index"),
    )


def _read_log(run_dir: Path, case: str) -> pd.DataFrame:
    path = run_dir / "planner_logs" / f"{case}_prediction_primary_econ_planner_log.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _read_ts(run_dir: Path, case: str) -> pd.DataFrame:
    path = run_dir / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def _last(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if df.empty or col not in df.columns:
        return default
    s = df[col].fillna(default).astype(float)
    return float(s.iloc[-1]) if len(s) else default


def _pump_integral(ts: pd.DataFrame, start_s: float, end_s: float) -> float:
    if ts.empty or "pump_total_rate_m3_min" not in ts.columns:
        return 0.0
    w = ts[(ts["t_s"] >= start_s) & (ts["t_s"] < end_s)]
    return float(w["pump_total_rate_m3_min"].fillna(0).abs().sum() / 60.0)


def _veto_reason_rows(log: pd.DataFrame) -> pd.DataFrame:
    if "relief_envelope_reason" not in log.columns:
        return pd.DataFrame(columns=["reason", "rows"])
    return (
        log["relief_envelope_reason"]
        .fillna("")
        .astype(str)
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="rows")
    )


def _per_case_counter_delta(log: pd.DataFrame, counter: str) -> int:
    return int(round(_last(log, counter, 0.0)))


def _target_axis_ratio(row: pd.Series) -> tuple[float, float, float]:
    pitch = abs(float(row.get("target_pitch_deg", 0.0)))
    roll = abs(float(row.get("target_roll_deg", 0.0)))
    major = max(pitch, roll)
    minor = min(pitch, roll)
    ratio = minor / major if major > 1e-9 else 0.0
    return minor, major, ratio


def _dominant_axis(row: pd.Series) -> str:
    pitch = abs(float(row.get("current_pitch_deg", 0.0)))
    roll = abs(float(row.get("current_roll_deg", 0.0)))
    if max(pitch, roll) <= 1e-9:
        return "none"
    return "pitch" if pitch >= roll else "roll"


def _safe_relief_refresh_rows(log: pd.DataFrame) -> pd.DataFrame:
    if log.empty:
        return pd.DataFrame()
    df = log.copy()
    required = [
        "current_time_s",
        "prediction_primary_event_reset",
        "reactive_floor_active",
        "reactive_floor_posture_metric_deg",
        "active_effectiveness_current_response_deg",
        "far_horizon_near_max",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = 0.0
    # A read-only approximation of the fixed medium A1 eligibility. This
    # intentionally estimates ceiling/headroom; the exact controller gate remains
    # authoritative in reruns.
    refresh = df["prediction_primary_event_reset"].fillna(0).astype(float) > 0.5
    floor = df["reactive_floor_active"].fillna(0).astype(float) > 0.5
    posture = df["reactive_floor_posture_metric_deg"].fillna(0).astype(float)
    resp = df["active_effectiveness_current_response_deg"].fillna(0).astype(float)
    near_relief = df["far_horizon_near_max"].fillna(999).astype(float) < 0.9
    in_band = (posture >= 3.0) & (posture < 5.0)
    not_worsening = resp >= -0.05
    out = df[refresh & ~floor & in_band & not_worsening & near_relief].copy()
    return out


def _collect() -> dict[str, pd.DataFrame]:
    by_case, by_ts = _metadata()
    reason_frames: list[pd.DataFrame] = []
    counters: list[dict[str, Any]] = []
    delay_rows: list[dict[str, Any]] = []
    ceiling_rows: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []

    for log_path in sorted((A1_DIR / "planner_logs").glob("*_planner_log.csv")):
        case = _case_key(log_path)
        timestamp = _timestamp_from_case(case)
        meta = by_case.get(case, {}) or by_ts.get(timestamp, {})
        log = pd.read_csv(log_path, low_memory=False)
        a0_log = _read_log(A0_DIR, case)
        a0_ts = _read_ts(A0_DIR, case)
        a1_ts = _read_ts(A1_DIR, case)

        reasons = _veto_reason_rows(log)
        reasons.insert(0, "case", case)
        reasons.insert(1, "case_source", meta.get("case_source", ""))
        reason_frames.append(reasons)

        counters.append(
            {
                "case": case,
                "case_source": meta.get("case_source", ""),
                "source_groups": meta.get("source_groups", ""),
                "eligible_count": _per_case_counter_delta(log, "relief_envelope_eligible_count"),
                "pump_opportunity_count": _per_case_counter_delta(
                    log, "relief_envelope_pump_opportunity_count"
                ),
                "target_refresh_delayed_count": _per_case_counter_delta(
                    log, "relief_envelope_target_refresh_delayed_count"
                ),
                "axis_shape_veto_count": _per_case_counter_delta(
                    log, "relief_envelope_axis_shape_veto_count"
                ),
                "debt_exit_count": _per_case_counter_delta(log, "relief_envelope_debt_exit_count"),
                "reintensification_veto_count": _per_case_counter_delta(
                    log, "relief_envelope_reintensification_veto_count"
                ),
                "direction_mismatch_veto_count": _per_case_counter_delta(
                    log, "relief_envelope_direction_mismatch_veto_count"
                ),
                "posture_worsening_exit_count": _per_case_counter_delta(
                    log, "relief_envelope_posture_worsening_exit_count"
                ),
            }
        )

        if "relief_envelope_target_refresh_delayed_count" in log.columns:
            delayed = (
                log["relief_envelope_target_refresh_delayed_count"]
                .fillna(0)
                .astype(float)
                .diff()
                .fillna(log["relief_envelope_target_refresh_delayed_count"].fillna(0).astype(float))
                > 0
            )
        else:
            delayed = pd.Series(False, index=log.index)
        for _, row in log[delayed].iterrows():
            t = float(row.get("current_time_s", 0.0))
            delay_rows.append(
                {
                    "case": case,
                    "case_source": meta.get("case_source", ""),
                    "source_groups": meta.get("source_groups", ""),
                    "current_time_s": t,
                    "reason": row.get("relief_envelope_reason", ""),
                    "duration_s": row.get("relief_envelope_current_duration_s", 0.0),
                    "safety_debt_used": row.get("relief_envelope_safety_debt_used", 0.0),
                    "near_limit_dwell_s": row.get("relief_envelope_near_limit_dwell_s", 0.0),
                    "posture_metric_deg": row.get("reactive_floor_posture_metric_deg", 0.0),
                    "target_err_mean_kg": row.get("active_effectiveness_target_err_mean_kg", 0.0),
                    "target_axis_minor_deg": _target_axis_ratio(row)[0],
                    "target_axis_major_deg": _target_axis_ratio(row)[1],
                    "target_axis_ratio": _target_axis_ratio(row)[2],
                    "pump_delta_600s_m3": _pump_integral(a1_ts, t, t + 600.0)
                    - _pump_integral(a0_ts, t, t + 600.0),
                    "pump_delta_1800s_m3": _pump_integral(a1_ts, t, t + 1800.0)
                    - _pump_integral(a0_ts, t, t + 1800.0),
                    "pump_delta_7200s_m3": _pump_integral(a1_ts, t, 7200.0)
                    - _pump_integral(a0_ts, t, 7200.0),
                }
            )

        safe_rows = _safe_relief_refresh_rows(a0_log)
        for _, row in safe_rows.iterrows():
            t = float(row.get("current_time_s", 0.0))
            minor, major, ratio = _target_axis_ratio(row)
            ceiling_rows.append(
                {
                    "case": case,
                    "case_source": meta.get("case_source", ""),
                    "source_groups": meta.get("source_groups", ""),
                    "current_time_s": t,
                    "branch_event_reset": int(row.get("prediction_primary_event_reset", 0)),
                    "branch_sustained_recompute": int(
                        row.get("sustained_active_recompute_active", 0)
                    ),
                    "posture_metric_deg": float(row.get("reactive_floor_posture_metric_deg", 0.0)),
                    "response_deg": float(
                        row.get("active_effectiveness_current_response_deg", 0.0)
                    ),
                    "near_max": float(row.get("far_horizon_near_max", 0.0)),
                    "target_axis_minor_deg": minor,
                    "target_axis_major_deg": major,
                    "target_axis_ratio": ratio,
                    "dominant_axis": _dominant_axis(row),
                    "pump_after_600s_m3_A0": _pump_integral(a0_ts, t, t + 600.0),
                    "pump_after_1800s_m3_A0": _pump_integral(a0_ts, t, t + 1800.0),
                    "pump_after_7200s_m3_A0": _pump_integral(a0_ts, t, 7200.0),
                }
            )

        if "relief_envelope_axis_shape_veto_count" in log.columns:
            veto = (
                log["relief_envelope_axis_shape_veto_count"]
                .fillna(0)
                .astype(float)
                .diff()
                .fillna(log["relief_envelope_axis_shape_veto_count"].fillna(0).astype(float))
                > 0
            )
        else:
            veto = pd.Series(False, index=log.index)
        for _, row in log[veto].iterrows():
            minor, major, ratio = _target_axis_ratio(row)
            near_norms = [
                float(row.get("raw_pressure_block0_norm", 0.0)),
                float(row.get("raw_pressure_block1_norm", 0.0)),
                float(row.get("raw_pressure_block2_norm", 0.0)),
            ]
            # Without component vectors in the planner log, this is a conservative
            # proxy for safe unlock: all near blocks stay low and the final near
            # block is not higher than the first by more than a small tolerance.
            near_all_low = max(near_norms) < 0.5
            near_not_rising = near_norms[2] <= near_norms[0] + 0.05
            axis_rows.append(
                {
                    "case": case,
                    "case_source": meta.get("case_source", ""),
                    "source_groups": meta.get("source_groups", ""),
                    "current_time_s": float(row.get("current_time_s", 0.0)),
                    "posture_metric_deg": float(row.get("reactive_floor_posture_metric_deg", 0.0)),
                    "target_axis_minor_deg": minor,
                    "target_axis_major_deg": major,
                    "target_axis_ratio": ratio,
                    "raw_pressure_block0_norm": near_norms[0],
                    "raw_pressure_block1_norm": near_norms[1],
                    "raw_pressure_block2_norm": near_norms[2],
                    "far_horizon_near_max": float(row.get("far_horizon_near_max", 0.0)),
                    "far_horizon_far_min": float(row.get("far_horizon_far_min", 0.0)),
                    "far_horizon_far_max": float(row.get("far_horizon_far_max", 0.0)),
                    "safe_unlock_proxy": int(near_all_low and near_not_rising),
                    "pump_after_600s_m3_A0": _pump_integral(
                        a0_ts, float(row.get("current_time_s", 0.0)), float(row.get("current_time_s", 0.0)) + 600.0
                    ),
                }
            )

    reason_table = pd.concat(reason_frames, ignore_index=True) if reason_frames else pd.DataFrame()
    return {
        "reason_rows": reason_table,
        "case_counters": pd.DataFrame(counters),
        "delay_episodes": pd.DataFrame(delay_rows),
        "addressable_windows": pd.DataFrame(ceiling_rows),
        "axis_veto_coverage": pd.DataFrame(axis_rows),
    }


def _write_outputs(tables: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> None:
    for name, df in tables.items():
        df.to_csv(dirs["raw"] / f"{name}.csv", index=False)
        df.to_csv(dirs["out"] / f"{name}.csv", index=False)

    counters = tables["case_counters"]
    delays = tables["delay_episodes"]
    windows = tables["addressable_windows"]
    axis = tables["axis_veto_coverage"]
    reason = tables["reason_rows"]
    reason_summary = (
        reason.groupby("reason", as_index=False)["rows"].sum().sort_values("rows", ascending=False)
        if not reason.empty
        else pd.DataFrame()
    )
    reason_summary.to_csv(dirs["out"] / "veto_reason_histogram.csv", index=False)
    reason_summary.to_csv(dirs["raw"] / "veto_reason_histogram.csv", index=False)

    total_addressable_600 = float(windows["pump_after_600s_m3_A0"].sum()) if not windows.empty else 0.0
    total_addressable_1800 = float(windows["pump_after_1800s_m3_A0"].sum()) if not windows.empty else 0.0
    axis_safe_unlock = int(axis["safe_unlock_proxy"].sum()) if not axis.empty else 0
    axis_safe_unlock_pump_600 = (
        float(axis.loc[axis["safe_unlock_proxy"].astype(bool), "pump_after_600s_m3_A0"].sum())
        if not axis.empty
        else 0.0
    )
    axis_safe_unlock_pump_rows = (
        int((axis.loc[axis["safe_unlock_proxy"].astype(bool), "pump_after_600s_m3_A0"] > 1.0).sum())
        if not axis.empty
        else 0
    )
    axis_veto_total = len(axis)
    delayed_count = len(delays)
    cap_binding = 0
    if not delays.empty:
        cap_binding = int(
            (
                (delays["duration_s"].astype(float) >= 900.0)
                | (delays["safety_debt_used"].astype(float) >= 600.0)
                | (delays["near_limit_dwell_s"].astype(float) >= 300.0)
            ).sum()
        )
    current_saved = 875.9384816824422
    ceiling_ratio = current_saved / total_addressable_600 if total_addressable_600 > 1e-9 else 0.0

    summary = pd.DataFrame(
        [
            {
                "metric": "current_locked_saved_m3",
                "value": current_saved,
                "interpretation": "A1 + single-axis guard net saving vs A0",
            },
            {
                "metric": "read_only_addressable_600s_pump_m3",
                "value": total_addressable_600,
                "interpretation": "gross A0 pump after safe-relief non-floor refresh windows",
            },
            {
                "metric": "read_only_addressable_1800s_pump_m3",
                "value": total_addressable_1800,
                "interpretation": "gross A0 pump over 30min after addressable windows",
            },
            {
                "metric": "current_saved_div_addressable_600s",
                "value": ceiling_ratio,
                "interpretation": "rough fraction of gross 600s ceiling already captured",
            },
            {
                "metric": "delay_episode_count",
                "value": delayed_count,
                "interpretation": "actual target refresh delays under current guard",
            },
            {
                "metric": "cap_binding_delay_count",
                "value": cap_binding,
                "interpretation": "delays at duration/debt/near-limit caps",
            },
            {
                "metric": "axis_shape_veto_rows",
                "value": axis_veto_total,
                "interpretation": "mixed-axis veto events under current guard",
            },
            {
                "metric": "axis_veto_safe_unlock_proxy_rows",
                "value": axis_safe_unlock,
                "interpretation": "veto rows whose near pressure stayed low and non-rising",
            },
            {
                "metric": "axis_veto_safe_unlock_proxy_pump_600s_m3",
                "value": axis_safe_unlock_pump_600,
                "interpretation": "gross A0 pump after safe-unlock proxy veto rows",
            },
            {
                "metric": "axis_veto_safe_unlock_proxy_rows_with_pump",
                "value": axis_safe_unlock_pump_rows,
                "interpretation": "safe-unlock proxy veto rows with >1 m3 A0 pump in next 600s",
            },
        ]
    )
    summary.to_csv(dirs["out"] / "saving_ceiling_summary.csv", index=False)
    summary.to_csv(dirs["raw"] / "saving_ceiling_summary.csv", index=False)

    recommendation = "measure_only"
    if total_addressable_600 > current_saved * 1.5 and axis_safe_unlock >= 10:
        recommendation = "consider_relief_axis_coverage_guard"
    elif cap_binding > 0:
        recommendation = "consider_small_cap_extension"
    else:
        recommendation = "lock_current_result"

    md = f"""# Relief Envelope Saving Ceiling Diagnostics

## Purpose

Read-only diagnostics before tuning the learned 0-60min relief-economy layer.
The goal is to avoid blind gate changes: first estimate the remaining pump-saving
ceiling and the size of any safe mixed-axis unlock subset.

## Summary

| metric | value |
| --- | ---: |
| current locked saving | {current_saved:.2f} m3 |
| gross addressable 600s pump | {total_addressable_600:.2f} m3 |
| gross addressable 1800s pump | {total_addressable_1800:.2f} m3 |
| current saved / gross 600s ceiling | {ceiling_ratio:.2%} |
| actual delay episodes | {delayed_count} |
| cap-binding delay episodes | {cap_binding} |
| axis-shape veto rows | {axis_veto_total} |
| safe-unlock proxy rows among axis vetoes | {axis_safe_unlock} |
| safe-unlock proxy 600s pump | {axis_safe_unlock_pump_600:.2f} m3 |
| safe-unlock proxy rows with pump | {axis_safe_unlock_pump_rows} |

## Veto / Exit Reason Histogram

{_md_table(reason_summary.head(20))}

## Interpretation

Recommendation: **{recommendation}**.

- If the gross addressable ceiling is only modestly above the current `875.94 m3`,
  aggressive tuning is unlikely to pay off.
- If many axis-shape vetoes have low, non-rising near pressure, the next practical
  lever is a relief-axis-coverage guard, not looser safety gates.
- If current delays were not cap-bound, increasing duration/debt budgets will not
  add much.

These diagnostics are approximate ceiling estimates, not a replacement for a
closed-loop rerun. They are meant to decide whether a rerun is worth doing.
"""
    for path in (dirs["out"] / "saving_ceiling_diagnostics.md", dirs["paper"] / "saving_ceiling_diagnostics.md"):
        path.write_text(md, encoding="utf-8")


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            if abs(value) < 1.0:
                return f"{value:.3f}"
            return f"{value:.2f}"
        return str(value)

    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    dirs = _dirs()
    tables = _collect()
    _write_outputs(tables, dirs)
    print(f"[done] {OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
