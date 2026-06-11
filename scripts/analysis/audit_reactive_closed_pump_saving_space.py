#!/usr/bin/env python3
"""Audit no-preview reactive pump-saving headroom from closed-baseline traces.

This is a conservative proxy audit, not a counterfactual plant simulation.  It
answers a narrow question before tuning a no-preview economy baseline:

    Do the frozen closed-baseline traces contain many pump starts that occur
    while the platform is already in a low-risk attitude envelope?

If the answer is no, actuator-only retuning is unlikely to create a strong
traditional no-preview economy baseline.  If the answer is yes, a strictly
bounded actuator-level Pareto scan is worth running.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = (
    REPO_ROOT
    / "outputs"
    / "wind_prediction"
    / "prediction_primary_baseline_v1_10case_2h_naive"
    / "timeseries"
)
OUT_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "reactive_closed_pump_saving_space_v1"
DT = 1.0

LOW_PITCH_DEG = 1.2
LOW_ROLL_DEG = 1.0
FOLLOWUP_S = (300, 600)
PUMP_ON_EPS = 1e-6


def work_m3(pump: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(pump, dtype=float)), dx=DT) / 60.0)


def _case_id_from_name(path: Path) -> str:
    stem = path.name.replace("_closed_only_timeseries.csv", "")
    # timestamp is always the final YYYY-MM-DD_HHMMSS section.
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if len(part) == 10 and part.count("-") == 2:
            return "_".join(parts[:i])
    return stem


def _window_max_abs(series: pd.Series, start_idx: int, horizon_s: int) -> float:
    end_idx = min(len(series), start_idx + int(round(horizon_s / DT)) + 1)
    return float(np.nanmax(np.abs(series.iloc[start_idx:end_idx].to_numpy(dtype=float))))


def _event_work(pump: np.ndarray, start_idx: int, on: np.ndarray) -> tuple[float, int]:
    end_idx = start_idx + 1
    while end_idx < len(on) and bool(on[end_idx]):
        end_idx += 1
    return work_m3(pump[start_idx:end_idx]), end_idx


def audit_trace(path: Path) -> tuple[list[dict], dict]:
    df = pd.read_csv(path)
    case_id = _case_id_from_name(path)
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    on = np.abs(pump) > PUMP_ON_EPS
    starts = np.flatnonzero(on & np.r_[True, ~on[:-1]])
    total_work = work_m3(pump)
    pitch = df["pitch_deg"].astype(float)
    roll = df["roll_deg"].astype(float)

    rows: list[dict] = []
    for event_idx, start_idx in enumerate(starts, start=1):
        event_work, end_idx = _event_work(pump, int(start_idx), on)
        pitch_now = abs(float(pitch.iloc[start_idx]))
        roll_now = abs(float(roll.iloc[start_idx]))
        pitch_max_5 = _window_max_abs(pitch, int(start_idx), 300)
        roll_max_5 = _window_max_abs(roll, int(start_idx), 300)
        pitch_max_10 = _window_max_abs(pitch, int(start_idx), 600)
        roll_max_10 = _window_max_abs(roll, int(start_idx), 600)
        low_now = pitch_now <= LOW_PITCH_DEG and roll_now <= LOW_ROLL_DEG
        low_5 = pitch_max_5 <= LOW_PITCH_DEG and roll_max_5 <= LOW_ROLL_DEG
        low_10 = pitch_max_10 <= LOW_PITCH_DEG and roll_max_10 <= LOW_ROLL_DEG
        # Conservative proxy: only call a start "low-risk avoidable" if the
        # attitude is already low at start and stays low for at least 5 minutes.
        avoidable_5 = bool(low_now and low_5)
        avoidable_10 = bool(low_now and low_10)
        rows.append(
            {
                "case_id": case_id,
                "event_index": event_idx,
                "start_time_s": float(df["t_s"].iloc[start_idx]),
                "duration_s": float((end_idx - int(start_idx)) * DT),
                "event_work_m3": event_work,
                "event_work_pct_of_case": event_work / max(total_work, 1e-9) * 100.0,
                "pitch_abs_start_deg": pitch_now,
                "roll_abs_start_deg": roll_now,
                "pitch_abs_max_5min_deg": pitch_max_5,
                "roll_abs_max_5min_deg": roll_max_5,
                "pitch_abs_max_10min_deg": pitch_max_10,
                "roll_abs_max_10min_deg": roll_max_10,
                "low_risk_now": int(low_now),
                "low_risk_5min": int(low_5),
                "low_risk_10min": int(low_10),
                "avoidable_proxy_5min": int(avoidable_5),
                "avoidable_proxy_10min": int(avoidable_10),
            }
        )

    event_df = pd.DataFrame(rows)
    if event_df.empty:
        summary = {
            "case_id": case_id,
            "pump_work_m3": total_work,
            "pump_start_events": 0,
            "avoidable_proxy_5min_events": 0,
            "avoidable_proxy_10min_events": 0,
            "avoidable_proxy_5min_work_m3": 0.0,
            "avoidable_proxy_10min_work_m3": 0.0,
            "avoidable_proxy_5min_work_pct": 0.0,
            "avoidable_proxy_10min_work_pct": 0.0,
            "pitch_abs_p95": float(np.percentile(np.abs(pitch), 95)),
            "roll_abs_p95": float(np.percentile(np.abs(roll), 95)),
        }
    else:
        avoid5 = event_df["avoidable_proxy_5min"].astype(bool)
        avoid10 = event_df["avoidable_proxy_10min"].astype(bool)
        work5 = float(event_df.loc[avoid5, "event_work_m3"].sum())
        work10 = float(event_df.loc[avoid10, "event_work_m3"].sum())
        summary = {
            "case_id": case_id,
            "pump_work_m3": total_work,
            "pump_start_events": int(len(event_df)),
            "avoidable_proxy_5min_events": int(avoid5.sum()),
            "avoidable_proxy_10min_events": int(avoid10.sum()),
            "avoidable_proxy_5min_work_m3": work5,
            "avoidable_proxy_10min_work_m3": work10,
            "avoidable_proxy_5min_work_pct": work5 / max(total_work, 1e-9) * 100.0,
            "avoidable_proxy_10min_work_pct": work10 / max(total_work, 1e-9) * 100.0,
            "pitch_abs_p95": float(np.percentile(np.abs(pitch), 95)),
            "roll_abs_p95": float(np.percentile(np.abs(roll), 95)),
        }
    return rows, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(TRACE_DIR.glob("*_closed_only_timeseries.csv"))
    if not paths:
        raise FileNotFoundError(f"no closed traces in {TRACE_DIR}")

    events: list[dict] = []
    summaries: list[dict] = []
    for path in paths:
        event_rows, summary = audit_trace(path)
        events.extend(event_rows)
        summaries.append(summary)

    event_df = pd.DataFrame(events)
    summary_df = pd.DataFrame(summaries).sort_values("case_id")
    event_df.to_csv(OUT_DIR / "reactive_pump_start_events.csv", index=False)
    summary_df.to_csv(OUT_DIR / "reactive_pump_saving_space_by_case.csv", index=False)

    total_work = float(summary_df["pump_work_m3"].sum())
    work5 = float(summary_df["avoidable_proxy_5min_work_m3"].sum())
    work10 = float(summary_df["avoidable_proxy_10min_work_m3"].sum())
    total_starts = int(summary_df["pump_start_events"].sum())
    starts5 = int(summary_df["avoidable_proxy_5min_events"].sum())
    starts10 = int(summary_df["avoidable_proxy_10min_events"].sum())
    aggregate = {
        "cases": int(len(summary_df)),
        "total_pump_work_m3": total_work,
        "total_pump_start_events": total_starts,
        "avoidable_proxy_5min_events": starts5,
        "avoidable_proxy_10min_events": starts10,
        "avoidable_proxy_5min_event_pct": starts5 / max(total_starts, 1) * 100.0,
        "avoidable_proxy_10min_event_pct": starts10 / max(total_starts, 1) * 100.0,
        "avoidable_proxy_5min_work_m3": work5,
        "avoidable_proxy_10min_work_m3": work10,
        "avoidable_proxy_5min_work_pct": work5 / max(total_work, 1e-9) * 100.0,
        "avoidable_proxy_10min_work_pct": work10 / max(total_work, 1e-9) * 100.0,
    }
    pd.DataFrame([aggregate]).to_csv(OUT_DIR / "reactive_pump_saving_space_aggregate.csv", index=False)

    verdict = "worth_actuator_pareto" if aggregate["avoidable_proxy_5min_work_pct"] > 10.0 else "low_reactive_headroom"
    report = [
        "# Reactive Closed Pump-saving Space Audit",
        "",
        "This audit is a conservative proxy, not a counterfactual plant replay.",
        "",
        "## Thresholds",
        "",
        f"- low-risk pitch envelope: `{LOW_PITCH_DEG:.2f} deg`",
        f"- low-risk roll envelope: `{LOW_ROLL_DEG:.2f} deg`",
        "- avoidable proxy: pump starts while low-risk now and remains low-risk for 5 or 10 min",
        "",
        "## Aggregate",
        "",
        f"- total pump work: `{total_work:.2f} m3`",
        f"- pump start events: `{total_starts}`",
        f"- 5 min low-risk proxy events: `{starts5}` ({aggregate['avoidable_proxy_5min_event_pct']:.1f}%)",
        f"- 5 min low-risk proxy work: `{work5:.2f} m3` ({aggregate['avoidable_proxy_5min_work_pct']:.1f}%)",
        f"- 10 min low-risk proxy events: `{starts10}` ({aggregate['avoidable_proxy_10min_event_pct']:.1f}%)",
        f"- 10 min low-risk proxy work: `{work10:.2f} m3` ({aggregate['avoidable_proxy_10min_work_pct']:.1f}%)",
        f"- verdict: `{verdict}`",
        "",
        "## Case Summary",
        "",
        "| case | pump work | starts | 5 min proxy work | 10 min proxy work | pitch p95 | roll p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        report.append(
            "| {case} | {work:.1f} | {starts} | {w5:.1f}% | {w10:.1f}% | {pitch:.2f} | {roll:.2f} |".format(
                case=row["case_id"],
                work=float(row["pump_work_m3"]),
                starts=int(row["pump_start_events"]),
                w5=float(row["avoidable_proxy_5min_work_pct"]),
                w10=float(row["avoidable_proxy_10min_work_pct"]),
                pitch=float(row["pitch_abs_p95"]),
                roll=float(row["roll_abs_p95"]),
            )
        )
    report.extend(
        [
            "",
            "## Interpretation Rule",
            "",
            "- If the 5 min proxy work is below about 5%, the frozen reactive baseline has little actuator-only economy headroom.",
            "- If it is above about 10%, a strictly actuator-level no-preview Pareto scan is worth running.",
            "- This audit deliberately does not change PI gains or add prediction logic.",
        ]
    )
    (OUT_DIR / "reactive_pump_saving_space_report.md").write_text("\n".join(report), encoding="utf-8")
    print(OUT_DIR / "reactive_pump_saving_space_report.md")


if __name__ == "__main__":
    main()
