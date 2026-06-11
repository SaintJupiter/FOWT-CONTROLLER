#!/usr/bin/env python3
"""Hard-gate audit for long continuous pump-saving validations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TANK_CAPACITY_KG = 1850.0 * 1025.0


def _timestamp_stem(ts: str) -> str:
    return str(ts).replace(":", "").replace(" ", "_")


def _dt_s(df: pd.DataFrame, default_dt_s: float) -> float:
    if "t_s" not in df.columns or len(df) < 2:
        return float(default_dt_s)
    diffs = np.diff(df["t_s"].to_numpy(dtype=float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return float(default_dt_s)
    return float(np.median(diffs))


def _time_gt5_s(df: pd.DataFrame, default_dt_s: float) -> float:
    dt = _dt_s(df, default_dt_s)
    pitch = df.get("pitch_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float)
    return float(np.sum(np.maximum(np.abs(pitch), np.abs(roll)) > 5.0) * dt)


def _p95_abs(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(np.percentile(np.abs(df[col].to_numpy(dtype=float)), 95))


def _last_values(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    return df[cols].iloc[-1].to_numpy(dtype=float)


def _mean_bool(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(np.mean(df[col].to_numpy(dtype=float) > 0.5))


def _sum_bool(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns or len(df) == 0:
        return 0
    return int(np.sum(df[col].to_numpy(dtype=float) > 0.5))


def _find_timeseries(timeseries_dir: Path, case_id: str, timestamp: str) -> tuple[Path, Path]:
    prefix = f"{case_id}_{_timestamp_stem(timestamp)}_"
    closed = sorted(timeseries_dir.glob(f"{prefix}closed_only_timeseries.csv"))
    if not closed:
        raise FileNotFoundError(f"closed-only time series not found for {prefix}")
    candidates = sorted(
        p
        for p in timeseries_dir.glob(f"{prefix}*_timeseries.csv")
        if not p.name.endswith("_closed_only_timeseries.csv")
    )
    if not candidates:
        raise FileNotFoundError(f"primary time series not found for {prefix}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"multiple primary time series for {prefix}: {names}")
    return closed[0], candidates[0]


def _case_row(
    summary_row: pd.Series,
    timeseries_dir: Path,
    tank_capacity_kg: float,
    default_dt_s: float,
) -> dict:
    case_id = str(summary_row["case_id"])
    timestamp = str(summary_row["timestamp"])
    closed_path, primary_path = _find_timeseries(timeseries_dir, case_id, timestamp)
    closed = pd.read_csv(closed_path)
    primary = pd.read_csv(primary_path)

    tank_cols = ["tank1_kg", "tank2_kg", "tank3_kg"]
    err_cols = ["err_tank1_kg", "err_tank2_kg", "err_tank3_kg"]
    closed_tank = _last_values(closed, tank_cols)
    primary_tank = _last_values(primary, tank_cols)
    closed_err = _last_values(closed, err_cols)
    primary_err = _last_values(primary, err_cols)

    tank_diff_pct = np.abs(primary_tank - closed_tank) / float(tank_capacity_kg) * 100.0
    target_err_diff_pct = np.abs(primary_err - closed_err) / float(tank_capacity_kg) * 100.0

    closed_time_gt5_s = _time_gt5_s(closed, default_dt_s)
    primary_time_gt5_s = _time_gt5_s(primary, default_dt_s)
    closed_pitch_p95 = _p95_abs(closed, "pitch_deg")
    primary_pitch_p95 = _p95_abs(primary, "pitch_deg")
    closed_roll_p95 = _p95_abs(closed, "roll_deg")
    primary_roll_p95 = _p95_abs(primary, "roll_deg")

    closed_latch = int(closed["pump_latch_switch_count"].iloc[-1]) if "pump_latch_switch_count" in closed else 0
    primary_latch = int(primary["pump_latch_switch_count"].iloc[-1]) if "pump_latch_switch_count" in primary else 0
    fallback_rows = _sum_bool(primary, "preview_primary_safety_fallback")
    fallback_frac = _mean_bool(primary, "preview_primary_safety_fallback")
    deadband_rows = _sum_bool(primary, "deadband_target_release_active")
    deadband_frac = _mean_bool(primary, "deadband_target_release_active")

    closed_pump = float(summary_row["closed_pump_work_m3"])
    primary_pump = float(summary_row["primary_pump_work_m3"])
    saving_pct = 0.0 if closed_pump <= 0.0 else (closed_pump - primary_pump) / closed_pump * 100.0

    d_time_gt5_s = primary_time_gt5_s - closed_time_gt5_s
    d_pitch_p95 = primary_pitch_p95 - closed_pitch_p95
    d_roll_p95 = primary_roll_p95 - closed_roll_p95

    vetoes: list[str] = []
    if saving_pct < 0.0:
        vetoes.append("negative_saving")
    if float(np.max(tank_diff_pct)) > 2.0:
        vetoes.append("terminal_tank_diff_gt2pct")
    if float(np.max(target_err_diff_pct)) > 2.0:
        vetoes.append("terminal_target_err_diff_gt2pct")
    if d_time_gt5_s > 0.0:
        vetoes.append("d_time_gt5_positive")
    if d_pitch_p95 > 0.0:
        vetoes.append("d_pitch_p95_positive")
    if d_roll_p95 > 0.0:
        vetoes.append("d_roll_p95_positive")
    if fallback_rows > 0:
        vetoes.append("fallback_nonzero")

    return {
        "case_id": case_id,
        "timestamp": timestamp,
        "label": str(summary_row.get("label", "")),
        "closed_pump_work_m3": closed_pump,
        "primary_pump_work_m3": primary_pump,
        "saving_pct": saving_pct,
        "closed_latch_switches": closed_latch,
        "primary_latch_switches": primary_latch,
        "deadband_active_rows": deadband_rows,
        "deadband_active_frac": deadband_frac,
        "closed_time_gt5_s": closed_time_gt5_s,
        "primary_time_gt5_s": primary_time_gt5_s,
        "d_time_gt5_s": d_time_gt5_s,
        "closed_pitch_p95": closed_pitch_p95,
        "primary_pitch_p95": primary_pitch_p95,
        "d_pitch_p95": d_pitch_p95,
        "closed_roll_p95": closed_roll_p95,
        "primary_roll_p95": primary_roll_p95,
        "d_roll_p95": d_roll_p95,
        "terminal_tank_diff_mean_pct": float(np.mean(tank_diff_pct)),
        "terminal_tank_diff_max_pct": float(np.max(tank_diff_pct)),
        "terminal_target_err_diff_mean_pct": float(np.mean(target_err_diff_pct)),
        "terminal_target_err_diff_max_pct": float(np.max(target_err_diff_pct)),
        "fallback_rows": fallback_rows,
        "fallback_frac": fallback_frac,
        "case_pass": len(vetoes) == 0,
        "vetoes": "none" if not vetoes else ";".join(vetoes),
        "closed_timeseries": str(closed_path),
        "primary_timeseries": str(primary_path),
    }


def _write_markdown(out_path: Path, rows: pd.DataFrame, gate: dict) -> None:
    lines = [
        "# Hard 24h Pump-Saving Gate Audit",
        "",
        "## Aggregate",
        "",
        f"- cases: `{gate['cases']}`",
        f"- aggregate saving: `{gate['aggregate_saving_pct']:.3f}%`",
        f"- mean case saving: `{gate['mean_case_saving_pct']:.3f}%`",
        f"- max terminal tank diff: `{gate['max_terminal_tank_diff_pct']:.3f}%`",
        f"- max terminal target_err diff: `{gate['max_terminal_target_err_diff_pct']:.3f}%`",
        f"- max d_time>5: `{gate['max_d_time_gt5_s']:.1f}s`",
        f"- max d_pitch_p95: `{gate['max_d_pitch_p95']:.3f} deg`",
        f"- max d_roll_p95: `{gate['max_d_roll_p95']:.3f} deg`",
        f"- total fallback rows: `{gate['fallback_rows']}`",
        f"- cases passing all per-case gates: `{gate['passed_cases']}/{gate['cases']}`",
        f"- hard gate pass: `{gate['hard_gate_pass']}`",
        "",
        "## Per Case",
        "",
        "| case | saving | tank diff max | target_err diff max | d_time>5 | d_pitch_p95 | d_roll_p95 | fallback | latch | deadband | pass | vetoes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows.to_dict("records"):
        lines.append(
            "| {case_id} | {saving_pct:.2f}% | {terminal_tank_diff_max_pct:.3f}% | "
            "{terminal_target_err_diff_max_pct:.3f}% | {d_time_gt5_s:.0f}s | "
            "{d_pitch_p95:.3f} | {d_roll_p95:.3f} | {fallback_rows} | "
            "{closed_latch_switches}->{primary_latch_switches} | {deadband_active_frac:.1%} | "
            "{case_pass} | {vetoes} |".format(**r)
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--tank-capacity-kg", type=float, default=DEFAULT_TANK_CAPACITY_KG)
    parser.add_argument("--default-dt-s", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = args.run_dir
    summary_path = run_dir / "casebook_summary.csv"
    timeseries_dir = run_dir / "timeseries"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not timeseries_dir.exists():
        raise FileNotFoundError(timeseries_dir)

    summary = pd.read_csv(summary_path)
    rows = pd.DataFrame(
        [
            _case_row(row, timeseries_dir, float(args.tank_capacity_kg), float(args.default_dt_s))
            for _, row in summary.iterrows()
        ]
    )
    closed_total = float(rows["closed_pump_work_m3"].sum())
    primary_total = float(rows["primary_pump_work_m3"].sum())
    aggregate_saving = 0.0 if closed_total <= 0.0 else (closed_total - primary_total) / closed_total * 100.0
    gate = {
        "cases": int(len(rows)),
        "aggregate_saving_pct": aggregate_saving,
        "mean_case_saving_pct": float(rows["saving_pct"].mean()) if len(rows) else 0.0,
        "max_terminal_tank_diff_pct": float(rows["terminal_tank_diff_max_pct"].max()) if len(rows) else 0.0,
        "max_terminal_target_err_diff_pct": float(rows["terminal_target_err_diff_max_pct"].max()) if len(rows) else 0.0,
        "max_d_time_gt5_s": float(rows["d_time_gt5_s"].max()) if len(rows) else 0.0,
        "max_d_pitch_p95": float(rows["d_pitch_p95"].max()) if len(rows) else 0.0,
        "max_d_roll_p95": float(rows["d_roll_p95"].max()) if len(rows) else 0.0,
        "fallback_rows": int(rows["fallback_rows"].sum()) if len(rows) else 0,
        "passed_cases": int(rows["case_pass"].sum()) if len(rows) else 0,
    }
    gate["hard_gate_pass"] = bool(
        gate["cases"] >= 8
        and gate["aggregate_saving_pct"] >= 18.0
        and gate["mean_case_saving_pct"] >= 18.0
        and gate["max_terminal_tank_diff_pct"] <= 2.0
        and gate["max_terminal_target_err_diff_pct"] <= 2.0
        and gate["max_d_time_gt5_s"] <= 0.0
        and gate["max_d_pitch_p95"] <= 0.0
        and gate["max_d_roll_p95"] <= 0.0
        and gate["fallback_rows"] == 0
        and gate["passed_cases"] == gate["cases"]
    )

    rows.to_csv(run_dir / "hard_gate_case_audit.csv", index=False)
    pd.DataFrame([gate]).to_csv(run_dir / "hard_gate_aggregate.csv", index=False)
    _write_markdown(run_dir / "hard_gate_audit.md", rows, gate)
    print(
        "hard_gate_pass={hard_gate_pass} cases={cases} aggregate_saving={aggregate_saving_pct:.3f}% "
        "mean_case_saving={mean_case_saving_pct:.3f}% passed_cases={passed_cases}/{cases}".format(
            **gate
        )
    )


if __name__ == "__main__":
    main()
