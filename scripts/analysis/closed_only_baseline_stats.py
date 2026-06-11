#!/usr/bin/env python3
"""Compute closed_only baseline statistics on existing A2-smoke timeseries.

Reads the 3 closed_only 10h runs already produced by run_a2_smoke.py and
extracts the metrics that an economic-form planner objective will be compared
against:

  attitude  : pitch_deg, roll_deg  -> mean / std / p50 / p95 / p99 / max
  pump cost : pump_total_rate_m3_min -> cumulative work, duty cycle
  startstop : pump_stage_switch_count -> per-hour rate
  duration  : fraction of time any pump is active

Outputs:
  outputs/wind_prediction/closed_only_baseline/baseline_stats.csv
  outputs/wind_prediction/closed_only_baseline/baseline_summary.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
src_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_smoke"
out_dir = repo_root / "outputs" / "wind_prediction" / "closed_only_baseline"
out_dir.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    ("2023-09-11_034000", "clean_low_pressure_normal"),
    ("2024-09-05_181000", "residual_high_normal"),
    ("2022-02-04_110000", "high_pressure_high_event"),
]


def percentiles(x: np.ndarray) -> dict:
    a = np.abs(x)
    return {
        "abs_mean": float(a.mean()),
        "abs_std": float(a.std()),
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p95": float(np.percentile(a, 95)),
        "abs_p99": float(np.percentile(a, 99)),
        "abs_max": float(a.max()),
        "rms": float(np.sqrt(np.mean(x ** 2))),
    }


def analyse(slug: str, group: str) -> dict:
    path = src_dir / f"{slug}_closed_only_timeseries.csv"
    df = pd.read_csv(path)
    dt = float(np.median(np.diff(df["t_s"].values))) if "t_s" in df.columns else 1.0
    n_steps = len(df)
    duration_h = n_steps * dt / 3600.0

    pitch_stats = percentiles(df["pitch_deg"].values)
    roll_stats = percentiles(df["roll_deg"].values)

    pump_rate = df["pump_total_rate_m3_min"].abs().values  # m^3/min
    pump_active = (pump_rate > 1e-6)
    pump_work_m3 = float(np.trapezoid(pump_rate, dx=dt) / 60.0)  # m^3 cumulative volume moved
    pump_duty_ratio = float(np.mean(pump_active))
    pump_active_minutes = float(pump_duty_ratio * n_steps * dt / 60.0)

    if "pump_stage_switch_count" in df.columns:
        switches = float(df["pump_stage_switch_count"].iloc[-1] - df["pump_stage_switch_count"].iloc[0])
    else:
        switches = float("nan")
    if "pump_latch_switch_count" in df.columns:
        latch_switches = float(df["pump_latch_switch_count"].iloc[-1] - df["pump_latch_switch_count"].iloc[0])
    else:
        latch_switches = float("nan")
    fullspeed_ratio = float(np.mean(df["pump_fullspeed_any"].values)) if "pump_fullspeed_any" in df.columns else float("nan")

    return {
        "window": slug,
        "group": group,
        "duration_h": duration_h,
        **{f"pitch_{k}": v for k, v in pitch_stats.items()},
        **{f"roll_{k}": v for k, v in roll_stats.items()},
        "pump_work_m3": pump_work_m3,
        "pump_work_m3_per_h": pump_work_m3 / max(duration_h, 1e-9),
        "pump_duty_ratio": pump_duty_ratio,
        "pump_active_minutes": pump_active_minutes,
        "pump_stage_switches": switches,
        "pump_stage_switches_per_h": switches / max(duration_h, 1e-9) if not np.isnan(switches) else float("nan"),
        "pump_latch_switches": latch_switches,
        "pump_fullspeed_ratio": fullspeed_ratio,
    }


def main() -> None:
    rows = [analyse(slug, group) for slug, group in WINDOWS]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "baseline_stats.csv", index=False)

    # Aggregate envelope candidates.
    pitch_p95 = df["pitch_abs_p95"].values
    roll_p95 = df["roll_abs_p95"].values
    pitch_max = df["pitch_abs_max"].values
    roll_max = df["roll_abs_max"].values

    summary_lines = [
        "# closed_only Baseline Statistics",
        "",
        "Source: existing A2-smoke 10-hour runs on 3 windows (no rerun).",
        "",
        "## Per-window summary",
        "",
        "| window | group | dur(h) | pitch_p95 | pitch_p99 | pitch_max | roll_p95 | roll_p99 | roll_max | pump_work(m^3) | pump_duty | switches/h |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        summary_lines.append(
            f"| {r['window']} | {r['group']} | {r['duration_h']:.1f} | "
            f"{r['pitch_abs_p95']:.3f} | {r['pitch_abs_p99']:.3f} | {r['pitch_abs_max']:.3f} | "
            f"{r['roll_abs_p95']:.3f} | {r['roll_abs_p99']:.3f} | {r['roll_abs_max']:.3f} | "
            f"{r['pump_work_m3']:.2f} | {r['pump_duty_ratio']:.3f} | {r['pump_stage_switches_per_h']:.2f} |"
        )

    summary_lines += [
        "",
        "## Envelope candidates",
        "",
        f"- pitch p95 across windows: min={pitch_p95.min():.3f}, max={pitch_p95.max():.3f}, mean={pitch_p95.mean():.3f}",
        f"- pitch p99 across windows: min={df['pitch_abs_p99'].min():.3f}, max={df['pitch_abs_p99'].max():.3f}, mean={df['pitch_abs_p99'].mean():.3f}",
        f"- pitch max across windows: min={pitch_max.min():.3f}, max={pitch_max.max():.3f}",
        f"- roll  p95 across windows: min={roll_p95.min():.3f}, max={roll_p95.max():.3f}, mean={roll_p95.mean():.3f}",
        f"- roll  p99 across windows: min={df['roll_abs_p99'].min():.3f}, max={df['roll_abs_p99'].max():.3f}, mean={df['roll_abs_p99'].mean():.3f}",
        f"- roll  max across windows: min={roll_max.min():.3f}, max={roll_max.max():.3f}",
        "",
        "## Pump cost baseline",
        "",
        f"- pump_work_m3_per_h: clean={df.iloc[0]['pump_work_m3_per_h']:.2f}, "
        f"residual_high={df.iloc[1]['pump_work_m3_per_h']:.2f}, "
        f"storm={df.iloc[2]['pump_work_m3_per_h']:.2f}",
        f"- pump_duty_ratio:    clean={df.iloc[0]['pump_duty_ratio']:.3f}, "
        f"residual_high={df.iloc[1]['pump_duty_ratio']:.3f}, "
        f"storm={df.iloc[2]['pump_duty_ratio']:.3f}",
        f"- pump_stage_switches_per_h: clean={df.iloc[0]['pump_stage_switches_per_h']:.2f}, "
        f"residual_high={df.iloc[1]['pump_stage_switches_per_h']:.2f}, "
        f"storm={df.iloc[2]['pump_stage_switches_per_h']:.2f}",
        "",
        "These are the metrics that an economic-form planner must reduce in",
        "exchange for any attitude-envelope concession.",
    ]

    (out_dir / "baseline_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print((out_dir / "baseline_summary.md").read_text())


if __name__ == "__main__":
    main()
