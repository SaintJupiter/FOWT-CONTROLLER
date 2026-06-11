#!/usr/bin/env python3
"""Make clear cumulative-pump comparison figures from relaxed closed runs."""
from __future__ import annotations

from pathlib import Path

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "closed_economy_relaxed_v1_5case_2h"
RUN_DIR = DEFAULT_RUN_DIR
TS_DIR = RUN_DIR / "timeseries"
FIG_DIR = RUN_DIR / "figures"

PROFILES = [
    ("v1_engineered_minimal", "v1 baseline", "#2f6f9f"),
    ("v2_actuator_smoothed_lb", "v2 smooth candidate", "#d85c27"),
    ("econ_stop700_restart1100_ramp1p5", "relaxed economy + mild ramp", "#126b44"),
    ("econ_stop700_restart1100", "relaxed economy", "#6b8f3f"),
]

CASES = {
    "01_onset_strong": "01 onset strong",
    "04_decay_strong": "04 decay strong",
}


def _find_trace(case_id: str, profile: str) -> Path:
    matches = sorted(TS_DIR.glob(f"{case_id}_*_{profile}_timeseries.csv"))
    if not matches:
        raise FileNotFoundError(f"missing trace for {case_id} / {profile}")
    return matches[0]


def _work_m3(df: pd.DataFrame) -> float:
    pump = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float))
    return float(np.trapezoid(pump, dx=1.0) / 60.0)


def _cumulative_work_m3(df: pd.DataFrame) -> np.ndarray:
    pump = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float))
    return np.cumsum(pump) / 60.0


def _rolling_median(values: np.ndarray, window_s: float = 30.0) -> np.ndarray:
    window = max(1, int(round(float(window_s))))
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def _summary(df: pd.DataFrame) -> dict:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    pitch = np.abs(df["pitch_deg"].to_numpy(dtype=float))
    roll = np.abs(df["roll_deg"].to_numpy(dtype=float))
    return {
        "work_m3": _work_m3(df),
        "duty_pct": float(np.mean(np.abs(pump) > 1e-6) * 100.0),
        "pitch_p95": float(np.percentile(pitch, 95)),
        "roll_p95": float(np.percentile(roll, 95)),
        "max_att": float(max(np.max(pitch), np.max(roll))),
        "latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
    }


def make_case(case_id: str, label: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traces = []
    for profile, display, color in PROFILES:
        try:
            df = pd.read_csv(_find_trace(case_id, profile))
        except FileNotFoundError:
            continue
        traces.append((profile, display, color, df, _summary(df)))
    if len(traces) < 2:
        raise FileNotFoundError(f"not enough traces for {case_id} in {TS_DIR}")
    t_min = traces[0][3]["t_s"].to_numpy(dtype=float) / 60.0
    base_work = traces[0][4]["work_m3"]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14.0, 9.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 0.95, 0.9, 0.9]},
    )

    ax = axes[0]
    for _, display, color, df, stats in traces:
        saving = (base_work - stats["work_m3"]) / max(base_work, 1e-9) * 100.0
        ax.plot(
            t_min,
            _cumulative_work_m3(df),
            color=color,
            lw=1.9,
            label=f"{display}: {stats['work_m3']:.1f} m3 ({saving:+.2f}%)",
        )
    ax.set_ylabel("Cumulative\npump work (m3)")
    ax.legend(frameon=False, ncol=1, fontsize=8.5, loc="upper left")
    ax.grid(True, color="#d7dce2", lw=0.75)

    ax = axes[1]
    for _, display, color, df, _stats in traces:
        pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
        ax.plot(t_min, _rolling_median(pump), color=color, lw=1.35, label=display)
    ax.set_ylabel("Total pump rate\n30s median")
    ax.grid(True, color="#d7dce2", lw=0.75)

    ax = axes[2]
    for _, display, color, df, stats in traces:
        att = np.maximum(
            np.abs(df["pitch_deg"].to_numpy(dtype=float)),
            np.abs(df["roll_deg"].to_numpy(dtype=float)),
        )
        ax.plot(
            t_min,
            att,
            color=color,
            lw=1.25,
            label=f"{display}: p95 pitch {stats['pitch_p95']:.2f} deg",
        )
    ax.axhline(4.0, color="#9c2f2f", lw=0.8, ls="--", alpha=0.72)
    ax.set_ylabel("Max attitude\n(deg)")
    ax.grid(True, color="#d7dce2", lw=0.75)

    ax = axes[3]
    for idx, (_profile, display, color, df, stats) in enumerate(traces):
        active = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float)) > 1e-6
        y0 = idx * 1.0
        ax.fill_between(t_min, y0, y0 + 0.72, where=active, step="post", color=color, alpha=0.72)
        ax.text(
            t_min[0],
            y0 + 0.36,
            f"{display} | duty {stats['duty_pct']:.1f}% | latch {stats['latch_switches']}",
            va="center",
            ha="right",
            fontsize=8,
        )
    ax.set_yticks([])
    ax.set_ylabel("Pump active")
    ax.set_xlabel("Time (min)")
    ax.grid(True, axis="x", color="#d7dce2", lw=0.75)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(f"{label}: closed baseline vs smooth/economy candidates", fontsize=13, y=1.01)
    out = FIG_DIR / f"{case_id}_v1_v2_relaxed_economy_cumulative_pump.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global RUN_DIR, TS_DIR, FIG_DIR
    RUN_DIR = Path(args.run_dir)
    if not RUN_DIR.is_absolute():
        RUN_DIR = REPO_ROOT / RUN_DIR
    TS_DIR = RUN_DIR / "timeseries"
    FIG_DIR = RUN_DIR / "figures"
    paths = [make_case(case_id, label) for case_id, label in CASES.items()]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
