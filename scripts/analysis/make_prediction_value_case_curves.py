#!/usr/bin/env python3
"""Plot prediction-source case curves for raw 1 Hz casebook outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

RUN_DIRS = {
    "learned": "outputs/wind_prediction/forecast_value_rawenv_learned_holdpause_v2_10case_2h",
    "persistence": "outputs/wind_prediction/forecast_value_rawenv_persistence_holdpause_v2_10case_2h",
    "oracle": "outputs/wind_prediction/forecast_value_rawenv_oracle_holdpause_v2_10case_2h",
}

COLORS = {
    "learned": "#1f77b4",
    "persistence": "#ff7f0e",
    "oracle": "#2ca02c",
}

ACTION_LEVEL = {
    "active_reverse_small": -1.0,
    "hold": 0.0,
    "pump_saving": 0.5,
    "active_small": 1.0,
    "active_medium": 2.0,
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _find_one(directory: Path, prefix: str, suffix: str) -> Path:
    matches = sorted(directory.glob(f"{prefix}*{suffix}"))
    if not matches:
        raise FileNotFoundError(f"no match in {directory} for {prefix}*{suffix}")
    return matches[0]


def _load_timeseries(run_dir: Path, case_id: str, source: str) -> pd.DataFrame:
    path = _find_one(
        run_dir / "timeseries",
        case_id,
        f"prediction_primary_{source}_rawenv_holdpause_v2_10case_2h_timeseries.csv",
    )
    df = pd.read_csv(path)
    df["t_min"] = df["t_s"].to_numpy(dtype=float) / 60.0
    rate = np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float))
    dt_min = float(np.median(np.diff(df["t_s"].to_numpy(dtype=float)))) / 60.0
    df["pump_cum_m3"] = np.cumsum(rate * dt_min)
    return df


def _load_planner(run_dir: Path, case_id: str, source: str) -> pd.DataFrame:
    path = _find_one(
        run_dir / "planner_logs",
        case_id,
        f"prediction_primary_{source}_rawenv_holdpause_v2_10case_2h_planner_log.csv",
    )
    df = pd.read_csv(path)
    df["t_min"] = df["current_time_s"].to_numpy(dtype=float) / 60.0
    df["action_level"] = df["first_action"].map(ACTION_LEVEL).fillna(np.nan)
    return df


def _load_bucket_compare(compare_csv: Path, case_id: str) -> pd.DataFrame:
    if not compare_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(compare_csv)
    mask = df["case_key"].astype(str).str.startswith(case_id)
    return df[mask].copy()


def _shade_bucket_deltas(ax: plt.Axes, bucket_compare: pd.DataFrame) -> None:
    if bucket_compare.empty:
        return
    for _, row in bucket_compare.iterrows():
        delta = float(row.get("learned_minus_persistence_pump_m3", 0.0))
        if abs(delta) < 5.0:
            continue
        b = int(row["bucket"])
        color = "#b8e0c2" if delta < 0.0 else "#f4c2c2"
        ax.axvspan(b * 10.0, (b + 1) * 10.0, color=color, alpha=0.22, lw=0)


def plot_case(case_id: str, out_dir: Path, compare_csv: Path) -> Path:
    ts = {
        source: _load_timeseries(_resolve(run_dir), case_id, source)
        for source, run_dir in RUN_DIRS.items()
    }
    planners = {
        source: _load_planner(_resolve(run_dir), case_id, source)
        for source, run_dir in RUN_DIRS.items()
    }
    bucket_compare = _load_bucket_compare(compare_csv, case_id)

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13.5, 11.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.3, 1.3, 1.4, 1.25]},
    )
    fig.suptitle(
        f"{case_id}: raw 1Hz execution, raw-envelope + hold-pause v2",
        fontsize=14,
        fontweight="bold",
    )

    base = ts["learned"]
    axes[0].plot(base["t_min"], base["wind_speed"], color="#444444", lw=1.2)
    axes0b = axes[0].twinx()
    axes0b.plot(base["t_min"], base["wind_dir_deg"], color="#888888", lw=0.9, alpha=0.75)
    axes[0].set_ylabel("wind speed\nm/s")
    axes0b.set_ylabel("wind dir deg")
    axes[0].grid(True, alpha=0.22)

    for ax in axes[1:]:
        _shade_bucket_deltas(ax, bucket_compare)

    for source, df in ts.items():
        axes[1].plot(df["t_min"], df["pump_cum_m3"], label=source, color=COLORS[source], lw=1.9)
    axes[1].set_ylabel("cumulative\npump m3")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(loc="upper left", ncol=3, frameon=False)

    for source, df in ts.items():
        axes[2].plot(
            df["t_min"],
            np.abs(df["pump_total_rate_m3_min"]),
            label=source,
            color=COLORS[source],
            lw=0.85,
            alpha=0.88,
        )
    axes[2].set_ylabel("raw pump\nm3/min")
    axes[2].grid(True, alpha=0.22)

    for source, df in ts.items():
        axes[3].plot(df["t_min"], df["pitch_deg"], color=COLORS[source], lw=0.9, alpha=0.9)
    axes[3].axhline(8.0, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[3].axhline(-8.0, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[3].set_ylabel("pitch deg\nraw 1Hz")
    axes[3].grid(True, alpha=0.22)

    for source, df in ts.items():
        axes[4].plot(df["t_min"], df["roll_deg"], color=COLORS[source], lw=0.9, alpha=0.9)
    for source, planner in planners.items():
        axes[4].step(
            planner["t_min"],
            planner["action_level"],
            where="post",
            color=COLORS[source],
            lw=1.1,
            alpha=0.62,
            linestyle="--",
        )
    axes[4].axhline(8.0, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[4].axhline(-8.0, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[4].set_ylabel("roll deg\n+ action")
    axes[4].set_xlabel("time min")
    axes[4].grid(True, alpha=0.22)
    axes[4].set_xlim(0, 120)

    note = (
        "Green background: learned raw bucket pump < persistence by >=5 m3. "
        "Red: learned > persistence by >=5 m3. "
        "Dashed steps on bottom panel: planner action level "
        "hold=0, small=1, medium=2."
    )
    fig.text(0.01, 0.005, note, fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])

    out_path = out_dir / f"{case_id}_learned_persistence_oracle_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_case_summary(cases: list[str], out_dir: Path) -> Path:
    rows = []
    for source, run_dir in RUN_DIRS.items():
        summary = pd.read_csv(_resolve(run_dir) / "casebook_summary.csv")
        summary = summary[summary["case_id"].isin(cases)].copy()
        summary["source"] = source
        rows.append(summary)
    df = pd.concat(rows, ignore_index=True)
    keep = [
        "case_id",
        "source",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_latch_switches",
        "d_pump_work_pct",
    ]
    out = df[keep].sort_values(["case_id", "source"])
    path = out_dir / "selected_case_source_summary.csv"
    out.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "02_onset_signflip",
            "06_signflip_high",
            "07_signflip_sustained",
            "09_high_pressure_event",
        ],
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/prediction_value_case_curves_holdpause_v2",
    )
    parser.add_argument(
        "--compare-csv",
        default=(
            "outputs/wind_prediction/prediction_signal_flow_rawenv_holdpause_v2_10case_2h/"
            "prediction_signal_flow_source_compare.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compare_csv = _resolve(args.compare_csv)
    paths = [plot_case(case, out_dir, compare_csv) for case in args.cases]
    summary_path = write_case_summary(list(args.cases), out_dir)
    print("Summary:", summary_path)
    for path in paths:
        print("Figure:", path)


if __name__ == "__main__":
    main()
