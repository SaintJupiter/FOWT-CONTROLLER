#!/usr/bin/env python3
"""Plot learned-vs-persistence raw 1Hz curves for adaptive+micro casebook runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

ACTION_LEVEL = {
    "active_reverse_small": -1.0,
    "hold": 0.0,
    "pump_saving": 0.5,
    "active_small": 1.0,
    "active_medium": 2.0,
}

COLORS = {
    "learned": "#1f77b4",
    "persistence": "#ff7f0e",
}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _find_one(directory: Path, case_id: str, suffix: str) -> Path:
    matches = sorted(directory.glob(f"{case_id}*{suffix}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one match for {case_id}*{suffix} in {directory}, got {len(matches)}")
    return matches[0]


def _load_timeseries(run_dir: Path, case_id: str) -> pd.DataFrame:
    path = _find_one(run_dir / "timeseries", case_id, "_timeseries.csv")
    df = pd.read_csv(path)
    df["t_min"] = df["t_s"].to_numpy(dtype=float) / 60.0
    dt_min = float(np.median(np.diff(df["t_s"].to_numpy(dtype=float)))) / 60.0
    df["pump_cum_m3"] = np.cumsum(np.abs(df["pump_total_rate_m3_min"].to_numpy(dtype=float)) * dt_min)
    return df


def _load_planner(run_dir: Path, case_id: str) -> pd.DataFrame:
    path = _find_one(run_dir / "planner_logs", case_id, "_planner_log.csv")
    df = pd.read_csv(path)
    df["t_min"] = df["current_time_s"].to_numpy(dtype=float) / 60.0
    df["action_level"] = df["first_action"].map(ACTION_LEVEL).fillna(np.nan)
    return df


def _summary_row(run_dir: Path, case_id: str) -> pd.Series:
    df = pd.read_csv(run_dir / "casebook_summary.csv")
    row = df[df["case_id"] == case_id]
    if row.empty:
        raise KeyError(case_id)
    return row.iloc[0]


def _shade_differences(ax: plt.Axes, learned: pd.DataFrame, persistence: pd.DataFrame) -> None:
    bucket_s = 600
    max_t = min(float(learned["t_s"].max()), float(persistence["t_s"].max()))
    for start in np.arange(0.0, max_t + 1e-9, bucket_s):
        end = min(start + bucket_s, max_t + 1.0)
        ml = (learned["t_s"] >= start) & (learned["t_s"] < end)
        mp = (persistence["t_s"] >= start) & (persistence["t_s"] < end)
        lp = float(np.abs(learned.loc[ml, "pump_total_rate_m3_min"]).sum() / 60.0)
        pp = float(np.abs(persistence.loc[mp, "pump_total_rate_m3_min"]).sum() / 60.0)
        delta = lp - pp
        if abs(delta) < 5.0:
            continue
        color = "#b8e0c2" if delta < 0.0 else "#f4c2c2"
        ax.axvspan(start / 60.0, end / 60.0, color=color, alpha=0.20, lw=0)


def plot_case(case_id: str, learned_dir: Path, persistence_dir: Path, out_dir: Path) -> Path:
    ts = {
        "learned": _load_timeseries(learned_dir, case_id),
        "persistence": _load_timeseries(persistence_dir, case_id),
    }
    planners = {
        "learned": _load_planner(learned_dir, case_id),
        "persistence": _load_planner(persistence_dir, case_id),
    }
    rows = {
        "learned": _summary_row(learned_dir, case_id),
        "persistence": _summary_row(persistence_dir, case_id),
    }
    pump_l = float(rows["learned"]["primary_pump_work_m3"])
    pump_p = float(rows["persistence"]["primary_pump_work_m3"])
    diff = pump_l - pump_p
    pct = (pump_l / pump_p - 1.0) * 100.0 if pump_p > 1e-9 else 0.0
    label = str(rows["learned"]["label"])

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(14.0, 12.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.25, 1.15, 1.2, 1.2, 1.1]},
    )
    fig.suptitle(
        f"{case_id}: {label} | learned {pump_l:.1f} m3 vs persistence {pump_p:.1f} m3 ({pct:.1f}%)",
        fontsize=13.5,
        fontweight="bold",
    )

    base = ts["learned"]
    axes[0].plot(base["t_min"], base["wind_speed"], color="#333333", lw=1.2, label="wind speed")
    ax0b = axes[0].twinx()
    ax0b.plot(base["t_min"], base["wind_dir_deg"], color="#999999", lw=0.9, alpha=0.8, label="wind dir")
    axes[0].set_ylabel("wind\nm/s")
    ax0b.set_ylabel("dir deg")
    axes[0].grid(True, alpha=0.22)

    for ax in axes[1:]:
        _shade_differences(ax, ts["learned"], ts["persistence"])

    for source, df in ts.items():
        axes[1].plot(df["t_min"], df["pump_cum_m3"], color=COLORS[source], lw=2.0, label=source)
    axes[1].set_ylabel("cum pump\nm3")
    axes[1].grid(True, alpha=0.22)
    axes[1].legend(loc="upper left", ncol=2, frameon=False)

    for source, df in ts.items():
        axes[2].plot(df["t_min"], np.abs(df["pump_total_rate_m3_min"]), color=COLORS[source], lw=0.75, alpha=0.9)
    axes[2].set_ylabel("raw pump\nm3/min")
    axes[2].grid(True, alpha=0.22)

    for source, df in ts.items():
        axes[3].plot(df["t_min"], df["pitch_deg"], color=COLORS[source], lw=0.9, alpha=0.95)
    axes[3].axhline(6.5, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[3].axhline(-6.5, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[3].set_ylabel("pitch deg")
    axes[3].grid(True, alpha=0.22)

    for source, df in ts.items():
        axes[4].plot(df["t_min"], df["roll_deg"], color=COLORS[source], lw=0.9, alpha=0.95)
    axes[4].axhline(5.5, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[4].axhline(-5.5, color="#aa3333", ls="--", lw=0.8, alpha=0.55)
    axes[4].set_ylabel("roll deg")
    axes[4].grid(True, alpha=0.22)

    for source, planner in planners.items():
        axes[5].step(
            planner["t_min"],
            planner["action_level"],
            where="post",
            color=COLORS[source],
            lw=1.5,
            alpha=0.85,
            label=f"{source} action",
        )
        if "relief_medium_cap_active" in planner.columns:
            cap = planner[planner["relief_medium_cap_active"].astype(int) == 1]
            axes[5].scatter(cap["t_min"], np.full(len(cap), 2.25), color=COLORS[source], s=28, marker="v")
        if "hold_risk_micro_action_active" in planner.columns:
            micro = planner[planner["hold_risk_micro_action_active"].astype(int) == 1]
            axes[5].scatter(micro["t_min"], np.full(len(micro), 2.45), color=COLORS[source], s=30, marker="*")
    for source, df in ts.items():
        if "preview_primary_safety_fallback" in df.columns:
            fb = df[df["preview_primary_safety_fallback"].astype(int) == 1]
            if not fb.empty:
                axes[5].scatter(
                    fb["t_min"].iloc[:: max(1, len(fb) // 80)],
                    np.full(len(fb.iloc[:: max(1, len(fb) // 80)]), -0.35),
                    color=COLORS[source],
                    s=8,
                    alpha=0.45,
                )
    axes[5].set_yticks([-1.0, 0.0, 0.5, 1.0, 2.0])
    axes[5].set_yticklabels(["rev", "hold", "save", "small", "medium"])
    axes[5].set_ylabel("planner\nmarkers")
    axes[5].set_xlabel("time min")
    axes[5].grid(True, alpha=0.22)
    axes[5].legend(loc="upper left", ncol=2, frameon=False)
    axes[5].set_xlim(0, 120)

    note = (
        "Blue=learned, orange=persistence. Green bands: learned uses >=5 m3 less pump in that 10-min bucket; "
        "red bands: learned uses more. Triangles=relief cap active; stars=hold-risk micro; bottom dots=fallback samples."
    )
    fig.text(0.01, 0.006, note, fontsize=9, color="#444444")
    fig.tight_layout(rect=[0, 0.025, 1, 0.955])

    out_path = out_dir / f"{case_id}_adaptive_micro_learned_vs_persistence.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-dir", default="outputs/wind_prediction/reliefcap_adaptive_v1_micro_pilot30_learned_2h")
    parser.add_argument("--persistence-dir", default="outputs/wind_prediction/reliefcap_adaptive_v1_micro_pilot30_persistence_2h")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "09_fr_relief_09",
            "14_sf_calibration_07",
            "21_b_decay_strong",
            "23_b_high_pressure_like",
        ],
    )
    parser.add_argument("--out-dir", default="outputs/wind_prediction/adaptive_micro_case_curves")
    args = parser.parse_args()

    learned_dir = _resolve(args.learned_dir)
    persistence_dir = _resolve(args.persistence_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for case_id in args.cases:
        paths.append(plot_case(case_id, learned_dir, persistence_dir, out_dir))
    summary = pd.DataFrame({"case_id": args.cases, "figure": [str(p) for p in paths]})
    summary.to_csv(out_dir / "figure_index.csv", index=False)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
