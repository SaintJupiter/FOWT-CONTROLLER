#!/usr/bin/env python3
"""Plot guard10 learned-relief LSTM comparisons.

This is a read-only review helper. It compares the standard learned LSTM,
relief-emphasized learned LSTM variants, PP oracle, and reactive baseline on
the guard10 case set. It does not run controllers or modify logs.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "wind_prediction" / "relief_lstm_guard10_review_figures"

RUNS = {
    "reactive": {
        "dir": REPO / "outputs/wind_prediction/reactive_vs_pp_oracle_guard10",
        "needle": "closed_only",
        "label": "Reactive",
        "color": "#555555",
        "ls": "--",
    },
    "pp_oracle": {
        "dir": REPO / "outputs/wind_prediction/baseline_overlays_oracle_guard10_2h",
        "needle": "prediction_primary",
        "label": "PP oracle",
        "color": "#1f77b4",
        "ls": "-.",
    },
    "pp_base": {
        "dir": REPO / "outputs/wind_prediction/tune_baseline_learned_guard10",
        "needle": "prediction_primary",
        "label": "PP baseline LSTM",
        "color": "#d62728",
        "ls": "-",
    },
    "pp_e5": {
        "dir": REPO / "outputs/wind_prediction/pp_learned_synth_relief_guard10",
        "needle": "prediction_primary",
        "label": "PP relief e5",
        "color": "#ff7f0e",
        "ls": "-",
    },
    "pp_e10": {
        "dir": REPO / "outputs/wind_prediction/pp_learned_relief_e10_guard10",
        "needle": "prediction_primary",
        "label": "PP relief e10",
        "color": "#2ca02c",
        "ls": "-",
    },
    "pp_e15": {
        "dir": REPO / "outputs/wind_prediction/pp_learned_relief_e15_guard10",
        "needle": "prediction_primary",
        "label": "PP relief e15",
        "color": "#9467bd",
        "ls": "-",
    },
}

CASES = [
    "fr_relief_09",
    "sf_holdout_02",
    "b_decay_strong",
    "b_signflip_fallback",
    "fr_relief_01",
]


def find_case_file(run_dir: Path, subdir: str, case: str, needle: str) -> Path | None:
    base = run_dir / subdir
    if not base.exists():
        return None
    for path in sorted(base.glob("*")):
        if case in path.name and needle in path.name:
            return path
    return None


def fallback_series(ts: pd.DataFrame) -> pd.Series:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "fallback_active",
        "safety_fallback_active",
    ):
        if col in ts.columns:
            return ts[col].fillna(0).astype(float)
    candidates = [c for c in ts.columns if "fallback" in c.lower() and ts[c].dtype.kind in "biuf"]
    if candidates:
        return ts[candidates[0]].fillna(0).astype(float)
    return pd.Series(np.zeros(len(ts)), index=ts.index)


def load_timeseries(run_key: str, case: str) -> pd.DataFrame | None:
    meta = RUNS[run_key]
    path = find_case_file(meta["dir"], "timeseries", case, meta["needle"])
    if path is None:
        return None
    ts = pd.read_csv(path, low_memory=False)
    ts["time_min"] = ts["t_s"].astype(float) / 60.0 if "t_s" in ts.columns else np.arange(len(ts)) / 60.0
    ts["abs_pitch_deg"] = ts["pitch_deg"].abs()
    ts["abs_roll_deg"] = ts["roll_deg"].abs() if "roll_deg" in ts.columns else 0.0
    ts["fallback_active_plot"] = fallback_series(ts)
    if "pump_total_rate_m3_min" in ts.columns:
        ts["cum_pump_m3"] = (ts["pump_total_rate_m3_min"].abs() / 60.0).cumsum()
    else:
        ts["cum_pump_m3"] = 0.0
    return ts


def load_planner(run_key: str, case: str) -> pd.DataFrame | None:
    meta = RUNS[run_key]
    path = find_case_file(meta["dir"], "planner_logs", case, meta["needle"])
    if path is None:
        return None
    return pd.read_csv(path, low_memory=False)


def metrics_for(run_key: str, case: str) -> dict[str, float] | None:
    ts = load_timeseries(run_key, case)
    if ts is None:
        return None
    return {
        "pump_m3": float(ts["cum_pump_m3"].iloc[-1]),
        "pitch_p95": float(np.percentile(ts["abs_pitch_deg"], 95)),
        "roll_p95": float(np.percentile(ts["abs_roll_deg"], 95)),
        "fallback_pct": float(ts["fallback_active_plot"].mean() * 100.0),
        "time_over3_s": float(((ts["abs_pitch_deg"] > 3.0) | (ts["abs_roll_deg"] > 3.0)).sum()),
    }


def plot_summary() -> pd.DataFrame:
    rows = []
    for key, meta in RUNS.items():
        for case in CASES + ["b_high_pressure_event", "b_residual_high", "lowrisk_quiet", "lowrisk_clean", "lowrisk_random_03"]:
            m = metrics_for(key, case)
            if m is not None:
                rows.append({"run": key, "label": meta["label"], "case": case, **m})
    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["run", "label"], sort=False)
        .agg(
            sum_pump_m3=("pump_m3", "sum"),
            max_pitch_p95=("pitch_p95", "max"),
            max_roll_p95=("roll_p95", "max"),
            sum_fallback_pct=("fallback_pct", "sum"),
            sum_time_over3_s=("time_over3_s", "sum"),
        )
        .reset_index()
    )
    agg.to_csv(OUT_DIR / "relief_lstm_guard10_review_summary.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(len(agg))
    labels = agg["label"].tolist()
    bars = [
        ("sum_pump_m3", "sum pump (m3)"),
        ("max_pitch_p95", "max pitch p95 (deg)"),
        ("max_roll_p95", "max roll p95 (deg)"),
        ("sum_fallback_pct", "sum fallback (pp)"),
    ]
    for ax, (col, title) in zip(axes, bars):
        colors = [RUNS[r]["color"] for r in agg["run"]]
        ax.bar(x, agg[col], color=colors, alpha=0.86)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        if "pitch" in col or "roll" in col:
            ax.axhline(3.0, color="#777777", lw=0.8, ls=":", label="3 deg")
            ax.axhline(5.0, color="#aa0000", lw=0.8, ls=":", label="5 deg")
    fig.suptitle("Guard10 aggregate comparison: baseline LSTM vs relief-emphasis variants", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_DIR / "summary_guard10_variants.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return df


def plot_case(case: str, include_variants: list[str]) -> None:
    data = {key: load_timeseries(key, case) for key in include_variants}
    data = {k: v for k, v in data.items() if v is not None}
    planners = {key: load_planner(key, case) for key in include_variants if key != "reactive"}

    fig, axes = plt.subplots(4, 1, figsize=(13, 10.5), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.7, 1.3, 1.8]})

    ax = axes[0]
    for key, ts in data.items():
        meta = RUNS[key]
        ax.plot(ts["time_min"], ts["abs_pitch_deg"], color=meta["color"], ls=meta["ls"], lw=1.5, label=meta["label"])
    ax.axhline(3.0, color="#777777", lw=0.8, ls=":")
    ax.axhline(5.0, color="#aa0000", lw=0.8, ls=":")
    ax.set_ylabel("|pitch| deg")
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1]
    for key, ts in data.items():
        meta = RUNS[key]
        ax.plot(ts["time_min"], ts["abs_roll_deg"], color=meta["color"], ls=meta["ls"], lw=1.4, label=meta["label"])
    ax.axhline(3.0, color="#777777", lw=0.8, ls=":")
    ax.axhline(5.0, color="#aa0000", lw=0.8, ls=":")
    ax.set_ylabel("|roll| deg")
    ax.grid(alpha=0.25)

    ax = axes[2]
    for key, ts in data.items():
        meta = RUNS[key]
        ax.plot(ts["time_min"], ts["cum_pump_m3"], color=meta["color"], ls=meta["ls"], lw=1.5, label=meta["label"])
    ax.set_ylabel("cum pump m3")
    ax.grid(alpha=0.25)

    ax = axes[3]
    offset = 0
    ytick_pos = []
    ytick_lab = []
    for key, df in planners.items():
        if df is None or "first_action" not in df.columns:
            continue
        meta = RUNS[key]
        bucket = df["bucket"].to_numpy() if "bucket" in df.columns else np.arange(len(df))
        t = bucket * 10.0 + 5.0
        actions = df["first_action"].astype(str).tolist()
        action_level = {
            "hold": 0,
            "pump_saving": 1,
            "active_small": 2,
            "active_medium": 3,
            "active_reverse_small": -1,
        }
        vals = [action_level.get(a, 0) + offset for a in actions]
        ax.step(t, vals, where="mid", color=meta["color"], lw=1.6, label=meta["label"])
        ytick_pos.append(offset)
        ytick_lab.append(meta["label"])
        offset += 4
    ax.set_ylabel("bucket action")
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_xlabel("time (min)")
    ax.set_xlim(0, 120)

    metrics_bits = []
    for key in include_variants:
        m = metrics_for(key, case)
        if m:
            metrics_bits.append(
                f"{RUNS[key]['label']}: pump={m['pump_m3']:.0f}, p95={m['pitch_p95']:.2f}, "
                f"roll95={m['roll_p95']:.2f}, fb={m['fallback_pct']:.1f}%"
            )
    fig.suptitle(case + "\n" + " | ".join(metrics_bits), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    safe = case.replace("/", "_")
    fig.savefig(OUT_DIR / f"case_{safe}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_summary()
    # Four-line version for the main visual judgement.
    main_runs = ["reactive", "pp_oracle", "pp_base", "pp_e15"]
    for case in CASES:
        plot_case(case, main_runs)
    # Variant-heavy plots for the three changed cases.
    variant_runs = ["pp_oracle", "pp_base", "pp_e5", "pp_e10", "pp_e15"]
    for case in ("fr_relief_09", "sf_holdout_02", "b_decay_strong"):
        plot_case(case + "_variants", []) if False else None
        # Reuse plot_case by temporarily plotting all PP variants without reactive.
        plot_case(case + "_pp_variants_dummy", []) if False else None
        data_case = case
        include = variant_runs
        plot_case(data_case, include)
        src = OUT_DIR / f"case_{data_case}_comparison.png"
        dst = OUT_DIR / f"case_{data_case}_pp_variants.png"
        src.replace(dst)
        plot_case(data_case, main_runs)
    print(f"wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
