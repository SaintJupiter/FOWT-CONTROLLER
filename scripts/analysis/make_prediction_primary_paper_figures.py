#!/usr/bin/env python3
"""Paper-style figures for prediction-primary ballast validation."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
FIG_DIR = OUT_ROOT / "paper_figures_prediction_primary_v1"

RUNS = {
    "Naive primary": OUT_ROOT / "prediction_primary_baseline_v1_5case_2h",
    "Bucket guard": OUT_ROOT / "prediction_primary_baseline_v1_5case_2h_guard_all_v1",
    "Risk gate": OUT_ROOT / "prediction_primary_baseline_v1_5case_2h_risk_gate_v1",
    "Hold-risk gate": OUT_ROOT / "prediction_primary_baseline_v1_5case_2h_hold_risk_gate_v1",
}

CASE_LABELS = {
    "01_onset_strong": "Onset",
    "04_decay_strong": "Decay",
    "06_signflip_high": "Sign flip",
    "08_lowrisk_quiet": "Low risk",
    "09_high_pressure_event": "Case 9",
}

COLORS = {
    "Closed baseline": "#1f4e79",
    "Naive primary": "#c7532c",
    "Bucket guard": "#7a5195",
    "Risk gate": "#228b7d",
    "Hold-risk gate": "#2f6f9f",
}


def _load_summaries() -> pd.DataFrame:
    rows = []
    for run_name, run_dir in RUNS.items():
        path = run_dir / "casebook_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["run"] = run_name
        rows.append(df)
    if not rows:
        raise FileNotFoundError("no casebook_summary.csv files found")
    out = pd.concat(rows, ignore_index=True)
    out["case_short"] = out["case_id"].map(CASE_LABELS).fillna(out["case_id"])
    out["pump_saving_pct"] = -out["d_pump_work_pct"].astype(float)
    out["attitude_penalty_p95_deg"] = np.maximum(
        out["d_pitch_p95"].astype(float), out["d_roll_p95"].astype(float)
    )
    out["fallback_pct"] = out["primary_safety_fallback_ratio"].astype(float) * 100.0
    return out


def _style_axis(ax) -> None:
    ax.grid(True, color="#d7dce2", lw=0.75, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")


def plot_case_matrix(summary: pd.DataFrame) -> Path:
    cases = list(CASE_LABELS)
    runs = list(RUNS)
    x = np.arange(len(cases), dtype=float)
    width = 0.18

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.5, 9.6),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 0.85]},
    )
    metrics = [
        ("pump_saving_pct", "Pump saving vs closed (%)", (-50, 100)),
        ("attitude_penalty_p95_deg", "Worst p95 attitude penalty (deg)", (0, 4.4)),
        ("fallback_pct", "Primary fallback time (%)", (0, 65)),
    ]
    for ax, (col, ylabel, ylim) in zip(axes, metrics):
        for i, run in enumerate(runs):
            vals = []
            for cid in cases:
                row = summary[(summary["case_id"].eq(cid)) & (summary["run"].eq(run))]
                vals.append(float(row[col].iloc[0]) if not row.empty else np.nan)
            offset = (i - (len(runs) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=run,
                color=COLORS[run],
                edgecolor="#ffffff",
                linewidth=0.8,
            )
        _style_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        if col == "pump_saving_pct":
            ax.axhline(0.0, color="#374151", lw=1.0)
            ax.text(0.0, 0.96, "higher is better", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        if col == "attitude_penalty_p95_deg":
            ax.axhline(0.2, color="#9c2f2f", lw=1.0, ls="--")
            ax.text(0.995, 0.2, "strict target 0.2 deg", ha="right", va="bottom", color="#9c2f2f", fontsize=8)
        if col == "fallback_pct":
            ax.axhline(20.0, color="#9c2f2f", lw=1.0, ls="--")
            ax.text(0.995, 20.0, "avg target 20%", ha="right", va="bottom", color="#9c2f2f", fontsize=8)
    axes[0].legend(ncol=4, frameon=False, loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([CASE_LABELS[c] for c in cases])
    fig.suptitle(
        "Prediction-primary variants: energy saving, attitude cost, and fallback burden",
        fontsize=14,
        y=1.02,
    )
    out = FIG_DIR / "fig_prediction_primary_case_matrix.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def _read_case_timeseries(run_dir: Path, case_prefix: str, variant: str) -> pd.DataFrame:
    ts_dir = run_dir / "timeseries"
    matches = sorted(ts_dir.glob(f"{case_prefix}_*_{variant}_timeseries.csv"))
    if not matches:
        raise FileNotFoundError(f"missing timeseries for {case_prefix} {variant}")
    return pd.read_csv(matches[0])


def plot_case9_mechanism() -> Path:
    run_dir = RUNS["Naive primary"]
    closed = _read_case_timeseries(run_dir, "09_high_pressure_event", "closed_only")
    primary = _read_case_timeseries(run_dir, "09_high_pressure_event", "prediction_primary_econ")
    log_matches = sorted((run_dir / "planner_logs").glob("09_high_pressure_event_*_planner_log.csv"))
    planner_log = pd.read_csv(log_matches[0]) if log_matches else pd.DataFrame()

    t = closed["t_s"].to_numpy(dtype=float) / 60.0
    tp = primary["t_s"].to_numpy(dtype=float) / 60.0
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(13.5, 10.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 0.9, 0.75]},
    )

    ax = axes[0]
    ws = primary.get("wind_speed", pd.Series(np.zeros(len(primary)))).to_numpy(dtype=float)
    wd = primary.get("wind_dir_deg", pd.Series(np.zeros(len(primary)))).to_numpy(dtype=float)
    ax.plot(tp, ws, color="#1d70b8", lw=1.7, label="wind speed")
    ax2 = ax.twinx()
    ax2.plot(tp, wd, color="#4b5563", lw=1.0, alpha=0.75, label="wind direction")
    ax.set_ylabel("Wind speed\n(m/s)")
    ax2.set_ylabel("Wind dir\n(deg)")
    _style_axis(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax.text(0.01, 0.88, "Replay wind blocks", transform=ax.transAxes, fontsize=9, color="#334155")

    series = [
        (axes[1], "pitch_deg", "Pitch angle (deg)"),
        (axes[2], "roll_deg", "Roll angle (deg)"),
    ]
    for ax, col, ylabel in series:
        ax.plot(t, closed[col].to_numpy(dtype=float), color=COLORS["Closed baseline"], lw=1.35, label="closed baseline")
        ax.plot(tp, primary[col].to_numpy(dtype=float), color=COLORS["Naive primary"], lw=1.35, label="prediction-primary")
        ax.axhline(0.0, color="#64748b", lw=0.8)
        ax.axhline(4.0, color="#b91c1c", lw=0.8, ls="--", alpha=0.85)
        ax.axhline(-4.0, color="#b91c1c", lw=0.8, ls="--", alpha=0.85)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    axes[1].legend(ncol=2, frameon=False, loc="upper left")

    ax = axes[3]
    ax.plot(t, closed["pump_total_rate_m3_min"].to_numpy(dtype=float), color=COLORS["Closed baseline"], lw=1.25)
    ax.plot(tp, primary["pump_total_rate_m3_min"].to_numpy(dtype=float), color=COLORS["Naive primary"], lw=1.25)
    ax.set_ylabel("Pump rate\n(m3/min)")
    _style_axis(ax)

    ax = axes[4]
    if not planner_log.empty:
        action_colors = {
            "hold": "#d9e2ec",
            "active_medium": "#f4a261",
            "active_small": "#e9c46a",
            "pump_saving": "#90be6d",
            "active_reverse_small": "#b56576",
        }
        for _, row in planner_log.iterrows():
            start = float(row["current_time_s"]) / 60.0
            end = start + 10.0
            action = str(row.get("first_action", "hold"))
            ax.axvspan(start, end, color=action_colors.get(action, "#cbd5e1"), alpha=0.9)
            ax.text(start + 5.0, 0.55, action.replace("_", "\n"), ha="center", va="center", fontsize=7)
    fallback = primary.get("preview_primary_safety_fallback", pd.Series(np.zeros(len(primary)))).to_numpy(dtype=float)
    if np.nanmax(fallback) > 0:
        ax.fill_between(tp, 0.0, 0.18, where=fallback > 0.5, color="#991b1b", alpha=0.75, step="post")
        ax.text(0.01, 0.08, "fallback", transform=ax.transAxes, fontsize=8, color="#991b1b")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("Planner\naction")
    ax.set_xlabel("Time (min)")
    _style_axis(ax)

    fig.suptitle("Case 9 mechanism: reduced pump activity with bounded pitch cost but residual roll penalty", fontsize=14, y=1.02)
    out = FIG_DIR / "fig_case09_mechanism.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_summary_table(summary: pd.DataFrame) -> Path:
    cols = [
        "run",
        "case_id",
        "closed_pump_work_m3",
        "primary_pump_work_m3",
        "d_pump_work_pct",
        "d_pitch_p95",
        "d_roll_p95",
        "primary_safety_fallback_ratio",
        "closed_latch_switches",
        "primary_latch_switches",
    ]
    out = FIG_DIR / "prediction_primary_variant_summary.csv"
    summary[cols].to_csv(out, index=False)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summaries()
    paths = [plot_case_matrix(summary), plot_case9_mechanism(), write_summary_table(summary)]
    report = FIG_DIR / "figure_manifest.md"
    report.write_text(
        "\n".join(
            [
                "# Prediction-primary Paper Figures v1",
                "",
                *[f"- {p.relative_to(REPO_ROOT)}" for p in paths],
                "",
                "Figure style follows recent FOWT control papers: multi-metric comparisons, time-domain response, and control-action/fallback context rather than a single improvement percentage.",
            ]
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
