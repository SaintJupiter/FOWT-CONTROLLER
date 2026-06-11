#!/usr/bin/env python3
"""Paper-style 10-case figures for prediction-primary validation.

The figures are intentionally multi-metric: a predictive controller that saves
pump work but increases attitude cost should be visible as a trade-off, not as a
single percentage number.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
FIG_DIR = OUT_ROOT / "paper_figures_prediction_primary_10case_v1"

RUNS = {
    "Naive primary": OUT_ROOT / "prediction_primary_baseline_v1_10case_2h_naive",
    "Hold-risk gate": OUT_ROOT / "prediction_primary_baseline_v1_10case_2h_hold_risk_gate",
}

CASE_LABELS = {
    "01_onset_strong": "01 onset",
    "02_onset_signflip": "02 onset flip",
    "03_onset_moderate": "03 onset mod.",
    "04_decay_strong": "04 decay",
    "05_decay_signflip": "05 decay flip",
    "06_signflip_high": "06 sign flip",
    "07_signflip_sustained": "07 sustained",
    "08_lowrisk_quiet": "08 quiet",
    "09_high_pressure_event": "09 high pressure",
    "10_residual_high": "10 residual",
}

CASE_GROUPS = {
    "01_onset_strong": "onset",
    "02_onset_signflip": "onset",
    "03_onset_moderate": "onset",
    "04_decay_strong": "decay",
    "05_decay_signflip": "decay",
    "06_signflip_high": "sign flip",
    "07_signflip_sustained": "sign flip",
    "08_lowrisk_quiet": "quiet",
    "09_high_pressure_event": "high pressure",
    "10_residual_high": "residual",
}

GROUP_COLORS = {
    "onset": "#c7532c",
    "decay": "#7a5195",
    "sign flip": "#1f78b4",
    "quiet": "#7a8f3a",
    "high pressure": "#d08c2f",
    "residual": "#228b7d",
}

RUN_COLORS = {
    "Closed baseline": "#1f4e79",
    "Naive primary": "#c7532c",
    "Hold-risk gate": "#2f6f9f",
}


def _style_axis(ax) -> None:
    ax.grid(True, color="#d7dce2", lw=0.75, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")


def _load_summaries() -> pd.DataFrame:
    rows = []
    for run, run_dir in RUNS.items():
        path = run_dir / "casebook_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["run"] = run
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out["case_label"] = out["case_id"].map(CASE_LABELS).fillna(out["case_id"])
    out["case_group"] = out["case_id"].map(CASE_GROUPS).fillna("other")
    out["pump_saving_pct"] = -out["d_pump_work_pct"].astype(float)
    out["attitude_penalty_p95_deg"] = np.maximum(
        out["d_pitch_p95"].astype(float), out["d_roll_p95"].astype(float)
    )
    out["fallback_pct"] = out["primary_safety_fallback_ratio"].astype(float) * 100.0
    out["switch_reduction_pct"] = (
        1.0
        - out["primary_latch_switches"].astype(float)
        / out["closed_latch_switches"].replace(0, np.nan).astype(float)
    ) * 100.0
    out.loc[out["closed_latch_switches"].astype(float) == 0.0, "switch_reduction_pct"] = 0.0
    return out


def plot_tradeoff_map(summary: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.8, 7.4), constrained_layout=True)

    paired = {
        case: summary[summary["case_id"].eq(case)].set_index("run")
        for case in CASE_LABELS
        if len(summary[summary["case_id"].eq(case)]) == len(RUNS)
    }

    for case, df in paired.items():
        n = df.loc["Naive primary"]
        h = df.loc["Hold-risk gate"]
        ax.annotate(
            "",
            xy=(h["pump_saving_pct"], h["attitude_penalty_p95_deg"]),
            xytext=(n["pump_saving_pct"], n["attitude_penalty_p95_deg"]),
            arrowprops={
                "arrowstyle": "->",
                "lw": 1.1,
                "color": "#5f6b7a",
                "alpha": 0.65,
                "shrinkA": 6,
                "shrinkB": 6,
            },
        )

    for run, marker in [("Naive primary", "o"), ("Hold-risk gate", "s")]:
        sub = summary[summary["run"].eq(run)]
        for group, group_df in sub.groupby("case_group"):
            sizes = 70.0 + 7.0 * group_df["fallback_pct"].clip(0, 55)
            ax.scatter(
                group_df["pump_saving_pct"],
                group_df["attitude_penalty_p95_deg"],
                s=sizes,
                marker=marker,
                c=GROUP_COLORS.get(group, "#607d8b"),
                edgecolors="white",
                linewidths=0.9,
                alpha=0.92,
                label=f"{run} / {group}",
            )
        for _, row in sub.iterrows():
            ax.text(
                row["pump_saving_pct"] + 1.6,
                row["attitude_penalty_p95_deg"] + 0.035,
                str(row["case_label"]).split()[0],
                fontsize=8,
                color="#263238",
            )

    ax.axvline(0.0, color="#374151", lw=1.0)
    ax.axhline(0.2, color="#9c2f2f", lw=1.0, ls="--")
    ax.axhline(1.0, color="#c47f1a", lw=0.9, ls=":")
    ax.text(0.01, 0.97, "better pump economy", transform=ax.transAxes, fontsize=9, va="top")
    ax.text(
        0.99,
        0.2,
        "strict attitude target",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#9c2f2f",
    )
    ax.set_xlim(-45, 100)
    ax.set_ylim(-0.35, max(4.2, float(summary["attitude_penalty_p95_deg"].max()) + 0.45))
    ax.set_xlabel("Pump-work saving against frozen closed baseline (%)")
    ax.set_ylabel("Worst p95 pitch/roll penalty (deg)")
    ax.set_title("Prediction-primary trade-off map across ten 2 h wind windows", fontsize=13)
    _style_axis(ax)

    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#777", markeredgecolor="white", markersize=8, label="Naive primary"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#777", markeredgecolor="white", markersize=8, label="Hold-risk gate"),
    ]
    group_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="white", markersize=8, label=group)
        for group, color in GROUP_COLORS.items()
    ]
    ax.legend(handles=handles + group_handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11))

    out = FIG_DIR / "fig_10case_tradeoff_map.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metric_heatmap(summary: pd.DataFrame) -> Path:
    cases = list(CASE_LABELS)
    metrics = [
        ("pump_saving_pct", "pump saving (%)", -100, 100, 0),
        ("d_pitch_p95", "pitch p95 delta (deg)", -4, 4, 0),
        ("d_roll_p95", "roll p95 delta (deg)", -4, 4, 0),
        ("fallback_pct", "fallback time (%)", 0, 55, 20),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(RUNS),
        figsize=(14.8, 8.6),
        sharex=True,
        constrained_layout=True,
    )
    for r, (metric, ylabel, vmin, vmax, center) in enumerate(metrics):
        for c, run in enumerate(RUNS):
            ax = axes[r, c]
            sub = summary[summary["run"].eq(run)].set_index("case_id")
            values = np.array([[float(sub.loc[case, metric]) for case in cases]])
            if metric == "pump_saving_pct":
                cmap = "RdYlGn"
            elif metric == "fallback_pct":
                cmap = "YlOrRd"
            else:
                cmap = "RdBu_r"
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax) if vmin < center < vmax else None
            im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
            for i, case in enumerate(cases):
                val = values[0, i]
                color = "white" if abs(val - center) > (vmax - vmin) * 0.28 else "#172033"
                if metric == "pump_saving_pct":
                    text = f"{val:+.0f}"
                elif metric == "fallback_pct":
                    text = f"{val:.0f}"
                else:
                    text = f"{val:+.1f}"
                ax.text(i, 0, text, ha="center", va="center", fontsize=8, color=color)
            ax.set_yticks([])
            ax.set_xticks(np.arange(len(cases)))
            if r == len(metrics) - 1:
                ax.set_xticklabels([CASE_LABELS[case] for case in cases], rotation=42, ha="right")
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(ylabel)
            if r == 0:
                ax.set_title(run)
            for spine in ax.spines.values():
                spine.set_color("#c8d0da")
            fig.colorbar(im, ax=ax, fraction=0.024, pad=0.015)

    fig.suptitle("Ten-case matrix: energy, attitude, and supervisor burden", fontsize=13, y=1.02)
    out = FIG_DIR / "fig_10case_metric_heatmap.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def _read_case_timeseries(run_dir: Path, case_prefix: str, variant: str) -> pd.DataFrame:
    matches = sorted((run_dir / "timeseries").glob(f"{case_prefix}_*_{variant}_timeseries.csv"))
    if not matches:
        raise FileNotFoundError(f"missing timeseries: {case_prefix} {variant}")
    return pd.read_csv(matches[0])


def _read_planner_log(run_dir: Path, case_prefix: str) -> pd.DataFrame:
    matches = sorted((run_dir / "planner_logs").glob(f"{case_prefix}_*_planner_log.csv"))
    return pd.read_csv(matches[0]) if matches else pd.DataFrame()


def _draw_action_strip(ax, planner_log: pd.DataFrame, ts: pd.DataFrame, label: str) -> None:
    action_colors = {
        "hold": "#d8e1ea",
        "active_medium": "#f4a261",
        "active_small": "#e9c46a",
        "pump_saving": "#90be6d",
        "active_reverse_small": "#b56576",
    }
    if not planner_log.empty:
        for _, row in planner_log.iterrows():
            start = float(row["current_time_s"]) / 60.0
            end = start + 10.0
            action = str(row.get("first_action", "hold"))
            ax.axvspan(start, end, color=action_colors.get(action, "#cbd5e1"), alpha=0.92)
    fallback = ts.get("preview_primary_safety_fallback", pd.Series(np.zeros(len(ts)))).to_numpy(dtype=float)
    t = ts["t_s"].to_numpy(dtype=float) / 60.0
    if np.nanmax(fallback) > 0:
        ax.fill_between(t, 0.0, 0.22, where=fallback > 0.5, color="#991b1b", alpha=0.78, step="post")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.text(0.01, 0.76, label, transform=ax.transAxes, fontsize=9, color="#172033")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#9aa4ad")


def plot_case_mechanism(case_prefix: str, title: str, filename: str) -> Path:
    naive_dir = RUNS["Naive primary"]
    hold_dir = RUNS["Hold-risk gate"]
    closed = _read_case_timeseries(naive_dir, case_prefix, "closed_only")
    naive = _read_case_timeseries(naive_dir, case_prefix, "prediction_primary_econ")
    hold = _read_case_timeseries(hold_dir, case_prefix, "prediction_primary_econ")

    t = closed["t_s"].to_numpy(dtype=float) / 60.0
    tn = naive["t_s"].to_numpy(dtype=float) / 60.0
    th = hold["t_s"].to_numpy(dtype=float) / 60.0

    fig, axes = plt.subplots(
        6,
        1,
        figsize=(13.8, 11.4),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.9, 1.0, 1.0, 0.9, 0.45, 0.45]},
    )

    ax = axes[0]
    wind_speed = naive.get("wind_speed", pd.Series(np.zeros(len(naive)))).to_numpy(dtype=float)
    wind_dir = naive.get("wind_dir_deg", pd.Series(np.zeros(len(naive)))).to_numpy(dtype=float)
    ax.plot(tn, wind_speed, color="#1d70b8", lw=1.6)
    ax.set_ylabel("Wind speed\n(m/s)")
    ax2 = ax.twinx()
    ax2.plot(tn, wind_dir, color="#45515f", lw=1.0, alpha=0.78)
    ax2.set_ylabel("Wind dir\n(deg)")
    _style_axis(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    for ax, col, ylabel in [
        (axes[1], "pitch_deg", "Pitch angle (deg)"),
        (axes[2], "roll_deg", "Roll angle (deg)"),
    ]:
        ax.plot(t, closed[col].to_numpy(dtype=float), color=RUN_COLORS["Closed baseline"], lw=1.25, label="closed baseline")
        ax.plot(tn, naive[col].to_numpy(dtype=float), color=RUN_COLORS["Naive primary"], lw=1.25, label="naive primary")
        ax.plot(th, hold[col].to_numpy(dtype=float), color=RUN_COLORS["Hold-risk gate"], lw=1.25, label="hold-risk gate")
        ax.axhline(0.0, color="#64748b", lw=0.8)
        ax.axhline(4.0, color="#9c2f2f", lw=0.8, ls="--", alpha=0.75)
        ax.axhline(-4.0, color="#9c2f2f", lw=0.8, ls="--", alpha=0.75)
        ax.set_ylabel(ylabel)
        _style_axis(ax)
    axes[1].legend(ncol=3, frameon=False, loc="upper left")

    ax = axes[3]
    ax.plot(t, closed["pump_total_rate_m3_min"].to_numpy(dtype=float), color=RUN_COLORS["Closed baseline"], lw=1.15)
    ax.plot(tn, naive["pump_total_rate_m3_min"].to_numpy(dtype=float), color=RUN_COLORS["Naive primary"], lw=1.15)
    ax.plot(th, hold["pump_total_rate_m3_min"].to_numpy(dtype=float), color=RUN_COLORS["Hold-risk gate"], lw=1.15)
    ax.set_ylabel("Pump rate\n(m3/min)")
    _style_axis(ax)

    _draw_action_strip(axes[4], _read_planner_log(naive_dir, case_prefix), naive, "naive primary actions")
    _draw_action_strip(axes[5], _read_planner_log(hold_dir, case_prefix), hold, "hold-risk actions + fallback")
    axes[5].set_xlabel("Time (min)")

    fig.suptitle(title, fontsize=13, y=1.02)
    out = FIG_DIR / filename
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def write_tables(summary: pd.DataFrame) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "run",
        "case_id",
        "case_group",
        "pump_saving_pct",
        "d_pitch_p95",
        "d_roll_p95",
        "attitude_penalty_p95_deg",
        "fallback_pct",
        "switch_reduction_pct",
        "closed_pump_work_m3",
        "primary_pump_work_m3",
    ]
    out_csv = FIG_DIR / "prediction_primary_10case_summary.csv"
    summary[cols].to_csv(out_csv, index=False)

    grouped = (
        summary.groupby("run")
        .agg(
            mean_pump_saving_pct=("pump_saving_pct", "mean"),
            median_pump_saving_pct=("pump_saving_pct", "median"),
            mean_attitude_penalty_deg=("attitude_penalty_p95_deg", "mean"),
            max_attitude_penalty_deg=("attitude_penalty_p95_deg", "max"),
            mean_fallback_pct=("fallback_pct", "mean"),
            cases_with_positive_saving=("pump_saving_pct", lambda x: int((x > 0).sum())),
            cases_with_attitude_penalty_below_1deg=("attitude_penalty_p95_deg", lambda x: int((x < 1.0).sum())),
        )
        .reset_index()
    )
    out_grouped = FIG_DIR / "prediction_primary_10case_grouped_metrics.csv"
    grouped.to_csv(out_grouped, index=False)
    return [out_csv, out_grouped]


def write_manifest(paths: list[Path], summary: pd.DataFrame) -> Path:
    grouped = (
        summary.groupby("run")
        .agg(
            mean_pump=("pump_saving_pct", "mean"),
            median_pump=("pump_saving_pct", "median"),
            mean_penalty=("attitude_penalty_p95_deg", "mean"),
            mean_fallback=("fallback_pct", "mean"),
        )
        .round(2)
    )
    metric_table = grouped.reset_index()
    table_lines = [
        "| run | mean pump saving (%) | median pump saving (%) | mean attitude penalty (deg) | mean fallback (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in metric_table.iterrows():
        table_lines.append(
            "| {run} | {mean_pump:.2f} | {median_pump:.2f} | {mean_penalty:.2f} | {mean_fallback:.2f} |".format(
                run=row["run"],
                mean_pump=float(row["mean_pump"]),
                median_pump=float(row["median_pump"]),
                mean_penalty=float(row["mean_penalty"]),
                mean_fallback=float(row["mean_fallback"]),
            )
        )

    lines = [
        "# Prediction-primary 10-case figure pack",
        "",
        "## Files",
        "",
        *[f"- `{p.relative_to(REPO_ROOT)}`" for p in paths],
        "",
        "## Aggregate metrics",
        "",
        *table_lines,
        "",
        "## Reading notes",
        "",
        "- `Naive primary` is the high-saving prediction-planning upper case; it should not be sold as a final robust controller.",
        "- `Hold-risk gate` is a conservative supervisor check; it improves case 09 roll penalty but can erase savings in fast sign-flip/decay windows.",
        "- The main technical bottleneck is now planner-level risk-envelope selection, not the frozen closed baseline.",
        "- Figure structure follows common FOWT-control reporting practice: multi-case metric matrices plus time-domain response and actuator/supervisor context.",
    ]
    out = FIG_DIR / "figure_manifest.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary = _load_summaries()
    paths = [
        plot_tradeoff_map(summary),
        plot_metric_heatmap(summary),
        plot_case_mechanism(
            "09_high_pressure_event",
            "Case 09: supervisor reduces roll penalty while retaining partial pump saving",
            "fig_case09_supervisor_mechanism.png",
        ),
        plot_case_mechanism(
            "04_decay_strong",
            "Case 04 boundary: fallback burden rises but planner risk is not fully resolved",
            "fig_case04_boundary_mechanism.png",
        ),
    ]
    paths.extend(write_tables(summary))
    paths.append(write_manifest(paths, summary))
    print(FIG_DIR)


if __name__ == "__main__":
    main()
