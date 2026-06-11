#!/usr/bin/env python3
"""Diagnostic figures for the frozen no-preview closed pump behavior."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
AUDIT_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "reactive_closed_pump_saving_space_v1"
FIG_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "closed_pump_diagnostics_v1"


CASE_LABELS = {
    "01_onset_strong": "01 onset strong",
    "02_onset_signflip": "02 onset signflip",
    "03_onset_moderate": "03 onset moderate",
    "04_decay_strong": "04 decay strong",
    "05_decay_signflip": "05 decay signflip",
    "06_signflip_high": "06 signflip high",
    "07_signflip_sustained": "07 signflip sustained",
    "08_lowrisk_quiet": "08 low-risk quiet",
    "09_high_pressure_event": "09 high pressure",
    "10_residual_high": "10 residual high",
}

COLORS = {
    "pitch": "#1f4e79",
    "roll": "#c7532c",
    "pump_total": "#1b7f79",
    "tank1": "#2f6f9f",
    "tank2": "#d08c2f",
    "tank3": "#7a5195",
    "target": "#5f6b7a",
    "err": "#9c2f2f",
}


def _style_axis(ax) -> None:
    ax.grid(True, color="#d7dce2", lw=0.75, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa4ad")
    ax.spines["bottom"].set_color("#9aa4ad")


def _case_id_from_path(path: Path) -> str:
    name = path.name
    for cid in CASE_LABELS:
        if name.startswith(f"{cid}_"):
            return cid
    return name.replace("_closed_only_timeseries.csv", "")


def _load_closed_traces() -> list[tuple[str, Path, pd.DataFrame]]:
    rows = []
    for path in sorted(TRACE_DIR.glob("*_closed_only_timeseries.csv")):
        cid = _case_id_from_path(path)
        if cid not in CASE_LABELS:
            continue
        rows.append((cid, path, pd.read_csv(path)))
    if not rows:
        raise FileNotFoundError(f"no closed-only traces found in {TRACE_DIR}")
    return rows


def _work_m3(pump: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(pump, dtype=float)), dx=1.0) / 60.0)


def _rolling_median(values: np.ndarray, dt_s: float, window_s: float = 30.0) -> np.ndarray:
    window = max(1, int(round(float(window_s) / max(float(dt_s), 1e-9))))
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def _visible_stage_idx_changes(df: pd.DataFrame) -> int:
    count = 0
    for col in ("pump_stage_idx1", "pump_stage_idx2", "pump_stage_idx3"):
        if col in df.columns:
            vals = df[col].to_numpy(dtype=float)
            if vals.size > 1:
                count += int(np.sum(np.diff(vals) != 0.0))
    return count


def _rate_tv_per_min(df: pd.DataFrame) -> float:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    t = df["t_s"].to_numpy(dtype=float)
    duration_min = max(float(t[-1] - t[0]) / 60.0, 1e-9) if t.size > 1 else 1e-9
    return float(np.sum(np.abs(np.diff(pump))) / duration_min)


def _pump_starts(df: pd.DataFrame) -> int:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    active = np.abs(pump) > 1e-6
    if active.size <= 1:
        return int(active[0]) if active.size else 0
    return int(np.sum(active[1:] & ~active[:-1]))


def _summary_row(case_id: str, df: pd.DataFrame) -> dict:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    return {
        "case_id": case_id,
        "pump_work_m3": _work_m3(pump),
        "pump_duty_pct": float(np.mean(np.abs(pump) > 1e-6) * 100.0),
        "pump_latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
        "pump_stage_switches": int(df["pump_stage_switch_count"].iloc[-1])
        if "pump_stage_switch_count" in df.columns
        else 0,
        "visible_stage_idx_changes": _visible_stage_idx_changes(df),
        "pump_total_starts": _pump_starts(df),
        "rate_tv_per_min": _rate_tv_per_min(df),
        "pitch_abs_p95": float(np.percentile(np.abs(df["pitch_deg"].to_numpy(dtype=float)), 95)),
        "roll_abs_p95": float(np.percentile(np.abs(df["roll_deg"].to_numpy(dtype=float)), 95)),
        "pump_backlog_p95_kg": float(np.percentile(df["pump_total_backlog_kg"].to_numpy(dtype=float), 95))
        if "pump_total_backlog_kg" in df.columns
        else np.nan,
    }


def plot_case_detail(case_id: str, df: pd.DataFrame) -> Path:
    label = CASE_LABELS.get(case_id, case_id)
    t = df["t_s"].to_numpy(dtype=float) / 60.0
    dt_s = float(np.nanmedian(np.diff(df["t_s"].to_numpy(dtype=float)))) if len(df) > 1 else 1.0
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14.0, 13.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [0.85, 1.0, 0.95, 1.05, 1.05, 0.75, 0.75]},
    )

    ax = axes[0]
    ax.plot(t, df["wind_speed"], color="#1d70b8", lw=1.5)
    ax.set_ylabel("Wind speed\n(m/s)")
    ax2 = ax.twinx()
    ax2.plot(t, df["wind_dir_deg"], color="#45515f", lw=1.0, alpha=0.78)
    ax2.set_ylabel("Wind dir\n(deg)")
    _style_axis(ax)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax = axes[1]
    ax.plot(t, df["pitch_deg"], color=COLORS["pitch"], lw=1.2, label="pitch")
    ax.plot(t, df["roll_deg"], color=COLORS["roll"], lw=1.2, label="roll")
    ax.axhline(0.0, color="#64748b", lw=0.8)
    ax.axhline(4.0, color="#9c2f2f", lw=0.8, ls="--", alpha=0.75)
    ax.axhline(-4.0, color="#9c2f2f", lw=0.8, ls="--", alpha=0.75)
    ax.set_ylabel("Attitude\n(deg)")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    _style_axis(ax)

    ax = axes[2]
    pump_total = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    ax.plot(t, pump_total, color=COLORS["pump_total"], lw=0.55, alpha=0.24, label="total raw")
    ax.plot(
        t,
        _rolling_median(pump_total, dt_s=dt_s, window_s=30.0),
        color=COLORS["pump_total"],
        lw=1.55,
        label="total 30s median",
    )
    for col, color, label_i in [
        ("pump_rate1_m3min", COLORS["tank1"], "tank 1"),
        ("pump_rate2_m3min", COLORS["tank2"], "tank 2"),
        ("pump_rate3_m3min", COLORS["tank3"], "tank 3"),
    ]:
        ax.plot(t, df[col], color=color, lw=0.9, alpha=0.82, label=label_i)
    ax.set_ylabel("Pump rate\n(m3/min)")
    ax.legend(ncol=4, frameon=False, loc="upper left")
    _style_axis(ax)

    ax = axes[3]
    for tank, color in [(1, COLORS["tank1"]), (2, COLORS["tank2"]), (3, COLORS["tank3"])]:
        ax.plot(t, df[f"tank{tank}_kg"] / 1000.0, color=color, lw=1.0, label=f"tank {tank}")
        ax.plot(t, df[f"target_tank{tank}_kg"] / 1000.0, color=color, lw=0.8, ls="--", alpha=0.65)
    ax.set_ylabel("Mass and target\n(t)")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    _style_axis(ax)

    ax = axes[4]
    for tank, color in [(1, COLORS["tank1"]), (2, COLORS["tank2"]), (3, COLORS["tank3"])]:
        ax.plot(t, df[f"err_tank{tank}_kg"] / 1000.0, color=color, lw=0.95, label=f"err {tank}")
    ax.axhline(0.0, color="#64748b", lw=0.8)
    ax.axhline(0.5, color="#9c2f2f", lw=0.7, ls=":", alpha=0.75)
    ax.axhline(-0.5, color="#9c2f2f", lw=0.7, ls=":", alpha=0.75)
    ax.set_ylabel("Tank error\n(t)")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    _style_axis(ax)

    ax = axes[5]
    if "pump_total_backlog_kg" in df.columns:
        ax.plot(t, df["pump_total_backlog_kg"] / 1000.0, color="#5f6b7a", lw=1.0, label="total backlog")
    if "cmd_gap_kg" in df.columns:
        ax.plot(t, df["cmd_gap_kg"] / 1000.0, color="#9c2f2f", lw=0.95, alpha=0.85, label="cmd gap")
    ax.set_ylabel("Backlog / gap\n(t)")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    _style_axis(ax)

    ax = axes[6]
    offset = 0.0
    for tank, color in [(1, COLORS["tank1"]), (2, COLORS["tank2"]), (3, COLORS["tank3"])]:
        if f"pump_latched{tank}" in df.columns:
            ax.fill_between(
                t,
                offset,
                offset + 0.8,
                where=df[f"pump_latched{tank}"].to_numpy(dtype=float) > 0.5,
                color=color,
                alpha=0.75,
                step="post",
            )
            ax.text(t[0], offset + 0.4, f"L{tank}", va="center", ha="right", fontsize=8)
        offset += 1.0
    if "pump_fullspeed_any" in df.columns:
        ax.fill_between(
            t,
            offset,
            offset + 0.8,
            where=df["pump_fullspeed_any"].to_numpy(dtype=float) > 0.5,
            color="#991b1b",
            alpha=0.55,
            step="post",
        )
        ax.text(t[0], offset + 0.4, "full", va="center", ha="right", fontsize=8)
    ax.set_ylim(-0.05, 3.95)
    ax.set_yticks([])
    ax.set_ylabel("Latch / full")
    ax.set_xlabel("Time (min)")
    _style_axis(ax)

    work = _work_m3(df["pump_total_rate_m3_min"].to_numpy(dtype=float))
    switches = int(df["pump_latch_switch_count"].iloc[-1]) if "pump_latch_switch_count" in df.columns else 0
    fig.suptitle(
        f"{label}: frozen closed pump diagnostics | work={work:.1f} m3, latch switches={switches}",
        fontsize=13,
        y=1.01,
    )
    out = FIG_DIR / f"{case_id}_closed_pump_diagnostics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overview(summary: pd.DataFrame) -> Path:
    summary = summary.copy().sort_values("case_id")
    x = np.arange(len(summary))
    labels = [CASE_LABELS.get(cid, cid) for cid in summary["case_id"]]

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(14.0, 12.2),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.8, 0.9, 0.9, 0.85]},
    )
    axes[0].bar(x, summary["pump_work_m3"], color="#1b7f79")
    axes[0].set_ylabel("Pump work\n(m3)")
    _style_axis(axes[0])

    axes[1].bar(x, summary["pump_duty_pct"], color="#2f6f9f")
    axes[1].set_ylabel("Pump duty\n(%)")
    _style_axis(axes[1])

    axes[2].bar(x - 0.18, summary["pump_latch_switches"], width=0.36, color="#d08c2f", label="latch")
    axes[2].bar(
        x + 0.18,
        summary["visible_stage_idx_changes"],
        width=0.36,
        color="#7a5195",
        label="visible stage idx",
    )
    axes[2].set_ylabel("Switch count")
    axes[2].legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axes[2])

    axes[3].bar(x, summary["rate_tv_per_min"], color="#5f6b7a")
    axes[3].set_ylabel("Pump-rate\nTV/min")
    _style_axis(axes[3])

    axes[4].bar(x - 0.18, summary["pitch_abs_p95"], width=0.36, color=COLORS["pitch"], label="pitch p95")
    axes[4].bar(x + 0.18, summary["roll_abs_p95"], width=0.36, color=COLORS["roll"], label="roll p95")
    axes[4].axhline(4.0, color="#9c2f2f", lw=0.8, ls="--")
    axes[4].set_ylabel("Attitude p95\n(deg)")
    axes[4].legend(frameon=False, ncol=2, loc="upper left")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(labels, rotation=38, ha="right")
    _style_axis(axes[4])

    fig.suptitle("Frozen closed baseline: pump burden and attitude overview", fontsize=13, y=1.01)
    out = FIG_DIR / "closed_pump_diagnostic_overview.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_low_risk_start_audit(summary: pd.DataFrame) -> Path | None:
    audit_path = AUDIT_DIR / "reactive_pump_saving_space_by_case.csv"
    if not audit_path.exists():
        return None
    audit = pd.read_csv(audit_path).sort_values("case_id")
    x = np.arange(len(audit))
    labels = [CASE_LABELS.get(cid, cid) for cid in audit["case_id"]]

    fig, axes = plt.subplots(2, 1, figsize=(14.0, 6.8), sharex=True, constrained_layout=True)
    axes[0].bar(x - 0.18, audit["avoidable_proxy_5min_work_pct"], width=0.36, color="#2f6f9f", label="5 min proxy")
    axes[0].bar(x + 0.18, audit["avoidable_proxy_10min_work_pct"], width=0.36, color="#7a5195", label="10 min proxy")
    axes[0].axhline(10.0, color="#9c2f2f", lw=0.9, ls="--")
    axes[0].set_ylabel("Proxy work\n(% case pump)")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    _style_axis(axes[0])

    axes[1].bar(x - 0.18, audit["avoidable_proxy_5min_events"], width=0.36, color="#2f6f9f", label="5 min")
    axes[1].bar(x + 0.18, audit["pump_start_events"], width=0.36, color="#d08c2f", label="all starts")
    axes[1].set_ylabel("Start count")
    axes[1].legend(frameon=False, ncol=2, loc="upper left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=38, ha="right")
    _style_axis(axes[1])
    fig.suptitle("Reactive closed low-risk pump-start proxy audit", fontsize=13, y=1.01)
    out = FIG_DIR / "closed_pump_low_risk_start_audit.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    traces = _load_closed_traces()
    rows = []
    paths = []
    for case_id, _, df in traces:
        rows.append(_summary_row(case_id, df))
        paths.append(plot_case_detail(case_id, df))
    summary = pd.DataFrame(rows).sort_values("case_id")
    summary.to_csv(FIG_DIR / "closed_pump_diagnostic_summary.csv", index=False)
    paths.insert(0, plot_overview(summary))
    audit_path = plot_low_risk_start_audit(summary)
    if audit_path is not None:
        paths.insert(1, audit_path)

    manifest = [
        "# Closed Pump Diagnostic Figures",
        "",
        "Source traces:",
        f"- `{TRACE_DIR.relative_to(REPO_ROOT)}`",
        "",
        "Files:",
        "",
        *[f"- `{p.relative_to(REPO_ROOT)}`" for p in paths],
        f"- `{(FIG_DIR / 'closed_pump_diagnostic_summary.csv').relative_to(REPO_ROOT)}`",
    ]
    (FIG_DIR / "figure_manifest.md").write_text("\n".join(manifest), encoding="utf-8")
    print(FIG_DIR)


if __name__ == "__main__":
    main()
