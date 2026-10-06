#!/usr/bin/env python3
"""Plot pitch/roll separated diagnostics for selected smooth180 casebook runs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _series(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)


def _pump_work_m3(df: pd.DataFrame) -> float:
    t = _series(df, "t_s")
    pump = _series(df, "pump_total_rate_m3_min")
    if len(t) < 2:
        return 0.0
    dt = np.diff(t, prepend=t[0])
    dt[0] = np.nanmedian(dt[1:]) if len(dt) > 1 else 1.0
    return float(np.nansum(pump * dt / 60.0))


def _latch_switches(df: pd.DataFrame) -> float:
    if "pump_latch_switch_count" not in df.columns:
        return math.nan
    return float(pd.to_numeric(df["pump_latch_switch_count"], errors="coerce").max())


def _time_over(axis: np.ndarray, threshold: float, dt_s: float = 1.0) -> float:
    return float(np.nansum(axis > threshold) * dt_s)


def _case_key(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) < 3:
        return path.stem
    return "_".join(parts[1:3])


def _build_pairs(timeseries_dir: Path) -> list[tuple[str, Path, Path]]:
    closed = sorted(timeseries_dir.glob("*_closed_only_timeseries.csv"))
    pairs: list[tuple[str, Path, Path]] = []
    for closed_path in closed:
        prefix = closed_path.name.replace("_closed_only_timeseries.csv", "")
        matches = sorted(timeseries_dir.glob(f"{prefix}_*_timeseries.csv"))
        primary = [p for p in matches if "_closed_only_" not in p.name]
        if len(primary) != 1:
            raise RuntimeError(f"Expected one primary timeseries for {closed_path.name}, found {len(primary)}")
        pairs.append((_case_key(closed_path), closed_path, primary[0]))
    return pairs


def _summary_row(case_key: str, closed: pd.DataFrame, primary: pd.DataFrame) -> dict[str, float | str]:
    n = min(len(closed), len(primary))
    closed = closed.iloc[:n].reset_index(drop=True)
    primary = primary.iloc[:n].reset_index(drop=True)
    cp = np.abs(_series(closed, "pitch_deg"))
    cr = np.abs(_series(closed, "roll_deg"))
    pp = np.abs(_series(primary, "pitch_deg"))
    pr = np.abs(_series(primary, "roll_deg"))
    ca = np.maximum(cp, cr)
    pa = np.maximum(pp, pr)
    d_pitch_p95 = float(np.nanpercentile(pp, 95) - np.nanpercentile(cp, 95))
    d_roll_p95 = float(np.nanpercentile(pr, 95) - np.nanpercentile(cr, 95))
    worse_axis = "pitch" if d_pitch_p95 >= d_roll_p95 else "roll"
    closed_pump = _pump_work_m3(closed)
    primary_pump = _pump_work_m3(primary)
    pump_change_pct = (
        (primary_pump - closed_pump) / closed_pump * 100.0 if abs(closed_pump) > 1e-9 else math.nan
    )
    closed_latch = _latch_switches(closed)
    primary_latch = _latch_switches(primary)
    latch_reduction_pct = (
        (closed_latch - primary_latch) / closed_latch * 100.0
        if not math.isnan(closed_latch) and abs(closed_latch) > 1e-9
        else math.nan
    )
    return {
        "case_key": case_key,
        "pump_change_pct": pump_change_pct,
        "latch_reduction_pct": latch_reduction_pct,
        "closed_pitch_p95": float(np.nanpercentile(cp, 95)),
        "primary_pitch_p95": float(np.nanpercentile(pp, 95)),
        "d_pitch_p95": d_pitch_p95,
        "closed_roll_p95": float(np.nanpercentile(cr, 95)),
        "primary_roll_p95": float(np.nanpercentile(pr, 95)),
        "d_roll_p95": d_roll_p95,
        "worse_p95_axis": worse_axis,
        "d_worse_axis_p95": max(d_pitch_p95, d_roll_p95),
        "closed_axis_p95": float(np.nanpercentile(ca, 95)),
        "primary_axis_p95": float(np.nanpercentile(pa, 95)),
        "d_axis_p95": float(np.nanpercentile(pa, 95) - np.nanpercentile(ca, 95)),
        "d_time_over_5_s": _time_over(pa, 5.0) - _time_over(ca, 5.0),
        "d_time_over_7p5_s": _time_over(pa, 7.5) - _time_over(ca, 7.5),
        "d_time_over_10_s": _time_over(pa, 10.0) - _time_over(ca, 10.0),
    }


def _plot_case(case_key: str, closed: pd.DataFrame, primary: pd.DataFrame, row: dict[str, float | str], out: Path) -> None:
    n = min(len(closed), len(primary))
    closed = closed.iloc[:n].reset_index(drop=True)
    primary = primary.iloc[:n].reset_index(drop=True)
    t_h = _series(primary, "t_s") / 3600.0
    cp = _series(closed, "pitch_deg")
    cr = _series(closed, "roll_deg")
    pp = _series(primary, "pitch_deg")
    pr = _series(primary, "roll_deg")
    ca = np.maximum(np.abs(cp), np.abs(cr))
    pa = np.maximum(np.abs(pp), np.abs(pr))
    pump_c = _series(closed, "pump_total_rate_m3_min")
    pump_p = _series(primary, "pump_total_rate_m3_min")

    fig, axes = plt.subplots(5, 1, figsize=(15, 12), sharex=True)
    fig.subplots_adjust(top=0.90, hspace=0.28)
    title = (
        f"{case_key}: pitch/roll separated smooth180 diagnostic\n"
        f"pump {row['pump_change_pct']:+.1f}%, latch reduction {row['latch_reduction_pct']:.1f}%, "
        f"d_p95 pitch/roll {row['d_pitch_p95']:+.2f}/{row['d_roll_p95']:+.2f} deg, "
        f"worse axis: {row['worse_p95_axis']}"
    )
    fig.suptitle(title, fontsize=15, x=0.06, ha="left")

    panels = [
        (axes[0], cp, pp, "Pitch (deg)", row["d_pitch_p95"]),
        (axes[1], cr, pr, "Roll (deg)", row["d_roll_p95"]),
    ]
    for ax, base, test, label, delta in panels:
        color = "#d94801" if float(delta) > 0 else "#2b8cbe"
        ax.plot(t_h, base, color="#667085", lw=0.9, ls=":", label="closed only")
        ax.plot(t_h, test, color=color, lw=1.0, label="current production")
        ax.axhline(5.0, color="#667085", lw=0.8, ls="--")
        ax.axhline(-5.0, color="#667085", lw=0.8, ls="--")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", ncol=2, frameon=False)

    axes[2].plot(t_h, ca, color="#667085", lw=0.9, ls=":", label="closed max(|pitch|, |roll|)")
    axes[2].plot(t_h, pa, color="#d94801", lw=1.0, label="primary max(|pitch|, |roll|)")
    for level in (5.0, 7.5, 10.0):
        axes[2].axhline(level, color="#667085", lw=0.8, ls="--")
        axes[2].text(t_h[-1], level, f" {level:g}", va="center", color="#667085", fontsize=8)
    axes[2].set_ylabel("Threshold helper\n(deg)")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper left", ncol=2, frameon=False)

    axes[3].plot(t_h, pump_c, color="#667085", lw=0.8, ls=":", label="closed only")
    axes[3].plot(t_h, pump_p, color="#d94801", lw=0.8, label="current production")
    axes[3].set_ylabel("Pump rate\n(m3/min)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="upper left", ncol=2, frameon=False)

    if "pump_latch_switch_count" in closed.columns and "pump_latch_switch_count" in primary.columns:
        axes[4].plot(t_h, _series(closed, "pump_latch_switch_count"), color="#667085", lw=0.9, ls=":", label="closed only")
        axes[4].plot(t_h, _series(primary, "pump_latch_switch_count"), color="#d94801", lw=0.9, label="current production")
    axes[4].set_ylabel("Latch\nswitches")
    axes[4].set_xlabel("Time in 6h window (h)")
    axes[4].grid(True, alpha=0.25)
    axes[4].legend(loc="upper left", ncol=2, frameon=False)

    fig.savefig(out / f"{case_key}_pitch_roll_trace.png", dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    timeseries_dir = args.run_dir / "timeseries"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_key, closed_path, primary_path in _build_pairs(timeseries_dir):
        closed = _read_csv(closed_path)
        primary = _read_csv(primary_path)
        row = _summary_row(case_key, closed, primary)
        rows.append(row)
        _plot_case(case_key, closed, primary, row, args.out_dir)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "pitch_roll_case_comparison.csv", index=False)
    (args.out_dir / "README.md").write_text(
        "# Pitch/Roll Separated Smooth180 Diagnostics\n\n"
        "These plots use the verified smooth180 run and show pitch and roll as separate first-class signals. "
        "`max(|pitch|, |roll|)` is retained only as a threshold helper for T>5/T>7.5/T>10 interpretation.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
