#!/usr/bin/env python3
"""Plot representative forecast-advised economy cases.

The plots compare v1.6 baseline against the forecast-advised economy profile on
posture and pump trajectories.  They are meant for visual diagnosis, not for a
new controller experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_pair(ts_dir: Path, case_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    closed_matches = sorted(ts_dir.glob(f"{case_prefix}_*_closed_only_timeseries.csv"))
    primary_matches = sorted(ts_dir.glob(f"{case_prefix}_*_prediction_primary_econ_timeseries.csv"))
    if not closed_matches or not primary_matches:
        raise FileNotFoundError(f"missing timeseries pair for {case_prefix} in {ts_dir}")
    return pd.read_csv(closed_matches[0]), pd.read_csv(primary_matches[0]), closed_matches[0].name


def _axis(df: pd.DataFrame) -> pd.Series:
    return df[["pitch_deg", "roll_deg"]].abs().max(axis=1)


def _cum_pump(df: pd.DataFrame) -> np.ndarray:
    t = df["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return np.cumsum(df["pump_total_rate_m3_min"].to_numpy(dtype=float) * dt / 60.0)


def _fallback_mask(df: pd.DataFrame) -> pd.Series:
    for col in (
        "preview_primary_safety_fallback",
        "primary_safety_fallback",
        "preview_primary_safety_hard_active",
    ):
        if col in df.columns:
            return df[col].fillna(0).astype(float) > 0
    return pd.Series(False, index=df.index)


def _shade_fallback(ax: plt.Axes, t_min: np.ndarray, mask: pd.Series) -> None:
    values = mask.to_numpy(dtype=bool)
    if not values.any():
        return
    start: float | None = None
    for i, active in enumerate(values):
        if active and start is None:
            start = float(t_min[i])
        if start is not None and (not active or i == len(values) - 1):
            end = float(t_min[i])
            ax.axvspan(start, end, color="#e76f51", alpha=0.15, lw=0)
            start = None


def _stats(closed: pd.DataFrame, primary: pd.DataFrame) -> dict[str, float]:
    t = closed["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    c_axis = _axis(closed)
    p_axis = _axis(primary)
    c_pump = float((closed["pump_total_rate_m3_min"] * dt / 60.0).sum())
    p_pump = float((primary["pump_total_rate_m3_min"] * dt / 60.0).sum())
    return {
        "a0_pump": c_pump,
        "candidate_pump": p_pump,
        "saving_pct": 100.0 * (c_pump - p_pump) / c_pump if c_pump else 0.0,
        "a0_time_gt5": float((c_axis > 5.0).sum() * dt),
        "candidate_time_gt5": float((p_axis > 5.0).sum() * dt),
        "delta_time_gt5": float(((p_axis > 5.0).sum() - (c_axis > 5.0).sum()) * dt),
        "a0_p95": float(np.percentile(c_axis, 95)),
        "candidate_p95": float(np.percentile(p_axis, 95)),
        "fallback_s": float(_fallback_mask(primary).sum() * dt),
    }


def plot_case(
    *,
    ts_dir: Path,
    case_prefix: str,
    title: str,
    out_path: Path,
) -> dict[str, float | str]:
    closed, primary, source_name = _load_pair(ts_dir, case_prefix)
    t_min = closed["t_s"].to_numpy(dtype=float) / 60.0
    p_t_min = primary["t_s"].to_numpy(dtype=float) / 60.0
    c_axis = _axis(closed)
    p_axis = _axis(primary)
    stats = _stats(closed, primary)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].plot(t_min, c_axis, label="A0 max-axis", color="#1f77b4", lw=2)
    axes[0].plot(p_t_min, p_axis, label="candidate max-axis", color="#d62728", lw=2)
    axes[0].axhline(5.0, color="#111111", lw=1, ls="--", label="5 deg floor")
    _shade_fallback(axes[0], p_t_min, _fallback_mask(primary))
    axes[0].set_ylabel("max-axis deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_min, closed["pitch_deg"], label="A0 pitch", color="#1f77b4", lw=1.4)
    axes[1].plot(t_min, closed["roll_deg"], label="A0 roll", color="#2ca02c", lw=1.4)
    axes[1].plot(p_t_min, primary["pitch_deg"], label="cand pitch", color="#d62728", lw=1.2, ls="--")
    axes[1].plot(p_t_min, primary["roll_deg"], label="cand roll", color="#ff7f0e", lw=1.2, ls="--")
    axes[1].axhline(5.0, color="#111111", lw=0.8, ls=":")
    axes[1].axhline(-5.0, color="#111111", lw=0.8, ls=":")
    _shade_fallback(axes[1], p_t_min, _fallback_mask(primary))
    axes[1].set_ylabel("pitch/roll deg")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_min, _cum_pump(closed), label="A0 cumulative pump", color="#1f77b4", lw=2)
    axes[2].plot(p_t_min, _cum_pump(primary), label="candidate cumulative pump", color="#d62728", lw=2)
    _shade_fallback(axes[2], p_t_min, _fallback_mask(primary))
    axes[2].set_ylabel("pump m3")
    axes[2].legend(loc="upper left", fontsize=9)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t_min, closed["pump_total_rate_m3_min"], label="A0 pump rate", color="#1f77b4", lw=1.2)
    axes[3].plot(p_t_min, primary["pump_total_rate_m3_min"], label="candidate pump rate", color="#d62728", lw=1.2)
    _shade_fallback(axes[3], p_t_min, _fallback_mask(primary))
    axes[3].set_xlabel("time min")
    axes[3].set_ylabel("m3/min")
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(True, alpha=0.25)

    subtitle = (
        f"saving={stats['saving_pct']:.1f}% | "
        f"time>5: A0 {stats['a0_time_gt5']:.0f}s -> cand {stats['candidate_time_gt5']:.0f}s "
        f"(delta {stats['delta_time_gt5']:+.0f}s) | "
        f"fallback={stats['fallback_s']:.0f}s | "
        f"p95: {stats['a0_p95']:.2f}->{stats['candidate_p95']:.2f} deg"
    )
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"case_prefix": case_prefix, "source": source_name, "plot": str(out_path), **stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    cases = [
        (
            "03",
            "Clean high-stable example: large pump saving, posture not worse",
            "clean_high_stable_case03.png",
        ),
        (
            "23",
            "Wide-pool bad example: high saving, much more time above 5 deg",
            "bad_timegt5_case23.png",
        ),
        (
            "74",
            "Fallback-heavy boundary example: pump saving collapses into safety fallback",
            "fallback_heavy_case74.png",
        ),
    ]
    rows = []
    for prefix, title, name in cases:
        rows.append(
            plot_case(
                ts_dir=args.ts_dir,
                case_prefix=prefix,
                title=title,
                out_path=args.out_dir / name,
            )
        )
    pd.DataFrame(rows).to_csv(args.out_dir / "representative_case_plot_summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
