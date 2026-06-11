#!/usr/bin/env python3
"""Regenerate paper-style figures for the validated W1 stable-event subtype."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/saintyoung/Desktop/FOWT-CONTROLLER-main")
RUN = ROOT / "outputs/wind_prediction/regime_conditioned_policy_development_v1/hard_24h_pump_gate_v1/w1_hi_attn_stable_broad20_d1_12h_v1"
OUT = ROOT / "FIXED_PUMP_SAVING_RESULTS_20260530/W1_high_attention_direction_stable_event_validated/figures"
TS = RUN / "timeseries"


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
            ax.axvspan(start, float(t_min[i]), color="#e76f51", alpha=0.15, lw=0)
            start = None


def _load_pair(case_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    closed = sorted(TS.glob(f"{case_id}_*_closed_only_timeseries.csv"))
    primary = sorted(TS.glob(f"{case_id}_*_w1_hi_attn_stable_broad20_d1_12h_timeseries.csv"))
    if not closed or not primary:
        raise FileNotFoundError(f"missing W1 stable pair for {case_id}")
    return pd.read_csv(closed[0]), pd.read_csv(primary[0])


def _stats(closed: pd.DataFrame, primary: pd.DataFrame) -> dict[str, float]:
    t = closed["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    c_axis = _axis(closed)
    p_axis = _axis(primary)
    c_pump = float((closed["pump_total_rate_m3_min"] * dt / 60.0).sum())
    p_pump = float((primary["pump_total_rate_m3_min"] * dt / 60.0).sum())
    return {
        "saving_pct": 100.0 * (c_pump - p_pump) / c_pump if c_pump else 0.0,
        "closed_time5": float((c_axis > 5.0).sum() * dt),
        "primary_time5": float((p_axis > 5.0).sum() * dt),
        "delta_time5": float(((p_axis > 5.0).sum() - (c_axis > 5.0).sum()) * dt),
        "closed_p95": float(np.percentile(c_axis, 95)),
        "primary_p95": float(np.percentile(p_axis, 95)),
        "fallback_s": float(_fallback_mask(primary).sum() * dt),
        "closed_pump": c_pump,
        "primary_pump": p_pump,
    }


def _plot_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    x = np.arange(len(df))
    colors = np.where(df["saving_pct"] >= 30, "#2ca25f", np.where(df["saving_pct"] >= 20, "#66c2a4", "#fdae61"))
    ax.bar(x, df["saving_pct"], color=colors, edgecolor="#333333", linewidth=0.45)
    ax.axhline(20, color="#333333", lw=1.0, ls="--", label="20% reference")
    ax.set_ylabel("pump saving (%)")
    ax.set_xlabel("case index")
    ax.set_title("W1 direction-stable future-event: broad-20 validation, 12h real windows")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i + 1:02d}" for i in range(len(df))])
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(0, max(60.0, float(df["saving_pct"].max()) + 8.0))

    ax2 = ax.twinx()
    ax2.plot(x, df["max_d_p95"], color="#d95f0e", marker="o", lw=1.7, label="max Delta p95 axis")
    ax2.set_ylabel("max Delta p95 axis (deg)")
    ax2.set_ylim(0, max(1.0, float(df["max_d_p95"].max()) + 0.2))

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2)
    subtitle = (
        f"aggregate saving 39.6%; min {df['saving_pct'].min():.1f}%, "
        f"median {df['saving_pct'].median():.1f}%; 0/20 worse; fallback 0; total time>5 -238s"
    )
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "F_w1_broad20_12h_distribution.png", dpi=180)
    plt.close(fig)


def _plot_case(row: pd.Series, out_name: str, note: str) -> None:
    closed, primary = _load_pair(str(row["case_id"]))
    t_min = closed["t_s"].to_numpy(dtype=float) / 60.0
    p_t_min = primary["t_s"].to_numpy(dtype=float) / 60.0
    c_axis = _axis(closed)
    p_axis = _axis(primary)
    stats = _stats(closed, primary)
    mask = _fallback_mask(primary)

    fig, axes = plt.subplots(5, 1, figsize=(12.8, 12.0), sharex=True)
    fig.suptitle(f"{row['case_id']} | W1 direction-stable future-event | {note}", fontsize=13, fontweight="bold")

    axes[0].plot(t_min, c_axis, label="A0 max-axis", color="#1f77b4", lw=2)
    axes[0].plot(p_t_min, p_axis, label="deadband max-axis", color="#d62728", lw=2)
    axes[0].axhline(5.0, color="#111111", lw=1, ls="--", label="5 deg floor")
    _shade_fallback(axes[0], p_t_min, mask)
    axes[0].set_ylabel("max-axis deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_min, closed["pitch_deg"], label="A0 pitch", color="#1f77b4", lw=1.3)
    axes[1].plot(t_min, closed["roll_deg"], label="A0 roll", color="#2ca02c", lw=1.3)
    axes[1].plot(p_t_min, primary["pitch_deg"], label="deadband pitch", color="#d62728", lw=1.2, ls="--")
    axes[1].plot(p_t_min, primary["roll_deg"], label="deadband roll", color="#ff7f0e", lw=1.2, ls="--")
    axes[1].axhline(5.0, color="#111111", lw=0.8, ls=":")
    axes[1].axhline(-5.0, color="#111111", lw=0.8, ls=":")
    _shade_fallback(axes[1], p_t_min, mask)
    axes[1].set_ylabel("pitch/roll deg")
    axes[1].legend(loc="upper right", ncol=4, fontsize=7)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_min, _cum_pump(closed), label="A0 cumulative pump", color="#1f77b4", lw=2)
    axes[2].plot(p_t_min, _cum_pump(primary), label="deadband cumulative pump", color="#d62728", lw=2)
    _shade_fallback(axes[2], p_t_min, mask)
    axes[2].set_ylabel("pump m3")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t_min, closed["pump_total_rate_m3_min"], label="A0 pump rate", color="#1f77b4", lw=1.1)
    axes[3].plot(p_t_min, primary["pump_total_rate_m3_min"], label="deadband pump rate", color="#d62728", lw=1.1)
    _shade_fallback(axes[3], p_t_min, mask)
    axes[3].set_ylabel("m3/min")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].grid(True, alpha=0.25)

    axes[4].plot(t_min, closed["wind_speed"], label="actual wind speed", color="#4c78a8", lw=1.4)
    for col, color in (
        ("preview_pressure_block0_norm", "#d62728"),
        ("preview_pressure_block1_norm", "#ff7f0e"),
        ("preview_pressure_block2_norm", "#2ca02c"),
    ):
        if col in primary.columns:
            axes[4].plot(p_t_min, primary[col], label=col.replace("preview_pressure_", ""), color=color, lw=1.0, alpha=0.85)
    axes[4].set_xlabel("time min")
    axes[4].set_ylabel("wind / pressure")
    axes[4].legend(loc="upper right", ncol=4, fontsize=7)
    axes[4].grid(True, alpha=0.25)

    subtitle = (
        f"saving={stats['saving_pct']:.1f}% | pump {stats['closed_pump']:.0f}->{stats['primary_pump']:.0f} m3 | "
        f"time>5 {stats['closed_time5']:.0f}s->{stats['primary_time5']:.0f}s ({stats['delta_time5']:+.0f}s) | "
        f"fallback={stats['fallback_s']:.0f}s | p95 {stats['closed_p95']:.2f}->{stats['primary_p95']:.2f} deg"
    )
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / out_name, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RUN / "casebook_summary.csv")
    df["saving_pct"] = -df["d_pump_work_pct"].astype(float)
    df["max_d_p95"] = df[["d_pitch_p95", "d_roll_p95"]].max(axis=1)

    low = df.sort_values("saving_pct").iloc[0]
    median = df.iloc[(df["saving_pct"] - df["saving_pct"].median()).abs().argsort()].iloc[0]
    high = df.sort_values("saving_pct").iloc[-1]

    _plot_distribution(df)
    _plot_case(low, "F_w1_low_bounded_case_09_hi_attn_stable_broad_20230101.png", "lowest-saving bounded case")
    _plot_case(median, "F_w1_typical_median_case_11_hi_attn_stable_broad_20230324.png", "typical median case")
    _plot_case(high, "F_w1_high_saving_case_08_hi_attn_stable_broad_20221109.png", "high-saving case")

    print("rewrote paper-style W1 figures")
    for name in (
        "F_w1_broad20_12h_distribution.png",
        "F_w1_low_bounded_case_09_hi_attn_stable_broad_20230101.png",
        "F_w1_typical_median_case_11_hi_attn_stable_broad_20230324.png",
        "F_w1_high_saving_case_08_hi_attn_stable_broad_20221109.png",
    ):
        print(OUT / name)


if __name__ == "__main__":
    main()
