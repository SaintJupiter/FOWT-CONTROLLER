#!/usr/bin/env python3
"""Plot diverse examples from the 52-case high-stable clean subset.

This is a diagnosis/communication helper.  It does not run a new controller;
it only reads existing matched A0/candidate time-series and draws the posture,
pump, wind, and forecast-block signals for representative cases.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1/clean_neutral_headroom_expansion_v1")
CASE_DELTA = ROOT / "paper_ready/high_stable_clean_combined52_case_delta.csv"
OUT_DIR = ROOT / "paper_ready/figures/high_stable_clean_52_examples_v1"
TS_DIRS = {
    "test": ROOT / "validation_test80_forecast_advised_v1_1h/timeseries",
    "validation": ROOT / "validation_validation80_forecast_advised_v1_1h/timeseries",
}


SELECTED = [
    # split, case_prefix, short key, reason
    ("test", 4, "mild_decay_clear", "mild decay: future load eases, so the controller avoids chasing a fading high load"),
    ("test", 18, "moderate_high_steady", "moderate-high steady: high load remains stable, no reversal/re-intensification warning"),
    ("test", 44, "mixed_steady_low_gain", "mixed steady, lower saving: still clean, but less pump headroom than the strongest cases"),
    ("validation", 49, "mild_decay_high_saving", "validation mild decay: large pump saving with no fallback and no time>5 increase"),
    ("validation", 71, "worst_p95_edge", "edge case: largest p95 increase inside the 52-case set, used to inspect posture cost"),
    ("validation", 9, "lowest_saving_validation", "lowest-saving validation case: checks that the mechanism is not just one easy pattern"),
]


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
            ax.axvspan(start, end, color="#e76f51", alpha=0.14, lw=0)
            start = None


def _load_pair(split: str, case_prefix: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    ts_dir = TS_DIRS[split]
    prefix = f"{case_prefix:02d}"
    closed_matches = sorted(ts_dir.glob(f"{prefix}_*_closed_only_timeseries.csv"))
    primary_matches = sorted(ts_dir.glob(f"{prefix}_*_prediction_primary_econ_timeseries.csv"))
    if not closed_matches or not primary_matches:
        raise FileNotFoundError(f"missing timeseries pair for {split} case {prefix} in {ts_dir}")
    return pd.read_csv(closed_matches[0]), pd.read_csv(primary_matches[0]), closed_matches[0].name


def _parse_label(label: str) -> dict[str, str]:
    out = {}
    subtype = re.search(r"subtype=([^|]+)", label)
    if subtype:
        out["subtype"] = subtype.group(1).strip()
    for key in ("early", "near_range", "far_range", "drop", "dir"):
        m = re.search(rf"{key}=([-0-9.]+)", label)
        if m:
            out[key] = m.group(1)
    return out


def _summary_stats(closed: pd.DataFrame, primary: pd.DataFrame) -> dict[str, float]:
    t = closed["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    c_axis = _axis(closed)
    p_axis = _axis(primary)
    c_pump = float((closed["pump_total_rate_m3_min"] * dt / 60.0).sum())
    p_pump = float((primary["pump_total_rate_m3_min"] * dt / 60.0).sum())
    return {
        "a0_pump_m3": c_pump,
        "candidate_pump_m3": p_pump,
        "saving_pct": 100.0 * (c_pump - p_pump) / c_pump if c_pump else 0.0,
        "a0_time_gt5_s": float((c_axis > 5.0).sum() * dt),
        "candidate_time_gt5_s": float((p_axis > 5.0).sum() * dt),
        "delta_time_gt5_s": float(((p_axis > 5.0).sum() - (c_axis > 5.0).sum()) * dt),
        "a0_p95_axis_deg": float(np.percentile(c_axis, 95)),
        "candidate_p95_axis_deg": float(np.percentile(p_axis, 95)),
        "delta_p95_axis_deg": float(np.percentile(p_axis, 95) - np.percentile(c_axis, 95)),
        "fallback_s": float(_fallback_mask(primary).sum() * dt),
    }


def plot_one(row: pd.Series, split: str, prefix: int, key: str, reason: str) -> dict[str, float | str]:
    closed, primary, source = _load_pair(split, prefix)
    meta = _parse_label(str(row["label"]))
    stats = _summary_stats(closed, primary)
    t_min = closed["t_s"].to_numpy(dtype=float) / 60.0
    p_t_min = primary["t_s"].to_numpy(dtype=float) / 60.0
    c_axis = _axis(closed)
    p_axis = _axis(primary)
    mask = _fallback_mask(primary)

    fig, axes = plt.subplots(5, 1, figsize=(12.8, 12.0), sharex=True)
    title = (
        f"{split} {prefix:02d} | {meta.get('subtype', 'unknown')} | {key}\n"
        f"forecast shape: early={meta.get('early', '?')} m/s, near_range={meta.get('near_range', '?')}, "
        f"far_range={meta.get('far_range', '?')}, drop={meta.get('drop', '?')}, dir_shift={meta.get('dir', '?')} deg"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].plot(t_min, c_axis, label="A0 max-axis", color="#1f77b4", lw=2)
    axes[0].plot(p_t_min, p_axis, label="candidate max-axis", color="#d62728", lw=2)
    axes[0].axhline(5.0, color="#111111", lw=1, ls="--", label="5 deg floor")
    _shade_fallback(axes[0], p_t_min, mask)
    axes[0].set_ylabel("max-axis deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_min, closed["pitch_deg"], label="A0 pitch", color="#1f77b4", lw=1.3)
    axes[1].plot(t_min, closed["roll_deg"], label="A0 roll", color="#2ca02c", lw=1.3)
    axes[1].plot(p_t_min, primary["pitch_deg"], label="cand pitch", color="#d62728", lw=1.2, ls="--")
    axes[1].plot(p_t_min, primary["roll_deg"], label="cand roll", color="#ff7f0e", lw=1.2, ls="--")
    axes[1].axhline(5.0, color="#111111", lw=0.8, ls=":")
    axes[1].axhline(-5.0, color="#111111", lw=0.8, ls=":")
    _shade_fallback(axes[1], p_t_min, mask)
    axes[1].set_ylabel("pitch/roll deg")
    axes[1].legend(loc="upper right", ncol=4, fontsize=7)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t_min, _cum_pump(closed), label="A0 cumulative pump", color="#1f77b4", lw=2)
    axes[2].plot(p_t_min, _cum_pump(primary), label="candidate cumulative pump", color="#d62728", lw=2)
    _shade_fallback(axes[2], p_t_min, mask)
    axes[2].set_ylabel("pump m3")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t_min, closed["pump_total_rate_m3_min"], label="A0 pump rate", color="#1f77b4", lw=1.1)
    axes[3].plot(p_t_min, primary["pump_total_rate_m3_min"], label="candidate pump rate", color="#d62728", lw=1.1)
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
        f"saving={stats['saving_pct']:.1f}% | time>5: "
        f"{stats['a0_time_gt5_s']:.0f}s -> {stats['candidate_time_gt5_s']:.0f}s "
        f"({stats['delta_time_gt5_s']:+.0f}s) | fallback={stats['fallback_s']:.0f}s | "
        f"p95={stats['a0_p95_axis_deg']:.2f}->{stats['candidate_p95_axis_deg']:.2f} "
        f"({stats['delta_p95_axis_deg']:+.2f})"
    )
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10)
    fig.text(0.5, 0.902, reason, ha="center", fontsize=9, color="#333333")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{split}_{prefix:02d}_{key}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    return {
        "split": split,
        "case_prefix": prefix,
        "case_id": row["case_id"],
        "subtype": meta.get("subtype", ""),
        "key": key,
        "reason": reason,
        "source": source,
        "plot": str(out_path),
        **stats,
        "early_ms": meta.get("early", ""),
        "near_range_ms": meta.get("near_range", ""),
        "far_range_ms": meta.get("far_range", ""),
        "drop_ms": meta.get("drop", ""),
        "dir_shift_deg": meta.get("dir", ""),
    }


def main() -> None:
    df = pd.read_csv(CASE_DELTA)
    rows: list[dict[str, float | str]] = []
    for split, prefix, key, reason in SELECTED:
        match = df[(df["split"] == split) & (df["case_prefix"].astype(int) == int(prefix))]
        if match.empty:
            raise ValueError(f"no summary row for {split} {prefix:02d}")
        rows.append(plot_one(match.iloc[0], split, prefix, key, reason))
    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_DIR / "selected_examples_summary.csv", index=False)
    print(summary[[
        "split",
        "case_prefix",
        "subtype",
        "saving_pct",
        "delta_time_gt5_s",
        "fallback_s",
        "delta_p95_axis_deg",
        "plot",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
