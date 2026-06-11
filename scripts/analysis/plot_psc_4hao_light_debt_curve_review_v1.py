#!/usr/bin/env python3
"""Plot No.4 light-debt cases against the 0hao baseline.

The figures are diagnostic: they show posture, max-axis envelope, pump rate,
and delta/correlation traces so we can check whether the No.4 curves are just
parallel loosened-tolerance copies of the 0hao curves.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = (
    REPO_ROOT
    / "outputs"
    / "wind_prediction"
    / "psc_selector_mixed_pool_v1"
    / "psc_4hao_light_debt_curve_review_v1"
)
TS_DIR = RUN_DIR / "timeseries"
FIG_DIR = RUN_DIR / "curve_review_figures"
SUMMARY_CSV = RUN_DIR / "curve_review_metrics.csv"
SUMMARY_MD = RUN_DIR / "curve_review_readout.md"

CASES = [
    {
        "case": "09_dual_relief_09",
        "run_prefix": "01_dual_relief_09_2023-10-13_123000",
        "name": "09 transient relief 严格安全",
        "plot_title": "09 transient relief / strict-safe",
    },
    {
        "case": "10_dual_relief_10",
        "run_prefix": "02_dual_relief_10_2023-09-19_010000",
        "name": "10 relief 轻债务 +8s gt5",
        "plot_title": "10 relief / light-debt +8s gt5",
    },
    {
        "case": "94_dual_boundary_06",
        "run_prefix": "08_dual_boundary_06_2023-10-03_063000",
        "name": "94 boundary 轻债务 +31s gt5",
        "plot_title": "94 boundary / light-debt +31s gt5",
    },
]

PRIMARY_SUFFIX = "psc_4hao_light_debt_regime_auto_v1"


def _read_pair(run_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    closed_path = TS_DIR / f"{run_prefix}_closed_only_timeseries.csv"
    primary_path = TS_DIR / f"{run_prefix}_{PRIMARY_SUFFIX}_timeseries.csv"
    if not closed_path.exists():
        raise FileNotFoundError(closed_path)
    if not primary_path.exists():
        raise FileNotFoundError(primary_path)
    return pd.read_csv(closed_path), pd.read_csv(primary_path)


def _time_min(df: pd.DataFrame) -> np.ndarray:
    if "time_s" in df.columns:
        return df["time_s"].to_numpy(dtype=float) / 60.0
    return np.arange(len(df), dtype=float) / 60.0


def _axis(df: pd.DataFrame) -> np.ndarray:
    return df[["pitch_deg", "roll_deg"]].abs().max(axis=1).to_numpy(dtype=float)


def _work(df: pd.DataFrame) -> float:
    return float(df["pump_total_rate_m3_min"].to_numpy(dtype=float).sum() / 60.0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) != len(b) or len(a) < 3:
        return float("nan")
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _fit_slope(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) != len(b) or len(a) < 3 or float(np.std(a)) <= 1e-12:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(a, b, 1)
    return float(slope), float(intercept)


def _episodes(mask: np.ndarray) -> int:
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0
    starts = arr & ~np.r_[False, arr[:-1]]
    return int(starts.sum())


def _metrics(
    case: str, name: str, closed: pd.DataFrame, primary: pd.DataFrame
) -> dict[str, float | str]:
    n = min(len(closed), len(primary))
    c = closed.iloc[:n].copy()
    p = primary.iloc[:n].copy()
    c_axis = _axis(c)
    p_axis = _axis(p)
    c_pitch = c["pitch_deg"].to_numpy(dtype=float)
    p_pitch = p["pitch_deg"].to_numpy(dtype=float)
    c_roll = c["roll_deg"].to_numpy(dtype=float)
    p_roll = p["roll_deg"].to_numpy(dtype=float)
    c_pump = c["pump_total_rate_m3_min"].to_numpy(dtype=float)
    p_pump = p["pump_total_rate_m3_min"].to_numpy(dtype=float)
    pitch_slope, pitch_intercept = _fit_slope(c_pitch, p_pitch)
    roll_slope, roll_intercept = _fit_slope(c_roll, p_roll)
    axis_slope, axis_intercept = _fit_slope(c_axis, p_axis)
    saved = _work(c) - _work(p)
    return {
        "case": case,
        "name": name,
        "baseline_pump_m3": _work(c),
        "fourhao_pump_m3": _work(p),
        "saved_m3": saved,
        "saving_pct": saved / max(_work(c), 1e-9) * 100.0,
        "baseline_time_gt5_s": float((c_axis > 5.0).sum()),
        "fourhao_time_gt5_s": float((p_axis > 5.0).sum()),
        "d_time_gt5_s": float((p_axis > 5.0).sum() - (c_axis > 5.0).sum()),
        "baseline_max_axis_deg": float(np.max(c_axis)),
        "fourhao_max_axis_deg": float(np.max(p_axis)),
        "d_max_axis_deg": float(np.max(p_axis) - np.max(c_axis)),
        "pitch_corr": _corr(c_pitch, p_pitch),
        "roll_corr": _corr(c_roll, p_roll),
        "axis_corr": _corr(c_axis, p_axis),
        "pitch_slope_4hao_vs_0hao": pitch_slope,
        "pitch_intercept_deg": pitch_intercept,
        "roll_slope_4hao_vs_0hao": roll_slope,
        "roll_intercept_deg": roll_intercept,
        "axis_slope_4hao_vs_0hao": axis_slope,
        "axis_intercept_deg": axis_intercept,
        "pitch_delta_std_deg": float(np.std(p_pitch - c_pitch)),
        "roll_delta_std_deg": float(np.std(p_roll - c_roll)),
        "axis_delta_std_deg": float(np.std(p_axis - c_axis)),
        "baseline_pump_switches": _episodes(c_pump > 1e-6),
        "fourhao_pump_switches": _episodes(p_pump > 1e-6),
    }


def _plot_case(case: dict[str, str], closed: pd.DataFrame, primary: pd.DataFrame) -> Path:
    n = min(len(closed), len(primary))
    c = closed.iloc[:n].copy()
    p = primary.iloc[:n].copy()
    t = _time_min(c)
    c_axis = _axis(c)
    p_axis = _axis(p)
    c_pitch = c["pitch_deg"].to_numpy(dtype=float)
    p_pitch = p["pitch_deg"].to_numpy(dtype=float)
    c_roll = c["roll_deg"].to_numpy(dtype=float)
    p_roll = p["roll_deg"].to_numpy(dtype=float)
    c_pump = c["pump_total_rate_m3_min"].to_numpy(dtype=float)
    p_pump = p["pump_total_rate_m3_min"].to_numpy(dtype=float)

    fig, axes = plt.subplots(5, 1, figsize=(13.5, 11.5), sharex=True)
    colors = {"0hao": "#2b6cb0", "4hao": "#c05621", "delta": "#4a5568"}

    axes[0].plot(t, c_pitch, color=colors["0hao"], lw=1.3, label="0hao pitch")
    axes[0].plot(t, p_pitch, color=colors["4hao"], lw=1.3, label="4hao pitch")
    axes[0].axhline(5.0, color="#999999", lw=0.8, ls="--")
    axes[0].axhline(-5.0, color="#999999", lw=0.8, ls="--")
    axes[0].set_ylabel("pitch deg")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)

    axes[1].plot(t, c_roll, color=colors["0hao"], lw=1.3, label="0hao roll")
    axes[1].plot(t, p_roll, color=colors["4hao"], lw=1.3, label="4hao roll")
    axes[1].axhline(5.0, color="#999999", lw=0.8, ls="--")
    axes[1].axhline(-5.0, color="#999999", lw=0.8, ls="--")
    axes[1].set_ylabel("roll deg")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)

    axes[2].plot(t, c_axis, color=colors["0hao"], lw=1.3, label="0hao max axis")
    axes[2].plot(t, p_axis, color=colors["4hao"], lw=1.3, label="4hao max axis")
    axes[2].axhline(5.0, color="#aa0000", lw=0.9, ls="--", label="5 deg")
    axes[2].set_ylabel("max |axis| deg")
    axes[2].legend(loc="upper right", ncol=3, frameon=False)

    axes[3].plot(t, c_pump, color=colors["0hao"], lw=1.0, label="0hao pump")
    axes[3].plot(t, p_pump, color=colors["4hao"], lw=1.0, label="4hao pump")
    axes[3].set_ylabel("pump m3/min")
    axes[3].legend(loc="upper right", ncol=2, frameon=False)

    axes[4].plot(t, p_pitch - c_pitch, color="#805ad5", lw=1.0, label="pitch delta")
    axes[4].plot(t, p_roll - c_roll, color="#2f855a", lw=1.0, label="roll delta")
    axes[4].plot(t, p_axis - c_axis, color=colors["delta"], lw=1.1, label="max-axis delta")
    axes[4].axhline(0.0, color="#888888", lw=0.8)
    axes[4].set_ylabel("4hao - 0hao deg")
    axes[4].set_xlabel("time min")
    axes[4].legend(loc="upper right", ncol=3, frameon=False)

    for ax in axes:
        ax.grid(True, color="#dddddd", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for x in range(20, 120, 20):
            ax.axvline(x, color="#dddddd", lw=0.7, ls=":")

    fig.suptitle(case["plot_title"], fontsize=14, y=0.995)
    out = FIG_DIR / f"{case['case']}_curve_review.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_scatter(case: dict[str, str], closed: pd.DataFrame, primary: pd.DataFrame) -> Path:
    n = min(len(closed), len(primary))
    c = closed.iloc[:n].copy()
    p = primary.iloc[:n].copy()
    c_axis = _axis(c)
    p_axis = _axis(p)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    pairs = [
        ("pitch", c["pitch_deg"].to_numpy(dtype=float), p["pitch_deg"].to_numpy(dtype=float)),
        ("roll", c["roll_deg"].to_numpy(dtype=float), p["roll_deg"].to_numpy(dtype=float)),
        ("max axis", c_axis, p_axis),
    ]
    for ax, (label, a, b) in zip(axes, pairs):
        ax.scatter(a, b, s=4, alpha=0.22, color="#2d3748")
        lo = float(min(np.min(a), np.min(b)))
        hi = float(max(np.max(a), np.max(b)))
        ax.plot([lo, hi], [lo, hi], color="#c05621", lw=1.0, ls="--", label="y=x")
        slope, intercept = _fit_slope(a, b)
        xs = np.array([lo, hi], dtype=float)
        ax.plot(xs, slope * xs + intercept, color="#2b6cb0", lw=1.0, label="fit")
        ax.set_title(f"{label}: slope={slope:.2f}, corr={_corr(a,b):.2f}")
        ax.set_xlabel("0hao")
        ax.set_ylabel("4hao")
        ax.grid(True, color="#dddddd", lw=0.7, alpha=0.75)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{case['plot_title']} - parallel/scale check", fontsize=13)
    out = FIG_DIR / f"{case['case']}_parallel_check.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    figure_rows = []
    for case in CASES:
        closed, primary = _read_pair(case["run_prefix"])
        rows.append(_metrics(case["case"], case["name"], closed, primary))
        curve = _plot_case(case, closed, primary)
        scatter = _plot_scatter(case, closed, primary)
        figure_rows.append((case["case"], case["name"], curve, scatter))

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    lines = [
        "# 4hao Light-Debt Curve Review",
        "",
        "Cases: `09`, `10`, and `94`, the no-fallback light-debt gate examples.",
        "",
        "## Metrics",
        "",
        "| 工况 | case id | saved m3 | saving % | d_gt5 s | d_max_axis deg | pitch corr/slope | roll corr/slope | axis corr/slope | pump episodes 0hao->4hao |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['name']} | {row['case']} | {float(row['saved_m3']):.1f} | {float(row['saving_pct']):.2f}% | "
            f"{float(row['d_time_gt5_s']):.0f} | {float(row['d_max_axis_deg']):+.2f} | "
            f"{float(row['pitch_corr']):.2f}/{float(row['pitch_slope_4hao_vs_0hao']):.2f} | "
            f"{float(row['roll_corr']):.2f}/{float(row['roll_slope_4hao_vs_0hao']):.2f} | "
            f"{float(row['axis_corr']):.2f}/{float(row['axis_slope_4hao_vs_0hao']):.2f} | "
            f"{int(row['baseline_pump_switches'])}->{int(row['fourhao_pump_switches'])} |"
        )
    lines.extend(
        [
            "",
            "## Figure Index",
            "",
            "| 工况 | case id | curve review | parallel check |",
            "| --- | --- | --- | --- |",
        ]
    )
    for case, name, curve, scatter in figure_rows:
        lines.append(
            f"| {name} | {case} | {curve.relative_to(REPO_ROOT)} | {scatter.relative_to(REPO_ROOT)} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            "- If it were pure tolerance loosening, the 4hao posture would look like a near-parallel shifted copy: high correlation, slope near 1, and low delta variability.",
            "- The reviewed cases instead show fewer pump episodes and larger posture excursions.  The behavior is closer to target holding / less frequent pump chasing than to a simple static tolerance offset.",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {SUMMARY_MD}")
    for _, _, curve, scatter in figure_rows:
        print(curve)
        print(scatter)


if __name__ == "__main__":
    main()
