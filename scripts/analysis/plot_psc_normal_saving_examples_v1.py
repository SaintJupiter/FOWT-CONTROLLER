#!/usr/bin/env python3
"""Plot representative normal saving examples for the 96-case selector result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
SAFETY_TS = BASE / "degradation_ladder_96case_pair" / "current_forecast_adaptive" / "timeseries"
ECON_TS = BASE / "degradation_ladder_96case_pair" / "learned_rawenv_mainline" / "timeseries"
DECISIONS = (
    BASE
    / "psc_structural_selector_96case_pair_3hao_v2_learned_v1"
    / "structural_selector_case_decisions.csv"
)
OUT = BASE / "psc_normal_saving_examples_v1"
FIG_DIR = OUT / "figures"
SUMMARY_CSV = OUT / "normal_example_metrics.csv"
SUMMARY_MD = OUT / "normal_example_readout.md"

CASES = [
    {
        "case_id": "25_dual_neutral_01",
        "prefix": "25_dual_neutral_01_2023-02-03_141000",
        "kind": "posture_better",
        "title": "25 neutral / posture better",
    },
    {
        "case_id": "34_dual_neutral_10",
        "prefix": "34_dual_neutral_10_2024-05-02_110000",
        "kind": "posture_better",
        "title": "34 neutral / posture better",
    },
    {
        "case_id": "39_dual_neutral_15",
        "prefix": "39_dual_neutral_15_2024-05-17_152000",
        "kind": "posture_flat",
        "title": "39 neutral / posture flat",
    },
    {
        "case_id": "05_dual_relief_05",
        "prefix": "05_dual_relief_05_2023-12-28_211000",
        "kind": "slightly_worse",
        "title": "05 relief / slight p95 debt",
    },
]


def _read_pair(prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    safety = pd.read_csv(SAFETY_TS / f"{prefix}_current_forecast_adaptive_timeseries.csv")
    econ = pd.read_csv(ECON_TS / f"{prefix}_learned_rawenv_mainline_timeseries.csv")
    n = min(len(safety), len(econ))
    return safety.iloc[:n].copy(), econ.iloc[:n].copy()


def _time_min(df: pd.DataFrame) -> np.ndarray:
    if "time_s" in df.columns:
        return df["time_s"].to_numpy(dtype=float) / 60.0
    return np.arange(len(df), dtype=float) / 60.0


def _axis(df: pd.DataFrame) -> np.ndarray:
    return df[["pitch_deg", "roll_deg"]].abs().max(axis=1).to_numpy(dtype=float)


def _pump_m3(df: pd.DataFrame) -> float:
    return float(df["pump_total_rate_m3_min"].to_numpy(dtype=float).sum() / 60.0)


def _case_metrics(case: dict[str, str], safety: pd.DataFrame, econ: pd.DataFrame) -> dict[str, object]:
    safety_axis = _axis(safety)
    econ_axis = _axis(econ)
    saved = _pump_m3(safety) - _pump_m3(econ)
    safety_mean = float(np.mean(safety_axis))
    econ_mean = float(np.mean(econ_axis))
    safety_p95 = float(np.quantile(safety_axis, 0.95))
    econ_p95 = float(np.quantile(econ_axis, 0.95))
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "baseline_pump_m3": _pump_m3(safety),
        "saving_pump_m3": _pump_m3(econ),
        "saved_m3": saved,
        "saving_pct": 100.0 * saved / max(_pump_m3(safety), 1e-9),
        "baseline_axis_mean_deg": safety_mean,
        "saving_axis_mean_deg": econ_mean,
        "d_axis_mean_deg": econ_mean - safety_mean,
        "d_axis_mean_pct": 100.0 * (econ_mean - safety_mean) / max(safety_mean, 1e-9),
        "baseline_axis_p95_deg": safety_p95,
        "saving_axis_p95_deg": econ_p95,
        "d_axis_p95_deg": econ_p95 - safety_p95,
        "d_axis_p95_pct": 100.0 * (econ_p95 - safety_p95) / max(safety_p95, 1e-9),
        "d_time_gt3_s": int((econ_axis > 3.0).sum() - (safety_axis > 3.0).sum()),
        "d_time_gt4_s": int((econ_axis > 4.0).sum() - (safety_axis > 4.0).sum()),
        "d_time_gt5_s": int((econ_axis > 5.0).sum() - (safety_axis > 5.0).sum()),
    }


def _plot_case(case: dict[str, str], safety: pd.DataFrame, econ: pd.DataFrame) -> Path:
    t = _time_min(safety)
    safety_axis = _axis(safety)
    econ_axis = _axis(econ)
    safety_pump = safety["pump_total_rate_m3_min"].to_numpy(dtype=float)
    econ_pump = econ["pump_total_rate_m3_min"].to_numpy(dtype=float)
    fig, axes = plt.subplots(5, 1, figsize=(13.5, 11.0), sharex=True)
    blue = "#2b6cb0"
    orange = "#c05621"
    gray = "#4a5568"
    axes[0].plot(t, safety["pitch_deg"], color=blue, lw=1.1, label="0hao pitch")
    axes[0].plot(t, econ["pitch_deg"], color=orange, lw=1.1, label="saving pitch")
    axes[0].axhline(5.0, color="#999999", lw=0.8, ls="--")
    axes[0].axhline(-5.0, color="#999999", lw=0.8, ls="--")
    axes[0].set_ylabel("pitch deg")
    axes[0].legend(loc="upper right", ncol=2, frameon=False)

    axes[1].plot(t, safety["roll_deg"], color=blue, lw=1.1, label="0hao roll")
    axes[1].plot(t, econ["roll_deg"], color=orange, lw=1.1, label="saving roll")
    axes[1].axhline(5.0, color="#999999", lw=0.8, ls="--")
    axes[1].axhline(-5.0, color="#999999", lw=0.8, ls="--")
    axes[1].set_ylabel("roll deg")
    axes[1].legend(loc="upper right", ncol=2, frameon=False)

    axes[2].plot(t, safety_axis, color=blue, lw=1.1, label="0hao max axis")
    axes[2].plot(t, econ_axis, color=orange, lw=1.1, label="saving max axis")
    axes[2].axhline(5.0, color="#aa0000", lw=0.8, ls="--", label="5 deg")
    axes[2].set_ylabel("max |axis| deg")
    axes[2].legend(loc="upper right", ncol=3, frameon=False)

    axes[3].plot(t, safety_pump, color=blue, lw=0.9, label="0hao pump")
    axes[3].plot(t, econ_pump, color=orange, lw=0.9, label="saving pump")
    axes[3].set_ylabel("pump m3/min")
    axes[3].legend(loc="upper right", ncol=2, frameon=False)

    axes[4].plot(t, econ_axis - safety_axis, color=gray, lw=1.0, label="max-axis delta")
    axes[4].axhline(0.0, color="#888888", lw=0.8)
    axes[4].set_ylabel("saving - 0hao deg")
    axes[4].set_xlabel("time min")
    axes[4].legend(loc="upper right", frameon=False)

    for ax in axes:
        ax.grid(True, color="#dddddd", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(case["title"], fontsize=14)
    out = FIG_DIR / f"{case['case_id']}_normal_saving_review.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _selector_averages() -> dict[str, float]:
    df = pd.read_csv(DECISIONS)
    selected = (
        (df["use_rawenv"].astype(float) == 1.0)
        & (df["rawenv_safe"].astype(float) == 1.0)
        & (df["pump_gain_m3"].astype(float) > 0.0)
    )
    sel = df[selected].copy()
    policy_gain = df["pump_gain_m3"].where(selected, 0.0)
    baseline = float(df["safety_pump_m3"].sum())
    return {
        "casebook_count": float(len(df)),
        "opened_count": float(len(sel)),
        "headline_saving_pct": 100.0 * float(policy_gain.sum()) / max(baseline, 1e-9),
        "opened_weighted_saving_pct": 100.0
        * float(sel["pump_gain_m3"].sum())
        / max(float(sel["safety_pump_m3"].sum()), 1e-9),
        "opened_simple_mean_pct": float(
            (100.0 * sel["pump_gain_m3"] / sel["safety_pump_m3"].clip(lower=1e-9)).mean()
        ),
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    figures = []
    for case in CASES:
        safety, econ = _read_pair(case["prefix"])
        rows.append(_case_metrics(case, safety, econ))
        figures.append((case, _plot_case(case, safety, econ)))
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    avg = _selector_averages()
    lines = [
        "# Normal Saving Examples",
        "",
        "These examples are selected from the normal safe-positive 96-case selector pool, not from the 4hao light-debt boundary subset.",
        "",
        "## Averages",
        "",
        f"- 96-case headline saving recomputed from the selector file: {avg['headline_saving_pct']:.2f}%",
        f"- Opened cases: {avg['opened_count']:.0f}/{avg['casebook_count']:.0f}",
        f"- Opened-case weighted saving: {avg['opened_weighted_saving_pct']:.2f}%",
        f"- Opened-case simple mean saving: {avg['opened_simple_mean_pct']:.2f}%",
        "",
        "## Examples",
        "",
        "| type | case | saved m3 | saving % | mean axis 0hao->saving | mean axis change % | p95 axis 0hao->saving | p95 axis change % | d_gt3 s | d_gt4 s | d_gt5 s | figure |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_case = {case["case_id"]: fig for case, fig in figures}
    for row in summary.to_dict("records"):
        lines.append(
            f"| {row['kind']} | {row['case_id']} | {float(row['saved_m3']):.1f} | "
            f"{float(row['saving_pct']):.2f}% | "
            f"{float(row['baseline_axis_mean_deg']):.2f}->{float(row['saving_axis_mean_deg']):.2f} | "
            f"{float(row['d_axis_mean_pct']):+.1f}% | "
            f"{float(row['baseline_axis_p95_deg']):.2f}->{float(row['saving_axis_p95_deg']):.2f} | "
            f"{float(row['d_axis_p95_pct']):+.1f}% | "
            f"{int(row['d_time_gt3_s'])} | {int(row['d_time_gt4_s'])} | {int(row['d_time_gt5_s'])} | "
            f"{by_case[str(row['case_id'])].relative_to(REPO_ROOT)} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(SUMMARY_MD)
    print(SUMMARY_CSV)
    for _, fig in figures:
        print(fig)


if __name__ == "__main__":
    main()
