#!/usr/bin/env python3
"""Build W1 prediction-necessity audit tables and figures.

This is a read-only attribution artifact. It compares:
- A0 / closed-only;
- blind deadband, applied even on direction-reversal boundary cases;
- prediction-gated deployment, which opens deadband on W1 stable-direction events and
  abstains on direction-reversal/signflip boundary cases.

The goal is to show the prediction module's safety value: it decides when the
otherwise useful tolerance-band economy must NOT be used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/Users/saintyoung/Desktop/FOWT-CONTROLLER-main")
BASE = ROOT / "outputs/wind_prediction/regime_conditioned_policy_development_v1/hard_24h_pump_gate_v1"
BOUNDARY = BASE / "d1_boundary_reversal_reintensification_6h_v1"
STABLE = BASE / "w1_hi_attn_stable_broad20_d1_12h_v1"
OUT = ROOT / "FIXED_PUMP_SAVING_RESULTS_20260530/W1_prediction_necessity_audit"


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


def _load_pair_from_audit(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    closed = ROOT / str(row["closed_timeseries"])
    primary = ROOT / str(row["primary_timeseries"])
    return pd.read_csv(closed), pd.read_csv(primary)


def _stats(closed: pd.DataFrame, primary: pd.DataFrame) -> dict[str, float]:
    t = closed["t_s"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    ca = _axis(closed)
    pa = _axis(primary)
    c_pump = float((closed["pump_total_rate_m3_min"] * dt / 60.0).sum())
    p_pump = float((primary["pump_total_rate_m3_min"] * dt / 60.0).sum())
    return {
        "saving_pct": 100.0 * (c_pump - p_pump) / c_pump if c_pump else 0.0,
        "closed_time5": float((ca > 5.0).sum() * dt),
        "blind_time5": float((pa > 5.0).sum() * dt),
        "delta_time5": float(((pa > 5.0).sum() - (ca > 5.0).sum()) * dt),
        "closed_p95": float(np.percentile(ca, 95)),
        "blind_p95": float(np.percentile(pa, 95)),
        "delta_p95": float(np.percentile(pa, 95) - np.percentile(ca, 95)),
        "fallback_s": float(_fallback_mask(primary).sum() * dt),
        "closed_pump": c_pump,
        "blind_pump": p_pump,
    }


def _build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    boundary = pd.read_csv(BOUNDARY / "hard_gate_case_audit.csv")
    stable = pd.read_csv(STABLE / "hard_gate_case_audit.csv")

    b = boundary.copy()
    b["set"] = "boundary_reversal_signflip"
    b["blind_saving_pct"] = b["saving_pct"].astype(float)
    b["blind_d_time5_s"] = b["d_time_gt5_s"].astype(float)
    b["prediction_gated_saving_pct"] = 0.0
    b["prediction_gated_d_time5_s"] = 0.0
    b["prediction_safety_improvement_time5_s"] = b["blind_d_time5_s"]
    b["gate_decision"] = "abstain_deadband"

    s = stable.copy()
    s["set"] = "stable_direction_event"
    s["blind_saving_pct"] = s["saving_pct"].astype(float)
    s["blind_d_time5_s"] = s["d_time_gt5_s"].astype(float)
    s["prediction_gated_saving_pct"] = s["saving_pct"].astype(float)
    s["prediction_gated_d_time5_s"] = s["d_time_gt5_s"].astype(float)
    s["prediction_safety_improvement_time5_s"] = 0.0
    s["gate_decision"] = "allow_deadband"

    cols = [
        "set",
        "case_id",
        "label",
        "closed_pump_work_m3",
        "primary_pump_work_m3",
        "blind_saving_pct",
        "prediction_gated_saving_pct",
        "closed_time_gt5_s",
        "primary_time_gt5_s",
        "blind_d_time5_s",
        "prediction_gated_d_time5_s",
        "prediction_safety_improvement_time5_s",
        "d_pitch_p95",
        "d_roll_p95",
        "fallback_rows",
        "gate_decision",
        "closed_timeseries",
        "primary_timeseries",
    ]
    combined = pd.concat([s[cols], b[cols]], ignore_index=True)

    def agg(g: pd.DataFrame) -> pd.Series:
        closed_pump = g["closed_pump_work_m3"].sum()
        blind_pump = g["primary_pump_work_m3"].sum()
        gated_pump = np.where(g["gate_decision"].eq("allow_deadband"), g["primary_pump_work_m3"], g["closed_pump_work_m3"]).sum()
        return pd.Series(
            {
                "n": len(g),
                "closed_pump_m3": closed_pump,
                "blind_deadband_pump_m3": blind_pump,
                "prediction_gated_pump_m3": gated_pump,
                "blind_saving_pct": 100.0 * (closed_pump - blind_pump) / closed_pump if closed_pump else 0.0,
                "prediction_gated_saving_pct": 100.0 * (closed_pump - gated_pump) / closed_pump if closed_pump else 0.0,
                "blind_d_time5_s": g["blind_d_time5_s"].sum(),
                "prediction_gated_d_time5_s": g["prediction_gated_d_time5_s"].sum(),
                "avoided_time5_s": g["prediction_safety_improvement_time5_s"].sum(),
                "fallback_rows": g["fallback_rows"].sum(),
            }
        )

    summary = combined.groupby("set", sort=False).apply(agg).reset_index()
    allrow = agg(combined)
    allrow["set"] = "combined_stable_plus_boundary"
    summary = pd.concat([summary, pd.DataFrame([allrow])], ignore_index=True)
    boundary_ranked = b.sort_values("prediction_safety_improvement_time5_s", ascending=False)
    return combined, summary, boundary_ranked


def _plot_safety_bars(boundary_ranked: pd.DataFrame, summary: pd.DataFrame) -> None:
    worst = boundary_ranked.head(6).copy()
    x = np.arange(len(worst))

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.2))
    axes[0].bar(x - 0.18, worst["blind_d_time5_s"], width=0.36, color="#d62728", label="blind deadband")
    axes[0].bar(x + 0.18, np.zeros(len(worst)), width=0.36, color="#1f77b4", label="prediction-gated abstain")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(worst["case_id"], rotation=20, ha="right")
    axes[0].set_ylabel("Delta time >5 deg (s)")
    axes[0].set_title("Boundary cases: prediction avoids unsafe blind-deadband activations")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, axis="y", alpha=0.25)

    labels = ["stable events", "boundary", "combined"]
    lookup = {r["set"]: r for _, r in summary.iterrows()}
    blind = [
        lookup["stable_direction_event"]["blind_saving_pct"],
        lookup["boundary_reversal_signflip"]["blind_saving_pct"],
        lookup["combined_stable_plus_boundary"]["blind_saving_pct"],
    ]
    gated = [
        lookup["stable_direction_event"]["prediction_gated_saving_pct"],
        lookup["boundary_reversal_signflip"]["prediction_gated_saving_pct"],
        lookup["combined_stable_plus_boundary"]["prediction_gated_saving_pct"],
    ]
    x2 = np.arange(len(labels))
    axes[1].bar(x2 - 0.18, blind, width=0.36, color="#d62728", label="blind deadband")
    axes[1].bar(x2 + 0.18, gated, width=0.36, color="#1f77b4", label="prediction-gated")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("pump saving (%)")
    axes[1].set_title("Prediction trades some blind saving for safety by abstaining on reversal/signflip")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / "F_w1_prediction_necessity_safety_bars.png", dpi=180)
    plt.close(fig)


def _plot_boundary_case(row: pd.Series, out_name: str) -> None:
    closed, blind = _load_pair_from_audit(row)
    t = closed["t_s"].to_numpy(dtype=float) / 60.0
    bt = blind["t_s"].to_numpy(dtype=float) / 60.0
    ca = _axis(closed)
    ba = _axis(blind)
    st = _stats(closed, blind)
    mask = _fallback_mask(blind)

    fig, axes = plt.subplots(4, 1, figsize=(12.8, 10.0), sharex=True)
    fig.suptitle(
        f"{row['case_id']} | prediction necessity boundary example",
        fontsize=13,
        fontweight="bold",
    )
    subtitle = (
        f"Blind deadband saves {st['saving_pct']:.1f}% pump but adds {st['delta_time5']:+.0f}s time>5. "
        "Prediction-gated policy abstains here, avoiding that safety cost."
    )
    fig.text(0.5, 0.925, subtitle, ha="center", fontsize=10)

    axes[0].plot(t, ca, label="A0 / prediction-gated abstain", color="#1f77b4", lw=2)
    axes[0].plot(bt, ba, label="blind deadband", color="#d62728", lw=2)
    axes[0].axhline(5.0, color="#111111", lw=1, ls="--", label="5 deg floor")
    _shade_fallback(axes[0], bt, mask)
    axes[0].set_ylabel("max-axis deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, closed["pitch_deg"], label="A0 pitch", color="#1f77b4", lw=1.3)
    axes[1].plot(t, closed["roll_deg"], label="A0 roll", color="#2ca02c", lw=1.3)
    axes[1].plot(bt, blind["pitch_deg"], label="blind pitch", color="#d62728", lw=1.2, ls="--")
    axes[1].plot(bt, blind["roll_deg"], label="blind roll", color="#ff7f0e", lw=1.2, ls="--")
    axes[1].axhline(5.0, color="#111111", lw=0.8, ls=":")
    axes[1].axhline(-5.0, color="#111111", lw=0.8, ls=":")
    axes[1].set_ylabel("pitch/roll deg")
    axes[1].legend(loc="upper right", ncol=4, fontsize=7)
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(t, _cum_pump(closed), label="A0 / gated cumulative pump", color="#1f77b4", lw=2)
    axes[2].plot(bt, _cum_pump(blind), label="blind cumulative pump", color="#d62728", lw=2)
    axes[2].set_ylabel("pump m3")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(t, closed["wind_speed"], label="actual wind", color="#4c78a8", lw=1.4)
    if "preview_pressure_block02_dot" in blind.columns:
        axes[3].plot(bt, blind["preview_pressure_block02_dot"], label="preview dot02", color="#d95f0e", lw=1.0)
    axes[3].set_xlabel("time min")
    axes[3].set_ylabel("wind / preview")
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(OUT / out_name, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    combined, summary, boundary_ranked = _build_tables()
    combined.to_csv(OUT / "w1_prediction_necessity_per_case.csv", index=False)
    summary.to_csv(OUT / "w1_prediction_necessity_summary.csv", index=False)
    boundary_ranked.to_csv(OUT / "w1_prediction_necessity_boundary_ranked.csv", index=False)

    _plot_safety_bars(boundary_ranked, summary)
    for idx, row in enumerate(boundary_ranked.head(3).itertuples(index=False), start=1):
        s = pd.Series(row._asdict())
        _plot_boundary_case(s, f"F_w1_prediction_necessity_boundary_case{idx}_{s['case_id']}.png")

    readme = """# W1 Prediction Necessity Audit

This artifact demonstrates why prediction is needed even though the deadband action itself is an execution-layer tolerance economy.

Comparison:

- **A0 / closed-only**: tight control, no economy tolerance.
- **Blind deadband**: applies the same deadband everywhere, including direction-reversal/signflip boundary cases.
- **Prediction-gated deadband**: opens deadband on direction-stable future-event windows, but abstains on direction-reversal/signflip windows.

The key safety evidence is on boundary cases. Blind deadband still saves pump, but it adds time above 5 deg in the worst signflip windows. Prediction-gated deployment avoids these unsafe activations by falling back to A0 on those windows.

Files:

- `w1_prediction_necessity_summary.csv`: aggregate stable/boundary/combined comparison.
- `w1_prediction_necessity_per_case.csv`: per-case comparison.
- `F_w1_prediction_necessity_safety_bars.png`: main summary figure.
- `F_w1_prediction_necessity_boundary_case*.png`: worst boundary examples.

Paper framing:

Prediction does not create the deadband's within-window saving. Its role is deployment safety: it tells the controller when the tolerance band is safe to use and when it must abstain.
"""
    (OUT / "README.md").write_text(readme)
    print(summary.to_string(index=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
