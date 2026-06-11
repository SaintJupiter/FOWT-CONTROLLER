#!/usr/bin/env python3
"""Closing attribution analyses for the budget100 Pareto result.

This script is read-only.  It uses existing A0 and budget100 outputs to:

1. Compare case-level regime selection against blind budget100 and random
   same-size selectors.
2. Decompose the worst catch-up boundary case from the natural relief/decay
   expansion casebook.

The point is not to tune the controller.  The point is to make the final paper
claim precise: budget100 is an aggressive Pareto mode whose pump saving is
strongest in forecast-identifiable opportunity regimes, with catch-up/fallback
as the operating boundary.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
FINAL = ROOT / "budget100_final_candidate_v1"
RELIEF = ROOT / "relief_decay_expansion_v1"
RAW = FINAL / "raw_tables"
PAPER = FINAL / "paper_ready"
FIG = PAPER / "figures"


METRICS = [
    "pump_m3",
    "time_gt3_s",
    "time_gt4_s",
    "time_gt45_s",
    "time_gt5_s",
    "idle_gt5_s",
    "fallback_time_s",
    "p95_max_axis_deg",
    "max_axis_deg",
]


def series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if np.isscalar(default):
            return s.fillna(default)
        return s.fillna(pd.Series(default, index=df.index))
    return pd.Series(default, index=df.index, dtype=float)


def fallback_flag(df: pd.DataFrame) -> np.ndarray:
    cols = [
        "target_lookup_fallback",
        "preview_primary_safety_fallback",
        "fallback_reason_missing_state",
        "fallback_reason_missing_table",
        "fallback_reason_missing_columns",
        "fallback_reason_invalid_state_id",
    ]
    out = np.zeros(len(df), dtype=bool)
    for col in cols:
        if col in df.columns:
            out |= series(df, col).to_numpy() > 0.5
    return out


def selected_metrics(table: pd.DataFrame, selected: np.ndarray, name: str) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "arm": name,
        "selected_cases": int(selected.sum()),
        "total_cases": int(len(table)),
    }
    for metric in METRICS:
        chosen = np.where(
            selected,
            table[f"{metric}_budget100"].to_numpy(dtype=float),
            table[f"{metric}_a0"].to_numpy(dtype=float),
        )
        base = table[f"{metric}_a0"].to_numpy(dtype=float)
        row[metric] = float(chosen.sum())
        row[f"delta_{metric}_vs_a0"] = float(chosen.sum() - base.sum())
    a0_pump = float(table["pump_m3_a0"].sum())
    row["pump_saved_m3_vs_a0"] = -float(row["delta_pump_m3_vs_a0"])
    row["pump_saving_pct_vs_a0"] = (
        100.0 * float(row["pump_saved_m3_vs_a0"]) / a0_pump if a0_pump > 1e-9 else 0.0
    )
    return row


def selection_attribution() -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(RAW / "locked53_budget100_expansion_case_table.csv")
    opportunity = table["opportunity_relief_decay_pump_ge150"].to_numpy(dtype=bool)
    all_cases = np.ones(len(table), dtype=bool)
    none = np.zeros(len(table), dtype=bool)
    rows = [
        selected_metrics(table, none, "A0_baseline"),
        selected_metrics(table, all_cases, "blind_budget100_all_cases"),
        selected_metrics(table, opportunity, "forecast_opportunity_selected_only"),
    ]

    rng = np.random.default_rng(20260525)
    n_select = int(opportunity.sum())
    random_rows = []
    for i in range(5000):
        selected = np.zeros(len(table), dtype=bool)
        selected[rng.choice(len(table), size=n_select, replace=False)] = True
        row = selected_metrics(table, selected, f"random_same_size_{i:04d}")
        random_rows.append(row)
    random_df = pd.DataFrame(random_rows)
    rows.append(
        {
            "arm": "random_same_size_mean",
            "selected_cases": n_select,
            "total_cases": int(len(table)),
            "pump_saved_m3_vs_a0": float(random_df["pump_saved_m3_vs_a0"].mean()),
            "pump_saving_pct_vs_a0": float(random_df["pump_saving_pct_vs_a0"].mean()),
            "delta_time_gt5_s_vs_a0": float(random_df["delta_time_gt5_s_vs_a0"].mean()),
            "delta_idle_gt5_s_vs_a0": float(random_df["delta_idle_gt5_s_vs_a0"].mean()),
            "delta_fallback_time_s_vs_a0": float(
                random_df["delta_fallback_time_s_vs_a0"].mean()
            ),
            "delta_p95_max_axis_deg_vs_a0": float(
                random_df["delta_p95_max_axis_deg_vs_a0"].mean()
            ),
        }
    )
    summary = pd.DataFrame(rows)

    opp_saving = float(
        summary.loc[
            summary["arm"] == "forecast_opportunity_selected_only",
            "pump_saved_m3_vs_a0",
        ].iloc[0]
    )
    random_df["beats_opportunity_saving"] = (
        random_df["pump_saved_m3_vs_a0"] >= opp_saving
    )
    return summary, random_df


def boundary_case() -> tuple[pd.DataFrame, Path]:
    deltas = pd.read_csv(RELIEF / "raw_tables" / "relief_decay_expansion_case_delta.csv")
    worst = deltas.sort_values("pump_saved_m3", ascending=True).iloc[0]
    case = str(worst["case"])
    a0_path = RELIEF / "a0_baseline" / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
    b_path = RELIEF / "budget100" / "timeseries" / f"{case}_prediction_primary_econ_timeseries.csv"
    a0 = pd.read_csv(a0_path, low_memory=False)
    b = pd.read_csv(b_path, low_memory=False)
    n = min(len(a0), len(b))
    a0 = a0.iloc[:n].copy()
    b = b.iloc[:n].copy()
    t = series(a0, "t_s", default=np.arange(n)).to_numpy() / 60.0
    a0_axis = np.maximum(np.abs(series(a0, "pitch_deg")), np.abs(series(a0, "roll_deg"))).to_numpy()
    b_axis = np.maximum(np.abs(series(b, "pitch_deg")), np.abs(series(b, "roll_deg"))).to_numpy()
    a0_pump = np.abs(series(a0, "pump_total_rate_m3_min").to_numpy())
    b_pump = np.abs(series(b, "pump_total_rate_m3_min").to_numpy())
    a0_fb = fallback_flag(a0)
    b_fb = fallback_flag(b)

    bucket = np.floor(series(a0, "t_s", default=np.arange(n)).to_numpy() / 600.0).astype(int)
    rows = []
    for k in sorted(np.unique(bucket)):
        mask = bucket == k
        rows.append(
            {
                "case": case,
                "bucket": int(k),
                "start_min": float(t[mask][0]),
                "end_min": float(t[mask][-1]),
                "a0_pump_m3": float(a0_pump[mask].sum() / 60.0),
                "budget100_pump_m3": float(b_pump[mask].sum() / 60.0),
                "delta_pump_m3": float((b_pump[mask].sum() - a0_pump[mask].sum()) / 60.0),
                "a0_max_axis_peak": float(np.nanmax(a0_axis[mask])),
                "budget100_max_axis_peak": float(np.nanmax(b_axis[mask])),
                "a0_fallback_s": int(a0_fb[mask].sum()),
                "budget100_fallback_s": int(b_fb[mask].sum()),
                "delta_fallback_s": int(b_fb[mask].sum() - a0_fb[mask].sum()),
            }
        )
    timeline = pd.DataFrame(rows)

    cum_a0 = np.cumsum(a0_pump) / 60.0
    cum_b = np.cumsum(b_pump) / 60.0
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(t, a0_axis, label="A0 max-axis", lw=1.4)
    axes[0].plot(t, b_axis, label="budget100 max-axis", lw=1.4)
    axes[0].axhline(5.0, color="crimson", ls="--", lw=1, label="5 deg hard floor")
    axes[0].set_ylabel("deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].plot(t, a0_pump, label="A0 pump rate", lw=1.2)
    axes[1].plot(t, b_pump, label="budget100 pump rate", lw=1.2)
    axes[1].set_ylabel("m3/min")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, cum_a0, label="A0 cumulative pump", lw=1.4)
    axes[2].plot(t, cum_b, label="budget100 cumulative pump", lw=1.4)
    axes[2].fill_between(t, cum_a0, cum_b, where=cum_b >= cum_a0, color="crimson", alpha=0.15)
    axes[2].set_ylabel("m3")
    axes[2].legend(loc="upper left", fontsize=8)

    axes[3].step(t, a0_fb.astype(int), where="post", label="A0 fallback", lw=1.1)
    axes[3].step(t, b_fb.astype(int), where="post", label="budget100 fallback", lw=1.1)
    axes[3].set_ylabel("fallback")
    axes[3].set_xlabel("time (min)")
    axes[3].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Boundary catch-up case: {case}")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig_path = FIG / "budget100_boundary_catchup_case.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)
    return timeline, fig_path


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c, "")
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    selection, random_df = selection_attribution()
    keep = [
        "arm",
        "selected_cases",
        "pump_saving_pct_vs_a0",
        "pump_saved_m3_vs_a0",
        "delta_time_gt5_s_vs_a0",
        "delta_idle_gt5_s_vs_a0",
        "delta_fallback_time_s_vs_a0",
        "delta_p95_max_axis_deg_vs_a0",
    ]
    selection[keep].to_csv(RAW / "budget100_regime_selection_negative_control.csv", index=False)
    random_df.to_csv(RAW / "budget100_random_same_size_selector_distribution.csv", index=False)

    timeline, fig_path = boundary_case()
    timeline.to_csv(RAW / "budget100_boundary_catchup_timeline.csv", index=False)

    opp = selection[selection["arm"] == "forecast_opportunity_selected_only"].iloc[0]
    rnd = selection[selection["arm"] == "random_same_size_mean"].iloc[0]
    blind = selection[selection["arm"] == "blind_budget100_all_cases"].iloc[0]
    random_beat = float(random_df["beats_opportunity_saving"].mean() * 100.0)

    text = [
        "# Budget100 Closing Attribution",
        "",
        "This is a read-only closing analysis using existing A0 and budget100 outputs. It does not tune the controller.",
        "",
        "## Regime Selection vs Controls",
        "",
        md_table(selection[keep], keep),
        "",
        f"A same-size random selector beats the fixed forecast-opportunity selector's pump saving in {random_beat:.1f}% of 5000 draws.",
        "",
        "Interpretation: the fixed relief/decay opportunity rule enriches for pump-saving cases compared with a random same-size selector, but blind budget100 over the whole locked53 casebook saves more total pump because it also acts on non-relief high-pump regimes. Therefore the forecast contribution should be framed as regime selection / applicability, not as the sole source of the aggressive-mode saving.",
        "",
        "## Boundary Case",
        "",
        f"The worst catch-up case timeline is saved to `{RAW / 'budget100_boundary_catchup_timeline.csv'}` and the figure is `{fig_path}`.",
        "",
        "This case is useful as the paper boundary example: aggressive economy holding can backfire when the apparent opportunity is actually a catch-up/fallback regime. Showing it explicitly strengthens the claim that budget100 is an optional Pareto mode with a visible operating boundary.",
        "",
        "## Bottom Line",
        "",
        f"- Forecast-opportunity selected mode on locked53: {opp['pump_saving_pct_vs_a0']:.2f}% pump saving, fallback delta {opp['delta_fallback_time_s_vs_a0']:.0f}s.",
        f"- Random same-size mean: {rnd['pump_saving_pct_vs_a0']:.2f}% pump saving, fallback delta {rnd['delta_fallback_time_s_vs_a0']:.0f}s.",
        f"- Blind budget100 all cases: {blind['pump_saving_pct_vs_a0']:.2f}% pump saving, fallback delta {blind['delta_fallback_time_s_vs_a0']:.0f}s.",
        "",
        "Use these numbers to avoid overclaiming: budget100 is broadly pump-saving, forecast relief/decay selection identifies a defensible target regime, and the boundary/catch-up case explains why it should not be presented as safety-neutral automation.",
        "",
    ]
    (PAPER / "budget100_closing_attribution.md").write_text(
        "\n".join(text), encoding="utf-8"
    )
    print("Wrote closing attribution analysis to", FINAL)


if __name__ == "__main__":
    main()
