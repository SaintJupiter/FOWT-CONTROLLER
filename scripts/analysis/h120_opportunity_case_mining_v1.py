#!/usr/bin/env python3
"""Mine relief/reversal opportunity cases from existing A0 vs budget runs.

This script is intentionally read-only with respect to controller behavior.  It
uses existing 1h A0/budget100 runs to identify forecast-observable transient
relief cases, summarize pump/comfort tradeoffs, and render a few representative
trajectory figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
CLOSING = ROOT / "closing_pareto_v1" / "raw_tables"
OUT = ROOT / "opportunity_case_mining_v1"
GROUPS = ("guard10_3600", "broader20_3600")
A0 = "a0_baseline"
BUDGET = "budget100"


@dataclass
class CasePaths:
    group: str
    case: str
    a0_ts: Path
    budget_ts: Path
    a0_log: Path


def case_from_path(path: Path) -> str:
    name = path.name
    name = re.sub(r"_prediction_primary_econ_timeseries\.csv$", "", name)
    name = re.sub(r"_prediction_primary_econ_planner_log\.csv$", "", name)
    return name


def safe_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if np.isscalar(default):
            return series.fillna(default)
        return series.fillna(pd.Series(default, index=df.index))
    return pd.Series(default, index=df.index, dtype=float)


def paths_for_group(group: str) -> list[CasePaths]:
    base = ROOT / group
    rows: list[CasePaths] = []
    for a0_ts in sorted((base / A0 / "timeseries").glob("*_timeseries.csv")):
        case = case_from_path(a0_ts)
        budget_ts = base / BUDGET / "timeseries" / a0_ts.name
        a0_log = base / A0 / "planner_logs" / a0_ts.name.replace(
            "_timeseries.csv", "_planner_log.csv"
        )
        if budget_ts.exists() and a0_log.exists():
            rows.append(CasePaths(group, case, a0_ts, budget_ts, a0_log))
    return rows


def planner_features(path: Path) -> dict[str, float | int | bool]:
    d = pd.read_csv(path, low_memory=False)
    b0 = safe_series(d, "raw_pressure_block0_norm")
    b1 = safe_series(d, "raw_pressure_block1_norm")
    b2 = safe_series(d, "raw_pressure_block2_norm")
    near_peak = pd.concat([b0, b1, b2], axis=1).max(axis=1)
    early_peak = pd.concat([b0, b1], axis=1).max(axis=1)
    near_drop = early_peak - b2

    far_cols = [
        "far_horizon_norm_60_80",
        "far_horizon_norm_80_100",
        "far_horizon_norm_100_120",
    ]
    far = [safe_series(d, c) for c in far_cols]
    far_min = pd.concat(far, axis=1).min(axis=1)
    far_max = pd.concat(far, axis=1).max(axis=1)
    near_max = safe_series(d, "far_horizon_near_max", default=np.nan)
    if near_max.isna().all():
        near_max = near_peak
    far_relief = near_max - far_min

    reversal = (
        (safe_series(d, "far_horizon_reversal") > 0.5)
        | (safe_series(d, "far_horizon_direction_shift") > 0.5)
        | (safe_series(d, "far_horizon_dir_shift_deg") >= 45.0)
        | (safe_series(d, "h120_scheduler_far_signflip_risk") > 0.5)
    )
    reintensification = safe_series(
        d, "h120_scheduler_far_reintensification_after_relief"
    ) > 0.5

    high_then_relief = (early_peak >= 0.65) & (near_drop >= 0.20)
    rise_then_fall = (b1 >= b0 + 0.10) & (b2 <= b1 - 0.20)
    h120_far_relief = (near_max >= 0.65) & (far_relief >= 0.25)
    h120_far_reintensify = (far_max >= far_min + 0.25) & (far_max >= 0.75)

    return {
        "decision_rows": int(len(d)),
        "near_high_rows": int((near_peak >= 0.65).sum()),
        "high_then_relief_rows": int(high_then_relief.sum()),
        "rise_then_fall_rows": int(rise_then_fall.sum()),
        "h120_far_relief_rows": int(h120_far_relief.sum()),
        "reversal_or_signflip_rows": int(reversal.sum()),
        "reintensification_rows": int(reintensification.sum()),
        "h120_far_reintensify_rows": int(h120_far_reintensify.sum()),
        "max_near_peak": float(near_peak.max()),
        "max_near_drop": float(near_drop.max()),
        "max_far_relief_drop": float(far_relief.max()),
        "mean_far_min_60_120": float(far_min.mean()),
        "forecast_relief_decay_candidate": bool(
            high_then_relief.sum() > 0 or h120_far_relief.sum() > 0
        ),
        "fast_callback_candidate": bool(
            high_then_relief.sum() > 0 or rise_then_fall.sum() > 0
        ),
        "h120_supervisory_relief_candidate": bool(h120_far_relief.sum() > 0),
        "direction_reversal_candidate": bool(reversal.sum() > 0),
    }


def load_case_metrics() -> pd.DataFrame:
    d = pd.read_csv(CLOSING / "closing_pareto_case_metrics.csv")
    a0 = d[d["arm"] == A0].copy()
    b = d[d["arm"] == BUDGET].copy()
    merged = a0.merge(
        b,
        on=["group", "case"],
        suffixes=("_a0", "_budget100"),
        validate="one_to_one",
    )
    merged["pump_saved_m3"] = merged["pump_m3_a0"] - merged["pump_m3_budget100"]
    merged["pump_saving_pct"] = np.where(
        merged["pump_m3_a0"] > 1e-9,
        100.0 * merged["pump_saved_m3"] / merged["pump_m3_a0"],
        0.0,
    )
    for metric in [
        "time_gt3_s",
        "time_gt4_s",
        "time_gt45_s",
        "time_gt5_s",
        "idle_gt5_s",
        "p95_max_axis_deg",
        "max_axis_deg",
    ]:
        merged[f"delta_{metric}"] = (
            merged[f"{metric}_budget100"] - merged[f"{metric}_a0"]
        )
    return merged


def subset_summary(d: pd.DataFrame) -> pd.DataFrame:
    subsets: list[tuple[str, pd.Series]] = [
        ("all_existing_cases", pd.Series(True, index=d.index)),
        ("forecast_relief_decay_ex_ante", d["forecast_relief_decay_candidate"]),
        ("fast_callback_ex_ante", d["fast_callback_candidate"]),
        ("h120_supervisory_relief_ex_ante", d["h120_supervisory_relief_candidate"]),
        ("direction_reversal_or_signflip", d["direction_reversal_candidate"]),
        (
            "demo_high_saving_budget_ge20pct",
            (d["pump_saving_pct"] >= 20.0) & (d["pump_m3_a0"] > 1e-6),
        ),
        (
            "demo_relief_and_budget_ge20pct",
            d["forecast_relief_decay_candidate"]
            & (d["pump_saving_pct"] >= 20.0)
            & (d["pump_m3_a0"] > 1e-6),
        ),
    ]
    rows = []
    total_a0 = d["pump_m3_a0"].sum()
    for name, mask in subsets:
        s = d[mask.fillna(False)]
        if s.empty:
            rows.append(
                {
                    "subset": name,
                    "cases": 0,
                    "a0_pump_m3": 0.0,
                    "budget100_pump_m3": 0.0,
                    "pump_saved_m3": 0.0,
                    "pump_saving_pct": 0.0,
                    "share_of_total_a0_pump_pct": 0.0,
                    "delta_time_gt3_s": 0,
                    "delta_time_gt4_s": 0,
                    "delta_time_gt5_s": 0,
                    "delta_idle_gt5_s": 0,
                    "mean_delta_p95_max_axis_deg": 0.0,
                    "max_delta_max_axis_deg": 0.0,
                }
            )
            continue
        a0_pump = float(s["pump_m3_a0"].sum())
        b_pump = float(s["pump_m3_budget100"].sum())
        rows.append(
            {
                "subset": name,
                "cases": int(len(s)),
                "a0_pump_m3": a0_pump,
                "budget100_pump_m3": b_pump,
                "pump_saved_m3": a0_pump - b_pump,
                "pump_saving_pct": 100.0 * (a0_pump - b_pump) / a0_pump
                if a0_pump > 1e-9
                else 0.0,
                "share_of_total_a0_pump_pct": 100.0 * a0_pump / total_a0
                if total_a0 > 1e-9
                else 0.0,
                "delta_time_gt3_s": int(s["delta_time_gt3_s"].sum()),
                "delta_time_gt4_s": int(s["delta_time_gt4_s"].sum()),
                "delta_time_gt5_s": int(s["delta_time_gt5_s"].sum()),
                "delta_idle_gt5_s": int(s["delta_idle_gt5_s"].sum()),
                "mean_delta_p95_max_axis_deg": float(
                    s["delta_p95_max_axis_deg"].mean()
                ),
                "max_delta_max_axis_deg": float(s["delta_max_axis_deg"].max()),
            }
        )
    return pd.DataFrame(rows)


def plot_case(case: pd.Series, case_paths: CasePaths, out_dir: Path) -> Path:
    a0 = pd.read_csv(case_paths.a0_ts, low_memory=False)
    b = pd.read_csv(case_paths.budget_ts, low_memory=False)
    log = pd.read_csv(case_paths.a0_log, low_memory=False)
    t = safe_series(a0, "t_s", default=np.arange(len(a0))).to_numpy() / 60.0
    tb = safe_series(b, "t_s", default=np.arange(len(b))).to_numpy() / 60.0
    a0_axis = np.maximum(
        np.abs(safe_series(a0, "pitch_deg").to_numpy()),
        np.abs(safe_series(a0, "roll_deg").to_numpy()),
    )
    b_axis = np.maximum(
        np.abs(safe_series(b, "pitch_deg").to_numpy()),
        np.abs(safe_series(b, "roll_deg").to_numpy()),
    )
    a0_pump = np.abs(safe_series(a0, "pump_total_rate_m3_min").to_numpy())
    b_pump = np.abs(safe_series(b, "pump_total_rate_m3_min").to_numpy())
    log_t = safe_series(log, "current_time_s", default=np.arange(len(log)) * 600).to_numpy() / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t, a0_axis, label="A0 v1.6 max axis", lw=1.8)
    axes[0].plot(tb, b_axis, label="budget100 max axis", lw=1.8)
    axes[0].axhline(5.0, color="crimson", ls="--", lw=1, label="hard floor 5 deg")
    axes[0].axhline(4.0, color="orange", ls=":", lw=1)
    axes[0].set_ylabel("max axis deg")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    axes[1].plot(t, a0_pump, label="A0 pump rate", lw=1.5)
    axes[1].plot(tb, b_pump, label="budget100 pump rate", lw=1.5)
    axes[1].set_ylabel("pump m3/min")
    axes[1].legend(loc="upper right", fontsize=8)

    for col, label, style in [
        ("raw_pressure_block0_norm", "0-20m", "-"),
        ("raw_pressure_block1_norm", "20-40m", "-"),
        ("raw_pressure_block2_norm", "40-60m", "-"),
        ("far_horizon_norm_60_80", "60-80m", "--"),
        ("far_horizon_norm_80_100", "80-100m", "--"),
        ("far_horizon_norm_100_120", "100-120m", "--"),
    ]:
        if col in log.columns:
            axes[2].plot(log_t, safe_series(log, col), style, marker="o", ms=3, label=label)
    axes[2].set_ylabel("pressure norm")
    axes[2].set_xlabel("time in case (min)")
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)

    title = (
        f"{case['group']} / {case['case']} | saved {case['pump_saved_m3']:.1f} m3 "
        f"({case['pump_saving_pct']:.1f}%), dT>5={case['delta_time_gt5_s']:.0f}s"
    )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / f"{case['group']}__{case['case']}.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def write_markdown(summary: pd.DataFrame, cases: pd.DataFrame, figure_paths: list[Path]) -> None:
    top_ex = cases.sort_values("pump_saved_m3", ascending=False).head(8)
    rel = summary[summary["subset"] == "forecast_relief_decay_ex_ante"].iloc[0]
    demo = summary[summary["subset"] == "demo_relief_and_budget_ge20pct"].iloc[0]
    allrow = summary[summary["subset"] == "all_existing_cases"].iloc[0]

    lines = [
        "# Opportunity Case Mining Summary",
        "",
        "## Purpose",
        "",
        "This read-only pass checks whether pump-saving should be presented through a regime-conditioned lens: cases where forecast pressure/wind rises and then relieves, so the controller can avoid chasing a transient posture/target.",
        "",
        "## Key Results",
        "",
        f"- Existing A0 vs budget100 casebook total: {int(allrow['cases'])} cases, pump saving {allrow['pump_saving_pct']:.2f}% ({allrow['pump_saved_m3']:.1f} m3).",
        f"- Forecast-observable relief/decay candidates: {int(rel['cases'])} cases, covering {rel['share_of_total_a0_pump_pct']:.1f}% of A0 pump, with {rel['pump_saving_pct']:.2f}% pump reduction under budget100.",
        f"- Favorable relief/decay demo subset with >=20% observed saving: {int(demo['cases'])} cases, {demo['pump_saving_pct']:.2f}% pump reduction. This subset is useful for figures, but because it uses observed saving, it should not replace the full-book result.",
        "",
        "## Interpretation",
        "",
        "- It is legitimate to add a regime-conditioned evaluation because most windows are low-wind/no-action and dilute the effect. The subset rule must be observable before control action, such as forecast relief/decay, pressure ramp-then-drop, or 60-120min supervisory relief.",
        "- It is not legitimate to claim the result-selected demo subset as the global controller average. Use it as case-study evidence only.",
        "- The 120min forecast can be kept as a supervisory/case-discovery signal: it identifies far relief, re-intensification, reversal, or sign flip context. The closed-loop pump-saving number should still be tied to the actual controller arm being evaluated.",
        "- Extending selected examples to 3-4h is useful for the narrative, because the operator can see multiple forecast-conditioned decisions and the pump/posture tradeoff over a longer process. It should be a case-study extension, not a replacement for the fixed-window validation metric.",
        "",
        "## Top Existing Demonstration Cases",
        "",
        "| group | case | saved m3 | saving % | d time>5 s | forecast relief/decay | h120 relief | reversal/signflip |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for _, r in top_ex.iterrows():
        lines.append(
            f"| {r['group']} | {r['case']} | {r['pump_saved_m3']:.1f} | "
            f"{r['pump_saving_pct']:.1f} | {r['delta_time_gt5_s']:.0f} | "
            f"{bool(r['forecast_relief_decay_candidate'])} | "
            f"{bool(r['h120_supervisory_relief_candidate'])} | "
            f"{bool(r['direction_reversal_candidate'])} |"
        )
    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
        ]
    )
    for p in figure_paths:
        lines.append(f"- `{p}`")
    lines.extend(
        [
            "",
            "## Recommended Paper Framing",
            "",
            "Use three separate statements: full-book result, forecast-defined opportunity-regime result, and selected visual case studies. This gives the paper a strong, honest story: the method is most valuable in transient relief/decay wind processes, while the full-book average remains reported for generality.",
            "",
        ]
    )
    (OUT / "paper_ready" / "opportunity_case_mining_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    (OUT / "raw_tables").mkdir(parents=True, exist_ok=True)
    fig_dir = OUT / "paper_ready" / "figures" / "opportunity_cases"
    fig_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_case_metrics()
    features = []
    path_map: dict[tuple[str, str], CasePaths] = {}
    for group in GROUPS:
        for p in paths_for_group(group):
            path_map[(p.group, p.case)] = p
            feat = {"group": p.group, "case": p.case}
            feat.update(planner_features(p.a0_log))
            features.append(feat)
    feat_df = pd.DataFrame(features)
    cases = metrics.merge(feat_df, on=["group", "case"], how="left")
    cases["paper_demo_candidate"] = (
        cases["forecast_relief_decay_candidate"].fillna(False)
        & (cases["pump_saving_pct"] >= 20.0)
        & (cases["pump_m3_a0"] > 1e-6)
    )
    cases["safeish_demo_candidate"] = cases["paper_demo_candidate"] & (
        cases["delta_time_gt5_s"] <= 120.0
    )
    cases.to_csv(OUT / "raw_tables" / "opportunity_case_table.csv", index=False)

    summary = subset_summary(cases)
    summary.to_csv(OUT / "raw_tables" / "opportunity_subset_summary.csv", index=False)

    # Pick a small visual set: high-saving relief/decay cases first, then top savers.
    visual = cases[cases["paper_demo_candidate"]].sort_values(
        ["safeish_demo_candidate", "pump_saved_m3"], ascending=[False, False]
    )
    if len(visual) < 4:
        visual = pd.concat(
            [visual, cases.sort_values("pump_saved_m3", ascending=False)]
        ).drop_duplicates(["group", "case"])
    visual = visual.head(5)
    figure_paths = []
    for _, r in visual.iterrows():
        p = path_map.get((r["group"], r["case"]))
        if p is not None:
            figure_paths.append(plot_case(r, p, fig_dir))

    write_markdown(summary, cases, figure_paths)
    print("Wrote:")
    print(OUT / "raw_tables" / "opportunity_case_table.csv")
    print(OUT / "raw_tables" / "opportunity_subset_summary.csv")
    print(OUT / "paper_ready" / "opportunity_case_mining_summary.md")
    for p in figure_paths:
        print(p)


if __name__ == "__main__":
    main()
