#!/usr/bin/env python3
"""Freeze budget100 as the practical regime-conditioned pump-saving candidate.

The script consolidates existing A0 / budget100 / budget95 runs.  It does not
change controller behavior or rerun simulations.  Outputs are intended for
paper-ready reporting of a regime-conditioned pump-comfort Pareto result.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/economy_pump_budget_closed_loop_probe_v1")
MINING = ROOT / "opportunity_case_mining_v1"
OUT = ROOT / "budget100_final_candidate_v1"
GROUPS = ("guard10_3600", "broader20_3600")
ARMS = ("a0_baseline", "budget100", "budget95")


def case_from_path(path: Path) -> str:
    return re.sub(r"_prediction_primary_econ_timeseries\.csv$", "", path.name)


def series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def case_metrics(path: Path) -> dict[str, float | int | str]:
    d = pd.read_csv(path, low_memory=False)
    pitch = series(d, "pitch_deg").to_numpy()
    roll = series(d, "roll_deg").to_numpy()
    axis = np.maximum(np.abs(pitch), np.abs(roll))
    pump = np.abs(series(d, "pump_total_rate_m3_min").to_numpy())
    fallback_cols = [
        "target_lookup_fallback",
        "preview_primary_safety_fallback",
        "fallback_reason_missing_state",
        "fallback_reason_missing_table",
        "fallback_reason_missing_columns",
        "fallback_reason_invalid_state_id",
    ]
    fallback_any = np.zeros(len(d), dtype=bool)
    for col in fallback_cols:
        if col in d.columns:
            fallback_any |= series(d, col).to_numpy() > 0.5
    return {
        "pump_m3": float(np.nansum(pump) / 60.0),
        "time_gt3_s": int(np.nansum(axis > 3.0)),
        "time_gt4_s": int(np.nansum(axis > 4.0)),
        "time_gt45_s": int(np.nansum(axis > 4.5)),
        "time_gt5_s": int(np.nansum(axis > 5.0)),
        "idle_gt5_s": int(np.nansum((axis > 5.0) & (pump < 0.5))),
        "fallback_time_s": int(np.nansum(fallback_any)),
        "fallback_ratio": float(np.nanmean(fallback_any)) if len(d) else 0.0,
        "p95_max_axis_deg": float(np.nanpercentile(axis, 95)) if len(d) else 0.0,
        "max_axis_deg": float(np.nanmax(axis)) if len(d) else 0.0,
    }


def load_all_metrics() -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        for arm in ARMS:
            ts_dir = ROOT / group / arm / "timeseries"
            if not ts_dir.exists():
                continue
            for path in sorted(ts_dir.glob("*_timeseries.csv")):
                row = case_metrics(path)
                row.update({"group": group, "arm": arm, "case": case_from_path(path)})
                rows.append(row)
    return pd.DataFrame(rows)


def subset_masks(case_table: pd.DataFrame) -> dict[str, set[tuple[str, str]]]:
    all_cases = set(case_table[["group", "case"]].apply(tuple, axis=1))
    opportunity = set(
        case_table[
            case_table["forecast_relief_decay_candidate"]
            & (case_table["pump_m3_a0"] >= 150.0)
        ][["group", "case"]].apply(tuple, axis=1)
    )
    h120_opportunity = set(
        case_table[
            case_table["h120_supervisory_relief_candidate"]
            & (case_table["pump_m3_a0"] >= 150.0)
        ][["group", "case"]].apply(tuple, axis=1)
    )
    demo = set(
        case_table[
            case_table["forecast_relief_decay_candidate"]
            & (case_table["pump_m3_a0"] >= 150.0)
            & (case_table["pump_saving_pct"] >= 20.0)
        ][["group", "case"]].apply(tuple, axis=1)
    )
    return {
        "full_30_casebook": all_cases,
        "opportunity_relief_decay_pump_ge150": opportunity,
        "h120_relief_pump_ge150": h120_opportunity,
        "demo_relief_decay_pump_ge150_and_saving_ge20": demo,
    }


def summarize_subset(metrics: pd.DataFrame, subset: set[tuple[str, str]]) -> pd.DataFrame:
    ref = metrics[
        (metrics["arm"] == "a0_baseline")
        & metrics[["group", "case"]].apply(tuple, axis=1).isin(subset)
    ]
    rows = []
    a0_pump = float(ref["pump_m3"].sum())
    for arm in ("a0_baseline", "budget100", "budget95"):
        arm_df = metrics[
            (metrics["arm"] == arm)
            & metrics[["group", "case"]].apply(tuple, axis=1).isin(subset)
        ]
        if arm_df.empty:
            continue
        row = {
            "arm": arm,
            "cases": int(len(arm_df)),
            "pump_m3": float(arm_df["pump_m3"].sum()),
            "pump_saved_m3_vs_a0": a0_pump - float(arm_df["pump_m3"].sum()),
            "pump_saving_pct_vs_a0": 100.0
            * (a0_pump - float(arm_df["pump_m3"].sum()))
            / a0_pump
            if a0_pump > 1e-9
            else 0.0,
            "time_gt3_s": int(arm_df["time_gt3_s"].sum()),
            "time_gt4_s": int(arm_df["time_gt4_s"].sum()),
            "time_gt45_s": int(arm_df["time_gt45_s"].sum()),
            "time_gt5_s": int(arm_df["time_gt5_s"].sum()),
            "idle_gt5_s": int(arm_df["idle_gt5_s"].sum()),
            "fallback_time_s": int(arm_df["fallback_time_s"].sum()),
            "fallback_ratio_mean": float(arm_df["fallback_ratio"].mean()),
            "p95_max_axis_deg_mean": float(arm_df["p95_max_axis_deg"].mean()),
            "max_axis_deg_max": float(arm_df["max_axis_deg"].max()),
        }
        if arm != "a0_baseline":
            for col in [
                "time_gt3_s",
                "time_gt4_s",
                "time_gt45_s",
                "time_gt5_s",
                "idle_gt5_s",
                "fallback_time_s",
            ]:
                row[f"delta_{col}_vs_a0"] = int(
                    arm_df[col].sum() - ref[col].sum()
                )
            row["delta_p95_mean_deg_vs_a0"] = float(
                arm_df["p95_max_axis_deg"].mean() - ref["p95_max_axis_deg"].mean()
            )
            row["delta_max_axis_max_deg_vs_a0"] = float(
                arm_df["max_axis_deg"].max() - ref["max_axis_deg"].max()
            )
        else:
            for col in [
                "time_gt3_s",
                "time_gt4_s",
                "time_gt45_s",
                "time_gt5_s",
                "idle_gt5_s",
                "fallback_time_s",
            ]:
                row[f"delta_{col}_vs_a0"] = 0
            row["delta_p95_mean_deg_vs_a0"] = 0.0
            row["delta_max_axis_max_deg_vs_a0"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def per_case_delta(metrics: pd.DataFrame, subset_name: str, subset: set[tuple[str, str]]) -> pd.DataFrame:
    a0 = metrics[
        (metrics["arm"] == "a0_baseline")
        & metrics[["group", "case"]].apply(tuple, axis=1).isin(subset)
    ]
    rows = []
    for arm in ("budget100", "budget95"):
        b = metrics[
            (metrics["arm"] == arm)
            & metrics[["group", "case"]].apply(tuple, axis=1).isin(subset)
        ]
        merged = b.merge(a0, on=["group", "case"], suffixes=("", "_a0"))
        for _, r in merged.iterrows():
            rows.append(
                {
                    "subset": subset_name,
                    "arm": arm,
                    "group": r["group"],
                    "case": r["case"],
                    "a0_pump_m3": r["pump_m3_a0"],
                    "pump_m3": r["pump_m3"],
                    "pump_saved_m3": r["pump_m3_a0"] - r["pump_m3"],
                    "pump_saving_pct": 100.0
                    * (r["pump_m3_a0"] - r["pump_m3"])
                    / r["pump_m3_a0"]
                    if r["pump_m3_a0"] > 1e-9
                    else 0.0,
                    "delta_time_gt5_s": r["time_gt5_s"] - r["time_gt5_s_a0"],
                    "delta_idle_gt5_s": r["idle_gt5_s"] - r["idle_gt5_s_a0"],
                    "delta_fallback_time_s": r["fallback_time_s"]
                    - r["fallback_time_s_a0"],
                    "delta_p95_max_axis_deg": r["p95_max_axis_deg"]
                    - r["p95_max_axis_deg_a0"],
                    "delta_max_axis_deg": r["max_axis_deg"] - r["max_axis_deg_a0"],
                }
            )
    return pd.DataFrame(rows)


def concentration(case_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, arm), d in case_delta.groupby(["subset", "arm"]):
        pos = d.copy()
        total = float(pos["pump_saved_m3"].sum())
        pos = pos.sort_values("pump_saved_m3", ascending=False)
        for n in (1, 3, 5):
            top = pos.head(n)
            rows.append(
                {
                    "subset": subset,
                    "arm": arm,
                    "top_n": n,
                    "top_saved_m3": float(top["pump_saved_m3"].sum()),
                    "share_of_total_saved_pct": 100.0
                    * float(top["pump_saved_m3"].sum())
                    / total
                    if abs(total) > 1e-9
                    else 0.0,
                    "top_cases": "; ".join(top["case"].tolist()),
                }
            )
    return pd.DataFrame(rows)


def fallback_concentration(case_delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, arm), d in case_delta.groupby(["subset", "arm"]):
        total = float(d["delta_fallback_time_s"].sum())
        positive = d[d["delta_fallback_time_s"] > 0].copy()
        positive = positive.sort_values("delta_fallback_time_s", ascending=False)
        positive_total = float(positive["delta_fallback_time_s"].sum())
        rows.append(
            {
                "subset": subset,
                "arm": arm,
                "cases": int(len(d)),
                "cases_with_added_fallback": int(len(positive)),
                "total_delta_fallback_time_s": total,
                "positive_added_fallback_time_s": positive_total,
                "max_case_delta_fallback_time_s": float(positive["delta_fallback_time_s"].max())
                if not positive.empty
                else 0.0,
                "top1_share_of_added_fallback_pct": 100.0
                * float(positive["delta_fallback_time_s"].head(1).sum())
                / positive_total
                if positive_total > 1e-9 and not positive.empty
                else 0.0,
                "top3_share_of_added_fallback_pct": 100.0
                * float(positive["delta_fallback_time_s"].head(3).sum())
                / positive_total
                if positive_total > 1e-9 and not positive.empty
                else 0.0,
                "top_fallback_cases": "; ".join(positive["case"].head(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.2f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    raw = OUT / "raw_tables"
    paper = OUT / "paper_ready"
    raw.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    metrics = load_all_metrics()
    case_table = pd.read_csv(MINING / "raw_tables" / "opportunity_case_table.csv")
    masks = subset_masks(case_table)

    summary_frames = []
    case_frames = []
    for name, subset in masks.items():
        s = summarize_subset(metrics, subset)
        s.insert(0, "subset", name)
        summary_frames.append(s)
        case_frames.append(per_case_delta(metrics, name, subset))
    summary = pd.concat(summary_frames, ignore_index=True)
    case_delta = pd.concat(case_frames, ignore_index=True)
    conc = concentration(case_delta)
    fallback_conc = fallback_concentration(case_delta)

    summary.to_csv(raw / "budget100_final_summary_table.csv", index=False)
    case_delta.to_csv(raw / "budget100_final_per_case_delta.csv", index=False)
    conc.to_csv(raw / "budget100_top_case_concentration.csv", index=False)
    fallback_conc.to_csv(raw / "budget100_fallback_concentration.csv", index=False)
    case_table.to_csv(raw / "budget100_opportunity_rule_manifest.csv", index=False)

    opp = summary[
        (summary["subset"] == "opportunity_relief_decay_pump_ge150")
        & summary["arm"].isin(["budget100", "budget95"])
    ][
        [
            "arm",
            "cases",
            "pump_saving_pct_vs_a0",
            "pump_saved_m3_vs_a0",
            "delta_time_gt5_s_vs_a0",
            "delta_idle_gt5_s_vs_a0",
            "delta_fallback_time_s_vs_a0",
            "delta_p95_mean_deg_vs_a0",
            "max_axis_deg_max",
        ]
    ]
    full = summary[
        (summary["subset"] == "full_30_casebook")
        & summary["arm"].isin(["a0_baseline", "budget100", "budget95"])
    ][
        [
            "arm",
            "cases",
            "pump_saving_pct_vs_a0",
            "pump_saved_m3_vs_a0",
            "delta_time_gt5_s_vs_a0",
            "delta_idle_gt5_s_vs_a0",
            "delta_fallback_time_s_vs_a0",
            "delta_p95_mean_deg_vs_a0",
        ]
    ]
    top = conc[
        (conc["subset"] == "opportunity_relief_decay_pump_ge150")
        & (conc["arm"] == "budget100")
    ]
    fallback_top = fallback_conc[
        (fallback_conc["subset"] == "opportunity_relief_decay_pump_ge150")
        & (fallback_conc["arm"] == "budget100")
    ]

    text = [
        "# Budget100 Final Candidate Decision",
        "",
        "## Decision",
        "",
        "`budget100` is frozen as the current best practical controller candidate for the regime-conditioned pump-saving result.",
        "",
        "It should be reported as a **regime-conditioned pump-comfort Pareto result**, not as a full-time average 20% saving and not as pure forecast-attributable saving.",
        "",
        "## Opportunity Rule",
        "",
        "The fixed opportunity domain is: forecast-observable relief/decay candidate plus A0 baseline pump demand >= 150 m3. This is defined before looking at the controller outcome and is therefore usable as a regime-conditioned evaluation rule.",
        "",
        "## Main Opportunity Result",
        "",
        md_table(
            opp,
            [
                "arm",
                "cases",
                "pump_saving_pct_vs_a0",
                "pump_saved_m3_vs_a0",
                "delta_time_gt5_s_vs_a0",
                "delta_idle_gt5_s_vs_a0",
                "delta_fallback_time_s_vs_a0",
                "delta_p95_mean_deg_vs_a0",
                "max_axis_deg_max",
            ],
        ),
        "",
        "Interpretation: `budget100` gives 19.66% pump reduction in the opportunity domain, approximately 20%. The result is not safety-neutral: time>5 increases by 93s and fallback time increases by 746s. The hard v1.6 floor / recovery / fallback logic remains intact, but the aggressive economy mode drives the system closer to the safety boundary more often. Therefore this must be reported as an optional pump-comfort/safety-margin Pareto point, not as a no-regression controller headline.",
        "",
        "## Full Casebook Context",
        "",
        md_table(
            full,
            [
                "arm",
                "cases",
                "pump_saving_pct_vs_a0",
                "pump_saved_m3_vs_a0",
                "delta_time_gt5_s_vs_a0",
                "delta_idle_gt5_s_vs_a0",
                "delta_fallback_time_s_vs_a0",
                "delta_p95_mean_deg_vs_a0",
            ],
        ),
        "",
        "The full 30-case result is larger for `budget100`, but it carries more high-posture exposure. The paper-facing claim should therefore focus on the fixed opportunity regime and disclose the full casebook context separately.",
        "",
        "## Budget100 vs Budget95",
        "",
        "`budget95` was tested as a minimal stronger-budget adjustment. It did not improve the opportunity result: pump saving fell from 19.66% to 16.11%. This non-monotonic behavior is consistent with catch-up/target-lifecycle effects; lower budget does not automatically mean lower total pump.",
        "",
        "## Top Case Concentration",
        "",
        md_table(top, ["top_n", "top_saved_m3", "share_of_total_saved_pct", "top_cases"]),
        "",
        "The result is concentrated enough that per-case deltas must be reported. This is acceptable for a regime-conditioned result, but it should not be presented as a uniform all-regime controller effect.",
        "",
        "## Fallback Concentration",
        "",
        md_table(
            fallback_top,
            [
                "cases_with_added_fallback",
                "total_delta_fallback_time_s",
                "positive_added_fallback_time_s",
                "max_case_delta_fallback_time_s",
                "top1_share_of_added_fallback_pct",
                "top3_share_of_added_fallback_pct",
                "top_fallback_cases",
            ],
        ),
        "",
        "The added fallback is a safety-margin cost. It is concentrated in a small number of cases, so the paper should show the per-case fallback distribution rather than hiding it inside an aggregate pump-saving percentage.",
        "",
        "## Paper Boundary",
        "",
        "- Do not write that `budget100` is a global 20% saving.",
        "- Do not write that it is pure forecast-attributable saving.",
        "- Do not write that 60-120min h120 directly contributes closed-loop pump saving.",
        "- Do write that v1.6 hard floor / fallback / recovery remain unchanged.",
        "- Do write that `budget100` is a practical aggressive economy mode for relief/decay + pump-opportunity regimes, with about 20% pump reduction and disclosed comfort / safety-margin cost.",
        "- Keep the clean no-regression automatic forecast result separate from this aggressive Pareto point.",
        "",
        "## Recommendation",
        "",
        "Use `budget100` as the final practical aggressive Pareto candidate, not as the no-regression headline. Stop budget-family parameter tuning. Additional work, if needed, should mine more fixed-rule relief/decay + pump-opportunity cases for statistical support, not change the subset definition based on results.",
        "",
    ]
    (paper / "budget100_final_candidate_summary.md").write_text("\n".join(text), encoding="utf-8")

    zh = [
        "# Budget100 最终候选结果简报",
        "",
        "## 结论",
        "",
        "`budget100` 固化为当前最实用的 regime-conditioned 节泵候选控制器。",
        "",
        "它的正确表述是：在预测可识别的回落/衰减且原控制器确实有泵量需求的工况中，形成一个 pump-comfort Pareto 节泵结果。不要写成全场景平均 20%，也不要写成纯预测直接带来 20%。",
        "",
        "## 机会域定义",
        "",
        "固定机会域：未来有 relief/decay 回落特征，并且 A0 基准泵量 >= 150 m3。这个规则不是按结果倒选，而是按运行前可解释的工况和基准泵量机会定义。",
        "",
        "## 主结果",
        "",
        md_table(
            opp,
            [
                "arm",
                "cases",
                "pump_saving_pct_vs_a0",
                "pump_saved_m3_vs_a0",
                "delta_time_gt5_s_vs_a0",
                "delta_idle_gt5_s_vs_a0",
                "delta_fallback_time_s_vs_a0",
                "delta_p95_mean_deg_vs_a0",
            ],
        ),
        "",
        "最重要的一行是 `budget100`：节泵 19.66%，可以写作约 20%；time>5 增加 93 秒；idle>5 减少 805 秒；fallback time 增加 746 秒。hard floor / recovery / fallback 没有被禁用，但系统更频繁地靠近并进入安全兜底层，因此它不是“安全无退化”的主结果，而是 aggressive pump-comfort / safety-margin Pareto 点。",
        "",
        "## 为什么不继续调 budget",
        "",
        "`budget95` 反而从 19.66% 降到 16.11%。这说明预算不是越低越好，继续小调预算容易触发后续补泵/catch-up，不值得继续扫。",
        "",
        "## 论文写法",
        "",
        "建议写：在回落/衰减且有实际泵量机会的工况中，本文 aggressive economy mode 可实现约 20% 节泵，同时保留 v1.6 hard floor，但需要披露姿态、fallback 和安全裕度代价。干净的自动预测节泵主结果应与这个 aggressive Pareto 点分开。",
        "",
    ]
    (paper / "budget100_final_candidate_summary_zh.md").write_text("\n".join(zh), encoding="utf-8")

    print("Wrote final candidate package to", OUT)


if __name__ == "__main__":
    main()
