#!/usr/bin/env python3
"""Audit the headroom for a No.4 moderate learned economy arm.

This is a read-only replay-metric audit.  It does not simulate a new
controller.  It answers the planning question that follows the 3hao v2
selector result: how much additional pump saving is needed to reach a 15%
headline, which unopened cases contain that water, and how much rawenv safety
debt appears when the existing aggressive arm is used there.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
DEFAULT_METRICS = (
    DEFAULT_BASE / "degradation_ladder_96case_pair" / "degradation_ladder_case_metrics.csv"
)
DEFAULT_DECISIONS = (
    DEFAULT_BASE
    / "psc_structural_selector_96case_pair_3hao_v2_learned_v1"
    / "structural_selector_case_decisions.csv"
)
DEFAULT_LABELS = DEFAULT_BASE / "replay_selector_labels_96case_pair.csv"
DEFAULT_OUTPUT = DEFAULT_BASE / "psc_4hao_moderate_arm_feasibility_v1"

SAFETY_ARM = "current_forecast_adaptive"
RAWENV_ARM = "learned_rawenv_mainline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-saving-pct", type=float, default=15.0)
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _read_case_deltas(metrics_path: Path) -> pd.DataFrame:
    metrics = pd.read_csv(metrics_path)
    _require_columns(
        metrics,
        [
            "arm",
            "case_id",
            "label",
            "pump_m3",
            "time_gt5_s",
            "time_gt6_s",
            "fallback_s",
            "max_axis_deg",
            "p95_axis_deg",
        ],
        "metrics",
    )
    safety = metrics[metrics["arm"] == SAFETY_ARM].copy()
    economy = metrics[metrics["arm"] == RAWENV_ARM].copy()
    if safety.empty or economy.empty:
        raise ValueError(f"metrics must contain both {SAFETY_ARM} and {RAWENV_ARM}")

    keep = [
        "case_id",
        "label",
        "pump_m3",
        "time_gt5_s",
        "time_gt6_s",
        "fallback_s",
        "max_axis_deg",
        "p95_axis_deg",
    ]
    df = safety[keep].merge(
        economy[keep],
        on=["case_id", "label"],
        suffixes=("_safety", "_rawenv"),
        validate="one_to_one",
    )
    df["d_pump_m3"] = df["pump_m3_rawenv"] - df["pump_m3_safety"]
    df["pump_gain_m3"] = (-df["d_pump_m3"]).clip(lower=0.0)
    df["d_time_gt5_s"] = df["time_gt5_s_rawenv"] - df["time_gt5_s_safety"]
    df["d_time_gt6_s"] = df["time_gt6_s_rawenv"] - df["time_gt6_s_safety"]
    df["d_fallback_s"] = df["fallback_s_rawenv"] - df["fallback_s_safety"]
    df["d_p95_axis_deg"] = df["p95_axis_deg_rawenv"] - df["p95_axis_deg_safety"]
    df["d_max_axis_deg"] = df["max_axis_deg_rawenv"] - df["max_axis_deg_safety"]
    df["metric_risk_marker"] = (
        (df["d_fallback_s"] > 0) | (df["d_time_gt5_s"] >= 60)
    ).astype(int)
    return df


def _case_number(case_id: str) -> int:
    try:
        return int(str(case_id).split("_", 1)[0])
    except Exception:
        return 10**9


def _extract_regime(label: str) -> str:
    text = str(label)
    key = "mixed_regime="
    if key in text:
        return text.split(key, 1)[1].split("|", 1)[0].strip()
    return "unknown"


def _merge_inputs(args: argparse.Namespace) -> pd.DataFrame:
    df = _read_case_deltas(args.metrics)
    labels = pd.read_csv(args.labels)
    decisions = pd.read_csv(args.decisions)
    _require_columns(labels, ["case_id", "rawenv_safe", "rawenv_pump_saves", "use_rawenv"], "labels")
    _require_columns(decisions, ["case_id", "use_economy", "selected_arm"], "decisions")

    label_cols = [
        c
        for c in [
            "case_id",
            "rawenv_safe",
            "rawenv_pump_saves",
            "use_rawenv",
        ]
        if c in labels.columns
    ]
    decision_cols = [
        c
        for c in [
            "case_id",
            "mixed_regime",
            "risk_marker",
            "risk_reason",
            "risk_severity_score",
            "use_economy",
            "decision_reason",
            "selected_arm",
            "learned_signal_rows",
            "learned_confidence_mean",
            "learned_prob_stable_mean",
            "learned_prob_transient_decay_mean",
            "learned_prob_reversal_signflip_mean",
            "learned_prob_reintensification_mean",
            "learned_prob_sustained_high_mean",
            "learned_prob_ramp_onset_mean",
            "learned_event_reversal_signflip_max",
            "learned_event_reintensification_max",
            "learned_event_attention_any_max",
            "learned_opportunity_mean",
            "learned_opportunity_max",
            "learned_risk_max",
        ]
        if c in decisions.columns
    ]
    df = df.merge(labels[label_cols], on="case_id", how="left", validate="one_to_one")
    df = df.merge(decisions[decision_cols], on="case_id", how="left", validate="one_to_one")
    if "mixed_regime" not in df.columns:
        df["mixed_regime"] = df["label"].map(_extract_regime)
    else:
        df["mixed_regime"] = df["mixed_regime"].fillna(df["label"].map(_extract_regime))
    df["risk_marker"] = df.get("risk_marker", df["metric_risk_marker"]).fillna(
        df["metric_risk_marker"]
    )
    df["risk_marker"] = df["risk_marker"].astype(int)
    df["use_economy"] = df["use_economy"].fillna(0).astype(int)
    df["case_sort"] = df["case_id"].map(_case_number)
    return df.sort_values("case_sort").reset_index(drop=True)


def _selected_pump(row: pd.Series, extra_rawenv: bool = False) -> float:
    if int(row["use_economy"]) or extra_rawenv:
        return float(row["pump_m3_rawenv"])
    return float(row["pump_m3_safety"])


def _selected_delta(row: pd.Series, col: str, extra_rawenv: bool = False) -> float:
    if int(row["use_economy"]) or extra_rawenv:
        return float(row[f"{col}_rawenv"] - row[f"{col}_safety"])
    return 0.0


def _scenario_from_mask(
    df: pd.DataFrame,
    baseline_pump: float,
    target_saving_pct: float,
    name: str,
    mask: pd.Series,
    note: str,
) -> dict[str, Any]:
    extra = mask.fillna(False).astype(bool)
    pump = sum(_selected_pump(r, bool(extra.loc[i])) for i, r in df.iterrows())
    saved = baseline_pump - pump
    opened = df[(df["use_economy"] == 1) | extra]
    extra_cases = df[extra]
    return {
        "scenario": name,
        "pump_m3": pump,
        "saved_m3": saved,
        "saving_pct": saved / max(baseline_pump, 1e-9) * 100.0,
        "target_gap_m3": max(0.0, baseline_pump * target_saving_pct / 100.0 - saved),
        "economy_cases": int(len(opened)),
        "extra_cases": int(extra.sum()),
        "extra_saved_m3": float(extra_cases["pump_gain_m3"].sum()),
        "extra_d_time_gt5_s": float(extra_cases["d_time_gt5_s"].sum()),
        "extra_d_fallback_s": float(extra_cases["d_fallback_s"].sum()),
        "extra_risk_cases": int(extra_cases["risk_marker"].sum()),
        "extra_case_ids": ";".join(extra_cases["case_id"].astype(str).tolist()),
        "note": note,
    }


def _candidate_category(row: pd.Series) -> str:
    if int(row["use_economy"]) == 1:
        return "already_opened_by_3hao_v2"
    if float(row["pump_gain_m3"]) <= 0:
        return "no_rawenv_pump_gain"
    if float(row["d_fallback_s"]) == 0 and float(row["d_time_gt5_s"]) <= 0:
        return "strict_unopened"
    if float(row["d_fallback_s"]) == 0 and float(row["d_time_gt5_s"]) <= 10:
        return "tiny_gt5_debt_appendix"
    if float(row["d_fallback_s"]) == 0 and float(row["d_time_gt5_s"]) < 60:
        return "small_gt5_debt_pareto"
    regime = str(row["mixed_regime"])
    if regime == "transient_peak_future_decay":
        return "4hao_relief_edge_candidate"
    if regime == "neutral_mhs_broader":
        return "4hao_neutral_highrisk_candidate"
    if regime == "direction_reversal_boundary":
        return "boundary_falsifier_not_first_release"
    return "other_unopened_rawenv_risk"


def _build_candidate_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["candidate_category"] = out.apply(_candidate_category, axis=1)
    out["rawenv_safety_debt_per_100m3_gt5_s"] = np.where(
        out["pump_gain_m3"] > 0,
        out["d_time_gt5_s"].clip(lower=0) / out["pump_gain_m3"] * 100.0,
        np.nan,
    )
    out["rawenv_safety_debt_per_100m3_fallback_s"] = np.where(
        out["pump_gain_m3"] > 0,
        out["d_fallback_s"].clip(lower=0) / out["pump_gain_m3"] * 100.0,
        np.nan,
    )
    priority = {
        "strict_unopened": 0,
        "tiny_gt5_debt_appendix": 1,
        "small_gt5_debt_pareto": 2,
        "4hao_relief_edge_candidate": 3,
        "4hao_neutral_highrisk_candidate": 4,
        "boundary_falsifier_not_first_release": 5,
        "other_unopened_rawenv_risk": 6,
        "no_rawenv_pump_gain": 7,
        "already_opened_by_3hao_v2": 8,
    }
    out["category_priority"] = out["candidate_category"].map(priority).fillna(99).astype(int)
    cols = [
        "case_id",
        "mixed_regime",
        "candidate_category",
        "use_economy",
        "use_rawenv",
        "risk_marker",
        "pump_m3_safety",
        "pump_m3_rawenv",
        "pump_gain_m3",
        "d_time_gt5_s",
        "d_time_gt6_s",
        "d_fallback_s",
        "d_p95_axis_deg",
        "d_max_axis_deg",
        "rawenv_safety_debt_per_100m3_gt5_s",
        "rawenv_safety_debt_per_100m3_fallback_s",
        "decision_reason",
        "risk_reason",
        "learned_signal_rows",
        "learned_confidence_mean",
        "learned_prob_transient_decay_mean",
        "learned_prob_reintensification_mean",
        "learned_prob_sustained_high_mean",
        "learned_prob_ramp_onset_mean",
        "learned_event_reintensification_max",
        "learned_opportunity_mean",
        "learned_risk_max",
        "label",
    ]
    cols = [c for c in cols if c in out.columns]
    return out.sort_values(["category_priority", "pump_gain_m3"], ascending=[True, False])[
        cols
    ].reset_index(drop=True)


def _summarize_groups(df: pd.DataFrame, target_deficit_m3: float) -> pd.DataFrame:
    unopened_gain = df[(df["use_economy"] == 0) & (df["pump_gain_m3"] > 0)].copy()
    rows: list[dict[str, Any]] = []
    for name, mask in {
        "all_unopened_rawenv_gain": unopened_gain.index == unopened_gain.index,
        "fallback0_time_gt5_le_10": (unopened_gain["d_fallback_s"] == 0)
        & (unopened_gain["d_time_gt5_s"] <= 10),
        "fallback0_time_gt5_lt_60": (unopened_gain["d_fallback_s"] == 0)
        & (unopened_gain["d_time_gt5_s"] < 60),
        "risky_relief_rawenv_gain": (unopened_gain["mixed_regime"] == "transient_peak_future_decay")
        & (unopened_gain["risk_marker"] == 1),
        "risky_neutral_rawenv_gain": (unopened_gain["mixed_regime"] == "neutral_mhs_broader")
        & (unopened_gain["risk_marker"] == 1),
        "boundary_rawenv_gain": (unopened_gain["mixed_regime"] == "direction_reversal_boundary")
        & (unopened_gain["pump_gain_m3"] > 0),
    }.items():
        sub = unopened_gain[mask]
        gain = float(sub["pump_gain_m3"].sum())
        rows.append(
            {
                "pool": name,
                "cases": int(len(sub)),
                "rawenv_gain_m3": gain,
                "required_fraction_to_close_15pct_gap": (
                    target_deficit_m3 / gain if gain > 0 else np.nan
                ),
                "rawenv_d_time_gt5_s": float(sub["d_time_gt5_s"].sum()),
                "rawenv_d_fallback_s": float(sub["d_fallback_s"].sum()),
                "case_ids": ";".join(sub["case_id"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def _write_readout(
    out_dir: Path,
    df: pd.DataFrame,
    scenarios: pd.DataFrame,
    groups: pd.DataFrame,
    target_saving_pct: float,
) -> None:
    baseline_pump = float(df["pump_m3_safety"].sum())
    rawenv_pump = float(df["pump_m3_rawenv"].sum())
    selected_pump = float(
        df.apply(lambda r: r["pump_m3_rawenv"] if int(r["use_economy"]) else r["pump_m3_safety"], axis=1).sum()
    )
    selected_saved = baseline_pump - selected_pump
    target_saved = baseline_pump * target_saving_pct / 100.0
    deficit = max(0.0, target_saved - selected_saved)
    strict_unopened = df[
        (df["use_economy"] == 0)
        & (df["pump_gain_m3"] > 0)
        & (df["d_fallback_s"] == 0)
        & (df["d_time_gt5_s"] <= 0)
    ]
    tiny = df[
        (df["use_economy"] == 0)
        & (df["pump_gain_m3"] > 0)
        & (df["d_fallback_s"] == 0)
        & (df["d_time_gt5_s"] > 0)
        & (df["d_time_gt5_s"] <= 10)
    ]
    under60 = df[
        (df["use_economy"] == 0)
        & (df["pump_gain_m3"] > 0)
        & (df["d_fallback_s"] == 0)
        & (df["d_time_gt5_s"] > 10)
        & (df["d_time_gt5_s"] < 60)
    ]

    best_groups = groups.set_index("pool")
    risky_neutral_fraction = best_groups.loc[
        "risky_neutral_rawenv_gain", "required_fraction_to_close_15pct_gap"
    ]
    risky_relief_fraction = best_groups.loc[
        "risky_relief_rawenv_gain", "required_fraction_to_close_15pct_gap"
    ]

    lines = [
        "# 4hao Moderate Arm Feasibility Audit",
        "",
        "This audit uses already-run 96-case replay metrics only.  It does not simulate a new arm.",
        "",
        "## Headline",
        "",
        f"- 0hao baseline pump: `{baseline_pump:.1f} m3`.",
        f"- 1hao rawenv all-on pump: `{rawenv_pump:.1f} m3`, saving `{(baseline_pump - rawenv_pump) / baseline_pump * 100.0:.2f}%`, but unsafe.",
        f"- 3hao v2 selected pump: `{selected_pump:.1f} m3`, saving `{selected_saved / baseline_pump * 100.0:.2f}%`.",
        f"- Target `{target_saving_pct:.2f}%` needs `{target_saved:.1f} m3` saved, so 4hao must add about `{deficit:.1f} m3` beyond 3hao v2.",
        "",
        "## Strict Selector Headroom",
        "",
        f"- Unopened strict rawenv candidates with `fallback=0` and `d_time_gt5<=0`: `{len(strict_unopened)}` case(s), `{strict_unopened['pump_gain_m3'].sum():.1f} m3` raw saving.",
        "- This means the strict two-arm selector space is effectively saturated; a real 4hao result has to change the action, not just open more 1hao cases.",
        "",
        "## No-Fallback Pareto Candidates",
        "",
        f"- Tiny-debt candidates (`fallback=0`, `0<d_time_gt5<=10s`): `{len(tiny)}` case(s), `{tiny['pump_gain_m3'].sum():.1f} m3` raw saving.",
        f"- Small-debt candidates (`fallback=0`, `10<d_time_gt5<60s`): `{len(under60)}` case(s), `{under60['pump_gain_m3'].sum():.1f} m3` raw saving.",
        "- These are useful as a Pareto appendix or as 4hao smoke-test seeds, but rawenv all-on would change the main claim from strict no-debt safety to a time-debt tradeoff.",
        "",
        "## 4hao Raw-Potential Requirement",
        "",
        f"- Closing the 15% gap would require only `{risky_neutral_fraction * 100.0:.1f}%` of the unopened risky neutral-like raw saving, if a mild arm could harvest it safely.",
        f"- It would require `{risky_relief_fraction * 100.0:.1f}%` of the unopened risky relief raw saving.",
        "- These fractions are planning numbers only.  Rawenv all-on in those same pools has large `time_gt5` and fallback debt, so safety must be proven by actual mild-arm replay.",
        "",
        "## Recommended Next Replay Seeds",
        "",
    ]
    seed_cols = [
        "case_id",
        "mixed_regime",
        "candidate_category",
        "pump_gain_m3",
        "d_time_gt5_s",
        "d_fallback_s",
    ]
    seeds = _build_candidate_table(df)
    seeds = seeds[
        seeds["candidate_category"].isin(
            [
                "tiny_gt5_debt_appendix",
                "strict_unopened",
                "small_gt5_debt_pareto",
                "4hao_relief_edge_candidate",
                "4hao_neutral_highrisk_candidate",
                "boundary_falsifier_not_first_release",
            ]
        )
    ].head(12)
    if seeds.empty:
        lines.append("- No unopened pump-gain seeds found.")
    else:
        lines.append("| case | regime | category | raw gain m3 | raw d_gt5 s | raw d_fallback s |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in seeds[seed_cols].to_dict("records"):
            lines.append(
                f"| {row['case_id']} | {row['mixed_regime']} | {row['candidate_category']} | "
                f"{float(row['pump_gain_m3']):.1f} | {float(row['d_time_gt5_s']):.0f} | "
                f"{float(row['d_fallback_s']):.0f} |"
            )
    lines.extend(
        [
            "",
            "## Scenario Table",
            "",
            "| scenario | saving % | saved m3 | target gap m3 | extra cases | extra raw gain m3 | extra d_gt5 s | extra d_fallback s |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scenarios.to_dict("records"):
        lines.append(
            f"| {row['scenario']} | {float(row['saving_pct']):.2f} | "
            f"{float(row['saved_m3']):.1f} | {float(row['target_gap_m3']):.1f} | "
            f"{int(row['extra_cases'])} | {float(row['extra_saved_m3']):.1f} | "
            f"{float(row['extra_d_time_gt5_s']):.0f} | {float(row['extra_d_fallback_s']):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Do not call 4hao a safe controller yet.",
            "- The most honest next step is a small real replay smoke: one tiny-debt relief case, one boundary falsifier, and a few risky neutral-like high-potential cases under a capped/budgeted mild profile.",
            "- Success criterion for a future 4hao smoke: incremental pump saving large enough to close roughly `883 m3` on the 96-case scale, with `fallback=0` and no material `time_gt5` debt.",
        ]
    )
    (out_dir / "4hao_moderate_arm_feasibility_readout.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _merge_inputs(args)
    baseline_pump = float(df["pump_m3_safety"].sum())
    selected_saved = float(
        df.apply(
            lambda r: r["pump_m3_safety"]
            - (r["pump_m3_rawenv"] if int(r["use_economy"]) else r["pump_m3_safety"]),
            axis=1,
        ).sum()
    )
    target_saved = baseline_pump * float(args.target_saving_pct) / 100.0
    deficit = max(0.0, target_saved - selected_saved)

    unopened_gain = (df["use_economy"] == 0) & (df["pump_gain_m3"] > 0)
    scenarios = [
        _scenario_from_mask(
            df,
            baseline_pump,
            float(args.target_saving_pct),
            "3hao_v2_strict",
            pd.Series(False, index=df.index),
            "Current 3hao v2 selected set.",
        ),
        _scenario_from_mask(
            df,
            baseline_pump,
            float(args.target_saving_pct),
            "3hao_v2_plus_rawenv_fallback0_gt5_le_10",
            unopened_gain & (df["d_fallback_s"] == 0) & (df["d_time_gt5_s"] <= 10),
            "Rawenv all-on for tiny-debt no-fallback unopened candidates.",
        ),
        _scenario_from_mask(
            df,
            baseline_pump,
            float(args.target_saving_pct),
            "3hao_v2_plus_rawenv_fallback0_gt5_lt_60",
            unopened_gain & (df["d_fallback_s"] == 0) & (df["d_time_gt5_s"] < 60),
            "Rawenv all-on for no-fallback candidates below the diagnostic disaster threshold.",
        ),
    ]
    risky_unopened_gain = unopened_gain & (df["risk_marker"] == 1)
    risky_gain_total = float(df.loc[risky_unopened_gain, "pump_gain_m3"].sum())
    for fraction in (0.05, 0.10, 0.20, 0.30, 0.40):
        saved = selected_saved + risky_gain_total * fraction
        scenarios.append(
            {
                "scenario": f"theoretical_4hao_{fraction:.0%}_of_unopened_risky_gain",
                "pump_m3": baseline_pump - saved,
                "saved_m3": saved,
                "saving_pct": saved / max(baseline_pump, 1e-9) * 100.0,
                "target_gap_m3": max(0.0, target_saved - saved),
                "economy_cases": int(df["use_economy"].sum()),
                "extra_cases": int(risky_unopened_gain.sum()),
                "extra_saved_m3": risky_gain_total * fraction,
                "extra_d_time_gt5_s": np.nan,
                "extra_d_fallback_s": np.nan,
                "extra_risk_cases": int(risky_unopened_gain.sum()),
                "extra_case_ids": ";".join(df.loc[risky_unopened_gain, "case_id"].astype(str)),
                "note": "Pump-only planning scenario; safety unknown until real mild-arm replay.",
            }
        )
    scenarios_df = pd.DataFrame(scenarios)
    candidates = _build_candidate_table(df)
    groups = _summarize_groups(df, deficit)

    df.to_csv(out_dir / "4hao_case_delta_table.csv", index=False)
    candidates.to_csv(out_dir / "4hao_candidate_pool.csv", index=False)
    scenarios_df.to_csv(out_dir / "4hao_scenario_summary.csv", index=False)
    groups.to_csv(out_dir / "4hao_headroom_by_pool.csv", index=False)
    _write_readout(out_dir, df, scenarios_df, groups, float(args.target_saving_pct))

    print(f"baseline_pump_m3={baseline_pump:.1f}")
    print(f"3hao_v2_saved_m3={selected_saved:.1f}")
    print(f"target_gap_m3={deficit:.1f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
