#!/usr/bin/env python3
"""Create an evidence-backed optimization roadmap for PSC No.4 regimes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
SAFE_VETO = BASE / "psc_4hao_safe_veto_addback_plan_20260603"
OUT = BASE / "psc_4hao_regime_specialized_optimization_20260603"


REGIME_PLAN = {
    "neutral_mhs_broader": {
        "condition_code": "P2/P3",
        "role": "mature_main_economy",
        "priority": "high",
        "decision": "enable",
        "optimization_goal": "Keep generalized_gate enabled and validate on fresh neutral/falsifier pools.",
        "success_gate": "saved_m3>=2600, fallback=0, time_gt7_delta<=0",
        "next_task": "Task 4",
    },
    "lowrisk_stable_redundant_candidate": {
        "condition_code": "lowrisk",
        "role": "safe_support_economy",
        "priority": "medium",
        "decision": "enable_with_monitoring",
        "optimization_goal": "Keep enabled as support economy but do not let sparse low-pump mass drive global claims.",
        "success_gate": "fallback=0, time_gt7_delta<=0, no false-trigger expansion",
        "next_task": "Task 4",
    },
    "transient_peak_future_decay": {
        "condition_code": "P1",
        "role": "conditional_addback",
        "priority": "high",
        "decision": "refine_rule",
        "optimization_goal": "Recover more of the outcome-informed P1 target without reopening fallback.",
        "success_gate": "saved_m3>=1100 stretch, fallback=0, time_gt7_delta<=0",
        "next_task": "Task 7",
    },
    "direction_reversal_boundary": {
        "condition_code": "W1",
        "role": "conditional_addback_and_warning",
        "priority": "high",
        "decision": "split_save_vs_warning",
        "optimization_goal": "Preserve low-pressure reversal saving and fail closed on high-pressure negative-drop catch-up cases.",
        "success_gate": "saved_m3>=1850, fallback=0, time_gt7_delta<=0",
        "next_task": "Task 5",
    },
    "reintensification_boundary": {
        "condition_code": "W2",
        "role": "conditional_addback_with_veto",
        "priority": "medium",
        "decision": "enable_clean_subcases",
        "optimization_goal": "Keep clean-direction/flat reintensification addback and add blocked-case telemetry.",
        "success_gate": "saved_m3>=700, fallback=0, time_gt7_delta<=0",
        "next_task": "Task 6",
    },
    "sustained_high_safety_event": {
        "condition_code": "W3/W7",
        "role": "safety_watch",
        "priority": "medium",
        "decision": "fail_closed",
        "optimization_goal": "Use as safety regression; only mine specialist if isolated screen finds material saving.",
        "success_gate": "no false trigger; optional branch requires addressable_saved_m3>=250",
        "next_task": "Task 8",
    },
    "quiet_low_opportunity": {
        "condition_code": "quiet",
        "role": "false_trigger_regression",
        "priority": "medium",
        "decision": "baseline_noop",
        "optimization_goal": "Keep baseline/no-op and prevent economy false triggers.",
        "success_gate": "gate_cases=0, fallback=0, saved_m3 not required",
        "next_task": "Task 8",
    },
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(SAFE_VETO / "safe_veto_addback_summary.csv")
    by_stratum = pd.read_csv(SAFE_VETO / "safe_veto_addback_by_stratum.csv")
    selected = pd.read_csv(SAFE_VETO / "rule_distilled_addback_selected_cases.csv")
    return summary, by_stratum, selected


def _wide_by_stratum(by_stratum: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum, group in by_stratum.groupby("stratum", sort=False):
        rec: dict[str, object] = {"stratum": stratum}
        for policy in ["baseline", "current_generalized_gate", "safe_veto", "rule_distilled_addback", "safe_veto_addback"]:
            hit = group[group["policy_name"].eq(policy)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            prefix = policy.replace("current_generalized_gate", "current").replace("rule_distilled_addback", "rule")
            for col in ["cases", "gate_cases", "baseline_pump_m3", "pump_m3", "saved_m3", "time_gt5_s", "time_gt7_s", "fallback_s", "worst_p95_axis_deg"]:
                rec[f"{prefix}_{col}"] = row[col]
        rows.append(rec)
    wide = pd.DataFrame(rows)
    wide["rule_vs_current_saved_delta_m3"] = wide["rule_saved_m3"] - wide["current_saved_m3"]
    wide["rule_vs_current_fallback_reduction_s"] = wide["current_fallback_s"] - wide["rule_fallback_s"]
    wide["rule_vs_current_time_gt7_reduction_s"] = wide["current_time_gt7_s"] - wide["rule_time_gt7_s"]
    return wide


def _matrix(by_stratum: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    wide = _wide_by_stratum(by_stratum)
    selected_counts = (
        selected.groupby(["stratum", "rule_addback_reason"], dropna=False)
        .size()
        .reset_index(name="selected_cases")
    )
    reason_text = (
        selected_counts.groupby("stratum")
        .apply(lambda g: "; ".join(f"{r.rule_addback_reason}:{int(r.selected_cases)}" for r in g.itertuples()))
        .reset_index(name="rule_reasons")
    )
    matrix = wide.merge(reason_text, on="stratum", how="left")
    for stratum, meta in REGIME_PLAN.items():
        for key, value in meta.items():
            matrix.loc[matrix["stratum"].eq(stratum), key] = value
    matrix["rule_reasons"] = matrix["rule_reasons"].fillna("none")
    order = [
        "condition_code",
        "stratum",
        "role",
        "priority",
        "decision",
        "rule_gate_cases",
        "rule_saved_m3",
        "current_saved_m3",
        "rule_vs_current_saved_delta_m3",
        "rule_fallback_s",
        "rule_time_gt7_s",
        "rule_vs_current_fallback_reduction_s",
        "rule_vs_current_time_gt7_reduction_s",
        "rule_reasons",
        "optimization_goal",
        "success_gate",
        "next_task",
    ]
    return matrix[order].sort_values(["priority", "condition_code"], ascending=[True, True])


def _task_dependencies() -> pd.DataFrame:
    rows = [
        (1, "Freeze current evidence and success envelope", "", "done"),
        (2, "Create regime-specialized optimization roadmap", "1", "done"),
        (3, "Implement runtime safe-veto rule dispatcher profile", "1,2", "pending"),
        (4, "Optimize P2/P3 and lowrisk mature economy branches", "3", "pending"),
        (5, "Optimize W1 direction-reversal conditional addback", "3", "pending"),
        (6, "Optimize W2 reintensification conditional addback", "3", "pending"),
        (7, "Refine P1 transient peak/future decay addback", "3", "pending"),
        (8, "Keep W3/W7 and quiet-low fail-closed with regression checks", "3", "pending"),
        (9, "Mine C3/gusty anti-chatter casebook before controller work", "2", "pending"),
        (10, "Run mixed smoke, 80-case replay, and fresh/falsifier validation", "4,5,6,7,8", "pending"),
        (11, "Package accepted and rejected regimes", "10", "pending"),
    ]
    return pd.DataFrame(rows, columns=["task_id", "title", "depends_on", "status"])


def _fmt(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for rec in df[cols].to_dict("records"):
        lines.append("| " + " | ".join(_fmt(rec[col]) for col in cols) + " |")
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, matrix: pd.DataFrame, deps: pd.DataFrame) -> None:
    rule = summary[summary["policy"].eq("rule_distilled_addback")].iloc[0]
    current = summary[summary["policy"].eq("current_generalized_gate")].iloc[0]
    safe = summary[summary["policy"].eq("safe_veto")].iloc[0]
    lines = [
        "# PSC No.4 Regime-Specialized Optimization Roadmap - 2026-06-03",
        "",
        "## Decision",
        "",
        "Proceed with regime-specialized optimization. The best path is a safe-veto runtime dispatcher with conditional rule addback, not a direct copy of old specialist profiles.",
        "",
        "## Headline Metrics",
        "",
        f"- Current generalized_gate: {current['saving_pct']:.2f}% saving, {current['saved_m3']:.1f} m3 saved, {current['fallback_s']:.0f}s fallback, {current['time_gt7_s']:.0f}s time>7.",
        f"- Safe-veto floor: {safe['saving_pct']:.2f}% saving, {safe['saved_m3']:.1f} m3 saved, {safe['fallback_s']:.0f}s fallback, {safe['time_gt7_s']:.0f}s time>7.",
        f"- Rule-distilled addback candidate: {rule['saving_pct']:.2f}% saving, {rule['saved_m3']:.1f} m3 saved, {rule['fallback_s']:.0f}s fallback, {rule['time_gt7_s']:.0f}s time>7.",
        "",
        "## Regime Optimization Matrix",
        "",
        _md_table(
            matrix,
            [
                "condition_code",
                "stratum",
                "decision",
                "rule_gate_cases",
                "rule_saved_m3",
                "rule_fallback_s",
                "rule_time_gt7_s",
                "next_task",
            ],
        ),
        "",
        "## Execution Order",
        "",
        _md_table(deps, ["task_id", "title", "depends_on", "status"]),
        "",
        "## Immediate Next Move",
        "",
        "Implement task 3 first: a runtime experimental profile that reproduces the rule-distilled safe-veto addback using pre-action telemetry. Then optimize P2/lowrisk, W1, W2, and P1 separately before mixed validation.",
    ]
    (OUT / "regime_optimization_roadmap.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, by_stratum, selected = _load()
    matrix = _matrix(by_stratum, selected)
    deps = _task_dependencies()
    summary.to_csv(OUT / "frozen_policy_summary.csv", index=False)
    matrix.to_csv(OUT / "regime_optimization_matrix.csv", index=False)
    deps.to_csv(OUT / "regime_task_dependencies.csv", index=False)
    _write_report(summary, matrix, deps)
    print(OUT / "regime_optimization_roadmap.md")
    print(matrix[["condition_code", "stratum", "decision", "rule_saved_m3", "rule_fallback_s", "next_task"]].to_string(index=False))


if __name__ == "__main__":
    main()
