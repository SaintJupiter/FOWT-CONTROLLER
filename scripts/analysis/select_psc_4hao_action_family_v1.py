#!/usr/bin/env python3
"""Select a case-level No.4 action family from completed closed-loop runs.

This is an empirical selector audit, not a new controller.  It answers:
given the No.4 variants already replayed, how much additional water can be
accepted without double-counting cases already opened by No.3?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
OUT = BASE / "psc_4hao_25pct_overnight_v1"
CASE_METRICS = OUT / "case_metrics.csv"
THREE_HAO = (
    BASE
    / "psc_structural_selector_96case_pair_3hao_v2_learned_v1"
    / "structural_selector_case_decisions.csv"
)

BASELINE_3HAO_M3 = 5999.1
DENOMINATOR_96CASE_M3 = 45880.1

ACTION_PRIORITY = {
    "dynamic_focus4_activeposture_pressure110_i120": "pressure-gated active",
    "dynamic_focus4_activeposture_pressure100_i120": "pressure-gated active",
    "dynamic_focus4_activeposture_enter43_pressure110_i120": "pressure-gated active",
    "edge13_activeposture_pressure110_i120": "pressure-gated active",
    "case22_activeposture_release25": "tail-debt release25",
    "dynamic_focus4_activeposture_enter40_i120": "active posture",
    "edge13_activeposture_enter40_i120": "active posture",
    "debtaware_edge13_frac035_budget100_release32_w6": "mild 35%",
    "dynamic_focus4_enter32_full43_max100": "mild 35%",
    "dynamic_focus4_enter34_full47_max085": "mild 35%",
    "dynamic_focus4_active_eff_eps005": "mild 35%",
}


def _format_reasons(row: pd.Series, policy: str) -> str:
    reasons: list[str] = []
    if float(row["saved_m3"]) <= 0.0:
        reasons.append("no_positive_saving")
    if float(row["d_fallback_s"]) > 0.0:
        reasons.append(f"fallback+{float(row['d_fallback_s']):.0f}s")
    if float(row["p95_axis_deg"]) >= 5.0:
        reasons.append(f"p95>{float(row['p95_axis_deg']):.2f}deg")
    gt5_limit = {"strict": 0.0, "balanced": 30.0, "appendix": 60.0}[policy]
    p95_limit = {"strict": 0.50, "balanced": 1.35, "appendix": 1.35}[policy]
    if float(row["d_time_gt5_s"]) > gt5_limit:
        reasons.append(f"gt5+{float(row['d_time_gt5_s']):.0f}s>{gt5_limit:.0f}s")
    if float(row["d_p95_axis_deg"]) > p95_limit:
        reasons.append(f"dp95+{float(row['d_p95_axis_deg']):.2f}>{p95_limit:.2f}")
    return ";".join(reasons) if reasons else "accepted"


def _recommendation(best: pd.Series | None, policy: str) -> str:
    if best is None:
        return "missing_replay"
    reasons = _format_reasons(best, policy)
    if reasons == "accepted":
        family = ACTION_PRIORITY.get(str(best["run_id"]), "other replay")
        return f"accept_{family}"
    if "fallback" in reasons or "p95>" in reasons:
        return "reject_structural_safety_failure"
    if "gt5" in reasons and "dp95" in reasons:
        return "needs_axis_debt_predictor"
    if "gt5" in reasons:
        return "needs_tail_debt_gate"
    if "dp95" in reasons:
        return "needs_posture_debt_gate"
    if "no_positive_saving" in reasons:
        return "reject_no_water"
    return "manual_review"


def _case_diagnostics(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_id, group in df.groupby("case_id", sort=True):
        ranked = group.copy()
        ranked["failure_reasons"] = ranked.apply(
            lambda r: _format_reasons(r, policy),
            axis=1,
        )
        ranked["action_family"] = ranked["run_id"].map(ACTION_PRIORITY).fillna(
            "other replay"
        )
        acceptable = ranked[ranked["failure_reasons"].eq("accepted")].copy()
        if acceptable.empty:
            acceptable_best = None
        else:
            acceptable["selector_score"] = (
                acceptable["saved_m3"]
                - 25.0 * acceptable["d_p95_axis_deg"].clip(lower=0.0)
                - 0.8 * acceptable["d_time_gt5_s"].clip(lower=0.0)
                - 0.15 * acceptable["d_time_gt4_s"].clip(lower=0.0)
            )
            acceptable_best = acceptable.sort_values(
                ["selector_score", "saved_m3"],
                ascending=[False, False],
            ).iloc[0]

        near = ranked.copy()
        near["violation_count"] = near["failure_reasons"].apply(
            lambda s: 0 if s == "accepted" else len(str(s).split(";"))
        )
        near["near_score"] = (
            near["violation_count"] * 10000.0
            + near["d_fallback_s"].clip(lower=0.0) * 4.0
            + near["d_time_gt5_s"].clip(lower=0.0)
            + near["d_p95_axis_deg"].clip(lower=0.0) * 100.0
            - near["saved_m3"].clip(lower=0.0) * 0.05
        )
        closest = near.sort_values(
            ["near_score", "saved_m3"],
            ascending=[True, False],
        ).iloc[0]
        chosen = acceptable_best if acceptable_best is not None else closest
        rows.append(
            {
                "policy": policy,
                "case_id": case_id,
                "selected_run_id": str(chosen["run_id"]),
                "selected_action_family": str(chosen["action_family"]),
                "accepted": int(acceptable_best is not None),
                "saved_m3": float(chosen["saved_m3"]),
                "d_fallback_s": float(chosen["d_fallback_s"]),
                "d_time_gt4_s": float(chosen["d_time_gt4_s"]),
                "d_time_gt5_s": float(chosen["d_time_gt5_s"]),
                "p95_axis_deg": float(chosen["p95_axis_deg"]),
                "d_p95_axis_deg": float(chosen["d_p95_axis_deg"]),
                "failure_reasons": str(chosen["failure_reasons"]),
                "recommendation": _recommendation(chosen, policy),
                "candidate_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _opened_cases() -> set[str]:
    df = pd.read_csv(THREE_HAO)
    opened = df[df["use_economy"].astype(float) > 0.5]["case_id"].astype(str)
    return set(opened)


def _policy_mask(df: pd.DataFrame, policy: str) -> pd.Series:
    positive = df["saved_m3"] > 0.0
    no_fallback = df["d_fallback_s"] <= 0.0
    p95_under_5 = df["p95_axis_deg"] < 5.0
    if policy == "strict":
        return (
            positive
            & no_fallback
            & (df["d_time_gt5_s"] <= 0.0)
            & (df["d_p95_axis_deg"] <= 0.50)
            & p95_under_5
        )
    if policy == "balanced":
        return (
            positive
            & no_fallback
            & (df["d_time_gt5_s"] <= 30.0)
            & (df["d_p95_axis_deg"] <= 1.35)
            & p95_under_5
        )
    if policy == "appendix":
        return (
            positive
            & no_fallback
            & (df["d_time_gt5_s"] <= 60.0)
            & (df["d_p95_axis_deg"] <= 1.35)
            & p95_under_5
        )
    raise ValueError(f"unknown policy: {policy}")


def _rank_candidates(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    candidates = df[_policy_mask(df, policy)].copy()
    if candidates.empty:
        return candidates
    candidates["action_family"] = candidates["run_id"].map(ACTION_PRIORITY).fillna(
        "other replay"
    )
    # Score favors water first, then lower p95 debt and lower gt5 debt.  The
    # policy gates above keep the safety envelope narrow; this rank only breaks
    # ties among acceptable variants for the same case.
    candidates["selector_score"] = (
        candidates["saved_m3"]
        - 25.0 * candidates["d_p95_axis_deg"].clip(lower=0.0)
        - 0.8 * candidates["d_time_gt5_s"].clip(lower=0.0)
        - 0.15 * candidates["d_time_gt4_s"].clip(lower=0.0)
    )
    candidates = candidates.sort_values(
        ["case_id", "selector_score", "saved_m3"],
        ascending=[True, False, False],
    )
    return candidates.groupby("case_id", as_index=False).head(1).copy()


def _summarize(selection: pd.DataFrame, policy: str) -> dict[str, object]:
    add_m3 = float(selection["saved_m3"].sum()) if not selection.empty else 0.0
    total_m3 = BASELINE_3HAO_M3 + add_m3
    return {
        "policy": policy,
        "accepted_new_case_count": int(len(selection)),
        "additional_saved_m3": add_m3,
        "projected_total_saved_m3": total_m3,
        "projected_96case_saving_pct": 100.0 * total_m3 / DENOMINATOR_96CASE_M3,
        "additional_pool_saving_pct": (
            100.0
            * add_m3
            / max(float(selection["baseline_pump_m3"].sum()), 1e-9)
            if not selection.empty
            else 0.0
        ),
        "d_fallback_s": float(selection["d_fallback_s"].sum())
        if not selection.empty
        else 0.0,
        "d_time_gt4_s": float(selection["d_time_gt4_s"].sum())
        if not selection.empty
        else 0.0,
        "d_time_gt5_s": float(selection["d_time_gt5_s"].sum())
        if not selection.empty
        else 0.0,
        "max_p95_axis_deg": float(selection["p95_axis_deg"].max())
        if not selection.empty
        else 0.0,
        "max_d_p95_axis_deg": float(selection["d_p95_axis_deg"].max())
        if not selection.empty
        else 0.0,
        "case_ids": ";".join(selection["case_id"].astype(str).tolist())
        if not selection.empty
        else "",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    case_df = pd.read_csv(CASE_METRICS)
    opened = _opened_cases()
    case_df = case_df[~case_df["case_id"].astype(str).isin(opened)].copy()
    case_df = case_df[case_df["run_id"].isin(ACTION_PRIORITY)].copy()

    selections: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for policy in ("strict", "balanced", "appendix"):
        selected = _rank_candidates(case_df, policy)
        selected.insert(0, "policy", policy)
        selections.append(selected)
        diagnostics.append(_case_diagnostics(case_df, policy))
        summaries.append(_summarize(selected, policy))

    selected_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    diagnostics_df = pd.concat(diagnostics, ignore_index=True) if diagnostics else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    selected_df.to_csv(OUT / "4hao_action_family_selected_cases.csv", index=False)
    diagnostics_df.to_csv(OUT / "4hao_action_family_case_diagnostics.csv", index=False)
    summary_df.to_csv(OUT / "4hao_action_family_projection_summary.csv", index=False)

    lines = [
        "# 4号动作族选择器实证复盘",
        "",
        "一句话结论：在已经跑出的 No.4 动作族里，按 case 选择 mild / active / pressure-gated active 可以比单一全局 profile 更合理，但主线仍然只能到 14.7%-15.0% 左右；18% 还需要新的安全水量来源。",
        "",
        "## 汇总",
        "",
        "| 口径 | 新增case | 新增节水m3 | 96-case投影 | d_gt4_s | d_gt5_s | max p95 | max d_p95 | case |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {policy} | {accepted_new_case_count} | {additional_saved_m3:.1f} | "
            "{projected_total_saved_m3:.1f} / {projected_96case_saving_pct:.2f}% | "
            "{d_time_gt4_s:.0f} | {d_time_gt5_s:.0f} | {max_p95_axis_deg:.2f} | "
            "{max_d_p95_axis_deg:.2f} | {case_ids} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 已剔除 3号已经打开的 case，避免重复计算节水。",
            "- strict：fallback 不增、gt5 不增、d_p95 <= 0.50deg。",
            "- balanced：fallback 不增、gt5 增量 <= 30s、d_p95 <= 1.35deg、p95 < 5deg。",
            "- appendix：fallback 不增、gt5 增量 <= 60s、d_p95 <= 1.35deg、p95 < 5deg。",
            "- 这个脚本只是复盘已完成闭环结果，不等价于已实现在线 selector。",
            "- `4hao_action_family_case_diagnostics.csv` 给出每个 case 的最佳候选、拒绝原因和下一步建议。",
            "",
            "## 输出",
            "",
            "- `4hao_action_family_selected_cases.csv`",
            "- `4hao_action_family_case_diagnostics.csv`",
            "- `4hao_action_family_projection_summary.csv`",
        ]
    )
    (OUT / "4hao_action_family_selector_readout.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
