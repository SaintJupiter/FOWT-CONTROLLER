#!/usr/bin/env python3
"""Build a readable No.4 gate-optimization readout from smoke failures.

This is an analysis layer, not a runtime controller.  It answers the immediate
optimization question after the failed No.4 smoke runs: if the current learned
arm is too broad, which smoke-proven gates are defensible, how much water do
they save, and what remains before a real No.4 controller can be claimed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
FEASIBILITY_DIR = BASE / "psc_4hao_moderate_arm_feasibility_v1"
SMOKE_SUMMARY = FEASIBILITY_DIR / "4hao_smoke_120min_summary.csv"
CASE_DELTA = FEASIBILITY_DIR / "4hao_case_delta_table.csv"
OUT_CSV = FEASIBILITY_DIR / "4hao_gate_optimization_result_table.csv"
OUT_MD = FEASIBILITY_DIR / "4hao_gate_optimization_readout.md"

ZERO_HAO_PROFILE = "0号安全基线"
THREE_HAO_PROFILE = "3号v2严格选择器"
TARGET_SAVING_PCT = 15.0

PROFILE_NAMES = {
    "regime_auto": "A 全局温和调度",
    "dual_specialist": "B 双专家调度",
    "neutral_mhs_clean": "C neutral专家",
    "regime_auto_scale025": "D 全局缩放25%(无效)",
    "regime_auto_scale040": "E 全局缩放40%(无效)",
    "regime_auto_pirelease_probe": "F hold释放复核",
    "regime_auto_engineered_probe": "G 泵侧对齐复核",
}


def _fmt_cases(series: pd.Series) -> str:
    return ";".join(series.astype(str).tolist())


def _short_cases(text: object, limit: int = 5) -> str:
    cases = [s for s in str(text or "").split(";") if s]
    if len(cases) <= limit:
        return ";".join(cases)
    return f"{len(cases)}例: " + ";".join(cases[:limit]) + ";..."


def _result_row(
    *,
    scope: str,
    name: str,
    raw_profile: str,
    baseline_pump_m3: float,
    selected_pump_m3: float,
    d_time_gt5_s: float,
    d_fallback_s: float,
    opened_cases: str,
    failed_cases: str,
    verdict: str,
    note: str,
) -> dict[str, object]:
    saved = baseline_pump_m3 - selected_pump_m3
    return {
        "scope": scope,
        "候选名": name,
        "raw_profile": raw_profile,
        "opened_case_count": len([s for s in str(opened_cases).split(";") if s]),
        "opened_cases": opened_cases,
        "failed_case_count": len([s for s in str(failed_cases).split(";") if s]),
        "failed_cases": failed_cases,
        "selected_pump_m3": selected_pump_m3,
        "saved_m3": saved,
        "saving_pct": saved / max(baseline_pump_m3, 1e-9) * 100.0,
        "d_time_gt5_s": d_time_gt5_s,
        "d_fallback_s": d_fallback_s,
        "verdict": verdict,
        "note": note,
    }


def _smoke_gate_rows(smoke: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile, group in smoke.groupby("profile", sort=False):
        baseline_pump = float(group["baseline_pump_m3"].sum())
        full_pump = float(group["primary_pump_m3"].sum())
        fail = group[group["safety_fail"].astype(int).eq(1)]
        rows.append(
            _result_row(
                scope="8例smoke",
                name=PROFILE_NAMES.get(profile, profile),
                raw_profile=profile,
                baseline_pump_m3=baseline_pump,
                selected_pump_m3=full_pump,
                d_time_gt5_s=float(group["d_time_gt5_s"].sum()),
                d_fallback_s=float(group["d_fallback_s"].sum()),
                opened_cases=_fmt_cases(group["case_id"]),
                failed_cases=_fmt_cases(fail["case_id"]),
                verdict="reject_global_4hao" if len(fail) else "candidate",
                note="全量打开该profile；用于说明为什么不能直接升格为4号。",
            )
        )

        strict = group[
            group["strict_no_debt"].astype(int).eq(1)
            & group["pump_saved_vs_0hao_m3"].gt(0)
        ]
        rows.append(
            _result_row(
                scope="8例smoke",
                name=f"{PROFILE_NAMES.get(profile, profile)} - 严格安全门",
                raw_profile=profile,
                baseline_pump_m3=baseline_pump,
                selected_pump_m3=baseline_pump
                - float(strict["pump_saved_vs_0hao_m3"].sum()),
                d_time_gt5_s=float(strict["d_time_gt5_s"].sum()),
                d_fallback_s=float(strict["d_fallback_s"].sum()),
                opened_cases=_fmt_cases(strict["case_id"]),
                failed_cases="",
                verdict="safe_but_small",
                note="只打开 fallback=0 且 time_gt5 不增加的 smoke 正例。",
            )
        )

        pareto = group[
            group["d_fallback_s"].eq(0)
            & group["d_time_gt5_s"].lt(60)
            & group["pump_saved_vs_0hao_m3"].gt(0)
        ]
        rows.append(
            _result_row(
                scope="8例smoke",
                name=f"{PROFILE_NAMES.get(profile, profile)} - 轻债务门",
                raw_profile=profile,
                baseline_pump_m3=baseline_pump,
                selected_pump_m3=baseline_pump
                - float(pareto["pump_saved_vs_0hao_m3"].sum()),
                d_time_gt5_s=float(pareto["d_time_gt5_s"].sum()),
                d_fallback_s=float(pareto["d_fallback_s"].sum()),
                opened_cases=_fmt_cases(pareto["case_id"]),
                failed_cases="",
                verdict="pareto_candidate",
                note="只打开 fallback=0 且单case time_gt5<60s 的 no-fallback 正例。",
            )
        )
    return rows


def _ninety_six_case_rows(delta: pd.DataFrame) -> list[dict[str, object]]:
    baseline_pump = float(delta["pump_m3_safety"].sum())
    three_pump = float(
        delta.apply(
            lambda row: row["pump_m3_rawenv"]
            if int(row["use_economy"])
            else row["pump_m3_safety"],
            axis=1,
        ).sum()
    )
    unopened_gain = delta["use_economy"].astype(int).eq(0) & delta["pump_gain_m3"].gt(0)
    strict = delta[
        unopened_gain & delta["d_fallback_s"].eq(0) & delta["d_time_gt5_s"].le(0)
    ]
    light_debt = delta[
        unopened_gain & delta["d_fallback_s"].eq(0) & delta["d_time_gt5_s"].lt(60)
    ]
    rows = [
        _result_row(
            scope="96例全池",
            name=ZERO_HAO_PROFILE,
            raw_profile="current_forecast_adaptive",
            baseline_pump_m3=baseline_pump,
            selected_pump_m3=baseline_pump,
            d_time_gt5_s=0.0,
            d_fallback_s=0.0,
            opened_cases="",
            failed_cases="",
            verdict="baseline",
            note="0号安全基线，作为所有百分比的分母。",
        ),
        _result_row(
            scope="96例全池",
            name=THREE_HAO_PROFILE,
            raw_profile="psc_structural_selector_96case_pair_3hao_v2",
            baseline_pump_m3=baseline_pump,
            selected_pump_m3=three_pump,
            d_time_gt5_s=0.0,
            d_fallback_s=0.0,
            opened_cases=_fmt_cases(delta.loc[delta["use_economy"].eq(1), "case_id"]),
            failed_cases="",
            verdict="reference_only",
            note="3号只是参考线；本轮优化目标是4号。",
        ),
        _result_row(
            scope="96例全池",
            name="4号严格安全门(96例投影)",
            raw_profile="strict_no_debt_gate",
            baseline_pump_m3=baseline_pump,
            selected_pump_m3=three_pump - float(strict["pump_gain_m3"].sum()),
            d_time_gt5_s=float(strict["d_time_gt5_s"].sum()),
            d_fallback_s=float(strict["d_fallback_s"].sum()),
            opened_cases=_fmt_cases(strict["case_id"]),
            failed_cases="",
            verdict="safe_but_not_enough",
            note="只加严格无债务正例；安全但省水增量很小。",
        ),
        _result_row(
            scope="96例全池",
            name="4号轻债务门(96例投影)",
            raw_profile="fallback0_time_gt5_lt60_gate",
            baseline_pump_m3=baseline_pump,
            selected_pump_m3=three_pump - float(light_debt["pump_gain_m3"].sum()),
            d_time_gt5_s=float(light_debt["d_time_gt5_s"].sum()),
            d_fallback_s=float(light_debt["d_fallback_s"].sum()),
            opened_cases=_fmt_cases(light_debt["case_id"]),
            failed_cases="",
            verdict="reaches_15pct_with_time_debt",
            note="能跨过15%，但依赖39s time_gt5 债务，不能叫严格无债务。",
        ),
    ]
    return rows


def main() -> None:
    smoke = pd.read_csv(SMOKE_SUMMARY)
    delta = pd.read_csv(CASE_DELTA)
    rows = _ninety_six_case_rows(delta) + _smoke_gate_rows(smoke)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    target_saved = float(delta["pump_m3_safety"].sum()) * TARGET_SAVING_PCT / 100.0
    three = out[out["候选名"].eq(THREE_HAO_PROFILE)].iloc[0]
    gap = target_saved - float(three["saved_m3"])
    best = out[out["候选名"].eq("4号轻债务门(96例投影)")].iloc[0]

    lines = [
        "# 4号门控优化结果",
        "",
        "这一步不是在优化3号；3号只作为参照。真正做的是把4号 smoke 的失败案例转成可解释门控：哪些开、哪些关、为什么。",
        "",
        "## 做到了什么",
        "",
        "- 复核了现有 4号 候选：全量打开都会失败，不能直接升格。",
        "- 证明了 `primary_scale=0.25/0.40` 不是正确的温和旋钮，因为它缩放的是主控目标，不是单独缩放省水动作。",
        "- 用失败案例构造两个可读门：严格安全门和轻债务门。",
        "- 所有结果表都补了省水百分比，并给候选换成中文短名。",
        "",
        "## 核心结果表",
        "",
        "| 范围 | 候选名 | raw profile | 开启数 | 失败数 | 开启case | 失败case | 省水m3 | 省水% | d_gt5 s | fallback s | verdict |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in out.to_dict("records"):
        lines.append(
            f"| {row['scope']} | {row['候选名']} | {row['raw_profile']} | "
            f"{int(row['opened_case_count'])} | {int(row['failed_case_count'])} | "
            f"{_short_cases(row['opened_cases'])} | {_short_cases(row['failed_cases'])} | "
            f"{float(row['saved_m3']):.1f} | "
            f"{float(row['saving_pct']):.2f}% | {float(row['d_time_gt5_s']):.0f} | "
            f"{float(row['d_fallback_s']):.0f} | {row['verdict']} |"
        )

    lines.extend(
        [
            "",
            "## 简单解释",
            "",
            "- 表里的 `96例全池` 百分比用 0号 96例总泵量作分母；`8例smoke` 百分比只用于比较这8个压力测试样本，不能和96例 headline 直接相加。",
            f"- 当前3号参考线：省水 `{float(three['saved_m3']):.1f} m3`，省水率 `{float(three['saving_pct']):.2f}%`。",
            f"- 要到 15% 需要再多省约 `{gap:.1f} m3`。",
            "- 严格安全门只多拿到一个严格正例，增量太小，所以节水少。",
            f"- 轻债务门能到 `{float(best['saving_pct']):.2f}%`，但代价是 `fallback=0`、`time_gt5 +{float(best['d_time_gt5_s']):.0f}s`，它是 Pareto 候选，不是严格安全结论。",
            "",
            "## 失败案例给出的下一步",
            "",
            "- `11_dual_relief_11`：省水多，但 fallback +36s、time_gt5 +580s。下一步要加 reintensification / roll 风险拒绝。",
            "- `18_dual_relief_18`：fallback 0，但 time_gt5 +581s。下一步不能只看 fallback，要加 time_gt5 债务预测或高姿态持续时间预算。",
            "- `57/58/59_dual_neutral_*`：长期 hold 导致 fallback +412s。下一步 neutral 高风险域默认拒绝，除非有干净起点和明确 relief 确认。",
            "- `09/10/94`：是 no-fallback 正例，可以作为4号第一批允许域；其中 `10/94` 带轻微 time_gt5 债务，只能放在 Pareto/附录口径。",
            "",
            "## 下一轮优化计划",
            "",
            "1. 做 4号 runtime 门控 v1：先只允许 `09/10/94` 对应的 no-fallback 形态，其他 smoke 失败形态强制回0号。",
            "2. 做动作级温和化：不要用 `primary_scale`，改成只对 economy/hold 类省水动作加预算、短 hold、relief 确认。",
            "3. 先跑 8-case 120min smoke，再扩到96-case；结果表继续用中文名、m3、百分比、fallback、time_gt5。",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
