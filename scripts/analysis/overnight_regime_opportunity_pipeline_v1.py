#!/usr/bin/env python3
"""Build the overnight regime-opportunity package.

The script consolidates the current source of truth, the read-only regime
mining pass, and existing specialist validation summaries into a morning-facing
decision package.  It does not tune controller thresholds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
MINING = ROOT / "regime_mining_v3"
OUT = ROOT / "overnight_regime_opportunity_v1"
RAW = OUT / "raw_tables"
PAPER = OUT / "paper_ready"
CASEBOOKS = OUT / "casebooks"
REGISTRY = ROOT / "paper_ready" / "regime_specialist_registry_final.json"
NEUTRAL_PROFILE = ROOT / "neutral_untyped_profile_v1" / "raw_tables" / "neutral_untyped_profile_rows.csv"


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}%"


def num(x: float | int | None, digits: int = 2) -> str:
    if x is None or x == "" or pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def source_manifest() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    neutral_overnight = read_csv(RAW / "neutral_mhs_overnight_extended_summary.csv")
    neutral_evidence = "31.48% clean-start / 11.97% broader"
    if not neutral_overnight.empty:
        neutral_summary = dict(zip(neutral_overnight["metric"], neutral_overnight["value"], strict=False))
        neutral_evidence += (
            f" / {neutral_summary.get('pump_saving_pct', '')} fixed-rule 96-case overnight"
        )
    allowed = [
        ("relief_decay_episode_auto_v1", "valid matched automatic relief/decay specialist", "18.13%"),
        ("neutral_mhs_clean_budget_v1", "second automatic candidate, clean-start gated", neutral_evidence),
        ("budget100", "operator/Pareto reference inside fixed opportunity domains", "17-23% regime-conditioned"),
        ("far_event_advisory", "h120 supervisory/advisory layer", "not direct pump control"),
    ]
    forbidden = [
        ("relief_decay_smart_release_v1 = 21.49%", "old duration-mismatched aggregate; do not cite"),
        ("dual_specialist_pump_saving_v1 as final policy", "boundary/catch-up harm blocks promotion"),
        ("global 20% saving", "not supported; savings are regime-conditioned"),
        ("120min direct closed-loop pump saving", "h120 role is supervisory here"),
    ]
    lines = [
        "# Overnight Run Manifest",
        "",
        "This manifest fixes the allowed sources for the overnight regime opportunity pass.",
        "",
        "## Allowed Evidence",
        "",
        markdown_table(["Item", "Role", "Current Evidence"], allowed),
        "",
        "## Forbidden / Deprecated Evidence",
        "",
        markdown_table(["Item", "Reason"], forbidden),
        "",
        "## Registry Snapshot",
        "",
        f"- registry_version: `{registry.get('version', 'missing')}`",
        "- hard safety invariant: v1.6 hard floor / fallback / recovery / pump penalty unchanged",
    ]
    (PAPER / "overnight_run_manifest.md").write_text("\n".join(lines), encoding="utf-8")


def prevalence_summary() -> pd.DataFrame:
    prev = read_csv(MINING / "raw_tables" / "regime_mining_v3_prevalence_all_splits.csv")
    if prev.empty:
        return prev
    total_rows = prev["rows"].sum()
    grouped = (
        prev.groupby("primary_regime", as_index=False)
        .agg(
            rows=("rows", "sum"),
            splits=("split", lambda s: ",".join(sorted(set(map(str, s))))),
            row_pct_mean=("row_pct", "mean"),
            episodes=("episode_count", "sum"),
            median_episode_minutes=("median_episode_minutes", "median"),
            max_episode_minutes=("max_episode_minutes", "max"),
            future_speed_max_median_ms=("future_speed_max_median_ms", "median"),
            peak_to_late_drop_median_ms=("peak_to_late_drop_median_ms", "median"),
            dir_shift_median_deg=("dir_shift_median_deg", "median"),
        )
        .sort_values("rows", ascending=False)
    )
    grouped["row_pct_scanned"] = 100.0 * grouped["rows"] / total_rows
    return grouped


def current_evidence() -> dict[str, dict[str, object]]:
    soft_smoke = read_csv(RAW / "transient_peak_soft_decay_smoke_summary.csv")
    soft_decay_status = "near-family candidate"
    soft_decay_evidence = "v2 table: n=2, high saving; needs smoke only"
    soft_decay_decision = "merge into transient-decay family if smoke is non-harmful"
    soft_decay_next = "short 1h smoke only"
    if not soft_smoke.empty:
        s = dict(zip(soft_smoke["metric"], soft_smoke["value"], strict=False))
        saving_text = str(s.get("pump_saving_pct", ""))
        try:
            saving_value = float(saving_text.rstrip("%"))
        except ValueError:
            saving_value = 0.0
        harmed_cases = int(float(s.get("harmed_cases", 0)))
        if saving_value < 5.0 or harmed_cases > 0:
            soft_decay_status = "tested_no_go_to_expand"
            soft_decay_evidence = (
                f"overnight smoke: {saving_text} pump saving, "
                f"{s.get('active_saved_cases', '')} saved cases, {harmed_cases} harmed cases"
            )
            soft_decay_decision = "do not expand; keep as diagnostic inside the decay family"
            soft_decay_next = "stop testing this branch"

    evidence: dict[str, dict[str, object]] = {
        "transient_peak_fast_decay": {
            "best_profile": "relief_decay_episode_auto_v1",
            "status": "mature automatic",
            "closed_loop_evidence": "18.13% matched automatic; 25.18% mixed subset",
            "risk": "time>5/fallback costs must be reported",
            "decision": "keep as main automatic specialist",
            "next_step": "package / cite; no more tuning",
        },
        "transient_peak_soft_decay": {
            "best_profile": "relief_decay_episode_auto_v1 family",
            "status": soft_decay_status,
            "closed_loop_evidence": soft_decay_evidence,
            "risk": "too few cases if left alone",
            "decision": soft_decay_decision,
            "next_step": soft_decay_next,
        },
        "residual_high_plateau": {
            "best_profile": "budget100 Pareto / no automatic promotion",
            "status": "operator/Pareto candidate",
            "closed_loop_evidence": "budget100 high; automatic recognition/separability failed",
            "risk": "plateau-good overlaps boundary-bad",
            "decision": "freeze; do not tune loose/strict again",
            "next_step": "paper boundary discussion",
        },
        "lowrisk_stable_redundant_candidate": {
            "best_profile": "none",
            "status": "background / sparse opportunity",
            "closed_loop_evidence": "high prevalence but low addressable pump; sparse high-saving cases",
            "risk": "top-case dominated",
            "decision": "do not build automatic specialist now",
            "next_step": "log as no-action/background",
        },
        "gusty_oscillation_candidate": {
            "best_profile": "gusty_oscillation_smoothing_v1",
            "status": "tested no-go for automatic",
            "closed_loop_evidence": "blind 17.68% with safety cost; smoothing -0.0004%",
            "risk": "blind saving is comfort trade, forecast smoothing did not capture it",
            "decision": "freeze after one-shot test",
            "next_step": "do not sweep thresholds",
        },
        "direction_reversal_boundary": {
            "best_profile": "none",
            "status": "veto / abstain",
            "closed_loop_evidence": "dual specialist boundary harm -8.15%; strict fix -10.69%",
            "risk": "catch-up/fallback",
            "decision": "must abstain to v1.6",
            "next_step": "use as negative control",
        },
        "reintensification_boundary": {
            "best_profile": "none",
            "status": "veto / abstain",
            "closed_loop_evidence": "boundary mechanism: apparent relief followed by second event",
            "risk": "catch-up/fallback",
            "decision": "must abstain to v1.6",
            "next_step": "use as negative control",
        },
        "far_only_h120_advisory": {
            "best_profile": "far_event_advisory",
            "status": "supervisory",
            "closed_loop_evidence": "h120 useful as advisory, not direct pump control",
            "risk": "do not claim closed-loop saving",
            "decision": "retain output/logging",
            "next_step": "paper architecture layer",
        },
        "quiet_low_opportunity": {
            "best_profile": "none",
            "status": "no action",
            "closed_loop_evidence": "little pump to save",
            "risk": "inflates prevalence but not impact",
            "decision": "ignore for pump-saving headline",
            "next_step": "background only",
        },
        "neutral_untyped": {
            "best_profile": "none",
            "status": "untyped pool",
            "closed_loop_evidence": "needs pump-opportunity filter before action",
            "risk": "too broad",
            "decision": "mine subtypes only if new mechanism appears",
            "next_step": "do not run blanket tests",
        },
        "residual_high_slow_decay": {
            "best_profile": "none",
            "status": "boundary-like candidate",
            "closed_loop_evidence": "v2 table shows pump saving with safety cost/concentration",
            "risk": "slow decay can become catch-up",
            "decision": "do not promote",
            "next_step": "boundary discussion",
        },
        "sustained_high_safety_event": {
            "best_profile": "v1.6 hard floor",
            "status": "safety regime",
            "closed_loop_evidence": "real load, not economy opportunity",
            "risk": "safety ownership",
            "decision": "no economy action",
            "next_step": "hard-floor layer",
        },
        "transient_decay_low_pump_proxy": {
            "best_profile": "none",
            "status": "low value",
            "closed_loop_evidence": "v2 table: low pump opportunity can become negative",
            "risk": "no material pump to remove",
            "decision": "drop",
            "next_step": "no validation",
        },
    }

    neutral = read_csv(
        ROOT
        / "neutral_moderate_high_steady_broader_v1"
        / "raw_tables"
        / "neutral_mhs_broader_clean_auto_learned_summary.csv"
    )
    if not neutral.empty:
        clean = neutral[neutral["group"] == "clean_32"]
        broader = neutral[neutral["group"] == "all_64_broader"]
        if not clean.empty and not broader.empty:
            evidence["neutral_moderate_high_steady_clean_start"] = {
                "best_profile": "neutral_mhs_clean_budget_v1",
                "status": "second automatic candidate",
                "closed_loop_evidence": (
                    f"{clean.iloc[0]['pump_saving_pct']:.2f}% clean-start; "
                    f"{broader.iloc[0]['pump_saving_pct']:.2f}% broader"
                ),
                "risk": "requires current-state clean-start gate",
                "decision": "keep as second specialist; verify if more clean-start cases are mined",
                "next_step": "package / optional broader confirmation",
            }
    neutral_overnight = read_csv(RAW / "neutral_mhs_overnight_extended_summary.csv")
    if not neutral_overnight.empty:
        summary = dict(zip(neutral_overnight["metric"], neutral_overnight["value"], strict=False))
        prior = evidence.get("neutral_moderate_high_steady_clean_start", {})
        prior_text = str(prior.get("closed_loop_evidence", ""))
        overnight_text = (
            f"overnight fixed-rule 96-case: {summary.get('pump_saving_pct', '')} saving, "
            f"{summary.get('saved_cases', '')}/{summary.get('cases', '')} saved cases, "
            f"{summary.get('harmed_cases', '')} harmed cases, "
            f"top-case share {summary.get('top_case_share', '')}, "
            f"max fallback delta {summary.get('max_d_fallback_ratio', '')}, "
            f"max p95-axis delta {summary.get('max_d_p95_max_axis', '')}"
        )
        evidence["neutral_moderate_high_steady_clean_start"] = {
            "best_profile": "neutral_mhs_clean_budget_v1",
            "status": "second automatic candidate, broadened",
            "closed_loop_evidence": (
                f"{prior_text}; {overnight_text}" if prior_text else overnight_text
            ),
            "risk": "requires current-state clean-start gate; broader pool dilutes saving",
            "decision": "keep as second specialist candidate; 96-case run supports non-single-case behavior",
            "next_step": "package as secondary result; do not tune budget tonight",
        }
    return evidence


def build_candidate_library() -> pd.DataFrame:
    prev = prevalence_summary()
    evidence = current_evidence()
    rows: list[dict[str, object]] = []
    for _, r in prev.iterrows():
        regime = str(r["primary_regime"])
        ev = evidence.get(
            regime,
            {
                "best_profile": "none",
                "status": "unmapped",
                "closed_loop_evidence": "no current closed-loop evidence",
                "risk": "unknown",
                "decision": "do not run until mechanism is clear",
                "next_step": "read-only review only",
            },
        )
        high_prevalence = float(r["row_pct_scanned"]) >= 1.0
        likely_actionable = ev["status"] in {
            "mature automatic",
            "second automatic candidate",
            "near-family candidate",
        }
        boundary = "boundary" in regime or "reintensification" in regime
        if regime == "transient_peak_soft_decay" and str(ev["status"]).startswith("tested_no_go"):
            priority = "freeze_or_background"
        elif regime == "transient_peak_soft_decay":
            priority = "validate_short"
        elif likely_actionable:
            priority = "keep"
        elif boundary:
            priority = "veto"
        elif high_prevalence:
            priority = "freeze_or_background"
        else:
            priority = "drop_low_value"
        rows.append(
            {
                "primary_regime": regime,
                "rows": int(r["rows"]),
                "row_pct_scanned": float(r["row_pct_scanned"]),
                "episode_count": int(r["episodes"]),
                "median_episode_minutes": float(r["median_episode_minutes"]),
                "max_episode_minutes": float(r["max_episode_minutes"]),
                "future_speed_max_median_ms": float(r["future_speed_max_median_ms"]),
                "peak_to_late_drop_median_ms": float(r["peak_to_late_drop_median_ms"]),
                "dir_shift_median_deg": float(r["dir_shift_median_deg"]),
                "best_profile": ev["best_profile"],
                "status": ev["status"],
                "closed_loop_evidence": ev["closed_loop_evidence"],
                "risk": ev["risk"],
                "decision": ev["decision"],
                "next_step": ev["next_step"],
                "candidate_priority": priority,
            }
        )
    # Add the state-gated neutral-MHS specialist because it is not a pure wind
    # shape regime in the mining table.
    ev = evidence.get("neutral_moderate_high_steady_clean_start")
    if ev:
        rows.append(
            {
                "primary_regime": "neutral_moderate_high_steady_clean_start",
                "rows": "",
                "row_pct_scanned": "",
                "episode_count": "",
                "median_episode_minutes": "",
                "max_episode_minutes": "",
                "future_speed_max_median_ms": "",
                "peak_to_late_drop_median_ms": "",
                "dir_shift_median_deg": "",
                "best_profile": ev["best_profile"],
                "status": ev["status"],
                "closed_loop_evidence": ev["closed_loop_evidence"],
                "risk": ev["risk"],
                "decision": ev["decision"],
                "next_step": ev["next_step"],
                "candidate_priority": "keep",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "regime_candidate_library_overnight.csv", index=False)
    return df


def write_action_matrix(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    action_map = {
        "transient_peak_fast_decay": (
            "delay/reduce chasing transient peak",
            "relief_decay_episode_auto_v1",
            "automatic specialist",
        ),
        "transient_peak_soft_decay": (
            "same as transient-decay, but require short smoke before merging",
            "relief_decay_episode_auto_v1 family",
            "short smoke candidate",
        ),
        "neutral_moderate_high_steady_clean_start": (
            "reduce economy tracking only from very clean posture",
            "neutral_mhs_clean_budget_v1",
            "second automatic specialist candidate",
        ),
        "residual_high_plateau": (
            "operator/Pareto budget only",
            "budget100 reference",
            "not automatic; recognition failed",
        ),
        "gusty_oscillation_candidate": (
            "one-shot smoothing already tested",
            "gusty_oscillation_smoothing_v1",
            "no-go for automatic",
        ),
        "lowrisk_stable_redundant_candidate": (
            "none now; too little addressable pump",
            "none",
            "background",
        ),
        "direction_reversal_boundary": ("release/abstain", "v1.6", "veto"),
        "reintensification_boundary": ("release/abstain", "v1.6", "veto"),
        "far_only_h120_advisory": ("supervisory warning", "far_event_advisory", "advisory"),
    }
    for _, r in candidates.iterrows():
        regime = str(r["primary_regime"])
        if regime == "transient_peak_soft_decay" and str(r["status"]).startswith("tested_no_go"):
            action, profile, role = (
                "no expansion after smoke; keep as diagnostic",
                "none",
                "tested no-go",
            )
        else:
            action, profile, role = action_map.get(
                regime,
                ("no dedicated action", str(r["best_profile"]), str(r["status"])),
            )
        rows.append(
            {
                "primary_regime": regime,
                "recommended_action": action,
                "profile_or_layer": profile,
                "role": role,
                "closed_loop_evidence": r["closed_loop_evidence"],
                "risk": r["risk"],
                "decision": r["decision"],
                "next_step": r["next_step"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "overnight_regime_action_matrix.csv", index=False)
    return df


def write_markdown_outputs(candidates: pd.DataFrame, matrix: pd.DataFrame) -> None:
    top = candidates.sort_values(
        by=["candidate_priority", "row_pct_scanned"],
        ascending=[True, False],
        na_position="last",
    )
    rows = []
    sortable = candidates.copy()
    sortable["_rows_sort"] = pd.to_numeric(sortable["rows"], errors="coerce")
    for _, r in sortable.sort_values("_rows_sort", ascending=False, na_position="last").iterrows():
        rows.append(
            [
                r["primary_regime"],
                r["rows"],
                num(r["row_pct_scanned"]),
                r["status"],
                r["closed_loop_evidence"],
                r["decision"],
            ]
        )
    lines = [
        "# Overnight Regime Candidate Prevalence Summary",
        "",
        "This table combines the refreshed read-only mining pass with current closed-loop evidence. It is not a controller tuning result.",
        "",
        markdown_table(
            ["Regime", "Rows", "Approx. Row %", "Status", "Closed-loop Evidence", "Decision"],
            rows,
        ),
    ]
    (PAPER / "regime_candidate_prevalence_summary.md").write_text("\n".join(lines), encoding="utf-8")

    matrix_rows = [
        [
            r["primary_regime"],
            r["recommended_action"],
            r["profile_or_layer"],
            r["role"],
            r["decision"],
        ]
        for _, r in matrix.iterrows()
    ]
    (PAPER / "regime_action_fit_matrix.md").write_text(
        "\n".join(
            [
                "# Regime Action Fit Matrix",
                "",
                markdown_table(
                    ["Regime", "Recommended Action", "Profile/Layer", "Role", "Decision"],
                    matrix_rows,
                ),
            ]
        ),
        encoding="utf-8",
    )

    best_rows = candidates[candidates["candidate_priority"].isin(["keep", "validate_short"])].copy()
    best = [
        "# Overnight Best Candidates",
        "",
        "## Keep / Promote",
        "",
    ]
    for _, r in best_rows.iterrows():
        best += [
            f"### {r['primary_regime']}",
            "",
            f"- status: {r['status']}",
            f"- evidence: {r['closed_loop_evidence']}",
            f"- decision: {r['decision']}",
            f"- next step: {r['next_step']}",
            "",
        ]
    best += [
        "## Freeze / No-Go Signals",
        "",
        "- residual plateau remains operator/Pareto because automatic recognition/separability failed.",
        "- gusty remains no-go for automatic because forecast smoothing did not capture the blind budget pool.",
        "- lowrisk is high-prevalence but low-impact/background, not a 20% route.",
        "- reversal/reintensification are boundary controls, not pump-saving opportunities.",
    ]
    (PAPER / "overnight_best_candidates.md").write_text("\n".join(best), encoding="utf-8")

    smoke = read_csv(RAW / "transient_peak_soft_decay_smoke_summary.csv")
    smoke_lines = [
        "## soft-decay 短测",
        "",
        "尚未完成或未找到 matched A0 对照。",
    ]
    if not smoke.empty:
        s = dict(zip(smoke["metric"], smoke["value"], strict=False))
        smoke_lines = [
            "## soft-decay 短测",
            "",
            f"- case 数：{s.get('cases', '')}",
            f"- 总节泵：{s.get('pump_saving_pct', '')}，泵量差：{s.get('pump_saved_m3', '')} m3",
            f"- 省泵 case：{s.get('active_saved_cases', '')}；变差 case：{s.get('harmed_cases', '')}",
            "- 判定：不扩展。它不是新的强机会域，应保留在回落家族的诊断里，不单独成熟。",
        ]

    neutral_lines = [
        "## neutral-MHS 扩展验证",
        "",
        "尚未完成或未找到 matched A0 对照。",
    ]
    neutral_overnight = read_csv(RAW / "neutral_mhs_overnight_extended_summary.csv")
    if not neutral_overnight.empty:
        n = dict(zip(neutral_overnight["metric"], neutral_overnight["value"], strict=False))
        neutral_lines = [
            "## neutral-MHS 扩展验证",
            "",
            f"- 固定规则样本数：{n.get('cases', '')}",
            f"- 总节泵：{n.get('pump_saving_pct', '')}，泵量差：{n.get('pump_saved_m3', '')} m3",
            f"- 省泵 case：{n.get('saved_cases', '')}；变差 case：{n.get('harmed_cases', '')}",
            f"- top-case share：{n.get('top_case_share', '')}",
            f"- 最大 fallback 增量：{n.get('max_d_fallback_ratio', '')}",
            f"- 最大 p95 姿态增量：{n.get('max_d_p95_max_axis', '')} deg",
            "- 判定：可保留为第二自动专家候选。它不是 20% 主数字，但证明不是单 case 偶然；更宽样本下收益被高风险相似窗口稀释到约 12%。",
        ]

    report = [
        "# 夜间工况优化总报告",
        "",
        "## 总结",
        "",
        "这轮夜间任务没有继续乱扫阈值，而是把工况挖掘、已有闭环证据、专家策略状态合成一张候选库。当前结论是：真正值得保留的是两个正向专家，其他高占比工况大多不是缺少调参，而是缺少安全可分离的自动边界。",
        "",
        "## 当前最有价值的结果",
        "",
        "1. **瞬时峰值后回落**：成熟自动主线，`relief_decay_episode_auto_v1`，matched 18.13%。原理是预测显示峰值会回落，所以不追马上会消失的临时负载。",
        "2. **中等偏高稳定 + 初始姿态很干净**：第二自动候选，`neutral_mhs_clean_budget_v1`，clean-start 31.48%，64-case broader 11.97%，今晚固定规则 96-case 为 12.14%。原理是姿态已经很干净，预测又稳定，不需要做过多 economy 精细跟踪。",
        "3. **方向反转 / 再增强 / catch-up**：必须 abstain，回到 v1.6。这里不是机会，是节泵边界。",
        "",
        "## 还有没有更多工况",
        "",
        "有更多风型，但不是都有控制价值。低风险工况占比很高，但本来泵就少；gusty 占比不低，但已测 smoothing 基本没有节泵；residual plateau 有 Pareto 机会，但自动识别和边界分离失败。",
        "",
        *smoke_lines,
        "",
        *neutral_lines,
        "",
        "## 明早决策",
        "",
        "- 如果只追论文可用性：停止再挖，写两个专家 + 边界负例 + h120 advisory。",
        "- transient soft-decay 已按短 smoke 测过；若其 summary 没有明显节泵，就不要再扩展。",
        "- 不建议再投入 plateau loose/strict、gusty 阈值、全局 dual dispatcher。",
    ]
    (PAPER / "overnight_regime_optimization_report_zh.md").write_text("\n".join(report), encoding="utf-8")


def build_soft_decay_smoke_casebook(max_cases: int = 12) -> Path:
    sources = [
        MINING / "raw_tables" / "casebooks_test" / "transient_peak_soft_decay_features.csv",
        MINING / "raw_tables" / "casebooks_validation" / "transient_peak_soft_decay_features.csv",
    ]
    frames = [read_csv(p) for p in sources if p.exists()]
    if not frames:
        raise FileNotFoundError("no transient_peak_soft_decay feature casebooks found")
    df = pd.concat(frames, ignore_index=True)
    df["score"] = (
        df["peak_to_late_drop_ms"].fillna(0.0)
        + 0.2 * df["near_max_ms"].fillna(0.0)
        - 0.03 * df["dir_shift_abs_max_deg"].fillna(0.0)
    )
    df = df.sort_values("score", ascending=False).head(max_cases)
    rows = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rows.append(
            {
                "case_id": f"soft_decay_smoke_{i:02d}",
                "timestamp": str(row["future_start"]),
                "label": (
                    "transient_peak_soft_decay_smoke"
                    f" | max={float(row['future_speed_max_ms']):.1f}"
                    f" | drop={float(row['peak_to_late_drop_ms']):.1f}"
                    f" | dir={float(row['dir_shift_abs_max_deg']):.0f}"
                ),
            }
        )
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    path = CASEBOOKS / "transient_peak_soft_decay_smoke_cases.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def neutral_mhs_score(df: pd.DataFrame) -> pd.Series:
    return (
        1.5 * df["early_max_ms"].fillna(0.0)
        + 0.5 * df["near_max_ms"].fillna(0.0)
        - 0.7 * df["near_range_ms"].fillna(0.0)
        - 0.7 * df["far_range_ms"].fillna(0.0)
        - 0.04 * df["dir_shift_abs_max_deg"].fillna(0.0)
    )


def spaced_top(df: pd.DataFrame, max_cases: int, min_spacing_s: int) -> pd.DataFrame:
    work = df.copy()
    work["selection_score"] = neutral_mhs_score(work)
    work = work.sort_values("selection_score", ascending=False)
    keep = []
    taken: list[pd.Timestamp] = []
    for _, row in work.iterrows():
        t = pd.to_datetime(row["future_start"])
        if all(abs((t - prev).total_seconds()) >= min_spacing_s for prev in taken):
            keep.append(row)
            taken.append(t)
        if len(keep) >= max_cases:
            break
    return pd.DataFrame(keep)


def build_neutral_mhs_overnight_casebook(max_per_split: int = 48) -> Path:
    if not NEUTRAL_PROFILE.exists():
        raise FileNotFoundError(NEUTRAL_PROFILE)
    df = pd.read_csv(NEUTRAL_PROFILE)
    pool = df[df["neutral_subtype"] == "neutral_moderate_high_steady"].copy()
    selected = pd.concat(
        [
            spaced_top(pool[pool["split"] == split], max_per_split, min_spacing_s=3 * 3600)
            for split in ("test", "validation")
        ],
        ignore_index=True,
    )
    selected.to_csv(RAW / "neutral_mhs_overnight_extended_selected_features.csv", index=False)
    rows = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        rows.append(
            {
                "case_id": f"neutral_mhs_overnight_{i:02d}",
                "timestamp": str(row["future_start"]),
                "label": (
                    "neutral_mhs_overnight_extended"
                    f" | split={row['split']}"
                    f" | early={float(row['early_max_ms']):.1f}"
                    f" | range={float(row['near_range_ms']):.1f}"
                    f" | dir={float(row['dir_shift_abs_max_deg']):.0f}"
                ),
            }
        )
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    path = CASEBOOKS / "neutral_mhs_overnight_extended_cases.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def summarize_soft_decay_smoke() -> None:
    cand_path = OUT / "transient_peak_soft_decay_episode_auto_smoke_1h" / "casebook_summary.csv"
    a0_path = OUT / "transient_peak_soft_decay_a0_smoke_1h" / "casebook_summary.csv"
    if not cand_path.exists() or not a0_path.exists():
        return
    cand = pd.read_csv(cand_path)
    a0 = pd.read_csv(a0_path)
    keep = [
        "case_id",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_safety_fallback_ratio",
    ]
    merged = a0[keep].merge(cand[keep], on="case_id", suffixes=("_a0", "_candidate"))
    merged["pump_saved_m3"] = (
        merged["primary_pump_work_m3_a0"].fillna(0.0)
        - merged["primary_pump_work_m3_candidate"].fillna(0.0)
    )
    merged["d_pitch_p95"] = (
        merged["primary_pitch_p95_candidate"].fillna(0.0)
        - merged["primary_pitch_p95_a0"].fillna(0.0)
    )
    merged["d_roll_p95"] = (
        merged["primary_roll_p95_candidate"].fillna(0.0)
        - merged["primary_roll_p95_a0"].fillna(0.0)
    )
    merged["d_fallback_ratio"] = (
        merged["primary_safety_fallback_ratio_candidate"].fillna(0.0)
        - merged["primary_safety_fallback_ratio_a0"].fillna(0.0)
    )
    closed = merged["primary_pump_work_m3_a0"].fillna(0.0).sum()
    primary = merged["primary_pump_work_m3_candidate"].fillna(0.0).sum()
    saved = merged["pump_saved_m3"].sum()
    saving = 100.0 * saved / closed if closed else 0.0
    active_saved_cases = int((merged["pump_saved_m3"] > 1.0).sum())
    harmed_cases = int((merged["pump_saved_m3"] < -1.0).sum())
    rows = [
        ["cases", len(merged)],
        ["closed_pump_m3", f"{closed:.2f}"],
        ["candidate_pump_m3", f"{primary:.2f}"],
        ["pump_saved_m3", f"{saved:.2f}"],
        ["pump_saving_pct", f"{saving:.2f}%"],
        ["active_saved_cases", active_saved_cases],
        ["harmed_cases", harmed_cases],
        ["mean_d_pitch_p95", f"{merged['d_pitch_p95'].mean():.3f}"],
        ["mean_d_roll_p95", f"{merged['d_roll_p95'].mean():.3f}"],
        ["max_d_fallback_ratio", f"{merged['d_fallback_ratio'].max():.3f}"],
    ]
    merged.to_csv(RAW / "transient_peak_soft_decay_smoke_case_delta.csv", index=False)
    with (RAW / "transient_peak_soft_decay_smoke_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    (PAPER / "transient_peak_soft_decay_smoke_summary.md").write_text(
        "\n".join(
            [
                "# Transient Soft-Decay Smoke Summary",
                "",
                "This is the only overnight short smoke candidate. It tests whether the existing relief/decay specialist can cover softer decay cases without creating a new branch.",
                "",
                markdown_table(["Metric", "Value"], rows),
                "",
                "Decision: promote only if the saving is material and the fallback/p95 deltas stay small. Otherwise keep soft-decay folded into diagnostics rather than a new specialist.",
            ]
        ),
        encoding="utf-8",
    )


def summarize_neutral_mhs_overnight() -> None:
    cand_path = OUT / "neutral_mhs_clean_auto_overnight_1h" / "casebook_summary.csv"
    a0_path = OUT / "neutral_mhs_a0_overnight_1h" / "casebook_summary.csv"
    if not cand_path.exists() or not a0_path.exists():
        return
    cand = pd.read_csv(cand_path)
    a0 = pd.read_csv(a0_path)
    keep = [
        "case_id",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_safety_fallback_ratio",
    ]
    merged = a0[keep].merge(cand[keep], on="case_id", suffixes=("_a0", "_candidate"))
    merged["pump_saved_m3"] = (
        merged["primary_pump_work_m3_a0"].fillna(0.0)
        - merged["primary_pump_work_m3_candidate"].fillna(0.0)
    )
    merged["d_p95_max_axis"] = (
        merged[["primary_pitch_p95_candidate", "primary_roll_p95_candidate"]].abs().max(axis=1)
        - merged[["primary_pitch_p95_a0", "primary_roll_p95_a0"]].abs().max(axis=1)
    )
    merged["d_fallback_ratio"] = (
        merged["primary_safety_fallback_ratio_candidate"].fillna(0.0)
        - merged["primary_safety_fallback_ratio_a0"].fillna(0.0)
    )
    a0_pump = merged["primary_pump_work_m3_a0"].fillna(0.0).sum()
    cand_pump = merged["primary_pump_work_m3_candidate"].fillna(0.0).sum()
    saved = merged["pump_saved_m3"].sum()
    positive = merged.loc[merged["pump_saved_m3"] > 0, "pump_saved_m3"]
    rows = [
        ["cases", len(merged)],
        ["a0_pump_m3", f"{a0_pump:.2f}"],
        ["candidate_pump_m3", f"{cand_pump:.2f}"],
        ["pump_saved_m3", f"{saved:.2f}"],
        ["pump_saving_pct", f"{100.0 * saved / a0_pump if a0_pump else 0.0:.2f}%"],
        ["saved_cases", int((merged["pump_saved_m3"] > 1.0).sum())],
        ["harmed_cases", int((merged["pump_saved_m3"] < -1.0).sum())],
        ["top_case_share", f"{positive.max() / saved if saved > 0 and len(positive) else 0.0:.3f}"],
        ["max_d_fallback_ratio", f"{merged['d_fallback_ratio'].max():.3f}"],
        ["max_d_p95_max_axis", f"{merged['d_p95_max_axis'].max():.3f}"],
    ]
    merged.to_csv(RAW / "neutral_mhs_overnight_extended_case_delta.csv", index=False)
    with (RAW / "neutral_mhs_overnight_extended_summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    (PAPER / "neutral_mhs_overnight_extended_summary.md").write_text(
        "\n".join(
            [
                "# Neutral MHS Overnight Extended Summary",
                "",
                "This is a broader fixed-rule confirmation run for the second specialist candidate. It does not change the controller or tune thresholds.",
                "",
                markdown_table(["Metric", "Value"], rows),
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    source_manifest()
    build_soft_decay_smoke_casebook()
    build_neutral_mhs_overnight_casebook()
    summarize_soft_decay_smoke()
    summarize_neutral_mhs_overnight()
    candidates = build_candidate_library()
    matrix = write_action_matrix(candidates)
    write_markdown_outputs(candidates, matrix)
    print(f"WROTE {PAPER / 'overnight_run_manifest.md'}")
    print(f"WROTE {RAW / 'regime_candidate_library_overnight.csv'}")
    print(f"WROTE {PAPER / 'overnight_regime_optimization_report_zh.md'}")
    print(f"WROTE {CASEBOOKS / 'transient_peak_soft_decay_smoke_cases.csv'}")


if __name__ == "__main__":
    main()
