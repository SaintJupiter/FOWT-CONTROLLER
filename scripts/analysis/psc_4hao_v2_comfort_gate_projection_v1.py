#!/usr/bin/env python3
"""Build a 96-case posture stratification and 4hao v2 comfort-gate projection.

This is an offline decision audit. It does not replay a new 4hao controller;
it uses the existing 0hao/1hao paired casebook plus the current 4hao candidate
pool to decide where a true mild-arm replay should focus next.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
DECISIONS = (
    BASE
    / "psc_structural_selector_96case_pair_3hao_v2_learned_v1"
    / "structural_selector_case_decisions.csv"
)
CANDIDATES = BASE / "psc_4hao_moderate_arm_feasibility_v1" / "4hao_candidate_pool.csv"
SCENARIOS = BASE / "psc_4hao_moderate_arm_feasibility_v1" / "4hao_scenario_summary.csv"
SAFETY_TS = BASE / "degradation_ladder_96case_pair" / "current_forecast_adaptive" / "timeseries"
ECON_TS = BASE / "degradation_ladder_96case_pair" / "learned_rawenv_mainline" / "timeseries"

OUT = BASE / "psc_4hao_v2_comfort_gate_projection_v1"
CASE_TABLE = OUT / "96case_posture_stratification.csv"
STRATA_SUMMARY = OUT / "96case_posture_stratification_summary.csv"
GATE_CANDIDATES = OUT / "4hao_v2_candidate_gate_table.csv"
GATE_PROJECTION = OUT / "4hao_v2_gate_projection.csv"
REPORT = OUT / "4hao_v2_recommendation.md"

ARM_SAFETY = "current_forecast_adaptive"
ARM_ECON = "learned_rawenv_mainline"


CANDIDATE_NAMES = {
    "already_opened_by_3hao_v2": "3号已打开",
    "strict_unopened": "4号候选A: 严格安全未打开",
    "tiny_gt5_debt_appendix": "4号候选B: 极小gt5债务",
    "small_gt5_debt_pareto": "4号候选C: 边界Pareto",
    "4hao_relief_edge_candidate": "拒绝/待温和化: relief边缘高风险",
    "4hao_neutral_highrisk_candidate": "拒绝/待温和化: neutral高风险",
    "boundary_falsifier_not_first_release": "拒绝: boundary反例",
    "no_rawenv_pump_gain": "无有效省水",
}

REGIME_NAMES = {
    "transient_peak_future_decay": "relief衰减",
    "moderate_high_steady": "中高稳态",
    "direction_reversal_boundary": "方向反转边界",
    "direction_reversal_catchup_boundary": "方向反转追赶边界",
    "neutral": "neutral",
    "stable": "稳定",
}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _scalar(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def _short_case_name(case_id: str, mixed_regime: str) -> str:
    parts = case_id.split("_")
    if len(parts) >= 4:
        family = parts[2]
        number = parts[3]
    else:
        family = case_id
        number = ""
    regime = REGIME_NAMES.get(mixed_regime, mixed_regime)
    return f"{case_id[:2]} {family}-{number} / {regime}"


def _find_ts(ts_dir: Path, case_id: str, arm_name: str) -> Path | None:
    matches = sorted(ts_dir.glob(f"{case_id}_*_{arm_name}_timeseries.csv"))
    return matches[0] if matches else None


@lru_cache(maxsize=None)
def _timeseries_metrics(case_id: str) -> dict[str, float | str]:
    safety_path = _find_ts(SAFETY_TS, case_id, ARM_SAFETY)
    econ_path = _find_ts(ECON_TS, case_id, ARM_ECON)
    base = {
        "case_id": case_id,
        "timeseries_status": "missing",
        "ts_rows": np.nan,
        "ts_d_time_gt3_s": np.nan,
        "ts_d_time_gt4_s": np.nan,
        "ts_d_time_gt5_s": np.nan,
        "ts_safety_p95_axis_deg": np.nan,
        "ts_economy_p95_axis_deg": np.nan,
        "ts_d_p95_axis_deg": np.nan,
        "ts_safety_mean_axis_deg": np.nan,
        "ts_economy_mean_axis_deg": np.nan,
        "ts_d_mean_axis_deg": np.nan,
        "ts_safety_max_axis_deg": np.nan,
        "ts_economy_max_axis_deg": np.nan,
        "ts_d_max_axis_deg": np.nan,
        "ts_economy_p95_margin_to_5deg": np.nan,
    }
    if safety_path is None or econ_path is None:
        return base

    usecols = ["pitch_deg", "roll_deg"]
    safety = pd.read_csv(safety_path, usecols=usecols)
    econ = pd.read_csv(econ_path, usecols=usecols)
    n = min(len(safety), len(econ))
    if n <= 0:
        return base
    safety_axis = safety.iloc[:n][["pitch_deg", "roll_deg"]].abs().max(axis=1).to_numpy(float)
    econ_axis = econ.iloc[:n][["pitch_deg", "roll_deg"]].abs().max(axis=1).to_numpy(float)
    safety_p95 = float(np.quantile(safety_axis, 0.95))
    econ_p95 = float(np.quantile(econ_axis, 0.95))
    return {
        "case_id": case_id,
        "timeseries_status": "ok",
        "ts_rows": float(n),
        "ts_d_time_gt3_s": float((econ_axis > 3.0).sum() - (safety_axis > 3.0).sum()),
        "ts_d_time_gt4_s": float((econ_axis > 4.0).sum() - (safety_axis > 4.0).sum()),
        "ts_d_time_gt5_s": float((econ_axis > 5.0).sum() - (safety_axis > 5.0).sum()),
        "ts_safety_p95_axis_deg": safety_p95,
        "ts_economy_p95_axis_deg": econ_p95,
        "ts_d_p95_axis_deg": econ_p95 - safety_p95,
        "ts_safety_mean_axis_deg": float(np.mean(safety_axis)),
        "ts_economy_mean_axis_deg": float(np.mean(econ_axis)),
        "ts_d_mean_axis_deg": float(np.mean(econ_axis) - np.mean(safety_axis)),
        "ts_safety_max_axis_deg": float(np.max(safety_axis)),
        "ts_economy_max_axis_deg": float(np.max(econ_axis)),
        "ts_d_max_axis_deg": float(np.max(econ_axis) - np.max(safety_axis)),
        "ts_economy_p95_margin_to_5deg": 5.0 - econ_p95,
    }


def _rawenv_stratum(row: pd.Series) -> str:
    gain = _scalar(row, "pump_gain_m3")
    d_gt5 = _scalar(row, "d_time_gt5_s")
    d_fallback = _scalar(row, "d_fallback_s")
    d_p95 = _scalar(row, "d_p95_axis_deg")
    if gain <= 1e-9:
        return "H 无有效省水"
    if d_fallback > 0 or d_gt5 >= 60:
        return "G 明显安全失败"
    if d_gt5 > 10:
        return "F 中等gt5债务/边界"
    if d_gt5 > 0:
        return "E 小gt5债务候选"
    if d_p95 <= -0.10:
        return "A 姿态更好且硬安全"
    if abs(d_p95) <= 0.10:
        return "B 姿态基本持平且硬安全"
    if d_p95 <= 0.50:
        return "C 轻姿态债务但硬安全"
    return "D p95姿态债务但硬安全"


def _casebook_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    decisions = pd.read_csv(DECISIONS)
    candidates = pd.read_csv(CANDIDATES)

    ts_rows = pd.DataFrame([_timeseries_metrics(str(case_id)) for case_id in decisions["case_id"]])
    df = decisions.merge(
        candidates[
            [
                "case_id",
                "candidate_category",
                "risk_marker",
                "rawenv_safety_debt_per_100m3_gt5_s",
                "rawenv_safety_debt_per_100m3_fallback_s",
            ]
        ],
        on="case_id",
        how="left",
        suffixes=("", "_candidate"),
    ).merge(ts_rows, on="case_id", how="left")

    df["candidate_name"] = df["candidate_category"].map(CANDIDATE_NAMES).fillna(df["candidate_category"])
    df["case_name"] = [
        _short_case_name(str(case_id), str(regime))
        for case_id, regime in zip(df["case_id"], df["mixed_regime"])
    ]
    df["rawenv_stratum"] = df.apply(_rawenv_stratum, axis=1)
    df["is_3hao_selected_arm_opened"] = df["selected_arm"].astype(str).eq(ARM_ECON)
    df["is_3hao_strict_rawenv_flag_opened"] = (
        _num(df["use_rawenv"]).eq(1.0) & _num(df["rawenv_safe"]).eq(1.0) & (_num(df["pump_gain_m3"]) > 0.0)
    )
    df["selected_saved_m3"] = _num(df["safety_pump_m3"]) - _num(df["selected_pump_m3"])
    df["selected_saving_pct_case"] = 100.0 * df["selected_saved_m3"] / _num(df["safety_pump_m3"]).clip(lower=1e-9)
    df["rawenv_saving_pct_case"] = 100.0 * _num(df["pump_gain_m3"]) / _num(df["safety_pump_m3"]).clip(lower=1e-9)
    df["rawenv_potential_gain_m3"] = _num(df["pump_gain_m3"]).clip(lower=0.0)
    df["selected_saved_m3_positive"] = _num(df["selected_saved_m3"]).clip(lower=0.0)

    total_baseline = float(_num(df["safety_pump_m3"]).sum())
    summary_rows = []
    order = [
        "A 姿态更好且硬安全",
        "B 姿态基本持平且硬安全",
        "C 轻姿态债务但硬安全",
        "D p95姿态债务但硬安全",
        "E 小gt5债务候选",
        "F 中等gt5债务/边界",
        "G 明显安全失败",
        "H 无有效省水",
    ]
    for stratum in order:
        part = df[df["rawenv_stratum"].eq(stratum)].copy()
        if part.empty:
            continue
        potential = float(part["rawenv_potential_gain_m3"].sum())
        selected_saved = float(part["selected_saved_m3_positive"].sum())
        summary_rows.append(
            {
                "rawenv_stratum": stratum,
                "case_count": int(len(part)),
                "3hao_selected_arm_opened_count": int(part["is_3hao_selected_arm_opened"].sum()),
                "3hao_strict_rawenv_flag_opened_count": int(part["is_3hao_strict_rawenv_flag_opened"].sum()),
                "rawenv_potential_saved_m3": potential,
                "rawenv_potential_saving_pct_of_96case": 100.0 * potential / max(total_baseline, 1e-9),
                "3hao_selected_saved_m3": selected_saved,
                "3hao_selected_saving_pct_of_96case": 100.0 * selected_saved / max(total_baseline, 1e-9),
                "sum_d_time_gt5_s_if_rawenv": float(_num(part["d_time_gt5_s"]).sum()),
                "sum_d_fallback_s_if_rawenv": float(_num(part["d_fallback_s"]).sum()),
                "max_d_p95_axis_deg_if_rawenv": float(_num(part["d_p95_axis_deg"]).max()),
                "min_p95_margin_to_5deg_if_rawenv": float(_num(part["ts_economy_p95_margin_to_5deg"]).min()),
                "example_cases": ";".join(part["case_id"].head(6).astype(str)),
            }
        )

    case_cols = [
        "case_id",
        "case_name",
        "mixed_regime",
        "rawenv_stratum",
        "candidate_name",
        "is_3hao_selected_arm_opened",
        "is_3hao_strict_rawenv_flag_opened",
        "safety_pump_m3",
        "economy_pump_m3",
        "selected_pump_m3",
        "pump_gain_m3",
        "rawenv_saving_pct_case",
        "selected_saved_m3",
        "selected_saving_pct_case",
        "d_p95_axis_deg",
        "ts_safety_p95_axis_deg",
        "ts_economy_p95_axis_deg",
        "ts_economy_p95_margin_to_5deg",
        "ts_d_time_gt3_s",
        "ts_d_time_gt4_s",
        "d_time_gt5_s",
        "d_time_gt6_s",
        "d_fallback_s",
        "d_max_axis_deg",
        "risk_reason",
        "decision_reason",
        "timeseries_status",
    ]
    case_table = df[case_cols].sort_values(["rawenv_stratum", "case_id"]).copy()
    summary = pd.DataFrame(summary_rows)
    return df, case_table, summary


def _gate_projection(casebook: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = pd.read_csv(SCENARIOS)
    candidates = pd.read_csv(CANDIDATES)
    ts_rows = pd.DataFrame([_timeseries_metrics(str(case_id)) for case_id in candidates["case_id"]])
    candidates = candidates.merge(ts_rows, on="case_id", how="left")
    candidates["candidate_name"] = candidates["candidate_category"].map(CANDIDATE_NAMES).fillna(
        candidates["candidate_category"]
    )
    candidates["case_name"] = [
        _short_case_name(str(case_id), str(regime))
        for case_id, regime in zip(candidates["case_id"], candidates["mixed_regime"])
    ]
    candidates["rawenv_saving_pct_case"] = (
        100.0 * _num(candidates["pump_gain_m3"]) / _num(candidates["pump_m3_safety"]).clip(lower=1e-9)
    )

    base_row = scenarios[scenarios["scenario"].eq("3hao_v2_strict")].iloc[0]
    base_saved = float(base_row["saved_m3"])
    total_baseline = float(base_row["pump_m3"] + base_row["saved_m3"])
    base_cases = int(base_row["economy_cases"])

    def available_extra(df: pd.DataFrame) -> pd.Series:
        return (
            ~df["candidate_category"].eq("already_opened_by_3hao_v2")
            & (_num(df["pump_gain_m3"]) > 0.0)
        )

    def make_projection(
        scenario: str,
        readable_name: str,
        mask: pd.Series,
        status: str,
        note: str,
    ) -> dict[str, object]:
        extra = candidates[mask].copy()
        extra_saved = float(_num(extra["pump_gain_m3"]).sum())
        saved = base_saved + extra_saved
        extra_cases = list(extra["case_id"].astype(str))
        return {
            "scenario": scenario,
            "readable_name": readable_name,
            "status": status,
            "pump_m3": total_baseline - saved,
            "saved_m3": saved,
            "saving_pct": 100.0 * saved / max(total_baseline, 1e-9),
            "economy_cases": base_cases + int(len(extra)),
            "extra_cases": int(len(extra)),
            "extra_saved_m3": extra_saved,
            "extra_saved_pct_of_96case": 100.0 * extra_saved / max(total_baseline, 1e-9),
            "extra_d_time_gt3_s": float(_num(extra["ts_d_time_gt3_s"]).sum()) if len(extra) else 0.0,
            "extra_d_time_gt4_s": float(_num(extra["ts_d_time_gt4_s"]).sum()) if len(extra) else 0.0,
            "extra_d_time_gt5_s": float(_num(extra["d_time_gt5_s"]).sum()) if len(extra) else 0.0,
            "extra_d_fallback_s": float(_num(extra["d_fallback_s"]).sum()) if len(extra) else 0.0,
            "max_extra_d_p95_axis_deg": float(_num(extra["d_p95_axis_deg"]).max()) if len(extra) else 0.0,
            "min_extra_p95_margin_to_5deg": float(_num(extra["ts_economy_p95_margin_to_5deg"]).min())
            if len(extra)
            else np.nan,
            "extra_case_ids": ";".join(extra_cases),
            "note": note,
        }

    extra_base = available_extra(candidates)
    strict_mask = (
        extra_base
        & (_num(candidates["risk_marker"]).eq(0.0))
        & (_num(candidates["d_fallback_s"]) <= 0.0)
        & (_num(candidates["d_time_gt5_s"]) <= 0.0)
        & (_num(candidates["ts_d_time_gt4_s"]) <= 0.0)
        & (_num(candidates["d_p95_axis_deg"]) <= 0.50)
    )
    appendix_mask = (
        extra_base
        & (_num(candidates["risk_marker"]).eq(0.0))
        & (_num(candidates["d_fallback_s"]) <= 0.0)
        & (_num(candidates["d_time_gt5_s"]) <= 10.0)
        & (_num(candidates["d_p95_axis_deg"]) <= 1.50)
    )
    relaxed_15_mask = (
        extra_base
        & (_num(candidates["risk_marker"]).eq(0.0))
        & (_num(candidates["d_fallback_s"]) <= 0.0)
        & (_num(candidates["d_time_gt5_s"]) < 60.0)
        & (_num(candidates["d_p95_axis_deg"]) <= 2.00)
    )

    rows = [
        make_projection(
            "3hao_current_reported",
            "3号当前主线",
            candidates["case_id"].eq("__none__"),
            "baseline",
            "Current scenario-summary口径: 39个economy cases, 约13.08%节水。",
        ),
        make_projection(
            "4hao_v2_strict_comfort_gate",
            "4号v2严格舒适门",
            strict_mask,
            "recommended_main",
            "只加fallback=0、gt5不增加、gt4不增加、p95姿态债务<=0.50deg的未打开候选。",
        ),
        make_projection(
            "4hao_v2_comfort_appendix",
            "4号v2观察附录门",
            appendix_mask,
            "appendix_only",
            "允许极小gt5债务(<=10s)且p95姿态债务<=1.50deg；用于画图和人工判断。",
        ),
        make_projection(
            "4hao_v2_relaxed_15pct_diagnostic",
            "15%诊断线(不建议主线)",
            relaxed_15_mask,
            "diagnostic_not_main",
            "为了达到15%会吃下94这类长时间姿态偏高样本，应作为反例压力测试。",
        ),
    ]

    for scenario_name in [
        "theoretical_4hao_5%_of_unopened_risky_gain",
        "theoretical_4hao_10%_of_unopened_risky_gain",
    ]:
        src = scenarios[scenarios["scenario"].eq(scenario_name)]
        if src.empty:
            continue
        src_row = src.iloc[0]
        rows.append(
            {
                "scenario": scenario_name,
                "readable_name": "真实温和arm泵量目标 " + scenario_name.split("_4hao_")[1].replace("_", " "),
                "status": "requires_real_replay",
                "pump_m3": float(src_row["pump_m3"]),
                "saved_m3": float(src_row["saved_m3"]),
                "saving_pct": float(src_row["saving_pct"]),
                "economy_cases": int(src_row["economy_cases"]),
                "extra_cases": int(src_row["extra_cases"]),
                "extra_saved_m3": float(src_row["extra_saved_m3"]),
                "extra_saved_pct_of_96case": 100.0 * float(src_row["extra_saved_m3"]) / max(total_baseline, 1e-9),
                "extra_d_time_gt3_s": np.nan,
                "extra_d_time_gt4_s": np.nan,
                "extra_d_time_gt5_s": np.nan,
                "extra_d_fallback_s": np.nan,
                "max_extra_d_p95_axis_deg": np.nan,
                "min_extra_p95_margin_to_5deg": np.nan,
                "extra_case_ids": str(src_row["extra_case_ids"]),
                "note": "这只是泵量空间估计，安全性必须靠真实4号温和arm闭环回放确认。",
            }
        )

    projection = pd.DataFrame(rows)

    candidates["gate_strict_main"] = strict_mask
    candidates["gate_appendix"] = appendix_mask
    candidates["gate_relaxed_15pct_diagnostic"] = relaxed_15_mask
    candidate_cols = [
        "case_id",
        "case_name",
        "mixed_regime",
        "candidate_name",
        "pump_m3_safety",
        "pump_m3_rawenv",
        "pump_gain_m3",
        "rawenv_saving_pct_case",
        "d_p95_axis_deg",
        "ts_safety_p95_axis_deg",
        "ts_economy_p95_axis_deg",
        "ts_economy_p95_margin_to_5deg",
        "ts_d_time_gt3_s",
        "ts_d_time_gt4_s",
        "d_time_gt5_s",
        "d_time_gt6_s",
        "d_fallback_s",
        "d_max_axis_deg",
        "gate_strict_main",
        "gate_appendix",
        "gate_relaxed_15pct_diagnostic",
        "risk_reason",
        "decision_reason",
        "label",
    ]
    gate_candidates = candidates[candidate_cols].sort_values(
        ["gate_strict_main", "gate_appendix", "gate_relaxed_15pct_diagnostic", "pump_gain_m3"],
        ascending=[False, False, False, False],
    )
    return projection, gate_candidates


def _markdown_table(rows: list[dict[str, object]], cols: list[tuple[str, str]]) -> list[str]:
    lines = []
    header = "| " + " | ".join(title for _, title in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines.extend([header, sep])
    for row in rows:
        values = []
        for key, _ in cols:
            value = row.get(key, "")
            if isinstance(value, float):
                if np.isnan(value):
                    values.append("")
                elif "pct" in key or key.endswith("_saving_pct"):
                    values.append(f"{value:.2f}%")
                elif "m3" in key:
                    values.append(f"{value:.1f}")
                elif "deg" in key:
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value:.1f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    casebook: pd.DataFrame,
    strata_summary: pd.DataFrame,
    projection: pd.DataFrame,
    gate_candidates: pd.DataFrame,
) -> None:
    total_pump = float(_num(casebook["safety_pump_m3"]).sum())
    selected_arm = casebook[casebook["is_3hao_selected_arm_opened"]].copy()
    strict_rawenv = casebook[casebook["is_3hao_strict_rawenv_flag_opened"]].copy()
    selected_saved = float(_num(selected_arm["selected_saved_m3"]).sum())
    strict_saved = float(_num(strict_rawenv["pump_gain_m3"]).sum())

    candidate_counts = (
        gate_candidates["candidate_name"].value_counts().rename_axis("candidate_name").reset_index(name="case_count")
    )
    candidate_count_rows = candidate_counts.to_dict("records")

    gate_rows = projection[
        [
            "readable_name",
            "status",
            "saved_m3",
            "saving_pct",
            "extra_cases",
            "extra_saved_m3",
            "extra_d_time_gt4_s",
            "extra_d_time_gt5_s",
            "extra_d_fallback_s",
            "max_extra_d_p95_axis_deg",
            "extra_case_ids",
        ]
    ].to_dict("records")

    strict_cases = gate_candidates[gate_candidates["gate_strict_main"]].copy()
    appendix_cases = gate_candidates[gate_candidates["gate_appendix"] & ~gate_candidates["gate_strict_main"]].copy()
    relaxed_only_cases = gate_candidates[
        gate_candidates["gate_relaxed_15pct_diagnostic"]
        & ~gate_candidates["gate_appendix"]
        & ~gate_candidates["gate_strict_main"]
    ].copy()

    def candidate_rows(df: pd.DataFrame) -> list[dict[str, object]]:
        rows = []
        for row in df.to_dict("records"):
            rows.append(
                {
                    "case_id": row["case_id"],
                    "case_name": row["case_name"],
                    "candidate_name": row["candidate_name"],
                    "pump_gain_m3": float(row["pump_gain_m3"]),
                    "rawenv_saving_pct_case": float(row["rawenv_saving_pct_case"]),
                    "d_p95_axis_deg": float(row["d_p95_axis_deg"]),
                    "ts_d_time_gt4_s": float(row["ts_d_time_gt4_s"]),
                    "d_time_gt5_s": float(row["d_time_gt5_s"]),
                    "d_fallback_s": float(row["d_fallback_s"]),
                }
            )
        return rows

    lines: list[str] = [
        "# 4号 v2 舒适门控投影与下一步处置",
        "",
        "## 这一步做了什么",
        "",
        "- 这一步不是重新实现一个真实4号控制器，而是把现有96-case 0号/1号配对结果重新分层，明确哪些省水是硬安全、哪些只是姿态债务、哪些是明显失败。",
        "- 结果文件统一把节水量(m3)和节水百分比(%)同时列出；姿态不再用平均姿态百分比做主指标，改用p95角度、d_p95角度、gt3/gt4/gt5时间和fallback。",
        "- 候选名字已经换成可读标签，例如“4号候选A: 严格安全未打开”“4号候选B: 极小gt5债务”“拒绝/待温和化: neutral高风险”。",
        "",
        "## 当前口径核对",
        "",
    ]
    lines.extend(
        _markdown_table(
            [
                {
                    "name": "3号当前主线(场景汇总口径)",
                    "cases": 39,
                    "saved_m3": 5999.149045747043,
                    "saving_pct": 13.075701166478515,
                    "note": "沿用旧汇总表，用来和之前13.08%对齐。",
                },
                {
                    "name": "3号严格rawenv安全正例(选择表复算)",
                    "cases": int(len(strict_rawenv)),
                    "saved_m3": strict_saved,
                    "saving_pct": 100.0 * strict_saved / max(total_pump, 1e-9),
                    "note": "严格 use_rawenv=1/rawenv_safe=1/pump_gain>0；少了一个低风险release口径差异样本。",
                },
                {
                    "name": "3号实际selected_arm复算",
                    "cases": int(len(selected_arm)),
                    "saved_m3": selected_saved,
                    "saving_pct": 100.0 * selected_saved / max(total_pump, 1e-9),
                    "note": "按 selected_arm=learned_rawenv_mainline 复算，和13.08%一致。",
                },
            ],
            [
                ("name", "口径"),
                ("cases", "打开case"),
                ("saved_m3", "节水m3"),
                ("saving_pct", "节水%"),
                ("note", "说明"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 96-case 分层概览",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            strata_summary.to_dict("records"),
            [
                ("rawenv_stratum", "1号全开相对0号的类型"),
                ("case_count", "case数"),
                ("3hao_selected_arm_opened_count", "3号已打开"),
                ("rawenv_potential_saved_m3", "1号潜在节水m3"),
                ("rawenv_potential_saving_pct_of_96case", "1号潜在节水%"),
                ("3hao_selected_saved_m3", "3号已拿到m3"),
                ("3hao_selected_saving_pct_of_96case", "3号已拿到%"),
                ("sum_d_time_gt5_s_if_rawenv", "若全开gt5增量s"),
                ("sum_d_fallback_s_if_rawenv", "若全开fallback增量s"),
                ("max_d_p95_axis_deg_if_rawenv", "最大d_p95 deg"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 4号 v2 门控投影",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            gate_rows,
            [
                ("readable_name", "方案"),
                ("status", "定位"),
                ("saved_m3", "总节水m3"),
                ("saving_pct", "总节水%"),
                ("extra_cases", "新增case"),
                ("extra_saved_m3", "新增节水m3"),
                ("extra_d_time_gt4_s", "新增gt4 s"),
                ("extra_d_time_gt5_s", "新增gt5 s"),
                ("extra_d_fallback_s", "新增fallback s"),
                ("max_extra_d_p95_axis_deg", "最大d_p95 deg"),
                ("extra_case_ids", "新增case id"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 候选池改名后的分布",
            "",
        ]
    )
    lines.extend(_markdown_table(candidate_count_rows, [("candidate_name", "候选类型"), ("case_count", "case数")]))
    lines.extend(
        [
            "",
            "## 可以作为主线、附录和反例的具体case",
            "",
            "### 主线: 严格舒适门",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            candidate_rows(strict_cases),
            [
                ("case_id", "case"),
                ("case_name", "易读名"),
                ("candidate_name", "候选类型"),
                ("pump_gain_m3", "节水m3"),
                ("rawenv_saving_pct_case", "case节水%"),
                ("d_p95_axis_deg", "d_p95 deg"),
                ("ts_d_time_gt4_s", "d_gt4 s"),
                ("d_time_gt5_s", "d_gt5 s"),
                ("d_fallback_s", "d_fallback s"),
            ],
        )
    )
    lines.extend(["", "### 附录: 可画图观察但不进主线", ""])
    lines.extend(
        _markdown_table(
            candidate_rows(appendix_cases),
            [
                ("case_id", "case"),
                ("case_name", "易读名"),
                ("candidate_name", "候选类型"),
                ("pump_gain_m3", "节水m3"),
                ("rawenv_saving_pct_case", "case节水%"),
                ("d_p95_axis_deg", "d_p95 deg"),
                ("ts_d_time_gt4_s", "d_gt4 s"),
                ("d_time_gt5_s", "d_gt5 s"),
                ("d_fallback_s", "d_fallback s"),
            ],
        )
    )
    lines.extend(["", "### 15%反例线: 不建议作为主结果", ""])
    lines.extend(
        _markdown_table(
            candidate_rows(relaxed_only_cases),
            [
                ("case_id", "case"),
                ("case_name", "易读名"),
                ("candidate_name", "候选类型"),
                ("pump_gain_m3", "节水m3"),
                ("rawenv_saving_pct_case", "case节水%"),
                ("d_p95_axis_deg", "d_p95 deg"),
                ("ts_d_time_gt4_s", "d_gt4 s"),
                ("d_time_gt5_s", "d_gt5 s"),
                ("d_fallback_s", "d_fallback s"),
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 简单结论",
            "",
            "1. 当前节水少，不是因为没有省水空间，而是因为3号只能在0号和1号之间选择；1号一旦全开会在很多case里带来gt5或fallback风险，3号只能放弃这些case。",
            "2. 如果4号v2仍然只是从现有1号全开结果里挑case，严格舒适门只会新增09一个case，总节水约13.15%，提升很小。",
            "3. 09+10作为观察附录能到约13.84%，但10已经有p95姿态债务和少量gt5债务。",
            "4. 要到15.56%必须把94也吃进去；94就是你担心的那种“没爆5度但整体姿态长期变差”的边界样本，所以不能作为主结果，只能当压力测试反例。",
            "5. 因此下一步不该继续靠放宽阈值硬吃case，而应该实现真实4号温和arm：短hold、小预算、p95/gt4/gt5/fallback门控，先在10、94和少量relief边缘case上回放，再扩到96-case。",
            "",
            "## 下一步执行计划",
            "",
            "1. 做真实4号 mild-arm replay：把1号全开省水动作缩成20%/30%/40%三档，加入单次hold时长和全局pump budget。",
            "2. 主线验收门：fallback=0、d_gt5=0、d_gt4不明显增加、d_p95_axis_deg<=0.50；附录门可允许d_gt5<=10s但必须单独标注。",
            "3. 先跑10、94、11、18、22、24这些失败/边界case，看温和动作能不能把姿态债务压下来，而不是只靠容忍度放宽。",
            "4. 再跑完整96-case，目标是主线稳定14%上下；如果要报15%+，必须明确是Pareto附录还是主线结果。",
            "",
            "## 输出文件",
            "",
            f"- 96-case逐case分层: `{CASE_TABLE}`",
            f"- 96-case分层汇总: `{STRATA_SUMMARY}`",
            f"- 4号v2候选门控表: `{GATE_CANDIDATES}`",
            f"- 4号v2方案投影表: `{GATE_PROJECTION}`",
            f"- 本报告: `{REPORT}`",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    casebook, case_table, strata_summary = _casebook_tables()
    projection, gate_candidates = _gate_projection(casebook)
    case_table.to_csv(CASE_TABLE, index=False)
    strata_summary.to_csv(STRATA_SUMMARY, index=False)
    gate_candidates.to_csv(GATE_CANDIDATES, index=False)
    projection.to_csv(GATE_PROJECTION, index=False)
    _write_report(casebook, strata_summary, projection, gate_candidates)
    print(f"Wrote {REPORT}")
    print(f"Wrote {GATE_PROJECTION}")
    print(f"Wrote {CASE_TABLE}")


if __name__ == "__main__":
    main()
