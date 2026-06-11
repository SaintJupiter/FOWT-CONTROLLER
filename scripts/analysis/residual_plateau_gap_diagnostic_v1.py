#!/usr/bin/env python3
"""Diagnose whether residual-high plateau is a second matureable regime.

This is deliberately read-only.  It answers two questions:

1. Does the residual_high_plateau subtype remain a real pump-saving opportunity
   when separated from slow-decay/reversal cases?
2. Is the failure of the automatic residual-high policy caused by a detector
   rule gap or by learned forecast error?
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
RAW = ROOT / "raw_tables"
PAPER = ROOT / "paper_ready"
OUT = ROOT / "residual_plateau_gap_diagnostic_v1"
OUT_RAW = OUT / "raw_tables"
OUT_PAPER = OUT / "paper_ready"


LOG_DIRS = {
    "guard10": {
        "learned": Path("outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/runs/guard10/v16_h120_learned_no_scheduler/planner_logs"),
        "oracle": Path("outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/guard10_v16_oracle_baseline/planner_logs"),
    },
    "broader20": {
        "learned": Path("outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/runs/broader20/v16_h120_learned_no_scheduler/planner_logs"),
        "oracle": Path("outputs/wind_prediction/h120_learned_risk_scheduler_probe_v1/broader20_v16_oracle_baseline/planner_logs"),
    },
}


def normalize_core(value: str) -> str:
    text = str(value).lower()
    text = text.replace("_prediction_primary_econ_planner_log.csv", "")
    text = text.replace("_prediction_primary_econ_timeseries.csv", "")
    text = re.sub(r"^\d+_", "", text)
    text = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{6}$", "", text)
    text = re.sub(r"^\d+_", "", text)
    return text


def log_index(log_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    if not log_dir.exists():
        return idx
    for path in sorted(log_dir.glob("*.csv")):
        idx[normalize_core(path.name)] = path
    return idx


def shape_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use raw blocks for regime-shape diagnosis.  The non-raw pressure_block*
    # fields may be internally weighted/attenuated by controller logic, which
    # can turn a true high-pressure plateau into an artificial decline.
    b0 = pd.to_numeric(
        df.get("raw_pressure_block0_norm", df.get("pressure_block0_norm", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    b1 = pd.to_numeric(
        df.get("raw_pressure_block1_norm", df.get("pressure_block1_norm", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    b2 = pd.to_numeric(
        df.get("raw_pressure_block2_norm", df.get("pressure_block2_norm", 0.0)),
        errors="coerce",
    ).fillna(0.0)
    near_max = pd.concat([b0, b1, b2], axis=1).max(axis=1)
    near_min = pd.concat([b0, b1, b2], axis=1).min(axis=1)
    near_range = near_max - near_min
    drop_0_60 = b0 - b2
    rise_0_60 = b2 - b0
    high = near_max >= 1.05
    plateau = high & (near_range < 0.15)
    decay = high & ((drop_0_60 >= 0.20) | ((near_range >= 0.30) & (b2 < b0)))
    rising = high & (rise_0_60 >= 0.15)
    slow_or_mixed = high & ~(plateau | decay | rising)
    return pd.DataFrame(
        {
            "b0": b0,
            "b1": b1,
            "b2": b2,
            "near_max": near_max,
            "near_range": near_range,
            "drop_0_60": drop_0_60,
            "rise_0_60": rise_0_60,
            "high": high,
            "plateau": plateau,
            "decay": decay,
            "rising": rising,
            "slow_or_mixed": slow_or_mixed,
        }
    )


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    value = df[col]
    if value.dtype == object:
        return value.fillna("").astype(str).str.lower().isin(["1", "true", "yes", "y"])
    return pd.to_numeric(value, errors="coerce").fillna(0) > 0


def veto_signal(df: pd.DataFrame) -> pd.Series:
    cols = [
        "far_horizon_reversal",
        "far_horizon_direction_shift",
        "far_horizon_hidden_intensification",
        "h120_scheduler_far_signflip_risk",
        "h120_scheduler_far_reintensification_after_relief",
        "h120_oracle_probe_far_reintensification",
    ]
    out = pd.Series(False, index=df.index)
    for col in cols:
        out = out | bool_col(df, col)
    return out


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def summarize_forecast_gap(case_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    learned_indexes = {name: log_index(paths["learned"]) for name, paths in LOG_DIRS.items()}
    oracle_indexes = {name: log_index(paths["oracle"]) for name, paths in LOG_DIRS.items()}

    target = case_table[
        case_table["fine_regime"].isin(
            [
                "residual_high_plateau_high_pump",
                "residual_high_slow_decay_or_moderate_relief",
                "direction_reversal_catchup_boundary",
            ]
        )
    ].copy()
    for _, case in target.iterrows():
        core_candidates = {
            normalize_core(case.get("case", "")),
            normalize_core(case.get("case_id", "")),
        }
        matched = False
        for set_name in LOG_DIRS:
            learned_idx = learned_indexes[set_name]
            oracle_idx = oracle_indexes[set_name]
            hit = None
            for core in core_candidates:
                if core in learned_idx and core in oracle_idx:
                    hit = core
                    break
            if hit is None:
                continue
            matched = True
            learned = pd.read_csv(learned_idx[hit])
            oracle = pd.read_csv(oracle_idx[hit])
            if "current_time_s" not in learned.columns or "current_time_s" not in oracle.columns:
                continue
            learned = learned.set_index("current_time_s")
            oracle = oracle.set_index("current_time_s")
            idx = learned.index.intersection(oracle.index)
            if len(idx) == 0:
                continue
            lf = shape_features(learned.loc[idx])
            of = shape_features(oracle.loc[idx])
            lv = veto_signal(learned.loc[idx])
            ov = veto_signal(oracle.loc[idx])

            row = {
                "case": case.get("case", ""),
                "case_id": case.get("case_id", ""),
                "fine_regime": case.get("fine_regime", ""),
                "set_name": set_name,
                "matched_core": hit,
                "rows": len(idx),
                "oracle_high_rows": int(of["high"].sum()),
                "learned_high_rows": int(lf["high"].sum()),
                "oracle_plateau_rows": int(of["plateau"].sum()),
                "learned_plateau_rows": int(lf["plateau"].sum()),
                "oracle_decay_rows": int(of["decay"].sum()),
                "learned_decay_rows": int(lf["decay"].sum()),
                "oracle_rising_rows": int(of["rising"].sum()),
                "learned_rising_rows": int(lf["rising"].sum()),
                "oracle_veto_rows": int(ov.sum()),
                "learned_veto_rows": int(lv.sum()),
                "mean_abs_b0_err": float((lf["b0"] - of["b0"]).abs().mean()),
                "mean_abs_b1_err": float((lf["b1"] - of["b1"]).abs().mean()),
                "mean_abs_b2_err": float((lf["b2"] - of["b2"]).abs().mean()),
                "oracle_plateau_frac": safe_div(float(of["plateau"].sum()), len(idx)),
                "learned_plateau_frac": safe_div(float(lf["plateau"].sum()), len(idx)),
                "oracle_veto_frac": safe_div(float(ov.sum()), len(idx)),
                "learned_veto_frac": safe_div(float(lv.sum()), len(idx)),
                "a0_pump_m3": case.get("a0_pump_m3", 0.0),
                "budget100_saving_pct": case.get("budget100_saving_pct", 0.0),
                "delta_time_gt5_s": case.get("delta_time_gt5_s", 0.0),
                "delta_fallback_time_s": case.get("delta_fallback_time_s", 0.0),
            }
            row["plateau_row_overlap"] = int((lf["plateau"] & of["plateau"]).sum())
            row["plateau_precision"] = safe_div(row["plateau_row_overlap"], row["learned_plateau_rows"])
            row["plateau_recall"] = safe_div(row["plateau_row_overlap"], row["oracle_plateau_rows"])
            row["veto_row_overlap"] = int((lv & ov).sum())
            row["veto_precision"] = safe_div(row["veto_row_overlap"], row["learned_veto_rows"])
            row["veto_recall"] = safe_div(row["veto_row_overlap"], row["oracle_veto_rows"])
            rows.append(row)
        if not matched:
            rows.append(
                {
                    "case": case.get("case", ""),
                    "case_id": case.get("case_id", ""),
                    "fine_regime": case.get("fine_regime", ""),
                    "set_name": "not_covered_by_existing_learned_oracle_logs",
                    "matched_core": "",
                    "rows": 0,
                    "a0_pump_m3": case.get("a0_pump_m3", 0.0),
                    "budget100_saving_pct": case.get("budget100_saving_pct", 0.0),
                    "delta_time_gt5_s": case.get("delta_time_gt5_s", 0.0),
                    "delta_fallback_time_s": case.get("delta_fallback_time_s", 0.0),
                }
            )

    case_gap = pd.DataFrame(rows)
    summary_rows = []
    covered = case_gap[case_gap["rows"].fillna(0) > 0]
    for regime, g in covered.groupby("fine_regime"):
        oracle_plateau = g["oracle_plateau_rows"].sum()
        learned_plateau = g["learned_plateau_rows"].sum()
        overlap = g["plateau_row_overlap"].sum()
        oracle_veto = g["oracle_veto_rows"].sum()
        learned_veto = g["learned_veto_rows"].sum()
        veto_overlap = g["veto_row_overlap"].sum()
        summary_rows.append(
            {
                "fine_regime": regime,
                "covered_cases": len(g),
                "rows": int(g["rows"].sum()),
                "plateau_precision": safe_div(overlap, learned_plateau),
                "plateau_recall": safe_div(overlap, oracle_plateau),
                "oracle_plateau_frac": safe_div(oracle_plateau, g["rows"].sum()),
                "learned_plateau_frac": safe_div(learned_plateau, g["rows"].sum()),
                "veto_precision": safe_div(veto_overlap, learned_veto),
                "veto_recall": safe_div(veto_overlap, oracle_veto),
                "oracle_veto_frac": safe_div(oracle_veto, g["rows"].sum()),
                "learned_veto_frac": safe_div(learned_veto, g["rows"].sum()),
                "mean_abs_b0_err": g["mean_abs_b0_err"].mean(),
                "mean_abs_b1_err": g["mean_abs_b1_err"].mean(),
                "mean_abs_b2_err": g["mean_abs_b2_err"].mean(),
            }
        )
    return case_gap, pd.DataFrame(summary_rows)


def summarize_residual_arms(case_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_path = ROOT / "residual_high_7200/raw_tables/candidate_matrix_case_deltas.csv"
    matrix = pd.read_csv(matrix_path)
    mapping = case_table[["case_id", "timestamp", "fine_regime"]].copy()
    mapping["case_id_norm"] = mapping["case_id"].map(normalize_core)
    matrix["case_id_norm"] = matrix["case_id"].map(normalize_core)
    matrix = matrix.merge(
        mapping[["case_id_norm", "timestamp", "fine_regime"]],
        on=["case_id_norm", "timestamp"],
        how="left",
    )
    matrix["fine_regime"] = matrix["fine_regime"].fillna("unmapped")
    rows = []
    for (regime, arm), g in matrix.groupby(["fine_regime", "arm"], dropna=False):
        baseline = g["pump_m3_baseline"].sum()
        saved = g["pump_saved_m3_vs_baseline"].sum()
        rows.append(
            {
                "fine_regime": regime,
                "arm": arm,
                "cases": len(g),
                "baseline_pump_m3": baseline,
                "arm_pump_m3": g["pump_m3"].sum(),
                "pump_saved_m3": saved,
                "pump_saving_pct": 100.0 * saved / baseline if baseline else 0.0,
                "delta_fallback_time_s": g["delta_fallback_time_s_vs_baseline"].sum(),
                "mean_delta_p95_deg": g["delta_p95_vs_baseline"].mean(),
                "top_case_saved_m3": g["pump_saved_m3_vs_baseline"].max(),
            }
        )
    arm_summary = pd.DataFrame(rows).sort_values(["fine_regime", "pump_saved_m3"], ascending=[True, False])

    reason_rows = []
    log_dir = ROOT / "residual_high_7200/residual_high_economy_v1/planner_logs"
    if log_dir.exists():
        for path in sorted(log_dir.glob("*.csv")):
            log = pd.read_csv(path)
            reason = log.get("economy_pump_budget_reason", pd.Series(dtype=object)).fillna("missing")
            counts = reason.value_counts()
            core = normalize_core(path.name)
            # Map by core substring, falling back to file core.
            fine = "unmapped"
            for _, r in case_table.iterrows():
                if normalize_core(r.get("case_id", "")) == core or normalize_core(r.get("case", "")) == core:
                    fine = r.get("fine_regime", "unmapped")
                    break
            for key, count in counts.items():
                reason_rows.append(
                    {
                        "case_core": core,
                        "fine_regime": fine,
                        "reason": key,
                        "rows": int(count),
                    }
                )
    reason_counts = pd.DataFrame(reason_rows)
    reason_summary = (
        reason_counts.groupby(["fine_regime", "reason"], dropna=False)["rows"].sum().reset_index()
        if not reason_counts.empty
        else pd.DataFrame(columns=["fine_regime", "reason", "rows"])
    )
    return matrix, arm_summary, reason_summary


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        else:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in out.columns) + " |")
    return "\n".join(lines)


def write_decision(
    arm_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    reason_summary: pd.DataFrame,
) -> None:
    plateau_arms = arm_summary[arm_summary["fine_regime"] == "residual_high_plateau_high_pump"]
    plateau_gap = gap_summary[gap_summary["fine_regime"] == "residual_high_plateau_high_pump"]
    boundary_gap = gap_summary[gap_summary["fine_regime"] == "direction_reversal_catchup_boundary"]

    lines = []
    lines.append("# Residual-High Plateau Gap Diagnostic\n")
    lines.append("## 结论\n")
    lines.append("`residual_high_plateau_high_pump` 是真实的第二机会域，但现在还不是成熟自动策略。\n")
    lines.append("- 宽松 `budget100` 在 plateau 子集上能明显省泵，并且没有表现成慢回落那种明显安全代价。\n")
    lines.append("- 当前 `residual_high_economy_v1` 没抓住这个机会，说明自动策略存在缺口。\n")
    lines.append("- 从已有 learned/oracle 对照日志看，诊断覆盖有限；在覆盖样本里需要看 learned 预测是否已经呈现 plateau。如果已经呈现，优先修规则；如果 learned 与 oracle 形态不一致，才考虑预测层二分类头。\n")
    lines.append("\n## Residual-high 子集各 arm 表现\n")
    lines.append(md_table(plateau_arms, digits=3))
    lines.append("\n## Learned forecast vs oracle/actual shape 诊断\n")
    lines.append(md_table(gap_summary, digits=3))
    lines.append("\n## residual_high_economy_v1 触发原因计数\n")
    lines.append(md_table(reason_summary, digits=0))
    lines.append("\n## 判断\n")
    if not plateau_gap.empty:
        rec = float(plateau_gap["plateau_recall"].iloc[0])
        prec = float(plateau_gap["plateau_precision"].iloc[0])
        if rec >= 0.50 and prec >= 0.50:
            lines.append("已有 learned 预测在覆盖样本里能较好识别 plateau 形态；下一步不应先训练模型，而应修 residual-high 的运行时规则/动作接口。\n")
        else:
            lines.append("已有 learned 预测对 plateau 的识别不够稳；如果要把 residual-high 做成成熟第二策略，需要考虑一个很小的 targeted binary head：plateau-vs-rising/reversal，而不是 9 类大分类器。\n")
    else:
        lines.append("现有 learned/oracle 对照日志未充分覆盖 plateau；下一步应先补一次固定 casebook 的 learned/oracle 形态审计，再决定是否训练预测头。\n")
    if not boundary_gap.empty:
        lines.append("boundary/reversal 类仍应作为 veto/release 边界，不应被包装成省泵目标。\n")
    lines.append("\n## 下一步\n")
    lines.append("1. 不训练 9-label regime classifier。\n")
    lines.append("2. 若 learned plateau 形态可分：优先修 residual-high detector/rule，把 budget100 的 plateau 行为转成可解释自动策略。\n")
    lines.append("3. 若 learned plateau 形态不可分：只考虑一个小的 plateau-vs-boundary 二分类预测头。\n")
    lines.append("4. residual-high slow-decay 继续作为边界/谨慎类，不和 plateau 混在一起。\n")
    (OUT_PAPER / "residual_plateau_gap_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    case_table = pd.read_csv(RAW / "regime_library_v2_case_table.csv")

    residual_matrix, arm_summary, reason_summary = summarize_residual_arms(case_table)
    gap_case, gap_summary = summarize_forecast_gap(case_table)

    residual_matrix.to_csv(OUT_RAW / "residual_high_candidate_matrix_with_fine_regime.csv", index=False)
    arm_summary.to_csv(OUT_RAW / "residual_high_arm_summary_by_fine_regime.csv", index=False)
    reason_summary.to_csv(OUT_RAW / "residual_high_economy_reason_summary.csv", index=False)
    gap_case.to_csv(OUT_RAW / "residual_plateau_forecast_gap_case_table.csv", index=False)
    gap_summary.to_csv(OUT_RAW / "residual_plateau_forecast_gap_summary.csv", index=False)
    write_decision(arm_summary, gap_summary, reason_summary)
    print(f"Wrote {OUT_RAW / 'residual_high_arm_summary_by_fine_regime.csv'}")
    print(f"Wrote {OUT_RAW / 'residual_plateau_forecast_gap_summary.csv'}")
    print(f"Wrote {OUT_PAPER / 'residual_plateau_gap_decision.md'}")


if __name__ == "__main__":
    main()
