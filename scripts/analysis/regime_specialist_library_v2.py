#!/usr/bin/env python3
"""Build a finer regime library and strategy map for pump-saving specialists.

The v1 taxonomy intentionally stayed conservative.  This pass separates more
forecast/state shapes so controller work can move from "one good relief regime"
to a typed library: mature specialists, candidates, veto/boundary regimes, and
no-action regimes.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
RAW = ROOT / "raw_tables"
PAPER = ROOT / "paper_ready"
CASEBOOKS = RAW / "casebooks_v2"


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    value = row.get(col, default)
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(row: pd.Series, col: str) -> bool:
    value = row.get(col, False)
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _text(row: pd.Series, col: str) -> str:
    value = row.get(col, "")
    if pd.isna(value):
        return ""
    return str(value).lower()


def classify_v2(row: pd.Series) -> tuple[str, str, str, str]:
    """Return fine_regime, role, policy, reason."""

    case = _text(row, "case")
    label = _text(row, "label")
    coarse = _text(row, "auto_regime")
    a0_pump = _num(row, "a0_pump_m3")
    saving_pct = _num(row, "budget100_saving_pct")
    max_peak = _num(row, "max_near_peak")
    near_drop = _num(row, "max_near_drop")
    far_drop = _num(row, "max_far_relief_drop")
    near_high_rows = _num(row, "near_high_rows")
    high_then_relief_rows = _num(row, "high_then_relief_rows")
    rise_then_fall_rows = _num(row, "rise_then_fall_rows")
    reversal_rows = _num(row, "reversal_or_signflip_rows")
    time_gt5 = _num(row, "time_gt5_s_a0")
    p95 = _num(row, "p95_max_axis_deg_a0")
    lowrisk = _bool(row, "is_lowrisk_label") or "lowrisk" in case or "lowrisk" in label

    relief_decay = (
        _bool(row, "forecast_relief_decay_candidate")
        or _bool(row, "fast_callback_candidate")
        or "relief_decay" in case
        or high_then_relief_rows > 0
        or rise_then_fall_rows > 0
        or ("transient_peak_future_decay" in coarse)
    )
    reversal = (
        _bool(row, "direction_reversal_candidate")
        or reversal_rows > 0
        or "signflip" in case
        or "_sf_" in case
        or "signflip" in label
    )
    reintensify = "reintens" in label or "reintens" in case
    far_only = (
        _bool(row, "h120_supervisory_relief_candidate")
        or far_drop >= 0.35
    ) and not relief_decay

    if reversal:
        return (
            "direction_reversal_catchup_boundary",
            "veto_boundary",
            "Release/veto economy holds; keep v1.6 floor. Do not use as pump-saving target.",
            "direction/sign flip can make a hold turn into later catch-up pump.",
        )
    if reintensify:
        return (
            "reintensification_after_relief_boundary",
            "veto_boundary",
            "Release/veto economy holds; keep v1.6 floor.",
            "forecast shape is not a clean decay; load returns after apparent relief.",
        )
    if time_gt5 >= 600.0 or p95 >= 5.25:
        return (
            "hard_floor_recovery_dominated",
            "safety_backbone",
            "v1.6 hard floor/recovery only; no economy relaxation.",
            "case already lives near/above hard floor, so saving actions risk safety margin.",
        )
    if relief_decay and a0_pump >= 150.0:
        if high_then_relief_rows > 0 or near_drop >= 0.30 or "relief_decay_exp" in case:
            return (
                "transient_peak_fast_decay_high_pump",
                "mature_automatic",
                "relief_decay_smart_release_v1",
                "forecast sees pump-worthy peak followed by clear 0-60min decay; avoid chasing transient peak.",
            )
        return (
            "transient_peak_soft_decay_high_pump",
            "mature_or_near_mature",
            "relief_decay_smart_release_v1 with conservative release guards",
            "decay signal exists but is softer; use the same specialist with stricter release.",
        )
    if relief_decay:
        return (
            "transient_decay_low_pump_opportunity",
            "low_value_supporting",
            "Log/advisory or very light economy hold; not a headline strategy.",
            "forecast shape is favorable but baseline pump is too small to save much.",
        )
    if max_peak >= 1.05 and near_high_rows >= 6 and a0_pump >= 150.0:
        if near_drop < 0.08 and far_drop < 0.12:
            return (
                "residual_high_plateau_high_pump",
                "candidate_not_mature",
                "Optional budget100 Pareto mode or future residual-high specialist; not global automatic yet.",
                "large pump pool exists, but simple automatic residual-high policy did not reproduce budget100 saving.",
            )
        return (
            "residual_high_slow_decay_or_moderate_relief",
            "candidate_not_mature",
            "Residual-high economy hold with early release; needs richer detector before promotion.",
            "some relief exists but not the clean peak-then-decay shape.",
        )
    if lowrisk or (max_peak < 0.80 and a0_pump >= 50.0):
        if saving_pct >= 20.0:
            return (
                "lowrisk_redundant_refill_high_saving_sparse",
                "candidate_sparse",
                "Very conservative redundant-refill suppression / refresh cooldown; needs more cases.",
                "low external load but controller still pumps; evidence is sparse and case-concentrated.",
            )
        return (
            "lowrisk_small_redundant_pump",
            "low_value_supporting",
            "Keep baseline or log as possible redundant-pump audit.",
            "low-risk shape exists but current saving is small.",
        )
    if far_only:
        return (
            "h120_far_relief_advisory_only",
            "advisory_only",
            "60-120min supervisory relief advisory; do not drive pump directly.",
            "far-horizon relief exists without a near executable pump-saving window.",
        )
    if a0_pump < 75.0:
        return (
            "low_pump_no_material_opportunity",
            "no_action",
            "Keep v1.6 baseline.",
            "baseline pump is already small, so there is little to save.",
        )
    return (
        "neutral_untyped_pump_opportunity",
        "needs_more_evidence",
        "No specialist yet; mine more features before controller changes.",
        "case has some pump but no clean deployable forecast/state shape.",
    )


def strategy_matrix() -> pd.DataFrame:
    rows = [
        {
            "fine_regime": "transient_peak_fast_decay_high_pump",
            "runtime_detector": "learned 0-60min forecast: high/rising near pressure followed by clear decay; no reversal/re-intensification/floor risk.",
            "strategy": "relief_decay_smart_release_v1",
            "status": "mature automatic",
            "controller_action": "Open economy saving mode: delay/hold non-urgent target pursuit, release on floor risk or decay failure.",
        },
        {
            "fine_regime": "transient_peak_soft_decay_high_pump",
            "runtime_detector": "same as above but weaker decay magnitude or fewer high-then-relief rows.",
            "strategy": "same specialist, conservative release",
            "status": "near mature",
            "controller_action": "Use smart release only if release/veto guards are clean; otherwise baseline.",
        },
        {
            "fine_regime": "transient_decay_low_pump_opportunity",
            "runtime_detector": "decay is visible, but baseline pump is small.",
            "strategy": "advisory/light hold only",
            "status": "supporting",
            "controller_action": "Do not spend controller complexity here; can log as forecast-correct low-opportunity case.",
        },
        {
            "fine_regime": "residual_high_plateau_high_pump",
            "runtime_detector": "near pressure high for many buckets, little decay, no reversal.",
            "strategy": "future residual-high specialist or optional budget100 Pareto mode",
            "status": "candidate not mature",
            "controller_action": "Do not auto-enable globally; if used, present as operator-selected Pareto mode with safety-margin cost.",
        },
        {
            "fine_regime": "residual_high_slow_decay_or_moderate_relief",
            "runtime_detector": "high load with some slow relief but not clean transient decay.",
            "strategy": "candidate early-release economy hold",
            "status": "candidate not mature",
            "controller_action": "Needs new detector; current residual_high_economy_v1 was too conservative.",
        },
        {
            "fine_regime": "lowrisk_redundant_refill_high_saving_sparse",
            "runtime_detector": "low forecast pressure, low floor risk, baseline still pumps.",
            "strategy": "conservative redundant-refill suppression / refresh cooldown",
            "status": "sparse candidate",
            "controller_action": "Mine more examples before automatic deployment; current evidence concentrated.",
        },
        {
            "fine_regime": "direction_reversal_catchup_boundary",
            "runtime_detector": "signflip, direction reversal, or axis reversal signature.",
            "strategy": "veto/release",
            "status": "negative boundary",
            "controller_action": "Do not hold; release economy saving mode because catch-up pump risk is high.",
        },
        {
            "fine_regime": "reintensification_after_relief_boundary",
            "runtime_detector": "forecast relief followed by renewed high pressure.",
            "strategy": "veto/release",
            "status": "negative boundary",
            "controller_action": "Do not hold through the second event; keep v1.6 recovery ready.",
        },
        {
            "fine_regime": "hard_floor_recovery_dominated",
            "runtime_detector": "time>5 or p95 already dominated by hard-floor domain.",
            "strategy": "v1.6 hard floor",
            "status": "safety only",
            "controller_action": "No economy strategy; safety controller owns this regime.",
        },
        {
            "fine_regime": "h120_far_relief_advisory_only",
            "runtime_detector": "60-120min far relief without near executable opportunity.",
            "strategy": "supervisory advisory",
            "status": "advisory only",
            "controller_action": "Output advisory/context; no direct pump action.",
        },
        {
            "fine_regime": "low_pump_no_material_opportunity",
            "runtime_detector": "baseline pump already low.",
            "strategy": "baseline",
            "status": "no action",
            "controller_action": "No pump-saving specialist.",
        },
        {
            "fine_regime": "neutral_untyped_pump_opportunity",
            "runtime_detector": "pump exists but no clean shape yet.",
            "strategy": "feature mining first",
            "status": "needs evidence",
            "controller_action": "Do not add gates until a deployable shape is found.",
        },
    ]
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, digits: int = 2) -> str:
    if df.empty:
        return "_No rows._"
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}")
        else:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(x))
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in out.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)

    src = RAW / "regime_taxonomy_case_table.csv"
    df = pd.read_csv(src)
    labels = df.apply(classify_v2, axis=1, result_type="expand")
    labels.columns = ["fine_regime", "strategy_role", "specialist_policy", "classification_reason"]
    df = pd.concat([df, labels], axis=1)

    rows = []
    for regime, g in df.groupby("fine_regime", dropna=False):
        a0 = g["a0_pump_m3"].sum()
        saved = g["budget100_saving_m3"].sum()
        rows.append(
            {
                "fine_regime": regime,
                "cases": len(g),
                "a0_pump_m3": a0,
                "budget100_saved_m3": saved,
                "budget100_saving_pct": 100.0 * saved / a0 if a0 else 0.0,
                "delta_time_gt5_s": g["delta_time_gt5_s"].sum(),
                "delta_idle_gt5_s": g["delta_idle_gt5_s"].sum(),
                "delta_fallback_time_s": g["delta_fallback_time_s"].sum(),
                "mean_delta_p95_deg": g["delta_p95_max_axis_deg"].mean(),
                "top_case_share_of_saved": g["budget100_saving_m3"].max() / saved if saved > 0 else 0.0,
                "strategy_role": g["strategy_role"].mode().iat[0],
                "specialist_policy": g["specialist_policy"].mode().iat[0],
            }
        )
    summary = pd.DataFrame(rows).sort_values(["budget100_saved_m3", "cases"], ascending=[False, False])
    matrix = strategy_matrix()

    case_cols = [
        "case",
        "case_id",
        "timestamp",
        "label",
        "auto_regime",
        "fine_regime",
        "strategy_role",
        "specialist_policy",
        "classification_reason",
        "a0_pump_m3",
        "budget100_saved_m3",
        "budget100_saving_pct",
        "delta_time_gt5_s",
        "delta_idle_gt5_s",
        "delta_fallback_time_s",
        "delta_p95_max_axis_deg",
        "max_near_peak",
        "max_near_drop",
        "max_far_relief_drop",
        "near_high_rows",
        "high_then_relief_rows",
        "rise_then_fall_rows",
        "reversal_or_signflip_rows",
    ]
    df[[c for c in case_cols if c in df.columns]].to_csv(RAW / "regime_library_v2_case_table.csv", index=False)
    summary.to_csv(RAW / "regime_library_v2_summary.csv", index=False)
    matrix.to_csv(RAW / "regime_specialist_strategy_matrix_v2.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    for regime, g in df.groupby("fine_regime"):
        cols = [c for c in ["case_id", "timestamp", "label"] if c in g.columns]
        if len(cols) < 3:
            continue
        cb = g[cols].copy()
        cb["label"] = cb["label"].astype(str) + f" | fine_regime={regime}"
        cb.to_csv(CASEBOOKS / f"{regime}_cases.csv", index=False)

    lines = []
    lines.append("# Regime Specialist Library v2\n")
    lines.append("这版把工况拆得更细：不是只找一个回落工况，而是把每类可运行时识别的风/压力形态都绑定到策略状态。\n")
    lines.append("## 核心结论\n")
    lines.append("- 已成熟自动策略仍然只有 `transient_peak_fast_decay_high_pump / transient_peak_soft_decay_high_pump`，对应 `relief_decay_smart_release_v1`。这是当前可以直接自动识别并起作用的主线。\n")
    lines.append("- `residual_high_*` 是最大潜力候选，但当前自动策略没有成熟；宽松 budget100 能省很多，但还不能说成稳健自动控制。\n")
    lines.append("- `lowrisk_*` 有省泵例子，但样本稀疏且贡献集中，适合作为补充机会域，不适合现在立刻写成主策略。\n")
    lines.append("- `direction_reversal_*` 和 `reintensification_*` 是边界/反例，作用是 veto/release，不是省泵目标。\n")
    lines.append("\n## 工况汇总\n")
    lines.append(md_table(summary, digits=3))
    lines.append("\n## 每类工况对应策略\n")
    lines.append(md_table(matrix, digits=3))
    lines.append("\n## 下一步优先级\n")
    lines.append("1. 论文主线：锁定回落类自动策略，做图和表，说明它如何识别临时峰值并避免追峰值泵水。\n")
    lines.append("2. 如果继续开发：优先攻 `residual_high_plateau_high_pump`，但要承认它目前不是自动成熟策略，需要新检测器，而不是简单调门限。\n")
    lines.append("3. `lowrisk_redundant_refill` 先继续找样本；样本足够前不要写成自动控制主线。\n")
    lines.append("4. 反转/再增强类作为边界图：解释为什么有些工况不能开省泵模式。\n")
    (PAPER / "regime_specialist_library_v2.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {RAW / 'regime_library_v2_case_table.csv'}")
    print(f"Wrote {RAW / 'regime_library_v2_summary.csv'}")
    print(f"Wrote {PAPER / 'regime_specialist_library_v2.md'}")


if __name__ == "__main__":
    main()
