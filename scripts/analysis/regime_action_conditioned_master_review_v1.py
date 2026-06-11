#!/usr/bin/env python3
"""Master review for regime-conditioned pump-saving development.

This script is deliberately read-only.  It collects existing regime-mining,
controller-result, and separability-audit artifacts into one decision table.

The review follows an action-conditioned procedure:

1. Is there a material pump-saving outcome for a fixed action?
2. Is the safety/fallback cost acceptable for the intended claim?
3. Can the useful cases be detected from pre-action forecast/state features?
4. Is prevalence large enough to justify further mining or controller work?

It does not run simulations and it does not tune thresholds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
RAW = ROOT / "raw_tables"
V3 = ROOT / "regime_mining_v3" / "raw_tables"
OUT = ROOT / "regime_action_conditioned_master_review_v1"
OUT_RAW = OUT / "raw_tables"
OUT_PAPER = OUT / "paper_ready"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(prevalence: pd.DataFrame, regime: str) -> float:
    if prevalence.empty:
        return float("nan")
    rows = prevalence[prevalence["primary_regime"] == regime]
    if rows.empty:
        return 0.0
    return float(rows["row_pct"].iloc[0])


def sum_pct(prevalence: pd.DataFrame, regimes: list[str]) -> float:
    return sum(pct(prevalence, r) for r in regimes)


def get_metric(df: pd.DataFrame, col: str, default: float | str = ""):
    if df.empty or col not in df.columns:
        return default
    return df[col].iloc[0]


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)

    test_prev = read_csv(V3 / "regime_mining_v3_test_prevalence.csv")
    val_prev = read_csv(V3 / "regime_mining_v3_validation_prevalence.csv")
    lib = read_csv(RAW / "regime_library_v2_summary.csv")
    relief = read_csv(RAW / "relief_decay_episode_auto_v1_24case_compare.csv")
    outcome_auc = read_csv(ROOT / "outcome_labeled_regime_audit_v1" / "raw_tables" / "relief_decay_outcome_detector_feature_auc.csv")
    outcome_rules = read_csv(ROOT / "outcome_labeled_regime_audit_v1" / "raw_tables" / "relief_decay_outcome_detector_rule_scan.csv")
    gusty = read_csv(ROOT / "regime_mining_v3" / "raw_tables" / "gusty_oscillation_smoothing_v1_screen_summary.csv")
    lowrisk = read_csv(ROOT / "lowrisk_redundant_opportunity_audit_v1" / "raw_tables" / "lowrisk_redundant_summary.csv")
    plateau_loo = read_csv(ROOT / "residual_plateau_separability_study_v1" / "raw_tables" / "loo_summary.csv")

    relief_row = relief[relief.get("arm", pd.Series(dtype=str)) == "episode_auto_v1"]
    relief_auc = outcome_auc[
        (outcome_auc.get("label", pd.Series(dtype=str)) == "episode_auto_good")
        & (~outcome_auc.get("ceiling_feature", pd.Series(dtype=bool)).astype(bool))
    ].sort_values("auc", ascending=False)
    relief_rule = outcome_rules[outcome_rules.get("label", pd.Series(dtype=str)) == "episode_auto_good"].head(1)

    gusty_row = gusty[gusty.get("arm", pd.Series(dtype=str)) == "gusty_smoothing_v1"]
    gusty_budget = gusty[gusty.get("arm", pd.Series(dtype=str)) == "budget100"]
    lowrisk_mat = lowrisk[lowrisk.get("group", pd.Series(dtype=str)) == "lowrisk_with_material_a0_pump_ge100"]
    plateau_bal = plateau_loo[
        (plateau_loo.get("run", pd.Series(dtype=str)) == "learned_h120")
        & (plateau_loo.get("objective", pd.Series(dtype=str)) == "balanced")
        & (plateau_loo.get("allow_two_feature", pd.Series(dtype=int)) == 0)
    ]
    plateau_zero = plateau_loo[
        (plateau_loo.get("run", pd.Series(dtype=str)) == "learned_h120")
        & (plateau_loo.get("objective", pd.Series(dtype=str)) == "zero_boundary")
        & (plateau_loo.get("allow_two_feature", pd.Series(dtype=int)) == 0)
    ]

    def lib_regime(name: str) -> pd.DataFrame:
        return lib[lib.get("fine_regime", pd.Series(dtype=str)) == name]

    fast_soft = ["transient_peak_fast_decay", "transient_peak_soft_decay"]
    boundary = ["direction_reversal_boundary", "reintensification_boundary"]
    safety_noop = [
        "quiet_low_opportunity",
        "transient_decay_low_pump_proxy",
        "sustained_high_safety_event",
        "far_only_h120_advisory",
    ]

    records = [
        {
            "family": "transient_peak_future_decay",
            "role": "mature automatic",
            "test_prevalence_pct": sum_pct(test_prev, fast_soft),
            "validation_prevalence_pct": sum_pct(val_prev, fast_soft),
            "fixed_action": "episode_auto_v1",
            "pump_saving_pct": float(get_metric(relief_row, "pump_saving_vs_A0_pct", 18.125)),
            "delta_time_gt5_s": float(get_metric(relief_row, "delta_time_gt5_vs_A0_s", 485.0)),
            "delta_fallback_s": float(get_metric(relief_row, "delta_fallback_vs_A0_s", 362.0)),
            "detector_evidence": f"runtime AUC={float(get_metric(relief_auc, 'auc', 0.993)):.3f}; best simple rule precision={float(get_metric(relief_rule, 'precision', 1.0)):.3f}, recall={float(get_metric(relief_rule, 'recall', 1.0)):.3f}",
            "method_decision": "keep as main automatic regime",
            "next_action": "No more tuning. Use as template for action-conditioned regime proof; optional mixed-set abstention only if needed for final packaging.",
        },
        {
            "family": "residual_high_plateau",
            "role": "operator/Pareto candidate, not automatic",
            "test_prevalence_pct": pct(test_prev, "residual_high_plateau"),
            "validation_prevalence_pct": pct(val_prev, "residual_high_plateau"),
            "fixed_action": "budget100 / plateau specialists",
            "pump_saving_pct": float(get_metric(lib_regime("residual_high_plateau_high_pump"), "budget100_saving_pct", 36.85)),
            "delta_time_gt5_s": float(get_metric(lib_regime("residual_high_plateau_high_pump"), "delta_time_gt5_s", -689.0)),
            "delta_fallback_s": float(get_metric(lib_regime("residual_high_plateau_high_pump"), "delta_fallback_time_s", -298.0)),
            "detector_evidence": f"learned-h120 balanced boundary FPR={float(get_metric(plateau_bal, 'boundary_fpr', 0.364)):.3f}; zero-boundary recall={float(get_metric(plateau_zero, 'plateau_recall', 0.0)):.3f}",
            "method_decision": "freeze automatic control; recognition bottleneck",
            "next_action": "Do not tune another gate. Only revisit if more labels allow a targeted plateau-vs-boundary recognizer.",
        },
        {
            "family": "gusty_oscillation",
            "role": "tested candidate, no-go for current automatic action",
            "test_prevalence_pct": pct(test_prev, "gusty_oscillation_candidate"),
            "validation_prevalence_pct": pct(val_prev, "gusty_oscillation_candidate"),
            "fixed_action": "gusty_smoothing_v1",
            "pump_saving_pct": float(get_metric(gusty_row, "saving_pct_vs_a0", -0.0004)),
            "delta_time_gt5_s": float(get_metric(gusty_row, "delta_time_gt5_s", 0.0)),
            "delta_fallback_s": float(get_metric(gusty_row, "delta_fallback_s", 0.0)),
            "detector_evidence": f"budget100 pool={float(get_metric(gusty_budget, 'saving_pct_vs_a0', 17.68)):.2f}% but safety-costly; smoothing fired and saved ~0",
            "method_decision": "freeze as future/Pareto candidate",
            "next_action": "No further threshold tuning. A different action would need a new outcome-label pass before coding.",
        },
        {
            "family": "lowrisk_stable_redundant",
            "role": "large background, sparse material opportunity",
            "test_prevalence_pct": pct(test_prev, "lowrisk_stable_redundant_candidate"),
            "validation_prevalence_pct": pct(val_prev, "lowrisk_stable_redundant_candidate"),
            "fixed_action": "none mature",
            "pump_saving_pct": float(get_metric(lowrisk_mat, "budget100_saving_pct", 19.34)),
            "delta_time_gt5_s": float(get_metric(lowrisk_mat, "delta_time_gt5_s", 7.0)),
            "delta_fallback_s": float(get_metric(lowrisk_mat, "delta_fallback_time_s", 0.0)),
            "detector_evidence": f"material subset cases={int(get_metric(lowrisk_mat, 'cases', 6))}; top-case share={float(get_metric(lowrisk_mat, 'top_case_share', 0.771)):.3f}",
            "method_decision": "not a strategy yet; prevalence is misleading",
            "next_action": "Do not build a branch. Mine only if a pre-action material-pump detector is found; otherwise baseline/advisory.",
        },
        {
            "family": "direction_reversal_or_reintensification",
            "role": "boundary / veto",
            "test_prevalence_pct": sum_pct(test_prev, boundary),
            "validation_prevalence_pct": sum_pct(val_prev, boundary),
            "fixed_action": "release / abstain",
            "pump_saving_pct": float(get_metric(lib_regime("direction_reversal_catchup_boundary"), "budget100_saving_pct", 4.21)),
            "delta_time_gt5_s": float(get_metric(lib_regime("direction_reversal_catchup_boundary"), "delta_time_gt5_s", -391.0)),
            "delta_fallback_s": float(get_metric(lib_regime("direction_reversal_catchup_boundary"), "delta_fallback_time_s", 495.0)),
            "detector_evidence": "Repeatedly appears as catch-up/fallback twin of savable regimes.",
            "method_decision": "veto, not saving target",
            "next_action": "Keep as negative control for any future detector; do not design pump-saving action here.",
        },
        {
            "family": "neutral_untyped",
            "role": "only remaining broad blind spot",
            "test_prevalence_pct": pct(test_prev, "neutral_untyped"),
            "validation_prevalence_pct": pct(val_prev, "neutral_untyped"),
            "fixed_action": "none",
            "pump_saving_pct": "",
            "delta_time_gt5_s": "",
            "delta_fallback_s": "",
            "detector_evidence": "High prevalence but no fixed action/outcome label yet.",
            "method_decision": "possible future mining, but only action-conditioned",
            "next_action": "If more progress is required, do one read-only pump-opportunity + action-ceiling screen here. Do not create a named regime by shape alone.",
        },
        {
            "family": "safety_or_no_opportunity",
            "role": "baseline / advisory / hard-floor owned",
            "test_prevalence_pct": sum_pct(test_prev, safety_noop),
            "validation_prevalence_pct": sum_pct(val_prev, safety_noop),
            "fixed_action": "v1.6 baseline / h120 advisory",
            "pump_saving_pct": "",
            "delta_time_gt5_s": "",
            "delta_fallback_s": "",
            "detector_evidence": "Either no material pump or hard safety controller owns the state.",
            "method_decision": "do not optimize for pump saving",
            "next_action": "Keep hard floor and advisory layers; no economy specialist.",
        },
    ]
    table = pd.DataFrame(records)
    table.to_csv(OUT_RAW / "regime_action_conditioned_master_review.csv", index=False)

    go = table[table["method_decision"].str.contains("main automatic", case=False, na=False)]
    freeze = table[table["method_decision"].str.contains("freeze|veto|not a strategy|do not", case=False, na=False)]

    def md_table(df: pd.DataFrame) -> str:
        cols = [
            "family",
            "role",
            "test_prevalence_pct",
            "validation_prevalence_pct",
            "fixed_action",
            "pump_saving_pct",
            "detector_evidence",
            "method_decision",
        ]
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df[cols].iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append(f"{val:.2f}")
                else:
                    vals.append(str(val))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    md = [
        "# Regime Action-Conditioned Master Review v1",
        "",
        "This is a read-only consolidation of existing evidence. It does not run new simulations or tune thresholds.",
        "",
        "## Method",
        "",
        "A regime is considered actionable only when a fixed action has an outcome label (pump saving with acceptable cost) and that outcome is separable from bad/neutral cases using pre-action forecast/state features.",
        "",
        "## Master Table",
        "",
        md_table(table),
        "",
        "## Decision",
        "",
        "- Keep `transient_peak_future_decay` as the mature automatic regime.",
        "- Do not continue controller tuning for plateau, gusty, or lowrisk under the current evidence.",
        "- Do not start a broad search for more hand-named regimes.",
        "- If additional progress is required, inspect only the remaining `neutral_untyped` pool with an action-conditioned outcome screen: first find material pump opportunity and a fixed candidate action, then test detector separability.",
        "",
        "## Practical Next Step",
        "",
        "The highest-value next step is not another simulation. It is a read-only neutral-pool opportunity screen that asks whether `neutral_untyped` contains a material pump-saving action class. If it does not, the taxonomy should freeze.",
        "",
    ]
    text = "\n".join(md)
    (OUT / "decision.md").write_text(text, encoding="utf-8")
    (OUT_PAPER / "regime_action_conditioned_master_review_summary.md").write_text(text, encoding="utf-8")

    print(table.to_string(index=False))
    print(f"\nWrote {OUT / 'decision.md'}")


if __name__ == "__main__":
    main()
