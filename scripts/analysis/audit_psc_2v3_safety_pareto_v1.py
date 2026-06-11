#!/usr/bin/env python3
"""Audit 2号 vs 3号 PSC safe-saving Pareto evidence.

This script is read-only.  It separates three evidence types:

1. fixed structural selector results on the 96-case pool;
2. existing 24-case held-out classifier evidence;
3. score-level LOO/per-regime threshold audits using already exported HGB
   scores, without retraining models.

The LOO audit is a stress test of score/threshold selection, not a full model
retraining claim.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("outputs/wind_prediction/psc_selector_mixed_pool_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "psc_2v3_safety_pareto_audit_v1",
    )
    parser.add_argument("--disaster-time-gt5-s", type=float, default=60.0)
    parser.add_argument("--disaster-fallback-s", type=float, default=0.0)
    return parser.parse_args()


def _open_mask(df: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(df.get("use_economy", 0), errors="coerce").fillna(0).astype(int).eq(1)


def _series_or_default(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _case_disaster(df: pd.DataFrame, args: argparse.Namespace, prefix: str = "") -> pd.Series:
    fallback = pd.to_numeric(df[f"{prefix}d_fallback_vs_safety_s"], errors="coerce").fillna(0.0)
    time_gt5 = pd.to_numeric(df[f"{prefix}d_time_gt5_vs_safety_s"], errors="coerce").fillna(0.0)
    return fallback.gt(float(args.disaster_fallback_s)) | time_gt5.ge(float(args.disaster_time_gt5_s))


def _saving_pct(saving_m3: float, baseline_pump_m3: float) -> float:
    return 100.0 * float(saving_m3) / max(float(baseline_pump_m3), 1e-9)


def _summarize_selected(
    name: str,
    selected: pd.DataFrame,
    baseline_pump_m3: float,
    args: argparse.Namespace,
    note: str = "",
) -> dict[str, Any]:
    opened = selected[_open_mask(selected)].copy()
    if "selected_d_pump_m3" in selected.columns:
        d_pump = pd.to_numeric(selected["selected_d_pump_m3"], errors="coerce").fillna(0.0)
        d_gt5 = pd.to_numeric(selected["selected_d_time_gt5_s"], errors="coerce").fillna(0.0)
        d_gt6 = pd.to_numeric(selected["selected_d_time_gt6_s"], errors="coerce").fillna(0.0)
        d_fb = pd.to_numeric(selected["selected_d_fallback_s"], errors="coerce").fillna(0.0)
        opened_dp = pd.to_numeric(opened["selected_d_pump_m3"], errors="coerce").fillna(0.0)
    elif "d_pump_vs_safety_m3" in selected.columns:
        d_pump = pd.to_numeric(selected["d_pump_vs_safety_m3"], errors="coerce").fillna(0.0)
        d_gt5 = pd.to_numeric(selected["d_time_gt5_vs_safety_s"], errors="coerce").fillna(0.0)
        d_gt6 = pd.to_numeric(selected["d_time_gt6_vs_safety_s"], errors="coerce").fillna(0.0)
        d_fb = pd.to_numeric(selected["d_fallback_vs_safety_s"], errors="coerce").fillna(0.0)
        opened_dp = pd.to_numeric(opened["d_pump_vs_safety_m3"], errors="coerce").fillna(0.0)
    else:
        d_pump = pd.to_numeric(selected["d_pump_m3"], errors="coerce").fillna(0.0)
        d_gt5 = pd.to_numeric(selected["d_time_gt5_s"], errors="coerce").fillna(0.0)
        d_gt6 = pd.to_numeric(selected["d_time_gt6_s"], errors="coerce").fillna(0.0)
        d_fb = pd.to_numeric(selected["d_fallback_s"], errors="coerce").fillna(0.0)
        opened_dp = pd.to_numeric(opened["d_pump_m3"], errors="coerce").fillna(0.0)

    if opened.empty:
        opened_disaster = opened_risk = 0
        worst_gt5 = worst_fb = 0.0
        opened_case_ids = opened_disaster_ids = opened_risk_ids = ""
    else:
        opened_delta = opened.copy()
        if "selected_d_time_gt5_s" in opened_delta.columns:
            opened_delta["d_time_gt5_vs_safety_s"] = opened_delta["selected_d_time_gt5_s"]
            opened_delta["d_fallback_vs_safety_s"] = opened_delta["selected_d_fallback_s"]
        elif "d_time_gt5_vs_safety_s" not in opened_delta.columns:
            opened_delta["d_time_gt5_vs_safety_s"] = opened_delta["d_time_gt5_s"]
            opened_delta["d_fallback_vs_safety_s"] = opened_delta["d_fallback_s"]
        opened_disaster_mask = _case_disaster(opened_delta, args)
        opened_disaster = int(opened_disaster_mask.sum())
        opened_risk = int(_series_or_default(opened, "risk_marker", 0.0).astype(int).sum())
        worst_gt5 = float(pd.to_numeric(opened_delta["d_time_gt5_vs_safety_s"], errors="coerce").fillna(0.0).max())
        worst_fb = float(pd.to_numeric(opened_delta["d_fallback_vs_safety_s"], errors="coerce").fillna(0.0).max())
        opened_case_ids = ";".join(opened["case_id"].astype(str).tolist())
        opened_disaster_ids = ";".join(opened.loc[opened_disaster_mask, "case_id"].astype(str).tolist())
        risk_mask = _series_or_default(opened, "risk_marker", 0.0).astype(int).eq(1)
        opened_risk_ids = ";".join(opened.loc[risk_mask, "case_id"].astype(str).tolist())

    saving_m3 = -float(d_pump.sum())
    return {
        "selector": name,
        "cases": int(len(selected)),
        "economy_cases": int(len(opened)),
        "saving_m3": saving_m3,
        "saving_pct": _saving_pct(saving_m3, baseline_pump_m3),
        "d_time_gt5_s": float(d_gt5.sum()),
        "d_time_gt6_s": float(d_gt6.sum()),
        "d_fallback_s": float(d_fb.sum()),
        "risk_marker_opened": opened_risk,
        "disaster_opened": opened_disaster,
        "worst_opened_d_time_gt5_s": worst_gt5,
        "worst_opened_d_fallback_s": worst_fb,
        "opened_case_ids": opened_case_ids,
        "opened_risk_case_ids": opened_risk_ids,
        "opened_disaster_case_ids": opened_disaster_ids,
        "top_opened_pump_gain_share": float((-opened_dp).clip(lower=0).max() / max(float((-opened_dp).clip(lower=0).sum()), 1e-9))
        if not opened.empty
        else 0.0,
        "note": note,
    }


def _baseline_pump(diag: pd.DataFrame) -> float:
    return float(pd.to_numeric(diag["safety_pump_m3"], errors="coerce").sum())


def _structural_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = args.root
    diag = pd.read_csv(root / "psc_case_diagnostic_96case_pair_frozen_validation_v1" / "case_diagnostic_table.csv")
    baseline = _baseline_pump(diag)
    rows: list[dict[str, Any]] = []

    zero = diag.copy()
    zero["use_economy"] = 0
    zero["d_pump_vs_safety_m3"] = 0.0
    zero["d_time_gt5_vs_safety_s"] = 0.0
    zero["d_time_gt6_vs_safety_s"] = 0.0
    zero["d_fallback_vs_safety_s"] = 0.0
    rows.append(_summarize_selected("0号_safe_baseline_fixed96", zero, baseline, args, "0号 fixed 96-case reference"))

    one = diag.copy()
    one["use_economy"] = 1
    rows.append(_summarize_selected("1号_learned_rawenv_all_on_fixed96", one, baseline, args, "1号 all-on; unsafe upper pump-saving stress"))

    for name, path, note in [
        (
            "2号_current_only_structural_fixed96",
            root / "psc_structural_selector_96case_pair_frozen_current_only_v1" / "structural_selector_case_decisions.csv",
            "2号 fixed structural current-only selector",
        ),
        (
            "3号_v1_learned_structural_fixed96",
            root / "psc_structural_selector_96case_pair_frozen_learned_v1" / "structural_selector_case_decisions.csv",
            "3号 v1 fixed structural learned selector",
        ),
        (
            "2号_current_only_structural_v2_fixed96",
            root / "psc_structural_selector_96case_pair_3hao_v2_current_only_v1" / "structural_selector_case_decisions.csv",
            "2号 v2-parameter current-only structural selector",
        ),
        (
            "3号_v2_learned_structural_fixed96",
            root / "psc_structural_selector_96case_pair_3hao_v2_learned_v1" / "structural_selector_case_decisions.csv",
            "3号 v2 learned structural selector; current canonical 96-case headline",
        ),
    ]:
        selected = pd.read_csv(path)
        rows.append(_summarize_selected(name, selected, baseline, args, note))

    oracle = diag.copy()
    oracle["use_economy"] = pd.to_numeric(oracle["use_rawenv"], errors="coerce").fillna(0).astype(int)
    closed = oracle["use_economy"].astype(int).eq(0)
    for col in ["d_pump_m3", "d_time_gt5_s", "d_time_gt6_s", "d_fallback_s"]:
        oracle.loc[closed, col] = 0.0
    rows.append(_summarize_selected("oracle_replay_label_fixed96", oracle, baseline, args, "Replay-label upper bound for 0号/1号 switching"))
    return pd.DataFrame(rows), diag


def _heldout_classifier_rows(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = args.root
    diag = pd.read_csv(root / "psc_case_diagnostic_24case_6h_v1" / "case_diagnostic_table.csv")
    baseline = _baseline_pump(diag)
    specs = [
        (
            "2号_current_only_HGB_24case_holdout",
            root / "replay_selector_classifier_24case_6h_current_only_hgb_v1" / "psc_replay_selector_case_predictions.csv",
            "current_only_replay_selector",
            "Existing 24-case HGB holdout; current-only opens disaster",
        ),
        (
            "3号_learned_HGB_24case_holdout",
            root / "replay_selector_classifier_24case_6h_hgb_v1" / "psc_replay_selector_case_predictions.csv",
            "learned_replay_selector",
            "Existing 24-case HGB holdout; learned HGB blocks disaster but is conservative",
        ),
        (
            "3号_learned_HGB_safetyfirst_24case_holdout",
            root / "replay_selector_classifier_24case_6h_safetyfirst_v1" / "psc_replay_selector_case_predictions.csv",
            "learned_replay_selector",
            "Safety-first selected learned HGB holdout",
        ),
    ]
    rows: list[dict[str, Any]] = []
    case_rows: list[pd.DataFrame] = []
    for name, path, selector, note in specs:
        df = pd.read_csv(path)
        sub = df[df["selector"].astype(str).eq(selector) & df["split"].astype(str).eq("holdout")].copy()
        rows.append(_summarize_selected(name, sub, baseline, args, note))
        sub["audit_selector"] = name
        case_rows.append(sub)
    cases = pd.concat(case_rows, ignore_index=True, sort=False)
    cases["disaster_opened"] = (_open_mask(cases) & _case_disaster(cases, args)).astype(int)
    return pd.DataFrame(rows), cases


def _choose_best_threshold(
    train: pd.DataFrame,
    safe_col: str,
    save_col: str,
    hard_col: str,
) -> tuple[float, float]:
    if train.empty:
        return 1.01, 1.01
    safe_values = sorted(set(np.linspace(0.20, 0.95, 31).round(4)) | set(pd.to_numeric(train[safe_col], errors="coerce").dropna().round(4)))
    save_values = sorted(set(np.linspace(0.20, 0.95, 31).round(4)) | set(pd.to_numeric(train[save_col], errors="coerce").dropna().round(4)))
    safe_arr = pd.to_numeric(train[safe_col], errors="coerce").to_numpy(float)
    save_arr = pd.to_numeric(train[save_col], errors="coerce").to_numpy(float)
    hard_arr = pd.to_numeric(train[hard_col], errors="coerce").fillna(0).to_numpy(int) == 0
    d_pump = pd.to_numeric(train["d_pump_m3"], errors="coerce").fillna(0.0).to_numpy(float)
    d_gt5 = pd.to_numeric(train["d_time_gt5_s"], errors="coerce").fillna(0.0).to_numpy(float)
    d_fb = pd.to_numeric(train["d_fallback_s"], errors="coerce").fillna(0.0).to_numpy(float)

    best: tuple[float, float, float, int, float, float] | None = None
    for st in safe_values:
        safe_mask = safe_arr >= float(st)
        for pt in save_values:
            mask = safe_mask & (save_arr >= float(pt)) & hard_arr
            if float(d_gt5[mask].sum()) <= 0.0 and float(d_fb[mask].sum()) <= 0.0:
                score = (float(d_pump[mask].sum()), -int(mask.sum()), float(st), float(pt), float(d_gt5[mask].sum()), float(d_fb[mask].sum()))
                if best is None or score < best:
                    best = score
    if best is None:
        return 1.01, 1.01
    return float(best[2]), float(best[3])


def _score_level_audit_for_table(
    diag: pd.DataFrame,
    dataset_name: str,
    source_label: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = "learned_hgb" if source_label == "3号_learned_hgb_scores" else "current_hgb"
    safe_col = f"{prefix}_score_rawenv_safe"
    save_col = f"{prefix}_score_rawenv_pump_saves"
    hard_col = f"{prefix}_hard_block"
    needed = [safe_col, save_col, hard_col]
    data = diag[[c for c in diag.columns if c in set(diag.columns)]].copy()
    data = data[data[safe_col].notna() & data[save_col].notna()].copy()
    baseline = _baseline_pump(diag)
    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []

    folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for case_id in data["case_id"].astype(str):
        eval_df = data[data["case_id"].astype(str).eq(case_id)].copy()
        train_df = data[~data["case_id"].astype(str).eq(case_id)].copy()
        folds.append((f"loocv_{case_id}", train_df, eval_df))
    for regime, eval_df in data.groupby("mixed_regime", sort=False):
        train_df = data[~data["mixed_regime"].astype(str).eq(str(regime))].copy()
        folds.append((f"regime_holdout_{regime}", train_df, eval_df.copy()))

    for fold, train_df, eval_df in folds:
        st, pt = _choose_best_threshold(train_df, safe_col, save_col, hard_col)
        opened_mask = (
            pd.to_numeric(eval_df[safe_col], errors="coerce").ge(st)
            & pd.to_numeric(eval_df[save_col], errors="coerce").ge(pt)
            & pd.to_numeric(eval_df[hard_col], errors="coerce").fillna(0).astype(int).eq(0)
        )
        eval_selected = eval_df.copy()
        eval_selected["use_economy"] = opened_mask.astype(int)
        summary = _summarize_selected(
            f"{source_label}_{dataset_name}_{fold}",
            eval_selected,
            baseline,
            args,
            "Score-level threshold-selection audit; no model retraining",
        )
        summary.update(
            {
                "dataset": dataset_name,
                "source": source_label,
                "fold": fold,
                "threshold_safe": st,
                "threshold_save": pt,
                "eval_cases": int(len(eval_df)),
                "train_cases": int(len(train_df)),
            }
        )
        rows.append(summary)
        for row in eval_selected.itertuples(index=False):
            opened = bool(getattr(row, "use_economy"))
            disaster = opened and (
                float(getattr(row, "d_fallback_s")) > float(args.disaster_fallback_s)
                or float(getattr(row, "d_time_gt5_s")) >= float(args.disaster_time_gt5_s)
            )
            case_rows.append(
                {
                    "dataset": dataset_name,
                    "source": source_label,
                    "fold": fold,
                    "case_id": row.case_id,
                    "mixed_regime": row.mixed_regime,
                    "opened": int(opened),
                    "disaster_opened": int(disaster),
                    "threshold_safe": st,
                    "threshold_save": pt,
                    "d_pump_m3": float(getattr(row, "d_pump_m3")) if opened else 0.0,
                    "d_time_gt5_s": float(getattr(row, "d_time_gt5_s")) if opened else 0.0,
                    "d_time_gt6_s": float(getattr(row, "d_time_gt6_s")) if opened else 0.0,
                    "d_fallback_s": float(getattr(row, "d_fallback_s")) if opened else 0.0,
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(case_rows)


def _score_level_audits(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = [
        ("96case", args.root / "psc_case_diagnostic_96case_pair_frozen_validation_v1" / "case_diagnostic_table.csv"),
        ("24case_6h", args.root / "psc_case_diagnostic_24case_6h_v1" / "case_diagnostic_table.csv"),
    ]
    summaries: list[pd.DataFrame] = []
    cases: list[pd.DataFrame] = []
    for dataset_name, path in tables:
        diag = pd.read_csv(path)
        for source in ["2号_current_hgb_scores", "3号_learned_hgb_scores"]:
            s, c = _score_level_audit_for_table(diag, dataset_name, source, args)
            summaries.append(s)
            cases.append(c)
    return pd.concat(summaries, ignore_index=True, sort=False), pd.concat(cases, ignore_index=True, sort=False)


def _aggregate_score_level(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, source), sub in summary.groupby(["dataset", "source"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "source": source,
                "folds": int(len(sub)),
                "folds_with_disaster": int(pd.to_numeric(sub["disaster_opened"], errors="coerce").fillna(0).gt(0).sum()),
                "total_disaster_opened": int(pd.to_numeric(sub["disaster_opened"], errors="coerce").fillna(0).sum()),
                "mean_saving_pct": float(pd.to_numeric(sub["saving_pct"], errors="coerce").fillna(0).mean()),
                "total_eval_saving_m3": float(pd.to_numeric(sub["saving_m3"], errors="coerce").fillna(0).sum()),
                "worst_opened_d_time_gt5_s": float(pd.to_numeric(sub["worst_opened_d_time_gt5_s"], errors="coerce").fillna(0).max()),
                "worst_opened_d_fallback_s": float(pd.to_numeric(sub["worst_opened_d_fallback_s"], errors="coerce").fillna(0).max()),
                "disaster_case_ids": ";".join(
                    sorted(
                        {
                            cid
                            for ids in sub.loc[pd.to_numeric(sub["disaster_opened"], errors="coerce").fillna(0).gt(0), "opened_disaster_case_ids"].astype(str)
                            for cid in ids.split(";")
                            if cid
                        }
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for rec in view.to_dict("records"):
        vals = []
        for col in view.columns:
            val = rec[col]
            if isinstance(val, (float, np.floating)):
                vals.append("nan" if math.isnan(float(val)) else f"{float(val):.4g}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structural, _ = _structural_rows(args)
    heldout, heldout_cases = _heldout_classifier_rows(args)
    score_summary, score_cases = _score_level_audits(args)
    score_agg = _aggregate_score_level(score_summary)

    structural.to_csv(args.output_dir / "fixed96_safe_saving_pareto.csv", index=False)
    heldout.to_csv(args.output_dir / "existing24_hgb_holdout_pareto.csv", index=False)
    heldout_cases.to_csv(args.output_dir / "existing24_hgb_holdout_cases.csv", index=False)
    score_summary.to_csv(args.output_dir / "score_level_threshold_fold_summary.csv", index=False)
    score_cases.to_csv(args.output_dir / "score_level_threshold_case_decisions.csv", index=False)
    score_agg.to_csv(args.output_dir / "score_level_threshold_aggregate.csv", index=False)

    key_structural = structural[
        [
            "selector",
            "saving_pct",
            "saving_m3",
            "economy_cases",
            "risk_marker_opened",
            "disaster_opened",
            "worst_opened_d_time_gt5_s",
            "worst_opened_d_fallback_s",
            "note",
        ]
    ].copy()
    key_heldout = heldout[
        [
            "selector",
            "saving_pct",
            "saving_m3",
            "economy_cases",
            "disaster_opened",
            "worst_opened_d_time_gt5_s",
            "worst_opened_d_fallback_s",
            "opened_disaster_case_ids",
            "note",
        ]
    ].copy()
    key_score = score_agg[
        [
            "dataset",
            "source",
            "folds",
            "folds_with_disaster",
            "total_disaster_opened",
            "mean_saving_pct",
            "worst_opened_d_time_gt5_s",
            "worst_opened_d_fallback_s",
            "disaster_case_ids",
        ]
    ].copy()

    lines = [
        "# 2号 vs 3号 PSC Safety Pareto Audit",
        "",
        "This audit separates fixed structural selectors from black-box/score-level threshold selection.",
        "",
        "## Fixed 96-Case Structural Pareto",
        "",
        _md_table(key_structural),
        "",
        "Interpretation: fixed structural 3号 improves water saving over 2号 while both keep zero diagnostic disasters on the 96-case pool.",
        "",
        "## Existing 24-Case HGB Holdout Evidence",
        "",
        _md_table(key_heldout),
        "",
        "Interpretation: current-only HGB opens a held-out boundary disaster, while learned HGB blocks the same holdout set but is conservative.",
        "",
        "## Score-Level LOO/Per-Regime Threshold Audit",
        "",
        _md_table(key_score),
        "",
        "Important caveat: this is a score-level threshold-selection stress test using exported HGB scores; it does not retrain models per fold.",
        "",
        "## Decision",
        "",
        "- The strongest present claim is not naive black-box learned classification.",
        "- The current evidence supports fixed structural 3号 as the safe selector path.",
        "- 2号 must remain visible as the non-ML baseline.",
        "- Safety robustness should be claimed only where the audit actually shows lower disaster exposure.",
        "",
    ]
    (args.output_dir / "psc_2v3_safety_pareto_readout.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output_dir / "psc_2v3_safety_pareto_readout.md")
    print(key_structural.to_string(index=False))
    print(key_heldout.to_string(index=False))
    print(key_score.to_string(index=False))


if __name__ == "__main__":
    main()
