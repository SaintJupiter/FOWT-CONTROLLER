#!/usr/bin/env python3
"""Build a 24-case PSC selector diagnostic table.

This is a read-only Stage-0 audit for the PSC 6h selector work.  It joins
replay-grounded labels, rawenv-vs-safety deltas, learned PSC features,
current-only proxy features, and selector decisions so structural gates can be
designed from visible case signatures instead of black-box coverage chasing.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


DEFAULT_ROOT = Path("outputs/wind_prediction/psc_selector_mixed_pool_v1")
DEFAULT_LADDER = DEFAULT_ROOT / "degradation_ladder_24case_6h_v1"
DEFAULT_LEARNED = DEFAULT_ROOT / "replay_selector_classifier_24case_6h_hgb_v1"
DEFAULT_CURRENT = DEFAULT_ROOT / "replay_selector_classifier_24case_6h_current_only_hgb_v1"


FEATURE_COLUMNS = [
    "signal_rows",
    "confidence_mean",
    "confidence_max",
    "prob_stable_mean",
    "prob_transient_decay_mean",
    "prob_reversal_signflip_mean",
    "prob_reintensification_mean",
    "prob_sustained_high_mean",
    "prob_ramp_onset_mean",
    "event_reversal_signflip_max",
    "event_reintensification_max",
    "event_attention_any_max",
    "time_to_attention_min_min",
    "opportunity_mean",
    "opportunity_max",
    "risk_max",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_LADDER / "degradation_ladder_case_metrics.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_ROOT / "replay_selector_labels_24case_6h_v1.csv")
    parser.add_argument("--learned-dir", type=Path, default=DEFAULT_LEARNED)
    parser.add_argument("--current-dir", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROOT / "psc_case_diagnostic_24case_6h_v1",
    )
    parser.add_argument("--safety-arm", default="current_forecast_adaptive")
    parser.add_argument("--economy-arm", default="learned_rawenv_mainline")
    parser.add_argument(
        "--risk-time-gt5-s",
        type=float,
        default=60.0,
        help="Diagnostic high-posture risk marker. Reported as a marker, not an immutable safety law.",
    )
    parser.add_argument(
        "--risk-fallback-s",
        type=float,
        default=0.0,
        help="Diagnostic fallback risk marker. Reported as a marker, not an immutable safety law.",
    )
    return parser.parse_args()


def _extract_mixed_regime(label: str) -> str:
    marker = "mixed_regime="
    text = str(label)
    if marker not in text:
        return "unknown"
    return text.split(marker, 1)[1].split("|", 1)[0].strip()


def _selector_rows(path: Path, preferred_selector: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if preferred_selector not in set(df["selector"].astype(str)):
        selectors = sorted(set(df["selector"].astype(str)))
        raise ValueError(f"{preferred_selector!r} missing in {path}; found {selectors}")
    keep = df[df["selector"].eq(preferred_selector)].copy()
    cols = [
        "case_id",
        "score_use_rawenv",
        "score_rawenv_safe",
        "score_rawenv_pump_saves",
        "hard_block",
        "use_economy",
        "selected_arm",
        "d_pump_vs_safety_m3",
        "d_time_gt5_vs_safety_s",
        "d_time_gt6_vs_safety_s",
        "d_fallback_vs_safety_s",
    ]
    return keep[[c for c in cols if c in keep.columns]].copy()


def _feature_rows(path: Path, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["case_id", "regime_argmax_mode", *FEATURE_COLUMNS]
    out = df[[c for c in cols if c in df.columns]].copy()
    return out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "case_id"})


def _wide_metrics(metrics: pd.DataFrame, safety_arm: str, economy_arm: str) -> pd.DataFrame:
    safety = metrics[metrics["arm"].eq(safety_arm)].set_index("case_id")
    economy = metrics[metrics["arm"].eq(economy_arm)].set_index("case_id")
    common = sorted(set(safety.index) & set(economy.index))
    rows: list[dict[str, Any]] = []
    for case_id in common:
        s = safety.loc[case_id]
        e = economy.loc[case_id]
        row = {
            "case_id": case_id,
            "label": str(s.get("label", e.get("label", ""))),
            "mixed_regime": _extract_mixed_regime(str(s.get("label", e.get("label", "")))),
            "safety_pump_m3": float(s["pump_m3"]),
            "economy_pump_m3": float(e["pump_m3"]),
            "d_pump_m3": float(e["pump_m3"] - s["pump_m3"]),
            "safety_time_gt5_s": float(s["time_gt5_s"]),
            "economy_time_gt5_s": float(e["time_gt5_s"]),
            "d_time_gt5_s": float(e["time_gt5_s"] - s["time_gt5_s"]),
            "safety_time_gt6_s": float(s["time_gt6_s"]),
            "economy_time_gt6_s": float(e["time_gt6_s"]),
            "d_time_gt6_s": float(e["time_gt6_s"] - s["time_gt6_s"]),
            "safety_fallback_s": float(s["fallback_s"]),
            "economy_fallback_s": float(e["fallback_s"]),
            "d_fallback_s": float(e["fallback_s"] - s["fallback_s"]),
            "safety_p95_axis_deg": float(s.get("p95_axis_deg", np.nan)),
            "economy_p95_axis_deg": float(e.get("p95_axis_deg", np.nan)),
            "d_p95_axis_deg": float(e.get("p95_axis_deg", np.nan) - s.get("p95_axis_deg", np.nan)),
            "safety_max_axis_deg": float(s.get("max_axis_deg", np.nan)),
            "economy_max_axis_deg": float(e.get("max_axis_deg", np.nan)),
            "d_max_axis_deg": float(e.get("max_axis_deg", np.nan) - s.get("max_axis_deg", np.nan)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_value(y: pd.Series, score: pd.Series, metric: str) -> float:
    mask = score.notna()
    yy = y[mask].astype(int)
    ss = score[mask].astype(float)
    if len(yy) == 0 or yy.nunique() < 2:
        return float("nan")
    try:
        if metric == "auc":
            return float(roc_auc_score(yy, ss))
        if metric == "ap":
            return float(average_precision_score(yy, ss))
    except ValueError:
        return float("nan")
    raise ValueError(metric)


def _candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    prefixes = ("learned_", "current_")
    score_cols = [
        "learned_score_use_rawenv",
        "learned_score_rawenv_safe",
        "learned_score_rawenv_pump_saves",
        "current_score_use_rawenv",
        "current_score_rawenv_safe",
        "current_score_rawenv_pump_saves",
    ]
    cols = []
    for col in df.columns:
        if col in score_cols:
            cols.append(col)
        elif col.startswith(prefixes) and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return sorted(set(cols))


def _threshold_rules(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    y_block = df["risk_marker"].astype(bool)
    y_positive = df["use_rawenv"].astype(bool)
    for col in feature_cols:
        values = df[col].replace([np.inf, -np.inf], np.nan).dropna().unique()
        if len(values) == 0:
            continue
        for direction in ("ge", "le"):
            for threshold in sorted(values):
                if direction == "ge":
                    pred_block = df[col].astype(float).ge(float(threshold))
                else:
                    pred_block = df[col].astype(float).le(float(threshold))
                tp = int((pred_block & y_block).sum())
                fp = int((pred_block & ~y_block).sum())
                fn = int((~pred_block & y_block).sum())
                tn = int((~pred_block & ~y_block).sum())
                if tp == 0:
                    continue
                positive_kept = int((~pred_block & y_positive).sum())
                positive_blocked = int((pred_block & y_positive).sum())
                safe_positive_kept = int((~pred_block & y_positive & ~y_block).sum())
                rows.append(
                    {
                        "feature": col,
                        "direction": direction,
                        "threshold": float(threshold),
                        "risk_recall": tp / max(tp + fn, 1),
                        "risk_precision": tp / max(tp + fp, 1),
                        "risk_tp": tp,
                        "risk_fp": fp,
                        "risk_fn": fn,
                        "risk_tn": tn,
                        "use_rawenv_kept": positive_kept,
                        "use_rawenv_blocked": positive_blocked,
                        "safe_use_rawenv_kept": safe_positive_kept,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["risk_recall", "risk_precision", "safe_use_rawenv_kept", "risk_fp"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
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


def _write_readout(out_dir: Path, table: pd.DataFrame, feature_scores: pd.DataFrame, rules: pd.DataFrame) -> None:
    must = table[table["risk_marker"].eq(1)].copy()
    positives = table[table["use_rawenv"].eq(1)].copy()
    blocked_positives = positives[positives["learned_hgb_use_economy"].eq(0)].copy()
    learned_open = table[table["learned_hgb_use_economy"].eq(1)].copy()
    current_open = table[table["current_hgb_use_economy"].eq(1)].copy()

    lines = [
        "# PSC 24-case 6h diagnostic readout",
        "",
        "This readout uses `risk_marker` as a diagnostic severe-risk flag, not as a mechanical scientific law.",
        "",
        "## Counts",
        "",
        f"- cases: {len(table)}",
        f"- use_rawenv positives: {int(table['use_rawenv'].sum())}",
        f"- rawenv_safe: {int(table['rawenv_safe'].sum())}",
        f"- rawenv_pump_saves: {int(table['rawenv_pump_saves'].sum())}",
        f"- risk_marker cases: {int(table['risk_marker'].sum())}",
        f"- learned HGB economy cases: {int(table['learned_hgb_use_economy'].sum())}",
        f"- current-only HGB economy cases: {int(table['current_hgb_use_economy'].sum())}",
        "",
        "## Risk Marker Cases",
        "",
        _md_table(
            must[
                [
                    "case_id",
                    "mixed_regime",
                    "use_rawenv",
                    "d_pump_m3",
                    "d_time_gt5_s",
                    "d_time_gt6_s",
                    "d_fallback_s",
                    "risk_reason",
                    "learned_hgb_use_economy",
                    "current_hgb_use_economy",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Positives Blocked By Learned HGB",
        "",
        _md_table(
            blocked_positives[
                [
                    "case_id",
                    "mixed_regime",
                    "d_pump_m3",
                    "d_time_gt5_s",
                    "d_time_gt6_s",
                    "d_fallback_s",
                    "learned_hgb_hard_block",
                    "learned_regime_argmax_mode",
                    "learned_event_attention_any_max",
                    "learned_event_reintensification_max",
                    "learned_prob_ramp_onset_mean",
                    "learned_prob_sustained_high_mean",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Learned HGB Opened Cases",
        "",
        _md_table(
            learned_open[
                [
                    "case_id",
                    "mixed_regime",
                    "use_rawenv",
                    "d_pump_m3",
                    "d_time_gt5_s",
                    "d_time_gt6_s",
                    "d_fallback_s",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Current-only HGB Opened Cases",
        "",
        _md_table(
            current_open[
                [
                    "case_id",
                    "mixed_regime",
                    "use_rawenv",
                    "d_pump_m3",
                    "d_time_gt5_s",
                    "d_time_gt6_s",
                    "d_fallback_s",
                ]
            ],
            max_rows=40,
        ),
        "",
        "## Best Single-Feature Risk Scores",
        "",
        _md_table(feature_scores.head(25), max_rows=25),
        "",
        "## Best Single-Threshold Risk Rules",
        "",
        _md_table(rules.head(25), max_rows=25),
        "",
        "## Regime Aggregate",
        "",
        _md_table(
            table.groupby("mixed_regime", dropna=False)
            .agg(
                cases=("case_id", "size"),
                use_rawenv=("use_rawenv", "sum"),
                risk_marker=("risk_marker", "sum"),
                d_pump_m3=("d_pump_m3", "sum"),
                d_time_gt5_s=("d_time_gt5_s", "sum"),
                d_time_gt6_s=("d_time_gt6_s", "sum"),
                d_fallback_s=("d_fallback_s", "sum"),
            )
            .reset_index(),
            max_rows=20,
        ),
        "",
    ]
    (out_dir / "diagnostic_readout.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.metrics)
    labels = pd.read_csv(args.labels)
    wide = _wide_metrics(metrics, args.safety_arm, args.economy_arm)
    label_keep = ["case_id", "rawenv_safe", "rawenv_pump_saves", "use_rawenv"]
    labels = labels[[c for c in label_keep if c in labels.columns]].copy()
    table = wide.merge(labels, on="case_id", how="left")

    learned_features = _feature_rows(args.learned_dir / "psc_replay_selector_dataset.csv", "learned")
    current_features = _feature_rows(args.current_dir / "psc_replay_selector_dataset.csv", "current")
    table = table.merge(learned_features, on="case_id", how="left").merge(current_features, on="case_id", how="left")

    learned_selector = _selector_rows(args.learned_dir / "psc_replay_selector_case_predictions.csv", "learned_replay_selector")
    learned_selector = learned_selector.rename(
        columns={c: f"learned_hgb_{c}" for c in learned_selector.columns if c != "case_id"}
    )
    current_selector = _selector_rows(args.current_dir / "psc_replay_selector_case_predictions.csv", "current_only_replay_selector")
    current_selector = current_selector.rename(
        columns={c: f"current_hgb_{c}" for c in current_selector.columns if c != "case_id"}
    )
    table = table.merge(learned_selector, on="case_id", how="left").merge(current_selector, on="case_id", how="left")

    risk_reasons = []
    fallback_delta = pd.to_numeric(table["d_fallback_s"], errors="coerce").fillna(0.0)
    time_gt5_delta = pd.to_numeric(table["d_time_gt5_s"], errors="coerce").fillna(0.0)
    for fallback_s, time_gt5_s in zip(fallback_delta, time_gt5_delta):
        reasons = []
        if float(fallback_s) > float(args.risk_fallback_s):
            reasons.append("fallback_delta_positive")
        if float(time_gt5_s) >= float(args.risk_time_gt5_s):
            reasons.append(f"d_time_gt5_ge_{int(args.risk_time_gt5_s)}s")
        risk_reasons.append(";".join(reasons))
    table["risk_reason"] = risk_reasons
    table["risk_marker"] = table["risk_reason"].astype(str).ne("").astype(int)
    table["risk_severity_score"] = (
        table["d_fallback_s"].clip(lower=0).astype(float) * 10.0
        + table["d_time_gt5_s"].clip(lower=0).astype(float)
        + 0.25 * table["d_time_gt6_s"].clip(lower=0).astype(float)
        + table["d_pump_m3"].clip(lower=0).astype(float) * 0.1
    )
    table["pump_gain_m3"] = (-table["d_pump_m3"]).clip(lower=0)

    feature_cols = _candidate_feature_columns(table)
    score_rows = []
    for col in feature_cols:
        score = pd.to_numeric(table[col], errors="coerce")
        auc_risk = _metric_value(table["risk_marker"], score, "auc")
        ap_risk = _metric_value(table["risk_marker"], score, "ap")
        auc_use = _metric_value(table["use_rawenv"], score, "auc")
        ap_use = _metric_value(table["use_rawenv"], score, "ap")
        score_rows.append(
            {
                "feature": col,
                "risk_auc": auc_risk,
                "risk_ap": ap_risk,
                "use_rawenv_auc": auc_use,
                "use_rawenv_ap": ap_use,
                "mean_risk": float(score[table["risk_marker"].eq(1)].mean(skipna=True)),
                "mean_use_rawenv": float(score[table["use_rawenv"].eq(1)].mean(skipna=True)),
                "mean_other": float(score[(table["risk_marker"].eq(0)) & (table["use_rawenv"].eq(0))].mean(skipna=True)),
            }
        )
    feature_scores = pd.DataFrame(score_rows)
    if not feature_scores.empty:
        feature_scores["risk_auc_or_inverse"] = (feature_scores["risk_auc"] - 0.5).abs() + 0.5
        feature_scores = feature_scores.sort_values(
            ["risk_auc_or_inverse", "risk_ap", "use_rawenv_ap"],
            ascending=[False, False, False],
        ).drop(columns=["risk_auc_or_inverse"])

    rules = _threshold_rules(table, feature_cols)

    preferred_cols = [
        "case_id",
        "mixed_regime",
        "label",
        "rawenv_safe",
        "rawenv_pump_saves",
        "use_rawenv",
        "risk_marker",
        "risk_reason",
        "risk_severity_score",
        "pump_gain_m3",
        "d_pump_m3",
        "d_time_gt5_s",
        "d_time_gt6_s",
        "d_fallback_s",
        "d_p95_axis_deg",
        "d_max_axis_deg",
        "learned_hgb_use_economy",
        "learned_hgb_hard_block",
        "current_hgb_use_economy",
        "current_hgb_hard_block",
        "learned_regime_argmax_mode",
        "current_regime_argmax_mode",
    ]
    remaining = [c for c in table.columns if c not in preferred_cols]
    table = table[[c for c in preferred_cols if c in table.columns] + remaining]

    table.to_csv(args.output_dir / "case_diagnostic_table.csv", index=False)
    feature_scores.to_csv(args.output_dir / "feature_separation_scores.csv", index=False)
    rules.to_csv(args.output_dir / "single_feature_threshold_rules.csv", index=False)
    _write_readout(args.output_dir, table, feature_scores, rules)

    print(args.output_dir / "case_diagnostic_table.csv")
    print(args.output_dir / "diagnostic_readout.md")
    print(
        table[
            [
                "case_id",
                "mixed_regime",
                "use_rawenv",
                "risk_marker",
                "d_pump_m3",
                "d_time_gt5_s",
                "d_time_gt6_s",
                "d_fallback_s",
                "learned_hgb_use_economy",
                "current_hgb_use_economy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
