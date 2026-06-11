#!/usr/bin/env python3
"""Outcome-labeled detector audit for the mature relief/decay action.

This is a read-only methodological audit.  It separates:

* outcome labels: did a fixed action actually save pump without safety cost?
* detector features: could runtime-available forecast/state features identify
  those good cases before action?

The purpose is to reduce circularity in the regime taxonomy.  It does not tune
the controller or run new simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import math

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
RAW = ROOT / "raw_tables"
OUT = ROOT / "outcome_labeled_regime_audit_v1"
OUT_RAW = OUT / "raw_tables"
OUT_PAPER = OUT / "paper_ready"

PER_CASE = RAW / "relief_decay_episode_auto_v1_24case_per_case.csv"
VIRTUAL_FEATURES = RAW / "relief_decay_virtual_episode_selector_features.csv"
CONTINUITY = RAW / "relief_decay_episode_continuity_diagnostic.csv"


FEATURES = [
    # Runtime learned forecast summary at t0 / early episode.
    "t0_early_max",
    "t0_drop",
    "t0_rise",
    "t0_dir",
    "t0_candidate",
    "first30_max_early",
    "first30_max_drop",
    "first30_min_dir",
    "first30_any_candidate",
    # Learned forecast trajectory over the 7200s episode diagnostic.
    "learn_mean_drop_pred",
    "learn_max_pred_early",
    "learn_candidate_rows",
    "learn_latch_rows",
    "learn_refresh_rows",
    "learn_event_reset_rows",
    # Actual/oracle future shape; these are marked as ceiling features in output.
    "early_max",
    "early_rise",
    "drop_mean",
    "actual_peak_to_late_drop_ms",
    "actual_high_ge16_total_s",
]

CEILING_FEATURES = {
    "early_max",
    "early_rise",
    "drop_mean",
    "actual_peak_to_late_drop_ms",
    "actual_high_ge16_total_s",
}


@dataclass(frozen=True)
class Rule:
    feature_a: str
    direction_a: str
    threshold_a: float
    feature_b: str | None = None
    direction_b: str | None = None
    threshold_b: float | None = None

    def text(self) -> str:
        first = f"{self.feature_a} {'>=' if self.direction_a == 'ge' else '<='} {self.threshold_a:.4g}"
        if self.feature_b is None:
            return first
        second = f"{self.feature_b} {'>=' if self.direction_b == 'ge' else '<='} {self.threshold_b:.4g}"
        return f"{first} AND {second}"

    def apply(self, df: pd.DataFrame) -> pd.Series:
        mask = _cond(df[self.feature_a], self.direction_a, self.threshold_a)
        if self.feature_b is not None:
            mask = mask & _cond(df[self.feature_b], self.direction_b or "ge", float(self.threshold_b))
        return mask.fillna(False)


def _cond(series: pd.Series, direction: str, threshold: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values >= threshold if direction == "ge" else values <= threshold


def ensure_dirs() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)


def auc_score(values: pd.Series, labels: pd.Series) -> tuple[float, str]:
    data = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "y": labels.astype(int)}).dropna()
    pos = data[data.y == 1].v.tolist()
    neg = data[data.y == 0].v.tolist()
    if not pos or not neg:
        return math.nan, "ge"
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
            total += 1
    auc_ge = wins / total if total else math.nan
    if auc_ge >= 0.5:
        return auc_ge, "ge"
    return 1.0 - auc_ge, "le"


def score_mask(mask: pd.Series, labels: pd.Series) -> dict[str, float]:
    pred = mask.astype(bool)
    y = labels.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    bal_acc = 0.5 * (recall + (tn / (tn + fp) if tn + fp else 0.0))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "balanced_accuracy": bal_acc,
    }


def thresholds(series: pd.Series) -> list[float]:
    vals = sorted(set(float(v) for v in pd.to_numeric(series, errors="coerce").dropna().tolist()))
    if not vals:
        return []
    mids = [(a + b) / 2.0 for a, b in zip(vals[:-1], vals[1:])]
    return sorted(set(vals + mids))


def best_rules(df: pd.DataFrame, label_col: str, features: list[str], max_pairs: int = 60) -> pd.DataFrame:
    labels = df[label_col].astype(int)
    candidates: list[dict[str, object]] = []
    top_features = features[:8]
    all_rules: list[Rule] = []
    for f in top_features:
        for direction in ("ge", "le"):
            for t in thresholds(df[f]):
                all_rules.append(Rule(f, direction, t))
    # Keep pair scan small: pair only strong single-feature candidates.
    pair_features = top_features[:6]
    pair_count = 0
    for fa, fb in combinations(pair_features, 2):
        for da in ("ge", "le"):
            for db in ("ge", "le"):
                for ta in thresholds(df[fa]):
                    for tb in thresholds(df[fb]):
                        all_rules.append(Rule(fa, da, ta, fb, db, tb))
                        pair_count += 1
                        if pair_count >= max_pairs * 200:
                            break
                    if pair_count >= max_pairs * 200:
                        break
                if pair_count >= max_pairs * 200:
                    break
            if pair_count >= max_pairs * 200:
                break
        if pair_count >= max_pairs * 200:
            break
    for rule in all_rules:
        stats = score_mask(rule.apply(df), labels)
        if stats["tp"] == 0:
            continue
        # Conservative objective: prefer precision, then recall, then balanced accuracy.
        objective = 3.0 * stats["precision"] + stats["recall"] + stats["balanced_accuracy"] - 0.2 * stats["fpr"]
        candidates.append({"label": label_col, "rule": rule.text(), "objective": objective, **stats})
    return pd.DataFrame(candidates).sort_values(
        ["precision", "recall", "balanced_accuracy", "objective"],
        ascending=[False, False, False, False],
    )


def load_table() -> pd.DataFrame:
    per = pd.read_csv(PER_CASE)
    virt = pd.read_csv(VIRTUAL_FEATURES)
    cont = pd.read_csv(CONTINUITY)
    df = per.merge(virt, on="case_id", how="left").merge(cont, on="case_id", how="left", suffixes=("", "_cont"))

    df["budget100_good"] = (
        (df["budget100_reference_pump_delta_vs_A0_m3"] >= 20.0)
        & (df["budget100_reference_pump_saving_pct"] >= 5.0)
        & (df["budget100_reference_time_gt5_delta_s"] <= 120.0)
        & (df["budget100_reference_fallback_delta_s"] <= 120.0)
    ).astype(int)
    df["episode_auto_good"] = (
        (df["episode_auto_v1_pump_delta_vs_A0_m3"] >= 20.0)
        & (df["episode_auto_v1_pump_saving_pct"] >= 5.0)
        & (df["episode_auto_v1_time_gt5_delta_s"] <= 120.0)
        & (df["episode_auto_v1_fallback_delta_s"] <= 120.0)
    ).astype(int)
    df["budget100_bad_or_neutral"] = 1 - df["budget100_good"]
    df["episode_auto_bad_or_neutral"] = 1 - df["episode_auto_good"]
    return df


def write_md(case_table: pd.DataFrame, auc_table: pd.DataFrame, rule_table: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame, cols: list[str], n: int | None = None) -> str:
        sub = df[cols].head(n) if n else df[cols]
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in sub.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                vals.append(f"{float(val):.3f}" if isinstance(val, (float, np.floating)) else str(val))
            rows.append("| " + " | ".join(vals) + " |")
        return "\n".join(rows)

    budget_good = int(case_table["budget100_good"].sum())
    episode_good = int(case_table["episode_auto_good"].sum())
    total = int(len(case_table))
    best_learned = auc_table[
        (auc_table["label"] == "episode_auto_good") & (~auc_table["ceiling_feature"])
    ].head(1)
    best_rule = rule_table[rule_table["label"] == "episode_auto_good"].head(1)
    learned_auc = float(best_learned["auc"].iloc[0]) if not best_learned.empty else math.nan
    rule_precision = float(best_rule["precision"].iloc[0]) if not best_rule.empty else math.nan
    rule_recall = float(best_rule["recall"].iloc[0]) if not best_rule.empty else math.nan
    budget_auc = auc_table[auc_table["label"] == "budget100_good"].head(10)
    episode_auc = auc_table[auc_table["label"] == "episode_auto_good"].head(10)
    budget_rules = rule_table[rule_table["label"] == "budget100_good"].head(8)
    episode_rules = rule_table[rule_table["label"] == "episode_auto_good"].head(8)

    lines = [
        "# Relief/Decay Outcome-Labeled Detector Audit",
        "",
        "This read-only audit separates the causal/outcome label from the runtime detector features.",
        "It uses the existing matched 24-case relief/decay results; no new controller tuning or simulation is introduced.",
        "",
        "## Outcome Labels",
        "",
        f"- `budget100_good`: {budget_good}/{total} cases meet pump-saving and safety-cost criteria.",
        f"- `episode_auto_good`: {episode_good}/{total} cases meet pump-saving and safety-cost criteria.",
        "",
        "A good case requires pump saving >=20 m3, saving >=5%, time>5 delta <=120s, and fallback delta <=120s.",
        "",
        "## Detector Separability",
        "",
        f"Best learned/runtime feature AUC for `episode_auto_good`: {learned_auc:.3f}.",
        f"Best simple rule for `episode_auto_good`: precision={rule_precision:.3f}, recall={rule_recall:.3f}.",
        "",
        "## Top Feature AUCs: Budget100 Opportunity",
        "",
        md_table(budget_auc, ["label", "feature", "auc", "direction", "ceiling_feature"]),
        "",
        "## Top Feature AUCs: Episode Auto",
        "",
        md_table(episode_auc, ["label", "feature", "auc", "direction", "ceiling_feature"]),
        "",
        "## Top Simple Rules: Budget100 Opportunity",
        "",
        md_table(budget_rules, ["label", "rule", "precision", "recall", "fpr", "balanced_accuracy"]),
        "",
        "## Top Simple Rules: Episode Auto",
        "",
        md_table(episode_rules, ["label", "rule", "precision", "recall", "fpr", "balanced_accuracy"]),
        "",
        "## Interpretation",
        "",
        "This is not a full representative prevalence proof.  It is a bounded methodological check on the already-matched 24-case relief/decay evidence.",
        "The key use is reviewer-facing: the mature regime is now described as an action-conditioned outcome label that is tested for pre-action separability, rather than only a hand-named wind-shape category.",
        "",
    ]
    (OUT / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_PAPER / "relief_decay_outcome_detector_audit_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    df = load_table()
    usable_features = [f for f in FEATURES if f in df.columns and pd.to_numeric(df[f], errors="coerce").notna().sum() >= 4]

    auc_rows: list[dict[str, object]] = []
    for label in ("budget100_good", "episode_auto_good"):
        labels = df[label].astype(int)
        for feature in usable_features:
            auc, direction = auc_score(df[feature], labels)
            if math.isnan(auc):
                continue
            auc_rows.append(
                {
                    "label": label,
                    "feature": feature,
                    "auc": auc,
                    "direction": direction,
                    "ceiling_feature": feature in CEILING_FEATURES,
                }
            )
    auc_table = pd.DataFrame(auc_rows).sort_values(["label", "auc"], ascending=[True, False])
    # Feed strongest non-ceiling runtime features into simple-rule scanner.
    runtime_features = (
        auc_table[~auc_table["ceiling_feature"]]
        .sort_values(["label", "auc"], ascending=[True, False])["feature"]
        .drop_duplicates()
        .tolist()
    )
    rule_tables = []
    for label in ("budget100_good", "episode_auto_good"):
        feature_order = (
            auc_table[(auc_table["label"] == label) & (~auc_table["ceiling_feature"])]
            .sort_values("auc", ascending=False)["feature"]
            .tolist()
        )
        rule_tables.append(best_rules(df, label, feature_order))
    rule_table = pd.concat(rule_tables, ignore_index=True)
    case_cols = [
        "case_id",
        "A0_primary_pump_work_m3",
        "budget100_reference_pump_delta_vs_A0_m3",
        "budget100_reference_pump_saving_pct",
        "budget100_reference_time_gt5_delta_s",
        "budget100_reference_fallback_delta_s",
        "episode_auto_v1_pump_delta_vs_A0_m3",
        "episode_auto_v1_pump_saving_pct",
        "episode_auto_v1_time_gt5_delta_s",
        "episode_auto_v1_fallback_delta_s",
        "budget100_good",
        "episode_auto_good",
    ] + usable_features
    case_table = df[case_cols].copy()
    case_table.to_csv(OUT_RAW / "relief_decay_outcome_detector_case_table.csv", index=False)
    auc_table.to_csv(OUT_RAW / "relief_decay_outcome_detector_feature_auc.csv", index=False)
    rule_table.to_csv(OUT_RAW / "relief_decay_outcome_detector_rule_scan.csv", index=False)
    write_md(case_table, auc_table, rule_table)
    print("cases", len(case_table), "features", len(usable_features))
    print(auc_table.head(10).to_string(index=False))
    print(rule_table.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
