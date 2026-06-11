#!/usr/bin/env python3
"""Read-only separability study for residual-high plateau vs boundary cases.

The controller experiments showed a clear pattern:

* Loose plateau recognition catches some true plateau pump-saving cases, but also
  admits direction-reversal/catch-up boundary cases.
* Strict recognition avoids the boundary, but captures almost no plateau pump.

This script asks the prior question before any new controller work:

Can plateau-good cases be separated from boundary-bad cases by a simple,
runtime-interpretable 0-60 / 60-120 forecast-shape rule?

It deliberately does not tune a controller.  It scans simple one- and two-feature
rules offline, evaluates leave-one-case-out (LOO) stability, and writes a
decision summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import math

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
IN_H120 = (
    ROOT
    / "residual_high_plateau_oracle_vs_learned_h120_recognition_v1"
    / "raw_tables"
    / "oracle_vs_learned_h120_case_features.csv"
)
IN_NEAR = (
    ROOT
    / "residual_high_plateau_boundary_recognition_v1"
    / "raw_tables"
    / "case_features_with_delta.csv"
)

OUT = ROOT / "residual_plateau_separability_study_v1"
OUT_RAW = OUT / "raw_tables"
OUT_PAPER = OUT / "paper_ready"


EXCLUDE_SUBSTRINGS = (
    "case_id",
    "short_case_id",
    "group",
    "run",
    "label",
    "y_plateau",
    "delta",
    "saving",
    "pump",
    "fallback",
    "p95",
    "b2_",
    "a0_",
    "candidate_rows",
    "active_rows",
    "hold_rows",
    "plateau_detected_rows",
    "far_available_rows",
    "forced_release_rows",
)


PRIORITY_FEATURES = [
    # Far-horizon / h120 shape.
    "first_far_min",
    "first_far_max",
    "first_far_over_near",
    "mean_far_over_near",
    "min_far_over_near",
    "max_far_over_near",
    "first_far_dir_shift_deg",
    "max_far_dir_shift_deg",
    "first_far_dot_60_120",
    "min_far_dot_60_120",
    "first_far_reversal",
    "any_far_reversal",
    "first_far_direction_shift",
    "any_far_direction_shift",
    "first_far_hidden_intensification",
    "any_far_hidden_intensification",
    "first_far_boundary_any",
    "any_far_boundary_any",
    # Near-horizon shape and state-shape context.
    "first_near_max",
    "first_near_last",
    "first_near_min",
    "first_near_range",
    "first_near_slope02",
    "first_near_drop02",
    "first_cos_02",
    "first_pressure_signflip_pitch_02",
    "first_pressure_signflip_roll_02",
    "first_pressure_dom_change_02",
    "first_posture_max_axis",
    "first_axis_balance_pressure0",
    "first_axis_balance_posture",
    "mean_near_range",
    "max_near_range",
    "mean_near_slope02",
    "max_near_slope02",
    "mean_cos_02",
    "min_cos_02",
    "mean_axis_balance_pressure0",
]


@dataclass(frozen=True)
class Condition:
    feature: str
    direction: str
    threshold: float

    def apply(self, df: pd.DataFrame) -> pd.Series:
        value = pd.to_numeric(df[self.feature], errors="coerce")
        if self.direction == "ge":
            return value >= self.threshold
        return value <= self.threshold

    def text(self) -> str:
        op = ">=" if self.direction == "ge" else "<="
        return f"{self.feature} {op} {self.threshold:.6g}"


@dataclass(frozen=True)
class Rule:
    conditions: tuple[Condition, ...]
    objective: str

    def apply(self, df: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            return pd.Series(False, index=df.index)
        mask = pd.Series(True, index=df.index)
        for condition in self.conditions:
            mask = mask & condition.apply(df)
        return mask.fillna(False)

    def text(self) -> str:
        if not self.conditions:
            return "NO_RULE"
        return " AND ".join(condition.text() for condition in self.conditions)


def ensure_dirs() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)


def normalize_near_table(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["case_id", "short_case_id"]
    for col in df.columns:
        if col in keep:
            continue
        if any(part in col for part in ("delta", "saving", "pump", "fallback", "p95")):
            continue
        if col in PRIORITY_FEATURES:
            keep.append(col)
    out = df[keep].copy()
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def load_dataset() -> pd.DataFrame:
    h120 = pd.read_csv(IN_H120)
    near = normalize_near_table(pd.read_csv(IN_NEAR))
    df = h120.merge(near, on=["case_id", "short_case_id"], how="left", suffixes=("", "_near"))
    df["y_plateau"] = pd.to_numeric(df["label_plateau"], errors="coerce").fillna(0).astype(int)
    for col in df.columns:
        if col in ("run", "case_id", "short_case_id", "group"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    return df


def predictor_features(df: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    for col in PRIORITY_FEATURES:
        if col in df.columns:
            candidates.append(col)
    for col in df.columns:
        if col == "y_plateau":
            continue
        if col in candidates:
            continue
        if any(part in col for part in EXCLUDE_SUBSTRINGS):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 4 and values.nunique(dropna=True) > 1:
            candidates.append(col)
    # Keep the scan interpretable and small: priority first, then numeric extras.
    return candidates


def auc_plateau_high(values: pd.Series, labels: pd.Series) -> float:
    data = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "y": labels}).dropna()
    pos = data[data["y"] == 1]["v"].tolist()
    neg = data[data["y"] == 0]["v"].tolist()
    if not pos or not neg:
        return math.nan
    wins = 0.0
    total = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
            total += 1
    return wins / total if total else math.nan


def midpoints(values: Iterable[float]) -> list[float]:
    vals = sorted({float(v) for v in values if pd.notna(v)})
    if not vals:
        return []
    thresholds = set(vals)
    for a, b in zip(vals[:-1], vals[1:]):
        thresholds.add((a + b) / 2.0)
    return sorted(thresholds)


def score_mask(mask: pd.Series, labels: pd.Series) -> dict[str, float]:
    pred = mask.astype(bool)
    y = labels.astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    pos = tp + fn
    neg = tn + fp
    recall = tp / pos if pos else 0.0
    fpr = fp / neg if neg else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    bal_acc = 0.5 * (recall + (tn / neg if neg else 0.0))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "plateau_kept": tp,
        "boundary_kept": fp,
        "plateau_total": pos,
        "boundary_total": neg,
        "precision": precision,
        "plateau_recall": recall,
        "boundary_fpr": fpr,
        "balanced_accuracy": bal_acc,
    }


def single_condition_scan(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    labels = df["y_plateau"]
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() < 4 or values.nunique(dropna=True) <= 1:
            continue
        for direction in ("ge", "le"):
            for threshold in midpoints(values):
                condition = Condition(feature, direction, threshold)
                stats = score_mask(condition.apply(df), labels)
                rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold": threshold,
                        "rule": condition.text(),
                        **stats,
                    }
                )
    return pd.DataFrame(rows)


def feature_auc_table(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    labels = df["y_plateau"]
    for feature in features:
        values = pd.to_numeric(df[feature], errors="coerce")
        if values.notna().sum() < 4 or values.nunique(dropna=True) <= 1:
            continue
        auc = auc_plateau_high(values, labels)
        rows.append(
            {
                "feature": feature,
                "auc_plateau_high": auc,
                "best_auc": max(auc, 1.0 - auc) if not math.isnan(auc) else math.nan,
                "preferred_direction": "high_for_plateau" if auc >= 0.5 else "low_for_plateau",
                "plateau_mean": values[labels == 1].mean(),
                "boundary_mean": values[labels == 0].mean(),
                "plateau_min": values[labels == 1].min(),
                "plateau_max": values[labels == 1].max(),
                "boundary_min": values[labels == 0].min(),
                "boundary_max": values[labels == 0].max(),
            }
        )
    return pd.DataFrame(rows).sort_values("best_auc", ascending=False)


def make_condition(row: pd.Series) -> Condition:
    return Condition(str(row["feature"]), str(row["direction"]), float(row["threshold"]))


def two_feature_scan(df: pd.DataFrame, single: pd.DataFrame, max_conditions: int = 28) -> pd.DataFrame:
    labels = df["y_plateau"]
    if single.empty:
        return pd.DataFrame()
    pool = single[
        (single["plateau_kept"] > 0)
        & (
            (single["boundary_kept"] <= 2)
            | (single["balanced_accuracy"] >= 0.65)
            | (single["plateau_recall"] >= 0.5)
        )
    ].copy()
    pool = pool.sort_values(
        ["boundary_kept", "balanced_accuracy", "plateau_kept"],
        ascending=[True, False, False],
    ).head(max_conditions)
    rows = []
    for _, a in pool.iterrows():
        for _, b in pool.iterrows():
            if str(a["feature"]) >= str(b["feature"]):
                continue
            rule = Rule((make_condition(a), make_condition(b)), objective="scan")
            stats = score_mask(rule.apply(df), labels)
            if stats["plateau_kept"] == 0:
                continue
            rows.append({"rule": rule.text(), **stats})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["boundary_kept", "balanced_accuracy", "plateau_kept"],
        ascending=[True, False, False],
    )


def choose_rule(train: pd.DataFrame, features: list[str], objective: str, max_two_feature: bool) -> Rule:
    single = single_condition_scan(train, features)
    candidate_rows = []
    if not single.empty:
        candidate_rows.extend(("single", row) for _, row in single.iterrows())
    if max_two_feature and not single.empty:
        two = two_feature_scan(train, single)
        if not two.empty:
            candidate_rows.extend(("two", row) for _, row in two.iterrows())
    if not candidate_rows:
        return Rule((), objective=objective)

    best_rule = Rule((), objective=objective)
    best_key = None
    for kind, row in candidate_rows:
        if kind == "single":
            rule = Rule((make_condition(row),), objective=objective)
            stats = score_mask(rule.apply(train), train["y_plateau"])
        else:
            conditions = []
            for part in str(row["rule"]).split(" AND "):
                tokens = part.split()
                if len(tokens) != 3:
                    continue
                conditions.append(Condition(tokens[0], "ge" if tokens[1] == ">=" else "le", float(tokens[2])))
            rule = Rule(tuple(conditions), objective=objective)
            stats = score_mask(rule.apply(train), train["y_plateau"])
        if objective == "zero_boundary":
            if stats["boundary_kept"] != 0 or stats["plateau_kept"] <= 0:
                continue
            key = (stats["plateau_kept"], stats["balanced_accuracy"], -len(rule.conditions))
        else:
            # Prefer balanced accuracy, then lower boundary false positive rate, then recall.
            key = (
                stats["balanced_accuracy"],
                -stats["boundary_fpr"],
                stats["plateau_recall"],
                -len(rule.conditions),
            )
        if best_key is None or key > best_key:
            best_key = key
            best_rule = rule
    return best_rule


def loo_eval(df: pd.DataFrame, features: list[str], run_name: str, objective: str, max_two_feature: bool) -> pd.DataFrame:
    rows = []
    for idx, held in df.iterrows():
        train = df.drop(index=idx).copy()
        test = df.loc[[idx]].copy()
        rule = choose_rule(train, features, objective=objective, max_two_feature=max_two_feature)
        pred = bool(rule.apply(test).iloc[0])
        train_stats = score_mask(rule.apply(train), train["y_plateau"])
        rows.append(
            {
                "run": run_name,
                "objective": objective,
                "allow_two_feature": int(max_two_feature),
                "heldout_case": held["short_case_id"],
                "heldout_group": held["group"],
                "heldout_label_plateau": int(held["y_plateau"]),
                "pred_plateau": int(pred),
                "correct": int(pred == bool(held["y_plateau"])),
                "rule": rule.text(),
                "train_plateau_kept": train_stats["plateau_kept"],
                "train_boundary_kept": train_stats["boundary_kept"],
                "train_balanced_accuracy": train_stats["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows)


def summarize_loo(loo: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in loo.groupby(["run", "objective", "allow_two_feature"]):
        run, objective, allow_two = keys
        y = group["heldout_label_plateau"].astype(int)
        p = group["pred_plateau"].astype(int)
        stats = score_mask(p.astype(bool), y)
        rows.append(
            {
                "run": run,
                "objective": objective,
                "allow_two_feature": allow_two,
                **stats,
                "cases": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["run", "objective", "allow_two_feature"]
    )


def markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._\n"
    subset = df[columns].head(max_rows).copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in subset.iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.3f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_decision(
    feature_auc: pd.DataFrame,
    single: pd.DataFrame,
    two: pd.DataFrame,
    loo_summary: pd.DataFrame,
    loo: pd.DataFrame,
) -> None:
    learned_loo = loo_summary[loo_summary["run"] == "learned_h120"]
    learned_zero = learned_loo[
        (learned_loo["objective"] == "zero_boundary") & (learned_loo["allow_two_feature"] == 1)
    ]
    learned_bal = learned_loo[
        (learned_loo["objective"] == "balanced") & (learned_loo["allow_two_feature"] == 1)
    ]
    zero_recall = float(learned_zero["plateau_recall"].iloc[0]) if len(learned_zero) else 0.0
    zero_fpr = float(learned_zero["boundary_fpr"].iloc[0]) if len(learned_zero) else 1.0
    bal_fpr = float(learned_bal["boundary_fpr"].iloc[0]) if len(learned_bal) else 1.0
    bal_recall = float(learned_bal["plateau_recall"].iloc[0]) if len(learned_bal) else 0.0

    if zero_fpr == 0.0 and zero_recall >= 0.5:
        decision = "CONDITIONAL-GO: a narrow interpretable separator exists, but it needs more mined plateau/boundary cases before controller wiring."
    elif bal_recall >= 0.6 and bal_fpr <= 0.2:
        decision = "CONDITIONAL-DIAGNOSTIC: the signal is informative, but the clean zero-boundary automatic rule is still too narrow."
    else:
        decision = "NO-GO for automatic plateau control now: learned h120 features do not provide a stable simple separator under LOO."

    lines = ["# Residual-high plateau separability study v1\n"]
    lines.append(f"## Decision\n\n{decision}\n")
    lines.append(
        "This is a read-only recognition study. It does not tune the controller. "
        "The question is whether residual-high plateau-good cases can be separated from "
        "direction-reversal/catch-up boundary cases by simple runtime forecast-shape features.\n"
    )
    lines.append("## Leave-one-case-out summary\n\n")
    lines.append(
        markdown_table(
            loo_summary,
            [
                "run",
                "objective",
                "allow_two_feature",
                "tp",
                "fp",
                "tn",
                "fn",
                "plateau_recall",
                "boundary_fpr",
                "balanced_accuracy",
            ],
        )
    )
    lines.append("\n## Top learned-h120 feature separation\n\n")
    lines.append(
        markdown_table(
            feature_auc[feature_auc["run"] == "learned_h120"],
            [
                "feature",
                "auc_plateau_high",
                "best_auc",
                "preferred_direction",
                "plateau_mean",
                "boundary_mean",
                "plateau_min",
                "plateau_max",
                "boundary_min",
                "boundary_max",
            ],
            max_rows=12,
        )
    )
    zero_single = single[(single["run"] == "learned_h120") & (single["boundary_kept"] == 0)].sort_values(
        ["plateau_kept", "balanced_accuracy"], ascending=[False, False]
    )
    lines.append("\n## Best zero-boundary learned-h120 single-feature rules\n\n")
    lines.append(
        markdown_table(
            zero_single,
            [
                "rule",
                "plateau_kept",
                "boundary_kept",
                "plateau_total",
                "boundary_total",
                "plateau_recall",
                "balanced_accuracy",
            ],
            max_rows=10,
        )
    )
    zero_two = two[(two["run"] == "learned_h120") & (two["boundary_kept"] == 0)].sort_values(
        ["plateau_kept", "balanced_accuracy"], ascending=[False, False]
    )
    lines.append("\n## Best zero-boundary learned-h120 two-feature rules\n\n")
    lines.append(
        markdown_table(
            zero_two,
            [
                "rule",
                "plateau_kept",
                "boundary_kept",
                "plateau_total",
                "boundary_total",
                "plateau_recall",
                "balanced_accuracy",
            ],
            max_rows=10,
        )
    )
    lines.append("\n## Interpretation\n\n")
    lines.append(
        "- A rule that is broad enough to catch many plateau cases also tends to admit boundary/catch-up cases. "
        "That matches the loose-vs-strict controller result.\n"
    )
    lines.append(
        "- Zero-boundary rules exist, but if they only retain a small plateau subset or fail under LOO, "
        "they are not a mature automatic second regime yet.\n"
    )
    lines.append(
        "- h120/f120 can still be useful as recognition context. But unless the LOO zero-boundary rule keeps a meaningful plateau share, "
        "the next mature controller should not be another gate tweak. Plateau should stay an operator/Pareto mode, or require a deliberately trained small plateau-vs-boundary head with more labels.\n"
    )
    (OUT / "decision.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_PAPER / "residual_plateau_separability_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    all_feature_auc = []
    all_single = []
    all_two = []
    all_loo = []
    for run_name, run_df in df.groupby("run"):
        run_df = run_df.copy().reset_index(drop=True)
        features = predictor_features(run_df)
        auc = feature_auc_table(run_df, features)
        auc.insert(0, "run", run_name)
        all_feature_auc.append(auc)

        single = single_condition_scan(run_df, features)
        if not single.empty:
            single.insert(0, "run", run_name)
            all_single.append(single)
            two = two_feature_scan(run_df, single)
            if not two.empty:
                two.insert(0, "run", run_name)
                # Add totals that are constant for convenience.
                two["plateau_total"] = int(run_df["y_plateau"].sum())
                two["boundary_total"] = int((1 - run_df["y_plateau"]).sum())
                all_two.append(two)
        for objective in ("zero_boundary", "balanced"):
            for allow_two in (False, True):
                all_loo.append(
                    loo_eval(
                        run_df,
                        features,
                        run_name=run_name,
                        objective=objective,
                        max_two_feature=allow_two,
                    )
                )

    feature_auc = pd.concat(all_feature_auc, ignore_index=True).sort_values(["run", "best_auc"], ascending=[True, False])
    single_rules = pd.concat(all_single, ignore_index=True).sort_values(
        ["run", "boundary_kept", "balanced_accuracy", "plateau_kept"],
        ascending=[True, True, False, False],
    )
    two_rules = (
        pd.concat(all_two, ignore_index=True).sort_values(
            ["run", "boundary_kept", "balanced_accuracy", "plateau_kept"],
            ascending=[True, True, False, False],
        )
        if all_two
        else pd.DataFrame()
    )
    loo = pd.concat(all_loo, ignore_index=True)
    loo_summary = summarize_loo(loo)

    df.to_csv(OUT_RAW / "separability_input_cases.csv", index=False)
    feature_auc.to_csv(OUT_RAW / "feature_auc.csv", index=False)
    single_rules.to_csv(OUT_RAW / "single_feature_rule_scan.csv", index=False)
    two_rules.to_csv(OUT_RAW / "two_feature_rule_scan.csv", index=False)
    loo.to_csv(OUT_RAW / "loo_rule_results.csv", index=False)
    loo_summary.to_csv(OUT_RAW / "loo_summary.csv", index=False)
    write_decision(feature_auc, single_rules, two_rules, loo_summary, loo)

    print("Wrote", OUT)
    print(loo_summary.to_string(index=False))


if __name__ == "__main__":
    main()
