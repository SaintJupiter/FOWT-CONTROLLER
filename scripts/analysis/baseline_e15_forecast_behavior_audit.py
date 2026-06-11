#!/usr/bin/env python3
"""Offline baseline/e15 forecast behavior audit.

This script intentionally does not run closed-loop control, train models, or
change planner settings. It compares fixed forecast sources on existing
10-minute dataset windows selected from prior case lists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.wind_prediction.ballast_planner import PlannerConfig, compute_pressure_blocks, norm_term
from src.wind_prediction.forecast_adapter import (
    CurrentOnlyForecastAdapter,
    ForecastModelAdapter,
    PersistenceMeanForecastAdapter,
)


RELIEF_THRESHOLD = 0.95
INTENSIFY_THRESHOLD = 1.05
EPS = 1e-6


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    model_dir: str | None = None


MODEL_SPECS = (
    ModelSpec(
        "baseline",
        "learned",
        "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
    ),
    ModelSpec(
        "e15",
        "learned",
        "outputs/wind_prediction/lstm_synth_relief_t030_e15_v1",
    ),
    ModelSpec("oracle", "oracle"),
    ModelSpec("persistence", "persistence"),
    ModelSpec("current_only", "current_only"),
)


def _read_cases(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "case_id" not in df.columns or "timestamp" not in df.columns:
        raise ValueError(f"{path} missing case_id/timestamp columns")
    out = pd.DataFrame()
    out["case_id"] = df["case_id"].astype(str)
    out["timestamp"] = pd.to_datetime(df["timestamp"])
    out["label"] = df["label"].astype(str) if "label" in df.columns else ""
    out["selection_group"] = (
        df["selection_group"].astype(str) if "selection_group" in df.columns else ""
    )
    out["source"] = source
    out["source_path"] = str(path)
    return out


def build_audit_cases() -> pd.DataFrame:
    sources = [
        (
            "pilot30",
            REPO_ROOT / "outputs/wind_prediction/ab_reliefcap030_pilot30_v1/pilot30_cases.csv",
        ),
        (
            "mechanism_grid_v2",
            REPO_ROOT / "outputs/wind_prediction/prediction_value_mechanism_grid_v2/mechanism_grid_cases.csv",
        ),
        (
            "guard10",
            REPO_ROOT / "outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv",
        ),
        (
            "broader20",
            REPO_ROOT
            / "outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/f60_relief_e15_holdout_cases.csv",
        ),
    ]
    frames = [_read_cases(path, source) for source, path in sources if path.exists()]
    if not frames:
        raise FileNotFoundError("no case source CSVs found")
    df = pd.concat(frames, ignore_index=True)
    guard = set()
    guard_ts = set()
    guard_path = REPO_ROOT / "outputs/wind_prediction/broad_optimization_work/guard10_broad_cases.csv"
    if guard_path.exists():
        g = pd.read_csv(guard_path)
        guard = {(str(r.case_id), pd.Timestamp(r.timestamp)) for r in g.itertuples()}
        guard_ts = {pd.Timestamp(r.timestamp) for r in g.itertuples()}
    df["is_guard10"] = [
        bool((str(r.case_id), pd.Timestamp(r.timestamp)) in guard or pd.Timestamp(r.timestamp) in guard_ts)
        for r in df.itertuples()
    ]
    df["case_family"] = df.apply(infer_case_family, axis=1)
    family_rank = {
        "pilot30": 0,
        "guard10": 1,
        "broader20": 2,
        "mechanism_grid_v2": 3,
    }
    df["_source_rank"] = df["source"].map(family_rank).fillna(99)
    df = (
        df.sort_values(["timestamp", "_source_rank", "case_id"])
        .drop_duplicates(subset=["timestamp"], keep="first")
        .drop(columns=["_source_rank"])
        .reset_index(drop=True)
    )
    return df


def infer_case_family(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("case_id", "")),
            str(row.get("selection_group", "")),
            str(row.get("label", "")),
        ]
    ).lower()
    if any(x in text for x in ("lowrisk", "quiet", "random low")):
        return "lowrisk"
    if any(x in text for x in ("signflip", "reversal")):
        return "signflip"
    if any(x in text for x in ("onset", "future_risk")):
        return "onset"
    if any(x in text for x in ("b/high", "b_high", "sustained_high", "high_pressure", "stronger_active")):
        return "b_high_or_sustained"
    if any(x in text for x in ("relief", "decay", "hold_risk")):
        return "future_relief_or_decay"
    if any(x in text for x in ("persistence", "flat")):
        return "flat_or_persistence"
    return "other"


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    idx = pd.read_csv(
        dataset_dir / "sample_index.csv.gz",
        parse_dates=["history_start", "history_end", "future_start", "future_end"],
    )
    arrays = {}
    for split in ("train", "validation", "test"):
        arrays[f"X_{split}"] = np.load(dataset_dir / f"X_{split}.npy", mmap_mode="r")
        arrays[f"y_uv_raw_{split}"] = np.load(dataset_dir / f"y_uv_raw_{split}.npy", mmap_mode="r")
    split_counts: dict[str, int] = {}
    offsets = []
    for split in idx["split"].tolist():
        n = split_counts.get(split, 0)
        offsets.append(n)
        split_counts[split] = n + 1
    idx["_split_offset"] = offsets
    return idx, arrays


def pressure_norms(uv: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    blocks = compute_pressure_blocks(np.asarray(uv, dtype=float), [1.0, 1.0, 1.0], cfg)
    return np.asarray(
        [norm_term(np.asarray(b["pressure_vec_raw"], dtype=float), cfg) for b in blocks],
        dtype=float,
    )


def predict_models(x: np.ndarray, y_oracle: np.ndarray, adapters: dict[str, object], cfg: PlannerConfig) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {"oracle": pressure_norms(y_oracle, cfg)}
    for name, adapter in adapters.items():
        result = adapter.predict_window(x)
        preds[name] = pressure_norms(result.wind_uv_raw, cfg)
    return preds


def true_regime(norms: np.ndarray) -> str:
    ratios = ratios_from_norms(norms)
    relief = bool(min(ratios[1], ratios[2]) < RELIEF_THRESHOLD)
    intensify = bool(max(ratios[1], ratios[2]) > INTENSIFY_THRESHOLD)
    direction_reversal = bool((ratios[1] - 1.0) * (ratios[2] - ratios[1]) < -0.01)
    if relief and intensify:
        return "mixed_swing"
    if relief:
        return "relief_only"
    if intensify:
        return "intensify_only"
    if direction_reversal:
        return "mixed_swing"
    return "flat"


def ratios_from_norms(norms: np.ndarray) -> np.ndarray:
    b0 = max(float(norms[0]), EPS)
    return np.asarray([1.0, float(norms[1]) / b0, float(norms[2]) / b0], dtype=float)


def relief_pred(norms: np.ndarray) -> bool:
    r = ratios_from_norms(norms)
    return bool(min(r[1], r[2]) < RELIEF_THRESHOLD)


def intensify_pred(norms: np.ndarray) -> bool:
    r = ratios_from_norms(norms)
    return bool(max(r[1], r[2]) > INTENSIFY_THRESHOLD)


def direction_agree(pred: np.ndarray, oracle: np.ndarray) -> float:
    pred_steps = np.sign(np.diff(pred))
    oracle_steps = np.sign(np.diff(oracle))
    return float(np.mean(pred_steps == oracle_steps))


def range_value(norms: np.ndarray) -> float:
    return float(np.max(norms) - np.min(norms))


def range_ratio(pred_norms: np.ndarray, oracle_norms: np.ndarray) -> float:
    oracle_range = range_value(oracle_norms)
    if oracle_range < 0.05:
        return float("nan")
    return range_value(pred_norms) / oracle_range


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if y_true.size == 0 or y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def classification_metrics(
    df: pd.DataFrame,
    model: str,
    target_col: str,
    pred_col: str,
    score_col: str,
) -> dict[str, float]:
    y = df[target_col].astype(bool).to_numpy()
    p = df[pred_col].astype(bool).to_numpy()
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision + recall) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "model": model,
        "target": target_col.replace("true_", ""),
        "n": int(len(df)),
        "support": int(np.sum(y)),
        "predicted_positive": int(np.sum(p)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "pr_auc": average_precision(y, df[score_col].to_numpy()),
    }


def make_selector_norms(row: pd.Series, selector: str) -> np.ndarray:
    b = np.asarray([row[f"baseline_b{i}"] for i in range(3)], dtype=float)
    e = np.asarray([row[f"e15_b{i}"] for i in range(3)], dtype=float)
    baseline_risk = bool(row["baseline_pred_intensify"])
    e15_relief = bool(row["e15_pred_relief"])
    if selector == "conservative_select":
        return b if bool(row["baseline_e15_disagree_relief"]) else e
    if selector == "relief_confirmed_select":
        return e if e15_relief and not baseline_risk else b
    if selector == "soft_blend_03":
        return 0.3 * e + 0.7 * b if e15_relief and not baseline_risk else b
    if selector == "soft_blend_05":
        return 0.5 * e + 0.5 * b if e15_relief and not baseline_risk else b
    raise ValueError(selector)


def bootstrap_stability(df: pd.DataFrame, model: str, pred_col: str, score_name: str, n_iter: int = 500) -> dict[str, float]:
    rng = np.random.default_rng(20260519)
    case_ids = np.asarray(sorted(df["case_id"].unique()))
    values = []
    for _ in range(n_iter):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        sub = pd.concat([df[df["case_id"] == c] for c in sampled], ignore_index=True)
        y = sub["true_relief"].astype(bool).to_numpy()
        p = sub[pred_col].astype(bool).to_numpy()
        tp = np.sum(y & p)
        fp = np.sum(~y & p)
        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / max(np.sum(y), 1)
        values.append((precision, recall))
    arr = np.asarray(values, dtype=float)
    return {
        "model": model,
        "metric": score_name,
        "precision_mean": float(np.nanmean(arr[:, 0])),
        "precision_p05": float(np.nanpercentile(arr[:, 0], 5)),
        "precision_p95": float(np.nanpercentile(arr[:, 0], 95)),
        "recall_mean": float(np.nanmean(arr[:, 1])),
        "recall_p05": float(np.nanpercentile(arr[:, 1], 5)),
        "recall_p95": float(np.nanpercentile(arr[:, 1], 95)),
    }


def md_table(obj: pd.DataFrame | pd.Series, floatfmt: str = ".3f") -> str:
    """Small markdown table helper to avoid depending on tabulate."""
    if isinstance(obj, pd.Series):
        df = obj.rename("count").reset_index()
    else:
        df = obj.copy()
    if df.empty:
        return "_empty_"
    headers = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for val in row.tolist():
            if isinstance(val, (float, np.floating)):
                if math.isnan(float(val)):
                    vals.append("nan")
                else:
                    vals.append(format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def evaluate_subsets(bucket_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subsets = {
        "all": bucket_df,
        "non_guard10": bucket_df[~bucket_df["is_guard10"]],
        "guard10_only": bucket_df[bucket_df["is_guard10"]],
    }
    models = ["baseline", "e15", "persistence", "current_only"]
    selectors = ["conservative_select", "relief_confirmed_select", "soft_blend_03", "soft_blend_05"]
    for subset_name, sub in subsets.items():
        if sub.empty:
            continue
        for model in models + selectors:
            for target, score_suffix in [("true_relief", "relief_score"), ("true_intensify", "intensify_score")]:
                pred_col = f"{model}_pred_{target.replace('true_', '')}"
                score_col = f"{model}_{score_suffix}"
                if pred_col not in sub.columns:
                    continue
                rec = classification_metrics(sub, model, target, pred_col, score_col)
                rec["subset"] = subset_name
                rows.append(rec)
    return pd.DataFrame(rows)


def write_markdown_reports(out_dir: Path, cases: pd.DataFrame, bucket_df: pd.DataFrame, metrics: pd.DataFrame, gated: pd.DataFrame) -> None:
    distribution = []
    distribution.append("# Audit Case Distribution\n")
    distribution.append(f"- Cases/windows: {len(cases)}\n")
    distribution.append(f"- Guard10 windows: {int(cases['is_guard10'].sum())}\n")
    distribution.append(f"- Non-guard10 windows: {int((~cases['is_guard10']).sum())}\n\n")
    distribution.append("## By Case Family\n\n")
    distribution.append(md_table(cases["case_family"].value_counts()))
    distribution.append("\n\n## By True Trend Regime\n\n")
    distribution.append(md_table(bucket_df["true_trend_regime"].value_counts()))
    distribution.append("\n\n## By Source\n\n")
    distribution.append(md_table(cases["source"].value_counts()))
    (out_dir / "audit_case_distribution.md").write_text("\n".join(distribution), encoding="utf-8")

    summary = []
    summary.append("# Baseline/e15 Forecast Behavior Audit\n")
    summary.append("This is an offline forecast audit only: no training, no controller change, no e15 mainline wiring.\n")
    summary.append("## Relief Metrics\n")
    show = metrics[(metrics["target"] == "relief") & (metrics["model"].isin(["baseline", "e15", "relief_confirmed_select", "soft_blend_03", "soft_blend_05"]))][
        ["subset", "model", "support", "predicted_positive", "precision", "recall", "f1", "false_positive_rate", "pr_auc"]
    ].copy()
    summary.append(md_table(show))
    summary.append("\n\n## Key Decision Notes\n")
    for subset in ["all", "non_guard10", "guard10_only"]:
        base = metrics[(metrics["subset"] == subset) & (metrics["model"] == "baseline") & (metrics["target"] == "relief")]
        e15 = metrics[(metrics["subset"] == subset) & (metrics["model"] == "e15") & (metrics["target"] == "relief")]
        gate = metrics[(metrics["subset"] == subset) & (metrics["model"] == "relief_confirmed_select") & (metrics["target"] == "relief")]
        if base.empty or e15.empty or gate.empty:
            continue
        b, e, g = base.iloc[0], e15.iloc[0], gate.iloc[0]
        summary.append(
            f"- {subset}: baseline P/R={b.precision:.3f}/{b.recall:.3f}, "
            f"e15 P/R={e.precision:.3f}/{e.recall:.3f}, "
            f"relief_confirmed P/R={g.precision:.3f}/{g.recall:.3f}."
        )
    summary.append("\n\n## Bootstrap Stability\n")
    boot = pd.read_csv(out_dir / "bootstrap_stability_table.csv")
    summary.append(md_table(boot))
    (out_dir / "forecast_behavior_summary.md").write_text("\n".join(summary), encoding="utf-8")

    gate_summary = []
    gate_summary.append("# Gated e15 Offline Selector Summary\n")
    gate_summary.append("Selectors are evaluated only as forecast outputs against oracle pressure blocks; no controller replay is performed.\n")
    gate_summary.append("## Selector Shape Metrics\n")
    gate_summary.append(md_table(gated))
    gate_summary.append("\n\n## Decision\n")
    gate_summary.append(
        "A selector should only advance to a default-off closed-loop probe if it improves false-relief behavior "
        "on non-guard10 without collapsing true relief recall or hurting lowrisk. See metrics table for the pass/fail call."
    )
    (out_dir / "gated_e15_offline_summary.md").write_text("\n".join(gate_summary), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/baseline_e15_forecast_behavior_audit_v1",
    )
    parser.add_argument(
        "--dataset-dir",
        default="data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = REPO_ROOT / args.dataset_dir
    idx, arrays = load_dataset(dataset_dir)
    cases = build_audit_cases()
    idx_lookup = {
        pd.Timestamp(r["future_start"]): (str(r["split"]), int(r["_split_offset"]))
        for _, r in idx.iterrows()
    }
    cases = cases[cases["timestamp"].isin(idx_lookup)].copy().reset_index(drop=True)

    cfg = PlannerConfig()
    adapters: dict[str, object] = {
        "baseline": ForecastModelAdapter(
            REPO_ROOT / "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1",
            dataset_dir=dataset_dir,
            device=args.device,
        ),
        "e15": ForecastModelAdapter(
            REPO_ROOT / "outputs/wind_prediction/lstm_synth_relief_t030_e15_v1",
            dataset_dir=dataset_dir,
            device=args.device,
        ),
        "persistence": PersistenceMeanForecastAdapter(dataset_dir=dataset_dir),
        "current_only": CurrentOnlyForecastAdapter(dataset_dir=dataset_dir),
    }

    rows = []
    for case in cases.itertuples():
        split, off = idx_lookup[pd.Timestamp(case.timestamp)]
        x = arrays[f"X_{split}"][off]
        y = arrays[f"y_uv_raw_{split}"][off]
        pred = predict_models(x, y, adapters, cfg)
        oracle = pred["oracle"]
        true_r = ratios_from_norms(oracle)
        regime = true_regime(oracle)
        rec = {
            "case_id": case.case_id,
            "timestamp": pd.Timestamp(case.timestamp),
            "source": case.source,
            "label": case.label,
            "case_family": case.case_family,
            "is_guard10": bool(case.is_guard10),
            "split": split,
            "true_trend_regime": regime,
            "true_relief": bool(regime == "relief_only"),
            "true_intensify": bool(regime == "intensify_only"),
            "true_relief_score": float(max(0.0, 1.0 - min(true_r[1], true_r[2]))),
            "true_intensify_score": float(max(0.0, max(true_r[1], true_r[2]) - 1.0)),
        }
        for model, norms in pred.items():
            ratios = ratios_from_norms(norms)
            for i in range(3):
                rec[f"{model}_b{i}"] = float(norms[i])
                rec[f"{model}_r{i}"] = float(ratios[i])
            rec[f"{model}_pred_relief"] = relief_pred(norms)
            rec[f"{model}_pred_intensify"] = intensify_pred(norms)
            rec[f"{model}_relief_score"] = float(max(0.0, 1.0 - min(ratios[1], ratios[2])))
            rec[f"{model}_intensify_score"] = float(max(0.0, max(ratios[1], ratios[2]) - 1.0))
            rec[f"{model}_shape_l2_to_oracle"] = float(np.linalg.norm(norms - oracle))
            rec[f"{model}_direction_agree"] = direction_agree(norms, oracle)
            rec[f"{model}_range"] = range_value(norms)
            rec[f"{model}_range_abs_error"] = abs(range_value(norms) - range_value(oracle))
            rec[f"{model}_range_ratio_to_oracle"] = range_ratio(norms, oracle)
        rec["baseline_e15_disagree_relief"] = bool(rec["baseline_pred_relief"] != rec["e15_pred_relief"])
        rec["baseline_risk_rejects_e15_relief"] = bool(rec["e15_pred_relief"] and rec["baseline_pred_intensify"])
        rows.append(rec)

    bucket_df = pd.DataFrame(rows)

    for selector in ["conservative_select", "relief_confirmed_select", "soft_blend_03", "soft_blend_05"]:
        norms_list = []
        for _, row in bucket_df.iterrows():
            norms = make_selector_norms(row, selector)
            norms_list.append(norms)
        arr = np.vstack(norms_list)
        for i in range(3):
            bucket_df[f"{selector}_b{i}"] = arr[:, i]
        bucket_df[f"{selector}_pred_relief"] = [relief_pred(x) for x in arr]
        bucket_df[f"{selector}_pred_intensify"] = [intensify_pred(x) for x in arr]
        bucket_df[f"{selector}_relief_score"] = [
            max(0.0, 1.0 - min(ratios_from_norms(x)[1], ratios_from_norms(x)[2]))
            for x in arr
        ]
        bucket_df[f"{selector}_intensify_score"] = [
            max(0.0, max(ratios_from_norms(x)[1], ratios_from_norms(x)[2]) - 1.0)
            for x in arr
        ]
        bucket_df[f"{selector}_shape_l2_to_oracle"] = [
            float(np.linalg.norm(arr[i] - bucket_df.loc[i, ["oracle_b0", "oracle_b1", "oracle_b2"]].to_numpy(dtype=float)))
            for i in range(len(bucket_df))
        ]
        bucket_df[f"{selector}_direction_agree"] = [
            direction_agree(arr[i], bucket_df.loc[i, ["oracle_b0", "oracle_b1", "oracle_b2"]].to_numpy(dtype=float))
            for i in range(len(bucket_df))
        ]
        bucket_df[f"{selector}_range"] = [range_value(x) for x in arr]
        bucket_df[f"{selector}_range_abs_error"] = [
            abs(range_value(arr[i]) - float(bucket_df.loc[i, "oracle_range"]))
            for i in range(len(bucket_df))
        ]
        bucket_df[f"{selector}_range_ratio_to_oracle"] = [
            range_ratio(
                arr[i],
                bucket_df.loc[i, ["oracle_b0", "oracle_b1", "oracle_b2"]].to_numpy(dtype=float),
            )
            for i in range(len(bucket_df))
        ]

    cases = cases.merge(
        bucket_df[["case_id", "timestamp", "true_trend_regime"]],
        on=["case_id", "timestamp"],
        how="left",
    )

    metrics = evaluate_subsets(bucket_df)

    regime_rows = []
    for subset_name, sub in {
        "all": bucket_df,
        "non_guard10": bucket_df[~bucket_df["is_guard10"]],
        "guard10_only": bucket_df[bucket_df["is_guard10"]],
    }.items():
        for regime, rsub in sub.groupby("true_trend_regime"):
            for model in ["baseline", "e15", "relief_confirmed_select", "soft_blend_03", "soft_blend_05"]:
                regime_rows.append(
                    {
                        "subset": subset_name,
                        "true_trend_regime": regime,
                        "model": model,
                        "n": len(rsub),
                        "pred_relief_rate": float(rsub[f"{model}_pred_relief"].mean()),
                        "pred_intensify_rate": float(rsub[f"{model}_pred_intensify"].mean()),
                        "mean_shape_l2_to_oracle": float(rsub[f"{model}_shape_l2_to_oracle"].mean()),
                        "mean_direction_agree": float(rsub[f"{model}_direction_agree"].mean()),
                        "mean_range_abs_error": float(rsub[f"{model}_range_abs_error"].mean()),
                        "mean_range_ratio_to_oracle": float(rsub[f"{model}_range_ratio_to_oracle"].replace([np.inf, -np.inf], np.nan).mean()),
                    }
                )
    regime_table = pd.DataFrame(regime_rows)

    selector_rows = []
    for subset_name, sub in {
        "all": bucket_df,
        "non_guard10": bucket_df[~bucket_df["is_guard10"]],
        "guard10_only": bucket_df[bucket_df["is_guard10"]],
    }.items():
        for model in ["baseline", "e15", "conservative_select", "relief_confirmed_select", "soft_blend_03", "soft_blend_05"]:
            true_relief_rejected = int(np.sum(sub["true_relief"] & ~sub[f"{model}_pred_relief"]))
            true_relief_count = int(sub["true_relief"].sum())
            onset_signflip = sub[sub["case_family"].isin(["onset", "signflip"])]
            selector_rows.append(
                {
                    "subset": subset_name,
                    "model": model,
                    "n": len(sub),
                    "mean_shape_l2_to_oracle": float(sub[f"{model}_shape_l2_to_oracle"].mean()),
                    "mean_direction_agree": float(sub[f"{model}_direction_agree"].mean()),
                    "mean_range_abs_error": float(sub[f"{model}_range_abs_error"].mean()),
                    "mean_range_ratio_to_oracle": float(sub[f"{model}_range_ratio_to_oracle"].replace([np.inf, -np.inf], np.nan).mean()),
                    "false_relief": int(np.sum(~sub["true_relief"] & sub[f"{model}_pred_relief"])),
                    "false_relief_rate": float(np.mean(~sub["true_relief"] & sub[f"{model}_pred_relief"])),
                    "true_relief_rejected": true_relief_rejected,
                    "true_relief_rejected_rate": true_relief_rejected / true_relief_count if true_relief_count else float("nan"),
                    "onset_signflip_false_relief": int(np.sum(~onset_signflip["true_relief"] & onset_signflip[f"{model}_pred_relief"])) if not onset_signflip.empty else 0,
                    "onset_signflip_false_relief_rate": float(np.mean(~onset_signflip["true_relief"] & onset_signflip[f"{model}_pred_relief"])) if not onset_signflip.empty else float("nan"),
                    "lowrisk_pred_relief_count": int(np.sum((sub["case_family"] == "lowrisk") & sub[f"{model}_pred_relief"])),
                }
            )
    gated_table = pd.DataFrame(selector_rows)

    disagreement = []
    for subset_name, sub in {
        "all": bucket_df,
        "non_guard10": bucket_df[~bucket_df["is_guard10"]],
        "guard10_only": bucket_df[bucket_df["is_guard10"]],
    }.items():
        ct = pd.crosstab(sub["baseline_pred_relief"], sub["e15_pred_relief"])
        for base_val in [False, True]:
            for e_val in [False, True]:
                disagreement.append(
                    {
                        "subset": subset_name,
                        "baseline_pred_relief": base_val,
                        "e15_pred_relief": e_val,
                        "count": int(ct.loc[base_val, e_val]) if base_val in ct.index and e_val in ct.columns else 0,
                    }
                )
    disagreement_table = pd.DataFrame(disagreement)

    boot = pd.DataFrame(
        [
            bootstrap_stability(bucket_df[~bucket_df["is_guard10"]], "baseline", "baseline_pred_relief", "non_guard10_relief"),
            bootstrap_stability(bucket_df[~bucket_df["is_guard10"]], "e15", "e15_pred_relief", "non_guard10_relief"),
            bootstrap_stability(bucket_df[~bucket_df["is_guard10"]], "relief_confirmed_select", "relief_confirmed_select_pred_relief", "non_guard10_relief"),
            bootstrap_stability(bucket_df, "baseline", "baseline_pred_relief", "all_relief"),
            bootstrap_stability(bucket_df, "e15", "e15_pred_relief", "all_relief"),
            bootstrap_stability(bucket_df, "relief_confirmed_select", "relief_confirmed_select_pred_relief", "all_relief"),
        ]
    )

    bucket_df.to_csv(out_dir / "forecast_behavior_bucket_table.csv", index=False)
    cases.to_csv(out_dir / "audit_cases.csv", index=False)
    metrics.to_csv(out_dir / "forecast_behavior_table.csv", index=False)
    regime_table.to_csv(out_dir / "forecast_behavior_regime_table.csv", index=False)
    disagreement_table.to_csv(out_dir / "model_disagreement_matrix.csv", index=False)
    boot.to_csv(out_dir / "bootstrap_stability_table.csv", index=False)
    gated_table.to_csv(out_dir / "gated_e15_rule_table.csv", index=False)
    gated_table.to_csv(out_dir / "gated_e15_regime_table.csv", index=False)
    gated_table.to_csv(out_dir / "gated_e15_action_proxy_table.csv", index=False)

    write_markdown_reports(out_dir, cases, bucket_df, metrics, gated_table)

    print(f"wrote {out_dir}")
    print(f"cases={len(cases)} guard10={int(cases['is_guard10'].sum())} non_guard10={int((~cases['is_guard10']).sum())}")


if __name__ == "__main__":
    main()
