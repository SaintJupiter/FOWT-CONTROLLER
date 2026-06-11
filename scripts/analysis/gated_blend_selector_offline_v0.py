#!/usr/bin/env python3
"""Offline dynamic baseline/e15 blend selector audit.

Consumes the existing baseline/e15 forecast behavior bucket table. It does not
train, replay closed-loop control, or modify controller logic.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RELIEF_THRESHOLD = 0.95
STRONG_RELIEF_THRESHOLD = 0.85
INTENSIFY_THRESHOLD = 1.05
STRONG_INTENSIFY_THRESHOLD = 1.15
EPS = 1e-6


def norms(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_b0"], row[f"{prefix}_b1"], row[f"{prefix}_b2"]], dtype=float)


def ratios(arr: np.ndarray) -> np.ndarray:
    b0 = max(float(arr[0]), EPS)
    return np.asarray([1.0, float(arr[1]) / b0, float(arr[2]) / b0], dtype=float)


def pred_relief(arr: np.ndarray) -> bool:
    r = ratios(arr)
    return bool(min(r[1], r[2]) < RELIEF_THRESHOLD)


def pred_strong_relief(arr: np.ndarray) -> bool:
    r = ratios(arr)
    return bool(min(r[1], r[2]) < STRONG_RELIEF_THRESHOLD)


def pred_intensify(arr: np.ndarray) -> bool:
    r = ratios(arr)
    return bool(max(r[1], r[2]) > INTENSIFY_THRESHOLD)


def pred_strong_intensify(arr: np.ndarray) -> bool:
    r = ratios(arr)
    return bool(max(r[1], r[2]) > STRONG_INTENSIFY_THRESHOLD)


def direction_agree(pred: np.ndarray, oracle: np.ndarray) -> float:
    return float(np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(oracle))))


def range_value(arr: np.ndarray) -> float:
    return float(np.max(arr) - np.min(arr))


def range_ratio(pred: np.ndarray, oracle: np.ndarray) -> float:
    oracle_range = range_value(oracle)
    if oracle_range < 0.05:
        return float("nan")
    return range_value(pred) / oracle_range


def relief_score(arr: np.ndarray) -> float:
    r = ratios(arr)
    return float(max(0.0, 1.0 - min(r[1], r[2])))


def intensify_score(arr: np.ndarray) -> float:
    r = ratios(arr)
    return float(max(0.0, max(r[1], r[2]) - 1.0))


def signflip_or_swing_proxy(row: pd.Series) -> bool:
    if str(row.get("case_family", "")) == "signflip":
        return True
    for prefix in ("baseline", "e15"):
        arr = norms(row, prefix)
        r = ratios(arr)
        if (r[1] - 1.0) * (r[2] - r[1]) < -0.01:
            return True
    b = norms(row, "baseline")
    e = norms(row, "e15")
    # Opposite trend direction between the two models is also a runtime-only
    # swing proxy. Oracle regime is deliberately not used here.
    if np.sign(b[2] - b[0]) * np.sign(e[2] - e[0]) < 0:
        return True
    return False


def dynamic_alpha(row: pd.Series) -> tuple[float, str]:
    b = norms(row, "baseline")
    e = norms(row, "e15")
    b_relief = pred_relief(b)
    e_relief = pred_relief(e)
    b_intensify = pred_intensify(b)
    b_strong_intensify = pred_strong_intensify(b)
    e_strong_relief = pred_strong_relief(e)
    baseline_flat = not b_relief and not b_intensify
    if b_strong_intensify:
        return 0.0, "baseline_strong_intensify"
    if b_intensify and e_relief:
        return 0.0, "baseline_intensify_rejects_e15_relief"
    if signflip_or_swing_proxy(row):
        return 0.0, "signflip_or_swing_proxy"
    if b_relief and e_relief:
        return 0.3, "both_relief_alpha03"
    if e_strong_relief and baseline_flat:
        return 0.3, "e15_strong_relief_baseline_flat_alpha03"
    if e_relief and baseline_flat:
        return 0.1, "e15_relief_baseline_flat_alpha01"
    return 0.0, "default_baseline"


def fixed_soft_blend(row: pd.Series, alpha: float) -> np.ndarray:
    b = norms(row, "baseline")
    e = norms(row, "e15")
    if pred_relief(e) and not pred_intensify(b):
        return alpha * e + (1.0 - alpha) * b
    return b


def classify_metrics(df: pd.DataFrame, model: str, target_col: str, pred_col: str, score_col: str) -> dict[str, float | int | str]:
    y = df[target_col].astype(bool).to_numpy()
    p = df[pred_col].astype(bool).to_numpy()
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision + recall) else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    return {
        "model": model,
        "target": target_col.replace("true_", ""),
        "n": int(len(df)),
        "support": int(y.sum()),
        "predicted_positive": int(p.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "pr_auc": average_precision(y, df[score_col].to_numpy(dtype=float)),
    }


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    if y.size == 0 or y.sum() == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev) * precision))


def add_selector(df: pd.DataFrame, name: str, values: list[np.ndarray], reasons: list[str] | None = None, alphas: list[float] | None = None) -> None:
    arr = np.vstack(values)
    for i in range(3):
        df[f"{name}_b{i}"] = arr[:, i]
    df[f"{name}_pred_relief"] = [pred_relief(x) for x in arr]
    df[f"{name}_pred_intensify"] = [pred_intensify(x) for x in arr]
    df[f"{name}_relief_score"] = [relief_score(x) for x in arr]
    df[f"{name}_intensify_score"] = [intensify_score(x) for x in arr]
    df[f"{name}_shape_l2_to_oracle"] = [
        float(np.linalg.norm(arr[i] - norms(df.iloc[i], "oracle"))) for i in range(len(df))
    ]
    df[f"{name}_direction_agree"] = [
        direction_agree(arr[i], norms(df.iloc[i], "oracle")) for i in range(len(df))
    ]
    df[f"{name}_range"] = [range_value(x) for x in arr]
    df[f"{name}_range_abs_error"] = [
        abs(range_value(arr[i]) - range_value(norms(df.iloc[i], "oracle"))) for i in range(len(df))
    ]
    df[f"{name}_range_ratio_to_oracle"] = [
        range_ratio(arr[i], norms(df.iloc[i], "oracle")) for i in range(len(df))
    ]
    if reasons is not None:
        df[f"{name}_reason"] = reasons
    if alphas is not None:
        df[f"{name}_weight"] = alphas


def subset_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": df,
        "non_guard10": df[~df["is_guard10"]],
        "guard10_only": df[df["is_guard10"]],
    }


def summarize_models(df: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    regime_rows = []
    disagreement_rows = []
    for subset, sub in subset_frames(df).items():
        for model in models:
            for target, score in (("true_relief", "relief_score"), ("true_intensify", "intensify_score")):
                metric = classify_metrics(sub, model, target, f"{model}_pred_{target.replace('true_', '')}", f"{model}_{score}")
                metric["subset"] = subset
                metric_rows.append(metric)
            true_relief = sub["true_relief"].astype(bool)
            pred_relief = sub[f"{model}_pred_relief"].astype(bool)
            onset_signflip_mixed = sub[
                sub["case_family"].isin(["onset", "signflip"]) | (sub["true_trend_regime"] == "mixed_swing")
            ]
            regime_rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "n": int(len(sub)),
                    "mean_shape_l2_to_oracle": float(sub[f"{model}_shape_l2_to_oracle"].mean()),
                    "mean_direction_agree": float(sub[f"{model}_direction_agree"].mean()),
                    "mean_range_abs_error": float(sub[f"{model}_range_abs_error"].mean()),
                    "mean_range_ratio_to_oracle": float(sub[f"{model}_range_ratio_to_oracle"].replace([np.inf, -np.inf], np.nan).mean()),
                    "false_relief": int((~true_relief & pred_relief).sum()),
                    "false_relief_rate": float((~true_relief & pred_relief).mean()),
                    "true_relief_rejected": int((true_relief & ~pred_relief).sum()),
                    "true_relief_rejected_rate": float((true_relief & ~pred_relief).sum() / max(int(true_relief.sum()), 1)),
                    "onset_signflip_mixed_false_relief": int((~onset_signflip_mixed["true_relief"].astype(bool) & onset_signflip_mixed[f"{model}_pred_relief"].astype(bool)).sum()) if not onset_signflip_mixed.empty else 0,
                    "onset_signflip_mixed_false_relief_rate": float((~onset_signflip_mixed["true_relief"].astype(bool) & onset_signflip_mixed[f"{model}_pred_relief"].astype(bool)).mean()) if not onset_signflip_mixed.empty else float("nan"),
                    "lowrisk_pred_relief_count": int(((sub["case_family"] == "lowrisk") & pred_relief).sum()),
                }
            )
        # Disagreement partitions are runtime-only model predictions plus true labels for evaluation.
        for reason, rsub in sub.groupby("dynamic_alpha_reason"):
            disagreement_rows.append(
                {
                    "subset": subset,
                    "dynamic_alpha_reason": reason,
                    "n": int(len(rsub)),
                    "mean_alpha": float(rsub["dynamic_alpha_weight"].mean()),
                    "true_regime_counts": json_counts(rsub["true_trend_regime"]),
                    "case_family_counts": json_counts(rsub["case_family"]),
                    "baseline_shape_l2": float(rsub["baseline_shape_l2_to_oracle"].mean()),
                    "e15_shape_l2": float(rsub["e15_shape_l2_to_oracle"].mean()),
                    "dynamic_alpha_shape_l2": float(rsub["dynamic_alpha_shape_l2_to_oracle"].mean()),
                    "dynamic_alpha_false_relief": int((~rsub["true_relief"].astype(bool) & rsub["dynamic_alpha_pred_relief"].astype(bool)).sum()),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(regime_rows), pd.DataFrame(disagreement_rows)


def json_counts(series: pd.Series) -> str:
    return "; ".join(f"{k}:{v}" for k, v in series.value_counts().items())


def bootstrap(df: pd.DataFrame, models: list[str], n_iter: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(20260519)
    rows = []
    for subset, sub in subset_frames(df).items():
        if sub.empty:
            continue
        cases = np.asarray(sorted(sub["case_id"].unique()))
        for model in models:
            vals = []
            for _ in range(n_iter):
                sampled = rng.choice(cases, size=len(cases), replace=True)
                boot = pd.concat([sub[sub["case_id"] == c] for c in sampled], ignore_index=True)
                y = boot["true_relief"].astype(bool).to_numpy()
                p = boot[f"{model}_pred_relief"].astype(bool).to_numpy()
                tp = np.sum(y & p)
                fp = np.sum(~y & p)
                fn = np.sum(y & ~p)
                precision = tp / (tp + fp) if tp + fp else np.nan
                recall = tp / (tp + fn) if tp + fn else np.nan
                false_relief_rate = fp / max(len(boot), 1)
                l2 = boot[f"{model}_shape_l2_to_oracle"].mean()
                vals.append((precision, recall, false_relief_rate, l2))
            arr = np.asarray(vals, dtype=float)
            rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "precision_mean": float(np.nanmean(arr[:, 0])),
                    "precision_p05": float(np.nanpercentile(arr[:, 0], 5)),
                    "precision_p95": float(np.nanpercentile(arr[:, 0], 95)),
                    "recall_mean": float(np.nanmean(arr[:, 1])),
                    "recall_p05": float(np.nanpercentile(arr[:, 1], 5)),
                    "recall_p95": float(np.nanpercentile(arr[:, 1], 95)),
                    "false_relief_rate_mean": float(np.nanmean(arr[:, 2])),
                    "shape_l2_mean": float(np.nanmean(arr[:, 3])),
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for val in row.tolist():
            if isinstance(val, (float, np.floating)):
                vals.append("nan" if math.isnan(float(val)) else format(float(val), floatfmt))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(out_dir: Path, metrics: pd.DataFrame, regime: pd.DataFrame, disagreement: pd.DataFrame, boot: pd.DataFrame) -> None:
    lines = [
        "# Gated Blend Selector Offline v0",
        "",
        "Offline forecast selector audit only: no training, no controller change, no closed-loop replay.",
        "",
        "## Relief Metrics",
        "",
    ]
    show = metrics[
        (metrics["target"] == "relief")
        & (metrics["model"].isin(["baseline", "e15", "soft_blend_03", "soft_blend_05", "dynamic_alpha"]))
    ][["subset", "model", "support", "predicted_positive", "precision", "recall", "f1", "false_positive_rate", "pr_auc"]]
    lines.append(md_table(show))
    lines.extend(["", "## Shape / Risk Proxy Metrics", ""])
    show_regime = regime[
        regime["model"].isin(["baseline", "e15", "soft_blend_03", "soft_blend_05", "dynamic_alpha"])
    ][[
        "subset",
        "model",
        "mean_shape_l2_to_oracle",
        "mean_direction_agree",
        "mean_range_abs_error",
        "false_relief",
        "true_relief_rejected_rate",
        "onset_signflip_mixed_false_relief_rate",
        "lowrisk_pred_relief_count",
    ]]
    lines.append(md_table(show_regime))
    lines.extend(["", "## Dynamic Alpha Disagreement Cells", ""])
    lines.append(md_table(disagreement[disagreement["subset"] == "non_guard10"]))
    lines.extend(["", "## Bootstrap", ""])
    lines.append(md_table(boot[boot["subset"].isin(["all", "non_guard10"])]))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Advance only if dynamic_alpha or soft_blend_03 is stable on non-guard10, reduces false relief versus raw e15, keeps recall, and does not degrade shape/action proxy versus baseline.",
        ]
    )
    (out_dir / "gated_blend_selector_offline_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/wind_prediction/baseline_e15_forecast_behavior_audit_v1/forecast_behavior_bucket_table.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/gated_blend_selector_offline_v0",
    )
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPO_ROOT / args.input)

    values = []
    reasons = []
    alphas = []
    for _, row in df.iterrows():
        alpha, reason = dynamic_alpha(row)
        alphas.append(alpha)
        reasons.append(reason)
        values.append(alpha * norms(row, "e15") + (1.0 - alpha) * norms(row, "baseline"))
    add_selector(df, "dynamic_alpha", values, reasons=reasons, alphas=alphas)

    add_selector(df, "soft_blend_03", [fixed_soft_blend(row, 0.3) for _, row in df.iterrows()])
    add_selector(df, "soft_blend_05", [fixed_soft_blend(row, 0.5) for _, row in df.iterrows()])

    models = ["baseline", "e15", "soft_blend_03", "soft_blend_05", "dynamic_alpha"]
    metrics, regime, disagreement = summarize_models(df, models)
    boot = bootstrap(df, models)

    df.to_csv(out_dir / "gated_blend_selector_bucket_table.csv", index=False)
    metrics.to_csv(out_dir / "gated_blend_selector_regime_table.csv", index=False)
    regime.to_csv(out_dir / "gated_blend_selector_action_proxy_table.csv", index=False)
    disagreement.to_csv(out_dir / "gated_blend_selector_disagreement_table.csv", index=False)
    boot.to_csv(out_dir / "gated_blend_selector_bootstrap.csv", index=False)
    write_summary(out_dir, metrics, regime, disagreement, boot)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
