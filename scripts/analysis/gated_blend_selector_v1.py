#!/usr/bin/env python3
"""Offline audit for baseline/e15 blend selector v1 candidates.

This is a forecast-selector audit only. It consumes the fixed forecast behavior
bucket table and does not train models, run closed-loop control, or change
controller settings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RELIEF_THRESHOLD = 0.95
WEAK_RELIEF_THRESHOLD = 0.98
STRONG_RELIEF_THRESHOLD = 0.85
INTENSIFY_THRESHOLD = 1.05
STRONG_INTENSIFY_THRESHOLD = 1.15
EPS = 1e-6


def norms(row: pd.Series, prefix: str) -> np.ndarray:
    return np.asarray([row[f"{prefix}_b0"], row[f"{prefix}_b1"], row[f"{prefix}_b2"]], dtype=float)


def ratios(arr: np.ndarray) -> np.ndarray:
    b0 = max(float(arr[0]), EPS)
    return np.asarray([1.0, float(arr[1]) / b0, float(arr[2]) / b0], dtype=float)


def pred_relief(arr: np.ndarray, threshold: float = RELIEF_THRESHOLD) -> bool:
    r = ratios(arr)
    return bool(min(r[1], r[2]) < threshold)


def relief_blocks_count(arr: np.ndarray, threshold: float = RELIEF_THRESHOLD) -> int:
    r = ratios(arr)
    return int(r[1] < threshold) + int(r[2] < threshold)


def pred_weak_relief(arr: np.ndarray) -> bool:
    return pred_relief(arr, WEAK_RELIEF_THRESHOLD)


def pred_confirmed_relief(arr: np.ndarray) -> bool:
    return relief_blocks_count(arr) >= 2


def pred_strong_relief(arr: np.ndarray) -> bool:
    return pred_relief(arr, STRONG_RELIEF_THRESHOLD)


def pred_intensify(arr: np.ndarray, threshold: float = INTENSIFY_THRESHOLD) -> bool:
    r = ratios(arr)
    return bool(max(r[1], r[2]) > threshold)


def pred_strong_intensify(arr: np.ndarray) -> bool:
    return pred_intensify(arr, STRONG_INTENSIFY_THRESHOLD)


def swing_proxy(b: np.ndarray, e: np.ndarray) -> bool:
    for arr in (b, e):
        r = ratios(arr)
        if (r[1] - 1.0) * (r[2] - r[1]) < -0.01:
            return True
    return bool(np.sign(b[2] - b[0]) * np.sign(e[2] - e[0]) < 0)


def relief_score(arr: np.ndarray) -> float:
    r = ratios(arr)
    return float(max(0.0, 1.0 - min(r[1], r[2])))


def intensify_score(arr: np.ndarray) -> float:
    r = ratios(arr)
    return float(max(0.0, max(r[1], r[2]) - 1.0))


def direction_agree(pred: np.ndarray, oracle: np.ndarray) -> float:
    return float(np.mean(np.sign(np.diff(pred)) == np.sign(np.diff(oracle))))


def range_value(arr: np.ndarray) -> float:
    return float(np.max(arr) - np.min(arr))


def range_ratio(pred: np.ndarray, oracle: np.ndarray) -> float:
    oracle_range = range_value(oracle)
    if oracle_range < 0.05:
        return float("nan")
    return range_value(pred) / oracle_range


def apply_risk_floor(b: np.ndarray, e: np.ndarray, alpha_candidate: float, floor_ratio: float) -> tuple[float, float]:
    allowed = float(alpha_candidate)
    for i in range(3):
        if b[i] <= EPS or e[i] >= b[i] * floor_ratio:
            continue
        max_alpha = (b[i] * (1.0 - floor_ratio)) / max(b[i] - e[i], EPS)
        allowed = min(allowed, max_alpha)
    alpha_final = float(np.clip(allowed, 0.0, alpha_candidate))
    return alpha_final, float(alpha_candidate - alpha_final)


def fixed_soft03(row: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    b = norms(row, "baseline")
    e = norms(row, "e15")
    alpha = 0.3 if pred_relief(e) and not pred_intensify(b) else 0.0
    return alpha * e + (1.0 - alpha) * b, {
        "alpha_candidate": alpha,
        "alpha_final": alpha,
        "reason": "soft03_relief_and_not_base_intensify" if alpha > 0 else "soft03_blocked",
    }


def dynamic_alpha(row: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    b = norms(row, "baseline")
    e = norms(row, "e15")
    b_relief = pred_relief(b)
    e_relief = pred_relief(e)
    b_intensify = pred_intensify(b)
    if pred_strong_intensify(b):
        alpha, reason = 0.0, "dynamic_strong_baseline_intensify"
    elif b_intensify and e_relief:
        alpha, reason = 0.0, "dynamic_baseline_intensify_vs_e15_relief_conflict"
    elif swing_proxy(b, e):
        alpha, reason = 0.0, "dynamic_swing_proxy_blocks"
    elif b_relief and e_relief:
        alpha, reason = 0.3, "dynamic_both_relief"
    elif pred_strong_relief(e) and not b_relief and not b_intensify:
        alpha, reason = 0.3, "dynamic_e15_strong_relief_baseline_neutral"
    elif e_relief and not b_relief and not b_intensify:
        alpha, reason = 0.1, "dynamic_e15_relief_baseline_neutral"
    else:
        alpha, reason = 0.0, "dynamic_no_rule_match"
    return alpha * e + (1.0 - alpha) * b, {
        "alpha_candidate": alpha,
        "alpha_final": alpha,
        "reason": reason,
    }


def risk_floor_blend_v1(row: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    b = norms(row, "baseline")
    e = norms(row, "e15")
    b_relief = pred_relief(b)
    e_relief = pred_relief(e)
    b_intensify = pred_intensify(b)
    if pred_strong_intensify(b):
        alpha, reason = 0.0, "risk_floor_strong_baseline_intensify"
    elif b_intensify and e_relief:
        alpha, reason = 0.0, "risk_floor_baseline_intensify_vs_e15_relief_conflict"
    elif swing_proxy(b, e):
        alpha, reason = 0.1, "risk_floor_swing_proxy_alpha01"
    elif b_relief and e_relief:
        alpha, reason = 0.5, "risk_floor_both_relief_alpha05"
    elif pred_strong_relief(e) and not b_relief and not b_intensify:
        alpha, reason = 0.3, "risk_floor_e15_strong_relief_baseline_flat_alpha03"
    elif e_relief and not b_relief and not b_intensify:
        alpha, reason = 0.1, "risk_floor_e15_relief_baseline_flat_alpha01"
    else:
        alpha, reason = 0.0, "risk_floor_no_rule_match"
    floor_ratio = 1.0
    floor_triggered = 0
    shrink = 0.0
    if alpha > 0.0:
        floor_ratio = 0.95
        if pred_confirmed_relief(b):
            floor_ratio = 0.85
        elif pred_weak_relief(b):
            floor_ratio = 0.90
        alpha_final, shrink = apply_risk_floor(b, e, alpha, floor_ratio)
        floor_triggered = int(shrink > 1e-9)
        if floor_triggered:
            reason = f"{reason}_floor_shrunk"
    else:
        alpha_final = 0.0
    return alpha_final * e + (1.0 - alpha_final) * b, {
        "alpha_candidate": alpha,
        "alpha_final": alpha_final,
        "floor_ratio": floor_ratio,
        "floor_triggered": floor_triggered,
        "floor_shrink_amount": shrink,
        "reason": reason,
    }


def confirmed_relief_alpha_v1(row: pd.Series) -> tuple[np.ndarray, dict[str, object]]:
    b = norms(row, "baseline")
    e = norms(row, "e15")
    e_blocks = relief_blocks_count(e)
    b_relief = pred_relief(b)
    b_intensify = pred_intensify(b)
    e_relief = pred_relief(e)
    e_strong = pred_strong_relief(e)
    swing = swing_proxy(b, e)
    baseline_flat = not b_relief and not b_intensify
    if b_intensify:
        alpha, reason = 0.0, "confirmed_baseline_intensify_blocks"
    elif swing:
        alpha, reason = 0.0, "confirmed_swing_proxy_blocks"
    elif b_relief and e_blocks >= 2:
        alpha, reason = 0.5, "confirmed_both_relief_two_blocks_alpha05"
    elif e_strong and baseline_flat:
        alpha, reason = 0.3, "confirmed_e15_strong_relief_baseline_flat_alpha03"
    elif e_relief and baseline_flat:
        alpha, reason = 0.0, "confirmed_single_block_relief_rejected"
    else:
        alpha, reason = 0.0, "confirmed_no_rule_match"
    return alpha * e + (1.0 - alpha) * b, {
        "alpha_candidate": alpha,
        "alpha_final": alpha,
        "e15_relief_blocks_count": e_blocks,
        "confirmation_pass": int(alpha > 0.0),
        "reason": reason,
    }


SELECTORS = {
    "soft_blend_03": fixed_soft03,
    "dynamic_alpha": dynamic_alpha,
    "risk_floor_blend_v1": risk_floor_blend_v1,
    "confirmed_relief_alpha_v1": confirmed_relief_alpha_v1,
}


def add_selector(df: pd.DataFrame, name: str, fn) -> None:
    arrs = []
    meta = []
    for _, row in df.iterrows():
        arr, info = fn(row)
        arrs.append(arr)
        meta.append(info)
    mat = np.vstack(arrs)
    oracle = df[["oracle_b0", "oracle_b1", "oracle_b2"]].to_numpy(dtype=float)
    new_cols: dict[str, object] = {
        f"{name}_b0": mat[:, 0],
        f"{name}_b1": mat[:, 1],
        f"{name}_b2": mat[:, 2],
        f"{name}_pred_relief": [pred_relief(x) for x in mat],
        f"{name}_pred_intensify": [pred_intensify(x) for x in mat],
        f"{name}_relief_score": [relief_score(x) for x in mat],
        f"{name}_intensify_score": [intensify_score(x) for x in mat],
        f"{name}_shape_l2_to_oracle": np.linalg.norm(mat - oracle, axis=1),
        f"{name}_direction_agree": [direction_agree(mat[i], oracle[i]) for i in range(len(df))],
        f"{name}_range_abs_error": [
            abs(range_value(mat[i]) - range_value(oracle[i])) for i in range(len(df))
        ],
        f"{name}_range_ratio_to_oracle": [range_ratio(mat[i], oracle[i]) for i in range(len(df))],
    }
    keys = sorted({k for d in meta for k in d})
    for key in keys:
        new_cols[f"{name}_{key}"] = [d.get(key, np.nan if key != "reason" else "") for d in meta]
    extra = pd.DataFrame(new_cols, index=df.index)
    for col in extra.columns:
        df[col] = extra[col]


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


def classify(df: pd.DataFrame, model: str, target: str) -> dict[str, float | int | str]:
    y = df[f"true_{target}"].astype(bool).to_numpy()
    p = df[f"{model}_pred_{target}"].astype(bool).to_numpy()
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision + recall) else float("nan")
    return {
        "model": model,
        "target": target,
        "n": len(df),
        "support": int(y.sum()),
        "predicted_positive": int(p.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / (fp + tn) if fp + tn else float("nan"),
        "pr_auc": average_precision(y, df[f"{model}_{target}_score"].to_numpy(dtype=float)),
    }


def subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "all": df,
        "non_guard10": df[~df["is_guard10"]],
        "guard10_only": df[df["is_guard10"]],
        "broader20": df[df.get("is_broader20", False)],
    }


def summarize(df: pd.DataFrame, models: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    regime_rows = []
    alpha_rows = []
    for subset, sub in subsets(df).items():
        if sub.empty:
            continue
        onset_signflip = sub[sub["case_family"].isin(["onset", "signflip"]) | sub["true_trend_regime"].eq("mixed_swing")]
        for model in models:
            for target in ("relief", "intensify"):
                row = classify(sub, model, target)
                row["subset"] = subset
                metric_rows.append(row)
            pred_relief = sub[f"{model}_pred_relief"].astype(bool)
            true_relief = sub["true_relief"].astype(bool)
            regime_rows.append(
                {
                    "subset": subset,
                    "model": model,
                    "n": len(sub),
                    "mean_shape_l2_to_oracle": float(sub[f"{model}_shape_l2_to_oracle"].mean()),
                    "mean_direction_agree": float(sub[f"{model}_direction_agree"].mean()),
                    "mean_range_abs_error": float(sub[f"{model}_range_abs_error"].mean()),
                    "false_relief": int((~true_relief & pred_relief).sum()),
                    "false_relief_rate": float((~true_relief & pred_relief).mean()),
                    "true_relief_rejected": int((true_relief & ~pred_relief).sum()),
                    "true_relief_rejected_rate": float((true_relief & ~pred_relief).sum() / max(int(true_relief.sum()), 1)),
                    "onset_signflip_mixed_false_relief": int((~onset_signflip["true_relief"].astype(bool) & onset_signflip[f"{model}_pred_relief"].astype(bool)).sum()) if not onset_signflip.empty else 0,
                    "lowrisk_pred_relief_count": int(((sub["case_family"] == "lowrisk") & pred_relief).sum()),
                }
            )
            if f"{model}_alpha_final" in sub.columns:
                alpha = sub[f"{model}_alpha_final"].fillna(0.0)
                alpha_rows.append(
                    {
                        "subset": subset,
                        "model": model,
                        "n": len(sub),
                        "alpha_gt0": int((alpha > 0).sum()),
                        "alpha_0": int((alpha == 0).sum()),
                        "alpha_01": int(np.isclose(alpha, 0.1).sum()),
                        "alpha_03": int(np.isclose(alpha, 0.3).sum()),
                        "alpha_05": int(np.isclose(alpha, 0.5).sum()),
                        "mean_alpha": float(alpha.mean()),
                        "floor_triggered": int(sub.get(f"{model}_floor_triggered", pd.Series(0, index=sub.index)).fillna(0).sum()),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(regime_rows), pd.DataFrame(alpha_rows)


def bootstrap(df: pd.DataFrame, models: list[str], n_iter: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(20260519)
    rows = []
    for subset, sub in subsets(df).items():
        if sub.empty:
            continue
        sub = sub.reset_index(drop=True)
        case_to_idx = {
            case: np.flatnonzero(sub["case_id"].to_numpy() == case)
            for case in sorted(sub["case_id"].unique())
        }
        cases = np.asarray(list(case_to_idx))
        for model in models:
            true_relief = sub["true_relief"].astype(bool).to_numpy()
            pred_relief = sub[f"{model}_pred_relief"].astype(bool).to_numpy()
            shape_l2 = sub[f"{model}_shape_l2_to_oracle"].to_numpy(dtype=float)
            values = []
            for _ in range(n_iter):
                sampled = rng.choice(cases, size=len(cases), replace=True)
                idx = np.concatenate([case_to_idx[c] for c in sampled])
                y = true_relief[idx]
                p = pred_relief[idx]
                tp = np.sum(y & p)
                fp = np.sum(~y & p)
                fn = np.sum(y & ~p)
                precision = tp / (tp + fp) if tp + fp else np.nan
                recall = tp / max(tp + fn, 1)
                values.append((precision, recall, float(np.mean(shape_l2[idx]))))
            arr = np.asarray(values, dtype=float)
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
                    "shape_l2_mean": float(np.nanmean(arr[:, 2])),
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


def write_summary(out_dir: Path, metrics: pd.DataFrame, regime: pd.DataFrame, alpha: pd.DataFrame, boot: pd.DataFrame) -> None:
    lines = [
        "# Gated Blend Selector v1 Offline Audit",
        "",
        "Offline forecast-selector audit only: no training, no controller change, no closed-loop replay.",
        "",
        "## Relief Metrics",
        "",
    ]
    models = ["baseline", "e15", "soft_blend_03", "dynamic_alpha", "risk_floor_blend_v1", "confirmed_relief_alpha_v1"]
    show = metrics[(metrics["target"] == "relief") & metrics["model"].isin(models)][
        ["subset", "model", "support", "predicted_positive", "precision", "recall", "f1", "false_positive_rate", "pr_auc"]
    ]
    lines.append(md_table(show))
    lines.extend(["", "## Shape / Risk Proxy", ""])
    show_regime = regime[regime["model"].isin(models)][
        ["subset", "model", "mean_shape_l2_to_oracle", "mean_direction_agree", "mean_range_abs_error", "false_relief", "true_relief_rejected_rate", "onset_signflip_mixed_false_relief", "lowrisk_pred_relief_count"]
    ]
    lines.append(md_table(show_regime))
    lines.extend(["", "## Alpha Distribution", ""])
    lines.append(md_table(alpha[alpha["model"].isin(models)]))
    lines.extend(["", "## Bootstrap", ""])
    lines.append(md_table(boot[boot["model"].isin(models) & boot["subset"].isin(["all", "non_guard10", "broader20"])]))
    lines.extend(["", "## Decision Note", ""])
    lines.append("Closed-loop should only be run for selectors that improve non-guard10/broader20 false-relief behavior without collapsing true-relief recall.")
    (out_dir / "selector_v1_offline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/wind_prediction/baseline_e15_forecast_behavior_audit_v1/forecast_behavior_bucket_table.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/wind_prediction/gated_blend_selector_v1",
    )
    args = parser.parse_args()
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPO_ROOT / args.input)
    broader_path = REPO_ROOT / "outputs/wind_prediction/f60_relief_e15_holdout_validation_v1/f60_relief_e15_holdout_cases.csv"
    if broader_path.exists():
        broader = pd.read_csv(broader_path)
        broader_keys = {
            (str(r.case_id), str(pd.Timestamp(r.timestamp)))
            for r in broader.itertuples(index=False)
        }
        df["is_broader20"] = [
            (str(r.case_id), str(pd.Timestamp(r.timestamp))) in broader_keys
            for r in df.itertuples(index=False)
        ]
    else:
        df["is_broader20"] = False

    for selector, fn in SELECTORS.items():
        add_selector(df, selector, fn)

    models = ["baseline", "e15", "soft_blend_03", "dynamic_alpha", "risk_floor_blend_v1", "confirmed_relief_alpha_v1"]
    metrics, regime, alpha = summarize(df, models)
    boot = bootstrap(df, models)

    df.to_csv(out_dir / "selector_v1_alpha_telemetry.csv", index=False)
    metrics.to_csv(out_dir / "selector_v1_offline_table.csv", index=False)
    regime.to_csv(out_dir / "selector_v1_regime_table.csv", index=False)
    alpha.to_csv(out_dir / "selector_v1_alpha_table.csv", index=False)
    boot.to_csv(out_dir / "selector_v1_bootstrap.csv", index=False)
    # Placeholder action proxy table: pressure-shape proxy is the safe offline
    # proxy for this pass. Full planner replay is intentionally left out until a
    # selector survives the non-guard10 forecast-behavior gate.
    regime.to_csv(out_dir / "selector_v1_action_proxy_table.csv", index=False)
    write_summary(out_dir, metrics, regime, alpha, boot)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
