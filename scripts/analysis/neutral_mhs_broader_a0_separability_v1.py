#!/usr/bin/env python3
"""Summarize broader neutral-MHS A0 runs and audit clean-start separability."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_moderate_high_steady_broader_v1"
RAW_DIR = BASE / "raw_tables"
CASEBOOKS = BASE / "casebooks"
A0 = BASE / "a0_1h" / "casebook_summary.csv"
FEATURES = RAW_DIR / "neutral_mhs_broader_selected_features.csv"
TS_DIR = BASE / "a0_1h" / "timeseries"


def auc_score(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    x = np.asarray(x).astype(float)
    mask = np.isfinite(x)
    y = y[mask]
    x = x[mask]
    pos = x[y == 1]
    neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg))
        wins += 0.5 * float(np.sum(p == neg))
    return wins / float(len(pos) * len(neg))


def safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series(np.full(len(df), default), index=df.index)


def max_abs_axis(df: pd.DataFrame) -> pd.Series:
    pitch = safe_col(df, "pitch_deg").abs()
    roll = safe_col(df, "roll_deg").abs()
    return pd.concat([pitch, roll], axis=1).max(axis=1)


def window(df: pd.DataFrame, seconds: float) -> pd.DataFrame:
    if "t_s" not in df.columns:
        return df.iloc[: max(1, min(len(df), int(seconds)))]
    t = pd.to_numeric(df["t_s"], errors="coerce")
    out = df.loc[t <= seconds]
    if out.empty:
        return df.iloc[:1]
    return out


def summarize_timeseries(case_id: str) -> dict[str, object]:
    matches = sorted(TS_DIR.glob(f"{case_id}_*_timeseries.csv"))
    if not matches:
        return {"case_id": case_id, "timeseries_found": 0}
    path = matches[0]
    df = pd.read_csv(path)
    axis = max_abs_axis(df)
    w60 = window(df, 60.0)
    w300 = window(df, 300.0)
    axis60 = max_abs_axis(w60)
    axis300 = max_abs_axis(w300)
    first = df.iloc[0]
    return {
        "case_id": case_id,
        "timeseries_found": 1,
        "initial_pitch_abs_deg": abs(float(first.get("pitch_deg", np.nan))),
        "initial_roll_abs_deg": abs(float(first.get("roll_deg", np.nan))),
        "initial_max_axis_deg": float(axis.iloc[0]),
        "max_axis_0_60s_deg": float(axis60.max()),
        "p95_axis_0_60s_deg": float(np.percentile(axis60, 95)),
        "mean_axis_0_60s_deg": float(axis60.mean()),
        "max_axis_0_300s_deg": float(axis300.max()),
        "p95_axis_0_300s_deg": float(np.percentile(axis300, 95)),
        "mean_axis_0_300s_deg": float(axis300.mean()),
        "initial_backlog_kg": float(first.get("pump_total_backlog_kg", np.nan)),
        "max_backlog_0_60s_kg": float(safe_col(w60, "pump_total_backlog_kg").max()),
        "initial_cmd_gap_kg": float(first.get("cmd_gap_kg", np.nan)),
        "max_cmd_gap_0_60s_kg": float(safe_col(w60, "cmd_gap_kg").max()),
        "initial_fullspeed_any": float(first.get("pump_fullspeed_any", 0.0)),
        "fullspeed_any_0_60s": float(safe_col(w60, "pump_fullspeed_any").max()),
        "initial_safety_active": float(first.get("preview_primary_safety_active", 0.0)),
        "safety_active_ratio_0_60s": float(safe_col(w60, "preview_primary_safety_active").mean()),
        "initial_safety_fallback": float(first.get("preview_primary_safety_fallback", 0.0)),
        "safety_fallback_ratio_0_60s": float(safe_col(w60, "preview_primary_safety_fallback").mean()),
        "source_timeseries": str(path),
    }


def best_threshold(df: pd.DataFrame, feature: str) -> dict[str, object]:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = df["clean_economy_opportunity"].astype(bool)
    valid = x.notna()
    x = x[valid]
    y = y[valid]
    best: dict[str, object] | None = None
    for direction in ("le", "ge"):
        for threshold in sorted(set(float(v) for v in x.to_numpy())):
            pred = x <= threshold if direction == "le" else x >= threshold
            tp = int((pred & y).sum())
            fp = int((pred & ~y).sum())
            fn = int((~pred & y).sum())
            tn = int((~pred & ~y).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            fpr = fp / (fp + tn) if fp + tn else 0.0
            score = (precision, -fpr, recall, tp)
            row = {
                "feature": feature,
                "direction": direction,
                "threshold": float(threshold),
                "tp_clean": tp,
                "fp_heavy": fp,
                "fn_clean": fn,
                "tn_heavy": tn,
                "precision": float(precision),
                "recall": float(recall),
                "fpr": float(fpr),
                "score_precision": float(score[0]),
                "score_neg_fpr": float(score[1]),
                "score_recall": float(score[2]),
                "score_tp": int(score[3]),
            }
            if best is None or score > (
                best["score_precision"],
                best["score_neg_fpr"],
                best["score_recall"],
                best["score_tp"],
            ):
                best = row
    assert best is not None
    return best


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.4g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)

    a0 = pd.read_csv(A0)
    feat = pd.read_csv(FEATURES).copy()
    feat["timestamp"] = feat["future_start"].astype(str)
    compact = feat[
        [
            "timestamp",
            "split",
            "early_max_ms",
            "peak_to_late_drop_ms",
            "near_range_ms",
            "far_range_ms",
            "dir_shift_abs_max_deg",
            "selection_score",
        ]
    ]
    df = a0.merge(compact, on="timestamp", how="left")
    df["p95_max_axis_deg"] = df[["primary_pitch_p95", "primary_roll_p95"]].abs().max(axis=1)
    df["material_a0_pump"] = df["primary_pump_work_m3"] >= 100.0
    df["clean_economy_opportunity"] = (
        (df["primary_pump_work_m3"] >= 100.0)
        & (df["primary_safety_fallback_ratio"] <= 0.01)
        & (df["p95_max_axis_deg"] <= 4.0)
    )
    df["safety_margin_heavy"] = (
        (df["primary_safety_fallback_ratio"] >= 0.05)
        | (df["p95_max_axis_deg"] >= 5.0)
    )

    state = pd.DataFrame([summarize_timeseries(str(cid)) for cid in df["case_id"]])
    merged = df.merge(state, on="case_id", how="left")
    merged.to_csv(RAW_DIR / "neutral_mhs_broader_a0_opportunity_per_case.csv", index=False)

    split_summary = (
        merged.groupby("split")
        .agg(
            cases=("case_id", "size"),
            total_a0_pump_m3=("primary_pump_work_m3", "sum"),
            clean_cases=("clean_economy_opportunity", "sum"),
            safety_margin_heavy_cases=("safety_margin_heavy", "sum"),
            median_initial_max_axis_deg=("initial_max_axis_deg", "median"),
            median_p95_max_axis_deg=("p95_max_axis_deg", "median"),
            median_fallback_ratio=("primary_safety_fallback_ratio", "median"),
            median_a0_pump_m3=("primary_pump_work_m3", "median"),
        )
        .reset_index()
    )
    split_summary["clean_case_share"] = split_summary["clean_cases"] / split_summary["cases"]
    split_summary.to_csv(RAW_DIR / "neutral_mhs_broader_a0_opportunity_summary.csv", index=False)

    features = [
        "initial_max_axis_deg",
        "max_axis_0_60s_deg",
        "p95_axis_0_60s_deg",
        "mean_axis_0_60s_deg",
        "max_axis_0_300s_deg",
        "p95_axis_0_300s_deg",
        "mean_axis_0_300s_deg",
        "initial_backlog_kg",
        "max_backlog_0_60s_kg",
        "initial_cmd_gap_kg",
        "max_cmd_gap_0_60s_kg",
        "initial_fullspeed_any",
        "fullspeed_any_0_60s",
        "initial_safety_active",
        "safety_active_ratio_0_60s",
        "initial_safety_fallback",
        "safety_fallback_ratio_0_60s",
    ]
    y = merged["clean_economy_opportunity"].astype(int).to_numpy()
    auc_rows = []
    for feature in features:
        x = pd.to_numeric(merged[feature], errors="coerce").to_numpy()
        auc = auc_score(y, x)
        auc_rows.append(
            {
                "feature": feature,
                "auc_clean_high": auc,
                "auc_clean_low": 1.0 - auc if np.isfinite(auc) else np.nan,
                "best_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                "clean_mean": float(pd.to_numeric(merged.loc[merged["clean_economy_opportunity"], feature], errors="coerce").mean()),
                "heavy_mean": float(pd.to_numeric(merged.loc[merged["safety_margin_heavy"], feature], errors="coerce").mean()),
            }
        )
    auc_table = pd.DataFrame(auc_rows).sort_values("best_auc", ascending=False)
    auc_table.to_csv(RAW_DIR / "neutral_mhs_broader_current_state_feature_auc.csv", index=False)

    thresholds = pd.DataFrame([best_threshold(merged, feature) for feature in features]).sort_values(
        ["precision", "fpr", "recall", "tp_clean"],
        ascending=[False, True, False, False],
    )
    thresholds.to_csv(RAW_DIR / "neutral_mhs_broader_current_state_candidate_thresholds.csv", index=False)

    clean = merged[merged["clean_economy_opportunity"]].copy()
    clean_casebook = pd.DataFrame(
        {
            "case_id": [f"neutral_mhs_broader_clean_{i+1:02d}" for i in range(len(clean))],
            "timestamp": clean["timestamp"].astype(str).values,
            "label": clean["label"].astype(str).values + " | clean_a0_opportunity",
        }
    )
    clean_casebook.to_csv(CASEBOOKS / "neutral_mhs_broader_clean_cases.csv", index=False)

    best_auc = auc_table.iloc[0]
    best_rule = thresholds.iloc[0]
    md = [
        "# Neutral MHS Broader A0 Opportunity and Separability Audit",
        "",
        "This audit applies the fixed broader neutral-MHS casebook rule, then checks whether current-state information can safely decide when the automatic economy budget should open.",
        "",
        "## A0 Opportunity Split",
        "",
        markdown_table(split_summary),
        "",
        "## Best Separability Signal",
        "",
        f"- best feature: `{best_auc['feature']}` with AUC {best_auc['best_auc']:.3f}",
        f"- best low-misfire rule: `{best_rule['feature']}` {best_rule['direction']} {best_rule['threshold']:.3f}",
        f"- clean caught: {int(best_rule['tp_clean'])}/{int(merged['clean_economy_opportunity'].sum())}",
        f"- heavy misfires: {int(best_rule['fp_heavy'])}/{int((~merged['clean_economy_opportunity']).sum())}",
        f"- precision {best_rule['precision']:.3f}, recall {best_rule['recall']:.3f}, FPR {best_rule['fpr']:.3f}",
        "",
        "## Interpretation",
        "",
        "The broader set deliberately includes lookalikes. A deployable policy must open only on the clean-start cases and abstain on the heavy current-posture cases.",
        "If the implemented conservative threshold `initial_max_axis_deg <= 0.06` remains close to the zero-misfire rule, the next validation is the learned automatic run on this same broader casebook.",
        "",
        "## Files",
        "",
        "- `raw_tables/neutral_mhs_broader_a0_opportunity_per_case.csv`",
        "- `raw_tables/neutral_mhs_broader_a0_opportunity_summary.csv`",
        "- `raw_tables/neutral_mhs_broader_current_state_feature_auc.csv`",
        "- `raw_tables/neutral_mhs_broader_current_state_candidate_thresholds.csv`",
        "- `casebooks/neutral_mhs_broader_clean_cases.csv`",
    ]
    (BASE / "neutral_mhs_broader_a0_separability_decision.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    print(split_summary.to_string(index=False))
    print()
    print(auc_table.head(5).to_string(index=False))
    print()
    print(thresholds.head(5).to_string(index=False))
    print(f"clean cases: {len(clean_casebook)}")


if __name__ == "__main__":
    main()
