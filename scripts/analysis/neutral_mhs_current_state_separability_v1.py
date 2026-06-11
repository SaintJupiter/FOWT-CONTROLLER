#!/usr/bin/env python3
"""Check whether current plant state separates clean neutral-MHS opportunities.

The expanded neutral_moderate_high_steady smoke showed a strong action
opportunity on the clean subset, but weak separation from forecast-shape
features alone. This read-only audit tests whether runtime state (current
posture / safety state / backlog) supplies the missing deployment gate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_moderate_high_steady_expanded_v1"
A0_TABLE = BASE / "raw_tables" / "neutral_mhs_expanded_a0_opportunity_per_case.csv"
TS_DIR = BASE / "a0_1h" / "timeseries"
RAW_DIR = BASE / "raw_tables"
PAPER_DIR = BASE / "paper_ready"


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


def summarize_timeseries(case_id: str) -> dict[str, float | str]:
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


def best_threshold_for_clean(
    df: pd.DataFrame, feature: str, directions: Iterable[str] = ("le", "ge")
) -> dict[str, float | str | int]:
    x = pd.to_numeric(df[feature], errors="coerce")
    y = df["clean_economy_opportunity"].astype(bool)
    valid = x.notna()
    x = x[valid]
    y = y[valid]
    best: dict[str, float | str | int] | None = None
    for direction in directions:
        thresholds = sorted(set(float(v) for v in x.to_numpy()))
        for th in thresholds:
            pred = x <= th if direction == "le" else x >= th
            tp = int((pred & y).sum())
            fp = int((pred & ~y).sum())
            fn = int((~pred & y).sum())
            tn = int((~pred & ~y).sum())
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            # Prefer clean separation, then recall, then coverage.
            score = (precision, -fpr, recall, tp)
            row = {
                "feature": feature,
                "direction": direction,
                "threshold": float(th),
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


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)

    per_case = pd.read_csv(A0_TABLE)
    state_rows = [summarize_timeseries(str(cid)) for cid in per_case["case_id"]]
    state = pd.DataFrame(state_rows)
    merged = per_case.merge(state, on="case_id", how="left")
    merged.to_csv(RAW_DIR / "neutral_mhs_current_state_per_case.csv", index=False)

    candidate_features = [
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
    auc_rows = []
    y = merged["clean_economy_opportunity"].astype(int).to_numpy()
    for feature in candidate_features:
        if feature not in merged.columns:
            continue
        x = pd.to_numeric(merged[feature], errors="coerce").to_numpy()
        auc = auc_score(y, x)
        auc_rows.append(
            {
                "feature": feature,
                "auc_clean_high": auc,
                "auc_clean_low": 1.0 - auc if np.isfinite(auc) else np.nan,
                "clean_mean": float(pd.to_numeric(merged.loc[merged["clean_economy_opportunity"], feature], errors="coerce").mean()),
                "heavy_mean": float(pd.to_numeric(merged.loc[merged["safety_margin_heavy"], feature], errors="coerce").mean()),
            }
        )
    auc_table = pd.DataFrame(auc_rows)
    auc_table["best_auc"] = auc_table[["auc_clean_high", "auc_clean_low"]].max(axis=1)
    auc_table = auc_table.sort_values("best_auc", ascending=False)
    auc_table.to_csv(RAW_DIR / "neutral_mhs_current_state_feature_auc.csv", index=False)

    thresholds = []
    for feature in candidate_features:
        if feature in merged.columns:
            thresholds.append(best_threshold_for_clean(merged, feature))
    threshold_table = pd.DataFrame(thresholds).sort_values(
        ["precision", "fpr", "recall", "tp_clean"],
        ascending=[False, True, False, False],
    )
    threshold_table.to_csv(RAW_DIR / "neutral_mhs_current_state_candidate_thresholds.csv", index=False)

    best_auc = auc_table.iloc[0].to_dict()
    best_rule = threshold_table.iloc[0].to_dict()
    clean = merged[merged["clean_economy_opportunity"]]
    heavy = merged[merged["safety_margin_heavy"]]

    md = [
        "# Neutral MHS Current-State Separability Audit",
        "",
        "This read-only audit asks whether the high-saving neutral moderate-high steady subset can be gated by runtime state, not just forecast shape.",
        "",
        "## Key Result",
        "",
        f"- cases: {len(merged)} ({len(clean)} clean economy opportunities, {len(heavy)} safety-margin-heavy cases)",
        f"- best current-state feature: `{best_auc['feature']}` with AUC {best_auc['best_auc']:.3f}",
        f"- best zero/low-misfire candidate rule: `{best_rule['feature']}` {best_rule['direction']} {best_rule['threshold']:.3f}",
        f"  - clean caught: {int(best_rule['tp_clean'])}/{len(clean)}",
        f"  - heavy misfires: {int(best_rule['fp_heavy'])}/{len(heavy)}",
        f"  - precision: {best_rule['precision']:.3f}, recall: {best_rule['recall']:.3f}, FPR: {best_rule['fpr']:.3f}",
        "",
        "## Interpretation",
        "",
    ]
    if float(best_auc["best_auc"]) >= 0.95 and int(best_rule["fp_heavy"]) == 0:
        md.extend(
            [
                "Runtime state cleanly separates the safe high-saving subset from safety-heavy lookalikes.",
                "This means neutral_moderate_high_steady is no longer blocked by action effect; it needs a conservative automatic gate combining forecast shape + current-state safety margin.",
            ]
        )
    elif float(best_auc["best_auc"]) >= 0.8:
        md.extend(
            [
                "Runtime state provides useful separation, but not enough to promote directly without a guarded mixed-set validation.",
                "The next step should be a default-off auto gate and a short mixed 32-case run, not parameter tuning.",
            ]
        )
    else:
        md.extend(
            [
                "Runtime state does not cleanly separate clean opportunities from safety-heavy lookalikes.",
                "This regime should remain a candidate/operator Pareto mode unless a better detector is found.",
            ]
        )
    md.extend(
        [
            "",
            "## Files",
            "",
            "- `raw_tables/neutral_mhs_current_state_per_case.csv`",
            "- `raw_tables/neutral_mhs_current_state_feature_auc.csv`",
            "- `raw_tables/neutral_mhs_current_state_candidate_thresholds.csv`",
        ]
    )
    (BASE / "current_state_separability_decision.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (PAPER_DIR / "neutral_mhs_current_state_separability_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    print("cases", len(merged))
    print("best_auc_feature", best_auc["feature"], f"{best_auc['best_auc']:.3f}")
    print(
        "best_rule",
        best_rule["feature"],
        best_rule["direction"],
        f"{best_rule['threshold']:.3f}",
        "tp",
        int(best_rule["tp_clean"]),
        "fp",
        int(best_rule["fp_heavy"]),
        "precision",
        f"{best_rule['precision']:.3f}",
        "recall",
        f"{best_rule['recall']:.3f}",
    )


if __name__ == "__main__":
    main()
