#!/usr/bin/env python3
"""Summarize expanded neutral moderate-high steady fixed-action smoke."""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_moderate_high_steady_expanded_v1"
OUT_RAW = BASE / "raw_tables"
OUT_PAPER = BASE / "paper_ready"

A0 = BASE / "a0_1h" / "casebook_summary.csv"
BUDGET = BASE / "budget100_clean_1h" / "casebook_summary.csv"
A0_OPP = OUT_RAW / "neutral_mhs_expanded_a0_opportunity_per_case.csv"
FEATURES = OUT_RAW / "neutral_moderate_high_steady_expanded_selected_features.csv"


FEATURE_COLS = [
    "early_max_ms",
    "near_max_ms",
    "far_max_ms",
    "peak_to_late_drop_ms",
    "near_range_ms",
    "far_range_ms",
    "block_range_0_60_ms",
    "block_slope_0_60_ms",
    "block_slope_60_120_ms",
    "dir_shift_abs_max_deg",
    "near_dir_shift_abs_max_deg",
    "far_dir_shift_abs_max_deg",
    "future_speed_ramp_max_ms",
    "future_vector_change_max_ms",
    "vector_reversal_cosine",
    "selection_score",
]


def auc(values: pd.Series, labels: pd.Series) -> tuple[float, str]:
    data = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "y": labels.astype(int)}).dropna()
    pos = data[data["y"] == 1]["v"].tolist()
    neg = data[data["y"] == 0]["v"].tolist()
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
    auc_ge = wins / total
    if auc_ge >= 0.5:
        return auc_ge, "ge"
    return 1.0 - auc_ge, "le"


def compact(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [
        "timestamp",
        "case_id",
        "primary_pump_work_m3",
        "primary_pitch_p95",
        "primary_roll_p95",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
    ]
    out = df[cols].copy()
    return out.rename(columns={c: f"{prefix}_{c}" for c in cols if c != "timestamp"})


def main() -> None:
    OUT_RAW.mkdir(parents=True, exist_ok=True)
    OUT_PAPER.mkdir(parents=True, exist_ok=True)
    a0 = pd.read_csv(A0)
    budget = pd.read_csv(BUDGET)
    opp = pd.read_csv(A0_OPP)
    features = pd.read_csv(FEATURES)
    features = features.copy()
    features["timestamp"] = features["future_start"].astype(str)

    clean_ts = set(opp[opp["clean_economy_opportunity"]]["timestamp"].astype(str))
    a0_clean = a0[a0["timestamp"].astype(str).isin(clean_ts)]
    merged = compact(a0_clean, "a0").merge(compact(budget, "budget100"), on="timestamp", how="inner")
    merged["pump_saved_m3"] = merged["a0_primary_pump_work_m3"] - merged["budget100_primary_pump_work_m3"]
    merged["pump_saving_pct"] = 100.0 * merged["pump_saved_m3"] / merged["a0_primary_pump_work_m3"].clip(lower=1e-9)
    merged["a0_p95_max_axis_deg"] = merged[["a0_primary_pitch_p95", "a0_primary_roll_p95"]].abs().max(axis=1)
    merged["budget100_p95_max_axis_deg"] = merged[
        ["budget100_primary_pitch_p95", "budget100_primary_roll_p95"]
    ].abs().max(axis=1)
    merged["delta_p95_max_axis_deg"] = merged["budget100_p95_max_axis_deg"] - merged["a0_p95_max_axis_deg"]
    merged["delta_fallback_ratio"] = (
        merged["budget100_primary_safety_fallback_ratio"] - merged["a0_primary_safety_fallback_ratio"]
    )
    merged.to_csv(OUT_RAW / "neutral_mhs_expanded_clean_budget100_delta.csv", index=False)

    total_a0 = float(merged["a0_primary_pump_work_m3"].sum())
    total_budget = float(merged["budget100_primary_pump_work_m3"].sum())
    saved = total_a0 - total_budget
    saving_pct = 100.0 * saved / max(total_a0, 1e-9)

    opp_join = opp[["timestamp", "clean_economy_opportunity", "safety_margin_heavy"]].merge(
        features[["timestamp", "split", *FEATURE_COLS]], on="timestamp", how="left"
    )
    label = opp_join["clean_economy_opportunity"].astype(int)
    auc_rows = []
    for col in FEATURE_COLS:
        score, direction = auc(opp_join[col], label)
        auc_rows.append(
            {
                "feature": col,
                "auc_clean_vs_heavy": score,
                "direction_for_clean": direction,
                "clean_mean": float(opp_join.loc[label == 1, col].mean()),
                "heavy_mean": float(opp_join.loc[label == 0, col].mean()),
            }
        )
    auc_table = pd.DataFrame(auc_rows).sort_values("auc_clean_vs_heavy", ascending=False)
    auc_table.to_csv(OUT_RAW / "neutral_mhs_expanded_clean_feature_auc.csv", index=False)

    split_xtab = pd.crosstab(opp_join["split"], opp_join["clean_economy_opportunity"])
    split_xtab.to_csv(OUT_RAW / "neutral_mhs_expanded_split_vs_clean.csv")

    summary = pd.DataFrame(
        [
            {
                "subset": "neutral_mhs_expanded_clean_1h",
                "cases": int(len(merged)),
                "a0_pump_m3": total_a0,
                "budget100_pump_m3": total_budget,
                "pump_saved_m3": saved,
                "pump_saving_pct": saving_pct,
                "max_delta_fallback_ratio": float(merged["delta_fallback_ratio"].max()),
                "mean_delta_p95_max_axis_deg": float(merged["delta_p95_max_axis_deg"].mean()),
                "top_case_share_of_saved": float(
                    merged["pump_saved_m3"].max() / saved if saved > 1e-9 else 0.0
                ),
                "best_preaction_auc_clean_vs_heavy": float(auc_table["auc_clean_vs_heavy"].iloc[0]),
                "best_preaction_feature": str(auc_table["feature"].iloc[0]),
            }
        ]
    )
    summary.to_csv(OUT_RAW / "neutral_mhs_expanded_clean_budget100_summary.csv", index=False)

    text = "\n".join(
        [
            "# Neutral Moderate-High Steady Expanded Fixed-Action Summary",
            "",
            "This is still a smoke result, not a mature automatic controller. It expands the prior 4-case clean subset to 16 clean cases selected by the fixed A0 opportunity screen.",
            "",
            f"- clean cases: {len(merged)}",
            f"- A0 pump: {total_a0:.2f} m3",
            f"- budget100 pump: {total_budget:.2f} m3",
            f"- pump saved: {saved:.2f} m3 ({saving_pct:.2f}%)",
            f"- max fallback-ratio delta: {float(merged['delta_fallback_ratio'].max()):.4f}",
            f"- mean p95 max-axis delta: {float(merged['delta_p95_max_axis_deg'].mean()):.3f} deg",
            f"- top-case share of saved pump: {float(summary['top_case_share_of_saved'].iloc[0]):.3f}",
            "",
            "## Recognition Caveat",
            "",
            f"The best pre-action shape feature separates clean vs safety-heavy with AUC {float(auc_table['auc_clean_vs_heavy'].iloc[0]):.3f} (`{auc_table['feature'].iloc[0]}`).",
            "However, all clean cases in this expanded smoke came from the test split and all safety-heavy cases came from validation. That is a strong split/domain warning, so this should not yet be promoted as an automatic regime.",
            "",
            "## Decision",
            "",
            "This is now the strongest new candidate beyond transient-peak/future-decay: a moderate-high steady clean-economy subset with ~30% short-horizon pump saving and low immediate cost. The next step should be a targeted cross-split/domain investigation of why validation windows are safety-heavy, not another controller tweak.",
            "",
        ]
    )
    (BASE / "decision.md").write_text(text, encoding="utf-8")
    (OUT_PAPER / "neutral_mhs_expanded_fixed_action_summary.md").write_text(text, encoding="utf-8")
    print(summary.to_string(index=False))
    print(auc_table.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
