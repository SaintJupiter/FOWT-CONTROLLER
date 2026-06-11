#!/usr/bin/env python3
"""Summarize baseline pump/headroom for representative neutral_untyped cases."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "neutral_untyped_baseline_headroom_v1"
RUN = BASE / "a0_representative_1h"
RAW = BASE / "raw_tables"
PAPER = BASE / "paper_ready"


def _duration_metrics(case_id: str) -> dict[str, float]:
    matches = sorted((RUN / "timeseries").glob(f"{case_id}_*_prediction_primary_econ_timeseries.csv"))
    path = matches[0] if matches else RUN / "timeseries" / f"{case_id}_prediction_primary_econ_timeseries.csv"
    if not path.exists():
        return {
            "duration_s": 0.0,
            "time_gt3_s": 0.0,
            "time_gt4_s": 0.0,
            "time_gt5_s": 0.0,
        }
    ts = pd.read_csv(path, usecols=["t_s", "pitch_deg", "roll_deg"])
    if len(ts) < 2:
        dt = 1.0
    else:
        dt = float(ts["t_s"].diff().dropna().median())
        if not dt or dt <= 0:
            dt = 1.0
    axis = ts[["pitch_deg", "roll_deg"]].abs().max(axis=1)
    return {
        "duration_s": float(len(ts) * dt),
        "time_gt3_s": float((axis > 3.0).sum() * dt),
        "time_gt4_s": float((axis > 4.0).sum() * dt),
        "time_gt5_s": float((axis > 5.0).sum() * dt),
    }


def _fmt_float(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(_fmt_float(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)

    summary_path = RUN / "casebook_summary.csv"
    features_path = RAW / "neutral_untyped_representative_selected_features.csv"
    summary = pd.read_csv(summary_path)
    features = pd.read_csv(features_path)
    if len(summary) != len(features):
        raise RuntimeError(f"row mismatch: summary={len(summary)} features={len(features)}")

    df = summary.copy()
    feature_cols = [
        "split",
        "future_speed_first_ms",
        "future_speed_max_ms",
        "early_max_ms",
        "near_max_ms",
        "far_max_ms",
        "peak_to_late_drop_ms",
        "near_range_ms",
        "far_range_ms",
        "block_slope_0_60_ms",
        "block_slope_60_120_ms",
        "dir_shift_abs_max_deg",
        "neutral_subtype",
        "sample_subtype",
    ]
    for col in feature_cols:
        if col in features.columns:
            df[col] = features[col].to_numpy()

    duration = pd.DataFrame([_duration_metrics(case_id) for case_id in df["case_id"]])
    df = pd.concat([df, duration], axis=1)

    df["p95_max_axis_deg"] = df[["primary_pitch_p95", "primary_roll_p95"]].abs().max(axis=1)
    df["material_a0_pump"] = df["primary_pump_work_m3"] >= 100.0
    df["high_a0_pump"] = df["primary_pump_work_m3"] >= 200.0
    df["clean_headroom"] = (
        (df["p95_max_axis_deg"] <= 3.5)
        & (df["time_gt5_s"] <= 1.0)
        & (df["primary_safety_fallback_ratio"] <= 0.001)
    )
    df["medium_headroom"] = (
        (df["p95_max_axis_deg"] <= 5.0)
        & (df["time_gt5_s"] <= 60.0)
        & (df["primary_safety_fallback_ratio"] <= 0.02)
        & ~df["clean_headroom"]
    )
    df["safety_heavy"] = (
        (df["p95_max_axis_deg"] > 5.0)
        | (df["time_gt5_s"] > 60.0)
        | (df["primary_safety_fallback_ratio"] > 0.02)
    )

    def classify(row: pd.Series) -> str:
        if not bool(row["material_a0_pump"]):
            return "low_pump_abstain"
        if bool(row["clean_headroom"]):
            return "clean_candidate"
        if bool(row["medium_headroom"]):
            return "graded_candidate"
        return "safety_boundary_abstain"

    df["headroom_bucket"] = df.apply(classify, axis=1)
    df["recommended_role"] = df["headroom_bucket"].map(
        {
            "clean_candidate": "eligible_for_conservative_hold_or_tracking_reduction",
            "graded_candidate": "eligible_only_for_headroom_scaled_pareto",
            "safety_boundary_abstain": "do_not_relax_use_v1_6_floor",
            "low_pump_abstain": "do_not_target_low_pump",
        }
    )

    per_case_cols = [
        "case_id",
        "sample_subtype",
        "split",
        "primary_pump_work_m3",
        "p95_max_axis_deg",
        "time_gt3_s",
        "time_gt4_s",
        "time_gt5_s",
        "primary_safety_fallback_ratio",
        "primary_latch_switches",
        "peak_to_late_drop_ms",
        "near_range_ms",
        "far_range_ms",
        "dir_shift_abs_max_deg",
        "headroom_bucket",
        "recommended_role",
    ]
    df[per_case_cols].to_csv(RAW / "neutral_untyped_baseline_headroom_per_case.csv", index=False)

    by_subtype = (
        df.groupby("sample_subtype")
        .agg(
            cases=("case_id", "size"),
            total_a0_pump_m3=("primary_pump_work_m3", "sum"),
            median_a0_pump_m3=("primary_pump_work_m3", "median"),
            high_pump_cases=("high_a0_pump", "sum"),
            clean_candidates=("clean_headroom", "sum"),
            graded_candidates=("medium_headroom", "sum"),
            safety_boundary_abstain=("safety_heavy", "sum"),
            median_p95_max_axis_deg=("p95_max_axis_deg", "median"),
            median_time_gt5_s=("time_gt5_s", "median"),
            median_fallback_ratio=("primary_safety_fallback_ratio", "median"),
            median_latch_switches=("primary_latch_switches", "median"),
            median_peak_to_late_drop_ms=("peak_to_late_drop_ms", "median"),
            median_near_range_ms=("near_range_ms", "median"),
            median_dir_shift_deg=("dir_shift_abs_max_deg", "median"),
        )
        .reset_index()
    )
    by_subtype["clean_or_graded_share"] = (
        by_subtype["clean_candidates"] + by_subtype["graded_candidates"]
    ) / by_subtype["cases"]
    by_subtype["safety_boundary_share"] = by_subtype["safety_boundary_abstain"] / by_subtype["cases"]
    by_subtype.to_csv(RAW / "neutral_untyped_baseline_headroom_summary.csv", index=False)

    by_bucket = (
        df.groupby("headroom_bucket")
        .agg(
            cases=("case_id", "size"),
            total_a0_pump_m3=("primary_pump_work_m3", "sum"),
            median_a0_pump_m3=("primary_pump_work_m3", "median"),
            median_p95_max_axis_deg=("p95_max_axis_deg", "median"),
            total_time_gt5_s=("time_gt5_s", "sum"),
            median_fallback_ratio=("primary_safety_fallback_ratio", "median"),
        )
        .reset_index()
    )
    by_bucket["case_share"] = by_bucket["cases"] / len(df)
    by_bucket["pump_share"] = by_bucket["total_a0_pump_m3"] / max(1e-9, df["primary_pump_work_m3"].sum())
    by_bucket.to_csv(RAW / "neutral_untyped_headroom_bucket_summary.csv", index=False)

    duplicate_count = int(
        len(df)
        - len(
            df[
                [
                    "primary_pump_work_m3",
                    "primary_pitch_p95",
                    "primary_roll_p95",
                    "primary_safety_fallback_ratio",
                ]
            ]
            .round(6)
            .drop_duplicates()
        )
    )

    total_pump = float(df["primary_pump_work_m3"].sum())
    candidate = df[df["headroom_bucket"].isin(["clean_candidate", "graded_candidate"])]
    safety = df[df["headroom_bucket"].eq("safety_boundary_abstain")]
    candidate_pump = float(candidate["primary_pump_work_m3"].sum())
    safety_pump = float(safety["primary_pump_work_m3"].sum())

    summary_cols = [
        "sample_subtype",
        "cases",
        "total_a0_pump_m3",
        "median_a0_pump_m3",
        "clean_candidates",
        "graded_candidates",
        "safety_boundary_abstain",
        "clean_or_graded_share",
        "safety_boundary_share",
        "median_p95_max_axis_deg",
        "median_time_gt5_s",
        "median_fallback_ratio",
    ]
    bucket_cols = [
        "headroom_bucket",
        "cases",
        "case_share",
        "total_a0_pump_m3",
        "pump_share",
        "median_a0_pump_m3",
        "median_p95_max_axis_deg",
        "total_time_gt5_s",
    ]

    decision = f"""# Neutral Untyped Baseline Headroom Decision v1

This is an A0-only, 1h representative screen for the broad `neutral_untyped` pool. It does not test a new controller. It asks whether this pool has real baseline pump and enough posture headroom to justify a future headroom-graded economy action.

## Top Line

- Total A0 pump in the 48-case representative sample: **{total_pump:.1f} m3**.
- Clean/graded candidates: **{len(candidate)}/{len(df)} cases**, carrying **{candidate_pump:.1f} m3** baseline pump (**{candidate_pump / total_pump:.1%}** of sample pump).
- Safety-boundary abstain cases: **{len(safety)}/{len(df)} cases**, carrying **{safety_pump:.1f} m3** baseline pump (**{safety_pump / total_pump:.1%}** of sample pump).
- Rounded duplicate metric rows: **{duplicate_count}**. This means some high-posture patterns repeat; treat this as a representative screen, not a final validation set.

## By Shape Subtype

{_markdown_table(by_subtype, summary_cols)}

## By Headroom Bucket

{_markdown_table(by_bucket, bucket_cols)}

## Interpretation

1. `neutral_untyped` is not a useless pool: it has real pump. The candidate part has enough A0 pump to justify one small follow-up smoke test.
2. It is also not one clean automatic saving regime. A large fraction of pump sits in `safety_boundary_abstain`, where p95/time>5/fallback are already near the safety domain. Those rows should fall back to v1.6 and must not be relaxed.
3. The usable mechanism is a **headroom-graded economy suppression**, not a blanket budget: clean rows can hold more, medium rows can suppress mildly, safety-boundary rows abstain.
4. This would be a state/headroom-conditioned Pareto extension of the neutral-MHS fixed-hold idea, not a pure forecast-attributable specialist. Forecast can provide boundary vetoes, but current posture headroom is the main safety key.

## Decision

Proceed only to a short smoke test if we continue: A0 vs one conservative headroom-graded profile on `clean_candidate + graded_candidate`, with `safety_boundary_abstain` as a negative-control set. Do not build a broad controller or run a long validation until that smoke shows material pump reduction without extra time>5/fallback.
"""
    (BASE / "decision.md").write_text(decision, encoding="utf-8")
    (PAPER / "neutral_untyped_baseline_headroom_decision.md").write_text(decision, encoding="utf-8")
    print(by_subtype.to_string(index=False))
    print()
    print(by_bucket.to_string(index=False))
    print(f"\nWrote {BASE / 'decision.md'}")


if __name__ == "__main__":
    main()
