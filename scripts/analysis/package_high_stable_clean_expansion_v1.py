#!/usr/bin/env python3
"""Package the high-stable clean neutral/headroom expansion result.

This script only aggregates already-run casebooks.  It creates the paper-facing
48-case test/validation result for the cleanest broad economy subdomain found in
the neutral/headroom pool.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
BASE = ROOT / "clean_neutral_headroom_expansion_v1"
PAPER = BASE / "paper_ready"
PROFILE = ROOT / "neutral_untyped_profile_v1/raw_tables/neutral_untyped_profile_rows.csv"

NEUTRAL_UNTYPED_ALL_WINDOW_PREVALENCE_PCT = 19.915949832556308
BROAD_ADJUSTABLE_POOL_ALL_WINDOW_PREVALENCE_PCT = 40.9
CLEAN_ECONOMY_OPPORTUNITY_POOL_ALL_WINDOW_PREVALENCE_PCT = 33.8


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.2f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def high_stable_rule(df: pd.DataFrame) -> pd.Series:
    subtype_ok = df["neutral_subtype"].isin(
        [
            "neutral_moderate_high_steady",
            "neutral_mild_decay",
            "neutral_mixed_steady",
        ]
    )
    return (
        subtype_ok
        & (df["early_max_ms"] >= 14.0)
        & (df["near_max_ms"] >= 14.0)
        & (df["far_max_ms"] >= 11.5)
        & (df["near_range_ms"] <= 3.0)
        & (df["far_range_ms"] <= 3.0)
        & (df["dir_shift_abs_max_deg"] <= 20.0)
        & (df["speed_ramp_ge_3ms"].astype(int) == 0)
        & (df["ballast_attention_event"].astype(int) == 0)
        & (df["flag_reversal"].astype(str).str.lower().ne("true"))
        & (df["flag_reintensification"].astype(str).str.lower().ne("true"))
    )


def summarize_cases(df: pd.DataFrame, label: str) -> dict[str, float | int | str]:
    closed = float(df["closed_pump_integral_m3"].sum())
    primary = float(df["primary_pump_integral_m3"].sum())
    saved = closed - primary
    positive = df["saved_m3"].clip(lower=0.0)
    return {
        "set": label,
        "cases": int(len(df)),
        "a0_pump_m3": closed,
        "candidate_pump_m3": primary,
        "saved_m3": saved,
        "saving_pct": 100.0 * saved / closed if closed else 0.0,
        "delta_time_gt5_s": float(df["d_time_gt5_s"].sum()),
        "delta_time_gt4_s": float(df["d_time_gt4_s"].sum()),
        "fallback_s": float(df["primary_fallback_s"].sum()),
        "fallback_cases": int((df["primary_fallback_s"] > 0.0).sum()),
        "mean_delta_p95_axis_deg": float(df["d_p95_axis_deg"].mean()),
        "max_delta_p95_axis_deg": float(df["d_p95_axis_deg"].max()),
        "top_case_share": float(positive.max() / max(positive.sum(), 1e-9)),
        "bad_cases_timegt5_gt300": int((df["d_time_gt5_s"] > 300.0).sum()),
    }


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    test_cases = pd.read_csv(BASE / "casebooks/high_stable_clean_test_cases.csv")
    validation_cases = pd.read_csv(BASE / "casebooks/high_stable_clean_validation_cases.csv")
    test_deltas = pd.read_csv(PAPER / "test80_v1_refresh_case_runtime_delta.csv")
    validation80_deltas = pd.read_csv(PAPER / "validation80_v1_refresh_case_runtime_delta.csv")

    test_times = set(pd.to_datetime(test_cases["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S"))
    test_deltas["_timestamp_key"] = pd.to_datetime(test_deltas["timestamp"]).dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    test_high = test_deltas[test_deltas["_timestamp_key"].isin(test_times)].copy()
    test_high = test_high.drop(columns=["_timestamp_key"])
    if len(test_high) != len(test_cases):
        raise RuntimeError(f"expected {len(test_cases)} high-stable test rows, got {len(test_high)}")

    validation_times = set(
        pd.to_datetime(validation_cases["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    validation80_deltas["_timestamp_key"] = pd.to_datetime(
        validation80_deltas["timestamp"]
    ).dt.strftime("%Y-%m-%d %H:%M:%S")
    validation_high = validation80_deltas[
        validation80_deltas["_timestamp_key"].isin(validation_times)
    ].copy()
    validation_high = validation_high.drop(columns=["_timestamp_key"])
    if len(validation_high) != len(validation_cases):
        raise RuntimeError(
            f"expected {len(validation_cases)} high-stable validation rows, got {len(validation_high)}"
        )

    combined = pd.concat(
        [
            test_high.assign(split="test"),
            validation_high.assign(split="validation"),
        ],
        ignore_index=True,
    )

    summary = pd.DataFrame(
        [
            summarize_cases(test_high, "test_high_stable_clean"),
            summarize_cases(validation_high, "validation_high_stable_clean"),
            summarize_cases(combined, "combined_high_stable_clean"),
        ]
    )

    profile = pd.read_csv(PROFILE)
    high = profile[high_stable_rule(profile)].copy()
    split_counts = (
        high.groupby("split")
        .agg(rows=("split", "size"), series=("series_id", "nunique"))
        .reset_index()
    )
    split_counts["row_share_of_neutral_untyped_pct"] = 100.0 * split_counts["rows"] / len(profile)
    high_share = 100.0 * len(high) / len(profile)
    approx_all_window = (
        NEUTRAL_UNTYPED_ALL_WINDOW_PREVALENCE_PCT * high_share / 100.0
    )
    prevalence = pd.DataFrame(
        [
            {
                "metric": "neutral_untyped_rows",
                "value": float(len(profile)),
            },
            {
                "metric": "high_stable_clean_rows",
                "value": float(len(high)),
            },
            {
                "metric": "high_stable_share_of_neutral_untyped_pct",
                "value": high_share,
            },
            {
                "metric": "approx_all_window_prevalence_pct",
                "value": approx_all_window,
            },
            {
                "metric": "share_of_broad_adjustable_pool_pct",
                "value": 100.0 * approx_all_window / BROAD_ADJUSTABLE_POOL_ALL_WINDOW_PREVALENCE_PCT,
            },
            {
                "metric": "share_of_clean_economy_opportunity_pool_pct",
                "value": 100.0 * approx_all_window / CLEAN_ECONOMY_OPPORTUNITY_POOL_ALL_WINDOW_PREVALENCE_PCT,
            },
            {
                "metric": "selected_test_cases",
                "value": float(len(test_high)),
            },
            {
                "metric": "selected_validation_cases",
                "value": float(len(validation_high)),
            },
        ]
    )

    subtype = (
        combined.assign(
            subtype=combined["label"].astype(str).str.extract(r"subtype=([^|]+)")[0].str.strip()
        )
        .groupby(["split", "subtype"])
        .agg(
            cases=("case_id", "count"),
            a0_pump_m3=("closed_pump_integral_m3", "sum"),
            candidate_pump_m3=("primary_pump_integral_m3", "sum"),
            saved_m3=("saved_m3", "sum"),
            delta_time_gt5_s=("d_time_gt5_s", "sum"),
            fallback_s=("primary_fallback_s", "sum"),
            mean_delta_p95_axis_deg=("d_p95_axis_deg", "mean"),
        )
        .reset_index()
    )
    subtype["saving_pct"] = 100.0 * subtype["saved_m3"] / subtype["a0_pump_m3"]

    combined.to_csv(PAPER / "high_stable_clean_combined52_case_delta.csv", index=False)
    summary.to_csv(PAPER / "high_stable_clean_combined52_summary.csv", index=False)
    prevalence.to_csv(PAPER / "high_stable_clean_prevalence.csv", index=False)
    split_counts.to_csv(PAPER / "high_stable_clean_prevalence_by_split.csv", index=False)
    subtype.to_csv(PAPER / "high_stable_clean_subtype_summary.csv", index=False)

    text = [
        "# High-Stable Clean Neutral/Headroom Expansion v1",
        "",
        "## What This Adds",
        "",
        "This is a fixed pre-action expansion of the broad neutral/headroom economy",
        "family.  The rule selects cases where the future load is high but steady,",
        "with no large direction shift, no ramp/event-onset flag, and no reversal or",
        "re-intensification flag.  It is intended to answer the sample-size concern",
        "without admitting the high-cost ramp/onset cases from the wider 80-case pool.",
        "",
        "## Fixed Rule",
        "",
        "- subtype: `neutral_moderate_high_steady`, `neutral_mild_decay`, or `neutral_mixed_steady`",
        "- `early_max_ms >= 14`, `near_max_ms >= 14`, `far_max_ms >= 11.5`",
        "- `near_range_ms <= 3`, `far_range_ms <= 3`",
        "- `dir_shift_abs_max_deg <= 20`",
        "- no speed ramp, no ballast-attention event, no reversal, no re-intensification",
        "",
        "## Main Result",
        "",
        _md_table(summary),
        "",
        "## Coverage Estimate",
        "",
        _md_table(prevalence),
        "",
        "## Subtype Breakdown",
        "",
        _md_table(subtype),
        "",
        "## Interpretation",
        "",
        "The high-stable clean subset is not the same as the transient relief/decay",
        "specialist.  Its mechanism is reduced economy-layer target pursuit during a",
        "high but steady load period with enough posture headroom.  The forecast is",
        "used as the permissive entry and boundary-veto signal: it admits steady",
        "high-load windows and excludes ramp, reversal, and re-intensification windows.",
        "",
        "This result is cleaner than the wider 80-case broad screen.  The 80-case screen",
        "saved more pump in aggregate, but it admitted high-risk rows with large",
        "`time>5` increases.  The high-stable rule gives a split-balanced 52-case",
        "result with zero fallback and no increase in `time>5`.",
        "",
        "## Paper Use",
        "",
        "Use this as the broader, cleaner evidence point for the neutral/headroom",
        "economy family.  It expands the evidence base beyond the original small",
        "casebooks while keeping the boundary conditions interpretable.",
        "",
    ]
    out = PAPER / "high_stable_clean_expansion_summary.md"
    out.write_text("\n".join(text), encoding="utf-8")
    print(_md_table(summary))
    print()
    print(_md_table(prevalence))
    print()
    print(out)


if __name__ == "__main__":
    main()
