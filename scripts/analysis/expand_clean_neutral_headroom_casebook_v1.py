#!/usr/bin/env python3
"""Expand the clean neutral/headroom casebook with fixed pre-outcome rules.

This script does not inspect controller outcomes.  It selects a larger,
time-spaced sample from the already-mined `neutral_untyped` pool so the broad
economy result can be validated on a more representative set rather than on a
small hand-curated screen.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
PROFILE = ROOT / "neutral_untyped_profile_v1/raw_tables/neutral_untyped_profile_rows.csv"
OUT = ROOT / "clean_neutral_headroom_expansion_v1"
RAW = OUT / "raw_tables"
CASEBOOKS = OUT / "casebooks"

MIN_SPACING_S = 6 * 3600
TARGET_PER_SPLIT = 80
BOUNDARY_PER_SPLIT = 24


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def clean_rule(df: pd.DataFrame) -> pd.Series:
    """Fixed, pre-outcome rule for the lower-cost neutral/headroom family."""

    subtype_ok = df["neutral_subtype"].isin(
        [
            "neutral_moderate_high_steady",
            "neutral_mild_decay",
            "neutral_mixed_steady",
        ]
    )
    return (
        subtype_ok
        & (df["early_max_ms"] >= 11.0)
        & (df["near_max_ms"] >= 11.0)
        & (df["near_range_ms"] <= 2.8)
        & (df["far_range_ms"] <= 3.0)
        & (df["dir_shift_abs_max_deg"] <= 25.0)
        & (df["early_rise_ms"] < 3.0)
        & (df["speed_ramp_ge_3ms"].astype(int) == 0)
        & (df["ballast_attention_event"].astype(int) == 0)
        & (df["flag_reversal"].astype(str).str.lower().ne("true"))
        & (df["flag_reintensification"].astype(str).str.lower().ne("true"))
    )


def boundary_rule(df: pd.DataFrame) -> pd.Series:
    """Adjacent non-clean rows used to test abstention / cost boundaries."""

    return (
        df["neutral_subtype"].isin(
            [
                "neutral_ramp_or_event_onset",
                "neutral_direction_shift",
                "neutral_far_rise_watch",
            ]
        )
        | (df["dir_shift_abs_max_deg"] > 30.0)
        | (df["early_rise_ms"] >= 3.0)
        | (df["speed_ramp_ge_3ms"].astype(int) == 1)
        | (df["ballast_attention_event"].astype(int) == 1)
        | (df["flag_reintensification"].astype(str).str.lower().eq("true"))
    )


def score_clean(df: pd.DataFrame) -> pd.Series:
    # High pump proxy, steadiness, low direction drift.  Uses wind-shape only.
    return (
        1.20 * df["early_max_ms"].fillna(0.0)
        + 0.80 * df["near_max_ms"].fillna(0.0)
        + 0.35 * df["peak_to_late_drop_ms"].fillna(0.0).clip(lower=-1.0, upper=3.0)
        - 0.45 * df["near_range_ms"].fillna(0.0)
        - 0.35 * df["far_range_ms"].fillna(0.0)
        - 0.035 * df["dir_shift_abs_max_deg"].fillna(0.0)
    )


def score_boundary(df: pd.DataFrame) -> pd.Series:
    # Pick hard adjacent rows, not random low-value rows.
    return (
        1.00 * df["early_max_ms"].fillna(0.0)
        + 0.80 * df["early_rise_ms"].fillna(0.0)
        + 0.030 * df["dir_shift_abs_max_deg"].fillna(0.0)
        + 1.50 * df["speed_ramp_ge_3ms"].fillna(0).astype(float)
        + 1.50 * df["ballast_attention_event"].fillna(0).astype(float)
    )


def spaced_stratified(
    df: pd.DataFrame,
    score: pd.Series,
    max_cases: int,
    *,
    split: str,
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a time-spaced sample with light subtype/intensity balance."""

    if df.empty:
        return df.copy(), pd.DataFrame()
    work = df.copy()
    work["selection_score"] = score.loc[work.index]
    work["intensity_bin"] = pd.qcut(
        work["early_max_ms"].rank(method="first"),
        q=min(4, max(1, len(work))),
        labels=False,
        duplicates="drop",
    )
    groups = [
        g.copy()
        for _, g in work.groupby(["neutral_subtype", "intensity_bin"], dropna=False)
    ]
    groups = [g.sort_values("selection_score", ascending=False) for g in groups if not g.empty]

    keep: list[pd.Series] = []
    taken: list[pd.Timestamp] = []
    progress = True
    while len(keep) < max_cases and progress:
        progress = False
        for g in groups:
            for idx, row in g.iterrows():
                if idx in {int(k.name) for k in keep if k.name is not None}:
                    continue
                t = pd.to_datetime(row["future_start"])
                if all(abs((t - prev).total_seconds()) >= MIN_SPACING_S for prev in taken):
                    keep.append(row)
                    taken.append(t)
                    progress = True
                    break
            if len(keep) >= max_cases:
                break

    selected = pd.DataFrame(keep)
    if selected.empty:
        selected = work.sort_values("selection_score", ascending=False).head(max_cases).copy()

    rows = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        label = (
            f"{prefix} | subtype={row['neutral_subtype']} | split={split} | "
            f"early={float(row['early_max_ms']):.2f} | "
            f"near_range={float(row['near_range_ms']):.2f} | "
            f"far_range={float(row['far_range_ms']):.2f} | "
            f"drop={float(row['peak_to_late_drop_ms']):.2f} | "
            f"dir={float(row['dir_shift_abs_max_deg']):.1f}"
        )
        rows.append(
            {
                "case_id": f"{prefix}_{split}_{i:03d}",
                "timestamp": str(row["future_start"]),
                "label": label,
            }
        )
    return selected, pd.DataFrame(rows)


def episode_summary(df: pd.DataFrame, selected: pd.DataFrame) -> dict[str, float | int]:
    if selected.empty:
        return {
            "available_rows": int(len(df)),
            "selected_cases": 0,
            "time_span_days": 0.0,
            "episode_count_gap24h": 0,
            "median_cases_per_episode_gap24h": 0.0,
        }
    times = pd.to_datetime(selected["future_start"]).sort_values().reset_index(drop=True)
    gaps = times.diff().dt.total_seconds().fillna(0.0)
    episode_id = (gaps > 24 * 3600).cumsum()
    per_episode = episode_id.value_counts()
    return {
        "available_rows": int(len(df)),
        "selected_cases": int(len(selected)),
        "time_span_days": float((times.max() - times.min()).total_seconds() / 86400.0),
        "episode_count_gap24h": int(episode_id.nunique()),
        "median_cases_per_episode_gap24h": float(per_episode.median()),
    }


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    CASEBOOKS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PROFILE)
    clean_pool = df[clean_rule(df)].copy()
    boundary_pool = df[boundary_rule(df) & ~clean_rule(df)].copy()

    selected_clean = []
    selected_boundary = []
    clean_casebooks = []
    boundary_casebooks = []
    summary_rows = []
    for split in ("test", "validation"):
        clean_split = clean_pool[clean_pool["split"].eq(split)].copy()
        boundary_split = boundary_pool[boundary_pool["split"].eq(split)].copy()
        sel_clean, cb_clean = spaced_stratified(
            clean_split,
            score_clean(clean_split),
            TARGET_PER_SPLIT,
            split=split,
            prefix="clean_neutral_headroom",
        )
        sel_boundary, cb_boundary = spaced_stratified(
            boundary_split,
            score_boundary(boundary_split),
            BOUNDARY_PER_SPLIT,
            split=split,
            prefix="neutral_boundary_control",
        )
        selected_clean.append(sel_clean)
        selected_boundary.append(sel_boundary)
        clean_casebooks.append(cb_clean)
        boundary_casebooks.append(cb_boundary)
        summary_rows.append(
            {
                "split": split,
                "pool": "clean_neutral_headroom",
                **episode_summary(clean_split, sel_clean),
                "early_max_median_ms": float(sel_clean["early_max_ms"].median()),
                "near_range_median_ms": float(sel_clean["near_range_ms"].median()),
                "far_range_median_ms": float(sel_clean["far_range_ms"].median()),
                "dir_shift_median_deg": float(sel_clean["dir_shift_abs_max_deg"].median()),
            }
        )
        summary_rows.append(
            {
                "split": split,
                "pool": "neutral_boundary_control",
                **episode_summary(boundary_split, sel_boundary),
                "early_max_median_ms": float(sel_boundary["early_max_ms"].median()),
                "near_range_median_ms": float(sel_boundary["near_range_ms"].median()),
                "far_range_median_ms": float(sel_boundary["far_range_ms"].median()),
                "dir_shift_median_deg": float(sel_boundary["dir_shift_abs_max_deg"].median()),
            }
        )

    selected_clean_df = pd.concat(selected_clean, ignore_index=True)
    selected_boundary_df = pd.concat(selected_boundary, ignore_index=True)
    clean_casebook = pd.concat(clean_casebooks, ignore_index=True)
    boundary_casebook = pd.concat(boundary_casebooks, ignore_index=True)
    mixed_casebook = pd.concat([clean_casebook, boundary_casebook], ignore_index=True)

    selected_clean_df.to_csv(RAW / "clean_neutral_headroom_selected_features.csv", index=False)
    selected_boundary_df.to_csv(RAW / "neutral_boundary_control_selected_features.csv", index=False)
    clean_casebook.to_csv(CASEBOOKS / "clean_neutral_headroom_expanded_cases.csv", index=False)
    boundary_casebook.to_csv(CASEBOOKS / "neutral_boundary_control_cases.csv", index=False)
    mixed_casebook.to_csv(CASEBOOKS / "clean_neutral_headroom_mixed_with_boundary_cases.csv", index=False)
    for split in ("test", "validation"):
        clean_casebook[clean_casebook["case_id"].str.contains(f"_{split}_")].to_csv(
            CASEBOOKS / f"clean_neutral_headroom_expanded_{split}_cases.csv",
            index=False,
        )
        boundary_casebook[boundary_casebook["case_id"].str.contains(f"_{split}_")].to_csv(
            CASEBOOKS / f"neutral_boundary_control_{split}_cases.csv",
            index=False,
        )

    subtype_summary = (
        clean_pool.groupby(["split", "neutral_subtype"])
        .agg(rows=("neutral_subtype", "size"), series=("series_id", "nunique"))
        .reset_index()
    )
    subtype_summary["selected_rows"] = 0
    sel_counts = (
        selected_clean_df.groupby(["split", "neutral_subtype"]).size().rename("selected_rows")
    )
    subtype_summary = subtype_summary.drop(columns=["selected_rows"]).merge(
        sel_counts.reset_index(),
        on=["split", "neutral_subtype"],
        how="left",
    )
    subtype_summary["selected_rows"] = subtype_summary["selected_rows"].fillna(0).astype(int)
    subtype_summary.to_csv(RAW / "clean_neutral_headroom_pool_subtype_summary.csv", index=False)

    coverage = pd.DataFrame(
        [
            {
                "metric": "neutral_untyped_rows",
                "value": float(len(df)),
            },
            {
                "metric": "clean_pool_rows",
                "value": float(len(clean_pool)),
            },
            {
                "metric": "clean_pool_row_share_of_neutral_pct",
                "value": 100.0 * len(clean_pool) / len(df),
            },
            {
                "metric": "boundary_control_rows",
                "value": float(len(boundary_pool)),
            },
            {
                "metric": "selected_clean_cases",
                "value": float(len(clean_casebook)),
            },
            {
                "metric": "selected_boundary_cases",
                "value": float(len(boundary_casebook)),
            },
        ]
    )
    coverage.to_csv(RAW / "clean_neutral_headroom_expansion_coverage.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RAW / "clean_neutral_headroom_expansion_selection_summary.csv", index=False)

    text = [
        "# Clean Neutral/Headroom Expansion v1",
        "",
        "Fixed pre-outcome expansion for the broad economy family.  This file only",
        "selects cases and reports coverage; it does not inspect controller outcomes.",
        "",
        "## Coverage",
        "",
        md_table(coverage),
        "",
        "## Selection Summary",
        "",
        md_table(summary),
        "",
        "## Casebooks",
        "",
        f"- `{CASEBOOKS / 'clean_neutral_headroom_expanded_cases.csv'}`",
        f"- `{CASEBOOKS / 'clean_neutral_headroom_expanded_test_cases.csv'}`",
        f"- `{CASEBOOKS / 'clean_neutral_headroom_expanded_validation_cases.csv'}`",
        f"- `{CASEBOOKS / 'neutral_boundary_control_cases.csv'}`",
        f"- `{CASEBOOKS / 'neutral_boundary_control_test_cases.csv'}`",
        f"- `{CASEBOOKS / 'neutral_boundary_control_validation_cases.csv'}`",
        f"- `{CASEBOOKS / 'clean_neutral_headroom_mixed_with_boundary_cases.csv'}`",
        "",
    ]
    (OUT / "decision.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


if __name__ == "__main__":
    main()
