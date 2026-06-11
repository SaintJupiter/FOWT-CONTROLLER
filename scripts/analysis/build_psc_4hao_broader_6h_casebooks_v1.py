#!/usr/bin/env python3
"""Build stratified real-window casebooks for PSC 4hao broader 6h validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
OUT = BASE / "psc_4hao_broader_6h_test_only_main_v1"
CASEBOOKS = OUT / "casebooks"
RAW = OUT / "raw_tables"
MIN_SPACING_HOURS = 6.0

RELIEF = (
    REPO
    / "outputs"
    / "wind_prediction"
    / "economy_pump_budget_closed_loop_probe_v1"
    / "relief_decay_expansion_v1"
    / "relief_decay_expansion_cases.csv"
)
NEUTRAL = (
    REPO
    / "outputs"
    / "wind_prediction"
    / "regime_conditioned_policy_development_v1"
    / "neutral_moderate_high_steady_broader_v1"
    / "casebooks"
    / "neutral_mhs_broader_cases.csv"
)
V3 = (
    REPO
    / "outputs"
    / "wind_prediction"
    / "regime_conditioned_policy_development_v1"
    / "regime_mining_v3"
    / "raw_tables"
)
STRESS = BASE / "psc_4hao_25pct_overnight_v1" / "expanded_top20_risky_gain_cases.csv"


def _read_casebook(path: Path, stratum: str, role: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df[["case_id", "timestamp", "label"]].copy()
    out["timestamp_dt"] = pd.to_datetime(out["timestamp"], errors="raise")
    out["source_case_id"] = out["case_id"].astype(str)
    out["source_file"] = str(path.relative_to(REPO))
    out["stratum"] = stratum
    out["validation_role"] = role
    out["selection_basis"] = source
    return out


def _spaced(df: pd.DataFrame, n: int, taken: set[pd.Timestamp] | None = None) -> pd.DataFrame:
    if taken is None:
        taken = set()
    work = df.sort_values("timestamp_dt").copy()
    keep = []
    for _, row in work.iterrows():
        t = row["timestamp_dt"]
        if any(abs((t - prev).total_seconds()) < MIN_SPACING_HOURS * 3600 for prev in taken):
            continue
        keep.append(row)
        taken.add(t)
        if len(keep) >= n:
            break
    return pd.DataFrame(keep)


def _v3_cases(split: str, regime: str) -> Path:
    return V3 / f"casebooks_{split}" / f"{regime}_cases.csv"


def _v3_test_casebook(regime: str, role: str, source: str) -> pd.DataFrame:
    return _read_casebook(_v3_cases("test", regime), regime, role, source + " (test split only)")


def _renumber(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["case_id"] = [f"{prefix}_{i:02d}" for i in range(1, len(out) + 1)]
    out["label"] = (
        out["label"].astype(str)
        + " | broader6h_stratum="
        + out["stratum"].astype(str)
        + " | validation_role="
        + out["validation_role"].astype(str)
        + " | source_case_id="
        + out["source_case_id"].astype(str)
    )
    return out


def _write_pool(name: str, df: pd.DataFrame) -> None:
    df = _renumber(df, name)
    df.to_csv(RAW / f"{name}_case_table.csv", index=False)
    df[["case_id", "timestamp", "label"]].to_csv(CASEBOOKS / f"{name}_cases.csv", index=False)


def main() -> None:
    CASEBOOKS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    positive_parts = [
        _read_casebook(RELIEF, "transient_peak_future_decay", "main_positive", "fixed-rule real-window relief/decay").head(16),
        _read_casebook(NEUTRAL, "neutral_mhs_broader", "main_positive", "fixed-rule real-window neutral steady").head(16),
    ]
    positive = pd.concat(positive_parts, ignore_index=True)

    negative_taken: set[pd.Timestamp] = set()
    negative_parts = []
    for regime, count in [
        ("direction_reversal_boundary", 12),
        ("reintensification_boundary", 8),
        ("sustained_high_safety_event", 8),
    ]:
        pool = _v3_test_casebook(
            regime,
            "main_negative",
            "regime_mining_v3 ex-ante future-shape mining",
        )
        negative_parts.append(_spaced(pool, count, negative_taken))
    negative = pd.concat(negative_parts, ignore_index=True)

    bg_taken: set[pd.Timestamp] = set()
    background_parts = []
    for regime, count in [
        ("quiet_low_opportunity", 10),
        ("lowrisk_stable_redundant_candidate", 10),
    ]:
        pool = _v3_test_casebook(
            regime,
            "main_background",
            "regime_mining_v3 ex-ante low-opportunity mining",
        )
        background_parts.append(_spaced(pool, count, bg_taken))
    background = pd.concat(background_parts, ignore_index=True)

    stress = _read_casebook(
        STRESS,
        "risky_top_gain_stress",
        "appendix_stress",
        "outcome-informed real-window high-risk/high-gain stress pool",
    )

    _write_pool("positive_pool", positive)
    _write_pool("negative_pool", negative)
    _write_pool("background_pool", background)
    _write_pool("stress_pool", stress)

    combined_main = pd.concat([positive, negative, background], ignore_index=True)
    _write_pool("main_broader_pool", combined_main)

    manifest = pd.concat(
        [
            positive.assign(pool="positive_pool"),
            negative.assign(pool="negative_pool"),
            background.assign(pool="background_pool"),
            stress.assign(pool="stress_pool"),
            combined_main.assign(pool="main_broader_pool"),
        ],
        ignore_index=True,
    )
    summary = (
        manifest.groupby(["pool", "validation_role", "stratum"], dropna=False)
        .agg(cases=("case_id", "size"))
        .reset_index()
    )
    manifest.to_csv(RAW / "broader6h_manifest.csv", index=False)
    summary.to_csv(RAW / "broader6h_casebook_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(CASEBOOKS)


if __name__ == "__main__":
    main()
