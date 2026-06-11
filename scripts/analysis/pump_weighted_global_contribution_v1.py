#!/usr/bin/env python3
"""Estimate pump-weighted global contribution for the current regime library.

The goal is to separate three quantities that are easy to conflate:

1. all-window prevalence;
2. in-regime pump saving;
3. estimated global pump contribution after pump-share weighting.

This is an analysis/packaging script only.  It does not run controllers or alter
controller behavior.
"""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
OUT = ROOT / "paper_ready" / "pump_weighted_global_contribution_v1"
OUT.mkdir(parents=True, exist_ok=True)

# Current source-of-truth for the matched relief/decay specialist result.
# Do not replace this with the conservative dispatcher-v2 row: that run is a
# different auto-dispatcher profile and is intentionally tracked separately.
RELIEF_DECAY_EPISODE_AUTO_V1_MATCHED_SAVING_PCT = 18.13


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _mean_pump(path: str, pump_col: str, cases_col: str = "cases", row: str | None = None, key: str = "group") -> float:
    df = _read(path)
    if row is not None:
        df = df[df[key].eq(row)]
    if df.empty:
        raise RuntimeError(f"missing row {row!r} in {path}")
    r = df.iloc[0]
    return float(r[pump_col]) / float(r[cases_col])


def _normalized_case_id(value: object) -> str:
    return re.sub(r"^\d+_", "", str(value))


def _split_corrected_neutral_summary() -> tuple[pd.DataFrame, dict[str, float]]:
    base = ROOT / "neutral_untyped_baseline_headroom_v1"
    mixed = _read(str(base / "forecast_advised_economy_v2_representative_48_1h/casebook_summary.csv"))
    validation = _read(str(base / "forecast_advised_economy_v2_validation_missing20_1h/casebook_summary.csv"))
    mixed["_norm_case_id"] = mixed["case_id"].map(_normalized_case_id)
    validation["_norm_case_id"] = validation["case_id"].map(_normalized_case_id)
    corrected = pd.concat(
        [
            mixed[~mixed["_norm_case_id"].isin(set(validation["_norm_case_id"]))],
            validation,
        ],
        ignore_index=True,
    )
    if int(corrected["_norm_case_id"].nunique()) != 48 or int(len(corrected)) != 48:
        raise RuntimeError(
            "expected split-corrected neutral representative to contain 48 unique cases"
        )
    corrected = corrected.drop(columns=["_norm_case_id"])
    closed = float(corrected["closed_pump_work_m3"].sum())
    primary = float(corrected["primary_pump_work_m3"].sum())
    active = int((corrected["preview_pump_suppression_ratio"] > 0).sum())
    metrics = {
        "cases": float(len(corrected)),
        "active_cases": float(active),
        "pump_intensity_m3_per_1h_case": closed / float(len(corrected)),
        "pump_saving_pct": 100.0 * (closed - primary) / closed if closed else 0.0,
        "fallback_rows_s": float(corrected["primary_safety_fallback_ratio"].sum() * 3600.0),
        "mean_d_pitch_p95": float(
            (corrected["primary_pitch_p95"] - corrected["closed_pitch_p95"]).mean()
        ),
        "mean_d_roll_p95": float(
            (corrected["primary_roll_p95"] - corrected["closed_roll_p95"]).mean()
        ),
    }
    corrected["subtype"] = corrected["label"].str.extract(r"subtype=([^|]+)")[0].str.strip()
    clean_subtypes = {
        "neutral_mild_decay",
        "neutral_mixed_steady",
        "neutral_moderate_high_steady",
    }
    clean = corrected[corrected["subtype"].isin(clean_subtypes)].copy()
    clean_closed = float(clean["closed_pump_work_m3"].sum())
    clean_primary = float(clean["primary_pump_work_m3"].sum())
    metrics.update(
        {
            "clean_cases": float(len(clean)),
            "clean_active_cases": float((clean["preview_pump_suppression_ratio"] > 0).sum()),
            "clean_pump_intensity_m3_per_1h_case": clean_closed / float(len(clean)),
            "clean_pump_saving_pct": 100.0 * (clean_closed - clean_primary) / clean_closed
            if clean_closed
            else 0.0,
            "clean_fallback_rows_s": float(clean["primary_safety_fallback_ratio"].sum() * 3600.0),
            "clean_row_fraction_of_neutral": float(len(clean)) / float(len(corrected)),
        }
    )
    return corrected, metrics


def _reason_counts(timeseries_dir: Path) -> tuple[int, int, Counter[str]]:
    cases = 0
    active_cases = 0
    reasons: Counter[str] = Counter()
    for path in sorted(timeseries_dir.glob("*_prediction_primary_econ_timeseries.csv")):
        cases += 1
        df = pd.read_csv(
            path,
            usecols=["preview_pump_suppression_active", "preview_pump_suppression_reason"],
        )
        active = df["preview_pump_suppression_active"].astype(str).isin(["1", "True", "true"]).any()
        active_cases += int(active)
        reasons.update(df["preview_pump_suppression_reason"].fillna("").astype(str).tolist())
    return cases, active_cases, reasons


def main() -> None:
    candidate = _read(str(ROOT / "overnight_regime_opportunity_v1/raw_tables/regime_candidate_library_overnight.csv"))
    v2 = _read(str(ROOT / "raw_tables/regime_library_v2_summary.csv"))
    neutral_corrected, neutral_metrics = _split_corrected_neutral_summary()

    # Baseline-pump intensity proxies, in m3 per 1h representative case/window.
    # These are deliberately sourced from already-run casebooks, not from wind speed alone.
    intensity: dict[str, tuple[float, str]] = {}
    intensity["lowrisk_stable_redundant_candidate"] = (
        _mean_pump(
            str(ROOT / "lowrisk_redundant_opportunity_audit_v1/raw_tables/lowrisk_redundant_summary.csv"),
            "a0_pump_m3",
            row="lowrisk_or_quiet_shape",
        ),
        "lowrisk_or_quiet_shape A0 pump / cases",
    )
    intensity["neutral_untyped"] = (
        float(neutral_metrics["pump_intensity_m3_per_1h_case"]),
        "neutral_untyped split-corrected representative 48 closed baseline pump / cases",
    )
    lowpump = v2[v2["fine_regime"].eq("low_pump_no_material_opportunity")].iloc[0]
    intensity["quiet_low_opportunity"] = (
        float(lowpump["a0_pump_m3"]) / float(lowpump["cases"]),
        "low_pump_no_material_opportunity A0 pump / cases",
    )
    gusty = _read(str(ROOT / "regime_mining_v3/raw_tables/gusty_oscillation_budget100_screen_summary.csv"))
    intensity["gusty_oscillation_candidate"] = (
        float(gusty[gusty["arm"].eq("a0_baseline")].iloc[0]["pump_m3"]) / 24.0,
        "gusty screen A0 pump / cases",
    )
    for primary, fine in [
        ("direction_reversal_boundary", "direction_reversal_catchup_boundary"),
        ("residual_high_plateau", "residual_high_plateau_high_pump"),
        ("transient_decay_low_pump_proxy", "transient_decay_low_pump_opportunity"),
        ("residual_high_slow_decay", "residual_high_slow_decay_or_moderate_relief"),
        ("transient_peak_soft_decay", "transient_peak_soft_decay_high_pump"),
        ("transient_peak_fast_decay", "transient_peak_fast_decay_high_pump"),
    ]:
        row = v2[v2["fine_regime"].eq(fine)].iloc[0]
        intensity[primary] = (
            float(row["a0_pump_m3"]) / float(row["cases"]),
            f"{fine} A0 pump / cases",
        )
    hard = v2[v2["fine_regime"].eq("low_pump_no_material_opportunity")].iloc[0]
    intensity["sustained_high_safety_event"] = (
        float(hard["a0_pump_m3"]) / float(hard["cases"]),
        "conservative low-pump proxy; safety events are not economy targets",
    )
    intensity["reintensification_boundary"] = intensity["direction_reversal_boundary"]
    intensity["far_only_h120_advisory"] = (
        intensity["quiet_low_opportunity"][0],
        "advisory-only low pump proxy",
    )

    rows: list[dict[str, object]] = []
    for _, r in candidate.dropna(subset=["row_pct_scanned"]).iterrows():
        name = str(r["primary_regime"])
        pump_per_window, source = intensity.get(name, (0.0, "no pump proxy"))
        row_pct = float(r["row_pct_scanned"])
        rows.append(
            {
                "primary_regime": name,
                "row_pct_all_windows": row_pct,
                "pump_intensity_m3_per_1h_case": pump_per_window,
                "pump_intensity_source": source,
                "pump_weight_proxy": row_pct * pump_per_window,
                "status": r.get("status", ""),
            }
        )
    pump_proxy = pd.DataFrame(rows)
    total_weight = float(pump_proxy["pump_weight_proxy"].sum())
    pump_proxy["estimated_pump_share_pct"] = 100.0 * pump_proxy["pump_weight_proxy"] / total_weight
    pump_proxy = pump_proxy.sort_values("estimated_pump_share_pct", ascending=False)
    pump_proxy.to_csv(OUT / "regime_pump_share_proxy.csv", index=False)

    # Positive families.  Neutral-MHS clean-start is reported as an operating
    # point of the broad safe-economy mechanism, not as an additive global family.
    rep_summary = _read(
        str(
            ROOT
            / "neutral_untyped_baseline_headroom_v1/paper_ready/forecast_advised_broad_economy_v1/broad_economy_core_metrics.csv"
        )
    )
    neutral_saving = float(
        rep_summary[rep_summary["set"].eq("split-corrected neutral_untyped 48")].iloc[0][
            "pump_saving_pct"
        ]
    )
    relief_saving = RELIEF_DECAY_EPISODE_AUTO_V1_MATCHED_SAVING_PCT
    mhs = _read(
        str(
            ROOT
            / "neutral_moderate_high_steady_broader_v1/raw_tables/neutral_mhs_broader_clean_auto_learned_summary.csv"
        )
    )
    mhs_clean = mhs[mhs["group"].eq("clean_32")].iloc[0]
    mhs_broad = mhs[mhs["group"].eq("all_64_broader")].iloc[0]

    def share(regime: str) -> float:
        match = pump_proxy[pump_proxy["primary_regime"].eq(regime)]
        return float(match.iloc[0]["estimated_pump_share_pct"]) if not match.empty else 0.0

    def rows_pct(regime: str) -> float:
        match = pump_proxy[pump_proxy["primary_regime"].eq(regime)]
        return float(match.iloc[0]["row_pct_all_windows"]) if not match.empty else 0.0

    contribution_rows = [
        {
            "family": "forecast_relief_decay_auto",
            "role": "forecast-causal specialist",
            "all_window_prevalence_pct": rows_pct("transient_peak_fast_decay"),
            "estimated_pump_share_pct": share("transient_peak_fast_decay"),
            "active_or_applicable_fraction": 1.0,
            "in_regime_saving_pct": relief_saving,
            "equal_window_global_contribution_pct": rows_pct("transient_peak_fast_decay")
            * relief_saving
            / 100.0,
            "pump_weighted_global_contribution_pct": share("transient_peak_fast_decay")
            * relief_saving
            / 100.0,
            "cost_note": "episode_auto_v1 matched source-of-truth; fallback/time>5 costs must be reported",
            "additivity_note": "additive with neutral/headroom family",
        },
        {
            "family": "safe_economy_broad_neutral_headroom",
            "role": "state/headroom economy mode, forecast-gated and forecast-vetoed",
            "all_window_prevalence_pct": rows_pct("neutral_untyped"),
            "estimated_pump_share_pct": share("neutral_untyped"),
            "active_or_applicable_fraction": neutral_metrics["active_cases"] / neutral_metrics["cases"],
            "in_regime_saving_pct": neutral_saving,
            "equal_window_global_contribution_pct": rows_pct("neutral_untyped")
            * neutral_saving
            / 100.0,
            "pump_weighted_global_contribution_pct": share("neutral_untyped")
            * neutral_saving
            / 100.0,
            "cost_note": (
                "split-corrected 48-case representative: fallback +"
                f"{neutral_metrics['fallback_rows_s']:.0f}s, p95 pitch "
                f"{neutral_metrics['mean_d_pitch_p95']:+.3f}deg"
            ),
            "additivity_note": "subsumes/broadens neutral-MHS; do not add neutral-MHS separately",
        },
        {
            "family": "cost_filtered_neutral_headroom_clean40",
            "role": "cleaner operating point of the same neutral/headroom economy mode",
            "all_window_prevalence_pct": rows_pct("neutral_untyped")
            * float(neutral_metrics["clean_row_fraction_of_neutral"]),
            "estimated_pump_share_pct": share("neutral_untyped")
            * (
                float(neutral_metrics["clean_row_fraction_of_neutral"])
                * float(neutral_metrics["clean_pump_intensity_m3_per_1h_case"])
                / float(neutral_metrics["pump_intensity_m3_per_1h_case"])
            ),
            "active_or_applicable_fraction": neutral_metrics["clean_active_cases"]
            / neutral_metrics["clean_cases"],
            "in_regime_saving_pct": neutral_metrics["clean_pump_saving_pct"],
            "equal_window_global_contribution_pct": rows_pct("neutral_untyped")
            * float(neutral_metrics["clean_row_fraction_of_neutral"])
            * neutral_metrics["clean_pump_saving_pct"]
            / 100.0,
            "pump_weighted_global_contribution_pct": share("neutral_untyped")
            * (
                float(neutral_metrics["clean_row_fraction_of_neutral"])
                * float(neutral_metrics["clean_pump_intensity_m3_per_1h_case"])
                / float(neutral_metrics["pump_intensity_m3_per_1h_case"])
            )
            * neutral_metrics["clean_pump_saving_pct"]
            / 100.0,
            "cost_note": (
                "excludes neutral_ramp_or_event_onset; fallback +"
                f"{neutral_metrics['clean_fallback_rows_s']:.0f}s across 40 cases"
            ),
            "additivity_note": "alternative to broad neutral/headroom, not additive with it",
        },
        {
            "family": "neutral_mhs_clean_start_operating_point",
            "role": "clean-start operating point of the safe-economy mechanism",
            "all_window_prevalence_pct": None,
            "estimated_pump_share_pct": None,
            "active_or_applicable_fraction": float(mhs_clean["active_cases"]) / float(mhs_clean["cases"]),
            "in_regime_saving_pct": float(mhs_clean["pump_saving_pct"]),
            "equal_window_global_contribution_pct": None,
            "pump_weighted_global_contribution_pct": None,
            "cost_note": f"clean-start {float(mhs_clean['pump_saving_pct']):.2f}%; broader 64-case {float(mhs_broad['pump_saving_pct']):.2f}%",
            "additivity_note": "not globally additive; use as evidence for the same safe-economy mechanism",
        },
    ]
    contribution = pd.DataFrame(contribution_rows)
    contribution.to_csv(OUT / "pump_weighted_global_contribution_table.csv", index=False)

    def contribution_sum(families: set[str], col: str) -> float:
        rows = contribution[contribution["family"].isin(families)]
        return float(rows[col].sum())

    broad_families = {
        "forecast_relief_decay_auto",
        "safe_economy_broad_neutral_headroom",
    }
    clean_families = {
        "forecast_relief_decay_auto",
        "cost_filtered_neutral_headroom_clean40",
    }
    total_equal_broad = contribution_sum(broad_families, "equal_window_global_contribution_pct")
    total_pump_weighted_broad = contribution_sum(
        broad_families, "pump_weighted_global_contribution_pct"
    )
    total_equal_clean = contribution_sum(clean_families, "equal_window_global_contribution_pct")
    total_pump_weighted_clean = contribution_sum(
        clean_families, "pump_weighted_global_contribution_pct"
    )

    # Forecast/replay sample coverage audit for the broad neutral representative set.
    audits = []
    for name, path in [
        (
            "legacy_representative_48_mixed_split",
            ROOT
            / "neutral_untyped_baseline_headroom_v1/forecast_advised_economy_v2_representative_48_1h/timeseries",
        ),
        (
            "validation_split_recovered20",
            ROOT
            / "neutral_untyped_baseline_headroom_v1/forecast_advised_economy_v2_validation_missing20_1h/timeseries",
        ),
        (
            "legacy_inactive20_oracle_wrong_split",
            ROOT
            / "neutral_untyped_baseline_headroom_v1/forecast_advised_economy_v2_oracle_missing20_1h/timeseries",
        ),
    ]:
        cases, active_cases, reasons = _reason_counts(path)
        total_rows = sum(reasons.values())
        audits.append(
            {
                "set": name,
                "cases": cases,
                "active_cases": active_cases,
                "inactive_cases": cases - active_cases,
                "missing_sample_rows": reasons.get("missing_sample", 0),
                "suppression_rows": reasons.get("forecast_advised_economy_suppression", 0),
                "plateau_no_candidate_rows": reasons.get("plateau_only_no_candidate", 0),
                "total_rows": total_rows,
                "missing_sample_row_ratio": reasons.get("missing_sample", 0) / total_rows
                if total_rows
                else 0.0,
            }
        )
    audit = pd.DataFrame(audits)
    audit.to_csv(OUT / "forecast_sample_coverage_audit.csv", index=False)

    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df[cols].iterrows():
            vals = []
            for col in cols:
                v = row[col]
                if pd.isna(v):
                    vals.append("n/a")
                elif isinstance(v, float):
                    if "pct" in col or "fraction" in col or "ratio" in col or "share" in col:
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(f"{v:.2f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    summary = f"""# Pump-Weighted Global Contribution Estimate

## Why This Exists

The project has three different percentages that must not be mixed:

1. in-regime saving, e.g. `{neutral_saving:.2f}%` inside the split-corrected
   48-case `neutral_untyped` representative set;
2. coverage of an opportunity pool, e.g. full coverage of the representative
   `neutral_untyped` split-corrected sample;
3. estimated global pump contribution, which must account for how much baseline
   pump each regime actually carries.

This file estimates (3) from already-run casebooks.  It is not a new controller
experiment.

## Main Contribution Table

{md_table(contribution, [
    "family",
    "role",
    "all_window_prevalence_pct",
    "estimated_pump_share_pct",
    "active_or_applicable_fraction",
    "in_regime_saving_pct",
    "equal_window_global_contribution_pct",
    "pump_weighted_global_contribution_pct",
    "cost_note",
    "additivity_note",
])}

Additive estimates:

- broad 48-case operating point, equal-window estimate:
  **{total_equal_broad:.2f}%**
- broad 48-case operating point, pump-weighted casebook estimate:
  **{total_pump_weighted_broad:.2f}%**
- cleaner 40-case operating point, equal-window estimate:
  **{total_equal_clean:.2f}%**
- cleaner 40-case operating point, pump-weighted casebook estimate:
  **{total_pump_weighted_clean:.2f}%**

The equal-window number is conservative because low-risk / quiet windows are
numerous but low-pump.  The pump-weighted number is more relevant, but it is
still an estimate because it uses mined casebook pump intensities rather than a
single fully unfiltered all-window replay denominator.

The broad 48-case and cleaner 40-case neutral/headroom rows are alternatives,
not additive.  The 40-case row excludes the ramp/onset subtype that produces
most of the extra `time>5` exposure in the full 48-case operating point.

## Pump Share Proxy

{md_table(pump_proxy.head(10), [
    "primary_regime",
    "row_pct_all_windows",
    "pump_intensity_m3_per_1h_case",
    "estimated_pump_share_pct",
    "status",
])}

## Forecast Sample Coverage Audit

{md_table(audit, [
    "set",
    "cases",
    "active_cases",
    "inactive_cases",
    "missing_sample_rows",
    "suppression_rows",
    "missing_sample_row_ratio",
])}

The legacy representative 48-case run mixed test and validation timestamps while
the replay interface loaded only the test split.  Its 20 inactive cases were
therefore `missing_sample` fail-closed rows.  Re-running those same 20 timestamps
with `--replay-split validation` recovers learned forecast samples and activates
all 20 cases.  The legacy oracle-missing row is retained only as evidence of the
split mismatch.

## Decision

Use the global result as the current pump-weighted estimate, with the split fix
disclosed:

- broad 48-case equal-window contribution: about **{total_equal_broad:.1f}%**;
- broad 48-case pump-weighted casebook estimate: about **{total_pump_weighted_broad:.1f}%**;
- cleaner 40-case equal-window contribution: about **{total_equal_clean:.1f}%**;
- cleaner 40-case pump-weighted casebook estimate: about **{total_pump_weighted_clean:.1f}%**.

Wording-safe headline: "the cleaner neutral/headroom point is already close to
the 20% pump-weighted target with much lower safety-margin cost; the full
48-case point exceeds the target but must be labeled as the aggressive Pareto
endpoint."

The paper should still report this as an estimate, not a universal all-window
replay, because regime pump shares are built from mined casebook intensities.
"""
    (OUT / "pump_weighted_global_contribution_summary.md").write_text(summary)


if __name__ == "__main__":
    main()
