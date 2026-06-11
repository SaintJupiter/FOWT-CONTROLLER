#!/usr/bin/env python3
"""Scan existing PSC replay outputs for safe extra pump-saving sources.

This is a source-mining step for the 18% saving plan. It does not introduce a
new controller. It asks whether already-completed No.4/fresh/broader runs hide
additional fallback-free case-level candidates that were not captured by the
previous action-family priority list.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
OUT = BASE / "psc_18pct_saving_20260602"
OVERNIGHT = BASE / "psc_4hao_25pct_overnight_v1"
THREE_HAO = BASE / "psc_structural_selector_96case_pair_3hao_v2_learned_v1" / "structural_selector_case_decisions.csv"
FRESH = BASE / "psc_4hao_refresh_validation_6h_fresh_20260601"
BROADER = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs"

NO3_SAVED_M3 = 5999.1490457470445
DENOMINATOR_96CASE_M3 = 45880.13269320306
TARGET_SAVED_M3 = 0.18 * DENOMINATOR_96CASE_M3


POLICY_LIMITS = {
    "strict": {"gt5": 0.0, "dp95": 0.50},
    "balanced": {"gt5": 30.0, "dp95": 1.35},
    "appendix": {"gt5": 60.0, "dp95": 1.35},
}


def _opened_by_no3() -> set[str]:
    df = pd.read_csv(THREE_HAO)
    return set(df.loc[df["use_economy"].astype(float) > 0.5, "case_id"].astype(str))


def _policy_mask(df: pd.DataFrame, policy: str) -> pd.Series:
    lim = POLICY_LIMITS[policy]
    return (
        (df["saved_m3"] > 0.0)
        & (df["d_fallback_s"] <= 0.0)
        & (df["d_time_gt5_s"] <= lim["gt5"])
        & (df["d_p95_axis_deg"] <= lim["dp95"])
        & (df["p95_axis_deg"] < 5.0)
    )


def _rank(df: pd.DataFrame, policy: str, source_pool: str) -> pd.DataFrame:
    cand = df[_policy_mask(df, policy)].copy()
    if cand.empty:
        return cand
    cand["policy"] = policy
    cand["source_pool"] = source_pool
    cand["selector_score"] = (
        cand["saved_m3"]
        - 25.0 * cand["d_p95_axis_deg"].clip(lower=0.0)
        - 0.8 * cand["d_time_gt5_s"].clip(lower=0.0)
        - 0.15 * cand["d_time_gt4_s"].clip(lower=0.0)
    )
    cand = cand.sort_values(
        ["case_id", "selector_score", "saved_m3"],
        ascending=[True, False, False],
    )
    return cand.groupby("case_id", as_index=False).head(1).copy()


def scan_96case_all_variants() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(OVERNIGHT / "case_metrics.csv")
    opened = _opened_by_no3()
    df = df[~df["case_id"].astype(str).isin(opened)].copy()
    df["source_pool"] = "96case_existing_4hao_variants"
    selections = []
    summaries = []
    for policy in POLICY_LIMITS:
        selected = _rank(df, policy, "96case_existing_4hao_variants")
        selections.append(selected)
        add_m3 = float(selected["saved_m3"].sum()) if not selected.empty else 0.0
        total_m3 = NO3_SAVED_M3 + add_m3
        summaries.append(
            {
                "source_pool": "96case_existing_4hao_variants",
                "policy": policy,
                "accepted_cases": int(len(selected)),
                "additional_saved_m3": add_m3,
                "projected_total_saved_m3": total_m3,
                "projected_saving_pct": 100.0 * total_m3 / DENOMINATOR_96CASE_M3,
                "gap_to_18pct_m3": TARGET_SAVED_M3 - total_m3,
                "d_fallback_s": float(selected["d_fallback_s"].sum()) if not selected.empty else 0.0,
                "d_time_gt4_s": float(selected["d_time_gt4_s"].sum()) if not selected.empty else 0.0,
                "d_time_gt5_s": float(selected["d_time_gt5_s"].sum()) if not selected.empty else 0.0,
                "max_p95_axis_deg": float(selected["p95_axis_deg"].max()) if not selected.empty else 0.0,
                "max_d_p95_axis_deg": float(selected["d_p95_axis_deg"].max()) if not selected.empty else 0.0,
                "case_ids": ";".join(selected["case_id"].astype(str).tolist()) if not selected.empty else "",
            }
        )
    selected_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return selected_df, pd.DataFrame(summaries)


def _normalize_delta_table(path: Path, source_pool: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "d_time_gt4_s" not in df.columns:
        df["d_time_gt4_s"] = 0.0
    if "stratum" not in df.columns:
        df["stratum"] = df["label"].astype(str).str.extract(r"mixed_regime=([^|]+)", expand=False).fillna("")
    if "validation_role" not in df.columns:
        df["validation_role"] = ""
    df["run_id"] = "refresh_on"
    df["candidate_name"] = "fresh_or_broader_refresh_on"
    df["source_pool"] = source_pool
    return df


def scan_fresh_and_broader() -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = [
        (FRESH / "mixed24" / "stage_a_refresh_on_case_deltas.csv", "fresh_mixed24_refresh_on"),
        (FRESH / "risky20" / "stage_a_refresh_on_case_deltas.csv", "fresh_risky20_refresh_on"),
        (BROADER / "negative_pool" / "refresh_on_case_deltas.csv", "broader_negative_refresh_on"),
        (BROADER / "background_pool" / "refresh_on_case_deltas.csv", "broader_background_refresh_on"),
    ]
    frames = [_normalize_delta_table(path, name) for path, name in sources if path.exists()]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    selections = []
    summaries = []
    for source_pool, group in df.groupby("source_pool", sort=True):
        for policy in POLICY_LIMITS:
            selected = _rank(group, policy, source_pool)
            selections.append(selected)
            summaries.append(
                {
                    "source_pool": source_pool,
                    "policy": policy,
                    "accepted_cases": int(len(selected)),
                    "accepted_saved_m3": float(selected["saved_m3"].sum()) if not selected.empty else 0.0,
                    "baseline_pump_m3": float(selected["pump_m3_baseline"].sum()) if not selected.empty else 0.0,
                    "accepted_pool_saving_pct": (
                        100.0 * float(selected["saved_m3"].sum()) / max(float(selected["pump_m3_baseline"].sum()), 1e-9)
                        if not selected.empty
                        else 0.0
                    ),
                    "d_fallback_s": float(selected["d_fallback_s"].sum()) if not selected.empty else 0.0,
                    "d_time_gt4_s": float(selected["d_time_gt4_s"].sum()) if not selected.empty else 0.0,
                    "d_time_gt5_s": float(selected["d_time_gt5_s"].sum()) if not selected.empty else 0.0,
                    "max_p95_axis_deg": float(selected["p95_axis_deg"].max()) if not selected.empty else 0.0,
                    "max_d_p95_axis_deg": float(selected["d_p95_axis_deg"].max()) if not selected.empty else 0.0,
                    "case_ids": ";".join(selected["case_id"].astype(str).tolist()) if not selected.empty else "",
                }
            )
    selected_df = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return selected_df, pd.DataFrame(summaries)


def write_readout(summary_96: pd.DataFrame, summary_fresh: pd.DataFrame) -> None:
    best = summary_96.sort_values("projected_saving_pct", ascending=False).iloc[0]
    lines = [
        "# PSC 18% New Source Scan",
        "",
        "This scan reuses already-completed replays. It is a mining pass, not a new controller.",
        "",
        "## 96-case All-Variant Result",
        "",
        "| policy | accepted | add m3 | projected | gap to 18% | d_gt5 | max p95 | cases |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_96.to_dict("records"):
        lines.append(
            "| {policy} | {accepted_cases} | {additional_saved_m3:.1f} | "
            "{projected_total_saved_m3:.1f} / {projected_saving_pct:.2f}% | "
            "{gap_to_18pct_m3:.1f} | {d_time_gt5_s:.0f} | {max_p95_axis_deg:.2f} | "
            "{case_ids} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Fresh/Broader Source Clues",
            "",
            "| source | policy | accepted | accepted m3 | d_gt5 | max p95 | cases |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary_fresh.to_dict("records"):
        lines.append(
            "| {source_pool} | {policy} | {accepted_cases} | {accepted_saved_m3:.1f} | "
            "{d_time_gt5_s:.0f} | {max_p95_axis_deg:.2f} | {case_ids} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                f"The best already-run 96-case all-variant scan reaches "
                f"{best['projected_saving_pct']:.2f}% with {best['d_time_gt5_s']:.0f}s "
                f"gt5 debt, leaving {best['gap_to_18pct_m3']:.1f} m3 to 18%."
            ),
            "",
            "So the current evidence still does not reach 18%. The most useful next work is not another selector over the same completed variants, but a new runtime-safe mechanism for high-saving boundary/relief cases.",
        ]
    )
    (OUT / "new_source_scan_readout.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selected_96, summary_96 = scan_96case_all_variants()
    selected_fresh, summary_fresh = scan_fresh_and_broader()
    selected_96.to_csv(OUT / "all_variant_96case_selected_cases.csv", index=False)
    summary_96.to_csv(OUT / "all_variant_96case_summary.csv", index=False)
    selected_fresh.to_csv(OUT / "fresh_broader_selected_safe_sources.csv", index=False)
    summary_fresh.to_csv(OUT / "fresh_broader_safe_source_summary.csv", index=False)
    write_readout(summary_96, summary_fresh)
    print(summary_96.to_string(index=False))
    print(summary_fresh.to_string(index=False))


if __name__ == "__main__":
    main()
