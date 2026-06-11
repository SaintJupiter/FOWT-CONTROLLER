#!/usr/bin/env python3
"""Summarize broad PSC No.4 generalization evidence.

This script consolidates the broader positive / negative / background 6h pools
after the 2026-06-03 direction change: optimize for mechanisms that generalize,
not for a fixed 18% target.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
RUNS = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs"
OUT = BASE / "psc_4hao_generalization_20260603"


def _load_cases() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pool in ["positive_pool", "negative_pool", "background_pool"]:
        path = RUNS / pool / "refresh_on_case_deltas.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["pool"] = pool
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no refresh_on_case_deltas.csv files under {RUNS}")
    out = pd.concat(frames, ignore_index=True)
    out["strict_accept"] = (
        (out["saved_m3"] > 0.0)
        & (out["d_fallback_s"] <= 0.0)
        & (out["d_time_gt5_s"] <= 0.0)
        & (out["d_time_gt4_s"] <= 0.0)
        & (out["p95_axis_deg"] < 3.75)
        & (out["d_p95_axis_deg"] <= 1.25)
    )
    out["mechanism_decision"] = "reject_or_negative_control"
    out.loc[
        out["stratum"].eq("neutral_mhs_broader") & out["strict_accept"],
        "mechanism_decision",
    ] = "promote_main_gate"
    out.loc[
        out["stratum"].eq("lowrisk_stable_redundant_candidate") & out["strict_accept"],
        "mechanism_decision",
    ] = "keep_as_small_free_gate"
    out.loc[
        out["stratum"].eq("transient_peak_future_decay") & out["strict_accept"],
        "mechanism_decision",
    ] = "keep_as_narrow_specialist"
    out.loc[
        out["stratum"].isin(["direction_reversal_boundary", "reintensification_boundary"])
        & out["strict_accept"],
        "mechanism_decision",
    ] = "use_as_release_rule_probe_only"
    return out


def _summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratum = (
        df.groupby(["pool", "stratum"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            strict_accept_cases=("strict_accept", "sum"),
            baseline_pump_m3=("pump_m3_baseline", "sum"),
            refresh_on_pump_m3=("pump_m3", "sum"),
            saved_m3=("saved_m3", "sum"),
            d_time_gt3_s=("d_time_gt3_s", "sum"),
            d_time_gt4_s=("d_time_gt4_s", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_p95_axis_deg=("p95_axis_deg", "max"),
            max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
        )
        .reset_index()
    )
    stratum["saving_pct"] = 100.0 * stratum["saved_m3"] / stratum["baseline_pump_m3"].clip(lower=1e-9)

    strict = (
        df[df["strict_accept"]]
        .groupby(["pool", "stratum", "mechanism_decision"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            baseline_pump_m3=("pump_m3_baseline", "sum"),
            saved_m3=("saved_m3", "sum"),
            d_time_gt4_s=("d_time_gt4_s", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            max_p95_axis_deg=("p95_axis_deg", "max"),
            max_d_p95_axis_deg=("d_p95_axis_deg", "max"),
            case_ids=("case_id", lambda x: ";".join(map(str, x))),
        )
        .reset_index()
    )
    if not strict.empty:
        strict["saving_pct"] = 100.0 * strict["saved_m3"] / strict["baseline_pump_m3"].clip(lower=1e-9)
    return stratum, strict


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in df[cols].to_dict("records"):
        vals = []
        for col in cols:
            value = rec[col]
            if isinstance(value, float):
                if col.endswith("_pct"):
                    vals.append(f"{value:.2f}%")
                elif col.endswith("_s"):
                    vals.append(f"{value:.0f}")
                elif col.endswith("_deg"):
                    vals.append(f"{value:.2f}")
                else:
                    vals.append(f"{value:.1f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_report(stratum: pd.DataFrame, strict: pd.DataFrame) -> None:
    neutral = stratum[stratum["stratum"].eq("neutral_mhs_broader")].iloc[0]
    relief = stratum[stratum["stratum"].eq("transient_peak_future_decay")].iloc[0]
    negative = stratum[stratum["pool"].eq("negative_pool")].copy()

    lines = [
        "# PSC No.4 Generalization Summary - 2026-06-03",
        "",
        "## Decision",
        "",
        "Do not optimize toward a fixed 18% target.  The broad result says the current refresh-on No.4 mechanism should be promoted only for the neutral/headroom family, kept narrow for clean relief cases, and rejected or released early in boundary/reintensification regimes.",
        "",
        "## Main Evidence",
        "",
        "- `neutral_mhs_broader` is the current broad positive domain: "
        f"{int(neutral['cases'])} cases, {neutral['saving_pct']:.2f}% saving, "
        f"fallback delta {neutral['d_fallback_s']:.0f}s, time>5 delta {neutral['d_time_gt5_s']:.0f}s.",
        "- `transient_peak_future_decay` is not safe as a broad always-on action: "
        f"{int(relief['cases'])} cases, {relief['saving_pct']:.2f}% saving, "
        f"but fallback delta {relief['d_fallback_s']:.0f}s and time>5 delta {relief['d_time_gt5_s']:.0f}s.",
        "- Negative/boundary pools confirm the same failure mode: pump can drop while time>4/time>5 and p95 posture debt rise.  These are release-rule and negative-control data, not mainline savings.",
        "",
        "## Stratum Summary",
        "",
        _md_table(
            stratum.sort_values(["pool", "stratum"]),
            [
                "pool",
                "stratum",
                "cases",
                "strict_accept_cases",
                "saved_m3",
                "saving_pct",
                "d_time_gt4_s",
                "d_time_gt5_s",
                "d_fallback_s",
                "max_p95_axis_deg",
                "max_d_p95_axis_deg",
            ],
        ),
        "",
        "## Strict Accepted Mechanism Slices",
        "",
        _md_table(
            strict.sort_values(["mechanism_decision", "pool", "stratum"]),
            [
                "mechanism_decision",
                "pool",
                "stratum",
                "cases",
                "saved_m3",
                "saving_pct",
                "d_time_gt4_s",
                "d_time_gt5_s",
                "d_fallback_s",
                "max_p95_axis_deg",
            ],
        )
        if not strict.empty
        else "_none_",
        "",
        "## Next Execution",
        "",
        "1. Build a `psc_4hao_generalized_gate_v1` profile that defaults to the neutral/headroom gate and rejects broad relief/boundary by default.",
        "2. Add a relief specialist only for strict-clean relief rows: no fallback, no time>4/time>5 debt, and p95-axis below the comfort band.",
        "3. Use direction-reversal and reintensification rows as negative controls and release-rule probes, not as additive saving pools.",
        "4. Re-run the full positive/negative/background pools after any controller change; acceptance is improvement in the stratum table, not a single target percentage.",
    ]
    (OUT / "psc_4hao_generalization_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    stratum, strict = _summaries(cases)
    cases.to_csv(OUT / "psc_4hao_generalization_case_table.csv", index=False)
    stratum.to_csv(OUT / "psc_4hao_generalization_stratum_summary.csv", index=False)
    strict.to_csv(OUT / "psc_4hao_generalization_strict_accept_summary.csv", index=False)
    _write_report(stratum, strict)
    print(OUT / "psc_4hao_generalization_readout.md")
    print(stratum.to_string(index=False))
    print(strict.to_string(index=False))


if __name__ == "__main__":
    main()
