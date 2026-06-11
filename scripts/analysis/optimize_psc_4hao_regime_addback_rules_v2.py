#!/usr/bin/env python3
"""Optimize PSC No.4 regime addback rules from the frozen 80-case evidence.

This is still an offline policy-distillation pass. It uses only case label
features that describe pre-action forecast shapes, not outcome metrics, for the
candidate rule itself. Outcome metrics are used only to score and audit the
candidate after selection.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
SAFE_VETO = BASE / "psc_4hao_safe_veto_addback_plan_20260603"
CASEBOOK_DIR = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks"
OUT = BASE / "psc_4hao_regime_addback_rules_v2_20260603"

SAFE_FRAMES = {"neutral_mhs_broader", "lowrisk_stable_redundant_candidate"}


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _load_cases() -> pd.DataFrame:
    cases = pd.read_csv(SAFE_VETO / "current_gate_case_outcomes_with_addback_flags.csv")
    for col in ("max", "drop", "dir", "early_max", "early_rise", "drop_mean"):
        if col not in cases.columns:
            cases[col] = cases["label"].map(lambda text, key=col: _extract_label_float(str(text), key))
    cases["safe_veto_selected"] = cases["stratum"].isin(SAFE_FRAMES)
    cases["regime_v2_selected"] = cases.apply(_regime_v2_selected, axis=1)
    cases["regime_v2_reason"] = cases.apply(_regime_v2_reason, axis=1)
    return cases


def _extract_label_float(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _regime_v2_reason(row: pd.Series) -> str:
    if bool(row.get("safe_veto_selected", False)):
        return "base_safe_frame"

    stratum = str(row.get("stratum", ""))
    max_v = _finite(row.get("max"))
    drop = _finite(row.get("drop"))
    direction = _finite(row.get("dir"))
    early_max = _finite(row.get("early_max"))
    early_rise = _finite(row.get("early_rise"))
    drop_mean = _finite(row.get("drop_mean"))

    if stratum == "direction_reversal_boundary":
        if max_v is not None and drop is not None and (max_v < 10.0 or drop > -5.0):
            return "w1_low_pressure_or_not_strong_negative_drop"
    if stratum == "reintensification_boundary":
        if direction is not None and drop is not None and (direction >= 30.0 or (direction <= 15.0 and abs(drop) <= 0.2)):
            return "w2_clean_direction_or_flat_reintensification"
    if stratum == "transient_peak_future_decay":
        if early_max is None or early_rise is None or drop_mean is None:
            return "fail_closed"
        mid_peak_moderate_rise = 19.0 <= early_max < 27.0 and 3.0 <= early_rise <= 5.2
        low_rise_deep_relief = 19.0 <= early_max <= 20.0 and early_rise <= 3.0 and drop_mean >= 6.5
        high_rise_moderate_relief = (
            21.0 <= early_max <= 24.5
            and 6.8 <= early_rise <= 9.5
            and 4.3 <= drop_mean <= 5.95
        )
        if mid_peak_moderate_rise:
            return "p1_mid_peak_moderate_rise"
        if low_rise_deep_relief:
            return "p1_low_rise_deep_relief"
        if high_rise_moderate_relief:
            return "p1_high_rise_moderate_relief"
    return "fail_closed"


def _regime_v2_selected(row: pd.Series) -> bool:
    return _regime_v2_reason(row) != "fail_closed"


def _choose_policy(df: pd.DataFrame, selector: Callable[[pd.Series], bool], name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in df.to_dict("records"):
        use_gate = selector(pd.Series(rec))
        row = dict(rec)
        row["policy_name"] = name
        row["chosen_arm"] = "current_generalized_gate" if use_gate else "baseline"
        if use_gate:
            source_prefix = ""
        else:
            source_prefix = "baseline_"
        row["chosen_pump_m3"] = float(row[f"{source_prefix}pump_m3"])
        row["chosen_time_gt3_s"] = float(row[f"{source_prefix}time_gt3_s"])
        row["chosen_time_gt4_s"] = float(row[f"{source_prefix}time_gt4_s"])
        row["chosen_time_gt5_s"] = float(row[f"{source_prefix}time_gt5_s"])
        row["chosen_time_gt7_s"] = float(row[f"{source_prefix}time_gt7_s"])
        row["chosen_fallback_s"] = float(row[f"{source_prefix}fallback_s"])
        row["chosen_p95_axis_deg"] = float(row[f"{source_prefix}p95_axis_deg"])
        row["chosen_max_axis_deg"] = float(row[f"{source_prefix}max_axis_deg"])
        row["chosen_saved_m3"] = float(row["baseline_pump_m3"]) - float(row["chosen_pump_m3"])
        rows.append(row)
    return pd.DataFrame(rows)


def _summary(policy_cases: pd.DataFrame) -> pd.DataFrame:
    baseline_total = float(policy_cases.groupby("policy_name")["baseline_pump_m3"].sum().iloc[0])
    rows: list[dict[str, object]] = []
    for policy, group in policy_cases.groupby("policy_name", sort=False):
        pump = float(group["chosen_pump_m3"].sum())
        rows.append(
            {
                "policy": policy,
                "cases": int(group["case_id"].nunique()),
                "gate_cases": int(group["chosen_arm"].eq("current_generalized_gate").sum()),
                "pump_m3": pump,
                "saved_m3": baseline_total - pump,
                "saving_pct": 100.0 * (baseline_total - pump) / max(baseline_total, 1e-9),
                "time_gt5_s": float(group["chosen_time_gt5_s"].sum()),
                "time_gt7_s": float(group["chosen_time_gt7_s"].sum()),
                "fallback_s": float(group["chosen_fallback_s"].sum()),
                "worst_p95_axis_deg": float(group["chosen_p95_axis_deg"].max()),
                "worst_max_axis_deg": float(group["chosen_max_axis_deg"].max()),
            }
        )
    return pd.DataFrame(rows)


def _by_stratum(policy_cases: pd.DataFrame) -> pd.DataFrame:
    return (
        policy_cases.groupby(["policy_name", "stratum"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            gate_cases=("chosen_arm", lambda s: int((s == "current_generalized_gate").sum())),
            baseline_pump_m3=("baseline_pump_m3", "sum"),
            pump_m3=("chosen_pump_m3", "sum"),
            saved_m3=("chosen_saved_m3", "sum"),
            time_gt5_s=("chosen_time_gt5_s", "sum"),
            time_gt7_s=("chosen_time_gt7_s", "sum"),
            fallback_s=("chosen_fallback_s", "sum"),
            worst_p95_axis_deg=("chosen_p95_axis_deg", "max"),
        )
        .reset_index()
    )


def _relative(summary: pd.DataFrame) -> pd.DataFrame:
    by_policy = summary.set_index("policy")

    def compare(candidate: str, reference: str) -> dict[str, object]:
        cand = by_policy.loc[candidate]
        ref = by_policy.loc[reference]

        def reduction(metric: str) -> float:
            denom = float(ref[metric])
            if denom <= 1e-9:
                return 0.0 if float(cand[metric]) <= 1e-9 else -100.0
            return 100.0 * (denom - float(cand[metric])) / denom

        return {
            "comparison": f"{candidate} vs {reference}",
            "saving_pct_delta_pp": float(cand["saving_pct"]) - float(ref["saving_pct"]),
            "saved_m3_delta": float(cand["saved_m3"]) - float(ref["saved_m3"]),
            "time_gt5_reduction_pct": reduction("time_gt5_s"),
            "time_gt7_reduction_pct": reduction("time_gt7_s"),
            "fallback_reduction_pct": reduction("fallback_s"),
        }

    return pd.DataFrame(
        [
            compare("regime_addback_v2", "rule_distilled_addback_v1"),
            compare("regime_addback_v2", "current_generalized_gate"),
            compare("regime_addback_v2", "safe_veto"),
            compare("regime_addback_v2", "safe_veto_addback_target"),
        ]
    )


def _load_source_casebooks() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CASEBOOK_DIR.glob("*_pool_cases.csv")]
    return pd.concat(frames, ignore_index=True).drop_duplicates("case_id", keep="first")


def _strip_number(case_id: str) -> str:
    parts = str(case_id).split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else str(case_id)


def _write_casebook(cases: pd.DataFrame) -> pd.DataFrame:
    source = _load_source_casebooks()
    selected = cases[cases["regime_v2_selected"]].copy()
    selected["source_case_id"] = selected["case_id"].map(_strip_number)
    merged = selected.merge(source, left_on="source_case_id", right_on="case_id", how="left", suffixes=("_synthetic", ""))
    out = pd.DataFrame(
        {
            "case_id": merged["case_id"],
            "timestamp": merged["timestamp"],
            "label": (
                merged["label"].fillna("")
                + " | regime_addback_v2_selected=1"
                + " | synthetic_case_id="
                + merged["case_id_synthetic"].fillna("")
                + " | regime_v2_reason="
                + merged["regime_v2_reason"].fillna("")
                + " | saved_m3="
                + merged["saved_m3"].round(3).astype(str)
            ),
            "stratum": merged["stratum"],
            "synthetic_case_id": merged["case_id_synthetic"],
            "regime_v2_reason": merged["regime_v2_reason"],
            "saved_m3": merged["saved_m3"],
            "d_time_gt5_s": merged["d_time_gt5_s"],
            "d_time_gt7_s": merged["d_time_gt7_s"],
            "fallback_s": merged["fallback_s"],
        }
    )
    out.to_csv(OUT / "regime_addback_v2_casebook.csv", index=False)
    return out


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for rec in df[cols].to_dict("records"):
        values = []
        for col in cols:
            value = rec[col]
            if isinstance(value, float):
                if col.endswith("_pct"):
                    values.append(f"{value:.2f}%")
                elif col.endswith("_pp"):
                    values.append(f"{value:+.2f}")
                elif col.endswith("_m3"):
                    values.append(f"{value:.1f}")
                elif col.endswith("_s"):
                    values.append(f"{value:.0f}")
                elif col.endswith("_deg"):
                    values.append(f"{value:.2f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(summary: pd.DataFrame, by_stratum: pd.DataFrame, relative: pd.DataFrame, selected: pd.DataFrame) -> None:
    v2_by = by_stratum[by_stratum["policy_name"].eq("regime_addback_v2")]
    p1 = selected[selected["stratum"].eq("transient_peak_future_decay")]
    lines = [
        "# PSC No.4 Regime Addback Rules v2 - 2026-06-03",
        "",
        "## Decision",
        "",
        "Promote the v2 offline rule candidate for runtime implementation planning. It keeps the mature safe-veto base, preserves W1/W2 safe addbacks, and materially improves P1 by using drop_mean to recover two safe transient-decay cases while rejecting a weak negative case.",
        "",
        "## Policy Summary",
        "",
        _md_table(
            summary,
            [
                "policy",
                "cases",
                "gate_cases",
                "pump_m3",
                "saved_m3",
                "saving_pct",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "worst_p95_axis_deg",
            ],
        ),
        "",
        "## Relative Gain",
        "",
        _md_table(
            relative,
            [
                "comparison",
                "saving_pct_delta_pp",
                "saved_m3_delta",
                "time_gt5_reduction_pct",
                "time_gt7_reduction_pct",
                "fallback_reduction_pct",
            ],
        ),
        "",
        "## v2 By Regime",
        "",
        _md_table(
            v2_by,
            [
                "stratum",
                "cases",
                "gate_cases",
                "saved_m3",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "worst_p95_axis_deg",
            ],
        ),
        "",
        "## P1 v2 Selected Cases",
        "",
        _md_table(
            p1,
            [
                "case_id",
                "regime_v2_reason",
                "early_max",
                "early_rise",
                "drop_mean",
                "saved_m3",
                "d_time_gt5_s",
                "d_time_gt7_s",
                "fallback_s",
            ],
        ),
        "",
        "## Runtime Rule Candidate",
        "",
        "- Base safe frames: enable neutral_mhs_broader and lowrisk_stable_redundant_candidate.",
        "- W1: enable direction_reversal_boundary when max < 10.0 or drop > -5.0.",
        "- W2: enable reintensification_boundary when dir >= 30.0, or dir <= 15.0 and abs(drop) <= 0.2.",
        "- P1 mid-peak: enable when 19.0 <= early_max < 27.0 and 3.0 <= early_rise <= 5.2.",
        "- P1 low-rise deep-relief: enable when 19.0 <= early_max <= 20.0 and early_rise <= 3.0 and drop_mean >= 6.5.",
        "- P1 high-rise moderate-relief: enable when 21.0 <= early_max <= 24.5 and 6.8 <= early_rise <= 9.5 and 4.3 <= drop_mean <= 5.95.",
        "- W3/W7 and quiet-low remain fail-closed/no-op regression strata.",
    ]
    (OUT / "regime_addback_v2_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    baseline = _choose_policy(cases, lambda _: False, "baseline")
    current = _choose_policy(cases, lambda _: True, "current_generalized_gate")
    safe_veto = _choose_policy(cases, lambda r: bool(r["safe_veto_selected"]), "safe_veto")
    rule_v1 = _choose_policy(cases, lambda r: bool(r["rule_addback_selected"]), "rule_distilled_addback_v1")
    target = _choose_policy(cases, lambda r: bool(r["safe_veto_selected"]) or bool(r["addback_candidate"]), "safe_veto_addback_target")
    v2 = _choose_policy(cases, lambda r: bool(r["regime_v2_selected"]), "regime_addback_v2")
    policy_cases = pd.concat([baseline, current, safe_veto, rule_v1, target, v2], ignore_index=True)
    summary = _summary(policy_cases)
    by_stratum = _by_stratum(policy_cases)
    relative = _relative(summary)
    selected = cases[cases["regime_v2_selected"]].copy()
    rejected = cases[~cases["regime_v2_selected"]].copy()
    casebook = _write_casebook(cases)

    cases.to_csv(OUT / "regime_addback_v2_case_outcomes.csv", index=False)
    policy_cases.to_csv(OUT / "regime_addback_v2_policy_cases.csv", index=False)
    summary.to_csv(OUT / "regime_addback_v2_summary.csv", index=False)
    by_stratum.to_csv(OUT / "regime_addback_v2_by_stratum.csv", index=False)
    relative.to_csv(OUT / "regime_addback_v2_relative_gain.csv", index=False)
    selected.to_csv(OUT / "regime_addback_v2_selected_cases.csv", index=False)
    rejected.to_csv(OUT / "regime_addback_v2_rejected_cases.csv", index=False)
    _write_report(summary, by_stratum, relative, selected)

    print(OUT / "regime_addback_v2_readout.md")
    print(summary.to_string(index=False))
    print(f"selected cases: {len(casebook)}")


if __name__ == "__main__":
    main()
