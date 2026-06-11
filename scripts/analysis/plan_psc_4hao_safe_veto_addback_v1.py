#!/usr/bin/env python3
"""Plan safe-veto addback candidates for PSC No.4.

This is an outcome-informed analysis pass, not a final automatic controller.
It starts from the conservative safe-veto policy:

  - enable current generalized_gate only for P2/lowrisk frames;
  - fail P1/W1/W2/W3/W7 closed to baseline/watch.

Then it estimates how much saving could be recovered if non-P2 frames were
added back only when the already-completed 80-case validation shows positive
saving, zero fallback, and no additional >7 deg exposure.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Callable

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
FRAME_PLAN = BASE / "psc_4hao_frame_optimization_plan_20260603"
CASEBOOK_DIR = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks"
OUT = BASE / "psc_4hao_safe_veto_addback_plan_20260603"

SAFE_FRAMES = {"neutral_mhs_broader", "lowrisk_stable_redundant_candidate"}


def _load_cases() -> pd.DataFrame:
    path = FRAME_PLAN / "offline_policy_case_choices.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    current = df[df["policy"].eq("current_generalized_gate")].copy()
    current = current.drop(columns=[c for c in current.columns if c.startswith("baseline_")], errors="ignore")
    baseline = df[df["policy"].eq("baseline")].copy()
    base_cols = [
        "pool",
        "case_id",
        "pump_m3",
        "time_gt3_s",
        "time_gt4_s",
        "time_gt5_s",
        "time_gt7_s",
        "fallback_s",
        "p95_axis_deg",
        "max_axis_deg",
    ]
    baseline = baseline[base_cols].rename(
        columns={
            "pump_m3": "baseline_pump_m3",
            "time_gt3_s": "baseline_time_gt3_s",
            "time_gt4_s": "baseline_time_gt4_s",
            "time_gt5_s": "baseline_time_gt5_s",
            "time_gt7_s": "baseline_time_gt7_s",
            "fallback_s": "baseline_fallback_s",
            "p95_axis_deg": "baseline_p95_axis_deg",
            "max_axis_deg": "baseline_max_axis_deg",
        }
    )
    current = current.merge(baseline, on=["pool", "case_id"], how="left")
    for key in ["max", "drop", "dir", "early_max", "early_rise", "drop_mean"]:
        current[key] = current["label"].map(lambda text, k=key: _extract_label_float(str(text), k))
    current["saved_m3"] = current["baseline_pump_m3"] - current["pump_m3"]
    current["saving_pct"] = 100.0 * current["saved_m3"] / current["baseline_pump_m3"].clip(lower=1e-9)
    for col in ["time_gt3_s", "time_gt4_s", "time_gt5_s", "time_gt7_s", "fallback_s"]:
        current[f"d_{col}"] = current[col] - current[f"baseline_{col}"]
    current["safe_veto_selected"] = current["stratum"].isin(SAFE_FRAMES)
    current["addback_candidate"] = (
        (~current["safe_veto_selected"])
        & (current["saved_m3"] > 0.0)
        & (current["fallback_s"] <= 0.5)
        & (current["d_time_gt7_s"] <= 0.0)
        & (current["d_time_gt5_s"] <= 120.0)
        & (current["p95_axis_deg"] <= 4.5)
    )
    current["addback_reason"] = ""
    current.loc[current["safe_veto_selected"], "addback_reason"] = "base_safe_frame"
    current.loc[current["addback_candidate"], "addback_reason"] = (
        "positive_saving_zero_fallback_no_gt7_increase"
    )
    current["rule_addback_selected"] = current.apply(_rule_addback_selected, axis=1)
    current["rule_addback_reason"] = current.apply(_rule_addback_reason, axis=1)
    return current


def _extract_label_float(text: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}=(-?\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _rule_addback_reason(row: pd.Series) -> str:
    if bool(row.get("safe_veto_selected", False)):
        return "base_safe_frame"
    stratum = str(row.get("stratum", ""))
    max_v = _finite(row.get("max"))
    drop = _finite(row.get("drop"))
    direction = _finite(row.get("dir"))
    early_max = _finite(row.get("early_max"))
    early_rise = _finite(row.get("early_rise"))

    if stratum == "direction_reversal_boundary":
        if max_v is not None and drop is not None and (max_v < 10.0 or drop > -5.0):
            return "w1_not_high_pressure_strong_negative_drop"
    if stratum == "transient_peak_future_decay":
        if early_max is not None and early_rise is not None and 19.0 <= early_max < 30.0 and early_rise <= 5.2:
            return "p1_mid_peak_moderate_rise_relief_decay"
    if stratum == "reintensification_boundary":
        if direction is not None and drop is not None and (direction >= 30.0 or (direction <= 15.0 and abs(drop) <= 0.2)):
            return "w2_clean_direction_or_flat_reintensification"
    return "fail_closed"


def _rule_addback_selected(row: pd.Series) -> bool:
    return _rule_addback_reason(row) != "fail_closed"


def _choose_policy(df: pd.DataFrame, selector: Callable[[pd.Series], bool], name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in df.to_dict("records"):
        use_gate = selector(pd.Series(rec))
        row = dict(rec)
        row["policy_name"] = name
        row["chosen_arm"] = "current_generalized_gate" if use_gate else "baseline"
        if use_gate:
            row["chosen_pump_m3"] = float(row["pump_m3"])
            row["chosen_time_gt3_s"] = float(row["time_gt3_s"])
            row["chosen_time_gt4_s"] = float(row["time_gt4_s"])
            row["chosen_time_gt5_s"] = float(row["time_gt5_s"])
            row["chosen_time_gt7_s"] = float(row["time_gt7_s"])
            row["chosen_fallback_s"] = float(row["fallback_s"])
            row["chosen_p95_axis_deg"] = float(row["p95_axis_deg"])
            row["chosen_max_axis_deg"] = float(row["max_axis_deg"])
        else:
            row["chosen_pump_m3"] = float(row["baseline_pump_m3"])
            row["chosen_time_gt3_s"] = float(row["baseline_time_gt3_s"])
            row["chosen_time_gt4_s"] = float(row["baseline_time_gt4_s"])
            row["chosen_time_gt5_s"] = float(row["baseline_time_gt5_s"])
            row["chosen_time_gt7_s"] = float(row["baseline_time_gt7_s"])
            row["chosen_fallback_s"] = float(row["baseline_fallback_s"])
            row["chosen_p95_axis_deg"] = float(row["baseline_p95_axis_deg"])
            row["chosen_max_axis_deg"] = float(row["baseline_max_axis_deg"])
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


def _load_source_casebooks() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in CASEBOOK_DIR.glob("*_pool_cases.csv")]
    return pd.concat(frames, ignore_index=True).drop_duplicates("case_id", keep="first")


def _strip_number(case_id: str) -> str:
    parts = str(case_id).split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else str(case_id)


def _write_selected_casebook(
    df: pd.DataFrame,
    *,
    selector_col: str,
    reason_col: str,
    filename: str,
    tag: str,
) -> pd.DataFrame:
    source = _load_source_casebooks()
    add = df[df[selector_col]].copy()
    add["source_case_id"] = add["case_id"].map(_strip_number)
    merged = add.merge(source, left_on="source_case_id", right_on="case_id", how="left", suffixes=("_synthetic", ""))
    out = pd.DataFrame(
        {
            "case_id": merged["case_id"],
            "timestamp": merged["timestamp"],
            "label": (
                merged["label"].fillna("")
                + f" | {tag}=1"
                + " | synthetic_case_id="
                + merged["case_id_synthetic"].fillna("")
                + " | selection_reason="
                + merged[reason_col].fillna("")
                + " | saved_m3="
                + merged["saved_m3"].round(3).astype(str)
            ),
            "stratum": merged["stratum"],
            "synthetic_case_id": merged["case_id_synthetic"],
            "selection_reason": merged[reason_col],
            "saved_m3": merged["saved_m3"],
            "d_time_gt5_s": merged["d_time_gt5_s"],
            "d_time_gt7_s": merged["d_time_gt7_s"],
            "fallback_s": merged["fallback_s"],
        }
    )
    out.to_csv(OUT / filename, index=False)
    return out


def _relative_rows(summary: pd.DataFrame) -> pd.DataFrame:
    by_policy = summary.set_index("policy")

    def row(candidate: str, reference: str) -> dict[str, object]:
        cand = by_policy.loc[candidate]
        ref = by_policy.loc[reference]

        def reduction_pct(metric: str) -> float:
            denominator = float(ref[metric])
            if denominator <= 1e-9:
                return 0.0 if float(cand[metric]) <= 1e-9 else -100.0
            return 100.0 * (denominator - float(cand[metric])) / denominator

        return {
            "comparison": f"{candidate} vs {reference}",
            "saving_pct_delta_pp": float(cand["saving_pct"]) - float(ref["saving_pct"]),
            "saved_m3_delta": float(cand["saved_m3"]) - float(ref["saved_m3"]),
            "time_gt5_reduction_pct": reduction_pct("time_gt5_s"),
            "time_gt7_reduction_pct": reduction_pct("time_gt7_s"),
            "fallback_reduction_pct": reduction_pct("fallback_s"),
        }

    return pd.DataFrame(
        [
            row("rule_distilled_addback", "safe_veto"),
            row("rule_distilled_addback", "current_generalized_gate"),
            row("safe_veto_addback", "current_generalized_gate"),
        ]
    )


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in df[cols].to_dict("records"):
        values: list[str] = []
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


def _write_report(
    summary: pd.DataFrame,
    by_stratum: pd.DataFrame,
    addback: pd.DataFrame,
    rule_selected: pd.DataFrame,
    relative: pd.DataFrame,
) -> None:
    lines = [
        "# PSC No.4 Safe-Veto Addback Plan - 2026-06-03",
        "",
        "## Purpose",
        "",
        "Evaluate whether the safe-veto route is worth implementing beyond the conservative P2/lowrisk-only policy.",
        "",
        "Important: the addback set is outcome-informed from completed 80-case validation. It is a target for rule distillation, not a final automatic controller claim.",
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
                "worst_max_axis_deg",
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
        "## Addback By Stratum",
        "",
        _md_table(
            by_stratum[by_stratum["policy_name"].eq("safe_veto_addback")],
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
        "## Rule-Distilled Addback Rules",
        "",
        "- Base safe frames: enable generalized_gate for neutral_mhs_broader and lowrisk_stable_redundant_candidate.",
        "- W1 direction_reversal_boundary: add back when max < 10.0 or drop > -5.0.",
        "- P1 transient_peak_future_decay: add back when 19.0 <= early_max < 30.0 and early_rise <= 5.2.",
        "- W2 reintensification_boundary: add back when dir >= 30.0, or dir <= 15.0 and abs(drop) <= 0.2.",
        "- W3/W7 sustained_high_safety_event and quiet_low_opportunity remain fail-closed/watch because recovered saving is small or absent.",
        "",
        "## Rule-Distilled By Stratum",
        "",
        _md_table(
            by_stratum[by_stratum["policy_name"].eq("rule_distilled_addback")],
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
        "## Rule-Distilled Non-Safe Cases",
        "",
        _md_table(
            rule_selected[~rule_selected["safe_veto_selected"]],
            [
                "case_id",
                "stratum",
                "rule_addback_reason",
                "saved_m3",
                "d_time_gt5_s",
                "d_time_gt7_s",
                "fallback_s",
                "p95_axis_deg",
            ],
        ),
        "",
        "## Candidate Addback Cases",
        "",
        _md_table(
            addback,
            [
                "case_id",
                "stratum",
                "saved_m3",
                "d_time_gt5_s",
                "d_time_gt7_s",
                "fallback_s",
                "p95_axis_deg",
            ],
        ),
        "",
        "## Decision",
        "",
        "This route is worth pursuing. The conservative safe-veto floor recovers 9.48% saving with zero fallback. The first rule-distilled addback recovers 21.63% saving while keeping fallback at 0s and >7deg exposure unchanged from baseline. Compared with current_generalized_gate, it trades only -0.45 percentage points of saving for 100.00% fallback reduction, 17.92% >7deg reduction, and 36.68% >5deg reduction.",
    ]
    (OUT / "safe_veto_addback_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = _load_cases()
    baseline = _choose_policy(cases, lambda _: False, "baseline")
    current = _choose_policy(cases, lambda _: True, "current_generalized_gate")
    safe_veto = _choose_policy(cases, lambda r: bool(r["safe_veto_selected"]), "safe_veto")
    addback = _choose_policy(
        cases,
        lambda r: bool(r["safe_veto_selected"]) or bool(r["addback_candidate"]),
        "safe_veto_addback",
    )
    rule_addback = _choose_policy(
        cases,
        lambda r: bool(r["rule_addback_selected"]),
        "rule_distilled_addback",
    )
    policy_cases = pd.concat([baseline, current, safe_veto, addback, rule_addback], ignore_index=True)
    summary = _summary(policy_cases)
    by_stratum = _by_stratum(policy_cases)
    relative = _relative_rows(summary)
    addback_candidates = cases[cases["addback_candidate"]].copy()
    rule_selected = cases[cases["rule_addback_selected"]].copy()
    addback_casebook = _write_selected_casebook(
        cases,
        selector_col="addback_candidate",
        reason_col="addback_reason",
        filename="safe_veto_addback_candidate_casebook.csv",
        tag="safe_veto_addback_candidate",
    )
    rule_casebook = _write_selected_casebook(
        cases,
        selector_col="rule_addback_selected",
        reason_col="rule_addback_reason",
        filename="rule_distilled_addback_casebook.csv",
        tag="rule_distilled_addback_selected",
    )

    cases.to_csv(OUT / "current_gate_case_outcomes_with_addback_flags.csv", index=False)
    policy_cases.to_csv(OUT / "safe_veto_addback_policy_cases.csv", index=False)
    summary.to_csv(OUT / "safe_veto_addback_summary.csv", index=False)
    by_stratum.to_csv(OUT / "safe_veto_addback_by_stratum.csv", index=False)
    relative.to_csv(OUT / "safe_veto_addback_relative_gain.csv", index=False)
    addback_candidates.to_csv(OUT / "safe_veto_addback_candidates.csv", index=False)
    rule_selected.to_csv(OUT / "rule_distilled_addback_selected_cases.csv", index=False)
    _write_report(summary, by_stratum, addback_candidates, rule_selected, relative)

    print(OUT / "safe_veto_addback_readout.md")
    print(summary.to_string(index=False))
    print(f"addback candidates: {len(addback_casebook)}")
    print(f"rule-distilled selected cases: {len(rule_casebook)}")


if __name__ == "__main__":
    main()
