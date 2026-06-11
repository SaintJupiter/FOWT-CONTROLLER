"""Evidence ledger for the 20% pump-saving ceiling question.

This script does not tune the controller. It consolidates existing locked
holdout and round-trip diagnostics to answer whether a safety-neutral 20%
pump reduction is supported by the observed pump pools.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/wind_prediction/pump_saving_20pct_ceiling_v1"
RAW = OUT / "raw_tables"
PAPER = OUT / "paper_ready"
DEBUG = OUT / "debug"

LOCKED = REPO / "outputs/wind_prediction/h120_relief_envelope_learned_a1_locked_holdout_v1"
ROUNDTRIP = REPO / "outputs/wind_prediction/relief_roundtrip_decomposition_step0_v1"
CASCADE = REPO / "outputs/wind_prediction/relief_economy_cascade_roundtrip_diagnostics_v1"
B_DEV = REPO / "outputs/wind_prediction/relief_adaptive_axis_cap_b1_dev_matrix_v1"


def ensure_dirs() -> None:
    for d in (OUT, RAW, PAPER, DEBUG):
        d.mkdir(parents=True, exist_ok=True)


def read_metric_table(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if not {"metric", "value"}.issubset(df.columns):
        raise ValueError(f"Expected metric/value table: {path}")
    return {str(row.metric): float(row.value) for row in df.itertuples(index=False)}


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def pct(x: float, denom: float, digits: int = 2) -> str:
    if denom == 0:
        return "nan"
    return f"{100.0 * x / denom:.{digits}f}%"


def markdown_table(rows: Iterable[dict[str, object]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(out)


def main() -> None:
    ensure_dirs()

    compare = pd.read_csv(LOCKED / "learned_a1_locked_holdout_compare_table.csv")
    per_case = pd.read_csv(LOCKED / "learned_a1_per_case_delta_table.csv")
    trigger = pd.read_csv(LOCKED / "learned_a1_trigger_diagnostics.csv")
    step0 = read_metric_table(ROUNDTRIP / "raw_tables/step0_summary.csv")
    cascade = read_metric_table(CASCADE / "raw_tables/cascade_roundtrip_summary.csv")
    kind = pd.read_csv(ROUNDTRIP / "raw_tables/roundtrip_kind_summary.csv")

    a0 = compare.loc[compare["arm"].eq("A0_learned_v16")].iloc[0]
    a1 = compare.loc[compare["arm"].eq("A1_learned_near_envelope_medium_axis_guard")].iloc[0]
    a0_pump = float(a0["sum_pump"])
    a1_pump = float(a1["sum_pump"])
    a1_saved = a0_pump - a1_pump
    a1_saved_pct = a1_saved / a0_pump

    total_reversible = float(step0["cycle_reversible_m3"])
    addressable_reversible = float(step0["addressable_cycle_reversible_m3"])
    addressable_pump = float(step0["A0_addressable_pump_m3"])
    roundtrip_reduction = float(cascade["roundtrip_proxy_reduction_m3"])
    a1_remaining_roundtrip = float(cascade["A1_roundtrip_proxy_m3"])
    post600 = float(cascade["saved_after_600s_m3"])

    target_rows = []
    for target_pct in [0.08, 0.10, 0.15, 0.20]:
        target_saving = target_pct * a0_pump
        target_rows.append(
            {
                "target_pump_reduction_pct": target_pct,
                "target_saving_m3": target_saving,
                "additional_saving_needed_vs_A1_m3": max(0.0, target_saving - a1_saved),
                "target_exceeds_total_reversible_proxy": target_saving > total_reversible,
                "target_exceeds_addressable_reversible_proxy": target_saving > addressable_reversible,
                "target_exceeds_observed_addressable_pump": target_saving > addressable_pump,
            }
        )
    target_gap = pd.DataFrame(target_rows)

    ceiling_rows = [
        {
            "pool_or_result": "A0_total_pump",
            "m3": a0_pump,
            "pct_of_A0": 1.0,
            "interpretation": "Locked 53-case baseline pump use.",
        },
        {
            "pool_or_result": "A1_current_saved",
            "m3": a1_saved,
            "pct_of_A0": a1_saved_pct,
            "interpretation": "Measured learned 0-60min relief-economy saving with single-axis guard.",
        },
        {
            "pool_or_result": "A0_addressable_pump_proxy",
            "m3": addressable_pump,
            "pct_of_A0": addressable_pump / a0_pump,
            "interpretation": "Pump occurring in read-only safe posture + near-relief windows; not all is avoidable.",
        },
        {
            "pool_or_result": "total_reversible_roundtrip_proxy",
            "m3": total_reversible,
            "pct_of_A0": total_reversible / a0_pump,
            "interpretation": "Upper diagnostic pool of reversible/chatter motion, including genuinely needed reversals.",
        },
        {
            "pool_or_result": "addressable_reversible_roundtrip_proxy",
            "m3": addressable_reversible,
            "pct_of_A0": addressable_reversible / a0_pump,
            "interpretation": "Reversible pool tagged safe and relief-adjacent; closer to free-savings headroom.",
        },
        {
            "pool_or_result": "A1_roundtrip_proxy_reduction",
            "m3": roundtrip_reduction,
            "pct_of_A0": roundtrip_reduction / a0_pump,
            "interpretation": "Portion of A1 saving explained by reduced round-trip proxy.",
        },
        {
            "pool_or_result": "A1_post_600s_cascade_saving",
            "m3": post600,
            "pct_of_A0": post600 / a0_pump,
            "interpretation": "Trajectory cascade saving beyond the immediate 600s trigger window.",
        },
        {
            "pool_or_result": "remaining_A1_roundtrip_proxy",
            "m3": a1_remaining_roundtrip,
            "pct_of_A0": a1_remaining_roundtrip / a0_pump,
            "interpretation": "Remaining reversible proxy after A1; not equivalent to safely avoidable pump.",
        },
    ]
    ceiling = pd.DataFrame(ceiling_rows)

    concentration = per_case.copy()
    concentration["pump_saved_m3"] = (-concentration["A1_minus_A0_pump_m3"]).clip(lower=0.0)
    concentration = concentration.sort_values("pump_saved_m3", ascending=False)
    concentration["cumulative_saved_m3"] = concentration["pump_saved_m3"].cumsum()
    concentration["share_of_A1_saving"] = concentration["pump_saved_m3"] / max(a1_saved, 1e-9)
    concentration["cumulative_share_of_A1_saving"] = concentration["cumulative_saved_m3"] / max(a1_saved, 1e-9)

    b1_path = B_DEV / "raw_tables/b1_dev_summary_metrics.csv"
    b2_path = B_DEV / "raw_tables/b2_dev_summary_metrics.csv"
    b_summary = []
    if b1_path.exists():
        b1 = pd.read_csv(b1_path)
        a1_dev = float(b1.loc[b1["arm"].eq("A1_current"), "pump_m3"].iloc[0])
        b1_dev = float(b1.loc[b1["arm"].eq("B1_adaptive_cap"), "pump_m3"].iloc[0])
        b_summary.append(
            {
                "experiment": "B1_adaptive_axis_cap_dev",
                "reference": "A1_current_dev",
                "delta_pump_m3": b1_dev - a1_dev,
                "interpretation": "No additional cap opportunities fired in the dev matrix.",
            }
        )
    if b2_path.exists():
        b2 = pd.read_csv(b2_path)
        a1_dev = float(b2.loc[b2["arm"].eq("A1_current"), "pump_m3"].iloc[0])
        b2_dev = float(b2.loc[b2["arm"].eq("B2_hold_extension_dev"), "pump_m3"].iloc[0])
        b_summary.append(
            {
                "experiment": "B2_hold_extension_dev",
                "reference": "A1_current_dev",
                "delta_pump_m3": b2_dev - a1_dev,
                "interpretation": "Looser hold/debt caused more catch-up than saving on dev.",
            }
        )
    b_summary_df = pd.DataFrame(b_summary)

    experiment_rows = [
        {
            "priority": 1,
            "experiment": "A1_negative_controls",
            "purpose": "Prove the 6.55% is forecast-attributable, not generic pump suppression.",
            "decision": "Run before making the result a final paper claim if not already done.",
        },
        {
            "priority": 2,
            "experiment": "comfort_pareto_probe",
            "purpose": "If 20% is required, quantify pump saved vs time>3/p95 comfort cost while hard floor stays untouched.",
            "decision": "This is the honest route to larger numbers; it changes the objective.",
        },
        {
            "priority": 3,
            "experiment": "offline_noncausal_pump_floor",
            "purpose": "Build a true optimizer lower bound for pump under the same safety envelope.",
            "decision": "Expensive and optional; use to strengthen the ceiling claim, not for immediate tuning.",
        },
        {
            "priority": 4,
            "experiment": "new_mechanism_only_after_ceiling_gap",
            "purpose": "Only design more controller levers if the floor/Pareto audit shows a large free pool remains.",
            "decision": "Avoid blind B1/B2-style tuning on the locked holdout.",
        },
    ]
    next_experiments = pd.DataFrame(experiment_rows)

    ceiling.to_csv(RAW / "ceiling_evidence_table.csv", index=False)
    target_gap.to_csv(RAW / "target_gap_table.csv", index=False)
    concentration.to_csv(RAW / "a1_case_concentration_table.csv", index=False)
    kind.to_csv(RAW / "roundtrip_kind_summary_copy.csv", index=False)
    b_summary_df.to_csv(RAW / "b1_b2_dev_result_table.csv", index=False)
    next_experiments.to_csv(RAW / "next_experiment_table.csv", index=False)

    paper_rows = [
        {
            "quantity": "A0 pump",
            "value": f"{fmt(a0_pump)} m3",
            "meaning": "Locked 53-case baseline.",
        },
        {
            "quantity": "A1 saving",
            "value": f"{fmt(a1_saved)} m3 ({pct(a1_saved, a0_pump)})",
            "meaning": "Current learned 0-60min relief-economy result.",
        },
        {
            "quantity": "20% target",
            "value": f"{fmt(0.20 * a0_pump)} m3",
            "meaning": "Required saving for the user's aspirational target.",
        },
        {
            "quantity": "Total reversible proxy",
            "value": f"{fmt(total_reversible)} m3 ({pct(total_reversible, a0_pump)})",
            "meaning": "Diagnostic upper pool before separating necessary vs avoidable reversals.",
        },
        {
            "quantity": "Safe relief-adjacent reversible proxy",
            "value": f"{fmt(addressable_reversible)} m3 ({pct(addressable_reversible, a0_pump)})",
            "meaning": "Closer estimate of free forecast-addressable round-trip pool.",
        },
        {
            "quantity": "A1 concentration",
            "value": f"top-1 {pct(float(step0['saving_top1_share']), 1.0, 1)}, top-5 {pct(float(step0['saving_top5_share']), 1.0, 1)}",
            "meaning": "Savings are real but narrow, not broad across the holdout.",
        },
    ]

    verdict = "NO-GO for 20% safety-neutral pump saving under the current architecture"
    if 0.20 * a0_pump <= total_reversible:
        verdict = "CONDITIONAL: 20% does not exceed total reversible proxy, but still needs optimizer proof"

    summary = f"""# Pump-Saving 20 Percent Ceiling Summary

## Verdict

**{verdict}.**

The current evidence supports the learned 0-60min relief-economy A1 result as a
real safety-neutral pump-saving layer, but it does **not** support a free 20%
reduction target. A 20% reduction would require `{fmt(0.20 * a0_pump)} m3`,
which is larger than the observed total reversible/chatter proxy
(`{fmt(total_reversible)} m3`). That total reversible proxy itself includes
motion that is likely physically necessary under true wind reversals, so it is
not a free-savings pool.

{markdown_table(paper_rows, ["quantity", "value", "meaning"])}

## What This Means

- A1 already saves `{fmt(a1_saved)} m3` (`{pct(a1_saved, a0_pump)}`) with no
  safety regression on the locked holdout.
- A1's saving is larger than the safe relief-adjacent reversible proxy
  (`{fmt(addressable_reversible)} m3`), because part of the gain comes from
  one-way over-pursuit and closed-loop cascade avoidance, not only local
  round-trip removal.
- The 20% target requires another
  `{fmt(max(0.0, 0.20 * a0_pump - a1_saved))} m3` beyond A1. The current B1/B2
  dev probes did not find that headroom: B1 had no extra cap actions and B2 was
  net worse due to catch-up.
- Therefore the honest path to 20% is not another hidden forecast trick. It is
  a pump-posture Pareto reframe: quantify how much additional pump can be saved
  if comfort/posture metrics are allowed to move while the v1.6 hard floor
  remains intact.

## Recommended Next Experiments

{markdown_table(next_experiments.to_dict("records"), ["priority", "experiment", "purpose", "decision"])}

## Paper Framing

Use A1 as the measured closed-loop pump-saving result and h120 as supervisory
advisory. If a larger number is required, present it as a Pareto frontier, not
as safety-neutral forecast saving.
"""

    decision = f"""# Pump-Saving 20 Percent Ceiling Decision

## Decision

**Do not keep tuning the current relief-economy architecture toward a claimed
20% safety-neutral pump reduction.**

The locked evidence says the current architecture has captured a clean
`{pct(a1_saved, a0_pump)}` saving. Pushing to 20% would require cutting into
pump that the diagnostics classify as demand-tracking, floor/recovery, or
non-addressable reversible motion.

## Evidence

- A0 total pump: `{fmt(a0_pump)} m3`.
- A1 pump: `{fmt(a1_pump)} m3`.
- A1 saving: `{fmt(a1_saved)} m3` (`{pct(a1_saved, a0_pump)}`).
- Required 20% saving: `{fmt(0.20 * a0_pump)} m3`.
- Total reversible proxy: `{fmt(total_reversible)} m3`
  (`{pct(total_reversible, a0_pump)}`).
- Safe relief-adjacent reversible proxy: `{fmt(addressable_reversible)} m3`
  (`{pct(addressable_reversible, a0_pump)}`).
- A1 round-trip proxy reduction: `{fmt(roundtrip_reduction)} m3`.
- A1 cascade saving after 600s: `{fmt(post600)} m3`.
- B1 adaptive cap dev delta vs A1: `{fmt(float(b_summary_df.loc[b_summary_df['experiment'].eq('B1_adaptive_axis_cap_dev'), 'delta_pump_m3'].iloc[0]) if not b_summary_df.empty and b_summary_df['experiment'].eq('B1_adaptive_axis_cap_dev').any() else 0.0)} m3`.
- B2 hold-extension dev delta vs A1: `{fmt(float(b_summary_df.loc[b_summary_df['experiment'].eq('B2_hold_extension_dev'), 'delta_pump_m3'].iloc[0]) if not b_summary_df.empty and b_summary_df['experiment'].eq('B2_hold_extension_dev').any() else 0.0)} m3`.

## What To Do Next

1. Freeze A1 as the current safety-neutral closed-loop saving result, pending
   negative-control wording if needed.
2. If the paper must show a larger pump reduction, run a single-knob comfort
   Pareto probe and explicitly report the posture cost.
3. If reviewers demand a hard ceiling, implement a true offline non-causal pump
   floor optimizer as a separate expensive validation, not as controller tuning.

## What Not To Do

- Do not tune B1/B2 on the locked 53-case holdout.
- Do not claim 20% as safety-neutral under the current evidence.
- Do not weaken the v1.6 hard floor or hide posture tradeoffs.
"""

    (PAPER / "pump_saving_20pct_ceiling_summary.md").write_text(summary)
    (OUT / "decision.md").write_text(decision)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
