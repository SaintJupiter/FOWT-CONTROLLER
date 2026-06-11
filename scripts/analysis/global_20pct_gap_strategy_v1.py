#!/usr/bin/env python3
"""Quantify the remaining gap to a pump-weighted 20% global target.

This is a planning/analysis script. It does not run controllers and does not
change controller behavior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("outputs/wind_prediction/regime_conditioned_policy_development_v1")
SRC = ROOT / "paper_ready" / "pump_weighted_global_contribution_v1"
OUT = ROOT / "paper_ready" / "global_20pct_gap_strategy_v1"
OUT.mkdir(parents=True, exist_ok=True)


TARGET_GLOBAL = 20.0


def main() -> None:
    pump_share = pd.read_csv(SRC / "regime_pump_share_proxy.csv")
    contribution = pd.read_csv(SRC / "pump_weighted_global_contribution_table.csv")

    current_additive = float(
        contribution["pump_weighted_global_contribution_pct"].dropna().sum()
    )
    gap = TARGET_GLOBAL - current_additive

    share = {
        str(r["primary_regime"]): float(r["estimated_pump_share_pct"])
        for _, r in pump_share.iterrows()
    }

    levers = [
        {
            "lever": "deepen_broad_neutral_headroom",
            "pump_share_pct": share.get("neutral_untyped", 0.0),
            "current_saving_pct": 21.24,
            "realistic_next_saving_pct": 26.0,
            "aggressive_saving_pct": 30.0,
            "mechanism": "increase economy suppression inside already-active neutral/headroom windows",
            "risk": "comfort/safety-margin cost if posture rides too high",
            "priority": 1,
        },
        {
            "lever": "gusty_pump_batching_or_hysteresis",
            "pump_share_pct": share.get("gusty_oscillation_candidate", 0.0),
            "current_saving_pct": 0.0,
            "realistic_next_saving_pct": 10.0,
            "aggressive_saving_pct": 25.0,
            "mechanism": "batch or hysteresis small alternating target pursuit instead of chasing each gust",
            "risk": "previous smoothing failed; must avoid delayed catch-up",
            "priority": 2,
        },
        {
            "lever": "lowrisk_no_churn_idle_mode",
            "pump_share_pct": share.get("lowrisk_stable_redundant_candidate", 0.0),
            "current_saving_pct": 0.0,
            "realistic_next_saving_pct": 3.0,
            "aggressive_saving_pct": 8.0,
            "mechanism": "suppress tiny economy refresh/chatter when posture and forecast are quiet",
            "risk": "large by count but low pump per case; top-case concentration risk",
            "priority": 3,
        },
        {
            "lever": "plateau_operator_or_headroom_guarded_mode",
            "pump_share_pct": share.get("residual_high_plateau", 0.0),
            "current_saving_pct": 0.0,
            "realistic_next_saving_pct": 8.0,
            "aggressive_saving_pct": 20.0,
            "mechanism": "operator/pareto steady-high hold only with large posture headroom",
            "risk": "recognition overlaps boundary; automatic version previously failed",
            "priority": 4,
        },
    ]
    lever_df = pd.DataFrame(levers)
    lever_df["realistic_global_add_pct"] = (
        lever_df["pump_share_pct"] * lever_df["realistic_next_saving_pct"] / 100.0
    )
    lever_df["aggressive_global_add_pct"] = (
        lever_df["pump_share_pct"] * lever_df["aggressive_saving_pct"] / 100.0
    )
    lever_df.to_csv(OUT / "global_20pct_candidate_levers.csv", index=False)

    scenarios = [
        {
            "scenario": "current_validated",
            "neutral_saving_pct": 21.24,
            "gusty_saving_pct": 0.0,
            "lowrisk_saving_pct": 0.0,
            "plateau_saving_pct": 0.0,
        },
        {
            "scenario": "neutral_deepening_only_moderate",
            "neutral_saving_pct": 26.0,
            "gusty_saving_pct": 0.0,
            "lowrisk_saving_pct": 0.0,
            "plateau_saving_pct": 0.0,
        },
        {
            "scenario": "neutral_deepening_only_aggressive",
            "neutral_saving_pct": 30.0,
            "gusty_saving_pct": 0.0,
            "lowrisk_saving_pct": 0.0,
            "plateau_saving_pct": 0.0,
        },
        {
            "scenario": "neutral_30_plus_gusty_25",
            "neutral_saving_pct": 30.0,
            "gusty_saving_pct": 25.0,
            "lowrisk_saving_pct": 0.0,
            "plateau_saving_pct": 0.0,
        },
        {
            "scenario": "neutral_30_gusty_25_lowrisk_5",
            "neutral_saving_pct": 30.0,
            "gusty_saving_pct": 25.0,
            "lowrisk_saving_pct": 5.0,
            "plateau_saving_pct": 0.0,
        },
        {
            "scenario": "neutral_28_gusty_20_lowrisk_8_plateau_15",
            "neutral_saving_pct": 28.0,
            "gusty_saving_pct": 20.0,
            "lowrisk_saving_pct": 8.0,
            "plateau_saving_pct": 15.0,
        },
    ]
    scenario_df = pd.DataFrame(scenarios)
    scenario_df["relief_decay_global_pct"] = float(
        contribution[
            contribution["family"].eq("forecast_relief_decay_auto")
        ].iloc[0]["pump_weighted_global_contribution_pct"]
    )
    scenario_df["neutral_global_pct"] = (
        share.get("neutral_untyped", 0.0) * scenario_df["neutral_saving_pct"] / 100.0
    )
    scenario_df["gusty_global_pct"] = (
        share.get("gusty_oscillation_candidate", 0.0)
        * scenario_df["gusty_saving_pct"]
        / 100.0
    )
    scenario_df["lowrisk_global_pct"] = (
        share.get("lowrisk_stable_redundant_candidate", 0.0)
        * scenario_df["lowrisk_saving_pct"]
        / 100.0
    )
    scenario_df["plateau_global_pct"] = (
        share.get("residual_high_plateau", 0.0)
        * scenario_df["plateau_saving_pct"]
        / 100.0
    )
    scenario_df["total_pump_weighted_global_pct"] = scenario_df[
        [
            "relief_decay_global_pct",
            "neutral_global_pct",
            "gusty_global_pct",
            "lowrisk_global_pct",
            "plateau_global_pct",
        ]
    ].sum(axis=1)
    scenario_df["gap_to_20_pct"] = TARGET_GLOBAL - scenario_df[
        "total_pump_weighted_global_pct"
    ]
    scenario_df.to_csv(OUT / "global_20pct_scenario_table.csv", index=False)

    md = [
        "# Global 20% Gap Strategy",
        "",
        "## Current Position",
        "",
        f"- current pump-weighted estimate: **{current_additive:.2f}%**",
        f"- target: **{TARGET_GLOBAL:.1f}%**",
        f"- remaining gap: **{gap:.2f} percentage points**",
        "",
        "The gap cannot be closed by the rare relief-decay family. It must come from",
        "large pump-share families: broad neutral/headroom first, then gusty, lowrisk,",
        "and possibly plateau as an operator/Pareto mode.",
        "",
        "## Candidate Lever Ranking",
        "",
        "| priority | lever | pump share | realistic add | aggressive add | risk |",
        "| ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for _, r in lever_df.sort_values("priority").iterrows():
        md.append(
            f"| {int(r['priority'])} | {r['lever']} | "
            f"{float(r['pump_share_pct']):.2f}% | "
            f"{float(r['realistic_global_add_pct']):.2f}% | "
            f"{float(r['aggressive_global_add_pct']):.2f}% | "
            f"{r['risk']} |"
        )
    md.extend(
        [
            "",
            "## Scenario Arithmetic",
            "",
            "| scenario | neutral | gusty | lowrisk | plateau | total global | gap to 20 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, r in scenario_df.iterrows():
        md.append(
            f"| {r['scenario']} | {float(r['neutral_saving_pct']):.1f}% | "
            f"{float(r['gusty_saving_pct']):.1f}% | "
            f"{float(r['lowrisk_saving_pct']):.1f}% | "
            f"{float(r['plateau_saving_pct']):.1f}% | "
            f"{float(r['total_pump_weighted_global_pct']):.2f}% | "
            f"{float(r['gap_to_20_pct']):+.2f}% |"
        )
    md.extend(
        [
            "",
            "## Practical Decision",
            "",
            "To reach a pump-weighted 20% global target, one lever is not enough.",
            "The shortest credible route is:",
            "",
            "1. push broad neutral/headroom from 21.24% toward 28-30% with a disclosed",
            "   Pareto/comfort setting;",
            "2. add a new gusty batching/hysteresis action that captures part of the",
            "   10.57% pump-share gusty pool;",
            "3. add only a small lowrisk no-churn mode if it is nearly free;",
            "4. keep plateau as operator/Pareto unless a boundary-safe gate appears.",
            "",
            "This is no longer a pure forecast-free-saving story. It is a broad",
            "pump-saving controller with forecast/state gates and explicit safety-margin",
            "accounting.",
        ]
    )
    (OUT / "global_20pct_gap_strategy.md").write_text("\n".join(md), encoding="utf-8")
    print(OUT / "global_20pct_gap_strategy.md")


if __name__ == "__main__":
    main()
