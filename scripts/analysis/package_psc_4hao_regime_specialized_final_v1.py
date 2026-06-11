#!/usr/bin/env python3
"""Package the PSC No.4 regime-specialized optimization final evidence."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
V2 = BASE / "psc_4hao_regime_addback_rules_v2_20260603"
OUT = BASE / "psc_4hao_regime_specialized_optimization_20260603"


REGIME_STATUS = {
    "neutral_mhs_broader": {
        "condition_code": "P2/P3",
        "status": "accepted_improved",
        "decision": "Keep generalized_gate enabled as mature economy branch.",
    },
    "lowrisk_stable_redundant_candidate": {
        "condition_code": "lowrisk",
        "status": "accepted_monitor",
        "decision": "Keep enabled as low-mass support branch; do not use as headline driver.",
    },
    "direction_reversal_boundary": {
        "condition_code": "W1",
        "status": "accepted_conditional",
        "decision": "Keep low-pressure / not-strong-negative-drop addback; high-pressure negative-drop case remains fail-closed warning.",
    },
    "reintensification_boundary": {
        "condition_code": "W2",
        "status": "accepted_conditional",
        "decision": "Keep clean-direction or flat-reintensification addback; other W2 cases remain vetoed.",
    },
    "transient_peak_future_decay": {
        "condition_code": "P1",
        "status": "accepted_improved",
        "decision": "Promote v2 refined P1 rule using early_max, early_rise, and drop_mean; recovers the outcome target without fallback.",
    },
    "sustained_high_safety_event": {
        "condition_code": "W3/W7",
        "status": "safe_no_go_fail_closed",
        "decision": "Keep fail-closed; available outcome target adds only 64.9 m3, below materiality threshold.",
    },
    "quiet_low_opportunity": {
        "condition_code": "quiet",
        "status": "regression_fail_closed",
        "decision": "Keep baseline/no-op; use as false-trigger regression pool.",
    },
}


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(V2 / "regime_addback_v2_summary.csv")
    by_stratum = pd.read_csv(V2 / "regime_addback_v2_by_stratum.csv")
    relative = pd.read_csv(V2 / "regime_addback_v2_relative_gain.csv")
    return summary, by_stratum, relative


def _regime_table(by_stratum: pd.DataFrame) -> pd.DataFrame:
    v2 = by_stratum[by_stratum["policy_name"].eq("regime_addback_v2")].copy()
    current = by_stratum[by_stratum["policy_name"].eq("current_generalized_gate")][
        ["stratum", "saved_m3", "fallback_s", "time_gt7_s"]
    ].rename(
        columns={
            "saved_m3": "current_saved_m3",
            "fallback_s": "current_fallback_s",
            "time_gt7_s": "current_time_gt7_s",
        }
    )
    table = v2.merge(current, on="stratum", how="left")
    table["saved_delta_vs_current_m3"] = table["saved_m3"] - table["current_saved_m3"]
    table["fallback_reduction_vs_current_s"] = table["current_fallback_s"] - table["fallback_s"]
    table["time_gt7_reduction_vs_current_s"] = table["current_time_gt7_s"] - table["time_gt7_s"]
    for stratum, meta in REGIME_STATUS.items():
        for key, value in meta.items():
            table.loc[table["stratum"].eq(stratum), key] = value
    cols = [
        "condition_code",
        "stratum",
        "status",
        "decision",
        "gate_cases",
        "saved_m3",
        "saved_delta_vs_current_m3",
        "fallback_s",
        "fallback_reduction_vs_current_s",
        "time_gt7_s",
        "time_gt7_reduction_vs_current_s",
        "worst_p95_axis_deg",
    ]
    return table[cols].sort_values("condition_code")


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
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


def _write_run_commands() -> None:
    lines = [
        "# PSC No.4 Regime-Specialized Optimization Run Commands",
        "",
        "```bash",
        "PYTHONPYCACHEPREFIX=/private/tmp/fowt_pycache ./.venv312/bin/python -m py_compile scripts/analysis/optimize_psc_4hao_regime_addback_rules_v2.py",
        "PYTHONPYCACHEPREFIX=/private/tmp/fowt_pycache ./.venv312/bin/python scripts/analysis/optimize_psc_4hao_regime_addback_rules_v2.py",
        "PYTHONPYCACHEPREFIX=/private/tmp/fowt_pycache ./.venv312/bin/python -m py_compile scripts/analysis/package_psc_4hao_regime_specialized_final_v1.py",
        "PYTHONPYCACHEPREFIX=/private/tmp/fowt_pycache ./.venv312/bin/python scripts/analysis/package_psc_4hao_regime_specialized_final_v1.py",
        "```",
        "",
        "Runtime implementation remains a separate follow-up: map label-derived early_max/early_rise/drop_mean/max/drop/dir to provider telemetry and add an experimental profile without changing current production profiles.",
    ]
    (OUT / "run_commands.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(summary: pd.DataFrame, relative: pd.DataFrame, regimes: pd.DataFrame) -> None:
    lines = [
        "# PSC No.4 Regime-Specialized Optimization Final Package - 2026-06-03",
        "",
        "## Final Decision",
        "",
        "Stop this optimization pass with `regime_addback_v2` as the best offline rule candidate. It gives every necessary regime either a material accepted improvement, a monitored support role, or a safe fail-closed/no-go decision.",
        "",
        "This is an offline distillation result ready for runtime implementation planning. It is not yet a provider-level deployed controller profile.",
        "",
        "## Policy Metrics",
        "",
        _md_table(
            summary,
            [
                "policy",
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
        "## Relative Gains",
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
        "## Regime Outcomes",
        "",
        _md_table(
            regimes,
            [
                "condition_code",
                "stratum",
                "status",
                "gate_cases",
                "saved_m3",
                "fallback_s",
                "time_gt7_s",
                "decision",
            ],
        ),
        "",
        "## Stop Rationale",
        "",
        "- P1 got the main significant improvement: +433.7 m3 versus v1 and now matches the outcome-informed P1 target.",
        "- W1 and W2 already meet their safe conditional targets and preserve fallback=0.",
        "- P2/P3 remains the mature economy backbone.",
        "- W3/W7 has only 64.9 m3 outcome-target saving, below the 250 m3 materiality threshold, so it stays fail-closed.",
        "- Quiet-low has no pump mass and stays a false-trigger regression pool.",
    ]
    (OUT / "final_execution_package.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, by_stratum, relative = _load()
    regimes = _regime_table(by_stratum)
    regimes.to_csv(OUT / "accepted_rejected_regime_outcomes.csv", index=False)
    summary.to_csv(OUT / "final_policy_metrics.csv", index=False)
    relative.to_csv(OUT / "final_relative_gains.csv", index=False)
    _write_run_commands()
    _write_report(summary, relative, regimes)
    print(OUT / "final_execution_package.md")
    print(regimes[["condition_code", "stratum", "status", "saved_m3", "fallback_s", "time_gt7_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
