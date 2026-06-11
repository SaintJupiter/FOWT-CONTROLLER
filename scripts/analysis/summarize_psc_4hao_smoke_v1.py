#!/usr/bin/env python3
"""Summarize 120-minute No.4 smoke replay candidates.

The smoke runs use a small casebook whose rows are renumbered by the casebook
runner.  This script maps them back to the original 96-case ids and compares
the candidate mild arms against the frozen 0hao current-forecast baseline.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = REPO_ROOT / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
FEASIBILITY_DIR = BASE / "psc_4hao_moderate_arm_feasibility_v1"
SMOKE_CASES = FEASIBILITY_DIR / "4hao_smoke_cases.csv"
FROZEN_METRICS = BASE / "degradation_ladder_96case_pair" / "degradation_ladder_case_metrics.csv"
OUTPUT_CSV = FEASIBILITY_DIR / "4hao_smoke_120min_summary.csv"
OUTPUT_MD = FEASIBILITY_DIR / "4hao_smoke_120min_readout.md"
PARAM_RESULT_CSV = FEASIBILITY_DIR / "4hao_smoke_120min_parameter_result_table.csv"

PROFILES = {
    "regime_auto": BASE / "psc_4hao_smoke_120min_regime_auto_learned_v1",
    "dual_specialist": BASE / "psc_4hao_smoke_120min_dual_specialist_learned_v1",
    "neutral_mhs_clean": BASE / "psc_4hao_smoke_120min_neutral_mhs_clean_learned_v1",
    "regime_auto_scale025": BASE / "psc_4hao_smoke_120min_regime_auto_learned_scale025_v1",
    "regime_auto_scale040": BASE / "psc_4hao_smoke_120min_regime_auto_learned_scale040_v1",
    "regime_auto_pirelease_probe": BASE / "psc_4hao_smoke_120min_regime_auto_pirelease_probe_v1",
    "regime_auto_engineered_probe": BASE / "psc_4hao_smoke_120min_regime_auto_engineered_probe_v1",
}

PROFILE_DISPLAY = {
    "regime_auto": "A 全局温和调度",
    "dual_specialist": "B 双专家调度",
    "neutral_mhs_clean": "C neutral专家",
    "regime_auto_scale025": "D 全局缩放25%(无效)",
    "regime_auto_scale040": "E 全局缩放40%(无效)",
    "regime_auto_pirelease_probe": "F hold释放复核",
    "regime_auto_engineered_probe": "G 泵侧对齐复核",
}

PROFILE_PARAMS = {
    "regime_auto": {
        "candidate_intent": "global learned regime-auto budget profile",
        "allocator_mode": "regime_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
    "dual_specialist": {
        "candidate_intent": "relief/decay plus neutral specialist dispatcher",
        "allocator_mode": "dual_specialist_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
    "neutral_mhs_clean": {
        "candidate_intent": "neutral moderate-high clean-start specialist",
        "allocator_mode": "neutral_mhs_clean_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
    "regime_auto_scale025": {
        "candidate_intent": "scale test, not valid final mild-action knob",
        "allocator_mode": "regime_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
    "regime_auto_scale040": {
        "candidate_intent": "scale test, not valid final mild-action knob",
        "allocator_mode": "regime_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
    "regime_auto_pirelease_probe": {
        "candidate_intent": "mechanism probe: release hold branch to PI",
        "allocator_mode": "regime_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
    "regime_auto_engineered_probe": {
        "candidate_intent": "mechanism probe: align primary pump profile with 0hao",
        "allocator_mode": "regime_auto",
        "economy_budget_m3": 100.0,
        "economy_forecast_smart": 1,
        "smart_posture_deg": 4.7,
        "smart_pressure_norm": 1.05,
        "relief_norm": 0.7,
        "relief_drop_norm": 0.2,
    },
}


def _axis_metrics(timeseries_path: Path) -> dict[str, float]:
    df = pd.read_csv(
        timeseries_path,
        usecols=[
            "pitch_deg",
            "roll_deg",
            "preview_primary_safety_fallback",
        ],
    )
    axis = df[["pitch_deg", "roll_deg"]].abs().max(axis=1)
    return {
        "time_gt5_s": float((axis > 5.0).sum()),
        "time_gt6_s": float((axis > 6.0).sum()),
        "max_axis_deg": float(axis.max()),
        "p95_axis_deg": float(axis.quantile(0.95)),
        "fallback_s": float(df["preview_primary_safety_fallback"].fillna(0).sum()),
    }


def _find_timeseries(run_dir: Path, run_case_id: str) -> Path:
    matches = sorted((run_dir / "timeseries").glob(f"{run_case_id}_*_timeseries.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one timeseries for {run_case_id}, found {len(matches)}")
    return matches[0]


def main() -> None:
    smoke_cases = pd.read_csv(SMOKE_CASES)
    frozen = pd.read_csv(FROZEN_METRICS)
    baseline = frozen[frozen["arm"] == "current_forecast_adaptive"].copy()
    baseline = baseline.set_index("case_id")
    rows: list[dict[str, float | str | int]] = []

    for profile, run_dir in PROFILES.items():
        if not (run_dir / "casebook_summary.csv").exists():
            continue
        summary = pd.read_csv(run_dir / "casebook_summary.csv")
        if len(summary) != len(smoke_cases):
            raise ValueError(f"{profile}: summary rows do not match smoke cases")
        for idx, run_row in summary.reset_index(drop=True).iterrows():
            original_case_id = str(smoke_cases.loc[idx, "case_id"])
            base_row = baseline.loc[original_case_id]
            ts_metrics = _axis_metrics(_find_timeseries(run_dir, str(run_row["case_id"])))
            primary_pump = float(run_row["primary_pump_work_m3"])
            base_pump = float(base_row["pump_m3"])
            d_pump = primary_pump - base_pump
            d_gt5 = ts_metrics["time_gt5_s"] - float(base_row["time_gt5_s"])
            d_gt6 = ts_metrics["time_gt6_s"] - float(base_row["time_gt6_s"])
            d_fallback = ts_metrics["fallback_s"] - float(base_row["fallback_s"])
            safety_fail = bool((d_fallback > 0) or (d_gt5 >= 60))
            strict_no_debt = bool((d_fallback == 0) and (d_gt5 <= 0))
            rows.append(
                {
                    "profile": profile,
                    "candidate_name": PROFILE_DISPLAY.get(profile, profile),
                    "case_id": original_case_id,
                    "run_case_id": run_row["case_id"],
                    "baseline_pump_m3": base_pump,
                    "primary_pump_m3": primary_pump,
                    "pump_saved_vs_0hao_m3": -d_pump,
                    "pump_saved_vs_0hao_pct": (-d_pump) / max(base_pump, 1e-9) * 100.0,
                    "baseline_time_gt5_s": float(base_row["time_gt5_s"]),
                    "primary_time_gt5_s": ts_metrics["time_gt5_s"],
                    "d_time_gt5_s": d_gt5,
                    "baseline_time_gt6_s": float(base_row["time_gt6_s"]),
                    "primary_time_gt6_s": ts_metrics["time_gt6_s"],
                    "d_time_gt6_s": d_gt6,
                    "baseline_fallback_s": float(base_row["fallback_s"]),
                    "primary_fallback_s": ts_metrics["fallback_s"],
                    "d_fallback_s": d_fallback,
                    "primary_p95_axis_deg": ts_metrics["p95_axis_deg"],
                    "primary_max_axis_deg": ts_metrics["max_axis_deg"],
                    "strict_no_debt": int(strict_no_debt),
                    "safety_fail": int(safety_fail),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    profile_summary = (
        out.groupby("profile")
        .agg(
            candidate_name=("candidate_name", "first"),
            cases=("case_id", "count"),
            baseline_pump_m3=("baseline_pump_m3", "sum"),
            primary_pump_m3=("primary_pump_m3", "sum"),
            pump_saved_vs_0hao_m3=("pump_saved_vs_0hao_m3", "sum"),
            d_time_gt5_s=("d_time_gt5_s", "sum"),
            d_time_gt6_s=("d_time_gt6_s", "sum"),
            d_fallback_s=("d_fallback_s", "sum"),
            strict_no_debt_cases=("strict_no_debt", "sum"),
            safety_fail_cases=("safety_fail", "sum"),
        )
        .reset_index()
    )
    profile_summary["pump_saved_vs_0hao_pct"] = (
        profile_summary["pump_saved_vs_0hao_m3"]
        / profile_summary["baseline_pump_m3"].clip(lower=1e-9)
        * 100.0
    )
    param_rows: list[dict[str, float | str | int]] = []
    for profile, run_dir in PROFILES.items():
        summary_path = run_dir / "casebook_summary.csv"
        if not summary_path.exists():
            continue
        first = pd.read_csv(summary_path, nrows=1).iloc[0].to_dict()
        result = profile_summary[profile_summary["profile"] == profile].iloc[0].to_dict()
        params = PROFILE_PARAMS.get(profile, {})
        verdict = "reject_global_4hao"
        if int(result["safety_fail_cases"]) == 0 and float(result["pump_saved_vs_0hao_m3"]) > 0:
            verdict = "candidate"
        elif str(profile).startswith("regime_auto_scale"):
            verdict = "invalid_knob"
        param_rows.append(
            {
                "profile": profile,
                "candidate_name": PROFILE_DISPLAY.get(profile, profile),
                "candidate_intent": params.get("candidate_intent", ""),
                "primary_control_profile": first.get("primary_control_profile", ""),
                "forecast_source_effective": first.get("forecast_source_effective", ""),
                "duration_min": 120,
                "primary_scale": first.get("primary_scale", ""),
                "safety_profile": first.get("primary_safety_profile", ""),
                "hold_target_mode": first.get("primary_hold_target_mode", ""),
                "planner_envelope_mode": first.get("planner_envelope_mode", ""),
                "planner_envelope_barrier": first.get("planner_envelope_barrier_active", ""),
                "economy_budget_enabled": 1,
                "economy_budget_m3": params.get("economy_budget_m3", ""),
                "allocator_mode": params.get("allocator_mode", ""),
                "economy_forecast_smart": params.get("economy_forecast_smart", ""),
                "smart_posture_deg": params.get("smart_posture_deg", ""),
                "smart_pressure_norm": params.get("smart_pressure_norm", ""),
                "relief_norm": params.get("relief_norm", ""),
                "relief_drop_norm": params.get("relief_drop_norm", ""),
                "relief_medium_cap_enabled": first.get("relief_medium_cap_enabled", ""),
                "relief_medium_cap_ratio": first.get("relief_medium_cap_ratio", ""),
                "relief_medium_cap_adaptive": first.get("relief_medium_cap_adaptive", ""),
                "baseline_pump_m3": result["baseline_pump_m3"],
                "saved_vs_0hao_m3": result["pump_saved_vs_0hao_m3"],
                "saved_vs_0hao_pct": result["pump_saved_vs_0hao_pct"],
                "d_time_gt5_s": result["d_time_gt5_s"],
                "d_time_gt6_s": result["d_time_gt6_s"],
                "d_fallback_s": result["d_fallback_s"],
                "strict_no_debt_cases": result["strict_no_debt_cases"],
                "safety_fail_cases": result["safety_fail_cases"],
                "verdict": verdict,
            }
        )
    param_result = pd.DataFrame(param_rows)
    param_result.to_csv(PARAM_RESULT_CSV, index=False)
    strict = out[(out["strict_no_debt"] == 1) & (out["pump_saved_vs_0hao_m3"] > 0)].copy()
    strict_summary = (
        strict.groupby("profile")
        .agg(
            strict_gain_cases=("case_id", "count"),
            strict_gain_m3=("pump_saved_vs_0hao_m3", "sum"),
            strict_case_ids=("case_id", lambda s: ";".join(s.astype(str))),
        )
        .reset_index()
    )
    pareto = out[
        (out["d_fallback_s"] == 0)
        & (out["d_time_gt5_s"] < 60)
        & (out["pump_saved_vs_0hao_m3"] > 0)
    ].copy()
    pareto_summary = (
        pareto.groupby("profile")
        .agg(
            pareto_cases=("case_id", "count"),
            pareto_gain_m3=("pump_saved_vs_0hao_m3", "sum"),
            pareto_d_gt5_s=("d_time_gt5_s", "sum"),
            pareto_case_ids=("case_id", lambda s: ";".join(s.astype(str))),
        )
        .reset_index()
    )

    lines = [
        "# 4hao 120min Smoke Replay Summary",
        "",
        "This is a real 120-minute replay comparison against the frozen 0hao current-forecast baseline for the 8-case No.4 smoke seed set.",
        "",
        "## What This Step Did",
        "",
        "- This step did not optimize 3hao.  It used 3hao only as the reference point that shows how much headroom remains.",
        "- The actual 4hao task here was to test whether any existing learned/budgeted economy profile can already serve as a moderate 4hao arm.",
        "- Result: none can be promoted directly.  A new 4hao action layer is needed.",
        "",
        "## Parameter Result Table",
        "",
        "| 候选名 | profile | intent | control profile | scale | allocator | budget m3 | forecast smart | posture/pressure gate | relief cap | 省水m3 | 省水% | d_gt5 s | d_fallback s | fail cases | verdict |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in param_result.to_dict("records"):
        relief_cap = (
            f"{int(float(row['relief_medium_cap_enabled']))}/"
            f"{float(row['relief_medium_cap_ratio']):.2f}/"
            f"{int(float(row['relief_medium_cap_adaptive']))}"
        )
        lines.append(
            f"| {row['candidate_name']} | {row['profile']} | {row['candidate_intent']} | {row['primary_control_profile']} | "
            f"{float(row['primary_scale']):.2f} | {row['allocator_mode']} | "
            f"{float(row['economy_budget_m3']):.0f} | {int(row['economy_forecast_smart'])} | "
            f"{float(row['smart_posture_deg']):.1f}deg/{float(row['smart_pressure_norm']):.2f} | "
            f"{relief_cap} | {float(row['saved_vs_0hao_m3']):.1f} | "
            f"{float(row['saved_vs_0hao_pct']):.2f}% | "
            f"{float(row['d_time_gt5_s']):.0f} | {float(row['d_fallback_s']):.0f} | "
            f"{int(row['safety_fail_cases'])} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "`relief cap` is `enabled/ratio/adaptive`.  All rows use learned forecast, 120min duration, raw planner envelope, envelope barrier on, pause hold-target mode, and default safety profile.",
            "",
            "## Profile Summary",
            "",
            "| 候选名 | profile | cases | 省水m3 | 省水% | d_gt5 s | d_gt6 s | d_fallback s | strict no-debt cases | safety-fail cases |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in profile_summary.to_dict("records"):
        lines.append(
            f"| {row['candidate_name']} | {row['profile']} | {int(row['cases'])} | {float(row['pump_saved_vs_0hao_m3']):.1f} | "
            f"{float(row['pump_saved_vs_0hao_pct']):.2f}% | "
            f"{float(row['d_time_gt5_s']):.0f} | {float(row['d_time_gt6_s']):.0f} | "
            f"{float(row['d_fallback_s']):.0f} | {int(row['strict_no_debt_cases'])} | "
            f"{int(row['safety_fail_cases'])} |"
        )

    lines.extend(
        [
            "",
            "## Strict Positive Subset",
            "",
            "| 候选名 | profile | strict gain cases | strict gain m3 | cases |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for row in strict_summary.to_dict("records"):
        lines.append(
            f"| {PROFILE_DISPLAY.get(row['profile'], row['profile'])} | {row['profile']} | {int(row['strict_gain_cases'])} | "
            f"{float(row['strict_gain_m3']):.1f} | {row['strict_case_ids']} |"
        )

    lines.extend(
        [
            "",
            "## No-Fallback Pareto Subset",
            "",
            "| 候选名 | profile | cases | gain m3 | d_gt5 s | cases |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in pareto_summary.to_dict("records"):
        lines.append(
            f"| {PROFILE_DISPLAY.get(row['profile'], row['profile'])} | {row['profile']} | {int(row['pareto_cases'])} | "
            f"{float(row['pareto_gain_m3']):.1f} | {float(row['pareto_d_gt5_s']):.0f} | "
            f"{row['pareto_case_ids']} |"
        )

    lines.extend(
        [
            "",
            "## Case-Level Table",
            "",
            "| 候选名 | profile | case | 省水m3 | 省水% | d_gt5 s | d_gt6 s | d_fallback s | strict | safety fail |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in out.sort_values(["profile", "case_id"]).to_dict("records"):
        lines.append(
            f"| {row['candidate_name']} | {row['profile']} | {row['case_id']} | "
            f"{float(row['pump_saved_vs_0hao_m3']):.1f} | {float(row['pump_saved_vs_0hao_pct']):.2f}% | "
            f"{float(row['d_time_gt5_s']):.0f} | "
            f"{float(row['d_time_gt6_s']):.0f} | {float(row['d_fallback_s']):.0f} | "
            f"{int(row['strict_no_debt'])} | {int(row['safety_fail'])} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- None of these existing mild profiles is safe as a global 4hao arm on the smoke set.",
            "- The strict no-debt signal is local and very small: only `09_dual_relief_09` is a strict pump-saving positive under these tests.",
            "- `10_dual_relief_10` and `94_dual_boundary_06` are no-fallback Pareto positives with small `time_gt5` debt, so they belong in an appendix or relaxed target, not in the strict main claim.",
            "- `11_dual_relief_11` and `57/58/59_dual_neutral_33/34/35` are useful falsifiers: they save pump but add fallback or large time_gt5 debt and must stay gated out or receive a much milder action.",
            "- `18_dual_relief_18` is also a falsifier for the current action strength: it has fallback `0` but adds about `581s` of `time_gt5`, so it must not be treated as safe yet.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT_CSV}")
    print(f"wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
