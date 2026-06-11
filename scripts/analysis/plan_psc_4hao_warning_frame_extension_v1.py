#!/usr/bin/env python3
"""Extend PSC No.4 frame planning from pump saving to warning/veto roles.

This is a read-only evidence pass. It reuses the completed PSC No.4 80-case
frame plan and the older ULTIMATE_PUMP_SAVING_CONCLUSION registry to answer a
specific follow-up question: whether W/C regimes should improve the controller
as warning, release, or anti-chatter mechanisms instead of being judged only by
pump saving.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "outputs" / "wind_prediction" / "psc_selector_mixed_pool_v1"
FRAME_PLAN = BASE / "psc_4hao_frame_optimization_plan_20260603"
ULTIMATE = REPO / "ULTIMATE_PUMP_SAVING_CONCLUSION"
OUT = BASE / "psc_4hao_warning_frame_extension_20260603"
RUNS = BASE / "psc_4hao_broader_6h_test_only_main_v1" / "runs"

STRATUM_ROLE = {
    "neutral_mhs_broader": {
        "old_id": "P2/P3",
        "control_role": "main economy enable",
        "safety_role": "keep ordinary hard-floor safety layer",
        "next_action": "keep enabled; do not spend optimization budget here first",
    },
    "transient_peak_future_decay": {
        "old_id": "P1 + W2/W6 veto overlays",
        "control_role": "conditional economy enable",
        "safety_role": "release/veto when relief is not durable or future load worsens",
        "next_action": "split clean transient decay from re-intensification and far-intensification",
    },
    "direction_reversal_boundary": {
        "old_id": "W1",
        "control_role": "split low-posture saving vs high-pressure warning",
        "safety_role": "warn/release on high-pressure, high-posture, or catch-up-prone reversal",
        "next_action": "audit W1 branch variables before any full dispatcher run",
    },
    "reintensification_boundary": {
        "old_id": "W2",
        "control_role": "not a saving target",
        "safety_role": "veto apparent relief and close economy relaxation",
        "next_action": "make it an override, not an arm selected for saving",
    },
    "sustained_high_safety_event": {
        "old_id": "W3/W7",
        "control_role": "baseline/watch",
        "safety_role": "risk watch; optional h120 safety margin when f120 data is available",
        "next_action": "keep saving off; evaluate warning lead time separately",
    },
    "lowrisk_stable_redundant_candidate": {
        "old_id": "lowrisk redundant / C-support",
        "control_role": "small clean economy support",
        "safety_role": "no warning role; verify zero false activation",
        "next_action": "keep as low-priority clean support branch",
    },
    "quiet_low_opportunity": {
        "old_id": "no-action",
        "control_role": "baseline/no-op",
        "safety_role": "no warning role",
        "next_action": "keep closed; use as false-trigger control",
    },
}

CANDIDATE_ROWS = [
    {
        "id": "C3",
        "name": "Gusty / repeated-peak hold-current with runtime release",
        "role": "anti-chatter economy candidate",
        "evidence": "12-case runtime-release validation saves 19.78%; time>5 +33s; fallback 0; latch switches 1633 -> 1036",
        "current_80case_status": "not explicitly represented as a PSC No.4 stratum",
        "next_action": "add 2-3 gusty/repeated-peak smoke cases before full dispatcher validation",
    },
    {
        "id": "W2",
        "name": "Re-intensification after relief",
        "role": "warning/veto",
        "evidence": "old registry treats as veto/warning; current 80-case reintensification has 5.38% saving but +10s fallback",
        "current_80case_status": "present as reintensification_boundary",
        "next_action": "override economy saving when predicted relief is temporary",
    },
    {
        "id": "W6",
        "name": "Ramp / far-intensification watch",
        "role": "warning/veto",
        "evidence": "old balanced saving action was -25.20%; h120 watch had excessive posture cost in rejected point",
        "current_80case_status": "partly mixed into transient/boundary risk",
        "next_action": "use only as close-saving warning unless a new f120 safety-watch run is planned",
    },
    {
        "id": "W7",
        "name": "High-attention direction-stable event risk",
        "role": "risk watch",
        "evidence": "registry marks as risk watch / close saving, not saving target",
        "current_80case_status": "present as sustained_high_safety_event with -48.32% saving under generalized_gate",
        "next_action": "force baseline/watch in dispatcher",
    },
]


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        return f"{value:.2f}"
    return str(value)


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for rec in df[cols].to_dict("records"):
        lines.append("| " + " | ".join(_fmt(rec[col]) for col in cols) + " |")
    return "\n".join(lines)


def _load_frame() -> pd.DataFrame:
    path = FRAME_PLAN / "current_generalized_gate_by_frame.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    for col in [
        "saving_pct",
        "time_gt5_s",
        "time_gt7_s",
        "fallback_s",
        "d_time_gt5_s",
        "d_time_gt7_s",
        "d_fallback_s",
        "worst_p95_axis_deg",
        "worst_max_axis_deg",
    ]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _load_policy_cases() -> pd.DataFrame:
    path = FRAME_PLAN / "offline_policy_case_choices.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    for col in ["pump_m3", "baseline_pump_m3", "saved_m3", "time_gt5_s", "time_gt7_s", "fallback_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _role_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rec in frame.to_dict("records"):
        role = STRATUM_ROLE.get(str(rec["stratum"]), {})
        rows.append(
            {
                "pool": rec["pool"],
                "stratum": rec["stratum"],
                "old_id": role.get("old_id", rec.get("old_id", "")),
                "cases": int(rec["cases"]),
                "current_saving_pct": float(rec["saving_pct"]),
                "current_gt5_s": float(rec["time_gt5_s"]),
                "current_gt7_s": float(rec["time_gt7_s"]),
                "current_fallback_s": float(rec["fallback_s"]),
                "gt5_delta_s": float(rec["d_time_gt5_s"]),
                "gt7_delta_s": float(rec["d_time_gt7_s"]),
                "fallback_delta_s": float(rec["d_fallback_s"]),
                "control_role": role.get("control_role", ""),
                "safety_role": role.get("safety_role", ""),
                "next_action": role.get("next_action", ""),
            }
        )
    return pd.DataFrame(rows)


def _policy_by_stratum(policy_cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    policies = [
        "current_generalized_gate",
        "per_case_min_pump_no_fallback_regret",
        "per_case_min_pump_tail_guard",
    ]
    for (policy, stratum), group in policy_cases[policy_cases["policy"].isin(policies)].groupby(["policy", "stratum"]):
        baseline = float(group["baseline_pump_m3"].sum())
        pump = float(group["pump_m3"].sum())
        rows.append(
            {
                "policy": policy,
                "stratum": stratum,
                "cases": int(group["case_id"].nunique()),
                "pump_m3": pump,
                "saved_m3": baseline - pump,
                "saving_pct": 100.0 * (baseline - pump) / max(baseline, 1e-9),
                "time_gt5_s": float(group["time_gt5_s"].sum()),
                "time_gt7_s": float(group["time_gt7_s"].sum()),
                "fallback_s": float(group["fallback_s"].sum()),
                "chosen_generalized": int((group["chosen_arm"] == "generalized_gate").sum()),
                "chosen_refresh": int((group["chosen_arm"] == "refresh_on").sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["stratum", "policy"])


def _smoke_plan(policy_cases: pd.DataFrame) -> pd.DataFrame:
    current = policy_cases[policy_cases["policy"].eq("current_generalized_gate")].copy()
    current["risk_score"] = current["fallback_s"] * 10.0 + current["time_gt7_s"] * 3.0 + current["time_gt5_s"]
    current["saving_score"] = current["saved_m3"]
    specs = [
        ("neutral_mhs_broader", 2, "protect P2/P3 main saving"),
        ("transient_peak_future_decay", 3, "split clean P1 from W2/W6 warning cases"),
        ("direction_reversal_boundary", 3, "split W1 low-posture saving from high-pressure warning"),
        ("reintensification_boundary", 2, "test W2 veto override"),
        ("sustained_high_safety_event", 1, "confirm W3/W7 baseline/watch behavior"),
        ("lowrisk_stable_redundant_candidate", 1, "keep small clean support branch honest"),
    ]
    rows: list[dict[str, object]] = []
    for stratum, count, reason in specs:
        sub = current[current["stratum"].eq(stratum)].copy()
        if sub.empty:
            continue
        if stratum == "neutral_mhs_broader":
            sub = sub.sort_values("saving_score", ascending=False)
        else:
            sub = sub.sort_values(["risk_score", "saving_score"], ascending=False)
        for _, rec in sub.head(count).iterrows():
            rows.append(
                {
                    "case_id": rec["case_id"],
                    "pool": rec["pool"],
                    "stratum": rec["stratum"],
                    "reason": reason,
                    "current_saved_m3": float(rec["saved_m3"]),
                    "current_gt5_s": float(rec["time_gt5_s"]),
                    "current_gt7_s": float(rec["time_gt7_s"]),
                    "current_fallback_s": float(rec["fallback_s"]),
                    "suggested_dispatcher_check": STRATUM_ROLE.get(stratum, {}).get("next_action", ""),
                }
            )
    return pd.DataFrame(rows)


def _source_case_id(synthetic_case_id: str) -> str:
    parts = synthetic_case_id.split("_", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else synthetic_case_id


def _smoke_casebook(smoke: pd.DataFrame) -> pd.DataFrame:
    casebook_frames: list[pd.DataFrame] = []
    for path in (BASE / "psc_4hao_broader_6h_test_only_main_v1" / "casebooks").glob("*_pool_cases.csv"):
        casebook_frames.append(pd.read_csv(path))
    if not casebook_frames or smoke.empty:
        return pd.DataFrame()
    casebooks = pd.concat(casebook_frames, ignore_index=True).drop_duplicates("case_id", keep="first")
    plan = smoke.copy()
    plan["source_case_id"] = plan["case_id"].map(_source_case_id)
    merged = plan.merge(casebooks, left_on="source_case_id", right_on="case_id", how="left", suffixes=("_synthetic", ""))
    out = pd.DataFrame(
        {
            "case_id": merged["case_id"],
            "timestamp": merged["timestamp"],
            "label": (
                merged["label"].fillna("")
                + " | smoke_reason="
                + merged["reason"].fillna("")
                + " | synthetic_case_id="
                + merged["case_id_synthetic"].fillna("")
            ),
            "source_synthetic_case_id": merged["case_id_synthetic"],
            "stratum": merged["stratum"],
            "pool": merged["pool"],
            "current_saved_m3": merged["current_saved_m3"],
            "current_gt5_s": merged["current_gt5_s"],
            "current_gt7_s": merged["current_gt7_s"],
            "current_fallback_s": merged["current_fallback_s"],
        }
    )
    return out


def _case_id_from_log(path: Path) -> str:
    parts = path.name.split("_")
    if len(parts) >= 4 and parts[1] in {"positive", "negative", "background"} and parts[2] == "pool":
        return "_".join(parts[:4])
    return "_".join(parts[:3])


def _telemetry_audit(policy_cases: pd.DataFrame) -> pd.DataFrame:
    case_stratum = (
        policy_cases[policy_cases["policy"].eq("current_generalized_gate")][["case_id", "pool", "stratum"]]
        .drop_duplicates()
        .copy()
    )
    log_cols = {
        "forecast_advised_economy_boundary_veto": "boundary_veto_rows",
        "forecast_advised_economy_posture_release": "posture_release_rows",
        "forecast_advised_runaway_release": "runaway_release_rows",
        "relief_envelope_reintensification_veto_count": "relief_reintensification_veto_rows",
        "h120_scheduler_remote_risk_trigger_count": "h120_remote_risk_rows",
        "gusty_hold_current_gate_evaluated": "gusty_gate_evaluated_rows",
        "gusty_hold_current_runtime_released": "gusty_runtime_release_rows",
    }
    rows: list[dict[str, object]] = []
    for path in RUNS.glob("*/generalized_gate/planner_logs/*.csv"):
        rec: dict[str, object] = {"case_id": _case_id_from_log(path)}
        df = pd.read_csv(path, usecols=lambda c: c in set(log_cols), low_memory=False)
        for col, out_col in log_cols.items():
            rec[out_col] = float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()) if col in df else 0.0
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    logs = pd.DataFrame(rows)
    merged = case_stratum.merge(logs, on="case_id", how="left").fillna(0.0)
    agg_rows: list[dict[str, object]] = []
    for (pool, stratum), group in merged.groupby(["pool", "stratum"], dropna=False):
        rec: dict[str, object] = {
            "pool": pool,
            "stratum": stratum,
            "cases": int(group["case_id"].nunique()),
        }
        for out_col in log_cols.values():
            rec[out_col] = float(group[out_col].sum())
        agg_rows.append(rec)
    return pd.DataFrame(agg_rows).sort_values(["pool", "stratum"])


def _read_registry_metrics() -> pd.DataFrame:
    path = ULTIMATE / "regime_numbering_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    keep = [
        "ID",
        "name",
        "type",
        "all_window_pct",
        "pump_share_pct",
        "in_regime_saving_pct",
        "time_gt5_delta_s",
        "fallback_delta_s",
        "status",
    ]
    df = pd.read_csv(path)
    return df[[c for c in keep if c in df.columns]]


def _write_report(
    role: pd.DataFrame,
    policy_by_stratum: pd.DataFrame,
    telemetry: pd.DataFrame,
    registry: pd.DataFrame,
    candidates: pd.DataFrame,
    smoke: pd.DataFrame,
    smoke_casebook: pd.DataFrame,
) -> None:
    current = role.copy()
    total_cases = int(current["cases"].sum())
    total_gt5 = float(current["current_gt5_s"].sum())
    total_gt7 = float(current["current_gt7_s"].sum())
    total_fallback = float(current["current_fallback_s"].sum())
    positive_saving = current[current["current_saving_pct"] > 0]
    weighted_saving = float((positive_saving["current_saving_pct"] * positive_saving["cases"]).sum()) / max(
        float(positive_saving["cases"].sum()), 1.0
    )

    current_gate = policy_by_stratum[policy_by_stratum["policy"].eq("current_generalized_gate")]
    no_fb = policy_by_stratum[policy_by_stratum["policy"].eq("per_case_min_pump_no_fallback_regret")]
    total_current_saved = float(current_gate["saved_m3"].sum())
    total_no_fb_saved = float(no_fb["saved_m3"].sum())

    lines = [
        "# PSC No.4 Warning / Frame Extension Plan - 2026-06-03",
        "",
        "## Purpose",
        "",
        "This pass continues the interrupted June 2 work. It extends the frame plan from pure pump saving to a three-role dispatcher: economy enable, warning/veto, and candidate anti-chatter release.",
        "",
        "No long simulation is run here. The report only reads existing 80-case PSC No.4 results, the old P/W/C registry, and warning-watch evidence.",
        "",
        "## Immediate Reading",
        "",
        f"- Current 80-case generalized_gate still saves material pump, but its risk exposure is concentrated: {total_gt5:.0f}s above 5 deg, {total_gt7:.0f}s above 7 deg, and {total_fallback:.0f}s fallback over {total_cases} cases.",
        f"- Among currently positive-saving frames, the simple case-weighted saving rate is {weighted_saving:.2f}%, but that mixes clean P2 with warning-owned boundary frames.",
        f"- A no-fallback offline selector raises saved pump from {total_current_saved:.1f} m3 to {total_no_fb_saved:.1f} m3. This is the useful headroom: the next optimization should approximate it with explainable W/C rules, not chase a global threshold.",
        "",
        "## Current Frames Reinterpreted As Control Roles",
        "",
        _md_table(
            role.sort_values(["pool", "stratum"]),
            [
                "stratum",
                "old_id",
                "cases",
                "current_saving_pct",
                "current_gt5_s",
                "current_gt7_s",
                "current_fallback_s",
                "control_role",
                "safety_role",
            ],
        ),
        "",
        "## Offline Headroom By Frame",
        "",
        _md_table(
            policy_by_stratum,
            [
                "stratum",
                "policy",
                "saving_pct",
                "saved_m3",
                "time_gt5_s",
                "time_gt7_s",
                "fallback_s",
                "chosen_generalized",
                "chosen_refresh",
            ],
        ),
        "",
        "## Old P/W/C Evidence To Reuse",
        "",
        _md_table(
            registry.head(14),
            [
                "ID",
                "type",
                "all_window_pct",
                "pump_share_pct",
                "in_regime_saving_pct",
                "time_gt5_delta_s",
                "fallback_delta_s",
                "status",
            ],
        )
        if not registry.empty
        else "Registry metrics file not found.",
        "",
        "## Current Generalized-Gate W/C Telemetry Audit",
        "",
        _md_table(
            telemetry,
            [
                "stratum",
                "cases",
                "boundary_veto_rows",
                "posture_release_rows",
                "runaway_release_rows",
                "relief_reintensification_veto_rows",
                "h120_remote_risk_rows",
                "gusty_gate_evaluated_rows",
                "gusty_runtime_release_rows",
            ],
        )
        if not telemetry.empty
        else "No generalized_gate planner logs found for telemetry audit.",
        "",
        "Reading: boundary-veto telemetry is present in the current run, but h120 remote-risk, gusty runtime-release, and explicit posture/runaway release paths do not materially fire in this 80-case generalized_gate validation. The dispatcher can therefore reuse existing fields, but it still needs a focused smoke run for W/C activation behavior.",
        "",
        "## C/W Extensions",
        "",
        _md_table(candidates, ["id", "role", "current_80case_status", "evidence", "next_action"]),
        "",
        "## Recommended Smoke Set",
        "",
        _md_table(
            smoke,
            [
                "case_id",
                "stratum",
                "reason",
                "current_saved_m3",
                "current_gt5_s",
                "current_gt7_s",
                "current_fallback_s",
            ],
        ),
        "",
        "Runnable smoke casebook:",
        "",
        f"`{OUT / 'next_smoke_casebook.csv'}`",
        "",
        _md_table(
            smoke_casebook,
            ["case_id", "timestamp", "stratum", "current_saved_m3", "current_gt5_s", "current_gt7_s", "current_fallback_s"],
        )
        if not smoke_casebook.empty
        else "Smoke casebook could not be materialized from current casebook files.",
        "",
        "## Decision",
        "",
        "Proceed with a frame dispatcher, but define the output of each frame differently:",
        "",
        "1. P2/P3 stays the main economy-enable branch.",
        "2. P1 is conditional economy only when relief is durable; W2/W6 overlays should veto it when re-intensification or far-worsening is predicted.",
        "3. W1 must split into low-posture saving and high-pressure warning/release before the full 80-case run.",
        "4. W3/W7 are not economy branches; they are watch/close-saving branches.",
        "5. C3 should be added as a separate anti-chatter candidate because the old evidence shows it can reduce latch churn with fallback 0, but it is not yet represented in the current PSC No.4 80-case strata.",
        "",
        "This means the next improvement is not just higher pump saving. It is reducing false economy activation in W frames while preserving P2 and adding C3 only after a small smoke validation.",
    ]
    (OUT / "psc_4hao_warning_frame_extension_readout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = _load_frame()
    policy_cases = _load_policy_cases()
    role = _role_matrix(frame)
    policy_by_stratum = _policy_by_stratum(policy_cases)
    telemetry = _telemetry_audit(policy_cases)
    registry = _read_registry_metrics()
    candidates = pd.DataFrame(CANDIDATE_ROWS)
    smoke = _smoke_plan(policy_cases)
    smoke_casebook = _smoke_casebook(smoke)

    role.to_csv(OUT / "current_frame_warning_role_matrix.csv", index=False)
    policy_by_stratum.to_csv(OUT / "offline_policy_by_stratum.csv", index=False)
    telemetry.to_csv(OUT / "current_generalized_gate_wc_telemetry_audit.csv", index=False)
    registry.to_csv(OUT / "old_regime_registry_metrics.csv", index=False)
    candidates.to_csv(OUT / "cw_extension_candidate_matrix.csv", index=False)
    smoke.to_csv(OUT / "next_smoke_case_plan.csv", index=False)
    smoke_casebook.to_csv(OUT / "next_smoke_casebook.csv", index=False)
    _write_report(role, policy_by_stratum, telemetry, registry, candidates, smoke, smoke_casebook)

    print(OUT / "psc_4hao_warning_frame_extension_readout.md")


if __name__ == "__main__":
    main()
