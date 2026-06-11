#!/usr/bin/env python3
"""Closed-loop reactive baseline freeze check.

This script compares three non-preview ballast baselines on five mechanism
windows:

* closed_raw: legacy plant pump defaults.
* closed_engineered_minimal: PI plus ordinary actuator constraints.
* closed_smooth_candidate: lightweight anti-chatter candidate for figures.

The purpose is to decide whether the smooth closed baseline is a fair,
non-predictive engineering baseline or whether it changes control performance
enough that a simpler engineered baseline should be frozen instead.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from defaults import DEFAULT_PUMP_CFG
from run_prediction_primary_casebook import TIMESTAMP_FMT, smooth_wind_trace
from run_validation import run_closed_loop_case
from wind_prediction.replay_dataset import Fino1ReplayDataset


DT = 1.0
DEFAULT_DURATION_S = 7200.0

CASES = [
    ("onset", "2024-11-27 19:40:00", "onset ramp"),
    ("decay", "2024-09-27 13:00:00", "decay ramp-down"),
    ("sustained_storm", "2024-09-05 18:10:00", "sustained high wind"),
    ("normal_low", "2021-12-20 14:30:00", "normal low-risk wind"),
    ("signflip", "2023-10-03 06:30:00", "high-pressure direction change"),
]


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("stiffness/matrix xlsx not found")


ENGINEERED_MINIMAL_PUMP_CFG = dict(DEFAULT_PUMP_CFG)

SMOOTH_CANDIDATE_PUMP_CFG = {
    **ENGINEERED_MINIMAL_PUMP_CFG,
    "pump_min_on_s": 30.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 20.0,
    "pump_stage_hysteresis_kg": 75.0,
    "pump_stage_min_dwell_s": 10.0,
}

SMOOTH_CANDIDATE_TARGET_SHAPE = {}

VARIANTS = {
    "closed_raw": {
        "pump_cfg": None,
        "target_shape_override_cfg": {},
        "role": "legacy raw reference",
    },
    "closed_engineered_minimal": {
        "pump_cfg": ENGINEERED_MINIMAL_PUMP_CFG,
        "target_shape_override_cfg": {},
        "role": "ordinary feedback controller with actuator constraints",
    },
    "closed_smooth_candidate": {
        "pump_cfg": SMOOTH_CANDIDATE_PUMP_CFG,
        "target_shape_override_cfg": SMOOTH_CANDIDATE_TARGET_SHAPE,
        "role": "lightweight anti-chatter candidate for figure baseline",
    },
}


PARAM_SOURCE_NOTES = {
    "pump_stop_err_kg": "implementation tolerance / pump latch deadband",
    "pump_restart_err_kg": "hysteretic restart threshold to avoid rapid on/off cycling",
    "pump_min_on_s": "minimum on dwell; standard actuator anti-chatter practice",
    "pump_min_off_s": "minimum off dwell; standard actuator anti-chatter practice",
    "pump_hold_before_stop_s": "near-target confirmation dwell",
    "pump_global_quiet_hold_s": "global quiet confirmation dwell",
    "pump_global_quiet_backlog_kg": "global near-target backlog threshold",
    "pump_target_quiet_rate_kg_s": "target-motion gate for stopping pumps",
    "pump_ramp_up_m3_min_per_s": "finite pump rate ramp",
    "pump_ramp_down_m3_min_per_s": "finite pump rate ramp",
    "pump_stage_hysteresis_kg": "stage-selection hysteresis",
    "pump_stage_min_dwell_s": "stage-selection dwell",
    "pump_low_end_stage_hysteresis_kg": "low-end stage-selection hysteresis",
    "pump_low_end_stage_min_dwell_s": "low-end stage-selection dwell",
    "deadband_target_release_enabled": "target release inside attitude deadband",
}


def pct_abs(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.abs(np.asarray(values, dtype=float)), q))


def work_m3(pump: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(pump, dtype=float)), dx=DT) / 60.0)


def summarize(df: pd.DataFrame) -> dict[str, float]:
    pump = df["pump_total_rate_m3_min"].to_numpy(dtype=float)
    pitch = df["pitch_deg"].to_numpy(dtype=float)
    roll = df["roll_deg"].to_numpy(dtype=float)
    return {
        "pump_work_m3": work_m3(pump),
        "pump_duty": float(np.mean(np.abs(pump) > 1e-6)),
        "pitch_abs_p95": pct_abs(pitch, 95),
        "roll_abs_p95": pct_abs(roll, 95),
        "pitch_abs_max": pct_abs(pitch, 100),
        "roll_abs_max": pct_abs(roll, 100),
        "pump_latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
        "pump_stage_switches": int(df["pump_stage_switch_count"].iloc[-1])
        if "pump_stage_switch_count" in df.columns
        else 0,
        "pump_quiet_stop_blocks": int(df["pump_quiet_stop_block_count"].iloc[-1])
        if "pump_quiet_stop_block_count" in df.columns
        else 0,
    }


def _finite_str(value: object) -> str:
    if value is None:
        return "None"
    try:
        v = float(value)
    except Exception:
        return str(value)
    if np.isinf(v):
        return "inf"
    if np.isnan(v):
        return "nan"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.6g}"


def build_parameter_audit(out_dir: Path) -> pd.DataFrame:
    keys = sorted(
        set().union(
            *(set(v.get("pump_cfg") or {}) for v in VARIANTS.values()),
            *(set(v.get("target_shape_override_cfg") or {}) for v in VARIANTS.values()),
        )
    )
    rows = []
    for key in keys:
        row = {
            "parameter": key,
            "source_note": PARAM_SOURCE_NOTES.get(key, "configuration difference"),
        }
        for variant, cfg in VARIANTS.items():
            value = None
            if cfg.get("pump_cfg") and key in cfg["pump_cfg"]:
                value = cfg["pump_cfg"][key]
            if cfg.get("target_shape_override_cfg") and key in cfg["target_shape_override_cfg"]:
                value = cfg["target_shape_override_cfg"][key]
            row[variant] = _finite_str(value)
        rows.append(row)
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "closed_baseline_parameter_audit.csv", index=False)
    return audit


def compare_variants(summary: pd.DataFrame, reference: str, candidate: str) -> pd.DataFrame:
    rows = []
    for case_id, group in summary.groupby("case_id", sort=False):
        ref_row = group[group["variant"].eq(reference)]
        cand_row = group[group["variant"].eq(candidate)]
        if ref_row.empty or cand_row.empty:
            continue
        ref = ref_row.iloc[0]
        cand = cand_row.iloc[0]
        pump_delta_pct = (
            (cand["pump_work_m3"] - ref["pump_work_m3"]) / max(float(ref["pump_work_m3"]), 1e-9) * 100.0
        )
        latch_delta_pct = (
            (cand["pump_latch_switches"] - ref["pump_latch_switches"])
            / max(float(ref["pump_latch_switches"]), 1e-9)
            * 100.0
        )
        rows.append(
            {
                "case_id": case_id,
                "reference": reference,
                "candidate": candidate,
                "d_pump_work_pct": pump_delta_pct,
                "d_pitch_p95_deg": float(cand["pitch_abs_p95"] - ref["pitch_abs_p95"]),
                "d_roll_p95_deg": float(cand["roll_abs_p95"] - ref["roll_abs_p95"]),
                "d_pitch_max_deg": float(cand["pitch_abs_max"] - ref["pitch_abs_max"]),
                "d_roll_max_deg": float(cand["roll_abs_max"] - ref["roll_abs_max"]),
                "d_latch_switch_pct": latch_delta_pct,
                "ref_latch_switches": int(ref["pump_latch_switches"]),
                "cand_latch_switches": int(cand["pump_latch_switches"]),
            }
        )
    return pd.DataFrame(rows)


def decision_from_delta(delta: pd.DataFrame) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if delta.empty:
        return "FAIL", ["missing comparison rows"]
    max_abs_pump = float(np.max(np.abs(delta["d_pump_work_pct"])))
    max_abs_pitch_p95 = float(np.max(np.abs(delta["d_pitch_p95_deg"])))
    max_abs_roll_p95 = float(np.max(np.abs(delta["d_roll_p95_deg"])))
    max_abs_pitch_max = float(np.max(np.abs(delta["d_pitch_max_deg"])))
    max_abs_roll_max = float(np.max(np.abs(delta["d_roll_max_deg"])))
    latch_reduction_cases = int(np.sum(delta["d_latch_switch_pct"] <= -30.0))
    if (
        max_abs_pump <= 1.0
        and max(max_abs_pitch_p95, max_abs_roll_p95) <= 0.10
        and max(max_abs_pitch_max, max_abs_roll_max) <= 0.30
        and latch_reduction_cases >= max(3, int(np.ceil(0.6 * len(delta))))
    ):
        return "STRONG_PASS", reasons
    if (
        max_abs_pump <= 2.0
        and max(max_abs_pitch_p95, max_abs_roll_p95) <= 0.15
        and max(max_abs_pitch_max, max_abs_roll_max) <= 0.50
        and latch_reduction_cases >= max(3, int(np.ceil(0.6 * len(delta))))
    ):
        return "ACCEPTABLE_PASS", reasons
    if max_abs_pump > 2.0:
        reasons.append(f"pump work changed by up to {max_abs_pump:.2f}%")
    if max(max_abs_pitch_p95, max_abs_roll_p95) > 0.15:
        reasons.append(
            f"p95 attitude changed by up to {max(max_abs_pitch_p95, max_abs_roll_p95):.3f} deg"
        )
    if max(max_abs_pitch_max, max_abs_roll_max) > 0.50:
        reasons.append(
            f"max attitude changed by up to {max(max_abs_pitch_max, max_abs_roll_max):.3f} deg"
        )
    if latch_reduction_cases < max(3, int(np.ceil(0.6 * len(delta)))):
        reasons.append(
            f"latch switches reduced >=30% in only {latch_reduction_cases}/{len(delta)} cases"
        )
    return "FAIL", reasons


def write_report(
    out_dir: Path,
    elapsed_s: float,
    audit: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: dict[str, pd.DataFrame],
    decision: str,
    reasons: list[str],
) -> None:
    lines = [
        "# Closed Baseline Freeze Check",
        "",
        f"- elapsed: `{elapsed_s:.1f}s`",
        "- duration per case: `120 min`",
        "- cases: onset, decay, sustained_storm, normal_low, signflip",
        "- compared variants: `closed_raw`, `closed_engineered_minimal`, `closed_smooth_candidate`",
        "- public-control context: research FOWT controllers commonly compare against transparent PI/PID/reference controllers rather than proprietary commercial controllers.",
        "- literature anchors: ROSCO open-source reference controller; active ballast PID with pump-flow constraints; FOWT MPC/preview studies versus PI/gain-scheduled feedback baselines.",
        "",
        "## Parameter Audit",
        "",
        "| parameter | raw | engineered_minimal | smooth_candidate | note |",
        "|---|---:|---:|---:|---|",
    ]
    for _, row in audit.iterrows():
        lines.append(
            f"| {row['parameter']} | {row['closed_raw']} | {row['closed_engineered_minimal']} | "
            f"{row['closed_smooth_candidate']} | {row['source_note']} |"
        )
    lines += [
        "",
        "## Summary Metrics",
        "",
        "| case | variant | pump m3 | pitch p95 | roll p95 | pitch max | roll max | latch switches |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['case_id']} | {row['variant']} | {row['pump_work_m3']:.2f} | "
            f"{row['pitch_abs_p95']:.3f} | {row['roll_abs_p95']:.3f} | "
            f"{row['pitch_abs_max']:.3f} | {row['roll_abs_max']:.3f} | "
            f"{int(row['pump_latch_switches'])} |"
        )
    for name, delta in deltas.items():
        lines += [
            "",
            f"## Delta: {name}",
            "",
            "| case | d_pump | d_pitch_p95 | d_roll_p95 | d_pitch_max | d_roll_max | d_latch |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in delta.iterrows():
            lines.append(
                f"| {row['case_id']} | {row['d_pump_work_pct']:+.2f}% | "
                f"{row['d_pitch_p95_deg']:+.3f} | {row['d_roll_p95_deg']:+.3f} | "
                f"{row['d_pitch_max_deg']:+.3f} | {row['d_roll_max_deg']:+.3f} | "
                f"{row['d_latch_switch_pct']:+.1f}% |"
            )
    recommended = (
        "closed_smooth_candidate" if decision in ("STRONG_PASS", "ACCEPTABLE_PASS") else "closed_engineered_minimal"
    )
    lines += [
        "",
        "## Decision",
        "",
        f"- smooth candidate vs engineered minimal: `{decision}`",
        f"- recommended frozen baseline: `{recommended}`",
    ]
    if reasons:
        lines += [f"- reason: {reason}" for reason in reasons]
    else:
        lines.append("- reason: smooth candidate preserves pump/attitude metrics while reducing chatter.")
    lines += [
        "",
        "Interpretation rule:",
        "",
        "- If `STRONG_PASS` or `ACCEPTABLE_PASS`, freeze `closed_smooth_candidate` as the non-preview reactive baseline.",
        "- If `FAIL`, use `closed_engineered_minimal` as the fair non-preview baseline or revise smooth parameters and rerun this check.",
        "",
        "Frozen baseline note:",
        "",
        "- The recommended baseline is non-preview PI feedback with pump stop/restart hysteresis, minimum on/off dwell, near-target stop confirmation, and finite pump-rate ramp.",
        "- It does not use future wind, pressure-block future information, planner actions, bucket guards, or hold-current prediction targets.",
    ]
    (out_dir / "closed_baseline_freeze_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    out_dir = repo_root / "outputs" / "wind_prediction" / "closed_baseline_freeze_v2"
    ts_dir = out_dir / "timeseries"
    for d in (out_dir, ts_dir):
        d.mkdir(parents=True, exist_ok=True)

    audit = build_parameter_audit(out_dir)
    excel_path = discover_excel()
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    n_steps = int(round(DEFAULT_DURATION_S / DT))
    row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))

    rows = []
    for case_id, ts_str, label in CASES:
        ts = pd.Timestamp(ts_str).to_pydatetime()
        wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
        wind_trace = smooth_wind_trace(wind_trace, transition_s=0.0)
        for variant, cfg in VARIANTS.items():
            run_name = f"{case_id}_{ts_str.replace(':', '').replace(' ', '_')}_{variant}"
            print(f"--- {run_name} ---", flush=True)
            _, timeseries = run_closed_loop_case(
                excel_path=excel_path,
                case_name=run_name,
                dt=DT,
                n_steps=n_steps,
                wind_trace=wind_trace,
                control_enabled=True,
                record_timeseries=True,
                experiment_protocol="main",
                platform_profile="default",
                start_from_heave_equilibrium=True,
                pump_cfg=cfg["pump_cfg"],
                target_shape_override_cfg=cfg["target_shape_override_cfg"],
            )
            df = pd.DataFrame(timeseries)
            df.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
            row = {
                "case_id": case_id,
                "timestamp": ts_str,
                "label": label,
                "variant": variant,
                "role": cfg["role"],
            }
            row.update(summarize(df))
            rows.append(row)
            print(
                f"   pump={row['pump_work_m3']:.2f} m3, "
                f"pitch95={row['pitch_abs_p95']:.3f}, roll95={row['roll_abs_p95']:.3f}, "
                f"latch={row['pump_latch_switches']}",
                flush=True,
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "closed_baseline_freeze_summary.csv", index=False)
    deltas = {
        "engineered_minimal_vs_raw": compare_variants(
            summary, "closed_raw", "closed_engineered_minimal"
        ),
        "smooth_candidate_vs_engineered_minimal": compare_variants(
            summary, "closed_engineered_minimal", "closed_smooth_candidate"
        ),
    }
    for name, delta in deltas.items():
        delta.to_csv(out_dir / f"{name}.csv", index=False)
    decision, reasons = decision_from_delta(deltas["smooth_candidate_vs_engineered_minimal"])
    freeze_meta = {
        "decision": decision,
        "reasons": reasons,
        "recommended_baseline": (
            "closed_smooth_candidate" if decision in ("STRONG_PASS", "ACCEPTABLE_PASS") else "closed_engineered_minimal"
        ),
        "cases": CASES,
        "variants": {
            name: {
                "role": cfg["role"],
                "pump_cfg": cfg["pump_cfg"],
                "target_shape_override_cfg": cfg["target_shape_override_cfg"],
            }
            for name, cfg in VARIANTS.items()
        },
    }
    (out_dir / "closed_baseline_freeze_meta.json").write_text(
        json.dumps(freeze_meta, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    write_report(
        out_dir=out_dir,
        elapsed_s=time.perf_counter() - t0,
        audit=audit,
        summary=summary,
        deltas=deltas,
        decision=decision,
        reasons=reasons,
    )
    print(f"Report: {out_dir / 'closed_baseline_freeze_report.md'}")
    print(f"DECISION {decision}; recommended={freeze_meta['recommended_baseline']}")


if __name__ == "__main__":
    main()
