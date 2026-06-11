#!/usr/bin/env python3
"""Strict actuator-only no-preview economy baseline check.

The scan deliberately does not touch PI gains, wind prediction, planner logic,
or safety guards.  It is only meant to answer whether ordinary actuator-layer
retuning can materially reduce pump work while keeping the same attitude level
as the frozen closed_baseline_v1.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "archive" / "legacy_fowt_control"))

from defaults import DEFAULT_PUMP_CFG
from run_prediction_primary_casebook import TIMESTAMP_FMT, _load_indexed_cases, discover_excel, smooth_wind_trace
from run_validation import run_closed_loop_case
from wind_prediction.replay_dataset import Fino1ReplayDataset


DT = 1.0
DEFAULT_DURATION_S = 7200.0
OUT_DIR = REPO_ROOT / "outputs" / "wind_prediction" / "closed_economy_pareto_v1"
DATASET_DIR = REPO_ROOT / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1"
CAL_CASE_IDS = {"01", "03", "04", "06", "08"}
VAL_CASE_IDS = {f"{i:02d}" for i in range(1, 11)}

BASELINE_CFG = {
    "pump_stop_err_kg": 300.0,
    "pump_restart_err_kg": 500.0,
    "pump_min_on_s": 20.0,
    "pump_min_off_s": 12.0,
    "pump_hold_before_stop_s": 10.0,
    "pump_global_quiet_hold_s": 0.0,
    "pump_global_quiet_backlog_kg": float("inf"),
    "pump_target_quiet_rate_kg_s": float("inf"),
    "pump_ramp_up_m3_min_per_s": 2.0,
    "pump_ramp_down_m3_min_per_s": 3.0,
}

# Small, actuator-only parameter space.  The wider/harder candidates are
# intentionally modest because this baseline must remain a normal reactive
# controller, not a deliberately lazy controller.
CANDIDATES = [
    ("baseline", {}),
    ("restart_575", {"pump_restart_err_kg": 575.0}),
    ("restart_650", {"pump_restart_err_kg": 650.0}),
    ("hold_15", {"pump_hold_before_stop_s": 15.0}),
    ("hold_20", {"pump_hold_before_stop_s": 20.0}),
    ("min_on_25", {"pump_min_on_s": 25.0}),
    ("min_on_30", {"pump_min_on_s": 30.0}),
    ("restart_575_hold_15", {"pump_restart_err_kg": 575.0, "pump_hold_before_stop_s": 15.0}),
    ("restart_650_hold_15", {"pump_restart_err_kg": 650.0, "pump_hold_before_stop_s": 15.0}),
    ("restart_575_min_on_25", {"pump_restart_err_kg": 575.0, "pump_min_on_s": 25.0}),
]


def _pump_cfg(overrides: dict) -> dict:
    cfg = dict(DEFAULT_PUMP_CFG)
    cfg.update(BASELINE_CFG)
    cfg.update(overrides)
    return cfg


def _safe_cfg_for_json(cfg: dict) -> dict:
    out = {}
    for key, value in cfg.items():
        try:
            v = float(value)
        except Exception:
            out[key] = value
            continue
        out[key] = "Infinity" if np.isinf(v) else v
    return out


def work_m3(pump: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(np.asarray(pump, dtype=float)), dx=DT) / 60.0)


def pct_abs(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.abs(np.asarray(values, dtype=float)), q))


def summarize_df(df: pd.DataFrame) -> dict:
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
        "latch_switches": int(df["pump_latch_switch_count"].iloc[-1])
        if "pump_latch_switch_count" in df.columns
        else 0,
    }


def _selected_cases(case_ids: set[str]) -> list[tuple[int, str, str, str]]:
    cases = _load_indexed_cases("")
    return [item for item in cases if f"{item[0]:02d}" in case_ids]


def run_cases(
    cases: list[tuple[int, str, str, str]],
    candidates: list[tuple[str, dict]],
    out_dir: Path,
    duration_s: float,
    write_timeseries: bool = False,
) -> pd.DataFrame:
    n_steps = int(round(duration_s / DT))
    excel_path = discover_excel()
    replay = Fino1ReplayDataset(dataset_dir=DATASET_DIR, split="test")
    ts_dir = out_dir / "timeseries"
    if write_timeseries:
        ts_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_num, case_id, ts_str, label in cases:
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        row_count = int(np.ceil(n_steps * DT / replay.update_interval_s))
        wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=row_count, dt_s=DT)
        wind_trace = smooth_wind_trace(wind_trace, 0.0)
        for candidate_name, overrides in candidates:
            run_name = f"{case_num:02d}_{case_id}_{ts_str.replace(':', '').replace(' ', '_')}_{candidate_name}"
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
                preview_trim_provider=None,
                pump_cfg=_pump_cfg(overrides),
                target_shape_override_cfg={},
            )
            df = pd.DataFrame(timeseries)
            if write_timeseries:
                df.to_csv(ts_dir / f"{run_name}_timeseries.csv", index=False)
            summary = summarize_df(df)
            rows.append(
                {
                    "case_num": int(case_num),
                    "case_id": case_id,
                    "timestamp": ts_str,
                    "label": label,
                    "candidate": candidate_name,
                    "overrides_json": json.dumps(_safe_cfg_for_json(overrides), sort_keys=True),
                    **summary,
                }
            )
            print(
                f"{run_name}: pump={summary['pump_work_m3']:.1f}, "
                f"pitch_p95={summary['pitch_abs_p95']:.3f}, roll_p95={summary['roll_abs_p95']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


def compare_to_baseline(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for case_num, group in summary.groupby("case_num", sort=False):
        base = group[group["candidate"].eq("baseline")]
        if base.empty:
            continue
        b = base.iloc[0]
        for _, r in group.iterrows():
            rows.append(
                {
                    "case_num": int(case_num),
                    "case_id": r["case_id"],
                    "candidate": r["candidate"],
                    "d_pump_work_pct": (r["pump_work_m3"] - b["pump_work_m3"])
                    / max(float(b["pump_work_m3"]), 1e-9)
                    * 100.0,
                    "pump_saving_pct": (b["pump_work_m3"] - r["pump_work_m3"])
                    / max(float(b["pump_work_m3"]), 1e-9)
                    * 100.0,
                    "d_pitch_p95": float(r["pitch_abs_p95"] - b["pitch_abs_p95"]),
                    "d_roll_p95": float(r["roll_abs_p95"] - b["roll_abs_p95"]),
                    "d_pitch_max": float(r["pitch_abs_max"] - b["pitch_abs_max"]),
                    "d_roll_max": float(r["roll_abs_max"] - b["roll_abs_max"]),
                    "d_latch_switches": int(r["latch_switches"] - b["latch_switches"]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_deltas(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in delta.groupby("candidate", sort=False):
        rows.append(
            {
                "candidate": candidate,
                "mean_pump_saving_pct": float(group["pump_saving_pct"].mean()),
                "median_pump_saving_pct": float(group["pump_saving_pct"].median()),
                "max_pitch_p95_worsen_deg": float(group["d_pitch_p95"].max()),
                "max_roll_p95_worsen_deg": float(group["d_roll_p95"].max()),
                "max_pitch_max_worsen_deg": float(group["d_pitch_max"].max()),
                "max_roll_max_worsen_deg": float(group["d_roll_max"].max()),
                "mean_latch_delta": float(group["d_latch_switches"].mean()),
                "valid_strict": int(
                    group["d_pitch_p95"].max() <= 0.10
                    and group["d_roll_p95"].max() <= 0.08
                    and group["d_pitch_max"].max() <= 0.0
                    and group["d_roll_max"].max() <= 0.0
                ),
                "valid_relaxed": int(
                    group["d_pitch_p95"].max() <= 0.15
                    and group["d_roll_p95"].max() <= 0.12
                    and group["d_pitch_max"].max() <= 0.10
                    and group["d_roll_max"].max() <= 0.10
                ),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["valid_strict", "mean_pump_saving_pct", "valid_relaxed"],
        ascending=[False, False, False],
    )


def choose_candidate(agg: pd.DataFrame) -> str:
    strict = agg[(agg["candidate"] != "baseline") & (agg["valid_strict"] == 1)]
    if not strict.empty and float(strict["mean_pump_saving_pct"].max()) > 0.5:
        return str(strict.iloc[0]["candidate"])
    return "baseline"


def write_report(
    out_dir: Path,
    cal_summary: pd.DataFrame,
    cal_delta: pd.DataFrame,
    cal_agg: pd.DataFrame,
    selected: str,
    val_summary: pd.DataFrame,
    val_delta: pd.DataFrame,
    val_agg: pd.DataFrame,
    elapsed_s: float,
) -> None:
    selected_row = val_agg[val_agg["candidate"].eq(selected)]
    selected_val = selected_row.iloc[0] if not selected_row.empty else None
    lines = [
        "# Closed Economy Pareto v1",
        "",
        f"- elapsed: `{elapsed_s:.1f}s`",
        "- information set: no forecast, no planner, no safety guard",
        "- PI gains: locked to `closed_baseline_v1`",
        "- tunable layer: pump actuator thresholds and dwell only",
        "- calibration cases: `01,03,04,06,08`",
        "- reserved validation cases: `01-10`",
        "",
        "## Selection",
        "",
        f"- selected candidate: `{selected}`",
        "",
    ]
    if selected_val is not None:
        lines.extend(
            [
                "## Reserved Validation Result",
                "",
                f"- mean pump saving vs closed_baseline_v1: `{selected_val['mean_pump_saving_pct']:.2f}%`",
                f"- median pump saving vs closed_baseline_v1: `{selected_val['median_pump_saving_pct']:.2f}%`",
                f"- max pitch p95 worsening: `{selected_val['max_pitch_p95_worsen_deg']:.3f} deg`",
                f"- max roll p95 worsening: `{selected_val['max_roll_p95_worsen_deg']:.3f} deg`",
                f"- strict valid: `{int(selected_val['valid_strict'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Calibration Aggregate",
            "",
            "| candidate | mean saving | max d_pitch_p95 | max d_roll_p95 | strict | relaxed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in cal_agg.iterrows():
        lines.append(
            f"| {r['candidate']} | {r['mean_pump_saving_pct']:.2f}% | {r['max_pitch_p95_worsen_deg']:.3f} | {r['max_roll_p95_worsen_deg']:.3f} | {int(r['valid_strict'])} | {int(r['valid_relaxed'])} |"
        )
    lines.extend(
        [
            "",
            "## Reserved Validation Aggregate",
            "",
            "| candidate | mean saving | max d_pitch_p95 | max d_roll_p95 | strict | relaxed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in val_agg.iterrows():
        lines.append(
            f"| {r['candidate']} | {r['mean_pump_saving_pct']:.2f}% | {r['max_pitch_p95_worsen_deg']:.3f} | {r['max_roll_p95_worsen_deg']:.3f} | {int(r['valid_strict'])} | {int(r['valid_relaxed'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `baseline` is selected, the strict actuator-only search did not find a no-preview economy baseline that saves meaningful pump work without changing attitude performance.",
            "- If a non-baseline candidate is selected, use it only as `closed_economy_pareto_v1`, not as a replacement for the frozen fairness baseline.",
            "- This is a reserved validation check, not an out-of-distribution generalization proof.",
        ]
    )
    (out_dir / "closed_economy_pareto_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal_dir = OUT_DIR / "calibration"
    val_dir = OUT_DIR / "reserved_validation"
    cal_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    cal_cases = _selected_cases(CAL_CASE_IDS)
    val_cases = _selected_cases(VAL_CASE_IDS)
    cal_summary = run_cases(cal_cases, CANDIDATES, cal_dir, DEFAULT_DURATION_S, write_timeseries=False)
    cal_delta = compare_to_baseline(cal_summary)
    cal_agg = aggregate_deltas(cal_delta)
    selected = choose_candidate(cal_agg)

    val_candidates = [("baseline", {})]
    if selected != "baseline":
        selected_overrides = dict(CANDIDATES[[name for name, _ in CANDIDATES].index(selected)][1])
        val_candidates.append((selected, selected_overrides))
    val_summary = run_cases(val_cases, val_candidates, val_dir, DEFAULT_DURATION_S, write_timeseries=False)
    val_delta = compare_to_baseline(val_summary)
    val_agg = aggregate_deltas(val_delta)

    cal_summary.to_csv(OUT_DIR / "calibration_summary.csv", index=False)
    cal_delta.to_csv(OUT_DIR / "calibration_deltas.csv", index=False)
    cal_agg.to_csv(OUT_DIR / "calibration_aggregate.csv", index=False)
    val_summary.to_csv(OUT_DIR / "reserved_validation_summary.csv", index=False)
    val_delta.to_csv(OUT_DIR / "reserved_validation_deltas.csv", index=False)
    val_agg.to_csv(OUT_DIR / "reserved_validation_aggregate.csv", index=False)

    meta = {
        "selected_candidate": selected,
        "candidate_grid": [
            {"name": name, "overrides": _safe_cfg_for_json(overrides)}
            for name, overrides in CANDIDATES
        ],
        "calibration_case_ids": sorted(CAL_CASE_IDS),
        "reserved_validation_case_ids": sorted(VAL_CASE_IDS),
        "strict_constraints": {
            "max_pitch_p95_worsen_deg": 0.10,
            "max_roll_p95_worsen_deg": 0.08,
            "max_pitch_max_worsen_deg": 0.0,
            "max_roll_max_worsen_deg": 0.0,
        },
    }
    (OUT_DIR / "closed_economy_pareto_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_report(
        OUT_DIR,
        cal_summary,
        cal_delta,
        cal_agg,
        selected,
        val_summary,
        val_delta,
        val_agg,
        time.perf_counter() - t0,
    )
    print(OUT_DIR / "closed_economy_pareto_report.md")


if __name__ == "__main__":
    main()
