#!/usr/bin/env python3
"""Frozen pump-first A2 suppression experiment.

The forecast-informed restart suppression path was classified as a redundant
default-off overlay during the 2026-06-05 control-chain cleanup. This script is
kept only as historical context and now fails fast instead of launching a long
experiment against a frozen interface.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "archive" / "legacy_fowt_control"))

from wind_prediction.ballast_planner import PlannerConfig
from wind_prediction.ballast_planner_provider import BallastPlannerPreviewProvider
from wind_prediction.replay_dataset import Fino1ReplayDataset


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
DT = 1.0
N_STEPS = 3600


def discover_excel() -> str:
    for d in (repo_root / "data", repo_root / "archive" / "legacy_fowt_control" / "data"):
        for x in d.rglob("*.xlsx"):
            if "stiffness" in x.name.lower() or "matrix" in x.name.lower():
                return str(x)
    raise FileNotFoundError("could not find stiffness/matrix workbook")


def percentiles(x: np.ndarray) -> dict[str, float]:
    a = np.abs(np.asarray(x, dtype=float))
    return {
        "abs_p50": float(np.percentile(a, 50)),
        "abs_p95": float(np.percentile(a, 95)),
        "abs_max": float(np.max(a)),
        "rms": float(np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))),
    }


def summarize_run(df: pd.DataFrame, dt: float) -> dict[str, float]:
    n = len(df)
    pitch = df.get("pitch_deg", pd.Series(np.zeros(n))).to_numpy(dtype=float)
    roll = df.get("roll_deg", pd.Series(np.zeros(n))).to_numpy(dtype=float)
    pump_rate = np.abs(df.get("pump_total_rate_m3_min", pd.Series(np.zeros(n))).to_numpy(dtype=float))
    pump_active = pump_rate > 1e-6
    first_hour_n = int(min(3600 / dt, n))
    out: dict[str, float] = {
        "n_steps": float(n),
        "pump_work_m3": float(np.trapezoid(pump_rate, dx=dt) / 60.0),
        "pump_work_first_hour": float(np.trapezoid(pump_rate[:first_hour_n], dx=dt) / 60.0),
        "pump_duty_ratio": float(np.mean(pump_active)),
        "pump_duty_first_hour": float(np.mean(pump_active[:first_hour_n])),
        "mean_rate_when_on": float(np.mean(pump_rate[pump_active])) if np.any(pump_active) else 0.0,
        "mean_rate_when_on_first_hour": (
            float(np.mean(pump_rate[:first_hour_n][pump_active[:first_hour_n]]))
            if np.any(pump_active[:first_hour_n])
            else 0.0
        ),
        "pump_latch_switch_count": float(df.get("pump_latch_switch_count", pd.Series([0])).max()),
        "pump_stage_switch_count": float(df.get("pump_stage_switch_count", pd.Series([0])).max()),
        "suppression_active_ratio": float(
            np.mean(df.get("preview_pump_suppression_active", pd.Series(np.zeros(n))).to_numpy(dtype=float) > 0.5)
        ),
        "suppression_blocked_tank_steps": float(
            np.sum(df.get("suppression_blocked_tanks", pd.Series(np.zeros(n))).to_numpy(dtype=float))
        ),
        "suppression_blocked_mass_m3": float(
            np.sum(df.get("suppression_blocked_mass_kg", pd.Series(np.zeros(n))).to_numpy(dtype=float)) / 1025.0
        ),
    }
    out.update({f"pitch_{k}": v for k, v in percentiles(pitch).items()})
    out.update({f"roll_{k}": v for k, v in percentiles(roll).items()})
    out["pitch_first_hour_p95"] = float(np.percentile(np.abs(pitch[:first_hour_n]), 95))
    out["pitch_first_hour_max"] = float(np.max(np.abs(pitch[:first_hour_n])))
    out["roll_first_hour_p95"] = float(np.percentile(np.abs(roll[:first_hour_n]), 95))
    out["roll_first_hour_max"] = float(np.max(np.abs(roll[:first_hour_n])))
    return out


def segment_summary(df: pd.DataFrame, dt: float) -> dict[str, float]:
    pump_rate = np.abs(df.get("pump_total_rate_m3_min", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float))
    out: dict[str, float] = {}
    segments = [
        ("0_1h", 0, 3600),
        ("1_2h", 3600, 7200),
        ("2_4h", 7200, 14400),
        ("4_6h", 14400, 21600),
        ("6_10h", 21600, 36000),
    ]
    for name, s_sec, e_sec in segments:
        s = int(max(0, min(len(df), round(s_sec / dt))))
        e = int(max(s, min(len(df), round(e_sec / dt))))
        seg = pump_rate[s:e]
        active = seg > 1e-6
        out[f"pump_work_{name}"] = float(np.trapezoid(seg, dx=dt) / 60.0) if len(seg) else 0.0
        out[f"pump_duty_{name}"] = float(np.mean(active)) if len(seg) else 0.0
        out[f"mean_rate_when_on_{name}"] = float(np.mean(seg[active])) if np.any(active) else 0.0
    return out


def load_windows(limit_per_group: int) -> list[dict[str, object]]:
    scan_dir = repo_root / "outputs" / "wind_prediction" / "transient_onset_scan"
    windows: list[dict[str, object]] = []
    for group, filename in [
        ("onset", "scan_top_onset.csv"),
        ("decay", "scan_top_decay.csv"),
        ("signflip", "scan_top_signflip.csv"),
    ]:
        df = pd.read_csv(scan_dir / filename).head(limit_per_group)
        for _, r in df.iterrows():
            windows.append({"group": group, **r.to_dict()})

    full = pd.read_csv(scan_dir / "scan_full.csv")
    low = full[
        (full["block0_norm"] <= 0.75)
        & (full["block1_norm"] <= 0.75)
        & (full["block2_norm"] <= 0.75)
    ].copy()
    if not low.empty:
        low["score"] = low[["block0_norm", "block1_norm", "block2_norm"]].max(axis=1)
        low = low.sort_values(["score", "timestamp"], ascending=[True, True]).head(limit_per_group)
        for _, r in low.iterrows():
            windows.append({"group": "lowrisk", **r.to_dict()})
    return windows


def make_provider(
    replay: Fino1ReplayDataset,
    ts: datetime,
    cfg: PlannerConfig,
    discounts: list[float],
    variant: dict[str, object],
) -> BallastPlannerPreviewProvider | None:
    if not variant["provider"]:
        return None
    return BallastPlannerPreviewProvider(
        replay_dataset=replay,
        start_timestamp=ts,
        cfg=cfg,
        block_discounts=discounts,
        bias_shape="event_decay",
        objective_mode="economic",
        setpoint_bias_sign=1.0,
        setpoint_channel_enabled=bool(variant["setpoint"]),
        ff_channel_enabled=False,
    )


def build_variants() -> list[dict[str, object]]:
    variants: list[dict[str, object]] = [
        {"tag": "closed_only", "provider": False, "setpoint": False, "suppression": False, "restart_err_kg": 0.0},
        {"tag": "v0_signpos", "provider": True, "setpoint": True, "suppression": False, "restart_err_kg": 0.0},
    ]
    for tons in (1, 2, 4):
        variants.append({
            "tag": f"suppression_only_{tons}t",
            "provider": True,
            "setpoint": False,
            "suppression": True,
            "restart_err_kg": float(tons * 1000),
        })
    for tons in (1, 2, 4):
        variants.append({
            "tag": f"signpos_suppression_{tons}t",
            "provider": True,
            "setpoint": True,
            "suppression": True,
            "restart_err_kg": float(tons * 1000),
        })
    return variants


def paired_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_key in summary_df["case_key"].unique():
        sub = summary_df[summary_df["case_key"] == case_key]
        base_rows = sub[sub["variant"] == "closed_only"]
        if base_rows.empty:
            continue
        base = base_rows.iloc[0]
        for _, r in sub.iterrows():
            if r["variant"] == "closed_only":
                continue
            rows.append({
                "case_key": case_key,
                "group": r["group"],
                "window": r["window"],
                "variant": r["variant"],
                "d_pump_total_pct": (r["pump_work_m3"] - base["pump_work_m3"]) / max(base["pump_work_m3"], 1e-9) * 100.0,
                "d_pump_1h_pct": (r["pump_work_first_hour"] - base["pump_work_first_hour"]) / max(base["pump_work_first_hour"], 1e-9) * 100.0,
                "d_pump_duty": r["pump_duty_ratio"] - base["pump_duty_ratio"],
                "d_mean_rate_when_on": r["mean_rate_when_on"] - base["mean_rate_when_on"],
                "d_latch_switch_count": r["pump_latch_switch_count"] - base["pump_latch_switch_count"],
                "d_stage_switch_count": r["pump_stage_switch_count"] - base["pump_stage_switch_count"],
                "d_pitch_1h_p95": r["pitch_first_hour_p95"] - base["pitch_first_hour_p95"],
                "d_roll_1h_p95": r["roll_first_hour_p95"] - base["roll_first_hour_p95"],
                "d_pitch_max": r["pitch_abs_max"] - base["pitch_abs_max"],
                "d_roll_max": r["roll_abs_max"] - base["roll_abs_max"],
                "budget_002_pass": int(
                    (r["pitch_first_hour_p95"] <= base["pitch_first_hour_p95"] + 0.02)
                    and (r["roll_first_hour_p95"] <= base["roll_first_hour_p95"] + 0.02)
                ),
                "budget_001_pass": int(
                    (r["pitch_first_hour_p95"] <= base["pitch_first_hour_p95"] + 0.01)
                    and (r["roll_first_hour_p95"] <= base["roll_first_hour_p95"] + 0.01)
                ),
                "suppression_active_ratio": r["suppression_active_ratio"],
                "suppression_blocked_tank_steps": r["suppression_blocked_tank_steps"],
                "suppression_blocked_mass_m3": r["suppression_blocked_mass_m3"],
            })
    return pd.DataFrame(rows)


def aggregate(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (group, variant), sub in delta_df.groupby(["group", "variant"], sort=False):
        rows.append({
            "group": group,
            "variant": variant,
            "n": len(sub),
            "d_pump_total_pct_mean": float(sub["d_pump_total_pct"].mean()),
            "d_pump_1h_pct_mean": float(sub["d_pump_1h_pct"].mean()),
            "d_pump_duty_mean": float(sub["d_pump_duty"].mean()),
            "d_mean_rate_when_on_mean": float(sub["d_mean_rate_when_on"].mean()),
            "d_latch_switch_count_mean": float(sub["d_latch_switch_count"].mean()),
            "d_pitch_1h_p95_mean": float(sub["d_pitch_1h_p95"].mean()),
            "d_roll_1h_p95_mean": float(sub["d_roll_1h_p95"].mean()),
            "budget_002_pass_rate": float(sub["budget_002_pass"].mean()),
            "budget_001_pass_rate": float(sub["budget_001_pass"].mean()),
            "suppression_active_ratio_mean": float(sub["suppression_active_ratio"].mean()),
            "suppression_blocked_tank_steps_mean": float(sub["suppression_blocked_tank_steps"].mean()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-group", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=repo_root / "outputs" / "wind_prediction" / "planner_a2_pump_first_suppression")
    parser.add_argument("--save-timeseries", action="store_true", help="write per-case timeseries CSVs; summaries are always written")
    args = parser.parse_args()
    raise RuntimeError(
        "run_a2_pump_first_suppression.py is frozen: preview pump suppression "
        "was removed from the active control path on 2026-06-05. Use "
        "scripts/analysis/run_prediction_primary_casebook.py with the canonical "
        "production profile instead."
    )

    t0 = time.perf_counter()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())
    base_cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]

    excel_path = discover_excel()
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    from defaults import clone_cfg  # noqa: F401
    from run_validation import run_closed_loop_case

    windows = load_windows(limit_per_group=max(1, int(args.limit_per_group)))
    variants = build_variants()
    summary_rows: list[dict[str, object]] = []
    segment_rows: list[dict[str, object]] = []
    issues: list[str] = []

    for win in windows:
        group = str(win["group"])
        ts_str = str(win["timestamp"])
        ts = datetime.strptime(ts_str, TIMESTAMP_FMT)
        case_key = f"{group}_{ts_str}"
        try:
            wind_trace = replay.build_wind_trace(start_timestamp=ts, row_count=N_STEPS // 60, dt_s=DT)
        except ValueError as e:
            issues.append(f"{case_key}: build_wind_trace failed ({e})")
            continue

        for variant in variants:
            tag = f"{group}_{ts_str.replace(':', '').replace(' ', '_')}_{variant['tag']}"
            print(f"\n--- {case_key} variant={variant['tag']} ---")
            provider = make_provider(replay, ts, base_cfg, discounts, variant)
            try:
                _, timeseries = run_closed_loop_case(
                    excel_path=excel_path,
                    case_name=tag,
                    dt=DT,
                    n_steps=int(wind_trace["n_steps"]),
                    wind_trace=wind_trace,
                    control_enabled=True,
                    record_timeseries=True,
                    experiment_protocol="main",
                    platform_profile="default",
                    start_from_heave_equilibrium=True,
                    preview_trim_provider=provider,
                )
            except Exception as e:
                issues.append(f"{tag}: {type(e).__name__}: {e}")
                print(f"   [FAIL] {e}")
                continue

            df = pd.DataFrame(timeseries)
            if args.save_timeseries:
                df.to_csv(out_dir / f"{tag}_timeseries.csv", index=False)
            if provider is not None and provider.records:
                pd.DataFrame(provider.records).to_csv(out_dir / f"{tag}_planner_log.csv", index=False)

            row = summarize_run(df, DT)
            row.update({
                "case_key": case_key,
                "group": group,
                "window": ts_str,
                "variant": variant["tag"],
                "speed_t0": float(win.get("speed_t0", np.nan)),
                "speed_block2": float(win.get("speed_block2", np.nan)),
                "block0_norm": float(win.get("block0_norm", np.nan)),
                "block1_norm": float(win.get("block1_norm", np.nan)),
                "block2_norm": float(win.get("block2_norm", np.nan)),
            })
            if provider is not None and provider.records:
                actions = [str(r["first_action"]) for r in provider.records]
                row["planner_action_mix"] = "|".join(f"{k}:{v}" for k, v in Counter(actions).most_common())
            summary_rows.append(row)

            seg = segment_summary(df, DT)
            seg.update({"case_key": case_key, "group": group, "window": ts_str, "variant": variant["tag"]})
            segment_rows.append(seg)
            print(
                f"   pump_total={row['pump_work_m3']:.1f} pump_1h={row['pump_work_first_hour']:.1f} "
                f"pitch_1h_p95={row['pitch_first_hour_p95']:.3f} "
                f"blocked_steps={row['suppression_blocked_tank_steps']:.0f}"
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "pump_first_summary.csv", index=False)
    pd.DataFrame(segment_rows).to_csv(out_dir / "pump_first_segments.csv", index=False)

    delta_df = paired_deltas(summary_df) if not summary_df.empty else pd.DataFrame()
    delta_df.to_csv(out_dir / "pump_first_delta_paired.csv", index=False)
    agg_df = aggregate(delta_df) if not delta_df.empty else pd.DataFrame()
    agg_df.to_csv(out_dir / "pump_first_aggregate.csv", index=False)

    lines = [
        "# A2 pump-first restart suppression",
        "",
        f"- elapsed: `{time.perf_counter() - t0:.1f}s`",
        f"- groups/windows: `{len(windows)}` windows x `{len(variants)}` variants",
        "- primary: reduce real pump work; pitch/roll are constraints",
        "- budget: report +0.02deg primary and +0.01deg strict first-hour p95 pass/fail",
        "",
        "## Aggregate vs closed_only",
        "",
        "| group | variant | n | d_pump_total | d_pump_1h | d_latch | d_pitch_1h_p95 | d_roll_1h_p95 | pass_0.02 | blocked_steps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    if not agg_df.empty:
        for _, r in agg_df.iterrows():
            lines.append(
                f"| {r['group']} | {r['variant']} | {int(r['n'])} | "
                f"{r['d_pump_total_pct_mean']:+.2f}% | {r['d_pump_1h_pct_mean']:+.2f}% | "
                f"{r['d_latch_switch_count_mean']:+.2f} | {r['d_pitch_1h_p95_mean']:+.4f} | "
                f"{r['d_roll_1h_p95_mean']:+.4f} | {r['budget_002_pass_rate']:.2f} | "
                f"{r['suppression_blocked_tank_steps_mean']:.1f} |"
            )

    lines += [
        "",
        "## Mechanism: segment pump-work delta vs closed_only",
        "",
        "| group | variant | window | 0-1h | 1-2h | 2-4h | 4-6h | 6-10h |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if not summary_df.empty:
        seg_df = pd.DataFrame(segment_rows)
        for case_key in seg_df["case_key"].unique():
            sub = seg_df[seg_df["case_key"] == case_key]
            base_rows = sub[sub["variant"] == "closed_only"]
            if base_rows.empty:
                continue
            base = base_rows.iloc[0]
            for _, r in sub.iterrows():
                if r["variant"] == "closed_only":
                    continue
                lines.append(
                    f"| {r['group']} | {r['variant']} | {r['window']} | "
                    f"{r['pump_work_0_1h'] - base['pump_work_0_1h']:+.1f} | "
                    f"{r['pump_work_1_2h'] - base['pump_work_1_2h']:+.1f} | "
                    f"{r['pump_work_2_4h'] - base['pump_work_2_4h']:+.1f} | "
                    f"{r['pump_work_4_6h'] - base['pump_work_4_6h']:+.1f} | "
                    f"{r['pump_work_6_10h'] - base['pump_work_6_10h']:+.1f} |"
                )

    lines += ["", "## Issues", ""]
    lines += [f"- {x}" for x in issues] if issues else ["None."]
    (out_dir / "pump_first_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {out_dir / 'pump_first_report.md'}")


if __name__ == "__main__":
    main()
