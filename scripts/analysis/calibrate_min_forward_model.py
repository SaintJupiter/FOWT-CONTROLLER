#!/usr/bin/env python3
"""Calibrate minimal forward models from closed_only timeseries.

Step 0b gate for prediction-primary redesign:
  1) Can a minimal pump actuator replay reproduce pump_work from target masses?
  2) Can a simple one-step ARX model predict pitch/roll well enough to support
     counterfactual scheduling?

This is a read-only analysis over existing CSVs; it does not run simulation or
change controller code.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "outputs" / "wind_prediction"
DEFAULT_OUT_DIR = DEFAULT_INPUT_ROOT / "min_forward_model_calibration"
TANKS = (1, 2, 3)


def discover_closed_only_files(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("*closed_only_timeseries.csv") if p.is_file())


def case_key(path: Path) -> str:
    name = path.stem
    if name.endswith("_timeseries"):
        name = name[: -len("_timeseries")]
    for prefix in ("onset_", "decay_", "signflip_", "lowrisk_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.endswith("_closed_only"):
        name = name[: -len("_closed_only")]
    return name


def dedupe_files(files: list[Path]) -> list[Path]:
    preferred_dirs = {
        "planner_a2_pump_first_suppression",
        "planner_a2_pump_first_suppression_full",
        "planner_a2_economic_expanded",
        "planner_a2_onset",
        "planner_a2_smoke",
    }

    def score(path: Path) -> tuple[int, int, str]:
        explicit = int(path.name.lower().startswith(("onset_", "decay_", "signflip_", "lowrisk_")))
        preferred = int(path.parent.name in preferred_dirs)
        return (explicit, preferred, str(path))

    chosen: dict[str, Path] = {}
    for path in sorted(files, key=score, reverse=True):
        chosen.setdefault(case_key(path), path)
    return sorted(chosen.values())


def scenario_group(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    for prefix in ("onset", "decay", "signflip", "lowrisk"):
        if name.startswith(prefix + "_"):
            return prefix
    if "onset" in parent:
        return "onset_existing"
    if "economic_expanded" in parent:
        return "expanded_existing"
    if "smoke" in parent:
        return "smoke_existing"
    return "other_existing"


def read_df(path: Path) -> pd.DataFrame | None:
    required = [
        "t_s",
        "wind_speed",
        "wind_dir_deg",
        "pitch_deg",
        "roll_deg",
        "pump_total_rate_m3_min",
    ]
    for tank in TANKS:
        required.extend(
            [
                f"tank{tank}_kg",
                f"target_tank{tank}_kg",
                f"pump_rate{tank}_m3min",
            ]
        )
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in required if c not in set(header.columns)]
    if missing:
        print(f"[SKIP] {path}: missing {missing[:5]}")
        return None
    return pd.read_csv(path, usecols=required)


def median_dt_s(df: pd.DataFrame) -> float:
    t = pd.to_numeric(df["t_s"], errors="coerce").to_numpy(dtype=float)
    diffs = np.diff(t[np.isfinite(t)])
    diffs = diffs[diffs > 1e-9]
    return float(np.median(diffs)) if len(diffs) else 1.0


def pump_schedule_points() -> list[tuple[float, float]]:
    return sorted(
        [
            (3000.0, 15.0),
            (2000.0, 14.0),
            (1000.0, 12.0),
            (700.0, 10.0),
            (500.0, 8.0),
            (300.0, 6.0),
            (200.0, 4.0),
            (0.0, 0.0),
        ],
        key=lambda x: x[0],
    )


def pump_stage_rate(abs_err: float) -> float:
    points = pump_schedule_points()
    e = float(abs_err)
    if e <= points[0][0]:
        return float(points[0][1])
    if e >= points[-1][0]:
        return float(points[-1][1])
    for i in range(len(points) - 1):
        e0, r0 = points[i]
        e1, r1 = points[i + 1]
        if e0 <= e <= e1:
            if abs(e1 - e0) < 1e-12:
                return float(r1)
            frac = (e - e0) / (e1 - e0)
            return float(r0 + frac * (r1 - r0))
    return float(points[-1][1])


def pump_stage_idx(abs_err: float, allow_zero_stage: bool) -> int:
    points = pump_schedule_points()
    e = float(abs_err)
    if e <= points[0][0]:
        idx = 0
    else:
        idx = len(points) - 2
        for cand in range(len(points) - 1):
            if e <= points[cand + 1][0]:
                idx = cand
                break
    if not allow_zero_stage and len(points) > 2:
        idx = max(1, idx)
    return int(idx)


def pump_rate_from_stage(abs_err: float, allow_zero_stage: bool) -> float:
    idx = pump_stage_idx(abs_err, allow_zero_stage=allow_zero_stage)
    points = pump_schedule_points()
    idx = int(np.clip(idx, 0, len(points) - 2))
    e0, r0 = points[idx]
    e1, r1 = points[idx + 1]
    e = float(np.clip(float(abs_err), e0, e1))
    if abs(e1 - e0) < 1e-12:
        return float(r1)
    frac = (e - e0) / (e1 - e0)
    return float(r0 + frac * (r1 - r0))


def apply_pump_mode_defaults(args: argparse.Namespace) -> None:
    if str(args.pump_mode).lower() == "legacy":
        args.pump_stop_err_kg = 150.0
        args.pump_restart_err_kg = 300.0
        args.pump_min_on_s = 12.0
        args.pump_min_off_s = 6.0
        args.pump_hold_before_stop_s = 0.0
        args.pump_ramp_up_m3_min_per_s = float("inf")
        args.pump_ramp_down_m3_min_per_s = float("inf")
    elif str(args.pump_mode).lower() == "default":
        args.pump_stop_err_kg = 300.0
        args.pump_restart_err_kg = 500.0
        args.pump_min_on_s = 20.0
        args.pump_min_off_s = 12.0
        args.pump_hold_before_stop_s = 10.0
        args.pump_ramp_up_m3_min_per_s = 2.0
        args.pump_ramp_down_m3_min_per_s = 3.0
    elif str(args.pump_mode).lower() == "custom":
        return
    else:
        raise ValueError("--pump-mode must be legacy/default/custom")


def ramp_towards(current: float, target: float, up_per_s: float, down_per_s: float, dt_s: float) -> float:
    if not math.isfinite(up_per_s) and target >= current:
        return float(target)
    if not math.isfinite(down_per_s) and target < current:
        return float(target)
    if target >= current:
        max_delta = up_per_s * dt_s
    else:
        max_delta = down_per_s * dt_s
    return float(current + np.clip(target - current, -max_delta, max_delta))


def replay_pump(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    dt_s = median_dt_s(df)
    rho = float(args.rho)
    n = len(df)
    masses = np.vstack(
        [pd.to_numeric(df[f"tank{tank}_kg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float) for tank in TANKS]
    ).T
    targets = np.vstack(
        [
            pd.to_numeric(df[f"target_tank{tank}_kg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
            for tank in TANKS
        ]
    ).T
    actual_rates = np.vstack(
        [
            pd.to_numeric(df[f"pump_rate{tank}_m3min"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for tank in TANKS
        ]
    ).T
    sim_rates = np.zeros_like(actual_rates)
    latched = np.zeros(3, dtype=bool)
    on_elapsed = np.zeros(3, dtype=float)
    off_elapsed = np.full(3, float(args.pump_min_off_s), dtype=float)
    near_target = np.zeros(3, dtype=float)
    smoothed_rate = np.zeros(3, dtype=float)

    for k in range(n):
        # Timeseries rows are recorded after plant.step(). Pump latch/rate for row k
        # were computed from the pre-step mass, which is row k-1 for k > 0.
        current = masses[k - 1] if k > 0 else masses[k]
        target = targets[k]
        for j in range(3):
            err = target[j] - current[j]
            abs_err = abs(float(err))
            if abs_err < float(args.pump_stop_err_kg):
                near_target[j] += dt_s
            else:
                near_target[j] = 0.0

            if latched[j]:
                on_elapsed[j] += dt_s
                off_elapsed[j] = 0.0
            else:
                off_elapsed[j] += dt_s
                on_elapsed[j] = 0.0

            local_stop_ready = (
                abs_err < float(args.pump_stop_err_kg)
                and on_elapsed[j] >= float(args.pump_min_on_s)
                and near_target[j] >= float(args.pump_hold_before_stop_s)
            )
            if latched[j]:
                if local_stop_ready:
                    latched[j] = False
                    on_elapsed[j] = 0.0
                    off_elapsed[j] = 0.0
                    near_target[j] = 0.0
            else:
                if abs_err > float(args.pump_restart_err_kg) and off_elapsed[j] >= float(args.pump_min_off_s):
                    latched[j] = True
                    on_elapsed[j] = 0.0
                    off_elapsed[j] = 0.0
                    near_target[j] = 0.0

            target_rate = pump_rate_from_stage(abs_err, allow_zero_stage=False) if latched[j] else 0.0
            smoothed_rate[j] = ramp_towards(
                current=smoothed_rate[j],
                target=target_rate,
                up_per_s=float(args.pump_ramp_up_m3_min_per_s),
                down_per_s=float(args.pump_ramp_down_m3_min_per_s),
                dt_s=dt_s,
            )
            sim_rates[k, j] = smoothed_rate[j]

    actual_total = np.sum(np.abs(actual_rates), axis=1)
    sim_total = np.sum(np.abs(sim_rates), axis=1)
    actual_work = float(np.sum(actual_total) * dt_s / 60.0)
    sim_work = float(np.sum(sim_total) * dt_s / 60.0)
    mae_rate = float(np.mean(np.abs(sim_total - actual_total)))
    return {
        "pump_actual_work_m3": actual_work,
        "pump_replay_work_m3": sim_work,
        "pump_work_rel_err_pct": float((sim_work - actual_work) / max(actual_work, 1e-9) * 100.0),
        "pump_total_rate_mae_m3_min": mae_rate,
        "pump_total_rate_actual_mean_m3_min": float(np.mean(actual_total)),
    }


def wind_features(df: pd.DataFrame) -> np.ndarray:
    ws = pd.to_numeric(df["wind_speed"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
    wd = np.deg2rad(pd.to_numeric(df["wind_dir_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float))
    ws = np.clip(ws, 0.0, 40.0)
    # Use ws^2 projected components as a minimal wind load proxy.
    return np.column_stack([ws * ws * np.cos(wd), ws * ws * np.sin(wd), ws])


def ballast_features(df: pd.DataFrame) -> np.ndarray:
    masses = np.vstack(
        [pd.to_numeric(df[f"tank{tank}_kg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float) for tank in TANKS]
    ).T
    masses = np.clip(masses, 0.0, 1850.0 * 1025.0)
    centered = masses - np.mean(masses, axis=1, keepdims=True)
    return centered / 100000.0


def build_attitude_dataset(files: list[Path]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for file_idx, path in enumerate(files):
        df = read_df(path)
        if df is None or len(df) < 2:
            continue
        pitch = pd.to_numeric(df["pitch_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        roll = pd.to_numeric(df["roll_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        pitch = np.clip(pitch, -20.0, 20.0)
        roll = np.clip(roll, -20.0, 20.0)
        wf = wind_features(df)
        bf = ballast_features(df)
        state = np.column_stack([pitch, roll])
        # One-step linear ARX: next attitude from current attitude + wind + ballast + bias.
        x = np.column_stack([np.ones(len(df) - 1), state[:-1], wf[:-1], bf[:-1]])
        y = state[1:]
        finite = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        x = x[finite]
        y = y[finite]
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        meta.append({"file_idx": file_idx, "path": str(path), "n_rows": len(df), "n_samples": len(x)})
    if not xs:
        raise ValueError("no finite attitude training samples")
    return np.vstack(xs), np.vstack(ys), meta


def fit_ridge(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = np.zeros(x.shape[1], dtype=float)
    scale = np.ones(x.shape[1], dtype=float)
    if x.shape[1] > 1:
        mean[1:] = np.mean(x[:, 1:], axis=0)
        scale[1:] = np.std(x[:, 1:], axis=0)
        scale[scale < 1e-9] = 1.0
    xs = (x - mean) / scale
    xs[:, 0] = 1.0
    xtx = xs.T @ xs
    reg = lam * np.eye(xtx.shape[0])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(xtx + reg, xs.T @ y)
    return coef, mean, scale


def apply_linear_model(x: np.ndarray, coef: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    xs = (np.asarray(x, dtype=float) - mean) / scale
    xs[:, 0] = 1.0
    return xs @ coef


def predict_rollout(
    df: pd.DataFrame,
    coef: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    horizon_s: float,
) -> dict[str, float]:
    dt_s = median_dt_s(df)
    h = max(1, int(round(horizon_s / max(dt_s, 1e-9))))
    pitch = pd.to_numeric(df["pitch_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
    roll = pd.to_numeric(df["roll_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
    pitch = np.clip(pitch, -20.0, 20.0)
    roll = np.clip(roll, -20.0, 20.0)
    wf = wind_features(df)
    bf = ballast_features(df)
    n_eval = max(0, len(df) - h)
    if n_eval <= 0:
        return {
            "pitch_rollout_rmse_deg": 0.0,
            "roll_rollout_rmse_deg": 0.0,
            "pitch_rollout_p95_abs_err_deg": 0.0,
            "roll_rollout_p95_abs_err_deg": 0.0,
        }
    pred = np.column_stack([pitch[:n_eval], roll[:n_eval]]).astype(float)
    finite_eval = np.isfinite(pred).all(axis=1)
    for step in range(h):
        idx = np.arange(n_eval) + step
        x = np.column_stack([np.ones(n_eval), pred, wf[idx], bf[idx]])
        finite_eval &= np.isfinite(x).all(axis=1)
        pred = apply_linear_model(x, coef=coef, mean=mean, scale=scale)
    actual = np.column_stack([pitch[h : h + n_eval], roll[h : h + n_eval]])
    finite_eval &= np.isfinite(pred).all(axis=1) & np.isfinite(actual).all(axis=1)
    pred = pred[finite_eval]
    actual = actual[finite_eval]
    if len(pred) == 0:
        return {
            "pitch_rollout_rmse_deg": math.inf,
            "roll_rollout_rmse_deg": math.inf,
            "pitch_rollout_p95_abs_err_deg": math.inf,
            "roll_rollout_p95_abs_err_deg": math.inf,
        }
    err = pred - actual
    return {
        "pitch_rollout_rmse_deg": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "roll_rollout_rmse_deg": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "pitch_rollout_p95_abs_err_deg": float(np.percentile(np.abs(err[:, 0]), 95)),
        "roll_rollout_p95_abs_err_deg": float(np.percentile(np.abs(err[:, 1]), 95)),
    }


def evaluate_attitude(files: list[Path], train_frac: float, horizon_s: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(files)
    split = max(1, min(len(files) - 1, int(round(len(files) * train_frac)))) if len(files) > 1 else 1
    train_files = files[:split]
    test_files = files[split:] if len(files) > 1 else files
    x_train, y_train, _ = build_attitude_dataset(train_files)
    coef, mean, scale = fit_ridge(x_train, y_train)

    rows: list[dict[str, Any]] = []
    for path in test_files:
        df = read_df(path)
        if df is None or len(df) < 2:
            continue
        pitch = pd.to_numeric(df["pitch_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        roll = pd.to_numeric(df["roll_deg"], errors="coerce").ffill().fillna(0.0).to_numpy(dtype=float)
        pitch = np.clip(pitch, -20.0, 20.0)
        roll = np.clip(roll, -20.0, 20.0)
        wf = wind_features(df)
        bf = ballast_features(df)
        x = np.column_stack([np.ones(len(df) - 1), pitch[:-1], roll[:-1], wf[:-1], bf[:-1]])
        actual = np.column_stack([pitch[1:], roll[1:]])
        finite = np.isfinite(x).all(axis=1) & np.isfinite(actual).all(axis=1)
        x = x[finite]
        actual = actual[finite]
        if len(x) == 0:
            continue
        pred = apply_linear_model(x, coef=coef, mean=mean, scale=scale)
        err = pred - actual
        rollout = predict_rollout(df, coef=coef, mean=mean, scale=scale, horizon_s=horizon_s)
        rows.append(
            {
                "file": str(path),
                "group": scenario_group(path),
                "case": case_key(path),
                "attitude_1step_pitch_rmse_deg": float(np.sqrt(np.mean(err[:, 0] ** 2))),
                "attitude_1step_roll_rmse_deg": float(np.sqrt(np.mean(err[:, 1] ** 2))),
                "attitude_1step_pitch_p95_abs_err_deg": float(np.percentile(np.abs(err[:, 0]), 95)),
                "attitude_1step_roll_p95_abs_err_deg": float(np.percentile(np.abs(err[:, 1]), 95)),
                "pitch_actual_p95_deg": float(np.percentile(np.abs(pitch), 95)),
                "roll_actual_p95_deg": float(np.percentile(np.abs(roll), 95)),
                **rollout,
            }
        )
    meta = {
        "train_files": [str(p) for p in train_files],
        "test_files": [str(p) for p in test_files],
        "coef_shape": list(coef.shape),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
    }
    return pd.DataFrame(rows), meta


def analyze_files(files: list[Path], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pump_rows: list[dict[str, Any]] = []
    for path in files:
        df = read_df(path)
        if df is None:
            continue
        row = {
            "file": str(path),
            "group": scenario_group(path),
            "case": case_key(path),
            "duration_h": float((pd.to_numeric(df["t_s"], errors="coerce").iloc[-1] + median_dt_s(df)) / 3600.0),
        }
        row.update(replay_pump(df, args=args))
        pump_rows.append(row)
        print(f"[PUMP] {row['group']} {row['case']}: err={row['pump_work_rel_err_pct']:+.2f}%")
    att_df, att_meta = evaluate_attitude(files, train_frac=float(args.train_frac), horizon_s=float(args.rollout_horizon_s))
    return pd.DataFrame(pump_rows), att_df, att_meta


def pass_fail(pump_df: pd.DataFrame, att_df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    pump_abs = np.abs(pump_df["pump_work_rel_err_pct"].to_numpy(dtype=float)) if not pump_df.empty else np.array([])
    pitch_rollout = att_df["pitch_rollout_p95_abs_err_deg"].to_numpy(dtype=float) if not att_df.empty else np.array([])
    roll_rollout = att_df["roll_rollout_p95_abs_err_deg"].to_numpy(dtype=float) if not att_df.empty else np.array([])
    return {
        "pump_median_abs_work_err_pct": float(np.median(pump_abs)) if len(pump_abs) else math.nan,
        "pump_p80_abs_work_err_pct": float(np.percentile(pump_abs, 80)) if len(pump_abs) else math.nan,
        "pitch_rollout_p95_abs_err_median_deg": float(np.median(pitch_rollout)) if len(pitch_rollout) else math.nan,
        "roll_rollout_p95_abs_err_median_deg": float(np.median(roll_rollout)) if len(roll_rollout) else math.nan,
        "pump_gate_pass_15pct": bool(len(pump_abs) and np.percentile(pump_abs, 80) <= float(args.pump_err_gate_pct)),
        "attitude_gate_pass": bool(
            len(pitch_rollout)
            and len(roll_rollout)
            and np.median(pitch_rollout) <= float(args.attitude_rollout_gate_deg)
            and np.median(roll_rollout) <= float(args.attitude_rollout_gate_deg)
        ),
    }


def write_report(out_dir: Path, files: list[Path], pump_df: pd.DataFrame, att_df: pd.DataFrame, gate: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Minimal Forward Model Calibration",
        "",
        "Read-only Step 0b. No controller changes and no closed-loop simulation.",
        "",
        "## Config",
        "",
        f"- deduped files: `{len(files)}`",
        f"- pump_mode: `{args.pump_mode}`",
        f"- pump stop/restart: `{float(args.pump_stop_err_kg):.1f}/{float(args.pump_restart_err_kg):.1f} kg`",
        f"- pump min on/off: `{float(args.pump_min_on_s):.1f}/{float(args.pump_min_off_s):.1f} s`",
        f"- pump error gate p80: `{float(args.pump_err_gate_pct):.1f}%`",
        f"- attitude rollout horizon: `{float(args.rollout_horizon_s):.0f}s`",
        f"- attitude rollout gate median p95 error: `{float(args.attitude_rollout_gate_deg):.3f}deg`",
        "",
        "## Gate",
        "",
        f"- pump median abs work error: `{gate['pump_median_abs_work_err_pct']:.2f}%`",
        f"- pump p80 abs work error: `{gate['pump_p80_abs_work_err_pct']:.2f}%`",
        f"- pitch rollout median p95 abs error: `{gate['pitch_rollout_p95_abs_err_median_deg']:.3f}deg`",
        f"- roll rollout median p95 abs error: `{gate['roll_rollout_p95_abs_err_median_deg']:.3f}deg`",
        f"- pump replay gate pass: `{int(gate['pump_gate_pass_15pct'])}`",
        f"- attitude model gate pass: `{int(gate['attitude_gate_pass'])}`",
        "",
    ]
    if gate["pump_gate_pass_15pct"] and gate["attitude_gate_pass"]:
        lines.append("Decision: PROCEED to Step 0c counterfactual oracle.")
    elif gate["pump_gate_pass_15pct"]:
        lines.append("Decision: pump actuator replay is usable, but attitude model is too weak for schedule optimization; improve model or limit Step 0c to pump-layer upper-bound only.")
    else:
        lines.append("Decision: STOP prediction-primary implementation until pump actuator replay is fixed; counterfactual pump scheduling would not be credible.")

    lines += [
        "",
        "## Pump Replay By Group",
        "",
        "| group | files | median_abs_work_err_pct | p80_abs_work_err_pct |",
        "|---|---:|---:|---:|",
    ]
    if not pump_df.empty:
        tmp = pump_df.copy()
        tmp["abs_err"] = np.abs(tmp["pump_work_rel_err_pct"])
        for group, sub in tmp.groupby("group", sort=True):
            lines.append(
                f"| {group} | {len(sub)} | {np.median(sub['abs_err']):.2f}% | {np.percentile(sub['abs_err'], 80):.2f}% |"
            )

    lines += [
        "",
        "## Outputs",
        "",
        "- `pump_replay_calibration.csv`",
        "- `attitude_arx_calibration.csv`",
        "- `calibration_meta.json`",
    ]
    (out_dir / "min_forward_model_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--pump-mode", choices=["legacy", "default", "custom"], default="legacy")
    ap.add_argument("--rho", type=float, default=1025.0)
    ap.add_argument("--pump-stop-err-kg", type=float, default=150.0)
    ap.add_argument("--pump-restart-err-kg", type=float, default=300.0)
    ap.add_argument("--pump-min-on-s", type=float, default=12.0)
    ap.add_argument("--pump-min-off-s", type=float, default=6.0)
    ap.add_argument("--pump-hold-before-stop-s", type=float, default=0.0)
    ap.add_argument("--pump-ramp-up-m3-min-per-s", type=float, default=float("inf"))
    ap.add_argument("--pump-ramp-down-m3-min-per-s", type=float, default=float("inf"))
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--rollout-horizon-s", type=float, default=600.0)
    ap.add_argument("--pump-err-gate-pct", type=float, default=15.0)
    ap.add_argument("--attitude-rollout-gate-deg", type=float, default=0.30)
    args = ap.parse_args()
    apply_pump_mode_defaults(args)

    files = discover_closed_only_files(Path(args.input_root))
    if not bool(args.no_dedupe):
        files = dedupe_files(files)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]
    if not files:
        raise FileNotFoundError(f"no closed_only_timeseries files under {args.input_root}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {len(files)} files.")
    pump_df, att_df, att_meta = analyze_files(files, args=args)
    gate = pass_fail(pump_df, att_df, args=args)
    pump_df.to_csv(out_dir / "pump_replay_calibration.csv", index=False)
    att_df.to_csv(out_dir / "attitude_arx_calibration.csv", index=False)
    (out_dir / "calibration_meta.json").write_text(
        json.dumps({"attitude": att_meta, "gate": gate}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_report(out_dir=out_dir, files=files, pump_df=pump_df, att_df=att_df, gate=gate, args=args)
    print(f"Report: {out_dir / 'min_forward_model_report.md'}")


if __name__ == "__main__":
    main()
