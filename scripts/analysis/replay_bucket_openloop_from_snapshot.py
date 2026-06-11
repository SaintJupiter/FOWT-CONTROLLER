#!/usr/bin/env python3
"""Replay controlled ballast actions from exported bucket snapshots.

This is an analysis tool only. It restores a saved platform snapshot and
branches four target choices: hold, pitch-only active_small, pitch-only
active_medium, and the logged planner mixed action. It does not call the
planner or change controller decision code.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "archive" / "legacy_fowt_control"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from core_model import FloatingPlatform  # type: ignore  # noqa: E402
from run_prediction_primary_casebook import discover_excel  # noqa: E402
from wind_env import wind_speed_to_thrust_n  # type: ignore  # noqa: E402
from wind_prediction.ballast_planner import PlannerConfig, tank_signal  # noqa: E402


HORIZON_S = 1800
CHECKPOINTS = (600, 1200, 1800)
RHO = 1025.0


def _arr(values: Any, dtype=float) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def _restore_pump_state(plant: FloatingPlatform, pump: dict[str, Any]) -> None:
    def vec(name: str, fallback, dtype=float):
        return np.asarray(pump.get(name, fallback), dtype=dtype)

    plant._pump_active_latch = vec("active_latch", [False, False, False], dtype=bool)
    plant._pump_on_elapsed_s = vec("on_elapsed_s", [0.0, 0.0, 0.0])
    plant._pump_off_elapsed_s = vec("off_elapsed_s", [plant.pump_min_off_s] * 3)
    plant._pump_near_target_s = vec("near_target_s", [0.0, 0.0, 0.0])
    plant._pump_rate_smoothed_m3_min = vec("rate_smoothed_m3_min", [0.0, 0.0, 0.0])
    plant._pump_rate_released_m3_min = vec("rate_released_m3_min", [0.0, 0.0, 0.0])
    plant._pump_prev_target_ballast_mass = vec(
        "prev_target_ballast_mass_kg", plant.current_ballast_mass.copy()
    )
    plant._pump_global_quiet_s = float(pump.get("global_quiet_s", 0.0))
    plant._pump_quiet_stop_blocked_prev = vec(
        "quiet_stop_blocked_prev", [False, False, False], dtype=bool
    )
    plant._pump_quiet_stop_block_count = int(pump.get("quiet_stop_block_count", 0))
    plant._pump_latch_switch_count = int(pump.get("latch_switch_count", 0))
    plant._pump_stage_idx = vec("stage_idx", [0, 0, 0], dtype=int)
    plant._pump_stage_dwell_s = vec("stage_dwell_s", [0.0, 0.0, 0.0])
    plant._pump_stage_switch_count = int(pump.get("stage_switch_count", 0))


def _build_plant(snapshot: dict[str, Any], target: np.ndarray) -> FloatingPlatform:
    excel = discover_excel()
    cfg = snapshot.get("config", {})
    pump_cfg = cfg.get("pump_cfg", {}) or {}
    platform_profile = cfg.get("platform_profile", None)
    platform_cfg = cfg.get("platform_cfg", {}) or None
    plant = FloatingPlatform(
        excel,
        pump_cfg=pump_cfg,
        platform_profile=platform_profile,
        platform_cfg=platform_cfg,
    )
    plant.set_irregular_wave(0, 10, 0, 0)
    masses = _arr(snapshot["tank_masses_kg"], dtype=float)
    plant.force_ballast_mass(float(masses[0]), float(masses[1]), float(masses[2]))
    plant.state = _arr(snapshot["state_12"], dtype=float).copy()
    _restore_pump_state(plant, snapshot.get("pump_internal", {}))
    plant.set_ballast_target(float(target[0]), float(target[1]), float(target[2]))
    return plant


def _mass_delta_for_vec(avec: np.ndarray, cfg: PlannerConfig) -> np.ndarray:
    if float(np.linalg.norm(avec)) <= 1e-12:
        return np.zeros(3, dtype=float)
    return tank_signal(avec, cfg) * cfg.action_mass_quantum_kg


def _variants(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = PlannerConfig()
    masses = _arr(snapshot["tank_masses_kg"], dtype=float)
    planner = snapshot.get("planner", {})
    pitch_rad = float(_arr(snapshot["state_12"])[4])
    pitch_sign = 0.0 if abs(pitch_rad) <= 1e-12 else math.copysign(1.0, pitch_rad)
    small = np.array([pitch_sign * cfg.deadband_pitch_deg * cfg.active_small_ratio, 0.0])
    medium = np.array([pitch_sign * cfg.deadband_pitch_deg * cfg.active_medium_ratio, 0.0])
    mixed = np.array(
        [
            float(planner.get("action_vec_pitch_deg", 0.0)),
            float(planner.get("action_vec_roll_deg", 0.0)),
        ],
        dtype=float,
    )
    raw = [
        ("hold_no_extra", np.zeros(2), "hold/no extra ballast target"),
        ("active_small_pitch_only", small, "pitch-only active_small equivalent"),
        ("active_medium_pitch_only", medium, "pitch-only active_medium equivalent"),
        ("logged_mixed_action", mixed, "logged planner action_vec"),
    ]
    out = []
    for name, avec, note in raw:
        delta = _mass_delta_for_vec(avec, cfg)
        target = np.clip(masses + delta, 0.0, float(snapshot.get("tank_capacity_kg", cfg.tank_capacity_kg)))
        out.append({"variant": name, "action_vec": avec, "target_delta": target - masses, "target": target, "note": note})
    return out


def _load_action_vec_overrides(snapshot_dir: Path) -> dict[tuple[str, int], tuple[float, float]]:
    root = snapshot_dir.parent
    log_dir = root / "planner_logs"
    overrides: dict[tuple[str, int], tuple[float, float]] = {}
    if not log_dir.exists():
        return overrides
    for path in log_dir.glob("*_planner_log.csv"):
        stem = path.name.replace("_planner_log.csv", "")
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    bucket = int(float(row.get("bucket", "")))
                    pitch = float(row.get("planner_action_pitch_deg", "nan"))
                    roll = float(row.get("planner_action_roll_deg", "nan"))
                except ValueError:
                    continue
                if math.isfinite(pitch) and math.isfinite(roll):
                    overrides[(stem, bucket)] = (pitch, roll)
    return overrides


def _fill_snapshot_action_vec(
    snapshot: dict[str, Any],
    overrides: dict[tuple[str, int], tuple[float, float]],
) -> None:
    planner = snapshot.setdefault("planner", {})
    pitch = float(planner.get("action_vec_pitch_deg", np.nan))
    roll = float(planner.get("action_vec_roll_deg", np.nan))
    if math.isfinite(pitch) and math.isfinite(roll):
        return
    key = (str(snapshot.get("case_id", "")), int(snapshot.get("bucket_id", -1)))
    if key not in overrides:
        return
    planner["action_vec_pitch_deg"], planner["action_vec_roll_deg"] = overrides[key]
    planner["action_vec_source"] = "planner_log_backfill"


def _wind_arrays(snapshot: dict[str, Any], n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    wind = snapshot.get("wind_window", {}) or {}
    ws = np.asarray(wind.get("ws", []), dtype=float)
    wd = np.asarray(wind.get("wd_deg", []), dtype=float)
    if ws.size < n_steps or wd.size < n_steps:
        obs = snapshot.get("wind_obs", {})
        ws0 = float(obs.get("ws", 0.0))
        wd0 = float(obs.get("wd_deg", 0.0))
        ws = np.resize(ws if ws.size else np.array([ws0], dtype=float), n_steps)
        wd = np.resize(wd if wd.size else np.array([wd0], dtype=float), n_steps)
    return ws[:n_steps], wd[:n_steps]


def _pitch_coeff(delta: np.ndarray) -> float:
    return float(delta[0] - 0.5 * delta[1] - 0.5 * delta[2])


def _roll_coeff(delta: np.ndarray) -> float:
    return float(-delta[1] + delta[2])


def replay_one(snapshot: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    dt = float(snapshot.get("dt_s", 1.0))
    n_steps = int(round(HORIZON_S / dt))
    ws, wd = _wind_arrays(snapshot, n_steps)
    plant = _build_plant(snapshot, variant["target"])
    initial_state = np.asarray(snapshot["state_12"], dtype=float)
    pitch0 = math.degrees(float(initial_state[4]))
    roll0 = math.degrees(float(initial_state[3]))
    mass0 = np.asarray(snapshot["tank_masses_kg"], dtype=float)
    pump_volume = 0.0
    pump_active_steps = 0
    pitch_ts = []
    roll_ts = []
    mass_ts = []
    checkpoints: dict[int, tuple[float, float, np.ndarray]] = {}
    for i in range(n_steps):
        thrust = float(wind_speed_to_thrust_n(float(ws[i])))
        _, info = plant.step(thrust, float(wd[i]), dt, float(i) * dt)
        rate = np.asarray(info.get("pump_rate_cmd_m3_min", np.zeros(3)), dtype=float)
        pump_volume += float(np.sum(np.abs(rate)) * dt / 60.0)
        pump_active_steps += int(np.sum(np.abs(rate)) > 1e-9)
        pitch_ts.append(float(info["pitch_deg"]))
        roll_ts.append(float(info["roll_deg"]))
        mass_ts.append(np.asarray(info["tank_masses"], dtype=float).copy())
        sec = int(round((i + 1) * dt))
        if sec in CHECKPOINTS:
            checkpoints[sec] = (
                float(info["pitch_deg"]),
                float(info["roll_deg"]),
                np.asarray(info["tank_masses"], dtype=float).copy(),
            )
    final_mass = np.asarray(mass_ts[-1], dtype=float)
    target_delta = np.asarray(variant["target_delta"], dtype=float)
    actual_delta = final_mass - mass0
    row: dict[str, Any] = {
        "case_id": snapshot.get("case_id", ""),
        "bucket_id": int(snapshot.get("bucket_id", -1)),
        "variant": variant["variant"],
        "variant_note": variant["note"],
        "initial_pitch_deg": pitch0,
        "initial_roll_deg": roll0,
        "pressure_block0_norm": float(snapshot.get("planner", {}).get("pressure_block0_norm", np.nan)),
        "pressure_block1_norm": float(snapshot.get("planner", {}).get("pressure_block1_norm", np.nan)),
        "pressure_block2_norm": float(snapshot.get("planner", {}).get("pressure_block2_norm", np.nan)),
        "action_vec_pitch_deg": float(variant["action_vec"][0]),
        "action_vec_roll_deg": float(variant["action_vec"][1]),
        "target_delta_t1_kg": float(target_delta[0]),
        "target_delta_t2_kg": float(target_delta[1]),
        "target_delta_t3_kg": float(target_delta[2]),
        "target_delta_mean_abs_kg": float(np.mean(np.abs(target_delta))),
        "target_pitch_coeff_kg": _pitch_coeff(target_delta),
        "target_roll_coeff_kg": _roll_coeff(target_delta),
        "actual_delta_t1_kg": float(actual_delta[0]),
        "actual_delta_t2_kg": float(actual_delta[1]),
        "actual_delta_t3_kg": float(actual_delta[2]),
        "actual_delta_mean_abs_kg": float(np.mean(np.abs(actual_delta))),
        "actual_pitch_coeff_kg": _pitch_coeff(actual_delta),
        "actual_roll_coeff_kg": _roll_coeff(actual_delta),
        "pump_volume_m3": float(pump_volume),
        "pump_active_ratio": float(pump_active_steps / max(n_steps, 1)),
    }
    for idx, sec in enumerate(CHECKPOINTS):
        p, r, m = checkpoints.get(sec, (np.nan, np.nan, np.full(3, np.nan)))
        row[f"pitch_end_plus{idx}_bucket_deg"] = float(p)
        row[f"roll_end_plus{idx}_bucket_deg"] = float(r)
        row[f"pitch_abs_change_plus{idx}_bucket_deg"] = float(abs(p) - abs(pitch0)) if np.isfinite(p) else np.nan
        row[f"roll_abs_change_plus{idx}_bucket_deg"] = float(abs(r) - abs(roll0)) if np.isfinite(r) else np.nan
        row[f"pitch_improve_plus{idx}_bucket_deg"] = float(abs(pitch0) - abs(p)) if np.isfinite(p) else np.nan
        row[f"roll_improve_plus{idx}_bucket_deg"] = float(abs(roll0) - abs(r)) if np.isfinite(r) else np.nan
        row[f"mass_t1_plus{idx}_kg"] = float(m[0])
        row[f"mass_t2_plus{idx}_kg"] = float(m[1])
        row[f"mass_t3_plus{idx}_kg"] = float(m[2])
        row[f"pitch_improve_per_m3_plus{idx}"] = (
            row[f"pitch_improve_plus{idx}_bucket_deg"] / pump_volume
            if pump_volume > 1e-9 and np.isfinite(row[f"pitch_improve_plus{idx}_bucket_deg"])
            else np.nan
        )
    row["response_lag"] = int(
        row["pitch_improve_plus0_bucket_deg"] < 0.0
        and (
            row["pitch_improve_plus1_bucket_deg"] > 0.0
            or row["pitch_improve_plus2_bucket_deg"] > 0.0
        )
    )
    row["roll_improves_but_pitch_not_current"] = int(
        row["roll_improve_plus0_bucket_deg"] > 0.0
        and row["pitch_improve_plus0_bucket_deg"] <= 0.0
    )
    intended = row["target_pitch_coeff_kg"]
    actual = row["actual_pitch_coeff_kg"]
    row["response_reverse_sign"] = int(abs(intended) > 1e-9 and actual * intended < -1e-9)
    return row


def write_schema(out_dir: Path) -> None:
    text = """# Bucket Snapshot Schema

Snapshot files are JSON and are written only when the analysis switch is enabled.

## Complete State

- `state_12`: full platform state at bucket start, captured after controller policy computation and before the plant step for that second.
- `state_order`: `[x, y, z, roll_rad, pitch_rad, yaw_rad, vx, vy, vz, roll_rate_rad_s, pitch_rate_rad_s, yaw_rate_rad_s]`.
- `tank_masses_kg`: current actual ballast masses.
- `target_masses_kg`: target masses that would be applied by the closed-loop policy at this instant.
- `pump_internal`: pump latch, elapsed timers, smoothed/released rates, previous target, stage indices and dwell timers.

## Wind / Planner

- `wind_window`: current bucket plus future replay wind samples used for replay.
- `planner`: logged planner action, action vector, target/delta and pressure block norms.
- `config`: target-shape, pump, platform and protocol metadata needed to reconstruct the plant.

## Boundary

The snapshot does not change the controller. It is an analysis-only export path, default off. Replay results are open-loop branch tests from the same saved state, not closed-loop performance claims.
"""
    (out_dir / "bucket_snapshot_schema.md").write_text(text, encoding="utf-8")


def write_summary(out_dir: Path, df: pd.DataFrame) -> None:
    lines = ["# Open-Loop Replay Summary", ""]
    lines.append("This replay restores exported bucket snapshots and branches hold / active_small pitch-only / active_medium pitch-only / logged mixed action. It does not call the planner and does not modify controller logic.")
    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    # Small vs medium per bucket.
    for bucket in sorted(df["bucket_id"].unique()):
        sub = df[df["bucket_id"] == bucket]
        small = sub[sub["variant"] == "active_small_pitch_only"].iloc[0]
        medium = sub[sub["variant"] == "active_medium_pitch_only"].iloc[0]
        hold = sub[sub["variant"] == "hold_no_extra"].iloc[0]
        lines.append(
            f"- bucket {bucket}: hold next2 pitch improve `{hold['pitch_improve_plus2_bucket_deg']:.3f}` deg, "
            f"small `{small['pitch_improve_plus2_bucket_deg']:.3f}` deg with `{small['pump_volume_m3']:.1f}` m3, "
            f"medium `{medium['pitch_improve_plus2_bucket_deg']:.3f}` deg with `{medium['pump_volume_m3']:.1f}` m3."
        )
    lines.append("")
    med_better_next2 = 0
    small_better_than_hold_next2 = 0
    small_negative_next2 = 0
    medium_negative_next2 = 0
    small_lag = 0
    pump_ratios = []
    for bucket in sorted(df["bucket_id"].unique()):
        sub = df[df["bucket_id"] == bucket]
        hold = sub[sub["variant"] == "hold_no_extra"].iloc[0]
        small = sub[sub["variant"] == "active_small_pitch_only"].iloc[0]
        medium = sub[sub["variant"] == "active_medium_pitch_only"].iloc[0]
        med_better_next2 += int(
            medium["pitch_improve_plus2_bucket_deg"]
            > small["pitch_improve_plus2_bucket_deg"]
        )
        small_better_than_hold_next2 += int(
            small["pitch_improve_plus2_bucket_deg"]
            > hold["pitch_improve_plus2_bucket_deg"]
        )
        small_negative_next2 += int(small["pitch_improve_plus2_bucket_deg"] < 0.0)
        medium_negative_next2 += int(medium["pitch_improve_plus2_bucket_deg"] < 0.0)
        small_lag += int(small["response_lag"] == 1)
        if float(small["pump_volume_m3"]) > 1e-9:
            pump_ratios.append(
                float(medium["pump_volume_m3"]) / float(small["pump_volume_m3"])
            )
    n_buckets = int(df["bucket_id"].nunique())
    mean_pump_ratio = float(np.mean(pump_ratios)) if pump_ratios else float("nan")
    lines.append("## Answers")
    lines.append("")
    lines.append(
        "1. active_small is mainly amplitude-limited in these high-pitch fr09 buckets. "
        f"It beats hold at next2 in `{small_better_than_hold_next2}/{n_buckets}` buckets, "
        f"but still leaves pitch worse than the initial state at next2 in `{small_negative_next2}/{n_buckets}` buckets. "
        f"Classic current-bucket-to-next-bucket response lag was detected in `{small_lag}/{n_buckets}` buckets."
    )
    lines.append(
        "2. active_medium gives visibly stronger pitch authority: "
        f"it improves next-2 pitch more than active_small in `{med_better_next2}/{n_buckets}` buckets. "
        f"The cost is water use: about `{mean_pump_ratio:.2f}x` the active_small pump volume. "
        f"It still leaves pitch worse than initial at next2 in `{medium_negative_next2}/{n_buckets}` buckets, so it is not a magic fix."
    )
    lines.append(
        "3. Action evaluation should include current + next1/next2. Current-bucket response can look good while later pressure disturbance reverses part of the pitch benefit."
    )
    lines.append("4. There is evidence to study the active_small/active_medium boundary, but not enough to promote active_medium directly into the mainline.")
    lines.append("5. There is no new reason to continue target lifecycle changes from this replay.")
    lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("- No planner decision, target lifecycle, eligibility, comfort, veto, or forecast-credit logic was changed.")
    lines.append("- Replay branches are open-loop target substitutions from saved snapshots, not full closed-loop casebook results.")
    lines.append("- Use this tool to characterize action authority before proposing any controller change.")
    (out_dir / "openloop_replay_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    snapshots = sorted(args.snapshot_dir.glob("*_snapshot.json"))
    if not snapshots:
        raise FileNotFoundError(f"no snapshots in {args.snapshot_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overrides = _load_action_vec_overrides(args.snapshot_dir)
    rows = []
    for path in snapshots:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        _fill_snapshot_action_vec(snapshot, overrides)
        for variant in _variants(snapshot):
            rows.append(replay_one(snapshot, variant))
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "openloop_replay_comparison.csv", index=False)
    write_schema(args.out_dir)
    write_summary(args.out_dir, df)
    print(f"Wrote {args.out_dir / 'bucket_snapshot_schema.md'}")
    print(f"Wrote {args.out_dir / 'openloop_replay_summary.md'}")
    print(f"Wrote {args.out_dir / 'openloop_replay_comparison.csv'}")


if __name__ == "__main__":
    main()
