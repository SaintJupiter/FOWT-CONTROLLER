#!/usr/bin/env python3
"""Cost replay for objective_mode 2-state switch (default-off prototype).

Decision logic (in provider per bucket):

  recovery mode when:
      persistent_high_posture_counter >= enter_buckets
      AND NOT future_rising_pressure
  -> effective cfg: smooth_huber attitude cost, w_attitude=3

  economic mode otherwise:
      preserves quadratic attitude cost, w_attitude=1

This script does NOT run the live state machine; it directly constructs the
two effective cfgs and evaluates the same 4 scenarios. The mode the state
machine would have selected is annotated, and the scalar that mode produces
is the winning answer.

Scenarios:
  S1 lowrisk_clean   (pitch=4.95, wind 1.5, no rising)
        Expected mode: recovery (persistent_high, not rising)
        Expected best: hold (smooth_huber zones lowrisk residual to 0)

  S2 fr_relief_09    (pitch=5.0, wind 18 decaying)
        Expected mode: recovery (persistent_high, decay = not rising)
        Expected best: active_small (residual high enough above zone)

  S3 fr01_weak       (pitch=3.0, wind 5 mild)
        Expected mode: recovery (persistent_high, not rising)
        Expected best: hold (residual ~ delta, zone wins)

  S4 onset_rising    (pitch=3.0, wind 5->10->15 rising)
        Expected mode: economic (future_rising signals economic)
        Expected best: active_medium or active_small (quadratic preempts)
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction.ballast_planner import (  # noqa: E402
    PlannerConfig,
    apply_posture_state_to_blocks,
    economic_scalar_cost,
    evaluate_sequence,
    pressure_proxy_vec,
    norm_term,
)


def make_blocks(cfg, mean_wind_ms, decay):
    rows = []
    speeds = [mean_wind_ms - i * decay for i in range(3)]
    for idx, sp in enumerate(speeds):
        uv = np.array([[-sp, 0.0], [-sp, 0.0]], dtype=float)
        raw = pressure_proxy_vec(uv, cfg)
        pvec = raw * float([1.0, 0.85, 0.7][idx])
        rows.append({
            "block_name": f"b{idx}",
            "pressure_vec": pvec,
            "pressure_vec_raw": raw,
            "pressure_norm": norm_term(pvec, cfg),
        })
    return rows


def determine_mode_for_scenario(cfg, pitch, blocks, persistent_buckets=3):
    """Simulate the provider's mode decision for this scenario."""
    posture_norm = abs(pitch) / cfg.deadband_pitch_deg
    block_norms_raw = [norm_term(np.asarray(b["pressure_vec_raw"], dtype=float), cfg) for b in blocks]
    future_rising = (
        max(block_norms_raw[1], block_norms_raw[2])
        > block_norms_raw[0] + float(cfg.recovery_mode_future_rising_norm_diff)
    )
    high = posture_norm > float(cfg.recovery_mode_posture_norm_threshold)
    # Simulate persistent counter ≥ enter_buckets:
    use_recovery = (high and persistent_buckets >= cfg.recovery_mode_enter_buckets and not future_rising)
    return "recovery" if use_recovery else "economic", future_rising, posture_norm


def evaluate_for_mode(mode, base_cfg, pitch, wind, decay):
    if mode == "recovery":
        eff = dataclasses.replace(
            base_cfg,
            attitude_zone_form="smooth_huber",
            attitude_zone_delta_norm=base_cfg.recovery_mode_zone_delta_norm,
            attitude_zone_terminal_delta_norm=base_cfg.recovery_mode_zone_delta_norm,
            w_attitude_residual=base_cfg.recovery_mode_w_attitude,
            w_terminal_residual=base_cfg.recovery_mode_w_terminal,
        )
    else:
        eff = base_cfg  # quadratic + w=1
    plant = {
        "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
        "pump_fullspeed_any": 0,
        "pump_total_backlog_kg": 0.0,
        "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        "posture_vec_deg": np.array([pitch, 0.0], dtype=float),
        "forecast_has_future": 1,
    }
    blocks = make_blocks(eff, wind, decay)
    adjusted, meta = apply_posture_state_to_blocks(blocks, plant, eff)
    plant["posture_state_norm"] = meta["posture_state_norm"]
    plant["posture_state_raw_norm"] = meta["posture_state_raw_norm"]
    plant["posture_state_credit"] = meta["posture_state_credit"]
    plant["posture_state_credit_reason"] = meta["posture_state_credit_reason"]
    sequences = [
        ("hold", "hold", "hold"),
        ("pump_saving", "hold", "hold"),
        ("active_small", "hold", "hold"),
        ("active_medium", "hold", "hold"),
    ]
    rows = []
    for seq in sequences:
        r = evaluate_sequence(seq, adjusted, plant, eff)
        if r["hard_reject_reason"]:
            rows.append((">".join(seq), float("inf"), r["hard_reject_reason"]))
            continue
        rows.append((">".join(seq), economic_scalar_cost(r, eff), ""))
    best = min((r for r in rows if r[2] == ""), key=lambda r: r[1])
    return rows, best


def main():
    cfg = PlannerConfig(
        objective_mode_active=True,  # exercise the prototype path
    )

    scenarios = [
        ("S1_lowrisk_clean",  4.95, 1.5,  0.0,  "recovery", "hold"),
        ("S2_fr_relief_09",   5.0,  18.0, 1.5,  "recovery", "active_small or active_medium"),
        ("S3_fr01_weak",      3.0,  5.0,  0.5,  "recovery", "hold"),
        ("S4_onset_rising",   3.0,  5.0, -5.0,  "economic", "active_small or active_medium"),
    ]
    print(f"{'scenario':22s} {'expected_mode':12s} {'mode_picked':12s} {'future_rising':14s} {'best':30s}  expected")
    for label, pitch, wind, decay, exp_mode, exp_best in scenarios:
        blocks = make_blocks(cfg, wind, decay)
        mode, future_rising, posture_norm = determine_mode_for_scenario(cfg, pitch, blocks, persistent_buckets=3)
        rows, best = evaluate_for_mode(mode, cfg, pitch, wind, decay)
        marker = "✓" if mode == exp_mode else "✗"
        print(f"{label:22s} {exp_mode:12s} {mode:12s} {str(bool(future_rising)):14s} {best[0]:30s}  {exp_best}  {marker}")


if __name__ == "__main__":
    main()
