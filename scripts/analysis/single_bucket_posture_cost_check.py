#!/usr/bin/env python3
"""Single-bucket cost-function diagnostic for posture-aware planner cost.

Tests the *math* of evaluate_sequence + economic_scalar_cost in a controlled
scenario, without running any closed-loop simulation.

Scenario: lowrisk_clean-like.
- Current pitch = 4.5 deg (well outside 1.2 deg envelope).
- Forecast: 3 blocks of low wind pressure (norm ~0.4, decaying slightly).
- Tank masses balanced near nominal.

Two controller configurations are tested:

  baseline:
    w_attitude_residual = 0
    w_terminal_residual = 0
    posture_state_residual_active = False
    posture_hold_barrier_active = False

  posture_aware:
    w_attitude_residual = 1.0
    w_terminal_residual = 1.0
    posture_state_residual_active = True
    posture_hold_barrier_active = False

For each config we evaluate four representative sequences:
  - hold > hold > hold
  - pump_saving > hold > hold
  - active_small > hold > hold
  - active_medium > hold > hold

For each sequence we print the scalar cost. The intent:

  In baseline: hold should win (planner ignores attitude).
  In posture_aware: an active sequence should beat hold.

This is a math check, not an experiment.
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


def synthesize_lowrisk_blocks(cfg: PlannerConfig) -> list[dict]:
    """Build 3 forecast blocks for a calm-wind window."""
    # ~5 m/s wind from ~270 deg, decaying slightly.
    uv = np.array(
        [
            [-5.0, 0.0],
            [-4.8, 0.0],
            [-4.6, 0.0],
            [-4.4, 0.0],
            [-4.2, 0.0],
            [-4.0, 0.0],
        ],
        dtype=float,
    )
    discounts = [1.0, 0.85, 0.7]
    blocks: list[dict] = []
    for idx, (name, s, e) in enumerate((("b0", 0, 2), ("b1", 2, 4), ("b2", 4, 6))):
        raw = pressure_proxy_vec(uv[s:e], cfg)
        pvec = raw * float(discounts[idx])
        blocks.append({
            "block_name": name,
            "pressure_vec": pvec,
            "pressure_vec_raw": raw,
            "pressure_norm": norm_term(pvec, cfg),
        })
    return blocks


def run_one(cfg: PlannerConfig, posture_pitch_deg: float, label: str) -> None:
    print(f"\n=== {label}: pitch={posture_pitch_deg:.2f} deg ===")

    plant_info: dict = {
        "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
        "pump_fullspeed_any": 0,
        "pump_total_backlog_kg": 0.0,
        "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        "posture_vec_deg": np.array([posture_pitch_deg, 0.0], dtype=float),
        "forecast_has_future": 1,
    }

    blocks = synthesize_lowrisk_blocks(cfg)
    print("forecast block pressure_norm:", [round(b["pressure_norm"], 3) for b in blocks])

    if cfg.posture_state_residual_active:
        adjusted_blocks, meta = apply_posture_state_to_blocks(blocks, plant_info, cfg)
        print(
            "posture_state injected: norm=%.3f raw_norm=%.3f credit=%.3f (%s)"
            % (
                meta["posture_state_norm"],
                meta["posture_state_raw_norm"],
                meta["posture_state_credit"],
                meta["posture_state_credit_reason"],
            )
        )
        # Push the same posture metadata into plant_info, mirroring what the
        # provider does at runtime.
        plant_info["posture_state_norm"] = meta["posture_state_norm"]
        plant_info["posture_state_raw_norm"] = meta["posture_state_raw_norm"]
        plant_info["posture_state_credit"] = meta["posture_state_credit"]
        plant_info["posture_state_credit_reason"] = meta["posture_state_credit_reason"]
        eval_blocks = adjusted_blocks
    else:
        eval_blocks = blocks

    print(
        "eval block pressure_norm (post-posture):",
        [round(norm_term(b["pressure_vec"], cfg), 3) for b in eval_blocks],
    )

    sequences = [
        ("hold", "hold", "hold"),
        ("pump_saving", "hold", "hold"),
        ("active_small", "hold", "hold"),
        ("active_medium", "hold", "hold"),
    ]
    rows = []
    for seq in sequences:
        result = evaluate_sequence(seq, eval_blocks, plant_info, cfg)
        if result["hard_reject_reason"]:
            rows.append((">".join(seq), float("inf"), result["hard_reject_reason"], result["costs"]))
            continue
        scalar = economic_scalar_cost(result, cfg)
        rows.append((">".join(seq), float(scalar), "", result["costs"]))

    # Print compact table.
    print(f"{'sequence':35s} {'scalar':>10s}  {'attitude':>9s} {'terminal':>9s} {'envelope':>9s} {'pump':>6s}  {'reject':s}")
    for name, scalar, reject, costs in rows:
        if reject:
            print(f"{name:35s} {'REJECT':>10s}  reason={reject}")
            continue
        print(
            f"{name:35s} {scalar:>10.3f}  {costs['attitude_residual_cost']:>9.3f} "
            f"{costs['terminal_residual_cost']:>9.3f} {costs['envelope_violation_cost']:>9.3f} "
            f"{costs['pump_work_cost']:>6.3f}"
        )

    best = min((r for r in rows if r[2] == ""), key=lambda r: r[1])
    print(f"BEST: {best[0]}  scalar={best[1]:.3f}")


def main() -> None:
    baseline_cfg = PlannerConfig()
    posture_aware_cfg = dataclasses.replace(
        baseline_cfg,
        w_attitude_residual=1.0,
        w_terminal_residual=1.0,
        posture_state_residual_active=True,
    )

    # Sanity: zero-posture should produce hold-wins in both configs.
    run_one(baseline_cfg, posture_pitch_deg=0.0, label="baseline (zero posture)")
    run_one(posture_aware_cfg, posture_pitch_deg=0.0, label="posture_aware (zero posture)")

    # The real test: large posture, low wind.
    run_one(baseline_cfg, posture_pitch_deg=4.5, label="baseline (4.5deg pitch, lowrisk)")
    run_one(posture_aware_cfg, posture_pitch_deg=4.5, label="posture_aware (4.5deg pitch, lowrisk)")


if __name__ == "__main__":
    main()
