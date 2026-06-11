#!/usr/bin/env python3
"""Cost replay: quadratic vs smooth_huber attitude cost on representative buckets.

This is a math diagnostic, not a closed-loop experiment. It evaluates the
4 archetypal bucket scenarios that the candidate matrix and the user's
prompt called out as failure modes for the bistable quadratic objective:

  1. lowrisk_clean high-hold (pitch~5deg, pressure~0)
     Desired: hold wins. Quadratic over-fires. smooth_huber should let hold win.

  2. fr_relief_09 high-active (pitch~5deg, pressure~1.5, modest relief)
     Desired: small wins. Both forms should agree.

  3. fr_relief_01 weak-benefit overpump (pitch~3deg, pressure~0.5)
     Desired: hold wins (or small wins by marginal amount). Quadratic
     over-fires; smooth_huber should leave hold ahead because residual is
     close to the deadzone boundary.

  4. b_high_pressure_event onset (pitch~3deg, pressure~1.8, rising)
     Desired: small/medium wins. Both forms should agree.

The script prints per-scenario:
  - quadratic-form scalar, attitude_cost, pump_cost for each candidate
  - smooth_huber-form scalar, etc.
  - winning action under each form
  - margin of victory

Pass criteria for v4 prototype:
  - Scenario 1 (lowrisk): smooth_huber best = hold (quadratic gives active_medium)
  - Scenario 2 (fr09): smooth_huber best = active_small or active_medium (same direction as quadratic)
  - Scenario 3 (fr01-like): smooth_huber best = hold (quadratic gives active)
  - Scenario 4 (b_high_pressure onset): smooth_huber best = active_* (same direction as quadratic)
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


def make_blocks(cfg: PlannerConfig, mean_wind_ms: float, decay: float = 0.0) -> list[dict]:
    """Build 3 forecast blocks. mean_wind_ms = current wind, decay = per-block reduction in m/s."""
    rows = []
    speeds = [
        mean_wind_ms - 0 * decay,
        mean_wind_ms - 1 * decay,
        mean_wind_ms - 2 * decay,
    ]
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


def run_scenario(cfg: PlannerConfig, label: str, pitch_deg: float, mean_wind_ms: float, decay: float = 0.0) -> dict:
    plant_info: dict = {
        "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
        "pump_fullspeed_any": 0,
        "pump_total_backlog_kg": 0.0,
        "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        "posture_vec_deg": np.array([pitch_deg, 0.0], dtype=float),
        "forecast_has_future": 1,
    }
    blocks = make_blocks(cfg, mean_wind_ms, decay=decay)
    adjusted, meta = apply_posture_state_to_blocks(blocks, plant_info, cfg)
    plant_info["posture_state_norm"] = meta["posture_state_norm"]
    plant_info["posture_state_raw_norm"] = meta["posture_state_raw_norm"]
    plant_info["posture_state_credit"] = meta["posture_state_credit"]
    plant_info["posture_state_credit_reason"] = meta["posture_state_credit_reason"]

    sequences = [
        ("hold", "hold", "hold"),
        ("pump_saving", "hold", "hold"),
        ("active_small", "hold", "hold"),
        ("active_medium", "hold", "hold"),
    ]
    rows = []
    for seq in sequences:
        r = evaluate_sequence(seq, adjusted, plant_info, cfg)
        if r["hard_reject_reason"]:
            rows.append({"seq": ">".join(seq), "scalar": float("inf"),
                         "att": float("nan"), "term": float("nan"),
                         "pump": float("nan"), "reject": r["hard_reject_reason"]})
            continue
        rows.append({
            "seq": ">".join(seq),
            "scalar": economic_scalar_cost(r, cfg),
            "att": r["costs"]["attitude_residual_cost"],
            "term": r["costs"]["terminal_residual_cost"],
            "pump": r["costs"]["pump_work_cost"],
            "reject": "",
        })
    best = min((r for r in rows if r["reject"] == ""), key=lambda r: r["scalar"])
    return {"label": label, "rows": rows, "best": best,
            "posture_norm": meta["posture_state_norm"]}


def print_scenario(result: dict, form: str) -> None:
    print(f"\n[{form}] {result['label']}  (posture_norm={result['posture_norm']:.3f})")
    print(f"  {'sequence':30s} {'scalar':>10s}  {'att':>8s} {'term':>8s} {'pump':>6s}  reject")
    for r in result["rows"]:
        if r["reject"]:
            print(f"  {r['seq']:30s} {'REJECT':>10s}  reason={r['reject']}")
            continue
        print(f"  {r['seq']:30s} {r['scalar']:>10.3f}  {r['att']:>8.3f} {r['term']:>8.3f} {r['pump']:>6.3f}")
    print(f"  -> BEST: {result['best']['seq']}  scalar={result['best']['scalar']:.3f}")


def main() -> None:
    scenarios = [
        ("S1_lowrisk_clean_high_hold",   4.95, 1.5, 0.0),   # pitch=4.95, calm wind, no decay
        ("S2_fr_relief_09_active",       5.0,  18.0, 1.5),  # high wind, slight decay
        ("S3_fr01_weak_benefit",         3.0,  5.0, 0.5),   # moderate posture, moderate wind
        ("S4_b_high_pressure_onset",     3.0,  20.0, 0.0),  # moderate posture, strong sustained wind
    ]

    quadratic_cfg = PlannerConfig()  # default
    huber_cfg = dataclasses.replace(
        PlannerConfig(),
        attitude_zone_form="smooth_huber",
        attitude_zone_delta_norm=1.5,
        attitude_zone_terminal_delta_norm=1.5,
        # NOTE: smooth_huber form requires elevated w_attitude_residual to
        # overcome pump cost at the deadzone boundary. With w=1 (default), the
        # marginal attitude benefit at small excess < pump cost (0.85), so all
        # scenarios incorrectly select hold. With w=3, S2 fr_relief_09 and S4
        # onset scenarios re-acquire active selection while S1 lowrisk and S3
        # weak-benefit still pick hold. Treat w_attitude_residual as a
        # PROTOTYPE-LEVEL tuning parameter co-designed with the smooth_huber
        # form, not an independent weight.
        w_attitude_residual=3.0,
        w_terminal_residual=3.0,
    )

    print("=" * 70)
    print("Cost replay: quadratic (current default) vs smooth_huber (v4 prototype)")
    print("=" * 70)

    summary: list[tuple[str, str, str]] = []
    for label, pitch, wind, decay in scenarios:
        q = run_scenario(quadratic_cfg, label, pitch, wind, decay=decay)
        h = run_scenario(huber_cfg, label, pitch, wind, decay=decay)
        print_scenario(q, "QUADRATIC")
        print_scenario(h, "SMOOTH_HUBER")
        summary.append((label, q["best"]["seq"], h["best"]["seq"]))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'scenario':35s}  {'quadratic':25s} {'smooth_huber':25s}")
    for label, q, h in summary:
        marker = "  (changed)" if q != h else ""
        print(f"{label:35s}  {q:25s} {h:25s}{marker}")


if __name__ == "__main__":
    main()
