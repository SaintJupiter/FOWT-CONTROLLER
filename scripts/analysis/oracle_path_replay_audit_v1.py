#!/usr/bin/env python3
"""Stage-2 pre-replay audit: emit oracle min-throughput tank paths and check
whether the V1.1 ceiling is likely to survive a real-plant replay.

It reuses the V1.1 LP solver (now returning the tank path) and computes, per case,
the dynamic-gap risk proxies:

- extrapolation: does the oracle tank trajectory stay within the observed
  (deadband vs closed) tank-delta range, where the fitted linear posture model is
  trustworthy?  Out-of-range => the static model is being extrapolated.
- rate utilisation: how close the oracle per-step pump increment runs to the rate
  cap (high => the path is aggressive and dynamics may bite).
- effective vs raw saving: tail-to-closed cost (large => saving is partly a terminal
  deferral, not real).
- terminal closure error.

Output: per-case audit CSV + emitted m_path CSVs for the future real replay.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import offline_min_throughput_oracle_v1 as O


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--resample-s", type=float, default=60.0)
    ap.add_argument("--terminal-tank-tol-kg", type=float, default=5000.0)
    ap.add_argument("--posture-limit-deg", type=float, default=3.0)
    ap.add_argument("--posture-margin-deg", type=float, default=0.0)
    ap.add_argument("--posture-bound-mode", type=str, default="deadband_envelope")
    ap.add_argument("--terminal-reference", type=str, default="closed")
    ap.add_argument("--pump-rate-limit-m3-min", type=float, default=8.0)
    ap.add_argument("--primary-label-substring", type=str, default="")
    ap.add_argument("--max-cases", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "paths").mkdir(exist_ok=True)
    pairs = O._load_case_pairs(args.run_dir, args.max_cases, args.primary_label_substring)

    rows = []
    for pair in pairs:
        closed = O._read_timeseries(pair.closed_path)
        primary = O._read_timeseries(pair.primary_path)
        closed_rs, primary_rs = O._paired_resampled(closed, primary, resample_s=args.resample_s)
        fit = O._fit_response_model(
            closed_rs, primary_rs, source="per_case",
            max_condition=1e6, min_rank=2, min_samples=8,
        )
        lp = O._solve_lp_oracle(
            closed_rs=closed_rs, primary_rs=primary_rs, fit=fit,
            posture_limit_deg=args.posture_limit_deg,
            posture_bound_mode=args.posture_bound_mode,
            posture_margin_deg=args.posture_margin_deg,
            terminal_tank_tol_kg=args.terminal_tank_tol_kg,
            terminal_reference=args.terminal_reference,
            tank_capacity_kg=O.DEFAULT_TANK_CAPACITY_KG,
            pump_rate_limit_m3_min=args.pump_rate_limit_m3_min,
            rho_kg_m3=O.RHO_KG_M3,
        )
        if lp.m_path is None or not (isinstance(lp.status, str) and lp.status.startswith("0")):
            rows.append({"case_id": pair.case_id, "status": lp.status, "usable": False})
            continue
        m = np.asarray(lp.m_path, dtype=float)
        mc = np.asarray(lp.m_closed_path, dtype=float)
        mp = primary_rs[list(O.TANK_COLS)].to_numpy(dtype=float)
        t = np.asarray(lp.t_path, dtype=float)
        # observed delta range (deadband - closed) per tank
        obs_delta = mp - mc
        orc_delta = m - mc
        obs_lo = obs_delta.min(axis=0); obs_hi = obs_delta.max(axis=0)
        # extrapolation fraction: rows where oracle delta is outside observed [lo,hi] on any tank
        below = orc_delta < (obs_lo - 1.0)
        above = orc_delta > (obs_hi + 1.0)
        extrap = np.any(below | above, axis=1)
        extrap_frac = float(np.mean(extrap))
        # how far outside (kg), worst
        outside = np.maximum(0.0, np.maximum(obs_lo - orc_delta, orc_delta - obs_hi))
        max_extrap_kg = float(np.max(outside))
        # rate utilisation: per-step |delta m| vs cap
        dt = np.diff(t)
        step_cap_kg = args.pump_rate_limit_m3_min * O.RHO_KG_M3 * dt / 60.0
        step_move = np.sum(np.abs(np.diff(m, axis=0)), axis=1)  # total kg moved per step
        rate_util = float(np.max(step_move / np.maximum(step_cap_kg * 3, 1e-9)))  # 3 tanks
        # save path
        pdf = pd.DataFrame({"t_s": t,
                            "m1_kg": m[:, 0], "m2_kg": m[:, 1], "m3_kg": m[:, 2],
                            "mc1_kg": mc[:, 0], "mc2_kg": mc[:, 1], "mc3_kg": mc[:, 2]})
        pdf.to_csv(args.out_dir / "paths" / f"{pair.case_id}_oracle_path.csv", index=False)
        rows.append({
            "case_id": pair.case_id, "status": lp.status, "usable": True,
            "fit_rmse_pitch": fit.rmse_pitch, "fit_rmse_roll": fit.rmse_roll,
            "oracle_pump_m3": lp.pump_m3, "oracle_effective_pump_m3": lp.effective_pump_m3,
            "tail_to_closed_m3": lp.tail_to_closed_m3,
            "extrap_frac": extrap_frac, "max_extrap_kg": max_extrap_kg,
            "rate_util_max": rate_util,
            "terminal_error_kg": lp.terminal_error_kg,
            "lp_posture_violation_max_deg": lp.posture_violation_max_deg,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "oracle_path_audit.csv", index=False)
    ok = df[df.get("usable", False) == True] if "usable" in df.columns else df
    print(f"cases={len(df)} usable={len(ok)}")
    if len(ok):
        cap = O.RHO_KG_M3  # for kg->m3 context
        print(f"extrap_frac:    mean={ok['extrap_frac'].mean():.2f} max={ok['extrap_frac'].max():.2f}  (low=trustworthy, oracle stays in observed range)")
        print(f"max_extrap_kg:  mean={ok['max_extrap_kg'].mean():.0f} max={ok['max_extrap_kg'].max():.0f}  (= {ok['max_extrap_kg'].max()/cap:.1f} m3 worst extrapolation)")
        print(f"rate_util_max:  mean={ok['rate_util_max'].mean():.2f} max={ok['rate_util_max'].max():.2f}  (<1 = within pump rate cap)")
        teff = (ok['tail_to_closed_m3'] / ok['oracle_effective_pump_m3']).replace([np.inf,-np.inf], np.nan)
        print(f"tail/effective: mean={teff.mean():.3f} max={teff.max():.3f}  (low=saving is real, not terminal deferral)")
        print(f"terminal_error: mean={ok['terminal_error_kg'].mean():.0f}kg max={ok['terminal_error_kg'].max():.0f}kg (tol={args.terminal_tank_tol_kg:.0f})")
        print(f"posture_viol:   max={ok['lp_posture_violation_max_deg'].max():.3f}deg")
        # verdict
        trust = (ok['extrap_frac'].max() < 0.25) and (ok['rate_util_max'].max() < 1.05) and (teff.max() < 0.2)
        print(f"\nPRE-REPLAY VERDICT: {'LIKELY-SURVIVES (build the real replay)' if trust else 'FRAGILE (oracle extrapolates / aggressive / terminal-deferred -> replay likely to shrink the ceiling)'}")


if __name__ == "__main__":
    main()
