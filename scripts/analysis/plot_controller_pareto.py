#!/usr/bin/env python3
"""Build Pareto plots across all controllers and cases.

This is a methodology-correction tool, not a new mechanism. It scans
outputs/wind_prediction for any 2h closed-loop timeseries on the canonical
cases (fr_relief_01/09, b_high_pressure_event, lowrisk_clean, lowrisk_random_03,
sf_holdout_02), classifies the controller by output-directory naming, and
plots pump vs pitch_p95 with fallback shown by color.

Intent: replace "candidate vs single baseline" verdicts with a Pareto-frontier
view that exposes which controllers actually dominate others.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASES = [
    'fr_relief_01', 'fr_relief_09', 'b_high_pressure_event',
    'lowrisk_clean', 'lowrisk_random_03', 'sf_holdout_02',
]


def classify(path: str, fname: str) -> str | None:
    """Map a timeseries-file path to a controller label."""
    p = path + '/' + fname
    if 'reactive_current_vs_learned_2case_2h_v1' in path and 'closed_only' in fname:
        return 'closed_only'
    if 'closed_only_baseline' in path:
        return 'closed_only'
    if 'reactive_current_vs_learned_2case_2h_v1' in path and 'prediction_primary_learned' in fname:
        return 'pred_primary_learned_old'
    if 'reactive' in path and 'reactive' in fname:
        return 'reactive_current'
    if 'current_only_reactive_primary_5case_2h' in path:
        return 'current_only_reactive'
    if 'no_unexplained_hold_v2a_current5_baseline' in path:
        return 'baseline_v2a_current5'
    if 'no_unexplained_hold_v2a_current5_candidate' in path:
        return 'no_unexplained_hold_v2a'
    if 'objective_mode_v3_prototype_5case_2h' in path:
        return 'obj_mode_v3'
    if 'hold_relief_debt_guard10_candidate' in path:
        return 'hold_relief_debt_v1'
    if 'active_intent_reproposal' in path:
        return 'active_intent_reproposal'
    return None


def compute_metrics(ts: pd.DataFrame) -> dict:
    pitch = ts['pitch_deg'].abs() if 'pitch_deg' in ts.columns else pd.Series([0.0])
    roll = ts['roll_deg'].abs() if 'roll_deg' in ts.columns else pd.Series([0.0])
    pump = (ts['pump_total_rate_m3_min'].abs() / 60).sum() if 'pump_total_rate_m3_min' in ts.columns else float('nan')
    fb_col = next((c for c in ts.columns if 'safety_fallback' in c.lower() or 'fallback_active' in c.lower()), None)
    fb = ts[fb_col].mean() if fb_col else 0.0
    t_over_3 = int(((pitch > 3) | (roll > 3)).sum())
    return {
        'pump_m3': float(pump),
        'pitch_p95': float(np.percentile(pitch, 95)),
        'pitch_max': float(pitch.max()),
        'roll_p95': float(np.percentile(roll, 95)),
        'fallback': float(fb),
        't_over_3_s': t_over_3,
    }


def collect():
    rows = []
    seen = set()
    for root, _, files in os.walk('outputs/wind_prediction'):
        if any(p in root for p in ['/figures', '/_eval', 'attitude_recovery_metrics', 'planner_logs']):
            continue
        for f in files:
            if not f.endswith('timeseries.csv'):
                continue
            case = next((c for c in CASES if c in f), None)
            if case is None:
                continue
            ctrl = classify(root, f)
            if ctrl is None:
                continue
            key = (case, ctrl)
            if key in seen:
                continue
            try:
                ts = pd.read_csv(os.path.join(root, f), low_memory=False)
            except Exception:
                continue
            m = compute_metrics(ts)
            m['case'] = case
            m['ctrl'] = ctrl
            rows.append(m)
            seen.add(key)
    return pd.DataFrame(rows)


def plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    color_map = {
        'closed_only': 'tab:blue',
        'reactive_current': 'tab:green',
        'current_only_reactive': 'tab:olive',
        'pred_primary_learned_old': 'tab:purple',
        'baseline_v2a_current5': 'tab:red',
        'no_unexplained_hold_v2a': 'tab:pink',
        'obj_mode_v3': 'tab:orange',
        'hold_relief_debt_v1': 'tab:brown',
        'active_intent_reproposal': 'tab:gray',
    }
    cases_with_data = df['case'].unique()
    n = len(cases_with_data)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 4 * ((n + 1) // 2)))
    axes = np.array(axes).flatten()
    for i, case in enumerate(sorted(cases_with_data)):
        ax = axes[i]
        sub = df[df.case == case].sort_values('pump_m3')
        for _, r in sub.iterrows():
            color = color_map.get(r.ctrl, 'black')
            # Marker size encodes fallback (higher fallback = bigger circle)
            size = 100 + 1000 * r.fallback
            ax.scatter(r.pump_m3, r.pitch_p95, c=color, s=size, edgecolors='black', linewidths=0.5, alpha=0.75, label=r.ctrl)
            ax.annotate(r.ctrl, (r.pump_m3, r.pitch_p95), fontsize=7,
                        xytext=(5, 3), textcoords='offset points')
        ax.set_xlabel('pump m³')
        ax.set_ylabel('pitch p95 °')
        ax.set_title(case)
        ax.grid(alpha=0.3)
        ax.axhline(3, color='gray', linestyle=':', linewidth=0.8, label='3° attention')
        ax.axhline(5, color='red', linestyle=':', linewidth=0.8, label='5° IEC')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Pareto map: pump cost vs pitch_p95 across controllers\n(circle size = fallback ratio)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = out_dir / 'controller_pareto_per_case.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'wrote {out}')


def main():
    df = collect()
    if df.empty:
        print('No timeseries found.')
        return
    out_dir = Path('outputs/wind_prediction/controller_pareto_review')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'controller_pareto_table.csv', index=False)
    print(df.sort_values(['case', 'pump_m3']).to_string(index=False))
    plot_pareto(df, out_dir)


if __name__ == '__main__':
    main()
