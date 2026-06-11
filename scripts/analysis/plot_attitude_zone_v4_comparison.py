#!/usr/bin/env python3
"""Visualize quadratic vs smooth_huber attitude_zone cost on 4 archetypes.

This is the figure that goes into the paper's limitations / future-work
discussion: it shows why the deadzone form fixes lowrisk over-pump but
loses onset preemption.

Per scenario, plots:
  - per-block hold-sequence cost (decomposed: attitude / terminal / envelope / pump)
  - per-block active_small-sequence cost
  - chosen first_action under each form
  - residual_norm per block under each form

Output: outputs/wind_prediction/attitude_zone_v4_figures/
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


SCENARIOS = [
    ("S1_lowrisk_clean", "lowrisk_clean: posture high, wind calm", 4.95, 1.5, 0.0),
    ("S2_fr_relief_09", "fr_relief_09: posture high, wind high decaying", 5.0, 18.0, 1.5),
    ("S3_fr01_weak", "fr01_weak: moderate posture, moderate wind", 3.0, 5.0, 0.5),
    ("S4_onset_rising", "onset_rising: low-to-medium wind RISING", 3.0, 5.0, -5.0),
]


def make_blocks(cfg, mean_wind_ms, decay):
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


def evaluate_scenario(cfg, pitch, wind, decay):
    plant = {
        "tank_masses": np.array([1108000.0, 1362000.0, 1362000.0], dtype=float),
        "pump_fullspeed_any": 0,
        "pump_total_backlog_kg": 0.0,
        "pump_rate_cmd_m3_min": np.array([0.0, 0.0, 0.0], dtype=float),
        "posture_vec_deg": np.array([pitch, 0.0], dtype=float),
        "forecast_has_future": 1,
    }
    blocks = make_blocks(cfg, wind, decay)
    adjusted, meta = apply_posture_state_to_blocks(blocks, plant, cfg)
    plant["posture_state_norm"] = meta["posture_state_norm"]
    plant["posture_state_raw_norm"] = meta["posture_state_raw_norm"]
    plant["posture_state_credit"] = meta["posture_state_credit"]
    plant["posture_state_credit_reason"] = meta["posture_state_credit_reason"]
    block_norms = [norm_term(b["pressure_vec"], cfg) for b in adjusted]

    sequences = [
        ("hold", "hold", "hold"),
        ("active_small", "hold", "hold"),
    ]
    out = {"block_norms": block_norms}
    for seq in sequences:
        r = evaluate_sequence(seq, adjusted, plant, cfg)
        key = seq[0]
        if r["hard_reject_reason"]:
            out[key] = None
            continue
        out[key] = {
            "scalar": economic_scalar_cost(r, cfg),
            "att": r["costs"]["attitude_residual_cost"],
            "term": r["costs"]["terminal_residual_cost"],
            "env": r["costs"]["envelope_violation_cost"],
            "env_term": r["costs"]["terminal_envelope_violation"],
            "pump": r["costs"]["pump_work_cost"],
            "duration": r["costs"]["pump_duration_cost"],
            "startstop": r["costs"]["startstop_cost"],
        }
    return out


def render_panel(ax, title, q_result, h_result):
    """Bar plot of hold vs active_small scalar cost under each form."""
    labels = ["quadratic\nhold", "quadratic\nactive_small", "smooth_huber\nhold", "smooth_huber\nactive_small"]
    qh = q_result.get("hold")
    qa = q_result.get("active_small")
    hh = h_result.get("hold")
    ha = h_result.get("active_small")
    values = [
        qh["scalar"] if qh else np.nan,
        qa["scalar"] if qa else np.nan,
        hh["scalar"] if hh else np.nan,
        ha["scalar"] if ha else np.nan,
    ]
    colors = ["#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    # Highlight winning bars per form.
    if qh and qa:
        winner_q = 0 if qh["scalar"] <= qa["scalar"] else 1
        bars[winner_q].set_edgecolor("black")
        bars[winner_q].set_linewidth(2.5)
    if hh and ha:
        winner_h = 2 if hh["scalar"] <= ha["scalar"] else 3
        bars[winner_h].set_edgecolor("black")
        bars[winner_h].set_linewidth(2.5)
    ax.set_ylabel("scalar cost")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def render_blocks(ax, title, q_norms, h_norms, delta):
    block_x = [0, 1, 2]
    ax.plot(block_x, q_norms, "o-", color="#1f77b4", label="block residual_norm (both forms identical, posture+pressure)")
    ax.axhline(delta, color="#ff7f0e", linestyle="--", label=f"smooth_huber zone (delta={delta})")
    ax.fill_between([-0.2, 2.2], 0, delta, color="#ff7f0e", alpha=0.15)
    ax.set_xlim(-0.2, 2.2)
    ax.set_xticks(block_x)
    ax.set_xticklabels(["block 0\n(0-20min)", "block 1\n(20-40min)", "block 2\n(40-60min)"])
    ax.set_ylabel("residual_norm (deadband units)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)


def main():
    out_dir = repo_root / "outputs" / "wind_prediction" / "attitude_zone_v4_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    quadratic_cfg = PlannerConfig()
    huber_cfg = dataclasses.replace(
        PlannerConfig(),
        attitude_zone_form="smooth_huber",
        attitude_zone_delta_norm=1.5,
        attitude_zone_terminal_delta_norm=1.5,
        w_attitude_residual=3.0,
        w_terminal_residual=3.0,
    )

    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    fig.suptitle(
        "attitude_zone_form: quadratic (default) vs smooth_huber (default-off)\n"
        "Hold-vs-active first-action cost on 4 archetypal scenarios",
        fontsize=12,
    )

    for row, (scenario_id, scenario_label, pitch, wind, decay) in enumerate(SCENARIOS):
        q_result = evaluate_scenario(quadratic_cfg, pitch, wind, decay)
        h_result = evaluate_scenario(huber_cfg, pitch, wind, decay)
        ax_left = axes[row, 0]
        ax_right = axes[row, 1]
        render_panel(ax_left, f"{scenario_id}  scalar cost", q_result, h_result)
        # block_norms identical between forms (posture+pressure rollout is the same)
        render_blocks(ax_right, f"{scenario_id}  block residual_norm",
                      q_result["block_norms"], h_result["block_norms"], huber_cfg.attitude_zone_delta_norm)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = out_dir / "attitude_zone_v4_quad_vs_huber_S1_S4.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
