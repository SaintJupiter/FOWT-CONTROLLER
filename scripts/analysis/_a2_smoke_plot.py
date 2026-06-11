#!/usr/bin/env python3
"""A2-smoke comparison figure (3 modes: closed_only / hold / event_decay)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
out_dir = repo_root / "outputs" / "wind_prediction" / "planner_a2_smoke"

WINDOWS = [
    ("2023-09-11_034000", "clean_low_pressure_normal", "2023-09-11 03:40"),
    ("2024-09-05_181000", "residual_high_normal",     "2024-09-05 18:10"),
    ("2022-02-04_110000", "high_pressure_high_event", "2022-02-04 11:00"),
]
MODE_STYLE = [
    ("closed_only",                       "closed_only",   "#1f77b4", "-"),
    ("closed_plus_planner_hold",          "planner (hold)", "#d62728", "-"),
    ("closed_plus_planner_event_decay",   "planner (event-decay)", "#2ca02c", "-"),
]


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    for i, (slug, group, ts) in enumerate(WINDOWS):
        ax = axes[i]
        bias_ax = ax.twinx()
        for mode_tag, label, color, ls in MODE_STYLE:
            path = out_dir / f"{slug}_{mode_tag}_timeseries.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            t_col = "t_s" if "t_s" in df.columns else ("t_sim_s" if "t_sim_s" in df.columns else None)
            t_min = df[t_col].astype(float).to_numpy() / 60.0 if t_col else (df.index.values / 60.0)
            ax.plot(t_min, df["pitch_deg"], label=label, color=color, lw=0.85, alpha=0.85, ls=ls)
            if mode_tag == "closed_plus_planner_event_decay" and "preview_pitch_bias_deg" in df.columns:
                bias_ax.plot(t_min, df["preview_pitch_bias_deg"], color="#999999",
                             lw=1.0, ls="--", alpha=0.7, label="event-decay bias")
        bias_ax.set_ylabel("pitch bias (deg)", color="#666666", fontsize=9)
        bias_ax.tick_params(axis='y', labelcolor="#666666")
        bias_ax.set_ylim(-0.5, 0.5)

        ax.set_title(f"{ts}  ·  {group}", fontsize=10)
        ax.set_ylabel("pitch (deg)", fontsize=9)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        bias_ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (min)")
    fig.suptitle("A2-smoke pitch trace: closed_only vs planner-hold vs planner-event-decay",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path = out_dir / "A2_smoke_pitch_compare.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
