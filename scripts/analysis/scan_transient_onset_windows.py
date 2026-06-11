#!/usr/bin/env python3
"""Scan FINO1 test split for transient-onset windows.

A "transient-onset" window is one where the *current* pressure proxy is small
(block0_norm < onset_low_threshold) but the *future* pressure proxy is large
(block2_norm > onset_high_threshold). This is the regime where a 60-min
preview planner can plausibly demonstrate predictive value: the closed-loop
controller, which is reactive only, has no access to that future information.

Outputs:
  outputs/wind_prediction/transient_onset_scan/scan_full.csv
  outputs/wind_prediction/transient_onset_scan/scan_top_onset.csv
  outputs/wind_prediction/transient_onset_scan/scan_summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from wind_prediction.ballast_planner import (
    PlannerConfig,
    compute_pressure_blocks,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset


ONSET_LOW = 0.5
ONSET_HIGH = 1.0
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def main() -> None:
    out_dir = repo_root / "outputs" / "wind_prediction" / "transient_onset_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_planner_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    sign_cfg = json.loads((base_planner_out / "diagnostics" / "a01_pressure_vec_sign_convention.json").read_text())
    discount_cfg = json.loads((base_planner_out / "diagnostics" / "a1_block_discount_config.json").read_text())

    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    discounts = discount_cfg["default_discount_blocks"]

    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )

    rows = []
    for ts in replay.iter_sample_timestamps():
        sample = replay.sample_for_history_end(ts)
        if sample is None:
            continue
        uv = np.asarray(sample.y_uv_raw, dtype=float)
        if uv.shape[0] < 6:
            continue
        try:
            blocks = compute_pressure_blocks(uv, discounts, cfg)
        except Exception:
            continue
        n0 = blocks[0]["pressure_norm"]
        n1 = blocks[1]["pressure_norm"]
        n2 = blocks[-1]["pressure_norm"]
        p0 = blocks[0]["pressure_vec"]
        p2 = blocks[-1]["pressure_vec"]
        sign_flip = int(float(np.dot(p0, p2)) < 0.0)
        # current speed (m/s) for diagnosis
        speed_now = float(np.linalg.norm(uv[1]))
        speed_block2 = float(np.mean(np.linalg.norm(uv[4:6], axis=1)))
        rows.append({
            "timestamp": ts.strftime(TIMESTAMP_FMT),
            "block0_norm": n0,
            "block1_norm": n1,
            "block2_norm": n2,
            "growth_norm": n2 - n0,
            "growth_ratio": n2 / max(n0, 1e-6),
            "sign_flip": sign_flip,
            "speed_t0": speed_now,
            "speed_block2": speed_block2,
            "speed_growth": speed_block2 - speed_now,
            "is_onset": int((n0 < ONSET_LOW) and (n2 > ONSET_HIGH)),
            "is_decay": int((n0 > ONSET_HIGH) and (n2 < ONSET_LOW)),
            "is_signflip_event": int(sign_flip and max(n0, n2) > ONSET_HIGH),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scan_full.csv", index=False)

    onset_df = df[df["is_onset"] == 1].sort_values("growth_norm", ascending=False).reset_index(drop=True)
    decay_df = df[df["is_decay"] == 1].sort_values("growth_norm", ascending=True).reset_index(drop=True)
    flip_df = df[df["is_signflip_event"] == 1].sort_values("block2_norm", ascending=False).reset_index(drop=True)

    onset_df.to_csv(out_dir / "scan_top_onset.csv", index=False)
    decay_df.to_csv(out_dir / "scan_top_decay.csv", index=False)
    flip_df.to_csv(out_dir / "scan_top_signflip.csv", index=False)

    lines = [
        "# Transient-onset scan (FINO1 test split)",
        "",
        f"- total samples scanned: `{len(df)}`",
        f"- onset windows (block0<{ONSET_LOW} AND block2>{ONSET_HIGH}): `{len(onset_df)}`",
        f"- decay windows (block0>{ONSET_HIGH} AND block2<{ONSET_LOW}): `{len(decay_df)}`",
        f"- sign-flip events (max>{ONSET_HIGH}): `{len(flip_df)}`",
        "",
        "## Top 15 onset windows (largest growth)",
        "",
        "| timestamp | block0 | block2 | growth | speed_t0 | speed_block2 | sign_flip |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in onset_df.head(15).iterrows():
        lines.append(
            f"| {r['timestamp']} | {r['block0_norm']:.3f} | {r['block2_norm']:.3f} | "
            f"{r['growth_norm']:+.3f} | {r['speed_t0']:.1f} | {r['speed_block2']:.1f} | {int(r['sign_flip'])} |"
        )
    lines += ["", "## Top 10 sign-flip events (likely direction reversal)", "",
              "| timestamp | block0 | block2 | speed_t0 | speed_block2 |",
              "|---|---|---|---|---|"]
    for _, r in flip_df.head(10).iterrows():
        lines.append(
            f"| {r['timestamp']} | {r['block0_norm']:.3f} | {r['block2_norm']:.3f} | "
            f"{r['speed_t0']:.1f} | {r['speed_block2']:.1f} |"
        )

    (out_dir / "scan_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "scan_summary.md").read_text())


if __name__ == "__main__":
    main()
