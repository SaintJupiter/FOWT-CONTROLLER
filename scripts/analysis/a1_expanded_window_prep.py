#!/usr/bin/env python3
"""A1-expanded window preparation.

Reads the existing 14 evaluation windows and:

  1. Re-labels them using a finer a1_group taxonomy:
       clean_low_pressure_normal  — low event, active_small qualifies at block-0 (small_term ≤ 1.0)
       residual_high_normal       — low event, but active_small does NOT qualify at block-0
       high_pressure_high_event   — unchanged
       combined_variability       — unchanged

     The split boundary (small_term ≤ 1.0) is derived from the A1.4 gate: norm_term
     uses the deadband as the unit, so ≤ 1.0 means "within one deadband unit".
     This matches the meaning of Rule B in the A1.4 necessity gate.

  2. Outputs the re-labelled window preview:
       outputs/.../planner_a1_dryrun/window_selection_expanded_preview.csv

  3. Outputs a per-block diagnostic for the 3 windows that are interesting:
     - 2023-03-22 (residual_high_normal, active_medium first action in A1.4)
     - 2024-09-05 (residual_high_normal, active_medium first action, pressure = 1.5 — same as high_pressure 2022-02-04)
     - 2022-02-04 (high_pressure_high_event, active_small first action in A1.4, block-0 residual equal to 2024-09-05)
       outputs/.../planner_a1_dryrun/diagnostics/a1_expanded_residual_comparison.csv

No planner logic is changed; no gate is touched.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import sys

import numpy as np
import pandas as pd


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
BLOCKS = (
    ("block1_0_20m", 0, 2),
    ("block2_20_40m", 2, 4),
    ("block3_40_60m", 4, 6),
)
FOCUS_WINDOWS = {
    "2023-03-22 00:20:00",
    "2024-09-05 18:10:00",
    "2022-02-04 11:00:00",
}


@dataclass(frozen=True)
class PlannerConfig:
    deadband_pitch_deg: float = 1.0
    deadband_roll_deg: float = 0.8
    pressure_sign_multiplier: float = -1.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_term(vec: np.ndarray, cfg: PlannerConfig) -> float:
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    return float(np.linalg.norm(vec / np.maximum(deadband, 1e-6)))


def pressure_proxy_vec(uv_block: np.ndarray, cfg: PlannerConfig, wind_ref: float = 12.0) -> np.ndarray:
    speed = np.sqrt(uv_block[:, 0] ** 2 + uv_block[:, 1] ** 2)
    mean_u = float(np.mean(uv_block[:, 0]))
    mean_v = float(np.mean(uv_block[:, 1]))
    mean_speed = float(np.mean(speed))
    mean_dir = float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
    mag = float(np.clip((max(mean_speed, 0.0) / max(wind_ref, 1.0)) ** 2, 0.0, 1.5))
    wd_rad = math.radians(mean_dir)
    raw = np.array(
        [
            -cfg.deadband_pitch_deg * mag * math.cos(wd_rad),
            cfg.deadband_roll_deg * mag * math.sin(wd_rad),
        ],
        dtype=float,
    )
    return cfg.pressure_sign_multiplier * raw


def action_vec_small(pressure_vec: np.ndarray, cfg: PlannerConfig, ratio: float) -> np.ndarray:
    deadband = np.array([cfg.deadband_pitch_deg, cfg.deadband_roll_deg], dtype=float)
    n = max(float(np.linalg.norm(pressure_vec)), 1e-9)
    return (pressure_vec / n) * deadband * ratio


def block_stats(uv_block: np.ndarray) -> dict[str, float]:
    speed = np.sqrt(uv_block[:, 0] ** 2 + uv_block[:, 1] ** 2)
    dirs = (np.rad2deg(np.arctan2(-uv_block[:, 0], -uv_block[:, 1])) + 360.0) % 360.0
    return {
        "wind_speed_mean": float(np.mean(speed)),
        "wind_speed_std": float(np.std(speed)),
        "wind_dir_mean": float(dirs.mean()),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "src"))
    from wind_prediction.replay_dataset import Fino1ReplayDataset

    base_out = repo_root / "outputs" / "wind_prediction" / "planner_a1_dryrun"
    diag_dir = base_out / "diagnostics"

    windows = pd.read_csv(base_out / "window_selection_preview.csv")
    sign_cfg = load_json(base_out / "diagnostics" / "a01_pressure_vec_sign_convention.json")
    discount_cfg = load_json(base_out / "diagnostics" / "a1_block_discount_config.json")
    a1_4_diag = pd.read_csv(diag_dir / "a1_4_medium_term_diagnostics.csv")
    a1_4_default = a1_4_diag[a1_4_diag["discount_profile"] == "default_discount"].copy()

    cfg = PlannerConfig(pressure_sign_multiplier=float(sign_cfg["planner_sign_multiplier"]))
    replay = Fino1ReplayDataset(
        dataset_dir=repo_root / "data" / "processed" / "wind_ml_10min" / "ballast_decision_fino1_meteo_aux_v1",
        split="test",
    )
    discounts = discount_cfg["default_discount_blocks"]

    # ------------------------------------------------------------------
    # 1. Re-label windows
    # ------------------------------------------------------------------
    diag_lookup = a1_4_default.set_index("prediction_timestamp")
    new_rows = []
    for _, win in windows.iterrows():
        ts = str(win["prediction_timestamp"])
        orig_group = str(win["a1_group"])
        new_group = orig_group
        small_term_b0 = np.nan
        if orig_group == "low_pressure_normal" and ts in diag_lookup.index:
            small_term_b0 = float(diag_lookup.loc[ts, "block0_small_term"])
            new_group = "clean_low_pressure_normal" if small_term_b0 <= 1.0 else "residual_high_normal"
        row = win.to_dict()
        row["a1_group_orig"] = orig_group
        row["a1_group"] = new_group
        row["block0_small_term"] = small_term_b0
        new_rows.append(row)
    expanded = pd.DataFrame(new_rows)
    expanded.to_csv(base_out / "window_selection_expanded_preview.csv", index=False)

    # Group count summary
    print("\n=== Window group counts (expanded) ===")
    print(expanded["a1_group"].value_counts().to_string())

    # ------------------------------------------------------------------
    # 2. Per-block diagnostic for the 3 focus windows
    # ------------------------------------------------------------------
    comparison_rows = []
    for _, win in windows.iterrows():
        ts = str(win["prediction_timestamp"])
        if ts not in FOCUS_WINDOWS:
            continue
        sample = replay.sample_for_history_end(datetime.strptime(ts, TIMESTAMP_FMT))
        if sample is None:
            print(f"  WARNING: no sample for {ts}")
            continue
        uv = np.asarray(sample.y_uv_raw, dtype=float)
        orig_group = str(win["a1_group"])

        block_rows: list[dict[str, Any]] = []
        for idx, (bname, bstart, bend) in enumerate(BLOCKS):
            uv_b = uv[bstart:bend]
            pvec = pressure_proxy_vec(uv_b, cfg) * float(discounts[idx])
            p_norm = norm_term(pvec, cfg)
            small_term_b = norm_term(pvec - action_vec_small(pvec, cfg, 0.15), cfg)
            med_term_b = norm_term(pvec - action_vec_small(pvec, cfg, 0.35), cfg)
            stats = block_stats(uv_b)
            block_rows.append(
                {
                    "block": bname,
                    "pressure_norm": round(p_norm, 4),
                    "small_term": round(small_term_b, 4),
                    "med_term": round(med_term_b, 4),
                    **{k: round(v, 4) for k, v in stats.items()},
                }
            )

        # block trend: speed change = block3 - block1
        spd_b1 = block_rows[0]["wind_speed_mean"]
        spd_b3 = block_rows[2]["wind_speed_mean"]
        dir_b1 = block_rows[0]["wind_dir_mean"]
        dir_b3 = block_rows[2]["wind_dir_mean"]
        dir_delta = (dir_b3 - dir_b1 + 180) % 360 - 180  # signed [-180, 180]

        for b in block_rows:
            comparison_rows.append(
                {
                    "prediction_timestamp": ts,
                    "a1_group_orig": orig_group,
                    "block0_small_term_gate": float(diag_lookup.loc[ts, "block0_small_term"]) if ts in diag_lookup.index else np.nan,
                    **b,
                    "wind_speed_change_b3_minus_b1": round(spd_b3 - spd_b1, 4),
                    "wind_dir_change_b3_minus_b1_deg": round(dir_delta, 1),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    out_path = diag_dir / "a1_expanded_residual_comparison.csv"
    comparison_df.to_csv(out_path, index=False)

    # Pretty print
    print("\n=== Per-block diagnostic (3 focus windows) ===")
    cols_show = [
        "prediction_timestamp", "a1_group_orig", "block",
        "pressure_norm", "small_term", "med_term",
        "wind_speed_mean", "wind_speed_change_b3_minus_b1", "wind_dir_change_b3_minus_b1_deg",
    ]
    print(comparison_df[cols_show].to_string(index=False))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
