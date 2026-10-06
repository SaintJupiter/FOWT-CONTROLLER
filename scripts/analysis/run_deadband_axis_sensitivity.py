#!/usr/bin/env python3
"""Planner-level pitch/roll deadband sensitivity scan.

This is intentionally lighter than a full closed-loop platform simulation. It
replays FINO1 forecast windows through the finite-candidate ballast planner and
compares how pitch/roll deadband choices change planner decisions.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from functools import cmp_to_key
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from wind_prediction import Fino1ReplayDataset
from wind_prediction.ballast_planner import (
    ACTIONS,
    DEFAULT_PLANT_INFO,
    PlannerConfig,
    apply_posture_state_to_blocks,
    compare_sequences,
    compute_pressure_blocks,
    evaluate_sequence,
    planner_action_vec,
    tank_signal,
)

DEFAULT_DATASET = ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_h240_f120_v1"
DEFAULT_OUT = ROOT / "outputs/deadband_axis_sensitivity_20260622"
DISCOUNTS = [1.0, 0.85, 0.70]
CONFIGS = [
    ("baseline_split_1p0_0p8", 1.0, 0.8),
    ("equal_1p0_1p0", 1.0, 1.0),
    ("equal_tight_0p8_0p8", 0.8, 0.8),
    ("equal_loose_1p2_1p2", 1.2, 1.2),
    ("split_loose_1p2_1p0", 1.2, 1.0),
    ("split_inverse_0p8_1p0", 0.8, 1.0),
]
POSTURE_CASES = [
    ("forecast_only_zero_posture", 0.0, 0.0),
    ("pitch_1p5deg", 1.5, 0.0),
    ("roll_1p5deg", 0.0, 1.5),
    ("pitch_roll_1p0deg", 1.0, 1.0),
    ("pitch2_roll1deg", 2.0, 1.0),
    ("pitch1_roll2deg", 1.0, 2.0),
]


def _sequence_space() -> list[tuple[str, str, str]]:
    return list(product(ACTIONS, repeat=3))


ALL_SEQUENCES = _sequence_space()


def _select_indices(replay: Fino1ReplayDataset, max_samples: int) -> list[int]:
    event_cols = list(replay.event_columns)
    attention_idx = event_cols.index("ballast_attention_event")
    y_event = replay._event_values()  # noqa: SLF001 - analysis script, avoids materializing all samples.
    positive = [idx for idx, row in enumerate(y_event) if float(row[attention_idx]) >= 0.5]
    neutral = [idx for idx, row in enumerate(y_event) if float(row[attention_idx]) < 0.5]
    half = max_samples // 2
    selected = positive[:half] + neutral[: max_samples - half]
    return sorted(selected)


def _plant_info_for_posture(pitch_deg: float, roll_deg: float) -> dict[str, Any]:
    plant = {
        key: np.array(value, dtype=float, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in DEFAULT_PLANT_INFO.items()
    }
    plant["posture_vec_deg"] = np.array([float(pitch_deg), float(roll_deg)], dtype=float)
    plant["forecast_has_future"] = 1
    return plant


def _best_for_sample(sample: Any, cfg: PlannerConfig, plant_info: dict[str, Any]) -> dict[str, Any]:
    uv = np.asarray(sample.y_uv_raw, dtype=float)
    blocks = compute_pressure_blocks(uv, DISCOUNTS, cfg)
    if "posture_vec_deg" in plant_info:
        blocks, meta = apply_posture_state_to_blocks(blocks, plant_info, cfg)
        plant_info = {**plant_info, **meta}
    evaluated = [evaluate_sequence(seq, blocks, plant_info, cfg) for seq in ALL_SEQUENCES]
    feasible = sorted(
        [row for row in evaluated if not row["hard_reject_reason"]],
        key=cmp_to_key(lambda a, b: compare_sequences(a, b, cfg)[0]),
    )
    if not feasible:
        return {
            "feasible_count": 0,
            "best_sequence": "",
            "first_action": "none",
            "first_delta_mean_kg": np.nan,
            "first_delta_abs_sum_kg": np.nan,
            "first_avec_pitch": np.nan,
            "first_avec_roll": np.nan,
            "first_tank_delta_1_kg": np.nan,
            "first_tank_delta_2_kg": np.nan,
            "first_tank_delta_3_kg": np.nan,
            "pressure_norm_b1": float(blocks[0]["pressure_norm"]),
            "pressure_norm_b2": float(blocks[1]["pressure_norm"]),
            "pressure_norm_b3": float(blocks[2]["pressure_norm"]),
        }
    best = feasible[0]
    first_action = best["sequence"][0]
    first_avec, _ = planner_action_vec(first_action, 0, blocks, plant_info, cfg)
    first_delta = tank_signal(first_avec, cfg) * cfg.action_mass_quantum_kg
    costs = {f"cost_{key}": float(value) for key, value in best["costs"].items()}
    return {
        "feasible_count": len(feasible),
        "best_sequence": ">".join(best["sequence"]),
        "first_action": first_action,
        "first_delta_mean_kg": float(np.mean(np.abs(first_delta))),
        "first_delta_abs_sum_kg": float(np.sum(np.abs(first_delta))),
        "first_avec_pitch": float(first_avec[0]),
        "first_avec_roll": float(first_avec[1]),
        "first_tank_delta_1_kg": float(first_delta[0]),
        "first_tank_delta_2_kg": float(first_delta[1]),
        "first_tank_delta_3_kg": float(first_delta[2]),
        "pressure_norm_b1": float(blocks[0]["pressure_norm"]),
        "pressure_norm_b2": float(blocks[1]["pressure_norm"]),
        "pressure_norm_b3": float(blocks[2]["pressure_norm"]),
        **costs,
    }


def _posture_probe_rows() -> list[dict[str, Any]]:
    poses = [
        ("same_0p8deg", 0.8, 0.8),
        ("same_0p9deg", 0.9, 0.9),
        ("pitch2_roll0p8", 2.0, 0.8),
        ("pitch0p8_roll2", 0.8, 2.0),
    ]
    rows: list[dict[str, Any]] = []
    for config_name, pitch_db, roll_db in CONFIGS:
        for pose_name, pitch, roll in poses:
            norm_pitch = pitch / pitch_db
            norm_roll = roll / roll_db
            rows.append(
                {
                    "config": config_name,
                    "deadband_pitch_deg": pitch_db,
                    "deadband_roll_deg": roll_db,
                    "pose": pose_name,
                    "pitch_deg": pitch,
                    "roll_deg": roll,
                    "pitch_deadband_units": norm_pitch,
                    "roll_deadband_units": norm_roll,
                    "combined_deadband_norm": float(np.hypot(norm_pitch, norm_roll)),
                }
            )
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{float(value):.4f}")
    headers = [str(col) for col in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def _write_markdown(
    out_dir: Path,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    action_dist: pd.DataFrame,
    posture_probe: pd.DataFrame,
    max_samples: int,
) -> None:
    lines = [
        "# Pitch/Roll Deadband Axis Sensitivity",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Samples: {max_samples} deterministic FINO1 test windows, half event-positive when available.",
        "- Scope: planner-level replay only; no full 1 Hz platform dynamics integration.",
        "- Forecast input: replay `y_uv_raw`, first 0-60 min consumed as three 20 min planner blocks.",
        "- Posture cases: one zero-current-posture case plus synthetic nonzero pitch/roll cases to exercise the state-feedback path.",
        "",
        "## Main Result",
        "",
        (
            "In the zero-current-posture forecast-only case, changing pitch/roll deadbands is almost "
            "invariant because the forecast proxy and action mapping use the same scale. Once current "
            "posture residual is injected, the same deadband choices change the normalized residual, "
            "selected first actions, and tank-delta proxy. This supports the narrower claim that "
            "axis-specific deadband tuning matters mainly through the state-feedback path; it does "
            "not by itself prove that `1.0/0.8` is globally optimal."
        ),
        "",
        "## Config Summary",
        "",
        _markdown_table(summary),
        "",
        "## Change Versus Baseline",
        "",
        _markdown_table(comparisons),
        "",
        "## First-Action Distribution",
        "",
        _markdown_table(action_dist),
        "",
        "## Current-Posture Normalization Probe",
        "",
        (
            "These rows show why different axis deadbands matter even before a pump command is "
            "computed: the same physical pitch/roll angle is interpreted as a different number "
            "of deadband units under different axis scales."
        ),
        "",
        _markdown_table(posture_probe),
        "",
        "## Suggested Paper Wording",
        "",
        (
            "`d_theta` and `d_phi` are treated as axis-specific control-scale parameters. They "
            "normalize pitch and roll into deadband units and enter the state-feedback residual, "
            "candidate action gating, and scoring. A planner-level sensitivity scan shows that "
            "forecast-only pressure is largely scale-invariant, while nonzero current posture makes "
            "selected actions and pump-demand proxies depend on the axis deadbands. Therefore the "
            "split-axis setting should be described as a state-feedback tuning parameter. The exact "
            "numerical pair remains an engineering setting to be validated by simulation rather than "
            "a literature-fixed value."
        ),
        "",
    ]
    (out_dir / "deadband_axis_sensitivity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=240)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    replay = Fino1ReplayDataset(dataset_dir=dataset_dir, split=args.split)
    selected = _select_indices(replay, max_samples=int(args.max_samples))

    rows: list[dict[str, Any]] = []
    for sample_idx in selected:
        sample = replay._materialize_sample(sample_idx)  # noqa: SLF001 - analysis script.
        event_map = {name: float(sample.y_event[idx]) for idx, name in enumerate(replay.event_columns)}
        for posture_case, posture_pitch, posture_roll in POSTURE_CASES:
            plant_info = _plant_info_for_posture(posture_pitch, posture_roll)
            if posture_case == "forecast_only_zero_posture":
                plant_info = DEFAULT_PLANT_INFO
            for config_name, pitch_db, roll_db in CONFIGS:
                cfg = PlannerConfig(deadband_pitch_deg=pitch_db, deadband_roll_deg=roll_db)
                result = _best_for_sample(sample, cfg, plant_info)
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "history_end": sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
                        "ballast_attention_event": event_map.get("ballast_attention_event", 0.0),
                        "posture_case": posture_case,
                        "posture_pitch_deg": posture_pitch,
                        "posture_roll_deg": posture_roll,
                        "config": config_name,
                        "deadband_pitch_deg": pitch_db,
                        "deadband_roll_deg": roll_db,
                        **result,
                    }
                )

    detail = pd.DataFrame(rows)
    detail.to_csv(out_dir / "deadband_axis_sensitivity_detail.csv", index=False)

    summary_rows = []
    for (posture_case, config_name), group in detail.groupby(["posture_case", "config"], sort=False):
        summary_rows.append(
            {
                "posture_case": posture_case,
                "config": config_name,
                "deadband_pitch_deg": float(group["deadband_pitch_deg"].iloc[0]),
                "deadband_roll_deg": float(group["deadband_roll_deg"].iloc[0]),
                "n": int(len(group)),
                "mean_feasible_count": float(group["feasible_count"].mean()),
                "mean_pressure_norm_b1": float(group["pressure_norm_b1"].mean()),
                "mean_terminal_residual_cost": float(group["cost_terminal_residual_cost"].mean()),
                "mean_attitude_residual_cost": float(group["cost_attitude_residual_cost"].mean()),
                "mean_pump_work_proxy": float(group["cost_pump_work_cost"].mean()),
                "mean_first_delta_abs_sum_kg": float(group["first_delta_abs_sum_kg"].mean()),
                "active_first_action_rate": float((group["first_action"] != "hold").mean()),
                "active_medium_first_rate": float((group["first_action"] == "active_medium").mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "deadband_axis_sensitivity_summary.csv", index=False)

    baseline = detail[detail["config"] == "baseline_split_1p0_0p8"].set_index(["posture_case", "sample_idx"])
    comparison_rows = []
    for (posture_case, config_name), group in detail.groupby(["posture_case", "config"], sort=False):
        joined = group.set_index(["posture_case", "sample_idx"]).join(
            baseline[["first_action", "best_sequence", "first_delta_abs_sum_kg", "cost_terminal_residual_cost"]],
            rsuffix="_baseline",
            how="inner",
        )
        comparison_rows.append(
            {
                "posture_case": posture_case,
                "config": config_name,
                "first_action_change_rate": float((joined["first_action"] != joined["first_action_baseline"]).mean()),
                "sequence_change_rate": float((joined["best_sequence"] != joined["best_sequence_baseline"]).mean()),
                "mean_delta_abs_sum_change_kg": float(
                    (joined["first_delta_abs_sum_kg"] - joined["first_delta_abs_sum_kg_baseline"]).mean()
                ),
                "mean_terminal_residual_change": float(
                    (joined["cost_terminal_residual_cost"] - joined["cost_terminal_residual_cost_baseline"]).mean()
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)
    comparisons.to_csv(out_dir / "deadband_axis_sensitivity_vs_baseline.csv", index=False)

    action_rows = []
    for (posture_case, config_name), group in detail.groupby(["posture_case", "config"], sort=False):
        counts = Counter(group["first_action"])
        for action in ACTIONS:
            action_rows.append(
                {
                    "posture_case": posture_case,
                    "config": config_name,
                    "first_action": action,
                    "count": int(counts.get(action, 0)),
                    "rate": float(counts.get(action, 0) / max(len(group), 1)),
                }
            )
    action_dist = pd.DataFrame(action_rows)
    action_dist.to_csv(out_dir / "deadband_axis_sensitivity_action_distribution.csv", index=False)

    posture_probe = pd.DataFrame(_posture_probe_rows())
    posture_probe.to_csv(out_dir / "deadband_axis_posture_normalization_probe.csv", index=False)

    _write_markdown(
        out_dir=out_dir,
        summary=summary,
        comparisons=comparisons,
        action_dist=action_dist,
        posture_probe=posture_probe,
        max_samples=len(selected),
    )

    print(f"Wrote {out_dir}")
    print(summary.to_string(index=False))
    print(comparisons.to_string(index=False))


if __name__ == "__main__":
    main()
