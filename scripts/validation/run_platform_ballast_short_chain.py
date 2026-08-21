#!/usr/bin/env python3
"""Run a short causal ballast-to-platform chain without a controller."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from fowt_platform import (
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    assemble_ballast_model_snapshot,
    IncrementalLoads,
    IncrementalPlatformModel,
    IncrementalState,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    simulate_execution_step,
)


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)


def run_short_chain(*, duration_s: float, step_s: float) -> list[dict[str, float]]:
    if duration_s <= 0.0 or step_s <= 0.0:
        raise ValueError("duration_s and step_s must be positive")
    step_count = int(round(duration_s / step_s))
    if not np.isclose(step_count * step_s, duration_s):
        raise ValueError("duration_s must be an integer multiple of step_s")

    density_kg_m3 = 1025.0
    runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    capacities = np.full(3, TANK_CAPACITY_KG)
    requested_target = REFERENCE_TANK_MASSES_KG + np.array(
        [-3_000.0, 3_000.0, 0.0]
    )
    execution_config = replace(
        ExecutionRolloutConfig(),
        block_duration_s=step_s,
        water_density_kg_m3=density_kg_m3,
        max_pump_rate_m3_min=1.0,
        target_slew_enabled=False,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (20_000.0, 1.0)),
        tank_capacity_kg=TANK_CAPACITY_KG,
        internal_step_s=min(1.0, step_s),
    )
    execution_state = ExecutionRolloutState(
        masses_kg=REFERENCE_TANK_MASSES_KG,
        target_masses_kg=REFERENCE_TANK_MASSES_KG,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
    )
    platform_state = IncrementalState.zeros()
    records: list[dict[str, float]] = []

    for step_index in range(step_count):
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=runtime_assembly,
            actual_tank_masses_kg=execution_state.actual_masses_kg,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=capacities,
            tank_coordinates_m=TANK_COORDINATES_M,
        )
        platform_state = IncrementalPlatformModel(
            snapshot.matrices
        ).advance_frozen_step(
            platform_state,
            IncrementalLoads(
                wind=np.zeros(6),
                wave=np.zeros(6),
                ballast=snapshot.incremental_ballast_load,
                other=np.zeros(6),
            ),
            duration_s=step_s,
        )
        execution = simulate_execution_step(
            execution_state,
            requested_target,
            execution_config,
        )
        execution_state = execution.state
        records.append(
            {
                "time_s": (step_index + 1) * step_s,
                "tank_1_kg": float(execution_state.actual_masses_kg[0]),
                "tank_2_kg": float(execution_state.actual_masses_kg[1]),
                "tank_3_kg": float(execution_state.actual_masses_kg[2]),
                "total_ballast_kg": float(np.sum(execution_state.actual_masses_kg)),
                "surge_m": float(platform_state.position[0]),
                "sway_m": float(platform_state.position[1]),
                "heave_m": float(platform_state.position[2]),
                "roll_rad": float(platform_state.position[3]),
                "pitch_rad": float(platform_state.position[4]),
                "yaw_rad": float(platform_state.position[5]),
            }
        )
    return records


def _write_csv(path: Path, records: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--step-s", type=float, default=1.0)
    parser.add_argument("--comparison-step-s", type=float, default=0.5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    primary = run_short_chain(duration_s=args.duration_s, step_s=args.step_s)
    comparison = run_short_chain(
        duration_s=args.duration_s,
        step_s=args.comparison_step_s,
    )
    if args.csv is not None:
        _write_csv(args.csv, primary)

    position_fields = ("surge_m", "sway_m", "heave_m", "roll_rad", "pitch_rad", "yaw_rad")
    final_difference = {
        field: primary[-1][field] - comparison[-1][field]
        for field in position_fields
    }
    print(
        json.dumps(
            {
                "duration_s": args.duration_s,
                "primary_step_s": args.step_s,
                "comparison_step_s": args.comparison_step_s,
                "primary_final": primary[-1],
                "comparison_final": comparison[-1],
                "final_position_difference": final_difference,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
