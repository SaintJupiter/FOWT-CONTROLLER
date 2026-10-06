#!/usr/bin/env python3
"""Fixed-snapshot local sensitivity check for normalized MPC priorities.

This is a robustness diagnostic, not parameter optimization.  Snapshot times,
tested priorities and multipliers are fixed before execution.  Every variant
starts from the same platform and pump state as its baseline.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from fowt_platform import (  # noqa: E402
    IncrementalState,
    ThreeTankDifferentialModes,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
)
from real_lstm_preview_fixture import (  # noqa: E402
    REFERENCE_MANIFEST,
    TANK_COORDINATES_M,
    assemble_planner_rotor_preview_from_record,
    infer_real_lstm_wind_record,
    prepare_real_lstm_preview_resources,
)
from preview_mpc_experiment_runtime import (  # noqa: E402
    initial_execution_state,
    run_preview_mpc_cycle,
)
from wind_prediction.replay_dataset import TIMESTAMP_FMT  # noqa: E402


FIXED_ORIGINS = (
    "2021-10-04 09:30:00",
    "2021-10-04 11:30:00",
    "2021-10-05 09:30:00",
)
SENSITIVITY_PRIORITIES = (
    "tank_throughput",
    "movement_change",
)
SENSITIVITY_FACTORS = (0.5, 2.0)


def _metrics(cycle: dict[str, Any]) -> dict[str, Any]:
    execution = cycle["execution"]
    plan = cycle["plan"]
    target_delta = np.asarray(plan["first_target_tank_mass_delta_kg"], dtype=float)
    peak = np.asarray(execution["actual_peak_abs_roll_pitch_deg"], dtype=float)
    terminal = np.asarray(execution["actual_end_roll_pitch_deg"], dtype=float)
    return {
        "plan_success": bool(plan["success"]),
        "selected_request_source": execution["selected_request_source"],
        "selected_posture_precheck_passed": bool(
            execution["selected_posture_precheck_passed"]
        ),
        "first_target_tank_mass_delta_kg": target_delta.tolist(),
        "first_target_delta_norm_kg": float(np.linalg.norm(target_delta)),
        "transferred_volume_m3": float(execution["transferred_volume_m3"]),
        "aggregate_pump_active_time_s": float(
            execution["aggregate_pump_active_time_s"]
        ),
        "peak_dominant_tilt_deg": float(np.max(peak)),
        "terminal_dominant_tilt_deg": float(np.max(np.abs(terminal))),
        "maximum_posture_slack_deg": float(np.max(plan["posture_slack_deg"])),
        "solver_iterations": int(execution["solver"]["iterations"]),
    }


def run_sensitivity(*, device: str = "cpu") -> dict[str, Any]:
    runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    modes = ThreeTankDifferentialModes(
        tank_coordinates_m=TANK_COORDINATES_M,
        gravity_m_s2=runtime.gravity_m_s2,
    )
    resources = prepare_real_lstm_preview_resources(device=device)
    records: list[dict[str, Any]] = []
    for origin_text in FIXED_ORIGINS:
        origin = datetime.strptime(origin_text, TIMESTAMP_FMT)
        source_record = infer_real_lstm_wind_record(
            resources=resources,
            origin=origin,
        )
        source_preview = assemble_planner_rotor_preview_from_record(
            resources=resources,
            source_record=source_record,
            platform_state=IncrementalState.zeros(),
            planner_forecast_mode="lstm",
            horizon_blocks=source_record.forecast.horizon_steps,
        )
        baseline_cycle, _, _ = run_preview_mpc_cycle(
            source_preview=source_preview,
            runtime=runtime,
            modes=modes,
            platform_state=IncrementalState.zeros(),
            execution_state=initial_execution_state(),
        )
        baseline = _metrics(baseline_cycle)
        baseline_delta = np.asarray(
            baseline["first_target_tank_mass_delta_kg"], dtype=float
        )
        records.append(
            {
                "origin": origin_text,
                "varied_priority": "baseline",
                "factor": 1.0,
                "metrics": baseline,
                "relative_to_baseline": {
                    "first_action_direction_reversed": False,
                    "first_target_delta_norm_ratio": 1.0,
                    "selected_request_source_changed": False,
                },
            }
        )
        for priority in SENSITIVITY_PRIORITIES:
            for factor in SENSITIVITY_FACTORS:
                cycle, _, _ = run_preview_mpc_cycle(
                    source_preview=source_preview,
                    runtime=runtime,
                    modes=modes,
                    platform_state=IncrementalState.zeros(),
                    execution_state=initial_execution_state(),
                    objective_priority_multipliers={priority: factor},
                )
                metrics = _metrics(cycle)
                delta = np.asarray(
                    metrics["first_target_tank_mass_delta_kg"], dtype=float
                )
                denominator = max(float(np.linalg.norm(baseline_delta)), 1.0e-12)
                reversed_direction = bool(
                    np.linalg.norm(delta) > 1.0e-9
                    and np.linalg.norm(baseline_delta) > 1.0e-9
                    and float(delta @ baseline_delta) < 0.0
                )
                records.append(
                    {
                        "origin": origin_text,
                        "varied_priority": priority,
                        "factor": factor,
                        "metrics": metrics,
                        "relative_to_baseline": {
                            "first_action_direction_reversed": reversed_direction,
                            "first_target_delta_norm_ratio": float(
                                np.linalg.norm(delta) / denominator
                            ),
                            "selected_request_source_changed": (
                                metrics["selected_request_source"]
                                != baseline["selected_request_source"]
                            ),
                        },
                    }
                )

    variants = [record for record in records if record["varied_priority"] != "baseline"]
    ratios = np.asarray(
        [
            record["relative_to_baseline"]["first_target_delta_norm_ratio"]
            for record in variants
        ],
        dtype=float,
    )
    by_priority = {}
    for priority in SENSITIVITY_PRIORITIES:
        subset = [record for record in variants if record["varied_priority"] == priority]
        priority_ratios = np.asarray(
            [
                record["relative_to_baseline"]["first_target_delta_norm_ratio"]
                for record in subset
            ],
            dtype=float,
        )
        by_priority[priority] = {
            "plan_failure_count": sum(
                not record["metrics"]["plan_success"] for record in subset
            ),
            "physical_precheck_failure_count": sum(
                not record["metrics"]["selected_posture_precheck_passed"]
                for record in subset
            ),
            "direction_reversal_count": sum(
                record["relative_to_baseline"]["first_action_direction_reversed"]
                for record in subset
            ),
            "first_target_delta_norm_ratio_range": [
                float(np.min(priority_ratios)),
                float(np.max(priority_ratios)),
            ],
        }
    return {
        "purpose": "fixed_snapshot_local_sensitivity_not_parameter_optimization",
        "fixed_origins": list(FIXED_ORIGINS),
        "varied_priorities": list(SENSITIVITY_PRIORITIES),
        "factors": list(SENSITIVITY_FACTORS),
        "variant_count": len(variants),
        "summary": {
            "plan_failure_count": sum(
                not record["metrics"]["plan_success"] for record in variants
            ),
            "physical_precheck_failure_count": sum(
                not record["metrics"]["selected_posture_precheck_passed"]
                for record in variants
            ),
            "first_action_direction_reversal_count": sum(
                record["relative_to_baseline"]["first_action_direction_reversed"]
                for record in variants
            ),
            "selected_request_source_change_count": sum(
                record["relative_to_baseline"]["selected_request_source_changed"]
                for record in variants
            ),
            "first_target_delta_norm_ratio_range": [
                float(np.min(ratios)),
                float(np.max(ratios)),
            ],
            "by_priority": by_priority,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = run_sensitivity(device=args.device)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output_json": str(args.output_json),
                    "summary": result["summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(payload)


if __name__ == "__main__":
    main()
