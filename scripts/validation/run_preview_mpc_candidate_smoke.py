#!/usr/bin/env python3
"""Run the predeclared preview-MPC candidates on three fixed six-hour cases.

The command is a chain smoke check only.  It verifies that every predeclared
candidate reaches the same LSTM six-block controller path and remains
executable under the configured low-order model limits.  It deliberately omits pump
performance contrasts and does not select a candidate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
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
    ThreeTankDifferentialModes,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
)
from preview_mpc_continuous_experiment import (  # noqa: E402
    CYCLE_COUNT,
    FIXED_CASE_ORIGINS,
    WARMUP_BLOCK_COUNT,
    preflight_source_operating_domain,
    prepare_source_records,
    run_variant,
)
from run_preview_mpc_parameter_study import (  # noqa: E402
    SCREENING_FORECAST_VARIANT,
    SCREENING_SOURCE_RECORD_FOLLOWUP_BLOCKS,
    read_and_verify_screening_registry,
)
from run_real_lstm_physical_cycle_smoke import (  # noqa: E402
    REFERENCE_MANIFEST,
    TANK_COORDINATES_M,
    prepare_smoke_resources,
)
from wind_prediction.preview_mpc import (  # noqa: E402
    MAX_SCALED_RECONSTRUCTION_VIOLATION,
)
from wind_prediction.replay_dataset import TIMESTAMP_FMT  # noqa: E402


DEFAULT_REGISTRY_PATH = (
    ROOT
    / "artifacts"
    / "preview_mpc_parameter_study_v3"
    / "predeclared_screening_registry.json"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compact_smoke_record(
    *,
    case_id: str,
    origin_time: str,
    candidate: dict[str, Any],
    registry: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    expected_multipliers = candidate["priority_multipliers"]
    actual_multipliers = result["objective_priority_multipliers"]
    planner_match = (
        result["planner_forecast_mode"]
        == SCREENING_FORECAST_VARIANT["planner_forecast_mode"]
        and result["planner_horizon_blocks"]
        == SCREENING_FORECAST_VARIANT["planner_horizon_blocks"]
    )
    identity_match = (
        result["preview_mpc_design_parameter_sha256"]
        == candidate["parameter_sha256"]
        and actual_multipliers == expected_multipliers
        and planner_match
    )
    aggregate = result["aggregate"]
    complete = aggregate["completed_cycle_count"] == CYCLE_COUNT
    internal_constraint_passed = all(
        aggregate[name] == 0
        for name in (
            "plan_failure_count",
            "fallback_count",
            "selected_precheck_failure_count",
            "no_safe_precheck_candidate_count",
            "actual_posture_limit_failure_count",
        )
    ) and (
        aggregate["maximum_scaled_constraint_violation"]
        <= MAX_SCALED_RECONSTRUCTION_VIOLATION
    )
    return {
        "case_id": case_id,
        "origin_time": origin_time,
        "candidate_id": candidate["candidate_id"],
        "candidate_registry_sha256": registry["candidate_registry_sha256"],
        "screening_configuration_sha256": registry[
            "screening_configuration_sha256"
        ],
        "expected_candidate_parameter_sha256": candidate["parameter_sha256"],
        "actual_candidate_parameter_sha256": result[
            "preview_mpc_design_parameter_sha256"
        ],
        "expected_priority_multipliers": expected_multipliers,
        "actual_priority_multipliers": actual_multipliers,
        "planner_forecast_mode": result["planner_forecast_mode"],
        "planner_horizon_blocks": result["planner_horizon_blocks"],
        "completed_cycle_count": aggregate["completed_cycle_count"],
        "continuous_duration_h": aggregate["continuous_duration_h"],
        "plant_load_sha256": result["plant_load_sha256"],
        "preconditioning_external_rotor_load_sha256": result["preconditioning"][
            "external_rotor_load_sha256"
        ],
        "preconditioning_duration_s": result["preconditioning"]["duration_s"],
        "plan_failure_count": aggregate["plan_failure_count"],
        "fallback_count": aggregate["fallback_count"],
        "selected_precheck_failure_count": aggregate[
            "selected_precheck_failure_count"
        ],
        "no_safe_precheck_candidate_count": aggregate[
            "no_safe_precheck_candidate_count"
        ],
        "actual_posture_limit_failure_count": aggregate[
            "actual_posture_limit_failure_count"
        ],
        "maximum_scaled_constraint_violation": aggregate[
            "maximum_scaled_constraint_violation"
        ],
        "identity_match": identity_match,
        "complete": complete,
        "internal_constraint_passed": internal_constraint_passed,
    }


def run_candidate_smoke(*, registry_path: Path, device: str = "cpu") -> dict[str, Any]:
    registry = read_and_verify_screening_registry(registry_path)
    runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    modes = ThreeTankDifferentialModes(
        tank_coordinates_m=TANK_COORDINATES_M,
        gravity_m_s2=runtime.gravity_m_s2,
    )
    resources = prepare_smoke_resources(device=device)
    records: list[dict[str, Any]] = []
    for case_index, origin_text in enumerate(FIXED_CASE_ORIGINS, start=1):
        origin = datetime.strptime(origin_text, TIMESTAMP_FMT)
        source_records = prepare_source_records(
            origin=origin,
            device=device,
            cycle_count=CYCLE_COUNT,
            lookahead_count=SCREENING_SOURCE_RECORD_FOLLOWUP_BLOCKS,
            resources=resources,
            warmup_block_count=WARMUP_BLOCK_COUNT,
        )
        domain_preflight = preflight_source_operating_domain(
            source_records=source_records,
            origin=origin,
            warmup_block_count=WARMUP_BLOCK_COUNT,
        )
        if not domain_preflight["supported"]:
            raise RuntimeError(
                f"smoke_{case_index:02d} is outside the source-bound nominal rotor "
                "operating domain; inspect the preflight report before candidates run"
            )
        case_plant_hashes: set[str] = set()
        case_preconditioning_hashes: set[str] = set()
        for candidate in registry["candidates"]:
            result = run_variant(
                name=candidate["candidate_id"],
                planner_forecast_mode=SCREENING_FORECAST_VARIANT[
                    "planner_forecast_mode"
                ],
                planner_horizon_blocks=SCREENING_FORECAST_VARIANT[
                    "planner_horizon_blocks"
                ],
                source_records=source_records,
                runtime=runtime,
                modes=modes,
                objective_priority_multipliers=candidate["priority_multipliers"],
                warmup_block_count=WARMUP_BLOCK_COUNT,
            )
            case_plant_hashes.add(result["plant_load_sha256"])
            case_preconditioning_hashes.add(
                result["preconditioning"]["external_rotor_load_sha256"]
            )
            records.append(
                _compact_smoke_record(
                    case_id=f"smoke_{case_index:02d}",
                    origin_time=origin_text,
                    candidate=candidate,
                    registry=registry,
                    result=result,
                )
            )
        if len(case_plant_hashes) != 1:
            raise RuntimeError("candidate smoke variants consumed different plant loads")
        if len(case_preconditioning_hashes) != 1:
            raise RuntimeError(
                "candidate smoke variants used different preconditioning loads"
            )
    passed = [
        item
        for item in records
        if (
            item["identity_match"]
            and item["complete"]
            and item["internal_constraint_passed"]
        )
    ]
    return {
        "purpose": "predeclared_preview_mpc_candidate_chain_smoke",
        "boundaries": {
            "is_parameter_selection": False,
            "is_performance_comparison": False,
            "uses_only_lstm_six_block_preview": True,
            "reports_only_identity_completeness_and_low_order_internal_constraints": True,
            "internal_constraint_scope": (
                "low_order_model_only_not_a_physical_platform_safety_claim"
            ),
        },
        "persisted_registry_path": str(registry_path.relative_to(ROOT)),
        "persisted_registry_file_sha256": _file_sha256(registry_path),
        "candidate_registry_sha256": registry["candidate_registry_sha256"],
        "screening_configuration_sha256": registry[
            "screening_configuration_sha256"
        ],
        "runtime_source_artifact_sha256": registry[
            "runtime_source_artifact_sha256"
        ],
        "run_count": len(records),
        "passed_run_count": len(passed),
        "all_runs_passed_identity_completeness_and_internal_constraints": len(passed)
        == len(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    result = run_candidate_smoke(
        registry_path=arguments.registry_json,
        device=arguments.device,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if arguments.output_json is None:
        print(payload)
    else:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(payload + "\n", encoding="utf-8")
        print(arguments.output_json)


if __name__ == "__main__":
    main()
