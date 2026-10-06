#!/usr/bin/env python3
"""Run the frozen ten-case preview-MPC candidate screen.

The command evaluates only the four candidates fixed before screening.  It
does not generate candidates, tune a weight from one case, or select a winner.
Its output is a compact set of case-level observations and non-scalar paired
summaries for later review.
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
    ThreeTankDifferentialModes,
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
)
from preview_mpc_parameter_study_protocol import (  # noqa: E402
    ParameterCandidate,
    ParameterStudyObservation,
    summarize_paired_candidate,
)
from preview_mpc_continuous_experiment import (  # noqa: E402
    CYCLE_COUNT,
    POSTURE_THRESHOLDS_DEG,
    WARMUP_BLOCK_COUNT,
    preflight_source_operating_domain,
    prepare_source_records,
    run_variant,
)
from real_lstm_preview_fixture import (  # noqa: E402
    REFERENCE_MANIFEST,
    TANK_COORDINATES_M,
    RealLstmWindRecord,
    prepare_real_lstm_preview_resources,
)
from run_preview_mpc_parameter_study import (  # noqa: E402
    SCREENING_FORECAST_VARIANT,
    SCREENING_SOURCE_RECORD_FOLLOWUP_BLOCKS,
    default_parameter_study_protocol,
    read_and_verify_screening_registry,
    screening_case_identity,
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
BASELINE_CANDIDATE_ID = "baseline_design"
STAGE_NAME = "multi_regime_screening"


def _frozen_candidates(registry: dict[str, Any]) -> tuple[ParameterCandidate, ...]:
    return tuple(
        ParameterCandidate(
            candidate_id=record["candidate_id"],
            parent_design_identity=record["parent_design_identity"],
            priority_multipliers=tuple(record["priority_multipliers"].items()),
            parameter_sha256=record["parameter_sha256"],
        )
        for record in registry["candidates"]
    )


def _observation_from_result(
    *,
    case_id: str,
    candidate: ParameterCandidate,
    result: dict[str, Any],
) -> ParameterStudyObservation:
    if not result.get("completed_requested_window", False):
        raise RuntimeError(
            "screening variant left the working model scope or otherwise "
            "terminated before the requested window completed"
        )
    aggregate = result["aggregate"]
    return ParameterStudyObservation(
        case_id=case_id,
        candidate_id=candidate.candidate_id,
        candidate_parameter_sha256=result["preview_mpc_design_parameter_sha256"],
        transferred_volume_m3=aggregate["transferred_volume_m3"],
        aggregate_pump_active_time_s=aggregate["aggregate_pump_active_time_s"],
        pump_start_count=aggregate["pump_start_count"],
        dominant_tilt_rms_deg=aggregate["dominant_tilt_rms_deg"],
        dominant_tilt_time_above_deg_s=tuple(
            (float(threshold), aggregate["dominant_tilt_time_above_deg_s"][str(threshold)])
            for threshold in POSTURE_THRESHOLDS_DEG
        ),
        plan_failure_count=aggregate["plan_failure_count"],
        fallback_count=aggregate["fallback_count"],
        precheck_failure_count=aggregate["selected_precheck_failure_count"],
        no_safe_candidate_count=aggregate["no_safe_precheck_candidate_count"],
        sampled_posture_violation_count=aggregate[
            "actual_posture_limit_failure_count"
        ],
    )


def _record(
    *,
    case_id: str,
    origin_time: str,
    stratum: str,
    candidate: ParameterCandidate,
    registry: dict[str, Any],
    result: dict[str, Any],
    observation: ParameterStudyObservation,
) -> dict[str, Any]:
    aggregate = result["aggregate"]
    identity_match = (
        result["preview_mpc_design_parameter_sha256"]
        == candidate.parameter_sha256
        and result["objective_priority_multipliers"] == candidate.multipliers
        and result["planner_forecast_mode"]
        == SCREENING_FORECAST_VARIANT["planner_forecast_mode"]
        and result["planner_horizon_blocks"]
        == SCREENING_FORECAST_VARIANT["planner_horizon_blocks"]
    )
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
        "stratum": stratum,
        "candidate_id": candidate.candidate_id,
        "candidate_registry_sha256": registry["candidate_registry_sha256"],
        "screening_configuration_sha256": registry[
            "screening_configuration_sha256"
        ],
        "runtime_source_artifact_sha256": registry[
            "runtime_source_artifact_sha256"
        ],
        "candidate_parameter_sha256": candidate.parameter_sha256,
        "actual_candidate_parameter_sha256": result[
            "preview_mpc_design_parameter_sha256"
        ],
        "objective_priority_multipliers": result[
            "objective_priority_multipliers"
        ],
        "planner_forecast_mode": result["planner_forecast_mode"],
        "planner_horizon_blocks": result["planner_horizon_blocks"],
        "completed_cycle_count": aggregate["completed_cycle_count"],
        "continuous_duration_h": aggregate["continuous_duration_h"],
        "plant_load_sha256": result["plant_load_sha256"],
        "preconditioning_external_rotor_load_sha256": result["preconditioning"][
            "external_rotor_load_sha256"
        ],
        "preconditioning_duration_s": result["preconditioning"]["duration_s"],
        "transferred_volume_m3": observation.transferred_volume_m3,
        "aggregate_pump_active_time_s": observation.aggregate_pump_active_time_s,
        "pump_start_count": observation.pump_start_count,
        "dominant_tilt_rms_deg": observation.dominant_tilt_rms_deg,
        "dominant_tilt_time_above_deg_s": {
            str(threshold): duration
            for threshold, duration in observation.dominant_tilt_time_above_deg_s
        },
        "plan_failure_count": observation.plan_failure_count,
        "fallback_count": observation.fallback_count,
        "selected_precheck_failure_count": observation.precheck_failure_count,
        "no_safe_precheck_candidate_count": observation.no_safe_candidate_count,
        "actual_posture_limit_failure_count": observation.sampled_posture_violation_count,
        "maximum_scaled_constraint_violation": aggregate[
            "maximum_scaled_constraint_violation"
        ],
        "identity_match": identity_match,
        "complete": complete,
        "internal_constraint_passed": internal_constraint_passed,
    }


def run_stage10_screening(
    *, registry_path: Path, device: str = "cpu"
) -> dict[str, Any]:
    """Run one fixed, complete ten-case screen without choosing a candidate."""

    registry = read_and_verify_screening_registry(registry_path)
    protocol = default_parameter_study_protocol()
    stage = protocol.stage(STAGE_NAME)
    if not stage.readiness["ready"]:
        raise RuntimeError(
            f"{STAGE_NAME} is not ready: {stage.readiness['issues']}"
        )
    if registry["screening_case_identity"] != screening_case_identity():
        raise ValueError(
            "persisted screening-case identity does not match the current "
            "frozen stage10 registry"
        )
    frozen_candidates = _frozen_candidates(registry)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in frozen_candidates}
    if set(candidate_by_id) != {
        "baseline_design",
        "terminal_posture_x2",
        "tank_throughput_x2",
        "movement_change_x2",
    }:
        raise ValueError("the frozen registry does not contain the four declared probes")

    runtime = assemble_volturnus_static_restoring_aligned_runtime_assembly(
        REFERENCE_MANIFEST,
        np.zeros((6, 6)),
    )
    modes = ThreeTankDifferentialModes(
        tank_coordinates_m=TANK_COORDINATES_M,
        gravity_m_s2=runtime.gravity_m_s2,
    )
    resources = prepare_real_lstm_preview_resources(device=device)
    observations: list[ParameterStudyObservation] = []
    records: list[dict[str, Any]] = []
    domain_preflights: list[dict[str, Any]] = []
    source_records_by_case: dict[str, list[RealLstmWindRecord]] = {}

    # Materialize and inspect every declared case before any candidate runs.
    # This prevents an early case from producing candidate results when a later
    # case makes the frozen screening set invalid.
    for case in stage.cases:
        origin = datetime.strptime(case.origin_time, TIMESTAMP_FMT)
        source_records = prepare_source_records(
            origin=origin,
            device=device,
            cycle_count=CYCLE_COUNT,
            lookahead_count=SCREENING_SOURCE_RECORD_FOLLOWUP_BLOCKS,
            resources=resources,
            warmup_block_count=WARMUP_BLOCK_COUNT,
        )
        source_records_by_case[case.case_id] = source_records
        domain_preflight = preflight_source_operating_domain(
            source_records=source_records,
            resources=resources,
            origin=origin,
            cycle_count=CYCLE_COUNT,
            warmup_block_count=WARMUP_BLOCK_COUNT,
        )
        domain_preflights.append({"case_id": case.case_id, **domain_preflight})

    unsupported_preflights = [
        item for item in domain_preflights if not item["supported"]
    ]
    if unsupported_preflights:
        return {
            "purpose": "predeclared_preview_mpc_ten_case_candidate_screening",
            "status": "source_operating_domain_rejected",
            "boundaries": {
                "candidate_set_frozen_before_run": True,
                "candidate_runs_started": False,
                "is_automatic_parameter_selection": False,
                "is_final_performance_validation": False,
                "is_physical_platform_safety_claim": False,
                "internal_constraint_scope": "low_order_model_only",
            },
            "stage": stage.as_dict(),
            "persisted_registry_path": str(registry_path.relative_to(ROOT)),
            "candidate_registry_sha256": registry["candidate_registry_sha256"],
            "screening_configuration_sha256": registry[
                "screening_configuration_sha256"
            ],
            "runtime_source_artifact_sha256": registry[
                "runtime_source_artifact_sha256"
            ],
            "source_operating_domain_preflights": domain_preflights,
            "unsupported_case_ids": [
                item["case_id"] for item in unsupported_preflights
            ],
            "run_count": 0,
            "records": [],
        }

    for case in stage.cases:
        source_records = source_records_by_case[case.case_id]
        case_source_record_hashes: set[str] = set()
        case_preconditioning_hashes: set[str] = set()
        for candidate in frozen_candidates:
            result = run_variant(
                name=candidate.candidate_id,
                planner_forecast_mode=SCREENING_FORECAST_VARIANT[
                    "planner_forecast_mode"
                ],
                planner_horizon_blocks=SCREENING_FORECAST_VARIANT[
                    "planner_horizon_blocks"
                ],
                source_records=source_records,
                resources=resources,
                runtime=runtime,
                modes=modes,
                objective_priority_multipliers=candidate.multipliers,
                warmup_block_count=WARMUP_BLOCK_COUNT,
            )
            case_source_record_hashes.add(
                result["source_record_sequence_sha256"]
            )
            case_preconditioning_hashes.add(
                result["preconditioning"]["external_rotor_load_sha256"]
            )
            observation = _observation_from_result(
                case_id=case.case_id,
                candidate=candidate,
                result=result,
            )
            observations.append(observation)
            records.append(
                _record(
                    case_id=case.case_id,
                    origin_time=case.origin_time,
                    stratum=case.stratum,
                    candidate=candidate,
                    registry=registry,
                    result=result,
                    observation=observation,
                )
            )
        if len(case_source_record_hashes) != 1:
            raise RuntimeError(
                "stage10 candidates consumed different recorded-wind sequences "
                f"for {case.case_id}"
            )
        if len(case_preconditioning_hashes) != 1:
            raise RuntimeError(
                "stage10 candidates used different preconditioning loads "
                f"for {case.case_id}"
            )
    integrity_passed = all(
        item["identity_match"]
        and item["complete"]
        and item["internal_constraint_passed"]
        for item in records
    )
    paired_summaries = (
        {
            candidate.candidate_id: summarize_paired_candidate(
                protocol=protocol,
                stage_name=stage.name,
                baseline_candidate_id=BASELINE_CANDIDATE_ID,
                candidate_id=candidate.candidate_id,
                frozen_candidates=frozen_candidates,
                observations=tuple(observations),
            )
            for candidate in frozen_candidates
            if candidate.candidate_id != BASELINE_CANDIDATE_ID
        }
        if integrity_passed
        else None
    )
    return {
        "purpose": "predeclared_preview_mpc_ten_case_candidate_screening",
        "status": "completed",
        "boundaries": {
            "candidate_set_frozen_before_run": True,
            "is_automatic_parameter_selection": False,
            "is_final_performance_validation": False,
            "is_physical_platform_safety_claim": False,
            "internal_constraint_scope": "low_order_model_only",
        },
        "stage": stage.as_dict(),
        "persisted_registry_path": str(registry_path.relative_to(ROOT)),
        "candidate_registry_sha256": registry["candidate_registry_sha256"],
        "screening_configuration_sha256": registry[
            "screening_configuration_sha256"
        ],
        "runtime_source_artifact_sha256": registry[
            "runtime_source_artifact_sha256"
        ],
        "source_operating_domain_preflights": domain_preflights,
        "run_count": len(records),
        "integrity_passed_run_count": sum(
            item["identity_match"]
            and item["complete"]
            and item["internal_constraint_passed"]
            for item in records
        ),
        "all_runs_passed_identity_completeness_and_internal_constraints": integrity_passed,
        "automatic_candidate_selection": None,
        "paired_summaries_available_for_review": integrity_passed,
        "paired_candidate_summaries": paired_summaries,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry-json", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    result = run_stage10_screening(
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
