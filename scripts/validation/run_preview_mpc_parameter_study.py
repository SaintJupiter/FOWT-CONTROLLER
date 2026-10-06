#!/usr/bin/env python3
"""Dry-run the staged preview-MPC parameter-study protocol.

No controller simulation or parameter search is performed.  The command
prints the frozen study structure, current stage readiness and the identity of
the unchanged baseline design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIRECTORY = ROOT / "src"
VALIDATION_DIRECTORY = Path(__file__).resolve().parent
for directory in (SRC_DIRECTORY, VALIDATION_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from preview_mpc_parameter_study_protocol import (  # noqa: E402
    ParameterStudyCase,
    ParameterStudyProtocol,
    ParameterStudyStage,
    freeze_candidate_registry,
)
from preview_mpc_parameter_study_cases import (  # noqa: E402
    load_frozen_staged_cases,
)
from preview_mpc_continuous_experiment import (  # noqa: E402
    FIXED_CASE_ORIGINS,
    implementation_identity,
    runtime_source_artifact_sha256,
)
from wind_prediction.preview_mpc_design import (  # noqa: E402
    research_preview_mpc_design_v4,
)


PREDECLARED_SCREENING_CANDIDATES = (
    {
        "candidate_id": "baseline_design",
        "priority_multipliers": {},
        "hypothesis": (
            "unmodified_research_preview_mpc_design_reference"
        ),
    },
    {
        "candidate_id": "tank_throughput_x2",
        "priority_multipliers": {"tank_throughput": 2.0},
        "hypothesis": (
            "stronger_total_tank_throughput_penalty_without_changing_"
            "posture_or_pump_constraints"
        ),
    },
    {
        "candidate_id": "movement_change_x2",
        "priority_multipliers": {"movement_change": 2.0},
        "hypothesis": (
            "stronger_continuity_preference_relative_to_the_current_"
            "unfinished_target_and_later_preview_actions"
        ),
    },
)
SCREENING_FORECAST_VARIANT = {
    "planner_forecast_mode": "lstm",
    "planner_horizon_blocks": 6,
    "decision_information": "controller_visible_learned_future_forecast",
}
# Each LSTM source record already carries the six future endpoints seen by the
# planner.  The rolling plant only needs one following recorded observation to
# advance the final controlled block, so the screening path must not create or
# preflight unused oracle-style tail records.
SCREENING_SOURCE_RECORD_FOLLOWUP_BLOCKS = 1


def _screening_configuration_sha256(
    *,
    candidate_registry_sha256: str,
    forecast_variant: dict[str, Any],
    screening_case_identity: dict[str, Any],
    base_design_parameter_sha256: str,
    implementation_bundle_sha256: str,
) -> str:
    payload = {
        "candidate_registry_sha256": candidate_registry_sha256,
        "screening_forecast_variant": forecast_variant,
        "screening_case_identity": screening_case_identity,
        "base_design_parameter_sha256": base_design_parameter_sha256,
        "implementation_bundle_sha256": implementation_bundle_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def screening_case_identity() -> dict[str, Any]:
    """Return the frozen ten-case identity used by the candidate screen."""

    staged_cases, staged_case_identity = load_frozen_staged_cases(
        repository_root=ROOT
    )
    cases = staged_cases["stage10"]
    case_records = [case.as_dict() for case in cases]
    case_registry_sha256 = hashlib.sha256(
        json.dumps(case_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "stage_name": "multi_regime_screening",
        "case_registry_sha256": case_registry_sha256,
        "selection_manifest_sha256": staged_case_identity[
            "selection_manifest_sha256"
        ],
        "stage10_cases_file_sha256": staged_case_identity["stage_file_sha256"][
            "stage10"
        ],
        "source_observations_sha256": staged_case_identity[
            "source_observations_sha256"
        ],
    }


def implementation_bundle_sha256() -> str:
    identity = implementation_identity()
    identity.update(
        {
            "parameter_study_protocol": hashlib.sha256(
                (VALIDATION_DIRECTORY / "preview_mpc_parameter_study_protocol.py")
                .read_bytes()
            ).hexdigest(),
            "parameter_study_case_registry": hashlib.sha256(
                (VALIDATION_DIRECTORY / "preview_mpc_parameter_study_cases.py")
                .read_bytes()
            ).hexdigest(),
        }
    )
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def default_parameter_study_protocol() -> ParameterStudyProtocol:
    design = research_preview_mpc_design_v4()
    design_record = design.as_dict()
    staged_cases, _ = load_frozen_staged_cases(repository_root=ROOT)
    smoke_cases = tuple(
        ParameterStudyCase(
            case_id=f"smoke_{index:02d}",
            origin_time=origin,
            source_event_id=f"legacy_smoke_event_{index:02d}",
            stratum="legacy_fixed_smoke_unclassified",
            duration_h=6.0,
            selection_basis=(
                "fixed_chain_case_declared_before_parameter_study_"
                "not_eligible_for_parameter_ranking"
            ),
        )
        for index, origin in enumerate(FIXED_CASE_ORIGINS, start=1)
    )
    return ParameterStudyProtocol(
        study_id="preview_mpc_staged_parameter_study_v3",
        base_design_identity=design.identity,
        base_design_parameter_sha256=design_record["parameter_sha256"],
        implementation_bundle_sha256=implementation_bundle_sha256(),
        tunable_priorities=(
            "terminal_posture",
            "tank_throughput",
            "movement_change",
        ),
        stages=(
            ParameterStudyStage(
                name="chain_smoke",
                role="smoke",
                cases=smoke_cases,
                minimum_case_count=3,
                minimum_stratum_count=1,
                minimum_cases_per_stratum=1,
                permits_parameter_selection=False,
            ),
            ParameterStudyStage(
                name="multi_regime_screening",
                role="screening",
                cases=staged_cases["stage10"],
                minimum_case_count=10,
                minimum_stratum_count=4,
                minimum_cases_per_stratum=2,
                permits_parameter_selection=True,
            ),
            ParameterStudyStage(
                name="independent_confirmation",
                role="confirmation",
                cases=staged_cases["stage20"],
                minimum_case_count=20,
                minimum_stratum_count=4,
                minimum_cases_per_stratum=5,
                permits_parameter_selection=False,
            ),
        ),
    )


def predeclared_screening_candidate_registry() -> dict[str, Any]:
    """Return four fixed directional probes without selecting or running one."""

    protocol = default_parameter_study_protocol()
    design = research_preview_mpc_design_v4()
    candidates = tuple(
        protocol.candidate(
            candidate_id=definition["candidate_id"],
            base_design=design,
            priority_multipliers=definition["priority_multipliers"],
        )
        for definition in PREDECLARED_SCREENING_CANDIDATES
    )
    registry = freeze_candidate_registry(candidates)
    hypotheses = {
        definition["candidate_id"]: definition["hypothesis"]
        for definition in PREDECLARED_SCREENING_CANDIDATES
    }
    registry.update(
        {
            "status": "predeclared_registry_not_executed",
            "screening_forecast_variant": SCREENING_FORECAST_VARIANT,
            "screening_case_identity": screening_case_identity(),
            "runtime_source_artifact_sha256": runtime_source_artifact_sha256(),
            "candidate_hypotheses": hypotheses,
            "multiplier_interpretation": (
                "factor_two_is_a_coarse_directional_probe_not_a_calibrated_"
                "or_optimal_parameter_value"
            ),
        }
    )
    registry["screening_configuration_sha256"] = _screening_configuration_sha256(
        candidate_registry_sha256=registry["candidate_registry_sha256"],
        forecast_variant=SCREENING_FORECAST_VARIANT,
        screening_case_identity=registry["screening_case_identity"],
        base_design_parameter_sha256=protocol.base_design_parameter_sha256,
        implementation_bundle_sha256=protocol.implementation_bundle_sha256,
    )
    return registry


def read_and_verify_screening_registry(path: Path) -> dict[str, Any]:
    """Load a screening registry bound to the current cases and implementation."""

    if not path.is_file():
        raise FileNotFoundError(f"predeclared candidate registry not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    persisted = payload.get("candidate_registry", payload)
    if not isinstance(persisted, dict):
        raise ValueError("candidate registry payload must contain an object")
    expected = predeclared_screening_candidate_registry()
    for key in (
        "candidate_registry_sha256",
        "screening_configuration_sha256",
        "screening_forecast_variant",
        "screening_case_identity",
        "runtime_source_artifact_sha256",
        "candidates",
    ):
        if persisted.get(key) != expected.get(key):
            raise ValueError(
                "persisted candidate registry does not match the current "
                f"predeclared registry for {key}"
            )
    if persisted.get("status") != "predeclared_registry_not_executed":
        raise ValueError("candidate registry is not a pre-execution registry")
    return persisted


def dry_run() -> dict[str, Any]:
    protocol = default_parameter_study_protocol()
    staged_cases, staged_case_identity = load_frozen_staged_cases(
        repository_root=ROOT
    )
    candidate_registry = predeclared_screening_candidate_registry()
    readiness = {stage.name: stage.readiness for stage in protocol.stages}
    return {
        "purpose": "parameter_study_protocol_dry_run_without_simulation",
        "is_parameter_search": False,
        "is_performance_result": False,
        "protocol": protocol.as_dict(),
        "staged_case_registry": staged_case_identity,
        "deferred_frozen_validation": {
            "case_count": len(staged_cases["stage30"]),
            "strata": sorted({case.stratum for case in staged_cases["stage30"]}),
            "rule": (
                "stage30_is_managed_by_validation_protocol_v2_and_must_not_be_"
                "used_for_parameter_selection"
            ),
        },
        "candidate_registry": candidate_registry,
        "stage_readiness": readiness,
        "next_required_input": (
            "run_stage10_only_after_review_of_the_lstm_six_block_chain_smoke"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = dry_run()
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "output_json": str(args.output_json),
                    "stage_readiness": result["stage_readiness"],
                    "next_required_input": result["next_required_input"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(payload)


if __name__ == "__main__":
    main()
