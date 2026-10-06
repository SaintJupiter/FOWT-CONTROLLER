"""Wind-prediction and ballast-control package.

Only the active preview-MPC application surface is advertised through
``__all__``. Historical convenience names remain lazily accessible for old
scripts, but they are compatibility paths rather than active controller APIs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "PreviewMPCApplication": (
        "preview_mpc_application",
        "PreviewMPCApplication",
    ),
    "PreviewMPCCycleAssembly": (
        "preview_mpc_application",
        "PreviewMPCCycleAssembly",
    ),
    "PreviewMPCSourceIdentity": (
        "preview_mpc_application",
        "PreviewMPCSourceIdentity",
    ),
    "PreviewMPCControlCycleInput": (
        "preview_mpc_control_cycle",
        "PreviewMPCControlCycleInput",
    ),
    "PreviewMPCControlCycleResult": (
        "preview_mpc_control_cycle",
        "PreviewMPCControlCycleResult",
    ),
    "PreviewMPCNoSafeCandidateError": (
        "preview_mpc_control_cycle",
        "PreviewMPCNoSafeCandidateError",
    ),
    "PreviewMPCFirstBlockCostReconciliation": (
        "preview_mpc_control_cycle",
        "PreviewMPCFirstBlockCostReconciliation",
    ),
    "PreviewMPCLifecycleCandidate": (
        "preview_mpc_control_cycle",
        "PreviewMPCLifecycleCandidate",
    ),
    "PreviewMPCSelectionReason": (
        "preview_mpc_control_cycle",
        "PreviewMPCSelectionReason",
    ),
    "PreviewMPCPlanningSource": (
        "preview_mpc_control_cycle",
        "PreviewMPCPlanningSource",
    ),
    "PreviewMPCTargetLifecycle": (
        "preview_mpc_control_cycle",
        "PreviewMPCTargetLifecycle",
    ),
    "run_preview_mpc_control_cycle": (
        "preview_mpc_control_cycle",
        "run_preview_mpc_control_cycle",
    ),
    "SourceBoundRotorPreview": (
        "source_bound_rotor_preview",
        "SourceBoundRotorPreview",
    ),
    "assemble_source_bound_nominal_rotor_preview": (
        "source_bound_rotor_preview",
        "assemble_source_bound_nominal_rotor_preview",
    ),
    "endpoint_average_rotor_load": (
        "source_bound_rotor_preview",
        "endpoint_average_rotor_load",
    ),
    "AttitudeBands": ("candidate_plan_evaluator", "AttitudeBands"),
    "BaselineScales": ("candidate_plan_evaluator", "BaselineScales"),
    "CandidatePlan": ("candidate_plan_evaluator", "CandidatePlan"),
    "CandidatePlanEvaluator": ("candidate_plan_evaluator", "CandidatePlanEvaluator"),
    "DEFAULT_STRUCTURE_PROFILES": (
        "candidate_plan_evaluator",
        "DEFAULT_STRUCTURE_PROFILES",
    ),
    "DEFAULT_WEIGHT_PROFILES": (
        "candidate_plan_evaluator",
        "DEFAULT_WEIGHT_PROFILES",
    ),
    "StructureProfile": ("candidate_plan_evaluator", "StructureProfile"),
    "WeightProfile": ("candidate_plan_evaluator", "WeightProfile"),
    "make_default_baseline_scales": (
        "candidate_plan_evaluator",
        "make_default_baseline_scales",
    ),
    "DatasetManifest": ("dataset_manifest", "DatasetManifest"),
    "DatasetValidationReport": ("dataset_manifest", "DatasetValidationReport"),
    "ForecastModelAdapter": ("forecast_adapter", "ForecastModelAdapter"),
    "ForecastResult": ("forecast_adapter", "ForecastResult"),
    "PersistenceMeanForecastAdapter": (
        "forecast_adapter",
        "PersistenceMeanForecastAdapter",
    ),
    "ForecastContract": ("forecast_contract", "ForecastContract"),
    "ForecastGeneralizedLoadAssembly": (
        "forecast_physical_load",
        "ForecastGeneralizedLoadAssembly",
    ),
    "ForecastPhysicalInputIdentity": (
        "forecast_physical_load",
        "ForecastPhysicalInputIdentity",
    ),
    "ForecastRotorLoadParameters": (
        "forecast_physical_load",
        "ForecastRotorLoadParameters",
    ),
    "ForecastRotorOperatingState": (
        "forecast_physical_load",
        "ForecastRotorOperatingState",
    ),
    "ForecastRotorKinematics": (
        "forecast_rotor_inflow",
        "ForecastRotorKinematics",
    ),
    "ForecastRotorRelativeInflow": (
        "forecast_rotor_inflow",
        "ForecastRotorRelativeInflow",
    ),
    "FuturePlatformKinematicsMode": (
        "forecast_physical_load",
        "FuturePlatformKinematicsMode",
    ),
    "FutureRotorOrientationMode": (
        "forecast_physical_load",
        "FutureRotorOrientationMode",
    ),
    "FORECAST_INTERVAL_LOAD_CONVENTION": (
        "forecast_platform_trajectory",
        "FORECAST_INTERVAL_LOAD_CONVENTION",
    ),
    "ForecastPlatformTrajectory": (
        "forecast_platform_trajectory",
        "ForecastPlatformTrajectory",
    ),
    "ForecastPlatformTrajectoryStep": (
        "forecast_platform_trajectory",
        "ForecastPlatformTrajectoryStep",
    ),
    "CurrentPlatformRhsDiagnostic": (
        "forecast_platform_rhs_diagnostic",
        "CurrentPlatformRhsDiagnostic",
    ),
    "ForecastPlatformRhsPointDiagnostic": (
        "forecast_platform_rhs_diagnostic",
        "ForecastPlatformRhsPointDiagnostic",
    ),
    "ForecastRhsBallastDiagnostic": (
        "forecast_rhs_ballast_diagnostic",
        "ForecastRhsBallastDiagnostic",
    ),
    "ForecastRhsBallastEndpointPreview": (
        "forecast_rhs_ballast_endpoint_preview",
        "ForecastRhsBallastEndpointPreview",
    ),
    "ForecastRhsBallastHorizonPreviews": (
        "forecast_rhs_ballast_endpoint_preview",
        "ForecastRhsBallastHorizonPreviews",
    ),
    "ForecastEndpointHeldResponse": (
        "forecast_endpoint_horizon_response",
        "ForecastEndpointHeldResponse",
    ),
    "ForecastEndpointHeldResponseChangeFromContinuation": (
        "forecast_endpoint_horizon_response",
        "ForecastEndpointHeldResponseChangeFromContinuation",
    ),
    "ForecastEndpointPostureAuthorization": (
        "forecast_endpoint_posture_authorization",
        "ForecastEndpointPostureAuthorization",
    ),
    "ForecastEndpointPostureAuthorizationReason": (
        "forecast_endpoint_posture_authorization",
        "ForecastEndpointPostureAuthorizationReason",
    ),
    "authorize_forecast_endpoint_from_posture_response": (
        "forecast_endpoint_posture_authorization",
        "authorize_forecast_endpoint_from_posture_response",
    ),
    "PhysicalCurrentCycleDecision": (
        "physical_current_cycle_decision",
        "PhysicalCurrentCycleDecision",
    ),
    "PhysicalCurrentCycleCandidate": (
        "physical_current_cycle_decision",
        "PhysicalCurrentCycleCandidate",
    ),
    "PhysicalCurrentCycleDecisionPolicy": (
        "physical_current_cycle_decision",
        "PhysicalCurrentCycleDecisionPolicy",
    ),
    "PhysicalCurrentCycleDecisionReason": (
        "physical_current_cycle_decision",
        "PhysicalCurrentCycleDecisionReason",
    ),
    "PhysicalCurrentCycleSelectionBasis": (
        "physical_current_cycle_decision",
        "PhysicalCurrentCycleSelectionBasis",
    ),
    "PhysicalCurrentCycleRoute": (
        "physical_current_cycle_decision",
        "PhysicalCurrentCycleRoute",
    ),
    "assemble_physical_current_cycle_decision": (
        "physical_current_cycle_decision",
        "assemble_physical_current_cycle_decision",
    ),
    "assemble_physical_current_cycle_decision_from_input": (
        "physical_current_cycle_decision",
        "assemble_physical_current_cycle_decision_from_input",
    ),
    "ForecastEndpointHeldResponseSet": (
        "forecast_endpoint_horizon_response",
        "ForecastEndpointHeldResponseSet",
    ),
    "assemble_forecast_endpoint_held_responses": (
        "forecast_endpoint_horizon_response",
        "assemble_forecast_endpoint_held_responses",
    ),
    "FirstForecastIntervalLifecycleRollout": (
        "forecast_first_interval_lifecycle_rollout",
        "FirstForecastIntervalLifecycleRollout",
    ),
    "FirstIntervalPhysicalDecisionInput": (
        "first_interval_physical_decision_input",
        "FirstIntervalPhysicalDecisionInput",
    ),
    "PhysicalForecastAdmission": (
        "physical_forecast_admission",
        "PhysicalForecastAdmission",
    ),
    "PhysicalForecastLeadReliabilityScope": (
        "physical_forecast_admission",
        "PhysicalForecastLeadReliabilityScope",
    ),
    "bind_physical_forecast_admission": (
        "physical_forecast_admission",
        "bind_physical_forecast_admission",
    ),
    "bind_physical_forecast_lead_reliability_scope": (
        "physical_forecast_admission",
        "bind_physical_forecast_lead_reliability_scope",
    ),
    "PhysicalLifecycleComparison": (
        "physical_lifecycle_comparison",
        "PhysicalLifecycleComparison",
    ),
    "PhysicalLifecycleChangeFromContinuation": (
        "physical_lifecycle_comparison",
        "PhysicalLifecycleChangeFromContinuation",
    ),
    "PhysicalLifecycleHorizonComparison": (
        "physical_lifecycle_comparison",
        "PhysicalLifecycleHorizonComparison",
    ),
    "PhysicalLifecycleHorizonOutcome": (
        "physical_lifecycle_comparison",
        "PhysicalLifecycleHorizonOutcome",
    ),
    "PhysicalLifecycleOutcome": (
        "physical_lifecycle_comparison",
        "PhysicalLifecycleOutcome",
    ),
    "assemble_physical_lifecycle_comparison_for_current_block": (
        "physical_lifecycle_comparison",
        "assemble_physical_lifecycle_comparison_for_current_block",
    ),
    "assemble_physical_lifecycle_horizon_comparison": (
        "physical_lifecycle_comparison",
        "assemble_physical_lifecycle_horizon_comparison",
    ),
    "PhysicalLifecycleSelectionFact": (
        "physical_lifecycle_selection",
        "PhysicalLifecycleSelectionFact",
    ),
    "PhysicalLifecycleSelectionReason": (
        "physical_lifecycle_selection",
        "PhysicalLifecycleSelectionReason",
    ),
    "PhysicalPostureLimits": (
        "physical_lifecycle_selection",
        "PhysicalPostureLimits",
    ),
    "HeldLifecycleForecastHorizonRollout": (
        "forecast_execution_block_lifecycle_rollout",
        "HeldLifecycleForecastHorizonRollout",
    ),
    "rollout_held_lifecycle_to_forecast_horizon": (
        "forecast_execution_block_lifecycle_rollout",
        "rollout_held_lifecycle_to_forecast_horizon",
    ),
    "assemble_physical_lifecycle_selection_for_current_block": (
        "physical_lifecycle_selection",
        "assemble_physical_lifecycle_selection_for_current_block",
    ),
    "select_physical_lifecycle_for_current_block": (
        "physical_lifecycle_selection",
        "select_physical_lifecycle_for_current_block",
    ),
    "PhysicalShadowCycleOutcome": (
        "physical_shadow_cycle",
        "PhysicalShadowCycleOutcome",
    ),
    "PhysicalShadowCycleRequest": (
        "physical_shadow_cycle",
        "PhysicalShadowCycleRequest",
    ),
    "assemble_physical_shadow_cycle": (
        "physical_shadow_cycle",
        "assemble_physical_shadow_cycle",
    ),
    "ForecastTrajectoryEnvironment": (
        "forecast_platform_trajectory",
        "ForecastTrajectoryEnvironment",
    ),
    "ForecastStageGrid": ("forecast_stage_grid", "ForecastStageGrid"),
    "ForecastStageWindow": ("forecast_stage_grid", "ForecastStageWindow"),
    "build_forecast_stage_grid": (
        "forecast_stage_grid",
        "build_forecast_stage_grid",
    ),
    "assemble_forecast_generalized_rotor_loads": (
        "forecast_physical_load",
        "assemble_forecast_generalized_rotor_loads",
    ),
    "assemble_forecast_rotor_relative_inflow": (
        "forecast_rotor_inflow",
        "assemble_forecast_rotor_relative_inflow",
    ),
    "rollout_frozen_forecast_platform_trajectory": (
        "forecast_platform_trajectory",
        "rollout_frozen_forecast_platform_trajectory",
    ),
    "rollout_first_forecast_interval_lifecycle": (
        "forecast_first_interval_lifecycle_rollout",
        "rollout_first_forecast_interval_lifecycle",
    ),
    "assemble_first_interval_physical_decision_input": (
        "first_interval_physical_decision_input",
        "assemble_first_interval_physical_decision_input",
    ),
    "assemble_first_interval_physical_cycle_input": (
        "physical_forecast_cycle_facts",
        "assemble_first_interval_physical_cycle_input",
    ),
    "assemble_replay_first_interval_physical_cycle_input": (
        "replay_physical_cycle_adapter",
        "assemble_replay_first_interval_physical_cycle_input",
    ),
    "ReplayPhysicalCycleEvidence": (
        "replay_physical_cycle_adapter",
        "ReplayPhysicalCycleEvidence",
    ),
    "assemble_replay_physical_cycle_evidence": (
        "replay_physical_cycle_adapter",
        "assemble_replay_physical_cycle_evidence",
    ),
    "assemble_replay_physical_rolling_cycle_input": (
        "replay_physical_cycle_adapter",
        "assemble_replay_physical_rolling_cycle_input",
    ),
    "assemble_resolved_physical_rolling_cycle_input": (
        "replay_physical_cycle_adapter",
        "assemble_resolved_physical_rolling_cycle_input",
    ),
    "ForecastExecutionBlockInterval": (
        "forecast_execution_block_lifecycle_rollout",
        "ForecastExecutionBlockInterval",
    ),
    "ForecastExecutionBlockLifecycleRollout": (
        "forecast_execution_block_lifecycle_rollout",
        "ForecastExecutionBlockLifecycleRollout",
    ),
    "rollout_current_execution_block_lifecycle": (
        "forecast_execution_block_lifecycle_rollout",
        "rollout_current_execution_block_lifecycle",
    ),
    "diagnose_current_platform_rhs": (
        "forecast_platform_rhs_diagnostic",
        "diagnose_current_platform_rhs",
    ),
    "diagnose_forecast_platform_rhs": (
        "forecast_platform_rhs_diagnostic",
        "diagnose_forecast_platform_rhs",
    ),
    "diagnose_forecast_rhs_ballast_redistribution": (
        "forecast_rhs_ballast_diagnostic",
        "diagnose_forecast_rhs_ballast_redistribution",
    ),
    "diagnose_forecast_horizon_rhs_ballast_redistributions": (
        "forecast_rhs_ballast_diagnostic",
        "diagnose_forecast_horizon_rhs_ballast_redistributions",
    ),
    "preview_forecast_rhs_ballast_endpoint": (
        "forecast_rhs_ballast_endpoint_preview",
        "preview_forecast_rhs_ballast_endpoint",
    ),
    "preview_forecast_rhs_ballast_horizon": (
        "forecast_rhs_ballast_endpoint_preview",
        "preview_forecast_rhs_ballast_horizon",
    ),
    "PhysicalEndpointReachability": (
        "physical_endpoint_reachability",
        "PhysicalEndpointReachability",
    ),
    "evaluate_physical_endpoint_reachability": (
        "physical_endpoint_reachability",
        "evaluate_physical_endpoint_reachability",
    ),
    "PhysicalExecutionCompensationDiagnostic": (
        "physical_execution_compensation",
        "PhysicalExecutionCompensationDiagnostic",
    ),
    "PhysicalOptionRhsComparison": (
        "physical_option_rhs_comparison",
        "PhysicalOptionRhsComparison",
    ),
    "PhysicalTargetLifecycleRhsDiagnostic": (
        "physical_option_rhs_comparison",
        "PhysicalTargetLifecycleRhsDiagnostic",
    ),
    "diagnose_actual_execution_compensation": (
        "physical_execution_compensation",
        "diagnose_actual_execution_compensation",
    ),
    "compare_reachable_physical_option_rhs": (
        "physical_option_rhs_comparison",
        "compare_reachable_physical_option_rhs",
    ),
    "diagnose_physical_target_lifecycle_rhs": (
        "physical_option_rhs_comparison",
        "diagnose_physical_target_lifecycle_rhs",
    ),
    "PhysicalForecastEndpointPreview": (
        "physical_forecast_endpoint_preview",
        "PhysicalForecastEndpointPreview",
    ),
    "preview_physical_endpoint_fractions": (
        "physical_forecast_endpoint_preview",
        "preview_physical_endpoint_fractions",
    ),
    "trace_physical_forecast_endpoint": (
        "physical_forecast_endpoint_preview",
        "trace_physical_forecast_endpoint",
    ),
    "PhysicalTargetLifecycle": (
        "physical_target_lifecycle",
        "PhysicalTargetLifecycle",
    ),
    "PhysicalTargetLifecycleFacts": (
        "physical_target_lifecycle",
        "PhysicalTargetLifecycleFacts",
    ),
    "PhysicalTargetLifecycleTrace": (
        "physical_target_lifecycle",
        "PhysicalTargetLifecycleTrace",
    ),
    "trace_physical_target_lifecycles": (
        "physical_target_lifecycle",
        "trace_physical_target_lifecycles",
    ),
    "trace_new_physical_target_lifecycle": (
        "physical_target_lifecycle",
        "trace_new_physical_target_lifecycle",
    ),
    "TankTargetEndpointRelation": (
        "physical_target_relation",
        "TankTargetEndpointRelation",
    ),
    "compare_tank_target_to_physical_endpoint": (
        "physical_target_relation",
        "compare_tank_target_to_physical_endpoint",
    ),
    "forecast_from_oracle_sample": (
        "forecast_contract",
        "forecast_from_oracle_sample",
    ),
    "speed_direction_from_uv": ("forecast_contract", "speed_direction_from_uv"),
    "validate_forecast_result": ("forecast_contract", "validate_forecast_result"),
    "AdvisoryOperatingPoint": ("far_event_advisory", "AdvisoryOperatingPoint"),
    "FarEventAdvisory": ("far_event_advisory", "FarEventAdvisory"),
    "FarEventAdvisoryResult": ("far_event_advisory", "FarEventAdvisoryResult"),
    "build_event_alarm_tables": ("far_event_advisory", "build_event_alarm_tables"),
    "build_scenario_table": ("far_event_advisory", "build_scenario_table"),
    "calibration_bins": ("far_event_advisory", "calibration_bins"),
    "far_event_label_from_uv": ("far_event_advisory", "far_event_label_from_uv"),
    "far_event_score_from_uv": ("far_event_advisory", "far_event_score_from_uv"),
    "pr_frontier": ("far_event_advisory", "pr_frontier"),
    "select_operating_point": ("far_event_advisory", "select_operating_point"),
    "ModelDecisionProvider": ("preview_trim_provider", "ModelDecisionProvider"),
    "ModelPreviewProvider": ("preview_trim_provider", "ModelPreviewProvider"),
    "OracleDecisionProvider": ("preview_trim_provider", "OracleDecisionProvider"),
    "OraclePreviewProvider": ("preview_trim_provider", "OraclePreviewProvider"),
    "Fino1ReplayDataset": ("replay_dataset", "Fino1ReplayDataset"),
    "ReplaySample": ("replay_dataset", "ReplaySample"),
    "ReplayTraceRow": ("replay_dataset", "ReplayTraceRow"),
}

_ACTIVE_PREVIEW_MPC_EXPORTS = {
    "PreviewMPCApplication",
    "PreviewMPCSourceIdentity",
    "PreviewMPCControlCycleResult",
    "PreviewMPCNoSafeCandidateError",
    "PreviewMPCPlanningSource",
    "PreviewMPCSelectionReason",
    "PreviewMPCTargetLifecycle",
    "SourceBoundRotorPreview",
    "assemble_source_bound_nominal_rotor_preview",
    "endpoint_average_rotor_load",
}


__all__ = sorted(_ACTIVE_PREVIEW_MPC_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
