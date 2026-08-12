"""Wind-prediction and ballast-control package.

Historical convenience exports are loaded lazily. This keeps the compact V2
controller independent at import time while preserving existing script imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
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

__all__ = sorted(_LAZY_EXPORTS)


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
