"""Strict, versioned configuration loader for the compact controller."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .controller_core import ControlCoreConfig
from .execution_rollout import ExecutionRolloutConfig
from .forecast_action_policy import ForecastActionPolicyConfig


CONTROLLER_CONFIG_SCHEMA_VERSION = "controller_core.v2"


@dataclass(frozen=True)
class LoadedControllerConfiguration:
    config: ControlCoreConfig
    source_path: Path
    sha256: str


def _reject_unknown(name: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} configuration keys: {unknown}")


def _required_mapping(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    if name not in document:
        raise ValueError(f"missing required {name} configuration section")
    value = document[name]
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} configuration section must be a JSON object")
    return dict(value)


def _json_object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON configuration key: {key}")
        value[key] = item
    return value


def controller_config_to_dict(config: ControlCoreConfig) -> dict[str, Any]:
    controller = asdict(config)
    execution = controller.pop("execution")
    forecast_policy = controller.pop("forecast_policy")
    return {
        "schema_version": CONTROLLER_CONFIG_SCHEMA_VERSION,
        "controller": controller,
        "execution": execution,
        "forecast_policy": forecast_policy,
    }


def controller_config_digest(document: Mapping[str, Any]) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_controller_config(document: Mapping[str, Any]) -> ControlCoreConfig:
    if not isinstance(document, Mapping):
        raise ValueError("controller configuration must be a JSON object")
    _reject_unknown(
        "root",
        document,
        {"schema_version", "controller", "execution", "forecast_policy"},
    )
    if document.get("schema_version") != CONTROLLER_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            "unsupported controller configuration schema_version: "
            f"{document.get('schema_version')!r}"
        )
    controller_values = _required_mapping(document, "controller")
    execution_values = _required_mapping(document, "execution")
    policy_values = _required_mapping(document, "forecast_policy")

    controller_fields = {field.name for field in fields(ControlCoreConfig)} - {
        "execution",
        "forecast_policy",
    }
    execution_fields = {field.name for field in fields(ExecutionRolloutConfig)}
    policy_fields = {field.name for field in fields(ForecastActionPolicyConfig)}
    _reject_unknown("controller", controller_values, controller_fields)
    _reject_unknown("execution", execution_values, execution_fields)
    _reject_unknown("forecast_policy", policy_values, policy_fields)

    for name in ("deadband_deg", "posture_priority_envelope_deg", "stage_discounts"):
        if name in controller_values:
            controller_values[name] = tuple(controller_values[name])
    if "pump_rate_schedule_m3_min" in execution_values:
        execution_values["pump_rate_schedule_m3_min"] = tuple(
            tuple(point) for point in execution_values["pump_rate_schedule_m3_min"]
        )
    if "stage_event_keys" in policy_values:
        policy_values["stage_event_keys"] = tuple(policy_values["stage_event_keys"])

    execution = ExecutionRolloutConfig(**execution_values)
    forecast_policy = ForecastActionPolicyConfig(**policy_values)
    return ControlCoreConfig(
        **controller_values,
        execution=execution,
        forecast_policy=forecast_policy,
    )


def load_controller_config(path: str | Path) -> LoadedControllerConfiguration:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    document = json.loads(
        source_path.read_text(encoding="utf-8"),
        object_pairs_hook=_json_object_without_duplicates,
    )
    config = parse_controller_config(document)
    normalized = controller_config_to_dict(config)
    return LoadedControllerConfiguration(
        config=config,
        source_path=source_path,
        sha256=controller_config_digest(normalized),
    )


__all__ = [
    "CONTROLLER_CONFIG_SCHEMA_VERSION",
    "LoadedControllerConfiguration",
    "controller_config_digest",
    "controller_config_to_dict",
    "load_controller_config",
    "parse_controller_config",
]
