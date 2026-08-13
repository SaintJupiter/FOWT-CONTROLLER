"""Versioned runtime inputs for the ballast planner."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .forecast_action_policy import (
    DEFAULT_STAGE_EVENT_KEYS,
    ForecastActionPolicyConfig,
)


_DEFAULT_PUMP_RATE_SCHEDULE_M3_MIN = (
    (0.0, 0.0),
    (200.0, 4.0),
    (300.0, 6.0),
    (500.0, 8.0),
    (700.0, 10.0),
    (1000.0, 12.0),
    (2000.0, 14.0),
    (3000.0, 15.0),
)


@dataclass(frozen=True)
class ExecutionRolloutRuntimeConfig:
    enabled: bool = False
    block_duration_s: float = 1200.0
    internal_step_s: float = 1.0
    water_density_kg_m3: float = 1025.0
    target_slew_enabled: bool = False
    target_slew_rate_m3_min: float = 10.0
    stop_error_kg: float = 300.0
    restart_error_kg: float = 500.0
    min_on_s: float = 20.0
    min_off_s: float = 12.0
    near_target_hold_s: float = 10.0
    ramp_up_m3_min_per_s: float = 2.0
    ramp_down_m3_min_per_s: float = 3.0
    max_pump_rate_m3_min: float = 15.0
    tank_capacity_kg: float = 1850.0 * 1025.0
    pump_rate_schedule_m3_min: tuple[tuple[float, float], ...] = (
        _DEFAULT_PUMP_RATE_SCHEDULE_M3_MIN
    )


@dataclass(frozen=True)
class TargetReplanRuntimeConfig:
    mode: str = "off"
    rate_noise_tolerance_deg_s: float = 0.002


@dataclass(frozen=True)
class PlannerRuntimeConfig:
    schema_version: str
    pressure_sign_multiplier: float
    default_discount_blocks: tuple[float, ...]
    pressure_aggregation_mode: str = "legacy_mean"
    lead_reliability_enabled: bool = False
    intrastage_stepwise_enabled: bool = False
    target_lifecycle_mode: str = "legacy"
    candidate_action_mode: str = "legacy_five"
    forecast_action_policy_mode: str = "off"
    forecast_action_policy: ForecastActionPolicyConfig = field(
        default_factory=ForecastActionPolicyConfig
    )
    target_replan_policy: TargetReplanRuntimeConfig = field(
        default_factory=TargetReplanRuntimeConfig
    )
    execution_rollout: ExecutionRolloutRuntimeConfig = (
        ExecutionRolloutRuntimeConfig()
    )


def validate_execution_target_refresh_contract(
    config: PlannerRuntimeConfig,
    *,
    event_reset_mode: str,
    bias_shape: str | None = None,
) -> None:
    """Reject runtime target semantics that the execution rollout cannot model.

    The execution-aware planner currently treats every non-hold candidate as a
    target update for the corresponding control cycle.  The provider must use
    the same rule; otherwise repeated actions are accumulated in preview while
    the real controller silently reuses a stale target.
    """

    if (
        config.execution_rollout.enabled
        and str(event_reset_mode) != "active_bucket"
    ):
        raise ValueError(
            "execution-aware planning requires event_reset_mode='active_bucket' "
            "so repeated active actions refresh the runtime target"
        )
    if (
        config.execution_rollout.enabled
        and bias_shape is not None
        and str(bias_shape) != "event_decay"
    ):
        raise ValueError(
            "execution-aware planning requires bias_shape='event_decay'; "
            "other bias modes bypass active-bucket target refresh"
        )
    if (
        config.target_lifecycle_mode == "planner_authoritative"
        and not config.execution_rollout.enabled
    ):
        raise ValueError(
            "planner-authoritative target lifecycle requires execution rollout "
            "so runtime commits the target evaluated during ranking"
        )


def _validated_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _validated_bool(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _non_negative_float(value: Any, *, field: str) -> float:
    parsed = _validated_float(value, field=field)
    if parsed < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def _positive_float(value: Any, *, field: str) -> float:
    parsed = _validated_float(value, field=field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be greater than zero")
    return parsed


def _parse_pump_rate_schedule(
    raw_schedule: Any,
    *,
    max_pump_rate_m3_min: float,
) -> tuple[tuple[float, float], ...]:
    field = "execution_rollout.pump_rate_schedule_m3_min"
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError(f"{field} must be a non-empty list")

    schedule: list[tuple[float, float]] = []
    for index, raw_point in enumerate(raw_schedule):
        point_field = f"{field}[{index}]"
        if not isinstance(raw_point, Mapping):
            raise ValueError(f"{point_field} must be an object")
        unknown = set(raw_point) - {"error_kg", "rate_m3_min"}
        missing = {"error_kg", "rate_m3_min"} - set(raw_point)
        if unknown:
            raise ValueError(
                f"{point_field} contains unknown fields: {sorted(unknown)}"
            )
        if missing:
            raise ValueError(
                f"{point_field} is missing fields: {sorted(missing)}"
            )
        error_kg = _non_negative_float(
            raw_point["error_kg"],
            field=f"{point_field}.error_kg",
        )
        rate_m3_min = _non_negative_float(
            raw_point["rate_m3_min"],
            field=f"{point_field}.rate_m3_min",
        )
        if rate_m3_min > max_pump_rate_m3_min:
            raise ValueError(
                f"{point_field}.rate_m3_min cannot exceed "
                "execution_rollout.max_pump_rate_m3_min"
            )
        schedule.append((error_kg, rate_m3_min))

    if schedule[0][0] != 0.0:
        raise ValueError(f"{field} must start at zero error")
    if any(
        later_error <= earlier_error
        for (earlier_error, _), (later_error, _) in zip(
            schedule,
            schedule[1:],
        )
    ):
        raise ValueError(f"{field} error thresholds must be strictly increasing")
    if any(
        later_rate < earlier_rate
        for (_, earlier_rate), (_, later_rate) in zip(
            schedule,
            schedule[1:],
        )
    ):
        raise ValueError(f"{field} pump rates must be non-decreasing")
    return tuple(schedule)


def _parse_execution_rollout(
    raw_rollout: Any,
) -> ExecutionRolloutRuntimeConfig:
    field = "execution_rollout"
    if not isinstance(raw_rollout, Mapping):
        raise ValueError(f"{field} must be an object")

    defaults = ExecutionRolloutRuntimeConfig()
    allowed_fields = {
        "enabled",
        "block_duration_s",
        "internal_step_s",
        "water_density_kg_m3",
        "target_slew_enabled",
        "target_slew_rate_m3_min",
        "stop_error_kg",
        "restart_error_kg",
        "min_on_s",
        "min_off_s",
        "near_target_hold_s",
        "ramp_up_m3_min_per_s",
        "ramp_down_m3_min_per_s",
        "max_pump_rate_m3_min",
        "tank_capacity_kg",
        "pump_rate_schedule_m3_min",
    }
    unknown = set(raw_rollout) - allowed_fields
    if unknown:
        raise ValueError(f"{field} contains unknown fields: {sorted(unknown)}")

    enabled = _validated_bool(
        raw_rollout.get("enabled", defaults.enabled),
        field=f"{field}.enabled",
    )
    block_duration_s = _positive_float(
        raw_rollout.get("block_duration_s", defaults.block_duration_s),
        field=f"{field}.block_duration_s",
    )
    internal_step_s = _positive_float(
        raw_rollout.get("internal_step_s", defaults.internal_step_s),
        field=f"{field}.internal_step_s",
    )
    if internal_step_s > block_duration_s:
        raise ValueError(
            "execution_rollout.internal_step_s cannot exceed block_duration_s"
        )
    water_density_kg_m3 = _positive_float(
        raw_rollout.get(
            "water_density_kg_m3",
            defaults.water_density_kg_m3,
        ),
        field=f"{field}.water_density_kg_m3",
    )
    target_slew_enabled = _validated_bool(
        raw_rollout.get(
            "target_slew_enabled",
            defaults.target_slew_enabled,
        ),
        field=f"{field}.target_slew_enabled",
    )
    target_slew_rate_m3_min = _non_negative_float(
        raw_rollout.get(
            "target_slew_rate_m3_min",
            defaults.target_slew_rate_m3_min,
        ),
        field=f"{field}.target_slew_rate_m3_min",
    )
    if target_slew_enabled and target_slew_rate_m3_min == 0.0:
        raise ValueError(
            "execution_rollout.target_slew_rate_m3_min must be greater than "
            "zero when target slew is enabled"
        )
    stop_error_kg = _non_negative_float(
        raw_rollout.get("stop_error_kg", defaults.stop_error_kg),
        field=f"{field}.stop_error_kg",
    )
    restart_error_kg = _non_negative_float(
        raw_rollout.get("restart_error_kg", defaults.restart_error_kg),
        field=f"{field}.restart_error_kg",
    )
    if restart_error_kg < stop_error_kg:
        raise ValueError(
            "execution_rollout.restart_error_kg cannot be smaller than "
            "stop_error_kg"
        )
    min_on_s = _non_negative_float(
        raw_rollout.get("min_on_s", defaults.min_on_s),
        field=f"{field}.min_on_s",
    )
    min_off_s = _non_negative_float(
        raw_rollout.get("min_off_s", defaults.min_off_s),
        field=f"{field}.min_off_s",
    )
    near_target_hold_s = _non_negative_float(
        raw_rollout.get(
            "near_target_hold_s",
            defaults.near_target_hold_s,
        ),
        field=f"{field}.near_target_hold_s",
    )
    ramp_up_m3_min_per_s = _positive_float(
        raw_rollout.get(
            "ramp_up_m3_min_per_s",
            defaults.ramp_up_m3_min_per_s,
        ),
        field=f"{field}.ramp_up_m3_min_per_s",
    )
    ramp_down_m3_min_per_s = _positive_float(
        raw_rollout.get(
            "ramp_down_m3_min_per_s",
            defaults.ramp_down_m3_min_per_s,
        ),
        field=f"{field}.ramp_down_m3_min_per_s",
    )
    max_pump_rate_m3_min = _positive_float(
        raw_rollout.get(
            "max_pump_rate_m3_min",
            defaults.max_pump_rate_m3_min,
        ),
        field=f"{field}.max_pump_rate_m3_min",
    )
    tank_capacity_kg = _positive_float(
        raw_rollout.get("tank_capacity_kg", defaults.tank_capacity_kg),
        field=f"{field}.tank_capacity_kg",
    )

    raw_schedule = raw_rollout.get("pump_rate_schedule_m3_min")
    if raw_schedule is None:
        schedule = defaults.pump_rate_schedule_m3_min
        if any(rate > max_pump_rate_m3_min for _, rate in schedule):
            raise ValueError(
                "execution_rollout.max_pump_rate_m3_min cannot be smaller "
                "than a default pump schedule rate"
            )
    else:
        schedule = _parse_pump_rate_schedule(
            raw_schedule,
            max_pump_rate_m3_min=max_pump_rate_m3_min,
        )

    return ExecutionRolloutRuntimeConfig(
        enabled=enabled,
        block_duration_s=block_duration_s,
        internal_step_s=internal_step_s,
        water_density_kg_m3=water_density_kg_m3,
        target_slew_enabled=target_slew_enabled,
        target_slew_rate_m3_min=target_slew_rate_m3_min,
        stop_error_kg=stop_error_kg,
        restart_error_kg=restart_error_kg,
        min_on_s=min_on_s,
        min_off_s=min_off_s,
        near_target_hold_s=near_target_hold_s,
        ramp_up_m3_min_per_s=ramp_up_m3_min_per_s,
        ramp_down_m3_min_per_s=ramp_down_m3_min_per_s,
        max_pump_rate_m3_min=max_pump_rate_m3_min,
        tank_capacity_kg=tank_capacity_kg,
        pump_rate_schedule_m3_min=schedule,
    )


def parse_planner_runtime_config(payload: Mapping[str, Any]) -> PlannerRuntimeConfig:
    schema_version = str(payload.get("schema_version", ""))
    if schema_version not in {"planner_runtime.v1", "planner_runtime.v2"}:
        raise ValueError(
            "Unsupported planner runtime schema: "
            f"{schema_version or '<missing>'}"
        )

    target_lifecycle_mode = str(payload.get("target_lifecycle_mode", "legacy"))
    if target_lifecycle_mode not in {"legacy", "planner_authoritative"}:
        raise ValueError(
            "target_lifecycle_mode must be 'legacy' or 'planner_authoritative'"
        )
    if (
        schema_version == "planner_runtime.v1"
        and target_lifecycle_mode != "legacy"
    ):
        raise ValueError(
            "planner_runtime.v1 cannot enable planner-authoritative lifecycle"
        )
    candidate_action_mode = str(
        payload.get("candidate_action_mode", "legacy_five")
    )
    if candidate_action_mode not in {
        "legacy_five",
        "explicit_target_lifecycle",
    }:
        raise ValueError(
            "candidate_action_mode must be 'legacy_five' or "
            "'explicit_target_lifecycle'"
        )
    if (
        schema_version == "planner_runtime.v1"
        and candidate_action_mode != "legacy_five"
    ):
        raise ValueError(
            "planner_runtime.v1 cannot enable explicit target-lifecycle actions"
        )

    pressure_sign = _validated_float(
        payload.get("pressure_sign_multiplier"),
        field="pressure_sign_multiplier",
    )
    if pressure_sign not in {-1.0, 1.0}:
        raise ValueError("pressure_sign_multiplier must be -1.0 or 1.0")

    raw_discounts = payload.get("default_discount_blocks")
    if not isinstance(raw_discounts, list) or not raw_discounts:
        raise ValueError("default_discount_blocks must be a non-empty list")
    discounts = tuple(
        _validated_float(value, field="default_discount_blocks")
        for value in raw_discounts
    )
    if any(value < 0.0 or value > 1.0 for value in discounts):
        raise ValueError("default_discount_blocks values must lie in [0, 1]")
    if any(later > earlier for earlier, later in zip(discounts, discounts[1:])):
        raise ValueError("default_discount_blocks must be non-increasing")

    evidence = payload.get("forecast_evidence", {})
    if not isinstance(evidence, Mapping):
        raise ValueError("forecast_evidence must be an object")
    pressure_aggregation_mode = str(
        evidence.get("pressure_aggregation_mode", "legacy_mean")
    )
    if pressure_aggregation_mode not in {"legacy_mean", "mean_step_force"}:
        raise ValueError(
            "forecast_evidence.pressure_aggregation_mode must be "
            "'legacy_mean' or 'mean_step_force'"
        )
    lead_reliability_enabled = _validated_bool(
        evidence.get("lead_reliability_enabled", False),
        field="forecast_evidence.lead_reliability_enabled",
    )
    intrastage_stepwise_enabled = _validated_bool(
        evidence.get("intrastage_stepwise_enabled", False),
        field="forecast_evidence.intrastage_stepwise_enabled",
    )
    if schema_version == "planner_runtime.v1" and (
        pressure_aggregation_mode != "legacy_mean"
        or lead_reliability_enabled
        or intrastage_stepwise_enabled
    ):
        raise ValueError("planner_runtime.v1 cannot enable forecast_evidence features")
    effective_pressure_mode = (
        "intrastage_stepwise"
        if intrastage_stepwise_enabled
        else pressure_aggregation_mode
    )

    raw_action_policy = payload.get("forecast_action_policy", {})
    if not isinstance(raw_action_policy, Mapping):
        raise ValueError("forecast_action_policy must be an object")
    action_policy_fields = {
        "mode",
        "stage_duration_s",
        "relative_speed_change_threshold",
        "minimum_speed_change_ms",
        "direction_consistency_min",
        "reversal_angle_deg",
        "continuous_reversal_points",
        "high_impact_reliability_min",
        "high_impact_event_probability_min",
        "stage_event_keys",
    }
    unknown_action_policy = set(raw_action_policy) - action_policy_fields
    if unknown_action_policy:
        raise ValueError(
            "forecast_action_policy contains unknown fields: "
            f"{sorted(unknown_action_policy)}"
        )
    forecast_action_policy_mode = str(raw_action_policy.get("mode", "off"))
    if forecast_action_policy_mode not in {"off", "shadow", "enforce"}:
        raise ValueError(
            "forecast_action_policy.mode must be 'off', 'shadow', or 'enforce'"
        )
    if schema_version == "planner_runtime.v1" and forecast_action_policy_mode != "off":
        raise ValueError("planner_runtime.v1 cannot enable forecast_action_policy")
    raw_stage_event_keys = raw_action_policy.get(
        "stage_event_keys",
        list(DEFAULT_STAGE_EVENT_KEYS),
    )
    if (
        not isinstance(raw_stage_event_keys, list)
        or not raw_stage_event_keys
        or any(not isinstance(value, str) or not value for value in raw_stage_event_keys)
    ):
        raise ValueError(
            "forecast_action_policy.stage_event_keys must be a non-empty list "
            "of strings"
        )
    continuous_reversal_points = raw_action_policy.get(
        "continuous_reversal_points",
        2,
    )
    if isinstance(continuous_reversal_points, bool) or not isinstance(
        continuous_reversal_points,
        int,
    ):
        raise ValueError(
            "forecast_action_policy.continuous_reversal_points must be an integer"
        )
    forecast_action_policy = ForecastActionPolicyConfig(
        enabled=forecast_action_policy_mode != "off",
        stage_duration_s=_positive_float(
            raw_action_policy.get("stage_duration_s", 1200.0),
            field="forecast_action_policy.stage_duration_s",
        ),
        relative_speed_change_threshold=_non_negative_float(
            raw_action_policy.get("relative_speed_change_threshold", 0.08),
            field="forecast_action_policy.relative_speed_change_threshold",
        ),
        minimum_speed_change_ms=_non_negative_float(
            raw_action_policy.get("minimum_speed_change_ms", 0.5),
            field="forecast_action_policy.minimum_speed_change_ms",
        ),
        direction_consistency_min=_validated_float(
            raw_action_policy.get("direction_consistency_min", 0.80),
            field="forecast_action_policy.direction_consistency_min",
        ),
        reversal_angle_deg=_positive_float(
            raw_action_policy.get("reversal_angle_deg", 120.0),
            field="forecast_action_policy.reversal_angle_deg",
        ),
        continuous_reversal_points=int(continuous_reversal_points),
        high_impact_reliability_min=_validated_float(
            raw_action_policy.get("high_impact_reliability_min", 0.65),
            field="forecast_action_policy.high_impact_reliability_min",
        ),
        high_impact_event_probability_min=_validated_float(
            raw_action_policy.get("high_impact_event_probability_min", 0.60),
            field="forecast_action_policy.high_impact_event_probability_min",
        ),
        stage_event_keys=tuple(raw_stage_event_keys),
    )
    if not 0.0 <= forecast_action_policy.direction_consistency_min <= 1.0:
        raise ValueError(
            "forecast_action_policy.direction_consistency_min must lie in [0, 1]"
        )
    if not 0.0 < forecast_action_policy.reversal_angle_deg <= 180.0:
        raise ValueError(
            "forecast_action_policy.reversal_angle_deg must lie in (0, 180]"
        )
    if forecast_action_policy.continuous_reversal_points < 2:
        raise ValueError(
            "forecast_action_policy.continuous_reversal_points must be at least 2"
        )
    if not 0.0 <= forecast_action_policy.high_impact_reliability_min <= 1.0:
        raise ValueError(
            "forecast_action_policy.high_impact_reliability_min must lie in [0, 1]"
        )
    if not 0.0 <= forecast_action_policy.high_impact_event_probability_min <= 1.0:
        raise ValueError(
            "forecast_action_policy.high_impact_event_probability_min must lie in [0, 1]"
        )

    raw_target_replan = payload.get("target_replan_policy", {})
    if not isinstance(raw_target_replan, Mapping):
        raise ValueError("target_replan_policy must be an object")
    unknown_target_replan = set(raw_target_replan) - {
        "mode",
        "rate_noise_tolerance_deg_s",
    }
    if unknown_target_replan:
        raise ValueError(
            "target_replan_policy contains unknown fields: "
            f"{sorted(unknown_target_replan)}"
        )
    target_replan_mode = str(raw_target_replan.get("mode", "off"))
    if target_replan_mode not in {"off", "shadow", "enforce"}:
        raise ValueError(
            "target_replan_policy.mode must be 'off', 'shadow', or 'enforce'"
        )
    if schema_version == "planner_runtime.v1" and target_replan_mode != "off":
        raise ValueError("planner_runtime.v1 cannot enable target_replan_policy")
    target_replan_policy = TargetReplanRuntimeConfig(
        mode=target_replan_mode,
        rate_noise_tolerance_deg_s=_non_negative_float(
            raw_target_replan.get("rate_noise_tolerance_deg_s", 0.002),
            field="target_replan_policy.rate_noise_tolerance_deg_s",
        ),
    )

    raw_rollout = payload.get("execution_rollout")
    if schema_version == "planner_runtime.v1":
        if raw_rollout is not None:
            if not isinstance(raw_rollout, Mapping):
                raise ValueError("execution_rollout must be an object")
            unknown = set(raw_rollout) - {"enabled"}
            if unknown:
                raise ValueError(
                    "planner_runtime.v1 cannot configure execution_rollout"
                )
            enabled = _validated_bool(
                raw_rollout.get("enabled", False),
                field="execution_rollout.enabled",
            )
            if enabled:
                raise ValueError(
                    "planner_runtime.v1 cannot enable execution_rollout"
                )
        execution_rollout = ExecutionRolloutRuntimeConfig(enabled=False)
    else:
        execution_rollout = (
            ExecutionRolloutRuntimeConfig(enabled=False)
            if raw_rollout is None
            else _parse_execution_rollout(raw_rollout)
        )
    if target_lifecycle_mode == "planner_authoritative" and not execution_rollout.enabled:
        raise ValueError(
            "planner-authoritative target lifecycle requires execution_rollout.enabled"
        )
    if execution_rollout.enabled and not math.isclose(
        execution_rollout.block_duration_s,
        forecast_action_policy.stage_duration_s,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "execution_rollout.block_duration_s must match "
            "forecast_action_policy.stage_duration_s so each candidate action "
            "is evaluated over its forecast stage"
        )
    if candidate_action_mode == "explicit_target_lifecycle":
        if target_lifecycle_mode != "planner_authoritative":
            raise ValueError(
                "explicit target-lifecycle actions require "
                "target_lifecycle_mode='planner_authoritative'"
            )
        if not execution_rollout.enabled:
            raise ValueError(
                "explicit target-lifecycle actions require execution_rollout.enabled"
            )
    if target_replan_policy.mode != "off":
        if candidate_action_mode != "explicit_target_lifecycle":
            raise ValueError(
                "target_replan_policy requires explicit target-lifecycle actions"
            )
        if not execution_rollout.enabled:
            raise ValueError(
                "target_replan_policy requires execution_rollout.enabled"
            )

    return PlannerRuntimeConfig(
        schema_version=schema_version,
        pressure_sign_multiplier=pressure_sign,
        default_discount_blocks=discounts,
        pressure_aggregation_mode=effective_pressure_mode,
        lead_reliability_enabled=lead_reliability_enabled,
        intrastage_stepwise_enabled=intrastage_stepwise_enabled,
        target_lifecycle_mode=target_lifecycle_mode,
        candidate_action_mode=candidate_action_mode,
        forecast_action_policy_mode=forecast_action_policy_mode,
        forecast_action_policy=forecast_action_policy,
        target_replan_policy=target_replan_policy,
        execution_rollout=execution_rollout,
    )


def load_planner_runtime_config(path: str | Path) -> PlannerRuntimeConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Planner runtime config not found: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Planner runtime config must contain a JSON object")
    return parse_planner_runtime_config(payload)
