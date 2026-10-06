"""Named design configuration and assembly for the preview MPC controller.

The optimizer owns the mathematical program, while validation scripts own
cases and reporting.  This module keeps the research-stage control design in
one place so changing an experiment cannot silently redefine the controller.
Runtime actuator reachability remains an input because it depends on the
current pump state and control-block duration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from fowt_platform import ThreeTankDifferentialModes
from wind_prediction.preview_mpc import (
    PreviewBlockModel,
    PreviewMPCConstraints,
    PreviewMPCController,
    PreviewMPCWeights,
)


_PRIORITY_NAMES = (
    "running_posture",
    "terminal_posture",
    "tank_movement",
    "tank_throughput",
    "movement_change",
    "posture_slack",
)

# Fixed objective reference from the active research working point:
# 1 m3/min for one 600 s control block at 1025 kg/m3. Runtime pump limits
# remain constraints and must not silently redefine the control preference.
_REFERENCE_TANK_MOVEMENT_SCALE_KG = 10_250.0


def _positive_vector(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} values must be positive")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nonnegative_vector(name: str, value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} values must be non-negative")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _nonnegative_priority(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class PreviewMPCDesign:
    """Research-stage control design independent of cases and pump state."""

    identity: str
    status: str
    maximum_abs_roll_pitch_rad: Any
    maximum_posture_slack_rad: Any
    absolute_maximum_abs_roll_pitch_rad: Any
    posture_objective_scale_rad: Any
    posture_slack_objective_scale_rad: float
    tank_movement_objective_scale_kg: float
    running_posture_priority: float
    terminal_posture_priority: float
    tank_movement_priority: float
    tank_throughput_priority: float
    movement_change_priority: float
    posture_slack_priority: float
    maximum_abs_working_model_scope_roll_pitch_rad: Any
    running_posture_reference_duration_s: float | None = None
    maximum_abs_total_tank_mass_equivalent_heave_m: float | None = None

    def __post_init__(self) -> None:
        identity = str(self.identity).strip()
        status = str(self.status).strip()
        if not identity:
            raise ValueError("identity must be non-empty")
        if not status:
            raise ValueError("status must be non-empty")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "status", status)
        object.__setattr__(
            self,
            "maximum_abs_roll_pitch_rad",
            _positive_vector(
                "maximum_abs_roll_pitch_rad",
                self.maximum_abs_roll_pitch_rad,
                (2,),
            ),
        )
        object.__setattr__(
            self,
            "maximum_posture_slack_rad",
            _nonnegative_vector(
                "maximum_posture_slack_rad",
                self.maximum_posture_slack_rad,
                (2,),
            ),
        )
        object.__setattr__(
            self,
            "absolute_maximum_abs_roll_pitch_rad",
            _positive_vector(
                "absolute_maximum_abs_roll_pitch_rad",
                self.absolute_maximum_abs_roll_pitch_rad,
                (2,),
            ),
        )
        object.__setattr__(
            self,
            "maximum_abs_working_model_scope_roll_pitch_rad",
            _positive_vector(
                "maximum_abs_working_model_scope_roll_pitch_rad",
                self.maximum_abs_working_model_scope_roll_pitch_rad,
                (2,),
            ),
        )
        object.__setattr__(
            self,
            "posture_objective_scale_rad",
            _positive_vector(
                "posture_objective_scale_rad",
                self.posture_objective_scale_rad,
                (2,),
            ),
        )
        for name in (
            "posture_slack_objective_scale_rad",
            "tank_movement_objective_scale_kg",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if np.any(
            self.maximum_abs_roll_pitch_rad + self.maximum_posture_slack_rad
            > self.absolute_maximum_abs_roll_pitch_rad + 1.0e-15
        ):
            raise ValueError(
                "nominal posture limit plus slack must not exceed the absolute limit"
            )
        if np.any(
            self.maximum_abs_working_model_scope_roll_pitch_rad
            > self.absolute_maximum_abs_roll_pitch_rad + 1.0e-15
        ):
            raise ValueError(
                "working model scope must not exceed the absolute posture limit"
            )
        for name in _PRIORITY_NAMES:
            field_name = f"{name}_priority"
            object.__setattr__(
                self,
                field_name,
                _nonnegative_priority(field_name, getattr(self, field_name)),
            )
        for name in (
            "running_posture_reference_duration_s",
            "maximum_abs_total_tank_mass_equivalent_heave_m",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite when set")
            object.__setattr__(self, name, value)

    @property
    def normalized_objective_priorities(self) -> dict[str, float]:
        return {
            name: float(getattr(self, f"{name}_priority"))
            for name in _PRIORITY_NAMES
        }

    def with_priority_multipliers(
        self,
        multipliers: Mapping[str, Any] | None,
    ) -> "PreviewMPCDesign":
        """Return a diagnostic variant without mutating the named design."""

        if multipliers is None:
            return self
        unknown = set(multipliers) - set(_PRIORITY_NAMES)
        if unknown:
            raise ValueError(
                f"unknown objective-priority multipliers: {sorted(unknown)}"
            )
        changes: dict[str, float] = {}
        for name, multiplier in multipliers.items():
            try:
                factor = float(multiplier)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "objective-priority multipliers must be finite and positive"
                ) from exc
            if not np.isfinite(factor) or factor <= 0.0:
                raise ValueError(
                    "objective-priority multipliers must be finite and positive"
                )
            changes[f"{name}_priority"] = float(
                getattr(self, f"{name}_priority") * factor
            )
        return replace(self, **changes)

    def _assemble_weights(self) -> PreviewMPCWeights:
        priorities = self.normalized_objective_priorities
        return PreviewMPCWeights.from_normalized_scales(
            roll_scale_rad=self.posture_objective_scale_rad[0],
            pitch_scale_rad=self.posture_objective_scale_rad[1],
            tank_movement_scale_kg=self.tank_movement_objective_scale_kg,
            posture_slack_scale_rad=self.posture_slack_objective_scale_rad,
            running_posture_priority=priorities["running_posture"],
            terminal_posture_priority=priorities["terminal_posture"],
            tank_movement_priority=priorities["tank_movement"],
            tank_throughput_priority=priorities["tank_throughput"],
            movement_change_priority=priorities["movement_change"],
            posture_slack_priority=priorities["posture_slack"],
        )

    def _assemble_constraints(
        self,
        *,
        maximum_abs_tank_mass_change_per_block_kg: Any,
    ) -> PreviewMPCConstraints:
        return PreviewMPCConstraints(
            maximum_abs_roll_rad=self.maximum_abs_roll_pitch_rad[0],
            maximum_abs_pitch_rad=self.maximum_abs_roll_pitch_rad[1],
            maximum_abs_tank_mass_change_per_block_kg=(
                maximum_abs_tank_mass_change_per_block_kg
            ),
            maximum_posture_slack_rad=self.maximum_posture_slack_rad,
        )

    def assemble_controller(
        self,
        *,
        block_model: PreviewBlockModel,
        ballast_modes: ThreeTankDifferentialModes,
        maximum_abs_tank_mass_change_per_block_kg: Any,
    ) -> PreviewMPCController:
        return PreviewMPCController(
            block_model=block_model,
            ballast_modes=ballast_modes,
            weights=self._assemble_weights(),
            constraints=self._assemble_constraints(
                maximum_abs_tank_mass_change_per_block_kg=(
                    maximum_abs_tank_mass_change_per_block_kg
                )
            ),
            running_posture_reference_duration_s=(
                self.running_posture_reference_duration_s
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        parameters = {
            "maximum_abs_roll_pitch_rad": (
                self.maximum_abs_roll_pitch_rad.tolist()
            ),
            "maximum_posture_slack_rad": (
                self.maximum_posture_slack_rad.tolist()
            ),
            "absolute_maximum_abs_roll_pitch_rad": (
                self.absolute_maximum_abs_roll_pitch_rad.tolist()
            ),
            "posture_objective_scale_rad": (
                self.posture_objective_scale_rad.tolist()
            ),
            "posture_slack_objective_scale_rad": (
                self.posture_slack_objective_scale_rad
            ),
            "tank_movement_objective_scale_kg": (
                self.tank_movement_objective_scale_kg
            ),
            "maximum_abs_working_model_scope_roll_pitch_rad": (
                self.maximum_abs_working_model_scope_roll_pitch_rad.tolist()
            ),
            "running_posture_reference_duration_s": (
                self.running_posture_reference_duration_s
            ),
            "maximum_abs_total_tank_mass_equivalent_heave_m": (
                self.maximum_abs_total_tank_mass_equivalent_heave_m
            ),
            "normalized_objective_priorities": (
                self.normalized_objective_priorities
            ),
        }
        parameter_sha256 = hashlib.sha256(
            json.dumps(
                parameters,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return {
            "identity": self.identity,
            "status": self.status,
            "parameter_sha256": parameter_sha256,
            "weight_design": (
                "physical_scales_with_explicit_dimensionless_priorities"
            ),
            "objective_time_contract": {
                "running_posture": (
                    "trapezoidal_integral_divided_by_fixed_reference_duration"
                    if self.running_posture_reference_duration_s is not None
                    else "trapezoidal_time_average_over_the_declared_horizon"
                ),
                "terminal_posture": "single_endpoint_penalty",
                "tank_movement": "sum_of_squared_tank_mass_change_per_block",
                "tank_throughput": "sum_of_absolute_tank_mass_change_per_block",
                "movement_change": (
                    "sum_of_squared_changes_between_adjacent_block_movements"
                ),
            },
            "objective_time_contract_status": (
                "fixed_reference_time_for_current_600s_block_and_3600s_horizon"
                if self.running_posture_reference_duration_s is not None
                else "explicit_current_prototype_definition_not_yet_approved_for_"
                "forecast_horizon_or_control_period_comparison"
            ),
            "running_posture_reference_duration_s": (
                self.running_posture_reference_duration_s
            ),
            "maximum_abs_total_tank_mass_equivalent_heave_m": (
                self.maximum_abs_total_tank_mass_equivalent_heave_m
            ),
            "maximum_abs_roll_pitch_deg": np.rad2deg(
                self.maximum_abs_roll_pitch_rad
            ).tolist(),
            "maximum_posture_slack_deg": np.rad2deg(
                self.maximum_posture_slack_rad
            ).tolist(),
            "posture_objective_scale_deg": np.rad2deg(
                self.posture_objective_scale_rad
            ).tolist(),
            "posture_slack_objective_scale_deg": float(
                np.rad2deg(self.posture_slack_objective_scale_rad)
            ),
            "tank_movement_objective_scale_kg": (
                self.tank_movement_objective_scale_kg
            ),
            "absolute_maximum_abs_roll_pitch_deg": np.rad2deg(
                self.absolute_maximum_abs_roll_pitch_rad
            ).tolist(),
            "maximum_abs_working_model_scope_roll_pitch_deg": np.rad2deg(
                self.maximum_abs_working_model_scope_roll_pitch_rad
            ).tolist(),
            "working_model_scope_status": (
                "conservative_research_scope_not_a_safety_limit_or_"
                "complete_dynamic_validation_claim"
            ),
            "normalized_objective_priorities": (
                self.normalized_objective_priorities
            ),
            "terminal_posture_cost_enabled": (
                self.terminal_posture_priority > 0.0
            ),
            "formal_recursive_feasibility_claimed": False,
        }


def research_preview_mpc_design_v1() -> PreviewMPCDesign:
    """Return the retained legacy comparison identity, not a calibrated optimum."""

    return PreviewMPCDesign(
        identity="research_preview_mpc_design_v1",
        status="stage_working_point_not_calibrated_or_optimal",
        maximum_abs_roll_pitch_rad=np.deg2rad([12.0, 12.0]),
        maximum_posture_slack_rad=np.deg2rad([3.0, 3.0]),
        absolute_maximum_abs_roll_pitch_rad=np.deg2rad([15.0, 15.0]),
        posture_objective_scale_rad=np.deg2rad([12.0, 12.0]),
        posture_slack_objective_scale_rad=float(np.deg2rad(3.0)),
        tank_movement_objective_scale_kg=153_750.0,
        running_posture_priority=1.0,
        terminal_posture_priority=1.0,
        tank_movement_priority=1.0e-2,
        tank_throughput_priority=5.0e-4,
        movement_change_priority=1.0e-3,
        posture_slack_priority=1.0e2,
        maximum_abs_working_model_scope_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
    )


def research_preview_mpc_design_v2() -> PreviewMPCDesign:
    """Return the scope-aligned working point used by active experiments.

    The QP posture domain and the provisional low-order model scope are both
    five degrees. Objective scales preserve the active 600 s, 1 m3/min
    working point while remaining independent of runtime actuator limits.
    """

    return PreviewMPCDesign(
        identity="research_preview_mpc_design_v2",
        status="scope_aligned_working_point_not_calibrated_or_optimal",
        maximum_abs_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
        maximum_posture_slack_rad=np.zeros(2),
        absolute_maximum_abs_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
        posture_objective_scale_rad=np.deg2rad([12.0, 12.0]),
        posture_slack_objective_scale_rad=float(np.deg2rad(3.0)),
        tank_movement_objective_scale_kg=_REFERENCE_TANK_MOVEMENT_SCALE_KG,
        running_posture_priority=1.0,
        terminal_posture_priority=1.0,
        tank_movement_priority=1.0e-2,
        tank_throughput_priority=5.0e-4,
        movement_change_priority=1.0e-3,
        posture_slack_priority=1.0e2,
        maximum_abs_working_model_scope_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
    )


def research_preview_mpc_design_v3() -> PreviewMPCDesign:
    """Return the fixed-time, explicit-total-mass-scope working design.

    The values define a reproducible research starting point. They are not a
    calibrated optimum and the equivalent-heave scope is a provisional model
    validity boundary rather than an operational safety limit.
    """

    return PreviewMPCDesign(
        identity="research_preview_mpc_design_v3",
        status=(
            "fixed_objective_time_and_mass_scope_working_point_"
            "not_calibrated_or_optimal"
        ),
        maximum_abs_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
        maximum_posture_slack_rad=np.zeros(2),
        absolute_maximum_abs_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
        posture_objective_scale_rad=np.deg2rad([12.0, 12.0]),
        posture_slack_objective_scale_rad=float(np.deg2rad(3.0)),
        tank_movement_objective_scale_kg=_REFERENCE_TANK_MOVEMENT_SCALE_KG,
        running_posture_priority=1.0,
        terminal_posture_priority=1.0,
        tank_movement_priority=0.0,
        tank_throughput_priority=5.0e-4,
        movement_change_priority=1.0e-3,
        posture_slack_priority=1.0e2,
        maximum_abs_working_model_scope_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
        running_posture_reference_duration_s=3_600.0,
        maximum_abs_total_tank_mass_equivalent_heave_m=0.01,
    )


def research_preview_mpc_design_v4() -> PreviewMPCDesign:
    """Return the first design with distinct control and model boundaries.

    Five degrees remains the nominal posture target. Pitch excursions up to
    ten degrees require an explicit penalized slack variable; ten degrees is
    the provisional pitch boundary supported by the existing source-consistent
    static comparison. Roll remains limited to five degrees because equivalent
    evidence is not yet available. This is a framework working point, not a
    calibrated operational limit.
    """

    return PreviewMPCDesign(
        identity="research_preview_mpc_design_v4",
        status=(
            "distinct_nominal_soft_and_model_boundaries_working_point_"
            "not_calibrated_or_optimal"
        ),
        maximum_abs_roll_pitch_rad=np.deg2rad([5.0, 5.0]),
        maximum_posture_slack_rad=np.deg2rad([0.0, 5.0]),
        absolute_maximum_abs_roll_pitch_rad=np.deg2rad([5.0, 10.0]),
        posture_objective_scale_rad=np.deg2rad([12.0, 12.0]),
        posture_slack_objective_scale_rad=float(np.deg2rad(3.0)),
        tank_movement_objective_scale_kg=_REFERENCE_TANK_MOVEMENT_SCALE_KG,
        running_posture_priority=1.0,
        terminal_posture_priority=1.0,
        tank_movement_priority=0.0,
        tank_throughput_priority=5.0e-4,
        movement_change_priority=1.0e-3,
        posture_slack_priority=1.0e2,
        maximum_abs_working_model_scope_roll_pitch_rad=np.deg2rad([5.0, 10.0]),
        running_posture_reference_duration_s=3_600.0,
        maximum_abs_total_tank_mass_equivalent_heave_m=0.01,
    )


__all__ = [
    "PreviewMPCDesign",
    "research_preview_mpc_design_v1",
    "research_preview_mpc_design_v2",
    "research_preview_mpc_design_v3",
    "research_preview_mpc_design_v4",
]
