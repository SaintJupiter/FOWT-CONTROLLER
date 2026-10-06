"""Trace one legacy decision beside a first-lead physical ballast preview.

This is a validation-side bridge, not a controller entry point.  It retains
two independent records formed at the same decision instant:

* the unmodified legacy controller decision; and
* the full, zero-net-mass physical ballast endpoint implied by the first
  future physical load point, together with a 10-minute pump preview.

The physical result is deliberately discarded after the trace.  It is not a
legacy candidate, does not select an action, and never updates controller or
plant state.  Its purpose is to expose whether source time, physical load,
three-tank endpoint, and actuator state agree before a later controller
redesign attempts to compare any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from fowt_platform import (
    BallastModelSnapshot,
    CurrentPosturePassiveLoadDiagnostic,
    IncrementalState,
    diagnose_current_posture_passive_load,
    diagnose_generalized_load_forecast_ballast_redistribution,
)
from wind_prediction.controller_core import (
    ControlCoreConfig,
    ControlDecision,
    ControlObservation,
    decide_control_cycle,
)
from wind_prediction.forecast_evidence import ForecastEvidence, validate_forecast_evidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)
from wind_prediction.physical_forecast_endpoint_preview import (
    PhysicalForecastEndpointPreview,
    preview_physical_endpoint_fractions,
)
from wind_prediction.physical_execution_compensation import (
    PhysicalExecutionCompensationDiagnostic,
    diagnose_actual_execution_compensation,
)
from wind_prediction.physical_target_relation import (
    TankTargetEndpointRelation,
    compare_tank_target_to_physical_endpoint,
)


_MASS_TOLERANCE_KG = 1.0e-8


def _three(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three finite values") from exc
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _two(name: str, value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite values") from exc
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain two finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _positive_scalar(name: str, value: Any) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite scalar") from exc
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar


@dataclass(frozen=True)
class DecisionPhysicalShadowRecord:
    """One same-origin legacy decision and independent physical endpoint trace.

    ``physical_preview`` always represents the full endpoint at the first
    discrete future forecast point.  It is not an alternative selected by the
    legacy controller.  The record retains it only for source, time, state and
    actuator-consistency review.
    """

    legacy_decision: ControlDecision
    current_posture_passive_load: CurrentPosturePassiveLoadDiagnostic
    physical_preview: PhysicalForecastEndpointPreview
    physical_execution_compensation: PhysicalExecutionCompensationDiagnostic
    legacy_target_relation: TankTargetEndpointRelation
    observed_current_enu_downwind_air_velocity_mps: Any
    platform_runtime_provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.legacy_decision, ControlDecision):
            raise TypeError("legacy_decision must be ControlDecision")
        if not isinstance(
            self.current_posture_passive_load,
            CurrentPosturePassiveLoadDiagnostic,
        ):
            raise TypeError(
                "current_posture_passive_load must be CurrentPosturePassiveLoadDiagnostic"
            )
        if not isinstance(self.physical_preview, PhysicalForecastEndpointPreview):
            raise TypeError("physical_preview must be PhysicalForecastEndpointPreview")
        if not isinstance(
            self.physical_execution_compensation,
            PhysicalExecutionCompensationDiagnostic,
        ):
            raise TypeError(
                "physical_execution_compensation must be "
                "PhysicalExecutionCompensationDiagnostic"
            )
        if not isinstance(self.legacy_target_relation, TankTargetEndpointRelation):
            raise TypeError("legacy_target_relation must be TankTargetEndpointRelation")
        if not self.legacy_decision.context.forecast_available:
            raise ValueError("legacy_decision must use a future forecast")
        if self.physical_preview.lead_index != 0:
            raise ValueError("physical_preview must represent the first forecast lead")
        expected_lead_s = float(
            self.physical_preview.load_assembly.input_identity.forecast_sample_period_s
        )
        if not math.isclose(
            self.physical_preview.lead_time_s,
            expected_lead_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("physical_preview must use the first forecast lead time")
        if not math.isclose(
            self.physical_preview.reachability.execution_duration_s,
            expected_lead_s,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                "physical preview duration must equal the first forecast lead time"
            )
        if not math.isclose(
            self.physical_preview.endpoint_sample.fraction,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("physical_preview must retain the full diagnostic endpoint")
        if self.physical_execution_compensation.preview is not self.physical_preview:
            raise ValueError(
                "physical execution compensation must describe the retained physical preview"
            )
        if (
            self.current_posture_passive_load.platform_snapshot
            is not self.physical_execution_compensation.platform_snapshot
        ):
            raise ValueError(
            "current posture passive load and execution compensation must share one snapshot"
            )
        if not np.allclose(
            self.legacy_target_relation.actual_tank_masses_kg,
            self.physical_preview.timed_diagnostic.actual_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "legacy_target_relation must use the physical preview's actual tank state"
            )
        if not np.allclose(
            self.legacy_target_relation.proposed_target_tank_masses_kg,
            self.legacy_decision.target_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "legacy_target_relation must use the legacy decision's target"
            )
        if not np.allclose(
            self.legacy_target_relation.physical_endpoint_tank_masses_kg,
            self.physical_preview.endpoint_sample.hypothetical_tank_masses_kg,
            rtol=0.0,
            atol=_MASS_TOLERANCE_KG,
        ):
            raise ValueError(
                "legacy_target_relation must use the physical preview endpoint"
            )

        observed = _two(
            "observed_current_enu_downwind_air_velocity_mps",
            self.observed_current_enu_downwind_air_velocity_mps,
        )
        scale = float(self.physical_preview.load_assembly.parameters.rotor_plane_speed_scale)
        recovered_current = (
            self.physical_preview.load_assembly.current_rotor_plane_enu_downwind_mps
            / scale
        )
        if not np.allclose(
            observed,
            recovered_current,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "physical preview must use the observation's original current ENU wind"
            )
        provenance = str(self.platform_runtime_provenance).strip()
        if not provenance:
            raise ValueError("platform_runtime_provenance must be a non-empty string")
        if (
            self.physical_execution_compensation.platform_snapshot.runtime_provenance
            != provenance
        ):
            raise ValueError(
                "physical execution compensation must use the retained platform provenance"
            )
        object.__setattr__(
            self,
            "observed_current_enu_downwind_air_velocity_mps",
            observed,
        )
        object.__setattr__(self, "platform_runtime_provenance", provenance)

    @property
    def is_shadow_only(self) -> bool:
        """Always true: this record cannot change controller or plant state."""

        return True


def trace_first_forecast_physical_shadow(
    *,
    observation: ControlObservation,
    evidence: ForecastEvidence,
    controller_config: ControlCoreConfig,
    current_wind_source: str,
    current_wind_observation_time: str,
    rotor_load_parameters: ForecastRotorLoadParameters,
    rotor_operating_state: ForecastRotorOperatingState,
    platform_snapshot: BallastModelSnapshot,
    platform_state: IncrementalState,
) -> DecisionPhysicalShadowRecord:
    """Build a non-persistent same-origin control and physical trace.

    The only physical point examined is the first discrete future forecast
    record.  For the FINO1 ten-minute sequence this is ``+600 s`` and the
    actuator preview is exactly 600 s.  The helper intentionally offers no
    lead selection, endpoint fraction selection, ranking, or state update.
    """

    if not isinstance(observation, ControlObservation):
        raise TypeError("observation must be ControlObservation")
    if not isinstance(evidence, ForecastEvidence):
        raise TypeError("evidence must be ForecastEvidence")
    if not isinstance(controller_config, ControlCoreConfig):
        raise TypeError("controller_config must be ControlCoreConfig")
    if not isinstance(rotor_load_parameters, ForecastRotorLoadParameters):
        raise TypeError("rotor_load_parameters must be ForecastRotorLoadParameters")
    if not isinstance(rotor_operating_state, ForecastRotorOperatingState):
        raise TypeError("rotor_operating_state must be ForecastRotorOperatingState")
    if not isinstance(platform_snapshot, BallastModelSnapshot):
        raise TypeError("platform_snapshot must be BallastModelSnapshot")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be IncrementalState")
    expected_posture_deg = np.degrees(platform_state.position[[4, 3]])
    expected_posture_rate_deg_s = np.degrees(platform_state.velocity[[4, 3]])
    if not np.allclose(
        observation.posture_deg,
        expected_posture_deg,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "platform_state pitch-roll posture must match observation.posture_deg"
        )
    if not np.allclose(
        observation.posture_rate_deg_s,
        expected_posture_rate_deg_s,
        rtol=0.0,
        atol=1.0e-6,
    ):
        raise ValueError(
            "platform_state pitch-roll rate must match observation.posture_rate_deg_s"
        )
    validate_forecast_evidence(evidence)
    if not evidence.provides_future_preview:
        raise ValueError("evidence must provide a future preview")
    if evidence.origin_time is None:
        raise ValueError("evidence.origin_time is required for a source-bound shadow")
    if not str(current_wind_observation_time).strip():
        raise ValueError(
            "current_wind_observation_time is required for a source-bound shadow"
        )

    actual = _three(
        "observation.execution_state.actual_masses_kg",
        observation.execution_state.actual_masses_kg,
    )
    if not np.allclose(
        actual,
        platform_snapshot.actual_tank_masses_kg,
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "platform_snapshot actual tank masses must match the controller observation"
        )
    capacities = _three(
        "platform_snapshot.tank_capacities_kg",
        platform_snapshot.tank_capacities_kg,
    )
    if not np.allclose(
        capacities,
        float(controller_config.execution.tank_capacity_kg),
        rtol=0.0,
        atol=_MASS_TOLERANCE_KG,
    ):
        raise ValueError(
            "physical tank capacities must match the controller execution capacity"
        )

    # The legacy decision is evaluated exactly as it was before this sidecar.
    legacy_decision = decide_control_cycle(observation, evidence, controller_config)
    load_assembly = assemble_forecast_generalized_rotor_loads(
        forecast=evidence,
        current_enu_downwind_air_velocity_mps=observation.current_wind_uv_ms,
        current_wind_source=current_wind_source,
        current_wind_observation_time=current_wind_observation_time,
        parameters=rotor_load_parameters,
        operating_state=rotor_operating_state,
    )
    profile = diagnose_generalized_load_forecast_ballast_redistribution(
        forecast=load_assembly.load_forecast,
        matrices=platform_snapshot.matrices,
        actual_tank_masses_kg=actual,
        tank_capacities_kg=capacities,
        tank_coordinates_m=platform_snapshot.tank_coordinates_m,
        gravity_m_s2=platform_snapshot.gravity_m_s2,
    )
    first_lead_duration_s = float(load_assembly.load_forecast.lead_time_at(0))
    physical_preview = preview_physical_endpoint_fractions(
        load_assembly=load_assembly,
        lead_index=0,
        timed_diagnostic=profile[0],
        actual_tank_masses_kg=actual,
        tank_capacities_kg=capacities,
        fractions=(1.0,),
        execution_state=observation.execution_state,
        execution_config=controller_config.execution,
        execution_duration_s=first_lead_duration_s,
        execution_start_time=current_wind_observation_time,
    )[0]
    physical_execution_compensation = diagnose_actual_execution_compensation(
        preview=physical_preview,
        platform_snapshot=platform_snapshot,
    )
    current_posture_passive_load = diagnose_current_posture_passive_load(
        platform_snapshot=platform_snapshot,
        platform_state=platform_state,
    )
    legacy_target_relation = compare_tank_target_to_physical_endpoint(
        actual_tank_masses_kg=actual,
        proposed_target_tank_masses_kg=legacy_decision.target_masses_kg,
        physical_endpoint_tank_masses_kg=(
            physical_preview.endpoint_sample.hypothetical_tank_masses_kg
        ),
        tank_capacities_kg=capacities,
        tank_coordinates_m=platform_snapshot.tank_coordinates_m,
        gravity_m_s2=platform_snapshot.gravity_m_s2,
    )
    return DecisionPhysicalShadowRecord(
        legacy_decision=legacy_decision,
        current_posture_passive_load=current_posture_passive_load,
        physical_preview=physical_preview,
        physical_execution_compensation=physical_execution_compensation,
        legacy_target_relation=legacy_target_relation,
        observed_current_enu_downwind_air_velocity_mps=observation.current_wind_uv_ms,
        platform_runtime_provenance=platform_snapshot.runtime_provenance,
    )


__all__ = [
    "DecisionPhysicalShadowRecord",
    "trace_first_forecast_physical_shadow",
]
