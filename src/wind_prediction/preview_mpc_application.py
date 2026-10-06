"""Deterministic application assembly for one preview-MPC control cycle.

Validation scripts provide case data and source-bound forecast facts.  This
module binds those facts to one ballast-dependent platform snapshot, one MPC
design, and one physical pump configuration.  :meth:`PreviewMPCApplication.assemble_cycle`
builds the typed input without side effects, while
:meth:`PreviewMPCApplication.run_cycle` is the public assemble-and-execute entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fowt_platform import (
    BallastModelSnapshot,
    BallastRuntimeAssembly,
    GeneralizedLoadForecast,
    IncrementalState,
    RotorGeneralizedLoad,
    ThreeTankDifferentialModes,
    assemble_ballast_model_snapshot,
)

from .execution_rollout import ExecutionRolloutConfig, ExecutionRolloutState
from .preview_mpc import (
    PreviewBlockModel,
    PreviewDisturbanceBlocks,
    PreviewMPCController,
    assemble_preview_disturbance_blocks,
)
from .preview_mpc_control_cycle import (
    PreviewMPCControlCycleResult,
    PreviewMPCControlCycleInput,
    PreviewMPCPlanningSource,
    run_preview_mpc_control_cycle,
)
from .preview_mpc_design import PreviewMPCDesign
from .run_identity import sha256_json


def _readonly_finite_array(
    name: str,
    value: Any,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and contain finite values")
    result = np.array(array, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _interval_loads(
    name: str,
    value: Any | None,
    *,
    horizon_steps: int,
) -> np.ndarray:
    if value is None:
        result = np.zeros((horizon_steps, 6), dtype=float)
        result.setflags(write=False)
        return result
    return _readonly_finite_array(name, value, (horizon_steps, 6))


@dataclass(frozen=True)
class PreviewMPCSourceIdentity:
    """Forecast identity known before controller-visible loads are assembled."""

    source: str
    model_version: str
    origin_time: str
    forecast_mode: str
    source_record_sha256: str
    sample_period_s: float
    lead_times_s: Any
    uses_future_information: bool

    def __post_init__(self) -> None:
        for name in ("source", "model_version", "origin_time", "forecast_mode"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        digest = str(self.source_record_sha256).strip().lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(
                "source_record_sha256 must be a 64-character SHA-256 digest"
            )
        object.__setattr__(self, "source_record_sha256", digest)
        sample_period = float(self.sample_period_s)
        if not np.isfinite(sample_period) or sample_period <= 0.0:
            raise ValueError("sample_period_s must be finite and positive")
        object.__setattr__(self, "sample_period_s", sample_period)
        lead_times = np.asarray(self.lead_times_s, dtype=float).reshape(-1)
        if (
            lead_times.size == 0
            or not np.all(np.isfinite(lead_times))
            or np.any(lead_times <= 0.0)
            or np.any(np.diff(lead_times) <= 0.0)
        ):
            raise ValueError("lead_times_s must be finite, positive, and increasing")
        lead_times = np.array(lead_times, dtype=float, copy=True)
        lead_times.setflags(write=False)
        object.__setattr__(self, "lead_times_s", lead_times)
        if type(self.uses_future_information) is not bool:
            raise ValueError("uses_future_information must be a boolean")

    @property
    def horizon_steps(self) -> int:
        return int(self.lead_times_s.size)


@dataclass(frozen=True)
class PreviewMPCCycleAssembly:
    """Cycle-local objects derived from one coherent physical snapshot."""

    design: PreviewMPCDesign
    snapshot: BallastModelSnapshot
    block_model: PreviewBlockModel
    disturbance_blocks: PreviewDisturbanceBlocks
    controller: PreviewMPCController
    control_cycle_input: PreviewMPCControlCycleInput
    maximum_abs_tank_mass_change_per_block_kg: Any
    maximum_abs_total_tank_mass_deviation_from_reference_kg: float

    def __post_init__(self) -> None:
        for name, expected_type in (
            ("design", PreviewMPCDesign),
            ("snapshot", BallastModelSnapshot),
            ("block_model", PreviewBlockModel),
            ("disturbance_blocks", PreviewDisturbanceBlocks),
            ("controller", PreviewMPCController),
            ("control_cycle_input", PreviewMPCControlCycleInput),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        maximum_change = _readonly_finite_array(
            "maximum_abs_tank_mass_change_per_block_kg",
            self.maximum_abs_tank_mass_change_per_block_kg,
            (3,),
        )
        if np.any(maximum_change <= 0.0):
            raise ValueError(
                "maximum_abs_tank_mass_change_per_block_kg must be positive"
            )
        object.__setattr__(
            self,
            "maximum_abs_tank_mass_change_per_block_kg",
            maximum_change,
        )
        total_mass_scope = float(
            self.maximum_abs_total_tank_mass_deviation_from_reference_kg
        )
        if not np.isfinite(total_mass_scope) or total_mass_scope <= 0.0:
            raise ValueError(
                "maximum_abs_total_tank_mass_deviation_from_reference_kg "
                "must be positive and finite"
            )
        object.__setattr__(
            self,
            "maximum_abs_total_tank_mass_deviation_from_reference_kg",
            total_mass_scope,
        )


@dataclass(frozen=True)
class PreviewMPCApplication:
    """Stable research assembly shared by scripts and control cycles.

    Runtime platform identity, ballast geometry, reference tank state and the
    base control design are fixed here.  Case-specific observations, forecasts
    and realised states remain explicit inputs to :meth:`assemble_cycle`.
    """

    runtime_assembly: BallastRuntimeAssembly
    ballast_modes: ThreeTankDifferentialModes
    design: PreviewMPCDesign
    reference_tank_masses_kg: Any
    tank_capacities_kg: Any
    tank_coordinates_m: Any

    def __post_init__(self) -> None:
        for name, expected_type in (
            ("runtime_assembly", BallastRuntimeAssembly),
            ("ballast_modes", ThreeTankDifferentialModes),
            ("design", PreviewMPCDesign),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        reference = _readonly_finite_array(
            "reference_tank_masses_kg",
            self.reference_tank_masses_kg,
            (3,),
        )
        capacities = _readonly_finite_array(
            "tank_capacities_kg",
            self.tank_capacities_kg,
            (3,),
        )
        coordinates = _readonly_finite_array(
            "tank_coordinates_m",
            self.tank_coordinates_m,
            (3, 3),
        )
        if np.any(capacities <= 0.0):
            raise ValueError("tank capacities must be positive")
        if np.any(reference < 0.0) or np.any(reference > capacities):
            raise ValueError("reference tank masses must remain within capacities")
        if not np.allclose(
            coordinates,
            self.ballast_modes.tank_coordinates_m,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "ballast modes and application must use the same tank coordinates"
            )
        if not np.isclose(
            self.runtime_assembly.gravity_m_s2,
            self.ballast_modes.gravity_m_s2,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "ballast modes and platform runtime must use the same gravity"
            )
        for name, value in (
            ("reference_tank_masses_kg", reference),
            ("tank_capacities_kg", capacities),
            ("tank_coordinates_m", coordinates),
        ):
            object.__setattr__(self, name, value)

    def assemble_cycle(
        self,
        *,
        source_identity: PreviewMPCSourceIdentity,
        rotor_load_forecast: GeneralizedLoadForecast,
        platform_state: IncrementalState,
        execution_state: ExecutionRolloutState,
        execution_config: ExecutionRolloutConfig,
        planner_first_block_rotor_load: RotorGeneralizedLoad,
        execution_first_block_rotor_load: RotorGeneralizedLoad,
        execution_first_block_rotor_load_substeps: (
            tuple[RotorGeneralizedLoad, ...] | None
        ) = None,
        wave_interval_loads: Any | None = None,
        other_interval_loads: Any | None = None,
    ) -> PreviewMPCCycleAssembly:
        """Assemble one typed control input without solving or executing it."""

        for name, value, expected_type in (
            ("source_identity", source_identity, PreviewMPCSourceIdentity),
            ("rotor_load_forecast", rotor_load_forecast, GeneralizedLoadForecast),
            ("platform_state", platform_state, IncrementalState),
            ("execution_state", execution_state, ExecutionRolloutState),
            ("execution_config", execution_config, ExecutionRolloutConfig),
            (
                "planner_first_block_rotor_load",
                planner_first_block_rotor_load,
                RotorGeneralizedLoad,
            ),
            (
                "execution_first_block_rotor_load",
                execution_first_block_rotor_load,
                RotorGeneralizedLoad,
            ),
        ):
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        if source_identity.horizon_steps != rotor_load_forecast.horizon_steps:
            raise ValueError(
                "planning source and rotor-load forecast horizons must match"
            )
        if not np.allclose(
            source_identity.lead_times_s,
            rotor_load_forecast.lead_times_s,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "planning source and rotor-load forecast lead times must match"
            )
        if not np.allclose(
            self.tank_capacities_kg,
            float(execution_config.tank_capacity_kg),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(
                "application and pump execution must use the same tank capacities"
            )

        horizon = rotor_load_forecast.horizon_steps
        wave = _interval_loads(
            "wave_interval_loads",
            wave_interval_loads,
            horizon_steps=horizon,
        )
        other = _interval_loads(
            "other_interval_loads",
            other_interval_loads,
            horizon_steps=horizon,
        )
        block_duration_s = float(execution_config.block_duration_s)
        snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=self.runtime_assembly,
            actual_tank_masses_kg=execution_state.actual_masses_kg,
            reference_tank_masses_kg=self.reference_tank_masses_kg,
            tank_capacities_kg=self.tank_capacities_kg,
            tank_coordinates_m=self.tank_coordinates_m,
        )
        block_model = PreviewBlockModel(
            matrices=snapshot.matrices,
            generalized_load_per_mode_kg=(
                self.ballast_modes.generalized_load_per_mode_kg
            ),
            block_duration_s=block_duration_s,
        )
        disturbance_blocks = assemble_preview_disturbance_blocks(
            rotor_load_forecast=rotor_load_forecast,
            current_incremental_ballast_load=snapshot.incremental_ballast_load,
            block_duration_s=block_duration_s,
            wave_interval_loads=wave,
            other_interval_loads=other,
        )
        planning_source = PreviewMPCPlanningSource(
            source=source_identity.source,
            model_version=source_identity.model_version,
            origin_time=source_identity.origin_time,
            forecast_mode=source_identity.forecast_mode,
            source_record_sha256=source_identity.source_record_sha256,
            controller_visible_loads_sha256=sha256_json(
                disturbance_blocks.generalized_loads.tolist()
            ),
            sample_period_s=source_identity.sample_period_s,
            lead_times_s=source_identity.lead_times_s,
            uses_future_information=source_identity.uses_future_information,
        )

        maximum_scheduled_rate_m3_min = max(
            rate for _, rate in execution_config.pump_rate_schedule_m3_min
        )
        maximum_change = np.full(
            3,
            maximum_scheduled_rate_m3_min
            * block_duration_s
            / 60.0
            * execution_config.water_density_kg_m3,
            dtype=float,
        )
        if self.design.maximum_abs_total_tank_mass_equivalent_heave_m is None:
            maximum_total_mass_deviation = float(
                3.0
                * maximum_scheduled_rate_m3_min
                * execution_config.internal_step_s
                / 60.0
                * execution_config.water_density_kg_m3
            )
        else:
            heave_restoring_n_m = float(snapshot.matrices.restoring_stiffness[2, 2])
            maximum_total_mass_deviation = float(
                heave_restoring_n_m
                * self.design.maximum_abs_total_tank_mass_equivalent_heave_m
                / self.runtime_assembly.gravity_m_s2
            )
            if maximum_total_mass_deviation <= 0.0:
                raise ValueError(
                    "positive heave restoring stiffness is required to convert "
                    "the equivalent-heave total-mass scope"
                )
        controller = self.design.assemble_controller(
            block_model=block_model,
            ballast_modes=self.ballast_modes,
            maximum_abs_tank_mass_change_per_block_kg=maximum_change,
        )
        cycle_input = PreviewMPCControlCycleInput(
            planning_source=planning_source,
            controller=controller,
            ballast_modes=self.ballast_modes,
            platform_state=platform_state,
            execution_state=execution_state,
            execution_config=execution_config,
            runtime_assembly=self.runtime_assembly,
            reference_tank_masses_kg=self.reference_tank_masses_kg,
            tank_capacities_kg=self.tank_capacities_kg,
            tank_coordinates_m=self.tank_coordinates_m,
            generalized_disturbance_loads=(
                disturbance_blocks.generalized_loads
            ),
            planner_first_block_rotor_load=planner_first_block_rotor_load,
            execution_first_block_rotor_load=execution_first_block_rotor_load,
            current_wave_load=wave[0],
            current_other_load=other[0],
            maximum_abs_roll_pitch_rad=(
                self.design.absolute_maximum_abs_roll_pitch_rad
            ),
            maximum_abs_working_model_scope_roll_pitch_rad=(
                self.design.maximum_abs_working_model_scope_roll_pitch_rad
            ),
            maximum_abs_total_tank_mass_deviation_from_reference_kg=(
                maximum_total_mass_deviation
            ),
            execution_first_block_rotor_load_substeps=(
                execution_first_block_rotor_load_substeps
            ),
        )
        return PreviewMPCCycleAssembly(
            design=self.design,
            snapshot=snapshot,
            block_model=block_model,
            disturbance_blocks=disturbance_blocks,
            controller=controller,
            control_cycle_input=cycle_input,
            maximum_abs_tank_mass_change_per_block_kg=maximum_change,
            maximum_abs_total_tank_mass_deviation_from_reference_kg=(
                maximum_total_mass_deviation
            ),
        )

    def run_cycle(
        self,
        *,
        source_identity: PreviewMPCSourceIdentity,
        rotor_load_forecast: GeneralizedLoadForecast,
        platform_state: IncrementalState,
        execution_state: ExecutionRolloutState,
        execution_config: ExecutionRolloutConfig,
        planner_first_block_rotor_load: RotorGeneralizedLoad,
        execution_first_block_rotor_load: RotorGeneralizedLoad,
        execution_first_block_rotor_load_substeps: (
            tuple[RotorGeneralizedLoad, ...] | None
        ) = None,
        wave_interval_loads: Any | None = None,
        other_interval_loads: Any | None = None,
    ) -> tuple[PreviewMPCCycleAssembly, PreviewMPCControlCycleResult]:
        """Assemble and execute one source-bound receding-horizon cycle."""

        assembly = self.assemble_cycle(
            source_identity=source_identity,
            rotor_load_forecast=rotor_load_forecast,
            platform_state=platform_state,
            execution_state=execution_state,
            execution_config=execution_config,
            planner_first_block_rotor_load=planner_first_block_rotor_load,
            execution_first_block_rotor_load=execution_first_block_rotor_load,
            execution_first_block_rotor_load_substeps=(
                execution_first_block_rotor_load_substeps
            ),
            wave_interval_loads=wave_interval_loads,
            other_interval_loads=other_interval_loads,
        )
        return (
            assembly,
            run_preview_mpc_control_cycle(assembly.control_cycle_input),
        )


__all__ = [
    "PreviewMPCApplication",
    "PreviewMPCCycleAssembly",
    "PreviewMPCSourceIdentity",
]
