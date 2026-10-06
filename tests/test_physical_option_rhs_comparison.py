import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import (
    BallastMassProperties,
    BallastRuntimeAssembly,
    IncrementalState,
    PlatformMatrices,
    assemble_ballast_model_snapshot,
    diagnose_generalized_load_forecast_ballast_redistribution,
    weight_stiffness_about_reference,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionTargetOperation,
    simulate_execution_step,
)
from wind_prediction.physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    PhysicalExecutionPlatformSubstep,
    advance_physical_execution_platform_path,
    advance_physical_execution_platform_substep,
)
from wind_prediction.controller_platform_handoff import (
    advance_controller_platform_path,
)
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
    assemble_forecast_generalized_rotor_loads,
)
from wind_prediction.forecast_platform_rhs_diagnostic import (
    diagnose_forecast_platform_rhs,
)
from wind_prediction.forecast_rhs_ballast_diagnostic import (
    diagnose_forecast_rhs_ballast_redistribution,
)
from wind_prediction.forecast_rhs_ballast_endpoint_preview import (
    preview_forecast_rhs_ballast_endpoint,
)
from wind_prediction.forecast_platform_trajectory import (
    ForecastTrajectoryEnvironment,
    rollout_frozen_forecast_platform_trajectory,
)
from wind_prediction.forecast_first_interval_lifecycle_rollout import (
    FirstForecastIntervalLifecycleRollout,
    rollout_first_forecast_interval_lifecycle,
)
from wind_prediction.first_interval_physical_decision_input import (
    assemble_first_interval_physical_decision_input,
)
from wind_prediction.physical_execution_compensation import (
    diagnose_actual_execution_compensation,
)
from wind_prediction.physical_forecast_endpoint_preview import (
    preview_physical_endpoint_fractions,
)
from wind_prediction.physical_option_rhs_comparison import (
    diagnose_physical_target_lifecycle_rhs,
    compare_reachable_physical_option_rhs,
)
from wind_prediction.physical_target_lifecycle import (
    PhysicalTargetLifecycle,
    trace_physical_target_lifecycles,
)


_ORIGIN = "2026-08-22T00:00:00"
_REFERENCE_MASSES = np.array([1_000.0, 1_000.0, 1_000.0])
_CAPACITIES = np.array([3_000.0, 3_000.0, 3_000.0])
_COORDINATES = np.array(
    [[-10.0, 0.0, -1.0], [5.0, 8.0, -1.0], [5.0, -8.0, -1.0]]
)


def _runtime_assembly(*, provenance: str = "physical_option_rhs_test"):
    properties = BallastMassProperties(
        total_mass_kg=100_000.0,
        center_of_mass_m=np.array([0.0, 0.0, -5.0]),
        inertia_about_reference_kg_m2=np.diag([1.0e7, 1.1e7, 1.2e7]),
    )
    return BallastRuntimeAssembly(
        base_matrices=PlatformMatrices(
            mass=np.diag([100_000.0, 100_000.0, 100_000.0, 1.0e7, 1.1e7, 1.2e7]),
            damping=np.diag([100.0, 100.0, 100.0, 1_000.0, 1_000.0, 1_000.0]),
            hydrostatic_stiffness=np.diag(
                [1_000.0, 1_000.0, 1_000.0, 100_000.0, 100_000.0, 100_000.0]
            ),
            mooring_stiffness=np.zeros((6, 6)),
            weight_stiffness=weight_stiffness_about_reference(
                total_mass_kg=properties.total_mass_kg,
                center_of_mass_m=properties.center_of_mass_m,
                gravity_m_s2=9.81,
            ),
        ),
        reference_mass_properties=properties,
        gravity_m_s2=9.81,
        provenance=provenance,
    )


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=1_200.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=0.1,
        tank_capacity_kg=float(_CAPACITIES[0]),
        internal_step_s=1.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 0.1), (10_000.0, 0.1)),
    )


def _prepared_records(
    *,
    execution_duration_s: float = 600.0,
    lead_reliability: tuple[float, float] = (0.9, 0.8),
    event_probs: dict[str, float] | None = None,
    future_uv_ms: np.ndarray | None = None,
):
    runtime = _runtime_assembly()
    baseline_snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime,
        actual_tank_masses_kg=_REFERENCE_MASSES,
        reference_tank_masses_kg=_REFERENCE_MASSES,
        tank_capacities_kg=_CAPACITIES,
        tank_coordinates_m=_COORDINATES,
    )
    future_uv = (
        np.array([[0.0, -10.0], [2.0, -12.0]], dtype=float)
        if future_uv_ms is None
        else np.asarray(future_uv_ms, dtype=float)
    )
    evidence = ForecastEvidence(
        source="physical_option_rhs_test",
        model_version="fixture",
        origin_time=_ORIGIN,
        sample_period_s=600.0,
        uv_ms=future_uv,
        lead_reliability=np.asarray(lead_reliability, dtype=float),
        event_probs=dict(event_probs or {}),
    )
    initial_state = IncrementalState(
        position=[0.0, 0.0, 0.0, 0.010, -0.015, 0.0],
        velocity=[0.0, 0.0, 0.0, -0.002, 0.003, 0.0],
    )
    assembly = assemble_forecast_generalized_rotor_loads(
        forecast=evidence,
        current_enu_downwind_air_velocity_mps=[0.0, -7.0],
        current_wind_source="physical_option_rhs_current_wind",
        current_wind_observation_time=_ORIGIN,
        parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=2.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 10.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        ),
        operating_state=ForecastRotorOperatingState(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=[0.8, 0.8],
            current_platform_reference_velocity_platform_mps=initial_state.velocity[:3],
            current_platform_angular_velocity_platform_radps=initial_state.velocity[3:],
            future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
            future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
            future_platform_kinematics_mode="frozen_zero",
            future_platform_kinematics_source="physical_option_rhs_fixture",
        ),
    )
    trajectory = rollout_frozen_forecast_platform_trajectory(
        load_assembly=assembly,
        platform_snapshot=baseline_snapshot,
        initial_state=initial_state,
        initial_state_time=_ORIGIN,
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            wave_interval_loads=np.zeros((2, 6)),
            other_interval_loads=np.zeros((2, 6)),
            forecast_origin_time=_ORIGIN,
            lead_times_s=[600.0, 1_200.0],
            provenance="physical_option_rhs_zero_nonrotor_fixture",
        ),
    )
    profile = diagnose_generalized_load_forecast_ballast_redistribution(
        forecast=assembly.load_forecast,
        matrices=baseline_snapshot.matrices,
        actual_tank_masses_kg=_REFERENCE_MASSES,
        tank_capacities_kg=_CAPACITIES,
        tank_coordinates_m=_COORDINATES,
        gravity_m_s2=9.81,
    )
    execution_state = ExecutionRolloutState(
        masses_kg=_REFERENCE_MASSES,
        target_masses_kg=_REFERENCE_MASSES,
        primary_target_kg=_REFERENCE_MASSES,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )
    preview = preview_physical_endpoint_fractions(
        load_assembly=assembly,
        lead_index=0,
        timed_diagnostic=profile[0],
        actual_tank_masses_kg=_REFERENCE_MASSES,
        tank_capacities_kg=_CAPACITIES,
        fractions=(1.0,),
        execution_state=execution_state,
        execution_config=_execution_config(),
        execution_duration_s=execution_duration_s,
        execution_start_time=_ORIGIN,
        trajectory=trajectory,
    )[0]
    compensation = diagnose_actual_execution_compensation(
        preview=preview,
        platform_snapshot=baseline_snapshot,
    )
    return runtime, trajectory, compensation


def _rhs_first_interval_input(runtime, trajectory, compensation):
    """Build the unranked first-interval facts from the direct RHS endpoint."""

    execution_state = compensation.preview.reachability.execution_start_state
    execution_config = compensation.preview.reachability.execution_config
    rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
    rhs_diagnostic = diagnose_forecast_rhs_ballast_redistribution(rhs_point=rhs_point)
    preview = preview_forecast_rhs_ballast_endpoint(
        rhs_diagnostic=rhs_diagnostic,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=rhs_point.lead_time_s,
        execution_start_time=_ORIGIN,
    )
    facts = trace_physical_target_lifecycles(
        execution_state=execution_state,
        execution_config=execution_config,
        execution_duration_s=rhs_point.lead_time_s,
        execution_start_time=_ORIGIN,
        new_track_preview=preview,
    )
    return assemble_first_interval_physical_decision_input(
        trajectory=trajectory,
        lifecycle_facts=facts,
        runtime_assembly=runtime,
        first_rhs_point=rhs_point,
    )


class PhysicalOptionRhsComparisonTests(unittest.TestCase):
    def test_rhs_derived_endpoint_reaches_first_interval_execution_without_old_fraction_route(self):
        runtime, trajectory, compensation = _prepared_records()
        execution_state = compensation.preview.reachability.execution_start_state
        execution_config = compensation.preview.reachability.execution_config
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        rhs_diagnostic = diagnose_forecast_rhs_ballast_redistribution(
            rhs_point=rhs_point
        )
        preview = preview_forecast_rhs_ballast_endpoint(
            rhs_diagnostic=rhs_diagnostic,
            execution_state=execution_state,
            execution_config=execution_config,
            execution_duration_s=rhs_point.lead_time_s,
            execution_start_time=_ORIGIN,
        )
        facts = trace_physical_target_lifecycles(
            execution_state=execution_state,
            execution_config=execution_config,
            execution_duration_s=rhs_point.lead_time_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )

        decision_input = assemble_first_interval_physical_decision_input(
            trajectory=trajectory,
            lifecycle_facts=facts,
            runtime_assembly=runtime,
            first_rhs_point=rhs_point,
        )
        rhs_response = diagnose_physical_target_lifecycle_rhs(
            rhs_point=rhs_point,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )

        self.assertIs(decision_input.new_track.lifecycle_trace, facts.new_track)
        self.assertIs(facts.new_track.source_preview, preview)
        np.testing.assert_allclose(
            decision_input.new_track.requested_target_masses_kg,
            rhs_diagnostic.hypothetical_tank_masses_kg,
        )
        np.testing.assert_allclose(
            rhs_response.reachable_snapshot.actual_tank_masses_kg,
            decision_input.new_track.reached_final_tank_masses_kg,
        )

    def test_first_interval_zero_tank_path_matches_frozen_tank_baseline(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
        )

        rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.continue_existing,
            runtime_assembly=runtime,
        )

        np.testing.assert_allclose(
            rollout.reached_final_tank_masses_kg,
            facts.continue_existing.execution_start_state.actual_masses_kg,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            rollout.execution_coupled_end_state.position,
            rollout.frozen_tank_baseline_end_state.position,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            rollout.execution_coupled_end_state.velocity,
            rollout.frozen_tank_baseline_end_state.velocity,
            atol=1.0e-10,
        )

    def test_first_interval_tracks_actual_pump_state_and_keeps_external_loads_fixed(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None

        rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )

        self.assertEqual(len(rollout.physical_path.substeps), 600)
        np.testing.assert_allclose(
            rollout.physical_path.substeps[0].ballast_snapshot.actual_tank_masses_kg,
            facts.new_track.execution_start_state.actual_masses_kg,
            atol=1.0e-12,
        )
        for previous, current in zip(
            rollout.physical_path.substeps,
            rollout.physical_path.substeps[1:],
        ):
            np.testing.assert_allclose(
                current.ballast_snapshot.actual_tank_masses_kg,
                previous.execution_step.state.actual_masses_kg,
                atol=1.0e-12,
            )
        frozen_loads = trajectory.steps[0].platform_step.loads
        for substep in rollout.physical_path.substeps:
            np.testing.assert_allclose(
                substep.platform_step.loads.wind,
                frozen_loads.wind,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                substep.platform_step.loads.wave,
                frozen_loads.wave,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                substep.platform_step.loads.other,
                frozen_loads.other,
                atol=1.0e-12,
            )
        np.testing.assert_allclose(
            rollout.reached_final_tank_masses_kg,
            facts.new_track.execution_step.state.actual_masses_kg,
            atol=1.0e-12,
        )

    def test_pump_lifecycle_state_changes_execution_path_not_forecast_physics(self):
        """Pump timing changes execution only after forecast physics is fixed."""

        runtime, trajectory, _ = _prepared_records()
        rhs_diagnostic = diagnose_forecast_rhs_ballast_redistribution(
            rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        )
        config = replace(_execution_config(), min_off_s=590.0)
        ready_state = ExecutionRolloutState(
            masses_kg=trajectory.platform_snapshot.actual_tank_masses_kg,
            target_masses_kg=trajectory.platform_snapshot.actual_tank_masses_kg,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            primary_target_kg=trajectory.platform_snapshot.actual_tank_masses_kg,
            pump_on_elapsed_s=np.zeros(3),
            pump_off_elapsed_s=np.full(3, 590.0),
            pump_near_target_s=np.zeros(3),
            pump_command_rates_m3_min=np.zeros(3),
            last_flow_directions=np.zeros(3),
        )
        delayed_state = replace(ready_state, pump_off_elapsed_s=np.zeros(3))

        def rollout_for(execution_state):
            preview = preview_forecast_rhs_ballast_endpoint(
                rhs_diagnostic=rhs_diagnostic,
                execution_state=execution_state,
                execution_config=config,
                execution_duration_s=rhs_diagnostic.lead_time_s,
                execution_start_time=_ORIGIN,
            )
            facts = trace_physical_target_lifecycles(
                execution_state=execution_state,
                execution_config=config,
                execution_duration_s=rhs_diagnostic.lead_time_s,
                execution_start_time=_ORIGIN,
                new_track_preview=preview,
            )
            assert facts.new_track is not None
            return preview, rollout_first_forecast_interval_lifecycle(
                trajectory=trajectory,
                lifecycle_trace=facts.new_track,
                runtime_assembly=runtime,
            )

        ready_preview, ready_rollout = rollout_for(ready_state)
        delayed_preview, delayed_rollout = rollout_for(delayed_state)

        self.assertIs(ready_preview.rhs_diagnostic, rhs_diagnostic)
        self.assertIs(delayed_preview.rhs_diagnostic, rhs_diagnostic)
        self.assertIs(ready_rollout.trajectory, trajectory)
        self.assertIs(delayed_rollout.trajectory, trajectory)
        self.assertIs(
            ready_rollout.frozen_tank_baseline_end_state,
            delayed_rollout.frozen_tank_baseline_end_state,
        )
        np.testing.assert_allclose(
            ready_preview.target_tank_masses_kg,
            delayed_preview.target_tank_masses_kg,
            atol=1.0e-12,
        )

        self.assertGreater(
            ready_rollout.transferred_volume_m3,
            delayed_rollout.transferred_volume_m3,
        )
        self.assertFalse(
            np.allclose(
                ready_rollout.reached_final_tank_masses_kg,
                delayed_rollout.reached_final_tank_masses_kg,
            )
        )
        self.assertGreater(
            np.linalg.norm(
                ready_rollout.execution_coupled_end_state.position
                - delayed_rollout.execution_coupled_end_state.position
            ),
            0.0,
        )

    def test_minimum_on_latch_changes_execution_path_not_forecast_physics(self):
        """An already running pump retains its dwell rule under frozen loads."""

        runtime, trajectory, _ = _prepared_records()
        masses = trajectory.platform_snapshot.actual_tank_masses_kg
        primary_target = masses + np.array([5.0, 0.0, 0.0])
        config = replace(
            _execution_config(),
            stop_error_kg=10.0,
            restart_error_kg=20.0,
            min_on_s=20.0,
        )
        running_state = ExecutionRolloutState(
            masses_kg=masses,
            target_masses_kg=primary_target,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.array([True, False, False]),
            primary_target_kg=primary_target,
            pump_on_elapsed_s=np.zeros(3),
            pump_off_elapsed_s=np.zeros(3),
            pump_near_target_s=np.zeros(3),
            pump_command_rates_m3_min=np.zeros(3),
            last_flow_directions=np.zeros(3),
        )
        idle_state = replace(
            running_state,
            pump_latched=np.zeros(3, dtype=bool),
        )

        def rollout_for(execution_state):
            facts = trace_physical_target_lifecycles(
                execution_state=execution_state,
                execution_config=config,
                execution_duration_s=600.0,
                execution_start_time=_ORIGIN,
            )
            return rollout_first_forecast_interval_lifecycle(
                trajectory=trajectory,
                lifecycle_trace=facts.continue_existing,
                runtime_assembly=runtime,
            )

        running_rollout = rollout_for(running_state)
        idle_rollout = rollout_for(idle_state)

        self.assertIs(running_rollout.trajectory, idle_rollout.trajectory)
        self.assertIs(
            running_rollout.frozen_tank_baseline_end_state,
            idle_rollout.frozen_tank_baseline_end_state,
        )
        self.assertGreater(
            running_rollout.transferred_volume_m3,
            idle_rollout.transferred_volume_m3,
        )
        self.assertGreater(
            running_rollout.reached_final_tank_masses_kg[0],
            idle_rollout.reached_final_tank_masses_kg[0],
        )

    def test_first_interval_replays_each_lifecycle_operation(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        traces = [facts.continue_existing, facts.release_to_current]
        assert facts.new_track is not None
        traces.append(facts.new_track)

        for trace in traces:
            rollout = rollout_first_forecast_interval_lifecycle(
                trajectory=trajectory,
                lifecycle_trace=trace,
                runtime_assembly=runtime,
            )
            np.testing.assert_allclose(
                rollout.reached_final_tank_masses_kg,
                trace.execution_step.state.actual_masses_kg,
                atol=1.0e-12,
            )

    def test_first_interval_rollout_exposes_existing_execution_facts(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None
        rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )

        self.assertIs(rollout.lifecycle, PhysicalTargetLifecycle.NEW_TRACK)
        np.testing.assert_allclose(
            rollout.requested_target_masses_kg,
            facts.new_track.requested_target_masses_kg,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            rollout.remaining_mass_to_target_kg,
            facts.new_track.remaining_mass_to_target_kg,
            atol=1.0e-12,
        )
        self.assertEqual(rollout.target_reached, facts.new_track.execution_step.target_reached)
        np.testing.assert_allclose(
            rollout.actual_tank_mass_delta_kg,
            rollout.reached_final_tank_masses_kg
            - facts.new_track.execution_start_state.actual_masses_kg,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            rollout.transferred_volume_m3,
            sum(
                substep.execution_step.transferred_volume_m3
                for substep in rollout.physical_path.substeps
            ),
        )
        self.assertAlmostEqual(
            rollout.aggregate_pump_active_time_s,
            sum(
                substep.execution_step.active_time_s
                for substep in rollout.physical_path.substeps
            ),
        )
        self.assertEqual(
            rollout.pump_start_count,
            sum(
                substep.execution_step.starts
                for substep in rollout.physical_path.substeps
            ),
        )

    def test_first_interval_decision_input_assembles_unranked_lifecycles(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )

        first_rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        decision_input = assemble_first_interval_physical_decision_input(
            trajectory=trajectory,
            lifecycle_facts=facts,
            runtime_assembly=runtime,
            first_rhs_point=first_rhs_point,
        )

        self.assertIs(decision_input.lifecycle_facts, facts)
        self.assertIs(decision_input.continue_existing.trajectory, trajectory)
        self.assertIs(decision_input.release_to_current.trajectory, trajectory)
        self.assertIsNotNone(decision_input.new_track)
        assert decision_input.new_track is not None
        self.assertIs(decision_input.new_track.trajectory, trajectory)
        self.assertIs(
            decision_input.continue_existing.lifecycle_trace.lifecycle,
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self.assertIs(
            decision_input.release_to_current.lifecycle_trace.lifecycle,
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        self.assertIs(
            decision_input.new_track.lifecycle_trace.lifecycle,
            PhysicalTargetLifecycle.NEW_TRACK,
        )
        self.assertIs(
            decision_input.response_for(PhysicalTargetLifecycle.CONTINUE_EXISTING),
            decision_input.continue_existing,
        )
        self.assertIs(
            decision_input.response_for("release_to_current"),
            decision_input.release_to_current,
        )
        self.assertIs(
            decision_input.response_for(PhysicalTargetLifecycle.NEW_TRACK),
            decision_input.new_track,
        )
        for response, diagnostic, trace in (
            (
                decision_input.continue_existing,
                decision_input.continue_existing_state_conditioned_rhs,
                facts.continue_existing,
            ),
            (
                decision_input.release_to_current,
                decision_input.release_to_current_state_conditioned_rhs,
                facts.release_to_current,
            ),
            (
                decision_input.new_track,
                decision_input.new_track_state_conditioned_rhs,
                facts.new_track,
            ),
        ):
            assert response is not None
            assert diagnostic is not None
            assert trace is not None
            self.assertIs(diagnostic.rhs_point, first_rhs_point)
            self.assertIs(
                diagnostic.rhs_point.platform_state,
                response.frozen_tank_baseline_end_state,
            )
            self.assertIsNot(
                diagnostic.rhs_point.platform_state,
                response.execution_coupled_end_state,
            )
            self.assertIs(diagnostic.runtime_assembly, runtime)
            self.assertIs(diagnostic.lifecycle_trace, trace)
            np.testing.assert_allclose(
                diagnostic.reachable_snapshot.actual_tank_masses_kg,
                response.reached_final_tank_masses_kg,
                atol=1.0e-12,
            )

    def test_first_interval_decision_input_allows_missing_new_track(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
        )

        decision_input = assemble_first_interval_physical_decision_input(
            trajectory=trajectory,
            lifecycle_facts=facts,
            runtime_assembly=runtime,
            first_rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0],
        )

        self.assertIsNone(facts.new_track)
        self.assertIsNone(decision_input.new_track)
        self.assertIsNone(decision_input.new_track_state_conditioned_rhs)
        with self.assertRaisesRegex(ValueError, "new_track is unavailable"):
            decision_input.response_for(PhysicalTargetLifecycle.NEW_TRACK)

    def test_first_interval_decision_input_rejects_wrong_origin_or_duration(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview

        wrong_origin_facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time="2026-08-22T00:10:00",
        )
        with self.assertRaisesRegex(ValueError, "execution start time must match"):
            assemble_first_interval_physical_decision_input(
                trajectory=trajectory,
                lifecycle_facts=wrong_origin_facts,
                runtime_assembly=runtime,
                first_rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0],
            )

        wrong_duration_facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=300.0,
            execution_start_time=_ORIGIN,
        )
        with self.assertRaisesRegex(ValueError, "execution duration must match"):
            assemble_first_interval_physical_decision_input(
                trajectory=trajectory,
                lifecycle_facts=wrong_duration_facts,
                runtime_assembly=runtime,
                first_rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0],
            )

    def test_first_interval_rejects_old_preview_rebound_to_another_trajectory(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        other_trajectory = rollout_frozen_forecast_platform_trajectory(
            load_assembly=trajectory.load_assembly,
            platform_snapshot=trajectory.platform_snapshot,
            initial_state=IncrementalState(
                position=trajectory.initial_state.position
                + np.array([0.0, 0.0, 0.0, 0.0, 0.001, 0.0]),
                velocity=trajectory.initial_state.velocity,
            ),
            initial_state_time=_ORIGIN,
            environment=trajectory.environment,
        )

        with self.assertRaisesRegex(ValueError, "matching forecast trajectory"):
            assemble_first_interval_physical_decision_input(
                trajectory=other_trajectory,
                lifecycle_facts=facts,
                runtime_assembly=runtime,
                first_rhs_point=diagnose_forecast_platform_rhs(
                    trajectory=other_trajectory
                )[0],
            )

    def test_old_preview_rejects_trajectory_with_different_ballast_context(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        snapshot = trajectory.platform_snapshot
        altered_coordinates = np.array(snapshot.tank_coordinates_m, copy=True)
        altered_coordinates[0, 0] += 0.25
        variants = (
            (
                replace(
                    snapshot,
                    tank_capacities_kg=snapshot.tank_capacities_kg + 1.0,
                ),
                "tank capacities",
            ),
            (
                replace(snapshot, tank_coordinates_m=altered_coordinates),
                "tank coordinates",
            ),
            (
                replace(snapshot, gravity_m_s2=snapshot.gravity_m_s2 - 0.01),
                "gravity",
            ),
        )

        for foreign_snapshot, message in variants:
            with self.subTest(context=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                replace(
                    preview,
                    trajectory=replace(
                        trajectory,
                        platform_snapshot=foreign_snapshot,
                    ),
                )

    def test_first_interval_decision_input_rejects_same_label_runtime_with_changed_dynamics(
        self,
    ):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
        )
        changed_runtime = replace(
            runtime,
            base_matrices=replace(
                runtime.base_matrices,
                damping=2.0 * runtime.base_matrices.damping,
            ),
        )

        with self.assertRaisesRegex(ValueError, "must reconstruct"):
            assemble_first_interval_physical_decision_input(
                trajectory=trajectory,
                lifecycle_facts=facts,
                runtime_assembly=changed_runtime,
                first_rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0],
            )

    def test_first_interval_decision_input_keeps_unreached_new_track_as_fact(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        slow_config = replace(
            _execution_config(),
            max_pump_rate_m3_min=1.0e-5,
            pump_rate_schedule_m3_min=((0.0, 1.0e-5), (10_000.0, 1.0e-5)),
        )
        profile = diagnose_generalized_load_forecast_ballast_redistribution(
            forecast=preview.load_assembly.load_forecast,
            matrices=trajectory.platform_snapshot.matrices,
            actual_tank_masses_kg=_REFERENCE_MASSES,
            tank_capacities_kg=_CAPACITIES,
            tank_coordinates_m=_COORDINATES,
            gravity_m_s2=9.81,
        )
        slow_preview = preview_physical_endpoint_fractions(
            load_assembly=preview.load_assembly,
            lead_index=0,
            timed_diagnostic=profile[0],
            actual_tank_masses_kg=_REFERENCE_MASSES,
            tank_capacities_kg=_CAPACITIES,
            fractions=(1.0,),
            execution_state=preview.reachability.execution_start_state,
            execution_config=slow_config,
            execution_duration_s=preview.lead_time_s,
            execution_start_time=_ORIGIN,
            trajectory=trajectory,
        )[0]
        self.assertFalse(slow_preview.reachability.endpoint_reached)
        facts = trace_physical_target_lifecycles(
            execution_state=slow_preview.reachability.execution_start_state,
            execution_config=slow_config,
            execution_duration_s=slow_preview.lead_time_s,
            execution_start_time=_ORIGIN,
            new_track_preview=slow_preview,
        )

        decision_input = assemble_first_interval_physical_decision_input(
            trajectory=trajectory,
            lifecycle_facts=facts,
            runtime_assembly=runtime,
            first_rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0],
        )

        self.assertIsNotNone(decision_input.new_track)
        assert decision_input.new_track is not None
        self.assertFalse(decision_input.new_track.lifecycle_trace.execution_step.target_reached)

    def test_first_interval_decision_input_rejects_rebound_lifecycle_response(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        decision_input = assemble_first_interval_physical_decision_input(
            trajectory=trajectory,
            lifecycle_facts=facts,
            runtime_assembly=runtime,
            first_rhs_point=diagnose_forecast_platform_rhs(trajectory=trajectory)[0],
        )

        with self.assertRaisesRegex(ValueError, "matching lifecycle trace"):
            replace(
                decision_input,
                release_to_current=decision_input.continue_existing,
            )

    def test_first_interval_decision_input_is_invariant_to_nonphysical_forecast_evidence(self):
        high_runtime, high_trajectory, high_compensation = _prepared_records(
            lead_reliability=(1.0, 1.0),
            event_probs={"event": 1.0},
        )
        low_runtime, low_trajectory, low_compensation = _prepared_records(
            lead_reliability=(0.0, 0.0),
            event_probs={"event": 0.0},
        )

        def decision_input_for(runtime, trajectory, compensation):
            preview = compensation.preview
            facts = trace_physical_target_lifecycles(
                execution_state=preview.reachability.execution_start_state,
                execution_config=preview.reachability.execution_config,
                execution_duration_s=preview.reachability.execution_duration_s,
                execution_start_time=_ORIGIN,
                new_track_preview=preview,
            )
            return assemble_first_interval_physical_decision_input(
                trajectory=trajectory,
                lifecycle_facts=facts,
                runtime_assembly=runtime,
                first_rhs_point=diagnose_forecast_platform_rhs(
                    trajectory=trajectory
                )[0],
            )

        high = decision_input_for(
            high_runtime,
            high_trajectory,
            high_compensation,
        )
        low = decision_input_for(low_runtime, low_trajectory, low_compensation)

        for high_response, low_response in zip(
            (
                high.continue_existing,
                high.release_to_current,
                high.new_track,
            ),
            (
                low.continue_existing,
                low.release_to_current,
                low.new_track,
            ),
        ):
            assert high_response is not None
            assert low_response is not None
            np.testing.assert_allclose(
                high_response.execution_coupled_end_state.position,
                low_response.execution_coupled_end_state.position,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                high_response.execution_coupled_end_state.velocity,
                low_response.execution_coupled_end_state.velocity,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                high_response.reached_final_tank_masses_kg,
                low_response.reached_final_tank_masses_kg,
                atol=1.0e-12,
            )

    def test_first_interval_physical_facts_change_with_future_wind_not_metadata(self):
        """A persistence counterfactual changes only the future wind records.

        The check is intentionally not a control-performance comparison.  It
        verifies that a changed future wind trajectory reaches the physical
        load, same-time RHS endpoint and ``new_track`` request, while the two
        target-lifecycle operations that depend only on the common pump state
        retain their request semantics.
        """

        forecast_runtime, forecast_trajectory, forecast_compensation = _prepared_records(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]], dtype=float)
        )
        persistence_runtime, persistence_trajectory, persistence_compensation = (
            _prepared_records(
                future_uv_ms=np.array([[0.0, -7.0], [0.0, -7.0]], dtype=float)
            )
        )

        forecast_input = _rhs_first_interval_input(
            forecast_runtime,
            forecast_trajectory,
            forecast_compensation,
        )
        persistence_input = _rhs_first_interval_input(
            persistence_runtime,
            persistence_trajectory,
            persistence_compensation,
        )

        forecast_load = forecast_trajectory.steps[0].rotor_load.generalized_load_platform
        persistence_load = persistence_trajectory.steps[0].rotor_load.generalized_load_platform
        self.assertFalse(np.allclose(forecast_load, persistence_load))

        assert forecast_input.new_track is not None
        assert persistence_input.new_track is not None
        self.assertFalse(
            np.allclose(
                forecast_input.new_track.requested_target_masses_kg,
                persistence_input.new_track.requested_target_masses_kg,
            )
        )

        np.testing.assert_allclose(
            forecast_input.continue_existing.requested_target_masses_kg,
            persistence_input.continue_existing.requested_target_masses_kg,
            atol=1.0e-12,
        )
        self.assertEqual(
            forecast_input.release_to_current.lifecycle_trace.execution_request.operation,
            persistence_input.release_to_current.lifecycle_trace.execution_request.operation,
        )

    def test_physical_path_non_integral_duration_matches_one_execution_step(self):
        runtime, trajectory, compensation = _prepared_records()
        duration_s = 2.5
        execution_state = compensation.preview.reachability.execution_start_state
        request = ExecutionRolloutRequest.track(
            compensation.preview.endpoint_sample.hypothetical_tank_masses_kg
        )
        path = advance_physical_execution_platform_path(
            platform_state=trajectory.initial_state,
            execution_state=execution_state,
            execution_request=request,
            execution_config=compensation.preview.reachability.execution_config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=trajectory.steps[0].rotor_load,
            wave_load=trajectory.steps[0].wave_load,
            other_load=trajectory.steps[0].other_load,
            duration_s=duration_s,
        )
        expected = simulate_execution_step(
            execution_state,
            request,
            replace(
                compensation.preview.reachability.execution_config,
                block_duration_s=duration_s,
            ),
        )

        self.assertEqual(len(path.substeps), 3)
        np.testing.assert_allclose(
            path.final_execution_state.actual_masses_kg,
            expected.state.actual_masses_kg,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            path.final_execution_state.target_masses_kg,
            expected.state.target_masses_kg,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            path.actual_tank_mass_delta_kg,
            expected.mass_delta_kg,
            atol=1.0e-12,
        )
        self.assertAlmostEqual(
            path.transferred_volume_m3,
            expected.transferred_volume_m3,
            places=12,
        )
        self.assertAlmostEqual(
            path.aggregate_pump_active_time_s,
            expected.active_time_s,
            places=12,
        )
        self.assertEqual(path.pump_start_count, expected.starts)
        self.assertEqual(path.pump_stop_count, expected.stops)
        self.assertEqual(
            path.pump_direction_switch_count,
            expected.direction_switches,
        )

    def test_first_interval_rejects_path_with_mismatched_execution_aggregate(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None
        rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )
        first = rollout.physical_path.substeps[0]
        inconsistent_first = replace(
            first,
            execution_step=replace(
                first.execution_step,
                transferred_volume_m3=first.execution_step.transferred_volume_m3 + 0.01,
            ),
        )
        inconsistent_path = PhysicalExecutionPlatformPath(
            substeps=(inconsistent_first, *rollout.physical_path.substeps[1:]),
            duration_s=rollout.physical_path.duration_s,
        )

        with self.assertRaisesRegex(ValueError, "transferred volume"):
            FirstForecastIntervalLifecycleRollout(
                trajectory=rollout.trajectory,
                lifecycle_trace=rollout.lifecycle_trace,
                runtime_assembly=rollout.runtime_assembly,
                physical_path=inconsistent_path,
                frozen_tank_baseline_end_state=rollout.frozen_tank_baseline_end_state,
                position_delta_from_frozen_tank_baseline=(
                    inconsistent_path.final_platform_state.position
                    - rollout.frozen_tank_baseline_end_state.position
                ),
                velocity_delta_from_frozen_tank_baseline=(
                    inconsistent_path.final_platform_state.velocity
                    - rollout.frozen_tank_baseline_end_state.velocity
                ),
            )

    def test_first_interval_rejects_latch_state_not_generated_by_execution(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None
        rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )
        first, second = rollout.physical_path.substeps[:2]
        tampered_first_state = replace(
            first.execution_step.state,
            pump_latched=np.zeros(3, dtype=bool),
        )
        tampered_first = replace(
            first,
            execution_step=replace(first.execution_step, state=tampered_first_state),
        )
        tampered_second = replace(
            second,
            start_execution_state=tampered_first_state,
        )
        tampered_path = PhysicalExecutionPlatformPath(
            substeps=(
                tampered_first,
                tampered_second,
                *rollout.physical_path.substeps[2:],
            ),
            duration_s=rollout.physical_path.duration_s,
        )

        with self.assertRaisesRegex(ValueError, "actuator outcome must follow"):
            FirstForecastIntervalLifecycleRollout(
                trajectory=rollout.trajectory,
                lifecycle_trace=rollout.lifecycle_trace,
                runtime_assembly=rollout.runtime_assembly,
                physical_path=tampered_path,
                frozen_tank_baseline_end_state=rollout.frozen_tank_baseline_end_state,
                position_delta_from_frozen_tank_baseline=(
                    tampered_path.final_platform_state.position
                    - rollout.frozen_tank_baseline_end_state.position
                ),
                velocity_delta_from_frozen_tank_baseline=(
                    tampered_path.final_platform_state.velocity
                    - rollout.frozen_tank_baseline_end_state.velocity
                ),
            )

    def test_physical_path_rejects_noncontinuous_substep_duration_or_state(self):
        runtime, trajectory, compensation = _prepared_records()
        path = advance_physical_execution_platform_path(
            platform_state=trajectory.initial_state,
            execution_state=compensation.preview.reachability.execution_start_state,
            execution_request=ExecutionRolloutRequest.track(
                compensation.preview.endpoint_sample.hypothetical_tank_masses_kg
            ),
            execution_config=compensation.preview.reachability.execution_config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=trajectory.steps[0].rotor_load,
            wave_load=trajectory.steps[0].wave_load,
            other_load=trajectory.steps[0].other_load,
            duration_s=2.5,
        )
        first, second, third = path.substeps
        with self.assertRaisesRegex(ValueError, "durations must sum"):
            PhysicalExecutionPlatformPath(
                substeps=(first, second),
                duration_s=path.duration_s,
            )

        with self.assertRaisesRegex(ValueError, "platform step must satisfy"):
            replace(
                third,
                platform_step=replace(
                    third.platform_step,
                    next_state=IncrementalState(
                        position=third.next_platform_state.position
                        + np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
                        velocity=third.next_platform_state.velocity,
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "pump start masses"):
            replace(
                first,
                start_execution_state=replace(
                    first.start_execution_state,
                    masses_kg=first.start_execution_state.actual_masses_kg
                    + np.array([1.0, 0.0, 0.0]),
                ),
            )

    def test_first_interval_rejects_path_with_noncanonical_substep_grid(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None
        reference_rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )
        compressed_substep = advance_physical_execution_platform_substep(
            platform_state=trajectory.initial_state,
            execution_state=facts.new_track.execution_start_state,
            execution_request=facts.new_track.execution_request,
            execution_config=facts.new_track.execution_config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=trajectory.steps[0].rotor_load,
            wave_load=trajectory.steps[0].wave_load,
            other_load=trajectory.steps[0].other_load,
            duration_s=facts.new_track.execution_duration_s,
        )
        compressed_path = PhysicalExecutionPlatformPath(
            substeps=(compressed_substep,),
            duration_s=facts.new_track.execution_duration_s,
        )
        np.testing.assert_allclose(
            compressed_path.final_execution_state.actual_masses_kg,
            reference_rollout.physical_path.final_execution_state.actual_masses_kg,
            atol=1.0e-12,
        )
        self.assertGreater(
            abs(
                compressed_path.final_platform_state.position[4]
                - reference_rollout.execution_coupled_end_state.position[4]
            ),
            1.0e-8,
        )

        with self.assertRaisesRegex(ValueError, "configured physical substeps"):
            FirstForecastIntervalLifecycleRollout(
                trajectory=reference_rollout.trajectory,
                lifecycle_trace=reference_rollout.lifecycle_trace,
                runtime_assembly=reference_rollout.runtime_assembly,
                physical_path=compressed_path,
                frozen_tank_baseline_end_state=(
                    reference_rollout.frozen_tank_baseline_end_state
                ),
                position_delta_from_frozen_tank_baseline=(
                    compressed_path.final_platform_state.position
                    - reference_rollout.frozen_tank_baseline_end_state.position
                ),
                velocity_delta_from_frozen_tank_baseline=(
                    compressed_path.final_platform_state.velocity
                    - reference_rollout.frozen_tank_baseline_end_state.velocity
                ),
            )

    def test_first_interval_rejects_rebound_initial_states_or_operation(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None
        rollout = rollout_first_forecast_interval_lifecycle(
            trajectory=trajectory,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )
        different_initial_state = IncrementalState(
            position=trajectory.initial_state.position
            + np.array([0.0, 0.0, 0.0, 0.0, 0.01, 0.0]),
            velocity=trajectory.initial_state.velocity,
        )
        wrong_initial_path = advance_physical_execution_platform_path(
            platform_state=different_initial_state,
            execution_state=facts.new_track.execution_start_state,
            execution_request=facts.new_track.execution_request,
            execution_config=facts.new_track.execution_config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=trajectory.steps[0].rotor_load,
            wave_load=trajectory.steps[0].wave_load,
            other_load=trajectory.steps[0].other_load,
            duration_s=facts.new_track.execution_duration_s,
        )
        with self.assertRaisesRegex(ValueError, "initial platform state"):
            FirstForecastIntervalLifecycleRollout(
                trajectory=rollout.trajectory,
                lifecycle_trace=rollout.lifecycle_trace,
                runtime_assembly=rollout.runtime_assembly,
                physical_path=wrong_initial_path,
                frozen_tank_baseline_end_state=rollout.frozen_tank_baseline_end_state,
                position_delta_from_frozen_tank_baseline=(
                    wrong_initial_path.final_platform_state.position
                    - rollout.frozen_tank_baseline_end_state.position
                ),
                velocity_delta_from_frozen_tank_baseline=(
                    wrong_initial_path.final_platform_state.velocity
                    - rollout.frozen_tank_baseline_end_state.velocity
                ),
            )

        rebound_operation_path = PhysicalExecutionPlatformPath(
            substeps=tuple(
                replace(
                    substep,
                    execution_step=replace(
                        substep.execution_step,
                        target_operation=ExecutionTargetOperation.RELEASE_TO_CURRENT,
                    ),
                )
                for substep in rollout.physical_path.substeps
            ),
            duration_s=rollout.physical_path.duration_s,
        )
        with self.assertRaisesRegex(ValueError, "target operation"):
            FirstForecastIntervalLifecycleRollout(
                trajectory=rollout.trajectory,
                lifecycle_trace=rollout.lifecycle_trace,
                runtime_assembly=rollout.runtime_assembly,
                physical_path=rebound_operation_path,
                frozen_tank_baseline_end_state=rollout.frozen_tank_baseline_end_state,
                position_delta_from_frozen_tank_baseline=(
                    rebound_operation_path.final_platform_state.position
                    - rollout.frozen_tank_baseline_end_state.position
                ),
                velocity_delta_from_frozen_tank_baseline=(
                    rebound_operation_path.final_platform_state.velocity
                    - rollout.frozen_tank_baseline_end_state.velocity
                ),
            )

    def test_first_interval_rejects_same_label_runtime_with_changed_dynamics(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
        )
        altered_damping = np.array(runtime.base_matrices.damping, copy=True)
        altered_damping[4, 4] += 1.0
        same_label_different_runtime = replace(
            runtime,
            base_matrices=replace(runtime.base_matrices, damping=altered_damping),
        )

        with self.assertRaisesRegex(ValueError, "must reconstruct"):
            rollout_first_forecast_interval_lifecycle(
                trajectory=trajectory,
                lifecycle_trace=facts.continue_existing,
                runtime_assembly=same_label_different_runtime,
            )

    def test_first_interval_rejects_execution_path_from_another_runtime(self):
        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        assert facts.new_track is not None
        altered_damping = np.array(runtime.base_matrices.damping, copy=True)
        altered_damping[5, 5] += 1.0
        foreign_runtime = replace(
            runtime,
            base_matrices=replace(runtime.base_matrices, damping=altered_damping),
        )
        foreign_path = advance_physical_execution_platform_path(
            platform_state=trajectory.initial_state,
            execution_state=facts.new_track.execution_start_state,
            execution_request=facts.new_track.execution_request,
            execution_config=facts.new_track.execution_config,
            runtime_assembly=foreign_runtime,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=trajectory.steps[0].rotor_load,
            wave_load=trajectory.steps[0].wave_load,
            other_load=trajectory.steps[0].other_load,
            duration_s=facts.new_track.execution_duration_s,
        )

        with self.assertRaisesRegex(ValueError, "configured runtime assembly"):
            FirstForecastIntervalLifecycleRollout(
                trajectory=trajectory,
                lifecycle_trace=facts.new_track,
                runtime_assembly=runtime,
                physical_path=foreign_path,
                frozen_tank_baseline_end_state=trajectory.steps[0].end_state,
                position_delta_from_frozen_tank_baseline=(
                    foreign_path.final_platform_state.position
                    - trajectory.steps[0].end_state.position
                ),
                velocity_delta_from_frozen_tank_baseline=(
                    foreign_path.final_platform_state.velocity
                    - trajectory.steps[0].end_state.velocity
                ),
            )

    def test_rebuilds_same_time_rhs_from_actual_reachable_tank_state(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]

        comparison = compare_reachable_physical_option_rhs(
            rhs_point=rhs_point,
            compensation=compensation,
            runtime_assembly=runtime,
        )

        np.testing.assert_allclose(
            comparison.reachable_snapshot.actual_tank_masses_kg,
            compensation.execution_final_tank_masses_kg,
        )
        np.testing.assert_allclose(
            comparison.reachable_loads.wind, rhs_point.loads.wind, atol=1.0e-12
        )
        np.testing.assert_allclose(
            comparison.reachable_loads.wave, rhs_point.loads.wave, atol=1.0e-12
        )
        np.testing.assert_allclose(
            comparison.reachable_loads.other, rhs_point.loads.other, atol=1.0e-12
        )
        np.testing.assert_allclose(
            comparison.reachable_loads.ballast,
            comparison.reachable_snapshot.incremental_ballast_load,
        )
        np.testing.assert_allclose(
            comparison.reachable_snapshot.matrices.mass
            @ comparison.reachable_state_derivative.velocity_rate,
            comparison.reachable_dynamic_rhs_generalized_load,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            comparison.delta_dynamic_rhs_from_frozen_baseline,
            (
                comparison.reachable_dynamic_rhs_generalized_load
                - rhs_point.dynamic_rhs_generalized_load
            ),
        )
        np.testing.assert_allclose(
            comparison.delta_acceleration_from_frozen_baseline,
            (
                comparison.reachable_state_derivative.velocity_rate
                - rhs_point.state_derivative.velocity_rate
            ),
        )
        self.assertGreater(
            np.linalg.norm(comparison.delta_dynamic_rhs_from_frozen_baseline),
            0.0,
        )

    def test_option_rhs_rejects_same_label_runtime_with_changed_dynamics(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        comparison = compare_reachable_physical_option_rhs(
            rhs_point=rhs_point,
            compensation=compensation,
            runtime_assembly=runtime,
        )
        altered_damping = np.array(runtime.base_matrices.damping, copy=True)
        altered_damping[4, 4] += 1.0
        same_label_different_runtime = replace(
            runtime,
            base_matrices=replace(runtime.base_matrices, damping=altered_damping),
        )

        with self.assertRaisesRegex(ValueError, "must reconstruct"):
            replace(comparison, runtime_assembly=same_label_different_runtime)

    def test_rejects_endpoint_preview_that_stops_before_matching_forecast_lead(self):
        runtime, trajectory, compensation = _prepared_records(execution_duration_s=30.0)
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]

        with self.assertRaisesRegex(ValueError, "pump preview through the matching lead time"):
            compare_reachable_physical_option_rhs(
                rhs_point=rhs_point,
                compensation=compensation,
                runtime_assembly=runtime,
            )

    def test_rejects_a_different_forecast_lead_or_runtime_assembly(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_points = diagnose_forecast_platform_rhs(trajectory=trajectory)

        with self.assertRaisesRegex(ValueError, "matching rhs forecast lead"):
            compare_reachable_physical_option_rhs(
                rhs_point=rhs_points[1],
                compensation=compensation,
                runtime_assembly=runtime,
            )

        with self.assertRaisesRegex(ValueError, "runtime_assembly provenance"):
            compare_reachable_physical_option_rhs(
                rhs_point=rhs_points[0],
                compensation=compensation,
                runtime_assembly=_runtime_assembly(provenance="other_runtime"),
            )

    def test_rejects_a_reachable_record_rebound_to_another_tank_snapshot(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        comparison = compare_reachable_physical_option_rhs(
            rhs_point=rhs_point,
            compensation=compensation,
            runtime_assembly=runtime,
        )

        with self.assertRaisesRegex(ValueError, "reachable_snapshot"):
            replace(
                comparison,
                reachable_snapshot=assemble_ballast_model_snapshot(
                    runtime_assembly=runtime,
                    actual_tank_masses_kg=_REFERENCE_MASSES,
                    reference_tank_masses_kg=_REFERENCE_MASSES,
                    tank_capacities_kg=_CAPACITIES,
                    tank_coordinates_m=_COORDINATES,
                ),
            )

    def test_endpoint_preview_matches_causally_ordered_platform_execution(self):
        """The same explicit target gives one actuator result in both paths.

        The frozen forecast trajectory and the P4 path intentionally use
        different tank-state histories, so this test does not compare their
        endpoint platform states.  It only checks the shared actuator
        semantics, per-substep tank handoff, and first-lead load identity.
        """

        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        execution_config = _execution_config()
        path = advance_controller_platform_path(
            platform_state=trajectory.initial_state,
            execution_state=preview.reachability.execution_start_state,
            execution_request=ExecutionRolloutRequest.track(
                preview.endpoint_sample.hypothetical_tank_masses_kg
            ),
            execution_config=execution_config,
            runtime_assembly=runtime,
            reference_tank_masses_kg=trajectory.platform_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=preview.load_assembly.future_rotor_loads[0],
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=preview.lead_time_s,
        )

        self.assertEqual(path.duration_s, preview.lead_time_s)
        self.assertEqual(path.duration_s, 600.0)
        self.assertEqual(
            preview.execution_start_time,
            preview.input_identity.forecast_origin_time,
        )
        np.testing.assert_allclose(
            path.substeps[0].ballast_snapshot.actual_tank_masses_kg,
            preview.reachability.execution_start_state.actual_masses_kg,
        )
        for previous, current in zip(path.substeps, path.substeps[1:]):
            np.testing.assert_allclose(
                current.ballast_snapshot.actual_tank_masses_kg,
                previous.next_execution_state.actual_masses_kg,
            )
        for substep in path.substeps:
            np.testing.assert_allclose(
                substep.platform_step.loads.wind,
                rhs_point.loads.wind,
            )
            np.testing.assert_allclose(substep.platform_step.loads.wave, np.zeros(6))
            np.testing.assert_allclose(substep.platform_step.loads.other, np.zeros(6))

        actual = path.final_execution_state
        expected = preview.reachability.execution_step.state
        for attribute in (
            "actual_masses_kg",
            "rate_limited_target_kg",
            "primary_target_masses_kg",
            "signed_flow_m3_min",
            "pump_on_elapsed_s",
            "pump_off_elapsed_s",
            "pump_near_target_s",
            "pump_command_rates_m3_min",
            "last_flow_directions",
        ):
            np.testing.assert_allclose(
                getattr(actual, attribute),
                getattr(expected, attribute),
            )
        np.testing.assert_array_equal(actual.pump_latched, expected.pump_latched)
        np.testing.assert_allclose(
            actual.actual_masses_kg,
            compensation.execution_final_tank_masses_kg,
        )

    def test_each_target_lifecycle_matches_causally_ordered_platform_execution(self):
        """Each unranked lifecycle request has one actuator result in P4."""

        runtime, trajectory, compensation = _prepared_records()
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        traces = (
            facts.continue_existing,
            facts.release_to_current,
            facts.new_track,
        )
        for trace in traces:
            path = advance_controller_platform_path(
                platform_state=trajectory.initial_state,
                execution_state=trace.execution_start_state,
                execution_request=trace.execution_request,
                execution_config=trace.execution_config,
                runtime_assembly=runtime,
                reference_tank_masses_kg=(
                    trajectory.platform_snapshot.reference_tank_masses_kg
                ),
                tank_capacities_kg=trajectory.platform_snapshot.tank_capacities_kg,
                tank_coordinates_m=trajectory.platform_snapshot.tank_coordinates_m,
                rotor_load=preview.load_assembly.future_rotor_loads[0],
                wave_load=np.zeros(6),
                other_load=np.zeros(6),
                duration_s=trace.execution_duration_s,
            )
            self.assertEqual(path.duration_s, trace.execution_duration_s)
            actual = path.final_execution_state
            expected = trace.execution_step.state
            for attribute in (
                "actual_masses_kg",
                "rate_limited_target_kg",
                "primary_target_masses_kg",
                "signed_flow_m3_min",
                "pump_on_elapsed_s",
                "pump_off_elapsed_s",
                "pump_near_target_s",
                "pump_command_rates_m3_min",
                "last_flow_directions",
            ):
                np.testing.assert_allclose(
                    getattr(actual, attribute),
                    getattr(expected, attribute),
                )
            np.testing.assert_array_equal(actual.pump_latched, expected.pump_latched)

    def test_lifecycle_traces_rebuild_at_one_first_forecast_rhs_point(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )

        diagnostics = tuple(
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_point,
                lifecycle_trace=trace,
                runtime_assembly=runtime,
            )
            for trace in (
                facts.continue_existing,
                facts.release_to_current,
                facts.new_track,
            )
        )

        for trace, diagnostic in zip(
            (
                facts.continue_existing,
                facts.release_to_current,
                facts.new_track,
            ),
            diagnostics,
        ):
            self.assertIs(diagnostic.lifecycle_trace, trace)
            np.testing.assert_allclose(
                diagnostic.reachable_snapshot.actual_tank_masses_kg,
                trace.actual_final_masses_kg,
            )
            np.testing.assert_allclose(
                diagnostic.reachable_loads.wind,
                rhs_point.loads.wind,
            )
            np.testing.assert_allclose(
                diagnostic.reachable_loads.wave,
                rhs_point.loads.wave,
            )
            np.testing.assert_allclose(
                diagnostic.reachable_loads.other,
                rhs_point.loads.other,
            )
            np.testing.assert_allclose(
                diagnostic.reachable_snapshot.matrices.mass
                @ diagnostic.reachable_state_derivative.velocity_rate,
                diagnostic.reachable_dynamic_rhs_generalized_load,
                atol=1.0e-10,
            )

    def test_lifecycle_labels_do_not_change_rhs_when_final_masses_match(self):
        runtime, trajectory, _ = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        state = ExecutionRolloutState(
            masses_kg=_REFERENCE_MASSES,
            target_masses_kg=_REFERENCE_MASSES,
            primary_target_kg=_REFERENCE_MASSES,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.full(3, 60.0),
        )
        facts = trace_physical_target_lifecycles(
            execution_state=state,
            execution_config=_execution_config(),
            execution_duration_s=600.0,
            execution_start_time=_ORIGIN,
        )

        continuation = diagnose_physical_target_lifecycle_rhs(
            rhs_point=rhs_point,
            lifecycle_trace=facts.continue_existing,
            runtime_assembly=runtime,
        )
        release = diagnose_physical_target_lifecycle_rhs(
            rhs_point=rhs_point,
            lifecycle_trace=facts.release_to_current,
            runtime_assembly=runtime,
        )

        self.assertNotEqual(
            continuation.lifecycle_trace.lifecycle,
            release.lifecycle_trace.lifecycle,
        )
        np.testing.assert_allclose(
            continuation.reachable_snapshot.actual_tank_masses_kg,
            release.reachable_snapshot.actual_tank_masses_kg,
        )
        np.testing.assert_allclose(
            continuation.reachable_dynamic_rhs_generalized_load,
            release.reachable_dynamic_rhs_generalized_load,
        )
        np.testing.assert_allclose(
            continuation.delta_acceleration_from_frozen_baseline,
            release.delta_acceleration_from_frozen_baseline,
        )

    def test_zero_endpoint_new_track_keeps_its_lifecycle_but_matches_same_rhs(self):
        runtime, trajectory, _ = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        load_assembly = trajectory.load_assembly
        timed = diagnose_generalized_load_forecast_ballast_redistribution(
            forecast=load_assembly.load_forecast,
            matrices=trajectory.platform_snapshot.matrices,
            actual_tank_masses_kg=_REFERENCE_MASSES,
            tank_capacities_kg=_CAPACITIES,
            tank_coordinates_m=_COORDINATES,
        )[0]
        zero_preview = preview_physical_endpoint_fractions(
            load_assembly=load_assembly,
            lead_index=0,
            timed_diagnostic=timed,
            actual_tank_masses_kg=_REFERENCE_MASSES,
            tank_capacities_kg=_CAPACITIES,
            fractions=(0.0,),
            execution_state=ExecutionRolloutState(
                masses_kg=_REFERENCE_MASSES,
                target_masses_kg=_REFERENCE_MASSES,
                primary_target_kg=_REFERENCE_MASSES,
                pump_rates_m3_min=np.zeros(3),
                pump_latched=np.zeros(3, dtype=bool),
                pump_off_elapsed_s=np.full(3, 60.0),
            ),
            execution_config=_execution_config(),
            execution_duration_s=600.0,
            execution_start_time=_ORIGIN,
            trajectory=trajectory,
        )[0]
        facts = trace_physical_target_lifecycles(
            execution_state=zero_preview.reachability.execution_start_state,
            execution_config=zero_preview.reachability.execution_config,
            execution_duration_s=zero_preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=zero_preview,
        )
        continuation = diagnose_physical_target_lifecycle_rhs(
            rhs_point=rhs_point,
            lifecycle_trace=facts.continue_existing,
            runtime_assembly=runtime,
        )
        new_track = diagnose_physical_target_lifecycle_rhs(
            rhs_point=rhs_point,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )

        self.assertNotEqual(
            continuation.lifecycle_trace.lifecycle,
            new_track.lifecycle_trace.lifecycle,
        )
        np.testing.assert_allclose(
            continuation.reachable_snapshot.actual_tank_masses_kg,
            new_track.reachable_snapshot.actual_tank_masses_kg,
        )
        np.testing.assert_allclose(
            continuation.reachable_dynamic_rhs_generalized_load,
            new_track.reachable_dynamic_rhs_generalized_load,
        )

    def test_lifecycle_rhs_rejects_wrong_time_second_lead_or_tampered_execution(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_points = diagnose_forecast_platform_rhs(trajectory=trajectory)
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )

        with self.assertRaisesRegex(ValueError, "execution start time"):
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_points[0],
                lifecycle_trace=replace(
                    facts.continue_existing,
                    execution_start_time="2026-08-22T00:10:00",
                ),
                runtime_assembly=runtime,
            )
        with self.assertRaisesRegex(ValueError, "first forecast lead"):
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_points[1],
                lifecycle_trace=facts.continue_existing,
                runtime_assembly=runtime,
            )
        short_facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=300.0,
            execution_start_time=_ORIGIN,
        )
        with self.assertRaisesRegex(ValueError, "execution duration"):
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_points[0],
                lifecycle_trace=short_facts.continue_existing,
                runtime_assembly=runtime,
            )
        mismatched_capacity = replace(
            facts.continue_existing,
            execution_config=replace(
                facts.continue_existing.execution_config,
                tank_capacity_kg=float(_CAPACITIES[0]) - 1.0,
            ),
        )
        with self.assertRaisesRegex(ValueError, "tank capacity"):
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_points[0],
                lifecycle_trace=mismatched_capacity,
                runtime_assembly=runtime,
            )
        tampered_step = replace(
            facts.continue_existing.execution_step,
            active_time_s=facts.continue_existing.execution_step.active_time_s + 1.0,
        )
        with self.assertRaisesRegex(ValueError, "must be reproducible"):
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_points[0],
                lifecycle_trace=replace(
                    facts.continue_existing,
                    execution_step=tampered_step,
                ),
                runtime_assembly=runtime,
            )

    def test_lifecycle_rhs_rejects_snapshot_from_another_runtime_assembly(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        diagnostic = diagnose_physical_target_lifecycle_rhs(
            rhs_point=rhs_point,
            lifecycle_trace=facts.new_track,
            runtime_assembly=runtime,
        )
        foreign_runtime = _runtime_assembly(provenance="foreign_runtime")
        foreign_snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=foreign_runtime,
            actual_tank_masses_kg=diagnostic.reachable_snapshot.actual_tank_masses_kg,
            reference_tank_masses_kg=(
                diagnostic.reachable_snapshot.reference_tank_masses_kg
            ),
            tank_capacities_kg=diagnostic.reachable_snapshot.tank_capacities_kg,
            tank_coordinates_m=diagnostic.reachable_snapshot.tank_coordinates_m,
        )
        with self.assertRaisesRegex(ValueError, "reachable_snapshot"):
            replace(diagnostic, reachable_snapshot=foreign_snapshot)

    def test_lifecycle_rhs_rejects_same_provenance_runtime_with_changed_matrices(self):
        runtime, trajectory, compensation = _prepared_records()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]
        preview = compensation.preview
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )
        altered_damping = np.array(runtime.base_matrices.damping, copy=True)
        altered_damping[4, 4] += 1.0
        same_label_different_runtime = replace(
            runtime,
            base_matrices=replace(runtime.base_matrices, damping=altered_damping),
        )

        with self.assertRaisesRegex(ValueError, "must reconstruct"):
            diagnose_physical_target_lifecycle_rhs(
                rhs_point=rhs_point,
                lifecycle_trace=facts.new_track,
                runtime_assembly=same_label_different_runtime,
            )

    def test_module_does_not_import_legacy_controller_or_candidate_ranking(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_option_rhs_comparison.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("controller", "candidate", "planner", "action_target")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )

    def test_decision_input_does_not_import_raw_evidence_or_legacy_policy(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "first_interval_physical_decision_input.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "forecast_evidence",
            "forecast_action_policy",
            "controller",
            "candidate",
            "planner",
            "safety",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
