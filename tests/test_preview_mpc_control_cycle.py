import unittest
from dataclasses import replace
from unittest.mock import PropertyMock, patch

import numpy as np

from fowt_platform import (
    GeneralizedLoadForecast,
    IncrementalState,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    ThreeTankDifferentialModes,
    assemble_ballast_model_snapshot,
)
from wind_prediction.preview_mpc_application import (
    PreviewMPCApplication,
    PreviewMPCSourceIdentity,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
)
from wind_prediction.preview_mpc import (
    PreviewBlockModel,
    PreviewMPCConstraints,
    PreviewMPCController,
    PreviewMPCWeights,
)
from wind_prediction.preview_mpc_control_cycle import (
    PreviewMPCControlCycleInput,
    PreviewMPCNoSafeCandidateError,
    PreviewMPCPlanningSource,
    PreviewMPCSelectionReason,
    PreviewMPCTargetLifecycle,
    run_preview_mpc_control_cycle,
)
from wind_prediction.preview_mpc_design import (
    research_preview_mpc_design_v1,
    research_preview_mpc_design_v3,
)
from wind_prediction.run_identity import sha256_json

from tests.test_preview_mpc_runtime import (
    CAPACITIES,
    COORDINATES,
    REFERENCE_MASSES,
    runtime,
)


def _zero_rotor_load() -> RotorGeneralizedLoad:
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.zeros(3),
        normal_load=RotorNormalLoad(
            thrust_n=0.0,
            force_platform_n=np.zeros(3),
        ),
        generalized_load_platform=np.zeros(6),
    )


def _pitch_rotor_load(moment_nm: float) -> RotorGeneralizedLoad:
    generalized_load = np.zeros(6)
    generalized_load[4] = float(moment_nm)
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.zeros(3),
        normal_load=RotorNormalLoad(
            thrust_n=0.0,
            force_platform_n=np.zeros(3),
        ),
        generalized_load_platform=generalized_load,
    )


def _rotor_load_from_generalized(load: np.ndarray) -> RotorGeneralizedLoad:
    return RotorGeneralizedLoad(
        relative_air_velocity_platform_mps=np.zeros(3),
        normal_load=RotorNormalLoad(
            thrust_n=0.0,
            force_platform_n=np.zeros(3),
        ),
        generalized_load_platform=np.asarray(load, dtype=float),
    )


def _execution_config() -> ExecutionRolloutConfig:
    return ExecutionRolloutConfig(
        block_duration_s=60.0,
        internal_step_s=10.0,
        water_density_kg_m3=1_000.0,
        max_pump_rate_m3_min=0.2,
        tank_capacity_kg=3_000.0,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 0.2), (10_000.0, 0.2)),
    )


def _execution_state(primary_target=REFERENCE_MASSES) -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=REFERENCE_MASSES,
        target_masses_kg=primary_target,
        primary_target_kg=primary_target,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )


def _controller(
    *,
    actual_masses=REFERENCE_MASSES,
    tank_movement=1.0e-9,
    movement_change=1.0e-10,
):
    assembly = runtime()
    modes = ThreeTankDifferentialModes(tank_coordinates_m=COORDINATES)
    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=assembly,
        actual_tank_masses_kg=actual_masses,
        reference_tank_masses_kg=REFERENCE_MASSES,
        tank_capacities_kg=CAPACITIES,
        tank_coordinates_m=COORDINATES,
    )
    block_model = PreviewBlockModel(
        matrices=snapshot.matrices,
        generalized_load_per_mode_kg=modes.generalized_load_per_mode_kg,
        block_duration_s=60.0,
        sample_fractions=(0.25, 0.5, 0.75, 1.0),
    )
    controller = PreviewMPCController(
        block_model=block_model,
        ballast_modes=modes,
        weights=PreviewMPCWeights(
            roll=1.0,
            pitch=1.0,
            tank_movement=tank_movement,
            movement_change=movement_change,
        ),
        constraints=PreviewMPCConstraints(
            maximum_abs_roll_rad=0.5,
            maximum_abs_pitch_rad=0.5,
            maximum_abs_tank_mass_change_per_block_kg=np.full(3, 500.0),
        ),
    )
    return assembly, modes, controller, snapshot


def _run_cycle(
    *,
    loads,
    primary_target=REFERENCE_MASSES,
    tank_movement=1.0e-9,
    movement_change=1.0e-10,
    execution_rotor_load=None,
    execution_rotor_load_substeps=None,
    planner_rotor_load=None,
    platform_state=None,
    execution_state=None,
    return_input=False,
    maximum_abs_total_tank_mass_deviation_from_reference_kg=1_000.0,
):
    resolved_execution_state = (
        _execution_state(primary_target)
        if execution_state is None
        else execution_state
    )
    assembly, modes, controller, snapshot = _controller(
        actual_masses=resolved_execution_state.actual_masses_kg,
        tank_movement=tank_movement,
        movement_change=movement_change,
    )
    rotor_loads = np.asarray(loads, dtype=float)
    generalized_disturbance_loads = (
        rotor_loads + snapshot.incremental_ballast_load[None, :]
    )
    resolved_planner_load = (
        _rotor_load_from_generalized(rotor_loads[0])
        if planner_rotor_load is None
        else planner_rotor_load
    )
    resolved_execution_load = (
        resolved_planner_load
        if execution_rotor_load is None
        else execution_rotor_load
    )
    cycle_input = PreviewMPCControlCycleInput(
        planning_source=PreviewMPCPlanningSource(
            source="test_forecast_fixture",
            model_version="test_model_v1",
            origin_time="2026-01-01 00:00:00",
            forecast_mode="test_preview",
            source_record_sha256="0" * 64,
            controller_visible_loads_sha256=sha256_json(
                generalized_disturbance_loads.tolist()
            ),
            sample_period_s=60.0,
            lead_times_s=60.0
            * np.arange(1, generalized_disturbance_loads.shape[0] + 1),
            uses_future_information=True,
        ),
        controller=controller,
        ballast_modes=modes,
        platform_state=(
            IncrementalState.zeros()
            if platform_state is None
            else platform_state
        ),
        execution_state=resolved_execution_state,
        execution_config=_execution_config(),
        runtime_assembly=assembly,
        reference_tank_masses_kg=REFERENCE_MASSES,
        tank_capacities_kg=CAPACITIES,
        tank_coordinates_m=COORDINATES,
        generalized_disturbance_loads=generalized_disturbance_loads,
        planner_first_block_rotor_load=resolved_planner_load,
        execution_first_block_rotor_load=resolved_execution_load,
        current_wave_load=np.zeros(6),
        current_other_load=np.zeros(6),
        maximum_abs_roll_pitch_rad=np.full(2, 0.5),
        maximum_abs_working_model_scope_roll_pitch_rad=np.full(2, 0.25),
        maximum_abs_total_tank_mass_deviation_from_reference_kg=(
            maximum_abs_total_tank_mass_deviation_from_reference_kg
        ),
        execution_first_block_rotor_load_substeps=(
            execution_rotor_load_substeps
        ),
    )
    if return_input:
        return cycle_input
    return run_preview_mpc_control_cycle(cycle_input)


class PreviewMPCControlCycleTests(unittest.TestCase):
    def test_application_assembles_one_source_bound_cycle_snapshot(self):
        assembly = runtime()
        modes = ThreeTankDifferentialModes(
            tank_coordinates_m=COORDINATES,
            gravity_m_s2=assembly.gravity_m_s2,
        )
        application = PreviewMPCApplication(
            runtime_assembly=assembly,
            ballast_modes=modes,
            design=research_preview_mpc_design_v3(),
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
        )
        load_forecast = GeneralizedLoadForecast(
            current_generalized_load=np.zeros(6),
            future_generalized_loads=np.zeros((3, 6)),
            lead_times_s=np.array([60.0, 120.0, 180.0]),
        )
        source_identity = PreviewMPCSourceIdentity(
            source="test_forecast_fixture",
            model_version="test_model_v1",
            origin_time="2026-01-01 00:00:00",
            forecast_mode="test_preview",
            source_record_sha256="0" * 64,
            sample_period_s=60.0,
            lead_times_s=load_forecast.lead_times_s,
            uses_future_information=True,
        )

        assembled, result = application.run_cycle(
            source_identity=source_identity,
            rotor_load_forecast=load_forecast,
            platform_state=IncrementalState.zeros(),
            execution_state=_execution_state(),
            execution_config=_execution_config(),
            planner_first_block_rotor_load=_zero_rotor_load(),
            execution_first_block_rotor_load=_zero_rotor_load(),
        )

        np.testing.assert_allclose(
            assembled.snapshot.actual_tank_masses_kg,
            REFERENCE_MASSES,
        )
        np.testing.assert_allclose(
            assembled.maximum_abs_tank_mass_change_per_block_kg,
            [200.0, 200.0, 200.0],
        )
        self.assertAlmostEqual(
            assembled.maximum_abs_total_tank_mass_deviation_from_reference_kg,
            assembled.snapshot.matrices.restoring_stiffness[2, 2]
            * 0.01
            / assembly.gravity_m_s2,
        )
        coarser_assembly = application.assemble_cycle(
            source_identity=source_identity,
            rotor_load_forecast=load_forecast,
            platform_state=IncrementalState.zeros(),
            execution_state=_execution_state(),
            execution_config=replace(_execution_config(), internal_step_s=20.0),
            planner_first_block_rotor_load=_zero_rotor_load(),
            execution_first_block_rotor_load=_zero_rotor_load(),
        )
        self.assertEqual(
            assembled.maximum_abs_total_tank_mass_deviation_from_reference_kg,
            coarser_assembly.maximum_abs_total_tank_mass_deviation_from_reference_kg,
        )
        self.assertEqual(
            assembled.control_cycle_input.planning_source.controller_visible_loads_sha256,
            sha256_json(assembled.disturbance_blocks.generalized_loads.tolist()),
        )
        self.assertTrue(result.free_plan.success, result.free_plan.message)

    def test_cycle_result_preserves_exact_planning_source(self):
        cycle_input = _run_cycle(
            loads=np.zeros((3, 6)),
            return_input=True,
        )

        result = run_preview_mpc_control_cycle(cycle_input)

        self.assertIs(result.planning_source, cycle_input.planning_source)
        self.assertEqual(
            result.planning_source.to_dict()["source_record_sha256"],
            "0" * 64,
        )

    def test_realised_load_substeps_are_used_only_after_selection(self):
        average = _pitch_rotor_load(1_000.0)
        result = _run_cycle(
            loads=np.zeros((1, 6)),
            execution_rotor_load=average,
            execution_rotor_load_substeps=tuple(
                _pitch_rotor_load(value)
                for value in np.linspace(0.0, 2_000.0, 6)
            ),
        )

        first_load = result.execution_path.substeps[0].platform_step.loads.wind[4]
        last_load = result.execution_path.substeps[-1].platform_step.loads.wind[4]
        self.assertLess(first_load, last_load)
        self.assertEqual(
            result.planning_source.controller_visible_loads_sha256,
            sha256_json(np.zeros((1, 6)).tolist()),
        )

    def test_planning_source_rejects_invalid_digest(self):
        cycle_input = _run_cycle(
            loads=np.zeros((3, 6)),
            return_input=True,
        )

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            replace(
                cycle_input.planning_source,
                source_record_sha256="not-a-digest",
            )

    def test_cycle_rejects_horizon_source_mismatch(self):
        cycle_input = _run_cycle(
            loads=np.zeros((3, 6)),
            return_input=True,
        )
        shorter_source = replace(
            cycle_input.planning_source,
            lead_times_s=np.array([60.0, 120.0]),
        )

        with self.assertRaisesRegex(ValueError, "generalized_disturbance_loads"):
            replace(cycle_input, planning_source=shorter_source)

    def test_cycle_rejects_block_duration_mismatch(self):
        cycle_input = _run_cycle(
            loads=np.zeros((3, 6)),
            return_input=True,
        )
        mismatched_config = replace(
            cycle_input.execution_config,
            block_duration_s=30.0,
        )

        with self.assertRaisesRegex(ValueError, "block durations"):
            replace(cycle_input, execution_config=mismatched_config)

    def test_cycle_rejects_first_block_planner_load_mismatch(self):
        cycle_input = _run_cycle(
            loads=np.zeros((3, 6)),
            return_input=True,
        )

        with self.assertRaisesRegex(ValueError, "first-block physical load"):
            replace(
                cycle_input,
                planner_first_block_rotor_load=_pitch_rotor_load(1.0),
            )

    def test_cycle_rejects_loads_not_bound_to_planning_source(self):
        cycle_input = _run_cycle(
            loads=np.zeros((3, 6)),
            return_input=True,
        )
        changed_loads = np.array(
            cycle_input.generalized_disturbance_loads,
            copy=True,
        )
        changed_loads[-1, 4] = 1.0

        with self.assertRaisesRegex(ValueError, "load digest"):
            replace(
                cycle_input,
                generalized_disturbance_loads=changed_loads,
            )

    def test_three_target_lifecycles_share_one_cycle_origin(self):
        loads = np.zeros((3, 6))
        loads[:, 4] = np.array([2_000.0, 5_000.0, -2_000.0])

        result = _run_cycle(loads=loads)

        self.assertEqual(
            {item.lifecycle for item in result.candidates},
            set(PreviewMPCTargetLifecycle),
        )
        for candidate in result.candidates:
            path = candidate.physical_precheck.execution_path
            np.testing.assert_allclose(
                path.substeps[0].start_execution_state.actual_masses_kg,
                REFERENCE_MASSES,
            )
            self.assertIsNotNone(candidate.tail_evaluation)
            tail = candidate.tail_evaluation
            np.testing.assert_allclose(
                tail.predicted_trajectory.initial_state.platform.position,
                path.final_platform_state.position,
            )
            np.testing.assert_allclose(
                tail.predicted_trajectory.initial_state.platform.velocity,
                path.final_platform_state.velocity,
            )
            np.testing.assert_allclose(
                candidate.tail_initial_tank_masses_kg,
                path.final_execution_state.actual_masses_kg,
            )
            self.assertEqual(tail.modal_increments_kg.shape[0], loads.shape[0] - 1)
        self.assertEqual(
            result.selected_lifecycle,
            PreviewMPCTargetLifecycle.TRACK_NEW_MPC_TARGET,
        )
        np.testing.assert_allclose(
            result.execution_request.target_masses_kg,
            result.free_plan.current_target_tank_masses_kg,
        )
        self.assertAlmostEqual(result.execution_path.duration_s, 60.0)

    def test_tail_rollout_uses_reached_tank_state_and_rebuilt_ballast_load(self):
        rotor_loads = np.zeros((3, 6))
        rotor_loads[:, 4] = np.array([2_000.0, 5_000.0, -2_000.0])
        result = _run_cycle(
            loads=rotor_loads,
            primary_target=np.array([1_200.0, 900.0, 900.0]),
        )
        candidate = next(
            item
            for item in result.candidates
            if item.lifecycle
            is PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET
        )
        tail = candidate.tail_evaluation
        self.assertIsNotNone(tail)

        assembly = runtime()
        modes = ThreeTankDifferentialModes(tank_coordinates_m=COORDINATES)
        tail_snapshot = assemble_ballast_model_snapshot(
            runtime_assembly=assembly,
            actual_tank_masses_kg=candidate.tail_initial_tank_masses_kg,
            reference_tank_masses_kg=REFERENCE_MASSES,
            tank_capacities_kg=CAPACITIES,
            tank_coordinates_m=COORDINATES,
        )
        tail_model = PreviewBlockModel(
            matrices=tail_snapshot.matrices,
            generalized_load_per_mode_kg=modes.generalized_load_per_mode_kg,
            block_duration_s=60.0,
            sample_fractions=(0.25, 0.5, 0.75, 1.0),
        )
        expected_first_tail = tail_model.propagate_block(
            initial_state=tail.predicted_trajectory.initial_state,
            modal_increment_kg=tail.modal_increments_kg[0],
            generalized_disturbance_load=(
                rotor_loads[1] + tail_snapshot.incremental_ballast_load
            ),
        )
        np.testing.assert_allclose(
            tail.predicted_trajectory.blocks[0].state_vectors,
            expected_first_tail.state_vectors,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_release_is_a_normal_candidate_and_resets_the_target(self):
        result = _run_cycle(
            loads=np.zeros((3, 6)),
            primary_target=np.array([1_200.0, 900.0, 900.0]),
            tank_movement=1.0e-4,
            movement_change=1.0e-12,
        )

        self.assertEqual(
            result.selected_lifecycle,
            PreviewMPCTargetLifecycle.RELEASE_TO_ACTUAL,
        )
        np.testing.assert_allclose(
            result.next_execution_state.primary_target_masses_kg,
            REFERENCE_MASSES,
        )
        np.testing.assert_allclose(
            result.next_execution_state.rate_limited_target_kg,
            REFERENCE_MASSES,
        )

    def test_future_strengthening_increases_the_first_compensation(self):
        steady = np.zeros((3, 6))
        steady[:, 4] = 2_000.0
        strengthening = np.zeros((3, 6))
        strengthening[:, 4] = np.array([2_000.0, 5_000.0, 9_000.0])

        steady_result = _run_cycle(loads=steady)
        strengthening_result = _run_cycle(loads=strengthening)

        steady_delta = (
            steady_result.execution_request.target_masses_kg - REFERENCE_MASSES
        )
        strengthening_delta = (
            strengthening_result.execution_request.target_masses_kg
            - REFERENCE_MASSES
        )
        self.assertEqual(
            strengthening_result.selected_lifecycle,
            PreviewMPCTargetLifecycle.TRACK_NEW_MPC_TARGET,
        )
        self.assertGreater(
            np.linalg.norm(strengthening_delta),
            np.linalg.norm(steady_delta),
        )
        self.assertGreater(float(steady_delta @ strengthening_delta), 0.0)

    def test_future_decline_reduces_the_first_compensation(self):
        steady = np.zeros((3, 6))
        steady[:, 4] = 5_000.0
        declining = np.zeros((3, 6))
        declining[:, 4] = np.array([5_000.0, 2_000.0, 0.0])

        steady_result = _run_cycle(loads=steady)
        declining_result = _run_cycle(loads=declining)

        steady_delta = (
            steady_result.execution_request.target_masses_kg - REFERENCE_MASSES
        )
        declining_delta = (
            declining_result.execution_request.target_masses_kg
            - REFERENCE_MASSES
        )
        self.assertEqual(
            declining_result.selected_lifecycle,
            PreviewMPCTargetLifecycle.TRACK_NEW_MPC_TARGET,
        )
        self.assertLess(
            np.linalg.norm(declining_delta),
            np.linalg.norm(steady_delta),
        )
        self.assertGreater(float(steady_delta @ declining_delta), 0.0)

    def test_future_reversal_can_defer_an_unnecessary_current_move(self):
        sustained = np.zeros((3, 6))
        sustained[:, 4] = 5_000.0
        reversing = np.zeros((3, 6))
        reversing[:, 4] = np.array([5_000.0, 0.0, -5_000.0])

        sustained_result = _run_cycle(loads=sustained)
        reversing_result = _run_cycle(loads=reversing)

        sustained_delta = (
            sustained_result.execution_request.target_masses_kg
            - REFERENCE_MASSES
        )
        reversing_delta = (
            reversing_result.execution_request.target_masses_kg
            - REFERENCE_MASSES
        )
        self.assertGreater(np.linalg.norm(sustained_delta), 0.0)
        np.testing.assert_allclose(reversing_delta, np.zeros(3), atol=1.0e-9)
        self.assertIn(
            reversing_result.selected_lifecycle,
            {
                PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET,
                PreviewMPCTargetLifecycle.RELEASE_TO_ACTUAL,
            },
        )

    def test_realised_future_load_cannot_change_current_selection(self):
        loads = np.zeros((3, 6))
        loads[:, 4] = np.array([2_000.0, 5_000.0, -2_000.0])
        realised = RotorGeneralizedLoad(
            relative_air_velocity_platform_mps=np.zeros(3),
            normal_load=RotorNormalLoad(
                thrust_n=0.0,
                force_platform_n=np.zeros(3),
            ),
            generalized_load_platform=np.array([0, 0, 0, 0, 2.0e5, 0]),
        )

        visible = _run_cycle(loads=loads)
        with_realised = _run_cycle(
            loads=loads,
            execution_rotor_load=realised,
        )

        self.assertEqual(
            with_realised.selected_lifecycle,
            visible.selected_lifecycle,
        )
        np.testing.assert_allclose(
            with_realised.execution_request.target_masses_kg,
            visible.execution_request.target_masses_kg,
        )
        self.assertFalse(
            np.allclose(
                with_realised.next_platform_state.position,
                visible.next_platform_state.position,
            )
        )

    def test_common_mass_change_is_recorded_instead_of_hidden(self):
        result = _run_cycle(
            loads=np.zeros((3, 6)),
            primary_target=np.array([1_200.0, 900.0, 900.0]),
            tank_movement=1.0e-12,
            movement_change=1.0e-2,
        )

        continue_candidate = next(
            item
            for item in result.candidates
            if item.lifecycle
            is PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET
        )
        actual_delta = continue_candidate.previewed_first_block_tank_delta_kg
        self.assertAlmostEqual(
            continue_candidate.unrepresented_common_mass_change_kg,
            float(np.sum(actual_delta)),
            places=9,
        )
        self.assertLess(continue_candidate.projection_residual_norm_kg, 1.0e-9)
        self.assertTrue(continue_candidate.common_mass_within_projection_scope)
        self.assertTrue(continue_candidate.projection_residual_within_tolerance)

    def test_physical_common_mass_change_is_not_free_in_candidate_score(self):
        result = _run_cycle(
            loads=np.zeros((3, 6)),
            primary_target=REFERENCE_MASSES + 20.0,
        )

        continue_candidate = next(
            item
            for item in result.candidates
            if item.lifecycle
            is PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET
        )
        self.assertGreater(
            continue_candidate.objective.tank_movement_by_block[0],
            0.0,
        )
        self.assertGreater(continue_candidate.comparison_objective, 0.0)

    def test_candidate_throughput_uses_actual_first_block_pump_volume(self):
        cycle_input = _run_cycle(
            loads=np.zeros((1, 6)),
            primary_target=np.array([1_200.0, 900.0, 900.0]),
            return_input=True,
        )
        controller = PreviewMPCController(
            block_model=cycle_input.controller.block_model,
            ballast_modes=cycle_input.ballast_modes,
            weights=PreviewMPCWeights(
                roll=1.0,
                pitch=1.0,
                tank_movement=0.0,
                tank_throughput=0.25,
                movement_change=1.0e-10,
            ),
            constraints=cycle_input.controller.constraints,
        )
        synthetic_transferred_volume_m3 = 1.25
        with patch(
            "wind_prediction.preview_mpc_control_cycle."
            "PhysicalExecutionPlatformPath.transferred_volume_m3",
            new_callable=PropertyMock,
            return_value=synthetic_transferred_volume_m3,
        ):
            result = run_preview_mpc_control_cycle(
                replace(cycle_input, controller=controller)
            )
        candidate = next(
            item
            for item in result.candidates
            if item.lifecycle
            is PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET
        )
        expected = (
            controller.weights.tank_throughput
            * synthetic_transferred_volume_m3
            * cycle_input.execution_config.water_density_kg_m3
        )
        self.assertAlmostEqual(
            candidate.objective.tank_throughput_by_block[0],
            expected,
        )

    def test_total_mass_scope_is_explicit_and_independent_of_internal_step(self):
        base = _run_cycle(
            loads=np.zeros((1, 6)),
            primary_target=REFERENCE_MASSES + 20.0,
            return_input=True,
            maximum_abs_total_tank_mass_deviation_from_reference_kg=50.0,
        )

        candidates = []
        for internal_step_s in (5.0, 20.0):
            result = run_preview_mpc_control_cycle(
                replace(
                    base,
                    execution_config=replace(
                        base.execution_config,
                        internal_step_s=internal_step_s,
                    ),
                )
            )
            candidate = next(
                item
                for item in result.candidates
                if item.lifecycle
                is PreviewMPCTargetLifecycle.CONTINUE_EXISTING_TARGET
            )
            candidates.append(candidate)

        for candidate in candidates:
            self.assertEqual(candidate.total_mass_scope_limit_kg, 50.0)
            self.assertFalse(candidate.total_mass_within_scope)
            self.assertGreater(
                candidate.maximum_abs_total_mass_deviation_from_reference_kg,
                50.0,
            )
            self.assertIn(
                "total_tank_mass_deviation_exceeds_scope",
                candidate.rejection_reason,
            )

    def test_single_block_horizon_needs_no_tail_replan(self):
        result = _run_cycle(loads=np.zeros((1, 6)))

        for candidate in result.candidates:
            self.assertIsNone(candidate.tail_evaluation)
            self.assertEqual(candidate.objective.running_posture_by_block.shape, (1,))
            self.assertEqual(candidate.objective.tank_movement_by_block.shape, (1,))
            np.testing.assert_allclose(
                candidate.tail_initial_tank_masses_kg,
                candidate.physical_precheck.execution_path.final_execution_state.actual_masses_kg,
            )

    def test_second_cycle_starts_from_first_cycle_realised_state(self):
        loads = np.zeros((3, 6))
        loads[:, 4] = np.array([2_000.0, 5_000.0, -2_000.0])

        first = _run_cycle(loads=loads)
        second = _run_cycle(
            loads=loads,
            platform_state=first.next_platform_state,
            execution_state=first.next_execution_state,
        )

        for candidate in second.candidates:
            start = candidate.physical_precheck.execution_path.substeps[0]
            np.testing.assert_allclose(
                start.start_platform_state.position,
                first.next_platform_state.position,
            )
            np.testing.assert_allclose(
                start.start_platform_state.velocity,
                first.next_platform_state.velocity,
            )
            np.testing.assert_allclose(
                start.start_execution_state.actual_masses_kg,
                first.next_execution_state.actual_masses_kg,
            )
            np.testing.assert_allclose(
                start.start_execution_state.primary_target_masses_kg,
                first.next_execution_state.primary_target_masses_kg,
            )

        actual_start = second.execution_path.substeps[0]
        np.testing.assert_allclose(
            actual_start.start_platform_state.position,
            first.next_platform_state.position,
        )
        np.testing.assert_allclose(
            actual_start.start_execution_state.actual_masses_kg,
            first.next_execution_state.actual_masses_kg,
        )

    def test_horizon_infeasible_first_block_safe_candidate_still_executes(self):
        loads = np.zeros((3, 6))
        loads[1:, 4] = 2.0e9

        result = _run_cycle(loads=loads)

        self.assertFalse(result.selected_from_horizon_feasible_options)
        self.assertTrue(result.selected_from_planner_load_precheck_safe_options)
        self.assertEqual(
            result.selection_reason,
            PreviewMPCSelectionReason.DEGRADED_FIRST_BLOCK_SAFE_NO_FULLY_ELIGIBLE_OPTION,
        )
        self.assertTrue(
            result.selected_candidate.physical_precheck.posture_precheck_passed
        )
        self.assertGreater(result.execution_path.duration_s, 0.0)

    def test_no_physically_safe_candidate_does_not_commit_an_action(self):
        rotor_load = _pitch_rotor_load(2.0e9)
        loads = np.tile(rotor_load.generalized_load_platform, (3, 1))

        with patch(
            "wind_prediction.preview_mpc_control_cycle."
            "advance_physical_execution_platform_path"
        ) as execute:
            with self.assertRaises(PreviewMPCNoSafeCandidateError) as caught:
                _run_cycle(
                    loads=loads,
                    planner_rotor_load=rotor_load,
                    execution_rotor_load=rotor_load,
                )

        execute.assert_not_called()

        candidates = caught.exception.candidates
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(
            all(
                not item.physical_precheck.posture_precheck_passed
                for item in candidates
            )
        )

    def test_candidate_outside_working_model_scope_is_not_committed(self):
        loads = np.zeros((3, 6))
        cycle_input = _run_cycle(loads=loads, return_input=True)
        cycle_input = replace(
            cycle_input,
            platform_state=IncrementalState(
                position=np.array([0.0, 0.0, 0.0, 0.30, 0.0, 0.0]),
                velocity=np.zeros(6),
            ),
        )

        with patch(
            "wind_prediction.preview_mpc_control_cycle."
            "advance_physical_execution_platform_path"
        ) as execute:
            with self.assertRaises(PreviewMPCNoSafeCandidateError) as caught:
                run_preview_mpc_control_cycle(cycle_input)

        execute.assert_not_called()
        self.assertTrue(
            all(
                not item.first_block_within_working_model_scope
                for item in caught.exception.candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
