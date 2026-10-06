from dataclasses import replace
import unittest
from pathlib import Path

import numpy as np

from fowt_platform import (
    assemble_volturnus_static_restoring_aligned_runtime_assembly,
    IncrementalState,
    RotorGeneralizedLoad,
    RotorNormalLoad,
    quasi_steady_rotor_generalized_load_from_enu_wind,
)
from wind_prediction.controller_core import (
    ControlAction,
    ControlCoreConfig,
    ControlObservation,
    decide_control_cycle,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutRequest,
    ExecutionRolloutState,
    ExecutionTargetOperation,
    simulate_execution_step,
)
from wind_prediction.controller_platform_handoff import (
    ControllerPlatformPath,
    ControllerPlatformSubstep,
    advance_controller_platform_path,
    advance_controller_platform_substep,
    control_observation_from_incremental_state,
)
from wind_prediction.physical_execution_platform_path import (
    PhysicalExecutionPlatformPath,
    PhysicalExecutionPlatformSubstep,
    advance_physical_execution_platform_path,
    advance_physical_execution_platform_substep,
)
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.forecast_evidence import ForecastEvidence


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = (
    ROOT
    / "configs"
    / "reference_platforms"
    / "volturnus_s_openfast_v1_1_16.json"
)
REFERENCE_TANK_MASSES_KG = np.array([1_108_000.0, 1_362_000.0, 1_362_000.0])
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ]
)


def _controller_config() -> ControlCoreConfig:
    execution = ExecutionRolloutConfig(
        block_duration_s=1200.0,
        internal_step_s=1.0,
        water_density_kg_m3=1025.0,
        max_pump_rate_m3_min=1.0,
        target_slew_enabled=False,
        stop_error_kg=0.0,
        restart_error_kg=0.0,
        min_on_s=0.0,
        min_off_s=0.0,
        near_target_hold_s=0.0,
        ramp_up_m3_min_per_s=np.inf,
        ramp_down_m3_min_per_s=np.inf,
        pump_rate_schedule_m3_min=((0.0, 1.0), (20_000.0, 1.0)),
        tank_capacity_kg=TANK_CAPACITY_KG,
    )
    return ControlCoreConfig(
        stage_duration_s=1200.0,
        stage_count=3,
        execution=execution,
        posture_priority_envelope_deg=(3.0, 3.0),
        w_residual=80.0,
        w_terminal_residual=80.0,
        w_pump_volume=0.2,
        forecast_policy=ForecastActionPolicyConfig(
            enabled=True,
            stage_duration_s=1200.0,
            high_impact_reliability_min=0.65,
            high_impact_event_probability_min=0.60,
        ),
    )


def _forecast() -> ForecastEvidence:
    return ForecastEvidence(
        source="handoff_test",
        model_version="test",
        origin_time="2026-08-19T00:00:00",
        sample_period_s=600.0,
        uv_ms=np.array([(0.0, -8.0)] * 6, dtype=float),
        lead_reliability=np.full(6, 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


def _changing_forecast() -> ForecastEvidence:
    return ForecastEvidence(
        source="handoff_test",
        model_version="test",
        origin_time="2026-08-20T00:00:00",
        sample_period_s=600.0,
        uv_ms=np.array(
            [
                (0.0, -10.0),
                (0.0, -12.0),
                (2.0, -11.0),
                (3.0, -9.0),
                (1.0, -8.0),
                (0.0, -7.0),
            ],
            dtype=float,
        ),
        lead_reliability=np.full(6, 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


class ControllerDecisionPlatformHandoffTests(unittest.TestCase):
    def test_legacy_handoff_exports_are_direct_physical_path_aliases(self):
        self.assertIs(ControllerPlatformPath, PhysicalExecutionPlatformPath)
        self.assertIs(ControllerPlatformSubstep, PhysicalExecutionPlatformSubstep)
        self.assertIs(
            advance_controller_platform_path,
            advance_physical_execution_platform_path,
        )
        self.assertIs(
            advance_controller_platform_substep,
            advance_physical_execution_platform_substep,
        )

    @staticmethod
    def _zero_rotor_load() -> RotorGeneralizedLoad:
        return RotorGeneralizedLoad(
            relative_air_velocity_platform_mps=np.zeros(3),
            normal_load=RotorNormalLoad(
                thrust_n=0.0,
                force_platform_n=np.zeros(3),
            ),
            generalized_load_platform=np.zeros(6),
        )

    def _advance_substep(
        self,
        *,
        platform_state: IncrementalState,
        execution_state: ExecutionRolloutState,
        request: ExecutionRolloutRequest,
        execution_config: ExecutionRolloutConfig,
        rotor_load: RotorGeneralizedLoad | None = None,
        tank_capacities_kg=None,
    ):
        runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            REFERENCE_MANIFEST,
            np.zeros((6, 6)),
        )
        self.assertEqual(
            runtime_assembly.provenance,
            "static_restoring_aligned_with_aux_frc_inertia_mooring",
        )
        return advance_controller_platform_substep(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=request,
            execution_config=execution_config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=(
                np.full(3, TANK_CAPACITY_KG)
                if tank_capacities_kg is None
                else tank_capacities_kg
            ),
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=(
                ControllerDecisionPlatformHandoffTests._zero_rotor_load()
                if rotor_load is None
                else rotor_load
            ),
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=1.0,
        )

    def _advance_path(
        self,
        *,
        platform_state: IncrementalState,
        execution_state: ExecutionRolloutState,
        request: ExecutionRolloutRequest,
        execution_config: ExecutionRolloutConfig,
        duration_s: float,
    ):
        runtime_assembly = assemble_volturnus_static_restoring_aligned_runtime_assembly(
            REFERENCE_MANIFEST,
            np.zeros((6, 6)),
        )
        return advance_controller_platform_path(
            platform_state=platform_state,
            execution_state=execution_state,
            execution_request=request,
            execution_config=execution_config,
            runtime_assembly=runtime_assembly,
            reference_tank_masses_kg=REFERENCE_TANK_MASSES_KG,
            tank_capacities_kg=np.full(3, TANK_CAPACITY_KG),
            tank_coordinates_m=TANK_COORDINATES_M,
            rotor_load=ControllerDecisionPlatformHandoffTests._zero_rotor_load(),
            wave_load=np.zeros(6),
            other_load=np.zeros(6),
            duration_s=duration_s,
        )

    def test_substep_rejects_platform_capacity_mismatch_with_execution_config(self):
        execution_state = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )

        with self.assertRaisesRegex(
            ValueError,
            "must match execution_config.tank_capacity_kg",
        ):
            self._advance_substep(
                platform_state=IncrementalState.zeros(),
                execution_state=execution_state,
                request=ExecutionRolloutRequest.track(REFERENCE_TANK_MASSES_KG),
                execution_config=_controller_config().execution,
                tank_capacities_kg=np.array(
                    [TANK_CAPACITY_KG, TANK_CAPACITY_KG - 1.0, TANK_CAPACITY_KG]
                ),
            )

    def test_decision_request_drives_short_execution_without_leaking_preview_state(self):
        config = _controller_config()
        execution_state = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        observation = ControlObservation(
            time_s=0.0,
            posture_deg=(4.5, 0.0),
            posture_rate_deg_s=(0.0, 0.0),
            current_wind_uv_ms=(0.0, -8.0),
            execution_state=execution_state,
        )

        decision = decide_control_cycle(observation, _forecast(), config)

        self.assertIn(
            decision.action,
            {ControlAction.NORMAL, ControlAction.STRENGTHEN},
        )
        self.assertEqual(
            decision.execution_request.operation,
            ExecutionTargetOperation.TRACK,
        )
        np.testing.assert_allclose(
            decision.execution_request.target_masses_kg,
            decision.target_masses_kg,
        )
        with self.assertRaisesRegex(TypeError, "execution_request"):
            self._advance_substep(
                platform_state=IncrementalState.zeros(),
                execution_state=execution_state,
                request=decision.first_execution,  # type: ignore[arg-type]
                execution_config=config.execution,
            )

        first = self._advance_substep(
            platform_state=IncrementalState.zeros(),
            execution_state=execution_state,
            request=decision.execution_request,
            execution_config=config.execution,
        )
        self.assertLess(
            np.linalg.norm(first.execution_step.mass_delta_kg),
            np.linalg.norm(decision.first_execution.mass_delta_kg),
        )
        np.testing.assert_allclose(
            first.ballast_snapshot.actual_tank_masses_kg,
            execution_state.actual_masses_kg,
        )
        np.testing.assert_allclose(first.next_platform_state.position, np.zeros(6))
        np.testing.assert_allclose(first.next_platform_state.velocity, np.zeros(6))

        second = self._advance_substep(
            platform_state=first.next_platform_state,
            execution_state=first.next_execution_state,
            request=decision.execution_request,
            execution_config=config.execution,
        )
        np.testing.assert_allclose(
            second.ballast_snapshot.actual_tank_masses_kg,
            first.next_execution_state.actual_masses_kg,
        )
        self.assertGreater(np.linalg.norm(second.next_platform_state.velocity[3:5]), 0.0)

        next_observation = control_observation_from_incremental_state(
            time_s=2.0,
            platform_state=second.next_platform_state,
            execution_state=second.next_execution_state,
            current_wind_uv_ms=(0.0, -8.0),
        )
        self.assertIs(next_observation.execution_state, second.next_execution_state)
        self.assertIsNot(next_observation.execution_state, decision.first_execution)

    def test_forecast_evidence_reaches_platform_only_through_explicit_rotor_load(self):
        config = _controller_config()
        execution_state = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        platform_state = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.0, np.radians(4.5), 0.0]),
            velocity=np.zeros(6),
        )
        evidence = _changing_forecast()
        current_wind_uv_ms = (0.0, -8.0)
        observation = control_observation_from_incremental_state(
            time_s=0.0,
            platform_state=platform_state,
            execution_state=execution_state,
            current_wind_uv_ms=current_wind_uv_ms,
        )
        decision = decide_control_cycle(observation, evidence, config)
        self.assertTrue(decision.context.forecast_available)
        self.assertGreater(
            np.linalg.norm(decision.context.stages[0].forecast_increment_deg),
            0.0,
        )

        rotor_load = quasi_steady_rotor_generalized_load_from_enu_wind(
            air_density_kg_m3=1.225,
            rotor_radius_m=100.0,
            thrust_coefficient=0.8,
            enu_downwind_air_velocity_mps=evidence.uv_ms[0],
            frozen_equilibrium_heading_rad=np.pi,
            platform_reference_velocity_platform_mps=platform_state.velocity[:3],
            platform_angular_velocity_platform_radps=platform_state.velocity[3:],
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 150.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        )
        substep = self._advance_substep(
            platform_state=platform_state,
            execution_state=execution_state,
            request=decision.execution_request,
            execution_config=config.execution,
            rotor_load=rotor_load,
        )

        np.testing.assert_allclose(
            substep.platform_step.loads.wind,
            rotor_load.generalized_load_platform,
        )
        self.assertGreater(np.linalg.norm(substep.platform_step.loads.wind), 0.0)
        np.testing.assert_allclose(
            substep.ballast_snapshot.actual_tank_masses_kg,
            execution_state.actual_masses_kg,
        )
        self.assertGreater(np.linalg.norm(substep.execution_step.mass_delta_kg), 0.0)

        next_observation = control_observation_from_incremental_state(
            time_s=1.0,
            platform_state=substep.next_platform_state,
            execution_state=substep.next_execution_state,
            # This test does not schedule a 600 s forecast stage onto a 1 s
            # physical substep. The forecast vector above is only an explicit
            # synthetic load input used to cover the P6-to-P4 interface.
            current_wind_uv_ms=current_wind_uv_ms,
        )
        self.assertIs(next_observation.execution_state, substep.next_execution_state)
        self.assertEqual(next_observation.current_wind_uv_ms, current_wind_uv_ms)
        self.assertAlmostEqual(
            next_observation.posture_deg[0],
            np.degrees(substep.next_platform_state.position[4]),
        )
        self.assertAlmostEqual(
            next_observation.posture_deg[1],
            np.degrees(substep.next_platform_state.position[3]),
        )

    def test_one_decision_period_uses_committed_request_and_returns_actual_state(self):
        """A 1200 s decision is held while the P4 handoff advances at 1 s."""

        config = _controller_config()
        initial_execution = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        fixed_wind_uv_ms = (0.0, -8.0)
        initial_platform_state = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, 0.0, np.radians(4.5), 0.0]),
            velocity=np.zeros(6),
        )
        initial_observation = control_observation_from_incremental_state(
            time_s=0.0,
            platform_state=initial_platform_state,
            execution_state=initial_execution,
            # P4.2 keeps the environment fixed to isolate state handoff.
            current_wind_uv_ms=fixed_wind_uv_ms,
        )
        decision = decide_control_cycle(initial_observation, _forecast(), config)
        request = decision.execution_request

        platform_state = initial_platform_state
        execution_state = initial_execution
        first_snapshot_masses = None
        second_snapshot_masses = None
        previous_execution_state = None
        for step_index in range(1200):
            substep = self._advance_substep(
                platform_state=platform_state,
                execution_state=execution_state,
                request=request,
                execution_config=config.execution,
            )
            if step_index == 0:
                first_snapshot_masses = substep.ballast_snapshot.actual_tank_masses_kg
            elif step_index == 1:
                second_snapshot_masses = substep.ballast_snapshot.actual_tank_masses_kg
                previous_execution_state = execution_state
            platform_state = substep.next_platform_state
            execution_state = substep.next_execution_state

        self.assertEqual(config.stage_duration_s, 1200.0)
        self.assertEqual(config.execution.block_duration_s, 1200.0)
        np.testing.assert_allclose(first_snapshot_masses, initial_execution.actual_masses_kg)
        self.assertIsNotNone(previous_execution_state)
        np.testing.assert_allclose(
            second_snapshot_masses,
            previous_execution_state.actual_masses_kg,
        )
        self.assertGreater(
            np.linalg.norm(execution_state.actual_masses_kg - initial_execution.actual_masses_kg),
            0.0,
        )
        self.assertGreater(np.linalg.norm(platform_state.position[3:5]), 0.0)

        next_observation = control_observation_from_incremental_state(
            time_s=1200.0,
            platform_state=platform_state,
            execution_state=execution_state,
            current_wind_uv_ms=fixed_wind_uv_ms,
        )
        self.assertIs(next_observation.execution_state, execution_state)
        self.assertIsNot(next_observation.execution_state, decision.first_execution)
        self.assertAlmostEqual(
            next_observation.posture_deg[0],
            np.degrees(platform_state.position[4]),
        )
        self.assertAlmostEqual(
            next_observation.posture_deg[1],
            np.degrees(platform_state.position[3]),
        )

    def test_incremental_state_observation_uses_pitch_roll_order_and_preserves_enu_wind(self):
        execution = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        state = IncrementalState(
            position=np.array([0.0, 0.0, 0.0, -0.2, 0.3, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0, -0.04, 0.05, 0.0]),
        )

        observation = control_observation_from_incremental_state(
            time_s=12.0,
            platform_state=state,
            execution_state=execution,
            current_wind_uv_ms=(3.0, -4.0),
        )

        self.assertAlmostEqual(observation.posture_deg[0], np.degrees(0.3))
        self.assertAlmostEqual(observation.posture_deg[1], np.degrees(-0.2))
        self.assertAlmostEqual(observation.posture_rate_deg_s[0], np.degrees(0.05))
        self.assertAlmostEqual(observation.posture_rate_deg_s[1], np.degrees(-0.04))
        # The handoff only converts platform attitude. Geographic ENU wind
        # remains unchanged until an explicit future load mapping uses heading.
        self.assertEqual(observation.current_wind_uv_ms, (3.0, -4.0))
        self.assertIs(observation.execution_state, execution)

    def test_track_and_release_requests_inherit_actual_actuator_state(self):
        config = _controller_config().execution
        initial = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        target = REFERENCE_TANK_MASSES_KG + np.array([10_000.0, 0.0, 0.0])

        tracked = self._advance_substep(
            platform_state=IncrementalState.zeros(),
            execution_state=initial,
            request=ExecutionRolloutRequest.track(target),
            execution_config=config,
        )
        np.testing.assert_allclose(
            tracked.next_execution_state.primary_target_masses_kg,
            target,
        )
        self.assertGreater(np.linalg.norm(tracked.execution_step.mass_delta_kg), 0.0)

        released = self._advance_substep(
            platform_state=tracked.next_platform_state,
            execution_state=tracked.next_execution_state,
            request=ExecutionRolloutRequest.release_to_current(),
            execution_config=config,
        )
        np.testing.assert_allclose(
            released.execution_step.requested_target_kg,
            tracked.next_execution_state.actual_masses_kg,
        )
        np.testing.assert_allclose(
            released.next_execution_state.primary_target_masses_kg,
            tracked.next_execution_state.actual_masses_kg,
        )
        self.assertEqual(
            released.execution_step.target_operation,
            ExecutionTargetOperation.RELEASE_TO_CURRENT,
        )

    def test_committed_track_path_matches_one_execution_rollout_with_target_slew(self):
        config = replace(
            _controller_config().execution,
            block_duration_s=5.5,
            internal_step_s=2.0,
            target_slew_enabled=True,
            target_slew_rate_m3_min=3.0,
            min_on_s=3.0,
            min_off_s=1.0,
            near_target_hold_s=1.0,
            ramp_up_m3_min_per_s=0.4,
            ramp_down_m3_min_per_s=0.3,
        )
        initial_execution = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        request = ExecutionRolloutRequest.track(
            REFERENCE_TANK_MASSES_KG + np.array([25_000.0, -12_000.0, 0.0])
        )

        path = self._advance_path(
            platform_state=IncrementalState.zeros(),
            execution_state=initial_execution,
            request=request,
            execution_config=config,
            duration_s=5.5,
        )
        direct = simulate_execution_step(initial_execution, request, config)

        self.assertEqual(len(path.substeps), 3)
        self.assertAlmostEqual(path.duration_s, 5.5)
        np.testing.assert_allclose(
            path.substeps[0].ballast_snapshot.actual_tank_masses_kg,
            initial_execution.actual_masses_kg,
        )
        np.testing.assert_allclose(
            path.substeps[1].ballast_snapshot.actual_tank_masses_kg,
            path.substeps[0].next_execution_state.actual_masses_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.actual_masses_kg,
            direct.state.actual_masses_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.rate_limited_target_kg,
            direct.state.rate_limited_target_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.primary_target_masses_kg,
            direct.state.primary_target_masses_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.signed_flow_m3_min,
            direct.state.signed_flow_m3_min,
        )
        np.testing.assert_array_equal(
            path.final_execution_state.pump_latched,
            direct.state.pump_latched,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_on_elapsed_s,
            direct.state.pump_on_elapsed_s,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_off_elapsed_s,
            direct.state.pump_off_elapsed_s,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_near_target_s,
            direct.state.pump_near_target_s,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_command_rates_m3_min,
            direct.state.pump_command_rates_m3_min,
        )
        np.testing.assert_allclose(
            path.final_execution_state.last_flow_directions,
            direct.state.last_flow_directions,
        )
        np.testing.assert_allclose(
            path.final_execution_state.previous_block_tank_mass_delta_kg,
            path.actual_tank_mass_delta_kg,
        )
        self.assertIs(
            path.final_execution_state,
            path.final_execution_state,
        )
        self.assertFalse(
            np.allclose(
                path.substeps[-1]
                .next_execution_state
                .previous_block_tank_mass_delta_kg,
                path.final_execution_state.previous_block_tank_mass_delta_kg,
            )
        )
        self.assertGreater(np.linalg.norm(path.final_platform_state.position[3:5]), 0.0)

        with self.assertRaisesRegex(ValueError, "must not exceed"):
            self._advance_path(
                platform_state=IncrementalState.zeros(),
                execution_state=initial_execution,
                request=request,
                execution_config=config,
                duration_s=5.500001,
            )

    def test_committed_release_path_matches_one_execution_rollout(self):
        config = replace(
            _controller_config().execution,
            block_duration_s=5.5,
            internal_step_s=2.0,
            target_slew_enabled=True,
            target_slew_rate_m3_min=3.0,
            min_on_s=3.0,
            min_off_s=1.0,
            near_target_hold_s=2.0,
            ramp_up_m3_min_per_s=0.4,
            ramp_down_m3_min_per_s=0.3,
        )
        initial_execution = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG + np.array([5_000.0, 0.0, 0.0]),
            primary_target_kg=REFERENCE_TANK_MASSES_KG + np.array([5_000.0, 0.0, 0.0]),
            pump_rates_m3_min=np.array([1.0, 0.0, 0.0]),
            pump_command_rates_m3_min=np.array([1.0, 0.0, 0.0]),
            pump_latched=np.array([True, False, False]),
            pump_on_elapsed_s=np.array([1.0, 0.0, 0.0]),
            pump_off_elapsed_s=np.array([0.0, 2.0, 2.0]),
            pump_near_target_s=np.zeros(3),
        )
        request = ExecutionRolloutRequest.release_to_current()
        path = self._advance_path(
            platform_state=IncrementalState(
                position=[0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
                velocity=np.zeros(6),
            ),
            execution_state=initial_execution,
            request=request,
            execution_config=config,
            duration_s=5.5,
        )
        direct = simulate_execution_step(initial_execution, request, config)

        self.assertEqual(len(path.substeps), 3)
        self.assertTrue(
            all(
                step.execution_step.target_operation
                is ExecutionTargetOperation.RELEASE_TO_CURRENT
                for step in path.substeps
            )
        )
        for substep in path.substeps:
            np.testing.assert_allclose(
                substep.execution_step.mass_delta_kg,
                np.zeros(3),
            )
            self.assertAlmostEqual(
                substep.execution_step.transferred_volume_m3,
                0.0,
            )
        np.testing.assert_allclose(
            path.final_execution_state.actual_masses_kg,
            direct.state.actual_masses_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.rate_limited_target_kg,
            direct.state.rate_limited_target_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.primary_target_masses_kg,
            direct.state.primary_target_masses_kg,
        )
        np.testing.assert_allclose(
            path.final_execution_state.signed_flow_m3_min,
            direct.state.signed_flow_m3_min,
        )
        np.testing.assert_array_equal(
            path.final_execution_state.pump_latched,
            direct.state.pump_latched,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_on_elapsed_s,
            direct.state.pump_on_elapsed_s,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_off_elapsed_s,
            direct.state.pump_off_elapsed_s,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_near_target_s,
            direct.state.pump_near_target_s,
        )
        np.testing.assert_allclose(
            path.final_execution_state.pump_command_rates_m3_min,
            direct.state.pump_command_rates_m3_min,
        )
        np.testing.assert_allclose(
            path.final_execution_state.last_flow_directions,
            direct.state.last_flow_directions,
        )
        np.testing.assert_allclose(
            path.final_execution_state.actual_masses_kg,
            initial_execution.actual_masses_kg,
        )
        self.assertAlmostEqual(
            sum(step.execution_step.transferred_volume_m3 for step in path.substeps),
            direct.transferred_volume_m3,
        )
        self.assertAlmostEqual(
            sum(step.execution_step.active_time_s for step in path.substeps),
            direct.active_time_s,
        )
        self.assertEqual(
            sum(step.execution_step.starts for step in path.substeps),
            direct.starts,
        )
        self.assertEqual(
            sum(step.execution_step.stops for step in path.substeps), direct.stops)
        self.assertEqual(
            sum(step.execution_step.direction_switches for step in path.substeps),
            direct.direction_switches,
        )
        self.assertGreater(direct.active_time_s, 0.0)
        self.assertNotEqual(path.final_platform_state.position[4], 0.01)

    def test_path_exports_whole_block_delta_when_pump_stops_before_final_substep(self):
        config = replace(
            _controller_config().execution,
            block_duration_s=5.5,
            internal_step_s=2.0,
            target_slew_enabled=False,
            stop_error_kg=0.0,
            restart_error_kg=0.0,
            min_on_s=0.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=np.inf,
        )
        initial_execution = ExecutionRolloutState(
            masses_kg=REFERENCE_TANK_MASSES_KG,
            target_masses_kg=REFERENCE_TANK_MASSES_KG,
            primary_target_kg=REFERENCE_TANK_MASSES_KG,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
        )
        requested_delta_kg = np.array([20.0, 0.0, 0.0])
        path = self._advance_path(
            platform_state=IncrementalState.zeros(),
            execution_state=initial_execution,
            request=ExecutionRolloutRequest.track(
                REFERENCE_TANK_MASSES_KG + requested_delta_kg
            ),
            execution_config=config,
            duration_s=5.5,
        )

        np.testing.assert_allclose(path.actual_tank_mass_delta_kg, requested_delta_kg)
        np.testing.assert_allclose(
            path.final_execution_state.previous_block_tank_mass_delta_kg,
            requested_delta_kg,
        )
        np.testing.assert_allclose(
            path.substeps[-1]
            .next_execution_state
            .previous_block_tank_mass_delta_kg,
            np.zeros(3),
        )


if __name__ == "__main__":
    unittest.main()
