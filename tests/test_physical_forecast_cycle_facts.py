from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest

import numpy as np

from fowt_platform import (
    BallastMassProperties,
    BallastRuntimeAssembly,
    IncrementalState,
    PlatformMatrices,
    assemble_ballast_model_snapshot,
    weight_stiffness_about_reference,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
)
from wind_prediction.controller_replay_adapter import ReplayForecastEvidenceSource
from wind_prediction.forecast_adapter import ForecastResult
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
)
from wind_prediction.forecast_platform_trajectory import (
    FORECAST_INTERVAL_LOAD_CONVENTION,
    ForecastTrajectoryEnvironment,
)
from wind_prediction.physical_forecast_cycle_facts import (
    assemble_first_interval_physical_cycle_input,
)
from wind_prediction.physical_current_cycle_decision import (
    PhysicalCurrentCycleDecisionPolicy,
    assemble_physical_current_cycle_decision_from_input,
)
from wind_prediction.physical_forecast_admission import PhysicalForecastAdmission
from wind_prediction.physical_lifecycle_selection import PhysicalPostureLimits
from wind_prediction.replay_physical_cycle_adapter import (
    assemble_replay_first_interval_physical_cycle_input,
    assemble_replay_physical_cycle_evidence,
)
from wind_prediction.physical_target_lifecycle import PhysicalTargetLifecycle


_ORIGIN = "2026-08-22T00:00:00"
_REFERENCE_MASSES = np.array([1_000.0, 1_000.0, 1_000.0])
_CAPACITIES = np.array([3_000.0, 3_000.0, 3_000.0])
_COORDINATES = np.array(
    [[-10.0, 0.0, -1.0], [5.0, 8.0, -1.0], [5.0, -8.0, -1.0]]
)


def _runtime() -> BallastRuntimeAssembly:
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
        provenance="physical_cycle_facts_test",
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


def _assert_same_execution_state(testcase: unittest.TestCase, left, right) -> None:
    """Compare the full actuator end state without reducing it to tank masses."""

    for attribute in (
        "actual_masses_kg",
        "rate_limited_target_kg",
        "primary_target_masses_kg",
        "signed_flow_m3_min",
        "pump_latched",
        "pump_on_elapsed_s",
        "pump_off_elapsed_s",
        "pump_near_target_s",
        "pump_command_rates_m3_min",
        "last_flow_directions",
    ):
        np.testing.assert_allclose(
            getattr(left, attribute),
            getattr(right, attribute),
            rtol=0.0,
            atol=1.0e-12,
            err_msg=attribute,
        )


def _assert_same_first_interval_response(
    testcase: unittest.TestCase,
    left,
    right,
) -> None:
    """Verify two factual lifecycle replays end at the same state."""

    _assert_same_execution_state(
        testcase,
        left.physical_path.final_execution_state,
        right.physical_path.final_execution_state,
    )
    for attribute in (
        "position",
        "velocity",
    ):
        np.testing.assert_allclose(
            getattr(left.execution_coupled_end_state, attribute),
            getattr(right.execution_coupled_end_state, attribute),
            rtol=0.0,
            atol=1.0e-12,
            err_msg=attribute,
        )
    for attribute in (
        "position_delta_from_frozen_tank_baseline",
        "velocity_delta_from_frozen_tank_baseline",
    ):
        np.testing.assert_allclose(
            getattr(left, attribute),
            getattr(right, attribute),
            rtol=0.0,
            atol=1.0e-12,
            err_msg=attribute,
        )


def _assert_same_lifecycle_responses(
    testcase: unittest.TestCase,
    left,
    right,
) -> None:
    """Compare every lifecycle instead of treating only new-track as evidence."""

    for lifecycle in (
        PhysicalTargetLifecycle.CONTINUE_EXISTING,
        PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        PhysicalTargetLifecycle.NEW_TRACK,
    ):
        _assert_same_first_interval_response(
            testcase,
            left.response_for(lifecycle),
            right.response_for(lifecycle),
        )


def _physical_facts(
    *,
    future_uv_ms: np.ndarray,
    lead_reliability: tuple[float, float] = (0.9, 0.8),
    event_probability: float = 0.6,
    origin_time: str | None = _ORIGIN,
    current_wind_observation_time: str = _ORIGIN,
    initial_platform_state_time: str = _ORIGIN,
    execution_state_time: str = _ORIGIN,
    environment_origin_time: str | None = _ORIGIN,
    environment_lead_times_s=None,
    current_enu_downwind_air_velocity_mps=(0.0, -7.0),
    current_reference_velocity_platform_mps=None,
    current_angular_velocity_platform_radps=None,
    current_wave_load=None,
    current_other_load=None,
    execution_state: ExecutionRolloutState | None = None,
    execution_config: ExecutionRolloutConfig | None = None,
):
    future_uv_ms = np.asarray(future_uv_ms, dtype=float)
    if future_uv_ms.ndim != 2 or future_uv_ms.shape[1] != 2 or future_uv_ms.shape[0] == 0:
        raise ValueError("future_uv_ms must have shape (H, 2) for positive H")
    future_steps = int(future_uv_ms.shape[0])
    runtime = _runtime()
    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime,
        actual_tank_masses_kg=_REFERENCE_MASSES,
        reference_tank_masses_kg=_REFERENCE_MASSES,
        tank_capacities_kg=_CAPACITIES,
        tank_coordinates_m=_COORDINATES,
    )
    if execution_state is None:
        execution_state = ExecutionRolloutState(
            masses_kg=_REFERENCE_MASSES,
            target_masses_kg=_REFERENCE_MASSES,
            primary_target_kg=_REFERENCE_MASSES,
            pump_rates_m3_min=np.zeros(3),
            pump_latched=np.zeros(3, dtype=bool),
            pump_off_elapsed_s=np.full(3, 60.0),
        )
    if execution_config is None:
        execution_config = _execution_config()
    evidence = ForecastEvidence(
        source="replay_lstm_fixture",
        model_version="fixture-v1",
        origin_time=origin_time,
        sample_period_s=600.0,
        uv_ms=future_uv_ms,
        lead_reliability=np.asarray(lead_reliability, dtype=float),
        event_probs={"wind_change": event_probability},
    )
    initial_state = IncrementalState(
        position=[0.0, 0.0, 0.0, 0.010, -0.015, 0.0],
        velocity=[0.0, 0.0, 0.0, -0.002, 0.003, 0.0],
    )
    angular_velocity = (
        initial_state.velocity[3:]
        if current_angular_velocity_platform_radps is None
        else current_angular_velocity_platform_radps
    )
    reference_velocity = (
        initial_state.velocity[:3]
        if current_reference_velocity_platform_mps is None
        else current_reference_velocity_platform_mps
    )
    if current_wave_load is None:
        current_wave_load = np.zeros(6)
    if current_other_load is None:
        current_other_load = np.zeros(6)
    environment_kwargs = {}
    if environment_origin_time is not None:
        environment_kwargs = {
            "forecast_origin_time": environment_origin_time,
            "lead_times_s": (
                [600.0 * (index + 1) for index in range(future_steps)]
                if environment_lead_times_s is None
                else environment_lead_times_s
            ),
        }
    return assemble_first_interval_physical_cycle_input(
        forecast=evidence,
        current_enu_downwind_air_velocity_mps=current_enu_downwind_air_velocity_mps,
        current_wind_source="same_origin_replay_observation",
        current_wind_observation_time=current_wind_observation_time,
        rotor_load_parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=2.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 10.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        ),
        rotor_operating_state=ForecastRotorOperatingState(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=[0.8] * future_steps,
            current_platform_reference_velocity_platform_mps=reference_velocity,
            current_platform_angular_velocity_platform_radps=angular_velocity,
            future_platform_reference_velocities_platform_mps=np.zeros((future_steps, 3)),
            future_platform_angular_velocities_platform_radps=np.zeros((future_steps, 3)),
            future_platform_kinematics_mode="frozen_zero",
            future_platform_kinematics_source="physical_cycle_facts_fixture",
        ),
        platform_snapshot=snapshot,
        initial_platform_state=initial_state,
        initial_platform_state_time=initial_platform_state_time,
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=current_wave_load,
            current_other_load=current_other_load,
            wave_interval_loads=np.zeros((future_steps, 6)),
            other_interval_loads=np.zeros((future_steps, 6)),
            provenance="explicit_zero_nonrotor_fixture",
            **environment_kwargs,
        ),
        runtime_assembly=runtime,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_state_time=execution_state_time,
    )


class _ReplayDataset:
    future_steps = 2
    event_columns = ("wind_change",)
    update_interval_s = 600.0

    def __init__(self, sample):
        self.sample = sample

    @staticmethod
    def simulation_timestamp(start_timestamp, current_time_s):
        return start_timestamp + timedelta(seconds=float(current_time_s))

    def sample_for_history_end(self, timestamp):
        if self.sample is None or timestamp != self.sample.history_end:
            return None
        return self.sample


class _InputOnlyForecastAdapter:
    """Small fake model whose future vector has no access to replay labels."""

    model_version = "input-only-fixture"
    provides_future_preview = True
    thresholds = {"wind_change": 0.5}

    @staticmethod
    def predict_window(x_window, timestamp=None):
        del x_window
        future_uv = np.array([[0.0, -10.0], [2.0, -12.0]], dtype=np.float32)
        return ForecastResult(
            wind_uv_raw=future_uv,
            wind_speed=np.array([10.0, np.sqrt(148.0)], dtype=np.float32),
            wind_dir_deg=np.array([0.0, 350.5377], dtype=np.float32),
            event_probs={"wind_change": 0.8},
            risk_levels={"wind_change": True},
            model_version="input-only-fixture",
            timestamp=timestamp,
        )


def _replay_source_physical_evidence(sample):
    """Run one replay source call into the physical-cycle fixture."""

    origin_time = sample.history_end.strftime("%Y-%m-%d %H:%M:%S")
    runtime = _runtime()
    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime,
        actual_tank_masses_kg=_REFERENCE_MASSES,
        reference_tank_masses_kg=_REFERENCE_MASSES,
        tank_capacities_kg=_CAPACITIES,
        tank_coordinates_m=_COORDINATES,
    )
    execution_state = ExecutionRolloutState(
        masses_kg=_REFERENCE_MASSES,
        target_masses_kg=_REFERENCE_MASSES,
        primary_target_kg=_REFERENCE_MASSES,
        pump_rates_m3_min=np.zeros(3),
        pump_latched=np.zeros(3, dtype=bool),
        pump_off_elapsed_s=np.full(3, 60.0),
    )
    initial_state = IncrementalState(
        position=[0.0, 0.0, 0.0, 0.010, -0.015, 0.0],
        velocity=[0.0, 0.0, 0.0, -0.002, 0.003, 0.0],
    )
    source = ReplayForecastEvidenceSource(
        replay_dataset=_ReplayDataset(sample),
        start_timestamp=sample.history_end,
        forecast_adapter=_InputOnlyForecastAdapter(),
        lead_reliability=np.array([1.0, 0.8]),
    )
    return assemble_replay_physical_cycle_evidence(
        forecast_source=source,
        current_time_s=0.0,
        rotor_load_parameters=ForecastRotorLoadParameters(
            air_density_kg_m3=1.225,
            rotor_radius_m=2.0,
            wind_reference_height_m=100.0,
            rotor_plane_height_m=100.0,
            power_law_shear_exponent=0.0,
            frozen_equilibrium_heading_rad=np.pi,
            rotor_force_application_point_from_reference_m=[0.0, 0.0, 10.0],
            downwind_rotor_normal_platform=[1.0, 0.0, 0.0],
        ),
        rotor_operating_state=ForecastRotorOperatingState(
            current_thrust_coefficient=0.8,
            future_thrust_coefficients=[0.8, 0.8],
            current_platform_reference_velocity_platform_mps=initial_state.velocity[:3],
            current_platform_angular_velocity_platform_radps=initial_state.velocity[3:],
            future_platform_reference_velocities_platform_mps=np.zeros((2, 3)),
            future_platform_angular_velocities_platform_radps=np.zeros((2, 3)),
            future_platform_kinematics_mode="frozen_zero",
            future_platform_kinematics_source="replay_source_physical_cycle_fixture",
        ),
        platform_snapshot=snapshot,
        initial_platform_state=initial_state,
        initial_platform_state_time=origin_time,
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            wave_interval_loads=np.zeros((2, 6)),
            other_interval_loads=np.zeros((2, 6)),
            forecast_origin_time=origin_time,
            lead_times_s=[600.0, 1_200.0],
            provenance="explicit_zero_nonrotor_replay_fixture",
        ),
        runtime_assembly=runtime,
        execution_state=execution_state,
        execution_config=_execution_config(),
        execution_state_time=origin_time,
    )


def _replay_source_physical_facts(sample):
    """Return the compatibility view used by the pre-existing tests."""

    evidence = _replay_source_physical_evidence(sample)
    assert evidence is not None
    return evidence.decision_input


class FirstIntervalPhysicalCycleFactsTests(unittest.TestCase):
    def test_replay_boundary_keeps_the_forecast_used_for_physical_input(self):
        origin = datetime(2026, 8, 22, 0, 0, 0)
        sample = SimpleNamespace(
            series_id="replay-evidence-fixture",
            x_window=np.zeros((12, 4), dtype=np.float32),
            history_start=origin - timedelta(minutes=110),
            history_end=origin,
            wind_obs={"ws": 8.0, "wd_deg": 0.0},
            y_uv_raw=np.zeros((2, 2), dtype=np.float32),
            y_event=np.zeros(1, dtype=np.float32),
        )

        evidence = _replay_source_physical_evidence(sample)

        self.assertIsNotNone(evidence)
        self.assertEqual(
            evidence.forecast.origin_time,
            origin.strftime("%Y-%m-%d %H:%M:%S"),
        )
        self.assertEqual(
            evidence.forecast.origin_time,
            evidence.decision_input.trajectory.initial_state_time,
        )
        self.assertEqual(
            evidence.forecast.source,
            evidence.decision_input.trajectory.load_assembly.input_identity.forecast_source,
        )

        decision = assemble_physical_current_cycle_decision_from_input(
            decision_input=evidence.decision_input,
            forecast_admission=PhysicalForecastAdmission(
                decision_input=evidence.decision_input,
                forecast=evidence.forecast,
                admitted=True,
                basis="fixture replay admission",
            ),
            posture_limits=PhysicalPostureLimits(
                max_abs_pitch_roll_rad=(1.0, 1.0),
                max_abs_pitch_roll_rate_rad_s=(1.0, 1.0),
                source="fixture broad posture range",
            ),
            policy=PhysicalCurrentCycleDecisionPolicy(
                posture_equivalence_margin=0.05,
            ),
        )
        self.assertIsNotNone(decision.current_execution_request)
        self.assertIsNotNone(decision.selected_current_block_rollout)
        self.assertIs(
            decision.selected_current_block_rollout.lifecycle_trace.execution_request,
            decision.current_execution_request,
        )

    def test_source_bound_replay_path_ignores_future_truth_labels(self):
        origin = datetime(2026, 8, 22, 0, 0, 0)
        shared = {
            "series_id": "same-origin-replay-fixture",
            "x_window": np.zeros((12, 4), dtype=np.float32),
            "history_start": origin - timedelta(minutes=110),
            "history_end": origin,
            "wind_obs": {"ws": 8.0, "wd_deg": 0.0},
        }
        first = SimpleNamespace(
            **shared,
            y_uv_raw=np.full((2, 2), 999.0, dtype=np.float32),
            y_event=np.array([0.0], dtype=np.float32),
        )
        second = SimpleNamespace(
            **shared,
            y_uv_raw=np.full((2, 2), -999.0, dtype=np.float32),
            y_event=np.array([1.0], dtype=np.float32),
        )

        first_facts = _replay_source_physical_facts(first)
        second_facts = _replay_source_physical_facts(second)

        self.assertFalse(np.array_equal(first.y_uv_raw, second.y_uv_raw))
        self.assertFalse(np.array_equal(first.y_event, second.y_event))
        first_loads = first_facts.trajectory.load_assembly
        second_loads = second_facts.trajectory.load_assembly
        self.assertEqual(
            first_loads.input_identity.current_wind_source,
            "replay_origin_observation",
        )
        np.testing.assert_allclose(
            first_loads.current_rotor_plane_enu_downwind_mps,
            [0.0, -8.0],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            first_loads.current_rotor_load.generalized_load_platform,
            second_loads.current_rotor_load.generalized_load_platform,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            first_loads.load_forecast.future_generalized_loads,
            second_loads.load_forecast.future_generalized_loads,
            rtol=0.0,
            atol=1.0e-12,
        )
        _assert_same_lifecycle_responses(self, first_facts, second_facts)

    def test_assembles_same_origin_unranked_physical_chain(self):
        facts = _physical_facts(future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]))

        preview = facts.new_track.lifecycle_trace.source_preview
        rhs_diagnostic = preview.rhs_diagnostic
        first_rhs_point = rhs_diagnostic.rhs_point
        self.assertEqual(facts.trajectory.initial_state_time, _ORIGIN)
        self.assertEqual(facts.lifecycle_facts.execution_start_time, _ORIGIN)
        self.assertEqual(
            facts.trajectory.interval_load_convention,
            FORECAST_INTERVAL_LOAD_CONVENTION,
        )
        self.assertIs(first_rhs_point.trajectory, facts.trajectory)
        self.assertIs(rhs_diagnostic.rhs_point, first_rhs_point)
        self.assertIs(
            preview.rhs_diagnostic,
            rhs_diagnostic,
        )
        self.assertIs(
            facts.lifecycle_facts.new_track.source_preview,
            preview,
        )
        self.assertEqual(first_rhs_point.lead_time_s, 600.0)
        self.assertEqual(
            facts.new_track.lifecycle_trace.lifecycle,
            PhysicalTargetLifecycle.NEW_TRACK,
        )
        self.assertEqual(
            facts.continue_existing.lifecycle_trace.lifecycle,
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self.assertEqual(
            facts.release_to_current.lifecycle_trace.lifecycle,
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        self.assertIs(facts.current_rhs.trajectory, facts.trajectory)
        self.assertEqual(
            len(facts.horizon_rhs_ballast_diagnostics),
            len(facts.trajectory.steps),
        )
        self.assertEqual(
            len(facts.horizon_endpoint_previews.previews),
            len(facts.trajectory.steps),
        )
        self.assertIs(
            facts.horizon_endpoint_previews.first_preview,
            preview,
        )
        self.assertIs(
            facts.horizon_endpoint_previews.preview_for_lead(0).rhs_diagnostic,
            rhs_diagnostic,
        )
        self.assertIs(
            facts.horizon_rhs_ballast_diagnostics[0].rhs_point,
            first_rhs_point,
        )
        self.assertIs(
            facts.current_rhs.posture_passive_load.platform_snapshot,
            facts.trajectory.platform_snapshot,
        )
        self.assertIs(
            facts.current_rhs.posture_passive_load.platform_state,
            facts.trajectory.initial_state,
        )

    def test_rejects_current_rhs_from_a_different_trajectory(self):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        foreign = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]])
        )

        with self.assertRaisesRegex(
            ValueError,
            "must use the shared forecast trajectory",
        ):
            replace(
                facts,
                current_rhs=foreign.current_rhs,
            )

    def test_future_wind_changes_new_track_but_not_existing_lifecycle_operations(self):
        forecast = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        persistence = _physical_facts(
            future_uv_ms=np.array([[0.0, -7.0], [0.0, -7.0]])
        )

        self.assertFalse(
            np.allclose(
                forecast.trajectory.load_assembly.load_forecast.future_generalized_loads,
                persistence.trajectory.load_assembly.load_forecast.future_generalized_loads,
            )
        )
        self.assertFalse(
            np.allclose(
                forecast.new_track.requested_target_masses_kg,
                persistence.new_track.requested_target_masses_kg,
            )
        )
        np.testing.assert_allclose(
            forecast.continue_existing.requested_target_masses_kg,
            persistence.continue_existing.requested_target_masses_kg,
        )
        np.testing.assert_allclose(
            forecast.release_to_current.requested_target_masses_kg,
            persistence.release_to_current.requested_target_masses_kg,
        )

    def test_current_observation_changes_current_load_not_forecast_shadow(self):
        future_uv = np.array([[0.0, -10.0], [2.0, -12.0]])
        reference = _physical_facts(
            future_uv_ms=future_uv,
            current_enu_downwind_air_velocity_mps=[0.0, -7.0],
        )
        changed_current = _physical_facts(
            future_uv_ms=future_uv,
            current_enu_downwind_air_velocity_mps=[3.0, -9.0],
        )

        self.assertFalse(
            np.allclose(
                reference.trajectory.load_assembly.current_rotor_load.generalized_load_platform,
                changed_current.trajectory.load_assembly.current_rotor_load.generalized_load_platform,
            )
        )
        self.assertFalse(
            np.allclose(
                reference.current_rhs.loads.wind,
                changed_current.current_rhs.loads.wind,
            )
        )
        self.assertFalse(
            np.allclose(
                reference.current_rhs.dynamic_rhs_generalized_load,
                changed_current.current_rhs.dynamic_rhs_generalized_load,
            )
        )
        np.testing.assert_allclose(
            reference.trajectory.load_assembly.load_forecast.future_generalized_loads,
            changed_current.trajectory.load_assembly.load_forecast.future_generalized_loads,
        )
        np.testing.assert_allclose(
            reference.new_track.requested_target_masses_kg,
            changed_current.new_track.requested_target_masses_kg,
        )
        _assert_same_lifecycle_responses(self, reference, changed_current)

    def test_forecast_tail_changes_later_shadow_not_first_interval_response(self):
        reference = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        changed_tail = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [-4.0, -14.0]])
        )

        np.testing.assert_allclose(
            reference.trajectory.steps[0].rotor_load.generalized_load_platform,
            changed_tail.trajectory.steps[0].rotor_load.generalized_load_platform,
        )
        self.assertFalse(
            np.allclose(
                reference.trajectory.steps[1].rotor_load.generalized_load_platform,
                changed_tail.trajectory.steps[1].rotor_load.generalized_load_platform,
            )
        )
        np.testing.assert_allclose(
            reference.new_track.requested_target_masses_kg,
            changed_tail.new_track.requested_target_masses_kg,
        )
        np.testing.assert_allclose(
            reference.horizon_rhs_ballast_diagnostics[0].hypothetical_tank_masses_kg,
            changed_tail.horizon_rhs_ballast_diagnostics[0].hypothetical_tank_masses_kg,
        )
        self.assertFalse(
            np.allclose(
                reference.horizon_rhs_ballast_diagnostics[1].hypothetical_tank_masses_kg,
                changed_tail.horizon_rhs_ballast_diagnostics[1].hypothetical_tank_masses_kg,
            )
        )
        np.testing.assert_allclose(
            reference.horizon_endpoint_previews.preview_for_lead(0).target_tank_masses_kg,
            changed_tail.horizon_endpoint_previews.preview_for_lead(0).target_tank_masses_kg,
        )
        self.assertFalse(
            np.allclose(
                reference.horizon_endpoint_previews.preview_for_lead(
                    1
                ).target_tank_masses_kg,
                changed_tail.horizon_endpoint_previews.preview_for_lead(
                    1
                ).target_tank_masses_kg,
            )
        )
        self.assertFalse(
            np.allclose(
                reference.trajectory.steps[1].end_state.position,
                changed_tail.trajectory.steps[1].end_state.position,
            )
        )
        _assert_same_lifecycle_responses(self, reference, changed_tail)

    def test_nonphysical_reliability_and_event_metadata_do_not_change_physical_facts(self):
        high = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            lead_reliability=(1.0, 1.0),
            event_probability=1.0,
        )
        low = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            lead_reliability=(0.0, 0.0),
            event_probability=0.0,
        )

        np.testing.assert_allclose(
            high.trajectory.load_assembly.load_forecast.future_generalized_loads,
            low.trajectory.load_assembly.load_forecast.future_generalized_loads,
        )
        np.testing.assert_allclose(
            high.new_track.requested_target_masses_kg,
            low.new_track.requested_target_masses_kg,
        )
        np.testing.assert_allclose(
            high.new_track.reached_final_tank_masses_kg,
            low.new_track.reached_final_tank_masses_kg,
        )

    def test_rejects_a_forecast_without_an_origin_time(self):
        with self.assertRaisesRegex(ValueError, "forecast.origin_time is required"):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                origin_time=None,
            )

    def test_rejects_a_current_wind_time_from_another_cycle(self):
        with self.assertRaisesRegex(
            ValueError,
            "current_wind_observation_time must match forecast.origin_time",
        ):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                current_wind_observation_time="2026-08-22T00:10:00",
            )

    def test_rejects_an_initial_platform_state_from_another_cycle(self):
        with self.assertRaisesRegex(
            ValueError,
            "initial_platform_state_time must match forecast.origin_time",
        ):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                initial_platform_state_time="2026-08-22T00:10:00",
            )

    def test_rejects_an_execution_state_from_another_cycle(self):
        with self.assertRaisesRegex(
            ValueError,
            "execution_state_time must match forecast.origin_time",
        ):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                execution_state_time="2026-08-22T00:10:00",
            )

    def test_rejects_unbound_or_misaligned_nonrotor_environment_loads(self):
        with self.assertRaisesRegex(
            ValueError,
            "environment must carry the forecast origin and lead-time grid",
        ):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                environment_origin_time=None,
            )
        with self.assertRaisesRegex(
            ValueError,
            "environment lead_times_s must match forecast lead times",
        ):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                environment_lead_times_s=[600.0, 1_201.0],
            )

    def test_rejects_current_rotor_kinematics_from_a_different_platform_state(self):
        with self.assertRaisesRegex(ValueError, "must match the initial platform state"):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                current_angular_velocity_platform_radps=[0.0, 0.0, 0.0],
            )
        with self.assertRaisesRegex(ValueError, "must match the initial platform state"):
            _physical_facts(
                future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
                current_reference_velocity_platform_mps=[1.0, 0.0, 0.0],
            )


if __name__ == "__main__":
    unittest.main()
