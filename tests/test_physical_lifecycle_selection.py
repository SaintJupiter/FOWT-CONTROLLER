from dataclasses import replace
from datetime import datetime, timedelta
import unittest
from unittest.mock import patch

import numpy as np

from fowt_platform.ballast_snapshot import assemble_ballast_model_snapshot
from tests.test_physical_forecast_cycle_facts import (
    _assert_same_execution_state,
    _execution_config,
    _physical_facts,
)
from wind_prediction.execution_rollout import ExecutionRolloutState
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.forecast_execution_block_lifecycle_rollout import (
    rollout_current_execution_block_lifecycle,
)
from wind_prediction.forecast_physical_load import (
    ForecastRotorLoadParameters,
    ForecastRotorOperatingState,
)
from wind_prediction.forecast_platform_trajectory import ForecastTrajectoryEnvironment
from wind_prediction.physical_forecast_admission import PhysicalForecastAdmission
from wind_prediction.physical_lifecycle_comparison import (
    PhysicalLifecycleComparison,
    PhysicalLifecycleOutcome,
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from wind_prediction.physical_lifecycle_selection import (
    PhysicalLifecycleSelectionReason,
    PhysicalPostureLimits,
    assemble_physical_lifecycle_selection_for_current_block,
    select_physical_lifecycle_for_current_block,
)
from wind_prediction.physical_shadow_cycle import (
    PhysicalShadowCycleOutcome,
    PhysicalShadowCycleRequest,
    assemble_physical_shadow_cycle,
)
from wind_prediction.physical_forecast_cycle_facts import (
    assemble_first_interval_physical_cycle_input,
)
from wind_prediction.physical_target_lifecycle import PhysicalTargetLifecycle


_DEFAULT_NEW_TRACK = object()


def _in_progress_execution_state() -> ExecutionRolloutState:
    return ExecutionRolloutState(
        masses_kg=np.array([1_000.0, 1_000.0, 1_000.0]),
        target_masses_kg=np.array([1_300.0, 700.0, 1_000.0]),
        primary_target_kg=np.array([1_300.0, 700.0, 1_000.0]),
        pump_rates_m3_min=np.array([0.1, -0.1, 0.0]),
        pump_latched=np.array([True, True, False]),
        pump_on_elapsed_s=np.zeros(3),
        pump_off_elapsed_s=np.zeros(3),
        pump_command_rates_m3_min=np.array([0.1, 0.1, 0.0]),
        last_flow_directions=np.array([1.0, -1.0, 0.0]),
    )


def _limits(
    max_abs_pitch_roll_rad=(0.1, 0.1),
    max_abs_pitch_roll_rate_rad_s=(0.1, 0.1),
) -> PhysicalPostureLimits:
    return PhysicalPostureLimits(
        max_abs_pitch_roll_rad=max_abs_pitch_roll_rad,
        max_abs_pitch_roll_rate_rad_s=max_abs_pitch_roll_rate_rad_s,
        source="selection_fixture_limits",
    )


def _bound_forecast_for(
    decision_input,
    *,
    lead_reliability=None,
    event_probs=None,
) -> ForecastEvidence:
    """Reconstruct an equivalent forecast solely for source-binding tests."""

    assembly = decision_input.trajectory.load_assembly
    identity = assembly.input_identity
    if identity.forecast_origin_time is None:
        raise AssertionError("physical fixture requires a forecast origin")
    horizon = identity.forecast_horizon_steps
    if lead_reliability is None:
        lead_reliability = np.ones(horizon)
    if event_probs is None:
        event_probs = {"fixture_event": 0.5}
    return ForecastEvidence(
        source=identity.forecast_source,
        model_version=identity.forecast_model_version,
        origin_time=identity.forecast_origin_time,
        sample_period_s=identity.forecast_sample_period_s,
        uv_ms=(
            np.asarray(assembly.future_rotor_plane_enu_downwind_mps, dtype=float)
            / assembly.parameters.rotor_plane_speed_scale
        ),
        lead_reliability=np.asarray(lead_reliability, dtype=float),
        event_probs=dict(event_probs),
    )


def _admission(
    decision_input,
    *,
    admitted=True,
    basis="fixture admission decision",
    forecast=None,
) -> PhysicalForecastAdmission:
    return PhysicalForecastAdmission(
        decision_input=decision_input,
        forecast=(
            _bound_forecast_for(decision_input) if forecast is None else forecast
        ),
        admitted=admitted,
        basis=basis,
    )


def _assemble_shadow_cycle_facts(
    *,
    origin_time: str,
    future_uv_ms,
    current_enu_downwind_air_velocity_mps,
    platform_state,
    execution_state,
    runtime_assembly,
    reference_snapshot,
    execution_config,
):
    """Build one fresh, synthetic shadow-cycle record from a prior shadow state."""

    future = np.asarray(future_uv_ms, dtype=float)
    snapshot = assemble_ballast_model_snapshot(
        runtime_assembly=runtime_assembly,
        actual_tank_masses_kg=execution_state.actual_masses_kg,
        reference_tank_masses_kg=reference_snapshot.reference_tank_masses_kg,
        tank_capacities_kg=reference_snapshot.tank_capacities_kg,
        tank_coordinates_m=reference_snapshot.tank_coordinates_m,
    )
    facts = assemble_first_interval_physical_cycle_input(
        forecast=ForecastEvidence(
            source="synthetic_shadow_chain_forecast",
            model_version="fixture-v1",
            origin_time=origin_time,
            sample_period_s=600.0,
            uv_ms=future,
            lead_reliability=np.linspace(0.9, 0.8, future.shape[0]),
            event_probs={"wind_change": 0.6},
        ),
        current_enu_downwind_air_velocity_mps=(
            current_enu_downwind_air_velocity_mps
        ),
        current_wind_source="synthetic_shadow_chain_observation",
        current_wind_observation_time=origin_time,
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
            future_thrust_coefficients=np.full(future.shape[0], 0.8),
            current_platform_reference_velocity_platform_mps=(
                platform_state.velocity[:3]
            ),
            current_platform_angular_velocity_platform_radps=(
                platform_state.velocity[3:]
            ),
            future_platform_reference_velocities_platform_mps=np.zeros(
                (future.shape[0], 3)
            ),
            future_platform_angular_velocities_platform_radps=np.zeros(
                (future.shape[0], 3)
            ),
            future_platform_kinematics_mode="frozen_zero",
            future_platform_kinematics_source="synthetic_shadow_chain_frozen_zero",
        ),
        platform_snapshot=snapshot,
        initial_platform_state=platform_state,
        initial_platform_state_time=origin_time,
        environment=ForecastTrajectoryEnvironment(
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            wave_interval_loads=np.zeros((future.shape[0], 6)),
            other_interval_loads=np.zeros((future.shape[0], 6)),
            forecast_origin_time=origin_time,
            lead_times_s=np.arange(1, future.shape[0] + 1, dtype=float) * 600.0,
            provenance="synthetic_shadow_chain_explicit_zero_nonrotor_loads",
        ),
        runtime_assembly=runtime_assembly,
        execution_state=execution_state,
        execution_config=execution_config,
        execution_state_time=origin_time,
    )
    return facts, snapshot


class PhysicalLifecycleSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_state=_in_progress_execution_state(),
            execution_config=replace(
                _execution_config(),
                min_on_s=1_500.0,
                ramp_down_m3_min_per_s=0.0,
            ),
        )
        lifecycle_facts = cls.facts.lifecycle_facts
        cls.continue_rollout = rollout_current_execution_block_lifecycle(
            trajectory=cls.facts.trajectory,
            lifecycle_trace=lifecycle_facts.continue_existing,
            runtime_assembly=cls.facts.runtime_assembly,
        )
        cls.release_rollout = rollout_current_execution_block_lifecycle(
            trajectory=cls.facts.trajectory,
            lifecycle_trace=lifecycle_facts.release_to_current,
            runtime_assembly=cls.facts.runtime_assembly,
        )
        assert lifecycle_facts.new_track is not None
        cls.new_track_rollout = rollout_current_execution_block_lifecycle(
            trajectory=cls.facts.trajectory,
            lifecycle_trace=lifecycle_facts.new_track,
            runtime_assembly=cls.facts.runtime_assembly,
        )

    def _select(
        self,
        *,
        limits,
        forecast_admitted=True,
        forecast_admission=None,
        new_track_rollout=_DEFAULT_NEW_TRACK,
    ):
        if new_track_rollout is _DEFAULT_NEW_TRACK:
            new_track_rollout = self.new_track_rollout
        if forecast_admission is None:
            forecast_admission = _admission(
                self.facts,
                admitted=forecast_admitted,
            )
        if not forecast_admission.admitted:
            new_track_rollout = None
        comparison = PhysicalLifecycleComparison(
            decision_input=self.facts,
            forecast_admission=forecast_admission,
            continue_existing=PhysicalLifecycleOutcome(
                lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
                rollout=self.continue_rollout,
            ),
            release_to_current=PhysicalLifecycleOutcome(
                lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                rollout=self.release_rollout,
            ),
            new_track=(
                None
                if new_track_rollout is None
                else PhysicalLifecycleOutcome(
                    lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
                    rollout=new_track_rollout,
                )
            ),
        )
        return select_physical_lifecycle_for_current_block(
            decision_input=self.facts,
            posture_limits=limits,
            forecast_admission=forecast_admission,
            comparison=comparison,
            horizon_comparison=assemble_physical_lifecycle_horizon_comparison(
                current_block_comparison=comparison,
            ),
        )

    @staticmethod
    def _shadow_request(
        facts,
        *,
        forecast_admitted=True,
        forecast_admission_reason="explicit synthetic shadow-chain admission",
        forecast_admission=None,
    ):
        if forecast_admission is None:
            forecast_admission = _admission(
                facts,
                admitted=forecast_admitted,
                basis=forecast_admission_reason,
            )
        return PhysicalShadowCycleRequest(
            decision_input=facts,
            posture_limits=_limits(),
            forecast_admission=forecast_admission,
        )

    def test_within_limits_continuation_is_preserved_without_ranking_alternatives(self):
        selection = self._select(limits=_limits())

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                PhysicalTargetLifecycle.NEW_TRACK,
            ),
        )
        self.assertIs(
            selection.selected_lifecycle,
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.PRESERVE_EXISTING,
        )

    def test_one_eligible_alternative_is_selected_after_continuation_exceeds_limit(self):
        selection = self._select(
            limits=_limits(
                max_abs_pitch_roll_rad=(0.020, 0.012),
                max_abs_pitch_roll_rate_rad_s=(0.020, 0.020),
            )
        )

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (PhysicalTargetLifecycle.RELEASE_TO_CURRENT,),
        )
        self.assertIs(
            selection.selected_lifecycle,
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.UNIQUE_ELIGIBLE_ALTERNATIVE,
        )

    def test_later_forecast_lead_can_replace_currently_feasible_continuation(self):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [15.0, 0.0]]),
            execution_state=_in_progress_execution_state(),
            execution_config=replace(
                _execution_config(),
                block_duration_s=600.0,
                min_on_s=1_500.0,
                ramp_down_m3_min_per_s=0.0,
            ),
        )
        admission = _admission(facts)
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=facts,
            forecast_admission=admission,
        )
        horizon = assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=comparison,
        )
        limit = _limits(
            max_abs_pitch_roll_rad=(0.025, 0.1),
            max_abs_pitch_roll_rate_rad_s=(0.1, 0.1),
        )

        self.assertLess(
            comparison.continue_existing.peak_abs_pitch_roll_rad[0],
            limit.max_abs_pitch_roll_rad[0],
        )
        self.assertGreater(
            horizon.continue_existing.peak_abs_pitch_roll_rad[0],
            limit.max_abs_pitch_roll_rad[0],
        )

        selection = select_physical_lifecycle_for_current_block(
            decision_input=facts,
            posture_limits=limit,
            forecast_admission=admission,
            comparison=comparison,
            horizon_comparison=horizon,
        )

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (PhysicalTargetLifecycle.RELEASE_TO_CURRENT,),
        )
        self.assertIs(
            selection.selected_lifecycle,
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        self.assertIs(
            selection.require_selected_rollout(),
            comparison.release_to_current.rollout,
        )

    def test_multiple_eligible_alternatives_remain_unselected(self):
        selection = self._select(
            limits=_limits(
                max_abs_pitch_roll_rad=(0.1, 0.1),
                max_abs_pitch_roll_rate_rad_s=(0.01138, 0.1),
            )
        )

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                PhysicalTargetLifecycle.NEW_TRACK,
            ),
        )
        self.assertIsNone(selection.selected_lifecycle)
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.MULTIPLE_ELIGIBLE_ALTERNATIVES,
        )

    def test_all_response_paths_outside_limits_remain_unselected(self):
        selection = self._select(
            limits=_limits(
                max_abs_pitch_roll_rad=(0.0151, 0.0101),
                max_abs_pitch_roll_rate_rad_s=(0.1, 0.1),
            )
        )

        self.assertEqual(selection.selection_eligible_lifecycles, ())
        self.assertIsNone(selection.selected_lifecycle)
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.NO_ELIGIBLE_LIFECYCLE,
        )

    def test_initial_state_outside_scope_is_not_written_as_candidate_failure(self):
        selection = self._select(
            limits=_limits(
                max_abs_pitch_roll_rad=(0.0149, 0.1),
                max_abs_pitch_roll_rate_rad_s=(0.1, 0.1),
            )
        )

        self.assertEqual(selection.selection_eligible_lifecycles, ())
        self.assertIsNone(selection.selected_lifecycle)
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.INITIAL_STATE_OUTSIDE_SELECTION_SCOPE,
        )

    def test_forecast_not_admitted_does_not_require_a_new_track_rollout(self):
        selection = self._select(
            limits=_limits(
                max_abs_pitch_roll_rad=(0.0149, 0.1),
                max_abs_pitch_roll_rate_rad_s=(0.1, 0.1),
            ),
            forecast_admitted=False,
            new_track_rollout=None,
        )

        self.assertIsNone(selection.new_track_rollout)
        self.assertEqual(selection.selection_eligible_lifecycles, ())
        self.assertIsNone(selection.selected_lifecycle)
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.FORECAST_NOT_ADMITTED,
        )

    def test_admitted_forecast_without_new_track_facts_compares_existing_paths(self):
        decision_input_without_new_track = replace(
            self.facts,
            lifecycle_facts=replace(self.facts.lifecycle_facts, new_track=None),
            new_track=None,
            new_track_state_conditioned_rhs=None,
        )
        forecast_admission = _admission(decision_input_without_new_track)
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=decision_input_without_new_track,
            forecast_admission=forecast_admission,
        )
        selection = select_physical_lifecycle_for_current_block(
            decision_input=decision_input_without_new_track,
            posture_limits=_limits(),
            forecast_admission=forecast_admission,
            comparison=comparison,
            horizon_comparison=assemble_physical_lifecycle_horizon_comparison(
                current_block_comparison=comparison,
            ),
        )

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
            ),
        )
        self.assertIs(
            selection.selected_lifecycle,
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.PRESERVE_EXISTING,
        )

    def test_admitted_forecast_rejects_an_omitted_existing_new_track_rollout(self):
        with self.assertRaisesRegex(ValueError, "new_track is required"):
            self._select(
                limits=_limits(),
                forecast_admitted=True,
                new_track_rollout=None,
            )

    def test_rejects_rollout_from_a_different_physical_fact_input(self):
        foreign_facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_state=_in_progress_execution_state(),
            execution_config=replace(
                _execution_config(),
                min_on_s=1_500.0,
                ramp_down_m3_min_per_s=0.0,
            ),
        )
        foreign_rollout = rollout_current_execution_block_lifecycle(
            trajectory=foreign_facts.trajectory,
            lifecycle_trace=foreign_facts.lifecycle_facts.continue_existing,
            runtime_assembly=foreign_facts.runtime_assembly,
        )

        with self.assertRaisesRegex(ValueError, "shared forecast trajectory"):
            PhysicalLifecycleComparison(
                decision_input=self.facts,
                forecast_admission=_admission(self.facts),
                continue_existing=PhysicalLifecycleOutcome(
                    lifecycle=PhysicalTargetLifecycle.CONTINUE_EXISTING,
                    rollout=foreign_rollout,
                ),
                release_to_current=PhysicalLifecycleOutcome(
                    lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                    rollout=self.release_rollout,
                ),
                new_track=PhysicalLifecycleOutcome(
                    lifecycle=PhysicalTargetLifecycle.NEW_TRACK,
                    rollout=self.new_track_rollout,
                ),
            )

    def test_rejects_forecast_admission_bound_to_different_physical_facts(self):
        foreign_facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_state=_in_progress_execution_state(),
            execution_config=replace(
                _execution_config(),
                min_on_s=1_500.0,
                ramp_down_m3_min_per_s=0.0,
            ),
        )

        with self.assertRaisesRegex(ValueError, "forecast_admission"):
            self._select(
                limits=_limits(),
                forecast_admission=_admission(foreign_facts),
            )

    def test_assembly_replays_the_complete_admitted_candidate_set(self):
        selection = assemble_physical_lifecycle_selection_for_current_block(
            decision_input=self.facts,
            posture_limits=_limits(),
            forecast_admission=_admission(self.facts),
        )

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                PhysicalTargetLifecycle.NEW_TRACK,
            ),
        )
        self.assertIsNotNone(selection.new_track_rollout)

    def test_assembly_allows_admitted_existing_paths_when_parent_has_no_new_track(self):
        decision_input_without_new_track = replace(
            self.facts,
            lifecycle_facts=replace(self.facts.lifecycle_facts, new_track=None),
            new_track=None,
            new_track_state_conditioned_rhs=None,
        )

        selection = assemble_physical_lifecycle_selection_for_current_block(
            decision_input=decision_input_without_new_track,
            posture_limits=_limits(),
            forecast_admission=_admission(decision_input_without_new_track),
        )

        self.assertEqual(
            selection.selection_eligible_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
            ),
        )
        self.assertIsNone(selection.new_track_rollout)

    def test_assembly_skips_new_track_when_forecast_is_not_admitted(self):
        from wind_prediction import physical_lifecycle_comparison

        original_rollout = (
            physical_lifecycle_comparison.rollout_current_execution_block_lifecycle
        )
        replayed_lifecycles = []

        def record_rollout(*, lifecycle_trace, **kwargs):
            replayed_lifecycles.append(lifecycle_trace.lifecycle)
            return original_rollout(lifecycle_trace=lifecycle_trace, **kwargs)

        with patch.object(
            physical_lifecycle_comparison,
            "rollout_current_execution_block_lifecycle",
            side_effect=record_rollout,
        ):
            selection = assemble_physical_lifecycle_selection_for_current_block(
                decision_input=self.facts,
                posture_limits=_limits(),
                forecast_admission=_admission(self.facts, admitted=False),
            )

        self.assertEqual(
            replayed_lifecycles,
            [
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
            ],
        )
        self.assertIsNone(selection.new_track_rollout)
        self.assertIsNone(selection.selected_lifecycle)
        self.assertIs(
            selection.selection_reason,
            PhysicalLifecycleSelectionReason.FORECAST_NOT_ADMITTED,
        )

    def test_assembly_does_not_silently_drop_a_required_rollout(self):
        from wind_prediction import physical_lifecycle_comparison

        original_rollout = (
            physical_lifecycle_comparison.rollout_current_execution_block_lifecycle
        )

        def fail_new_track_rollout(*, lifecycle_trace, **kwargs):
            if lifecycle_trace.lifecycle is PhysicalTargetLifecycle.NEW_TRACK:
                raise RuntimeError("new-track fixture failure")
            return original_rollout(lifecycle_trace=lifecycle_trace, **kwargs)

        with patch.object(
            physical_lifecycle_comparison,
            "rollout_current_execution_block_lifecycle",
            side_effect=fail_new_track_rollout,
        ):
            with self.assertRaisesRegex(RuntimeError, "new-track fixture failure"):
                assemble_physical_lifecycle_selection_for_current_block(
                    decision_input=self.facts,
                    posture_limits=_limits(),
                    forecast_admission=_admission(self.facts),
                )

    def test_selected_rollout_is_the_only_explicit_cross_cycle_state_source(self):
        selected = self._select(limits=_limits())

        self.assertIs(
            selected.require_selected_rollout(),
            selected.continue_existing_rollout,
        )
        self.assertIs(
            selected.require_selected_execution_request(),
            selected.continue_existing_rollout.lifecycle_trace.execution_request,
        )

        abstained = self._select(
            limits=_limits(),
            forecast_admitted=False,
        )
        with self.assertRaisesRegex(ValueError, "selection abstained"):
            abstained.require_selected_rollout()
        with self.assertRaisesRegex(ValueError, "selection abstained"):
            abstained.require_selected_execution_request()

    def test_selected_shadow_rollout_seeds_a_fresh_next_cycle_without_reusing_facts(self):
        selection = self._select(limits=_limits())
        selected_rollout = selection.require_selected_rollout()
        origin_snapshot = selection.decision_input.trajectory.platform_snapshot
        first_origin = selection.decision_input.trajectory.initial_state_time
        next_origin = (
            datetime.fromisoformat(first_origin)
            + timedelta(seconds=selected_rollout.duration_s)
        ).isoformat()

        next_facts, next_snapshot = _assemble_shadow_cycle_facts(
            origin_time=next_origin,
            future_uv_ms=np.array([[0.0, -11.0], [1.0, -12.0]]),
            current_enu_downwind_air_velocity_mps=(0.0, -10.0),
            platform_state=selected_rollout.final_platform_state,
            runtime_assembly=selected_rollout.runtime_assembly,
            execution_state=selected_rollout.final_execution_state,
            execution_config=selected_rollout.lifecycle_trace.execution_config,
            reference_snapshot=origin_snapshot,
        )

        self.assertEqual(next_facts.trajectory.initial_state_time, next_origin)
        self.assertEqual(
            next_facts.trajectory.initial_state_time,
            "2026-08-22T00:20:00",
        )
        np.testing.assert_allclose(
            next_facts.trajectory.initial_state.position,
            selected_rollout.final_platform_state.position,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            next_facts.trajectory.initial_state.velocity,
            selected_rollout.final_platform_state.velocity,
            rtol=0.0,
            atol=1.0e-12,
        )
        _assert_same_execution_state(
            self,
            next_facts.lifecycle_facts.continue_existing.execution_start_state,
            selected_rollout.final_execution_state,
        )
        np.testing.assert_allclose(
            next_snapshot.actual_tank_masses_kg,
            selected_rollout.final_execution_state.actual_masses_kg,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            next_snapshot.reference_tank_masses_kg,
            origin_snapshot.reference_tank_masses_kg,
            rtol=0.0,
            atol=1.0e-12,
        )
        self.assertFalse(
            np.allclose(
                next_snapshot.actual_tank_masses_kg,
                next_snapshot.reference_tank_masses_kg,
                rtol=0.0,
                atol=1.0e-12,
            )
        )
        self.assertIsNot(
            next_facts.lifecycle_facts.continue_existing,
            selection.decision_input.lifecycle_facts.continue_existing,
        )
        self.assertIsNot(next_facts.trajectory, selection.decision_input.trajectory)

    def test_six_hour_shadow_chain_keeps_selected_state_and_time_continuous(self):
        execution_config = replace(
            _execution_config(),
            internal_step_s=30.0,
            min_on_s=1_500.0,
            ramp_down_m3_min_per_s=0.0,
        )
        current_facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [1.0, -11.0]]),
            execution_state=_in_progress_execution_state(),
            execution_config=execution_config,
        )
        reference_snapshot = current_facts.trajectory.platform_snapshot
        current_time = datetime.fromisoformat(
            current_facts.trajectory.initial_state_time
        )
        start_time = current_time
        saw_actual_tank_change = False
        previous_outcome = None

        for block_index in range(18):
            outcome = assemble_physical_shadow_cycle(
                request=self._shadow_request(
                    current_facts,
                    forecast_admission_reason=(
                        "explicit synthetic smoke admission without policy claim"
                    ),
                ),
                previous=previous_outcome,
            )
            selection = outcome.selection
            self.assertIs(
                selection.selected_lifecycle,
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
            )
            self.assertIs(
                selection.selection_reason,
                PhysicalLifecycleSelectionReason.PRESERVE_EXISTING,
            )
            selected_rollout = outcome.require_selected_rollout()
            self.assertAlmostEqual(selected_rollout.duration_s, 1_200.0)
            self.assertEqual(
                current_facts.trajectory.initial_state_time,
                current_time.isoformat(),
            )
            if not np.allclose(
                selected_rollout.final_execution_state.actual_masses_kg,
                current_facts.lifecycle_facts.continue_existing.execution_start_state.actual_masses_kg,
                rtol=0.0,
                atol=1.0e-12,
            ):
                saw_actual_tank_change = True

            next_time = current_time + timedelta(seconds=selected_rollout.duration_s)
            if block_index == 17:
                current_time = next_time
                break

            next_facts, next_snapshot = _assemble_shadow_cycle_facts(
                origin_time=next_time.isoformat(),
                future_uv_ms=np.array(
                    [
                        [0.5 * ((block_index % 3) - 1), -10.0 - (block_index % 2)],
                        [1.0, -11.0 - (block_index % 3)],
                    ]
                ),
                current_enu_downwind_air_velocity_mps=(
                    0.25 * ((block_index % 3) - 1),
                    -8.0 - (block_index % 2),
                ),
                platform_state=selected_rollout.final_platform_state,
                execution_state=selected_rollout.final_execution_state,
                runtime_assembly=selected_rollout.runtime_assembly,
                reference_snapshot=reference_snapshot,
                execution_config=selected_rollout.lifecycle_trace.execution_config,
            )
            np.testing.assert_allclose(
                next_facts.trajectory.initial_state.position,
                selected_rollout.final_platform_state.position,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                next_facts.trajectory.initial_state.velocity,
                selected_rollout.final_platform_state.velocity,
                rtol=0.0,
                atol=1.0e-12,
            )
            _assert_same_execution_state(
                self,
                next_facts.lifecycle_facts.continue_existing.execution_start_state,
                selected_rollout.final_execution_state,
            )
            np.testing.assert_allclose(
                next_snapshot.actual_tank_masses_kg,
                selected_rollout.final_execution_state.actual_masses_kg,
                rtol=0.0,
                atol=1.0e-12,
            )
            np.testing.assert_allclose(
                next_snapshot.reference_tank_masses_kg,
                reference_snapshot.reference_tank_masses_kg,
                rtol=0.0,
                atol=1.0e-12,
            )
            self.assertIsNot(next_facts.trajectory, current_facts.trajectory)
            self.assertIsNot(
                next_facts.lifecycle_facts.continue_existing,
                current_facts.lifecycle_facts.continue_existing,
            )
            self.assertIs(next_facts.runtime_assembly, current_facts.runtime_assembly)
            current_facts = next_facts
            current_time = next_time
            previous_outcome = outcome

        self.assertEqual(current_time, start_time + timedelta(hours=6))
        self.assertTrue(saw_actual_tank_change)

    def test_shadow_cycle_abstention_exposes_no_cross_cycle_state(self):
        outcome = assemble_physical_shadow_cycle(
            request=self._shadow_request(
                self.facts,
                forecast_admitted=False,
                forecast_admission_reason="synthetic rejected forecast",
            )
        )

        self.assertFalse(outcome.is_selected)
        self.assertIsNone(outcome.selected_rollout)
        self.assertIs(
            outcome.selection.selection_reason,
            PhysicalLifecycleSelectionReason.FORECAST_NOT_ADMITTED,
        )
        with self.assertRaisesRegex(ValueError, "shadow cycle abstained"):
            outcome.require_selected_rollout()
        with self.assertRaisesRegex(ValueError, "selection abstained"):
            outcome.require_selected_execution_request()

        with self.assertRaisesRegex(ValueError, "shadow cycle abstained"):
            assemble_physical_shadow_cycle(
                request=self._shadow_request(self.facts),
                previous=outcome,
            )

    def test_shadow_cycle_outcome_rejects_mismatched_request_semantics(self):
        limits = _limits()
        admitted = _admission(
            self.facts,
            basis="synthetic admitted forecast",
        )
        selection = assemble_physical_lifecycle_selection_for_current_block(
            decision_input=self.facts,
            posture_limits=limits,
            forecast_admission=admitted,
        )
        conflicting_request = PhysicalShadowCycleRequest(
            decision_input=self.facts,
            posture_limits=limits,
            forecast_admission=_admission(
                self.facts,
                admitted=False,
                basis="synthetic admitted forecast",
            ),
        )
        with self.assertRaisesRegex(ValueError, "forecast_admission"):
            PhysicalShadowCycleOutcome(
                request=conflicting_request,
                selection=selection,
                selected_rollout=selection.require_selected_rollout(),
            )

        conflicting_reason = PhysicalShadowCycleRequest(
            decision_input=self.facts,
            posture_limits=limits,
            forecast_admission=_admission(
                self.facts,
                basis="different synthetic admission record",
            ),
        )
        with self.assertRaisesRegex(ValueError, "forecast_admission"):
            PhysicalShadowCycleOutcome(
                request=conflicting_reason,
                selection=selection,
                selected_rollout=selection.require_selected_rollout(),
            )

    def test_shadow_cycle_rejects_reused_or_stale_next_cycle_facts(self):
        first_outcome = assemble_physical_shadow_cycle(
            request=self._shadow_request(self.facts)
        )

        with self.assertRaisesRegex(ValueError, "fresh forecast trajectory"):
            assemble_physical_shadow_cycle(
                request=self._shadow_request(self.facts),
                previous=first_outcome,
            )

        selected_rollout = first_outcome.require_selected_rollout()
        reference_snapshot = self.facts.trajectory.platform_snapshot
        next_origin = (
            datetime.fromisoformat(self.facts.trajectory.initial_state_time)
            + timedelta(seconds=selected_rollout.duration_s)
        ).isoformat()
        stale_facts, _ = _assemble_shadow_cycle_facts(
            origin_time=next_origin,
            future_uv_ms=np.array([[0.0, -11.0], [1.0, -12.0]]),
            current_enu_downwind_air_velocity_mps=(0.0, -10.0),
            platform_state=self.facts.trajectory.initial_state,
            execution_state=selected_rollout.final_execution_state,
            runtime_assembly=selected_rollout.runtime_assembly,
            reference_snapshot=reference_snapshot,
            execution_config=selected_rollout.lifecycle_trace.execution_config,
        )
        with self.assertRaisesRegex(ValueError, "initial platform state"):
            assemble_physical_shadow_cycle(
                request=self._shadow_request(stale_facts),
                previous=first_outcome,
            )

        stale_pump_facts, _ = _assemble_shadow_cycle_facts(
            origin_time=next_origin,
            future_uv_ms=np.array([[0.0, -11.0], [1.0, -12.0]]),
            current_enu_downwind_air_velocity_mps=(0.0, -10.0),
            platform_state=selected_rollout.final_platform_state,
            execution_state=(
                self.facts.lifecycle_facts.continue_existing.execution_start_state
            ),
            runtime_assembly=selected_rollout.runtime_assembly,
            reference_snapshot=reference_snapshot,
            execution_config=selected_rollout.lifecycle_trace.execution_config,
        )
        with self.assertRaisesRegex(ValueError, "pump state"):
            assemble_physical_shadow_cycle(
                request=self._shadow_request(stale_pump_facts),
                previous=first_outcome,
            )

    def test_shadow_cycle_rejects_unannounced_model_or_pump_config_change(self):
        first_outcome = assemble_physical_shadow_cycle(
            request=self._shadow_request(self.facts)
        )
        selected_rollout = first_outcome.require_selected_rollout()
        reference_snapshot = self.facts.trajectory.platform_snapshot
        next_origin = (
            datetime.fromisoformat(self.facts.trajectory.initial_state_time)
            + timedelta(seconds=selected_rollout.duration_s)
        ).isoformat()

        shifted_coordinates = np.array(
            reference_snapshot.tank_coordinates_m,
            dtype=float,
            copy=True,
        )
        shifted_coordinates[0, 0] += 0.1
        changed_definition = assemble_ballast_model_snapshot(
            runtime_assembly=selected_rollout.runtime_assembly,
            actual_tank_masses_kg=selected_rollout.final_execution_state.actual_masses_kg,
            reference_tank_masses_kg=reference_snapshot.reference_tank_masses_kg,
            tank_capacities_kg=reference_snapshot.tank_capacities_kg,
            tank_coordinates_m=shifted_coordinates,
        )
        changed_model_facts, _ = _assemble_shadow_cycle_facts(
            origin_time=next_origin,
            future_uv_ms=np.array([[0.0, -11.0], [1.0, -12.0]]),
            current_enu_downwind_air_velocity_mps=(0.0, -10.0),
            platform_state=selected_rollout.final_platform_state,
            execution_state=selected_rollout.final_execution_state,
            runtime_assembly=selected_rollout.runtime_assembly,
            reference_snapshot=changed_definition,
            execution_config=selected_rollout.lifecycle_trace.execution_config,
        )
        with self.assertRaisesRegex(ValueError, "reference tank definition"):
            assemble_physical_shadow_cycle(
                request=self._shadow_request(changed_model_facts),
                previous=first_outcome,
            )

        changed_config_facts, _ = _assemble_shadow_cycle_facts(
            origin_time=next_origin,
            future_uv_ms=np.array([[0.0, -11.0], [1.0, -12.0]]),
            current_enu_downwind_air_velocity_mps=(0.0, -10.0),
            platform_state=selected_rollout.final_platform_state,
            execution_state=selected_rollout.final_execution_state,
            runtime_assembly=selected_rollout.runtime_assembly,
            reference_snapshot=reference_snapshot,
            execution_config=replace(
                selected_rollout.lifecycle_trace.execution_config,
                min_on_s=(
                    selected_rollout.lifecycle_trace.execution_config.min_on_s + 1.0
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "pump execution configuration"):
            assemble_physical_shadow_cycle(
                request=self._shadow_request(changed_config_facts),
                previous=first_outcome,
            )

    def test_limits_require_positive_finite_pitch_roll_pairs(self):
        with self.assertRaisesRegex(ValueError, "two positive finite values"):
            _limits(max_abs_pitch_roll_rad=(0.0, 0.1))
        with self.assertRaisesRegex(ValueError, "two positive finite values"):
            _limits(max_abs_pitch_roll_rate_rad_s=(0.1, np.nan))


if __name__ == "__main__":
    unittest.main()
