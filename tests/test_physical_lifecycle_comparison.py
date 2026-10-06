import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import _physical_facts
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.execution_rollout import ExecutionRolloutState
from wind_prediction.physical_forecast_admission import PhysicalForecastAdmission
from wind_prediction.physical_lifecycle_comparison import (
    PhysicalLifecycleComparison,
    PhysicalLifecycleOutcome,
    assemble_physical_lifecycle_comparison_for_current_block,
)
from wind_prediction.physical_target_lifecycle import PhysicalTargetLifecycle


def _bound_forecast(decision_input) -> ForecastEvidence:
    assembly = decision_input.trajectory.load_assembly
    identity = assembly.input_identity
    return ForecastEvidence(
        source=identity.forecast_source,
        model_version=identity.forecast_model_version,
        origin_time=identity.forecast_origin_time,
        sample_period_s=identity.forecast_sample_period_s,
        uv_ms=(
            np.asarray(assembly.future_rotor_plane_enu_downwind_mps, dtype=float)
            / assembly.parameters.rotor_plane_speed_scale
        ),
        lead_reliability=np.ones(identity.forecast_horizon_steps),
        event_probs={"fixture_event": 0.5},
    )


def _admission(decision_input, *, admitted=True) -> PhysicalForecastAdmission:
    return PhysicalForecastAdmission(
        decision_input=decision_input,
        forecast=_bound_forecast(decision_input),
        admitted=admitted,
        basis="fixture source-bound admission",
    )


def _in_progress_execution_state() -> ExecutionRolloutState:
    """Fixture with a stored target that pumps are already pursuing."""

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


class PhysicalLifecycleComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
        )

    def test_assembly_collects_one_same_origin_outcome_per_available_lifecycle(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )

        self.assertEqual(
            comparison.available_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                PhysicalTargetLifecycle.NEW_TRACK,
            ),
        )
        for lifecycle in comparison.available_lifecycles:
            outcome = comparison.outcome_for(lifecycle)
            rollout = outcome.rollout
            self.assertIs(rollout.trajectory, self.facts.trajectory)
            self.assertIs(rollout.runtime_assembly, self.facts.runtime_assembly)
            self.assertEqual(outcome.lifecycle, lifecycle)
            self.assertAlmostEqual(
                outcome.transferred_volume_m3,
                sum(
                    interval.physical_path.transferred_volume_m3
                    for interval in rollout.intervals
                ),
            )
            self.assertAlmostEqual(
                outcome.aggregate_pump_active_time_s,
                sum(
                    interval.physical_path.aggregate_pump_active_time_s
                    for interval in rollout.intervals
                ),
            )
            np.testing.assert_allclose(
                outcome.end_pitch_roll_rad,
                rollout.final_platform_state.position[[4, 3]],
            )
            np.testing.assert_allclose(
                outcome.peak_abs_pitch_roll_rad,
                rollout.post_origin_peak_abs_pitch_roll_rad,
            )

    def test_rejected_admission_excludes_forecast_specific_new_track_outcome(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts, admitted=False),
        )

        self.assertEqual(
            comparison.available_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
            ),
        )
        self.assertIsNone(comparison.new_track)
        with self.assertRaisesRegex(ValueError, "not available"):
            comparison.outcome_for(PhysicalTargetLifecycle.NEW_TRACK)

    def test_outcome_rejects_a_lifecycle_label_that_disagrees_with_its_rollout(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )

        with self.assertRaisesRegex(ValueError, "must match rollout.lifecycle"):
            PhysicalLifecycleOutcome(
                lifecycle=PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
                rollout=comparison.continue_existing.rollout,
            )

    def test_comparison_requires_the_admission_to_bind_the_same_physical_facts(self):
        other_facts = _physical_facts(
            future_uv_ms=np.array([[1.0, -11.0], [3.0, -13.0]]),
        )
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )

        with self.assertRaisesRegex(ValueError, "decision_input"):
            PhysicalLifecycleComparison(
                decision_input=other_facts,
                forecast_admission=_admission(self.facts),
                continue_existing=comparison.continue_existing,
                release_to_current=comparison.release_to_current,
                new_track=comparison.new_track,
            )

    def test_outcome_arrays_are_detached_from_the_rollout_state(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )
        result = comparison.new_track.end_pitch_roll_rad
        with self.assertRaises(ValueError):
            result[0] += 1.0

        np.testing.assert_allclose(
            comparison.new_track.end_pitch_roll_rad,
            comparison.new_track.rollout.final_platform_state.position[[4, 3]],
        )

    def test_release_remaining_mass_projects_the_execution_block_result(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )

        np.testing.assert_allclose(
            comparison.release_to_current.remaining_mass_to_target_kg,
            comparison.release_to_current.rollout.block_end_remaining_mass_to_target_kg,
        )

    def test_change_from_continuation_uses_one_explicit_difference_convention(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )

        change = comparison.change_from_continuation(
            PhysicalTargetLifecycle.NEW_TRACK
        )
        self.assertEqual(change.lifecycle, PhysicalTargetLifecycle.NEW_TRACK)
        self._assert_change_projection(change)

    def test_release_change_keeps_the_same_difference_convention_while_pumping(self):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_state=_in_progress_execution_state(),
        )
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=facts,
            forecast_admission=_admission(facts),
        )
        change = comparison.change_from_continuation(
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT
        )

        self.assertNotEqual(change.aggregate_pump_active_time_delta_s, 0.0)
        self._assert_change_projection(change)

    def _assert_change_projection(self, change) -> None:
        alternative = change.alternative
        continuation = change.continuation
        self.assertAlmostEqual(
            change.transferred_volume_delta_m3,
            alternative.transferred_volume_m3 - continuation.transferred_volume_m3,
        )
        self.assertAlmostEqual(
            change.aggregate_pump_active_time_delta_s,
            alternative.aggregate_pump_active_time_s
            - continuation.aggregate_pump_active_time_s,
        )
        self.assertEqual(
            change.pump_start_count_delta,
            alternative.pump_start_count - continuation.pump_start_count,
        )
        self.assertEqual(
            change.pump_stop_count_delta,
            alternative.pump_stop_count - continuation.pump_stop_count,
        )
        self.assertEqual(
            change.pump_direction_switch_count_delta,
            alternative.pump_direction_switch_count
            - continuation.pump_direction_switch_count,
        )
        np.testing.assert_allclose(
            change.end_pitch_roll_delta_rad,
            alternative.end_pitch_roll_rad - continuation.end_pitch_roll_rad,
        )
        np.testing.assert_allclose(
            change.peak_abs_pitch_roll_delta_rad,
            alternative.peak_abs_pitch_roll_rad
            - continuation.peak_abs_pitch_roll_rad,
        )
        np.testing.assert_allclose(
            change.end_pitch_roll_rate_delta_rad_s,
            alternative.end_pitch_roll_rate_rad_s
            - continuation.end_pitch_roll_rate_rad_s,
        )
        np.testing.assert_allclose(
            change.peak_abs_pitch_roll_rate_delta_rad_s,
            alternative.peak_abs_pitch_roll_rate_rad_s
            - continuation.peak_abs_pitch_roll_rate_rad_s,
        )
        np.testing.assert_allclose(
            change.reached_final_tank_mass_delta_kg,
            alternative.reached_final_tank_masses_kg
            - continuation.reached_final_tank_masses_kg,
        )

    def test_change_from_continuation_rejects_its_reference_operation(self):
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=self.facts,
            forecast_admission=_admission(self.facts),
        )

        with self.assertRaisesRegex(ValueError, "reference"):
            comparison.change_from_continuation(
                PhysicalTargetLifecycle.CONTINUE_EXISTING
            )


if __name__ == "__main__":
    unittest.main()
