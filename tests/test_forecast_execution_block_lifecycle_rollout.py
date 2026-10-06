from dataclasses import replace
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import (
    _execution_config,
    _physical_facts,
    _runtime,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from wind_prediction.forecast_execution_block_lifecycle_rollout import (
    ForecastExecutionBlockInterval,
    ForecastExecutionBlockLifecycleRollout,
    rollout_current_execution_block_lifecycle,
)
from wind_prediction.physical_execution_platform_path import (
    advance_physical_execution_platform_path,
)


def _state_matches(testcase: unittest.TestCase, left, right) -> None:
    np.testing.assert_allclose(left.position, right.position, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(left.velocity, right.velocity, rtol=0.0, atol=1e-12)


def _pump_state_matches(testcase: unittest.TestCase, left, right) -> None:
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
            atol=1e-12,
            err_msg=attribute,
        )


def _physical_paths_match(testcase: unittest.TestCase, left, right) -> None:
    testcase.assertAlmostEqual(left.duration_s, right.duration_s)
    testcase.assertEqual(len(left.substeps), len(right.substeps))
    for actual, expected in zip(left.substeps, right.substeps):
        testcase.assertAlmostEqual(actual.duration_s, expected.duration_s)
        _state_matches(testcase, actual.start_platform_state, expected.start_platform_state)
        _state_matches(
            testcase,
            actual.platform_step.next_state,
            expected.platform_step.next_state,
        )
        _pump_state_matches(
            testcase,
            actual.start_execution_state,
            expected.start_execution_state,
        )
        _pump_state_matches(
            testcase,
            actual.execution_step.state,
            expected.execution_step.state,
        )
        for attribute in (
            "actual_tank_masses_kg",
            "reference_tank_masses_kg",
            "tank_capacities_kg",
            "tank_coordinates_m",
            "incremental_ballast_load",
        ):
            np.testing.assert_allclose(
                getattr(actual.ballast_snapshot, attribute),
                getattr(expected.ballast_snapshot, attribute),
                rtol=0.0,
                atol=1e-12,
                err_msg=attribute,
            )
        for channel in ("wind", "wave", "ballast", "other"):
            np.testing.assert_allclose(
                getattr(actual.platform_step.loads, channel),
                getattr(expected.platform_step.loads, channel),
                rtol=0.0,
                atol=1e-12,
                err_msg=channel,
            )
        testcase.assertIs(
            actual.execution_step.target_operation,
            expected.execution_step.target_operation,
        )
        for attribute in (
            "requested_target_kg",
            "shaped_target_kg",
            "mass_delta_kg",
            "pump_volume_m3",
            "pump_runtime_s",
        ):
            np.testing.assert_allclose(
                getattr(actual.execution_step, attribute),
                getattr(expected.execution_step, attribute),
                rtol=0.0,
                atol=1e-12,
                err_msg=attribute,
            )


class ForecastExecutionBlockLifecycleRolloutTests(unittest.TestCase):
    def _new_track(self, future_uv_ms: np.ndarray, **kwargs):
        facts = _physical_facts(future_uv_ms=future_uv_ms, **kwargs)
        trace = facts.lifecycle_facts.new_track
        assert trace is not None
        return facts, trace

    def test_first_interval_exactly_reuses_the_existing_lifecycle_replay(self):
        facts, trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]])
        )

        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )

        self.assertEqual(len(rollout.intervals), 2)
        self.assertAlmostEqual(rollout.duration_s, 1_200.0)
        first = rollout.intervals[0]
        _physical_paths_match(self, first.physical_path, facts.new_track.physical_path)
        _state_matches(
            self,
            first.final_platform_state,
            facts.new_track.execution_coupled_end_state,
        )
        _pump_state_matches(
            self,
            first.final_execution_state,
            facts.new_track.physical_path.final_execution_state,
        )
        self.assertEqual(rollout.intervals[1].lead_index, 1)
        _state_matches(
            self,
            rollout.intervals[1].physical_path.substeps[0].start_platform_state,
            first.final_platform_state,
        )
        _pump_state_matches(
            self,
            rollout.intervals[1].physical_path.substeps[0].start_execution_state,
            first.final_execution_state,
        )

    def test_rollout_exposes_only_read_only_facts_from_its_existing_path(self):
        facts, trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )

        self.assertIs(rollout.start_platform_state, facts.trajectory.initial_state)
        _pump_state_matches(self, rollout.start_execution_state, trace.execution_start_state)
        np.testing.assert_allclose(
            rollout.block_end_requested_target_masses_kg,
            rollout.intervals[-1].physical_path.substeps[-1].execution_step.requested_target_kg,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rollout.actual_tank_mass_delta_kg,
            rollout.final_execution_state.actual_masses_kg
            - rollout.start_execution_state.actual_masses_kg,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rollout.block_end_remaining_mass_to_target_kg,
            rollout.block_end_requested_target_masses_kg
            - rollout.final_execution_state.actual_masses_kg,
            rtol=0.0,
            atol=1e-12,
        )

        path_states = [
            substep.next_platform_state
            for interval in rollout.intervals
            for substep in interval.physical_path.substeps
        ]
        expected_peak_position = np.max(
            np.abs([state.position[[4, 3]] for state in path_states]), axis=0
        )
        expected_peak_velocity = np.max(
            np.abs([state.velocity[[4, 3]] for state in path_states]), axis=0
        )
        np.testing.assert_allclose(
            rollout.post_origin_peak_abs_pitch_roll_rad,
            expected_peak_position,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rollout.post_origin_peak_abs_pitch_roll_rate_rad_s,
            expected_peak_velocity,
            rtol=0.0,
            atol=1e-12,
        )
        self.assertFalse(rollout.block_end_requested_target_masses_kg.flags.writeable)
        self.assertFalse(rollout.actual_tank_mass_delta_kg.flags.writeable)
        self.assertFalse(
            rollout.block_end_remaining_mass_to_target_kg.flags.writeable
        )
        self.assertFalse(rollout.post_origin_peak_abs_pitch_roll_rad.flags.writeable)
        self.assertFalse(
            rollout.post_origin_peak_abs_pitch_roll_rate_rad_s.flags.writeable
        )

    def test_later_forecast_load_changes_only_the_later_held_interval(self):
        reference_facts, reference_trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        changed_facts, changed_trace = self._new_track(
            np.array([[0.0, -10.0], [-4.0, -14.0]])
        )

        reference = rollout_current_execution_block_lifecycle(
            trajectory=reference_facts.trajectory,
            lifecycle_trace=reference_trace,
            runtime_assembly=_runtime(),
        )
        changed = rollout_current_execution_block_lifecycle(
            trajectory=changed_facts.trajectory,
            lifecycle_trace=changed_trace,
            runtime_assembly=_runtime(),
        )

        _state_matches(
            self,
            reference.intervals[0].final_platform_state,
            changed.intervals[0].final_platform_state,
        )
        _pump_state_matches(
            self,
            reference.intervals[0].final_execution_state,
            changed.intervals[0].final_execution_state,
        )
        self.assertFalse(
            np.allclose(
                reference.intervals[1].final_platform_state.position,
                changed.intervals[1].final_platform_state.position,
            )
        )
        np.testing.assert_allclose(
            reference_trace.requested_target_masses_kg,
            changed_trace.requested_target_masses_kg,
                rtol=0.0,
                atol=1e-12,
            )

    def test_block_duration_may_end_inside_a_later_forecast_interval(self):
        facts, trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_config=replace(
                _execution_config(),
                block_duration_s=900.0000000005,
            ),
        )

        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )

        self.assertAlmostEqual(rollout.duration_s, 900.0000000005, places=12)
        self.assertEqual(len(rollout.intervals), 2)
        self.assertAlmostEqual(rollout.intervals[1].start_time_s, 600.0)
        self.assertAlmostEqual(
            rollout.intervals[1].end_time_s,
            900.0000000005,
            places=12,
        )
        self.assertAlmostEqual(
            rollout.intervals[1].physical_path.duration_s,
            300.0000000005,
            places=12,
        )
        self.assertLessEqual(rollout.intervals[-1].end_time_s, 900.0000000005)

    def test_release_request_is_derived_from_each_substep_current_mass(self):
        in_progress_state = ExecutionRolloutState(
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
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_state=in_progress_state,
            execution_config=replace(
                _execution_config(),
                min_on_s=1_500.0,
                ramp_down_m3_min_per_s=0.0,
            ),
        )
        trace = facts.lifecycle_facts.release_to_current

        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )

        self.assertTrue(rollout.intervals[0].physical_path.substeps[0].start_execution_state.pump_latched[0])
        for interval in rollout.intervals:
            for substep in interval.physical_path.substeps:
                self.assertIs(
                    substep.execution_step.target_operation,
                    ExecutionTargetOperation.RELEASE_TO_CURRENT,
                )
                np.testing.assert_allclose(
                    substep.execution_step.requested_target_kg,
                    substep.start_execution_state.actual_masses_kg,
                    rtol=0.0,
                    atol=1e-12,
                )

        last_substep = rollout.intervals[-1].physical_path.substeps[-1]
        np.testing.assert_allclose(
            rollout.block_end_requested_target_masses_kg,
            last_substep.execution_step.requested_target_kg,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            rollout.block_end_remaining_mass_to_target_kg,
            last_substep.execution_step.requested_target_kg
            - rollout.final_execution_state.actual_masses_kg,
            rtol=0.0,
            atol=1e-12,
        )
        # A release request is reconstructed at every physical substep. Its
        # final value may equal the trace-origin request, so the substep-local
        # construction verified above is the relevant invariant.

    def test_record_rejects_a_path_with_a_foreign_target_operation(self):
        facts, trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )
        altered_substeps = tuple(
            replace(
                substep,
                execution_step=replace(
                    substep.execution_step,
                    target_operation=ExecutionTargetOperation.RELEASE_TO_CURRENT,
                ),
            )
            for substep in rollout.intervals[0].physical_path.substeps
        )
        altered_first = ForecastExecutionBlockInterval(
            trajectory_step=rollout.intervals[0].trajectory_step,
            start_time_s=rollout.intervals[0].start_time_s,
            end_time_s=rollout.intervals[0].end_time_s,
            physical_path=replace(
                rollout.intervals[0].physical_path,
                substeps=altered_substeps,
            ),
        )

        with self.assertRaisesRegex(ValueError, "declared lifecycle request"):
            ForecastExecutionBlockLifecycleRollout(
                trajectory=rollout.trajectory,
                lifecycle_trace=rollout.lifecycle_trace,
                runtime_assembly=rollout.runtime_assembly,
                intervals=(altered_first, *rollout.intervals[1:]),
                duration_s=rollout.duration_s,
            )

    def test_record_rejects_a_snapshot_with_altered_runtime_dynamics(self):
        facts, trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )
        foreign_runtime = replace(
            rollout.runtime_assembly,
            base_matrices=replace(
                rollout.runtime_assembly.base_matrices,
                damping=(
                    rollout.runtime_assembly.base_matrices.damping
                    + np.diag([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                ),
            ),
        )
        first_step = rollout.trajectory.steps[0]
        foreign_path = advance_physical_execution_platform_path(
            platform_state=rollout.trajectory.initial_state,
            execution_state=rollout.lifecycle_trace.execution_start_state,
            execution_request=rollout.lifecycle_trace.execution_request,
            execution_config=rollout.lifecycle_trace.execution_config,
            runtime_assembly=foreign_runtime,
            reference_tank_masses_kg=(
                rollout.trajectory.platform_snapshot.reference_tank_masses_kg
            ),
            tank_capacities_kg=rollout.trajectory.platform_snapshot.tank_capacities_kg,
            tank_coordinates_m=rollout.trajectory.platform_snapshot.tank_coordinates_m,
            rotor_load=first_step.rotor_load,
            wave_load=first_step.wave_load,
            other_load=first_step.other_load,
            duration_s=first_step.duration_s,
        )
        altered_first = ForecastExecutionBlockInterval(
            trajectory_step=rollout.intervals[0].trajectory_step,
            start_time_s=rollout.intervals[0].start_time_s,
            end_time_s=rollout.intervals[0].end_time_s,
            physical_path=foreign_path,
        )

        with self.assertRaisesRegex(ValueError, "runtime assembly"):
            ForecastExecutionBlockLifecycleRollout(
                trajectory=rollout.trajectory,
                lifecycle_trace=rollout.lifecycle_trace,
                runtime_assembly=rollout.runtime_assembly,
                intervals=(altered_first, *rollout.intervals[1:]),
                duration_s=rollout.duration_s,
            )

    def test_hold_request_is_not_replanned_at_the_second_forecast_interval(self):
        facts, trace = self._new_track(
            np.array([[0.0, -10.0], [2.0, -12.0]])
        )
        rollout = rollout_current_execution_block_lifecycle(
            trajectory=facts.trajectory,
            lifecycle_trace=trace,
            runtime_assembly=_runtime(),
        )

        for interval in rollout.intervals:
            for substep in interval.physical_path.substeps:
                self.assertIs(
                    substep.execution_step.target_operation,
                    trace.execution_request.operation,
                )
                np.testing.assert_allclose(
                    substep.execution_step.requested_target_kg,
                    trace.requested_target_masses_kg,
                    rtol=0.0,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
