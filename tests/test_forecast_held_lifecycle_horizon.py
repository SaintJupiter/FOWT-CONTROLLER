from dataclasses import replace
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import (
    _assert_same_execution_state,
    _execution_config,
    _physical_facts,
)
from wind_prediction.execution_rollout import (
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from wind_prediction.forecast_execution_block_lifecycle_rollout import (
    HeldLifecycleForecastHorizonRollout,
    rollout_current_execution_block_lifecycle,
    rollout_held_lifecycle_to_forecast_horizon,
)
from wind_prediction.physical_target_lifecycle import PhysicalTargetLifecycle


def _one_lead_block_config():
    return replace(_execution_config(), block_duration_s=600.0)


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


def _prefix(facts, lifecycle: PhysicalTargetLifecycle):
    traces = {
        PhysicalTargetLifecycle.CONTINUE_EXISTING: facts.lifecycle_facts.continue_existing,
        PhysicalTargetLifecycle.RELEASE_TO_CURRENT: facts.lifecycle_facts.release_to_current,
        PhysicalTargetLifecycle.NEW_TRACK: facts.lifecycle_facts.new_track,
    }
    trace = traces[lifecycle]
    if trace is None:
        raise AssertionError("fixture must provide the requested lifecycle trace")
    return rollout_current_execution_block_lifecycle(
        trajectory=facts.trajectory,
        lifecycle_trace=trace,
        runtime_assembly=facts.runtime_assembly,
    )


class HeldLifecycleForecastHorizonRolloutTests(unittest.TestCase):
    def test_holds_a_completed_first_block_through_all_remaining_forecast_leads(self):
        facts = _physical_facts(
            future_uv_ms=np.array(
                [
                    [0.0, -10.0],
                    [2.0, -12.0],
                    [1.0, -11.0],
                    [-1.0, -9.0],
                    [-2.0, -8.0],
                    [0.5, -10.5],
                ]
            ),
            lead_reliability=(0.9, 0.8, 0.8, 0.7, 0.7, 0.6),
            execution_config=_one_lead_block_config(),
        )
        prefix = _prefix(facts, PhysicalTargetLifecycle.NEW_TRACK)

        held = rollout_held_lifecycle_to_forecast_horizon(prefix=prefix)

        self.assertIsInstance(held, HeldLifecycleForecastHorizonRollout)
        self.assertIs(held.prefix, prefix)
        self.assertEqual(len(held.intervals), len(facts.trajectory.steps))
        self.assertIs(held.intervals[0], prefix.intervals[0])
        self.assertAlmostEqual(
            held.duration_s,
            facts.trajectory.steps[-1].end_time_s,
        )
        previous_platform_state = prefix.final_platform_state
        previous_execution_state = prefix.final_execution_state
        for lead_index, continuation in enumerate(held.continuation_intervals, start=1):
            self.assertIs(continuation.trajectory_step, facts.trajectory.steps[lead_index])
            np.testing.assert_allclose(
                continuation.physical_path.substeps[0].platform_step.loads.wind,
                facts.trajectory.steps[lead_index].rotor_load.generalized_load_platform,
            )
            np.testing.assert_allclose(
                continuation.physical_path.substeps[0].start_platform_state.position,
                previous_platform_state.position,
            )
            np.testing.assert_allclose(
                continuation.physical_path.substeps[0].start_platform_state.velocity,
                previous_platform_state.velocity,
            )
            _assert_same_execution_state(
                self,
                continuation.physical_path.substeps[0].start_execution_state,
                previous_execution_state,
            )
            previous_platform_state = continuation.final_platform_state
            previous_execution_state = continuation.final_execution_state

        target = prefix.lifecycle_trace.execution_request.target_masses_kg
        for substep in continuation.physical_path.substeps:
            self.assertIs(
                substep.execution_step.target_operation,
                ExecutionTargetOperation.TRACK,
            )
            np.testing.assert_allclose(substep.execution_step.requested_target_kg, target)

    def test_release_remains_dynamic_when_held_after_the_first_block(self):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_state=_in_progress_execution_state(),
            execution_config=_one_lead_block_config(),
        )
        prefix = _prefix(facts, PhysicalTargetLifecycle.RELEASE_TO_CURRENT)

        held = rollout_held_lifecycle_to_forecast_horizon(prefix=prefix)

        self.assertEqual(held.lifecycle, PhysicalTargetLifecycle.RELEASE_TO_CURRENT)
        for substep in held.continuation_intervals[0].physical_path.substeps:
            self.assertIs(
                substep.execution_step.target_operation,
                ExecutionTargetOperation.RELEASE_TO_CURRENT,
            )
            np.testing.assert_allclose(
                substep.execution_step.requested_target_kg,
                substep.start_execution_state.actual_masses_kg,
            )

    def test_rejects_an_incomplete_horizon_after_a_first_lead_prefix(self):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_config=_one_lead_block_config(),
        )
        prefix = _prefix(facts, PhysicalTargetLifecycle.CONTINUE_EXISTING)

        with self.assertRaisesRegex(ValueError, "cover every remaining forecast lead"):
            HeldLifecycleForecastHorizonRollout(
                prefix=prefix,
                continuation_intervals=(),
            )

    def test_rejects_a_prefix_that_ends_inside_a_forecast_lead(self):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_config=replace(_execution_config(), block_duration_s=900.0),
        )
        prefix = _prefix(facts, PhysicalTargetLifecycle.CONTINUE_EXISTING)

        with self.assertRaisesRegex(ValueError, "lead boundary"):
            rollout_held_lifecycle_to_forecast_horizon(prefix=prefix)

    def test_requires_a_completed_execution_block_prefix(self):
        with self.assertRaisesRegex(TypeError, "prefix"):
            rollout_held_lifecycle_to_forecast_horizon(prefix=object())


if __name__ == "__main__":
    unittest.main()
