import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from wind_prediction.execution_rollout import (
    ExecutionRolloutConfig,
    ExecutionRolloutState,
    ExecutionTargetOperation,
)
from wind_prediction.physical_forecast_endpoint_preview import (
    trace_physical_forecast_endpoint,
)
from wind_prediction.physical_target_lifecycle import (
    PhysicalTargetLifecycle,
    trace_physical_target_lifecycles,
)

from tests.test_physical_forecast_endpoint_preview import _prepared_preview


_ORIGIN = "2026-08-22T00:00:00"


def _new_track_preview():
    assembly, _, timed, sample, reachability = _prepared_preview()
    return trace_physical_forecast_endpoint(
        load_assembly=assembly,
        lead_index=0,
        timed_diagnostic=timed,
        endpoint_sample=sample,
        reachability=reachability,
        execution_start_time=_ORIGIN,
    )


class PhysicalTargetLifecycleTests(unittest.TestCase):
    def test_traces_three_lifecycles_from_one_execution_start(self):
        preview = _new_track_preview()
        facts = trace_physical_target_lifecycles(
            execution_state=preview.reachability.execution_start_state,
            execution_config=preview.reachability.execution_config,
            execution_duration_s=preview.reachability.execution_duration_s,
            execution_start_time=_ORIGIN,
            new_track_preview=preview,
        )

        self.assertIsNotNone(facts.new_track)
        self.assertIs(
            facts.new_track.execution_step,
            preview.reachability.execution_step,
        )
        self.assertIs(facts.new_track.source_preview, preview)
        self.assertIs(
            facts.new_track.lifecycle,
            PhysicalTargetLifecycle.NEW_TRACK,
        )
        self.assertIs(
            facts.continue_existing.lifecycle,
            PhysicalTargetLifecycle.CONTINUE_EXISTING,
        )
        self.assertIs(
            facts.release_to_current.lifecycle,
            PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
        )
        for trace in (
            facts.continue_existing,
            facts.release_to_current,
            facts.new_track,
        ):
            np.testing.assert_allclose(
                trace.execution_start_state.actual_masses_kg,
                preview.reachability.execution_start_state.actual_masses_kg,
            )
            self.assertEqual(
                trace.execution_duration_s,
                preview.reachability.execution_duration_s,
            )
            self.assertEqual(trace.execution_start_time, _ORIGIN)
        self.assertEqual(facts.execution_start_time, _ORIGIN)
        np.testing.assert_allclose(
            facts.continue_existing.requested_target_masses_kg,
            preview.reachability.execution_start_state.primary_target_masses_kg,
        )
        self.assertIs(
            facts.release_to_current.execution_request.operation,
            ExecutionTargetOperation.RELEASE_TO_CURRENT,
        )

    def test_new_track_rejects_mismatched_start_config_or_duration(self):
        preview = _new_track_preview()
        state = preview.reachability.execution_start_state
        config = preview.reachability.execution_config
        duration = preview.reachability.execution_duration_s

        with self.assertRaisesRegex(ValueError, "pump start state"):
            trace_physical_target_lifecycles(
                execution_state=replace(state, pump_latched=np.ones(3, dtype=bool)),
                execution_config=config,
                execution_duration_s=duration,
                execution_start_time=_ORIGIN,
                new_track_preview=preview,
            )
        with self.assertRaisesRegex(ValueError, "execution configuration"):
            trace_physical_target_lifecycles(
                execution_state=state,
                execution_config=replace(config, min_on_s=1.0),
                execution_duration_s=duration,
                execution_start_time=_ORIGIN,
                new_track_preview=preview,
            )
        with self.assertRaisesRegex(ValueError, "execution duration"):
            trace_physical_target_lifecycles(
                execution_state=state,
                execution_config=config,
                execution_duration_s=duration / 2.0,
                execution_start_time=_ORIGIN,
                new_track_preview=preview,
            )
        with self.assertRaisesRegex(ValueError, "execution start time"):
            trace_physical_target_lifecycles(
                execution_state=state,
                execution_config=config,
                execution_duration_s=duration,
                execution_start_time="2026-08-22T00:10:00",
                new_track_preview=preview,
            )

    def test_release_can_preserve_an_inflight_pump_state(self):
        state = ExecutionRolloutState(
            masses_kg=[500.0, 500.0, 500.0],
            target_masses_kg=[700.0, 500.0, 500.0],
            primary_target_kg=[700.0, 500.0, 500.0],
            pump_rates_m3_min=[0.5, 0.0, 0.0],
            pump_latched=[True, False, False],
            pump_on_elapsed_s=[0.0, 0.0, 0.0],
            pump_off_elapsed_s=[0.0, 100.0, 100.0],
            pump_command_rates_m3_min=[0.5, 0.0, 0.0],
            last_flow_directions=[1.0, 0.0, 0.0],
        )
        config = ExecutionRolloutConfig(
            block_duration_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=1.0,
            tank_capacity_kg=1_000.0,
            internal_step_s=1.0,
            stop_error_kg=1.0,
            restart_error_kg=1.0,
            min_on_s=100.0,
            min_off_s=0.0,
            near_target_hold_s=0.0,
            ramp_up_m3_min_per_s=np.inf,
            ramp_down_m3_min_per_s=0.0,
            pump_rate_schedule_m3_min=((0.0, 0.0), (1.0, 1.0)),
        )

        facts = trace_physical_target_lifecycles(
            execution_state=state,
            execution_config=config,
            execution_duration_s=10.0,
            execution_start_time=_ORIGIN,
        )

        release = facts.release_to_current
        np.testing.assert_allclose(
            release.requested_target_masses_kg,
            state.actual_masses_kg,
        )
        self.assertGreater(release.execution_step.active_time_s, 0.0)
        self.assertTrue(release.execution_step.state.pump_latched[0])

    def test_continue_tracks_primary_target_while_release_resets_a_slewed_target(self):
        state = ExecutionRolloutState(
            masses_kg=[500.0, 500.0, 500.0],
            target_masses_kg=[600.0, 500.0, 500.0],
            primary_target_kg=[800.0, 500.0, 500.0],
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
            pump_off_elapsed_s=[100.0, 100.0, 100.0],
        )
        config = ExecutionRolloutConfig(
            block_duration_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=1.0,
            tank_capacity_kg=1_000.0,
            target_slew_enabled=True,
            target_slew_rate_m3_min=0.6,
            pump_rate_schedule_m3_min=((0.0, 0.0), (1.0, 1.0)),
        )

        facts = trace_physical_target_lifecycles(
            execution_state=state,
            execution_config=config,
            execution_duration_s=10.0,
            execution_start_time=_ORIGIN,
        )

        continuation = facts.continue_existing
        release = facts.release_to_current
        np.testing.assert_allclose(
            continuation.requested_target_masses_kg,
            state.primary_target_masses_kg,
        )
        self.assertFalse(
            np.allclose(
                continuation.requested_target_masses_kg,
                state.rate_limited_target_kg,
            )
        )
        self.assertTrue(release.execution_step.target_slew_reset)
        np.testing.assert_allclose(
            release.requested_target_masses_kg,
            state.actual_masses_kg,
        )

    def test_rejects_a_lifecycle_preview_past_the_current_control_block(self):
        state = ExecutionRolloutState(
            masses_kg=[500.0, 500.0, 500.0],
            target_masses_kg=[500.0, 500.0, 500.0],
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
        )
        config = ExecutionRolloutConfig(
            block_duration_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=1.0,
            tank_capacity_kg=1_000.0,
            pump_rate_schedule_m3_min=((0.0, 0.0), (1.0, 1.0)),
        )

        with self.assertRaisesRegex(ValueError, "execution_config.block_duration_s"):
            trace_physical_target_lifecycles(
                execution_state=state,
                execution_config=config,
                execution_duration_s=11.0,
                execution_start_time=_ORIGIN,
            )

    def test_rejects_non_string_execution_start_time(self):
        state = ExecutionRolloutState(
            masses_kg=[500.0, 500.0, 500.0],
            target_masses_kg=[500.0, 500.0, 500.0],
            pump_rates_m3_min=[0.0, 0.0, 0.0],
            pump_latched=[False, False, False],
        )
        config = ExecutionRolloutConfig(
            block_duration_s=10.0,
            water_density_kg_m3=1_000.0,
            max_pump_rate_m3_min=1.0,
            tank_capacity_kg=1_000.0,
            pump_rate_schedule_m3_min=((0.0, 0.0), (1.0, 1.0)),
        )

        for invalid_time in (None, 0, 1.5, "   "):
            with self.subTest(invalid_time=invalid_time):
                with self.assertRaisesRegex(ValueError, "execution_start_time"):
                    trace_physical_target_lifecycles(
                        execution_state=state,
                        execution_config=config,
                        execution_duration_s=10.0,
                        execution_start_time=invalid_time,
                    )

    def test_module_does_not_import_legacy_controller_or_candidate_ranking(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "physical_target_lifecycle.py"
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


if __name__ == "__main__":
    unittest.main()
