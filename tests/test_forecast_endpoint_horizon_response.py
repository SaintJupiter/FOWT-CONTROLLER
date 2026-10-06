from dataclasses import replace
import ast
from pathlib import Path
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import _execution_config, _physical_facts
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.physical_forecast_admission import bind_physical_forecast_admission
from wind_prediction.physical_lifecycle_comparison import (
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from wind_prediction.forecast_endpoint_horizon_response import (
    assemble_forecast_endpoint_held_responses,
)


def _admission(facts, *, admitted=True):
    assembly = facts.trajectory.load_assembly
    identity = assembly.input_identity
    forecast = ForecastEvidence(
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
    return bind_physical_forecast_admission(
        decision_input=facts,
        forecast=forecast,
        admitted=admitted,
        basis="fixture source-bound admission",
    )


class ForecastEndpointHeldResponseTests(unittest.TestCase):
    def _responses(self, *, future_uv_ms, admitted=True):
        future_uv_ms = np.asarray(future_uv_ms, dtype=float)
        facts = _physical_facts(
            future_uv_ms=future_uv_ms,
            lead_reliability=tuple(np.ones(future_uv_ms.shape[0])),
            execution_config=replace(_execution_config(), block_duration_s=600.0),
        )
        comparison = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=facts,
            forecast_admission=_admission(facts, admitted=admitted),
        )
        return facts, comparison, assemble_forecast_endpoint_held_responses(
            current_block_comparison=comparison,
        )

    def test_admitted_horizon_keeps_one_independent_held_response_per_endpoint(self):
        facts, comparison, response_set = self._responses(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0], [-3.0, -8.0]],
        )

        self.assertEqual(
            len(response_set.responses),
            len(facts.horizon_endpoint_previews.previews),
        )
        self.assertIs(
            response_set.response_for_lead(0).current_block_rollout,
            comparison.new_track.rollout,
        )
        self.assertFalse(hasattr(response_set, "selected_response"))
        for index, response in enumerate(response_set.responses):
            preview = facts.horizon_endpoint_previews.preview_for_lead(index)
            self.assertIs(response.endpoint_preview, preview)
            self.assertIs(response.lifecycle_trace.source_preview, preview)
            self.assertEqual(response.lead_index, index)
            self.assertEqual(response.lead_time_s, preview.lead_time_s)
            self.assertIs(
                response.horizon_rollout.prefix,
                response.current_block_rollout,
            )
            self.assertAlmostEqual(
                response.horizon_rollout.duration_s,
                facts.trajectory.steps[-1].end_time_s,
            )
            np.testing.assert_allclose(
                response.target_tank_masses_kg,
                preview.target_tank_masses_kg,
            )
            self.assertFalse(response.target_tank_masses_kg.flags.writeable)

    def test_changed_tail_forecast_changes_only_later_endpoint_target_facts(self):
        reference_facts, _, reference = self._responses(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0], [-3.0, -8.0]],
        )
        changed_facts, _, changed = self._responses(
            future_uv_ms=[[0.0, -10.0], [8.0, -4.0], [-9.0, -2.0]],
        )

        np.testing.assert_allclose(
            reference.response_for_lead(0).target_tank_masses_kg,
            changed.response_for_lead(0).target_tank_masses_kg,
        )
        np.testing.assert_allclose(
            reference.response_for_lead(0).current_block_end_pitch_roll_rad,
            changed.response_for_lead(0).current_block_end_pitch_roll_rad,
        )
        self.assertFalse(
            np.allclose(
                reference.response_for_lead(1).target_tank_masses_kg,
                changed.response_for_lead(1).target_tank_masses_kg,
            )
        )
        self.assertIsNot(
            reference_facts.trajectory,
            changed_facts.trajectory,
        )

    def test_nonadmitted_forecast_keeps_endpoint_response_set_empty(self):
        _, comparison, response_set = self._responses(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0]],
            admitted=False,
        )

        self.assertFalse(comparison.forecast_admission.admitted)
        self.assertEqual(response_set.responses, ())

    def test_response_changes_keep_one_shared_continuation_reference(self):
        _, comparison, response_set = self._responses(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0], [-3.0, -8.0]],
        )
        horizon = assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=comparison,
        )

        changes = response_set.changes_from_continuation(
            horizon_comparison=horizon,
        )

        self.assertEqual(len(changes), len(response_set.responses))
        self.assertFalse(hasattr(changes[0], "selected_response"))
        for index, change in enumerate(changes):
            self.assertIs(change.response, response_set.response_for_lead(index))
            self.assertIs(change.continuation, horizon.continue_existing)
            self.assertAlmostEqual(
                change.horizon_transferred_volume_delta_m3,
                change.response.horizon_transferred_volume_m3
                - horizon.continue_existing.transferred_volume_m3,
            )
            np.testing.assert_allclose(
                change.horizon_end_pitch_roll_delta_rad,
                change.response.horizon_end_pitch_roll_rad
                - horizon.continue_existing.end_pitch_roll_rad,
            )
            self.assertFalse(change.horizon_end_pitch_roll_delta_rad.flags.writeable)

    def test_response_change_exposes_same_time_posture_histories(self):
        _, comparison, response_set = self._responses(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0], [-3.0, -8.0]],
        )
        horizon = assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=comparison,
        )
        change = response_set.change_from_continuation(
            horizon_comparison=horizon,
            lead_index=2,
        )

        def history(rollout):
            sample_times_s = []
            angles = []
            rates = []
            for interval in rollout.intervals:
                time_s = interval.start_time_s
                for substep in interval.physical_path.substeps:
                    time_s += substep.duration_s
                    sample_times_s.append(time_s)
                    angles.append(substep.next_platform_state.position[[4, 3]])
                    rates.append(substep.next_platform_state.velocity[[4, 3]])
            return (
                np.asarray(sample_times_s),
                np.asarray(angles),
                np.asarray(rates),
            )

        endpoint_times, endpoint_angles, endpoint_rates = history(
            change.response.horizon_rollout
        )
        continuation_times, continuation_angles, continuation_rates = history(
            change.continuation.rollout
        )
        np.testing.assert_allclose(endpoint_times, continuation_times)
        np.testing.assert_allclose(change.horizon_sample_times_s, endpoint_times)
        np.testing.assert_allclose(
            change.horizon_pitch_roll_delta_rad,
            endpoint_angles - continuation_angles,
        )
        np.testing.assert_allclose(
            change.horizon_pitch_roll_rate_delta_rad_s,
            endpoint_rates - continuation_rates,
        )
        np.testing.assert_allclose(
            change.horizon_abs_pitch_roll_delta_rad,
            np.abs(endpoint_angles) - np.abs(continuation_angles),
        )
        self.assertFalse(change.horizon_sample_times_s.flags.writeable)
        self.assertFalse(change.horizon_pitch_roll_delta_rad.flags.writeable)

    def test_response_change_rejects_a_horizon_from_other_current_block(self):
        _, comparison, response_set = self._responses(
            future_uv_ms=[[0.0, -10.0], [2.0, -12.0]],
        )
        _, other_comparison, _ = self._responses(
            future_uv_ms=[[1.0, -8.0], [-1.0, -12.0]],
        )
        other_horizon = assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=other_comparison,
        )

        with self.assertRaisesRegex(ValueError, "must retain the response set"):
            response_set.change_from_continuation(
                horizon_comparison=other_horizon,
                lead_index=0,
            )

    def test_module_avoids_legacy_policy_and_forecast_quality_dependencies(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_endpoint_horizon_response.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "candidate",
            "planner",
            "action_target",
            "controller",
            "policy",
            "event",
            "reliability",
            "score",
            "select",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
