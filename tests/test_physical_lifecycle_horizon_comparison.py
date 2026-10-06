from dataclasses import replace
import unittest

import numpy as np

from tests.test_physical_forecast_cycle_facts import _execution_config, _physical_facts
from wind_prediction.forecast_evidence import ForecastEvidence
from wind_prediction.physical_forecast_admission import bind_physical_forecast_admission
from wind_prediction.physical_lifecycle_comparison import (
    assemble_physical_lifecycle_comparison_for_current_block,
    assemble_physical_lifecycle_horizon_comparison,
)
from wind_prediction.physical_target_lifecycle import PhysicalTargetLifecycle


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


class PhysicalLifecycleHorizonComparisonTests(unittest.TestCase):
    def _comparison(self, *, admitted=True):
        facts = _physical_facts(
            future_uv_ms=np.array([[0.0, -10.0], [2.0, -12.0]]),
            execution_config=replace(_execution_config(), block_duration_s=600.0),
        )
        current = assemble_physical_lifecycle_comparison_for_current_block(
            decision_input=facts,
            forecast_admission=_admission(facts, admitted=admitted),
        )
        return facts, current, assemble_physical_lifecycle_horizon_comparison(
            current_block_comparison=current,
        )

    def test_horizon_outcomes_reuse_current_block_prefixes_without_selection(self):
        facts, current, horizon = self._comparison()

        self.assertEqual(horizon.available_lifecycles, current.available_lifecycles)
        self.assertFalse(hasattr(horizon, "selected_lifecycle"))
        for outcome in horizon.outcomes:
            prefix = current.outcome_for(outcome.lifecycle)
            self.assertIs(outcome.current_block_outcome, prefix)
            self.assertIs(outcome.rollout.prefix, prefix.rollout)
            self.assertAlmostEqual(
                outcome.rollout.duration_s,
                facts.trajectory.steps[-1].end_time_s,
            )
            self.assertEqual(len(outcome.rollout.intervals), len(facts.trajectory.steps))
            self.assertAlmostEqual(
                outcome.transferred_volume_m3,
                sum(
                    interval.physical_path.transferred_volume_m3
                    for interval in outcome.rollout.intervals
                ),
            )
            self.assertFalse(outcome.reached_final_tank_masses_kg.flags.writeable)

    def test_nonadmitted_forecast_does_not_create_a_new_track_horizon(self):
        _, current, horizon = self._comparison(admitted=False)

        self.assertIsNone(current.new_track)
        self.assertIsNone(horizon.new_track)
        self.assertEqual(
            horizon.available_lifecycles,
            (
                PhysicalTargetLifecycle.CONTINUE_EXISTING,
                PhysicalTargetLifecycle.RELEASE_TO_CURRENT,
            ),
        )


if __name__ == "__main__":
    unittest.main()
