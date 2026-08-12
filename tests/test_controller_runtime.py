import unittest
from dataclasses import replace

import numpy as np

from wind_prediction.controller import (
    ControlCoreConfig,
    ControllerMeasurements,
    ControllerRuntimeState,
    ForecastAssistedBallastController,
    ForecastEvidence,
)
from wind_prediction.execution_rollout import ExecutionRolloutConfig
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig


def _config():
    return ControlCoreConfig(
        stage_duration_s=1200.0,
        stage_count=3,
        execution=ExecutionRolloutConfig(
            block_duration_s=1200.0,
            internal_step_s=60.0,
            target_slew_enabled=False,
        ),
        forecast_policy=ForecastActionPolicyConfig(
            enabled=True,
            stage_duration_s=1200.0,
            high_impact_reliability_min=0.65,
            high_impact_event_probability_min=0.60,
        ),
    )


def _measurements(time_s, posture=(2.2, 0.0)):
    return ControllerMeasurements(
        time_s=time_s,
        posture_deg=posture,
        posture_rate_deg_s=(0.0, 0.0),
        current_wind_uv_ms=(0.0, -10.0),
    )


def _forecast():
    vectors = np.array(
        [
            (0.0, -10.0),
            (0.0, -11.0),
            (0.0, -12.0),
            (0.0, -12.0),
            (0.0, -11.0),
            (0.0, -10.0),
        ],
        dtype=float,
    )
    return ForecastEvidence(
        source="runtime_test",
        model_version="test",
        origin_time="2026-08-12T00:00:00",
        sample_period_s=600.0,
        uv_ms=vectors,
        lead_reliability=np.full(6, 0.9),
        event_probs={
            "attention_event_0_20m": 0.9,
            "attention_event_20_40m": 0.9,
            "attention_event_40_60m": 0.9,
        },
        provides_future_preview=True,
    )


class ControllerRuntimeTests(unittest.TestCase):
    def test_public_controller_advances_one_complete_cycle(self):
        controller = ForecastAssistedBallastController(_config())
        state = ControllerRuntimeState.initialize(
            (900_000.0, 900_000.0, 900_000.0)
        )

        result = controller.step(_measurements(0.0), state, _forecast())

        self.assertEqual(result.state.cycle_index, 1)
        self.assertEqual(result.state.last_action, result.decision.action)
        np.testing.assert_allclose(
            result.state.execution.primary_target_masses_kg,
            result.decision.target_masses_kg,
        )
        self.assertIn("cycle", result.trace)
        self.assertIn("state_after_execution", result.trace)
        self.assertEqual(
            result.trace["selected_action"],
            result.decision.action.value,
        )

    def test_controller_runs_consecutive_cycles_without_forecast(self):
        controller = ForecastAssistedBallastController(_config())
        state = ControllerRuntimeState.initialize(
            (900_000.0, 900_000.0, 900_000.0)
        )

        first = controller.step(_measurements(0.0), state, None)
        second = controller.step(
            _measurements(1200.0, posture=(1.8, 0.2)),
            first.state,
            None,
        )

        self.assertEqual(second.state.cycle_index, 2)
        self.assertFalse(second.decision.context.forecast_available)
        self.assertEqual(second.state.last_decision_time_s, 1200.0)

    def test_runtime_rejects_non_advancing_time(self):
        controller = ForecastAssistedBallastController(_config())
        state = ControllerRuntimeState.initialize(
            (900_000.0, 900_000.0, 900_000.0)
        )
        first = controller.step(_measurements(0.0), state, None)

        with self.assertRaisesRegex(ValueError, "must advance"):
            controller.step(_measurements(0.0), first.state, None)

    def test_config_rejects_mismatched_decision_and_execution_intervals(self):
        with self.assertRaisesRegex(ValueError, "execution.block_duration_s"):
            replace(
                _config(),
                execution=ExecutionRolloutConfig(block_duration_s=600.0),
            )

    def test_initial_state_requires_exactly_three_tanks(self):
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            ControllerRuntimeState.initialize((900_000.0, 900_000.0))


if __name__ == "__main__":
    unittest.main()
