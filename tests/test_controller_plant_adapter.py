import json
import unittest
from dataclasses import replace

import numpy as np

from wind_prediction.controller import (
    CompactControllerPlantAdapter,
    ControlCoreConfig,
    ForecastEvidence,
)
from wind_prediction.execution_rollout import ExecutionRolloutConfig
from wind_prediction.forecast_action_policy import ForecastActionPolicyConfig
from wind_prediction.controller_plant_adapter import wind_observation_to_uv_ms


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


def _forecast_source(time_s, wind_obs):
    del time_s, wind_obs
    vectors = np.array([(0.0, -12.0)] * 6, dtype=float)
    return ForecastEvidence(
        source="plant_adapter_test",
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


def _state(pitch_deg=4.5, roll_deg=0.0):
    state = np.zeros(12, dtype=float)
    state[4] = np.radians(pitch_deg)
    state[3] = np.radians(roll_deg)
    return state


def _plant_info(masses):
    values = np.asarray(masses, dtype=float)
    return {
        "tank_masses": values,
        "target_ballast_mass": values,
        "pump_rate_cmd_m3_min": np.zeros(3),
        "pump_latched": np.zeros(3, dtype=bool),
    }


class ControllerPlantAdapterTests(unittest.TestCase):
    def test_wind_conversion_matches_forecast_convention(self):
        np.testing.assert_allclose(
            wind_observation_to_uv_ms({"ws": 10.0, "wd_deg": 90.0}),
            (-10.0, 0.0),
            atol=1e-12,
        )

    def test_adapter_returns_exact_primary_target_and_trace(self):
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=(900_000.0,) * 3,
            forecast_source=_forecast_source,
            config=_config(),
        )
        output = adapter.compute(
            _state(),
            {"ws": 10.0, "wd_deg": 0.0},
            _plant_info((900_000.0,) * 3),
            0.0,
        )

        self.assertEqual(output["source"], "compact_controller_v2")
        self.assertEqual(output["preview_primary_enabled"], 1)
        self.assertEqual(len(output["preview_primary_target_kg"]), 3)
        trace = json.loads(output["preview_controller_v2_trace_json"])
        self.assertEqual(trace["selected_action"], output["preview_primary_action"])
        self.assertEqual(len(adapter.records), 1)

    def test_adapter_replans_once_per_bucket_and_uses_actual_feedback(self):
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=(900_000.0,) * 3,
            forecast_source=None,
            config=_config(),
        )
        first = adapter.compute(
            _state(),
            {"ws": 10.0, "wd_deg": 0.0},
            _plant_info((900_000.0,) * 3),
            0.0,
        )
        between = adapter.compute(
            _state(),
            {"ws": 10.0, "wd_deg": 0.0},
            _plant_info((890_000.0, 905_000.0, 905_000.0)),
            600.0,
        )
        second = adapter.compute(
            _state(),
            {"ws": 10.0, "wd_deg": 0.0},
            _plant_info((890_000.0, 905_000.0, 905_000.0)),
            1200.0,
        )

        self.assertEqual(len(adapter.records), 2)
        self.assertEqual(
            between["preview_primary_target_kg"],
            first["preview_primary_target_kg"],
        )
        self.assertNotEqual(
            second["preview_primary_target_kg"],
            first["preview_primary_target_kg"],
        )

    def test_adapter_requires_exact_platform_and_tank_state(self):
        adapter = CompactControllerPlantAdapter(
            initial_tank_masses_kg=(900_000.0,) * 3,
            forecast_source=None,
            config=_config(),
        )
        with self.assertRaisesRegex(ValueError, "exactly 12"):
            adapter.compute(
                np.zeros(6),
                {"ws": 10.0, "wd_deg": 0.0},
                _plant_info((900_000.0,) * 3),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
