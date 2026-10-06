"""Tests for binding one replay origin into the rolling physical chain."""

from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest

import numpy as np

from wind_prediction.controller_replay_adapter import ReplayForecastEvidenceSource
from wind_prediction.forecast_adapter import ForecastResult
from wind_prediction.replay_physical_cycle_adapter import (
    assemble_replay_physical_rolling_cycle_input,
    assemble_resolved_physical_rolling_cycle_input,
)


class _Dataset:
    future_steps = 3
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


class _ForecastAdapter:
    model_version = "rolling-adapter-fixture"
    provides_future_preview = True
    thresholds = {"wind_change": 0.6}

    @staticmethod
    def predict_window(x_window, timestamp=None):
        del x_window
        future_uv = np.array(
            [[1.0, -9.0], [2.0, -11.0], [3.0, -13.0]], dtype=np.float32
        )
        wind_dir_deg = (
            np.degrees(np.arctan2(-future_uv[:, 0], -future_uv[:, 1])) + 360.0
        ) % 360.0
        return ForecastResult(
            wind_uv_raw=future_uv,
            wind_speed=np.linalg.norm(future_uv, axis=1),
            wind_dir_deg=wind_dir_deg.astype(np.float32),
            event_probs={"wind_change": 0.8},
            risk_levels={"wind_change": True},
            model_version="rolling-adapter-fixture",
            timestamp=timestamp,
        )


def _source(sample):
    return ReplayForecastEvidenceSource(
        replay_dataset=_Dataset(sample),
        start_timestamp=sample.history_end if sample is not None else datetime(2026, 8, 24),
        forecast_adapter=_ForecastAdapter(),
        lead_reliability=np.array([1.0, 0.9, 0.8]),
    )


def _assemble(source):
    return assemble_replay_physical_rolling_cycle_input(
        forecast_source=source,
        current_time_s=0.0,
        start_time_s=0.0,
        current_thrust_coefficient=0.72,
        future_thrust_coefficients=[0.72, 0.70, 0.68],
        thrust_coefficient_source="fixed rolling-adapter operating state",
        current_nacelle_yaw_relative_platform_rad=0.0,
        current_nacelle_yaw_source="fixed rolling-adapter nacelle yaw",
        forecast_admitted=True,
        forecast_admission_basis="fixture admission",
        current_wave_load=np.arange(6, dtype=float),
        current_other_load=-np.arange(6, dtype=float),
        future_wave_interval_loads=np.full((3, 6), 2.0),
        future_other_interval_loads=np.full((3, 6), -3.0),
        nonrotor_load_provenance="explicit rolling-adapter fixture loads",
    )


class ReplayPhysicalRollingCycleAdapterTests(unittest.TestCase):
    def test_uses_one_replay_origin_for_current_observation_and_forecast_grid(self):
        origin = datetime(2026, 8, 24, 0, 0, 0)
        sample = SimpleNamespace(
            series_id="rolling-adapter-fixture",
            x_window=np.zeros((12, 4), dtype=np.float32),
            history_start=origin - timedelta(minutes=110),
            history_end=origin,
            wind_obs={"ws": 10.0, "wd_deg": 90.0},
            y_uv_raw=np.full((3, 2), 999.0, dtype=np.float32),
        )

        cycle = _assemble(_source(sample))

        self.assertIsNotNone(cycle)
        assert cycle is not None
        self.assertEqual(cycle.forecast.origin_time, "2026-08-24 00:00:00")
        self.assertEqual(
            cycle.current_wind_observation_time, cycle.forecast.origin_time
        )
        self.assertEqual(
            cycle.environment.forecast_origin_time, cycle.forecast.origin_time
        )
        np.testing.assert_allclose(
            cycle.current_enu_downwind_air_velocity_mps,
            [-10.0, 0.0],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            cycle.forecast.uv_ms,
            [[1.0, -9.0], [2.0, -11.0], [3.0, -13.0]],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            cycle.environment.lead_times_s,
            [600.0, 1_200.0, 1_800.0],
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            cycle.environment.current_wave_load,
            np.arange(6, dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            cycle.environment.wave_interval_loads,
            np.full((3, 6), 2.0),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_missing_replay_evidence_remains_missing_without_current_only_fallback(self):
        self.assertIsNone(_assemble(_source(None)))

    def test_resolved_inputs_can_be_assembled_after_operating_state_resolution(self):
        origin = datetime(2026, 8, 24, 0, 0, 0)
        sample = SimpleNamespace(
            series_id="rolling-adapter-fixture",
            x_window=np.zeros((12, 4), dtype=np.float32),
            history_start=origin - timedelta(minutes=110),
            history_end=origin,
            wind_obs={"ws": 10.0, "wd_deg": 90.0},
            y_uv_raw=np.full((3, 2), 999.0, dtype=np.float32),
        )
        source_bound = _source(sample).source_bound_forecast_and_current_enu_wind(0.0)
        self.assertIsNotNone(source_bound)
        assert source_bound is not None
        forecast, current_wind = source_bound

        cycle = assemble_resolved_physical_rolling_cycle_input(
            forecast=forecast,
            current_enu_downwind_air_velocity_mps=current_wind,
            current_wind_source="already_sourced_fixture_observation",
            current_wind_observation_time=forecast.origin_time,
            start_time_s=0.0,
            current_thrust_coefficient=0.72,
            future_thrust_coefficients=[0.72, 0.70, 0.68],
            thrust_coefficient_source="fixture operating-state resolution",
            current_nacelle_yaw_relative_platform_rad=0.0,
            current_nacelle_yaw_source="fixture yaw",
            forecast_admitted=True,
            forecast_admission_basis="fixture admission",
            current_wave_load=np.zeros(6),
            current_other_load=np.zeros(6),
            future_wave_interval_loads=np.zeros((3, 6)),
            future_other_interval_loads=np.zeros((3, 6)),
            nonrotor_load_provenance="fixture nonrotor loads",
        )

        self.assertIs(cycle.forecast, forecast)
        self.assertEqual(cycle.current_wind_observation_time, forecast.origin_time)
        np.testing.assert_allclose(
            cycle.current_enu_downwind_air_velocity_mps,
            current_wind,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            cycle.future_thrust_coefficients,
            [0.72, 0.70, 0.68],
            rtol=0.0,
            atol=1.0e-12,
        )


if __name__ == "__main__":
    unittest.main()
