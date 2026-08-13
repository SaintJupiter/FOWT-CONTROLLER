import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys

import numpy as np

from wind_prediction.attitude_rollout import (
    AttitudeRolloutParameters,
    rollout_attitude,
)


class AttitudeRolloutTests(unittest.TestCase):
    @staticmethod
    def _params(**overrides):
        values = {
            "enabled": True,
            "metric_sample_interval_s": 0.5,
            "thresholds_deg": (1.0, 2.0, 5.0),
        }
        values.update(overrides)
        return AttitudeRolloutParameters(**values)

    def test_disabled_default_does_not_create_rollout(self):
        result = rollout_attitude(
            [0.0, 0.0],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0]],
            [20.0],
        )
        self.assertIsNone(result)

    def test_zero_state_and_zero_inputs_remain_zero(self):
        result = rollout_attitude(
            [0.0, 0.0],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [30.0, 45.0],
            parameters=self._params(),
        )

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result.pitch_deg, 0.0, atol=1e-12)
        np.testing.assert_allclose(result.roll_deg, 0.0, atol=1e-12)
        self.assertAlmostEqual(result.metrics.vector_rms_deg, 0.0)
        self.assertEqual(result.metrics.dominant_exceedance_s[1.0], 0.0)

    def test_default_parameters_match_frozen_legacy_plant_working_condition(self):
        root = Path(__file__).resolve().parents[1]
        legacy_dir = root / "archive" / "legacy_fowt_control"
        sys.path.insert(0, str(legacy_dir))
        try:
            from core_model import FloatingPlatform

            with redirect_stdout(io.StringIO()):
                plant = FloatingPlatform(
                    None,
                    platform_profile="default",
                    allow_linear_mooring_fallback=True,
                )
        finally:
            sys.path.remove(str(legacy_dir))

        params = AttitudeRolloutParameters()
        self.assertEqual(
            params.parameter_source,
            "legacy_fowt_default_working_condition_v1",
        )
        self.assertEqual(params.pitch_inertia_kg_m2, plant.M_total[4, 4])
        self.assertEqual(params.roll_inertia_kg_m2, plant.M_total[3, 3])
        self.assertEqual(params.pitch_stiffness_nm_rad, plant.K_hydro[4])
        self.assertEqual(params.roll_stiffness_nm_rad, plant.K_hydro[3])
        self.assertEqual(params.pitch_damping_ratio, plant.zetas[4])
        self.assertEqual(params.roll_damping_ratio, plant.zetas[3])
        self.assertEqual(params.gravity_m_s2, plant.g)
        np.testing.assert_allclose(
            np.asarray(params.tank_xy_m),
            plant.tank_pos[:, :2],
            rtol=0.0,
            atol=1e-12,
        )

        expected_pitch_damping = 2.0 * plant.zetas[4] * np.sqrt(
            plant.M_total[4, 4] * plant.K_hydro[4]
        )
        expected_roll_damping = 2.0 * plant.zetas[3] * np.sqrt(
            plant.M_total[3, 3] * plant.K_hydro[3]
        )
        self.assertEqual(expected_pitch_damping, plant.C_lin[4, 4])
        self.assertEqual(expected_roll_damping, plant.C_lin[3, 3])

    def test_free_decay_period_matches_default_plant_parameters(self):
        params = self._params(metric_sample_interval_s=0.05)
        result = rollout_attitude(
            [2.0, 0.0],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0]],
            [90.0],
            parameters=params,
        )

        self.assertIsNotNone(result)
        pitch = result.pitch_deg
        peaks = np.flatnonzero(
            (pitch[1:-1] > pitch[:-2]) & (pitch[1:-1] >= pitch[2:])
        ) + 1
        self.assertGreaterEqual(len(peaks), 2)
        observed_period_s = result.times_s[peaks[1]] - result.times_s[peaks[0]]
        omega_n = np.sqrt(
            params.pitch_stiffness_nm_rad / params.pitch_inertia_kg_m2
        )
        expected_period_s = 2.0 * np.pi / (
            omega_n * np.sqrt(1.0 - params.pitch_damping_ratio**2)
        )
        self.assertAlmostEqual(observed_period_s, expected_period_s, delta=0.1)

    def test_static_ballast_moment_uses_plant_pitch_roll_signs(self):
        params = self._params(metric_sample_interval_s=2.0)
        front = rollout_attitude(
            [0.0, 0.0],
            [[10_000.0, 0.0, 0.0]],
            [[0.0, 0.0]],
            [600.0],
            parameters=params,
        )
        port = rollout_attitude(
            [0.0, 0.0],
            [[0.0, 10_000.0, 0.0]],
            [[0.0, 0.0]],
            [600.0],
            parameters=params,
        )
        starboard = rollout_attitude(
            [0.0, 0.0],
            [[0.0, 0.0, 10_000.0]],
            [[0.0, 0.0]],
            [600.0],
            parameters=params,
        )

        self.assertIsNotNone(front)
        self.assertIsNotNone(port)
        self.assertIsNotNone(starboard)
        self.assertGreater(front.stages[0].ballast_moment_nm[0], 0.0)
        self.assertLess(port.stages[0].ballast_moment_nm[1], 0.0)
        self.assertGreater(starboard.stages[0].ballast_moment_nm[1], 0.0)
        self.assertGreater(front.stages[0].end_state.pitch_deg, 0.0)
        self.assertLess(port.stages[0].end_state.roll_deg, 0.0)
        self.assertGreater(starboard.stages[0].end_state.roll_deg, 0.0)

    def test_reverse_ballast_moment_reduces_positive_pitch_response(self):
        no_action = rollout_attitude(
            [2.0, 0.0],
            [[0.0, 0.0, 0.0]],
            [[0.0, 0.0]],
            [5.0],
            parameters=self._params(),
        )
        reverse_action = rollout_attitude(
            [2.0, 0.0],
            [[-50_000.0, 25_000.0, 25_000.0]],
            [[0.0, 0.0]],
            [5.0],
            parameters=self._params(),
        )

        self.assertIsNotNone(no_action)
        self.assertIsNotNone(reverse_action)
        self.assertLess(
            reverse_action.stages[0].end_state.pitch_deg,
            no_action.stages[0].end_state.pitch_deg,
        )
        self.assertLess(
            reverse_action.stages[0].metrics.pitch_rms_deg,
            no_action.stages[0].metrics.pitch_rms_deg,
        )
        self.assertLess(reverse_action.stages[0].ballast_moment_nm[0], 0.0)

    def test_stronger_future_wind_increases_peak_and_threshold_exposure(self):
        mild = rollout_attitude(
            [0.0, 0.0],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [40.0, 40.0],
            parameters=self._params(),
        )
        strong = rollout_attitude(
            [0.0, 0.0],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0], [4.0e8, 2.0e8]],
            [40.0, 40.0],
            parameters=self._params(),
        )

        self.assertIsNotNone(mild)
        self.assertIsNotNone(strong)
        self.assertGreater(strong.metrics.dominant_peak_deg, 5.0)
        self.assertGreater(strong.metrics.vector_rms_deg, mild.metrics.vector_rms_deg)
        self.assertGreater(strong.metrics.dominant_exceedance_s[2.0], 0.0)

    def test_long_horizon_and_overdamped_parameters_remain_finite(self):
        result = rollout_attitude(
            [3.0, -2.0],
            [[10_000.0, -5_000.0, -5_000.0]] * 3,
            [[2.0e8, -1.0e8]] * 3,
            [7_200.0, 7_200.0, 7_200.0],
            initial_pitch_roll_rate_deg_s=[0.2, -0.1],
            parameters=self._params(
                pitch_damping_ratio=1.2,
                roll_damping_ratio=1.5,
                metric_sample_interval_s=60.0,
            ),
        )

        self.assertIsNotNone(result)
        for values in (
            result.pitch_deg,
            result.roll_deg,
            result.pitch_rate_deg_s,
            result.roll_rate_deg_s,
        ):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.isfinite(result.metrics.vector_rms_deg))
        self.assertEqual(result.metrics.duration_s, 21_600.0)

    def test_default_parameters_remain_bounded_for_six_hours(self):
        params = self._params(metric_sample_interval_s=30.0)
        result = rollout_attitude(
            [4.0, -3.0],
            [[0.0, 0.0, 0.0]] * 3,
            [[2.0e8, -1.0e8]] * 3,
            [7_200.0, 7_200.0, 7_200.0],
            initial_pitch_roll_rate_deg_s=[0.15, -0.12],
            parameters=params,
        )

        self.assertIsNotNone(result)
        equilibrium_pitch_deg = np.degrees(
            2.0e8 / params.pitch_stiffness_nm_rad
        )
        equilibrium_roll_deg = np.degrees(
            -1.0e8 / params.roll_stiffness_nm_rad
        )
        self.assertTrue(np.all(np.isfinite(result.pitch_deg)))
        self.assertTrue(np.all(np.isfinite(result.roll_deg)))
        # The non-zero initial rates create a finite first-cycle overshoot.  This
        # test guards numerical boundedness and convergence, not a safety limit.
        self.assertLess(np.max(np.abs(result.pitch_deg)), 10.0)
        self.assertLess(np.max(np.abs(result.roll_deg)), 10.0)
        self.assertAlmostEqual(
            result.stages[-1].end_state.pitch_deg,
            equilibrium_pitch_deg,
            delta=1e-6,
        )
        self.assertAlmostEqual(
            result.stages[-1].end_state.roll_deg,
            equilibrium_roll_deg,
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
