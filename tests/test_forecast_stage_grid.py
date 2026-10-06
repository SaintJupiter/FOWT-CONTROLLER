import unittest

from wind_prediction.forecast_stage_grid import build_forecast_stage_grid


class ForecastStageGridTests(unittest.TestCase):
    def test_default_six_ten_minute_points_form_three_twenty_minute_stages(self):
        grid = build_forecast_stage_grid(
            sample_period_s=600.0,
            stage_duration_s=1200.0,
            stage_count=3,
            available_point_count=6,
        )

        self.assertEqual(grid.points_per_stage, 2)
        self.assertEqual(grid.required_point_count, 6)
        self.assertEqual(
            [
                (
                    window.point_indices,
                    window.terminal_point_index,
                    window.lead_start_s,
                    window.lead_end_s,
                )
                for window in grid.windows
            ],
            [
                ((0, 1), 1, 600.0, 1200.0),
                ((2, 3), 3, 1800.0, 2400.0),
                ((4, 5), 5, 3000.0, 3600.0),
            ],
        )

    def test_rejects_a_forecast_period_that_cannot_partition_stages(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            build_forecast_stage_grid(
                sample_period_s=900.0,
                stage_duration_s=1200.0,
                stage_count=3,
            )

    def test_rejects_insufficient_discrete_future_records(self):
        with self.assertRaisesRegex(ValueError, "need 6 points, got 5"):
            build_forecast_stage_grid(
                sample_period_s=600.0,
                stage_duration_s=1200.0,
                stage_count=3,
                available_point_count=5,
            )


if __name__ == "__main__":
    unittest.main()
