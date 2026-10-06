import unittest

import numpy as np

from fowt_platform import GeneralizedLoadForecast
from wind_prediction.preview_mpc import assemble_preview_disturbance_blocks


class PreviewDisturbanceBlockTests(unittest.TestCase):
    def test_endpoint_forecasts_become_piecewise_linear_interval_averages(self):
        forecast = GeneralizedLoadForecast(
            current_generalized_load=[0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            future_generalized_loads=[
                [0.0, 0.0, 0.0, 0.0, 30.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 50.0, 0.0],
            ],
            lead_times_s=[600.0, 1_200.0],
        )
        ballast = np.array([0.0, 0.0, 0.0, 2.0, -4.0, 0.0])

        result = assemble_preview_disturbance_blocks(
            rotor_load_forecast=forecast,
            current_incremental_ballast_load=ballast,
            block_duration_s=600.0,
        )

        np.testing.assert_allclose(
            result.generalized_loads,
            [
                [0.0, 0.0, 0.0, 2.0, 16.0, 0.0],
                [0.0, 0.0, 0.0, 2.0, 36.0, 0.0],
            ],
        )
        self.assertEqual(
            result.endpoint_mapping,
            "piecewise_linear_endpoints_to_interval_average_constant_load",
        )

    def test_current_ballast_load_is_present_in_every_forecast_block(self):
        forecast = GeneralizedLoadForecast(
            current_generalized_load=np.zeros(6),
            future_generalized_loads=np.zeros((3, 6)),
            lead_times_s=[600.0, 1_200.0, 1_800.0],
        )
        ballast = np.array([0.0, 0.0, -12.0, 4.0, 6.0, 0.0])
        result = assemble_preview_disturbance_blocks(
            rotor_load_forecast=forecast,
            current_incremental_ballast_load=ballast,
            block_duration_s=600.0,
        )

        np.testing.assert_allclose(
            result.generalized_loads,
            np.repeat(ballast.reshape(1, 6), 3, axis=0),
        )

    def test_mismatched_forecast_grid_is_rejected(self):
        forecast = GeneralizedLoadForecast(
            current_generalized_load=np.zeros(6),
            future_generalized_loads=np.zeros((2, 6)),
            lead_times_s=[300.0, 900.0],
        )

        with self.assertRaisesRegex(ValueError, "consecutive block endpoints"):
            assemble_preview_disturbance_blocks(
                rotor_load_forecast=forecast,
                current_incremental_ballast_load=np.zeros(6),
                block_duration_s=600.0,
            )


if __name__ == "__main__":
    unittest.main()
