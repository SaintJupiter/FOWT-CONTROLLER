import ast
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import GeneralizedLoadForecast


class GeneralizedLoadForecastTests(unittest.TestCase):
    def make_forecast(self):
        return GeneralizedLoadForecast(
            current_generalized_load=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            future_generalized_loads=[
                [10.0, 0.0, 0.0, 0.0, 20.0, 0.0],
                [11.0, 0.0, 0.0, 0.0, 21.0, 0.0],
                [12.0, 0.0, 0.0, 0.0, 22.0, 0.0],
            ],
            lead_times_s=[600.0, 1200.0, 1800.0],
        )

    def test_preserves_discrete_load_points_and_positive_lead_times(self):
        forecast = self.make_forecast()

        self.assertEqual(forecast.horizon_steps, 3)
        self.assertEqual(forecast.lead_time_at(0), 600.0)
        self.assertEqual(forecast.lead_time_at(2), 1800.0)
        np.testing.assert_allclose(
            forecast.future_load_at(0),
            [10.0, 0.0, 0.0, 0.0, 20.0, 0.0],
        )
        np.testing.assert_allclose(
            forecast.future_load_at(2),
            [12.0, 0.0, 0.0, 0.0, 22.0, 0.0],
        )

    def test_constructor_copies_inputs_and_accessor_cannot_mutate_the_contract(self):
        current = np.zeros(6)
        future = np.arange(12, dtype=float).reshape(2, 6)
        leads = np.array([600.0, 1200.0])
        forecast = GeneralizedLoadForecast(
            current_generalized_load=current,
            future_generalized_loads=future,
            lead_times_s=leads,
        )
        current[0] = 99.0
        future[0, 0] = 99.0
        leads[0] = 99.0
        returned = forecast.future_load_at(0)
        returned[0] = -1.0

        self.assertEqual(forecast.current_generalized_load[0], 0.0)
        self.assertEqual(forecast.future_load_at(0)[0], 0.0)
        self.assertEqual(forecast.lead_time_at(0), 600.0)

    def test_rejects_invalid_shapes_nonpositive_or_nonmonotonic_leads_and_bad_indices(self):
        with self.assertRaisesRegex(ValueError, "shape \\(6,\\)"):
            GeneralizedLoadForecast(
                current_generalized_load=[0.0] * 5,
                future_generalized_loads=np.zeros((2, 6)),
                lead_times_s=[600.0, 1200.0],
            )
        with self.assertRaisesRegex(ValueError, "shape \\(H, 6\\)"):
            GeneralizedLoadForecast(
                current_generalized_load=np.zeros(6),
                future_generalized_loads=np.zeros(6),
                lead_times_s=[600.0],
            )
        for leads in ([0.0, 600.0], [600.0, 600.0], [1200.0, 600.0]):
            with self.subTest(leads=leads), self.assertRaisesRegex(
                ValueError, "positive and strictly increasing"
            ):
                GeneralizedLoadForecast(
                    current_generalized_load=np.zeros(6),
                    future_generalized_loads=np.zeros((2, 6)),
                    lead_times_s=leads,
                )

        forecast = self.make_forecast()
        for index in (-1, 3):
            with self.subTest(index=index), self.assertRaises(IndexError):
                forecast.future_load_at(index)
        with self.assertRaises(TypeError):
            forecast.lead_time_at(True)

    def test_does_not_depend_on_wind_prediction_or_execution_modules(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "fowt_platform"
            / "generalized_load_forecast.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertFalse(any(name.startswith("wind_prediction") for name in imported_modules))
        self.assertFalse(any("execution" in name for name in imported_modules))


if __name__ == "__main__":
    unittest.main()
