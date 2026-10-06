import ast
from pathlib import Path
import unittest

import numpy as np

from fowt_platform import (
    GeneralizedLoadForecast,
    PlatformMatrices,
    diagnose_forecast_ballast_redistribution,
    diagnose_generalized_load_forecast_ballast_redistribution,
)


TANK_COORDINATES_M = np.array(
    [
        [2.0, 0.0, -1.0],
        [-1.0, 2.0, -1.0],
        [-1.0, -2.0, -1.0],
    ]
)
CAPACITIES_KG = np.full(3, 1_000.0)
ACTUAL_MASSES_KG = np.full(3, 500.0)


def _matrices_with_pitch_roll_stiffness(stiffness_pitch_roll):
    restoring = np.eye(6)
    restoring[np.ix_([4, 3], [4, 3])] = stiffness_pitch_roll
    return PlatformMatrices(
        mass=np.eye(6),
        damping=np.zeros((6, 6)),
        hydrostatic_stiffness=restoring,
        mooring_stiffness=np.zeros((6, 6)),
    )


class ForecastBallastProfileDiagnosticTests(unittest.TestCase):
    def make_forecast(self):
        return GeneralizedLoadForecast(
            current_generalized_load=np.zeros(6),
            future_generalized_loads=[
                [0.0, 0.0, 0.0, 20.0, -30.0, 0.0],
                [0.0, 0.0, 0.0, -50.0, 40.0, 0.0],
            ],
            lead_times_s=[600.0, 1200.0],
        )

    def diagnose_profile(self, *, actual=ACTUAL_MASSES_KG):
        return diagnose_generalized_load_forecast_ballast_redistribution(
            forecast=self.make_forecast(),
            matrices=_matrices_with_pitch_roll_stiffness(np.diag([200.0, 100.0])),
            actual_tank_masses_kg=actual,
            tank_capacities_kg=CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

    def test_returns_one_time_ordered_independent_diagnostic_per_future_load(self):
        profile = self.diagnose_profile()

        self.assertEqual(len(profile), 2)
        self.assertEqual([item.lead_time_s for item in profile], [600.0, 1200.0])
        np.testing.assert_allclose(profile[0].actual_tank_masses_kg, ACTUAL_MASSES_KG)
        self.assertFalse(profile[0].actual_tank_masses_kg.flags.writeable)
        np.testing.assert_allclose(profile[0].tank_capacities_kg, CAPACITIES_KG)
        self.assertFalse(profile[0].tank_capacities_kg.flags.writeable)
        np.testing.assert_allclose(profile[0].tank_coordinates_m, TANK_COORDINATES_M)
        self.assertFalse(profile[0].tank_coordinates_m.flags.writeable)
        self.assertEqual(profile[0].gravity_m_s2, 10.0)
        np.testing.assert_allclose(
            profile[0].diagnostic.restoring_diagnostic.relative_pitch_roll_load_nm,
            [-30.0, 20.0],
        )
        np.testing.assert_allclose(
            profile[1].diagnostic.restoring_diagnostic.relative_pitch_roll_load_nm,
            [40.0, -50.0],
        )

    def test_each_profile_entry_matches_a_direct_one_lead_diagnostic(self):
        forecast = self.make_forecast()
        matrices = _matrices_with_pitch_roll_stiffness(np.diag([200.0, 100.0]))
        profile = diagnose_generalized_load_forecast_ballast_redistribution(
            forecast=forecast,
            matrices=matrices,
            actual_tank_masses_kg=ACTUAL_MASSES_KG,
            tank_capacities_kg=CAPACITIES_KG,
            tank_coordinates_m=TANK_COORDINATES_M,
            gravity_m_s2=10.0,
        )

        for index, timed in enumerate(profile):
            with self.subTest(index=index):
                direct = diagnose_forecast_ballast_redistribution(
                    matrices=matrices,
                    current_generalized_load=forecast.current_generalized_load,
                    future_generalized_load=forecast.future_load_at(index),
                    actual_tank_masses_kg=ACTUAL_MASSES_KG,
                    tank_capacities_kg=CAPACITIES_KG,
                    tank_coordinates_m=TANK_COORDINATES_M,
                    gravity_m_s2=10.0,
                )
                np.testing.assert_allclose(
                    timed.diagnostic.allocation.target_tank_masses_kg,
                    direct.allocation.target_tank_masses_kg,
                )
                np.testing.assert_allclose(
                    timed.diagnostic.remaining_relative_pitch_roll_load_nm,
                    direct.remaining_relative_pitch_roll_load_nm,
                )
                np.testing.assert_allclose(timed.actual_tank_masses_kg, ACTUAL_MASSES_KG)

    def test_later_leads_reuse_the_current_actual_tank_state_not_prior_endpoints(self):
        actual = np.array([120.0, 540.0, 810.0])
        profile = self.diagnose_profile(actual=actual)

        self.assertFalse(
            np.allclose(
                profile[0].diagnostic.allocation.target_tank_masses_kg,
                actual,
            )
        )
        np.testing.assert_allclose(
            profile[1].diagnostic.allocation.tank_mass_deltas_kg,
            profile[1].diagnostic.allocation.target_tank_masses_kg - actual,
        )
        self.assertAlmostEqual(
            float(np.sum(profile[1].diagnostic.allocation.tank_mass_deltas_kg)),
            0.0,
        )

    def test_rejects_an_untyped_forecast_contract(self):
        with self.assertRaisesRegex(TypeError, "GeneralizedLoadForecast"):
            diagnose_generalized_load_forecast_ballast_redistribution(
                forecast=object(),
                matrices=_matrices_with_pitch_roll_stiffness(np.eye(2)),
                actual_tank_masses_kg=ACTUAL_MASSES_KG,
                tank_capacities_kg=CAPACITIES_KG,
                tank_coordinates_m=TANK_COORDINATES_M,
            )

    def test_module_does_not_depend_on_control_or_execution_layers(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "fowt_platform"
            / "forecast_ballast_profile_diagnostic.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = ("wind_prediction", "controller", "execution", "pump")
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
