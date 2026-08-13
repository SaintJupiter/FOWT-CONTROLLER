import json
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = REPO_ROOT / "archive" / "legacy_fowt_control"
sys.path.insert(0, str(LEGACY_DIR))

from core_model import FloatingPlatform  # noqa: E402


class SubmittedPaperDefaultRegressionTests(unittest.TestCase):
    @staticmethod
    def _plant(profile):
        stiffness_path = LEGACY_DIR / "data" / "副本水平刚度曲线.xlsx"
        return FloatingPlatform(
            str(stiffness_path),
            platform_profile=profile,
            allow_linear_mooring_fallback=False,
        )

    def test_frozen_default_profile_behavior(self):
        fixture_path = (
            REPO_ROOT
            / "tests"
            / "fixtures"
            / "submitted_paper_default_regression_v1.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        plant = self._plant("default")
        plant.wave_components = []

        self.assertEqual(plant.platform_profile_name, fixture["platform_profile_name"])
        for name, expected in fixture["modes"].items():
            self.assertEqual(getattr(plant, name), expected)
        np.testing.assert_allclose(plant.M_total, fixture["M_total"], rtol=1e-12)
        np.testing.assert_allclose(plant.K_hydro, fixture["K_hydro"], rtol=1e-12)
        np.testing.assert_allclose(plant.C_lin, fixture["C_lin"], rtol=1e-12)

        state = np.asarray(fixture["state"], dtype=float)
        thrust = float(fixture["thrust_n"])
        direction = float(fixture["wind_direction_deg"])
        loads = plant.generalized_load_components(17.0, state, thrust, direction)
        derivative = plant._dynamics(17.0, state, thrust, direction)

        np.testing.assert_allclose(loads["total"], fixture["total_load"], rtol=1e-11)
        np.testing.assert_allclose(
            derivative,
            fixture["state_derivative"],
            rtol=1e-11,
        )

    def test_default_aliases_have_identical_short_execution(self):
        plants = [self._plant(profile) for profile in (None, "", "default")]
        for plant in plants:
            plant.wave_components = []
            plant.set_ballast_target(
                *(plant.current_ballast_mass + np.array([1200.0, -800.0, 400.0]))
            )
            for index in range(20):
                plant.step(750_000.0, 25.0, 0.1, index * 0.1)

        reference = plants[-1]
        for plant in plants[:-1]:
            self.assertEqual(plant.platform_profile_name, "default")
            np.testing.assert_array_equal(plant.state, reference.state)
            np.testing.assert_array_equal(
                plant.current_ballast_mass,
                reference.current_ballast_mass,
            )
            np.testing.assert_array_equal(
                plant.target_ballast_mass,
                reference.target_ballast_mass,
            )


if __name__ == "__main__":
    unittest.main()
