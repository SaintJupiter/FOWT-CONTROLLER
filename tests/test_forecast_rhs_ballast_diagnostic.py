import ast
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from tests.test_forecast_platform_rhs_diagnostic import _trajectory
from wind_prediction.forecast_platform_rhs_diagnostic import (
    diagnose_forecast_platform_rhs,
)
from wind_prediction.forecast_rhs_ballast_diagnostic import (
    ForecastRhsBallastDiagnostic,
    diagnose_forecast_horizon_rhs_ballast_redistributions,
    diagnose_forecast_rhs_ballast_redistribution,
)


class ForecastRhsBallastDiagnosticTests(unittest.TestCase):
    def test_uses_one_matched_endpoint_state_equation_and_tank_snapshot(self):
        trajectory = _trajectory()
        rhs_point = diagnose_forecast_platform_rhs(trajectory=trajectory)[0]

        diagnostic = diagnose_forecast_rhs_ballast_redistribution(rhs_point=rhs_point)

        self.assertIs(diagnostic.rhs_point, rhs_point)
        self.assertEqual(diagnostic.lead_index, rhs_point.lead_index)
        self.assertEqual(diagnostic.lead_time_s, rhs_point.lead_time_s)
        np.testing.assert_allclose(
            diagnostic.requested_ballast_pitch_roll_load_nm,
            -rhs_point.dynamic_rhs_generalized_load[[4, 3]],
        )
        np.testing.assert_allclose(
            diagnostic.frozen_matrix_direct_load_residual_pitch_roll_nm,
            rhs_point.dynamic_rhs_generalized_load[[4, 3]]
            + diagnostic.incremental_ballast_generalized_load[[4, 3]],
        )
        np.testing.assert_allclose(
            diagnostic.frozen_matrix_direct_load_residual_pitch_roll_nm,
            -diagnostic.allocation.residual_pitch_roll_moment_nm,
        )
        np.testing.assert_allclose(
            diagnostic.hypothetical_tank_masses_kg,
            diagnostic.allocation.target_tank_masses_kg,
        )
        self.assertAlmostEqual(
            float(np.sum(diagnostic.allocation.tank_mass_deltas_kg)),
            0.0,
        )

        relative_external_load = (
            trajectory.load_assembly.load_forecast.future_load_at(0)
            - trajectory.load_assembly.load_forecast.current_generalized_load
        )[[4, 3]]
        self.assertFalse(
            np.allclose(
                diagnostic.requested_ballast_pitch_roll_load_nm,
                -relative_external_load,
            )
        )

    def test_rejects_a_request_or_residual_rebound_from_the_rhs_point(self):
        rhs_point = diagnose_forecast_platform_rhs(trajectory=_trajectory())[0]
        diagnostic = diagnose_forecast_rhs_ballast_redistribution(rhs_point=rhs_point)

        with self.assertRaisesRegex(ValueError, "must counter the matching dynamic RHS"):
            replace(
                diagnostic,
                requested_ballast_pitch_roll_load_nm=(
                    diagnostic.requested_ballast_pitch_roll_load_nm
                    + np.array([1.0, 0.0])
                ),
            )

    def test_retains_one_source_bound_ballast_fact_per_forecast_lead(self):
        trajectory = _trajectory()

        diagnostics = diagnose_forecast_horizon_rhs_ballast_redistributions(
            trajectory=trajectory
        )

        self.assertEqual(len(diagnostics), len(trajectory.steps))
        for index, diagnostic in enumerate(diagnostics):
            self.assertIs(diagnostic.rhs_point.trajectory, trajectory)
            self.assertEqual(diagnostic.lead_index, index)
            self.assertEqual(diagnostic.lead_time_s, trajectory.steps[index].end_time_s)
            self.assertAlmostEqual(
                float(np.sum(diagnostic.allocation.tank_mass_deltas_kg)),
                0.0,
            )
        with self.assertRaisesRegex(ValueError, "frozen-matrix direct-load residual"):
            replace(
                diagnostic,
                frozen_matrix_direct_load_residual_pitch_roll_nm=(
                    diagnostic.frozen_matrix_direct_load_residual_pitch_roll_nm
                    + np.array([0.0, 1.0])
                ),
            )

    def test_module_does_not_import_raw_forecast_or_control_logic(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "wind_prediction"
            / "forecast_rhs_ballast_diagnostic.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        forbidden_fragments = (
            "forecast_evidence",
            "controller",
            "candidate",
            "planner",
            "execution",
            "pump",
        )
        self.assertFalse(
            any(
                any(fragment in module for fragment in forbidden_fragments)
                for module in imported_modules
            )
        )


if __name__ == "__main__":
    unittest.main()
