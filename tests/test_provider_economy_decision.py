import dataclasses
import unittest

import numpy as np

from src.wind_prediction.provider_economy import (
    ProviderEconomyMixin,
    _EconomySignals,
)


def _signals(**overrides):
    values = {}
    for field in dataclasses.fields(_EconomySignals):
        if field.name in {"posture", "vec0", "vec2"}:
            values[field.name] = np.zeros(2, dtype=float)
        elif field.name == "mode":
            values[field.name] = "budget"
        elif field.type is bool or field.type == "bool":
            values[field.name] = False
        else:
            values[field.name] = 0.0
    values.update(overrides)
    return _EconomySignals(**values)


class ProviderEconomyDecisionTests(unittest.TestCase):
    def test_budget_mode_uses_explicit_signal_bundle(self):
        provider = ProviderEconomyMixin.__new__(ProviderEconomyMixin)
        provider.economy_pump_budget_forecast_smart = False

        should_hold, reason = provider._economy_budget_mode_decision(_signals())

        self.assertTrue(should_hold)
        self.assertEqual(reason, "budget_exhausted_hold")

    def test_budget_mode_releases_for_forecast_floor_risk(self):
        provider = ProviderEconomyMixin.__new__(ProviderEconomyMixin)
        provider.economy_pump_budget_forecast_smart = True

        should_hold, reason = provider._economy_budget_mode_decision(
            _signals(floor_risk=True)
        )

        self.assertFalse(should_hold)
        self.assertEqual(reason, "budget_smart_floor_risk_release")


if __name__ == "__main__":
    unittest.main()
