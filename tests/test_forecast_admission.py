import unittest

from wind_prediction.forecast_admission import (
    EVENT_RISK_KEYS,
    TrustedEventGateConfig,
    trusted_event_gate,
)


class TrustedEventGateTests(unittest.TestCase):
    def _config(self, **overrides):
        values = {
            "enabled": True,
            "attention_threshold": 0.6,
            "highwind_threshold": 0.8,
            "dynamic_threshold": 0.7,
            "attention_min_heads": 2,
            "dynamic_min_heads": 2,
            "dynamic_enabled": True,
        }
        values.update(overrides)
        return TrustedEventGateConfig(**values)

    def test_disabled_gate_preserves_all_probabilities(self):
        raw = {"ballast_attention_event": 0.2, EVENT_RISK_KEYS[0]: 0.3}
        effective, diagnostics = trusted_event_gate(
            raw, self._config(enabled=False)
        )
        self.assertEqual(effective, raw)
        self.assertEqual(diagnostics["trusted"], 1)
        self.assertEqual(diagnostics["reason"], "disabled")

    def test_untrusted_signal_clears_controller_risk_channels(self):
        raw = {
            "ballast_attention_event": 0.4,
            EVENT_RISK_KEYS[0]: 0.5,
            EVENT_RISK_KEYS[1]: 0.4,
            EVENT_RISK_KEYS[2]: 0.3,
            "unrelated_probability": 0.9,
        }
        effective, diagnostics = trusted_event_gate(raw, self._config())
        self.assertEqual(diagnostics["trusted"], 0)
        self.assertEqual(diagnostics["reason"], "untrusted_event_signal")
        self.assertEqual(effective["ballast_attention_event"], 0.0)
        self.assertTrue(all(effective[key] == 0.0 for key in EVENT_RISK_KEYS))
        self.assertEqual(effective["unrelated_probability"], 0.9)

    def test_high_wind_event_is_admitted(self):
        raw = {"future_speed_ge_train_p95": 0.9, EVENT_RISK_KEYS[0]: 0.2}
        effective, diagnostics = trusted_event_gate(raw, self._config())
        self.assertEqual(effective, raw)
        self.assertEqual(diagnostics["trusted"], 1)
        self.assertEqual(diagnostics["reason"], "highwind")

    def test_attention_and_dynamic_multihead_reasons_are_combined(self):
        raw = {
            "ballast_attention_event": 0.7,
            EVENT_RISK_KEYS[0]: 0.7,
            "speed_ramp_ge_3ms": 0.8,
            "direction_shift_ge_45deg": 0.8,
        }
        _, diagnostics = trusted_event_gate(raw, self._config())
        self.assertEqual(diagnostics["trusted"], 1)
        self.assertEqual(
            diagnostics["reason"],
            "attention_multihead+dynamic_multihead",
        )


if __name__ == "__main__":
    unittest.main()
