import unittest

from src.wind_prediction.provider_config import ProviderConfig


class ProviderConfigTests(unittest.TestCase):
    def test_flattens_disjoint_sections(self):
        config = ProviderConfig(
            planning={"objective_mode": "economic"},
            forecast={"trusted_event_gate_enabled": True},
            target={"primary_hold_target_mode": "pause"},
        )

        self.assertEqual(
            config.to_legacy_kwargs(),
            {
                "objective_mode": "economic",
                "trusted_event_gate_enabled": True,
                "primary_hold_target_mode": "pause",
            },
        )

    def test_rejects_option_in_multiple_sections(self):
        config = ProviderConfig(
            planning={"objective_mode": "economic"},
            experimental={"objective_mode": "lex"},
        )

        with self.assertRaisesRegex(ValueError, "multiple sections"):
            config.to_legacy_kwargs()

    def test_fingerprint_is_order_independent(self):
        left = ProviderConfig(planning={"b": 2, "a": 1})
        right = ProviderConfig(planning={"a": 1, "b": 2})
        self.assertEqual(left.fingerprint(), right.fingerprint())


if __name__ == "__main__":
    unittest.main()
