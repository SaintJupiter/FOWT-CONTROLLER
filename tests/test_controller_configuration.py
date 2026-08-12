import json
import unittest
from pathlib import Path

from wind_prediction.controller_configuration import (
    CONTROLLER_CONFIG_SCHEMA_VERSION,
    controller_config_digest,
    controller_config_to_dict,
    load_controller_config,
    parse_controller_config,
)


ROOT = Path(__file__).resolve().parents[1]


class ControllerConfigurationTests(unittest.TestCase):
    def test_repository_configuration_loads_and_round_trips(self):
        loaded = load_controller_config(ROOT / "configs/controller_core_v2.json")
        document = controller_config_to_dict(loaded.config)
        reparsed = parse_controller_config(document)

        self.assertEqual(document["schema_version"], CONTROLLER_CONFIG_SCHEMA_VERSION)
        self.assertEqual(reparsed, loaded.config)
        self.assertEqual(loaded.sha256, controller_config_digest(document))
        self.assertEqual(len(loaded.sha256), 64)

    def test_unknown_configuration_key_is_rejected(self):
        loaded = load_controller_config(ROOT / "configs/controller_core_v2.json")
        document = controller_config_to_dict(loaded.config)
        document["controller"]["mystery_overlay"] = True

        with self.assertRaisesRegex(ValueError, "mystery_overlay"):
            parse_controller_config(document)

    def test_schema_version_is_required(self):
        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        document["schema_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_controller_config(document)


if __name__ == "__main__":
    unittest.main()
