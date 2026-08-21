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
from wind_prediction.controller import ForecastAssistedBallastController


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

    def test_missing_legacy_demand_scale_uses_the_historical_deadband_value(self):
        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        document["controller"].pop("legacy_demand_axis_scale_deg")

        config = parse_controller_config(document)

        self.assertIsNone(config.legacy_demand_axis_scale_deg)
        self.assertEqual(
            config.resolved_legacy_demand_axis_scale_deg,
            config.deadband_deg,
        )
        normalized = controller_config_to_dict(config)
        self.assertEqual(
            tuple(normalized["controller"]["legacy_demand_axis_scale_deg"]),
            config.deadband_deg,
        )
        reparsed = parse_controller_config(normalized)
        self.assertEqual(
            reparsed.legacy_demand_axis_scale_deg,
            config.deadband_deg,
        )

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

    def test_all_configuration_sections_are_required_objects(self):
        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        for section in ("controller", "execution", "forecast_policy"):
            with self.subTest(section=section, mode="missing"):
                changed = dict(document)
                changed.pop(section)
                with self.assertRaisesRegex(ValueError, "missing required"):
                    parse_controller_config(changed)
            with self.subTest(section=section, mode="wrong_type"):
                changed = dict(document)
                changed[section] = []
                with self.assertRaisesRegex(ValueError, "JSON object"):
                    parse_controller_config(changed)

    def test_string_booleans_are_rejected(self):
        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        document["execution"]["target_slew_enabled"] = "false"
        with self.assertRaisesRegex(ValueError, "boolean"):
            parse_controller_config(document)

    def test_invalid_execution_and_forecast_values_are_rejected(self):
        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        document["execution"]["max_pump_rate_m3_min"] = -1.0
        with self.assertRaisesRegex(ValueError, "max_pump_rate"):
            parse_controller_config(document)

        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        document["forecast_policy"]["direction_consistency_min"] = 2.0
        with self.assertRaisesRegex(ValueError, "direction_consistency"):
            parse_controller_config(document)

        document = json.loads(
            (ROOT / "configs/controller_core_v2.json").read_text(encoding="utf-8")
        )
        document["controller"]["minimum_action_demand_ratio"] = 1.1
        with self.assertRaisesRegex(ValueError, "minimum_action_demand_ratio"):
            parse_controller_config(document)

    def test_public_constructor_uses_file_configuration_and_digest(self):
        loaded = load_controller_config(ROOT / "configs/controller_core_v2.json")
        controller = ForecastAssistedBallastController.from_config_file(
            loaded.source_path
        )
        self.assertEqual(controller.config, loaded.config)
        self.assertEqual(controller.config_sha256, loaded.sha256)

    def test_duplicate_json_keys_are_rejected(self):
        path = ROOT / "configs/controller_core_v2.json"
        raw = path.read_text(encoding="utf-8")
        duplicate = raw.replace(
            '"schema_version": "controller_core.v2",',
            '"schema_version": "controller_core.v2",\n  "schema_version": "controller_core.v2",',
            1,
        )
        temp_path = ROOT / "configs/.controller_core_duplicate_test.json"
        try:
            temp_path.write_text(duplicate, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                load_controller_config(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
