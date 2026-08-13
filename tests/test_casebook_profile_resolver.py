from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from wind_prediction.casebook_profile_resolver import (
    resolve_primary_control_profile,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "analysis" / "run_prediction_primary_casebook.py"
REGISTRY_PATH = REPO_ROOT / "configs" / "casebook_profile_registry_v1.json"


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("casebook_runner_for_test", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import casebook runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CasebookProfileResolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runner = _load_runner_module()

    def test_production_profile_resolves_to_explicit_settings(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--primary-control-profile",
                "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
            ],
        ):
            args = self.runner.parse_args()

        resolution = resolve_primary_control_profile(
            args=args,
            primary_safety_override=self.runner.primary_safety_override,
            profile_registry_path=REGISTRY_PATH,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(resolution.registry_group, "production")
        self.assertEqual(args.primary_hold_target_mode, "pause")
        self.assertTrue(args.planner_envelope_raw)
        self.assertTrue(args.planner_envelope_barrier)
        self.assertTrue(args.planner_posture_state_residual)
        self.assertTrue(args.relief_medium_cap)

    def test_explicit_cli_value_overrides_named_profile_default(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--primary-control-profile",
                "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
                "--planner-posture-state-gain",
                "1.0",
            ],
        ):
            args = self.runner.parse_args()

        resolve_primary_control_profile(
            args=args,
            primary_safety_override=self.runner.primary_safety_override,
            profile_registry_path=REGISTRY_PATH,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(args.planner_posture_state_gain, 1.0)

    def test_named_profile_default_applies_without_explicit_cli_value(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                str(RUNNER_PATH),
                "--primary-control-profile",
                "rawenv_holdpause_barrier_reliefcap_adaptive_v1",
            ],
        ):
            args = self.runner.parse_args()

        resolve_primary_control_profile(
            args=args,
            primary_safety_override=self.runner.primary_safety_override,
            profile_registry_path=REGISTRY_PATH,
            repo_root=REPO_ROOT,
        )

        self.assertEqual(args.planner_posture_state_gain, 0.4)

    def test_unknown_profile_is_rejected_before_simulation(self) -> None:
        args = SimpleNamespace(
            primary_control_profile="not_registered",
            allow_isolated_profile=False,
        )
        with self.assertRaises(ValueError):
            resolve_primary_control_profile(
                args=args,
                primary_safety_override=lambda _name: {},
                profile_registry_path=REGISTRY_PATH,
                repo_root=REPO_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
