import ast
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODULES = (
    "src/wind_prediction/controller.py",
    "src/wind_prediction/controller_runtime.py",
    "src/wind_prediction/controller_plant_adapter.py",
    "src/wind_prediction/controller_configuration.py",
    "src/wind_prediction/controller_core.py",
    "src/wind_prediction/action_plan.py",
    "src/wind_prediction/ballast_allocation.py",
    "src/wind_prediction/execution_rollout.py",
    "src/wind_prediction/forecast_action_policy.py",
    "src/wind_prediction/forecast_evidence.py",
)


class CompactControllerArchitectureTests(unittest.TestCase):
    def test_active_controller_does_not_import_provider_stack(self):
        forbidden_prefixes = (
            "provider_",
            "ballast_planner_provider",
            "ballast_planner",
            "replay_dataset",
        )
        violations = []
        for relative_path in ACTIVE_MODULES:
            path = ROOT / relative_path
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    imported = (node.module or "").split(".")[-1]
                    if imported.startswith(forbidden_prefixes):
                        violations.append(f"{relative_path}:{node.lineno}:{imported}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name.split(".")[-1]
                        if imported.startswith(forbidden_prefixes):
                            violations.append(
                                f"{relative_path}:{node.lineno}:{imported}"
                            )
        self.assertEqual(violations, [])

    def test_public_facade_exports_the_runtime_entry(self):
        from wind_prediction import controller

        self.assertIn("ForecastAssistedBallastController", controller.__all__)
        self.assertIn("ControllerRuntimeState", controller.__all__)
        self.assertIn("ForecastEvidence", controller.__all__)
        self.assertIn("load_controller_config", controller.__all__)

    def test_active_controller_modules_remain_compact(self):
        oversized = []
        for relative_path in ACTIVE_MODULES:
            path = ROOT / relative_path
            line_count = len(path.read_text(encoding="utf-8").splitlines())
            if line_count > 1200:
                oversized.append(f"{relative_path}:{line_count}")
        self.assertEqual(oversized, [])

    def test_public_controller_import_does_not_eagerly_load_legacy_modules(self):
        script = (
            "import sys; import wind_prediction.controller; "
            "blocked=('candidate_plan_evaluator','far_event_advisory',"
            "'preview_trim_provider','replay_dataset','ballast_planner'); "
            "print(','.join(name for name in blocked "
            "if 'wind_prediction.' + name in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
