import ast
import json
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class ArchitectureConfigTests(unittest.TestCase):
    def test_production_config_references_current_smoke_gate(self):
        production = json.loads(
            (REPO_ROOT / "configs" / "production_controller_v1.json").read_text()
        )
        gate_path = REPO_ROOT / production["control_chain_smoke_gate"]
        self.assertTrue(gate_path.is_file())

        gate = json.loads(gate_path.read_text())
        self.assertEqual(gate["schema_version"], "2.0")
        reference_path = REPO_ROOT / gate["reference_summary_csv"]
        self.assertTrue(reference_path.is_file())

    def test_provider_architecture_does_not_regress_to_monolith(self):
        provider_dir = REPO_ROOT / "src" / "wind_prediction"
        facade = provider_dir / "ballast_planner_provider.py"
        self.assertLessEqual(len(facade.read_text().splitlines()), 120)

        behavior_modules = [
            path
            for path in provider_dir.glob("provider_*.py")
            if path.name not in {"provider_telemetry.py", "provider_settings.py"}
        ]
        largest_module = max(len(path.read_text().splitlines()) for path in behavior_modules)
        self.assertLessEqual(largest_module, 2000)

        runtime_tree = ast.parse((provider_dir / "provider_runtime.py").read_text())
        runtime_methods = [
            node
            for node in ast.walk(runtime_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        largest_runtime_method = max(
            node.end_lineno - node.lineno + 1 for node in runtime_methods
        )
        self.assertLessEqual(largest_runtime_method, 250)

    def test_closed_loop_policy_keeps_cycle_stages_explicit(self):
        policy_path = (
            REPO_ROOT / "archive" / "legacy_fowt_control" / "controllers_extras.py"
        )
        policy_tree = ast.parse(policy_path.read_text(encoding="utf-8-sig"))
        policy_class = next(
            node
            for node in policy_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ClosedLoopPolicy"
        )
        methods = {
            node.name: node
            for node in policy_class.body
            if isinstance(node, ast.FunctionDef)
        }
        stage_names = {
            "_resolve_cycle_setpoints",
            "_compute_feedback_target",
            "_select_primary_target",
            "_commit_target_transaction",
        }
        self.assertTrue(stage_names.issubset(methods))
        for name in stage_names:
            method = methods[name]
            self.assertLessEqual(method.end_lineno - method.lineno + 1, 180, name)

        source = policy_path.read_text(encoding="utf-8-sig")
        self.assertNotIn("class CommandRateLimiter", source)


if __name__ == "__main__":
    unittest.main()
