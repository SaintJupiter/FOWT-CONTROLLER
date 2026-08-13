import ast
import unittest
from pathlib import Path


PROVIDER_STATE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "wind_prediction"
    / "provider_state.py"
)


def _self_assignments(function: ast.FunctionDef) -> set[str]:
    fields: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                fields.add(target.attr)
    return fields


class ProviderStateResetTests(unittest.TestCase):
    def test_constructor_delegates_runtime_state_to_reset(self):
        tree = ast.parse(PROVIDER_STATE.read_text())
        provider = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {
            node.name: node for node in provider.body if isinstance(node, ast.FunctionDef)
        }

        reset_calls = [
            node
            for node in ast.walk(methods["__init__"])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr == "reset"
        ]
        self.assertEqual(len(reset_calls), 1)

        constructor_fields = _self_assignments(methods["__init__"])
        reset_fields = _self_assignments(methods["reset"])
        self.assertFalse(
            constructor_fields & reset_fields,
            "runtime state must have a single initialization owner",
        )


if __name__ == "__main__":
    unittest.main()
