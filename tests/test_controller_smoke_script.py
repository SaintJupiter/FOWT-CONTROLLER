import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ControllerSmokeScriptTests(unittest.TestCase):
    def test_smoke_script_runs_the_versioned_public_controller(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/validation/run_controller_core_smoke.py")],
            cwd=ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["schema_version"], "controller_core_smoke.v2")
        self.assertEqual(payload["completed_cycles"], 3)
        self.assertEqual(len(payload["controller_config"]["sha256"]), 64)
        self.assertEqual(len(payload["traces"]), 3)


if __name__ == "__main__":
    unittest.main()
