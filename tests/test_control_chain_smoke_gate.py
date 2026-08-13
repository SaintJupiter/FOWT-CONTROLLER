import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "analysis" / "check_control_chain_smoke_gate.py"
SPEC = importlib.util.spec_from_file_location("control_chain_smoke_gate", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ControlChainTimeseriesContractTests(unittest.TestCase):
    def _write(self, directory, *, final_offset=0.0):
        path = Path(directory) / "case_a_example_timeseries.csv"
        row = {
            "ballast_total_kg": 60.0,
            "ballast_total_delta_kg": 0.0,
        }
        for tank, mass, delta, net_rate in (
            (1, 10.0, 1.0, 0.1),
            (2, 20.0, -1.0, -0.1),
            (3, 30.0, 0.0, 0.0),
        ):
            row[f"tank{tank}_kg"] = mass
            row[f"tank{tank}_mass_delta_kg"] = delta
            row[f"pump_rate{tank}_m3min"] = abs(net_rate)
            row[f"pump_net_rate{tank}_m3min"] = net_rate
            row[f"target_tank{tank}_kg"] = mass
            row[f"target_final_tank{tank}_kg"] = mass + final_offset
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return row

    def _contract(self, row):
        return {
            "required_columns": list(row),
            "target_abs_max": 1e-6,
            "mass_balance_abs_max": 1e-6,
            "pump_rate_abs_max": 1e-6,
        }

    def test_valid_timeseries_contract_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self._write(directory)
            failures = MODULE._timeseries_contract_failures(
                Path(directory), ["case_a"], self._contract(row)
            )
            self.assertEqual(failures, [])

    def test_target_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            row = self._write(directory, final_offset=1.0)
            failures = MODULE._timeseries_contract_failures(
                Path(directory), ["case_a"], self._contract(row)
            )
            self.assertTrue(any("final target/plant target gap" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
