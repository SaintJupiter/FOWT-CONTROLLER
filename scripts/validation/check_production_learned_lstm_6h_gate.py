#!/usr/bin/env python3
"""Preflight the production learned-LSTM six-hour validation gate."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_RUNNER = "scripts/analysis/run_prediction_primary_casebook.py"
DEFAULT_CONFIG = "configs/production_learned_lstm_6h_gate_v1.json"


class GateConfigError(ValueError):
    """Raised when the learned-LSTM gate is incomplete or ambiguous."""


@dataclass(frozen=True)
class GateReport:
    config_path: Path
    cases_csv: Path
    dataset_dir: Path
    model_dir: Path
    planner_runtime_config: Path
    output_dir: Path
    run_identity_path: Path
    run_identity_present: bool
    command: tuple[str, ...]


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateConfigError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateConfigError(f"{path} must contain a JSON object")
    return value


def _command(raw: dict[str, Any]) -> tuple[str, ...]:
    value = raw.get("casebook_command")
    if not isinstance(value, list) or len(value) < 2:
        raise GateConfigError("casebook_command must contain a Python executable and runner")
    command = tuple(str(item) for item in value)
    if command[1] != CANONICAL_RUNNER:
        raise GateConfigError(f"gate must use {CANONICAL_RUNNER}")
    if any("\x00" in item or "\n" in item for item in command):
        raise GateConfigError("casebook_command contains an invalid argument")
    return command


def _option(command: tuple[str, ...], name: str) -> str:
    indices = [index for index, value in enumerate(command) if value == name]
    if len(indices) != 1:
        raise GateConfigError(f"{name} must appear exactly once")
    index = indices[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise GateConfigError(f"{name} must have one explicit value")
    return command[index + 1]


def _flag(command: tuple[str, ...], name: str, *, required: bool) -> None:
    count = command.count(name)
    if required and count != 1:
        raise GateConfigError(f"{name} must appear exactly once")
    if not required and count:
        raise GateConfigError(f"{name} is forbidden in this gate")


def _contract(raw: dict[str, Any]) -> dict[str, Any]:
    value = raw.get("gate_contract")
    if not isinstance(value, dict):
        raise GateConfigError("gate_contract must be a JSON object")
    return value


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise GateConfigError(f"{label} must be {expected!r}, got {actual!r}")


def _validate_cases(
    cases_path: Path,
    expected_cases: tuple[str, ...],
    expected_count: int,
) -> None:
    if not cases_path.is_file():
        raise GateConfigError(f"cases CSV is missing: {cases_path}")
    with cases_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_count:
        raise GateConfigError(
            f"cases CSV must contain {expected_count} row(s), found {len(rows)}"
        )
    case_ids = tuple(str(row.get("case_id", "")).strip() for row in rows)
    if any(not case_id for case_id in case_ids):
        raise GateConfigError("every cases CSV row must define case_id")
    _assert_equal(case_ids, expected_cases, "cases CSV case ids")
    if any(not str(row.get("timestamp", "")).strip() for row in rows):
        raise GateConfigError("every cases CSV row must define timestamp")


def _validate_assets(base: Path, names: Iterable[str], label: str) -> None:
    if not base.is_dir():
        raise GateConfigError(f"{label} directory is missing: {base}")
    missing = [name for name in names if not (base / str(name)).is_file()]
    if missing:
        raise GateConfigError(f"{label} assets are missing: {', '.join(missing)}")


def _validate_run_identity(
    path: Path,
    identity_contract: dict[str, Any],
    *,
    repo_root: Path,
    profile: str,
    output_dir: Path,
) -> None:
    identity = _load_object(path)
    _assert_equal(
        identity.get("schema_version"),
        identity_contract.get("schema_version"),
        "run identity schema_version",
    )
    _assert_equal(identity.get("control_profile"), profile, "run identity control_profile")
    _assert_equal(identity.get("forecast_source"), "learned", "run identity forecast_source")
    _assert_equal(
        identity.get("forecast_model_version"),
        identity_contract.get("forecast_model_version"),
        "run identity forecast_model_version",
    )
    expected_output = str(output_dir.relative_to(repo_root))
    _assert_equal(
        identity.get("output_directory"),
        expected_output,
        "run identity output_directory",
    )
    required_fields = identity_contract.get("required_fields", [])
    if not isinstance(required_fields, list):
        raise GateConfigError("run_identity_contract.required_fields must be a list")
    missing = [str(field) for field in required_fields if field not in identity]
    if missing:
        raise GateConfigError(f"run identity fields are missing: {', '.join(missing)}")
    input_files = identity.get("input_files")
    if not isinstance(input_files, dict):
        raise GateConfigError("run identity input_files must be an object")
    for group in ("model", "dataset", "configuration", "cases", "plant"):
        files = input_files.get(group)
        if not isinstance(files, dict) or not files:
            raise GateConfigError(f"run identity input group is missing: {group}")
        unavailable = [
            str(name)
            for name, fingerprint in files.items()
            if not isinstance(fingerprint, dict)
            or fingerprint.get("status") != "present"
            or fingerprint.get("sha256") in (None, "", "missing")
        ]
        if unavailable:
            raise GateConfigError(
                f"run identity has unavailable {group} inputs: {', '.join(unavailable)}"
            )


def validate_gate_config(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    repo_root: str | Path = REPO_ROOT,
    require_run_identity: bool = False,
) -> GateReport:
    root = Path(repo_root).resolve()
    source = _resolve(root, config_path)
    raw = _load_object(source)
    _assert_equal(
        raw.get("schema_version"),
        "production_learned_lstm_6h_gate.v1",
        "schema_version",
    )
    command = _command(raw)
    contract = _contract(raw)

    duration = float(_option(command, "--duration-s"))
    expected_duration = float(contract.get("duration_s", 0.0))
    _assert_equal(duration, 21600.0, "--duration-s")
    _assert_equal(expected_duration, 21600.0, "gate_contract.duration_s")

    forecast_source = _option(command, "--forecast-source")
    _assert_equal(forecast_source, contract.get("forecast_source"), "--forecast-source")
    _assert_equal(forecast_source, "learned", "--forecast-source")
    forbidden_sources = {str(item) for item in contract.get("forbidden_forecast_sources", [])}
    if forecast_source in forbidden_sources:
        raise GateConfigError(f"forecast source {forecast_source!r} is forbidden")

    profile = _option(command, "--primary-control-profile")
    _assert_equal(profile, contract.get("primary_control_profile"), "control profile")
    forbidden_profiles = {str(item) for item in contract.get("forbidden_control_profiles", [])}
    if profile in forbidden_profiles or profile == "manual":
        raise GateConfigError(f"control profile {profile!r} is forbidden")

    replay_split = _option(command, "--replay-split")
    _assert_equal(replay_split, contract.get("replay_split"), "--replay-split")
    _assert_equal(replay_split, "test", "--replay-split")
    _flag(
        command,
        "--primary-only",
        required=bool(contract.get("require_primary_only", False)),
    )
    _flag(command, "--reactive-primary-only", required=False)
    _flag(command, "--allow-isolated-profile", required=False)

    cases_csv = _resolve(root, _option(command, "--cases-csv"))
    expected_raw = raw.get("expected_cases")
    if not isinstance(expected_raw, list) or not expected_raw:
        raise GateConfigError("expected_cases must be a non-empty list")
    expected_cases = tuple(str(item) for item in expected_raw)
    command_cases = tuple(
        item.strip() for item in _option(command, "--case-ids").split(",") if item.strip()
    )
    _assert_equal(command_cases, expected_cases, "--case-ids")
    expected_count = int(contract.get("case_count", 0))
    _assert_equal(expected_count, 1, "gate_contract.case_count")
    _validate_cases(cases_csv, expected_cases, expected_count)

    dataset_value = _option(command, "--dataset-dir")
    model_value = _option(command, "--model-dir")
    runtime_value = _option(command, "--planner-runtime-config")
    _assert_equal(dataset_value, contract.get("dataset_dir"), "--dataset-dir")
    _assert_equal(model_value, contract.get("model_dir"), "--model-dir")
    _assert_equal(runtime_value, contract.get("planner_runtime_config"), "planner runtime")
    dataset_dir = _resolve(root, dataset_value)
    model_dir = _resolve(root, model_value)
    runtime_path = _resolve(root, runtime_value)

    assets = raw.get("required_assets")
    if not isinstance(assets, dict):
        raise GateConfigError("required_assets must be a JSON object")
    _validate_assets(dataset_dir, assets.get("dataset", []), "dataset")
    _validate_assets(model_dir, assets.get("model", []), "model")
    runtime = _load_object(runtime_path)
    _assert_equal(
        runtime.get("schema_version"),
        contract.get("planner_runtime_schema"),
        "planner runtime schema_version",
    )

    stiffness_path = _resolve(root, _option(command, "--stiffness-file"))
    if not stiffness_path.is_file():
        raise GateConfigError(f"stiffness file is missing: {stiffness_path}")

    output_dir = _resolve(root, _option(command, "--out-dir"))
    identity_contract = raw.get("run_identity_contract")
    if not isinstance(identity_contract, dict) or not identity_contract.get("required"):
        raise GateConfigError("run_identity_contract.required must be true")
    identity_path = _resolve(root, str(identity_contract.get("path", "")))
    _assert_equal(identity_path.parent, output_dir, "run identity output directory")
    identity_present = identity_path.is_file()
    if require_run_identity and not identity_present:
        raise GateConfigError(f"run identity is missing: {identity_path}")
    if identity_present:
        _validate_run_identity(
            identity_path,
            identity_contract,
            repo_root=root,
            profile=profile,
            output_dir=output_dir,
        )

    return GateReport(
        config_path=source,
        cases_csv=cases_csv,
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        planner_runtime_config=runtime_path,
        output_dir=output_dir,
        run_identity_path=identity_path,
        run_identity_present=identity_present,
        command=command,
    )


def _runner_dry_config(report: GateReport) -> None:
    command = list(report.command)
    command[0] = sys.executable
    command[1] = str(_resolve(REPO_ROOT, command[1]))
    command.append("--dry-config")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--require-run-identity",
        action="store_true",
        help="also require and validate run_identity.json from a completed gate run",
    )
    parser.add_argument(
        "--runner-dry-config",
        action="store_true",
        help="invoke the canonical runner with --dry-config; no simulation is executed",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the exact single-case six-hour command after validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = validate_gate_config(
            args.config,
            repo_root=REPO_ROOT,
            require_run_identity=bool(args.require_run_identity),
        )
        if args.runner_dry_config:
            _runner_dry_config(report)
    except (GateConfigError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("PASS: production learned-LSTM six-hour gate is reproducible")
    print(f"  config: {report.config_path.relative_to(REPO_ROOT)}")
    print(f"  cases: {report.cases_csv.relative_to(REPO_ROOT)}")
    print(f"  dataset: {report.dataset_dir.relative_to(REPO_ROOT)}")
    print(f"  model: {report.model_dir.relative_to(REPO_ROOT)}")
    print(f"  planner runtime: {report.planner_runtime_config.relative_to(REPO_ROOT)}")
    identity_state = "present and valid" if report.run_identity_present else "required after execution"
    print(f"  run identity: {identity_state}")
    if args.print_command:
        print("  command:")
        print("    " + " ".join(report.command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
