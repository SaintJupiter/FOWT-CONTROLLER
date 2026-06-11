#!/usr/bin/env python3
"""Static audit for the forecast-assisted control chain.

This script intentionally runs no simulations. It is a fail-fast architecture
check for provider size, profile sprawl, redundant overlay references, and
control-contract coverage.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "production_controller_v1.json"


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str
    status: str
    summary: str
    evidence: dict[str, Any]
    recommendation: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_count(path: Path) -> int:
    return len(_read_text(path).splitlines())


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _floatish(raw: Any, default: float = math.nan) -> float:
    try:
        if raw in ("", None):
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _resolve_repo_path(raw: str | None) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _class_init_param_count(path: Path, class_name: str) -> int:
    tree = ast.parse(_read_text(path), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = list(item.args.posonlyargs) + list(item.args.args) + list(item.args.kwonlyargs)
                    return max(0, len(args) - 1)
    return 0


def _dataclass_fields(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(_read_text(path), filename=str(path))
    fields: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fields.add(item.target.id)
        break
    return fields


def _token_counts(text: str, tokens: list[str]) -> dict[str, int]:
    return {token: len(re.findall(rf"\b{re.escape(token)}\b", text)) for token in tokens}


def _first_line_number(text: str, needle: str) -> int | None:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return None


def _registry_tokens(registry: dict[str, Any], group: str) -> list[str]:
    tokens: list[str] = []
    for item in registry.get(group, []):
        if isinstance(item, dict):
            tokens.extend(str(token) for token in item.get("tokens", []))
    return sorted(set(tokens))


def _registry_ids(registry: dict[str, Any], group: str) -> list[str]:
    ids: list[str] = []
    for item in registry.get(group, []):
        if isinstance(item, dict) and "id" in item:
            ids.append(str(item["id"]))
    return sorted(ids)


def _literal_strings(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Dict):
        out: set[str] = set()
        for item in [*node.keys, *node.values]:
            if item is not None:
                out |= _literal_strings(item)
        return out
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: set[str] = set()
        for item in node.elts:
            out |= _literal_strings(item)
        return out
    return set()


def _primary_control_choices(path: Path) -> set[str]:
    tree = ast.parse(_read_text(path), filename=str(path))
    choices: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        arg_names = _literal_strings(ast.Tuple(elts=list(node.args), ctx=ast.Load()))
        if "--primary-control-profile" not in arg_names:
            continue
        for keyword in node.keywords:
            if keyword.arg == "choices":
                choices |= _literal_strings(keyword.value)
    return choices


def _assigned_literal_strings(path: Path, names: set[str]) -> set[str]:
    tree = ast.parse(_read_text(path), filename=str(path))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            target_names = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            }
            if target_names & names:
                values |= _literal_strings(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names:
                values |= _literal_strings(node.value)
    return values


def _literal_add_argument_flags(path: Path) -> set[str]:
    tree = ast.parse(_read_text(path), filename=str(path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for value in _literal_strings(ast.Tuple(elts=list(node.args), ctx=ast.Load())):
            if value.startswith("--"):
                flags.add(value)
    return flags


def _primary_control_branch_names(path: Path) -> set[str]:
    tree = ast.parse(_read_text(path), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        has_primary_name = any(
            isinstance(item, ast.Name) and item.id == "primary_control_profile"
            for item in operands
        )
        if not has_primary_name:
            continue
        for item in operands:
            names |= _literal_strings(item)
    return names


def _severity_for_count(count: int, warn: int, critical: int | None = None) -> str:
    if critical is not None and count >= critical:
        return "critical"
    if count >= warn:
        return "high"
    return "ok"


def build_findings(config: dict[str, Any]) -> list[Finding]:
    provider_path = REPO_ROOT / "src" / "wind_prediction" / "ballast_planner_provider.py"
    planner_path = REPO_ROOT / "src" / "wind_prediction" / "ballast_planner.py"
    casebook_path = REPO_ROOT / "scripts" / "analysis" / "run_prediction_primary_casebook.py"
    profile_gate_path = REPO_ROOT / "src" / "wind_prediction" / "casebook_profile_gate.py"
    contracts_path = REPO_ROOT / "src" / "wind_prediction" / "control_contracts.py"
    forecast_contract_path = REPO_ROOT / "src" / "wind_prediction" / "forecast_contract.py"
    target_lifecycle_path = REPO_ROOT / "src" / "wind_prediction" / "target_lifecycle.py"
    safety_supervisor_path = REPO_ROOT / "src" / "wind_prediction" / "safety_supervisor.py"

    thresholds = config.get("static_audit_thresholds", {})
    provider_warn = int(thresholds.get("provider_line_count_warn", 8000))
    provider_param_warn = int(thresholds.get("provider_init_param_count_warn", 120))
    profile_warn = int(thresholds.get("casebook_profile_count_warn", 20))
    redundant_warn = int(thresholds.get("redundant_token_occurrences_warn", 100))

    findings: list[Finding] = []

    provider_lines = _line_count(provider_path)
    findings.append(
        Finding(
            id="provider_size",
            severity=_severity_for_count(provider_lines, provider_warn, critical=12000),
            status="confirmed",
            summary="BallastPlannerPreviewProvider remains too large for reliable control reasoning.",
            evidence={"file": str(provider_path.relative_to(REPO_ROOT)), "line_count": provider_lines},
            recommendation="Extract telemetry, target lifecycle, safety supervisor, and forecast trust before adding new signal logic.",
        )
    )

    init_params = _class_init_param_count(provider_path, "BallastPlannerPreviewProvider")
    findings.append(
        Finding(
            id="provider_init_params",
            severity=_severity_for_count(init_params, provider_param_warn, critical=250),
            status="confirmed",
            summary="Provider constructor exposes too many control and diagnostic switches.",
            evidence={"file": str(provider_path.relative_to(REPO_ROOT)), "init_param_count": init_params},
            recommendation="Move profile identity to config and extract grouped option objects before behavior changes.",
        )
    )

    forbidden = list(config.get("forbidden_in_production", []))
    provider_text = _read_text(provider_path)
    planner_text = _read_text(planner_path)
    casebook_text = _read_text(casebook_path)
    profile_gate_text = _read_text(profile_gate_path) if profile_gate_path.exists() else ""
    target_lifecycle_text = (
        _read_text(target_lifecycle_path) if target_lifecycle_path.exists() else ""
    )
    safety_supervisor_text = (
        _read_text(safety_supervisor_path) if safety_supervisor_path.exists() else ""
    )
    all_runtime_text = provider_text + "\n" + planner_text + "\n" + casebook_text
    counts = _token_counts(all_runtime_text, forbidden)
    total_redundant_refs = sum(counts.values())
    findings.append(
        Finding(
            id="forbidden_production_tokens",
            severity=_severity_for_count(total_redundant_refs, redundant_warn, critical=500),
            status="confirmed",
            summary="Forbidden or diagnostic mechanisms still occupy a large runtime and CLI surface.",
            evidence={
                "total_occurrences": total_redundant_refs,
                "top_tokens": dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)[:12]),
            },
            recommendation="Prune stale profile paths first, then remove redundant default-off mechanisms behind smoke gates.",
        )
    )

    overlay_registry_path = _resolve_repo_path(config.get("overlay_freeze_registry"))
    overlay_registry: dict[str, Any] = {}
    if overlay_registry_path is not None and overlay_registry_path.exists():
        overlay_registry = _load_json(overlay_registry_path)
    redundant_tokens = _registry_tokens(overlay_registry, "redundant_freeze_delete_candidate")
    diagnostic_tokens = _registry_tokens(overlay_registry, "diagnostic_keep_default_off")
    candidate_tokens = _registry_tokens(overlay_registry, "candidate_recheck")
    production_tokens = _registry_tokens(overlay_registry, "production_keep")
    redundant_counts = _token_counts(all_runtime_text, redundant_tokens)
    diagnostic_counts = _token_counts(all_runtime_text, diagnostic_tokens)
    candidate_counts = _token_counts(all_runtime_text, candidate_tokens)
    production_counts = _token_counts(all_runtime_text, production_tokens)
    findings.append(
        Finding(
            id="overlay_freeze_registry",
            severity="high" if sum(redundant_counts.values()) else "ok",
            status="needs_pruning" if sum(redundant_counts.values()) else "clean",
            summary="Default-off overlays are now classified before runtime deletion.",
            evidence={
                "registry": str(overlay_registry_path.relative_to(REPO_ROOT))
                if overlay_registry_path
                else None,
                "production_keep_count": len(_registry_ids(overlay_registry, "production_keep")),
                "diagnostic_keep_count": len(_registry_ids(overlay_registry, "diagnostic_keep_default_off")),
                "candidate_recheck_count": len(_registry_ids(overlay_registry, "candidate_recheck")),
                "redundant_delete_candidate_count": len(
                    _registry_ids(overlay_registry, "redundant_freeze_delete_candidate")
                ),
                "production_token_occurrences": sum(production_counts.values()),
                "diagnostic_token_occurrences": sum(diagnostic_counts.values()),
                "candidate_token_occurrences": sum(candidate_counts.values()),
                "redundant_token_occurrences": sum(redundant_counts.values()),
                "top_redundant_tokens": dict(
                    sorted(redundant_counts.items(), key=lambda item: item[1], reverse=True)[:15]
                ),
            },
            recommendation="Delete redundant groups only after production guard passes and the three-case smoke gate is defined.",
        )
    )

    telemetry_defaults_path = REPO_ROOT / "src" / "wind_prediction" / "control_telemetry_defaults.py"
    telemetry_defaults_text = (
        _read_text(telemetry_defaults_path) if telemetry_defaults_path.exists() else ""
    )
    pump_runtime_state_tokens = [
        "self._pump_suppression_",
        "self.pump_suppression_",
        "def _suppression_decision",
    ]
    pump_runtime_residue = [
        token for token in pump_runtime_state_tokens if token in provider_text
    ]
    pump_compatibility_helpers = [
        "frozen_pump_suppression_record_defaults",
        "frozen_pump_suppression_preview_defaults",
    ]
    missing_pump_compatibility_helpers = [
        name for name in pump_compatibility_helpers if name not in telemetry_defaults_text
    ]
    pump_runtime_state_removed = (
        not pump_runtime_residue and not missing_pump_compatibility_helpers
    )
    findings.append(
        Finding(
            id="pump_suppression_runtime_state_removed",
            severity="ok" if pump_runtime_state_removed else "high",
            status="passed" if pump_runtime_state_removed else "failed",
            summary="Frozen pump_suppression telemetry must not be backed by provider runtime state.",
            evidence={
                "provider_file": str(provider_path.relative_to(REPO_ROOT)),
                "telemetry_defaults_file": str(
                    telemetry_defaults_path.relative_to(REPO_ROOT)
                ),
                "provider_runtime_residue": pump_runtime_residue,
                "missing_compatibility_helpers": missing_pump_compatibility_helpers,
            },
            recommendation="Keep pump_suppression compatibility columns in control_telemetry_defaults.py; do not reintroduce provider-side fake state.",
        )
    )

    target_default_fields = {
        "_primary_anchor_masses_kg",
        "_primary_delta_kg",
        "_primary_target_kg",
        "_primary_target_initialized",
        "_primary_target_refreshed",
        "_primary_target_reused",
        "_primary_target_resumed",
        "_primary_target_last_reset_s",
        "_primary_refresh_owner",
        "_primary_refresh_owner_pending",
        "_paused_primary_target_kg",
        "_paused_primary_delta_kg",
        "_paused_primary_avec",
        "_paused_primary_action",
        "_paused_primary_valid",
    }
    missing_target_default_fields = sorted(
        name for name in target_default_fields if name not in target_lifecycle_text
    )
    target_helper_call_count = provider_text.count(
        "apply_primary_target_state_defaults("
    )
    old_default_block_residue = (
        '_primary_anchor_masses_kg = np.asarray(\n            self._default_plant_info["tank_masses"]'
        in provider_text
    )
    direct_target_mass_reads = provider_text.count(
        'plant_info.get("tank_masses", self._primary_anchor_masses_kg)'
    )
    target_lifecycle_defaults_extracted = (
        target_lifecycle_path.exists()
        and "def apply_primary_target_state_defaults" in target_lifecycle_text
        and "def plant_primary_masses" in target_lifecycle_text
        and target_helper_call_count >= 2
        and not old_default_block_residue
        and direct_target_mass_reads == 0
        and not missing_target_default_fields
    )
    findings.append(
        Finding(
            id="target_lifecycle_defaults_extracted",
            severity="ok" if target_lifecycle_defaults_extracted else "medium",
            status="passed" if target_lifecycle_defaults_extracted else "failed",
            summary="Primary target default state should be centralized outside the provider.",
            evidence={
                "provider_file": str(provider_path.relative_to(REPO_ROOT)),
                "target_lifecycle_file": str(
                    target_lifecycle_path.relative_to(REPO_ROOT)
                ),
                "helper_call_count": target_helper_call_count,
                "old_default_block_residue": old_default_block_residue,
                "direct_target_mass_reads": direct_target_mass_reads,
                "missing_default_fields": missing_target_default_fields,
            },
            recommendation="Keep initialization and reset on the target lifecycle helper before extracting pause/resume/release behavior.",
        )
    )

    reactive_floor_default_fields = {
        "_reactive_floor_latched",
        "_reactive_floor_enter_start_s",
        "_reactive_floor_active",
        "_reactive_floor_veto_active",
        "_reactive_floor_reason",
        "_reactive_floor_veto_reason",
        "_reactive_floor_posture_metric_deg",
        "_reactive_floor_max_axis_deg",
        "_reactive_floor_theta_total_deg",
        "_reactive_floor_current_response_deg",
        "_reactive_floor_target_stale",
        "_reactive_floor_target_err_mean_kg",
        "_reactive_floor_target_age_s",
        "_reactive_floor_pump_idle",
        "_reactive_floor_elapsed_s",
        "_reactive_floor_resolved_action",
        "_reactive_floor_medium_delay_active",
        "_reactive_floor_forecast_relief_clear",
        "_reactive_floor_short_risk_low",
        "_reactive_floor_delta_mean_kg",
        "_reactive_floor_count",
        "_reactive_floor_veto_count",
        "_reactive_floor_trigger_class",
        "_reactive_floor_theta_only_elapsed_s",
        "_reactive_floor_theta_only_gate_pass",
        "_reactive_floor_theta_only_gate_reason",
        "_reactive_floor_last_refresh_s",
        "_reactive_floor_post_exit_active",
        "_reactive_floor_post_exit_reason",
        "_reactive_floor_post_exit_released",
        "_reactive_floor_post_exit_release_count",
        "_reactive_floor_post_exit_cap_count",
        "_reactive_floor_post_exit_delta_before_kg",
        "_reactive_floor_post_exit_delta_after_kg",
        "_reactive_floor_post_exit_posture_metric_deg",
        "_reactive_floor_post_exit_current_response_deg",
        "_reactive_floor_post_exit_episode_active",
        "_reactive_floor_post_exit_prev_metric_deg",
    }
    missing_reactive_floor_default_fields = sorted(
        name for name in reactive_floor_default_fields if name not in safety_supervisor_text
    )
    reactive_floor_default_helper_calls = provider_text.count(
        "apply_reactive_floor_state_defaults("
    )
    reactive_floor_posture_metrics_calls = provider_text.count(
        "reactive_floor_posture_metrics("
    )
    provider_theta_total_method_residue = "def _theta_total_deg" in provider_text
    safety_supervisor_first_slice = (
        safety_supervisor_path.exists()
        and "def apply_reactive_floor_state_defaults" in safety_supervisor_text
        and "def reactive_floor_posture_metrics" in safety_supervisor_text
        and "def posture_vector_deg" in safety_supervisor_text
        and reactive_floor_default_helper_calls >= 2
        and reactive_floor_posture_metrics_calls >= 2
        and not provider_theta_total_method_residue
        and not missing_reactive_floor_default_fields
    )
    findings.append(
        Finding(
            id="safety_supervisor_first_slice",
            severity="ok" if safety_supervisor_first_slice else "medium",
            status="passed" if safety_supervisor_first_slice else "failed",
            summary="Reactive-floor safety state and posture metrics should be centralized before deeper supervisor extraction.",
            evidence={
                "provider_file": str(provider_path.relative_to(REPO_ROOT)),
                "safety_supervisor_file": str(
                    safety_supervisor_path.relative_to(REPO_ROOT)
                ),
                "default_helper_call_count": reactive_floor_default_helper_calls,
                "posture_metrics_call_count": reactive_floor_posture_metrics_calls,
                "provider_theta_total_method_residue": provider_theta_total_method_residue,
                "missing_default_fields": missing_reactive_floor_default_fields,
            },
            recommendation="Keep reactive-floor state defaults and posture metrics in safety_supervisor.py, then move veto/action decisions behind the same module.",
        )
    )

    pruned_overlay_guard_names = {
        "PRUNED_OVERLAY_FLAG_GUARDS",
        "PRUNED_OVERLAY_VALUE_GUARDS",
        "PRUNED_OVERLAY_TEXT_GUARDS",
        "PRUNED_OVERLAY_BOOLEAN_OPTIONAL_GUARDS",
    }
    frozen_cli_flags = sorted(
        value
        for value in _assigned_literal_strings(casebook_path, pruned_overlay_guard_names)
        if value.startswith("--")
    )
    literal_add_argument_flags = _literal_add_argument_flags(casebook_path)
    directly_exposed_frozen_flags = sorted(
        set(frozen_cli_flags) & literal_add_argument_flags
    )
    frozen_cli_helper_present = (
        "_register_pruned_overlay_legacy_args(parser)" in casebook_text
        and "_enforce_pruned_overlay_flags(args)" in casebook_text
        and "help=argparse.SUPPRESS" in casebook_text
    )
    frozen_cli_hidden = bool(frozen_cli_flags) and frozen_cli_helper_present and not directly_exposed_frozen_flags
    findings.append(
        Finding(
            id="frozen_legacy_cli_hidden",
            severity="ok" if frozen_cli_hidden else "high",
            status="passed" if frozen_cli_hidden else "failed",
            summary="Frozen legacy overlay CLI flags should remain hidden compatibility inputs with fail-fast enforcement.",
            evidence={
                "casebook_file": str(casebook_path.relative_to(REPO_ROOT)),
                "frozen_flag_count": len(frozen_cli_flags),
                "helper_present": frozen_cli_helper_present,
                "directly_exposed_frozen_flags": directly_exposed_frozen_flags[:50],
            },
            recommendation="Keep frozen overlay flags in the central hidden-argument registry; do not re-add explicit argparse blocks.",
        )
    )

    smoke_gate_path = _resolve_repo_path(config.get("control_chain_smoke_gate"))
    smoke_gate: dict[str, Any] = {}
    smoke_failures: list[str] = []
    smoke_reference_rows = 0
    if smoke_gate_path is None or not smoke_gate_path.exists():
        smoke_failures.append("smoke_gate_config_missing")
    else:
        smoke_gate = _load_json(smoke_gate_path)
        expected_cases = [str(case) for case in smoke_gate.get("expected_cases", [])]
        reference_raw = smoke_gate.get("reference_summary_csv")
        if not reference_raw:
            smoke_failures.append("reference_summary_csv_missing")
        else:
            reference_path = _resolve_repo_path(str(reference_raw))
            if reference_path is None or not reference_path.exists():
                smoke_failures.append("reference_summary_csv_not_found")
            else:
                ref_rows = _read_csv_rows(reference_path)
                smoke_reference_rows = len(ref_rows)
                ref_case_ids = {str(row.get("case_id", "")) for row in ref_rows}
                missing_cases = sorted(set(expected_cases) - ref_case_ids)
                if missing_cases:
                    smoke_failures.append(f"reference_missing_cases={missing_cases}")
                for row in ref_rows:
                    case_id = str(row.get("case_id", ""))
                    if expected_cases and case_id not in set(expected_cases):
                        continue
                    for column in smoke_gate.get("required_zero_ratio_columns", []):
                        if abs(_floatish(row.get(str(column), "0"), 0.0)) > 1e-9:
                            smoke_failures.append(f"{case_id}:{column}_not_zero")
                    case_thresholds = smoke_gate.get("case_thresholds", {}).get(case_id, {})
                    for key, limit in case_thresholds.items():
                        if not str(key).endswith("_max"):
                            continue
                        column = str(key)[:-4]
                        value = _floatish(row.get(column))
                        if not math.isfinite(value) or value > float(limit):
                            smoke_failures.append(f"{case_id}:{column}_over_max")
    findings.append(
        Finding(
            id="control_chain_smoke_gate",
            severity="critical" if smoke_failures else "ok",
            status="failed" if smoke_failures else "passed",
            summary="Three-case smoke gate must pass before behavior-affecting overlay deletion.",
            evidence={
                "config": str(smoke_gate_path.relative_to(REPO_ROOT)) if smoke_gate_path else None,
                "expected_cases": smoke_gate.get("expected_cases", []),
                "reference_summary_csv": smoke_gate.get("reference_summary_csv"),
                "reference_rows": smoke_reference_rows,
                "failures": smoke_failures,
            },
            recommendation="Run check_control_chain_smoke_gate.py on any new deletion smoke output before broad casebooks.",
        )
    )

    production = str(config.get("production_profile", {}).get("primary_control_profile", ""))
    diagnostics = set(config.get("diagnostic_profiles_allowed", []))
    registry_path = _resolve_repo_path(config.get("casebook_profile_registry"))
    registry: dict[str, Any] = {}
    if registry_path is not None and registry_path.exists():
        registry = _load_json(registry_path)
    registry_groups = {
        "production": set(registry.get("production", [])),
        "diagnostic": set(registry.get("diagnostic", [])),
        "historical_diagnostic": set(registry.get("historical_diagnostic", [])),
        "isolated_experiment": set(registry.get("isolated_experiment", [])),
        "prune_candidate": set(registry.get("prune_candidate", [])),
    }
    known_registered = set().union(*registry_groups.values()) if registry_groups else set()
    known_allowed = {production, *diagnostics}
    choice_profiles = _primary_control_choices(casebook_path)
    branch_profiles = _primary_control_branch_names(casebook_path)
    profiles = sorted(choice_profiles | branch_profiles | known_registered)
    registry_gated_entry = (
        "--primary-control-profile" in casebook_text
        and 'metavar="PROFILE"' in casebook_text
        and not choice_profiles
    )
    runtime_or_choice_profiles = choice_profiles | branch_profiles
    unknown_profiles = sorted(
        name
        for name in runtime_or_choice_profiles
        if name not in known_registered and name not in known_allowed
    )
    registered_without_branch = sorted(known_registered - branch_profiles)
    branch_without_registry = sorted(branch_profiles - known_registered)
    unregistered_severity = "high" if unknown_profiles else "ok"
    sprawl_severity = _severity_for_count(len(profiles), profile_warn, critical=60)
    findings.append(
        Finding(
            id="casebook_profile_sprawl",
            severity=sprawl_severity,
            status="confirmed",
            summary="Casebook script still contains many profile names beyond the canonical production profile.",
            evidence={
                "file": str(casebook_path.relative_to(REPO_ROOT)),
                "profile_count": len(profiles),
                "choice_profile_count": len(choice_profiles),
                "branch_profile_count": len(branch_profiles),
                "registry_gated_entry": registry_gated_entry,
                "production_profile": production,
                "diagnostic_profile_count": len(diagnostics),
                "registry": str(registry_path.relative_to(REPO_ROOT)) if registry_path else None,
                "registered_counts": {
                    key: len(value) for key, value in registry_groups.items()
                },
                "registered_without_runtime_branch_count": len(registered_without_branch),
                "sample_registered_without_runtime_branch": registered_without_branch[:20],
                "runtime_branch_without_registry_count": len(branch_without_registry),
                "sample_runtime_branch_without_registry": branch_without_registry[:20],
                "unknown_or_experimental_profile_count": len(unknown_profiles),
                "sample_unknown_or_experimental_profiles": unknown_profiles[:20],
            },
            recommendation="Keep the registry-gated CLI entry; move isolated runtime branches to isolated configs or delete failed branches.",
        )
    )
    findings.append(
        Finding(
            id="casebook_profile_registry",
            severity=unregistered_severity,
            status="covered" if not unknown_profiles else "incomplete",
            summary="Casebook profile names should be registered before pruning or promotion.",
            evidence={
                "profile_count": len(profiles),
                "registered_profile_count": len(known_registered),
                "unregistered_profile_count": len(unknown_profiles),
                "unregistered_profiles": unknown_profiles[:50],
                "argparse_choice_profile_count": len(choice_profiles),
                "runtime_branch_profile_count": len(branch_profiles),
                "registry_gated_entry": registry_gated_entry,
                "registered_without_runtime_branch_count": len(registered_without_branch),
                "sample_registered_without_runtime_branch": registered_without_branch[:50],
            },
            recommendation="Add any missing runtime branch profile to the registry; keep profile names out of argparse choices.",
        )
    )

    production_profile = config.get("production_profile", {})
    if not isinstance(production_profile, dict):
        production_profile = {}
    frozen_tokens = set(forbidden) | set(redundant_tokens)
    forbidden_enabled = [
        name
        for name in sorted(frozen_tokens)
        if bool(production_profile.get(name, False))
    ]
    registry_rules = registry.get("rules", {}) if registry else {}
    production_count_max = int(registry_rules.get("production_count_max", 1))
    diagnostic_count_max = int(registry_rules.get("diagnostic_count_max", len(diagnostics)))
    production_registry = registry_groups.get("production", set())
    diagnostic_registry = registry_groups.get("diagnostic", set())
    production_config_issues: list[str] = []
    if production not in production_registry:
        production_config_issues.append("production_profile_not_registered")
    if len(production_registry) > production_count_max:
        production_config_issues.append("too_many_production_profiles")
    if len(diagnostic_registry) > diagnostic_count_max:
        production_config_issues.append("too_many_diagnostic_profiles")
    if forbidden_enabled:
        production_config_issues.append("forbidden_switch_enabled")
    findings.append(
        Finding(
            id="production_profile_guard",
            severity="critical" if production_config_issues else "ok",
            status="failed" if production_config_issues else "passed",
            summary="Canonical production config must not enable forbidden overlays or drift from the registry.",
            evidence={
                "production_profile": production,
                "production_registry_count": len(production_registry),
                "diagnostic_registry_count": len(diagnostic_registry),
                "issues": production_config_issues,
                "forbidden_enabled": forbidden_enabled,
            },
            recommendation="Keep production_controller_v1 as the only promotion surface; move experiments into isolated configs.",
        )
    )

    isolated_profiles = sorted(registry_groups.get("isolated_experiment", set()))
    prune_candidates = sorted(registry_groups.get("prune_candidate", set()))
    isolated_gate_present = (
        "--allow-isolated-profile" in casebook_text
        and "enforce_casebook_profile_gate" in casebook_text
        and "enforce_casebook_profile_gate" in profile_gate_text
        and "ISOLATED_PROFILE_UNLOCK_ENV" in profile_gate_text
        and "FOWT_ALLOW_ISOLATED_PROFILE" in profile_gate_text
    )
    findings.append(
        Finding(
            id="profile_pruning_queue",
            severity="medium" if isolated_profiles and isolated_gate_present else (
                "high" if isolated_profiles else "ok"
            ),
            status=(
                "isolated_gate_active"
                if isolated_profiles and isolated_gate_present
                else ("needs_triage" if isolated_profiles else "empty")
            ),
            summary="Most casebook profiles are isolated experiments and should not remain in the production casebook surface.",
            evidence={
                "isolated_experiment_count": len(isolated_profiles),
                "prune_candidate_count": len(prune_candidates),
                "isolated_profile_gate_present": isolated_gate_present,
                "sample_isolated_experiments": isolated_profiles[:20],
                "prune_candidates": prune_candidates[:50],
            },
            recommendation="Keep isolated profiles locked behind --allow-isolated-profile; attach failed/no-go evidence and move safe deletions into prune_candidate before deleting branches.",
        )
    )

    dry_config_line = _first_line_number(casebook_text, "if bool(args.dry_config):")
    mkdir_line = _first_line_number(casebook_text, "d.mkdir(parents=True, exist_ok=True)")
    dry_config_no_output_side_effect = (
        dry_config_line is not None
        and mkdir_line is not None
        and dry_config_line < mkdir_line
    )
    findings.append(
        Finding(
            id="dry_config_no_output_side_effect",
            severity="ok" if dry_config_no_output_side_effect else "medium",
            status="passed" if dry_config_no_output_side_effect else "failed",
            summary="Dry configuration checks should not create casebook output directories.",
            evidence={
                "file": str(casebook_path.relative_to(REPO_ROOT)),
                "dry_config_line": dry_config_line,
                "mkdir_line": mkdir_line,
                "dry_config_returns_before_mkdir": dry_config_no_output_side_effect,
            },
            recommendation="Keep --dry-config before output directory creation so short checks do not add experimental data.",
        )
    )

    required_contracts = config.get("required_control_contracts", {})
    contract_classes = {
        "forecast_signal": "ForecastSignal",
        "supervisor_decision": "SupervisorDecision",
        "control_command": "ControlCommand",
    }
    missing_by_contract: dict[str, list[str]] = {}
    for key, class_name in contract_classes.items():
        fields = _dataclass_fields(contracts_path, class_name)
        required = set(required_contracts.get(key, []))
        missing_by_contract[class_name] = sorted(required - fields)
    has_missing = any(missing_by_contract.values())
    findings.append(
        Finding(
            id="control_contracts",
            severity="high" if has_missing else "ok",
            status="present" if not has_missing else "incomplete",
            summary="Control-facing contracts exist as behavior-neutral target interfaces.",
            evidence={
                "file": str(contracts_path.relative_to(REPO_ROOT)),
                "missing_required_fields": missing_by_contract,
            },
            recommendation="Wire these contracts only after production config and stale-profile pruning are stable.",
        )
    )

    forecast_contract_fields = _dataclass_fields(forecast_contract_path, "ForecastContract")
    control_semantic_fields = {
        "horizon_minutes",
        "staleness_s",
        "confidence",
        "allowed_effects",
        "fail_closed_action",
    }
    missing_forecast_semantics = sorted(control_semantic_fields - forecast_contract_fields)
    findings.append(
        Finding(
            id="forecast_contract_v1_scope",
            severity="medium" if missing_forecast_semantics else "ok",
            status="known_gap" if missing_forecast_semantics else "covered",
            summary="Existing ForecastContract remains a numeric ML-output contract, not a control-authority contract.",
            evidence={
                "file": str(forecast_contract_path.relative_to(REPO_ROOT)),
                "missing_control_semantics": missing_forecast_semantics,
            },
            recommendation="Keep ForecastContract for numeric validation; use ForecastSignal for controller-facing authority.",
        )
    )

    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Exit nonzero if any finding is critical.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = _load_json(config_path)
    findings = build_findings(config)

    payload = {
        "ok": True,
        "config": str(config_path.relative_to(REPO_ROOT)),
        "finding_count": len(findings),
        "findings": [asdict(finding) for finding in findings],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for finding in findings:
            print(
                f"[{finding.severity.upper()}] {finding.id}: {finding.summary} "
                f"evidence={finding.evidence}"
            )
            print(f"  recommendation: {finding.recommendation}")

    if args.fail_on_critical and any(f.severity == "critical" for f in findings):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
