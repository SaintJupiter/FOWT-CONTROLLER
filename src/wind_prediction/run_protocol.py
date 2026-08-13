from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .run_identity import RunIdentity as ContentAddressedRunIdentity
from .run_identity import build_run_identity


@dataclass(frozen=True)
class RunIdentity:
    """Stable identity for one reproducible experiment run."""

    kind: str
    primary_label: str
    primary_control_profile: str
    forecast_source_requested: str
    forecast_source_effective: str
    replay_split: str
    cases_source: str
    primary_only: bool
    reactive_primary_only: bool


@dataclass(frozen=True)
class RunInputs:
    dataset_dir: str
    model_dir: str | None
    stiffness_file: str | None
    planner_runtime_config: str | None
    cases_csv: str | None
    case_ids: list[str]
    duration_s: float
    skip_figures: bool


@dataclass(frozen=True)
class RunConfiguration:
    closed_pump_profile: str
    primary_pump_profile: str
    primary_safety_profile: str
    primary_scale: float
    event_reset_mode: str
    primary_hold_target_mode: str
    planner_envelope_mode: str
    planner_envelope_barrier_active: bool
    planner_posture_hold_barrier_active: bool
    planner_posture_state_residual_active: bool
    planner_posture_state_gain: float
    planner_pressure_norm_cap: float
    forecast_pump_suppression: bool
    relief_medium_cap: bool
    hold_relief_debt: bool
    recovery_mode: bool
    hold_comfort_release: bool


@dataclass(frozen=True)
class RunOutputs:
    out_dir: str
    summary_csv: str
    report_md: str
    protocol_json: str
    timeseries_dir: str
    planner_log_dir: str


@dataclass(frozen=True)
class RunResult:
    completed_cases: int = 0
    issue_count: int = 0
    elapsed_s: float | None = None
    aggregate_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RunProtocol:
    schema_version: str
    created_at_utc: str
    identity: RunIdentity
    inputs: RunInputs
    configuration: RunConfiguration
    outputs: RunOutputs
    result: RunResult
    environment: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _display_path(path: str | Path | None, repo_root: Path) -> str | None:
    if path in ("", None):
        return None
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    try:
        return str(p.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(p.resolve())


def _split_case_ids(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _mean_metric(summary_rows: Any, column: str) -> float | None:
    if summary_rows is None or not hasattr(summary_rows, "__getitem__"):
        return None
    try:
        values = summary_rows[column]
    except Exception:
        return None
    try:
        return float(values.mean())
    except Exception:
        return None


def _sum_metric(summary_rows: Any, column: str) -> float | None:
    if summary_rows is None or not hasattr(summary_rows, "__getitem__"):
        return None
    try:
        values = summary_rows[column]
    except Exception:
        return None
    try:
        return float(values.sum())
    except Exception:
        return None


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def summarize_casebook_metrics(summary_rows: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for out_name, column, reducer in (
        ("primary_pump_work_m3_sum", "primary_pump_work_m3", _sum_metric),
        ("primary_pitch_p95_mean", "primary_pitch_p95", _mean_metric),
        ("primary_roll_p95_mean", "primary_roll_p95", _mean_metric),
        ("primary_time_over_5deg_s_sum", "primary_time_over_5deg_s", _sum_metric),
        ("primary_time_over_7p5deg_s_sum", "primary_time_over_7p5deg_s", _sum_metric),
        ("primary_time_over_10deg_s_sum", "primary_time_over_10deg_s", _sum_metric),
        ("d_pump_work_pct_mean", "d_pump_work_pct", _mean_metric),
        ("d_pitch_p95_mean", "d_pitch_p95", _mean_metric),
        ("d_roll_p95_mean", "d_roll_p95", _mean_metric),
    ):
        value = _finite_or_none(reducer(summary_rows, column))
        if value is not None:
            metrics[out_name] = value
    return metrics


def build_casebook_run_protocol(
    *,
    args: argparse.Namespace,
    cfg: Any,
    repo_root: Path,
    out_dir: Path,
    dataset_dir: Path,
    closed_pump_profile: str,
    primary_pump_profile: str,
    primary_scale: float,
    forecast_source_effective: str,
    completed_cases: int = 0,
    issue_count: int = 0,
    elapsed_s: float | None = None,
    summary_rows: Any = None,
    created_at_utc: str | None = None,
) -> RunProtocol:
    cases_csv = _display_path(getattr(args, "cases_csv", ""), repo_root)
    cases_source = cases_csv or (
        "case_ids" if _split_case_ids(getattr(args, "case_ids", "")) else "built_in"
    )
    primary_label = str(getattr(args, "primary_label", "prediction_primary"))
    out_dir = Path(out_dir)
    planner_envelope_mode = (
        "discounted" if bool(getattr(cfg, "envelope_use_discount", False)) else "raw"
    )
    return RunProtocol(
        schema_version="run_protocol.v1",
        created_at_utc=created_at_utc or _utc_now_iso(),
        identity=RunIdentity(
            kind="prediction_primary_casebook",
            primary_label=primary_label,
            primary_control_profile=str(getattr(args, "primary_control_profile", "")),
            forecast_source_requested=str(getattr(args, "forecast_source", "")),
            forecast_source_effective=str(forecast_source_effective),
            replay_split=str(getattr(args, "replay_split", "")),
            cases_source=str(cases_source),
            primary_only=bool(getattr(args, "primary_only", False)),
            reactive_primary_only=bool(getattr(args, "reactive_primary_only", False)),
        ),
        inputs=RunInputs(
            dataset_dir=str(_display_path(dataset_dir, repo_root)),
            model_dir=_display_path(getattr(args, "model_dir", ""), repo_root),
            stiffness_file=_display_path(
                getattr(args, "stiffness_file", ""),
                repo_root,
            ),
            planner_runtime_config=_display_path(
                getattr(args, "planner_runtime_config", ""),
                repo_root,
            ),
            cases_csv=cases_csv,
            case_ids=_split_case_ids(getattr(args, "case_ids", "")),
            duration_s=float(getattr(args, "duration_s", 0.0)),
            skip_figures=bool(getattr(args, "skip_figures", False)),
        ),
        configuration=RunConfiguration(
            closed_pump_profile=str(closed_pump_profile),
            primary_pump_profile=str(primary_pump_profile),
            primary_safety_profile=str(getattr(args, "primary_safety_profile", "")),
            primary_scale=float(primary_scale),
            event_reset_mode=str(getattr(args, "event_reset_mode", "")),
            primary_hold_target_mode=str(getattr(args, "primary_hold_target_mode", "")),
            planner_envelope_mode=planner_envelope_mode,
            planner_envelope_barrier_active=bool(
                getattr(cfg, "envelope_barrier_active", False)
            ),
            planner_posture_hold_barrier_active=bool(
                getattr(cfg, "posture_hold_barrier_active", False)
            ),
            planner_posture_state_residual_active=bool(
                getattr(cfg, "posture_state_residual_active", False)
            ),
            planner_posture_state_gain=float(getattr(cfg, "posture_state_gain", 0.0)),
            planner_pressure_norm_cap=float(getattr(cfg, "pressure_norm_cap", 0.0)),
            forecast_pump_suppression=bool(
                getattr(args, "pump_suppression", False)
                or getattr(args, "forecast_advised_economy_suppression_only", False)
            ),
            relief_medium_cap=bool(getattr(args, "relief_medium_cap", False)),
            hold_relief_debt=bool(getattr(args, "hold_relief_debt", False)),
            recovery_mode=bool(getattr(args, "recovery_mode", False)),
            hold_comfort_release=bool(getattr(args, "hold_comfort_release", False)),
        ),
        outputs=RunOutputs(
            out_dir=str(_display_path(out_dir, repo_root)),
            summary_csv=str(_display_path(out_dir / "casebook_summary.csv", repo_root)),
            report_md=str(_display_path(out_dir / "casebook_report.md", repo_root)),
            protocol_json=str(_display_path(out_dir / "run_protocol.json", repo_root)),
            timeseries_dir=str(_display_path(out_dir / "timeseries", repo_root)),
            planner_log_dir=str(_display_path(out_dir / "planner_logs", repo_root)),
        ),
        result=RunResult(
            completed_cases=int(completed_cases),
            issue_count=int(issue_count),
            elapsed_s=float(elapsed_s) if elapsed_s is not None else None,
            aggregate_metrics=summarize_casebook_metrics(summary_rows),
        ),
        environment={
            "python_executable": str(Path(os.sys.executable).resolve()),
            "cwd": str(Path.cwd()),
        },
    )


def write_run_protocol(path: str | Path, protocol: RunProtocol | Mapping[str, Any]) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = protocol.to_dict() if isinstance(protocol, RunProtocol) else dict(protocol)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return out_path


_MODEL_IDENTITY_FILES = (
    "lstm_best.pt",
    "lstm_config.json",
    "lstm_event_thresholds.json",
)
_DATASET_IDENTITY_FILES = (
    "metadata.json",
    "scaler_train.json",
)
_CASEBOOK_PATH_ARGUMENTS = {
    "blend_baseline_model_dir",
    "blend_relief_model_dir",
    "bucket_snapshot_dir",
    "cases_csv",
    "dataset_dir",
    "forced_prefix_actions",
    "model_dir",
    "planner_runtime_config",
    "stiffness_file",
}


def _resolve_input_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _json_compatible(value: Any, *, repo_root: Path) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_compatible(asdict(value), repo_root=repo_root)
    if isinstance(value, Path):
        return _display_path(value, repo_root)
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item, repo_root=repo_root)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item, repo_root=repo_root) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _casebook_model_files(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> dict[str, Path]:
    source = str(getattr(args, "forecast_source", ""))
    directories: list[tuple[str, str | Path]] = []
    if source == "learned":
        directories.append(("forecast", getattr(args, "model_dir", "")))
    elif source.startswith("blend_"):
        directories.extend(
            (
                ("baseline", getattr(args, "blend_baseline_model_dir", "")),
                ("relief", getattr(args, "blend_relief_model_dir", "")),
            )
        )

    files: dict[str, Path] = {}
    for prefix, supplied_dir in directories:
        model_dir = _resolve_input_path(supplied_dir, repo_root)
        for filename in _MODEL_IDENTITY_FILES:
            files[f"{prefix}_{filename}"] = model_dir / filename
    return files


def _casebook_effective_config(
    *,
    protocol: RunProtocol,
    args: argparse.Namespace,
    planner_config: Any,
    repo_root: Path,
) -> dict[str, Any]:
    argument_values: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "out_dir":
            continue
        if key in _CASEBOOK_PATH_ARGUMENTS and value not in (None, ""):
            argument_values[key] = _display_path(value, repo_root)
        else:
            argument_values[key] = value
    return {
        "runner": "prediction_primary_casebook",
        "protocol_schema": protocol.schema_version,
        "protocol_identity": asdict(protocol.identity),
        "protocol_configuration": asdict(protocol.configuration),
        "arguments": _json_compatible(argument_values, repo_root=repo_root),
        "planner_config": _json_compatible(planner_config, repo_root=repo_root),
    }


def build_casebook_run_identity(
    *,
    protocol: RunProtocol,
    args: argparse.Namespace,
    planner_config: Any,
    repo_root: str | Path,
    out_dir: str | Path,
    dataset_dir: str | Path,
    forecast_source_effective: str,
    created_at_utc: str | None = None,
    repository_state: Any = None,
    python_version: str | None = None,
    dependency_versions: Mapping[str, str] | None = None,
) -> ContentAddressedRunIdentity:
    """Build the immutable sidecar identity for a casebook protocol.

    Result counters and elapsed time are intentionally excluded. The identity
    changes only when the effective experiment definition or a declared input
    file changes.
    """

    root = Path(repo_root).expanduser().resolve()
    dataset_root = _resolve_input_path(dataset_dir, root)
    planner_runtime = getattr(args, "planner_runtime_config", "")
    cases_csv = getattr(args, "cases_csv", "")
    stiffness_file = getattr(args, "stiffness_file", "")

    additional_groups: dict[str, Mapping[str, str | Path]] = {
        "configuration": {
            "planner_runtime_config": planner_runtime,
        },
    }
    if cases_csv:
        additional_groups["cases"] = {"cases_csv": cases_csv}
    if stiffness_file:
        additional_groups["plant"] = {"stiffness_file": stiffness_file}

    return build_run_identity(
        repo_root=root,
        effective_config=_casebook_effective_config(
            protocol=protocol,
            args=args,
            planner_config=planner_config,
            repo_root=root,
        ),
        model_files=_casebook_model_files(args, repo_root=root),
        dataset_files={
            filename: dataset_root / filename
            for filename in _DATASET_IDENTITY_FILES
        },
        additional_file_groups=additional_groups,
        output_directory=out_dir,
        control_profile=protocol.identity.primary_control_profile,
        forecast_source=protocol.identity.forecast_source_requested,
        forecast_model_version=str(forecast_source_effective),
        created_at_utc=created_at_utc,
        repository_state=repository_state,
        python_version=python_version,
        dependency_versions=dependency_versions,
    )


def write_run_identity(
    path: str | Path,
    identity: ContentAddressedRunIdentity | Mapping[str, Any],
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        identity.to_dict()
        if isinstance(identity, ContentAddressedRunIdentity)
        else dict(identity)
    )
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return out_path
