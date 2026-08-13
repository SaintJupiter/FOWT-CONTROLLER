"""Reproducible, staged six-hour controller-validation protocol."""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


CASEBOOK_RUNNER = Path("scripts/analysis/run_prediction_primary_casebook.py")
SMOKE_CHECKER = Path("scripts/analysis/check_control_chain_smoke_gate.py")
SIX_HOURS_S = 21_600.0
DIAGNOSTIC_CASE_COUNTS = (1, 3)
EVALUATION_STAGE_CASE_COUNTS = (10, 20, 30)
STAGE_CASE_COUNTS = (1, 3, 10, 20, 30)
MAX_CASES_PER_EXPERIMENT = 30
REQUIRED_ALGORITHMS = (
    "posture_feedback",
    "no_future_prediction",
    "prediction_assisted",
)

INTEGRITY_LAYER = "integrity"
SAFETY_LAYER = "safety"
PERFORMANCE_LAYER = "performance"


def enforce_development_experiment_bounds(
    *,
    duration_s: float,
    case_count: int,
) -> None:
    """Reject development runs outside the six-hour, 30-case boundary."""

    if float(duration_s) != SIX_HOURS_S:
        raise ValueError(
            "architecture v2 development experiments require "
            f"--duration-s {SIX_HOURS_S:g}; got {float(duration_s):g}"
        )
    if int(case_count) < 1:
        raise ValueError(
            "architecture v2 development experiments require at least one case"
        )
    if int(case_count) > MAX_CASES_PER_EXPERIMENT:
        raise ValueError(
            "architecture v2 development experiments cannot exceed "
            f"{MAX_CASES_PER_EXPERIMENT} cases; got {int(case_count)}"
        )


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _finite(value: float | int | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid case timestamp {value!r}") from exc


@dataclass(frozen=True)
class ValidationIssue:
    """One machine-readable failure or audit warning."""

    severity: str
    code: str
    message: str
    layer: str = INTEGRITY_LAYER


@dataclass(frozen=True)
class CaseTimeInterval:
    """Half-open source-data interval [start, end) for one validation case."""

    case_id: str
    start: datetime
    end: datetime

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CaseTimeInterval":
        case_id = str(raw.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("case interval is missing case_id")
        start = _parse_datetime(str(raw.get("start", "")))
        end = _parse_datetime(str(raw.get("end", "")))
        if end <= start:
            raise ValueError(f"case interval {case_id!r} must have end after start")
        return cls(case_id=case_id, start=start, end=end)

    @property
    def duration_s(self) -> float:
        return float((self.end - self.start).total_seconds())


def find_case_interval_overlaps(
    intervals: Sequence[CaseTimeInterval],
) -> tuple[tuple[str, str], ...]:
    """Return all overlapping case-id pairs; touching endpoints do not overlap."""

    seen: set[str] = set()
    for interval in intervals:
        if interval.case_id in seen:
            raise ValueError(f"duplicate case interval id {interval.case_id!r}")
        seen.add(interval.case_id)

    ordered = sorted(intervals, key=lambda item: (item.start, item.end, item.case_id))
    overlaps: list[tuple[str, str]] = []
    active: list[CaseTimeInterval] = []
    for current in ordered:
        active = [item for item in active if item.end > current.start]
        overlaps.extend((item.case_id, current.case_id) for item in active)
        active.append(current)
    return tuple(overlaps)


def validate_algorithm_physical_configs(
    configs: Mapping[str, Mapping[str, Any]],
    *,
    required_algorithms: Sequence[str] = REQUIRED_ALGORITHMS,
) -> tuple[ValidationIssue, ...]:
    """Require all compared algorithms to use identical physical settings."""

    missing = [name for name in required_algorithms if name not in configs]
    if missing:
        return (
            ValidationIssue(
                "error",
                "physical_config_missing_algorithm",
                f"missing physical configuration for: {', '.join(missing)}",
            ),
        )

    issues: list[ValidationIssue] = []
    reference_name = required_algorithms[0]
    reference = configs[reference_name]
    for name in required_algorithms[1:]:
        candidate = configs[name]
        keys = sorted(set(reference) | set(candidate))
        mismatches = [key for key in keys if reference.get(key) != candidate.get(key)]
        if mismatches:
            issues.append(
                ValidationIssue(
                    "error",
                    "physical_config_mismatch",
                    f"{name} differs from {reference_name} in: "
                    + ", ".join(mismatches),
                )
            )
    return tuple(issues)


@dataclass(frozen=True)
class ParameterFreezeManifest:
    """Audit record separating parameter development from the 30-case check."""

    parameter_set_id: str = ""
    frozen: bool = False
    frozen_after_stage: int | None = None
    tuning_stage_case_counts: tuple[int, ...] = ()
    config_sha256: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ParameterFreezeManifest":
        tuning_raw = raw.get("tuning_stage_case_counts", [])
        if not isinstance(tuning_raw, list):
            raise TypeError("tuning_stage_case_counts must be a list")
        frozen_after_raw = raw.get("frozen_after_stage")
        return cls(
            parameter_set_id=str(raw.get("parameter_set_id", "")).strip(),
            frozen=bool(raw.get("frozen", False)),
            frozen_after_stage=(
                None if frozen_after_raw is None else int(frozen_after_raw)
            ),
            tuning_stage_case_counts=tuple(int(value) for value in tuning_raw),
            config_sha256=str(raw.get("config_sha256", "")).strip(),
        )

    def validate_for_stage(self, stage_case_count: int) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        invalid_stages = sorted(
            count
            for count in self.tuning_stage_case_counts
            if count not in (1, 3, 10, 20)
        )
        if invalid_stages:
            issues.append(
                ValidationIssue(
                    "error",
                    "evaluation_results_used_for_tuning",
                    "only 1/3/10/20-case stages may be used for development: "
                    + ", ".join(str(value) for value in invalid_stages),
                )
            )

        if self.frozen_after_stage in (1, 3, 10, 20):
            post_freeze_tuning = sorted(
                count
                for count in self.tuning_stage_case_counts
                if count > self.frozen_after_stage
            )
            if post_freeze_tuning:
                issues.append(
                    ValidationIssue(
                        "error",
                        "post_freeze_tuning_recorded",
                        "tuning stages occur after frozen_after_stage: "
                        + ", ".join(str(value) for value in post_freeze_tuning),
                    )
                )

        if stage_case_count != 30:
            return tuple(issues)
        if not self.frozen:
            issues.append(
                ValidationIssue(
                    "error",
                    "parameters_not_frozen",
                    "parameters must be frozen before the 30-case stage",
                )
            )
        if self.frozen_after_stage not in (1, 3, 10, 20):
            issues.append(
                ValidationIssue(
                    "error",
                    "invalid_parameter_freeze_stage",
                    "frozen_after_stage must be one of 1, 3, 10, or 20",
                )
            )
        valid_hash = (
            len(self.config_sha256) == 64
            and all(char in "0123456789abcdefABCDEF" for char in self.config_sha256)
        )
        if not self.parameter_set_id or not valid_hash:
            issues.append(
                ValidationIssue(
                    "error",
                    "parameter_identity_missing",
                    "30-case evaluation requires parameter_set_id and a 64-character "
                    "hexadecimal config_sha256",
                )
            )
        return tuple(issues)


@dataclass(frozen=True)
class StageGate:
    case_count: int
    evidence_level: str
    require_performance_metrics: bool = False
    require_positive_ci_lower: bool = False
    require_class_metrics: bool = False


# One- and three-case runs are diagnostics only. Algorithm direction and
# parameter choices are not evaluated until the stratified 10-case stage.
# No stage has a fixed pump-reduction target, and the submitted v1 result is
# not a v2 regression gate.
STAGE_GATES: dict[int, StageGate] = {
    1: StageGate(1, "structural_smoke"),
    3: StageGate(3, "cross_case_diagnostic"),
    10: StageGate(10, "minimum_directional_screen", require_performance_metrics=True),
    20: StageGate(20, "independent_confirmation", require_performance_metrics=True),
    30: StageGate(
        30,
        "bounded_validation_evidence",
        require_performance_metrics=True,
        require_positive_ci_lower=True,
        require_class_metrics=True,
    ),
}


@dataclass(frozen=True)
class StageMetrics:
    """Aggregate metrics for one staged six-hour validation result."""

    case_count: int
    pump_reduction_vs_posture_pct: float | None = None
    pump_reduction_vs_no_future_pct: float | None = None
    improved_case_count: int | None = None
    pump_reduction_ci_lower_pct: float | None = None
    class_pump_reduction_pct: Mapping[str, float] = field(default_factory=dict)
    time_over_2deg_delta_pp: float | None = None
    time_over_3deg_delta_pp: float | None = None
    time_over_5deg_delta_pp: float | None = None
    new_over_7p5deg_case_count: int | None = None
    max_case_over_7p5deg_increase_s: float | None = None
    time_over_10deg_increase_s: float | None = None
    pitch_rms_increase_pct: float | None = None
    roll_rms_increase_pct: float | None = None
    pitch_p95_increase_pct: float | None = None
    roll_p95_increase_pct: float | None = None
    preview_actual_pump_mismatch_pct: float | None = None
    preview_actual_target_mismatch_pct: float | None = None


@dataclass(frozen=True)
class StageEvaluation:
    stage_case_count: int
    evidence_level: str
    issues: tuple[ValidationIssue, ...]
    performance_evaluated: bool

    def _layer_passed(self, layer: str) -> bool:
        return not any(
            issue.severity == "error" and issue.layer == layer
            for issue in self.issues
        )

    @property
    def integrity_passed(self) -> bool:
        return self._layer_passed(INTEGRITY_LAYER)

    @property
    def safety_passed(self) -> bool:
        return self._layer_passed(SAFETY_LAYER)

    @property
    def performance_passed(self) -> bool | None:
        if not self.performance_evaluated:
            return None
        return self._layer_passed(PERFORMANCE_LAYER)

    @property
    def passed(self) -> bool:
        performance_ok = self.performance_passed
        return (
            self.integrity_passed
            and self.safety_passed
            and performance_ok is not False
        )

    @property
    def supports_expanded_validation(self) -> bool | None:
        """Report evidence at 30 cases; never authorize an automatic run."""

        return self.passed if self.stage_case_count == 30 else None

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")


def _require_metric(
    issues: list[ValidationIssue],
    *,
    name: str,
    value: float | int | None,
    layer: str,
) -> bool:
    if _finite(value):
        return True
    issues.append(
        ValidationIssue(
            "error",
            "metric_missing",
            f"required metric {name} is missing",
            layer,
        )
    )
    return False


def _apply_upper_bound(
    issues: list[ValidationIssue],
    *,
    name: str,
    value: float | int | None,
    maximum: float,
    layer: str,
) -> None:
    if not _require_metric(issues, name=name, value=value, layer=layer):
        return
    if float(value) > maximum:
        issues.append(
            ValidationIssue(
                "error",
                "guardrail_exceeded",
                f"{name}={float(value):.3f} exceeds {maximum:.3f}",
                layer,
            )
        )


def evaluate_validation_stage(
    metrics: StageMetrics,
    *,
    parameter_manifest: ParameterFreezeManifest | None = None,
    structural_issues: Sequence[ValidationIssue] = (),
) -> StageEvaluation:
    """Evaluate integrity, safety, and result amplitude as separate layers."""

    if metrics.case_count not in STAGE_GATES:
        raise ValueError(
            f"unsupported validation stage {metrics.case_count}; "
            f"expected one of {STAGE_CASE_COUNTS}"
        )
    gate = STAGE_GATES[metrics.case_count]
    issues = list(structural_issues)

    if parameter_manifest is not None:
        issues.extend(parameter_manifest.validate_for_stage(metrics.case_count))
    elif metrics.case_count == 30:
        issues.append(
            ValidationIssue(
                "error",
                "parameter_manifest_missing",
                "30-case evaluation requires a parameter-freeze manifest",
            )
        )

    # Integrity and safety are hard gates from the first case onward. One case
    # can reveal a violation, but it cannot establish general performance.
    _apply_upper_bound(
        issues,
        name="preview_actual_pump_mismatch_pct",
        value=metrics.preview_actual_pump_mismatch_pct,
        maximum=1.0,
        layer=INTEGRITY_LAYER,
    )
    _apply_upper_bound(
        issues,
        name="preview_actual_target_mismatch_pct",
        value=metrics.preview_actual_target_mismatch_pct,
        maximum=1.0,
        layer=INTEGRITY_LAYER,
    )

    safety_bounds = (
        ("time_over_2deg_delta_pp", metrics.time_over_2deg_delta_pp, 4.5),
        ("time_over_3deg_delta_pp", metrics.time_over_3deg_delta_pp, 1.6),
        ("time_over_5deg_delta_pp", metrics.time_over_5deg_delta_pp, 0.10),
        (
            "max_case_over_7p5deg_increase_s",
            metrics.max_case_over_7p5deg_increase_s,
            60.0,
        ),
        ("time_over_10deg_increase_s", metrics.time_over_10deg_increase_s, 0.0),
        ("pitch_rms_increase_pct", metrics.pitch_rms_increase_pct, 10.0),
        ("roll_rms_increase_pct", metrics.roll_rms_increase_pct, 10.0),
        ("pitch_p95_increase_pct", metrics.pitch_p95_increase_pct, 10.0),
        ("roll_p95_increase_pct", metrics.roll_p95_increase_pct, 10.0),
    )
    for name, value, maximum in safety_bounds:
        _apply_upper_bound(
            issues,
            name=name,
            value=value,
            maximum=maximum,
            layer=SAFETY_LAYER,
        )
    if _require_metric(
        issues,
        name="new_over_7p5deg_case_count",
        value=metrics.new_over_7p5deg_case_count,
        layer=SAFETY_LAYER,
    ) and int(metrics.new_over_7p5deg_case_count) > 0:
        issues.append(
            ValidationIssue(
                "error",
                "new_high_posture_case",
                "prediction-assisted control introduced a new >7.5-degree case",
                SAFETY_LAYER,
            )
        )

    performance_evaluated = gate.require_performance_metrics
    if performance_evaluated:
        for name, value in (
            (
                "pump_reduction_vs_posture_pct",
                metrics.pump_reduction_vs_posture_pct,
            ),
            (
                "pump_reduction_vs_no_future_pct",
                metrics.pump_reduction_vs_no_future_pct,
            ),
        ):
            _require_metric(
                issues, name=name, value=value, layer=PERFORMANCE_LAYER
            )

        if _require_metric(
            issues,
            name="improved_case_count",
            value=metrics.improved_case_count,
            layer=PERFORMANCE_LAYER,
        ):
            improved = int(metrics.improved_case_count)
            if improved < 0 or improved > metrics.case_count:
                issues.append(
                    ValidationIssue(
                        "error",
                        "improved_case_count_invalid",
                        "improved_case_count must be within the stage case count",
                        INTEGRITY_LAYER,
                    )
                )

        if (
            _finite(metrics.pump_reduction_vs_posture_pct)
            and float(metrics.pump_reduction_vs_posture_pct) > 30.0
        ):
            issues.append(
                ValidationIssue(
                    "warning",
                    "pump_reduction_above_audit_threshold",
                    "pump reduction exceeds 30%; audit target lifecycle, deadband, "
                    "execution parity, and posture trade-offs before accepting it",
                    PERFORMANCE_LAYER,
                )
            )

    if gate.require_positive_ci_lower:
        if _require_metric(
            issues,
            name="pump_reduction_ci_lower_pct",
            value=metrics.pump_reduction_ci_lower_pct,
            layer=PERFORMANCE_LAYER,
        ) and float(metrics.pump_reduction_ci_lower_pct) <= 0.0:
            issues.append(
                ValidationIssue(
                    "error",
                    "pump_reduction_ci_not_positive",
                    "pump-reduction confidence-interval lower bound must be positive",
                    PERFORMANCE_LAYER,
                )
            )

    if gate.require_class_metrics:
        if not metrics.class_pump_reduction_pct:
            issues.append(
                ValidationIssue(
                    "error",
                    "class_metrics_missing",
                    "30-case evaluation requires every case class to be reported",
                    PERFORMANCE_LAYER,
                )
            )
        else:
            for name, value in sorted(metrics.class_pump_reduction_pct.items()):
                if not _finite(value) or float(value) < 0.0:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "class_pump_reduction_negative",
                            f"class {name!r} has pump reduction {value!r}; retain "
                            "the result as an applicability-boundary observation",
                            PERFORMANCE_LAYER,
                        )
                    )

    return StageEvaluation(
        stage_case_count=metrics.case_count,
        evidence_level=gate.evidence_level,
        issues=tuple(issues),
        performance_evaluated=performance_evaluated,
    )


@dataclass(frozen=True)
class StagedValidationManifest:
    """Strict-validation extension embedded in a protocol JSON file."""

    stage_case_count: int
    case_intervals: tuple[CaseTimeInterval, ...] = ()
    algorithm_physical_configs: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    parameter_manifest: ParameterFreezeManifest | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StagedValidationManifest":
        stage = int(raw.get("stage_case_count", 0))
        if stage not in STAGE_GATES:
            raise ValueError(
                f"stage_case_count must be one of {STAGE_CASE_COUNTS}, got {stage}"
            )
        intervals_raw = raw.get("case_intervals", [])
        if not isinstance(intervals_raw, list):
            raise TypeError("case_intervals must be a list")
        physical_raw = raw.get("algorithm_physical_configs", {})
        if not isinstance(physical_raw, dict):
            raise TypeError("algorithm_physical_configs must be an object")
        for name, config in physical_raw.items():
            if not isinstance(config, dict):
                raise TypeError(f"physical configuration for {name!r} must be an object")
        parameter_raw = raw.get("parameter_manifest")
        if parameter_raw is not None and not isinstance(parameter_raw, dict):
            raise TypeError("parameter_manifest must be an object")
        return cls(
            stage_case_count=stage,
            case_intervals=tuple(
                CaseTimeInterval.from_mapping(item) for item in intervals_raw
            ),
            algorithm_physical_configs=physical_raw,
            parameter_manifest=(
                None
                if parameter_raw is None
                else ParameterFreezeManifest.from_mapping(parameter_raw)
            ),
        )

    def validate(self, *, expected_cases: Sequence[str]) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        expected = tuple(str(value) for value in expected_cases)
        if len(expected) != len(set(expected)):
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_expected_case_id",
                    "expected_cases contains duplicate case identifiers",
                )
            )
        if len(expected) != self.stage_case_count:
            issues.append(
                ValidationIssue(
                    "error",
                    "stage_case_count_mismatch",
                    f"stage declares {self.stage_case_count} cases but expected_cases "
                    f"contains {len(expected)}",
                )
            )

        interval_ids = tuple(item.case_id for item in self.case_intervals)
        missing = sorted(set(expected) - set(interval_ids))
        extra = sorted(set(interval_ids) - set(expected))
        if missing:
            issues.append(
                ValidationIssue(
                    "error",
                    "case_interval_missing",
                    "missing case intervals for: " + ", ".join(missing),
                )
            )
        if extra:
            issues.append(
                ValidationIssue(
                    "error",
                    "unexpected_case_interval",
                    "unexpected case intervals for: " + ", ".join(extra),
                )
            )
        if len(interval_ids) != len(set(interval_ids)):
            issues.append(
                ValidationIssue(
                    "error",
                    "duplicate_case_interval_id",
                    "case_intervals contains duplicate case identifiers",
                )
            )
        for interval in self.case_intervals:
            if interval.duration_s != SIX_HOURS_S:
                issues.append(
                    ValidationIssue(
                        "error",
                        "case_interval_not_six_hours",
                        f"case {interval.case_id!r} lasts {interval.duration_s:g}s; "
                        "each case must last 21600s",
                    )
                )
        if len(interval_ids) == len(set(interval_ids)):
            for left, right in find_case_interval_overlaps(self.case_intervals):
                issues.append(
                    ValidationIssue(
                        "error",
                        "case_interval_overlap",
                        f"case intervals overlap: {left} and {right}",
                    )
                )

        if not self.algorithm_physical_configs:
            issues.append(
                ValidationIssue(
                    "error",
                    "physical_configs_missing",
                    "staged validation requires physical settings for all algorithms",
                )
            )
        else:
            issues.extend(
                validate_algorithm_physical_configs(self.algorithm_physical_configs)
            )

        if self.parameter_manifest is not None:
            issues.extend(
                self.parameter_manifest.validate_for_stage(self.stage_case_count)
            )
        elif self.stage_case_count == 30:
            issues.append(
                ValidationIssue(
                    "error",
                    "parameter_manifest_missing",
                    "30-case protocol requires a parameter-freeze manifest",
                )
            )
        return tuple(issues)


@dataclass(frozen=True)
class ControlValidationProtocol:
    """One explicit simulation command plus its result-checking contract."""

    source_path: Path
    schema_version: str
    casebook_command: tuple[str, ...]
    expected_cases: tuple[str, ...]
    repo_root: Path
    staged_validation: StagedValidationManifest | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        repo_root: str | Path,
    ) -> "ControlValidationProtocol":
        root = Path(repo_root).resolve()
        source = _resolve(root, path).resolve()
        with source.open("r", encoding="utf-8") as handle:
            raw: Any = json.load(handle)
        if not isinstance(raw, dict):
            raise TypeError(f"{source} must contain a JSON object")

        command_raw = raw.get("casebook_command")
        if not isinstance(command_raw, list) or not command_raw:
            raise ValueError(f"{source} must define a non-empty casebook_command")
        command = tuple(str(item) for item in command_raw)
        if len(command) < 2 or Path(command[1]) != CASEBOOK_RUNNER:
            raise ValueError(
                "validation protocol must use "
                f"{CASEBOOK_RUNNER.as_posix()} as its casebook runner"
            )
        if any("\x00" in item or "\n" in item for item in command):
            raise ValueError("validation command contains an invalid argument")

        expected_raw = raw.get("expected_cases")
        if not isinstance(expected_raw, list) or not expected_raw:
            raise ValueError(f"{source} must define expected_cases")
        staged_raw = raw.get("staged_validation")
        if staged_raw is not None and not isinstance(staged_raw, dict):
            raise TypeError("staged_validation must be an object")

        protocol = cls(
            source_path=source,
            schema_version=str(raw.get("schema_version", "")),
            casebook_command=command,
            expected_cases=tuple(str(item) for item in expected_raw),
            repo_root=root,
            staged_validation=(
                None
                if staged_raw is None
                else StagedValidationManifest.from_mapping(staged_raw)
            ),
        )
        protocol._required_option_value("--out-dir")
        return protocol

    def _required_option_value(self, option: str) -> str:
        try:
            index = self.casebook_command.index(option)
            value = self.casebook_command[index + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"validation protocol is missing required option {option}"
            ) from exc
        if not value or value.startswith("--"):
            raise ValueError(f"validation protocol has no value for {option}")
        return value

    @property
    def summary_csv(self) -> Path:
        out_dir = _resolve(self.repo_root, self._required_option_value("--out-dir"))
        return out_dir / "casebook_summary.csv"

    def manifest_issues(self) -> tuple[ValidationIssue, ...]:
        issues: list[ValidationIssue] = []
        if len(self.expected_cases) > MAX_CASES_PER_EXPERIMENT:
            issues.append(
                ValidationIssue(
                    "error",
                    "experiment_case_limit_exceeded",
                    f"one experiment cannot exceed {MAX_CASES_PER_EXPERIMENT} cases",
                )
            )
        if self.staged_validation is not None:
            issues.extend(
                self.staged_validation.validate(expected_cases=self.expected_cases)
            )
            try:
                duration_s = float(self._required_option_value("--duration-s"))
            except ValueError:
                duration_s = math.nan
            if duration_s != SIX_HOURS_S:
                issues.append(
                    ValidationIssue(
                        "error",
                        "staged_duration_not_six_hours",
                        "staged validation requires --duration-s 21600",
                    )
                )
        return tuple(issues)

    def runner_argv(self, python_executable: str | Path | None = None) -> list[str]:
        boundary_errors = [
            issue for issue in self.manifest_issues() if issue.severity == "error"
        ]
        if boundary_errors:
            raise ValueError(
                "validation protocol cannot run: "
                + "; ".join(issue.message for issue in boundary_errors)
            )
        argv = list(self.casebook_command)
        argv[0] = str(python_executable or sys.executable)
        argv[1] = str(_resolve(self.repo_root, argv[1]))
        return argv

    def checker_argv(self, python_executable: str | Path | None = None) -> list[str]:
        python = str(python_executable or sys.executable)
        return [
            python,
            str(_resolve(self.repo_root, SMOKE_CHECKER)),
            "--config",
            str(self.source_path),
            "--summary-csv",
            str(self.summary_csv),
        ]
