"""Protocol objects for staged, case-paired preview-MPC parameter studies.

This module does not search for parameters.  It freezes how cases, candidate
designs and observations must be identified before expensive simulations are
run, so a result cannot be selected from one favourable case after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from typing import Any, Mapping

import numpy as np

from wind_prediction.preview_mpc_design import PreviewMPCDesign


_STAGE_ROLES = {"smoke", "screening", "confirmation"}


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonempty(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _finite_nonnegative(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _sha256_digest(name: str, value: Any) -> str:
    digest = _nonempty(name, value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a SHA256 hex digest")
    return digest


@dataclass(frozen=True)
class ParameterStudyCase:
    case_id: str
    origin_time: str
    source_event_id: str
    stratum: str
    duration_h: float
    selection_basis: str

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "origin_time",
            "source_event_id",
            "stratum",
            "selection_basis",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        try:
            datetime.fromisoformat(self.origin_time)
        except ValueError as exc:
            raise ValueError("origin_time must be an ISO-format timestamp") from exc
        duration = float(self.duration_h)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_h must be finite and positive")
        object.__setattr__(self, "duration_h", duration)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "origin_time": self.origin_time,
            "source_event_id": self.source_event_id,
            "stratum": self.stratum,
            "duration_h": self.duration_h,
            "selection_basis": self.selection_basis,
        }


@dataclass(frozen=True)
class ParameterStudyStage:
    name: str
    role: str
    cases: tuple[ParameterStudyCase, ...]
    minimum_case_count: int
    minimum_stratum_count: int
    minimum_cases_per_stratum: int
    permits_parameter_selection: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty("name", self.name))
        role = _nonempty("role", self.role)
        if role not in _STAGE_ROLES:
            raise ValueError(f"role must be one of {sorted(_STAGE_ROLES)}")
        object.__setattr__(self, "role", role)
        cases = tuple(self.cases)
        if not all(isinstance(case, ParameterStudyCase) for case in cases):
            raise TypeError("cases must contain ParameterStudyCase values")
        object.__setattr__(self, "cases", cases)
        for name in (
            "minimum_case_count",
            "minimum_stratum_count",
            "minimum_cases_per_stratum",
        ):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if type(self.permits_parameter_selection) is not bool:
            raise ValueError("permits_parameter_selection must be a boolean")
        if role != "screening" and self.permits_parameter_selection:
            raise ValueError("only the screening stage may select parameters")

    @property
    def readiness(self) -> dict[str, Any]:
        strata = {case.stratum for case in self.cases}
        issues = []
        if len(self.cases) < self.minimum_case_count:
            issues.append("insufficient_case_count")
        if len(strata) < self.minimum_stratum_count:
            issues.append("insufficient_stratum_count")
        stratum_counts = {
            stratum: sum(case.stratum == stratum for case in self.cases)
            for stratum in strata
        }
        if any(
            count < self.minimum_cases_per_stratum
            for count in stratum_counts.values()
        ):
            issues.append("insufficient_cases_in_at_least_one_stratum")
        return {
            "ready": not issues,
            "case_count": len(self.cases),
            "stratum_count": len(strata),
            "stratum_case_counts": stratum_counts,
            "issues": issues,
        }

    @property
    def case_registry_sha256(self) -> str:
        return _json_sha256([case.as_dict() for case in self.cases])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "minimum_case_count": self.minimum_case_count,
            "minimum_stratum_count": self.minimum_stratum_count,
            "minimum_cases_per_stratum": self.minimum_cases_per_stratum,
            "permits_parameter_selection": self.permits_parameter_selection,
            "case_registry_sha256": self.case_registry_sha256,
            "readiness": self.readiness,
            "cases": [case.as_dict() for case in self.cases],
        }


@dataclass(frozen=True)
class ParameterCandidate:
    candidate_id: str
    parent_design_identity: str
    priority_multipliers: tuple[tuple[str, float], ...]
    parameter_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _nonempty("candidate_id", self.candidate_id),
        )
        object.__setattr__(
            self,
            "parent_design_identity",
            _nonempty("parent_design_identity", self.parent_design_identity),
        )
        multipliers = tuple(
            sorted(
                (str(name), float(value))
                for name, value in self.priority_multipliers
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for _, value in multipliers):
            raise ValueError("priority multipliers must be finite and positive")
        if len({name for name, _ in multipliers}) != len(multipliers):
            raise ValueError("priority multiplier names must be unique")
        object.__setattr__(self, "priority_multipliers", multipliers)
        sha = _sha256_digest("parameter_sha256", self.parameter_sha256)
        object.__setattr__(self, "parameter_sha256", sha)

    @property
    def multipliers(self) -> dict[str, float]:
        return dict(self.priority_multipliers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "parent_design_identity": self.parent_design_identity,
            "priority_multipliers": self.multipliers,
            "parameter_sha256": self.parameter_sha256,
        }


@dataclass(frozen=True)
class ParameterStudyProtocol:
    study_id: str
    base_design_identity: str
    base_design_parameter_sha256: str
    implementation_bundle_sha256: str
    tunable_priorities: tuple[str, ...]
    stages: tuple[ParameterStudyStage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _nonempty("study_id", self.study_id))
        object.__setattr__(
            self,
            "base_design_identity",
            _nonempty("base_design_identity", self.base_design_identity),
        )
        sha = _sha256_digest(
            "base_design_parameter_sha256",
            self.base_design_parameter_sha256,
        )
        object.__setattr__(self, "base_design_parameter_sha256", sha)
        object.__setattr__(
            self,
            "implementation_bundle_sha256",
            _sha256_digest(
                "implementation_bundle_sha256",
                self.implementation_bundle_sha256,
            ),
        )
        priorities = tuple(_nonempty("tunable priority", item) for item in self.tunable_priorities)
        if len(set(priorities)) != len(priorities):
            raise ValueError("tunable priorities must be unique")
        object.__setattr__(self, "tunable_priorities", priorities)
        stages = tuple(self.stages)
        if not stages or not all(isinstance(item, ParameterStudyStage) for item in stages):
            raise ValueError("stages must contain ParameterStudyStage values")
        if {stage.role for stage in stages} != _STAGE_ROLES:
            raise ValueError("protocol must contain smoke, screening and confirmation")
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("stage names must be unique")
        case_ids = [case.case_id for stage in stages for case in stage.cases]
        origins = [case.origin_time for stage in stages for case in stage.cases]
        event_ids = [case.source_event_id for stage in stages for case in stage.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("cases must not be reused across study stages")
        if len(set(origins)) != len(origins):
            raise ValueError("case origins must not be reused across study stages")
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("source events must not be reused across study stages")
        cases = [case for stage in stages for case in stage.cases]
        for index, left in enumerate(cases):
            left_start = datetime.fromisoformat(left.origin_time)
            left_end = left_start + timedelta(hours=left.duration_h)
            for right in cases[index + 1 :]:
                right_start = datetime.fromisoformat(right.origin_time)
                right_end = right_start + timedelta(hours=right.duration_h)
                if max(left_start, right_start) < min(left_end, right_end):
                    raise ValueError(
                        "study cases must not use overlapping time windows: "
                        f"{left.case_id}, {right.case_id}"
                    )
        object.__setattr__(self, "stages", stages)

    def stage(self, name: str) -> ParameterStudyStage:
        try:
            return next(stage for stage in self.stages if stage.name == name)
        except StopIteration as exc:
            raise KeyError(f"unknown parameter-study stage: {name}") from exc

    def candidate(
        self,
        *,
        candidate_id: str,
        base_design: PreviewMPCDesign,
        priority_multipliers: Mapping[str, Any] | None = None,
    ) -> ParameterCandidate:
        if base_design.identity != self.base_design_identity:
            raise ValueError("candidate base design identity does not match protocol")
        if base_design.as_dict()["parameter_sha256"] != self.base_design_parameter_sha256:
            raise ValueError("candidate base design parameters do not match protocol")
        multipliers = {} if priority_multipliers is None else dict(priority_multipliers)
        unknown = set(multipliers) - set(self.tunable_priorities)
        if unknown:
            raise ValueError(f"candidate changes non-tunable priorities: {sorted(unknown)}")
        variant = base_design.with_priority_multipliers(multipliers)
        return ParameterCandidate(
            candidate_id=candidate_id,
            parent_design_identity=base_design.identity,
            priority_multipliers=tuple(multipliers.items()),
            parameter_sha256=variant.as_dict()["parameter_sha256"],
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "base_design_identity": self.base_design_identity,
            "base_design_parameter_sha256": self.base_design_parameter_sha256,
            "implementation_bundle_sha256": self.implementation_bundle_sha256,
            "tunable_priorities": list(self.tunable_priorities),
            "comparison_rule": "same_candidates_on_every_case_within_a_stage",
            "selection_rule": (
                "screening_only_no_single_weighted_score_confirmation_is_holdout"
            ),
            "selection_sequence": [
                "reject_any_plan_execution_or_safety_failure",
                "reject_predeclared_posture_guardrail_violation",
                "compare_casewise_pump_volume_change_against_frozen_baseline",
                "use_iqr_and_improved_case_fraction_as_stability_checks",
            ],
            "candidate_registry_rule": (
                "freeze_before_screening_and_do_not_add_candidates_after_results"
            ),
            "confirmation_rule": (
                "freeze_candidate_and_acceptance_conditions_before_one_time_holdout"
            ),
            "reported_metrics": {
                "primary": ["transferred_volume_m3"],
                "secondary": [
                    "aggregate_pump_active_time_s",
                    "pump_start_count",
                ],
                "posture_guardrails": [
                    "dominant_tilt_rms_deg",
                    "dominant_tilt_time_above_deg_s",
                ],
                "validity": [
                    "plan_failure_count",
                    "fallback_count",
                    "precheck_failure_count",
                    "no_safe_candidate_count",
                    "sampled_posture_violation_count",
                ],
            },
            "stages": [stage.as_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class ParameterStudyObservation:
    case_id: str
    candidate_id: str
    candidate_parameter_sha256: str
    transferred_volume_m3: float
    aggregate_pump_active_time_s: float
    dominant_tilt_rms_deg: float
    dominant_tilt_time_above_deg_s: tuple[tuple[float, float], ...]
    pump_start_count: int = 0
    plan_failure_count: int = 0
    fallback_count: int = 0
    precheck_failure_count: int = 0
    no_safe_candidate_count: int = 0
    sampled_posture_violation_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _nonempty("case_id", self.case_id))
        object.__setattr__(
            self,
            "candidate_id",
            _nonempty("candidate_id", self.candidate_id),
        )
        object.__setattr__(
            self,
            "candidate_parameter_sha256",
            _sha256_digest(
                "candidate_parameter_sha256",
                self.candidate_parameter_sha256,
            ),
        )
        for name in (
            "transferred_volume_m3",
            "aggregate_pump_active_time_s",
            "dominant_tilt_rms_deg",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(name, getattr(self, name)),
            )
        thresholds = tuple(
            sorted(
                (
                    _finite_nonnegative("posture threshold", threshold),
                    _finite_nonnegative("posture threshold time", duration),
                )
                for threshold, duration in self.dominant_tilt_time_above_deg_s
            )
        )
        if len({threshold for threshold, _ in thresholds}) != len(thresholds):
            raise ValueError("posture thresholds must be unique")
        object.__setattr__(self, "dominant_tilt_time_above_deg_s", thresholds)
        for name in (
            "pump_start_count",
            "plan_failure_count",
            "fallback_count",
            "precheck_failure_count",
            "no_safe_candidate_count",
            "sampled_posture_violation_count",
        ):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


def summarize_paired_candidate(
    *,
    protocol: ParameterStudyProtocol,
    stage_name: str,
    baseline_candidate_id: str,
    candidate_id: str,
    frozen_candidates: tuple[ParameterCandidate, ...],
    observations: tuple[ParameterStudyObservation, ...],
    pump_volume_tolerance_m3: float = 1.0e-6,
) -> dict[str, Any]:
    """Summarize paired case differences without constructing a scalar score."""

    stage = protocol.stage(stage_name)
    tolerance = _finite_nonnegative(
        "pump_volume_tolerance_m3",
        pump_volume_tolerance_m3,
    )
    candidate_by_id = {item.candidate_id: item for item in frozen_candidates}
    if len(candidate_by_id) != len(frozen_candidates):
        raise ValueError("frozen candidate ids must be unique")
    for required_id in (baseline_candidate_id, candidate_id):
        if required_id not in candidate_by_id:
            raise ValueError(f"candidate {required_id} is absent from frozen registry")
    by_key: dict[tuple[str, str], ParameterStudyObservation] = {}
    for observation in observations:
        frozen = candidate_by_id.get(observation.candidate_id)
        if frozen is not None and (
            observation.candidate_parameter_sha256 != frozen.parameter_sha256
        ):
            raise ValueError(
                f"candidate parameter identity changed for {observation.candidate_id}"
            )
        key = (observation.case_id, observation.candidate_id)
        if key in by_key:
            raise ValueError(f"duplicate observation for {key}")
        by_key[key] = observation
    expected_cases = {case.case_id for case in stage.cases}
    expected_keys = {
        (case_id, candidate)
        for case_id in expected_cases
        for candidate in (baseline_candidate_id, candidate_id)
    }
    selected_keys = {
        key for key in by_key if key[0] in expected_cases and key[1] in {
            baseline_candidate_id,
            candidate_id,
        }
    }
    if selected_keys != expected_keys:
        missing = sorted(expected_keys - selected_keys)
        extra = sorted(selected_keys - expected_keys)
        raise ValueError(f"paired observations incomplete; missing={missing}, extra={extra}")

    rows = []
    for case in stage.cases:
        baseline = by_key[(case.case_id, baseline_candidate_id)]
        candidate = by_key[(case.case_id, candidate_id)]
        baseline_thresholds = dict(baseline.dominant_tilt_time_above_deg_s)
        candidate_thresholds = dict(candidate.dominant_tilt_time_above_deg_s)
        if baseline_thresholds.keys() != candidate_thresholds.keys():
            raise ValueError(
                f"posture thresholds differ for case {case.case_id}: "
                f"baseline={sorted(baseline_thresholds)}, "
                f"candidate={sorted(candidate_thresholds)}"
            )
        relative_pump_change = (
            100.0
            * (candidate.transferred_volume_m3 - baseline.transferred_volume_m3)
            / baseline.transferred_volume_m3
            if baseline.transferred_volume_m3 > 0.0
            else np.nan
        )
        absolute_pump_change = (
            candidate.transferred_volume_m3 - baseline.transferred_volume_m3
        )
        rows.append(
            {
                "case": case,
                "relative_pump_change_percent": relative_pump_change,
                "absolute_pump_change_m3": absolute_pump_change,
                "pump_time_change_s": (
                    candidate.aggregate_pump_active_time_s
                    - baseline.aggregate_pump_active_time_s
                ),
                "rms_posture_change_deg": (
                    candidate.dominant_tilt_rms_deg
                    - baseline.dominant_tilt_rms_deg
                ),
                "posture_threshold_time_change_s": {
                    threshold: candidate_thresholds[threshold]
                    - baseline_thresholds[threshold]
                    for threshold in baseline_thresholds
                },
                "candidate_improved_pump_volume": (
                    absolute_pump_change < -tolerance
                ),
                "candidate_unchanged_pump_volume": (
                    abs(absolute_pump_change) <= tolerance
                ),
                "candidate_failures": candidate.plan_failure_count,
                "candidate_fallbacks": candidate.fallback_count,
                "candidate_precheck_failures": candidate.precheck_failure_count,
                "candidate_no_safe_candidates": candidate.no_safe_candidate_count,
                "candidate_posture_violations": (
                    candidate.sampled_posture_violation_count
                ),
                "pump_start_count_change": (
                    candidate.pump_start_count - baseline.pump_start_count
                ),
            }
        )
    relative = np.asarray(
        [row["relative_pump_change_percent"] for row in rows],
        dtype=float,
    )
    finite_relative = relative[np.isfinite(relative)]
    absolute_changes = np.asarray(
        [row["absolute_pump_change_m3"] for row in rows],
        dtype=float,
    )

    def pump_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
        values = np.asarray(
            [item["relative_pump_change_percent"] for item in items],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        return {
            "case_count": len(items),
            "median_relative_pump_change_percent": (
                float(np.median(values)) if values.size else None
            ),
            "improved_case_fraction": float(
                np.mean([item["candidate_improved_pump_volume"] for item in items])
            ),
            "unchanged_case_fraction": float(
                np.mean([item["candidate_unchanged_pump_volume"] for item in items])
            ),
        }

    by_stratum = {}
    for stratum in sorted({case.stratum for case in stage.cases}):
        by_stratum[stratum] = pump_summary(
            [row for row in rows if row["case"].stratum == stratum]
        )
    threshold_time_changes = {}
    if rows:
        thresholds = sorted(rows[0]["posture_threshold_time_change_s"])
        for threshold in thresholds:
            changes = np.asarray(
                [
                    row["posture_threshold_time_change_s"][threshold]
                    for row in rows
                ],
                dtype=float,
            )
            threshold_time_changes[str(threshold)] = {
                "median_change_s": float(np.median(changes)),
                "maximum_increase_s": float(np.max(changes)),
            }
    validity = {
        "baseline_plan_failure_count": int(
            sum(baseline.plan_failure_count for baseline in (
                by_key[(case.case_id, baseline_candidate_id)] for case in stage.cases
            ))
        ),
        "candidate_plan_failure_count": int(
            sum(row["candidate_failures"] for row in rows)
        ),
        "baseline_fallback_count": int(
            sum(baseline.fallback_count for baseline in (
                by_key[(case.case_id, baseline_candidate_id)] for case in stage.cases
            ))
        ),
        "candidate_fallback_count": int(
            sum(row["candidate_fallbacks"] for row in rows)
        ),
        "baseline_precheck_failure_count": int(
            sum(baseline.precheck_failure_count for baseline in (
                by_key[(case.case_id, baseline_candidate_id)] for case in stage.cases
            ))
        ),
        "candidate_precheck_failure_count": int(
            sum(row["candidate_precheck_failures"] for row in rows)
        ),
        "baseline_no_safe_candidate_count": int(
            sum(baseline.no_safe_candidate_count for baseline in (
                by_key[(case.case_id, baseline_candidate_id)] for case in stage.cases
            ))
        ),
        "candidate_no_safe_candidate_count": int(
            sum(row["candidate_no_safe_candidates"] for row in rows)
        ),
        "baseline_sampled_posture_violation_count": int(
            sum(baseline.sampled_posture_violation_count for baseline in (
                by_key[(case.case_id, baseline_candidate_id)] for case in stage.cases
            ))
        ),
        "candidate_sampled_posture_violation_count": int(
            sum(row["candidate_posture_violations"] for row in rows)
        ),
    }
    return {
        "stage": stage.name,
        "stage_role": stage.role,
        "parameter_selection_permitted": stage.permits_parameter_selection,
        "baseline_candidate_id": baseline_candidate_id,
        "candidate_id": candidate_id,
        "case_count": len(rows),
        "paired_case_set_complete": True,
        "pump_volume": {
            "median_absolute_change_m3": float(np.median(absolute_changes)),
            "median_relative_change_percent": (
                float(np.median(finite_relative)) if finite_relative.size else None
            ),
            "relative_change_valid_case_count": int(finite_relative.size),
            "interquartile_range_percent": (
                np.quantile(finite_relative, [0.25, 0.75]).tolist()
                if finite_relative.size
                else None
            ),
            "improved_case_fraction": float(
                np.mean([row["candidate_improved_pump_volume"] for row in rows])
            ),
            "unchanged_case_fraction": float(
                np.mean([row["candidate_unchanged_pump_volume"] for row in rows])
            ),
            "comparison_tolerance_m3": tolerance,
            "by_stratum": by_stratum,
        },
        "pump_active_time_change_s": {
            "median": float(np.median([row["pump_time_change_s"] for row in rows])),
            "maximum": float(np.max([row["pump_time_change_s"] for row in rows])),
        },
        "pump_start_count_change": {
            "median": float(
                np.median([row["pump_start_count_change"] for row in rows])
            ),
            "maximum": int(max(row["pump_start_count_change"] for row in rows)),
        },
        "posture_rms_change_deg": {
            "median": float(np.median([row["rms_posture_change_deg"] for row in rows])),
            "maximum": float(np.max([row["rms_posture_change_deg"] for row in rows])),
        },
        "posture_threshold_time_change_s": threshold_time_changes,
        "validity": validity,
        "execution_and_internal_constraint_records_valid": all(
            value == 0 for value in validity.values()
        ),
        "automatic_scalar_score": None,
    }


def freeze_candidate_registry(
    candidates: tuple[ParameterCandidate, ...],
) -> dict[str, Any]:
    """Return a stable candidate-set identity without choosing a candidate."""

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidate registry must not be empty")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("candidate ids must be unique")
    records = [candidate.as_dict() for candidate in candidates]
    return {
        "candidate_count": len(records),
        "candidates": records,
        "candidate_registry_sha256": _json_sha256(records),
        "frozen_before_screening": True,
    }


__all__ = [
    "ParameterCandidate",
    "ParameterStudyCase",
    "ParameterStudyObservation",
    "ParameterStudyProtocol",
    "ParameterStudyStage",
    "freeze_candidate_registry",
    "summarize_paired_candidate",
]
