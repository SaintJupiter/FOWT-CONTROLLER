"""Shadow-mode error statistics for the lightweight attitude rollout.

This module compares predicted pitch/roll attitude with the measured attitude at
the next rollout stage.  It is deliberately disconnected from candidate ranking
and provider runtime: the output is evidence only and cannot authorize a control
action.

Sign agreement is reported in two forms.  ``value_sign`` compares the predicted
and measured attitude signs.  ``change_sign`` compares their changes from the
same current attitude and is the more relevant diagnostic for candidate-action
ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AttitudeShadowCriteria:
    """Configurable screening criteria for a future ranking experiment.

    Passing these checks means that the rollout has enough shadow evidence to be
    tested as an additional ranking feature.  It does not enable ranking and is
    not a claim of controller validation.
    """

    report_only: bool = True
    min_samples: int = 100
    min_distinct_cases: int = 10
    max_axis_mae_deg: float = 0.75
    max_axis_p95_abs_error_deg: float = 2.0
    max_axis_abs_bias_deg: float = 0.30
    min_change_sign_agreement: float = 0.75
    sign_tolerance_deg: float = 0.05
    quantiles: tuple[float, ...] = (0.50, 0.90, 0.95)

    def __post_init__(self) -> None:
        if int(self.min_samples) <= 0:
            raise ValueError("min_samples must be positive")
        if int(self.min_distinct_cases) <= 0:
            raise ValueError("min_distinct_cases must be positive")
        for name, value in (
            ("max_axis_mae_deg", self.max_axis_mae_deg),
            ("max_axis_p95_abs_error_deg", self.max_axis_p95_abs_error_deg),
            ("max_axis_abs_bias_deg", self.max_axis_abs_bias_deg),
            ("sign_tolerance_deg", self.sign_tolerance_deg),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= float(self.min_change_sign_agreement) <= 1.0:
            raise ValueError("min_change_sign_agreement must lie in [0, 1]")
        q = np.asarray(self.quantiles, dtype=float)
        if q.ndim != 1 or q.size == 0 or not np.all(np.isfinite(q)):
            raise ValueError("quantiles must be a non-empty finite sequence")
        if np.any(q <= 0.0) or np.any(q >= 1.0) or np.any(np.diff(q) <= 0.0):
            raise ValueError("quantiles must be strictly increasing within (0, 1)")
        if 0.95 not in tuple(float(value) for value in q):
            raise ValueError("quantiles must include 0.95 for the evidence screen")


@dataclass(frozen=True)
class AxisShadowStatistics:
    sample_count: int
    bias_deg: float
    mae_deg: float
    rmse_deg: float
    abs_error_quantiles_deg: Mapping[float, float]
    value_sign_comparable_count: int
    value_sign_agreement: float
    change_sign_comparable_count: int
    change_sign_agreement: float


@dataclass(frozen=True)
class AttitudeShadowResiduals:
    error_deg: np.ndarray
    abs_error_deg: np.ndarray
    vector_error_deg: np.ndarray
    value_sign_agreement: np.ndarray
    value_sign_comparable: np.ndarray
    change_sign_agreement: np.ndarray | None
    change_sign_comparable: np.ndarray | None


@dataclass(frozen=True)
class AttitudeShadowEvidence:
    meets_ranking_trial_evidence: bool
    status: str
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]
    report_only: bool
    sorting_authority_enabled: bool = False


@dataclass(frozen=True)
class AttitudeShadowReport:
    sample_count: int
    distinct_case_count: int
    pitch: AxisShadowStatistics
    roll: AxisShadowStatistics
    vector_error_quantiles_deg: Mapping[float, float]
    residuals: AttitudeShadowResiduals
    evidence: AttitudeShadowEvidence

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly summary without the per-sample residual arrays."""

        def axis_dict(axis: AxisShadowStatistics) -> dict[str, object]:
            return {
                "sample_count": axis.sample_count,
                "bias_deg": axis.bias_deg,
                "mae_deg": axis.mae_deg,
                "rmse_deg": axis.rmse_deg,
                "abs_error_quantiles_deg": dict(axis.abs_error_quantiles_deg),
                "value_sign_comparable_count": axis.value_sign_comparable_count,
                "value_sign_agreement": axis.value_sign_agreement,
                "change_sign_comparable_count": axis.change_sign_comparable_count,
                "change_sign_agreement": axis.change_sign_agreement,
            }

        return {
            "sample_count": self.sample_count,
            "distinct_case_count": self.distinct_case_count,
            "pitch": axis_dict(self.pitch),
            "roll": axis_dict(self.roll),
            "vector_error_quantiles_deg": dict(self.vector_error_quantiles_deg),
            "evidence": {
                "meets_ranking_trial_evidence": (
                    self.evidence.meets_ranking_trial_evidence
                ),
                "status": self.evidence.status,
                "checks": dict(self.evidence.checks),
                "reasons": list(self.evidence.reasons),
                "report_only": self.evidence.report_only,
                "sorting_authority_enabled": (
                    self.evidence.sorting_authority_enabled
                ),
            },
        }


def _as_attitude_array(name: str, values: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (n_samples, 2) in pitch/roll order")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _quantile_map(values: np.ndarray, quantiles: Sequence[float]) -> dict[float, float]:
    return {
        float(q): float(np.quantile(values, float(q)))
        for q in quantiles
    }


def _sign_comparison(
    predicted: np.ndarray,
    actual: np.ndarray,
    tolerance_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    predicted_sign = np.where(
        np.abs(predicted) <= tolerance_deg,
        0,
        np.sign(predicted),
    )
    actual_sign = np.where(
        np.abs(actual) <= tolerance_deg,
        0,
        np.sign(actual),
    )
    comparable = (predicted_sign != 0) | (actual_sign != 0)
    agreement = comparable & (predicted_sign == actual_sign)
    return agreement, comparable


def _agreement_fraction(agreement: np.ndarray, comparable: np.ndarray) -> float:
    count = int(np.count_nonzero(comparable))
    if count == 0:
        return float("nan")
    return float(np.count_nonzero(agreement) / count)


def _axis_statistics(
    error: np.ndarray,
    value_agreement: np.ndarray,
    value_comparable: np.ndarray,
    change_agreement: np.ndarray | None,
    change_comparable: np.ndarray | None,
    quantiles: Sequence[float],
) -> AxisShadowStatistics:
    if change_agreement is None or change_comparable is None:
        change_count = 0
        change_fraction = float("nan")
    else:
        change_count = int(np.count_nonzero(change_comparable))
        change_fraction = _agreement_fraction(change_agreement, change_comparable)
    return AxisShadowStatistics(
        sample_count=int(error.size),
        bias_deg=float(np.mean(error)),
        mae_deg=float(np.mean(np.abs(error))),
        rmse_deg=float(np.sqrt(np.mean(error * error))),
        abs_error_quantiles_deg=_quantile_map(np.abs(error), quantiles),
        value_sign_comparable_count=int(np.count_nonzero(value_comparable)),
        value_sign_agreement=_agreement_fraction(value_agreement, value_comparable),
        change_sign_comparable_count=change_count,
        change_sign_agreement=change_fraction,
    )


def _build_evidence(
    *,
    sample_count: int,
    distinct_case_count: int,
    pitch: AxisShadowStatistics,
    roll: AxisShadowStatistics,
    criteria: AttitudeShadowCriteria,
) -> AttitudeShadowEvidence:
    p95 = 0.95
    checks = {
        "sample_count": sample_count >= criteria.min_samples,
        "distinct_case_count": distinct_case_count >= criteria.min_distinct_cases,
        "pitch_mae": pitch.mae_deg <= criteria.max_axis_mae_deg,
        "roll_mae": roll.mae_deg <= criteria.max_axis_mae_deg,
        "pitch_p95_abs_error": (
            pitch.abs_error_quantiles_deg.get(p95, float("inf"))
            <= criteria.max_axis_p95_abs_error_deg
        ),
        "roll_p95_abs_error": (
            roll.abs_error_quantiles_deg.get(p95, float("inf"))
            <= criteria.max_axis_p95_abs_error_deg
        ),
        "pitch_bias": abs(pitch.bias_deg) <= criteria.max_axis_abs_bias_deg,
        "roll_bias": abs(roll.bias_deg) <= criteria.max_axis_abs_bias_deg,
        "pitch_change_sign": (
            math.isfinite(pitch.change_sign_agreement)
            and pitch.change_sign_agreement >= criteria.min_change_sign_agreement
        ),
        "roll_change_sign": (
            math.isfinite(roll.change_sign_agreement)
            and roll.change_sign_agreement >= criteria.min_change_sign_agreement
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    enough_data = checks["sample_count"] and checks["distinct_case_count"]
    meets = all(checks.values())
    if not enough_data:
        status = "insufficient_shadow_data"
    elif meets:
        status = "eligible_for_controlled_ranking_trial"
    else:
        status = "shadow_accuracy_not_ready"
    return AttitudeShadowEvidence(
        meets_ranking_trial_evidence=meets,
        status=status,
        checks=checks,
        reasons=reasons,
        report_only=bool(criteria.report_only),
        sorting_authority_enabled=False,
    )


def evaluate_attitude_rollout_shadow(
    predicted_next_pitch_roll_deg: Sequence[Sequence[float]],
    actual_next_pitch_roll_deg: Sequence[Sequence[float]],
    *,
    current_pitch_roll_deg: Sequence[Sequence[float]] | None = None,
    case_ids: Sequence[str] | None = None,
    criteria: AttitudeShadowCriteria | None = None,
) -> AttitudeShadowReport:
    """Compare shadow predictions with actual next-stage pitch/roll attitude.

    Arrays use ``[pitch, roll]`` order.  ``current_pitch_roll_deg`` is optional
    for basic error reporting but required for change-direction evidence.  Case
    identifiers are required to satisfy the independent-case evidence check.
    """

    cfg = criteria or AttitudeShadowCriteria()
    predicted = _as_attitude_array(
        "predicted_next_pitch_roll_deg", predicted_next_pitch_roll_deg
    )
    actual = _as_attitude_array(
        "actual_next_pitch_roll_deg", actual_next_pitch_roll_deg
    )
    if predicted.shape != actual.shape:
        raise ValueError("predicted and actual attitude arrays must have equal shape")

    current = None
    if current_pitch_roll_deg is not None:
        current = _as_attitude_array("current_pitch_roll_deg", current_pitch_roll_deg)
        if current.shape != actual.shape:
            raise ValueError("current and next-stage attitude arrays must have equal shape")

    if case_ids is None:
        distinct_case_count = 0
    else:
        if len(case_ids) != predicted.shape[0]:
            raise ValueError("case_ids length must equal the number of samples")
        normalized_case_ids = tuple(str(case_id).strip() for case_id in case_ids)
        if any(not case_id for case_id in normalized_case_ids):
            raise ValueError("case_ids must not contain empty identifiers")
        distinct_case_count = len(set(normalized_case_ids))

    error = predicted - actual
    abs_error = np.abs(error)
    vector_error = np.linalg.norm(error, axis=1)
    value_agreement, value_comparable = _sign_comparison(
        predicted,
        actual,
        cfg.sign_tolerance_deg,
    )

    if current is None:
        change_agreement = None
        change_comparable = None
    else:
        change_agreement, change_comparable = _sign_comparison(
            predicted - current,
            actual - current,
            cfg.sign_tolerance_deg,
        )

    pitch = _axis_statistics(
        error[:, 0],
        value_agreement[:, 0],
        value_comparable[:, 0],
        None if change_agreement is None else change_agreement[:, 0],
        None if change_comparable is None else change_comparable[:, 0],
        cfg.quantiles,
    )
    roll = _axis_statistics(
        error[:, 1],
        value_agreement[:, 1],
        value_comparable[:, 1],
        None if change_agreement is None else change_agreement[:, 1],
        None if change_comparable is None else change_comparable[:, 1],
        cfg.quantiles,
    )
    residuals = AttitudeShadowResiduals(
        error_deg=error,
        abs_error_deg=abs_error,
        vector_error_deg=vector_error,
        value_sign_agreement=value_agreement,
        value_sign_comparable=value_comparable,
        change_sign_agreement=change_agreement,
        change_sign_comparable=change_comparable,
    )
    evidence = _build_evidence(
        sample_count=int(predicted.shape[0]),
        distinct_case_count=distinct_case_count,
        pitch=pitch,
        roll=roll,
        criteria=cfg,
    )
    return AttitudeShadowReport(
        sample_count=int(predicted.shape[0]),
        distinct_case_count=distinct_case_count,
        pitch=pitch,
        roll=roll,
        vector_error_quantiles_deg=_quantile_map(vector_error, cfg.quantiles),
        residuals=residuals,
        evidence=evidence,
    )


__all__ = [
    "AttitudeShadowCriteria",
    "AttitudeShadowEvidence",
    "AttitudeShadowReport",
    "AttitudeShadowResiduals",
    "AxisShadowStatistics",
    "evaluate_attitude_rollout_shadow",
]
