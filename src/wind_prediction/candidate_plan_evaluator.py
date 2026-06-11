from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .forecast_adapter import ForecastResult


DEFAULT_ALLOC_MATRIX = np.array(
    [
        [-1.0, 0.0],
        [0.5, 1.0],
        [0.5, -1.0],
    ],
    dtype=float,
)


SEGMENT_SPECS = (
    ("0_20m", slice(0, 2), 1.00, "attention_event_0_20m"),
    ("20_40m", slice(2, 4), 0.65, "attention_event_20_40m"),
    ("40_60m", slice(4, 6), 0.35, "attention_event_40_60m"),
)


@dataclass(frozen=True)
class CandidatePlan:
    name: str
    pitch_sp_deg: float
    roll_sp_deg: float
    family: str


@dataclass(frozen=True)
class BaselineScales:
    pitch_deg: float
    roll_deg: float
    pump_rate_m3_min: float
    backlog_kg: float
    wind_speed_mps: float
    deadband_pitch_deg: float
    deadband_roll_deg: float
    saturation_margin_ratio: float = 0.08
    eps: float = 1e-6

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class WeightProfile:
    name: str
    lambda_att: float
    lambda_pump: float
    lambda_switch: float
    lambda_sat: float
    gamma_risk: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) if k != "name" else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class StructureProfile:
    name: str
    future_weight: float
    posture_weight: float
    activation_near: float
    activation_mid: float
    activation_far: float
    activation_delta: float
    activation_posture: float
    active_blend_activation: float
    active_blend_near: float
    risk_mult_near: float
    risk_mult_mid: float
    risk_mult_far: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) if k != "name" else v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class AttitudeBands:
    pitch_rms_band_deg: float
    roll_rms_band_deg: float
    exceed_time_band_s: float
    eps: float = 1e-6

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


DEFAULT_WEIGHT_PROFILES: dict[str, WeightProfile] = {
    "balanced": WeightProfile(
        name="balanced",
        lambda_att=1.00,
        lambda_pump=1.00,
        lambda_switch=0.85,
        lambda_sat=1.00,
        gamma_risk=1.00,
    ),
    "attitude_priority": WeightProfile(
        name="attitude_priority",
        lambda_att=1.00,
        lambda_pump=0.80,
        lambda_switch=0.70,
        lambda_sat=1.00,
        gamma_risk=1.15,
    ),
    "pump_priority": WeightProfile(
        name="pump_priority",
        lambda_att=1.00,
        lambda_pump=1.25,
        lambda_switch=1.05,
        lambda_sat=1.10,
        gamma_risk=0.85,
    ),
    "switch_smooth": WeightProfile(
        name="switch_smooth",
        lambda_att=1.00,
        lambda_pump=1.05,
        lambda_switch=1.30,
        lambda_sat=1.05,
        gamma_risk=0.95,
    ),
    "risk_aggressive": WeightProfile(
        name="risk_aggressive",
        lambda_att=1.00,
        lambda_pump=0.90,
        lambda_switch=0.75,
        lambda_sat=1.00,
        gamma_risk=1.35,
    ),
    "risk_conservative": WeightProfile(
        name="risk_conservative",
        lambda_att=1.00,
        lambda_pump=1.10,
        lambda_switch=1.05,
        lambda_sat=1.10,
        gamma_risk=0.70,
    ),
}


DEFAULT_STRUCTURE_PROFILES: dict[str, StructureProfile] = {
    "near_focus": StructureProfile(
        name="near_focus",
        future_weight=0.70,
        posture_weight=0.30,
        activation_near=0.55,
        activation_mid=0.25,
        activation_far=0.10,
        activation_delta=0.10,
        activation_posture=0.10,
        active_blend_activation=0.45,
        active_blend_near=0.55,
        risk_mult_near=0.60,
        risk_mult_mid=0.25,
        risk_mult_far=0.15,
    ),
    "smooth_view": StructureProfile(
        name="smooth_view",
        future_weight=0.60,
        posture_weight=0.40,
        activation_near=0.40,
        activation_mid=0.25,
        activation_far=0.15,
        activation_delta=0.10,
        activation_posture=0.10,
        active_blend_activation=0.55,
        active_blend_near=0.45,
        risk_mult_near=0.45,
        risk_mult_mid=0.30,
        risk_mult_far=0.25,
    ),
    "risk_focus": StructureProfile(
        name="risk_focus",
        future_weight=0.75,
        posture_weight=0.25,
        activation_near=0.60,
        activation_mid=0.25,
        activation_far=0.10,
        activation_delta=0.05,
        activation_posture=0.05,
        active_blend_activation=0.35,
        active_blend_near=0.65,
        risk_mult_near=0.70,
        risk_mult_mid=0.20,
        risk_mult_far=0.10,
    ),
}


def make_default_baseline_scales(
    deadband_pitch_deg: float = 1.0,
    deadband_roll_deg: float = 0.8,
) -> BaselineScales:
    return BaselineScales(
        pitch_deg=max(abs(float(deadband_pitch_deg)), 1e-3),
        roll_deg=max(abs(float(deadband_roll_deg)), 1e-3),
        pump_rate_m3_min=1.0,
        backlog_kg=1000.0,
        wind_speed_mps=12.0,
        deadband_pitch_deg=float(deadband_pitch_deg),
        deadband_roll_deg=float(deadband_roll_deg),
        saturation_margin_ratio=0.08,
        eps=1e-6,
    )


class CandidatePlanEvaluator:
    """Lightweight slow-layer decision maker with normalized scoring.

    The formal version avoids absolute heuristic magnitudes as much as possible:
    candidate amplitudes are generated from controller deadbands, while the cost
    function is normalized with closed-only baseline statistics so that a small
    number of high-level weights encodes relative priorities.
    """

    def __init__(
        self,
        tank_capacity_kg: float = 1850.0 * 1025.0,
        baseline_scales: BaselineScales | None = None,
        weight_profile: WeightProfile | None = None,
        structure_profile: StructureProfile | None = None,
        selection_mode: str = "weighted_cost_v1",
        attitude_bands: AttitudeBands | None = None,
        wind_reference_mps: float = 12.0,
        active_ratio_low: float = 0.25,
        active_ratio_high: float = 0.50,
        pump_saving_ratio: float = 0.12,
        upper_capacity_guard_ratio: float = 0.92,
        lower_capacity_guard_ratio: float = 0.08,
        reversal_guard_ratio: float = 0.50,
        backlog_guard_multiplier: float = 1.00,
    ) -> None:
        self.tank_capacity_kg = float(tank_capacity_kg)
        self.scales = baseline_scales if baseline_scales is not None else make_default_baseline_scales()
        self.weight_profile = (
            weight_profile if weight_profile is not None else DEFAULT_WEIGHT_PROFILES["balanced"]
        )
        self.structure_profile = (
            structure_profile
            if structure_profile is not None
            else DEFAULT_STRUCTURE_PROFILES["near_focus"]
        )
        self.selection_mode = str(selection_mode)
        self.attitude_bands = (
            attitude_bands
            if attitude_bands is not None
            else AttitudeBands(
                pitch_rms_band_deg=max(self.scales.deadband_pitch_deg, self.scales.eps),
                roll_rms_band_deg=max(self.scales.deadband_roll_deg, self.scales.eps),
                exceed_time_band_s=600.0,
                eps=self.scales.eps,
            )
        )
        self.active_ratio_low = float(active_ratio_low)
        self.active_ratio_high = float(active_ratio_high)
        self.pump_saving_ratio = float(pump_saving_ratio)
        self.wind_reference_mps = float(max(wind_reference_mps, 1.0))
        self.upper_capacity_guard_ratio = float(upper_capacity_guard_ratio)
        self.lower_capacity_guard_ratio = float(lower_capacity_guard_ratio)
        self.reversal_guard_ratio = float(reversal_guard_ratio)
        self.backlog_guard_multiplier = float(backlog_guard_multiplier)
        self._last_selected = np.zeros(2, dtype=float)
        self.records: list[dict[str, Any]] = []
        self.candidate_records: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._last_selected[:] = 0.0
        self.records = []
        self.candidate_records = []

    def metadata(self) -> dict[str, Any]:
        return {
            "candidate_ratios": {
                "active_low": float(self.active_ratio_low),
                "active_high": float(self.active_ratio_high),
                "pump_saving": float(self.pump_saving_ratio),
            },
            "engineering_reference": {
                "wind_reference_mps": float(self.wind_reference_mps),
            },
            "baseline_scales": self.scales.to_dict(),
            "weight_profile": self.weight_profile.to_dict(),
            "structure_profile": self.structure_profile.to_dict(),
            "selection_mode": str(self.selection_mode),
            "attitude_bands": self.attitude_bands.to_dict(),
            "constraint_config": {
                "upper_capacity_guard_ratio": float(self.upper_capacity_guard_ratio),
                "lower_capacity_guard_ratio": float(self.lower_capacity_guard_ratio),
                "reversal_guard_ratio": float(self.reversal_guard_ratio),
                "backlog_guard_multiplier": float(self.backlog_guard_multiplier),
            },
        }

    def evaluate(
        self,
        state: np.ndarray,
        wind_obs: dict[str, Any] | None,
        plant_info_prev: dict[str, Any] | None,
        forecast: ForecastResult,
    ) -> dict[str, Any]:
        current_pitch = float(np.degrees(state[4]))
        current_roll = float(np.degrees(state[3]))
        current_ws = float((wind_obs or {}).get("ws", 0.0))
        current_wd = float((wind_obs or {}).get("wd_deg", 0.0))

        segment_blob = self._build_segment_blob(forecast)
        risk_blob = self._build_risk_blob(forecast)
        desired_vec, desired_meta = self._desired_setpoint_vector(
            current_ws=current_ws,
            current_wd=current_wd,
            current_pitch_deg=current_pitch,
            current_roll_deg=current_roll,
            segment_blob=segment_blob,
            risk_blob=risk_blob,
        )
        candidates = self._build_candidates(
            desired_vec=desired_vec,
            desired_meta=desired_meta,
            risk_blob=risk_blob,
            plant_info_prev=plant_info_prev,
        )

        feasible_rows: list[dict[str, Any]] = []
        screened_rows: list[dict[str, Any]] = []
        candidate_diag_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            feasible, reason, proxy = self._check_hard_constraints(
                candidate=candidate,
                plant_info_prev=plant_info_prev,
            )
            if not feasible:
                row = {
                    "candidate_name": candidate.name,
                    "screen_reason": reason,
                    "pitch_sp_deg": float(candidate.pitch_sp_deg),
                    "roll_sp_deg": float(candidate.roll_sp_deg),
                    "hard_ok": 0,
                    "hard_reject_reason": reason,
                    "attitude_eligible": 0,
                    "attitude_reject_reason": "hard_reject",
                    "attitude_risk": np.inf,
                    "pump_rank_tuple": "",
                    "selection_mode": str(self.selection_mode),
                }
                screened_rows.append(row)
                candidate_diag_rows.append(row)
                continue

            score = self._score_candidate(
                candidate=candidate,
                desired_vec=desired_vec,
                current_pitch_deg=current_pitch,
                current_roll_deg=current_roll,
                plant_info_prev=plant_info_prev,
                risk_blob=risk_blob,
                proxy=proxy,
            )
            attitude_diag = self._attitude_eligible(
                candidate=candidate,
                desired_vec=desired_vec,
                desired_meta=desired_meta,
                current_pitch_deg=current_pitch,
                current_roll_deg=current_roll,
                risk_blob=risk_blob,
            )
            score["hard_ok"] = 1
            score["hard_reject_reason"] = ""
            score["attitude_eligible"] = int(attitude_diag["eligible"])
            score["attitude_reject_reason"] = str(attitude_diag["reject_reason"])
            score["attitude_risk"] = float(attitude_diag["attitude_risk"])
            score["attitude_pitch_proxy_deg"] = float(attitude_diag["pitch_rms_proxy_deg"])
            score["attitude_roll_proxy_deg"] = float(attitude_diag["roll_rms_proxy_deg"])
            score["attitude_exceed_proxy_s"] = float(attitude_diag["pose_exceed_time_proxy_s"])
            score["pump_rank_tuple"] = self._pump_burden_tuple(
                candidate=candidate,
                plant_info_prev=plant_info_prev,
                proxy=proxy,
                score_row=score,
            )
            score["selection_mode"] = str(self.selection_mode)
            feasible_rows.append(score)
            candidate_diag_rows.append(score)

        if feasible_rows:
            if self.selection_mode == "constraint_first_v1":
                selected, feasible_rows = self._select_constraint_first(feasible_rows)
            else:
                feasible_rows.sort(key=lambda row: float(row["cost_total"]))
                selected = feasible_rows[0]
        else:
            selected = {
                "candidate_name": "hold_fallback",
                "pitch_sp_deg": 0.0,
                "roll_sp_deg": 0.0,
                "cost_total": np.inf,
                "cost_attitude": np.inf,
                "cost_pump": np.inf,
                "cost_switch": np.inf,
                "cost_saturation": np.inf,
                "alignment": 0.0,
                "feasible": 0,
                "hard_ok": 0,
                "hard_reject_reason": "all_screened",
                "attitude_eligible": 0,
                "attitude_reject_reason": "all_screened",
                "attitude_risk": np.inf,
                "pump_rank_tuple": "",
                "selection_mode": str(self.selection_mode),
            }

        selected_vec = np.array(
            [float(selected["pitch_sp_deg"]), float(selected["roll_sp_deg"])],
            dtype=float,
        )
        self._last_selected[:] = selected_vec

        record = {
            "timestamp": forecast.timestamp,
            "selected_plan": str(selected["candidate_name"]),
            "selected_pitch_sp_deg": float(selected_vec[0]),
            "selected_roll_sp_deg": float(selected_vec[1]),
            "selected_action_signature": f"{selected_vec[0]:+.4f},{selected_vec[1]:+.4f}",
            "desired_pitch_proxy_deg": float(desired_vec[0]),
            "desired_roll_proxy_deg": float(desired_vec[1]),
            "activation_score": float(desired_meta["activation_score"]),
            "global_risk": float(risk_blob["global_risk"]),
            "risk_0_20m": float(risk_blob["0_20m"]),
            "risk_20_40m": float(risk_blob["20_40m"]),
            "risk_40_60m": float(risk_blob["40_60m"]),
            "cost_total": float(selected["cost_total"]),
            "selection_mode": str(self.selection_mode),
            "weight_profile": str(self.weight_profile.name),
            "screened_count": int(len(screened_rows)),
            "feasible_count": int(len(feasible_rows)),
            "rank_order": ">".join(str(row["candidate_name"]) for row in feasible_rows),
        }
        for row in feasible_rows:
            prefix = str(row["candidate_name"])
            record[f"{prefix}_cost_total"] = float(row["cost_total"])
            record[f"{prefix}_cost_attitude"] = float(row["cost_attitude"])
            record[f"{prefix}_cost_pump"] = float(row["cost_pump"])
            record[f"{prefix}_cost_switch"] = float(row["cost_switch"])
            record[f"{prefix}_cost_saturation"] = float(row["cost_saturation"])
            record[f"{prefix}_alignment"] = float(row["alignment"])
            record[f"{prefix}_action_norm"] = float(row.get("action_ratio", 0.0))
            record[f"{prefix}_attitude_risk"] = float(row.get("attitude_risk", np.nan))
            record[f"{prefix}_attitude_eligible"] = int(row.get("attitude_eligible", 0))
            record[f"{prefix}_pump_rank_tuple"] = str(row.get("pump_rank_tuple", ""))
        self.records.append(record)
        for row in candidate_diag_rows:
            self.candidate_records.append(
                {
                    "timestamp": forecast.timestamp,
                    "selection_mode": str(self.selection_mode),
                    "selected_plan": str(selected["candidate_name"]),
                    "selected": int(str(row["candidate_name"]) == str(selected["candidate_name"])),
                    "candidate_name": str(row["candidate_name"]),
                    "pitch_sp_deg": float(row.get("pitch_sp_deg", 0.0)),
                    "roll_sp_deg": float(row.get("roll_sp_deg", 0.0)),
                    "hard_ok": int(row.get("hard_ok", 0)),
                    "hard_reject_reason": str(row.get("hard_reject_reason", "")),
                    "attitude_eligible": int(row.get("attitude_eligible", 0)),
                    "attitude_reject_reason": str(row.get("attitude_reject_reason", "")),
                    "attitude_risk": float(row.get("attitude_risk", np.inf)),
                    "pump_rank_tuple": str(row.get("pump_rank_tuple", "")),
                    "cost_total": float(row.get("cost_total", np.inf)),
                    "cost_attitude": float(row.get("cost_attitude", np.inf)),
                    "cost_pump": float(row.get("cost_pump", np.inf)),
                    "cost_switch": float(row.get("cost_switch", np.inf)),
                    "cost_saturation": float(row.get("cost_saturation", np.inf)),
                }
            )

        output = {
            "pitch_bias_deg": float(selected_vec[0]),
            "roll_bias_deg": float(selected_vec[1]),
            "pitch_sp_deg": float(selected_vec[0]),
            "roll_sp_deg": float(selected_vec[1]),
            "source": "prediction_candidate_decision",
            "preview_mode": "candidate_plan_evaluation",
            "risk_level": str(desired_meta["risk_level"]),
            "selected_plan": str(selected["candidate_name"]),
            "selection_mode": str(self.selection_mode),
            "weight_profile": str(self.weight_profile.name),
            "decision_activation": float(desired_meta["activation_score"]),
            "desired_pitch_proxy_deg": float(desired_vec[0]),
            "desired_roll_proxy_deg": float(desired_vec[1]),
            "cost_total": float(selected["cost_total"]),
            "cost_attitude": float(selected["cost_attitude"]),
            "cost_pump": float(selected["cost_pump"]),
            "cost_switch": float(selected["cost_switch"]),
            "cost_saturation": float(selected["cost_saturation"]),
            "segment_risk_0_20m": float(risk_blob["0_20m"]),
            "segment_risk_20_40m": float(risk_blob["20_40m"]),
            "segment_risk_40_60m": float(risk_blob["40_60m"]),
            "ballast_attention_event_prob": float(risk_blob["global_risk"]),
            "screened_candidate_count": int(len(screened_rows)),
            "feasible_candidate_count": int(len(feasible_rows)),
            "candidate_score_snapshot": ";".join(
                f"{row['candidate_name']}={row['cost_total']:.3f}" for row in feasible_rows[:5]
            ),
            "screened_snapshot": ";".join(
                f"{row['candidate_name']}:{row['screen_reason']}" for row in screened_rows[:5]
            ),
        }
        return output

    def _deadband_vec(self) -> np.ndarray:
        return np.array(
            [self.scales.deadband_pitch_deg, self.scales.deadband_roll_deg],
            dtype=float,
        )

    def _build_segment_blob(self, forecast: ForecastResult) -> dict[str, dict[str, float]]:
        blob: dict[str, dict[str, float]] = {}
        uv = np.asarray(forecast.wind_uv_raw, dtype=float)
        speed = np.asarray(forecast.wind_speed, dtype=float)
        for seg_name, seg_slice, base_weight, _risk_name in SEGMENT_SPECS:
            uv_seg = uv[seg_slice]
            speed_seg = speed[seg_slice]
            mean_u = float(np.mean(uv_seg[:, 0]))
            mean_v = float(np.mean(uv_seg[:, 1]))
            mean_speed = float(np.mean(speed_seg))
            mean_dir = float((np.rad2deg(np.arctan2(-mean_u, -mean_v)) + 360.0) % 360.0)
            blob[seg_name] = {
                "mean_speed": mean_speed,
                "mean_dir_deg": mean_dir,
                "base_weight": float(base_weight),
            }
        return blob

    def _build_risk_blob(self, forecast: ForecastResult) -> dict[str, float]:
        event_probs = dict(forecast.event_probs)
        return {
            "global_risk": float(event_probs.get("ballast_attention_event", 0.0)),
            "0_20m": float(event_probs.get("attention_event_0_20m", 0.0)),
            "20_40m": float(event_probs.get("attention_event_20_40m", 0.0)),
            "40_60m": float(event_probs.get("attention_event_40_60m", 0.0)),
        }

    def _desired_setpoint_vector(
        self,
        current_ws: float,
        current_wd: float,
        current_pitch_deg: float,
        current_roll_deg: float,
        segment_blob: dict[str, dict[str, float]],
        risk_blob: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        current_vec = self._wind_to_setpoint_proxy(current_ws, current_wd)

        weighted_sum = np.zeros(2, dtype=float)
        weight_total = 0.0
        for seg_name, _seg_slice, base_weight, _risk_name in SEGMENT_SPECS:
            info = segment_blob[seg_name]
            risk_val = float(risk_blob[seg_name])
            seg_weight = float(base_weight) * (0.20 + risk_val)
            weighted_sum += seg_weight * self._wind_to_setpoint_proxy(
                ws=info["mean_speed"],
                wd_deg=info["mean_dir_deg"],
            )
            weight_total += seg_weight

        future_vec = weighted_sum / max(weight_total, self.scales.eps)
        delta_vec = future_vec - current_vec
        posture_vec = np.array(
            [
                -np.clip(
                    current_pitch_deg / max(self.scales.deadband_pitch_deg, self.scales.eps),
                    -1.5,
                    1.5,
                )
                * 0.35
                * self.scales.deadband_pitch_deg,
                -np.clip(
                    current_roll_deg / max(self.scales.deadband_roll_deg, self.scales.eps),
                    -1.5,
                    1.5,
                )
                * 0.35
                * self.scales.deadband_roll_deg,
            ],
            dtype=float,
        )
        desired_vec = (
            self.structure_profile.future_weight * delta_vec
            + self.structure_profile.posture_weight * posture_vec
        )

        deadband_vec = self._deadband_vec()
        delta_norm = float(np.linalg.norm(delta_vec / np.maximum(deadband_vec, self.scales.eps)))
        posture_norm = float(
            np.linalg.norm(
                np.array(
                    [
                        current_pitch_deg / max(self.scales.deadband_pitch_deg, self.scales.eps),
                        current_roll_deg / max(self.scales.deadband_roll_deg, self.scales.eps),
                    ],
                    dtype=float,
                )
            )
        )
        future_norm = float(np.linalg.norm(future_vec / np.maximum(deadband_vec, self.scales.eps)))
        coupling = 0.0
        future_mag = float(np.linalg.norm(delta_vec))
        posture_mag = float(np.linalg.norm(posture_vec))
        if future_mag > self.scales.eps and posture_mag > self.scales.eps:
            coupling = float(
                np.clip(
                    np.dot(delta_vec, posture_vec) / (future_mag * posture_mag),
                    -1.0,
                    1.0,
                )
            )
        aligned_coupling = max(0.0, coupling)
        activation_score = float(
            np.clip(
                self.structure_profile.activation_near * risk_blob["0_20m"]
                + self.structure_profile.activation_mid * risk_blob["20_40m"]
                + self.structure_profile.activation_far * risk_blob["40_60m"]
                + self.structure_profile.activation_delta * min(future_norm, 1.0)
                + self.structure_profile.activation_posture * min(posture_norm, 1.0),
                0.0,
                1.25,
            )
        )

        if risk_blob["0_20m"] >= 0.60:
            risk_level = "high"
        elif risk_blob["20_40m"] >= 0.40 or risk_blob["global_risk"] >= 0.45:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return desired_vec, {
            "activation_score": activation_score,
            "delta_norm": delta_norm,
            "posture_norm": posture_norm,
            "future_norm": future_norm,
            "posture_alignment": coupling,
            "aligned_coupling": aligned_coupling,
            "risk_level": risk_level,
        }

    def _build_candidates(
        self,
        desired_vec: np.ndarray,
        desired_meta: dict[str, Any],
        risk_blob: dict[str, float],
        plant_info_prev: dict[str, Any] | None,
    ) -> list[CandidatePlan]:
        desired_norm = float(np.linalg.norm(desired_vec))
        if desired_norm < self.scales.eps:
            direction = np.zeros(2, dtype=float)
        else:
            direction = desired_vec / desired_norm

        previous = self._last_selected.copy()
        if float(np.linalg.norm(previous)) > self.scales.eps:
            previous_dir = previous / max(float(np.linalg.norm(previous)), self.scales.eps)
        else:
            previous_dir = direction

        deadband_vec = self._deadband_vec()
        activation = float(np.clip(desired_meta.get("activation_score", 0.0), 0.0, 1.0))
        near_risk = float(np.clip(risk_blob.get("0_20m", 0.0), 0.0, 1.0))
        mid_risk = float(np.clip(risk_blob.get("20_40m", 0.0), 0.0, 1.0))
        far_risk = float(np.clip(risk_blob.get("40_60m", 0.0), 0.0, 1.0))
        near_weight = 0.70 * near_risk + 0.20 * mid_risk + 0.10 * far_risk
        desired_mag = float(np.linalg.norm(desired_vec))
        deadband_mag = float(np.linalg.norm(deadband_vec))
        posture_pressure = float(np.clip(desired_meta.get("posture_norm", 0.0), 0.0, 1.25))
        future_pressure = float(np.clip(desired_meta.get("future_norm", 0.0), 0.0, 1.25))
        aligned_coupling = float(np.clip(desired_meta.get("aligned_coupling", 0.0), 0.0, 1.0))
        active_trigger = float(
            np.clip(
                0.55 * near_risk
                + 0.20 * mid_risk
                + 0.10 * activation
                + 0.10 * min(posture_pressure, 1.0)
                + 0.05 * aligned_coupling,
                0.0,
                1.20,
            )
        )
        active_enable = float(
            np.clip(
                0.65 * near_risk
                + 0.15 * future_pressure
                + 0.15 * min(posture_pressure, 1.0)
                + 0.05 * aligned_coupling,
                0.0,
                1.25,
            )
        )
        active_blend = np.clip(
            self.structure_profile.active_blend_activation * activation
            + self.structure_profile.active_blend_near * near_weight
            + 0.15 * aligned_coupling,
            0.0,
            1.0,
        )
        active_ready = (
            near_risk >= 0.50
            and future_pressure >= 0.10
            and (posture_pressure >= 0.18 or aligned_coupling >= 0.25)
        )
        active_gate = 1.0 if (active_ready and active_enable >= 0.52) else 0.10
        active_cap = (
            self.active_ratio_low + (self.active_ratio_high - self.active_ratio_low) * active_blend
        ) * deadband_mag
        active_mag = min(active_cap, desired_mag * (0.70 + 0.45 * active_trigger))
        active_mag *= active_gate * (0.85 + 0.25 * aligned_coupling)
        if near_risk >= 0.65 and posture_pressure >= 0.35 and aligned_coupling >= 0.20:
            active_mag = max(
                active_mag,
                min(
                    active_cap,
                    (0.12 + 0.10 * active_trigger + 0.08 * aligned_coupling) * deadband_mag,
                ),
            )
        pump_saving_dir = 0.55 * direction + 0.45 * previous_dir
        pump_saving_norm = float(np.linalg.norm(pump_saving_dir))
        if pump_saving_norm > self.scales.eps:
            pump_saving_dir /= pump_saving_norm
        else:
            pump_saving_dir = np.zeros(2, dtype=float)
        pump_saving_cap = self.pump_saving_ratio * deadband_mag
        pump_saving_mag = min(
            pump_saving_cap,
            desired_mag * (0.28 + 0.15 * near_weight + 0.05 * min(posture_pressure, 1.0)),
        )
        if near_risk < 0.30 and desired_mag < 0.18 * deadband_mag:
            pump_saving_mag *= 0.65
        if int((plant_info_prev or {}).get("pump_fullspeed_any", 0)):
            pump_saving_mag *= 0.80
        if active_mag < 0.02 * deadband_mag:
            active_mag = 0.0
        if pump_saving_mag < 0.015 * deadband_mag:
            pump_saving_mag = 0.0
        plans = [
            CandidatePlan("hold", 0.0, 0.0, "hold"),
            CandidatePlan(
                "active_adjust",
                *(direction * active_mag),
                "risk",
            ),
            CandidatePlan(
                "pump_saving",
                *(pump_saving_dir * pump_saving_mag),
                "pump",
            ),
        ]
        cleaned: list[CandidatePlan] = []
        for plan in plans:
            pitch = float(np.clip(plan.pitch_sp_deg, -deadband_vec[0], deadband_vec[0]))
            roll = float(np.clip(plan.roll_sp_deg, -deadband_vec[1], deadband_vec[1]))
            cleaned.append(CandidatePlan(plan.name, pitch, roll, plan.family))
        return cleaned

    def _check_hard_constraints(
        self,
        candidate: CandidatePlan,
        plant_info_prev: dict[str, Any] | None,
    ) -> tuple[bool, str, dict[str, float]]:
        cand_vec = np.array([candidate.pitch_sp_deg, candidate.roll_sp_deg], dtype=float)
        deadband_vec = np.maximum(self._deadband_vec(), self.scales.eps)
        if np.any(np.abs(cand_vec) > deadband_vec + 1e-9):
            return False, "deadband_cap", {}

        tank_masses = np.asarray(
            (plant_info_prev or {}).get("tank_masses", np.zeros(3, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if tank_masses.size < 3:
            tank_masses = np.pad(tank_masses, (0, max(0, 3 - tank_masses.size)))
        mass_ratio = tank_masses[:3] / max(self.tank_capacity_kg, self.scales.eps)
        tank_signal = DEFAULT_ALLOC_MATRIX @ (cand_vec / deadband_vec)

        pushing_upper = np.any((mass_ratio >= self.upper_capacity_guard_ratio) & (tank_signal > 0.0))
        pulling_lower = np.any((mass_ratio <= self.lower_capacity_guard_ratio) & (tank_signal < 0.0))
        if pushing_upper or pulling_lower:
            return False, "capacity_guard", {
                "saturation_proxy": 1.0,
            }

        previous_norm = float(np.linalg.norm(self._last_selected / deadband_vec))
        if previous_norm > 0.05:
            delta_ratio = float(np.max(np.abs((cand_vec - self._last_selected) / deadband_vec)))
            if float(np.dot(cand_vec, self._last_selected)) < 0.0 and delta_ratio > self.reversal_guard_ratio:
                return False, "reversal_guard", {
                    "saturation_proxy": 0.0,
                }

        pump_fullspeed_any = int((plant_info_prev or {}).get("pump_fullspeed_any", 0))
        total_backlog = float((plant_info_prev or {}).get("pump_total_backlog_kg", 0.0))
        backlog_guard_kg = self.backlog_guard_multiplier * max(self.scales.backlog_kg, 1.0)
        if pump_fullspeed_any and candidate.name == "active_adjust":
            return False, "fullspeed_guard", {
                "saturation_proxy": 0.0,
            }
        if total_backlog >= backlog_guard_kg and candidate.name == "active_adjust":
            return False, "backlog_guard", {
                "saturation_proxy": min(total_backlog / max(backlog_guard_kg, 1.0), 2.0),
            }

        upper_margin = self.upper_capacity_guard_ratio - mass_ratio
        lower_margin = mass_ratio - self.lower_capacity_guard_ratio
        directional_margin = np.where(tank_signal > 0.0, upper_margin, lower_margin)
        min_margin = float(np.min(directional_margin))
        stress = float(np.max(np.abs(tank_signal)))
        margin_scale = max(self.scales.saturation_margin_ratio, self.scales.eps)
        saturation_proxy = max(0.0, 1.0 - min_margin / margin_scale) * stress

        return True, "", {
            "saturation_proxy": saturation_proxy,
        }

    def _compute_attitude_risk(
        self,
        candidate: CandidatePlan,
        desired_vec: np.ndarray,
        current_pitch_deg: float,
        current_roll_deg: float,
        risk_blob: dict[str, float],
    ) -> dict[str, float | str | bool]:
        cand_vec = np.array([candidate.pitch_sp_deg, candidate.roll_sp_deg], dtype=float)
        deadband_vec = np.maximum(self._deadband_vec(), self.scales.eps)
        posture_vec = np.array(
            [
                -np.clip(
                    current_pitch_deg / max(self.scales.deadband_pitch_deg, self.scales.eps),
                    -1.5,
                    1.5,
                )
                * 0.35
                * self.scales.deadband_pitch_deg,
                -np.clip(
                    current_roll_deg / max(self.scales.deadband_roll_deg, self.scales.eps),
                    -1.5,
                    1.5,
                )
                * 0.35
                * self.scales.deadband_roll_deg,
            ],
            dtype=float,
        )
        desired_residual = np.abs(desired_vec - cand_vec)
        posture_residual = np.abs(posture_vec - cand_vec)
        near_pressure = float(
            np.clip(
                0.65 * float(risk_blob["0_20m"])
                + 0.20 * float(risk_blob["20_40m"])
                + 0.15 * float(risk_blob["global_risk"]),
                0.0,
                1.25,
            )
        )
        pitch_proxy_deg = float(0.65 * posture_residual[0] + 0.35 * near_pressure * desired_residual[0])
        roll_proxy_deg = float(0.65 * posture_residual[1] + 0.35 * near_pressure * desired_residual[1])
        desired_gap = float(np.linalg.norm((desired_vec - cand_vec) / deadband_vec))
        residual_posture = float(np.linalg.norm((posture_vec - cand_vec) / deadband_vec))
        exceed_norm = float(np.clip(0.55 * near_pressure * desired_gap + 0.45 * residual_posture, 0.0, 2.0))
        exceed_proxy_s = float(exceed_norm * self.attitude_bands.exceed_time_band_s)
        pitch_risk = pitch_proxy_deg / max(self.attitude_bands.pitch_rms_band_deg, self.attitude_bands.eps)
        roll_risk = roll_proxy_deg / max(self.attitude_bands.roll_rms_band_deg, self.attitude_bands.eps)
        exceed_risk = exceed_proxy_s / max(self.attitude_bands.exceed_time_band_s, self.attitude_bands.eps)
        risk_components = {
            "pitch_rms_band": float(pitch_risk),
            "roll_rms_band": float(roll_risk),
            "pose_exceed_band": float(exceed_risk),
        }
        attitude_risk = max(risk_components.values())
        reject_reason = ""
        if attitude_risk > 1.0:
            reject_reason = max(risk_components, key=risk_components.get)
        return {
            "pitch_rms_proxy_deg": pitch_proxy_deg,
            "roll_rms_proxy_deg": roll_proxy_deg,
            "pose_exceed_time_proxy_s": exceed_proxy_s,
            "attitude_risk": float(attitude_risk),
            "reject_reason": reject_reason,
            "eligible": bool(attitude_risk <= 1.0),
        }

    def _attitude_eligible(
        self,
        candidate: CandidatePlan,
        desired_vec: np.ndarray,
        desired_meta: dict[str, Any],
        current_pitch_deg: float,
        current_roll_deg: float,
        risk_blob: dict[str, float],
    ) -> dict[str, float | str | bool]:
        _ = desired_meta
        return self._compute_attitude_risk(
            candidate=candidate,
            desired_vec=desired_vec,
            current_pitch_deg=current_pitch_deg,
            current_roll_deg=current_roll_deg,
            risk_blob=risk_blob,
        )

    def _pump_burden_tuple(
        self,
        candidate: CandidatePlan,
        plant_info_prev: dict[str, Any] | None,
        proxy: dict[str, float],
        score_row: dict[str, Any],
    ) -> tuple[float, float, float, float]:
        pump_rate_cmd = (plant_info_prev or {}).get("pump_rate_cmd_m3_min", [0.0, 0.0, 0.0])
        pump_total_rate = float(np.sum(np.asarray(pump_rate_cmd, dtype=float)))
        pump_active = pump_total_rate > self.scales.eps
        if candidate.name == "hold":
            startstop_count = 0.0
        elif candidate.name == "pump_saving":
            startstop_count = 1.0 if pump_active else 1.75
        else:
            startstop_count = 1.5 if pump_active else 2.5

        deadband_vec = np.maximum(self._deadband_vec(), self.scales.eps)
        action_ratio = float(score_row.get("action_ratio", 0.0))
        cand_vec = np.array([candidate.pitch_sp_deg, candidate.roll_sp_deg], dtype=float)
        cand_norm = float(np.linalg.norm(cand_vec))
        previous_norm = float(np.linalg.norm(self._last_selected / deadband_vec))
        if previous_norm > self.scales.eps and action_ratio > self.scales.eps and cand_norm > self.scales.eps:
            direction_switch = max(
                0.0,
                -float(np.dot(cand_vec, self._last_selected))
                / (cand_norm * max(float(np.linalg.norm(self._last_selected)), self.scales.eps)),
            )
        else:
            direction_switch = 0.0
        if candidate.name == "active_adjust":
            direction_switch += 0.15

        pump_work = float(score_row.get("cost_pump", np.inf))
        sat_ratio = float(proxy.get("saturation_proxy", 0.0))
        return (float(startstop_count), float(direction_switch), float(pump_work), float(sat_ratio))

    def _select_constraint_first(
        self,
        feasible_rows: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        eligible_rows = [row for row in feasible_rows if int(row.get("attitude_eligible", 0)) == 1]
        if eligible_rows:
            eligible_rows.sort(
                key=lambda row: (
                    tuple(row.get("pump_rank_tuple", (np.inf, np.inf, np.inf, np.inf))),
                    float(row.get("attitude_risk", np.inf)),
                    float(row.get("cost_total", np.inf)),
                )
            )
            selected = eligible_rows[0]
            ranked_rows = sorted(
                feasible_rows,
                key=lambda row: (
                    0 if int(row.get("attitude_eligible", 0)) == 1 else 1,
                    tuple(row.get("pump_rank_tuple", (np.inf, np.inf, np.inf, np.inf))),
                    float(row.get("attitude_risk", np.inf)),
                    float(row.get("cost_total", np.inf)),
                ),
            )
            return selected, ranked_rows

        ranked_rows = sorted(
            feasible_rows,
            key=lambda row: (
                float(row.get("attitude_risk", np.inf)),
                tuple(row.get("pump_rank_tuple", (np.inf, np.inf, np.inf, np.inf))),
                float(row.get("cost_total", np.inf)),
            ),
        )
        return ranked_rows[0], ranked_rows

    def _score_candidate(
        self,
        candidate: CandidatePlan,
        desired_vec: np.ndarray,
        current_pitch_deg: float,
        current_roll_deg: float,
        plant_info_prev: dict[str, Any] | None,
        risk_blob: dict[str, float],
        proxy: dict[str, float],
    ) -> dict[str, Any]:
        cand_vec = np.array([candidate.pitch_sp_deg, candidate.roll_sp_deg], dtype=float)
        deadband_vec = np.maximum(self._deadband_vec(), self.scales.eps)
        cand_norm = float(np.linalg.norm(cand_vec))
        desired_norm = float(np.linalg.norm(desired_vec))
        if cand_norm > self.scales.eps and desired_norm > self.scales.eps:
            alignment = float(np.dot(cand_vec, desired_vec) / (cand_norm * desired_norm))
        else:
            alignment = 0.0

        desired_gap = float(np.linalg.norm((desired_vec - cand_vec) / deadband_vec))
        action_ratio = float(np.linalg.norm(cand_vec / deadband_vec))
        delta_ratio = float(np.linalg.norm((cand_vec - self._last_selected) / deadband_vec))

        posture_severity = float(
            np.linalg.norm(
                np.array(
                    [
                        current_pitch_deg / max(self.scales.deadband_pitch_deg, self.scales.eps),
                        current_roll_deg / max(self.scales.deadband_roll_deg, self.scales.eps),
                    ],
                    dtype=float,
                )
            )
        )
        posture_vec = np.array(
            [
                -np.clip(
                    current_pitch_deg / max(self.scales.deadband_pitch_deg, self.scales.eps),
                    -1.5,
                    1.5,
                )
                * 0.35
                * self.scales.deadband_pitch_deg,
                -np.clip(
                    current_roll_deg / max(self.scales.deadband_roll_deg, self.scales.eps),
                    -1.5,
                    1.5,
                )
                * 0.35
                * self.scales.deadband_roll_deg,
            ],
            dtype=float,
        )
        residual_posture = float(np.linalg.norm((posture_vec - cand_vec) / deadband_vec))

        global_risk = float(risk_blob["global_risk"])
        near_risk = float(risk_blob["0_20m"])
        mid_risk = float(risk_blob["20_40m"])
        far_risk = float(risk_blob["40_60m"])
        near_pressure = float(np.clip(0.65 * near_risk + 0.20 * mid_risk + 0.15 * global_risk, 0.0, 1.25))
        risk_multiplier = 1.0 + self.weight_profile.gamma_risk * (
            self.structure_profile.risk_mult_near * near_risk
            + self.structure_profile.risk_mult_mid * mid_risk
            + self.structure_profile.risk_mult_far * far_risk
        )
        posture_alignment = 0.0
        posture_mag = float(np.linalg.norm(posture_vec))
        if cand_norm > self.scales.eps and posture_mag > self.scales.eps:
            posture_alignment = float(
                np.clip(
                    np.dot(cand_vec, posture_vec) / (cand_norm * posture_mag),
                    -1.0,
                    1.0,
                )
            )
        aligned_posture = max(0.0, posture_alignment)

        attitude_cost = self.weight_profile.lambda_att * risk_multiplier * (
            0.50 * desired_gap
            + 0.15 * ((1.0 - alignment) / 2.0)
            + 0.25 * near_pressure * residual_posture
            + 0.10 * max(0.0, desired_norm - action_ratio)
        ) * (1.0 + 0.20 * posture_severity)
        if candidate.name == "hold":
            attitude_cost *= 1.0 + 0.55 * near_pressure + 0.18 * max(0.0, posture_severity - 0.25)
        elif candidate.name == "active_adjust":
            attitude_cost *= max(0.55, 1.0 - 0.32 * near_pressure - 0.10 * aligned_posture)
        elif candidate.name == "pump_saving":
            attitude_cost *= 1.0 + 0.14 * max(0.0, near_pressure - 0.30)

        pump_fullspeed_any = int((plant_info_prev or {}).get("pump_fullspeed_any", 0))
        pump_rate_cmd = (plant_info_prev or {}).get("pump_rate_cmd_m3_min", [0.0, 0.0, 0.0])
        pump_total_rate = float(np.sum(np.asarray(pump_rate_cmd, dtype=float)))
        total_backlog = float((plant_info_prev or {}).get("pump_total_backlog_kg", 0.0))
        pump_rate_ratio = min(
            pump_total_rate / max(self.scales.pump_rate_m3_min, self.scales.eps),
            3.0,
        )
        backlog_ratio = min(
            total_backlog / max(self.scales.backlog_kg, 1.0),
            3.0,
        )
        pump_context = 1.0 + 0.20 * pump_rate_ratio + 0.15 * backlog_ratio + 0.10 * pump_fullspeed_any
        pump_cost = self.weight_profile.lambda_pump * (
            0.75 * action_ratio * pump_context
            + 0.15 * action_ratio * max(0.0, 1.0 - global_risk)
            + 0.10 * pump_fullspeed_any * action_ratio
        )
        if candidate.name == "hold":
            pump_cost = self.weight_profile.lambda_pump * (
                0.03 * (pump_rate_ratio + 0.6 * backlog_ratio + 0.3 * pump_fullspeed_any)
            )
        if candidate.name == "active_adjust":
            pump_cost *= max(0.75, 1.0 - 0.20 * near_pressure - 0.08 * aligned_posture)
        elif candidate.name == "pump_saving":
            pump_cost *= 0.70

        previous_norm = float(np.linalg.norm(self._last_selected / deadband_vec))
        if previous_norm > self.scales.eps and action_ratio > self.scales.eps:
            reversal_score = max(
                0.0,
                -float(np.dot(cand_vec, self._last_selected))
                / (cand_norm * max(float(np.linalg.norm(self._last_selected)), self.scales.eps)),
            )
        else:
            reversal_score = 0.0
        switch_cost = self.weight_profile.lambda_switch * (
            0.60 * delta_ratio + 0.40 * reversal_score
        )
        if candidate.name == "hold":
            switch_cost *= 0.35
        if candidate.name == "active_adjust":
            switch_cost *= max(0.80, 1.0 - 0.15 * near_pressure - 0.05 * aligned_posture)
        elif candidate.name == "pump_saving":
            switch_cost *= 0.72

        saturation_cost = self.weight_profile.lambda_sat * float(proxy.get("saturation_proxy", 0.0))

        total_cost = attitude_cost + pump_cost + switch_cost + saturation_cost
        return {
            "candidate_name": candidate.name,
            "pitch_sp_deg": float(candidate.pitch_sp_deg),
            "roll_sp_deg": float(candidate.roll_sp_deg),
            "cost_total": float(total_cost),
            "cost_attitude": float(attitude_cost),
            "cost_pump": float(pump_cost),
            "cost_switch": float(switch_cost),
            "cost_saturation": float(saturation_cost),
            "alignment": float(alignment),
            "action_ratio": float(action_ratio),
            "feasible": 1,
        }

    def _wind_to_setpoint_proxy(self, ws: float, wd_deg: float) -> np.ndarray:
        speed_scale = self.wind_reference_mps
        magnitude = float(np.clip((max(float(ws), 0.0) / speed_scale) ** 2, 0.0, 1.5))
        wd_rad = np.radians(float(wd_deg))
        deadband_vec = self._deadband_vec()
        return np.array(
            [
                -deadband_vec[0] * magnitude * np.cos(wd_rad),
                deadband_vec[1] * magnitude * np.sin(wd_rad),
            ],
            dtype=float,
        )
