"""Experiment resources for the source-bound real-LSTM rotor preview.

Paths and frozen reference identities belong to the validation layer.  This
module loads them and delegates all physical conversion to typed library
functions.  It contains no ballast policy or platform execution logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from fowt_platform import (
    IncrementalState,
    NominalBelowRatedRotorSchedule,
    RotorGeneralizedLoad,
    RotorPerformanceTable,
    enu_wind_to_platform,
    inspect_nominal_below_rated_operating_inputs,
    load_rosco_nominal_below_rated_schedule_from_zip,
    load_rosco_rotor_performance_table_from_zip,
)
from wind_prediction.controller_replay_adapter import ReplayForecastEvidenceSource
from wind_prediction.forecast_adapter import ForecastModelAdapter
from wind_prediction.forecast_evidence import (
    ForecastEvidence,
    lead_reliability_from_metrics,
    validate_forecast_evidence,
)
from wind_prediction.forecast_physical_load import ForecastRotorLoadParameters
from wind_prediction.forecast_rotor_inflow import (
    ForecastRotorKinematics,
    assemble_forecast_rotor_relative_inflow,
)
from wind_prediction.replay_dataset import Fino1ReplayDataset
from wind_prediction.run_identity import sha256_json
from wind_prediction.source_bound_rotor_preview import (
    SourceBoundRotorPreview,
    assemble_source_bound_nominal_rotor_preview,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIRECTORY = (
    ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
)
MODEL_DIRECTORY = (
    ROOT / "outputs/wind_prediction/lstm_segmented_fino1_meteo_aux_v1"
)
REFERENCE_MANIFEST = (
    ROOT / "configs/reference_platforms/volturnus_s_openfast_v1_1_16.json"
)
ROTOR_PREVIEW_MANIFEST = (
    ROOT
    / "configs/reference_platforms/iea15mw_nominal_rotor_preview_v1_1_16.json"
)
REFERENCE_TANK_MASSES_KG = np.array(
    [1_108_000.0, 1_362_000.0, 1_362_000.0]
)
TANK_CAPACITY_KG = 1_896_250.0
TANK_COORDINATES_M = np.array(
    [
        [46.2, 0.0, -10.0],
        [-23.1, 46.2 * 0.866, -10.0],
        [-23.1, -46.2 * 0.866, -10.0],
    ],
    dtype=float,
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, Any]:
    with ROTOR_PREVIEW_MANIFEST.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported rotor preview manifest schema")
    source_model = manifest["source_model"]
    archive = ROOT / source_model["archive_path"]
    expected_archive_sha256 = source_model["archive_sha256"]
    if _file_sha256(archive) != expected_archive_sha256:
        raise ValueError("rotor preview source archive SHA-256 mismatch")
    reference = manifest["reference_state"]
    if _file_sha256(ROOT / reference["path"]) != reference["sha256"]:
        raise ValueError("rotor preview reference-state SHA-256 mismatch")
    return manifest


@dataclass(frozen=True)
class RealLstmPreviewResources:
    """Read-only model and source resources shared across replay origins."""

    device: str
    replay: Fino1ReplayDataset
    forecast_adapter: ForecastModelAdapter
    lead_reliability: np.ndarray
    performance_table: RotorPerformanceTable
    nominal_schedule: NominalBelowRatedRotorSchedule
    rotor_parameters: ForecastRotorLoadParameters
    operating_assumptions: dict[str, Any]
    manifest_sha256: str


@dataclass(frozen=True)
class RealLstmWindRecord:
    """One immutable LSTM forecast and its same-origin wind observation.

    The record is independent of platform motion. It can therefore be inferred
    once and converted under the realised state of each controller variant
    without rerunning the neural network.
    """

    forecast: ForecastEvidence
    current_enu_downwind_wind_mps: Any

    def __post_init__(self) -> None:
        validate_forecast_evidence(self.forecast)
        current = np.asarray(self.current_enu_downwind_wind_mps, dtype=float)
        if current.shape != (2,) or not np.all(np.isfinite(current)):
            raise ValueError(
                "current_enu_downwind_wind_mps must have shape (2,) and be finite"
            )
        stored = np.array(current, dtype=float, copy=True)
        stored.setflags(write=False)
        object.__setattr__(self, "current_enu_downwind_wind_mps", stored)

    def source_record_payload(self) -> dict[str, Any]:
        forecast = self.forecast
        return {
            "forecast": {
                "source": forecast.source,
                "model_version": forecast.model_version,
                "origin_time": forecast.origin_time,
                "sample_period_s": float(forecast.sample_period_s),
                "uv_ms": np.asarray(forecast.uv_ms, dtype=float).tolist(),
                "event_probs": {
                    str(key): float(value)
                    for key, value in sorted(forecast.event_probs.items())
                },
                "lead_reliability": np.asarray(
                    forecast.lead_reliability,
                    dtype=float,
                ).tolist(),
                "provides_future_preview": bool(
                    forecast.provides_future_preview
                ),
                "metadata": dict(forecast.metadata),
            },
            "current_observation": {
                "enu_downwind_wind_mps": (
                    self.current_enu_downwind_wind_mps.tolist()
                ),
                "source": "fino1_replay_origin_observation",
                "observation_time": str(forecast.origin_time),
            },
        }

    @property
    def source_record_sha256(self) -> str:
        return sha256_json(self.source_record_payload())


PlannerWindForecastMode = Literal[
    "current_observation",
    "lstm",
    "persistence",
    "recorded_oracle",
]


def planner_forecast_evidence_from_record(
    *,
    source_record: RealLstmWindRecord,
    planner_forecast_mode: PlannerWindForecastMode,
    horizon_blocks: int,
    oracle_records: Sequence[RealLstmWindRecord] | None = None,
) -> ForecastEvidence:
    """Select future wind content before any platform-load conversion."""

    if not isinstance(source_record, RealLstmWindRecord):
        raise TypeError("source_record must be RealLstmWindRecord")
    if int(horizon_blocks) != horizon_blocks or horizon_blocks <= 0:
        raise ValueError("horizon_blocks must be a positive integer")
    horizon = int(horizon_blocks)
    source = source_record.forecast
    current = source_record.current_enu_downwind_wind_mps
    if planner_forecast_mode == "current_observation":
        if horizon != 1:
            raise ValueError(
                "current_observation mode requires exactly one control block"
            )
        uv = current[None, :]
        reliability = np.ones(1, dtype=float)
        evidence_source = "fino1_current_observation"
        model_version = "current_observation_only"
        provides_future_preview = True
        event_probs: dict[str, float] = {}
        metadata = {
            "planner_forecast_mode": planner_forecast_mode,
            "is_learned_future_prediction": False,
        }
    elif planner_forecast_mode == "lstm":
        if horizon > source.horizon_steps:
            raise ValueError("horizon_blocks exceeds the LSTM forecast horizon")
        uv = source.uv_ms[:horizon]
        reliability = source.lead_reliability[:horizon]
        evidence_source = source.source
        model_version = source.model_version
        provides_future_preview = True
        event_probs = dict(source.event_probs)
        metadata = {
            **dict(source.metadata),
            "planner_forecast_mode": planner_forecast_mode,
        }
    elif planner_forecast_mode == "persistence":
        if horizon > source.horizon_steps:
            raise ValueError("horizon_blocks exceeds the available forecast grid")
        uv = np.repeat(current[None, :], horizon, axis=0)
        reliability = np.ones(horizon, dtype=float)
        evidence_source = "fino1_observation_persistence"
        model_version = "observation_persistence"
        provides_future_preview = True
        event_probs = {}
        metadata = {
            "planner_forecast_mode": planner_forecast_mode,
            "is_learned_future_prediction": False,
        }
    elif planner_forecast_mode == "recorded_oracle":
        if oracle_records is None or len(oracle_records) < horizon:
            raise ValueError(
                "recorded_oracle mode requires one future source record per lead"
            )
        selected = tuple(oracle_records[:horizon])
        if not all(isinstance(record, RealLstmWindRecord) for record in selected):
            raise TypeError("oracle_records must contain RealLstmWindRecord values")
        uv = np.vstack(
            [record.current_enu_downwind_wind_mps for record in selected]
        )
        reliability = np.ones(horizon, dtype=float)
        evidence_source = "fino1_recorded_future_oracle"
        model_version = "noncausal_recorded_future_diagnostic"
        provides_future_preview = True
        event_probs = {}
        metadata = {
            "planner_forecast_mode": planner_forecast_mode,
            "noncausal_diagnostic": True,
            "recorded_future_origin_times": [
                str(record.forecast.origin_time) for record in selected
            ],
        }
    else:
        raise ValueError(f"unsupported planner_forecast_mode={planner_forecast_mode!r}")

    return ForecastEvidence(
        source=evidence_source,
        model_version=model_version,
        origin_time=source.origin_time,
        sample_period_s=source.sample_period_s,
        uv_ms=uv,
        event_probs=event_probs,
        lead_reliability=reliability,
        provides_future_preview=provides_future_preview,
        metadata=metadata,
    )


def prepare_real_lstm_preview_resources(
    *,
    device: str = "cpu",
) -> RealLstmPreviewResources:
    manifest = _load_manifest()
    assumptions = dict(manifest["operating_assumptions"])
    if assumptions.get("current_nacelle_yaw_mode") not in {
        "fixed_relative_platform",
        "align_with_current_horizontal_wind_at_control_update",
    }:
        raise ValueError("unsupported current_nacelle_yaw_mode")
    if assumptions.get("future_platform_kinematics_mode") != "frozen_zero":
        raise ValueError(
            "the current real-LSTM fixture supports only explicit frozen-zero "
            "future platform kinematics"
        )
    if assumptions.get("future_rotor_orientation_mode") != "frozen_current":
        raise ValueError(
            "the current real-LSTM fixture supports only frozen-current future "
            "rotor orientation"
        )
    replay = Fino1ReplayDataset(DATASET_DIRECTORY, split="test")
    adapter = ForecastModelAdapter(MODEL_DIRECTORY, device=device)
    expected_lead_minutes = tuple(
        int(round(step * replay.update_interval_s / 60.0))
        for step in range(1, int(replay.future_steps) + 1)
    )
    lead_reliability = lead_reliability_from_metrics(
        MODEL_DIRECTORY / "lstm_regression_metrics.csv",
        expected_lead_minutes=expected_lead_minutes,
    )
    lead_reliability.setflags(write=False)
    source_model = manifest["source_model"]
    archive = ROOT / source_model["archive_path"]
    archive_sha256 = source_model["archive_sha256"]
    performance_table = load_rosco_rotor_performance_table_from_zip(
        archive,
        source_model["rotor_performance_member"],
        expected_archive_sha256=archive_sha256,
    )
    nominal_schedule = load_rosco_nominal_below_rated_schedule_from_zip(
        archive,
        source_model["rosco_tuning_member"],
        expected_archive_sha256=archive_sha256,
    )
    return RealLstmPreviewResources(
        device=str(device),
        replay=replay,
        forecast_adapter=adapter,
        lead_reliability=lead_reliability,
        performance_table=performance_table,
        nominal_schedule=nominal_schedule,
        rotor_parameters=ForecastRotorLoadParameters(**manifest["load_parameters"]),
        operating_assumptions=assumptions,
        manifest_sha256=_file_sha256(ROTOR_PREVIEW_MANIFEST),
    )


def infer_real_lstm_wind_record(
    *,
    resources: RealLstmPreviewResources,
    origin: datetime,
) -> RealLstmWindRecord:
    """Run the LSTM once and bind its output to the same-origin observation."""

    if not isinstance(resources, RealLstmPreviewResources):
        raise TypeError("resources must be RealLstmPreviewResources")
    source = ReplayForecastEvidenceSource(
        replay_dataset=resources.replay,
        start_timestamp=origin,
        forecast_adapter=resources.forecast_adapter,
        lead_reliability=resources.lead_reliability,
    )
    source_bound = source.source_bound_forecast_and_current_enu_wind(0.0)
    if source_bound is None:
        raise ValueError("no replay sample is available at the requested origin")
    forecast, current_wind = source_bound
    return RealLstmWindRecord(
        forecast=forecast,
        current_enu_downwind_wind_mps=current_wind,
    )


def _current_nacelle_yaw_relative_platform_rad(
    *,
    resources: RealLstmPreviewResources,
    current_enu_downwind_wind_mps: Any,
) -> float:
    """Resolve the explicit cycle-start nacelle orientation assumption."""

    assumptions = resources.operating_assumptions
    mode = assumptions["current_nacelle_yaw_mode"]
    if mode == "fixed_relative_platform":
        return float(assumptions["current_nacelle_yaw_relative_platform_rad"])
    horizontal = enu_wind_to_platform(
        np.asarray(current_enu_downwind_wind_mps, dtype=float),
        resources.rotor_parameters.frozen_equilibrium_heading_rad,
    )
    if np.linalg.norm(horizontal) <= 1.0e-12:
        raise ValueError("current horizontal wind is too small to define nacelle yaw")
    reference_normal = resources.rotor_parameters.downwind_rotor_normal_platform
    return float(
        np.arctan2(horizontal[1], horizontal[0])
        - np.arctan2(reference_normal[1], reference_normal[0])
    )


def _kinematics_for_platform_state(
    *,
    resources: RealLstmPreviewResources,
    forecast: ForecastEvidence,
    platform_state: IncrementalState,
    current_enu_downwind_wind_mps: Any,
    current_nacelle_yaw_override_rad: float | None = None,
) -> ForecastRotorKinematics:
    assumptions = resources.operating_assumptions
    horizon = forecast.horizon_steps
    return ForecastRotorKinematics(
        current_platform_reference_velocity_platform_mps=(
            platform_state.velocity[:3]
        ),
        current_platform_angular_velocity_platform_radps=(
            platform_state.velocity[3:]
        ),
        future_platform_reference_velocities_platform_mps=np.zeros((horizon, 3)),
        future_platform_angular_velocities_platform_radps=np.zeros((horizon, 3)),
        future_platform_kinematics_mode=(
            assumptions["future_platform_kinematics_mode"]
        ),
        future_platform_kinematics_source=(
            assumptions["future_platform_kinematics_source"]
        ),
        current_nacelle_yaw_relative_platform_rad=(
            _current_nacelle_yaw_relative_platform_rad(
                resources=resources,
                current_enu_downwind_wind_mps=current_enu_downwind_wind_mps,
            )
            if current_nacelle_yaw_override_rad is None
            else float(current_nacelle_yaw_override_rad)
        ),
        future_rotor_orientation_mode=(
            assumptions["future_rotor_orientation_mode"]
        ),
        future_rotor_orientation_source=(
            assumptions["future_rotor_orientation_source"]
        ),
    )


def inspect_real_lstm_wind_record_operating_domain(
    *,
    resources: RealLstmPreviewResources,
    source_record: RealLstmWindRecord,
    platform_state: IncrementalState,
    include_future: bool = True,
) -> dict[str, Any]:
    """Inspect source support without converting unsupported points to loads."""

    kinematics = _kinematics_for_platform_state(
        resources=resources,
        forecast=source_record.forecast,
        platform_state=platform_state,
        current_enu_downwind_wind_mps=(
            source_record.current_enu_downwind_wind_mps
        ),
    )
    relative_inflow = assemble_forecast_rotor_relative_inflow(
        forecast=source_record.forecast,
        current_enu_downwind_air_velocity_mps=(
            source_record.current_enu_downwind_wind_mps
        ),
        current_wind_source="fino1_replay_origin_observation",
        current_wind_observation_time=str(source_record.forecast.origin_time),
        parameters=resources.rotor_parameters,
        kinematics=kinematics,
    )
    report = inspect_nominal_below_rated_operating_inputs(
        schedule=resources.nominal_schedule,
        performance_table=resources.performance_table,
        rotor_radius_m=resources.rotor_parameters.rotor_radius_m,
        current_normal_inflow_speed_mps=(
            relative_inflow.current_normal_relative_inflow_mps
        ),
        future_normal_inflow_speeds_mps=(
            relative_inflow.future_normal_relative_inflows_mps
        ),
    )
    if include_future:
        return {**report, "checked_input_scope": "current_and_forecast"}
    current = report["records"][:1]
    return {
        **report,
        "records": current,
        "supported": bool(current[0]["supported"]),
        "unsupported_record_count": int(not bool(current[0]["supported"])),
        "checked_input_scope": "current_only",
    }


def assemble_real_lstm_rotor_preview_from_record(
    *,
    resources: RealLstmPreviewResources,
    source_record: RealLstmWindRecord,
    platform_state: IncrementalState,
) -> SourceBoundRotorPreview:
    """Convert one already-bound wind record under the current platform state."""

    if not isinstance(resources, RealLstmPreviewResources):
        raise TypeError("resources must be RealLstmPreviewResources")
    if not isinstance(source_record, RealLstmWindRecord):
        raise TypeError("source_record must be RealLstmWindRecord")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be IncrementalState")
    forecast = source_record.forecast
    current_wind = source_record.current_enu_downwind_wind_mps
    assumptions = resources.operating_assumptions
    kinematics = _kinematics_for_platform_state(
        resources=resources,
        forecast=forecast,
        platform_state=platform_state,
        current_enu_downwind_wind_mps=current_wind,
    )
    preview = assemble_source_bound_nominal_rotor_preview(
        forecast=forecast,
        current_enu_downwind_wind_mps=current_wind,
        current_wind_source="fino1_replay_origin_observation",
        current_wind_observation_time=str(forecast.origin_time),
        platform_state=platform_state,
        kinematics=kinematics,
        parameters=resources.rotor_parameters,
        schedule=resources.nominal_schedule,
        performance_table=resources.performance_table,
        operating_mode_source=assumptions["operating_mode_source"],
    )
    if preview.source_record_sha256 != source_record.source_record_sha256:
        raise RuntimeError(
            "rotor-load conversion changed the bound wind source record"
        )
    return preview


def assemble_planner_rotor_preview_from_record(
    *,
    resources: RealLstmPreviewResources,
    source_record: RealLstmWindRecord,
    platform_state: IncrementalState,
    planner_forecast_mode: PlannerWindForecastMode,
    horizon_blocks: int,
    oracle_records: Sequence[RealLstmWindRecord] | None = None,
) -> SourceBoundRotorPreview:
    """Convert one selected planner wind sequence under common assumptions."""

    forecast = planner_forecast_evidence_from_record(
        source_record=source_record,
        planner_forecast_mode=planner_forecast_mode,
        horizon_blocks=horizon_blocks,
        oracle_records=oracle_records,
    )
    kinematics = _kinematics_for_platform_state(
        resources=resources,
        forecast=forecast,
        platform_state=platform_state,
        current_enu_downwind_wind_mps=(
            source_record.current_enu_downwind_wind_mps
        ),
    )
    assumptions = resources.operating_assumptions
    return assemble_source_bound_nominal_rotor_preview(
        forecast=forecast,
        current_enu_downwind_wind_mps=(
            source_record.current_enu_downwind_wind_mps
        ),
        current_wind_source="fino1_replay_origin_observation",
        current_wind_observation_time=str(source_record.forecast.origin_time),
        platform_state=platform_state,
        kinematics=kinematics,
        parameters=resources.rotor_parameters,
        schedule=resources.nominal_schedule,
        performance_table=resources.performance_table,
        operating_mode_source=assumptions["operating_mode_source"],
    )


def assemble_recorded_interval_rotor_load_substeps(
    *,
    resources: RealLstmPreviewResources,
    current_record: RealLstmWindRecord,
    following_record: RealLstmWindRecord,
    platform_state: IncrementalState,
    substep_count: int,
) -> tuple[RotorGeneralizedLoad, ...]:
    """Reconstruct realised rotor loads between adjacent wind observations.

    Horizontal ENU wind vectors are linearly reconstructed at physical-substep
    midpoints and then passed through the same relative-inflow, operating-state
    and rotor-load conversion used by controller-visible forecasts.  The
    following observation is plant-only evidence and must be attached only
    after the current control request has been selected.
    """

    if not isinstance(resources, RealLstmPreviewResources):
        raise TypeError("resources must be RealLstmPreviewResources")
    if not isinstance(current_record, RealLstmWindRecord) or not isinstance(
        following_record,
        RealLstmWindRecord,
    ):
        raise TypeError("current_record and following_record must be wind records")
    if not isinstance(platform_state, IncrementalState):
        raise TypeError("platform_state must be IncrementalState")
    if int(substep_count) != substep_count or substep_count <= 0:
        raise ValueError("substep_count must be a positive integer")
    count = int(substep_count)
    current_time = datetime.strptime(
        str(current_record.forecast.origin_time),
        "%Y-%m-%d %H:%M:%S",
    )
    following_time = datetime.strptime(
        str(following_record.forecast.origin_time),
        "%Y-%m-%d %H:%M:%S",
    )
    sample_period_s = float(current_record.forecast.sample_period_s)
    if abs((following_time - current_time).total_seconds() - sample_period_s) > 1.0e-9:
        raise ValueError("wind records must be adjacent on the forecast sample grid")

    start_wind = current_record.current_enu_downwind_wind_mps
    end_wind = following_record.current_enu_downwind_wind_mps
    interval_nacelle_yaw = _current_nacelle_yaw_relative_platform_rad(
        resources=resources,
        current_enu_downwind_wind_mps=start_wind,
    )
    loads: list[RotorGeneralizedLoad] = []
    assumptions = resources.operating_assumptions
    for index in range(count):
        fraction = (index + 0.5) / count
        wind = (1.0 - fraction) * start_wind + fraction * end_wind
        evidence = ForecastEvidence(
            source="fino1_recorded_interval_reconstruction",
            model_version="linear_enu_midpoint_v1",
            origin_time=str(current_record.forecast.origin_time),
            sample_period_s=sample_period_s / count,
            uv_ms=np.asarray(wind, dtype=float)[None, :],
            event_probs={},
            lead_reliability=np.ones(1, dtype=float),
            provides_future_preview=True,
            metadata={
                "plant_only_realised_disturbance": True,
                "interpolation_fraction": fraction,
            },
        )
        kinematics = _kinematics_for_platform_state(
            resources=resources,
            forecast=evidence,
            platform_state=platform_state,
            current_enu_downwind_wind_mps=wind,
            current_nacelle_yaw_override_rad=interval_nacelle_yaw,
        )
        preview = assemble_source_bound_nominal_rotor_preview(
            forecast=evidence,
            current_enu_downwind_wind_mps=wind,
            current_wind_source="fino1_recorded_interval_reconstruction",
            current_wind_observation_time=str(current_record.forecast.origin_time),
            platform_state=platform_state,
            kinematics=kinematics,
            parameters=resources.rotor_parameters,
            schedule=resources.nominal_schedule,
            performance_table=resources.performance_table,
            operating_mode_source=assumptions["operating_mode_source"],
        )
        loads.append(preview.load_assembly.current_rotor_load)
    return tuple(loads)


def assemble_real_lstm_rotor_preview(
    *,
    resources: RealLstmPreviewResources,
    origin: datetime,
    platform_state: IncrementalState,
) -> SourceBoundRotorPreview:
    """Infer one source record and convert it under the current platform state."""

    return assemble_real_lstm_rotor_preview_from_record(
        resources=resources,
        source_record=infer_real_lstm_wind_record(
            resources=resources,
            origin=origin,
        ),
        platform_state=platform_state,
    )


__all__ = [
    "DATASET_DIRECTORY",
    "MODEL_DIRECTORY",
    "REFERENCE_MANIFEST",
    "REFERENCE_TANK_MASSES_KG",
    "ROTOR_PREVIEW_MANIFEST",
    "RealLstmPreviewResources",
    "RealLstmWindRecord",
    "TANK_CAPACITY_KG",
    "TANK_COORDINATES_M",
    "PlannerWindForecastMode",
    "assemble_planner_rotor_preview_from_record",
    "assemble_recorded_interval_rotor_load_substeps",
    "assemble_real_lstm_rotor_preview",
    "assemble_real_lstm_rotor_preview_from_record",
    "infer_real_lstm_wind_record",
    "inspect_real_lstm_wind_record_operating_domain",
    "planner_forecast_evidence_from_record",
    "prepare_real_lstm_preview_resources",
]
