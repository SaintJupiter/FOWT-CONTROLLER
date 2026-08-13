"""Economy, relief, and long-horizon experimental policy helpers."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from .ballast_planner import (
    action_vec,
    compute_pressure_blocks,
    norm_term,
)
from .forecast_contract import (
    ForecastContract,
    forecast_from_oracle_sample,
    validate_forecast_result,
)
from .forecast_evidence import (
    ForecastEvidence,
    evidence_from_result,
    lead_reliability_from_metrics,
)
from .forecast_admission import (
    EVENT_RISK_KEYS as _EVENT_RISK_KEYS,
)




class ProviderHorizonMixin:
    """Long-horizon forecast behavior isolated from economy policies."""

    def _forecast_reliability_source(self) -> tuple[Path | None, tuple[Any, ...]]:
        """Resolve the reliability source and return a content-aware identity."""

        adapter = self.forecast_adapter
        model_dir = getattr(adapter, "model_dir", None)
        if model_dir is None:
            baseline = getattr(adapter, "baseline", None)
            model_dir = getattr(baseline, "model_dir", None)
        if model_dir is None:
            return None, ("model_dir", None, "metrics", None, "missing")

        resolved_model_dir = Path(model_dir).expanduser().resolve()
        metrics_path = resolved_model_dir / "lstm_regression_metrics.csv"
        try:
            payload = metrics_path.read_bytes()
        except FileNotFoundError:
            return metrics_path, (
                "model_dir",
                str(resolved_model_dir),
                "metrics",
                str(metrics_path),
                "missing",
            )
        except OSError as exc:
            return metrics_path, (
                "model_dir",
                str(resolved_model_dir),
                "metrics",
                str(metrics_path),
                "unavailable",
                type(exc).__name__,
                getattr(exc, "errno", None),
            )
        return metrics_path, (
            "model_dir",
            str(resolved_model_dir),
            "metrics",
            str(metrics_path),
            "sha256",
            hashlib.sha256(payload).hexdigest(),
        )

    def _set_forecast_reliability_diagnostic(
        self,
        *,
        status: str,
        source: Path | None,
        detail: str,
    ) -> None:
        self._forecast_lead_reliability_status = str(status)
        self._forecast_lead_reliability_source = (
            str(source) if source is not None else None
        )
        self._forecast_lead_reliability_detail = str(detail)

    def _forecast_lead_reliability_metadata(self) -> dict[str, Any]:
        """Expose reliability provenance without changing the numeric contract."""

        return {
            "status": str(
                getattr(self, "_forecast_lead_reliability_status", "missing")
            ),
            "source": getattr(self, "_forecast_lead_reliability_source", None),
            "detail": str(
                getattr(self, "_forecast_lead_reliability_detail", "not_loaded")
            ),
        }

    def _forecast_lead_reliability(self, horizon_steps: int) -> np.ndarray:
        """Return a cached, held-out reliability value for every forecast lead."""

        steps = max(int(horizon_steps), 0)
        required = bool(getattr(self.cfg, "lead_reliability_enabled", False))
        if steps == 0:
            self._set_forecast_reliability_diagnostic(
                status="loaded",
                source=None,
                detail="empty_horizon",
            )
            return np.ones(0, dtype=float)

        metrics_path, source_identity = self._forecast_reliability_source()
        cache_key = (steps, required, source_identity)
        cached = getattr(self, "_forecast_lead_reliability_cache", None)
        if isinstance(cached, dict) and cached.get("key") == cache_key:
            status = str(cached["status"])
            self._set_forecast_reliability_diagnostic(
                status=status,
                source=metrics_path,
                detail=str(cached["detail"]),
            )
            if required and status != "loaded":
                raise ValueError(
                    "lead reliability is enabled but its source is "
                    f"{status}: {cached['detail']}"
                )
            return np.asarray(cached["values"], dtype=float).copy()

        reliability = np.ones(steps, dtype=float)
        status = "missing"
        detail = "forecast model directory is unavailable"
        if metrics_path is not None:
            try:
                parsed = lead_reliability_from_metrics(metrics_path)
            except (FileNotFoundError, OSError):
                status = "missing"
                detail = f"reliability metrics are unavailable at {metrics_path}"
            except ValueError as exc:
                status = "shape_mismatch"
                detail = str(exc)
            else:
                if parsed.shape != (steps,):
                    status = "shape_mismatch"
                    detail = (
                        f"reliability metrics contain {parsed.size} leads; "
                        f"forecast requires {steps}"
                    )
                else:
                    status = "loaded"
                    detail = f"loaded {steps} lead reliability values"
                    reliability = parsed

        self._forecast_lead_reliability_cache_key = cache_key
        self._forecast_lead_reliability_cache = {
            "key": cache_key,
            "values": reliability.copy(),
            "status": status,
            "detail": detail,
        }
        self._set_forecast_reliability_diagnostic(
            status=status,
            source=metrics_path,
            detail=detail,
        )
        if required and status != "loaded":
            raise ValueError(
                "lead reliability is enabled but its source is "
                f"{status}: {detail}"
            )
        return reliability

    def _forecast_uv(self, sample) -> ForecastEvidence:
        contract = ForecastContract(
            future_steps=int(self.replay_dataset.future_steps),
            event_columns=tuple(self.replay_dataset.event_columns),
            require_event_probs=self.forecast_adapter is None,
        )
        if self.forecast_adapter is None:
            forecast = forecast_from_oracle_sample(
                y_uv_raw=sample.y_uv_raw,
                y_event=sample.y_event,
                event_columns=self.replay_dataset.event_columns,
                model_version="oracle_future",
                timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
            )
            self._last_forecast_blend_diagnostic = {}
            return evidence_from_result(
                forecast,
                source=str(forecast.model_version),
                sample_period_s=float(self.replay_dataset.update_interval_s),
                provides_future_preview=True,
            )
        forecast = self.forecast_adapter.predict_window(
            sample.x_window,
            timestamp=sample.history_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        forecast = validate_forecast_result(forecast, contract)
        # Capture optional blend diagnostic. Only BlendedForecastAdapter
        # populates this; plain ForecastModelAdapter leaves it unset.
        diag = getattr(self.forecast_adapter, "last_diagnostic", None)
        self._last_forecast_blend_diagnostic = dict(diag) if isinstance(diag, dict) else {}
        lead_reliability = self._forecast_lead_reliability(
            np.asarray(forecast.wind_uv_raw).shape[0]
        )
        return evidence_from_result(
            forecast,
            source=str(forecast.model_version),
            sample_period_s=float(self.replay_dataset.update_interval_s),
            provides_future_preview=bool(
                getattr(self.forecast_adapter, "provides_future_preview", True)
            ),
            lead_reliability=lead_reliability,
            metadata={
                "blend_diagnostic": self._last_forecast_blend_diagnostic,
                "lead_reliability": self._forecast_lead_reliability_metadata(),
                "event_thresholds": self._forecast_event_thresholds_metadata(),
            },
        )

    def _forecast_event_thresholds_metadata(self) -> dict[str, float]:
        """Expose the held-out event thresholds used by the loaded model.

        The action policy consumes these values as authorization thresholds for
        high-impact actions.  Keeping them in ``ForecastEvidence`` prevents a
        second, unrelated set of thresholds from drifting into the controller.
        """

        raw = getattr(self.forecast_adapter, "thresholds", {})
        if not isinstance(raw, dict):
            return {}
        thresholds: dict[str, float] = {}
        for name, value in raw.items():
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed) and 0.0 <= parsed <= 1.0:
                thresholds[str(name)] = parsed
        return thresholds


    def _oracle_preemptive_blocks(self, sample) -> list[dict[str, Any]]:
        """Oracle pressure blocks for the preemptive ceiling test.

        The main planner can still use a learned forecast. This helper reads
        the replay sample's true future only for the default-off preemptive
        diagnostic, so the result measures an oracle ceiling rather than a
        learned-model claim.
        """
        block_discounts = self.block_discounts
        uv = np.asarray(sample.y_uv_raw, dtype=float)
        return compute_pressure_blocks(uv, block_discounts, self.cfg)


    def _oracle_preemptive_log_fields(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}oracle_preemptive_enabled": int(
                self.oracle_preemptive_prevent_enabled
            ),
            f"{prefix}oracle_preemptive_active": int(self._oracle_preemptive_active),
            f"{prefix}oracle_preemptive_reason": self._oracle_preemptive_reason,
            f"{prefix}oracle_preemptive_count": int(self._oracle_preemptive_count),
            f"{prefix}oracle_preemptive_posture_metric_deg": float(
                self._oracle_preemptive_posture_metric_deg
            ),
            f"{prefix}oracle_preemptive_current_norm": float(
                self._oracle_preemptive_current_norm
            ),
            f"{prefix}oracle_preemptive_future_max_norm": float(
                self._oracle_preemptive_future_max_norm
            ),
            f"{prefix}oracle_preemptive_rise_norm": float(
                self._oracle_preemptive_rise_norm
            ),
            f"{prefix}oracle_preemptive_delta_mean_kg": float(
                self._oracle_preemptive_delta_mean_kg
            ),
        }


    def _reset_h120_oracle_probe_bucket_state(self, reason: str = "not_evaluated") -> None:
        self._h120_oracle_probe_active = False
        self._h120_oracle_probe_reason = str(reason)
        self._h120_oracle_probe_action_effect = "none"
        self._h120_oracle_probe_effective_medium_delay_s = float(
            self.reactive_floor_medium_delay_s
        )
        self._h120_oracle_probe_far_persistent_high = False
        self._h120_oracle_probe_far_intensification = False
        self._h120_oracle_probe_far_reintensification = False
        self._h120_oracle_probe_posture_metric_deg = 0.0
        self._h120_oracle_probe_far_max_norm = 0.0
        self._h120_oracle_probe_far_min_norm = 0.0
        self._h120_oracle_probe_near_last_norm = 0.0
        self._h120_oracle_probe_far_rise_norm = 0.0
        self._h120_oracle_probe_delta_mean_kg = 0.0
        self._h120_oracle_probe_preemptive_active = False
        self._h120_oracle_probe_short_delay_active = False
        self._h120_oracle_probe_early_stop_veto_active = False


    def _h120_oracle_probe_refresh_signals(
        self,
        plant_info: dict[str, Any] | None = None,
    ) -> None:
        if not self.h120_oracle_probe_enabled:
            self._h120_oracle_probe_reason = "disabled"
            return
        if not self.far_horizon_enabled:
            self._h120_oracle_probe_reason = "far_horizon_disabled"
            return
        if not self._far_horizon_available:
            self._h120_oracle_probe_reason = self._far_horizon_reason
            return

        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        near_last = float(norms[2])
        far = [float(x) for x in norms[3:6]]
        far_min = float(min(far))
        far_max = float(max(far))
        far_rise = far_max - near_last
        persistent_high = bool(far_min >= float(self.h120_oracle_probe_far_high_norm))
        intensification = bool(
            far_max >= float(self.h120_oracle_probe_far_high_norm)
            and far_rise >= float(self.h120_oracle_probe_intensify_margin_norm)
        )
        reintensification = bool(
            far[2] >= float(self.h120_oracle_probe_far_high_norm)
            and far[2] >= far[0] + float(self.h120_oracle_probe_intensify_margin_norm)
        )
        posture = np.zeros(2, dtype=float)
        if plant_info is not None:
            posture = np.asarray(
                plant_info.get("posture_vec_deg", posture),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        posture_abs = np.abs(posture[:2])
        pitch_abs = abs(float(posture[0]))
        roll_abs = abs(float(posture[1]))
        posture_balance = float(
            np.min(posture_abs) / max(float(np.max(posture_abs)), 1e-9)
        )

        self._h120_oracle_probe_posture_metric_deg = float(max_axis)
        self._h120_oracle_probe_far_max_norm = far_max
        self._h120_oracle_probe_far_min_norm = far_min
        self._h120_oracle_probe_near_last_norm = near_last
        self._h120_oracle_probe_far_rise_norm = far_rise
        self._h120_oracle_probe_far_persistent_high = persistent_high
        self._h120_oracle_probe_far_intensification = intensification
        self._h120_oracle_probe_far_reintensification = reintensification
        if persistent_high:
            self._h120_oracle_probe_reason = "far_persistent_high"
        elif intensification:
            self._h120_oracle_probe_reason = "far_intensification"
        elif reintensification:
            self._h120_oracle_probe_reason = "far_reintensification"
        else:
            self._h120_oracle_probe_reason = "no_far_high_pressure"


    def _h120_oracle_probe_far_pressure_risk(self) -> bool:
        return bool(
            self._h120_oracle_probe_far_persistent_high
            or self._h120_oracle_probe_far_intensification
            or self._h120_oracle_probe_far_reintensification
            or self._far_horizon_hidden_intensification
        )


    def _h120_oracle_probe_prepare_floor_delay(
        self,
        plant_info: dict[str, Any],
    ) -> None:
        self._h120_oracle_probe_refresh_signals(plant_info)
        self._h120_oracle_probe_effective_medium_delay_s = float(
            self.reactive_floor_medium_delay_s
        )
        if not self.h120_oracle_probe_enabled:
            return
        if self.reactive_floor_medium_delay_s <= 0.0:
            self._h120_oracle_probe_reason = "medium_delay_disabled"
            return
        if not self._h120_oracle_probe_far_pressure_risk():
            return
        short_delay = min(
            float(self.reactive_floor_medium_delay_s),
            float(self.h120_oracle_probe_short_delay_s),
        )
        if short_delay < float(self.reactive_floor_medium_delay_s):
            self._h120_oracle_probe_active = True
            self._h120_oracle_probe_short_delay_active = True
            self._h120_oracle_probe_action_effect = "shorten_medium_delay"
            self._h120_oracle_probe_effective_medium_delay_s = short_delay
            self._h120_oracle_probe_short_delay_count += 1
            self._h120_oracle_probe_count += 1


    def _h120_oracle_probe_preemptive_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        floor_refresh_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        self._h120_oracle_probe_preemptive_active = False
        self._h120_oracle_probe_delta_mean_kg = 0.0
        if not self.h120_oracle_probe_enabled:
            return False, np.zeros(2, dtype=float), "disabled"
        self._h120_oracle_probe_refresh_signals(plant_info)
        if floor_refresh_pending or self._reactive_floor_latched:
            self._h120_oracle_probe_reason = "floor_already_active"
            return False, np.zeros(2, dtype=float), "floor_already_active"
        if not self._h120_oracle_probe_far_pressure_risk():
            return False, np.zeros(2, dtype=float), self._h120_oracle_probe_reason

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        self._h120_oracle_probe_posture_metric_deg = float(max_axis)
        if max_axis < float(self.h120_oracle_probe_preemptive_enter_deg):
            self._h120_oracle_probe_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.h120_oracle_probe_preemptive_floor_deg):
            self._h120_oracle_probe_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"
        since_last = float(current_time) - float(self._h120_oracle_probe_last_preemptive_s)
        if since_last < float(self.h120_oracle_probe_cooldown_s):
            self._h120_oracle_probe_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"

        avec = action_vec("active_small", posture, self.cfg)
        delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        self._h120_oracle_probe_delta_mean_kg = delta_mean
        if delta_mean < float(self.h120_oracle_probe_min_delta_kg):
            self._h120_oracle_probe_reason = "delta_too_small"
            return False, avec, "delta_too_small"

        self._h120_oracle_probe_active = True
        self._h120_oracle_probe_preemptive_active = True
        self._h120_oracle_probe_action_effect = "preemptive_active_small"
        self._h120_oracle_probe_reason = "far_pressure_preemptive_active_small"
        self._h120_oracle_probe_preemptive_count += 1
        self._h120_oracle_probe_count += 1
        self._h120_oracle_probe_last_preemptive_s = float(current_time)
        return True, avec, "far_pressure_preemptive_active_small"


    def _h120_oracle_probe_log_fields(self, prefix: str = "") -> dict[str, Any]:
        return {
            f"{prefix}h120_oracle_probe_enabled": int(self.h120_oracle_probe_enabled),
            f"{prefix}h120_oracle_probe_active": int(self._h120_oracle_probe_active),
            f"{prefix}h120_oracle_probe_reason": self._h120_oracle_probe_reason,
            f"{prefix}h120_oracle_probe_action_effect": (
                self._h120_oracle_probe_action_effect
            ),
            f"{prefix}h120_oracle_probe_count": int(self._h120_oracle_probe_count),
            f"{prefix}h120_oracle_probe_short_delay_active": int(
                self._h120_oracle_probe_short_delay_active
            ),
            f"{prefix}h120_oracle_probe_short_delay_count": int(
                self._h120_oracle_probe_short_delay_count
            ),
            f"{prefix}h120_oracle_probe_early_stop_veto_active": int(
                self._h120_oracle_probe_early_stop_veto_active
            ),
            f"{prefix}h120_oracle_probe_early_stop_veto_count": int(
                self._h120_oracle_probe_early_stop_veto_count
            ),
            f"{prefix}h120_oracle_probe_preemptive_active": int(
                self._h120_oracle_probe_preemptive_active
            ),
            f"{prefix}h120_oracle_probe_preemptive_count": int(
                self._h120_oracle_probe_preemptive_count
            ),
            f"{prefix}h120_oracle_probe_effective_medium_delay_s": float(
                self._h120_oracle_probe_effective_medium_delay_s
            ),
            f"{prefix}h120_oracle_probe_far_persistent_high": int(
                self._h120_oracle_probe_far_persistent_high
            ),
            f"{prefix}h120_oracle_probe_far_intensification": int(
                self._h120_oracle_probe_far_intensification
            ),
            f"{prefix}h120_oracle_probe_far_reintensification": int(
                self._h120_oracle_probe_far_reintensification
            ),
            f"{prefix}h120_oracle_probe_posture_metric_deg": float(
                self._h120_oracle_probe_posture_metric_deg
            ),
            f"{prefix}h120_oracle_probe_near_last_norm": float(
                self._h120_oracle_probe_near_last_norm
            ),
            f"{prefix}h120_oracle_probe_far_min_norm": float(
                self._h120_oracle_probe_far_min_norm
            ),
            f"{prefix}h120_oracle_probe_far_max_norm": float(
                self._h120_oracle_probe_far_max_norm
            ),
            f"{prefix}h120_oracle_probe_far_rise_norm": float(
                self._h120_oracle_probe_far_rise_norm
            ),
            f"{prefix}h120_oracle_probe_delta_mean_kg": float(
                self._h120_oracle_probe_delta_mean_kg
            ),
        }


    def _reset_h120_scheduler_bucket_state(self, reason: str = "not_evaluated") -> None:
        self._h120_scheduler_risk_tier = "normal"
        self._h120_scheduler_reason = str(reason)
        self._h120_scheduler_far_persistent_high_pressure = False
        self._h120_scheduler_far_intensification = False
        self._h120_scheduler_far_reintensification_after_relief = False
        self._h120_scheduler_far_direction_consistent_with_current_posture = False
        self._h120_scheduler_far_signflip_risk = False
        self._h120_scheduler_effective_medium_delay_s = float(
            self.reactive_floor_medium_delay_s
        )
        self._h120_scheduler_remote_risk_active = False
        self._h120_scheduler_early_stop_suppressed_active = False
        self._h120_scheduler_floor_episode_active_for_budget = False
        self._h120_scheduler_far_max_norm = 0.0
        self._h120_scheduler_far_min_norm = 0.0
        self._h120_scheduler_near_max_norm = 0.0
        self._h120_scheduler_near_last_norm = 0.0
        self._h120_scheduler_prefloor_active = False
        self._h120_scheduler_prefloor_reason = str(reason)
        self._h120_scheduler_prefloor_delta_mean_kg = 0.0


    def _h120_scheduler_refresh(self, plant_info: dict[str, Any] | None = None) -> None:
        """Recompute the remote-risk tier from the 60-120min far-horizon blocks.

        Idempotent per bucket: call once at the start of the floor phase. Sets
        the effective delayed-medium delay and the remote_risk flag used by the
        early_stop guard. Does NOT touch target generation, does NOT decide
        whether high posture recovers, does NOT use relief to weaken the floor.
        """
        self._reset_h120_scheduler_bucket_state("not_evaluated")
        if not self.h120_risk_scheduler_enabled:
            self._h120_scheduler_reason = "disabled"
            return
        # When enabled, the scheduler OWNS the delay; base = normal-tier delay.
        self._h120_scheduler_effective_medium_delay_s = float(
            self.h120_scheduler_delay_normal_s
        )
        floor_episode_active = bool(
            self._reactive_floor_latched or self._reactive_floor_post_exit_episode_active
        )
        if not floor_episode_active:
            self._h120_scheduler_early_stop_suppressed_this_episode = 0
        self._h120_scheduler_floor_episode_active_for_budget = floor_episode_active
        if not self.far_horizon_enabled:
            self._h120_scheduler_reason = "far_horizon_disabled"
            return
        if not self._far_horizon_available:
            self._h120_scheduler_reason = self._far_horizon_reason
            return

        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        near = [float(x) for x in norms[:3]]
        far = [float(x) for x in norms[3:6]]
        near_max = float(max(near))
        near_last = float(near[2])
        far_min = float(min(far))
        far_max = float(max(far))
        high_norm = float(self.h120_scheduler_far_high_norm)
        intensify_margin = float(self.h120_scheduler_intensify_margin_norm)
        relief_margin = float(self.h120_scheduler_relief_margin_norm)

        persistent_high = bool(far_min >= high_norm)
        intensification = bool(far_max >= near_last + intensify_margin)
        near_relief = bool(near_last <= near_max - relief_margin)
        reintensification = bool(
            near_relief and far_max >= near_last + intensify_margin
        )

        # Direction consistency: far pressure vector aligned with current
        # posture (would push attitude further in the offending direction).
        direction_consistent = False
        posture = np.zeros(2, dtype=float)
        if plant_info is not None:
            posture = np.asarray(
                plant_info.get("posture_vec_deg", posture), dtype=float
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            posture = posture[:2]
        posture_norm = float(np.linalg.norm(posture))
        far_vecs = (
            self._far_horizon_vecs[3:6]
            if len(self._far_horizon_vecs) >= 6
            else []
        )
        if far_vecs and posture_norm > 1e-6:
            far_vec_sum = np.sum(np.asarray(far_vecs, dtype=float), axis=0)
            fv_norm = float(np.linalg.norm(far_vec_sum))
            if fv_norm > 1e-6:
                cos = float(np.dot(far_vec_sum, posture) / (fv_norm * posture_norm))
                # Pure direction alignment, decoupled from magnitude so the
                # tier rule can combine it explicitly with the danger flags.
                direction_consistent = bool(cos > 0.3)

        signflip_risk = bool(
            self._far_horizon_reversal or self._far_horizon_direction_shift
        )

        # Direction-gated tier rule (controller-relevant far-risk shapes):
        #   far_danger = far pressure stays high / rises / re-intensifies.
        #   high_risk ONLY when far_danger AND the far pressure direction is
        #   aligned with the current posture (would worsen the SAME axis).
        #   far_danger with mismatched direction -> risk_aware (cautious, not
        #   full escalation): a strong remote wind that does not load the
        #   current attitude axis should not trigger aggressive medium.
        #   signflip/reversal alone -> risk_aware. Otherwise normal.
        far_danger = bool(persistent_high or intensification or reintensification)
        if far_danger and direction_consistent:
            tier = "high_risk"
        elif far_danger or signflip_risk:
            tier = "risk_aware"
        else:
            tier = "normal"

        delay_map = {
            "normal": self.h120_scheduler_delay_normal_s,
            "risk_aware": self.h120_scheduler_delay_risk_aware_s,
            "high_risk": self.h120_scheduler_delay_high_risk_s,
        }
        self._h120_scheduler_risk_tier = tier
        self._h120_scheduler_reason = "ok"
        self._h120_scheduler_far_persistent_high_pressure = persistent_high
        self._h120_scheduler_far_intensification = intensification
        self._h120_scheduler_far_reintensification_after_relief = reintensification
        self._h120_scheduler_far_direction_consistent_with_current_posture = direction_consistent
        self._h120_scheduler_far_signflip_risk = signflip_risk
        self._h120_scheduler_effective_medium_delay_s = float(delay_map[tier])
        self._h120_scheduler_remote_risk_active = bool(tier == "high_risk")
        self._h120_scheduler_far_max_norm = far_max
        self._h120_scheduler_far_min_norm = far_min
        self._h120_scheduler_near_max_norm = near_max
        self._h120_scheduler_near_last_norm = near_last


    def _h120_scheduler_try_suppress_early_stop(self) -> bool:
        """Return True if early_stop should be suppressed this bucket.

        Only when remote risk (high_risk tier) is active AND the per-episode
        suppress budget is not exhausted. Increments the suppress counters.
        Never permanently disables early_stop — bounded by max_per_episode.
        """
        if not self.h120_risk_scheduler_enabled:
            return False
        if not self._h120_scheduler_remote_risk_active:
            return False
        if (
            self._h120_scheduler_early_stop_suppressed_this_episode
            >= self.h120_scheduler_early_stop_suppress_max_per_episode
        ):
            return False
        self._h120_scheduler_early_stop_suppressed_this_episode += 1
        self._h120_scheduler_early_stop_suppressed_count += 1
        self._h120_scheduler_early_stop_suppressed_active = True
        return True


    def _h120_scheduler_prefloor_probe_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        floor_refresh_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        """Conservative pre-floor coupling probe for oracle far-risk tests.

        Default-off.  Unlike scheduler v1's floor-internal levers, this can
        refresh a small posture-oriented target before the floor enters,
        but only in the 3-5 degree band and only under high-risk, direction-
        consistent far pressure.  It never weakens the floor and never uses
        relief suppression.
        """
        self._h120_scheduler_prefloor_active = False
        self._h120_scheduler_prefloor_delta_mean_kg = 0.0
        if not (
            self.h120_risk_scheduler_enabled
            and self.h120_scheduler_prefloor_probe_enabled
        ):
            self._h120_scheduler_prefloor_reason = "disabled"
            return False, np.zeros(2, dtype=float), "disabled"
        if floor_refresh_pending or self._reactive_floor_latched:
            self._h120_scheduler_prefloor_reason = "floor_already_active"
            return False, np.zeros(2, dtype=float), "floor_already_active"
        self._h120_scheduler_refresh(plant_info)
        if not self._h120_scheduler_remote_risk_active:
            self._h120_scheduler_prefloor_reason = "not_high_risk"
            return False, np.zeros(2, dtype=float), "not_high_risk"
        if not self._h120_scheduler_far_direction_consistent_with_current_posture:
            self._h120_scheduler_prefloor_reason = "direction_not_consistent"
            return False, np.zeros(2, dtype=float), "direction_not_consistent"

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        if max_axis < float(self.h120_scheduler_prefloor_enter_deg):
            self._h120_scheduler_prefloor_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.h120_scheduler_prefloor_floor_deg):
            self._h120_scheduler_prefloor_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"
        since_last = float(current_time) - float(self._h120_scheduler_prefloor_last_s)
        if since_last < float(self.h120_scheduler_prefloor_cooldown_s):
            self._h120_scheduler_prefloor_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"

        avec = action_vec(self.h120_scheduler_prefloor_action, posture, self.cfg)
        delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        if (
            float(self.h120_scheduler_prefloor_max_delta_kg) > 0.0
            and delta_mean > float(self.h120_scheduler_prefloor_max_delta_kg)
        ):
            scale = float(self.h120_scheduler_prefloor_max_delta_kg) / max(
                delta_mean, 1e-9
            )
            avec = avec * scale
            delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        self._h120_scheduler_prefloor_delta_mean_kg = delta_mean
        if delta_mean < float(self.h120_scheduler_prefloor_min_delta_kg):
            self._h120_scheduler_prefloor_reason = "delta_too_small"
            return False, avec, "delta_too_small"

        self._h120_scheduler_prefloor_active = True
        self._h120_scheduler_prefloor_reason = (
            f"prefloor_{self.h120_scheduler_prefloor_action}"
        )
        self._h120_scheduler_prefloor_count += 1
        self._h120_scheduler_prefloor_last_s = float(current_time)
        return True, avec, str(self._h120_scheduler_prefloor_reason)


    def _h120_axis_micro_prepare_needed(
        self,
        plant_info: dict[str, Any],
        current_time: float,
        higher_priority_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        """Corrected pre-floor axis-aware micro prepare for oracle ceiling.

        Distinct from the prefloor target-refresh probe (NO-GO): this fires
        ONLY on the direction-clean + real-rise eligible window validated by
        the Step 1 audit, and applies a small single-axis nudge (capped well
        below active_small) on the CURRENT dominant posture axis.  It never
        touches the floor, never refreshes a posture target wholesale, never
        uses the far signal to pick the axis (axis = current posture).
        """
        self._h120_axis_micro_active = False
        self._h120_axis_micro_axis = "none"
        self._h120_axis_micro_delta_mean_kg = 0.0
        if not (self.h120_axis_micro_enabled and self.far_horizon_enabled):
            self._h120_axis_micro_reason = "disabled"
            return False, np.zeros(2, dtype=float), "disabled"
        if higher_priority_pending or self._reactive_floor_latched:
            self._h120_axis_micro_reason = "floor_or_higher_active"
            return False, np.zeros(2, dtype=float), "floor_or_higher_active"

        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        pitch_abs, roll_abs = abs(float(posture[0])), abs(float(posture[1]))
        max_axis = max(pitch_abs, roll_abs)
        if max_axis < float(self.h120_axis_micro_enter_deg):
            self._h120_axis_micro_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.h120_axis_micro_floor_deg):
            self._h120_axis_micro_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"

        # eligibility: pump idle + worsening (reuse floor-computed quantities)
        if not bool(self._reactive_floor_pump_idle):
            self._h120_axis_micro_reason = "pump_not_idle"
            return False, np.zeros(2, dtype=float), "pump_not_idle"
        resp = float(self._reactive_floor_current_response_deg)
        worsening = math.isfinite(resp) and resp <= -float(
            self.h120_axis_micro_worsening_eps_deg
        )
        if not worsening:
            self._h120_axis_micro_reason = "not_worsening"
            return False, np.zeros(2, dtype=float), "not_worsening"

        # direction-clean + real-rise gate (Step 1 validated)
        if self._far_horizon_reversal or self._far_horizon_direction_shift:
            self._h120_axis_micro_reason = "direction_not_clean"
            return False, np.zeros(2, dtype=float), "direction_not_clean"
        if float(self._far_horizon_far_max) < float(
            self.h120_axis_micro_rise_floor_norm
        ):
            self._h120_axis_micro_reason = "no_far_rise"
            return False, np.zeros(2, dtype=float), "no_far_rise"

        since_last = float(current_time) - float(self._h120_axis_micro_last_s)
        if since_last < float(self.h120_axis_micro_cooldown_s):
            self._h120_axis_micro_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"

        # axis-aware micro: active_small direction, single dominant axis, capped
        base = action_vec("active_small", posture, self.cfg)
        base = np.asarray(base, dtype=float).reshape(2)
        axis_micro = np.zeros(2, dtype=float)
        if pitch_abs >= roll_abs:
            axis_micro[0] = base[0]
            self._h120_axis_micro_axis = "pitch"
        else:
            axis_micro[1] = base[1]
            self._h120_axis_micro_axis = "roll"
        delta_mean = float(np.mean(np.abs(self._primary_mass_delta_kg(axis_micro))))
        if delta_mean <= 0.0:
            self._h120_axis_micro_reason = "zero_action"
            return False, np.zeros(2, dtype=float), "zero_action"
        if delta_mean > float(self.h120_axis_micro_delta_kg):
            axis_micro = axis_micro * (
                float(self.h120_axis_micro_delta_kg) / max(delta_mean, 1e-9)
            )
            delta_mean = float(
                np.mean(np.abs(self._primary_mass_delta_kg(axis_micro)))
            )
        self._h120_axis_micro_delta_mean_kg = delta_mean
        self._h120_axis_micro_active = True
        self._h120_axis_micro_reason = f"axis_micro_{self._h120_axis_micro_axis}"
        self._h120_axis_micro_count += 1
        self._h120_axis_micro_last_s = float(current_time)
        return True, axis_micro, str(self._h120_axis_micro_reason)


    def _h120_axis_micro_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_axis_micro_enabled": int(self.h120_axis_micro_enabled),
            f"{p}h120_axis_micro_active": int(self._h120_axis_micro_active),
            f"{p}h120_axis_micro_reason": str(self._h120_axis_micro_reason),
            f"{p}h120_axis_micro_axis": str(self._h120_axis_micro_axis),
            f"{p}h120_axis_micro_count": int(self._h120_axis_micro_count),
            f"{p}h120_axis_micro_delta_mean_kg": float(
                self._h120_axis_micro_delta_mean_kg
            ),
        }


    def _h120_floor_shaping_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_floor_shaping_mode": str(self.h120_floor_shaping_mode),
            f"{p}h120_floor_shaping_active": int(self._h120_floor_shaping_active),
            f"{p}h120_floor_shaping_reason": str(self._h120_floor_shaping_reason),
            f"{p}h120_floor_shaping_effect": str(self._h120_floor_shaping_effect),
            f"{p}h120_floor_shaping_axis": str(self._h120_floor_shaping_axis),
            f"{p}h120_floor_shaping_count": int(self._h120_floor_shaping_count),
            f"{p}h120_floor_shaping_delta_mean_kg": float(
                self._h120_floor_shaping_delta_mean_kg
            ),
            f"{p}h120_floor_shaping_far_max_norm": float(self._far_horizon_far_max),
            f"{p}h120_floor_shaping_current_response_deg": float(
                self._reactive_floor_current_response_deg
            ),
        }


    def _h120_pareto_mode_selector_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_pareto_mode_selector_enabled": int(
                self.h120_pareto_mode_selector_enabled
            ),
            f"{p}h120_pareto_mode_selector_active": int(
                self._h120_pareto_mode_selector_active
            ),
            f"{p}h120_pareto_mode_selector_reason": str(
                self._h120_pareto_mode_selector_reason
            ),
            f"{p}h120_pareto_mode_selector_selected_mode": str(
                self._h120_pareto_mode_selector_selected_mode
            ),
            f"{p}h120_pareto_mode_selector_selected_effect": str(
                self._h120_pareto_mode_selector_selected_effect
            ),
            f"{p}h120_pareto_mode_selector_episode_active": int(
                self._h120_pareto_mode_selector_episode_active
            ),
            f"{p}h120_pareto_mode_selector_episode_mode": str(
                self._h120_pareto_mode_selector_episode_mode
            ),
            f"{p}h120_pareto_mode_selector_count": int(
                self._h120_pareto_mode_selector_count
            ),
            f"{p}h120_pareto_mode_selector_switch_count": int(
                self._h120_pareto_mode_selector_switch_count
            ),
            f"{p}h120_pareto_mode_selector_far_persistent_high": int(
                self._h120_pareto_mode_selector_far_persistent_high
            ),
            f"{p}h120_pareto_mode_selector_far_intensification": int(
                self._h120_pareto_mode_selector_far_intensification
            ),
            f"{p}h120_pareto_mode_selector_far_reintensification": int(
                self._h120_pareto_mode_selector_far_reintensification
            ),
            f"{p}h120_pareto_mode_selector_far_relief": int(
                self._h120_pareto_mode_selector_far_relief
            ),
            f"{p}h120_pareto_mode_selector_far_direction_consistent": int(
                self._h120_pareto_mode_selector_far_direction_consistent
            ),
            f"{p}h120_pareto_mode_selector_recovery_slow": int(
                self._h120_pareto_mode_selector_recovery_slow
            ),
            f"{p}h120_pareto_mode_selector_far_max_norm": float(
                self._h120_pareto_mode_selector_far_max_norm
            ),
            f"{p}h120_pareto_mode_selector_far_min_norm": float(
                self._h120_pareto_mode_selector_far_min_norm
            ),
            f"{p}h120_pareto_mode_selector_near_last_norm": float(
                self._h120_pareto_mode_selector_near_last_norm
            ),
            f"{p}h120_pareto_mode_selector_current_response_deg": float(
                self._h120_pareto_mode_selector_current_response_deg
            ),
        }


    def _h120_scheduler_log_fields(self, prefix: str = "") -> dict[str, Any]:
        p = str(prefix)
        return {
            f"{p}h120_scheduler_enabled": int(self.h120_risk_scheduler_enabled),
            f"{p}h120_scheduler_risk_tier": str(self._h120_scheduler_risk_tier),
            f"{p}h120_scheduler_reason": str(self._h120_scheduler_reason),
            f"{p}h120_scheduler_far_persistent_high_pressure": int(
                self._h120_scheduler_far_persistent_high_pressure
            ),
            f"{p}h120_scheduler_far_intensification": int(
                self._h120_scheduler_far_intensification
            ),
            f"{p}h120_scheduler_far_reintensification_after_relief": int(
                self._h120_scheduler_far_reintensification_after_relief
            ),
            f"{p}h120_scheduler_far_direction_consistent_with_current_posture": int(
                self._h120_scheduler_far_direction_consistent_with_current_posture
            ),
            f"{p}h120_scheduler_far_signflip_risk": int(
                self._h120_scheduler_far_signflip_risk
            ),
            f"{p}h120_scheduler_effective_medium_delay_s": float(
                self._h120_scheduler_effective_medium_delay_s
            ),
            f"{p}h120_scheduler_remote_risk_active": int(
                self._h120_scheduler_remote_risk_active
            ),
            f"{p}h120_scheduler_floor_episode_active_for_budget": int(
                self._h120_scheduler_floor_episode_active_for_budget
            ),
            f"{p}h120_scheduler_early_stop_suppressed_active": int(
                self._h120_scheduler_early_stop_suppressed_active
            ),
            f"{p}h120_scheduler_early_stop_suppressed_count": int(
                self._h120_scheduler_early_stop_suppressed_count
            ),
            f"{p}h120_scheduler_remote_risk_trigger_count": int(
                self._h120_scheduler_remote_risk_trigger_count
            ),
            f"{p}h120_scheduler_far_max_norm": float(self._h120_scheduler_far_max_norm),
            f"{p}h120_scheduler_far_min_norm": float(self._h120_scheduler_far_min_norm),
            f"{p}h120_scheduler_near_max_norm": float(self._h120_scheduler_near_max_norm),
            f"{p}h120_scheduler_near_last_norm": float(self._h120_scheduler_near_last_norm),
            f"{p}h120_scheduler_prefloor_probe_enabled": int(
                self.h120_scheduler_prefloor_probe_enabled
            ),
            f"{p}h120_scheduler_prefloor_action": str(
                self.h120_scheduler_prefloor_action
            ),
            f"{p}h120_scheduler_prefloor_active": int(
                self._h120_scheduler_prefloor_active
            ),
            f"{p}h120_scheduler_prefloor_reason": str(
                self._h120_scheduler_prefloor_reason
            ),
            f"{p}h120_scheduler_prefloor_count": int(
                self._h120_scheduler_prefloor_count
            ),
            f"{p}h120_scheduler_prefloor_delta_mean_kg": float(
                self._h120_scheduler_prefloor_delta_mean_kg
            ),
        }


    def _h120_pareto_mode_selector_refresh(
        self,
        plant_info: dict[str, Any] | None = None,
    ) -> None:
        """Default-off oracle mode selector for the static floor Pareto modes.

        This is a diagnostic selector only. It does not alter the floor logic;
        it only records which static mode the oracle would choose among:
        - v1.4 economy: active_small + early_stop
        - v1.6 balanced: active_small + delayed medium
        - v1.5 safety: active_medium + early_stop
        """
        self._h120_pareto_mode_selector_active = False
        self._h120_pareto_mode_selector_reason = "disabled"
        self._h120_pareto_mode_selector_selected_mode = "v16_delay1200"
        self._h120_pareto_mode_selector_selected_effect = "balanced"
        self._h120_pareto_mode_selector_far_persistent_high = False
        self._h120_pareto_mode_selector_far_intensification = False
        self._h120_pareto_mode_selector_far_reintensification = False
        self._h120_pareto_mode_selector_far_relief = False
        self._h120_pareto_mode_selector_far_direction_consistent = False
        self._h120_pareto_mode_selector_recovery_slow = False
        self._h120_pareto_mode_selector_far_max_norm = 0.0
        self._h120_pareto_mode_selector_far_min_norm = 0.0
        self._h120_pareto_mode_selector_near_last_norm = 0.0
        self._h120_pareto_mode_selector_current_response_deg = float("nan")
        if not self._reactive_floor_latched:
            self._h120_pareto_mode_selector_episode_active = False
            self._h120_pareto_mode_selector_episode_mode = "v16_delay1200"
        if not self.h120_pareto_mode_selector_enabled:
            return
        if not self.far_horizon_enabled or not self._far_horizon_available:
            self._h120_pareto_mode_selector_reason = "far_horizon_unavailable"
            return

        norms = list(self._far_horizon_norms)
        while len(norms) < 6:
            norms.append(0.0)
        near_last = float(norms[2])
        far = [float(x) for x in norms[3:6]]
        far_min = float(min(far))
        far_max = float(max(far))
        high_norm = float(self.h120_pareto_mode_selector_far_high_norm)
        intensify_margin = float(self.h120_pareto_mode_selector_intensify_margin_norm)
        relief_margin = float(self.h120_pareto_mode_selector_relief_margin_norm)
        persistent_high = bool(far_min >= high_norm)
        intensification = bool(far_max >= near_last + intensify_margin)
        near_relief = bool(near_last <= float(norms[0]) - relief_margin)
        reintensification = bool(near_relief and far_max >= near_last + intensify_margin)

        posture = np.zeros(2, dtype=float)
        if plant_info is not None:
            posture = np.asarray(
                plant_info.get("posture_vec_deg", posture),
                dtype=float,
            ).reshape(-1)
            if posture.size < 2:
                posture = np.pad(posture, (0, 2 - posture.size))
            posture = posture[:2]
        posture_norm = float(np.linalg.norm(posture))
        far_vecs = self._far_horizon_vecs[3:6] if len(self._far_horizon_vecs) >= 6 else []
        direction_consistent = False
        if far_vecs and posture_norm > 1e-6:
            far_vec_sum = np.sum(np.asarray(far_vecs, dtype=float), axis=0)
            fv_norm = float(np.linalg.norm(far_vec_sum))
            if fv_norm > 1e-6:
                cos = float(np.dot(far_vec_sum, posture) / (fv_norm * posture_norm))
                direction_consistent = bool(cos > 0.3)
        current_response = float(self._reactive_floor_current_response_deg)
        recovery_slow = bool(math.isfinite(current_response) and current_response <= float(self.h120_pareto_mode_selector_slow_response_eps_deg))

        self._h120_pareto_mode_selector_far_persistent_high = persistent_high
        self._h120_pareto_mode_selector_far_intensification = intensification
        self._h120_pareto_mode_selector_far_reintensification = reintensification
        self._h120_pareto_mode_selector_far_relief = near_relief
        self._h120_pareto_mode_selector_far_direction_consistent = direction_consistent
        self._h120_pareto_mode_selector_recovery_slow = recovery_slow
        self._h120_pareto_mode_selector_far_max_norm = far_max
        self._h120_pareto_mode_selector_far_min_norm = far_min
        self._h120_pareto_mode_selector_near_last_norm = near_last
        self._h120_pareto_mode_selector_current_response_deg = current_response

        # Conservative three-way rule:
        # - relief / weak far danger -> v1.4 economy
        # - persistent high / intensification / reintensification with slow
        #   recovery or direction support -> v1.5 safety
        # - otherwise -> v1.6 balanced
        if (not persistent_high and not intensification and not reintensification) or (
            near_relief and not direction_consistent
        ):
            self._h120_pareto_mode_selector_selected_mode = "v14_early_stop"
            self._h120_pareto_mode_selector_selected_effect = "economy"
            self._h120_pareto_mode_selector_reason = "relief_or_weak_far_risk"
        elif (persistent_high or intensification or reintensification) and (
            direction_consistent or recovery_slow
        ):
            self._h120_pareto_mode_selector_selected_mode = "v15_medium"
            self._h120_pareto_mode_selector_selected_effect = "safety"
            self._h120_pareto_mode_selector_reason = "far_danger_or_slow_recovery"
        else:
            self._h120_pareto_mode_selector_selected_mode = "v16_delay1200"
            self._h120_pareto_mode_selector_selected_effect = "balanced"
            self._h120_pareto_mode_selector_reason = "default_balanced"

        if self._h120_pareto_mode_selector_episode_active:
            self._h120_pareto_mode_selector_selected_mode = str(
                self._h120_pareto_mode_selector_episode_mode
            )
            if self._h120_pareto_mode_selector_selected_mode == "v14_early_stop":
                self._h120_pareto_mode_selector_selected_effect = "economy"
            elif self._h120_pareto_mode_selector_selected_mode == "v15_medium":
                self._h120_pareto_mode_selector_selected_effect = "safety"
            else:
                self._h120_pareto_mode_selector_selected_effect = "balanced"
            self._h120_pareto_mode_selector_reason = (
                f"episode_latched:{self._h120_pareto_mode_selector_reason}"
            )

        self._h120_pareto_mode_selector_active = True
        if (
            self._h120_pareto_mode_selector_selected_mode
            != self._h120_pareto_mode_selector_last_mode
        ):
            self._h120_pareto_mode_selector_switch_count += 1
            self._h120_pareto_mode_selector_last_mode = self._h120_pareto_mode_selector_selected_mode
        self._h120_pareto_mode_selector_count += 1


    def _oracle_preemptive_needed(
        self,
        sample,
        plant_info: dict[str, Any],
        current_time: float,
        floor_refresh_pending: bool,
    ) -> tuple[bool, np.ndarray, str]:
        self._oracle_preemptive_active = False
        self._oracle_preemptive_reason = "disabled"
        self._oracle_preemptive_posture_metric_deg = 0.0
        self._oracle_preemptive_current_norm = 0.0
        self._oracle_preemptive_future_max_norm = 0.0
        self._oracle_preemptive_rise_norm = 0.0
        self._oracle_preemptive_delta_mean_kg = 0.0
        if not self.oracle_preemptive_prevent_enabled:
            return False, np.zeros(2, dtype=float), "disabled"
        if floor_refresh_pending or self._reactive_floor_latched:
            self._oracle_preemptive_reason = "floor_already_active"
            return False, np.zeros(2, dtype=float), "floor_already_active"
        posture = np.asarray(
            plant_info.get("posture_vec_deg", np.zeros(2, dtype=float)),
            dtype=float,
        ).reshape(-1)
        if posture.size < 2:
            posture = np.pad(posture, (0, 2 - posture.size))
        posture = posture[:2]
        max_axis = max(abs(float(posture[0])), abs(float(posture[1])))
        self._oracle_preemptive_posture_metric_deg = float(max_axis)
        if max_axis < float(self.oracle_preemptive_enter_deg):
            self._oracle_preemptive_reason = "posture_below_enter"
            return False, np.zeros(2, dtype=float), "posture_below_enter"
        if max_axis >= float(self.oracle_preemptive_floor_deg):
            self._oracle_preemptive_reason = "posture_at_floor"
            return False, np.zeros(2, dtype=float), "posture_at_floor"
        since_last = float(current_time) - float(self._oracle_preemptive_last_s)
        if since_last < float(self.oracle_preemptive_cooldown_s):
            self._oracle_preemptive_reason = "cooldown"
            return False, np.zeros(2, dtype=float), "cooldown"
        oracle_blocks = self._oracle_preemptive_blocks(sample)
        if len(oracle_blocks) < 3:
            self._oracle_preemptive_reason = "insufficient_oracle_blocks"
            return False, np.zeros(2, dtype=float), "insufficient_oracle_blocks"
        raw_norms = [
            norm_term(
                np.asarray(
                    b.get("pressure_vec_raw", b.get("pressure_vec", np.zeros(2))),
                    dtype=float,
                ),
                self.cfg,
            )
            for b in oracle_blocks[:3]
        ]
        current_norm = float(raw_norms[0])
        future_max = float(max(raw_norms[1:3]))
        rise = future_max - current_norm
        self._oracle_preemptive_current_norm = current_norm
        self._oracle_preemptive_future_max_norm = future_max
        self._oracle_preemptive_rise_norm = rise
        if future_max < float(self.oracle_preemptive_min_future_norm):
            self._oracle_preemptive_reason = "future_pressure_low"
            return False, np.zeros(2, dtype=float), "future_pressure_low"
        if rise < float(self.oracle_preemptive_rise_margin_norm):
            self._oracle_preemptive_reason = "future_not_rising_enough"
            return False, np.zeros(2, dtype=float), "future_not_rising_enough"
        avec = action_vec("active_small", posture, self.cfg)
        delta = float(np.mean(np.abs(self._primary_mass_delta_kg(avec))))
        self._oracle_preemptive_delta_mean_kg = delta
        if delta < float(self.oracle_preemptive_min_delta_kg):
            self._oracle_preemptive_reason = "proposal_delta_below_threshold"
            return False, avec, "proposal_delta_below_threshold"
        self._oracle_preemptive_active = True
        self._oracle_preemptive_reason = "oracle_preemptive_refresh"
        self._oracle_preemptive_count += 1
        self._oracle_preemptive_last_s = float(current_time)
        return True, avec, "oracle_preemptive_refresh"


    def _event_risk_scales(self, event_probs: dict[str, float]) -> tuple[list[float], list[float]]:
        probs = [float(event_probs.get(key, 0.0)) for key in _EVENT_RISK_KEYS]
        return probs, [1.0, 1.0, 1.0]
