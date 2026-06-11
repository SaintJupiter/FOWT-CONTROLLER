from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ForecastResult:
    wind_uv_raw: np.ndarray
    wind_speed: np.ndarray
    wind_dir_deg: np.ndarray
    event_probs: dict[str, float]
    risk_levels: dict[str, bool]
    model_version: str
    timestamp: str | None = None


class ForecastModelAdapter:
    """Load a trained recurrent preview model and run one-window inference."""

    provides_future_preview = True

    def __init__(
        self,
        model_dir: str | Path,
        dataset_dir: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "lstm_config.json").read_text(encoding="utf-8"))
        self.threshold_blob = json.loads(
            (self.model_dir / "lstm_event_thresholds.json").read_text(encoding="utf-8")
        )
        if dataset_dir is None:
            dataset_dir = self.config["dataset_dir"]
        self.dataset_dir = Path(dataset_dir)
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        self.scaler = json.loads((self.dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))
        self.event_columns = list(self.metadata["event_columns"])
        self.thresholds = {
            str(name): float(value)
            for name, value in self.threshold_blob.get("thresholds", {}).items()
        }
        self.model_version = f"{self.config['model_type']}_dual_head_preview"
        self._device_request = str(device)
        self._torch: Any = None
        self._model: Any = None

    def _lazy_load_runtime(self) -> None:
        if self._model is not None:
            return

        import torch
        from torch import nn

        class WindRNN(nn.Module):
            def __init__(
                self,
                input_size: int,
                hidden_size: int,
                future_steps: int,
                event_count: int,
                num_layers: int,
                dropout: float,
                model_type: str,
                far_hint_head: bool = False,
            ) -> None:
                super().__init__()
                recurrent_dropout = dropout if num_layers > 1 else 0.0
                recurrent_cls = nn.LSTM if model_type.lower() == "lstm" else nn.GRU
                self.model_type = model_type.lower()
                self.future_steps = future_steps
                self.recurrent = recurrent_cls(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=recurrent_dropout,
                    batch_first=True,
                )
                self.shared = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.regression_head = nn.Linear(hidden_size, future_steps * 2)
                self.event_head = nn.Linear(hidden_size, event_count)
                self.far_hint_head = nn.Linear(hidden_size, 1) if far_hint_head else None

            def forward(self, x):
                if self.model_type == "lstm":
                    _, (hidden, _) = self.recurrent(x)
                else:
                    _, hidden = self.recurrent(x)
                features = self.shared(hidden[-1])
                uv = self.regression_head(features).view(x.shape[0], self.future_steps, 2)
                event_logits = self.event_head(features)
                return uv, event_logits

        requested = self._device_request
        if requested == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(requested)
        self._torch = torch
        self._device = device
        model = WindRNN(
            input_size=int(self.config["input_size"]),
            hidden_size=int(self.config["hidden_size"]),
            future_steps=int(self.config["future_steps"]),
            event_count=int(self.config["event_count"]),
            num_layers=int(self.config["num_layers"]),
            dropout=float(self.config["dropout"]),
            model_type=str(self.config["model_type"]),
            far_hint_head=bool(self.config.get("far_hint_auxiliary_head", False)),
        ).to(device)
        checkpoint = torch.load(self.model_dir / "lstm_best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._model = model

    def _full_scaled_prediction(self, pred_regression, x):
        if not bool(self.config.get("residual_regression", False)):
            return pred_regression
        u_idx, v_idx = [int(v) for v in self.config["uv_feature_indices"]]
        current_uv = x[:, -1, [u_idx, v_idx]].unsqueeze(1)
        return pred_regression + current_uv

    def _inverse_scale_uv(self, pred_uv_scaled: np.ndarray) -> np.ndarray:
        target_scaler = self.scaler["target_uv_scaler"]
        mean = np.array(
            [target_scaler["wind_u_ms"]["mean"], target_scaler["wind_v_ms"]["mean"]],
            dtype=np.float32,
        )
        std = np.array(
            [target_scaler["wind_u_ms"]["std"], target_scaler["wind_v_ms"]["std"]],
            dtype=np.float32,
        )
        return pred_uv_scaled * std + mean

    @staticmethod
    def _speed_direction_from_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        direction = (np.rad2deg(np.arctan2(-uv[:, 0], -uv[:, 1])) + 360.0) % 360.0
        return speed.astype(np.float32), direction.astype(np.float32)

    def predict_window(self, x_window: np.ndarray, timestamp: str | None = None) -> ForecastResult:
        self._lazy_load_runtime()
        x = np.asarray(x_window, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, :, :]
        expected = (int(self.metadata["history_steps"]), int(self.config["input_size"]))
        if x.shape[1:] != expected:
            raise ValueError(f"expected window shape (*, {expected[0]}, {expected[1]}), got {x.shape}")

        tensor = self._torch.from_numpy(x).to(self._device)
        with self._torch.no_grad():
            pred_uv_scaled, event_logits = self._model(tensor)
            pred_uv_scaled = self._full_scaled_prediction(pred_uv_scaled, tensor)
            pred_uv_scaled_np = pred_uv_scaled.detach().cpu().numpy()[0].astype(np.float32)
            event_prob_np = self._torch.sigmoid(event_logits).detach().cpu().numpy()[0].astype(np.float32)

        pred_uv_raw = self._inverse_scale_uv(pred_uv_scaled_np)
        speed, direction = self._speed_direction_from_uv(pred_uv_raw)
        event_probs = {
            name: float(event_prob_np[idx]) if idx < len(event_prob_np) else 0.0
            for idx, name in enumerate(self.event_columns)
        }
        risk_levels = {
            name: bool(event_probs[name] >= self.thresholds.get(name, 0.5))
            for name in self.event_columns
        }
        return ForecastResult(
            wind_uv_raw=pred_uv_raw,
            wind_speed=speed,
            wind_dir_deg=direction,
            event_probs=event_probs,
            risk_levels=risk_levels,
            model_version=self.model_version,
            timestamp=timestamp,
        )


class PersistenceMeanForecastAdapter:
    """Persistence baseline using the latest history-window mean wind vector.

    The FINO1 replay table is already sampled at 10-minute resolution, so the
    default ``history_minutes=10`` repeats the last 10-minute mean u/v vector
    across the full forecast horizon.
    """

    provides_future_preview = True

    def __init__(
        self,
        dataset_dir: str | Path,
        history_minutes: float = 10.0,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.metadata = json.loads((self.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        self.scaler = json.loads((self.dataset_dir / "scaler_train.json").read_text(encoding="utf-8"))
        self.feature_columns = list(self.metadata["feature_columns"])
        self.future_steps = int(self.metadata["future_steps"])
        self.history_steps = int(self.metadata["history_steps"])
        resolution_min = float(self.metadata.get("input_resolution_minutes", 10.0))
        self.history_mean_steps = max(1, int(round(float(history_minutes) / resolution_min)))
        self.history_mean_steps = min(self.history_mean_steps, self.history_steps)
        self.u_idx = self.feature_columns.index("wind_u_ms")
        self.v_idx = self.feature_columns.index("wind_v_ms")
        self.model_version = f"persistence_last_{int(round(self.history_mean_steps * resolution_min))}min_mean"

    def _inverse_scale_history_uv(self, x: np.ndarray) -> np.ndarray:
        feature_scaler = self.scaler["feature_scaler"]
        mean = np.array(
            [feature_scaler["wind_u_ms"]["mean"], feature_scaler["wind_v_ms"]["mean"]],
            dtype=np.float32,
        )
        std = np.array(
            [feature_scaler["wind_u_ms"]["std"], feature_scaler["wind_v_ms"]["std"]],
            dtype=np.float32,
        )
        uv_scaled = x[:, [self.u_idx, self.v_idx]].astype(np.float32)
        return uv_scaled * std + mean

    @staticmethod
    def _speed_direction_from_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        direction = (np.rad2deg(np.arctan2(-uv[:, 0], -uv[:, 1])) + 360.0) % 360.0
        return speed.astype(np.float32), direction.astype(np.float32)

    def predict_window(self, x_window: np.ndarray, timestamp: str | None = None) -> ForecastResult:
        x = np.asarray(x_window, dtype=np.float32)
        if x.ndim == 3:
            x = x[0]
        expected = (self.history_steps, len(self.feature_columns))
        if x.shape != expected:
            raise ValueError(f"expected window shape {expected}, got {x.shape}")
        history_uv = self._inverse_scale_history_uv(x[-self.history_mean_steps :])
        uv_mean = np.mean(history_uv, axis=0).astype(np.float32)
        pred_uv_raw = np.repeat(uv_mean[None, :], repeats=self.future_steps, axis=0)
        speed, direction = self._speed_direction_from_uv(pred_uv_raw)
        return ForecastResult(
            wind_uv_raw=pred_uv_raw,
            wind_speed=speed,
            wind_dir_deg=direction,
            event_probs={},
            risk_levels={},
            model_version=self.model_version,
            timestamp=timestamp,
        )


class CurrentOnlyForecastAdapter(PersistenceMeanForecastAdapter):
    """No-preview baseline: expose only current wind information to the planner.

    The adapter does not use measured future wind or a learned preview. It
    repeats the latest observed mean wind across the planner horizon, while
    ``provides_future_preview=False`` prevents forecast-only guards from treating
    that repetition as real evidence of future relief. This is deliberately
    conservative: unknown future should not look like zero wind.
    """

    provides_future_preview = False

    def __init__(
        self,
        dataset_dir: str | Path,
        history_minutes: float = 10.0,
        current_block_steps: int = 2,
    ) -> None:
        super().__init__(dataset_dir=dataset_dir, history_minutes=history_minutes)
        self.current_block_steps = max(1, int(current_block_steps))
        resolution_min = float(self.metadata.get("input_resolution_minutes", 10.0))
        self.model_version = (
            f"current_only_repeat_last_{int(round(self.history_mean_steps * resolution_min))}"
            "min"
        )

    def predict_window(self, x_window: np.ndarray, timestamp: str | None = None) -> ForecastResult:
        x = np.asarray(x_window, dtype=np.float32)
        if x.ndim == 3:
            x = x[0]
        expected = (self.history_steps, len(self.feature_columns))
        if x.shape != expected:
            raise ValueError(f"expected window shape {expected}, got {x.shape}")
        history_uv = self._inverse_scale_history_uv(x[-self.history_mean_steps :])
        uv_mean = np.mean(history_uv, axis=0).astype(np.float32)
        pred_uv_raw = np.repeat(uv_mean[None, :], self.future_steps, axis=0)
        speed, direction = self._speed_direction_from_uv(pred_uv_raw)
        return ForecastResult(
            wind_uv_raw=pred_uv_raw,
            wind_speed=speed,
            wind_dir_deg=direction,
            event_probs={},
            risk_levels={},
            model_version=self.model_version,
            timestamp=timestamp,
        )


class BlendedForecastAdapter:
    """Default-off baseline/e15 forecast blender for offline closed-loop probes."""

    provides_future_preview = True

    def __init__(
        self,
        baseline_model_dir: str | Path,
        relief_model_dir: str | Path,
        dataset_dir: str | Path | None = None,
        mode: str = "soft_blend_03",
        device: str = "cpu",
    ) -> None:
        if mode not in (
            "soft_blend_03",
            "dynamic_alpha",
            "risk_floor_blend_v1",
            "confirmed_relief_alpha_v1",
        ):
            raise ValueError(f"unsupported blended forecast mode: {mode}")
        self.mode = str(mode)
        self.baseline = ForecastModelAdapter(
            model_dir=baseline_model_dir,
            dataset_dir=dataset_dir,
            device=device,
        )
        self.relief = ForecastModelAdapter(
            model_dir=relief_model_dir,
            dataset_dir=dataset_dir,
            device=device,
        )
        self.model_version = f"blended_{self.mode}"
        # Per-call diagnostic for closed-loop attribution. Populated by
        # ``predict_window`` and read by the casebook runner to surface
        # bucket-level blend decisions in the planner_log telemetry. Only
        # active when a blended forecast source is selected — has no effect
        # on baseline / e15 / oracle / persistence runs.
        self.last_diagnostic: dict[str, Any] = {
            "blend_mode": self.mode,
            "blend_alpha": 0.0,
            "blend_alpha_candidate": 0.0,
            "blend_floor_ratio": 1.0,
            "blend_floor_triggered": 0,
            "blend_floor_shrink_amount": 0.0,
            "blend_baseline_relief": 0,
            "blend_baseline_weak_relief": 0,
            "blend_baseline_confirmed_relief": 0,
            "blend_baseline_intensify": 0,
            "blend_baseline_strong_intensify": 0,
            "blend_e15_relief": 0,
            "blend_e15_strong_relief": 0,
            "blend_e15_relief_blocks_count": 0,
            "blend_e15_intensify": 0,
            "blend_swing_proxy": 0,
            "blend_confirmation_pass": 0,
            "blend_rule_reason": "not_evaluated",
        }

    @staticmethod
    def _speed_direction_from_uv(uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        direction = (np.rad2deg(np.arctan2(-uv[:, 0], -uv[:, 1])) + 360.0) % 360.0
        return speed.astype(np.float32), direction.astype(np.float32)

    @staticmethod
    def _ratios(uv: np.ndarray) -> tuple[float, float]:
        speed = np.sqrt(uv[:, 0] ** 2 + uv[:, 1] ** 2)
        blocks = []
        for start, end in ((0, 2), (2, 4), (4, 6)):
            block = speed[start:end]
            blocks.append(float(np.mean(block)) if block.size else 0.0)
        b0 = max(blocks[0], 1e-6)
        return blocks[1] / b0, blocks[2] / b0

    @classmethod
    def _relief_blocks_count(cls, uv: np.ndarray, threshold: float = 0.95) -> int:
        r1, r2 = cls._ratios(uv)
        return int(r1 < threshold) + int(r2 < threshold)

    @classmethod
    def _relief(cls, uv: np.ndarray, threshold: float = 0.95) -> bool:
        r1, r2 = cls._ratios(uv)
        return min(r1, r2) < threshold

    @classmethod
    def _weak_relief(cls, uv: np.ndarray) -> bool:
        return cls._relief(uv, threshold=0.98)

    @classmethod
    def _confirmed_relief(cls, uv: np.ndarray) -> bool:
        return cls._relief_blocks_count(uv, threshold=0.95) >= 2

    @classmethod
    def _strong_relief(cls, uv: np.ndarray) -> bool:
        return cls._relief(uv, threshold=0.85)

    @classmethod
    def _intensify(cls, uv: np.ndarray, threshold: float = 1.05) -> bool:
        r1, r2 = cls._ratios(uv)
        return max(r1, r2) > threshold

    @classmethod
    def _strong_intensify(cls, uv: np.ndarray) -> bool:
        return cls._intensify(uv, threshold=1.15)

    @classmethod
    def _swing_proxy(cls, baseline_uv: np.ndarray, relief_uv: np.ndarray) -> bool:
        for uv in (baseline_uv, relief_uv):
            r1, r2 = cls._ratios(uv)
            if (r1 - 1.0) * (r2 - r1) < -0.01:
                return True
        base_delta = cls._ratios(baseline_uv)[1] - 1.0
        relief_delta = cls._ratios(relief_uv)[1] - 1.0
        return bool(np.sign(base_delta) * np.sign(relief_delta) < 0)

    def _alpha(
        self, baseline_uv: np.ndarray, relief_uv: np.ndarray
    ) -> tuple[float, str, dict[str, float | int]]:
        # Returns (alpha, reason). Reason is a stable string code for
        # bucket-level attribution telemetry. Decision logic is unchanged
        # from the prior float-returning version; the reason strings simply
        # name the branch each call took. Default-off telemetry only.
        diag: dict[str, float | int] = {
            "alpha_candidate": 0.0,
            "floor_ratio": 1.0,
            "floor_triggered": 0,
            "floor_shrink_amount": 0.0,
            "confirmation_pass": 0,
        }
        if self.mode == "soft_blend_03":
            relief_ok = self._relief(relief_uv)
            base_intensify = self._intensify(baseline_uv)
            if relief_ok and not base_intensify:
                diag["alpha_candidate"] = 0.3
                return 0.3, "soft03_relief_and_not_base_intensify", diag
            if not relief_ok:
                return 0.0, "soft03_no_relief", diag
            return 0.0, "soft03_baseline_intensify_blocks", diag
        # mode == "dynamic_alpha"
        baseline_relief = self._relief(baseline_uv)
        relief_relief = self._relief(relief_uv)
        baseline_intensify = self._intensify(baseline_uv)
        if self.mode == "dynamic_alpha":
            if self._strong_intensify(baseline_uv):
                return 0.0, "dynamic_strong_baseline_intensify", diag
            if baseline_intensify and relief_relief:
                return 0.0, "dynamic_baseline_intensify_vs_e15_relief_conflict", diag
            if self._swing_proxy(baseline_uv, relief_uv):
                return 0.0, "dynamic_swing_proxy_blocks", diag
            if baseline_relief and relief_relief:
                diag["alpha_candidate"] = 0.3
                return 0.3, "dynamic_both_relief", diag
            if self._strong_relief(relief_uv) and not baseline_relief and not baseline_intensify:
                diag["alpha_candidate"] = 0.3
                return 0.3, "dynamic_e15_strong_relief_baseline_neutral", diag
            if relief_relief and not baseline_relief and not baseline_intensify:
                diag["alpha_candidate"] = 0.1
                return 0.1, "dynamic_e15_relief_baseline_neutral", diag
            return 0.0, "dynamic_no_rule_match", diag

        if self.mode == "confirmed_relief_alpha_v1":
            relief_blocks = self._relief_blocks_count(relief_uv)
            e15_strong_relief = self._strong_relief(relief_uv)
            swing = self._swing_proxy(baseline_uv, relief_uv)
            baseline_flat = not baseline_relief and not baseline_intensify
            confirmed = relief_blocks >= 2 or (e15_strong_relief and not baseline_intensify)
            diag["confirmation_pass"] = int(confirmed and not swing)
            if baseline_intensify:
                return 0.0, "confirmed_baseline_intensify_blocks", diag
            if swing:
                return 0.0, "confirmed_swing_proxy_blocks", diag
            if baseline_relief and relief_blocks >= 2:
                diag["alpha_candidate"] = 0.5
                return 0.5, "confirmed_both_relief_two_blocks_alpha05", diag
            if e15_strong_relief and baseline_flat and confirmed:
                diag["alpha_candidate"] = 0.3
                return 0.3, "confirmed_e15_strong_relief_baseline_flat_alpha03", diag
            if relief_relief and baseline_flat:
                return 0.0, "confirmed_single_block_relief_rejected", diag
            return 0.0, "confirmed_no_rule_match", diag

        # mode == "risk_floor_blend_v1"
        if self._strong_intensify(baseline_uv):
            return 0.0, "risk_floor_strong_baseline_intensify", diag
        if baseline_intensify and relief_relief:
            return 0.0, "risk_floor_baseline_intensify_vs_e15_relief_conflict", diag
        swing = self._swing_proxy(baseline_uv, relief_uv)
        if swing:
            diag["alpha_candidate"] = 0.1
            return 0.1, "risk_floor_swing_proxy_alpha01", diag
        baseline_flat = not baseline_relief and not baseline_intensify
        alpha_candidate = 0.0
        reason = "risk_floor_no_rule_match"
        if baseline_relief and relief_relief:
            alpha_candidate = 0.5
            reason = "risk_floor_both_relief_alpha05"
        elif self._strong_relief(relief_uv) and baseline_flat:
            alpha_candidate = 0.3
            reason = "risk_floor_e15_strong_relief_baseline_flat_alpha03"
        elif relief_relief and baseline_flat:
            alpha_candidate = 0.1
            reason = "risk_floor_e15_relief_baseline_flat_alpha01"
        if alpha_candidate <= 0.0:
            return 0.0, reason, diag
        floor_ratio = 0.95
        if self._confirmed_relief(baseline_uv):
            floor_ratio = 0.85
        elif self._weak_relief(baseline_uv):
            floor_ratio = 0.90
        alpha_final, shrink = self._apply_risk_floor(
            baseline_uv, relief_uv, alpha_candidate, floor_ratio
        )
        diag["alpha_candidate"] = float(alpha_candidate)
        diag["floor_ratio"] = float(floor_ratio)
        diag["floor_triggered"] = int(shrink > 1e-9)
        diag["floor_shrink_amount"] = float(shrink)
        if shrink > 1e-9:
            reason = f"{reason}_floor_shrunk"
        return alpha_final, reason, diag

    @classmethod
    def _apply_risk_floor(
        cls,
        baseline_uv: np.ndarray,
        relief_uv: np.ndarray,
        alpha_candidate: float,
        floor_ratio: float,
    ) -> tuple[float, float]:
        base_speed = np.sqrt(baseline_uv[:, 0] ** 2 + baseline_uv[:, 1] ** 2)
        relief_speed = np.sqrt(relief_uv[:, 0] ** 2 + relief_uv[:, 1] ** 2)
        candidate = float(alpha_candidate)
        allowed = candidate
        for start, end in ((0, 2), (2, 4), (4, 6)):
            b = float(np.mean(base_speed[start:end]))
            e = float(np.mean(relief_speed[start:end]))
            if b <= 1e-6 or e >= b * floor_ratio:
                continue
            # Mean speed under blend is linear in alpha only when directions
            # match exactly. This conservative scalar bound still prevents
            # the common false-relief failure: e15 pulling magnitude far below
            # baseline in neutral/risky buckets.
            max_alpha = (b * (1.0 - floor_ratio)) / max(b - e, 1e-6)
            allowed = min(allowed, max_alpha)
        alpha_final = float(np.clip(allowed, 0.0, candidate))
        return alpha_final, float(candidate - alpha_final)

    def predict_window(self, x_window: np.ndarray, timestamp: str | None = None) -> ForecastResult:
        base = self.baseline.predict_window(x_window, timestamp=timestamp)
        relief = self.relief.predict_window(x_window, timestamp=timestamp)
        alpha, rule_reason, alpha_diag = self._alpha(base.wind_uv_raw, relief.wind_uv_raw)
        # Per-bucket diagnostic for closed-loop attribution. Computed
        # independently of the alpha decision so we can see WHY each
        # branch was taken, not just the final alpha. All sub-flag checks
        # use the exact same predicates the _alpha rule chain uses, so
        # they are consistent with the rule_reason returned above.
        self.last_diagnostic = {
            "blend_mode": self.mode,
            "blend_alpha": float(alpha),
            "blend_alpha_candidate": float(alpha_diag.get("alpha_candidate", alpha)),
            "blend_floor_ratio": float(alpha_diag.get("floor_ratio", 1.0)),
            "blend_floor_triggered": int(alpha_diag.get("floor_triggered", 0)),
            "blend_floor_shrink_amount": float(alpha_diag.get("floor_shrink_amount", 0.0)),
            "blend_baseline_relief": int(self._relief(base.wind_uv_raw)),
            "blend_baseline_weak_relief": int(self._weak_relief(base.wind_uv_raw)),
            "blend_baseline_confirmed_relief": int(self._confirmed_relief(base.wind_uv_raw)),
            "blend_baseline_intensify": int(self._intensify(base.wind_uv_raw)),
            "blend_baseline_strong_intensify": int(self._strong_intensify(base.wind_uv_raw)),
            "blend_e15_relief": int(self._relief(relief.wind_uv_raw)),
            "blend_e15_strong_relief": int(self._strong_relief(relief.wind_uv_raw)),
            "blend_e15_relief_blocks_count": int(self._relief_blocks_count(relief.wind_uv_raw)),
            "blend_e15_intensify": int(self._intensify(relief.wind_uv_raw)),
            "blend_swing_proxy": int(self._swing_proxy(base.wind_uv_raw, relief.wind_uv_raw)),
            "blend_confirmation_pass": int(alpha_diag.get("confirmation_pass", 0)),
            "blend_rule_reason": rule_reason,
        }
        pred_uv_raw = (
            float(alpha) * np.asarray(relief.wind_uv_raw, dtype=np.float32)
            + (1.0 - float(alpha)) * np.asarray(base.wind_uv_raw, dtype=np.float32)
        )
        speed, direction = self._speed_direction_from_uv(pred_uv_raw)
        event_keys = set(base.event_probs) | set(relief.event_probs)
        event_probs = {
            key: max(float(base.event_probs.get(key, 0.0)), float(relief.event_probs.get(key, 0.0)))
            for key in event_keys
        }
        risk_keys = set(base.risk_levels) | set(relief.risk_levels)
        risk_levels = {
            key: bool(base.risk_levels.get(key, False) or relief.risk_levels.get(key, False))
            for key in risk_keys
        }
        return ForecastResult(
            wind_uv_raw=pred_uv_raw.astype(np.float32),
            wind_speed=speed,
            wind_dir_deg=direction,
            event_probs=event_probs,
            risk_levels=risk_levels,
            model_version=f"{self.model_version}_alpha{alpha:.2f}",
            timestamp=timestamp,
        )
