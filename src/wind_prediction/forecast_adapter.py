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
            name: float(event_prob_np[idx])
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
