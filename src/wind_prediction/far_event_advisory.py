from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .forecast_adapter import ForecastModelAdapter

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"
FAR_BLOCK_COLUMNS = ("far_horizon_norm_60_80", "far_horizon_norm_80_100", "far_horizon_norm_100_120")
FAR_BLOCK_OFFSETS_MIN = (60, 80, 100)
EVENT_FLAG_COLUMNS = (
    "speed_ramp_ge_3ms",
    "direction_shift_ge_45deg",
    "vector_change_ge_train_p90",
    "future_speed_ge_train_p95",
    "ballast_attention_event",
)
SCENARIO_COLUMNS = (
    "quiet_no_event",
    "ramp_only",
    "direction_shift_only",
    "vector_change_only",
    "high_speed_only",
    "compound_multi_flag",
    "attention_event",
)

# Severity bands derived from the test calibration (score -> empirical event rate):
# below WATCH_ENTER the calibrated event rate is < ~1.5%; HIGH_CONFIDENCE is the score
# above which the calibrated event rate is ~0.97. These are display bands only; the
# binary alarm latch is governed by the hysteresis (tau_on / tau_off) elsewhere.
WATCH_ENTER_SCORE = 0.43
HIGH_CONFIDENCE_SCORE = 1.117
SEVERITY_LEVELS = ("unavailable", "clear", "watch", "advisory", "high_confidence_advisory")


def classify_severity(
    score: float | None,
    *,
    tau_on: float,
    alarm: bool | None = None,
    watch_enter: float = WATCH_ENTER_SCORE,
    high_confidence: float = HIGH_CONFIDENCE_SCORE,
) -> str:
    """Map a far-event score (+ optional latched alarm) to a display severity level.

    ``alarm`` lets a streaming caller pass the hysteresis-latched state so the level
    stays ``advisory`` while the alarm is held between tau_off and tau_on. When
    ``alarm`` is None the instantaneous ``score >= tau_on`` is used.
    """
    if score is None or not np.isfinite(float(score)):
        return "unavailable"
    s = float(score)
    latched = (s >= float(tau_on)) if alarm is None else bool(alarm)
    if latched:
        return "high_confidence_advisory" if s >= float(high_confidence) else "advisory"
    if s >= float(watch_enter):
        return "watch"
    return "clear"


@dataclass(frozen=True)
class AdvisoryOperatingPoint:
    selection_objective: str
    tau_on: float
    tau_off: float
    validation_precision: float
    validation_recall: float
    validation_fpr: float
    validation_f1: float
    test_precision: float
    test_recall: float
    test_fpr: float
    test_f1: float
    mean_lead_min: float
    selection_note: str = ""


@dataclass(frozen=True)
class FarEventAdvisoryResult:
    far_event_score: float
    far_event_prob: float
    alarm: bool
    lead_estimate_min: float
    regime_tag: str
    confidence: float
    tau_on: float
    tau_off: float
    model_version: str
    timestamp: str | None = None
    severity: str = "unavailable"


def _ensure_2d_uv(uv: np.ndarray) -> np.ndarray:
    arr = np.asarray(uv, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3 or arr.shape[-1] != 2 or arr.shape[1] < 12:
        raise ValueError(f"expected uv shape [N, >=12, 2], got {arr.shape}")
    return arr


def far_block_scores_from_uv(uv: np.ndarray, wind_ref: float = 12.0, cap: float = 1.5) -> np.ndarray:
    arr = _ensure_2d_uv(uv)
    blocks = []
    for start, end in ((6, 8), (8, 10), (10, 12)):
        speed = np.sqrt(arr[:, start:end, 0] ** 2 + arr[:, start:end, 1] ** 2).mean(axis=1)
        block = np.clip((np.maximum(speed, 0.0) / float(wind_ref)) ** 2, 0.0, float(cap))
        blocks.append(block.astype(np.float32))
    return np.stack(blocks, axis=1)


def far_event_score_from_uv(uv: np.ndarray, wind_ref: float = 12.0, cap: float = 1.5) -> np.ndarray:
    return far_block_scores_from_uv(uv, wind_ref=wind_ref, cap=cap).max(axis=1)


def far_event_label_from_uv(uv: np.ndarray, theta: float = 1.0) -> np.ndarray:
    return (far_event_score_from_uv(uv) >= float(theta)).astype(np.int8)


def predict_far_event_score(
    adapter: ForecastModelAdapter,
    x_window: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    adapter._lazy_load_runtime()
    torch = adapter._torch
    x = np.asarray(x_window, dtype=np.float32)
    if x.ndim == 2:
        x = x[None, :, :]
    scores: list[np.ndarray] = []
    for start in range(0, len(x), int(batch_size)):
        xb = torch.from_numpy(x[start : start + int(batch_size)]).to(adapter._device)
        with torch.no_grad():
            pred_uv_scaled, _ = adapter._model(xb)
            pred_uv_scaled = adapter._full_scaled_prediction(pred_uv_scaled, xb)
        pred_uv_raw = adapter._inverse_scale_uv(pred_uv_scaled.detach().cpu().numpy().astype(np.float32))
        scores.append(far_event_score_from_uv(pred_uv_raw))
    return np.concatenate(scores, axis=0)


def _binary_counts(label: np.ndarray, fire: np.ndarray) -> dict[str, float]:
    label = np.asarray(label, dtype=bool)
    fire = np.asarray(fire, dtype=bool)
    tp = int(np.logical_and(label, fire).sum())
    fp = int(np.logical_and(~label, fire).sum())
    fn = int(np.logical_and(label, ~fire).sum())
    tn = int(np.logical_and(~label, ~fire).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "support": float(label.sum()),
        "negative": float((~label).sum()),
        "fires": float(fire.sum()),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


def pr_frontier(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    round_to: int = 4,
) -> pd.DataFrame:
    score = np.asarray(scores, dtype=float).reshape(-1)
    label = np.asarray(labels, dtype=int).reshape(-1)
    if thresholds is None:
        thresholds = np.unique(np.round(score, round_to))
    rows = []
    for tau in thresholds:
        fire = score >= float(tau)
        row = {"tau": float(tau), **_binary_counts(label, fire)}
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["tau"], ascending=True).reset_index(drop=True)
    return out


def select_operating_point(
    frontier: pd.DataFrame,
    *,
    objective: str = "max_f1",
    precision_floor: float = 0.90,
    recall_floor: float = 0.95,
) -> dict[str, Any]:
    if frontier.empty:
        raise ValueError("frontier is empty")
    f = frontier.copy()
    if objective == "max_f1":
        best = f.sort_values(
            ["f1", "precision", "recall", "tau"],
            ascending=[False, False, False, True],
        ).iloc[0]
        note = "validation max-F1"
    elif objective == "precision_floor":
        filt = f[f["precision"] >= float(precision_floor)]
        if filt.empty:
            best = f.sort_values(["precision", "recall", "tau"], ascending=[False, False, True]).iloc[0]
            note = f"precision_floor_unreachable@{precision_floor:.2f}"
        else:
            best = filt.sort_values(["recall", "precision", "tau"], ascending=[False, False, True]).iloc[0]
            note = f"validation best recall with precision>= {precision_floor:.2f}"
    elif objective == "recall_floor":
        filt = f[f["recall"] >= float(recall_floor)]
        if filt.empty:
            best = f.sort_values(["recall", "precision", "tau"], ascending=[False, False, True]).iloc[0]
            note = f"recall_floor_unreachable@{recall_floor:.2f}"
        else:
            best = filt.sort_values(["precision", "recall", "tau"], ascending=[False, False, True]).iloc[0]
            note = f"validation best precision with recall>= {recall_floor:.2f}"
    else:
        raise ValueError(f"unsupported objective={objective!r}")
    return {
        "objective": objective,
        "note": note,
        "tau": float(best["tau"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "fpr": float(best["fpr"]),
        "f1": float(best["f1"]),
        "support": int(best["support"]),
        "fires": int(best["fires"]),
    }


def calibration_bins(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "score": np.asarray(scores, dtype=float).reshape(-1),
            "label": np.asarray(labels, dtype=int).reshape(-1),
        }
    )
    if frame.empty:
        return pd.DataFrame(columns=["bin", "score_min", "score_max", "count", "event_rate", "mean_score"])
    try:
        frame["bin"] = pd.qcut(frame["score"], q=min(int(n_bins), frame["score"].nunique()), duplicates="drop")
    except ValueError:
        frame["bin"] = pd.cut(frame["score"], bins=min(int(n_bins), max(int(frame["score"].nunique()), 1)))
    rows = []
    for idx, (bin_label, group) in enumerate(frame.groupby("bin", dropna=True), start=1):
        rows.append(
            {
                "bin": idx,
                "score_min": float(group["score"].min()),
                "score_max": float(group["score"].max()),
                "count": int(len(group)),
                "event_rate": float(group["label"].mean()) if len(group) else np.nan,
                "mean_score": float(group["score"].mean()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _scenario_masks(sample_index: pd.DataFrame) -> dict[str, np.ndarray]:
    frame = sample_index.copy()
    for col in EVENT_FLAG_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0
    ramp = frame["speed_ramp_ge_3ms"].astype(bool).to_numpy()
    shift = frame["direction_shift_ge_45deg"].astype(bool).to_numpy()
    vector = frame["vector_change_ge_train_p90"].astype(bool).to_numpy()
    high = frame["future_speed_ge_train_p95"].astype(bool).to_numpy()
    count = ramp.astype(int) + shift.astype(int) + vector.astype(int) + high.astype(int)
    any_event = frame["ballast_attention_event"].astype(bool).to_numpy()
    return {
        "quiet_no_event": ~(ramp | shift | vector | high),
        "ramp_only": ramp & ~shift & ~vector & ~high,
        "direction_shift_only": shift & ~ramp & ~vector & ~high,
        "vector_change_only": vector & ~ramp & ~shift & ~high,
        "high_speed_only": high & ~ramp & ~shift & ~vector,
        "compound_multi_flag": count >= 2,
        "attention_event": any_event,
    }


def classify_regime(row: pd.Series) -> str:
    sample = pd.DataFrame([row])
    masks = _scenario_masks(sample)
    for key in (
        "compound_multi_flag",
        "high_speed_only",
        "ramp_only",
        "direction_shift_only",
        "vector_change_only",
        "attention_event",
        "quiet_no_event",
    ):
        if bool(np.asarray(masks[key])[0]):
            return key
    return "quiet_no_event"


def build_scenario_table(
    sample_index: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float,
    *,
    split: str,
) -> pd.DataFrame:
    frame = sample_index.copy().reset_index(drop=True)
    frame["score"] = np.asarray(scores, dtype=float).reshape(-1)
    frame["label"] = np.asarray(labels, dtype=int).reshape(-1)
    fire = frame["score"] >= float(tau)
    masks = _scenario_masks(frame)
    rows: list[dict[str, Any]] = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            continue
        metrics = _binary_counts(frame["label"].to_numpy()[mask], fire.to_numpy()[mask])
        rows.append(
            {
                "split": split,
                "scenario": name,
                "count": int(mask.sum()),
                "base_rate": float(frame.loc[mask, "label"].mean()),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "scenario"], ascending=[False, True]).reset_index(drop=True)


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def hysteresis_alarm_state(
    scores: np.ndarray,
    *,
    tau_on: float,
    tau_off: float | None = None,
    min_on_rows: int = 1,
    min_off_rows: int = 1,
) -> np.ndarray:
    score = np.asarray(scores, dtype=float).reshape(-1)
    if tau_off is None:
        tau_off = max(float(tau_on) - 0.05, 0.0)
    state = False
    on_streak = 0
    off_streak = 0
    alarm = np.zeros(len(score), dtype=bool)
    for idx, value in enumerate(score):
        if not state:
            if value >= float(tau_on):
                on_streak += 1
            else:
                on_streak = 0
            if on_streak >= int(min_on_rows):
                state = True
                on_streak = 0
                off_streak = 0
        else:
            if value < float(tau_off):
                off_streak += 1
            else:
                off_streak = 0
            if off_streak >= int(min_off_rows):
                state = False
                off_streak = 0
        alarm[idx] = state
    return alarm


def build_event_alarm_tables(
    sample_index: pd.DataFrame,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    tau_on: float,
    tau_off: float | None = None,
    horizon_min: float = 120.0,
    min_on_rows: int = 1,
    min_off_rows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = sample_index.copy().reset_index(drop=True)
    frame["score"] = np.asarray(scores, dtype=float).reshape(-1)
    frame["label"] = np.asarray(labels, dtype=int).reshape(-1)
    frame["alarm"] = hysteresis_alarm_state(
        frame["score"].to_numpy(),
        tau_on=float(tau_on),
        tau_off=tau_off,
        min_on_rows=min_on_rows,
        min_off_rows=min_off_rows,
    )
    frame["future_start"] = pd.to_datetime(frame["future_start"])
    frame["future_end"] = pd.to_datetime(frame["future_end"])
    frame["scenario"] = frame.apply(classify_regime, axis=1)
    horizon = pd.Timedelta(minutes=float(horizon_min))
    truth_rows: list[dict[str, Any]] = []
    alarm_rows: list[dict[str, Any]] = []
    for series_id, group in frame.sort_values(["series_id", "future_start"]).groupby("series_id", sort=False):
        g = group.reset_index(drop=True)
        truth_runs = _contiguous_runs(g["label"].to_numpy() > 0)
        alarm_runs = _contiguous_runs(g["alarm"].to_numpy() > 0)
        truth_starts = [g.loc[s, "future_start"] for s, _ in truth_runs]
        truth_ends = [g.loc[e, "future_end"] for _, e in truth_runs]
        alarm_starts = [g.loc[s, "future_start"] for s, _ in alarm_runs]
        alarm_ends = [g.loc[e, "future_end"] for _, e in alarm_runs]
        for ep_idx, ((s_idx, e_idx), start_t, end_t) in enumerate(zip(truth_runs, truth_starts, truth_ends), start=1):
            window = (g["future_start"] >= (start_t - horizon)) & (g["future_start"] < start_t) & g["alarm"]
            alarm_times = g.loc[window, "future_start"]
            hit = not alarm_times.empty
            lead_min = float((start_t - alarm_times.min()).total_seconds() / 60.0) if hit else np.nan
            truth_rows.append(
                {
                    "split": split,
                    "series_id": series_id,
                    "truth_episode_id": f"{series_id}::truth{ep_idx:03d}",
                    "truth_start": start_t.strftime(TIMESTAMP_FMT),
                    "truth_end": end_t.strftime(TIMESTAMP_FMT),
                    "truth_len_rows": int(e_idx - s_idx + 1),
                    "hit": int(hit),
                    "lead_min": lead_min,
                    "first_alarm_time": alarm_times.min().strftime(TIMESTAMP_FMT) if hit else "",
                    "scenario": g.loc[s_idx:e_idx, "scenario"].mode().iloc[0] if not g.loc[s_idx:e_idx, "scenario"].empty else "unknown",
                }
            )
        for ep_idx, ((s_idx, e_idx), start_t, end_t) in enumerate(zip(alarm_runs, alarm_starts, alarm_ends), start=1):
            window = (g["future_start"] >= start_t) & (g["future_start"] <= start_t + horizon) & (g["label"] > 0)
            truth_times = g.loc[window, "future_start"]
            hit = not truth_times.empty
            alarm_rows.append(
                {
                    "split": split,
                    "series_id": series_id,
                    "alarm_episode_id": f"{series_id}::alarm{ep_idx:03d}",
                    "alarm_start": start_t.strftime(TIMESTAMP_FMT),
                    "alarm_end": end_t.strftime(TIMESTAMP_FMT),
                    "alarm_len_rows": int(e_idx - s_idx + 1),
                    "hit": int(hit),
                    "first_truth_time": truth_times.min().strftime(TIMESTAMP_FMT) if hit else "",
                    "scenario": g.loc[s_idx:e_idx, "scenario"].mode().iloc[0] if not g.loc[s_idx:e_idx, "scenario"].empty else "unknown",
                }
            )
    truth_df = pd.DataFrame(truth_rows)
    alarm_df = pd.DataFrame(alarm_rows)
    total_hours = max(len(frame) * 10.0 / 60.0, 1.0)
    summary = pd.DataFrame(
        [
            {
                "split": split,
                "tau_on": float(tau_on),
                "tau_off": float(tau_off if tau_off is not None else max(float(tau_on) - 0.05, 0.0)),
                "rows": int(len(frame)),
                "alarm_duty_cycle": float(frame["alarm"].mean()) if len(frame) else np.nan,
                "truth_episodes": int(len(truth_df)),
                "truth_hits": int(truth_df["hit"].sum()) if not truth_df.empty else 0,
                "truth_recall": float(truth_df["hit"].mean()) if len(truth_df) else np.nan,
                "median_lead_min": float(np.nanmedian(truth_df.loc[truth_df["hit"] > 0, "lead_min"])) if truth_df["hit"].any() else np.nan,
                "mean_lead_min": float(np.nanmean(truth_df.loc[truth_df["hit"] > 0, "lead_min"])) if truth_df["hit"].any() else np.nan,
                "alarm_episodes": int(len(alarm_df)),
                "false_alarm_episodes": int((alarm_df["hit"] == 0).sum()) if not alarm_df.empty else 0,
                "false_alarm_episodes_per_day": float(((alarm_df["hit"] == 0).sum()) / (total_hours / 24.0)) if len(alarm_df) else np.nan,
                "true_alarm_episodes_per_day": float(((alarm_df["hit"] == 1).sum()) / (total_hours / 24.0)) if len(alarm_df) else np.nan,
            }
        ]
    )
    return truth_df, alarm_df, summary


def build_confidence_lookup(calibration: pd.DataFrame) -> list[dict[str, float]]:
    if calibration.empty:
        return []
    rows = []
    for row in calibration.to_dict("records"):
        rows.append(
            {
                "score_min": float(row["score_min"]),
                "score_max": float(row["score_max"]),
                "confidence": float(row["event_rate"]),
                "count": int(row["count"]),
            }
        )
    return rows


def lookup_confidence(score: float, calibration_lookup: list[dict[str, float]] | None = None) -> float:
    if not calibration_lookup:
        return float("nan")
    value = float(score)
    for row in calibration_lookup:
        if value <= row["score_max"]:
            if value >= row["score_min"]:
                return float(row["confidence"])
    return float(calibration_lookup[-1]["confidence"])


def estimate_lead_from_point(mean_lead_min: float | None, alarm: bool) -> float:
    if not alarm:
        return 0.0
    if mean_lead_min is None or not np.isfinite(mean_lead_min):
        return float("nan")
    return float(mean_lead_min)


def advisory_result(
    score: float,
    *,
    tau_on: float,
    tau_off: float | None = None,
    mean_lead_min: float | None = None,
    regime_tag: str = "unknown",
    confidence_lookup: list[dict[str, float]] | None = None,
    model_version: str = "unknown",
    timestamp: str | None = None,
) -> FarEventAdvisoryResult:
    if tau_off is None:
        tau_off = max(float(tau_on) - 0.05, 0.0)
    alarm = float(score) >= float(tau_on)
    confidence = lookup_confidence(score, confidence_lookup)
    if not np.isfinite(confidence):
        confidence = float(np.clip(score / 1.5, 0.0, 1.0))
    return FarEventAdvisoryResult(
        far_event_score=float(score),
        far_event_prob=float(np.clip(score / 1.5, 0.0, 1.0)),
        alarm=bool(alarm),
        lead_estimate_min=estimate_lead_from_point(mean_lead_min, alarm),
        regime_tag=str(regime_tag),
        confidence=float(confidence),
        tau_on=float(tau_on),
        tau_off=float(tau_off),
        model_version=str(model_version),
        timestamp=timestamp,
        severity=classify_severity(score, tau_on=tau_on, alarm=alarm),
    )


class FarEventAdvisory:
    """Thin advisory-only wrapper around the learned h120 preview model."""

    def __init__(
        self,
        model_dir: str | Path,
        dataset_dir: str | Path,
        *,
        tau_on: float,
        tau_off: float | None = None,
        mean_lead_min: float | None = None,
        device: str = "cpu",
        batch_size: int = 8192,
        confidence_lookup: list[dict[str, float]] | None = None,
    ) -> None:
        self.adapter = ForecastModelAdapter(model_dir=model_dir, dataset_dir=dataset_dir, device=device)
        self.tau_on = float(tau_on)
        self.tau_off = float(tau_off if tau_off is not None else max(self.tau_on - 0.05, 0.0))
        self.mean_lead_min = mean_lead_min
        self.batch_size = int(batch_size)
        self.confidence_lookup = confidence_lookup or []
        self.model_version = self.adapter.model_version

    def score_window(self, x_window: np.ndarray) -> float:
        score = far_event_score_from_uv(self.adapter.predict_window(x_window).wind_uv_raw)
        return float(score[0])

    def predict_window(
        self,
        x_window: np.ndarray,
        *,
        regime_row: pd.Series | dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> FarEventAdvisoryResult:
        forecast = self.adapter.predict_window(x_window, timestamp=timestamp)
        score = float(far_event_score_from_uv(forecast.wind_uv_raw)[0])
        regime_tag = "unknown"
        if regime_row is not None:
            regime_tag = classify_regime(pd.Series(regime_row))
        return advisory_result(
            score,
            tau_on=self.tau_on,
            tau_off=self.tau_off,
            mean_lead_min=self.mean_lead_min,
            regime_tag=regime_tag,
            confidence_lookup=self.confidence_lookup,
            model_version=forecast.model_version,
            timestamp=timestamp,
        )
