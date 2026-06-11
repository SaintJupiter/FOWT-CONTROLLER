#!/usr/bin/env python3
"""Sweep control-facing trusted-event gates for segmented wind events.

The raw GRU event heads are useful but too noisy for direct control use. This
script evaluates lightweight post-model gates on cached test probabilities, so
we can choose a control signal by false-alarm cost rather than by classifier F1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
DEFAULT_MODEL_DIR = REPO_ROOT / "outputs/wind_prediction/gru_segmented_fino1_meteo_aux_v1"
DEFAULT_PROB_PATH = (
    REPO_ROOT
    / "outputs/wind_prediction/gru_segmented_event_recognition_by_regime_20260604/test_event_probabilities.npy"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/wind_prediction/gru_segmented_trusted_event_gate_sweep_20260604"


ATTENTION_KEYS = (
    "ballast_attention_event",
    "attention_event_0_20m",
    "attention_event_20_40m",
    "attention_event_40_60m",
)
DYNAMIC_KEYS = (
    "speed_ramp_ge_3ms",
    "direction_shift_ge_45deg",
    "vector_change_ge_train_p90",
)
HIGHWIND_KEY = "future_speed_ge_train_p95"


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def binary_counts(
    *,
    y_event: np.ndarray,
    pred: np.ndarray,
    event_idx: dict[str, int],
) -> dict[str, int | float]:
    true_any = y_event.any(axis=1)
    stable = ~true_any
    pred = pred.astype(bool)
    tp = int((true_any & pred).sum())
    fp = int((stable & pred).sum())
    fn = int((true_any & ~pred).sum())
    tn = int((stable & ~pred).sum())
    high = y_event[:, event_idx[HIGHWIND_KEY]]
    dynamic = np.zeros(len(y_event), dtype=bool)
    for key in DYNAMIC_KEYS:
        dynamic |= y_event[:, event_idx[key]]
    attention = y_event[:, event_idx["ballast_attention_event"]]
    return {
        "windows": int(len(pred)),
        "tp_hit": tp,
        "fp_false_alarm": fp,
        "fn_miss": fn,
        "tn_correct_stable": tn,
        "predicted_nonstable_count": int(pred.sum()),
        "accuracy": ratio(tp + tn, len(pred)),
        "precision": ratio(tp, tp + fp),
        "recall_detection": ratio(tp, int(true_any.sum())),
        "stable_false_alarm_rate": ratio(fp, int(stable.sum())),
        "miss_rate": ratio(fn, int(true_any.sum())),
        "highwind_recall": ratio(int((high & pred).sum()), int(high.sum())),
        "dynamic_recall": ratio(int((dynamic & pred).sum()), int(dynamic.sum())),
        "attention_recall": ratio(int((attention & pred).sum()), int(attention.sum())),
    }


def gate_prediction(
    *,
    probs: np.ndarray,
    event_idx: dict[str, int],
    highwind_threshold: float,
    attention_threshold: float,
    attention_min_heads: int,
    dynamic_threshold: float | None = None,
    dynamic_min_heads: int = 2,
) -> np.ndarray:
    highwind_hit = probs[:, event_idx[HIGHWIND_KEY]] >= float(highwind_threshold)
    attention_hits = np.zeros(probs.shape[0], dtype=int)
    for key in ATTENTION_KEYS:
        attention_hits += probs[:, event_idx[key]] >= float(attention_threshold)
    pred = highwind_hit | (attention_hits >= int(attention_min_heads))
    if dynamic_threshold is not None:
        dynamic_hits = np.zeros(probs.shape[0], dtype=int)
        for key in DYNAMIC_KEYS:
            dynamic_hits += probs[:, event_idx[key]] >= float(dynamic_threshold)
        pred |= dynamic_hits >= int(dynamic_min_heads)
    return pred


def build_rows(
    *,
    y_event: np.ndarray,
    probs: np.ndarray,
    event_columns: list[str],
    baseline_thresholds: np.ndarray,
) -> list[dict[str, int | float | str]]:
    event_idx = {name: idx for idx, name in enumerate(event_columns)}
    rows: list[dict[str, int | float | str]] = []

    def add(name: str, pred: np.ndarray, **params: float | int | str) -> None:
        rows.append(
            {
                "gate": name,
                **params,
                **binary_counts(y_event=y_event, pred=pred, event_idx=event_idx),
            }
        )

    add("baseline_any_best_f1", (probs >= baseline_thresholds.reshape(1, -1)).any(axis=1))
    add(
        "recommended_hi090_att075_k2",
        gate_prediction(
            probs=probs,
            event_idx=event_idx,
            highwind_threshold=0.90,
            attention_threshold=0.75,
            attention_min_heads=2,
        ),
        highwind_threshold=0.90,
        attention_threshold=0.75,
        attention_min_heads=2,
        dynamic_enabled=0,
    )
    add(
        "recommended_hi090_att075_k2_dyn085_k2",
        gate_prediction(
            probs=probs,
            event_idx=event_idx,
            highwind_threshold=0.90,
            attention_threshold=0.75,
            attention_min_heads=2,
            dynamic_threshold=0.85,
            dynamic_min_heads=2,
        ),
        highwind_threshold=0.90,
        attention_threshold=0.75,
        attention_min_heads=2,
        dynamic_enabled=1,
        dynamic_threshold=0.85,
        dynamic_min_heads=2,
    )

    for highwind_threshold in np.linspace(0.90, 0.98, 9):
        for attention_threshold in np.linspace(0.70, 0.90, 9):
            for attention_min_heads in (1, 2, 3):
                pred = gate_prediction(
                    probs=probs,
                    event_idx=event_idx,
                    highwind_threshold=float(highwind_threshold),
                    attention_threshold=float(attention_threshold),
                    attention_min_heads=int(attention_min_heads),
                )
                add(
                    f"hi{highwind_threshold:.2f}_att{attention_threshold:.2f}_k{attention_min_heads}",
                    pred,
                    highwind_threshold=float(highwind_threshold),
                    attention_threshold=float(attention_threshold),
                    attention_min_heads=int(attention_min_heads),
                    dynamic_enabled=0,
                )
                for dynamic_threshold in (0.80, 0.85, 0.90, 0.95):
                    pred_dyn = gate_prediction(
                        probs=probs,
                        event_idx=event_idx,
                        highwind_threshold=float(highwind_threshold),
                        attention_threshold=float(attention_threshold),
                        attention_min_heads=int(attention_min_heads),
                        dynamic_threshold=float(dynamic_threshold),
                        dynamic_min_heads=2,
                    )
                    add(
                        (
                            f"hi{highwind_threshold:.2f}_att{attention_threshold:.2f}"
                            f"_k{attention_min_heads}_dyn{dynamic_threshold:.2f}_k2"
                        ),
                        pred_dyn,
                        highwind_threshold=float(highwind_threshold),
                        attention_threshold=float(attention_threshold),
                        attention_min_heads=int(attention_min_heads),
                        dynamic_enabled=1,
                        dynamic_threshold=float(dynamic_threshold),
                        dynamic_min_heads=2,
                    )
    return rows


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df.loc[:, columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(out_path: Path, frame: pd.DataFrame) -> None:
    view = frame.copy()
    for col in (
        "accuracy",
        "precision",
        "recall_detection",
        "stable_false_alarm_rate",
        "miss_rate",
        "highwind_recall",
        "dynamic_recall",
        "attention_recall",
    ):
        view[col] = view[col].map(pct)

    key = view[
        view["gate"].isin(
            [
                "baseline_any_best_f1",
                "recommended_hi090_att075_k2",
                "recommended_hi090_att075_k2_dyn085_k2",
            ]
        )
    ]
    fpr5 = view[frame["stable_false_alarm_rate"] <= 0.05].head(12)
    fpr3 = view[frame["stable_false_alarm_rate"] <= 0.03].head(12)
    lines = [
        "# Segmented Event Trusted Gate Sweep",
        "",
        "## Key Gates",
        "",
        markdown_table(
            key,
            [
                "gate",
                "tp_hit",
                "fp_false_alarm",
                "fn_miss",
                "precision",
                "recall_detection",
                "stable_false_alarm_rate",
                "highwind_recall",
                "dynamic_recall",
                "attention_recall",
            ],
        ),
        "",
        "## Best Under 5% Stable False Alarm",
        "",
        markdown_table(
            fpr5,
            [
                "gate",
                "tp_hit",
                "fp_false_alarm",
                "fn_miss",
                "precision",
                "recall_detection",
                "stable_false_alarm_rate",
                "highwind_recall",
                "dynamic_recall",
            ],
        ),
        "",
        "## Best Under 3% Stable False Alarm",
        "",
        markdown_table(
            fpr3,
            [
                "gate",
                "tp_hit",
                "fp_false_alarm",
                "fn_miss",
                "precision",
                "recall_detection",
                "stable_false_alarm_rate",
                "highwind_recall",
                "dynamic_recall",
            ],
        ),
        "",
        "## Recommendation",
        "",
        (
            "Use `recommended_hi090_att075_k2` first for control-facing experiments: "
            "`future_speed_ge_train_p95 >= 0.90 OR at least two of "
            "ballast/0-20/20-40/40-60 attention heads >= 0.75`. It is the cleaner "
            "low-false-alarm starting point; dynamic heads remain optional because "
            "their precision is weaker."
        ),
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prob-path", type=Path, default=DEFAULT_PROB_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    model_dir = args.model_dir.resolve()
    prob_path = args.prob_path.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    event_columns = list(metadata["event_columns"])
    thresholds = json.loads((model_dir / "lstm_event_thresholds.json").read_text(encoding="utf-8"))[
        "thresholds"
    ]
    baseline_thresholds = np.asarray([thresholds[name] for name in event_columns], dtype=np.float32)
    y_event = np.load(dataset_dir / "y_event_test.npy").astype(bool)
    probs = np.load(prob_path)

    rows = build_rows(
        y_event=y_event,
        probs=probs,
        event_columns=event_columns,
        baseline_thresholds=baseline_thresholds,
    )
    frame = pd.DataFrame(rows)
    frame["selection_score"] = (
        frame["recall_detection"]
        - 2.0 * frame["stable_false_alarm_rate"]
        + 0.4 * frame["highwind_recall"]
        + 0.1 * frame["dynamic_recall"]
    )
    frame = frame.sort_values(
        ["selection_score", "recall_detection", "precision"],
        ascending=[False, False, False],
    )
    frame.to_csv(out_dir / "trusted_event_gate_sweep.csv", index=False)
    write_report(out_dir / "trusted_event_gate_sweep_report.md", frame)
    print(frame.head(20).to_string(index=False))
    print(f"[done] wrote {out_dir / 'trusted_event_gate_sweep.csv'}")
    print(f"[done] wrote {out_dir / 'trusted_event_gate_sweep_report.md'}")


if __name__ == "__main__":
    main()
