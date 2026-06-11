#!/usr/bin/env python3
"""Report segmented wind-event recognition quality by operating regime.

This is a diagnostic script for the GRU segmented wind model. It treats each
10-minute test window as a multi-label event-recognition sample, then writes a
confusion "ledger" by regime: true occurrences, predicted occurrences, hits,
misses, false alarms, precision/recall/FPR, and accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

# The local PyTorch/numeric stack can initialize two OpenMP runtimes on macOS.
# This diagnostic is read-only and deterministic; the env flag avoids an import
# abort without changing model weights or labels.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "modeling"))

from train_ballast_lstm import build_model  # noqa: E402


DEFAULT_DATASET_DIR = REPO_ROOT / "data/processed/wind_ml_10min/ballast_decision_fino1_meteo_aux_v1"
DEFAULT_MODEL_DIR = REPO_ROOT / "outputs/wind_prediction/gru_segmented_fino1_meteo_aux_v1"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/wind_prediction/gru_segmented_event_recognition_by_regime_20260604"


def pct(value: float | int | np.floating | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{float(value) * 100.0:.1f}%"


def ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def infer_event_probabilities(
    *,
    dataset_dir: Path,
    model_dir: Path,
    out_dir: Path,
    batch_size: int,
    force: bool,
) -> np.ndarray:
    cache_path = out_dir / "test_event_probabilities.npy"
    if cache_path.exists() and not force:
        return np.load(cache_path)

    config = load_json(model_dir / "lstm_config.json")
    x_test = np.load(dataset_dir / "X_test.npy", mmap_mode="r")
    event_count = int(config.get("event_count", 8))
    model = build_model(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        future_steps=int(config["future_steps"]),
        event_count=event_count,
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        model_type=str(config["model_type"]),
        conv_kernel_size=int(config.get("conv_kernel_size", 3)),
        far_hint_head=bool(config.get("far_hint_auxiliary_head", False)),
    )
    checkpoint = torch_load(model_dir / "lstm_best.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probs = np.empty((x_test.shape[0], event_count), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, x_test.shape[0], batch_size):
            end = min(start + batch_size, x_test.shape[0])
            batch = torch.from_numpy(np.array(x_test[start:end], dtype=np.float32, copy=True))
            _, event_logits = model(batch)
            probs[start:end] = torch.sigmoid(event_logits).cpu().numpy()
            if start == 0 or end == x_test.shape[0] or (start // batch_size) % 10 == 0:
                print(f"[infer] {end}/{x_test.shape[0]}", flush=True)

    np.save(cache_path, probs)
    return probs


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int | float]:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)
    tp = int(np.logical_and(yt, yp).sum())
    fp = int(np.logical_and(~yt, yp).sum())
    fn = int(np.logical_and(yt, ~yp).sum())
    tn = int(np.logical_and(~yt, ~yp).sum())
    n = int(yt.size)
    true_count = tp + fn
    pred_count = tp + fp
    return {
        "windows": n,
        "true_count": true_count,
        "predicted_count": pred_count,
        "tp_hit": tp,
        "fp_false_alarm": fp,
        "fn_miss": fn,
        "tn_correct_stable": tn,
        "accuracy": ratio(tp + tn, n),
        "precision": ratio(tp, pred_count),
        "recall_detection": ratio(tp, true_count),
        "miss_rate": ratio(fn, true_count),
        "false_alarm_rate": ratio(fp, fp + tn),
        "true_rate": ratio(true_count, n),
        "predicted_rate": ratio(pred_count, n),
    }


def any_event_rows(regimes: Iterable[tuple[str, str, np.ndarray]], y_event: np.ndarray, pred_event: np.ndarray) -> list[dict]:
    y_any = y_event.any(axis=1)
    p_any = pred_event.any(axis=1)
    rows = []
    for regime_key, regime_desc, mask in regimes:
        mask = mask.astype(bool)
        stats = binary_confusion(y_any[mask], p_any[mask])
        true_stable = int((~y_any[mask]).sum())
        pred_stable = int((~p_any[mask]).sum())
        row = {
            "regime": regime_key,
            "regime_desc": regime_desc,
            "true_nonstable_count": stats.pop("true_count"),
            "predicted_nonstable_count": stats.pop("predicted_count"),
            "true_stable_count": true_stable,
            "predicted_stable_count": pred_stable,
            **stats,
        }
        rows.append(row)
    return rows


def per_event_rows(
    regimes: Iterable[tuple[str, str, np.ndarray]],
    event_columns: list[str],
    y_event: np.ndarray,
    pred_event: np.ndarray,
) -> list[dict]:
    rows = []
    for regime_key, regime_desc, mask in regimes:
        mask = mask.astype(bool)
        for event_idx, event_name in enumerate(event_columns):
            stats = binary_confusion(y_event[mask, event_idx], pred_event[mask, event_idx])
            rows.append(
                {
                    "regime": regime_key,
                    "regime_desc": regime_desc,
                    "event": event_name,
                    **stats,
                }
            )
    return rows


def describe_any_event(row: pd.Series) -> str:
    recall = row["recall_detection"]
    fpr = row["false_alarm_rate"]
    if pd.isna(fpr) and not pd.isna(recall):
        if recall >= 0.85:
            return "事件工况：检出高，漏成 stable 少。"
        if recall >= 0.65:
            return "事件工况：检出中等，仍有漏成 stable。"
        return "事件工况：漏成 stable 偏多。"
    if pd.isna(recall):
        return "该切片无真实非稳定样本，主要看误报。"
    if recall < 0.50 and fpr < 0.02:
        return "偏保守：漏检多，误报低。"
    if recall < 0.50 and fpr >= 0.02:
        return "漏检多，且有误报。"
    if recall >= 0.75 and fpr < 0.02:
        return "识别较稳：检出高、误报低。"
    if recall >= 0.75:
        return "检出高，但稳定误报偏高。"
    return "中等：仍有明显漏检。"


def describe_event(row: pd.Series) -> str:
    recall = row["recall_detection"]
    precision = row["precision"]
    if pd.isna(recall):
        return "无真实出现，主要看误报。"
    if recall < 0.40 and precision >= 0.70:
        return "很保守：报了较准，但漏很多。"
    if recall < 0.40:
        return "检出弱：漏检是主问题。"
    if precision < 0.50:
        return "偏敏感：误报/串类较多。"
    if recall >= 0.70 and precision >= 0.70:
        return "相对可靠。"
    return "可用但需提升。"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    view = df.loc[:, columns].copy()
    headers = [str(col) for col in columns]

    def cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(cell(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def build_regimes(index_test: pd.DataFrame, y_event: np.ndarray, metadata: dict) -> list[tuple[str, str, np.ndarray]]:
    n = len(index_test)
    event_columns = list(metadata["event_columns"])
    event_sum = y_event.sum(axis=1)
    thresholds = metadata["event_thresholds"]

    ramp = index_test["future_speed_ramp_max_ms"].to_numpy()
    direction = index_test["future_dir_shift_abs_max_deg"].to_numpy()
    speed = index_test["future_speed_max_ms"].to_numpy()
    vector_change = index_test["future_vector_change_max_ms"].to_numpy()

    speed_p95 = float(thresholds["future_speed_train_p95_ms"])
    vector_p90 = float(thresholds["vector_change_train_p90_ms"])

    regimes: list[tuple[str, str, np.ndarray]] = [
        ("all_test", "全 test", np.ones(n, dtype=bool)),
        ("true_stable", "真实稳定窗口：8 个事件头均未出现", event_sum == 0),
        ("true_nonstable", "真实非稳定窗口：至少 1 个事件头出现", event_sum > 0),
        ("single_event", "单事件窗口：仅 1 个事件头出现", event_sum == 1),
        ("multi_event", "复合事件窗口：2 个及以上事件头出现", event_sum >= 2),
        ("ramp_3_6ms", "速度爬升 3-6 m/s", (ramp >= 3.0) & (ramp < 6.0)),
        ("ramp_ge_6ms", "速度爬升 >=6 m/s", ramp >= 6.0),
        ("direction_45_90deg", "方向变化 45-90 deg", (direction >= 45.0) & (direction < 90.0)),
        ("direction_ge_90deg", "方向变化 >=90 deg", direction >= 90.0),
        ("future_speed_ge_train_p95", f"未来风速 >= train p95 ({speed_p95:.2f} m/s)", speed >= speed_p95),
        ("vector_change_ge_train_p90", f"矢量变化 >= train p90 ({vector_p90:.2f} m/s)", vector_change >= vector_p90),
    ]

    for event_idx, event_name in enumerate(event_columns):
        regimes.append((f"true_{event_name}", f"真实 {event_name}", y_event[:, event_idx].astype(bool)))

    # Preserve order but drop empty slices so percentages remain meaningful.
    return [(key, desc, mask) for key, desc, mask in regimes if int(mask.sum()) > 0]


def write_markdown_report(
    *,
    out_path: Path,
    model_dir: Path,
    dataset_dir: Path,
    thresholds: dict[str, float],
    any_df: pd.DataFrame,
    event_df: pd.DataFrame,
) -> None:
    any_view = any_df.copy()
    for col in ["accuracy", "precision", "recall_detection", "miss_rate", "false_alarm_rate", "true_rate", "predicted_rate"]:
        any_view[col] = any_view[col].map(pct)
    any_view["judgement"] = any_df.apply(describe_any_event, axis=1)

    key_regimes = [
        "all_test",
        "true_stable",
        "true_nonstable",
        "single_event",
        "multi_event",
        "ramp_3_6ms",
        "ramp_ge_6ms",
        "direction_45_90deg",
        "direction_ge_90deg",
        "future_speed_ge_train_p95",
        "vector_change_ge_train_p90",
    ]
    any_main = any_view[any_view["regime"].isin(key_regimes)]

    event_all = event_df[event_df["regime"] == "all_test"].copy()
    for col in ["accuracy", "precision", "recall_detection", "miss_rate", "false_alarm_rate", "true_rate", "predicted_rate"]:
        event_all[col] = event_all[col].map(pct)
    event_all["judgement"] = event_df[event_df["regime"] == "all_test"].apply(describe_event, axis=1).values

    event_by_true_regime = event_df[
        event_df["regime"].str.startswith("true_") & (event_df["regime"] != "true_stable") & (event_df["regime"] != "true_nonstable")
    ].copy()
    event_by_true_regime = event_by_true_regime[event_by_true_regime["regime"].str.replace("true_", "", regex=False) == event_by_true_regime["event"]]
    for col in ["accuracy", "precision", "recall_detection", "miss_rate", "false_alarm_rate", "true_rate", "predicted_rate"]:
        event_by_true_regime[col] = event_by_true_regime[col].map(pct)
    event_by_true_regime["judgement"] = "该工况自身事件头检出情况"

    lines = [
        "# GRU Segmented Event Recognition By Regime",
        "",
        f"- model_dir: `{model_dir}`",
        f"- dataset_dir: `{dataset_dir}`",
        "- split: `test`",
        "- prediction rule: event probability >= validation-selected best-F1 threshold",
        f"- thresholds: `{json.dumps(thresholds, ensure_ascii=False)}`",
        "",
        "## Stable / Non-Stable Ledger",
        "",
        markdown_table(
            any_main,
            [
                "regime",
                "windows",
                "true_nonstable_count",
                "predicted_nonstable_count",
                "tp_hit",
                "fn_miss",
                "fp_false_alarm",
                "accuracy",
                "recall_detection",
                "false_alarm_rate",
                "judgement",
            ],
        ),
        "",
        "## Event Head Ledger - All Test",
        "",
        markdown_table(
            event_all,
            [
                "event",
                "windows",
                "true_count",
                "predicted_count",
                "tp_hit",
                "fp_false_alarm",
                "fn_miss",
                "precision",
                "recall_detection",
                "false_alarm_rate",
                "judgement",
            ],
        ),
        "",
        "## Event Head Ledger - Matching True Regime",
        "",
        markdown_table(
            event_by_true_regime,
            [
                "event",
                "windows",
                "true_count",
                "predicted_count",
                "tp_hit",
                "fp_false_alarm",
                "fn_miss",
                "precision",
                "recall_detection",
                "false_alarm_rate",
                "judgement",
            ],
        ),
        "",
        "## Files",
        "",
        "- `any_event_confusion_by_regime.csv`: stable/non-stable confusion by regime.",
        "- `event_head_confusion_by_regime.csv`: per-event-head confusion by regime.",
        "- `test_event_probabilities.npy`: cached event probabilities for reproducible reruns.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--force-infer", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_json(dataset_dir / "metadata.json")
    event_columns = list(metadata["event_columns"])
    threshold_doc = load_json(model_dir / "lstm_event_thresholds.json")
    thresholds = threshold_doc["thresholds"]
    threshold_array = np.asarray([thresholds[event] for event in event_columns], dtype=np.float32)

    y_event = np.load(dataset_dir / "y_event_test.npy").astype(bool)
    probs = infer_event_probabilities(
        dataset_dir=dataset_dir,
        model_dir=model_dir,
        out_dir=out_dir,
        batch_size=args.batch_size,
        force=args.force_infer,
    )
    pred_event = probs >= threshold_array.reshape(1, -1)

    usecols = ["split", *metadata["sample_summary_columns"], *event_columns]
    sample_index = pd.read_csv(dataset_dir / "sample_index.csv.gz", usecols=usecols)
    index_test = sample_index[sample_index["split"] == "test"].reset_index(drop=True)
    if len(index_test) != y_event.shape[0]:
        raise RuntimeError(f"sample_index test rows ({len(index_test)}) != y_event_test rows ({y_event.shape[0]})")

    regimes = build_regimes(index_test, y_event, metadata)
    any_df = pd.DataFrame(any_event_rows(regimes, y_event, pred_event))
    event_df = pd.DataFrame(per_event_rows(regimes, event_columns, y_event, pred_event))

    any_df.to_csv(out_dir / "any_event_confusion_by_regime.csv", index=False)
    event_df.to_csv(out_dir / "event_head_confusion_by_regime.csv", index=False)
    pd.DataFrame(probs, columns=event_columns).to_csv(out_dir / "test_event_probabilities.csv.gz", index=False)
    write_markdown_report(
        out_path=out_dir / "recognition_report.md",
        model_dir=model_dir,
        dataset_dir=dataset_dir,
        thresholds=thresholds,
        any_df=any_df,
        event_df=event_df,
    )

    print(f"[done] wrote {out_dir / 'recognition_report.md'}")
    print(f"[done] wrote {out_dir / 'any_event_confusion_by_regime.csv'}")
    print(f"[done] wrote {out_dir / 'event_head_confusion_by_regime.csv'}")


if __name__ == "__main__":
    main()
